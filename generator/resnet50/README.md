# 算法需要提供给手动映射的文件

以ResNet50为例说明算法需要提供给手动映射的文件:

1. 算法模型, 参考`resnet.py`, 能通过`resnet.resnet50`获得模型类, 算法模型中加入Cut模块, 代码如下所示:
```python
class Cut(nn.Module):
    def __init__(self, en: bool = False, in_cut_start: int = 4, type_out: int = 1):
        """
        result = (x >> (2 * in_cut_start).clip()
        type_out: 0-int32, 1-int8
        """
        super(Cut, self).__init__()
        self.en = en
        self.in_cut_start = in_cut_start
        self.type_out = type_out
        if type_out == 0:
            self.max, self.min = 0x7fffffff, -0x80000000
        elif type_out == 1:
            self.max, self.min = 0x7f, -0x80
        else:
            raise ValueError
    
    def forward(self, x: Tensor) -> Tensor:
        if self.en:
            return x.div(2 ** (2 * self.in_cut_start)).floor().clamp(min=self.min, max=self.max)
        return x

    def __repr__(self):
        return '{}(en={}, in_cut_start={:d}, type_out={:d})'.format(self.__class__.__name__,
                                                                    self.en,
                                                                    self.in_cut_start, self.type_out)
```
Cut模块用于在模型中引入截取 (即引入量化信息), 该模块需要加到所有需要量化的算子后

**截取模块的加入需要对映射具有一定的了解**, 这里只对一般情况进行说明: 一般来说, 所有包含乘法和加法的操作后需要引入截取模块, 例如Conv, BN, 全连接, AvgPool, 加法等, 如果Conv后紧跟着BN, 因为相邻的Conv和BN可以融合, 映射阶段只映射Conv. 此外, 截取模块必须加在MaxPool或ReLU之后, 因为映射阶段实际是通过设置Soma的截取参数来进行截取, 避免额外生成专用于截取的Soma原语

**加法和Concat**等操作的截取模块生成比较复杂, 需要考虑量化策略和映射的实现, 要具体情况具体分析

2. 截取参数字典, 参考`resnet_50_cut_parameters.py`, 字典的key为Cut模块在模型中的名字, value为截取的参数值, 该字典会在模型和映射两部分用到, 模型中将参数值赋给对应的cut模块, 映射时将参数值赋给对应的Soma原语进行截取

3. 数据处理器, 参考`data_handler.py`中的`ResNetDateHandler`类, 这个类实现的功能包括: 进行一次模型的前向推理, 从而可以通过`handler.parameters['LAYER_NAME']['input/output/bias/weight']`获得对应数据; 类中实现了`tensor_split`方法, 调用该方法可为映射的结果检查生成参考数据 (这个方法不用改)

可能需要修改的代码以注释的形式表明:
```python
class ResNetDataHandler:
    
    def __init__(self, seed=5):
        torch.manual_seed(seed)
        self.__model = resnet50(pretrained=False, progress=True, batch_norm=False, quantization=True)  # 需要替换成算法模型
        x = torch.randn((1, 3, 224, 224))  # 算法模型的实际输入形状
        x.mul_(128).floor_().clamp_(min=-128, max=127)
        internal_data_input = []
        internal_data_output = []
        internal_name = []
        self.__internal = OrderedDict()

        def hook_fn(module, input, output):
            assert (type(input) is tuple and len(input) == 1)
            internal_data_output.append(output.clone().detach())
            internal_data_input.append(input[0].clone().detach())

        for name, module in self.__model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear,
                                       nn.AdaptiveAvgPool2d, nn.BatchNorm2d, Cut)):  # 如果有其他需要check的计算, 需要在这里加上
                continue
            module.register_forward_hook(hook_fn)
            internal_name.append(name)
        _ = self.__model(x)
        for name, data_in, data_out in zip(internal_name, internal_data_input, internal_data_output):
            self.__internal[name] = {
                'input': np.array(data_in.squeeze(0), dtype=np.int32),
                'output': np.array(data_out.squeeze(0), dtype=np.int32)
            }

        for name, p in self.__model.named_parameters():
            layer_name, p_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            assert (self.__internal.get(layer_name) is not None)
            if p_name == 'bias':
                self.__internal[layer_name][p_name] = np.array(p.clone().detach(), dtype=np.int32)
            elif p_name == 'weight':
                self.__internal[layer_name][p_name] = np.array(p.clone().detach(), dtype=np.int8)
            else:
                raise ValueError

    @property
    def names(self):
        return self.__internal.keys()

    @property
    def parameters(self):
        return self.__internal

    @staticmethod
    def tensor_split(raw_data, split_dict, data_type, alignment, dims, is_weight=False):
        """
        raw_data:   待拆分的数据，需要为np.array类型
                    [C, H, W] or [C_out, C_in, Ky, Kx] or [Bias] or [C_out, C_in]
        split_dict: 拆分方法
                    {
                        (0, 0): ((0, 16), (0, 224), (0, 224)),
                    }
        data_type:  数据类型： 0 - int32; 1 - int 8; others - not support
        alignment:  对齐，每个方向需要对其的元素的个数： (16, None, None) - 代表第一个维度16个数字对齐
        dims:       存储时的优先顺序， 左侧优先
        """
        result = {}
        if data_type == 0:
            dtype = np.int32
            successive_length = 1
        elif data_type == 1:
            dtype = np.int8
            successive_length = 4
        else:
            raise NotImplementedError
        assert (raw_data.dtype == dtype)
        for position in split_dict.keys():
            assert (result.get(position) is None)
            # check whether start position well aligned
            for align, (start_position, _) in zip(alignment, split_dict[position]):
                if align is not None:
                    # assert (start_position % align == 0)
                    if start_position % align != 0:
                        warnings.warn(
                            'start position: {:d} cannot be divisible by alignment: {:d}'.format(start_position, align))
            new_data, new_shape, new_slice = [], [], []
            partial_data = raw_data[tuple([slice(*i) for i in split_dict[position]])]
            shape = partial_data.shape
            assert (len(shape) == len(alignment))
            for align, item in zip(alignment, shape):
                new_slice.append(slice(0, item))
                if align is not None:
                    item = ceil(item / align) * align
                new_shape.append(item)
            new_partial_data = np.zeros(tuple(new_shape), dtype=dtype)
            new_partial_data[tuple(new_slice)] = partial_data
            new_partial_data = new_partial_data.transpose(dims)
            if len(raw_data) == 4:
                warnings.warn('Data has 4 dims, but is_weight is False, So is_weight is forced to be True! ' +
                              'This may result in Error when saving data!')
            if is_weight:    # weight
                flatten_new_partial_data = np.array([], dtype=dtype)
                for c_out_group in range(new_partial_data.shape[0] // alignment[0]):
                    flatten_new_partial_data = np.append(
                        flatten_new_partial_data,
                        new_partial_data[c_out_group * alignment[0]: (c_out_group + 1) * alignment[0]].ravel(order='F'))
            else:
                flatten_new_partial_data = new_partial_data.ravel(order='F')
            result[position] = flatten_new_partial_data.reshape(-1, successive_length).tolist()
            # assert (len(flatten_new_partial_data) % successive_length == 0)
            # cnt, temp_list = 0, []
            # while cnt < len(flatten_new_partial_data):
            #     temp_list.append(flatten_new_partial_data[cnt])
            #     if (cnt + 1) % successive_length == 0:
            #         new_data.append(temp_list)
            #         temp_list = []
            #     cnt += 1
            # result[position] = new_data
        return result
```