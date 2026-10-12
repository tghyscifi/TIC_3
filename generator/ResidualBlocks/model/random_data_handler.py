import numpy as np
from numpy.lib.shape_base import split
import os
import sys

sys.path.append(os.getcwd())
from generator.mapping_utils.data_handler import DataHandler
from math import ceil
import torch
import torch.nn as nn
from collections import OrderedDict
from generator.ResidualBlocks.model.random_model import random_data_gen, Cut, model_data_gen


class ModelDataHandler(DataHandler):

    def __init__(self, seed=7, model='random'):
        super(ModelDataHandler, self).__init__()

        # 获取模型和测试输入输出数据
        if model=='random':
            input_x, output_y, self.__model = random_data_gen()
        else:
            input_x, output_y, self.__model = model_data_gen()

        # 检查数据是否量化，量化的数据是否越界
        DataHandler.check_model(self.__model)

        internal_data_input = []
        internal_data_output = []
        internal_name = []
        self.__internal = OrderedDict()

        def hook_fn(module, input, output):
            """hook function for input/output tensor extraction
            获取整个模型中各个module输入输出数据并添加到列表中

            Raises:
                输入是tuple并且长度为1（单个输入）
            """
            assert (type(input) is tuple and len(input) == 1)
            assert (input[0].max() < 2**31 - 1 and input[0].min() > -2**31)
            assert (output.max() < 2 ** 31 - 1 and output.min() > -2 ** 31)
            internal_data_output.append(output.clone().detach())
            internal_data_input.append(input[0].clone().detach())

        # 添加hook function
        for name, module in self.__model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear,
                                       nn.AdaptiveAvgPool2d, nn.BatchNorm2d, Cut, nn.Tanh)):
                continue
            # if not isinstance(module, nn.ReLU):
            #     continue
            # print(name)
            module.register_forward_hook(hook_fn)
            internal_name.append(name)

        # 运行网络（运行hook function）
        y = self.__model(input_x)

        self.__internal['output'] = np.array(y.squeeze(0).detach(), dtype=np.int32)

        # 将所有module的名字、输入输出等汇总到一个Dict中
        for name, data_in, data_out in zip(internal_name, internal_data_input, internal_data_output):  # zip函数可以将可迭代对象打包成元组列表
            # print(name)
            # print(data_in)
            # print(data_out)
            self.__internal[name] = {
                'input': np.array(data_in.squeeze(0), dtype=np.int32),
                'output': np.array(data_out.squeeze(0), dtype=np.int32)
            }

        # 将所有module的参数汇总到Dict中
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


if __name__ == '__main__':
    from itertools import product

    handler = ModelDataHandler(model='real')

    print(handler.names)
    print(handler.parameters['activation_out'])
