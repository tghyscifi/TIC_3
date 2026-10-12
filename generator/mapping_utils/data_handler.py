import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from generator.resnet50.resnet import resnet50, Cut
from math import ceil
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings


class DataHandler:
    def __init__(self):
        pass

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
        assert (raw_data.dtype == dtype), "The data type must be the same as the raw data type"

        for position in split_dict.keys():  # keys(0,0)是干什么的？
            assert (result.get(position) is None)

            # check whether the split dict overflows，拆分方法中指定的第i个数是否大于数据的第i个维度长度
            for i, (_, end_position) in enumerate(split_dict[position]):
                if end_position > raw_data.shape[i]:
                    warnings.warn(
                        "split dict overflows! end position {:d} > shape {:d}".format(end_position, raw_data.shape[i]))

            # check whether start position well aligned
            for align, (start_position, _) in zip(alignment, split_dict[position]):
                if align is not None:
                    # assert (start_position % align == 0)
                    if start_position % align != 0:  # start position是干嘛的？
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
            if len(raw_data.shape) == 4 and (not is_weight):
                warnings.warn('Data has 4 dims, but is_weight is False, So is_weight is forced to be True! ' +
                              'This may result in Error when saving data!')
            if is_weight:  # weight
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

    @staticmethod
    def lut_gen(func, max_abs_float_input: float, input_width: int, input_q_factor: int, output_width: int,
                output_q_factor: int, signed=True):
        """
        Generate LUT for Tianjic X1 (Using torch)
        parameters:
            func: nonlinear function. Could be str or function: 'sigmoid', 'tanh'; OR lambda x: 1/x;
            max_abs_float_input: max abs value of input data (float);
            input_width: width of input data (Quantized data for inference)
            input_q_factor: Quantized data = input float data * input_q_factor
            output_width: width of output data (Quantized data for inference)
            output_q_factor: Quantized data = output float data * output_q_factor
        """
        if isinstance(func, str):
            if func == 'sigmoid':
                def func(var):
                    return torch.sigmoid(var)
            elif func == 'tanh':
                def func(var):
                    return torch.tanh(var)
            elif func == 'x':
                def func(var):
                    return var
            else:
                raise NotImplementedError
        if output_width == 8 and signed:
            out_type = 1
            alignment = 16
            out_dtype = np.int8
        elif output_width == 32 and signed:
            out_type = 0
            alignment = 8
            out_dtype = np.int32
        else:
            raise NotImplementedError
        if signed:
            iw = input_width - 1
            ow = output_width - 1
            x_float = (max_abs_float_input / input_q_factor * torch.arange(-2 ** iw, 2 ** iw)).clamp(
                min=-max_abs_float_input, max=max_abs_float_input)
            y_float = func(x_float)
            y_int = torch.floor(
                (y_float * output_q_factor).clamp(min=-2 ** ow, max=2 ** ow - 1)).detach().numpy().astype(out_dtype)
            y_int = np.append(y_int[2 ** ow:], y_int[0: 2 ** ow])
        else:
            raise NotImplementedError
        y_int = DataHandler.tensor_split(raw_data=y_int, split_dict={(0, 0): ((0, 2 ** output_width),)},
                                         data_type=out_type, alignment=(alignment,), dims=(0,), is_weight=False)
        return y_int[(0, 0)]

    @staticmethod
    def check_model(model):
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if module.bias is not None:
                    assert module.bias.data.min() >= -0x80000000 and module.bias.data.max() <= 0x7fffffff
                    assert torch.sum(module.bias.data - module.bias.data.round()) == 0
                assert module.weight.data.min() >= -0x80 and module.weight.data.max() <= 0x7f
                assert torch.sum(module.weight.data - module.weight.data.round()) == 0


if __name__ == '__main__':
    from itertools import product

    handler = DataHandler()

    q_in_factor = 127
    q_out_factor = 127

    lut = np.array(handler.lut_gen('sigmoid', 1., 8, q_in_factor, 8, q_out_factor))
    import torch

    while True:
        rand = torch.randn((2048,)).clamp(-1, 1)
        x = rand.mul(q_in_factor).floor().mul(1. / q_in_factor)
        x_bar = np.floor(rand.detach().numpy() * q_in_factor) * (1. / q_in_factor)

        # y1 = np.array(torch.sigmoid(x) * q_out_factor, dtype=np.int8)
        y1 = np.array(x * q_out_factor, dtype=np.int8)

        qx = np.floor(np.array(x * q_in_factor, dtype=np.int32)).astype(np.uint8).tolist()
        qx_bar = np.array(torch.floor(x * q_in_factor)).astype(np.uint8).tolist()
        y2 = lut[qx]

        if np.sum(y1 - y2) != 0:
            yyy = 1
        print('..')
