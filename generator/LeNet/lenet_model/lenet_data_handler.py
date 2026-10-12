import numpy as np
from numpy.lib.shape_base import split
import os
import sys

sys.path.append(os.getcwd())
from generator.LeNet.lenet_model.lenet import LeNet, Cut
from generator.mapping_utils.data_handler import DataHandler
from math import ceil
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings
from generator.LeNet.lenet_cut_config import QuantizationConfig


class LeNetDataHandler(DataHandler):

    def __init__(self, seed=7):
        super(LeNetDataHandler, self).__init__()
        torch.manual_seed(seed)
        x = torch.randn((1, 1, 28, 28))

        self.__model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)
        ckpt_path = 'generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth'
        if ckpt_path is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(ckpt_path, map_location=device)
            self.__model.load_state_dict(pretrained_dict)
        else:
            for name, module in self.__model.named_modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.data.mul_(128).round_().clamp_(-128, 127)
                    module.bias.data.mul_(128 * 128).round_().clamp_(-0x80000000, 0x7fffffff)
                elif isinstance(module, nn.Linear):
                    module.weight.data.mul_(128).round_().clamp_(-128, 127)
                    module.bias.data.mul_(128 * 128).round_().clamp_(-0x80000000, 0x7fffffff)

        quantize = QuantizationConfig()
        for name, module in self.__model.named_modules():
            if isinstance(module, Cut):
                module.en = True
                module.in_cut_start = quantize[name]

        DataHandler.check_model(self.__model)

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
                                       nn.AdaptiveAvgPool2d, nn.BatchNorm2d, Cut)):
                continue
            module.register_forward_hook(hook_fn)
            internal_name.append(name)
        y = self.__model(x)
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


if __name__ == '__main__':
    from itertools import product

    handler = LeNetDataHandler()

    xx = 1
