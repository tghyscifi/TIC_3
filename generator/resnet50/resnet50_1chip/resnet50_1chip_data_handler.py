import numpy as np
from numpy.lib.shape_base import split
import os
import sys

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_1chip.resnet50_1chip_q import resnet50_1chip, Cut
from generator.mapping_utils.data_handler import DataHandler
from math import ceil
import torch
import torch.nn as nn
from collections import OrderedDict
import warnings


class ResNetDataHandler(DataHandler):

    def __init__(self, seed=5, ckpt=None):
        super(ResNetDataHandler, self).__init__()
        torch.manual_seed(seed)
        if ckpt is None:
            ckpt_path = None
        elif ckpt == 0:
            ckpt_path = './generator/resnet50/checkpoints/resnet50_1chip_9class_0_final_90.4.pth'
        elif ckpt == 1:
            ckpt_path = './generator/resnet50/checkpoints/resnet50_1chip_9class_1_final_88.0.pth'
        elif ckpt == 2:
            ckpt_path = './generator/resnet50/checkpoints/resnet50_1chip_9class_2_final_88.0.pth'
        elif ckpt == 3:
            ckpt_path = './generator/resnet50/checkpoints/resnet50_1chip_9class_3_final_84.9.pth'
        else:
            raise ValueError
        self.__model = resnet50_1chip(pretrained=ckpt_path, batch_norm=False, cut_en=True)
        # print(self.__model)

        if ckpt is None:
            x = torch.randn((1, 3, 224, 224)).mul(128).round().clamp(-128, 127)
            for name, module in self.__model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.data = module.weight.data.mul(128).round().clamp(-128, 127)
                    module.bias.data = module.bias.data.mul(128 * 128).round().clamp(-0x80000000, 0x7fffffff)
        else:
            x = torch.randn((1, 3, 224, 224))
            self.__model.quantize(ckpt=ckpt)

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

    def inference(self, x):
        return self.__model(x)


if __name__ == '__main__':
    from itertools import product

    handler = ResNetDataHandler(ckpt=0)
    print(handler.names)
