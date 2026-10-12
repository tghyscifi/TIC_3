import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from generator.SoundSNN.sound_snn import SNN
import torch
import torch.nn as nn
from generator.mapping_utils.data_handler import DataHandler
from generator.SoundSNN.sound_snn import LIFFCNeuron
from generator.SoundSNN.snn_config import SNNConfig

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SNNDataHandler(DataHandler):
    def __init__(self, seed=5, pretrained=False):
        super(DataHandler, self).__init__()
        torch.manual_seed(seed)

        snn_config = SNNConfig()
        self.__model = SNN(snn_config)

        if pretrained:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load('generator/SoundSNN/quantized_model.pth', map_location=device)
            self.__model.load_state_dict(state_dict)
        else:
            for name, module in self.__model.named_modules():    
                if isinstance(module, LIFFCNeuron):
                    module.kernel.weight.data = module.kernel.weight.data.mul(128).round().clamp(-128, 127)
                    module.kernel.bias.data = module.kernel.bias.data.mul(128).round().clamp(-2**31, 2**31 - 1)
        
        # inference
        x = torch.randn((1, 39, 16))
        x = x.mul(128).round().clamp(-128, 127)
        _ = self.__model(x)
        self.__internal = self.__model.summary

    @property
    def names(self):
        return self.__internal.keys()

    @property
    def parameters(self):
        return self.__internal


if __name__ == '__main__':
    handler = SNNDataHandler()
    print(handler.names)
