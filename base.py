import math
import torch
from abc import ABCMeta, abstractmethod


class QModule(torch.nn.Module):
    __meta_class__ = ABCMeta
    
    def __init__(self):
        super(QModule, self).__init__()
        self.quantization_mode = False
        self.aware_mode = False
        
    @abstractmethod
    def count_q_params(self):
        self.quantization_mode = False
        self.aware_mode = False
        
    @abstractmethod
    def quantize(self):
        self.quantization_mode = True
        self.aware_mode = False
    
    @abstractmethod
    def aware(self):
        self.quantization_mode = False
        self.aware_mode = True
    
    @abstractmethod
    def dequantize(self):
        pass
    
    
class QModel(QModule):
    def __init__(self, bit_shift_unit):
        super(QModel, self).__init__()
        self.bit_shift_unit = 2  ## TianjicX1硬件参数
        
    def count_q_params(self):
        QModule.count_q_params()
        for _, module in self.named_modules():
            if isinstance(QModule):
                module.count_q_params(self.bit_shift_unit)
    

class QConv2d(torch.nn.Conv2d, QModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None):
        torch.nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        self.weight_scale = None
        self.bit_shift = None
        
    def count_q_params(self, bit_shift_unit):
        '''卷积中计算量化参数
        weight_absmax * weight_scale = 128
        weight_scale = 2^(bit_shift_unit * n)
        bit_shift = bit_shift_unit * n = log_2 (128 / weight_absmax)
        n = round(log_2 (128 / weight_absmax) / bit_shift_unit)
        这里取整方法可以有很多
        '''
        weight_absmax = self.weight.data.abs().max()
        temp = math.log(128 / weight_absmax, 2) / bit_shift_unit
        if temp - math.floor(temp) >= 0.75:  # 经验公式
            n = math.ceil(temp)
        else:
            n = math.floor(temp)
        self.bit_shift = bit_shift_unit * n
        self.weight_scale = 2 ** self.bit_shift
        
    def forward(self, x):
        pass
        
    def quantize(self):
        QModule.quantize()
        
    def aware(self):
        QModule.aware()
        
        
if __name__ == '__main__':
    qconv = QConv2d(3, 16, 3)
    conv = torch.nn.Conv2d(3, 16, 3)
    print(conv.weight.data.abs().max())
    qconv.load_state_dict(conv.state_dict())
        
    


        