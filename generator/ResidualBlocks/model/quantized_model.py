from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from generator.ResidualBlocks.model.quantiation_config import QuantizationConfig, BITS, INT_MAX, INT_MIN, TANH_LIMIT
import os
import numpy as np
import random

feature_in_W = 60
feature_in_H = 560


def cut(x, in_cut_start: int = 0, en=True, bit_width=8):
    """
    result = (x >> (2 * in_cut_start).clip()
    type_out: 0-int32, 1-int8
    """
    if en:
        in_cut_start = in_cut_start
        qmax = 2 ** (bit_width - 1) - 1
        qmin = - 2 ** (bit_width - 1)
        return x.div(2 ** (2 * in_cut_start)).floor().clamp(min=qmin, max=qmax)
    else:
        return x


class QuantizResBlock(nn.Module):
    def __init__(self, param_conv1, param_conv2, activation_func, weight_dict, in_cut_start_dict):
        super(QuantizResBlock, self).__init__()
        self.conv1 = nn.Conv2d(param_conv1[0],
                               param_conv1[1],
                               param_conv1[2],
                               param_conv1[3],
                               param_conv1[4],
                               )
        self.activation1 = activation_func
        self.conv2 = nn.Conv2d(param_conv2[0],
                               param_conv2[1],
                               param_conv2[2],
                               param_conv2[3],
                               param_conv2[4],
                               )
        self.activation2 = activation_func
        self.weight_dict = weight_dict
        self.in_cut_start_dict = in_cut_start_dict


    def quantize(self):
        self.conv1.weight.data = self.conv1.weight.data.mul(self.weight_dict['conv1']).round().clamp(INT_MIN, INT_MAX)
        self.conv1.bias.data = self.conv1.bias.data.mul(self.weight_dict['conv1'] * (2**(BITS-1))).round()
        self.conv2.weight.data = self.conv2.weight.data.mul(self.weight_dict['conv2']).round().clamp(INT_MIN, INT_MAX)
        self.conv2.bias.data = self.conv2.bias.data.mul(self.weight_dict['conv2'] * (2**(BITS-1))).round()



    def forward(self, x):
        x1 = self.conv1(x)

        x1 = cut(x1, in_cut_start=self.in_cut_start_dict['conv1'], en=True, bit_width=BITS)

        x1 = self.activation1(x1)

        x1 = self.conv2(x1)

        x1 = cut(x1, in_cut_start=self.in_cut_start_dict['conv2'], en=True, bit_width=BITS)

        x1 = self.activation2(x1)

        x = x + x1  # 不能用+=会导致梯度无法传导

        x = cut(x, in_cut_start=self.in_cut_start_dict['add'], en=True, bit_width=BITS)

        return x


class QuantizNet(nn.Module):
    def __init__(self, qconfig) -> None:
        super().__init__()
        
        # self.model_params = params.model_params
        in_dim = 4
        out_dim = 6
        fnum = 32
        fw = 3

        self.in_cut_start_dict = qconfig['in_cut_start']
        self.weight_dict = qconfig['weight']

        self.conv_in = nn.Conv2d(in_dim, fnum, fw, 1, (fw - 1) // 2, bias=True)
        self.act_in = nn.ReLU()

        self.res0 = QuantizResBlock(param_conv1=[fnum, fnum, fw, 1, (fw - 1) // 2],
                                    param_conv2=[fnum, fnum, fw, 1, (fw - 1) // 2],
                                    activation_func=nn.ReLU(),
                                    weight_dict=self.weight_dict['res0'],
                                    in_cut_start_dict=self.in_cut_start_dict['res0'])
        self.res1 = QuantizResBlock(param_conv1=[fnum, fnum, fw, 1, (fw - 1) // 2],
                                    param_conv2=[fnum, fnum, fw, 1, (fw - 1) // 2],
                                    activation_func=nn.ReLU(),
                                    weight_dict=self.weight_dict['res1'],
                                    in_cut_start_dict=self.in_cut_start_dict['res1'])
        self.res2 = QuantizResBlock(param_conv1=[fnum, fnum, fw, 1, (fw - 1) // 2],
                                    param_conv2=[fnum, fnum, fw, 1, (fw - 1) // 2],
                                    activation_func=nn.ReLU(),
                                    weight_dict=self.weight_dict['res2'],
                                    in_cut_start_dict=self.in_cut_start_dict['res2'])

        self.conv_out = nn.Conv2d(fnum, out_dim, fw, 1, (fw - 1) // 2, bias=True)
        self.act_out = nn.Tanh()


    def clone(self, pretrained_model):
        self.conv_in.weight.data.copy_(pretrained_model.input_conv.weight.data)
        self.conv_in.bias.data.copy_(pretrained_model.input_conv.bias.data)

        self.res0.conv1.weight.data.copy_(pretrained_model.res_block_0.conv1.weight.data)
        self.res0.conv1.bias.data.copy_(pretrained_model.res_block_0.conv1.bias.data)
        self.res0.conv2.weight.data.copy_(pretrained_model.res_block_0.conv2.weight.data)
        self.res0.conv2.bias.data.copy_(pretrained_model.res_block_0.conv2.bias.data)

        self.res1.conv1.weight.data.copy_(pretrained_model.res_block_1.conv1.weight.data)
        self.res1.conv1.bias.data.copy_(pretrained_model.res_block_1.conv1.bias.data)
        self.res1.conv2.weight.data.copy_(pretrained_model.res_block_1.conv2.weight.data)
        self.res1.conv2.bias.data.copy_(pretrained_model.res_block_1.conv2.bias.data)

        self.res2.conv1.weight.data.copy_(pretrained_model.res_block_2.conv1.weight.data)
        self.res2.conv1.bias.data.copy_(pretrained_model.res_block_2.conv1.bias.data)
        self.res2.conv2.weight.data.copy_(pretrained_model.res_block_2.conv2.weight.data)
        self.res2.conv2.bias.data.copy_(pretrained_model.res_block_2.conv2.bias.data)

        
        self.conv_out.weight.data.copy_(pretrained_model.output_conv.weight.data)
        self.conv_out.bias.data.copy_(pretrained_model.output_conv.bias.data)


    def quantize(self):

        self.res0.quantize()
        self.res1.quantize()
        self.res2.quantize()

        self.conv_in.weight.data = self.conv_in.weight.data.mul(self.weight_dict['conv_in']).round().clamp(INT_MIN, INT_MAX)
        self.conv_in.bias.data = self.conv_in.bias.data.mul(self.weight_dict['conv_in'] * (2**(BITS-1))).round()
        
        self.conv_out.weight.data = self.conv_out.weight.data.mul(self.weight_dict['conv_out']).round().clamp(INT_MIN, INT_MAX)
        self.conv_out.bias.data = self.conv_out.bias.data.mul(self.weight_dict['conv_out'] * (2**(BITS-1))).round()


    def forward(self, x):
        # x = (x - 0.5) * 2
        #
        # x = x.mul((2**(BITS-1))).round().clamp(INT_MIN, INT_MAX)

        x = self.conv_in(x)
        x = cut(x, in_cut_start=self.in_cut_start_dict['conv_in'], en=True, bit_width=BITS)

        x = self.act_in(x)

        x = self.res0(x)
        # print(x)
        x = self.res1(x)

        x = self.res2(x)

        x = self.conv_out(x)

        x = cut(x, in_cut_start=self.in_cut_start_dict['conv_out'], en=True, bit_width=BITS)

        x = x / ((2**(BITS-1)) / TANH_LIMIT)
        x = self.act_out(x)
        x = torch.floor((x * (2**(BITS-1))).clamp(INT_MIN, INT_MAX))

        return x


def model_data_gen():
    def seed_torch(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch(7)
    input_rand = torch.randint(-128, 127, (1, 4, feature_in_H, feature_in_W)).to(torch.float)

    qconfig = QuantizationConfig()
    net = QuantizNet(qconfig)

    net.load_state_dict(torch.load("generator/ResidualBlocks/model/pretrained_model/ckpt0", map_location='cpu'))
    net.quantize()

    with torch.no_grad():
        output_act = net(input_rand)

    return input_rand, output_act, net


if __name__ == "__main__":
    def seed_torch(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    qconfig = QuantizationConfig()

    temp_quan = QuantizNet(qconfig)
    temp_quan.load_state_dict(torch.load("generator/ResidualBlocks/model/pretrained_model/ckpt0", map_location='cpu'))
    temp_quan.quantize()

    seed_torch(7)
    rgbd = torch.randint(-128, 127, (1, 4, feature_in_H, feature_in_W)).to(torch.float)

    net1 = temp_quan.to("cpu")

    net1.eval()

    outputs1 = net1(rgbd)

    outputs1 = outputs1.to('cpu')

    print(outputs1)