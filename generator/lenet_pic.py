import numpy as np
import sys
import os
import torch
import torch.nn as nn
import warnings
import math
from itertools import product
from collections import OrderedDict
from math import ceil
from PIL import Image   # 新增：用于读取真实图片
sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.LeNet.lenet_model.lenet import LeNet, Cut
from generator.mapping_utils.data_handler import DataHandler
from generator.LeNet.lenet_cut_config import QuantizationConfig
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from collections import OrderedDict
import warnings


# ================ lenet.py ================
# 用于量化权重和INT8激活
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_min, x_max, x_range):
        ctx.save_for_backward(x, torch.tensor(x_min), torch.tensor(x_max))
        x = x.mul(x_range).round().clamp(-128, 127).mul(1 / x_range)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask0 = torch.where(x < x_min, zeros, ones)
        mask1 = torch.where(x > x_max, zeros, ones)
        mask = mask0 * mask1
        grad = grad_output * mask
        return grad, None, None, None


# 用于量化偏置和INT32激活
class FakeQuantizeWithoutLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_range):
        ctx.save_for_backward(x)
        x = x.mul(x_range).round().mul(1 / x_range)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Cut(nn.Module):
    def __init__(self, en: bool = False, in_cut_start: int = 4, type_out: int = 1):
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

    def forward(self, x):
        if self.en:
            return x.div(2 ** (2 * self.in_cut_start)).floor().clamp(min=self.min, max=self.max)
        else:
            return x

    def __repr__(self):
        return '{}(en={}, in_cut_start={:d}, type_out={:d})'.format(self.__class__.__name__, self.en,
                                                                    self.in_cut_start, self.type_out)


class LeNet(torch.nn.Module):
    def __init__(self, batch_norm=True, clamp=False, aware_test=None,
                 cut_en=False, aware=False):
        super(LeNet, self).__init__()

        self.batch_norm = batch_norm
        self.clamp = clamp
        self.cut_en = cut_en
        self.aware = aware
        self.aware_test = aware_test

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=not batch_norm)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(6)
        self.cut1 = Cut(en=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=not batch_norm)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(16)
        self.cut2 = Cut(en=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(400, 120, bias=not batch_norm)
        if self.batch_norm:
            self.fc_bn1 = nn.BatchNorm1d(120)
        self.fc_cut1 = Cut(en=True)
        self.fc_relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(120, 84, bias=not batch_norm)
        if self.batch_norm:
            self.fc_bn2 = nn.BatchNorm1d(84)
        self.fc_cut2 = Cut(en=True)
        self.fc_relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(84, 10, bias=not batch_norm)
        self.fc_cut3 = Cut(en=True)

    def forward(self, x):
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        if self.cut_en:
            x = x.mul(128).round().clamp(-128, 127)

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        if self.cut_en:
            x = self.cut1(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.cut_en:
            x = self.cut2(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        if self.batch_norm:
            x = self.fc_bn1(x)
        if self.cut_en:
            x = self.fc_cut1(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.fc_relu1(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.fc_bn2(x)
        if self.cut_en:
            x = self.fc_cut2(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.fc_relu2(x)

        x = self.fc3(x)

        x = self.fc_cut3(x)

        return x


def lenet(pretrained=None, batch_norm=True, clamp=False, aware_test=None, cut_en=False, aware=False):
    model = LeNet(batch_norm=batch_norm, clamp=clamp, aware_test=aware_test, cut_en=cut_en, aware=aware)
    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained, map_location=device)
        if pretrained_dict.get('state_dict') is not None:
            pretrained_dict = pretrained_dict['state_dict']
        model.load_state_dict(pretrained_dict, strict=False)
    return model


# ================ lenet_cut_config.py ================
class QuantizationConfig:
    def __init__(self, name='in_cut_start'):
        self.__name = name

        self.__weight_parameters = {
            'conv1': 256,
            'conv2': 256,
            'fc1': 1024,
            'fc2': 256,
            'fc3': 256,
        }

        self.__cut_parameters = {
            'cut1': int(math.log(self.__weight_parameters['conv1'], 4)),
            'cut2': int(math.log(self.__weight_parameters['conv2'], 4)),
            'fc_cut1': int(math.log(self.__weight_parameters['fc1'], 4)),
            'fc_cut2': int(math.log(self.__weight_parameters['fc2'], 4)),
            'fc_cut3': int(math.log(self.__weight_parameters['fc3'] * 4, 4))
        }

    def __getitem__(self, item):
        if self.__name == 'in_cut_start':
            result = self.__cut_parameters.get(item)
        else:
            assert self.__name == 'weight'
            result = self.__weight_parameters.get(item)
        if result is None:
            warnings.warn('Module \'{}\' do not have in_cut_start parameter!'.format(item))
        return result


class LeNetDataHandler(DataHandler):
    # 修改：增加 input_image_path 参数，默认为 None（使用随机输入）
    def __init__(self, seed=7, input_image_path=None):
        super(LeNetDataHandler, self).__init__()
        torch.manual_seed(seed)
        
        if input_image_path is not None:
            x = self._load_and_preprocess_image(input_image_path)
        else:
            x = torch.randn((1, 1, 28, 28))

        self.__model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)
        ckpt_path = './package/lenet_mnist.pth'
        if ckpt_path is not None and os.path.exists(ckpt_path):
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
        
        for name, module in self.__model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
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
            if self.__internal.get(layer_name) is not None:
                if p_name == 'bias':
                    self.__internal[layer_name][p_name] = np.array(p.clone().detach(), dtype=np.int32)
                elif p_name == 'weight':
                    self.__internal[layer_name][p_name] = np.array(p.clone().detach(), dtype=np.int8)
                else:
                    raise ValueError

    def _load_and_preprocess_image(self, image_path):
        """加载单张灰度图片并预处理为模型输入格式"""
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 255.0) * 2.0 - 1.0
        x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        return x

    @property
    def names(self):
        return self.__internal.keys()

    @property
    def parameters(self):
        return self.__internal


# ================ lenet_data.py ================
def generate_data(handler):
    g_data = {
        'conv1': {},
        'cut1': {},
        'max_pool1': {},
        'conv2': {},
        'max_pool2': {},
        'fc1': {},
        'fc_relu1': {},
        'fc2': {},
        'fc_relu2': {},
        'fc3': {},
        'fc_cut3': {}
    }

    g_data['conv1']['input'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv1']['input'].astype(np.int8),
        split_dict={(0, 0): ((0, 1), (0, 28), (0, 28))},
        data_type=1, alignment=(None, None, 16),
        dims=(2, 1, 0))
    g_data['conv1']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv1']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 6), (0, 1), (0, 5), (0, 5))},
        data_type=1, alignment=(32, None, None, None),
        dims=(0, 3, 2, 1), is_weight=True)
    g_data['conv1']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv1']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 6), )},
        data_type=0, alignment=(32,), dims=(0,))
    g_data['cut1']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['cut1']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 6), (0, 24), (0, 24))},
        data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    g_data['max_pool1']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['max_pool1']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 6), (0, 12), (0, 12))},
        data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    g_data['conv2']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv2']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 16), (0, 6), (0, 5), (0, 5))},
        data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    g_data['conv2']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv2']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 16),)},
        data_type=0, alignment=(32,), dims=(0,))
    g_data['max_pool2']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['max_pool2']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 16), (0, 5), (0, 5))},
        data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    g_data['fc1']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc1']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 120), (0, 400))},
        data_type=1, alignment=(32, None),
        dims=(0, 1), is_weight=True)
    g_data['fc1']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc1']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 120),)},
        data_type=0, alignment=(32,), dims=(0,))

    g_data['fc2']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc2']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 84), (0, 120))},
        data_type=1, alignment=(32, None),
        dims=(0, 1), is_weight=True)
    g_data['fc2']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc2']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 84),)},
        data_type=0, alignment=(32,), dims=(0,))

    g_data['fc3']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc3']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 10), (0, 84))},
        data_type=1, alignment=(32, None),
        dims=(0, 1), is_weight=True)
    g_data['fc3']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc3']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 10),)},
        data_type=0, alignment=(32,), dims=(0,))

    g_data['fc_cut3']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc_cut3']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 10), )},
        data_type=1, alignment=(16,),
        dims=(0,))

    return g_data


# ================ g0.py ================
def gen_0_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, out_data_en=False, chip=(0, 0),
                     in_data_en=False):
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=32,
                        num_in=28, length_ciso=1, num_ciso=28, length_out=32, num_out=28,
                        type_in=1, type_out=1,
                        data_in=data['conv1']['input'][(0, 0)])
            if not out_data_en:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=32 * 28 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x380, addr_din_length=16 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=32 * 28 // 8 - 1,
                            A_offset=0, Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    return map_config


# ================ g2.py ================
def gen_2_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, in_data_en=False, out_data_en=False,
                     chip=(0, 0), init_data=False, send_to_fpga=False):
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=16,
                        num_in=1, length_ciso=1, num_ciso=1, length_out=16, num_out=1,
                        type_in=1, type_out=1,
                        data_in=data['fc_cut3']['output'][(0, 0)] if init_data else None)
            if not out_data_en and core_x == 7:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en or core_x != 7,
                         receive_en=in_data_en or core_x != 0, send_num=16 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=16 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            if core_x == 7:
                if not send_to_fpga:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=1 - core_x, Y=-1 - core_y, A=0, pack_per_Rhead=16 // 8 - 1,
                                    A_offset=0, Const=0, EN=1)
                else:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-128, A=0, pack_per_Rhead=16 // 8 - 1,
                                    A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=16 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            if in_data_en or core_x != 0:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x0000 >> 2, length_in=16,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=16, num_out=1,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    return map_config


def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False):
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=16,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=16, num_out=1,
                            type_in=1, type_out=1,
                            data_in=data['fc_cut3']['output'][(0, 0)] if init_data else None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=16 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x380, addr_din_length=32 * 28 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=16 // 8 - 1,
                            A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x0000 >> 2, length_in=32,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=32, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    offset = 1

    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p81(px=28, py=28, cin=1, cout=6, kx=5, ky=5, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x10320 >> 2, addr_out=0x1040 >> 2,
                       ina_type=1, inb_type=1, load_bias=2,
                       data_x=None if in_data_en else data['conv1']['input'][(core_x, core_y)],
                       data_w=data['conv1']['weight'][(core_x, core_y)],
                       data_b=data['conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1040 >> 2, addr_out=0x0400 >> 2,
                        cin=32, cout=6, px=24, py=24,
                        kx=2, ky=2, sx=2, sy=2, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['cut1'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=12, py=12, cin=6, cout=16, kx=5, ky=5, sx=1, sy=1,
                       addr_ina=0x400 >> 2, addr_inb=0x103A0 >> 2, addr_bias=0x11660 >> 2, addr_out=0x1040 >> 2,
                       pad_top=1, pad_down=1, pad_left=1, pad_right=1,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['conv2']['weight'][(core_x, core_y)],
                       data_b=data['conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1040 >> 2, addr_out=0x1A40 >> 2,
                        cin=32, cout=16, px=10, py=10,
                        kx=2, ky=2, sx=2, sy=2, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['cut2'], row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=400, cout=120, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x1A40 >> 2, addr_inb=0x116e0 >> 2, addr_bias=0x1dee0 >> 2, addr_out=0x400 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['fc1']['weight'][(core_x, core_y)],
                       data_b=data['fc1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x400 >> 2, addr_out=0x1000 >> 2,
                        cin=128, cout=120, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['fc_cut1'], row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=120, cout=84, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x1000 >> 2, addr_inb=0x8000 >> 2, addr_bias=0xAD00 >> 2, addr_out=0x400 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['fc2']['weight'][(core_x, core_y)],
                       data_b=data['fc2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x400 >> 2, addr_out=0x600 >> 2,
                        cin=96, cout=84, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['fc_cut2'], row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=84, cout=10, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x600 >> 2, addr_inb=0x1e0e0 >> 2, addr_bias=0x1eb60 >> 2, addr_out=0x1000 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['fc3']['weight'][(core_x, core_y)],
                       data_b=data['fc3']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1000 >> 2, addr_out=0x2000 >> 2,
                        cin=32, cout=10, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['fc_cut3'], row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


# ================ lenet_check.py ================
def check(data, compare, empty_offset=0):
    data_1 = {}
    for i in range(32):
        data_1[i] = {}

    chip = (0, 0)

    data_1[empty_offset + 1][(chip, (0, 0))] = [{
        'data': data['max_pool1']['output'][(0, 0)],
        'addr_start': 0x0400 >> 2,
        'type': 1
    }]

    data_1[empty_offset + 2][(chip, (0, 0))] = [{
        'data': data['max_pool2']['output'][(0, 0)],
        'addr_start': 0x1A40 >> 2,
        'type': 1
    }]

    data_1[empty_offset + 5][(chip, (0, 0))] = [{
        'data': data['fc_cut3']['output'][(0, 0)],
        'addr_start': 0x2000 >> 2,
        'type': 1
    }]

    chip = (0, 0)
    data_1[empty_offset + 0][(chip, (1, 0))] = [{
        'data': data['fc_cut3']['output'][(0, 0)],
        'addr_start': 0x8380,
        'type': 1
    }]

    chip = (0, 0)
    data_1[empty_offset + 0][(chip, (0, 1))] = [
        {
            'data': data['fc_cut3']['output'][(0, 0)],
            'addr_start': 0x0000,
            'type': 1
        },
        {
            'data': data['fc_cut3']['output'][(0, 0)],
            'addr_start': 0x8380,
            'type': 1
        }
    ]

    compare.add_ref_data(data_1)


# ================ 主程序入口 ================
if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description='LeNet simulation with real image input')
    parser.add_argument('--input_image', type=str, default=None,
                        help='Path to input image (grayscale, any size). If not provided, random input will be used.')
    args = parser.parse_args()

    case_file_name = 'LeNet_pic'
    chip = (0, 0)
    send_to_fpga = True

    # 根据命令行参数决定输入
    if args.input_image is not None:
        print(f"Using real image from: {args.input_image}")
        handler = LeNetDataHandler(input_image_path=args.input_image)
    else:
        print("No input image provided, using random input.")
        handler = LeNetDataHandler()   # 随机输入

    data = generate_data(handler)
    cuts = QuantizationConfig()
    config = MapConfigGen()

    phase = np.zeros(50).astype(int)
    phase[0] = 1
    phase[:] = 1

    clock_in_phase = 20_0000
    config_0 = gen_0_map_config(phase, clock_in_phase, size_x=1, size_y=1, data=data, out_data_en=True, chip=(0, 0),
                                in_data_en=not send_to_fpga)
    config.add_config(config_0, core_offset=(1, 0))

    config_1 = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, in_cut_start_dict=cuts,
                                in_data_en=True, out_data_en=True, chip=chip, data=data, init_data=True)
    config.add_config(config_1, core_offset=(0, 0))

    config_2 = gen_2_map_config(phase, clock_in_phase, size_x=8, size_y=1, data=data, in_data_en=True,
                                out_data_en=True, chip=(0, 0), init_data=True, send_to_fpga=send_to_fpga)
    config.add_config(config_2, core_offset=(0, 1))

    MapConfigGen.add_router_info(map_config=config.map_config)

    prim = {
        'axon': None, 'soma1': None, 'router': None,
        'soma2': p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x10000 >> 2, length_in=1024,
                     num_in=12, length_ciso=1, num_ciso=12, length_out=1024, num_out=12,
                     type_in=1, type_out=1, data_in=None)
    }
    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    config.map_config['sim_clock'] = 10_0000
    config.map_config['step_clock'] = {
        ((0, 0), 0): (10_0000 - 1, 10_0000)
    }

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "/simulator/Out_files/" + case_file_name + "/"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        if os.path.exists('cmp_out'):
            os.system('rm -rf cmp_out')
        os.chdir(c_path)

    test_config = {
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config.map_config, test_config)
    assert tester.run_test()
