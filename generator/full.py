#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: snn_hardware_mapping.py
功能: SNN (Spiking LeNet) 的硬件映射配置生成与模拟器测试。
使用方法:
    python snn_hardware_mapping.py --ckpt path/to/snn_weight.pth
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import product

# 导入 SNN 相关库
import snntorch as snn
from snntorch import surrogate

# 导入硬件映射所需的 primitives 和工具
from generator.mapping_utils.prims import p41, p06, p08, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.mapping_utils.data_handler import DataHandler
from generator.mapping_utils.print_memory  import print_mem_map
# 导入测试引擎
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch

sys.path.append(os.getcwd())

# ====================== SNN LeNet 模型 ======================
class SNNLeNet(nn.Module):
    def __init__(self, beta=0.9, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.beta = beta

        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=2, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(12, 32, kernel_size=5, stride=2, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(32 * 7 * 7, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x, num_steps=25):
        batch_size = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            flat = spk2.view(batch_size, -1)
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec, dim=0)

# ====================== SNN 数据处理器 ======================
class SNNDataHandler(DataHandler):
    def __init__(self, model, ckpt_path=r'E:\\Tsinghua\\tianji\\tianjic_x1_software_0814\\best_snn_mnist.pth', seed=7):
        super().__init__()
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading SNN checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("No checkpoint provided, using random initialized weights.")
        # 量化权重和偏置
        self._quantize_parameters()
        self._extract_internal_data()

    def _quantize_parameters(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data.mul_(128).round_().clamp_(-128, 127)
                if module.bias is not None:
                    module.bias.data.mul_(128*128).round_().clamp_(-0x80000000, 0x7fffffff)

    def _extract_internal_data(self):
        self.internal = OrderedDict()
        x = torch.randn(1, 1, 28, 28).to(self.device)
        hooks = []
        def hook_fn(name):
            def hook(module, input, output):
                self.internal[name] = {
                    'input': input[0].detach().cpu().numpy().squeeze(0),
                    'output': output.detach().cpu().numpy().squeeze(0)
                }
            return hook
        for name, module in self.model.named_modules():
            if name in ['conv1', 'conv2', 'fc1']:
                hooks.append(module.register_forward_hook(hook_fn(name)))
        with torch.no_grad():
            _ = self.model(x, num_steps=1)   # 只取一个时间步
        for h in hooks:
            h.remove()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_name = name
                if layer_name in self.internal:
                    self.internal[layer_name]['weight'] = module.weight.detach().cpu().numpy()
                    if module.bias is not None:
                        self.internal[layer_name]['bias'] = module.bias.detach().cpu().numpy()

    @property
    def parameters(self):
        return self.internal

# ====================== SNN 数据生成 ======================
def generate_snn_data(handler):
    data = {
        'conv1': {},
        'conv2': {},
        'fc1': {},
        'input_loader': {},
    }

    # 输入数据 - 放入字典，键为 (0,0)
    raw_input = handler.parameters['conv1']['input'].astype(np.int8)
    input_split = DataHandler.tensor_split(
        raw_data=raw_input,
        split_dict={(0, 0): ((0, 1), (0, 28), (0, 28))},
        data_type=1,
        alignment=(None, None, 16),
        dims=(2, 1, 0)
    )
    data['input_loader']['data'] = { (0,0): input_split[(0,0)] }

    # conv1 权重和偏置
    raw_w1 = handler.parameters['conv1']['weight'].astype(np.int8)
    w1_split = DataHandler.tensor_split(
        raw_data=raw_w1,
        split_dict={(0, 0): ((0, 12), (0, 1), (0, 5), (0, 5))},
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 3, 2, 1),
        is_weight=True
    )
    data['conv1']['weight'] = { (0,0): w1_split[(0,0)] }
    raw_b1 = handler.parameters['conv1']['bias'].astype(np.int32)
    b1_split = DataHandler.tensor_split(
        raw_data=raw_b1,
        split_dict={(0, 0): ((0, 12),)},
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )
    data['conv1']['bias'] = { (0,0): b1_split[(0,0)] }

    # conv2 权重和偏置
    raw_w2 = handler.parameters['conv2']['weight'].astype(np.int8)
    w2_split = DataHandler.tensor_split(
        raw_data=raw_w2,
        split_dict={(0, 0): ((0, 32), (0, 12), (0, 5), (0, 5))},
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )
    data['conv2']['weight'] = { (0,0): w2_split[(0,0)] }
    raw_b2 = handler.parameters['conv2']['bias'].astype(np.int32)
    b2_split = DataHandler.tensor_split(
        raw_data=raw_b2,
        split_dict={(0, 0): ((0, 32),)},
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )
    data['conv2']['bias'] = { (0,0): b2_split[(0,0)] }

    # fc1 权重和偏置
    raw_w3 = handler.parameters['fc1']['weight'].astype(np.int8)
    w3_split = DataHandler.tensor_split(
        raw_data=raw_w3,
        split_dict={(0, 0): ((0, 10), (0, 1568))},
        data_type=1,
        alignment=(32, None),
        dims=(0, 1),
        is_weight=True
    )
    data['fc1']['weight'] = { (0,0): w3_split[(0,0)] }
    raw_b3 = handler.parameters['fc1']['bias'].astype(np.int32)
    b3_split = DataHandler.tensor_split(
        raw_data=raw_b3,
        split_dict={(0, 0): ((0, 10),)},
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )
    data['fc1']['bias'] = { (0,0): b3_split[(0,0)] }

    return data

# ====================== 硬件映射配置生成 ======================
def gen_0_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, out_data_en=False, chip=(0,0), in_data_en=False):
    """
        Group0: FPGA 数据加载
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {0: {'clock': clock_in_phase, 'mode': 1}}
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        map_config[(chip,0)][0][(chip, (core_x, core_y))] = {'prims': []}
    phase_group = map_config[(chip,0)][0]

    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            # 从 data['input_loader']['data'] 字典中获取输入
            input_data = data['input_loader']['data'][(core_x, core_y)] if data else None
            soma1 = p06(addr_in=0x0000>>2, addr_out=0x9000, addr_ciso=0x0000>>2,
                        length_in=32, num_in=28, length_ciso=1, num_ciso=28,
                        length_out=32, num_out=28, type_in=1, type_out=1,
                        data_in=input_data)
            if not out_data_en:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en,
                         send_num=32*28//8-1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=16//8-1,
                         addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0,
                            pack_per_Rhead=32*28//8-1, A_offset=0, Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})
    return map_config

def gen_1_map_config_snn(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict, data=None,
                         in_data_en=False, out_data_en=False, chip=(0,0), init_data=False):
    """ Group1: SNN LeNet 计算核心（包含卷积+LIF）"""
    map_config = {
        'sim_clock': None,
        (chip, 0): {0: {'clock': clock_in_phase, 'mode': 1}}
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        map_config[(chip,0)][0][(chip, (core_x, core_y))] = {'prims': []}
    phase_group = map_config[(chip,0)][0]

    # 数据交互 Phase0
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x2000>>2, addr_out=0x9000, addr_ciso=0x0000>>2,
                            length_in=16, num_in=1, length_ciso=1, num_ciso=1,
                            length_out=16, num_out=1, type_in=1, type_out=1,
                            data_in=None)  # 不从外部加载，从地址读取
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en,
                         send_num=16//8-1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=32*28//8-1,
                         addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0,
                            pack_per_Rhead=16//8-1, A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000>>2, addr_ciso=0x0000>>2,
                            length_in=32, num_in=28, length_ciso=1, num_ciso=28,
                            length_out=32, num_out=28, type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    offset = 1
    # Phase1: conv1 + LIF1 (neuron count = 12*14*14 = 2352)
    if phase_en[offset]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p81(px=28, py=28, cin=1, cout=12, kx=5, ky=5, sx=2, sy=2,
                       addr_ina=0x0000>>2, addr_inb=0x10000>>2, addr_bias=0x10320>>2, addr_out=0x1040>>2,
                       ina_type=1, inb_type=1, load_bias=2,
                       data_x=None if in_data_en else data['input_loader']['data'][(core_x, core_y)],
                       data_w=data['conv1']['weight'][(core_x, core_y)],
                       data_b=data['conv1']['bias'][(core_x, core_y)])
            soma1 = p08(neuron_num=784, group_num=3, row_ck_on=0,
                        seed=0, Vth0=1, Vleaky_adpt_en=False, Vleaky_alpha=0, Vleaky_beta=0, ref_len=0,
                        addr_uin=0x1040>>2, addr_v=0x2000>>2, addr_s=0x1080>>2,
                        addr_vm=0x3000>>2, addr_vtheta=0x4000>>2, addr_para=0x5000>>2,
                        in_cut_start=in_cut_start_dict.get('cut1',0), in_row_max=0,
                        fire_type=4, Tw_en=False, Tw_len=0, Tw_cnt=0, Vinit=0,
                        VR=0, VL=-10, dV=0, Rst_mode=0, Vth_adpt_en=False, Vth_alpha=0, Vth_beta=0, Vth_Incre=0,
                        VM_const_en=True, VM_const=0, VM_len=784,
                        Vtheta_const_en=True, Vtheta_const=0, Vtheta_len=784,
                        ref_cnt_const_en=True, ref_cnt_const=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': None, 'soma2': None})
    # Phase2: conv2 + LIF2 (neuron count = 32*7*7 = 1568)
    if phase_en[offset+1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=14, cin=12, cout=32, kx=5, ky=5, sx=2, sy=2,
                       addr_ina=0x1080>>2, addr_inb=0x103A0>>2, addr_bias=0x11660>>2, addr_out=0x1040>>2,
                       pad_top=2, pad_down=2, pad_left=2, pad_right=2,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['conv2']['weight'][(core_x, core_y)],
                       data_b=data['conv2']['bias'][(core_x, core_y)])
            soma1 = p08(neuron_num=784, group_num=2, row_ck_on=0,
                        seed=0, Vth0=1, Vleaky_adpt_en=False, Vleaky_alpha=0, Vleaky_beta=0, ref_len=0,
                        addr_uin=0x1040>>2, addr_v=0x2000>>2, addr_s=0x1080>>2,
                        addr_vm=0x3000>>2, addr_vtheta=0x4000>>2, addr_para=0x5000>>2,
                        in_cut_start=in_cut_start_dict.get('cut2',0), in_row_max=0,
                        fire_type=4, Tw_en=False, Tw_len=0, Tw_cnt=0, Vinit=0,
                        VR=0, VL=-10, dV=0, Rst_mode=0, Vth_adpt_en=False, Vth_alpha=0, Vth_beta=0, Vth_Incre=0,
                        VM_const_en=True, VM_const=0, VM_len=784,
                        Vtheta_const_en=True, Vtheta_const=0, Vtheta_len=784,
                        ref_cnt_const_en=True, ref_cnt_const=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': None, 'soma2': None})
    # Phase3: fc1 + LIF3 (neuron count = 10)
    if phase_en[offset+2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=1568, cout=10, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x1080>>2, addr_inb=0x116e0>>2, addr_bias=0x1dee0>>2, addr_out=0x1000>>2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['fc1']['weight'][(core_x, core_y)],
                       data_b=data['fc1']['bias'][(core_x, core_y)])
            soma1 = p08(neuron_num=12, group_num=1, row_ck_on=0,
                        seed=0, Vth0=1, Vleaky_adpt_en=False, Vleaky_alpha=0, Vleaky_beta=0, ref_len=0,
                        addr_uin=0x1000>>2, addr_v=0x2000>>2, addr_s=0x2000>>2,
                        addr_vm=0x3000>>2, addr_vtheta=0x4000>>2, addr_para=0x5000>>2,
                        in_cut_start=in_cut_start_dict.get('fc_cut3',0), in_row_max=0,
                        fire_type=4, Tw_en=False, Tw_len=0, Tw_cnt=0, Vinit=0,
                        VR=0, VL=-10, dV=0, Rst_mode=0, Vth_adpt_en=False, Vth_alpha=0, Vth_beta=0, Vth_Incre=0,
                        VM_const_en=True, VM_const=0, VM_len=12,
                        Vtheta_const_en=True, Vtheta_const=0, Vtheta_len=12,
                        ref_cnt_const_en=True, ref_cnt_const=0)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': None, 'soma2': None})
    return map_config

def gen_2_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, in_data_en=False, out_data_en=False,
                     chip=(0,0), init_data=False, send_to_fpga=False):
    """
        Group2: 输出收集与发送
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {0: {'clock': clock_in_phase, 'mode': 1}}
    }
    for core_y, core_x in product(range(size_y), range(size_x)):
        map_config[(chip,0)][0][(chip, (core_x, core_y))] = {'prims': []}
    phase_group = map_config[(chip,0)][0]

    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x0000>>2, addr_out=0x9000, addr_ciso=0x0000>>2,
                        length_in=16, num_in=1, length_ciso=1, num_ciso=1,
                        length_out=16, num_out=1, type_in=1, type_out=1,
                        data_in=None)  # 不从外部加载，从地址读取
            if not out_data_en and core_x == 7:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en or core_x != 7,
                         receive_en=in_data_en or core_x != 0, send_num=16//8-1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=16//8-1,
                         addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            if core_x == 7:
                if not send_to_fpga:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=1-core_x, Y=-1-core_y, A=0,
                                    pack_per_Rhead=16//8-1, A_offset=0, Const=0, EN=1)
                else:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-128, A=0,
                                    pack_per_Rhead=16//8-1, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0,
                                pack_per_Rhead=16//8-1, A_offset=0, Const=0, EN=1)
            soma2 = None
            if in_data_en or core_x != 0:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000>>2, addr_ciso=0x0000>>2,
                            length_in=16, num_in=1, length_ciso=1, num_ciso=1,
                            length_out=16, num_out=1, type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})
    return map_config

# ====================== 主程序 ======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate hardware mapping for SNN LeNet")
    parser.add_argument('--ckpt', type=str, default=None,
                        help="Path to pretrained weights (optional)")
    args = parser.parse_args()

    case_file_name = 'SNN_LeNet_003'
    chip = (0, 0)
    send_to_fpga = True
    clock_in_phase = 200_000

    # 使能所有 phase (0..5)
    phase = np.ones(50, dtype=int)
    phase[:] = 1

    config = MapConfigGen()

    print("=== SNN LeNet Hardware Mapping ===")
    model = SNNLeNet()
    handler = SNNDataHandler(model, ckpt_path=args.ckpt)
    data = generate_snn_data(handler)
    in_cut_start_dict = {'cut1': 2, 'cut2': 2, 'fc_cut3': 2}

    config_0 = gen_0_map_config(phase, clock_in_phase, size_x=1, size_y=1, data=data,
                                out_data_en=True, chip=(0,0), in_data_en=not send_to_fpga)
    config.add_config(config_0, core_offset=(1, 0))

    config_1 = gen_1_map_config_snn(phase, clock_in_phase, size_x=1, size_y=1,
                                    in_cut_start_dict=in_cut_start_dict, data=data,
                                    in_data_en=True, out_data_en=True, chip=chip, init_data=True)
    config.add_config(config_1, core_offset=(0, 0))

    config_2 = gen_2_map_config(phase, clock_in_phase, size_x=8, size_y=1, data=data,
                                in_data_en=True, out_data_en=True, chip=(0,0),
                                init_data=False, send_to_fpga=send_to_fpga)
    config.add_config(config_2, core_offset=(0, 1))

    # 添加全局 router 信息
    MapConfigGen.add_router_info(map_config=config.map_config)

    # 添加一个额外的 p06 原语到所有 core 的开始
    prim_extra = {
        'axon': None, 'soma1': None, 'router': None,
        'soma2': p06(addr_in=0x0000>>2, addr_out=0x8400, addr_ciso=0x10000>>2,
                     length_in=1024, num_in=12, length_ciso=1, num_ciso=12,
                     length_out=1024, num_out=12, type_in=1, type_out=1, data_in=None)
    }
    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim_extra)

    # 设置时钟参数
    config.map_config['sim_clock'] = 100_000
    config.map_config['step_clock'] = {((0,0),0): (100_000-1, 100_000)}

    # 输出目录
    out_files_path = os.path.join(os.getcwd(), "simulator", "Out_files", case_file_name)
    os.makedirs(out_files_path, exist_ok=True)
    # 清理旧的 cmp_out 目录
    cmp_out_path = os.path.join(out_files_path, "cmp_out")
    if os.path.exists(cmp_out_path):
        os.system(f"rm -rf {cmp_out_path}")

    # 测试配置
    test_config = {
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config.map_config, test_config)
    assert tester.run_test(), "Hardware simulation failed!"
    print("Successfully generated and tested mapping for SNN mode.")
