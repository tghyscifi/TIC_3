import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41, p02
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_5chips.G7_data import generate_g7_data
from generator.resnet50.data_handler import ResNetDataHandler
from itertools import product


def gen_g7_map_config(phase_en, clock_in_phase, size_x, size_y, chip, cuts, static_data=None,
                      in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, init_data=None):
    """
        ResNet-50 5-Chip Group6
        core array : 2 * 14
    """
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

    # ********************* 组间数据传输 *******************************
    offset = 2

    # 接收1 * 14 * 512  发送1 * 14 * 512
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x9400 >> 2,
                       addr_inb=0x9400 >> 2, addr_bias=0x9400 >> 2, addr_out=0x9400 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x1c800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=512,
                            num_in=14, length_ciso=1, num_ciso=14, length_out=512, num_out=14, type_in=1, type_out=1,
                            data_in=static_data['g7_add_output1'][(core_x, core_y)] if init_data else None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=895, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=895, A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x19000 >> 2, addr_ciso=0x0000 >> 2, length_in=512,
                            num_in=14, length_ciso=1, num_ciso=14, length_out=512, num_out=14,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 接收1 * 14 * 512  发送1 * 14 * 512
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x1e400 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=512,
                            num_in=14, length_ciso=1, num_ciso=14, length_out=512, num_out=14, type_in=1, type_out=1,
                            data_in=static_data['g7_add_output2'][(core_x, core_y)] if init_data else None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=895, receive_num=0,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=895, A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x1AC00 >> 2, addr_ciso=0x0000 >> 2, length_in=512,
                            num_in=14, length_ciso=1, num_ciso=14, length_out=512, num_out=14,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # ************************ 计算部分 **************************

    # Convc3c1 + Relu
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if static_data is None:
                data_x = []
                data_w = []
                data_b = []
            else:
                data_x = static_data['conv3c1_input'][(core_x, core_y)]
                data_w = static_data['conv3c1_weight'][(core_x, core_y)]
                data_b = static_data['conv3c1_bias'][(core_x, core_y)]
            axon = p41(px=28, py=1, cin=512, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x4000 >> 2, addr_out=0x1C800 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=data_x, data_w=data_w, data_b=data_b)
            if core_x < 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8400 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
            elif core_x >= 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8780 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
            elif core_x < 7 and core_y == 1:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8B00 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
            else:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8E80 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)

            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2})
    # 权重发送1
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
    # 权重发送2
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
    # 第二次卷积                                                                      
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=28, py=1, cin=512, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x4000 >> 2, addr_out=0x1C800 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=None, data_w=None, data_b=None)
            if core_x < 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8E80 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = None
            elif core_x >= 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8400 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8400 >> 2, addr_out=0x9200 >> 2, addr_ciso=0x8780 >> 2, length_in=32,
                            num_in=28, length_ciso=32, num_ciso=28, length_out=64, num_out=28,
                            type_in=1, type_out=1)
            elif core_x < 7 and core_y == 1:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8780 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8780 >> 2, addr_out=0x9200 >> 2, addr_ciso=0x8B00 >> 2, length_in=32,
                            num_in=28, length_ciso=32, num_ciso=28, length_out=64, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8B00 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8B00 >> 2, addr_out=0x9200 >> 2, addr_ciso=0x8E80 >> 2, length_in=32,
                            num_in=28, length_ciso=32, num_ciso=28, length_out=64, num_out=28,
                            type_in=1, type_out=1)
            router = None

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2})

    # 权重发送1                                                                 
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
    # 权重发送2
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
    # 第三次卷积
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=28, py=1, cin=512, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x4000 >> 2, addr_out=0x1C800 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=None, data_w=None, data_b=None)
            if core_x < 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8B00 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8B00 >> 2, addr_out=0x9200 >> 2, addr_ciso=0x8E80 >> 2, length_in=32,
                            num_in=28, length_ciso=32, num_ciso=28, length_out=64, num_out=28,
                            type_in=1, type_out=1)
            elif core_x >= 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8E80 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = None
            elif core_x < 7 and core_y == 1:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8400 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8400 >> 2, addr_out=0x9900 >> 2, addr_ciso=0x9200 >> 2, length_in=32,
                            num_in=28, length_ciso=64, num_ciso=28, length_out=96, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8780 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8780 >> 2, addr_out=0x9900 >> 2, addr_ciso=0x9200 >> 2, length_in=32,
                            num_in=28, length_ciso=64, num_ciso=28, length_out=96, num_out=28,
                            type_in=1, type_out=1)
            router = None

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2})

    # 权重发送1
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
    # 权重发送2
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                # 0x24000是MEM3地址
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
    # 第四次卷积
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=28, py=1, cin=512, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x4000 >> 2, addr_out=0x1C800 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=None, data_w=None, data_b=None)
            if core_x < 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8780 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8400 >> 2, addr_out=0x9900 >> 2, addr_ciso=0x8780 >> 2, length_in=32,
                            num_in=28, length_ciso=32, num_ciso=28, length_out=64, num_out=28,
                            type_in=1, type_out=1)
            elif core_x >= 7 and core_y == 0:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8B00 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8B00 >> 2, addr_out=0x9900 >> 2, addr_ciso=0x8E80 >> 2, length_in=32,
                            num_in=28, length_ciso=32, num_ciso=28, length_out=64, num_out=28,
                            type_in=1, type_out=1)
            elif core_x < 7 and core_y == 1:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8E80 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x9900 >> 2, addr_out=0x1C800 >> 2, addr_ciso=0x8E80 >> 2, length_in=96,
                            num_in=28, length_ciso=32, num_ciso=28, length_out=128, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x8400 >> 2, cin=32, cout=32, px=28, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut1'],
                            row_ck_on=1, in_row_max=1)
                soma2 = p06(addr_in=0x8400 >> 2, addr_out=0x1C800 >> 2, addr_ciso=0x9900 >> 2, length_in=32,
                            num_in=28, length_ciso=96, num_ciso=28, length_out=128, num_out=28,
                            type_in=1, type_out=1)
            router = None

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2})

    if phase_en[offset + 10]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7 and core_y == 0:
                if core_x == 0:
                    soma1 = p06(addr_in=0x9900 >> 2, addr_out=0xA380 >> 2, addr_ciso=0x9200 >> 2, length_in=64,
                                num_in=28, length_ciso=64, num_ciso=28, length_out=128, num_out=28,
                                type_in=1, type_out=1)
                else:
                    soma1 = p06(addr_in=0x9900 >> 2, addr_out=0x1C800 >> 2, addr_ciso=0x9200 >> 2, length_in=64,
                                num_in=28, length_ciso=64, num_ciso=28, length_out=128, num_out=28,
                                type_in=1, type_out=1)
            elif core_x >= 7 and core_y == 0:
                soma1 = p06(addr_in=0x9200 >> 2, addr_out=0x1C800 >> 2, addr_ciso=0x9900 >> 2, length_in=64,
                            num_in=28, length_ciso=64, num_ciso=28, length_out=128, num_out=28,
                            type_in=1, type_out=1)
            elif core_x < 7 and core_y == 1:
                soma1 = None
            else:
                soma1 = None
            axon = None
            soma2 = None
            router = None

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2})

    # 权重发送1
    if phase_en[offset + 11]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1,
                            type_out=1)  # 0x24000是MEM3地址
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1,
                            type_out=1)  # 0x24000是MEM3地址
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16, type_in=1,
                            type_out=1)  # 0x24000是MEM3地址
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1024 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1024 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1024 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=512,
                            num_in=16, length_ciso=1, num_ciso=16, length_out=512, num_out=16,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
    # 权重发送2
    if phase_en[offset + 12]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 7:
                axon = None
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1,
                            type_out=1)  # 0x24000是MEM3地址
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif core_y == 0 and core_x >= 7:
                axon = None
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1,
                            type_out=1)  # 0x24000是MEM3地址
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=51, length_out=128, num_out=65, type_in=1,
                            type_out=1)  # 0x24000是MEM3地址
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1040 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=1040 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=-1, A=0, pack_per_Rhead=1040 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x2000 >> 2, addr_ciso=0x10000 >> 2, length_in=128,
                            num_in=65, length_ciso=1, num_ciso=65, length_out=128, num_out=65,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})

    # 发送数据
    if phase_en[offset + 13]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x + core_y * size_x) == 0:
                axon = None
                soma1 = None
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=0,
                             data_in=None)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xB180 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=28 * 128, num_out=1,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif (core_x + core_y * size_x) in [1, 3, 4, 5]:
                axon = None
                soma1 = p06(addr_in=0x1C800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=28 * 128, num_out=1,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=2,
                             addr_din_base=0x1000 >> 2, addr_din_length=1344 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                x_dict = {1: -1, 3: -2, 4: -3, 5: -4}
                a_dict = {1: 0, 3: 0, 4: 448, 5: 896}
                router.addRHead(S=0, T=1, P=0, Q=0, X=x_dict[core_x + core_y * size_x], Y=0,
                                A=a_dict[core_x + core_y * size_x],
                                pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x9580 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=28 * 128, num_out=3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif (core_x + core_y * size_x) in [2, 6]:
                axon = None
                soma1 = None
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=2,
                             addr_din_base=0x1000 >> 2, addr_din_length=1344 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=0,
                             data_in=None)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x9580 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=28 * 128, num_out=3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif (core_x + core_y * size_x) in [10, 14, 18, 22, 26, 27]:
                axon = None
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x1C800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=28 * 128, num_out=1,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                x_dict = {7: -5, 8: -6, 9: -7, 11: -8, 12: -9, 13: -10, 15: 3, 16: 2,
                          17: 1, 19: 0, 20: -1, 21: -2, 23: -3, 24: -4, 25: -5}
                a_dict = {7: 0, 8: 448, 9: 896, 11: 0, 12: 448, 13: 896, 15: 0, 16: 448,
                          17: 896, 19: 0, 20: 448, 21: 896, 23: 0, 24: 448, 25: 896}
                router.addRHead(S=0, T=1, P=0, Q=0, X=x_dict[core_x + core_y * size_x],
                                Y=0 if (core_x + core_y * size_x) <= 13 else -1,
                                A=a_dict[core_x + core_y * size_x],
                                pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})

    # 多播
    if phase_en[offset + 14]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if 0 <= (core_x + core_y * size_x) <= 6:
                if (core_x + core_y * size_x) == 0:
                    addr_in = 0xA380 >> 2
                    send_num = 896 - 1
                else:
                    addr_in = 0x9580 >> 2
                    send_num = 1344 - 1
                axon = None
                soma1 = p06(addr_in=addr_in, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=2 if (core_x + core_y * size_x) == 0 else 3, length_ciso=1,
                            num_ciso=2 if (core_x + core_y * size_x) == 0 else 3, length_out=28 * 128,
                            num_out=2 if (core_x + core_y * size_x) == 0 else 3,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=send_num, receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=1, X=7, Y=0, A=0, pack_per_Rhead=send_num, A_offset=0, Const=0, EN=1)
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                soma1 = None
                if 7 <= (core_x + core_y * size_x) <= 13:
                    cxy = 1
                    nx = 0
                    ny = 1
                elif 14 <= (core_x + core_y * size_x) <= 20:
                    cxy = 0
                    nx = 0
                    ny = 0
                else:
                    cxy = 1
                    nx = -7
                    ny = 0
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                             addr_din_base=0x1000 >> 2,
                             addr_din_length=(896 - 1) if (core_x + core_y * size_x) % 7 == 0 else (1344 - 1),
                             addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=0,
                             data_in=None, cxy=cxy, nx=nx, ny=ny,
                             relay_num=(896 - 1) if (core_x + core_y * size_x) % 7 == 0 else (1344 - 1))
                addr_out = 0xA380 >> 2 if (core_x + core_y * size_x) % 7 == 0 else 0x9580 >> 2
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=addr_out, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=2 if (core_x + core_y * size_x) % 7 == 0 else 3, length_ciso=1,
                            num_ciso=2 if (core_x + core_y * size_x) % 7 == 0 else 3, length_out=28 * 128,
                            num_out=2 if (core_x + core_y * size_x) % 7 == 0 else 3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})

    # 发送数据
    if phase_en[offset + 15]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x + core_y * size_x) in [0, 1, 5]:
                axon = None
                soma1 = None
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=2,
                             addr_din_base=0x1000 >> 2, addr_din_length=(1344 - 1),
                             addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=0,
                             data_in=None)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xBF80 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=28 * 128, num_out=3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif (core_x + core_y * size_x) in [2, 3, 4, 6]:
                axon = None
                soma1 = p06(addr_in=0x1C800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=28 * 128, num_out=1,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1,
                             receive_num=1 if (core_x + core_y * size_x) == 6 else 2,
                             addr_din_base=0x1000 >> 2,
                             addr_din_length=(896 - 1) if (core_x + core_y * size_x) == 0 else (1344 - 1),
                             addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                x_dict = {2: -2, 3: -3, 4: -4, 6: -5}
                a_dict = {2: 0, 3: 448, 4: 896, 6: 0}
                router.addRHead(S=0, T=1, P=0, Q=0, X=x_dict[core_x + core_y * size_x], Y=0,
                                A=a_dict[core_x + core_y * size_x],
                                pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xBF80 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=2 if (core_x + core_y * size_x) == 6 else 3, length_ciso=1,
                            num_ciso=2 if (core_x + core_y * size_x) == 6 else 3, length_out=28 * 128,
                            num_out=2 if (core_x + core_y * size_x) == 6 else 3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            elif (core_x + core_y * size_x) in [9, 13, 17, 21, 25]:
                axon = None
                soma1 = None
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x1C800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=28 * 128, num_out=1,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                x_dict = {7: -6, 8: -7, 10: -8, 11: -9, 12: -10, 14: 3, 15: 2, 16: 1,
                          18: 0, 19: -1, 20: -2, 22: -3, 23: -4, 24: -5, 26: -6, 27: -7}
                a_dict = {7: 448, 8: 896, 10: 0, 11: 448, 12: 896, 14: 0, 15: 448, 16: 896,
                          18: 0, 19: 448, 20: 896, 22: 0, 23: 448, 24: 896, 26: 0, 27: 448}
                router.addRHead(S=0, T=1, P=0, Q=0, X=x_dict[core_x + core_y * size_x],
                                Y=0 if (core_x + core_y * size_x) <= 13 else -1,
                                A=a_dict[core_x + core_y * size_x],
                                pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})

    # 多播
    if phase_en[offset + 16]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if 0 <= (core_x + core_y * size_x) <= 6:
                axon = None
                soma1 = p06(addr_in=0xBF80 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=2 if (core_x + core_y * size_x) == 6 else 3, length_ciso=1,
                            num_ciso=2 if (core_x + core_y * size_x) == 6 else 3, length_out=28 * 128,
                            num_out=2 if (core_x + core_y * size_x) == 6 else 3,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0,
                             send_num=(896 - 1) if (core_x + core_y * size_x) == 6 else (1344 - 1), receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=1, X=7, Y=0, A=0,
                                pack_per_Rhead=(896 - 1) if (core_x + core_y * size_x) == 6 else (1344 - 1),
                                A_offset=0, Const=0, EN=1)
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})
            else:
                axon = None
                soma1 = None
                if 7 <= (core_x + core_y * size_x) <= 13:
                    cxy = 1
                    nx = 0
                    ny = 1
                elif 14 <= (core_x + core_y * size_x) <= 20:
                    cxy = 0
                    nx = 0
                    ny = 0
                else:
                    cxy = 1
                    nx = -7
                    ny = 0
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0,
                             receive_num=0, addr_din_length=(896 - 1) if (core_x + 1) % 7 == 0 else (1344 - 1),
                             addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=0,
                             data_in=None, cxy=cxy, nx=nx, ny=ny,
                             relay_num=(896 - 1) if (core_x + core_y * size_x) in [13, 27] else (1344 - 1))
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xBF80 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=2 if (core_x + 1) % 7 == 0 else 3, length_ciso=1,
                            num_ciso=2 if (core_x + 1) % 7 == 0 else 3, length_out=28 * 128,
                            num_out=2 if (core_x + 1) % 7 == 0 else 3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})

    if phase_en[offset + 17]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x % 7 == 0:
                py = 5
                pad_top = 1
                pad_down = 0
                addr_ina = 0xA380 >> 2
            elif (core_x + 1) % 7 == 0:
                py = 5
                pad_top = 0
                pad_down = 1
                addr_ina = 0x9580 >> 2
            else:
                pad_top = 0
                pad_down = 0
                py = 6
                addr_ina = 0x9580 >> 2
            data_w = static_data['conv3c2_weight'][(core_x, core_y)]
            data_b = static_data['conv3c2_bias'][(core_x, core_y)]
            axon = p41(px=28, py=py, cin=128, cout=32, kx=3, ky=3, sx=1, sy=1, addr_ina=addr_ina,
                       addr_inb=0x10000 >> 2, addr_bias=0xFF80 >> 2, addr_out=0x1C800 >> 2, ina_type=1, inb_type=1,
                       pad_left=1, pad_right=1, pad_top=pad_top, pad_down=pad_down,
                       load_bias=2, data_x=None, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x1C800 >> 2, addr_out=0x1E400 >> 2, cin=32, cout=32, px=28, py=4, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut2'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2})

    # 通道方向整理
    if phase_en[offset + 18]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x // 7) % 2 == 0:
                axon = None
                soma1 = p06(addr_in=0x1E400 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 32,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 32, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x1E400 >> 2, addr_out=0x8400 >> 2, addr_ciso=0x21000 >> 2, length_in=32,
                            num_in=4 * 28, length_ciso=32, num_ciso=4 * 28, length_out=64, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x1E400 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 32,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 32, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x8400 >> 2, addr_ciso=0x1E400 >> 2, length_in=32,
                            num_in=4 * 28, length_ciso=32, num_ciso=4 * 28, length_out=64, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    if phase_en[offset + 19]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 0:
                axon = None
                soma1 = p06(addr_in=0x8400 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 64,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 64, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=896 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x8400 >> 2, addr_out=0x1C800 >> 2, addr_ciso=0x21000 >> 2, length_in=64,
                            num_in=4 * 28, length_ciso=64, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x8400 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 64,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 64, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=896 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x1C800 >> 2, addr_ciso=0x8400 >> 2, length_in=64,
                            num_in=4 * 28, length_ciso=64, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    if phase_en[offset + 20]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            data_w = static_data['conv3c3_weight'][(core_x, core_y)]
            data_b = static_data['conv3c3_bias'][(core_x, core_y)]
            axon = p41(px=28, py=4, cin=128, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1C800 >> 2,
                       addr_inb=0x4200 >> 2, addr_bias=0x8200 >> 2, addr_out=0x8400 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=None, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x8400 >> 2, addr_out=0xBC00 >> 2, cin=128, cout=128, px=28, py=4, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut3'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2})

    router_dict = {
        0: ((0, 0), (1, 0), (2, 0), (3, 0)),
        7: ((-7, 0), (-6, 0), (-5, 0), (-4, 0)),
        14: ((0, -1), (1, -1), (2, -1), (3, -1)),
        21: ((-7, -1), (-6, -1), (-5, -1), (-4, -1)),
        1: ((3, 0), (4, 0), (5, 0), (6, 0)),
        8: ((-4, 0), (-3, 0), (-2, 0), (-1, 0)),
        15: ((3, -1), (4, -1), (5, -1), (6, -1)),
        22: ((-4, -1), (-3, -1), (-2, -1), (-1, -1)),
        2: ((6, 0), (7, 0), (8, 0), (9, 0)),
        9: ((-1, 0), (0, 0), (1, 0), (2, 0)),
        16: ((6, -1), (7, -1), (8, -1), (9, -1)),
        23: ((-1, -1), (0, -1), (1, -1), (2, -1)),
        3: ((9, 0), (10, 0), (-3, 1), (-2, 1)),
        10: ((2, 0), (3, 0), (-10, 1), (-9, 1)),
        17: ((9, -1), (10, -1), (-3, 0), (-2, 0)),
        24: ((2, -1), (3, -1), (-10, 0), (-9, 0)),
        4: ((-2, 1), (-1, 1), (0, 1), (1, 1)),
        11: ((-9, 1), (-8, 1), (-7, 1), (-6, 1)),
        18: ((-2, 0), (-1, 0), (0, 0), (1, 0)),
        25: ((-9, 0), (-8, 0), (-7, 0), (-6, 0)),
        5: ((1, 1), (2, 1), (3, 1), (4, 1)),
        12: ((-6, 1), (-5, 1), (-4, 1), (-3, 1)),
        19: ((1, 0), (2, 0), (3, 0), (4, 0)),
        26: ((-6, 0), (-5, 0), (-4, 0), (-3, 0)),
        6: ((4, 1), (5, 1), (6, 1), (7, 1)),
        13: ((-3, 1), (-2, 1), (-1, 1), (0, 1)),
        20: ((4, 0), (5, 0), (6, 0), (7, 0)),
        27: ((-3, 0), (-2, 0), (-1, 0), (0, 0)),
    }

    if phase_en[offset + 21]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0xBC00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                        num_in=4, length_ciso=1, num_ciso=4, length_out=14 * 128, num_out=4, type_in=1, type_out=1)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][0][0],
                            Y=router_dict[core_x + core_y * size_x][0][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][1][0],
                            Y=router_dict[core_x + core_y * size_x][1][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][2][0],
                            Y=router_dict[core_x + core_y * size_x][2][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][3][0],
                            Y=router_dict[core_x + core_y * size_x][3][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x1C800 >> 2, addr_ciso=0x0000 >> 2, length_in=512,
                        num_in=14, length_ciso=1, num_ciso=14, length_out=512, num_out=14,
                        type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 22]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0xC300 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                        num_in=4, length_ciso=1, num_ciso=4, length_out=14 * 128, num_out=4,
                        type_in=1, type_out=1)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][0][0],
                            Y=router_dict[core_x + core_y * size_x][0][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][1][0],
                            Y=router_dict[core_x + core_y * size_x][1][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][2][0],
                            Y=router_dict[core_x + core_y * size_x][2][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            router.addRHead(S=0, T=1, P=0, Q=0,
                            X=router_dict[core_x + core_y * size_x][3][0],
                            Y=router_dict[core_x + core_y * size_x][3][1],
                            A=16 * ((core_x + core_y * size_x) // 7),
                            pack_per_Rhead=224 - 1, A_offset=48, Const=15, EN=1)
            soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x1E400 >> 2, addr_ciso=0x0000 >> 2, length_in=512,
                        num_in=14, length_ciso=1, num_ciso=14, length_out=512, num_out=14,
                        type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Add
    if phase_en[offset + 23]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=512, px=1, py=28,
                       addr_in=0x19000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x8400 >> 2, data_x=None)
            soma1 = pX5(mode='max', addr_in=0x8400 >> 2, addr_out=0x1c800 >> 2, cin=512, cout=512, px=1, py=28,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.2.cut5'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    case_file_name = 'R00007'

    phase = np.zeros(50).astype(int)  # 39-49表示组件数据传输的Phase
    cuts = [4, 4, 4, 1]

    # Conv3c1
    phase[0] = 1
    # 数据整理
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1
    phase[4] = 1
    phase[5] = 1
    phase[6] = 1
    phase[7] = 1
    phase[8] = 1
    phase[9] = 1
    phase[10] = 1
    phase[11] = 1
    phase[12] = 1
    phase[13] = 1
    phase[14] = 1
    phase[15] = 1
    phase[16] = 1
    phase[17] = 1
    phase[18] = 1
    phase[19] = 1
    phase[20] = 1
    phase[21] = 1
    phase[22] = 1
    phase[23] = 1

    handler = ResNetDataHandler()
    static_data = generate_g7_data(handler, size_y=2, size_x=14)
    config = gen_g7_map_config(phase, clock_in_phase=100_000, size_x=14, size_y=2, chip=(0, 0), cuts=cuts,
                               static_data=static_data, in_data_en=False, out_data_en=False)
    MapConfigGen.add_router_info(map_config=config, group_idx_list=[0], chip_x_num=1, chip_y_num=1)

    config['sim_clock'] = 300_000

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "/simulator/Out_files/" + case_file_name + "/"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        del_command = 'rd/s/q cmp_out'
        os.system(del_command)
        os.chdir(c_path)

    test_config = {
        'tb_name': case_file_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
