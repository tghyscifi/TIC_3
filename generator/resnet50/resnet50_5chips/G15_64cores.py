import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41, p02
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler


def gen_g15_map_config(phase_en, clock_in_phase, size_x, size_y, cuts, data=None,
                       in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0), init_data=None):
    """
        ResNet-50 5-Chip Group-15
        core array : 4 * 16
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

    # data in
    if data is None:
        data = {
            'layer4.0.conv1': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L41
            'layer4.0.conv2': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L42
            'layer4.0.conv3': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L43
        }

    # ******** 数据交互 ********
    offset = 2

    # 发送 12, 12, 12, 13 * 128  接收 7*7*256
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0xe500 >> 2,
                       addr_inb=0xe500 >> 2, addr_bias=0xe500 >> 2, addr_out=0xe500 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en and (core_y in [0, 1]):
                soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=128, cout=128, px=12, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer4.0.cut5']['output'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_y in [0, 1]),
                         receive_en=in_data_en, send_num=191, receive_num=7,
                         addr_din_base=0x380, addr_din_length=1567, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=1567, nx=0, ny=0, data_in=None)
            if out_data_en and (core_y in [0, 1]):
                if core_x // 4 == 0:
                    dst_x, dst_y = 0, -1
                elif core_x // 4 == 1:
                    dst_x, dst_y = 4, -1
                elif core_x // 4 == 2:
                    dst_x, dst_y = 8, -1
                elif core_x // 4 == 3:
                    dst_x, dst_y = 12, -1
                else:
                    raise ValueError
                a = (core_y * 12 * 512 + core_x % 4 * 128) // 8
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a, pack_per_Rhead=191,
                                A_offset=48, Const=15, EN=1)
            if in_data_en:
                if core_y == 0:
                    if core_x == 15:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 1:
                    if core_x == 15:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 2, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 2:
                    if core_x == 15:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 3, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 3:
                    if core_x == 15:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2, 3]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 4, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                else:
                    raise ValueError
            soma2 = p06(addr_in=0x8380, addr_out=0x1cf00 >> 2, addr_ciso=0, length_in=256, length_out=256,
                        length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 12, 12, 12, 13 * 128
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1, px = None, 0
            if out_data_en and (core_y in [2, 3]):
                px = 12 if core_y == 2 else 13
                soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=128, cout=128, px=px, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer4.0.cut5']['output'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_y in [2, 3]),
                         receive_en=0, send_num=px * 128 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=1599, nx=0, ny=0, data_in=None)
            if out_data_en and (core_y in [2, 3]):
                if core_x // 4 == 0:
                    dst_x, dst_y = 0, -1
                elif core_x // 4 == 1:
                    dst_x, dst_y = 4, -1
                elif core_x // 4 == 2:
                    dst_x, dst_y = 8, -1
                elif core_x // 4 == 3:
                    dst_x, dst_y = 12, -1
                else:
                    raise ValueError
                a = ((core_y - 2) * 12 * 512 + (core_x % 4) * 128) // 8
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=px * 128 // 8 - 1,
                                A_offset=48, Const=15, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********

    # L41 卷积，部分和收发
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=7, py=7, cin=256, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1cf00 >> 2,
                       addr_inb=0x500 >> 2, addr_bias=0x0000 >> 2, addr_out=0x19000 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None if in_data_en else data['layer4.0.conv1']['input'][(core_x, core_y)],
                       data_w=data['layer4.0.conv1']['weight'][(core_x, core_y)],
                       data_b=data['layer4.0.conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=32, cout=32, px=7, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            if core_y in [0]:
                addr_din_length = 831
            else:
                addr_din_length = 767
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=783, receive_num=3, addr_din_base=0x380,
                         addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                length = 13 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                length = 12 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=208, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=192, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=192, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=192, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                length = 12 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=416, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=384, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            else:
                length = 12 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=624, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=576, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=576, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=576, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19000 >> 2, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41 部分和求和，流水ReLU
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0]:
                px, py = 1, 13      # px, py = 1, 13  这个行流水的时候结果有问题， 改为不流水
            else:
                px, py = 4, 3
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=32, px=px, py=py,
                       addr_in=0x19000 >> 2, addr_bias=0x0, addr_out=0xe500 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0xe500 >> 2, addr_out=0xfe60 >> 2, cin=32, cout=32, px=px, py=py, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut1'],
                        row_ck_on=0, in_row_max=3)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41e 卷积，部分和收发 1/3 - 24/49
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=4, py=6, cin=256, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1cf00 >> 2,
                       addr_inb=0x2500 >> 2, addr_bias=0x80 >> 2, addr_out=0x19000 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=data['layer4.0.downsample.0']['weight'][(core_x, core_y)],
                       data_b=data['layer4.0.downsample.0']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=128, cout=128, px=4, py=6, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=3, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19000 >> 2, addr_ciso=0, length_in=6 * 128, length_out=6 * 128,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41e 部分和求和，流水Cut， 1/3 - 24/49
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=128, px=6, py=1,
                       addr_in=0x19000 >> 2, addr_bias=0x0, addr_out=0xe500 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xe500 >> 2, addr_out=0x19000 >> 2, cin=128, cout=128, px=6, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut4'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41e 卷积，部分和收发 2/3 - 24/49
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=4, py=6, cin=256, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1e700 >> 2,
                       addr_inb=0x2500 >> 2, addr_bias=0x80 >> 2, addr_out=0x19300 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=None,
                       data_b=None)
            soma1 = pX5(mode='max', addr_in=0x19300 >> 2, addr_out=0x9000, cin=128, cout=128, px=4, py=6, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=3, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19300 >> 2, addr_ciso=0, length_in=6 * 128, length_out=6 * 128,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41e 部分和求和，流水Cut， 2/3 - 24/49
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=128, px=6, py=1,
                       addr_in=0x19300 >> 2, addr_bias=0x0, addr_out=0xe500 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xe500 >> 2, addr_out=0x19300 >> 2, cin=128, cout=128, px=6, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut4'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41e 卷积，部分和收发 3/3 - 1/49
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=256, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1ff00 >> 2,
                       addr_inb=0x2500 >> 2, addr_bias=0x80 >> 2, addr_out=0x19600 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=None,
                       data_b=None)
            soma1 = pX5(mode='max', addr_in=0x19600 >> 2, addr_out=0x9000, cin=128, cout=128, px=1, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=1)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=63, receive_num=3, addr_din_base=0x380,
                         addr_din_length=255, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.Receive_en = 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=64, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=128, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=192, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19600 >> 2, addr_ciso=0, length_in=1 * 128, length_out=1 * 128,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41e 部分和求和，流水Cut， 3/3 - 1/49
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=128, px=1, py=1,
                       addr_in=0x19600 >> 2, addr_bias=0x0, addr_out=0xe500 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xe500 >> 2, addr_out=0x19600 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut4'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41 数据整理，发送
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            px = 13 if core_y == 0 else 12
            soma1 = pX5(mode='max', addr_in=0xfe60 >> 2, addr_out=0x9000, cin=32, cout=32, px=px, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=px * 32 // 8 - 1, receive_num=15,
                         addr_din_base=0x380, addr_din_length=783, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            if (core_x, core_y) in [(0, 0), (4, 1), (8, 2), (12, 3)]:
                router.Receive_en = 1
            if core_x in [0, 1, 2, 3]:
                dst_x, dst_y = 0, 0
            elif core_x in [4, 5, 6, 7]:
                dst_x, dst_y = 4, 1
            elif core_x in [8, 9, 10, 11]:
                dst_x, dst_y = 8, 2
            elif core_x in [12, 13, 14, 15]:
                dst_x, dst_y = 12, 3
            else:
                raise ValueError
            if core_y == 0:
                a = (0 + (core_x - (core_x // 4 * 4)) * 32) // 8
            else:
                a = (13 * 128 + (core_y - 1) * 12 * 128 + (core_x - (core_x // 4 * 4)) * 32) // 8
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y,
                            A=a, pack_per_Rhead=px * 32 // 8 - 1, A_offset=12, Const=3, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L41 整理数据 -- 横向多播
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=783, receive_num=0, addr_din_base=0x380,
                         addr_din_length=783, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x380, addr_dout_length=784 // 2 - 1, soma_in_en=0, cxy=1, relay_num=783,
                         nx=1, ny=0, data_in=None)
            if (core_x, core_y) in [(0, 0), (4, 1), (8, 2), (12, 3)]:
                router.Send_en, router.Receive_en, router.CXY = 1, 0, 0
            if (core_x, core_y) in [(15, 0), (15, 1), (15, 2), (0, 3)]:
                router.CXY = 0
            if core_y == 0:
                router.Nx = 1
            elif core_y == 1:
                if core_x in [1, 2, 3]:
                    router.Nx = -1
                elif core_x == 0:
                    router.Nx = 5
                else:
                    router.Nx = 1
            elif core_y == 2:
                if core_x in [1, 2, 3, 4, 5, 6, 7]:
                    router.Nx = -1
                elif core_x == 0:
                    router.Nx = 9
                else:
                    router.Nx = 1
            elif core_y == 3:
                if core_x in [13, 14]:
                    router.Nx = 1
                elif core_x == 15:
                    router.Nx = -4
                else:
                    router.Nx = -1
            router.addRHead(S=0, T=1, P=0, Q=1, X=1 if core_y in [0, 3] else -1, Y=0,
                            A=0, pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0xe500 >> 2, addr_ciso=0x0000, length_in=7 * 128, length_out=7 * 128,
                        length_ciso=1, num_in=7, num_ciso=7, num_out=7, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L42 卷积，部分和收发
    if phase_en[offset + 10]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=7, py=7, cin=128, cout=32, kx=3, ky=3, sx=1, sy=1, addr_ina=0xe500 >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x280 >> 2, addr_out=0x19680 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=1, pad_down=1, pad_left=1, pad_right=1,
                       data_x=None,
                       data_w=data['layer4.0.conv2']['weight'][(core_x, core_y)],
                       data_b=data['layer4.0.conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19680 >> 2, addr_out=0x9000, cin=32, cout=32, px=7, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            if core_y in [0]:
                addr_din_length = 831
            else:
                addr_din_length = 767
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=783, receive_num=3, addr_din_base=0x380,
                         addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                length = 13 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                length = 12 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=208, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=192, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=192, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=192, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                length = 12 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=416, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=384, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            else:
                length = 12 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=624, pack_per_Rhead=207, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=576, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=576, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=576, pack_per_Rhead=191, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19680 >> 2, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L42 部分和求和，流水ReLU，发送
    if phase_en[offset + 11]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # 改为不行流水
            if core_y in [0]:
                px, py = 1, 13
            else:
                px, py = 4, 3
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=32, px=px, py=py,
                       addr_in=0x19680 >> 2, addr_bias=0x0, addr_out=0xe500 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0xe500 >> 2, addr_out=0x19680 >> 2, cin=32, cout=32, px=px, py=py, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut2'],
                        row_ck_on=0, in_row_max=3)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L42 部分和求和，流水ReLU，发送
    if phase_en[offset + 12]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # 改为不行流水
            if core_y in [0]:
                px, py = 1, 13
            else:
                px, py = 4, 3
            axon = None
            soma1 = pX5(mode='max', addr_in=0x19680 >> 2, addr_out=0x9000, cin=32, cout=32, px=px, py=py, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=px * py * 32 // 8 - 1, receive_num=15,
                         addr_din_base=0x380, addr_din_length=783, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            if (core_x, core_y) in [(0, 0), (4, 1), (8, 2), (12, 3)]:
                router.Receive_en = 1
            if core_x in [0, 1, 2, 3]:
                dst_x, dst_y = 0, 0
            elif core_x in [4, 5, 6, 7]:
                dst_x, dst_y = 4, 1
            elif core_x in [8, 9, 10, 11]:
                dst_x, dst_y = 8, 2
            elif core_x in [12, 13, 14, 15]:
                dst_x, dst_y = 12, 3
            else:
                raise ValueError
            if core_y == 0:
                a = (0 + (core_x - (core_x // 4 * 4)) * 32) // 8
            else:
                a = (13 * 128 + (core_y - 1) * 12 * 128 + (core_x - (core_x // 4 * 4)) * 32) // 8
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y,
                            A=a, pack_per_Rhead=px * py * 32 // 8 - 1, A_offset=12, Const=3, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L42 整理数据 -- 横向多播
    if phase_en[offset + 13]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=783, receive_num=0, addr_din_base=0x380,
                         addr_din_length=783, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x380, addr_dout_length=(784 // 2) - 1, soma_in_en=0, cxy=1, relay_num=783,
                         nx=0, ny=0, data_in=None)
            if (core_x, core_y) in [(0, 0), (4, 1), (8, 2), (12, 3)]:
                router.Send_en, router.Receive_en, router.CXY = 1, 0, 0
            if (core_x, core_y) in [(15, 0), (15, 1), (15, 2), (0, 3)]:
                router.CXY = 0
            if core_y == 0:
                router.Nx = 1
            elif core_y == 1:
                if core_x in [1, 2, 3]:
                    router.Nx = -1
                elif core_x == 0:
                    router.Nx = 5
                else:
                    router.Nx = 1
            elif core_y == 2:
                if core_x in [1, 2, 3, 4, 5, 6, 7]:
                    router.Nx = -1
                elif core_x == 0:
                    router.Nx = 9
                else:
                    router.Nx = 1
            elif core_y == 3:
                if core_x in [13, 14]:
                    router.Nx = 1
                elif core_x == 15:
                    router.Nx = -4
                else:
                    router.Nx = -1
            router.addRHead(S=0, T=1, P=0, Q=1, X=1 if core_y in [0, 3] else -1, Y=0,
                            A=0, pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x1e780 >> 2, addr_ciso=0x0000, length_in=7 * 128, length_out=7 * 128,
                        length_ciso=1, num_in=7, num_ciso=7, num_out=7, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Ye 进行整理，每个得到13*128， 12*128， 12*128， 13*128
    if phase_en[offset + 14]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_y in [0]:
                px = 13
            else:
                px = 12
            soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=128, cout=128, px=px, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                        in_row_max=3, data_in=None)
            if core_y == 3:
                addr_din_length = 13 * 128 // 8 - 1
                receive_num = 2
            else:
                addr_din_length = 12 * 128 // 8 - 1
                receive_num = 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=px * 128 // 8 - 1, receive_num=receive_num,
                         addr_din_base=0x380, addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=192, pack_per_Rhead=15, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
            if core_y == 3:
                length = 13 * 128
            else:
                length = 12 * 128
            soma2 = p06(addr_in=0x8380, addr_out=0xe500 >> 2, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L43 卷积，部分和收发 1/3 - 24/49
    if phase_en[offset + 15]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=4, py=6, cin=128, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1e780 >> 2,
                       addr_inb=0xa500 >> 2, addr_bias=0x300 >> 2, addr_out=0x19000 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=data['layer4.0.conv3']['weight'][(core_x, core_y)],
                       data_b=data['layer4.0.conv3']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=128, cout=128, px=4, py=6, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=3, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19000 >> 2, addr_ciso=0, length_in=6 * 128, length_out=6 * 128,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L43 部分和求和，流水Cut， 1/3 - 24/49
    if phase_en[offset + 16]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=128, px=6, py=1,
                       addr_in=0x19000 >> 2, addr_bias=0x0, addr_out=0xeb80 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xeb80 >> 2, addr_out=0x19000 >> 2, cin=128, cout=128, px=6, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut3'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L43 卷积，部分和收发 2/3 - 24/49
    if phase_en[offset + 17]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=4, py=6, cin=128, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1f380 >> 2,
                       addr_inb=0xa500 >> 2, addr_bias=0x300 >> 2, addr_out=0x19300 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=None,
                       data_b=None)
            soma1 = pX5(mode='max', addr_in=0x19300 >> 2, addr_out=0x9000, cin=128, cout=128, px=4, py=6, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=3, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19300 >> 2, addr_ciso=0, length_in=6 * 128, length_out=6 * 128,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L43 部分和求和，流水Cut， 2/3 - 24/49
    if phase_en[offset + 18]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=128, px=6, py=1,
                       addr_in=0x19300 >> 2, addr_bias=0x0, addr_out=0xeb80 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xeb80 >> 2, addr_out=0x19300 >> 2, cin=128, cout=128, px=6, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut3'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L43 卷积，部分和收发 3/3 - 1/49
    if phase_en[offset + 19]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=128, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1ff80 >> 2,
                       addr_inb=0xa500 >> 2, addr_bias=0x300 >> 2, addr_out=0x19600 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=None,
                       data_b=None)
            soma1 = pX5(mode='max', addr_in=0x19600 >> 2, addr_out=0x9000, cin=128, cout=128, px=1, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=1)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=63, receive_num=3, addr_din_base=0x380,
                         addr_din_length=255, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.Receive_en = 1
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=64, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=128, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=192, pack_per_Rhead=63, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x19600 >> 2, addr_ciso=0, length_in=1 * 128, length_out=1 * 128,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L43 部分和求和，流水Cut， 3/3 - 1/49
    if phase_en[offset + 20]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=128, px=1, py=1,
                       addr_in=0x19600 >> 2, addr_bias=0x0, addr_out=0xeb80 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xeb80 >> 2, addr_out=0x19600 >> 2, cin=128, cout=128, px=1, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut3'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Y43 进行整理，每个得到12*128， 12*128， 12*128， 13*128
    if phase_en[offset + 21]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_y in [0]:
                px = 13
            else:
                px = 12
            soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=128, cout=128, px=px, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                        in_row_max=3, data_in=None)
            if core_y == 3:
                addr_din_length = 13 * 128 // 8 - 1
                receive_num = 2
            else:
                addr_din_length = 12 * 128 // 8 - 1
                receive_num = 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=px * 128 // 8 - 1, receive_num=receive_num,
                         addr_din_base=0x380, addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=192, pack_per_Rhead=15, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=96, pack_per_Rhead=95, A_offset=0, Const=0, EN=1)
            if core_y == 3:
                length = 13 * 128
                addr_out = 0xeb80 >> 2
            else:
                length = 12 * 128
                addr_out = 0xeb00 >> 2
            soma2 = p06(addr_in=0x8380, addr_out=addr_out, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L43 + shortcut 部分和求和，流水ReLU
    if phase_en[offset + 22]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 3:
                px = 13
            else:
                px = 12
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=px, py=1,
                       addr_in=0xe500 >> 2, addr_bias=0x0, addr_out=0x1c100 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x1c100 >> 2, addr_out=0x19000 >> 2, cin=128, cout=128, px=px, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer4.0.cut5'],
                        row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00015'
    chip = (1, 2)
    cuts = Resnet50Cuts()
    offset = 2

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[offset + 0] = 1  # L41 卷积，部分和收发
    phase[offset + 1] = 1  # L41 部分和求和，流水ReLU

    phase[offset + 2] = 1  # L41e 卷积，部分和收发 1/3 - 24/49
    phase[offset + 3] = 1  # L41e 部分和求和，流水Cut， 1/3 - 24/49
    phase[offset + 4] = 1  # L41e 卷积，部分和收发 2/3 - 24/49
    phase[offset + 5] = 1  # L41e 部分和求和，流水Cut， 2/3 - 24/49
    phase[offset + 6] = 1  # L41e 卷积，部分和收发 3/3 - 1/49
    phase[offset + 7] = 1  # L41e 部分和求和，流水Cut， 3/3 - 1/49

    phase[offset + 8] = 1  # 为L42整理数据，发送部分
    phase[offset + 9] = 1  # 为L42整理数据，横向多播部分

    phase[offset + 10] = 1  # L42 卷积，发送
    phase[offset + 11] = 1  # L42 部分和求和，流水ReLU，发送
    phase[offset + 12] = 1  # L42 整理数据 -- 横向多播

    phase[offset + 13] = 1  # Ye 进行整理，每个得到12*128， 12*128， 12*128， 13*128

    phase[offset + 14] = 1  # L43 卷积，部分和收发 1/3 - 24/49
    phase[offset + 15] = 1  # L43 部分和求和，流水Cut， 1/3 - 24/49
    phase[offset + 16] = 1  # L43 卷积，部分和收发 2/3 - 24/49
    phase[offset + 17] = 1  # L43 部分和求和，流水Cut， 2/3 - 24/49
    phase[offset + 18] = 1  # L43 卷积，部分和收发 3/3 - 1/49
    phase[offset + 19] = 1  # L43 部分和求和，流水Cut， 3/3 - 1/49

    phase[offset + 20] = 1  # Y43 进行整理，每个得到12*128， 12*128， 12*128， 13*128

    phase[offset + 21] = 1  # L43 + shortcut 部分和求和，流水ReLU

    # phase[0:14] = 0

    from generator.resnet50.resnet50_5chips.G15_data import generate_g15_data

    handler = ResNetDataHandler()
    data = generate_g15_data(handler, size_y=4, size_x=16)

    clock_in_phase = 100_000
    config = gen_g15_map_config(phase, clock_in_phase=clock_in_phase, size_x=16, size_y=4, data=data, in_data_en=False,
                                out_data_en=False, chip=chip, cuts=cuts)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "\\simulator\\Out_files\\" + case_file_name + "\\"
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
