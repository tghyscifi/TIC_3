import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41, p02
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler


def gen_g10_map_config(phase_en, clock_in_phase, size_x, size_y, cuts, data=None,
                       in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0), init_data=None):
    """
        ResNet-50 5-Chip Group-10
        core array : 4 * 8
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
            'layer3.1.conv1': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L26
            'layer3.1.conv2': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L27
            'layer3.1.conv3': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L28
        }

    # ******** 数据交互 ********
    offset = 4

    # 发送 7*7*64  接收 49*256
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x8980 >> 2,
                       addr_inb=0x8980 >> 2, addr_bias=0x8980 >> 2, addr_out=0x8980 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en and (core_x % 2 == 0):
                soma1 = pX5(mode='max', addr_in=0x1e780 >> 2, addr_out=0x9000, cin=64, cout=64, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.1.cut5']['output1'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_x % 2 == 0),
                         receive_en=in_data_en, send_num=7 * 7 * 64 // 8 - 1, receive_num=1,
                         addr_din_base=0x380, addr_din_length=1567, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=1567, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    dst_x, dst_y = 8, 0
                elif core_y == 1:
                    dst_x, dst_y = 8, 1
                elif core_y == 2:
                    dst_x, dst_y = 8, 2
                elif core_y == 3:
                    dst_x, dst_y = 8, 3
                else:
                    raise ValueError
                a = ((core_x // 2) * 64) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 64 // 8 - 1,
                                A_offset=24, Const=7, EN=1)
            if in_data_en:
                if core_y == 0:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 1:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 2, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 2:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 3, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 3:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2, 3]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 4, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                else:
                    raise ValueError
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0, length_in=256, length_out=256,
                        length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 7*7*64  接收 49*256
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x8980 >> 2,
                       addr_inb=0x8980 >> 2, addr_bias=0x8980 >> 2, addr_out=0x8980 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en and (core_x % 2 == 0):
                soma1 = pX5(mode='max', addr_in=0x1f3c0 >> 2, addr_out=0x9000, cin=64, cout=64, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.1.cut5']['output2'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_x % 2 == 0),
                         receive_en=in_data_en, send_num=7 * 7 * 64 // 8 - 1, receive_num=1,
                         addr_din_base=0x380, addr_din_length=1567, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=1567, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    dst_x, dst_y = 8, 0
                elif core_y == 1:
                    dst_x, dst_y = 8, 1
                elif core_y == 2:
                    dst_x, dst_y = 8, 2
                elif core_y == 3:
                    dst_x, dst_y = 8, 3
                else:
                    raise ValueError
                a = ((core_x // 2) * 64) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 64 // 8 - 1,
                                A_offset=24, Const=7, EN=1)
            if in_data_en:
                if core_y == 0:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 1:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 2, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 2:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 3, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 3:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2, 3]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 4, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                else:
                    raise ValueError
            soma2 = p06(addr_in=0x8380, addr_out=0x13100 >> 2, addr_ciso=0, length_in=256, length_out=256,
                        length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 7*7*64  接收 49*256
    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x8980 >> 2,
                       addr_inb=0x8980 >> 2, addr_bias=0x8980 >> 2, addr_out=0x8980 >> 2, axon_delay=True,
                       L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en and (core_x % 2 == 1):
                soma1 = pX5(mode='max', addr_in=0x1e780 >> 2, addr_out=0x9000, cin=64, cout=64, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.1.cut5']['output1'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_x % 2 == 1),
                         receive_en=in_data_en, send_num=7 * 7 * 64 // 8 - 1, receive_num=1,
                         addr_din_base=0x380, addr_din_length=1567, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=1567, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    dst_x, dst_y = 8, 0
                elif core_y == 1:
                    dst_x, dst_y = 8, 1
                elif core_y == 2:
                    dst_x, dst_y = 8, 2
                elif core_y == 3:
                    dst_x, dst_y = 8, 3
                else:
                    raise ValueError
                a = ((core_x // 2) * 64) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 64 // 8 - 1,
                                A_offset=24, Const=7, EN=1)
            if in_data_en:
                if core_y == 0:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 1:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 2, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 2:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 3, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 3:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2, 3]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 4, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                else:
                    raise ValueError
            soma2 = p06(addr_in=0x8380, addr_out=0x16200 >> 2, addr_ciso=0, length_in=256, length_out=256,
                        length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 7*7*64  接收 49*256
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if out_data_en and (core_x % 2 == 1):
                soma1 = pX5(mode='max', addr_in=0x1f3c0 >> 2, addr_out=0x9000, cin=64, cout=64, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.1.cut5']['output2'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_x % 2 == 1),
                         receive_en=in_data_en, send_num=7 * 7 * 64 // 8 - 1, receive_num=1,
                         addr_din_base=0x380, addr_din_length=1567, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=1567, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    dst_x, dst_y = 8, 0
                elif core_y == 1:
                    dst_x, dst_y = 8, 1
                elif core_y == 2:
                    dst_x, dst_y = 8, 2
                elif core_y == 3:
                    dst_x, dst_y = 8, 3
                else:
                    raise ValueError
                a = ((core_x // 2) * 64) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 64 // 8 - 1,
                                A_offset=24, Const=7, EN=1)
            if in_data_en:
                if core_y == 0:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 1:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 2, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 2:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 3, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                elif core_y == 3:
                    if core_x == 7:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_x in [1, 2, 3]:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                    elif core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 4, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
                else:
                    raise ValueError
            soma2 = p06(addr_in=0x8380, addr_out=0x19300 >> 2, addr_ciso=0, length_in=256, length_out=256,
                        length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********

    # L26 卷积，部分和收发  1/2
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=7, cin=256, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=0x180 >> 2, addr_bias=0x0000 >> 2, addr_out=0x1c400 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None if in_data_en else data['layer3.1.conv1']['input1'][(core_x, core_y)],
                       data_w=data['layer3.1.conv1']['weight'][(core_x, core_y)],
                       data_b=data['layer3.1.conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1c400 >> 2, addr_out=0x9000, cin=32, cout=32, px=14, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            if core_y in [0, 1]:
                addr_din_length = 1599
            else:
                addr_din_length = 1535
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1567, receive_num=3, addr_din_base=0x380,
                         addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x1c400 >> 2, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L26 部分和求和，流水ReLU 1/2
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0, 1]:
                px, py = 5, 5
            else:
                px, py = 4, 6
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=32, px=px, py=py,
                       addr_in=0x1c400 >> 2, addr_bias=0x0, addr_out=0x8980 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x8980 >> 2, addr_out=0xe140 >> 2, cin=32, cout=32, px=px, py=py, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.1.cut1'],
                        row_ck_on=1,
                        in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L26 卷积，部分和收发  2/2
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=7, cin=256, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x16200 >> 2,
                       addr_inb=0x180 >> 2, addr_bias=0x0000 >> 2, addr_out=0x1c400 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None if in_data_en else data['layer3.1.conv1']['input2'][(core_x, core_y)],
                       data_w=None,
                       data_b=None)
            soma1 = pX5(mode='max', addr_in=0x1c400 >> 2, addr_out=0x9000, cin=32, cout=32, px=14, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            if core_y in [0, 1]:
                addr_din_length = 1599
            else:
                addr_din_length = 1535
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1567, receive_num=3, addr_din_base=0x380,
                         addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x1c400 >> 2, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L26 部分和求和，流水ReLU 2/2 -- X1 split出shortcut部分 - 7*14*64=6.125
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0, 1]:
                px, py = 5, 5
            else:
                px, py = 4, 6
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=32, px=px, py=py,
                       addr_in=0x1c400 >> 2, addr_bias=0x0, addr_out=0x8980 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x8980 >> 2, addr_out=0xe460 >> 2, cin=32, cout=32, px=px, py=py, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.1.cut1'],
                        row_ck_on=1,
                        in_row_max=3)
            router = None
            addr_in = 0x10000 if core_x % 2 == 0 else 0x16200
            addr_in += core_x // 2 * 64
            length_in = 256
            soma2 = p06(addr_in=addr_in >> 2, addr_out=0xe780 >> 2, addr_ciso=0x0 >> 2, length_in=length_in,
                        num_in=7 * 14, length_ciso=1, num_ciso=7 * 14, length_out=64, num_out=7 * 14, type_in=1,
                        type_out=1, in_cut_start=0, row_ck_on=0, in_row_max=0, data_in=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L27整理数据，发送部分
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_y // 2 == 0:
                length = 25 * 32
            else:
                length = 24 * 32
            soma1 = p06(addr_in=0xe140 >> 2, addr_out=0x9000, addr_ciso=0xe460 >> 2, length_in=length,
                        length_out=length * 2, length_ciso=length, num_in=1, num_ciso=1, num_out=1, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=length * 2 // 8 - 1, receive_num=15,
                         addr_din_base=0x380, addr_din_length=1567, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            dst_x, dst_y = core_x // 2 * 2, core_x // 2
            if (core_x, core_y) == (dst_x, dst_y):
                router.Receive_en = 1
            if core_y == 0:
                a = 0
            elif core_y == 1:
                a = 25 * 64
            elif core_y == 2:
                a = 50 * 64
            else:
                a = 74 * 64
            a1 = a + core_x % 2 * 32
            a2 = a1 + 98 * 64
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a1 // 8,
                            pack_per_Rhead=length // 8 - 1, A_offset=4, Const=3, EN=1)
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a2 // 8,
                            pack_per_Rhead=length // 8 - 1, A_offset=4, Const=3, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L27整理数据，横向多播部分
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=1567, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1567, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x380, addr_dout_length=1568 // 2 - 1, soma_in_en=0, cxy=1, relay_num=1567,
                         nx=1, ny=0, data_in=None)
            if (core_x, core_y) in [(0, 0), (2, 1), (4, 2), (6, 3)]:
                router.Send_en, router.Receive_en, router.CXY = 1, 0, 0
            if core_y == 0:
                router.Nx = 1
            elif core_y == 1:
                if core_x in [1]:
                    router.Nx = -1
                elif core_x == 0:
                    router.Nx = 3
                else:
                    router.Nx = 1
            elif core_y == 2:
                if core_x in [1, 2, 3]:
                    router.Nx = -1
                elif core_x == 0:
                    router.Nx = 5
                else:
                    router.Nx = 1
            elif core_y == 3:
                if core_x in [1, 2, 3, 4, 5]:
                    router.Nx = -1
                elif core_x == 0:
                    router.Nx = 7
                else:
                    router.Nx = 1
            if core_x == 7:
                router.CXY = 0
            router.addRHead(S=0, T=1, P=0, Q=1, X=1 if core_y == 0 else -1, Y=0,
                            A=0, pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0x0000, length_in=14 * 64, length_out=14 * 64,
                        length_ciso=1, num_in=14, num_ciso=14, num_out=14, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # L27 卷积，部分和收发  1/2
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=8, cin=64, cout=32, kx=3, ky=3, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=0x2180 >> 2, addr_bias=0x80 >> 2, addr_out=0x13100 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=1, pad_down=0, pad_left=1, pad_right=1,
                       data_x=None,
                       data_w=data['layer3.1.conv2']['weight'][(core_x, core_y)],
                       data_b=data['layer3.1.conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x13100 >> 2, addr_out=0x9000, cin=32, cout=32, px=14, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            if core_y in [0, 1]:
                addr_din_length = 1599
            else:
                addr_din_length = 1535
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1567, receive_num=3, addr_din_base=0x380,
                         addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x13100 >> 2, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L27 部分和求和，流水ReLU 1/2
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0, 1]:
                px, py = 5, 5
            else:
                px, py = 4, 6
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=32, px=px, py=py,
                       addr_in=0x13100 >> 2, addr_bias=0x0, addr_out=0x8980 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x8980 >> 2, addr_out=0x1f9c0 >> 2, cin=32, cout=32, px=px, py=py, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.1.cut2'],
                        row_ck_on=1,
                        in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L27 卷积，部分和收发  2/2
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=8, cin=64, cout=32, kx=3, ky=3, sx=1, sy=1, addr_ina=0x11500 >> 2,
                       addr_inb=0x2180 >> 2, addr_bias=0x80 >> 2, addr_out=0x13100 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=1, pad_left=1, pad_right=1,
                       data_x=None,
                       data_w=None,
                       data_b=None)
            soma1 = pX5(mode='max', addr_in=0x13100 >> 2, addr_out=0x9000, cin=32, cout=32, px=14, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=3)
            if core_y in [0, 1]:
                addr_din_length = 1599
            else:
                addr_din_length = 1535
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1567, receive_num=3, addr_din_base=0x380,
                         addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=1,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_y == 0:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 1:
                length = 25 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=400, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=384, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            elif core_y == 2:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=800, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=768, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            else:
                length = 24 * 32
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-3, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-2, A=1200, pack_per_Rhead=399, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=1152, pack_per_Rhead=383, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x13100 >> 2, addr_ciso=0, length_in=length, length_out=length,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L26 部分和求和，流水ReLU 2/2
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0, 1]:
                px, py = 5, 5
            else:
                px, py = 4, 6
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=4, sx=1, sy=1, cin=32, px=px, py=py,
                       addr_in=0x13100 >> 2, addr_bias=0x0, addr_out=0x8980 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x8980 >> 2, addr_out=0x1fce0 >> 2, cin=32, cout=32, px=px, py=py, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.1.cut2'],
                        row_ck_on=1,
                        in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 1/4 - 发送 1 2 行
    if phase_en[offset + 10]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            length = 25 * 32
            soma1 = p06(addr_in=0x1f9c0 >> 2, addr_out=0x9000, addr_ciso=0x0, length_in=length,
                        length_out=length, length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=length // 8 - 1, receive_num=15,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            dst_x, dst_y = 0, 0
            if (core_x, core_y) == (dst_x, dst_y):
                router.Receive_en = 1
            if core_y % 2 == 0:
                a = 0
            else:
                a = 25 * 256
            a1 = a + core_x * 32
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a1 // 8,
                            pack_per_Rhead=length // 8 - 1, A_offset=28, Const=3, EN=1)
            if core_y in [2, 3]:
                if (core_x, core_y) != (0, 0):
                    soma1, router = None, None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 1/4 - 多播 1 2 行
    if phase_en[offset + 11]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            length = 25 * 32
            if core_y % 2 == 0:
                if core_x == 7:
                    nx, ny = 0, 1
                else:
                    nx, ny = 1, 0
            else:
                if core_x == 0:
                    nx, ny = 0, 1
                else:
                    nx, ny = -1, 0
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=length * 8 * 2 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x380, addr_dout_length=length * 8 * 2 // 16 - 1,
                         soma_in_en=0, cxy=1, relay_num=length * 8 * 2 // 8 - 1, nx=nx, ny=ny, data_in=None)
            if (core_x, core_y) in [(0, 0)]:
                router.Send_en, router.Receive_en, router.CXY = 1, 0, 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0, pack_per_Rhead=length * 8 * 2 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if (core_x, core_y) == (0, 3):
                router.CXY = 0
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0x0000, length_in=length, length_out=length,
                        length_ciso=1, num_in=16, num_ciso=16, num_out=16, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 2/4 - 发送 3 4 行
    if phase_en[offset + 12]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            length = 24 * 32
            soma1 = p06(addr_in=0x1f9c0 >> 2, addr_out=0x9000, addr_ciso=0x0, length_in=length,
                        length_out=length, length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=length // 8 - 1, receive_num=15,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            dst_x, dst_y = 0, 0
            if (core_x, core_y) == (dst_x, dst_y):
                router.Receive_en = 1
            if core_y % 2 == 0:
                a = 0
            else:
                a = 24 * 256
            a1 = a + core_x * 32
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a1 // 8,
                            pack_per_Rhead=length // 8 - 1, A_offset=28, Const=3, EN=1)
            if core_y in [0, 1]:
                soma1 = None
                router.Send_en = 0
                if (core_x, core_y) != (0, 0):
                    router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 2/4 - 多播 3 4 行
    if phase_en[offset + 13]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            length = 24 * 32
            if core_y % 2 == 0:
                if core_x == 7:
                    nx, ny = 0, 1
                else:
                    nx, ny = 1, 0
            else:
                if core_x == 0:
                    nx, ny = 0, 1
                else:
                    nx, ny = -1, 0
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=length * 8 * 2 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x380, addr_dout_length=length * 8 * 2 // 16 - 1,
                         soma_in_en=0, cxy=1, relay_num=length * 8 * 2 // 8 - 1, nx=nx, ny=ny, data_in=None)
            if (core_x, core_y) in [(0, 0)]:
                router.Send_en, router.Receive_en, router.CXY = 1, 0, 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0, pack_per_Rhead=length * 8 * 2 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if (core_x, core_y) == (0, 3):
                router.CXY = 0
            soma2 = p06(addr_in=0x8380, addr_out=0x13200 >> 2, addr_ciso=0x0000, length_in=length, length_out=length,
                        length_ciso=1, num_in=16, num_ciso=16, num_out=16, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 3/4 - 发送 1 2 行
    if phase_en[offset + 14]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            length = 25 * 32
            soma1 = p06(addr_in=0x1fce0 >> 2, addr_out=0x9000, addr_ciso=0x0, length_in=length,
                        length_out=length, length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=length // 8 - 1, receive_num=15,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            dst_x, dst_y = 0, 0
            if (core_x, core_y) == (dst_x, dst_y):
                router.Receive_en = 1
            if core_y % 2 == 0:
                a = 0
            else:
                a = 25 * 256
            a1 = a + core_x * 32
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a1 // 8,
                            pack_per_Rhead=length // 8 - 1, A_offset=28, Const=3, EN=1)
            if core_y in [2, 3]:
                if (core_x, core_y) != (0, 0):
                    soma1, router = None, None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 3/4 - 多播 1 2 行
    if phase_en[offset + 15]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            length = 25 * 32
            if core_y % 2 == 0:
                if core_x == 7:
                    nx, ny = 0, 1
                else:
                    nx, ny = 1, 0
            else:
                if core_x == 0:
                    nx, ny = 0, 1
                else:
                    nx, ny = -1, 0
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=length * 8 * 2 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x380, addr_dout_length=length * 8 * 2 // 16 - 1,
                         soma_in_en=0, cxy=1, relay_num=length * 8 * 2 // 8 - 1, nx=nx, ny=ny, data_in=None)
            if (core_x, core_y) in [(0, 0)]:
                router.Send_en, router.Receive_en, router.CXY = 1, 0, 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0, pack_per_Rhead=length * 8 * 2 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if (core_x, core_y) == (0, 3):
                router.CXY = 0
            soma2 = p06(addr_in=0x8380, addr_out=0x16200 >> 2, addr_ciso=0x0000, length_in=length, length_out=length,
                        length_ciso=1, num_in=16, num_ciso=16, num_out=16, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 4/4 - 发送 3 4 行
    if phase_en[offset + 16]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            length = 24 * 32
            soma1 = p06(addr_in=0x1fce0 >> 2, addr_out=0x9000, addr_ciso=0x0, length_in=length,
                        length_out=length, length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1,
                        type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=length // 8 - 1, receive_num=15,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            dst_x, dst_y = 0, 0
            if (core_x, core_y) == (dst_x, dst_y):
                router.Receive_en = 1
            if core_y % 2 == 0:
                a = 0
            else:
                a = 24 * 256
            a1 = a + core_x * 32
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y, A=a1 // 8,
                            pack_per_Rhead=length // 8 - 1, A_offset=28, Const=3, EN=1)
            if core_y in [0, 1]:
                soma1 = None
                router.Send_en = 0
                if (core_x, core_y) != (0, 0):
                    router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 为L28整理数据 - 4/4 - 多播 3 4 行
    if phase_en[offset + 17]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            length = 24 * 32
            if core_y % 2 == 0:
                if core_x == 7:
                    nx, ny = 0, 1
                else:
                    nx, ny = 1, 0
            else:
                if core_x == 0:
                    nx, ny = 0, 1
                else:
                    nx, ny = -1, 0
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=length * 8 * 2 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=length * 8 * 2 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x380, addr_dout_length=length * 8 * 2 // 16 - 1,
                         soma_in_en=0, cxy=1, relay_num=length * 8 * 2 // 8 - 1, nx=nx, ny=ny, data_in=None)
            if (core_x, core_y) in [(0, 0)]:
                router.Send_en, router.Receive_en, router.CXY = 1, 0, 0
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0, pack_per_Rhead=length * 8 * 2 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if (core_x, core_y) == (0, 3):
                router.CXY = 0
            soma2 = p06(addr_in=0x8380, addr_out=0x19400 >> 2, addr_ciso=0x0000, length_in=length, length_out=length,
                        length_ciso=1, num_in=16, num_ciso=16, num_out=16, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L28 卷积，流水cut，发送  14*14*32 -> 7*14*64
    if phase_en[offset + 18]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=14, cin=256, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=0x6980 >> 2, addr_bias=0x100 >> 2, addr_out=0x8980 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=data['layer3.1.conv3']['weight'][(core_x, core_y)],
                       data_b=data['layer3.1.conv3']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x8980 >> 2, addr_out=0x9000, cin=32, cout=32, px=14, py=14, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer3.1.cut3'],
                        row_ck_on=1,
                        in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=783, receive_num=1, addr_din_base=0x380,
                         addr_din_length=783, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            if core_x % 2 == 0:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=391, A_offset=4, Const=3, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=391, A_offset=4, Const=3, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=4, pack_per_Rhead=391, A_offset=4, Const=3, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=4, pack_per_Rhead=391, A_offset=4, Const=3, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0xcf00 >> 2, addr_ciso=0, length_in=7 * 64, length_out=7 * 64,
                        length_ciso=1, num_in=14, num_ciso=14, num_out=14, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L28 + shortcut 部分和求和，流水ReLU
    if phase_en[offset + 19]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=64, px=14, py=7,
                       addr_in=0xcf00 >> 2, addr_bias=0x0, addr_out=0x10000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x10000 >> 2, addr_out=0x1e780 >> 2, cin=64, cout=64, px=14, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.1.cut5'],
                        row_ck_on=1,
                        in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00010'
    chip = (1, 2)

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1  # L26 卷积，发送()  1/2
    phase[1] = 1  # L26 部分和求和，流水ReLU 1/2
    phase[2] = 1  # L26 卷积，发送()  2/2
    phase[3] = 1  # L26 部分和求和，流水ReLU 2/2  -- X1 split出shortcut部分 - 7*14*64=6.125

    phase[4] = 1  # 为L27整理数据，发送部分
    phase[5] = 1  # 为L27整理数据，横向多播部分

    phase[6] = 1  # L27 卷积，发送()  1/2
    phase[7] = 1  # L27 部分和求和，流水ReLU 1/2
    phase[8] = 1  # L27 卷积，发送()  2/2
    phase[9] = 1  # L27 部分和求和，流水ReLU 2/2

    phase[10] = 1  # 为L28整理数据 - 1/4 - 发送 1 2 行
    phase[11] = 1  # 为L28整理数据 - 1/4 - 多播 1 2 行
    phase[12] = 1  # 为L28整理数据 - 2/4 - 发送 3 4 行
    phase[13] = 1  # 为L28整理数据 - 2/4 - 多播 3 4 行
    phase[14] = 1  # 为L28整理数据 - 3/4 - 发送 1 2 行
    phase[15] = 1  # 为L28整理数据 - 3/4 - 多播 1 2 行
    phase[16] = 1  # 为L28整理数据 - 4/4 - 发送 3 4 行
    phase[17] = 1  # 为L28整理数据 - 4/4 - 多播 3 4 行

    phase[18] = 1  # L28 卷积，流水cut，发送  14*14*32 -> 7*14*64

    phase[19] = 1  # L28 + shortcut 求和，流水ReLU

    from generator.resnet50.resnet50_5chips.G10_data import generate_g10_data

    handler = ResNetDataHandler()
    data = generate_g10_data(handler, size_y=4, size_x=8)

    clock_in_phase = 100_000
    config = gen_g10_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=4, data=data, in_data_en=False,
                                out_data_en=False, chip=chip)
    MapConfigGen.add_router_info(map_config=config, group_idx_list=[0], chip_x_num=5, chip_y_num=5)

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
