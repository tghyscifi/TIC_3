import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41, p02
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler


def gen_g9_1_map_config(phase_en, clock_in_phase, size_x, size_y, cuts, data=None,
                        in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0), init_data=None):
    """
        ResNet-50 5-Chip Group-9-1
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
            'layer3.0.conv1': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L23
            'layer3.0.conv2': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L24
            'layer3.0.conv3': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L25
        }

    # ******** 数据交互 ********
    offset = 4

    # 发送 7*7*64  接收 24*512
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0xf500 >> 2,
                       addr_inb=0xf500 >> 2, addr_bias=0xf500 >> 2, addr_out=0xf500 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en and (core_y == 0):
                soma1 = pX5(mode='max', addr_in=0x18000 >> 2, addr_out=0x9000, cin=128, cout=128, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.0.cut5']['output'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_y == 0),
                         receive_en=in_data_en, send_num=7 * 7 * 128 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=24 * 512 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=24 * 512 // 8 - 1, nx=0, ny=0, data_in=None)
            if out_data_en and (core_y == 0):
                if core_x // 2 == 0:
                    dst_x, dst_y = 0, 4
                elif core_x // 2 == 1:
                    dst_x, dst_y = 1, 5
                elif core_x // 2 == 2:
                    dst_x, dst_y = 2, 6
                elif core_x // 2 == 3:
                    dst_x, dst_y = 3, 7
                else:
                    raise ValueError
                a = ((core_x % 2) * 128) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 128 // 8 - 1,
                                A_offset=16, Const=15, EN=1)
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
            soma2 = p06(addr_in=0x8380, addr_out=0x9300 >> 2, addr_ciso=0, length_in=512, length_out=512,
                        length_ciso=1, num_in=24, num_ciso=24, num_out=24, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 7*7*64   接收 25*512
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0xf500 >> 2,
                       addr_inb=0xf500 >> 2, addr_bias=0xf500 >> 2, addr_out=0xf500 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en and (core_y == 1):
                soma1 = pX5(mode='max', addr_in=0x18000 >> 2, addr_out=0x9000, cin=128, cout=128, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.0.cut5']['output'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_y == 1),
                         receive_en=in_data_en, send_num=7 * 7 * 128 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=25 * 512 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=25 * 512 // 8 - 1, nx=0, ny=0, data_in=None)
            if out_data_en and (core_y == 1):
                if core_x // 2 == 0:
                    dst_x, dst_y = 0, 4
                elif core_x // 2 == 1:
                    dst_x, dst_y = 1, 5
                elif core_x // 2 == 2:
                    dst_x, dst_y = 2, 6
                elif core_x // 2 == 3:
                    dst_x, dst_y = 3, 7
                else:
                    raise ValueError
                a = ((core_x % 2) * 128) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 128 // 8 - 1,
                                A_offset=16, Const=15, EN=1)
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
            soma2 = p06(addr_in=0x8380, addr_out=0xc300 >> 2, addr_ciso=0, length_in=512, length_out=512,
                        length_ciso=1, num_in=25, num_ciso=25, num_out=25, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 7*7*64
    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0xf500 >> 2,
                       addr_inb=0xf500 >> 2, addr_bias=0xf500 >> 2, addr_out=0xf500 >> 2, axon_delay=True,
                       L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en and (core_y == 2):
                soma1 = pX5(mode='max', addr_in=0x18000 >> 2, addr_out=0x9000, cin=128, cout=128, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.0.cut5']['output'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_y == 2),
                         receive_en=0, send_num=7 * 7 * 128 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en and (core_y == 2):
                if core_x // 2 == 0:
                    dst_x, dst_y = 0, 4
                elif core_x // 2 == 1:
                    dst_x, dst_y = 1, 5
                elif core_x // 2 == 2:
                    dst_x, dst_y = 2, 6
                elif core_x // 2 == 3:
                    dst_x, dst_y = 3, 7
                else:
                    raise ValueError
                a = ((core_x % 2) * 128) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 128 // 8 - 1,
                                A_offset=16, Const=15, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送 7*7*64
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if out_data_en and (core_y == 3):
                soma1 = pX5(mode='max', addr_in=0x18000 >> 2, addr_out=0x9000, cin=128, cout=128, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.0.cut5']['output'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=out_data_en and (core_y == 3),
                         receive_en=0, send_num=7 * 7 * 128 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en and (core_y == 3):
                if core_x // 2 == 0:
                    dst_x, dst_y = 0, 4
                elif core_x // 2 == 1:
                    dst_x, dst_y = 1, 5
                elif core_x // 2 == 2:
                    dst_x, dst_y = 2, 6
                elif core_x // 2 == 3:
                    dst_x, dst_y = 3, 7
                else:
                    raise ValueError
                a = ((core_x % 2) * 128) // 8
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=a,
                                pack_per_Rhead=7 * 7 * 128 // 8 - 1,
                                A_offset=16, Const=15, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********

    # L23 卷积，流水ReLU，发送()
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=7, py=7, cin=512, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x9300 >> 2,
                       addr_inb=0x10000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x18000 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None if in_data_en else data['layer3.0.conv1']['input'][(core_x, core_y)],
                       data_w=data['layer3.0.conv1']['weight'][(core_x, core_y)],
                       data_b=data['layer3.0.conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x18000 >> 2, addr_out=0x9000, cin=32, cout=32, px=7, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.0.cut1'],
                        row_ck_on=1, in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=195, receive_num=7, addr_din_base=0x3c0,
                         addr_din_length=1567, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            if (core_x, core_y) in [(0, 0), (7, 1), (0, 2), (7, 3)]:
                router.Receive_en = 1
            if core_y in [0, 1]:
                if core_x < 4:
                    dst_x, dst_y = 0, 0
                else:
                    dst_x, dst_y = 7, 1
            else:
                if core_x < 4:
                    dst_x, dst_y = 0, 2
                else:
                    dst_x, dst_y = 7, 3
            if dst_y % 2 == 0:
                a = abs(dst_y - core_y) * 784 + abs(dst_x - core_x) * 4
            else:
                a = (1 - abs(dst_y - core_y)) * 784 + (3 - abs(dst_x - core_x)) * 4
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y,
                            A=a, pack_per_Rhead=195, A_offset=12, Const=3, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 整理数据 -- 横向多播
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=1567, receive_num=0, addr_din_base=0x3c0,
                         addr_din_length=1567, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x3c0, addr_dout_length=1568 // 2 - 1, soma_in_en=0, cxy=1, relay_num=1567,
                         nx=1 if core_y % 2 == 0 else -1, ny=0, data_in=None)
            if (core_x, core_y) in [(0, 0), (7, 1), (0, 2), (7, 3)]:
                router.Send_en, router.Receive_en = 1, 0
            if core_x in [0, 7]:
                router.CXY = 0
            router.addRHead(S=0, T=1, P=0, Q=1, X=1 if core_y % 2 == 0 else -1, Y=0,
                            A=0, pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x83c0, addr_out=0x18700 >> 2, addr_ciso=0x0000, length_in=7 * 128, length_out=7 * 128,
                        length_ciso=1, num_in=14, num_ciso=14, num_out=14, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 整理数据 -- 1-3， 2-4之间发送一行交叠数据
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            addr_in = 0x1b100 >> 2 if core_y in [0, 1] else 0x18700 >> 2
            soma1 = p06(addr_in=addr_in, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=128, length_out=128,
                        length_ciso=1, num_in=14, num_ciso=14, num_out=14, type_in=1, type_out=1, in_cut_start=0,
                        in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=223, receive_num=0,
                         addr_din_base=0x3c0, addr_din_length=223, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2 if core_y in [0, 1] else -2,
                            A=0, pack_per_Rhead=223, A_offset=0, Const=0, EN=1)
            addr_out = 0x1b800 >> 2 if core_y in [0, 1] else 0x18000 >> 2
            soma2 = p06(addr_in=0x83c0, addr_out=addr_out, addr_ciso=0x0000, length_in=128, length_out=128,
                        length_ciso=1, num_in=14, num_ciso=14, num_out=14, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L24 卷积, 1-2, 3-4行互相交换3.5*14*32的数据，方便求部分和
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0, 1]:
                addr_ina = 0x18700 >> 2
                pad_top, pad_down = 1, 0
            else:
                addr_ina = 0x18000 >> 2
                pad_top, pad_down = 0, 1
            axon = p41(px=14, py=8, cin=128, cout=32, kx=3, ky=3, sx=1, sy=1, addr_ina=addr_ina,
                       addr_inb=0x300 >> 2, addr_bias=0x80 >> 2, addr_out=0x1bf00 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=pad_top, pad_down=pad_down, pad_left=1,
                       pad_right=1, data_x=None, data_w=data['layer3.0.conv2']['weight'][(core_x, core_y)],
                       data_b=data['layer3.0.conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1bf00 >> 2, addr_out=0x9000, cin=32, cout=32, px=14, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0, row_ck_on=1,
                        in_row_max=2)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1567, receive_num=1, addr_din_base=0x3c0,
                         addr_din_length=1567, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            if core_y in [0, 2]:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=784, pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=784, pack_per_Rhead=783, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x83c0, addr_out=0x9300 >> 2, addr_ciso=0x0000, length_in=7 * 32, length_out=7 * 32,
                        length_ciso=1, num_in=14, num_ciso=14, num_out=14, type_in=0, type_out=0,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L24 求部分和，流水ReLU，发送
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=32, px=7, py=7,
                       addr_in=0x9300 >> 2, addr_bias=0x0, addr_out=0x18000 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x18000 >> 2, addr_out=0x9000, cin=32, cout=32, px=7, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.0.cut2'],
                        row_ck_on=1, in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=195, receive_num=7, addr_din_base=0x3c0,
                         addr_din_length=1567, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            if (core_x, core_y) in [(0, 0), (7, 1), (0, 2), (7, 3)]:
                router.Receive_en = 1
            if core_y in [0, 1]:
                if core_x < 4:
                    dst_x, dst_y = 0, 0
                else:
                    dst_x, dst_y = 7, 1
            else:
                if core_x < 4:
                    dst_x, dst_y = 0, 2
                else:
                    dst_x, dst_y = 7, 3
            if dst_y % 2 == 0:
                a = abs(dst_y - core_y) * 784 + abs(dst_x - core_x) * 4
            else:
                a = (1 - abs(dst_y - core_y)) * 784 + (3 - abs(dst_x - core_x)) * 4
            router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y - core_y,
                            A=a, pack_per_Rhead=195, A_offset=12, Const=3, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 整理数据 -- 横向多播
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=1567, receive_num=0, addr_din_base=0x3c0,
                         addr_din_length=1567, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x3c0, addr_dout_length=1568 // 2 - 1, soma_in_en=0, cxy=1, relay_num=1567,
                         nx=1 if core_y % 2 == 0 else -1, ny=0, data_in=None)
            if (core_x, core_y) in [(0, 0), (7, 1), (0, 2), (7, 3)]:
                router.Send_en, router.Receive_en = 1, 0
            if core_x in [0, 7]:
                router.CXY = 0
            router.addRHead(S=0, T=1, P=0, Q=1, X=1 if core_y % 2 == 0 else -1, Y=0,
                            A=0, pack_per_Rhead=1567, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x83c0, addr_out=0x9300 >> 2, addr_ciso=0x0000, length_in=7 * 128, length_out=7 * 128,
                        length_ciso=1, num_in=14, num_ciso=14, num_out=14, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # L25 卷积1/2
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=7, py=7, cin=128, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x9300 >> 2,
                       addr_inb=0x14000 >> 2, addr_bias=0x100 >> 2, addr_out=0x18200 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=data['layer3.0.conv3']['weight'][(core_x, core_y)],
                       data_b=data['layer3.0.conv3']['bias'][(core_x, core_y)])
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # L25 部分和收发1/2
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_y in [0, 2]:
                addr_in = 0x1b200 >> 2
                px, py = 5, 5
                send_num = 1599  # 5 * 5 * 128 * 4 // 8 - 1
                din_length = 1535
                addr_out = 0x1b200 >> 2
                px2, py2 = 4, 6
            else:
                addr_in = 0x18200 >> 2
                px, py = 4, 6
                send_num = 1535  # 4 * 6 * 128 * 4 // 8 - 1
                din_length = 1599
                addr_out = 0x18000 >> 2
                px2, py2 = 5, 5
            soma1 = pX5(mode='max', addr_in=addr_in, addr_out=0x9000, cin=128, cout=128, px=px, py=py,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0,
                        row_ck_on=0, in_row_max=2, data_in=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=send_num, receive_num=0, addr_din_base=0x380,
                         addr_din_length=din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1 if core_y % 2 == 0 else -1,
                            A=0, pack_per_Rhead=send_num, A_offset=0, Const=0, EN=1)
            soma2 = pX5(mode='max', addr_in=0x8380, addr_out=addr_out, cin=128, cout=128, px=px2, py=py2,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0,
                        row_ck_on=0, in_row_max=2)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # L25 求和，流水截取 - 1/2
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0, 2]:
                addr_in = 0x18200 >> 2
                px, py = 4, 6
            else:
                addr_in = 0x18000 >> 2
                px, py = 5, 5
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=px, py=py,
                       addr_in=addr_in, addr_bias=0x0, addr_out=0xdd00 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xdd00 >> 2, addr_out=0xc400 >> 2, cin=128, cout=128, px=px, py=py,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=cuts['layer3.0.cut3'], row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L25 卷积2/2
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=7, py=7, cin=128, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0xab80 >> 2,
                       addr_inb=0x14000 >> 2, addr_bias=0x100 >> 2, addr_out=0x18200 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=data['layer3.0.conv3']['weight'][(core_x, core_y)],  # 其实上面已经初始化过weight和bias了
                       data_b=data['layer3.0.conv3']['bias'][(core_x, core_y)])
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # L25 部分和收发2/2
    if phase_en[offset + 10]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_y in [0, 2]:
                addr_in = 0x1b200 >> 2
                px, py = 5, 5
                send_num = 1599  # 5 * 5 * 128 * 4 // 8 - 1
                din_length = 1535
                addr_out = 0x1b200 >> 2
                px2, py2 = 4, 6
            else:
                addr_in = 0x18200 >> 2
                px, py = 4, 6
                send_num = 1535  # 4 * 6 * 128 * 4 // 8 - 1
                din_length = 1599
                addr_out = 0x18000 >> 2
                px2, py2 = 5, 5
            soma1 = pX5(mode='max', addr_in=addr_in, addr_out=0x9000, cin=128, cout=128, px=px, py=py,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0,
                        row_ck_on=0, in_row_max=2, data_in=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=send_num, receive_num=0, addr_din_base=0x380,
                         addr_din_length=din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1 if core_y % 2 == 0 else -1,
                            A=0, pack_per_Rhead=send_num, A_offset=0, Const=0, EN=1)
            soma2 = pX5(mode='max', addr_in=0x8380, addr_out=addr_out, cin=128, cout=128, px=px2, py=py2,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80000000, type_in=0, type_out=0, in_cut_start=0,
                        row_ck_on=0, in_row_max=2)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # L25 求和，流水截取 - 2/2
    if phase_en[offset + 11]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y in [0, 2]:
                addr_in = 0x18200 >> 2
                px, py = 4, 6
            else:
                addr_in = 0x18000 >> 2
                px, py = 5, 5
            axon = p02(avg_pooling_en=False, ina_type=0, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=px, py=py,
                       addr_in=addr_in, addr_bias=0x0, addr_out=0xdd00 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0xdd00 >> 2, addr_out=0xd080 >> 2, cin=128, cout=128, px=px, py=py,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=cuts['layer3.0.cut3'], row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L25 数据整理
    if phase_en[offset + 12]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_y in [0, 2]:
                addr_in = 0xd080 >> 2
                length_in = 3072  # 24 * 128
                send_num = 383
                din_length = 399
            else:
                addr_in = 0xc400 >> 2
                length_in = 3200  # 25 * 128
                send_num = 399
                din_length = 383
            soma1 = p06(addr_in=addr_in, addr_out=0x9000, addr_ciso=0x0, length_in=length_in,
                        length_out=length_in, length_ciso=1, num_in=1, num_ciso=1, num_out=1,
                        type_in=1, type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x350)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=send_num, receive_num=0, addr_din_base=0x380,
                         addr_din_length=din_length, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1 if core_y in [0, 2] else -1, A=0, pack_per_Rhead=send_num,
                            A_offset=0, Const=0, EN=1)
            if core_y in [0, 2]:
                soma2 = p06(addr_in=0x8380, addr_out=0xd000 >> 2, addr_ciso=0x0, length_in=3200, length_out=3200,
                            length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1, type_out=1, in_cut_start=0,
                            in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            else:
                soma2 = p06(addr_in=0x8380, addr_out=0xc480 >> 2, addr_ciso=0x0, length_in=3072, length_out=3072,
                            length_ciso=1, num_in=1, num_ciso=1, num_out=1, type_in=1, type_out=1, in_cut_start=0,
                            in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L25 数据搬运（只是方便后面操作）
    if phase_en[offset + 13]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if core_y in [0, 2]:
                addr_in = 0xc400 >> 2
            else:
                addr_in = 0xc480 >> 2
            soma1 = p06(addr_in=addr_in, addr_out=0x9300 >> 2, addr_ciso=0x0, length_in=7 * 128,
                        length_out=7 * 128, length_ciso=1, num_in=7, num_ciso=7, num_out=7,
                        type_in=1, type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            router, soma2 = None, None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # shortcut 求和，流水ReLU
    if phase_en[offset + 14]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=7, py=7,
                       addr_in=0x9300 >> 2, addr_bias=0x0, addr_out=0x19880 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x19880 >> 2, addr_out=0x18000 >> 2, cin=128, cout=128, px=7, py=7,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer3.0.cut5'],
                        row_ck_on=1, in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R000091'
    cuts = Resnet50Cuts()

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1  # L23 卷积，流水ReLU，发送()
    phase[1] = 1  # 整理数据 -- 横向多播
    phase[2] = 1  # 整理数据 -- 1-3， 2-4之间发送一行交叠数据

    phase[3] = 1  # L24 卷积, 1-2, 3-4行互相交换3.5*14*32的数据，方便求部分和
    phase[4] = 1  # L24 求部分和，流水ReLU，发送
    phase[5] = 1  # 整理数据 -- 横向多播

    phase[6] = 1  # L25 卷积1/2
    phase[7] = 1  # L25 部分和收发1/2
    phase[8] = 1  # L25 求和，流水ReLU - 1/2
    phase[9] = 1  # L25 卷积2/2
    phase[10] = 1  # L25 部分和收发2/2
    phase[11] = 1  # L25 求和，流水ReLU - 2/2
    phase[12] = 1  # L25 数据整理
    phase[13] = 1  # L25 数据整理

    phase[14] = 1  # shortcut 求和，流水ReLU

    from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data

    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)

    clock_in_phase = 50_000
    config = gen_g9_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=4, data=data, in_data_en=False,
                                 out_data_en=False, cuts=cuts)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[((0, 0), 0)][0][((0, 0), (0, 0))]['prims']) * clock_in_phase)

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
