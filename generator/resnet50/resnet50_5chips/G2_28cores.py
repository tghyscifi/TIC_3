import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41, p02
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler


def gen_g2_map_config(phase_en, clock_in_phase, size_x, size_y, cuts, data=None,
                      in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0),
                      init_data=True):
    """
        ResNet-50 5-Chip Group-2
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

    # data in
    if data is None:
        data = {
            'layer1.0.conv1': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L2
            'layer1.0.conv2': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L3
            'layer1.0.conv3': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L4
            'layer1.0.downsample.0': {
                'input': [],
                'weight': [],
                'bias': []
            },  # L4e
        }

    # ******** 数据交互 ********
    offset = 5

    # 接收4*28*64  发送 1*28*128 * 2
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x17500 >> 2,
                       addr_inb=0x17500 >> 2, addr_bias=0x17500 >> 2, addr_out=0x17500 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en:
                soma1 = p06(addr_in=0x9000 >> 2, addr_out=0x9000, addr_ciso=0xc800 >> 2, length_in=14*128,
                            length_out=28*128, length_ciso=14*128, num_in=2, num_ciso=2, num_out=2, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['layer1.0.cut5']['in_1'][(core_x, core_y)] if init_data else None,
                            data_ciso=data['layer1.0.cut5']['ciso_1'][(core_x, core_y)] if init_data else None)
            if core_x in [0, 7]:
                receive_num = 0
            else:
                receive_num = 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=in_data_en, send_num=895, receive_num=receive_num,
                         addr_din_base=0x380, addr_din_length=4 * 28 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                else:

                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x5800 >> 2, addr_ciso=0, length_in=28 * 64, length_out=28 * 64,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    #  发送 1*28*128 * 2
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x17500 >> 2,
                       addr_inb=0x17500 >> 2, addr_bias=0x17500 >> 2, addr_out=0x17500 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en:
                soma1 = p06(addr_in=0x9e00 >> 2, addr_out=0x9000, addr_ciso=0xd600 >> 2, length_in=14*128,
                            length_out=28*128, length_ciso=14*128, num_in=2, num_ciso=2, num_out=2, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['layer1.0.cut5']['in_2'][(core_x, core_y)] if init_data else None,
                            data_ciso=data['layer1.0.cut5']['ciso_2'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=0, send_num=895, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                else:

                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    #  发送 1*28*128 * 2
    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x17500 >> 2,
                       addr_inb=0x17500 >> 2, addr_bias=0x17500 >> 2, addr_out=0x17500 >> 2, axon_delay=True,
                       L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en:
                soma1 = p06(addr_in=0xac00 >> 2, addr_out=0x9000, addr_ciso=0xe400 >> 2, length_in=14*128,
                            length_out=28*128, length_ciso=14*128, num_in=2, num_ciso=2, num_out=2, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['layer1.0.cut5']['in_3'][(core_x, core_y)] if init_data else None,
                            data_ciso=data['layer1.0.cut5']['ciso_3'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=0, send_num=895, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                else:

                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    #  发送 1*28*128 * 2
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if out_data_en:
                soma1 = p06(addr_in=0xba00 >> 2, addr_out=0x9000, addr_ciso=0xf200 >> 2, length_in=14*128,
                            length_out=28*128, length_ciso=14*128, num_in=2, num_ciso=2, num_out=2, type_in=1,
                            type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0,
                            data_in=data['layer1.0.cut5']['in_4'][(core_x, core_y)] if init_data else None,
                            data_ciso=data['layer1.0.cut5']['ciso_4'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=0, send_num=895, receive_num=0,
                         addr_din_base=0x380, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=0, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=3, A=448, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                else:

                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=16, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=2, A=464, pack_per_Rhead=223,
                                    A_offset=16, Const=15, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 将上下两行接收的4*28*64互相发送，然后整理成4*56*64
    if phase_en[4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if in_data_en:
                soma1 = pX5(mode='max', addr_in=0x5800 >> 2, addr_out=0x9000, cin=64, cout=64, px=28, py=4, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=in_data_en,
                         receive_en=in_data_en, send_num=4 * 28 * 64 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=4 * 28 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if in_data_en:
                if core_y == 0:
                    x, y = 0, 1
                else:
                    x, y = 0, -1
                router.addRHead(S=0, T=1, P=0, Q=0, X=x, Y=y, A=0, pack_per_Rhead=4 * 28 * 64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if core_y == 0:
                addr_in, addr_ciso = 0x5800 >> 2, 0x8380
            else:
                addr_in, addr_ciso = 0x8380, 0x5800 >> 2
            soma2 = p06(addr_in=addr_in, addr_out=0x2000 >> 2, addr_ciso=addr_ciso, length_in=28 * 64,
                        length_out=56 * 64,
                        length_ciso=28 * 64, num_in=4, num_ciso=4, num_out=4, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 ********

    # L2 卷积，流水ReLU
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=56, py=4, cin=64, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x2000 >> 2,
                       addr_inb=0x10500 >> 2, addr_bias=0x10000 >> 2, addr_out=0x5800 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None if in_data_en else data['layer1.0.conv1']['input'][(core_x, core_y)],
                       data_w=data['layer1.0.conv1']['weight'][(core_x, core_y)],
                       data_b=data['layer1.0.conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x5800 >> 2, addr_out=0x17500 >> 2, cin=32, cout=32, px=56, py=4, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer1.0.cut1'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 整理数据 -- 上下互发送4*56*32的数据
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x17500 >> 2, addr_out=0x9000, addr_ciso=0x0000, length_in=56 * 32, length_out=56 * 32,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=1, type_out=1, in_cut_start=0,
                        in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=895, receive_num=0, addr_din_base=0x400,
                         addr_din_length=895, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1 if core_y == 0 else -1,
                            A=0, pack_per_Rhead=895, A_offset=0, Const=0, EN=1)
            if core_y == 0:
                addr_in, addr_ciso = 0x17500 >> 2, 0x8400
            else:
                addr_in, addr_ciso = 0x8400, 0x17500 >> 2
            soma2 = p06(addr_in=addr_in, addr_out=0x6600 >> 2, addr_ciso=addr_ciso, length_in=32, length_out=64,
                        length_ciso=32, num_in=4 * 56, num_ciso=4 * 56, num_out=4 * 56, type_in=1, type_out=1,
                        in_cut_start=0,
                        in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 整理数据 -- 左边向右发1*56*64的交叠数据
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x9000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=64, length_out=64,
                        length_ciso=1, num_in=56, num_ciso=56, num_out=56, type_in=1, type_out=1, in_cut_start=0,
                        in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            receive_en = 0 if core_x == 0 else 1
            send_en = 0 if core_x == size_x - 1 else 1
            router = p09(rhead_mode=1, send_en=send_en, receive_en=receive_en, send_num=447, receive_num=0,
                         addr_din_base=0x400, addr_din_length=447, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0,
                            A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8400, addr_out=0x5800 >> 2, addr_ciso=0x0000, length_in=64, length_out=64,
                        length_ciso=1, num_in=56, num_ciso=56, num_out=56, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if core_x == 0:
                soma2 = None
            if core_x == size_x - 1:
                soma1 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 整理数据 -- 右边向左发1*56*64的交叠数据
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x6600 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=64, length_out=64,
                        length_ciso=1, num_in=56, num_ciso=56, num_out=56, type_in=1, type_out=1, in_cut_start=0,
                        in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            receive_en = 0 if core_x == size_x - 1 else 1
            send_en = 0 if core_x == 0 else 1
            router = p09(rhead_mode=1, send_en=send_en, receive_en=receive_en, send_num=447, receive_num=0,
                         addr_din_base=0x400, addr_din_length=447, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=447, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8400, addr_out=0x9e00 >> 2, addr_ciso=0x0000, length_in=64, length_out=64,
                        length_ciso=1, num_in=56, num_ciso=56, num_out=56, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if core_x == 0:
                soma1 = None
            if core_x == size_x - 1:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L3 卷积，流水ReLU
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x == 0:
                py = 5
                pad_top, pad_down = 1, 0
                addr_ina = 0x6600 >> 2
            elif core_x == size_x - 1:
                py = 5
                pad_top, pad_down = 0, 1
                addr_ina = 0x5800 >> 2
            else:
                py = 6
                pad_top, pad_down = 0, 0
                addr_ina = 0x5800 >> 2
            axon = p41(px=56, py=py, cin=64, cout=32, kx=3, ky=3, sx=1, sy=1, addr_ina=addr_ina,
                       addr_inb=0x10d00 >> 2, addr_bias=0x10080 >> 2, addr_out=0xac00 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=pad_top, pad_down=pad_down, pad_left=1, pad_right=1,
                       data_x=None, data_w=data['layer1.0.conv2']['weight'][(core_x, core_y)],
                       data_b=data['layer1.0.conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0xac00 >> 2, addr_out=0x17500 >> 2, cin=32, cout=32, px=56, py=4, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer1.0.cut2'],
                        row_ck_on=1,
                        in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 整理数据 -- 上下互发送4*56*32的数据
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x17500 >> 2, addr_out=0x9000, addr_ciso=0x0000, length_in=56 * 32, length_out=56 * 32,
                        length_ciso=1, num_in=4, num_ciso=4, num_out=4, type_in=1, type_out=1, in_cut_start=0,
                        in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=895, receive_num=0, addr_din_base=0x400,
                         addr_din_length=895, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1 if core_y == 0 else -1,
                            A=0, pack_per_Rhead=895, A_offset=0, Const=0, EN=1)
            if core_y == 0:
                addr_in, addr_ciso = 0x17500 >> 2, 0x8400
            else:
                addr_in, addr_ciso = 0x8400, 0x17500 >> 2
            soma2 = p06(addr_in=addr_in, addr_out=0x5800 >> 2, addr_ciso=addr_ciso, length_in=32, length_out=64,
                        length_ciso=32, num_in=4 * 56, num_ciso=4 * 56, num_out=4 * 56, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L4 卷积，流水截取，搬运X2到Mem1
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=16, cin=64, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x5800 >> 2,
                       addr_inb=0x15500 >> 2, addr_bias=0x10100 >> 2, addr_out=0x17500 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0, data_x=None,
                       data_w=data['layer1.0.conv3']['weight'][(core_x, core_y)],
                       data_b=data['layer1.0.conv3']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x17500 >> 2, addr_out=0x9000 >> 2, cin=128, cout=128, px=14, py=16, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer1.0.cut3'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = p06(addr_in=0x2000 >> 2, addr_out=0x17500 >> 2, addr_ciso=0x0000, length_in=64, length_out=64,
                        length_ciso=1, num_in=4 * 56, num_ciso=4 * 56, num_out=4 * 56, type_in=1, type_out=1,
                        in_cut_start=0,
                        in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # L4e卷积，流水截取
    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=14, py=16, cin=64, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x17500 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x10300 >> 2, addr_out=0x1ad00 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0, data_x=None,
                       data_w=data['layer1.0.downsample.0']['weight'][(core_x, core_y)],
                       data_b=data['layer1.0.downsample.0']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1ad00 >> 2, addr_out=0x2000 >> 2, cin=128, cout=128, px=14, py=16,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=cuts['layer1.0.cut4'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 求和，流水ReLU
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=14, py=16,
                       addr_in=0x2000 >> 2, addr_bias=0x0, addr_out=0x17500 >> 2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0)
            soma1 = pX5(mode='max', addr_in=0x17500 >> 2, addr_out=0x9000 >> 2, cin=128, cout=128, px=14, py=16,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['layer1.0.cut5'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00002'

    chip = (0, 1)

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1  # L1卷积计算，流水ReLU
    phase[1] = 1  # 整理数据 -- 上下互发送4*56*32的数据

    phase[2] = 1  # 整理数据 -- 左边向右发1*56*64的交叠数据
    phase[3] = 1  # 整理数据 -- 右边向左发1*56*64的交叠数据
    phase[4] = 1  # L3 卷积，流水ReLU
    phase[5] = 1  # 整理数据 -- 上下互发送4*56*32的数据

    phase[6] = 1  # L4 卷积，流水截取
    phase[7] = 1  # L4e卷积，流水截取
    phase[8] = 1  # 求和，流水ReLU

    from generator.resnet50.resnet50_5chips.G2_data import generate_g2_data

    handler = ResNetDataHandler()
    data = generate_g2_data(handler, size_y=2, size_x=14)

    clock_in_phase = 50_000
    config = gen_g2_map_config(phase, clock_in_phase=clock_in_phase, size_x=14, size_y=2, data=data, in_data_en=False,
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
