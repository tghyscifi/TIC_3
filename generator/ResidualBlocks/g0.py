import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.ResidualBlocks.g1_data_gen import generate_data
from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
from generator.ResidualBlocks.g1 import gen_1_map_config
from itertools import product


W0_ADDR = 0x0000
B0_ADDR = 0x120
W1_ADDR = (0x10000 >> 2)
B1_ADDR = (0x10000 >> 2) + 0x900

LUT_ADDR = 0x140

X0_ADDR = (0x10000 >> 2) + 0x920 + 0x1A40
V0_ADDR = 0x180

CACHE_ADDR = (0x10000 >> 2) + 0x920 + 0x1A40 + 0x400

X1_ADDR = 0x180 + 0x1A40 + 0x1E0
V1_ADDR = (0x10000 >> 2) + 0x920

OVERLAP_LENGTH= 0x1E0
V1_LENGTH = 0x1A40
X0_LENGTH = 0x400
V0_LENGTH = 0x1A40
X1_LENGTH = 0x1A40
OUT_LENGTH = V1_LENGTH // 2



def gen_0_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, out_data_en=True, chip=(0, 0),
                     in_data_en=False, last_layer_in_data_en=False, last_layer_out_data_en=False):
    """
        Residual Block: Group 0
        Data Source
        core_x * core_y: 8 * 5
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

    # core间数据流配置
    core_data_flow = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        core_data_flow[(core_x, core_y)] = {
            "prev": {},
            "next": {}
        }
        if core_x == 0 and core_y == 0:
            core_data_flow[(core_x, core_y)]["prev"] = {}
            core_data_flow[(core_x, core_y)]["next"]["X"] = 1
            core_data_flow[(core_x, core_y)]["next"]["Y"] = 0
        elif core_y == 4 and core_x == 7:
            core_data_flow[(core_x, core_y)]["next"] = {}
            core_data_flow[(core_x, core_y)]["prev"]["X"] = -1
            core_data_flow[(core_x, core_y)]["prev"]["Y"] = 0
        elif core_y % 2 == 0 and 7 > core_x > 0:
            core_data_flow[(core_x, core_y)]["prev"]["X"] = -1
            core_data_flow[(core_x, core_y)]["prev"]["Y"] = 0
            core_data_flow[(core_x, core_y)]["next"]["X"] = 1
            core_data_flow[(core_x, core_y)]["next"]["Y"] = 0
        elif (core_y % 2 == 1) and 7 > core_x > 0:
            core_data_flow[(core_x, core_y)]["prev"]["X"] = 1
            core_data_flow[(core_x, core_y)]["prev"]["Y"] = 0
            core_data_flow[(core_x, core_y)]["next"]["X"] = -1
            core_data_flow[(core_x, core_y)]["next"]["Y"] = 0
        elif core_y % 2 == 0 and core_x == 0:
            core_data_flow[(core_x, core_y)]["prev"]["X"] = 0
            core_data_flow[(core_x, core_y)]["prev"]["Y"] = -1
            core_data_flow[(core_x, core_y)]["next"]["X"] = 1
            core_data_flow[(core_x, core_y)]["next"]["Y"] = 0
        elif core_y % 2 == 0 and core_x == 7:
            core_data_flow[(core_x, core_y)]["prev"]["X"] = -1
            core_data_flow[(core_x, core_y)]["prev"]["Y"] = 0
            core_data_flow[(core_x, core_y)]["next"]["X"] = 0
            core_data_flow[(core_x, core_y)]["next"]["Y"] = 1
        elif core_y % 2 == 1 and core_x == 0:
            core_data_flow[(core_x, core_y)]["prev"]["X"] = 1
            core_data_flow[(core_x, core_y)]["prev"]["Y"] = 0
            core_data_flow[(core_x, core_y)]["next"]["X"] = 0
            core_data_flow[(core_x, core_y)]["next"]["Y"] = 1
        elif core_y % 2 == 1 and core_x == 7:
            core_data_flow[(core_x, core_y)]["prev"]["X"] = 0
            core_data_flow[(core_x, core_y)]["prev"]["Y"] = -1
            core_data_flow[(core_x, core_y)]["next"]["X"] = -1
            core_data_flow[(core_x, core_y)]["next"]["Y"] = 0

    # 核间数据流显示
    # for core_y in range(size_y):
    #     if core_y % 2 == 1:
    #         for core_x in range(size_x-1, -1, -1):
    #             print((core_x, core_y), core_data_flow[(core_x, core_y)])
    #     else:
    #         for core_x in range(size_x):
    #             print((core_x, core_y), core_data_flow[(core_x, core_y)])

    # ******** 数据收发 ********
    # 向后续的residual block发送
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=13, L5_num=20, A2S2_mode=True)
            soma1 = p06(addr_in=V0_ADDR, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=32,
                        num_in=300, length_ciso=1, num_ciso=300, length_out=32, num_out=300,
                        type_in=1, type_out=1)
                        # data_in=data['conv0']['input'][(core_x, core_y)][0:2400])
            if (not out_data_en) and (not last_layer_in_data_en) and (not in_data_en):
                axon = None
            if (not out_data_en):
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=last_layer_in_data_en, send_num=32 * 300 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=32 * 300 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=32 * 300 // 16 - 1, soma_in_en=1)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-8, Y=0, A=0, pack_per_Rhead=32 * 300 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            if last_layer_in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=X1_ADDR, addr_ciso=(0x0000 >> 2), length_in=32,
                            num_in=300, length_ciso=1, num_ciso=300, length_out=32, num_out=300,
                            type_in=1, type_out=1)
            else:
                soma2 = p06(addr_in=X1_ADDR, addr_out=0x8400, addr_ciso=(0x0000 >> 2), length_in=32,
                            num_in=840, length_ciso=1, num_ciso=1, length_out=1, num_out=1,
                            type_in=1, type_out=1,
                            data_in=data['cut2']['output'][(core_x, core_y)])
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=13, L5_num=20, A2S2_mode=True)
            soma1 = p06(addr_in=V0_ADDR+0x960, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=32,
                        num_in=300, length_ciso=1, num_ciso=300, length_out=32, num_out=300,
                        type_in=1, type_out=1)
                        # data_in=data['conv0']['input'][(core_x, core_y)][2400:4800])
            if (not out_data_en) and (not last_layer_in_data_en) and (not in_data_en):
                axon = None
            if (not out_data_en):
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=last_layer_in_data_en, send_num=32 * 300 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=32 * 300 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=32 * 300 // 16 - 1, soma_in_en=1)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-8, Y=0, A=0, pack_per_Rhead=32 * 300 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            if last_layer_in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=X1_ADDR + 0x960, addr_ciso=(0x0000 >> 2), length_in=32,
                            num_in=300, length_ciso=1, num_ciso=300, length_out=32, num_out=300,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=4, L5_num=46, A2S2_mode=True)
            soma1 = p06(addr_in=V0_ADDR+0x960+0x960, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=32,
                        num_in=240, length_ciso=1, num_ciso=240, length_out=32, num_out=240,
                        type_in=1, type_out=1)
                        # data_in=data['conv0']['input'][(core_x, core_y)][4800:6720])
            if (not out_data_en) and (not last_layer_in_data_en) and (not in_data_en):
                axon = None
            if (not out_data_en):
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=last_layer_in_data_en, send_num=32 * 240 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=32 * 240 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=32 * 240 // 16 - 1, soma_in_en=1)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-8, Y=0, A=0, pack_per_Rhead=32 * 240 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            if last_layer_in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=X1_ADDR+0x960+0x960, addr_ciso=(0x0000 >> 2), length_in=32,
                            num_in=240, length_ciso=1, num_ciso=240, length_out=32, num_out=240,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # 接收或初始化输入
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            router = None
            soma2 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=False, receive_en=in_data_en, send_num=0,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=64 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1)
            if not in_data_en:
                axon = None
                soma1 = p06(addr_in=X0_ADDR,
                            addr_out=0x8400,
                            addr_ciso=0x0000,
                            length_in=64, num_in=64,
                            length_ciso=1, num_ciso=1,
                            length_out=1, num_out=1,
                            type_in=1, type_out=1,
                            data_in=data['conv_in']['input'][(core_x, core_y)])
                router = None
                soma2 = None
            else:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x3F00,
                           addr_inb=0x3F00, addr_bias=0x3F00, addr_out=0x3F00, axon_delay=True,
                           L4_num=9, L5_num=223, A2S2_mode=True)
                soma2 = p06(addr_in=0x8400,
                            addr_out=X0_ADDR,
                            addr_ciso=0x4000,
                            length_in=64, num_in=64,
                            length_ciso=1, num_ciso=64,
                            length_out=64, num_out=64,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            router = None
            soma2 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=last_layer_out_data_en, receive_en=False, send_num=16 * 420 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=64 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1)
            if last_layer_out_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                           addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                           L4_num=68, L5_num=52, A2S2_mode=True)
                soma1 = p06(addr_in=X1_ADDR - OVERLAP_LENGTH,
                            addr_out=0x9000,
                            addr_ciso=0x0000,
                            length_in=16, num_in=60 * 7,
                            length_ciso=1, num_ciso=60 * 7,
                            length_out=16, num_out=60 * 7,
                            type_in=1, type_out=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-5, A=0, pack_per_Rhead=16 * 420 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            router = None
            soma2 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=last_layer_out_data_en, receive_en=False, send_num=16 * 420 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=64 * 64 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1)
            if last_layer_out_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                           addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                           L4_num=68, L5_num=52, A2S2_mode=True)
                soma1 = p06(addr_in=X1_ADDR - OVERLAP_LENGTH + OUT_LENGTH // 2,
                            addr_out=0x9000,
                            addr_ciso=0x0000,
                            length_in=16, num_in=60 * 7,
                            length_ciso=1, num_ciso=60 * 7,
                            length_out=16, num_out=60 * 7,
                            type_in=1, type_out=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-5, A=0, pack_per_Rhead=16 * 420 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    offset = 6
    # ******** 开始计算 ********
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p81(px=62, py=16, cin=4, cout=32, kx=3, ky=3, sx=1, sy=1,
                       addr_ina=X0_ADDR, addr_inb=W0_ADDR, addr_bias=B0_ADDR, addr_out=CACHE_ADDR,
                       ina_type=1, inb_type=1, load_bias=2,
                       data_w=data['conv_in']['weight'][(core_x, core_y)],
                       data_b=data['conv_in']['bias'][(core_x, core_y)]
                       )
            # axon = None
            soma1 = pX5(mode='max', addr_in=CACHE_ADDR, addr_out=V0_ADDR,
                        cin=32, cout=32, px=60, py=14,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=4,
                        row_ck_on=1, in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 发送next数据，接收next数据
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                soma1 = p06(addr_in=X1_ADDR + X1_LENGTH - OVERLAP_LENGTH, addr_out=0x9000, addr_ciso=0x10000 >> 2,
                            length_in=32, num_in=60, length_ciso=1, num_ciso=60, length_out=32, num_out=60,
                            type_in=1, type_out=1)

            # router = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1,
                         send_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
                         receive_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
                         send_num=60 * 32 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x380, addr_din_length=60 * 32 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=60 * 32 // 16 - 1,
                         soma_in_en=1)
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                router.addRHead(S=0, T=1, P=0, Q=0,
                                X=core_data_flow[(core_x, core_y)]["next"]["X"],
                                Y=core_data_flow[(core_x, core_y)]["next"]["Y"],
                                A=0, pack_per_Rhead=60 * 32 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                soma2 = p06(addr_in=0x8380, addr_out=X1_ADDR - OVERLAP_LENGTH, addr_ciso=0x4000, length_in=32,
                            num_in=60, length_ciso=1, num_ciso=60, length_out=32, num_out=60,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append(
                {'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # conv0 发送prev数据，接收prev数据
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                soma1 = p06(addr_in=X1_ADDR, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=32,
                            num_in=60, length_ciso=1, num_ciso=60, length_out=32, num_out=60,
                            type_in=1, type_out=1)

            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x400)
            router = p09(rhead_mode=1,
                         send_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
                         receive_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
                         send_num=60 * 32 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=60 * 32 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=60 * 32 // 16 - 1,
                         soma_in_en=1)
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                router.addRHead(S=0, T=1, P=0, Q=0,
                                X=core_data_flow[(core_x, core_y)]["prev"]["X"],
                                Y=core_data_flow[(core_x, core_y)]["prev"]["Y"],
                                A=0, pack_per_Rhead=60 * 32 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                soma2 = p06(addr_in=0x8400, addr_out=X1_ADDR + X1_LENGTH, addr_ciso=0x4000,
                            length_in=32,
                            num_in=60, length_ciso=1, num_ciso=60, length_out=32, num_out=60,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append(
                {'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # conv_out

    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x == 0 and core_y == 0):
                axon = p41(px=60, py=15, cin=32, cout=32, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X1_ADDR, addr_inb=W1_ADDR, addr_bias=B1_ADDR, addr_out=CACHE_ADDR,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=1, pad_down=0, pad_left=1,
                           data_w=data['conv_out']['weight'][(core_x, core_y)],
                           data_b=data['conv_out']['bias'][(core_x, core_y)])
            elif (core_x == 7 and core_y == 4):
                axon = p41(px=60, py=15, cin=32, cout=32, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X1_ADDR - OVERLAP_LENGTH, addr_inb=W1_ADDR, addr_bias=B1_ADDR, addr_out=CACHE_ADDR,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=0, pad_down=1, pad_left=1,
                           data_w=data['conv_out']['weight'][(core_x, core_y)],
                           data_b=data['conv_out']['bias'][(core_x, core_y)])
            else:
                axon = p41(px=60, py=16, cin=32, cout=32, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X1_ADDR - OVERLAP_LENGTH, addr_inb=W1_ADDR, addr_bias=B1_ADDR, addr_out=CACHE_ADDR,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=0, pad_down=0, pad_left=1,
                           data_w=data['conv_out']['weight'][(core_x, core_y)],
                           data_b=data['conv_out']['bias'][(core_x, core_y)])

            soma1 = pX5(mode='max', addr_in=CACHE_ADDR, addr_out=V1_ADDR,
                        cin=32, cout=16, px=60, py=14,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=5,
                        row_ck_on=1, in_row_max=3)
            router = None
            # soma2 = None
            soma2 = p07(addr_in=V1_ADDR,
                        addr_out=X1_ADDR - OVERLAP_LENGTH,
                        group_num=14*60,
                        neuron_num=16,
                        lut_dw=1,
                        type_in=1, type_out=1,
                        addr_lut=0x140,
                        data_lut=data['activation_out']['lut'])
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})





    return map_config


if __name__ == '__main__':

    case_file_name = 'residual_block_g0'
    chip_0 = (0, 0)

    handler = ModelDataHandler()
    data = generate_data(handler, 8, 5)

    # send_to_fpga = True

    config = MapConfigGen()

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1
    phase[4] = 1
    phase[5] = 1
    phase[6] = 1
    phase[7] = 1
    phase[8] = 1
    phase[9] = 1
    # phase[10] = 1

    # phase[:] = 1

    clock_in_phase = 20_0000
    config_0 = gen_0_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5,
                                in_data_en=False, out_data_en=True, chip=chip_0, data=data, last_layer_in_data_en=True)

    config.add_config(config_0, core_offset=(8, 0))
    #
    config_1 = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5, in_cut_start_dict=None,
                                in_data_en=True, out_data_en=True, chip=chip_0, data=data, init_data=False)
    config.add_config(config_1, core_offset=(0, 0))

    MapConfigGen.add_router_info(map_config=config.map_config)

    # prim = {
    #     'axon': None, 'soma1': None, 'router': None,
    #     'soma2': p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x10000 >> 2, length_in=1024,
    #                  num_in=12, length_ciso=1, num_ciso=12, length_out=1024, num_out=12,
    #                  type_in=1, type_out=1, data_in=None)
    # }
    #
    # MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)


    clock_0 = 25_0000 - 1
    clock_1 = 25_0000
    step_exe_number = 1
    config.sim_clock = clock_1 * 3

    MapConfigGen.set_step_clock(config.map_config, clock_0=clock_0, clock_1=clock_1)
    MapConfigGen.set_step_exe_number(config.map_config, step_exe_number)

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

    tester = TestEngine(config.map_config, test_config)
    assert tester.run_test()

