import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
from generator.ResidualBlocks.g1_data_gen import generate_data

from itertools import product
import os


PX = 112
PY = 200
C = 32
Si = 40
Sj = 1
px = PX // Sj
py = PY // Si


OVERLAP_LENGTH = 0x380
V1_LENGTH = 4480
X0_LENGTH = 4480
V0_LENGTH = 4480
X1_LENGTH = V0_LENGTH
CACHE_LENGTH = 7168

W0_ADDR = 14080
B0_ADDR = 32736
W1_ADDR = 30432
B1_ADDR = 14048

X0_ADDR = 24160 + OVERLAP_LENGTH  # 0x8000 - OVERLAP_LENGTH - X0_LENGTH
V0_ADDR = 7776 + OVERLAP_LENGTH  # 0x920 + OVERLAP_LENGTH

CACHE_ADDR_0 = 16992

X1_ADDR = V0_ADDR
CACHE_ADDR_1 = 608
V1_ADDR = 20576

CACHE_ADDR_2 = 6880

OUT_ADDR = 20575


def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False):
    """
        Residual Block: Group 1
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

    # ******** 数据交互 ********

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

    # 加载输入
    # 无外部输入情况已通过校对
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # print(len(data['conv0']['input'][(core_x, core_y)]))
            axon = None
            router = None
            if out_data_en or in_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000,
                           addr_inb=0x0000, addr_bias=0x0000, addr_out=0x0000, axon_delay=True,
                           L4_num=13, L5_num=20, A2S2_mode=True)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x400)
                router = p09(rhead_mode=1,
                             send_en=out_data_en,
                             receive_en=in_data_en,
                             send_num=300 * 32 // 8 - 1,
                             receive_num=0,
                             addr_din_base=0x400, addr_din_length=300 * 32 // 8 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=300 * 32 // 16 - 1, soma_in_en=1)
            soma1 = None
            soma2 = None
            if out_data_en:
                soma1 = p06(addr_in=OUT_ADDR,
                            addr_out=0x9000,
                            addr_ciso=0x4000,
                            length_in=32, num_in=300,
                            length_ciso=1, num_ciso=300,
                            length_out=32, num_out=300,
                            type_in=1, type_out=1)
                            # data_in=data['cut2']['output'][(core_x, core_y)][0:2400])
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=5, A=0, pack_per_Rhead=32 * 300 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if init_data:
                soma2 = p06(addr_in=X0_ADDR,
                            addr_out=0x8400,
                            addr_ciso=0x0000,
                            length_in=C, num_in=px * py,
                            length_ciso=1, num_ciso=1,
                            length_out=1, num_out=1,
                            type_in=1, type_out=1,
                            data_in=data['conv0']['input'][(core_x, core_y)])
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=X0_ADDR, addr_ciso=0x4000, length_in=32,
                            num_in=300, length_ciso=1, num_ciso=300, length_out=32, num_out=300,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            router = None
            if out_data_en or in_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x3F00,
                           addr_inb=0x3F00, addr_bias=0x3F00, addr_out=0x3F00, axon_delay=True,
                           L4_num=13, L5_num=20, A2S2_mode=True)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x400)
                router = p09(rhead_mode=1,
                             send_en=out_data_en,
                             receive_en=in_data_en,
                             send_num=300 * 32 // 8 - 1,
                             receive_num=0,
                             addr_din_base=0x400, addr_din_length=300 * 32 // 8 - 1,
                             addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=300 * 32 // 16 - 1, soma_in_en=1)
            soma1 = None
            soma2 = None
            if out_data_en:
                soma1 = p06(addr_in=OUT_ADDR + 0x960,
                            addr_out=0x9000,
                            addr_ciso=0x4000,
                            length_in=32, num_in=300,
                            length_ciso=1, num_ciso=300,
                            length_out=32, num_out=300,
                            type_in=1, type_out=1)
                            # data_in=data['cut2']['output'][(core_x, core_y)][2400:4800])
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=5, A=0, pack_per_Rhead=32 * 300 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=X0_ADDR + 0x960, addr_ciso=0x4000, length_in=32,
                            num_in=300, length_ciso=1, num_ciso=300, length_out=32, num_out=300,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            router = None
            if in_data_en or out_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x3F00,
                           addr_inb=0x3F00, addr_bias=0x3F00, addr_out=0x3F00, axon_delay=True,
                           L4_num=4, L5_num=46, A2S2_mode=True)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x400)
                router = p09(rhead_mode=1,
                             send_en=out_data_en,
                             receive_en=in_data_en,
                             send_num=240 * 32 // 8 - 1,
                             receive_num=0,
                             addr_din_base=0x400, addr_din_length=240 * 32 // 8 - 1,
                             addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=240 * 32 // 16 - 1, soma_in_en=1)
            soma1 = None
            soma2 = None
            if out_data_en:
                soma1 = p06(addr_in=OUT_ADDR + 0x960 + 0x960,
                            addr_out=0x9000,
                            addr_ciso=0x4000,
                            length_in=32, num_in=240,
                            length_ciso=1, num_ciso=240,
                            length_out=32, num_out=240,
                            type_in=1, type_out=1)
                            # data_in=data['cut2']['output'][(core_x, core_y)][4800:6720])
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=5, A=0, pack_per_Rhead=32 * 240 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=X0_ADDR + 0x12C0, addr_ciso=0x4000, length_in=32,
                            num_in=240, length_ciso=1, num_ciso=240, length_out=32, num_out=240,
                            type_in=1, type_out=1)

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    offset = 3
    # ******** 开始计算 ********
    # 发送next数据，接收next数据
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                soma1 = p06(addr_in=X0_ADDR + X0_LENGTH - OVERLAP_LENGTH, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)

            # router = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x380)
            router = p09(rhead_mode=1,
                         send_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
                         receive_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
                         send_num=px * C // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x380, addr_din_length=px * C // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=px * C // 16 - 1, soma_in_en=1)
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                router.addRHead(S=0, T=1, P=0, Q=0,
                                X=core_data_flow[(core_x, core_y)]["next"]["X"],
                                Y=core_data_flow[(core_x, core_y)]["next"]["Y"],
                                A=0, pack_per_Rhead=px * C // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                soma2 = p06(addr_in=0x8380, addr_out=X0_ADDR - OVERLAP_LENGTH, addr_ciso=0x0000, length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # conv0 发送prev数据，接收prev数据
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 =None
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                soma1 = p06(addr_in=X0_ADDR, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)

            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x400)
            router = p09(rhead_mode=1,
                         send_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
                         receive_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
                         send_num=px * C // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=px * C // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=px * C // 16 - 1, soma_in_en=1)
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                router.addRHead(S=0, T=1, P=0, Q=0,
                                X=core_data_flow[(core_x, core_y)]["prev"]["X"],
                                Y=core_data_flow[(core_x, core_y)]["prev"]["Y"],
                                A=0, pack_per_Rhead=px * C // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                soma2 = p06(addr_in=0x8400, addr_out=X0_ADDR + X0_LENGTH, addr_ciso=0x0000, length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})


    # Conv0 relu
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x == 0 and core_y == 0):
                axon = p41(px=px, py=py+1, cin=C, cout=C, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X0_ADDR, addr_inb=W0_ADDR, addr_bias=B0_ADDR, addr_out=CACHE_ADDR_0,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=1, pad_down=0, pad_left=1,
                           data_w=data['conv0']['weight'][(core_x, core_y)],
                           data_b=data['conv0']['bias'][(core_x, core_y)])
            elif (core_x == 7 and core_y == 4):
                axon = p41(px=px, py=py+1, cin=C, cout=C, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X0_ADDR-OVERLAP_LENGTH, addr_inb=W0_ADDR, addr_bias=B0_ADDR, addr_out=CACHE_ADDR_0,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=0, pad_down=1, pad_left=1,
                           data_w=data['conv0']['weight'][(core_x, core_y)],
                           data_b=data['conv0']['bias'][(core_x, core_y)])
            else:
                axon = p41(px=px, py=py+2, cin=C, cout=C, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X0_ADDR-OVERLAP_LENGTH, addr_inb=W0_ADDR, addr_bias=B0_ADDR, addr_out=CACHE_ADDR_0,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=0, pad_down=0, pad_left=1,
                           data_w=data['conv0']['weight'][(core_x, core_y)],
                           data_b=data['conv0']['bias'][(core_x, core_y)])
            soma1 = None
            soma1 = pX5(mode='max', addr_in=CACHE_ADDR_0, addr_out=V0_ADDR,
                        cin=C, cout=C, px=px, py=py,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=3,
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # conv1卷积的交叠数据: 向后发一行
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1= None
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                soma1 = p06(addr_in=X1_ADDR + X1_LENGTH - OVERLAP_LENGTH, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)

            # print(core_x, core_y)
            # print("send", core_data_flow[(core_x, core_y)]["next"])
            # print("recv", (len(core_data_flow[(core_x, core_y)]["prev"]) != 0))

            # router = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x400)
            router = p09(rhead_mode=1,
                         send_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
                         receive_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
                         send_num=px * C // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=px * C // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=px*C//16 - 1, soma_in_en=1)
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                router.addRHead(S=0, T=1, P=0, Q=0,
                                X=core_data_flow[(core_x, core_y)]["next"]["X"],
                                Y=core_data_flow[(core_x, core_y)]["next"]["Y"],
                                A=0, pack_per_Rhead=px * C // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                soma2 = p06(addr_in=0x8400, addr_out=X1_ADDR - OVERLAP_LENGTH, addr_ciso=0x4000, length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})


    # conv1 发送prev数据，接收prev数据
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                soma1 = p06(addr_in=X1_ADDR, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)

            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x400)
            router = p09(rhead_mode=1,
                         send_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
                         receive_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
                         send_num=px * C // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=px * C // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=px*C//16 - 1, soma_in_en=1)
            if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
                router.addRHead(S=0, T=1, P=0, Q=0,
                                X=core_data_flow[(core_x, core_y)]["prev"]["X"],
                                Y=core_data_flow[(core_x, core_y)]["prev"]["Y"],
                                A=0, pack_per_Rhead=px * C // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
                soma2 = p06(addr_in=0x8400, addr_out=X1_ADDR + X1_LENGTH, addr_ciso=0x4000,
                            length_in=C,
                            num_in=px, length_ciso=1, num_ciso=px, length_out=C, num_out=px,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append(
                {'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # conv1 relu
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x == 0 and core_y == 0):
                axon = p41(px=px, py=py+1, cin=C, cout=C, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X1_ADDR, addr_inb=W1_ADDR, addr_bias=B1_ADDR, addr_out=CACHE_ADDR_1,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=1, pad_down=0, pad_left=1,
                           data_w=data['conv1']['weight'][(core_x, core_y)],
                           data_b=data['conv1']['bias'][(core_x, core_y)])
            elif (core_x == 7 and core_y == 4):
                axon = p41(px=px, py=py+1, cin=C, cout=C, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X1_ADDR - OVERLAP_LENGTH, addr_inb=W1_ADDR, addr_bias=B1_ADDR, addr_out=CACHE_ADDR_1,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=0, pad_down=1, pad_left=1,
                           data_w=data['conv1']['weight'][(core_x, core_y)],
                           data_b=data['conv1']['bias'][(core_x, core_y)])
            else:
                axon = p41(px=px, py=py+2, cin=C, cout=C, kx=3, ky=3, sx=1, sy=1,
                           addr_ina=X1_ADDR - OVERLAP_LENGTH, addr_inb=W1_ADDR, addr_bias=B1_ADDR, addr_out=CACHE_ADDR_1,
                           ina_type=1, inb_type=1, load_bias=2,
                           pad_right=1, pad_top=0, pad_down=0, pad_left=1,
                           data_w=data['conv1']['weight'][(core_x, core_y)],
                           data_b=data['conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=CACHE_ADDR_1, addr_out=V1_ADDR,
                        cin=C, cout=C, px=px, py=py,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=3,
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 求和，流水截取
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=C, px=px, py=py,
                       addr_in=V1_ADDR, addr_bias=0x0, addr_out=CACHE_ADDR_2, pad_top=0, pad_down=0, pad_left=0,
                       pad_right=0, bias_length=0, constant_b=0)

            # 不知道为什么跑不通，用X5替代
            # soma1 = p06(addr_in=0x2030, addr_out=(0x10000 >> 2) + 0x9B0, addr_ciso=0x8400,
            #             length_in=32, num_in=16,
            #             length_ciso=1, num_ciso=16,
            #             length_out=32, num_out=16,
            #             type_in=0, type_out=1,
            #             in_cut_start=1,
            #             row_ck_on=0, in_row_max=2, data_in=[], data_ciso=[])

            soma1 = pX5(mode='max', addr_in=CACHE_ADDR_2, addr_out=20500,
                        cin=C, cout=C, px=px, py=py,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=0,
                        row_ck_on=1, in_row_max=2)
            # soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})


    return map_config


if __name__ == '__main__':

    case_file_name = 'residual_block_gu'
    chip = (0, 0)

    handler = ModelDataHandler(model='real')
    g1_data = generate_data(handler, 8, 5)

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

    # phase[:] = 1

    clock_in_phase = 20_0000
    config_1 = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5, in_cut_start_dict=None,
                                in_data_en=False, out_data_en=False, chip=chip, data=g1_data, init_data=True)
    config.add_config(config_1, core_offset=(0, 0))

    MapConfigGen.add_router_info(map_config=config.map_config)

    # prim = {
    #     'axon': None, 'soma1': None, 'router': None,
    #     'soma2': p06(addr_in=0x7fff, addr_out=0x8400, addr_ciso=0x7fff >> 2, length_in=1,
    #                  num_in=1, length_ciso=1, num_ciso=1, length_out=1, num_out=1,
    #                  type_in=1, type_out=1, data_in=None)
    # }
    #
    # MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    config.map_config['sim_clock'] = 200_000
    config.map_config['step_clock'] = {
        ((0, 0), 0): (200_000 - 1, 200_000)
    }

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
