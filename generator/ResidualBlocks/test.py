import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.ResidualBlocks.g1_data_gen import generate_data
from generator.ResidualBlocks.test_data_gen import generate_test_data
from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
from generator.ResidualBlocks.g1 import gen_1_map_config
from itertools import product



def gen_test_map_config(phase_en, clock_in_phase, size_x, size_y, data, chip=(0, 0)):

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
    core_data_flow[(0, 0)] = {
        "prev": {},
        "next": {}
    }
    core_data_flow[(0, 0)]["prev"]["X"] = 1
    core_data_flow[(0, 0)]["prev"]["Y"] = 0
    core_data_flow[(0, 0)]["next"]["X"] = 0
    core_data_flow[(0, 0)]["next"]["Y"] = 1

    core_data_flow[(0, 1)] = {
        "prev": {},
        "next": {}
    }
    core_data_flow[(0, 1)]["prev"]["X"] = 0
    core_data_flow[(0, 1)]["prev"]["Y"] = -1
    core_data_flow[(0, 1)]["next"]["X"] = 1
    core_data_flow[(0, 1)]["next"]["Y"] = 0

    core_data_flow[(1, 1)] = {
        "prev": {},
        "next": {}
    }
    core_data_flow[(1, 1)]["prev"]["X"] = -1
    core_data_flow[(1, 1)]["prev"]["Y"] = 0
    core_data_flow[(1, 1)]["next"]["X"] = 0
    core_data_flow[(1, 1)]["next"]["Y"] = -1

    core_data_flow[(1, 0)] = {
        "prev": {},
        "next": {}
    }
    core_data_flow[(1, 0)]["prev"]["X"] = 0
    core_data_flow[(1, 0)]["prev"]["Y"] = 1
    core_data_flow[(1, 0)]["next"]["X"] = -1
    core_data_flow[(1, 0)]["next"]["Y"] = 0

    # ******** 数据收发 ********
    # C = 32
    # W = 8
    # H = 24
    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p26(addr_in=0x0000, addr_out=0x8400, addr_ciso=0x10000 >> 2,
    #                     length_in=W * C, num_in=H,
    #                     length_ciso=32, num_ciso=H,
    #                     length_out=C, num_out=H,
    #                     type_in=1, type_out=1, data_in=[])
    #
    #         router = None
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})




    # 列交叠
    C = 32
    W = 6
    H = 7
    W_overlapped = W + 2

    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C,
    #                     num_in=H * W, length_ciso=1, num_ciso=H * W, length_out=C, num_out=H * W,
    #                     type_in=1, type_out=1, data_in=[])
    #                     # data_in=data['nonoverlapping'][(0, 0)])
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #         router = p09(rhead_mode=1, send_en=True, receive_en=True, send_num=C * H * W // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=(C * H * W_overlapped) // 8 - 1, addr_rhead_base=addr_rhead_base,
    #                      addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=C * H * W // 16 - 1, soma_in_en=1)
    #
    #         router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=C // 8, pack_per_Rhead=C * H * W // 8 - 1,
    #                         A_offset=(C + C) // 8, Const=W * C // 8 - 1, EN=1)
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})
    #
    # # 发送prev
    # if phase_en[1]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p26(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2,
    #                     length_in=W * C, num_in=H,
    #                     length_ciso=1, num_ciso=H,
    #                     length_out=C, num_out=H,
    #                     type_in=1, type_out=1)
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #         router = p09(rhead_mode=1, send_en=True, receive_en=True,
    #                      send_num=C * H // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=C * H * W_overlapped // 8 - 1,
    #                      addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
    #                      addr_dout_base=0x1000, addr_dout_length=C * H // 16 - 1, soma_in_en=1)
    #
    #         router.addRHead(S=0, T=1, P=0, Q=0,
    #                         X=core_data_flow[(core_x, core_y)]['prev']['X'],
    #                         Y=core_data_flow[(core_x, core_y)]['prev']['Y'],
    #                         A=0, pack_per_Rhead=C * H // 8 - 1,
    #                         A_offset=((W + 1) * C) // 8, Const=C // 8 - 1, EN=1)
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})
    # #
    # # 发送next
    # if phase_en[2]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p26(addr_in=0x0000 + (((W - 1) * C) >> 2), addr_out=0x9000, addr_ciso=0x10000 >> 2,
    #                     length_in=W * C, num_in=H,
    #                     length_ciso=1, num_ciso=H,
    #                     length_out=C, num_out=H,
    #                     type_in=1, type_out=1)
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #         router = p09(rhead_mode=1, send_en=True, receive_en=True,
    #                      send_num=C * H // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=C * H * W_overlapped // 8 - 1,
    #                      addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
    #                      addr_dout_base=0x1000, addr_dout_length=C * H // 16 - 1, soma_in_en=1)
    #
    #         router.addRHead(S=0, T=1, P=0, Q=0,
    #                         X=core_data_flow[(core_x, core_y)]['next']['X'],
    #                         Y=core_data_flow[(core_x, core_y)]['next']['Y'],
    #                         A=((W + 1) * C) // 8, pack_per_Rhead=C * H // 8 - 1,
    #                         A_offset=((W + 1) * C) // 8, Const=C // 8 - 1, EN=1)
    #
    #         soma2 = p06(addr_in=0x8400, addr_out=0x0000, addr_ciso=0x4000,
    #                     length_in=C, num_in=H * W_overlapped,
    #                     length_ciso=1, num_ciso=H * W_overlapped,
    #                     length_out=C, num_out=H * W_overlapped,
    #                     type_in=1, type_out=1)
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # # 发送prev
    # if phase_en[1]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p26(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2,
    #                     length_in=W * C, num_in=H,
    #                     length_ciso=1, num_ciso=H,
    #                     length_out=C, num_out=H,
    #                     type_in=1, type_out=1)
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #         router = p09(rhead_mode=1, send_en=True, receive_en=True,
    #                      send_num=C * H // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=C * H // 8 - 1,
    #                      addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
    #                      addr_dout_base=0x1000, addr_dout_length=C * H // 16 - 1, soma_in_en=1)
    #
    #         router.addRHead(S=0, T=1, P=0, Q=0,
    #                         X=core_data_flow[(core_x, core_y)]['prev']['X'],
    #                         Y=core_data_flow[(core_x, core_y)]['prev']['Y'],
    #                         A=0, pack_per_Rhead=C * H // 8 - 1,
    #                         A_offset=0, Const=0, EN=1)
    #         soma2 = p06(addr_in=0x8400, addr_out=0x2000, addr_ciso=0x0000,
    #                     length_in=C, num_in=H,
    #                     length_ciso=W * C, num_ciso=H,
    #                     length_out=17 * C, num_out=H,
    #                     type_in=1, type_out=1)
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # 行交叠
    # C = 32
    # W = 20
    # H = 16
    # H_overlapped = H + 2
    #
    # OVERLAP_LENGTH = (W * C) >> 2
    # X0_ADDR = 0x0000 + OVERLAP_LENGTH
    # X0_LENGTH = (C * W * H) >> 2

    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = None
    #         if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
    #             soma1 = p06(addr_in=X0_ADDR + X0_LENGTH - OVERLAP_LENGTH, addr_out=0x9000, addr_ciso=0x10000 >> 2,
    #                         length_in=C, num_in=W,
    #                         length_ciso=1, num_ciso=W,
    #                         length_out=C, num_out=W,
    #                         type_in=1, type_out=1)
    #
    #         # router = None
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x380)
    #         router = p09(rhead_mode=1,
    #                      send_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
    #                      receive_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
    #                      send_num=W * C // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x380, addr_din_length=W * C // 8 - 1, addr_rhead_base=addr_rhead_base,
    #                      addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=W*C//16-1, soma_in_en=1)
    #         if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
    #             router.addRHead(S=0, T=1, P=0, Q=0,
    #                             X=core_data_flow[(core_x, core_y)]["next"]["X"],
    #                             Y=core_data_flow[(core_x, core_y)]["next"]["Y"],
    #                             A=0, pack_per_Rhead=W * C // 8 - 1,
    #                             A_offset=0, Const=0, EN=1)
    #         if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
    #             soma2 = p06(addr_in=0x8380, addr_out=X0_ADDR - OVERLAP_LENGTH, addr_ciso=0x4000,
    #                         length_in=C, num_in=W,
    #                         length_ciso=1, num_ciso=W,
    #                         length_out=C, num_out=W,
    #                         type_in=1, type_out=1)
    #         else:
    #             soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})
    #
    #
    # if phase_en[1]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 =None
    #         if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
    #             soma1 = p06(addr_in=X0_ADDR, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C,
    #                         num_in=W, length_ciso=1, num_ciso=W, length_out=C, num_out=W,
    #                         type_in=1, type_out=1)
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'], limit=0x400)
    #         router = p09(rhead_mode=1,
    #                      send_en=(len(core_data_flow[(core_x, core_y)]["prev"]) != 0),
    #                      receive_en=(len(core_data_flow[(core_x, core_y)]["next"]) != 0),
    #                      send_num=W * C // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=W * C // 8 - 1, addr_rhead_base=addr_rhead_base,
    #                      addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=W * C // 16 - 1, soma_in_en=1)
    #         if len(core_data_flow[(core_x, core_y)]["prev"]) != 0:
    #             router.addRHead(S=0, T=1, P=0, Q=0,
    #                             X=core_data_flow[(core_x, core_y)]["prev"]["X"],
    #                             Y=core_data_flow[(core_x, core_y)]["prev"]["Y"],
    #                             A=0, pack_per_Rhead=W * C // 8 - 1,
    #                             A_offset=0, Const=0, EN=1)
    #         if len(core_data_flow[(core_x, core_y)]["next"]) != 0:
    #             soma2 = p06(addr_in=0x8400, addr_out=X0_ADDR + X0_LENGTH, addr_ciso=0x4000, length_in=C,
    #                         num_in=W, length_ciso=1, num_ciso=W, length_out=C, num_out=W,
    #                         type_in=1, type_out=1)
    #         else:
    #             soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # 输出通道拆分
    C = 32
    W = 10
    H = 12
    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         if core_x == 0:
    #             soma1 = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C//2,
    #                         num_in=H * W, length_ciso=1, num_ciso=H * W, length_out=C//2, num_out=H * W,
    #                         type_in=1, type_out=1, data_in=[])
    #                         # data_in=data['test_out1'][(0, 0)])
    #         else:
    #             soma1 = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C//2,
    #                         num_in=H * W, length_ciso=1, num_ciso=H * W, length_out=C//2, num_out=H * W,
    #                         type_in=1, type_out=1, data_in=[])
    #                         # data_in=data['test_out2'][(0, 0)])
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #         router = p09(rhead_mode=1, send_en=True, receive_en=True, send_num=C // 2 * H * W // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=(C * H * W) // 8 - 1, addr_rhead_base=addr_rhead_base,
    #                      addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=C // 2 * H * W // 16 - 1, soma_in_en=1)
    #
    #         if core_x == 0:
    #             router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=0, pack_per_Rhead=C // 2 * H * W // 8 - 1,
    #                             A_offset=(C // 2) // 8, Const=C // 2 // 8 - 1, EN=1)
    #         else:
    #             router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=0, A=(C // 2) // 8, pack_per_Rhead=C // 2 * H * W // 8 - 1,
    #                             A_offset=(C // 2) // 8, Const=C // 2 // 8 - 1, EN=1)
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})
    #
    # if phase_en[1]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C//2,
    #                     num_in=H * W, length_ciso=1, num_ciso=H * W, length_out=C//2, num_out=H * W,
    #                     type_in=1, type_out=1)
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #         router = p09(rhead_mode=1, send_en=True, receive_en=True, send_num=C // 2 * H * W // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=(C * H * W) // 8 - 1, addr_rhead_base=addr_rhead_base,
    #                      addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=C // 2 * H * W // 16 - 1, soma_in_en=1)
    #
    #         if core_x == 0:
    #             router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=C // 2 * H * W // 8 - 1,
    #                             A_offset=(C // 2) // 8, Const=C // 2 // 8 - 1, EN=1)
    #         else:
    #             router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=(C // 2) // 8, pack_per_Rhead=C // 2 * H * W // 8 - 1,
    #                             A_offset=(C // 2) // 8, Const=C // 2 // 8 - 1, EN=1)
    #
    #         soma2 = None
    #         soma2 = p06(addr_in=0x8400, addr_out=0x0000, addr_ciso=0x10000 >> 2, length_in=C,
    #                     num_in=H * W, length_ciso=1, num_ciso=H * W, length_out=C, num_out=H * W,
    #                     type_in=1, type_out=1)
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # 直接搬移发送
    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         if core_x == 0:
    #             soma1 = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C//2,
    #                         num_in=H * W, length_ciso=1, num_ciso=H * W, length_out=C//2, num_out=H * W,
    #                         type_in=1, type_out=1, data_in=[])
    #                         # data_in=data['test_out1'][(0, 0)])
    #         else:
    #             soma1 = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=C//2,
    #                         num_in=H * W, length_ciso=1, num_ciso=H * W, length_out=C//2, num_out=H * W,
    #                         type_in=1, type_out=1, data_in=[])
    #                         # data_in=data['test_out2'][(0, 0)])
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #         router = p09(rhead_mode=1, send_en=True, receive_en=True, send_num=C // 2 * H * W // 8 - 1,
    #                      receive_num=0,
    #                      addr_din_base=0x400, addr_din_length=(C // 2 * H * W) // 8 - 1, addr_rhead_base=addr_rhead_base,
    #                      addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=C // 2 * H * W // 16 - 1, soma_in_en=1)
    #
    #         if core_x == 0:
    #             router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=C // 2 * H * W // 8 - 1,
    #                             A_offset=0, Const=0, EN=1)
    #         else:
    #             router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=C // 2 * H * W // 8 - 1,
    #                             A_offset=0, Const=0, EN=1)
    #         soma2 = None
    #         if core_x == 0:
    #             soma2 = p06(addr_in=0x0000, addr_out=0x4000, addr_ciso=0x8400, length_in=C//2,
    #                         num_in=H * W, length_ciso=C//2, num_ciso=H * W, length_out=C, num_out=H * W,
    #                         type_in=1, type_out=1)
    #         else:
    #             soma2 = p06(addr_in=0x8400, addr_out=0x4000, addr_ciso=0x0000, length_in=C // 2,
    #                         num_in=H * W, length_ciso=C // 2, num_ciso=H * W, length_out=C, num_out=H * W,
    #                         type_in=1, type_out=1)
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # conv-relu
    # PX = 112
    # PY = 7
    # Cin = 32
    # Cout = 32
    # pad_W = 1
    # pad_H = 0
    # PX_O = PX + 2 * pad_W - 2
    # PY_O = PY + 2 * pad_H - 2
    # IN_ROW_MAX = 2
    #
    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = p41(px=PX, py=PY, cin=Cin, cout=Cout, kx=3, ky=3, sx=1, sy=1,
    #                    addr_ina=0x4000, addr_inb=0x0000, addr_bias=0x900, addr_out=0x6000,
    #                    ina_type=1, inb_type=1, load_bias=2,
    #                    pad_right=pad_W, pad_top=pad_H, pad_down=pad_H, pad_left=pad_W,
    #                    data_w=[],
    #                    data_b=[])
    #         soma1 = pX5(mode='max', addr_in=0x6000, addr_out=0x2000,
    #                     cin=Cout, cout=Cout, px=PX_O, py=PY_O,
    #                     kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
    #                     in_cut_start=2,
    #                     row_ck_on=1, in_row_max=IN_ROW_MAX)
    #         router = None
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # 求和截取
    # C = 48
    # PX = 13
    # PY = 14
    # ADDR_CACHE = 0x2000
    # OUT_ADDR = 0x4000
    #
    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1,
    #                    cin=C, px=PX, py=PY,
    #                    addr_in=0x0020, addr_bias=0x0000, addr_out=ADDR_CACHE,
    #                    pad_top=0, pad_down=0, pad_left=0, pad_right=0,
    #                    bias_length=0, constant_b=0, data_x=[])
    #
    #         soma1 = pX5(mode='max', addr_in=ADDR_CACHE, addr_out=OUT_ADDR,
    #                     cin=C, cout=C, px=PX, py=PY,
    #                     kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
    #                     in_cut_start=0,
    #                     row_ck_on=1,
    #                     in_row_max=2)
    #         # soma1 = None
    #         router = None
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # LUT
    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = None
    #         router = None
    #         soma2 = None
    #         soma2 = p07(addr_in=0x0000,
    #                     addr_out=0x4000,
    #                     group_num=15,
    #                     neuron_num=50 * 16,
    #                     lut_dw=1,
    #                     type_in=1, type_out=1,
    #                     addr_lut=0x2000,
    #                     data_lut=data['activation_out']['lut'])
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # P81
    W0_ADDR = 0x0000
    B0_ADDR = 0x120
    X0_ADDR = (0x10000 >> 2) + 0x920 + 0x1A40
    V0_ADDR = 0x180
    CACHE_ADDR = (0x10000 >> 2) + 0x920 + 0x1A40 + 0x400


    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p81(px=62, py=16, cin=4, cout=32, kx=3, ky=3, sx=1, sy=1,
                       addr_ina=X0_ADDR, addr_inb=W0_ADDR, addr_bias=B0_ADDR, addr_out=CACHE_ADDR,
                       ina_type=1, inb_type=1, load_bias=2,
                       data_w=[],
                       data_b=[]
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


    return map_config


if __name__ == '__main__':

    case_file_name = 'test'
    chip_0 = (0, 0)

    handler = ModelDataHandler()
    data = generate_test_data()

    # send_to_fpga = True

    config = MapConfigGen()

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1
    phase[1] = 1
    phase[2] = 1
    # phase[3] = 1
    # phase[4] = 1
    # phase[5] = 1
    # phase[6] = 1
    # phase[7] = 1
    # phase[8] = 1
    # phase[9] = 1
    # phase[10] = 1

    # phase[:] = 1

    clock_in_phase = 20_0000
    config_0 = gen_test_map_config(phase, clock_in_phase, 2, 2, data, chip=chip_0)

    config.add_config(config_0, core_offset=(0, 0))

    MapConfigGen.add_router_info(map_config=config.map_config)

    # prim = {
    #     'axon': None, 'soma1': None, 'router': None,
    #     'soma2': p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x10000 >> 2, length_in=1024,
    #                  num_in=12, length_ciso=1, num_ciso=12, length_out=1024, num_out=12,
    #                  type_in=1, type_out=1, data_in=None)
    # }
    #
    # MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)


    clock_0 = 5_0000 - 1
    clock_1 = 5_0000
    step_exe_number = 1
    config.sim_clock = clock_1 * 1

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

