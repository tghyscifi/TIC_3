import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.prims import p04, p06, p26, p09, pX5, p41, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_5chips.G1_data import generate_g1_data
from generator.resnet50.data_handler import ResNetDataHandler
from itertools import product


def gen_buffer_map_config(phase_en, clock_in_phase, size_x, size_y,
                          g1_en=False, chip=(0, 0),
                          g0_en=False, send_to_fpga=False):
    """
        ResNet-50 5-Chip Buffer
        core array : 3 * 1
    """

    phase_group = {}

    # 接收 发送 Yfc = 32
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if g0_en or send_to_fpga:
                soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=32,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=32, num_out=1,
                            type_in=1, type_out=1, data_in=None)
            router = p09(rhead_mode=1, send_en=g0_en or send_to_fpga if (core_x, core_y) == (2, 0) else 1,
                         receive_en=g1_en if (core_x, core_y) == (0, 0) else 1,
                         send_num=32 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=32 // 8 - 1, addr_rhead_base=0x300,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, nx=0, ny=0, relay_num=0, data_in=None)
            if core_x != 2:
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0,
                                pack_per_Rhead=32 // 8 - 1, A_offset=0, Const=0, EN=1)
            else:
                if send_to_fpga:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-128, A=0,
                                    pack_per_Rhead=32 // 8 - 1, A_offset=0, Const=0, EN=1)
                else:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=3 - core_x, Y=0 - core_y, A=0,
                                    pack_per_Rhead=32 // 8 - 1, A_offset=0, Const=0, EN=g0_en)
            if not router.Receive_en:
                soma2 = None
            else:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2,
                            length_in=32,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=32, num_out=1,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))] = {'axon': axon, 'soma1': soma1, 'router': router,
                                                     'soma2': soma2}
    else:
        for core_y, core_x in product(range(size_y), range(size_x)):
            phase_group[(chip, (core_x, core_y))] = {'axon': None, 'soma1': None, 'router': None,
                                                     'soma2': None}

    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
                (chip, (0, 0)): {
                    'prims': [phase_group[(chip, (0, 0))]]
                },
                (chip, (1, 0)): {
                    'prims': [phase_group[(chip, (1, 0))]]
                },
                (chip, (2, 0)): {
                    'prims': [phase_group[(chip, (2, 0))]]
                }
            },
        }
    }
    return map_config
