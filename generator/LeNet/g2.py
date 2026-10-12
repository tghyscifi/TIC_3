import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_2_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, in_data_en=False, out_data_en=False,
                     chip=(0, 0), init_data=False, send_to_fpga=False):
    """
        Obstacle: Group 2
        8 cores
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
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=16,
                        num_in=1, length_ciso=1, num_ciso=1, length_out=16, num_out=1,
                        type_in=1, type_out=1,
                        data_in=data['fc_cut3']['output'][(0, 0)] if init_data else None)
            if not out_data_en and core_x == 7:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en or core_x != 7,
                         receive_en=in_data_en or core_x != 0, send_num=16 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=16 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            if core_x == 7:
                if not send_to_fpga:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=1 - core_x, Y=-1 - core_y, A=0, pack_per_Rhead=16 // 8 - 1,
                                    A_offset=0, Const=0, EN=1)
                else:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-128, A=0, pack_per_Rhead=16 // 8 - 1,
                                    A_offset=0, Const=0, EN=1)
            else:
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=16 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            if in_data_en or core_x != 0:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x0000 >> 2, length_in=16,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=16, num_out=1,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config
