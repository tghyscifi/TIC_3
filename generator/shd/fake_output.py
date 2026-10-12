import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81, p08
from generator.mapping_utils.map_config_gen import MapConfigGen

from itertools import product
import os


bits_in_ternary = 2
ternary_in_16B = 16 * 8 // bits_in_ternary
ternary_in_4B = 4 * 8 // bits_in_ternary
ternary_in_8B = 8 * 8 // bits_in_ternary

input_neuron_number = 700
input_neuron_grouped_by_16B = int(np.ceil(700 / ternary_in_16B) * ternary_in_16B)
input_bytes = int(np.ceil(700 / ternary_in_16B)) * 16
input_neuron_grouped_by_4B = int(np.ceil(700 / ternary_in_4B) * ternary_in_4B)
input_neuron_grouped_by_8B = int(np.ceil(700 / ternary_in_8B) * ternary_in_8B)

def gen_map_config_for_shd_out(phase_en, clock_in_phase, size_x, size_y, chip=(0, 0)):
    """
        Input Layer: Group 0
        core_x * core_y: 1 * 1
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

    # 加载输入
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # print(len(data['conv0']['input'][(core_x, core_y)]))
            axon = None
            soma1 = None
            router = None
            soma2 = None



            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # print(len(data['conv0']['input'][(core_x, core_y)]))
            axon = None
            soma1 = None
            router = None
            soma2 = None

            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])

            router = p09(rhead_mode=1,
                         send_en=False,
                         receive_en=True,
                         send_num=input_bytes // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=32 // 8 - 1,  # 输入时用p06+p09发送，p06一定是16B分组的
                         addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=input_bytes // 16 - 1,
                         soma_in_en=False)

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':

    case_file_name = 'shd_fake_output'

