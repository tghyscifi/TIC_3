import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81, p08
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.SoundSNN.snn_config import SNNConfig


def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, config=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=None,
                     delay_l4=None, delay_l5=None):
    """
        Obstacle: Group 1
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
    offset = 0
    # ******** 开始计算 ********
    # Conv计算
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=16, cout=128, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x18000 >> 2,
                       addr_out=0x8000 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=data['fc1'][0]['input'][(core_x, core_y)], 
                       data_w=data['fc1'][0]['weight'][(core_x, core_y)], 
                       data_b=data['fc1'][0]['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x8000 >> 2, addr_out=0x4000 >> 2,
                        cin=128, cout=128, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=0, row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os
    from generator.SoundSNN.data_handler import SNNDataHandler
    from generator.SoundSNN.g1_data import generate_g1_data

    case_file_name = 'SNN_1'
    chip = (0, 0)
    phase_offset = 0
    delay = (0,) * 9

    phase = np.zeros(50).astype(int)
    # 39~49 表示组间数据传输的Phase

    phase[phase_offset + 0] = 1

    handler = SNNDataHandler()
    data = generate_g1_data(handler, size_y=1, size_x=1, sequence_length=39)

    clock_in_phase = 150_000
    snn_config = SNNConfig()
    config = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, data=data,
                              config=snn_config, in_data_en=False, out_data_en=False, chip=chip)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)

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

    tester = TestEngine(config, test_config)
    assert tester.run_test()
