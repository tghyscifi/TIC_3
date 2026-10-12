import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler

from generator.resnet50.resnet50_5chips.G9_1_32cores import gen_g9_1_map_config
from generator.resnet50.resnet50_5chips.G9_2_16cores import gen_g9_2_map_config
from generator.resnet50.resnet50_5chips.G9_instant import gen_g9_instant_map_config
from copy import deepcopy


def gen_g9_map_config(phase_en, clock_in_phase, cuts, data=None, add_empty_phase=False,
                      in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0), init_data=None):
    """
        ResNet-50 5-Chip Group-9
        core array : 6 * 8
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,  # 4 * 8 array
                'mode': 1,
            },
            1: {
                'clock': clock_in_phase,  # 2 *8 array
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(2, 6), range(8)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    for core_y, core_x in product(range(2), range(8)):
        map_config[(chip, 0)][1][(chip, (core_x, core_y))] = {
            'prims': []
        }

    offset = 4

    if add_empty_phase:
        start_instant_pi_num = offset + 2 + 1
    else:
        start_instant_pi_num = offset + 2

    config_instant = gen_g9_instant_map_config(clock_in_phase=clock_in_phase, chip=chip, rhead_base=(0x350, 0x350),
                                               start_instant_pi_num=start_instant_pi_num, receive_pi_addr_base=0x360,
                                               data=None)

    config_0 = gen_g9_1_map_config(deepcopy(phase_en), clock_in_phase=clock_in_phase, size_x=8, size_y=4, data=data,
                                   cuts=cuts,
                                   in_data_en=in_data_en, out_data_en=out_data_en, delay_l4=delay_l4,
                                   delay_l5=delay_l5, chip=chip, init_data=init_data)

    config_1 = gen_g9_2_map_config(deepcopy(phase_en), clock_in_phase=clock_in_phase, size_x=8, size_y=2, data=data,
                                   cuts=cuts,
                                   in_data_en=in_data_en, out_data_en=out_data_en, delay_l4=delay_l4,
                                   delay_l5=delay_l5, chip=chip)
    #
    for core_y, core_x in product(range(6), range(8)):
        if core_y in [0, 1]:
            map_config[(chip, 0)][1][(chip, (core_x, core_y))] = config_1[(chip, 0)][0][(chip, (core_x, core_y))]
        else:
            map_config[(chip, 0)][0][(chip, (core_x, core_y))] = config_0[(chip, 0)][0][(chip, (core_x, core_y - 2))]

    # 插入接收shortcut的静态原语
    for core_y, core_x in product(range(2, 6), range(8)):
        if phase_en[offset + 14] == 1:
            map_config[(chip, 0)][0][(chip, (core_x, core_y))]['prims'].insert(
                offset + 14, config_instant[(chip, 0)][0][(chip, (core_x, core_y))]['prims'][0])
            map_config[(chip, 0)][0][(chip, (core_x, core_y))]['registers'] = \
                config_instant[(chip, 0)][0][(chip, (core_x, core_y))]['registers']

    # 插入shortcut发送的即时原语
    for core_y, core_x in product(range(2), range(8)):
        if phase_en[offset + 14] == 1:
            map_config[(chip, 0)][1][(chip, (core_x, core_y))]['instant_prims'] = \
                config_instant[(chip, 0)][1][(chip, (core_x, core_y))]['instant_prims']
            map_config[(chip, 0)][1][(chip, (core_x, core_y))]['registers'] = \
                config_instant[(chip, 0)][1][(chip, (core_x, core_y))]['registers']

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00009'
    cuts = Resnet50Cuts()
    chip = (1, 0)

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0:50] = 1

    from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data

    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)

    clock_in_phase = 200_000
    config = gen_g9_map_config(phase_en=phase, clock_in_phase=clock_in_phase, cuts=cuts, data=data,
                               in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=chip)

    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][1][(chip, (0, 0))]['prims']) * clock_in_phase)

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
        'test_group_phase': [(0, 1), (1, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
