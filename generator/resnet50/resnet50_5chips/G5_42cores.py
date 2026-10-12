import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler

from generator.resnet50.resnet50_5chips.G5_28cores import gen_g5_map_config0
from generator.resnet50.resnet50_5chips.G5_14cores import gen_g5_map_config1
from generator.resnet50.resnet50_5chips.G5_instant import gen_g5_instant_map_config
from copy import deepcopy


def gen_g5_map_config(phase_en, clock_in_phase, cuts, data=None, add_empty_phase=False,
                      in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0), init_data=True):
    """
        ResNet-50 5-Chip Group-5
        core array : 3 * 14
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {
                'clock': clock_in_phase,  # 2 * 14 array
                'mode': 1,
            },
            1: {
                'clock': clock_in_phase,  # 1 * 14 array
                'mode': 1,
            }
        }
    }
    offset = 4
    if add_empty_phase:
        start_instant_pi_num = offset + 2 + 1
    else:
        start_instant_pi_num = offset + 2
    for core_y, core_x in product(range(2), range(14)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    for core_y, core_x in product(range(2, 3), range(14)):
        map_config[(chip, 0)][1][(chip, (core_x, core_y))] = {
            'prims': []
        }

    config_instant = gen_g5_instant_map_config(clock_in_phase=clock_in_phase, chip=chip, rhead_base=(0x350, 0x350),
                                               start_instant_pi_num=start_instant_pi_num,
                                               receive_pi_addr_base=0x360)  # Core参数

    config_0 = gen_g5_map_config0(deepcopy(phase_en), clock_in_phase=clock_in_phase, size_x=14, size_y=2,
                                  static_data=data,
                                  cuts=cuts, chip=chip, in_data_en=in_data_en, out_data_en=out_data_en,
                                  delay_l4=delay_l4, delay_l5=delay_l5, init_data=init_data)

    config_1 = gen_g5_map_config1(deepcopy(phase_en), clock_in_phase=clock_in_phase, size_x=14, size_y=1,
                                  static_data=data,
                                  cuts=cuts, chip=chip, in_data_en=in_data_en, out_data_en=out_data_en,
                                  delay_l4=delay_l4, delay_l5=delay_l5)

    for core_y, core_x in product(range(3), range(14)):
        if core_y == 2:
            map_config[(chip, 0)][1][(chip, (core_x, core_y))] = config_1[(chip, 0)][0][(chip, (core_x, core_y - 2))]
        else:
            map_config[(chip, 0)][0][(chip, (core_x, core_y))] = config_0[(chip, 0)][0][(chip, (core_x, core_y))]

    # 插入接收shortcut的静态原语
    for core_y, core_x in product(range(2), range(14)):
        if phase_en[offset + 9] == 1:
            map_config[(chip, 0)][0][(chip, (core_x, core_y))]['prims'].insert(
                offset + 9, config_instant[(chip, 0)][0][(chip, (core_x, core_y))]['prims'][0])
            map_config[(chip, 0)][0][(chip, (core_x, core_y))]['prims'].insert(
                offset + 10, config_instant[(chip, 0)][0][(chip, (core_x, core_y))]['prims'][1])
            map_config[(chip, 0)][0][(chip, (core_x, core_y))]['registers'] = \
                config_instant[(chip, 0)][0][(chip, (core_x, core_y))]['registers']

    # 插入shortcut发送的即时原语
    for core_y, core_x in product(range(2, 3), range(14)):
        if phase_en[offset + 9] == 1:
            map_config[(chip, 0)][1][(chip, (core_x, core_y))]['instant_prims'] = \
                config_instant[(chip, 0)][1][(chip, (core_x, core_y))]['instant_prims']
            map_config[(chip, 0)][1][(chip, (core_x, core_y))]['registers'] = \
                config_instant[(chip, 0)][1][(chip, (core_x, core_y))]['registers']

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00005'
    cuts = Resnet50Cuts()
    chip = (1, 0)

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0:50] = 1

    from generator.resnet50.resnet50_5chips.G5_data import generate_g5_data

    handler = ResNetDataHandler()
    data = generate_g5_data(handler, size_y=2, size_x=14)

    clock_in_phase = 200_000
    config = gen_g5_map_config(phase_en=phase, clock_in_phase=clock_in_phase, cuts=cuts, data=data,
                               in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=chip)

    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, max(len(config[(chip, 0)][1][(chip, (0, 2))]['prims']) * clock_in_phase,
                                           len(config[(chip, 0)][0][(chip, chip)]['prims']) * clock_in_phase))

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    c_path = os.getcwd()
    out_files_path = os.getcwd() + "/simulator/Out_files/" + case_file_name + "/"
    if os.path.exists(out_files_path):
        os.chdir(out_files_path)
        del_command = 'rm -r cmp_out'
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
