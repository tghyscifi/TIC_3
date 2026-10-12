import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import p06
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
from generator.ResidualBlocks.g1_data_gen import generate_data

from itertools import product
import os


def test_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                    in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False):
    """
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
    # 无外部输入情况已通过校对
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # print(len(data['conv0']['input'][(core_x, core_y)]))
            axon = None
            if not in_data_en:
                soma1 = p06(addr_in=(0x10000 >> 2) + 0x2D0,
                            addr_out=(0x10000 >> 2) + 0x2D0,
                            addr_ciso=0x0000,
                            length_in=32, num_in=630,
                            length_ciso=1, num_ciso=630,
                            length_out=32, num_out=630,
                            type_in=1, type_out=1,
                            data_in=None)
                router = None
            else:
                soma1 = None
                router = None  # 在有外部输入时需要配置，但是因为7行数据的在mem2中放不下，一个phase收不完，有点麻烦，所以暂时不配
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':

    case_file_name = 'p06_test'
    chip = (0, 0)

    config = MapConfigGen()

    phase = np.zeros(1).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1

    clock_in_phase = 20_0000
    config_1 = test_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1)
    config.add_config(config_1, core_offset=(0, 0))

    MapConfigGen.add_router_info(map_config=config.map_config)

    config.map_config['sim_clock'] = 10_0000
    config.map_config['step_clock'] = {
        ((0, 0), 0): (10_0000 - 1, 10_0000)
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
