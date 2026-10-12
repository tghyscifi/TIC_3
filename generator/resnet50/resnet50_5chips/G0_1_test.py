import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.mapping_utils.map_config_gen import MapConfigGen

if __name__ == '__main__':
    case_file_name = 'T00001'

    map_config = {
        'sim_clock': 400000,
        ((0, 0), 0): {
            # 'step_exe_number': 1,
            0: {
                'clock': 100000,
                'mode': 1,
            }
        },
        ((1, 0), 0): {
            # 'step_exe_number': 1,
            1: {
                'clock': 100000,
                'mode': 1,
            }
        }
    }

    phase = np.zeros(50).astype(int)  # 39-49表示组件数据传输的Phase

    phase[0] = 1
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1

    phase[4] = 1

    phase[5] = 1
    phase[6] = 1
    phase[7] = 1
    phase[8] = 1
    #
    # handler = ResNetDataHandler()
    # data = generate_g1_data(handler, size_y=2, size_x=16)
    data = None

    delay_l4 = (28, ) * 9
    delay_l5 = (28, ) * 9

    from generator.resnet50.resnet50_5chips.G1_IB_test import gen_g1_ib_map_config
    from generator.resnet50.resnet50_5chips.G0_OB import gen_g0_ob_map_config

    config0 = gen_g0_ob_map_config(phase, clock_in_phase=100_000, data=data, in_data_en=True,
                                   out_data_en=True, chip=(0, 0), delay_l4=delay_l4, delay_l5=delay_l5)
    config1 = gen_g1_ib_map_config(phase, clock_in_phase=100_000, data=data, in_data_en=True,
                                   out_data_en=True, chip=(1, 0), delay_l4=delay_l4, delay_l5=delay_l5)

    for core_x, core_y in [(0, 0), (1, 0), (2, 0), (3, 0)]:
        map_config[((0, 0), 0)][0][((0, 0), (core_x, core_y))] = config0[((0, 0), 0)][0][((0, 0), (core_x, core_y))]
        map_config[((1, 0), 0)][1][((1, 0), (core_x, core_y))] = config1[((1, 0), 0)][0][((1, 0), (core_x, core_y))]

    MapConfigGen.add_router_info(map_config=map_config)

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
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()
