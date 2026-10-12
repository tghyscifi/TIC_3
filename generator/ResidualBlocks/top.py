import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.ResidualBlocks.g1_data_gen import generate_data
from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
from generator.ResidualBlocks.g0 import gen_0_map_config
from generator.ResidualBlocks.g1 import gen_1_map_config
from generator.ResidualBlocks.g2 import gen_2_map_config
from generator.ResidualBlocks.g3 import gen_3_map_config
from itertools import product



if __name__ == '__main__':

    case_file_name = 'residual_blocks'
    chip_0 = (0, 0)

    handler = ModelDataHandler(model='real')
    data = generate_data(handler, 8, 5)

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
    phase[10] = 1

    # phase[:] = 1

    clock_in_phase = 20_0000
    config_0 = gen_0_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5,
                                in_data_en=False, out_data_en=True, chip=chip_0, data=data, last_layer_in_data_en=True)
    config.add_config(config_0, core_offset=(8, 0))

    config_1 = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5, in_cut_start_dict=None,
                                in_data_en=True, out_data_en=True, chip=chip_0, data=data, init_data=False)
    config.add_config(config_1, core_offset=(0, 0))

    config_2 = gen_2_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5, in_cut_start_dict=None,
                                in_data_en=True, out_data_en=True, chip=chip_0, data=data, init_data=False)
    config.add_config(config_2, core_offset=(0, 5))

    config_3 = gen_3_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5, in_cut_start_dict=None,
                                in_data_en=True, out_data_en=True, chip=chip_0, data=data, init_data=False)
    config.add_config(config_3, core_offset=(8, 5))

    MapConfigGen.add_router_info(map_config=config.map_config)

    # prim = {
    #     'axon': None, 'soma1': None, 'router': None,
    #     'soma2': p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x10000 >> 2, length_in=1024,
    #                  num_in=12, length_ciso=1, num_ciso=12, length_out=1024, num_out=12,
    #                  type_in=1, type_out=1, data_in=None)
    # }
    #
    # MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)


    clock_0 = 25_0000 - 1
    clock_1 = 25_0000
    step_exe_number = 1
    config.sim_clock = clock_1 * 5

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
