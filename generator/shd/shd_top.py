import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81, p08

from generator.zhl_nmi22.shd.shd_input import gen_map_config_for_input
from generator.zhl_nmi22.shd.shd_hidden import gen_map_config_for_hidden
from generator.zhl_nmi22.shd.shd_output import gen_map_config_for_output
from generator.zhl_nmi22.shd.fake_input import gen_map_config_for_shd_in
from generator.zhl_nmi22.shd.fake_output import gen_map_config_for_shd_out
from generator.mapping_utils.map_config_gen import MapConfigGen

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


if __name__ == '__main__':

    case_file_name = 'shd_top'
    chip = (0, 0)

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
    # phase[10] = 1
    # phase[11] = 1
    # phase[12] = 1

    # phase[:] = 1

    clock_in_phase = 20_0000

    # config_2 = gen_map_config_for_shd_in(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, chip=(0, 1))
    # config.add_config(config_2, core_offset=(15, 0))

    config_2 = gen_map_config_for_input(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, chip=chip,
                                        data_input=False)
    config.add_config(config_2, core_offset=(0, 0))

    config_1 = gen_map_config_for_hidden(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=2, chip=chip,
                                         data_input=True, data_output=True, shuffle=True)
    config.add_config(config_1, core_offset=(1, 0))

    config_3 = gen_map_config_for_output(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, chip=chip,
                                         data_output=False)
    config.add_config(config_3, core_offset=(2, 0))

    # config_3 = gen_map_config_for_shd_out(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, chip=(1, 0))
    # config.add_config(config_3, core_offset=(3, 9))

    MapConfigGen.add_router_info(map_config=config.map_config)

    prim = {
        'axon': None, 'soma1': None, 'router': None,
        'soma2': p06(addr_in=0x7ff0, addr_out=0x3ff0, addr_ciso=0x8600, length_in=1,
                     num_in=1, length_ciso=1, num_ciso=1, length_out=1, num_out=1,
                     type_in=1, type_out=1, data_in=None)
    }

    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    clock_0 = 1750
    clock_1 = clock_0 + 1
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
