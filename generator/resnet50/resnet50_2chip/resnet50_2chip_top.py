import os
import sys

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_2chip.data_gen import resnet50_data
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_2chip.G1_32cores import gen_g1_map_config
from generator.resnet50.resnet50_2chip.G1_IB import gen_g1_ib_map_config
from generator.resnet50.resnet50_2chip.G0_OB import gen_g0_ob_map_config
from generator.resnet50.resnet50_2chip.G2_28cores import gen_g2_map_config
from generator.resnet50.resnet50_2chip.G3_28cores import gen_g3_map_config
from generator.resnet50.resnet50_2chip.G4_28cores import gen_g4_map_config
from generator.resnet50.resnet50_2chip.G5_32cores import gen_g5_map_config
from generator.resnet50.resnet50_1chip.GBuffer import gen_buffer_map_config
import numpy as np
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch
import json
from generator.mapping_utils.gen_router_info_json import RouterInfoJsonGen
from generator.mapping_utils.prims import p06
from copy import deepcopy
from generator.resnet50.resnet50_2chip.quantization_config_2chip import QuantizationConfig


def main():
    case_file_name = 'Q00733'
    send_to_fpga = True
    ckpt_num = 0
    add_empty_phase = True

    save_router_json = False
    run = True

    config = MapConfigGen()
    cuts = QuantizationConfig('in_cut_start', ckpt=ckpt_num)
    data_all = resnet50_data(ckpt=ckpt_num)

    clock_in_phase = 0xfffff
    phase = np.zeros(35).astype(int)

    delay_base = 80
    delay_l4_3 = (delay_base,) * 9
    delay_l5_3 = (delay_base,) * 9

    # **** Group En ****
    g0_en = 1

    g1_ib_en = 1
    g1_en = 1
    g2_en = 1
    g3_en = 1
    g4_en = 1

    g5_en = 1
    gbuffer_en = 1

    # **** Phase En ****
    phase[0] = 1
    phase[1] = 0
    phase[2] = 0
    phase[3] = 0
    phase[4] = 0
    phase[5] = 0
    phase[6] = 0
    phase[7] = 0
    phase[8] = 0
    phase[9] = 0

    phase[:] = 1

    clock_0 = 250_0000 - 1
    clock_1 = 250_0000
    step_exe_number = 1
    config.sim_clock = clock_1 * 1
    inin_data = True

    if g0_en:
        g0_config = gen_g0_ob_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                         in_data_en=gbuffer_en, out_data_en=g1_ib_en, delay_l4=delay_l4_3,
                                         delay_l5=delay_l5_3, chip=(0, 0), send_to_fpga=send_to_fpga)
        config.add_config(g0_config)

    if g1_ib_en:
        g1_ib_config = gen_g1_ib_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                            in_data_en=g0_en, out_data_en=g1_en, delay_l4=delay_l4_3,
                                            delay_l5=delay_l5_3, chip=(0, 0), init_data=inin_data)
        config.add_config(g1_ib_config)

    if g1_en:
        g1_config = gen_g1_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=16, size_y=2,
                                      cuts=cuts, static_data=data_all, g1_ib_en=g1_ib_en, g2_en=g2_en,
                                      delay_l4=delay_l4_3, delay_l5=delay_l5_3, chip=(0, 0), init_data=inin_data,)
        config.add_config(g1_config, core_offset=(0, 1))

    if g2_en:
        g2_config = gen_g2_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=14, size_y=2,
                                      cuts=cuts, data=data_all, in_data_en=g1_en,
                                      out_data_en=g3_en, delay_l4=delay_l4_3, delay_l5=delay_l5_3, chip=(0, 0),
                                      init_data=inin_data)
        config.add_config(g2_config, core_offset=(2, 3))

    if g3_en:
        g3_config = gen_g3_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=14, size_y=2,
                                      cuts=cuts, data=data_all, in_data_en=g2_en,
                                      out_data_en=g4_en, delay_l4=delay_l4_3, delay_l5=delay_l5_3, chip=(0, 0),
                                      init_data=inin_data)
        config.add_config(g3_config, core_offset=(2, 5))

    if g4_en:
        g4_config = gen_g4_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=14, size_y=2,
                                      cuts=cuts, data=data_all, in_data_en=g3_en,
                                      out_data_en=g1_en, delay_l4=delay_l4_3, delay_l5=delay_l5_3, chip=(0, 0),
                                      init_data=inin_data)
        config.add_config(g4_config, core_offset=(2, 7))

    if g5_en:
        g5_config = gen_g5_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=16, size_y=2,
                                      cuts=cuts, static_data=data_all, g4_en=g4_en, g0_en=gbuffer_en,
                                      delay_l4=delay_l4_3, delay_l5=delay_l5_3, chip=(0, 1),
                                      init_data=inin_data)
        config.add_config(g5_config, core_offset=(0, 0))

    if gbuffer_en:
        gbuffer_config = gen_buffer_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=3,
                                               size_y=1, g1_en=g5_en, chip=(0, 0), g0_en=g0_en,
                                               send_to_fpga=send_to_fpga)
        config.add_config(gbuffer_config, core_offset=(7, 0))

    MapConfigGen.add_router_info(config.map_config)
    MapConfigGen.set_step_clock(config.map_config, clock_0=clock_0, clock_1=clock_1)
    MapConfigGen.set_step_exe_number(config.map_config, step_exe_number)

    if add_empty_phase:
        prim = {
            'axon': None, 'soma1': None, 'router': None,
            'soma2': p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x10000 >> 2, length_in=1024,
                         num_in=12, length_ciso=1, num_ciso=12, length_out=1024, num_out=12,
                         type_in=1, type_out=1, data_in=None)
        }
        MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    if save_router_json:
        router_json = RouterInfoJsonGen.gen_router_info_json(config.map_config, mode=0)
        # router_json = json.dumps(router_json, sort_keys=True, indent=4, separators=(',', ': '))
        with open('./temp/config/' + case_file_name + '.json', 'w') as f:
            json.dump(router_json, f, indent=4, separators=(',', ':'), sort_keys=True)

    if run:
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
            'debug_file_switch': HardwareDebugFileSwitch().close_all.multi_chip.close_burst.dict,
            'test_group_phase': [(0, 1)]
        }

        tester = TestEngine(config.map_config, test_config)
        # assert tester.run_test(exe_name='TianjicX1_SIM_11_11.exe')
        assert tester.run_test()


if __name__ == '__main__':
    main()
