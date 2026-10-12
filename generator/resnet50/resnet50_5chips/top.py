import os
import sys

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G15_64cores import gen_g15_map_config
from generator.resnet50.resnet50_5chips.G16_64cores import gen_g16_map_config
from generator.resnet50.resnet50_5chips.G17_64cores import gen_g17_map_config
from generator.resnet50.resnet50_5chips.data_gen import resnet50_data
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_5chips.G16_IB import gen_g16_ib_map_config
from generator.resnet50.resnet50_5chips.G15_OB import gen_g15_ob_map_config
from generator.resnet50.resnet50_5chips.G14_32cores import gen_g14_map_config
from generator.resnet50.resnet50_5chips.G13_32cores import gen_g13_map_config
from generator.resnet50.resnet50_5chips.G13_IB import gen_g13_ib_map_config
from generator.resnet50.resnet50_5chips.G12_OB import gen_g12_ob_map_config
from generator.resnet50.resnet50_5chips.G12_32cores import gen_g12_map_config
from generator.resnet50.resnet50_5chips.G11_32cores import gen_g11_map_config
from generator.resnet50.resnet50_5chips.G10_32cores import gen_g10_map_config
from generator.resnet50.resnet50_5chips.G9_48cores import gen_g9_map_config
from generator.resnet50.resnet50_5chips.G9_IB import gen_g9_ib_map_config
from generator.resnet50.resnet50_5chips.G1_32cores import gen_g1_map_config
from generator.resnet50.resnet50_5chips.G1_IB import gen_g1_ib_map_config
from generator.resnet50.resnet50_5chips.G0_OB import gen_g0_ob_map_config
from generator.resnet50.resnet50_5chips.G2_28cores import gen_g2_map_config
from generator.resnet50.resnet50_5chips.G3_28cores import gen_g3_map_config
from generator.resnet50.resnet50_5chips.G4_28cores import gen_g4_map_config
from generator.resnet50.resnet50_5chips.G4_OB import gen_g4_ob_map_config
from generator.resnet50.resnet50_5chips.G5_IB import gen_g5_ib_map_config
from generator.resnet50.resnet50_5chips.G5_42cores import gen_g5_map_config
from generator.resnet50.resnet50_5chips.G6_28cores import gen_g6_map_config
from generator.resnet50.resnet50_5chips.G7_28cores import gen_g7_map_config
from generator.resnet50.resnet50_5chips.G8_28cores import gen_g8_map_config
from generator.resnet50.resnet50_5chips.G8_OB import gen_g8_ob_map_config
import numpy as np
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch
import json
from generator.mapping_utils.gen_router_info_json import RouterInfoJsonGen
from generator.mapping_utils.prims import p06
from copy import deepcopy
from generator.resnet50.resnet50_imagenet_quantization_config import QuantizationConfig


def main():
    case_file_name = 'Q00425'
    save_router_json = False
    run = True

    config = MapConfigGen()
    cuts = QuantizationConfig('in_cut_start')
    data_all = resnet50_data()

    add_empty_phase = True

    clock_in_phase = 0xfffff
    phase = np.zeros(35).astype(int)

    delay_base = 30
    delay_l4_1 = (delay_base,) * 9
    delay_l5_1 = (delay_base,) * 9
    delay_l4_2 = (delay_base + 5,) * 9
    delay_l5_2 = (delay_base + 5,) * 9
    delay_l4_3 = (delay_base + 10,) * 9
    delay_l5_3 = (delay_base + 10,) * 9

    # **** Group En ****
    g0_en = 1

    g1_ib_en = 1
    g1_en = 1
    g2_en = 1
    g3_en = 1
    g4_en = 1
    g4_ob_en = 1

    g5_ib_en = 1
    g5_en = 1
    g6_en = 1
    g7_en = 1
    g8_en = 1
    g8_ob_en = 1

    g9_ib_en = 1
    g9_en = 1
    g10_en = 1
    g11_en = 1
    g12_en = 1
    g12_ob_en = 1

    g13_ib_en = 1
    g13_en = 1
    g14_en = 1
    g15_en = 1
    g15_ob_en = 1

    g16_ib_en = 1
    g16_en = 1
    g17_en = 1

    # **** Phase En ****
    phase[0] = 1
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1
    phase[4] = 0
    phase[5] = 0
    phase[6] = 0
    phase[7] = 0
    phase[8] = 0
    phase[9] = 0

    phase[:] = 1

    clock_0 = 350_000 - 1
    clock_1 = 350_000
    step_exe_number = 1
    config.sim_clock = clock_1 * 1
    inin_data = True

    if g0_en:
        g0_config = gen_g0_ob_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                         in_data_en=g1_en, out_data_en=g1_ib_en, delay_l4=delay_l4_3,
                                         delay_l5=delay_l5_3, chip=(0, 0))
        config.add_config(g0_config)

    if g1_ib_en:
        g1_ib_config = gen_g1_ib_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                            in_data_en=g0_en, out_data_en=g1_en, delay_l4=delay_l4_3,
                                            delay_l5=delay_l5_3, chip=(0, 0), init_data=inin_data)
        config.add_config(g1_ib_config)

    if g1_en:
        g1_config = gen_g1_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=16, size_y=2,
                                      cuts=cuts, static_data=data_all, g1_ib_en=g1_ib_en, g2_en=g2_en, g17_en=g17_en,
                                      g0_en=g0_en, delay_l4=delay_l4_3, delay_l5=delay_l5_3, chip=(0, 0),
                                      init_data=inin_data)
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
                                      out_data_en=g4_ob_en, delay_l4=delay_l4_3, delay_l5=delay_l5_3, chip=(0, 0),
                                      init_data=inin_data)
        config.add_config(g4_config, core_offset=(2, 7))

    if g4_ob_en:
        g4_ob_config = gen_g4_ob_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                            in_data_en=g4_en, out_data_en=g5_ib_en, delay_l4=delay_l4_3,
                                            delay_l5=delay_l5_3, init_data=inin_data)
        config.add_config(g4_ob_config)

    if g5_ib_en:
        g5_ib_config = gen_g5_ib_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                            in_data_en=g4_ob_en, out_data_en=g5_en, delay_l4=delay_l4_3,
                                            delay_l5=delay_l5_3, init_data=inin_data, chip=(0, 1))
        config.add_config(g5_ib_config)

    if g5_en:
        g5_config = gen_g5_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, cuts=cuts, data=data_all,
                                      in_data_en=g5_ib_en, out_data_en=g6_en, delay_l4=delay_l4_2, delay_l5=delay_l5_2,
                                      chip=(0, 1), init_data=inin_data, add_empty_phase=add_empty_phase)
        config.add_config(g5_config, core_offset=(0, 0))

    if g6_en:
        g6_config = gen_g6_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, cuts=cuts,
                                      static_data=data_all,
                                      size_x=14, size_y=2,
                                      in_data_en=g5_en, out_data_en=g7_en, delay_l4=delay_l4_2, delay_l5=delay_l5_2,
                                      chip=(0, 1), init_data=inin_data)
        config.add_config(g6_config, core_offset=(0, 3))

    if g7_en:
        g7_config = gen_g7_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, cuts=cuts,
                                      static_data=data_all,
                                      size_x=14, size_y=2,
                                      in_data_en=g6_en, out_data_en=g8_en, delay_l4=delay_l4_2, delay_l5=delay_l5_2,
                                      chip=(0, 1), init_data=inin_data)
        config.add_config(g7_config, core_offset=(0, 5))

    if g8_en:
        g8_config = gen_g8_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, cuts=cuts,
                                      static_data=data_all,
                                      size_x=14, size_y=2,
                                      in_data_en=g7_en, out_data_en=g8_ob_en, delay_l4=delay_l4_2, delay_l5=delay_l5_2,
                                      chip=(0, 1), init_data=inin_data)
        config.add_config(g8_config, core_offset=(0, 7))

    if g8_ob_en:
        g8_ob_config = gen_g8_ob_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                            in_data_en=g8_en, out_data_en=g9_ib_en, delay_l4=delay_l4_2,
                                            delay_l5=delay_l5_2, init_data=inin_data, chip=(0, 1))
        config.add_config(g8_ob_config)

    if g9_ib_en:
        g9_ib_config = gen_g9_ib_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                            in_data_en=g8_ob_en, out_data_en=g9_en, delay_l4=delay_l4_2,
                                            delay_l5=delay_l5_2, chip=(0, 2), init_data=inin_data)
        config.add_config(g9_ib_config, core_offset=(0, 0))

    if g9_en:
        g9_config = gen_g9_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase,
                                      cuts=cuts, data=data_all, in_data_en=g9_ib_en, out_data_en=g10_en,
                                      delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(0, 2), init_data=inin_data,
                                      add_empty_phase=add_empty_phase)
        config.add_config(g9_config, core_offset=(0, 0))

    if g10_en:
        g10_config = gen_g10_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=8, size_y=4,
                                        cuts=cuts, data=data_all, in_data_en=g9_en, out_data_en=g11_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(0, 2), init_data=inin_data)
        config.add_config(g10_config, core_offset=(0, 6))

    if g11_en:
        g11_config = gen_g11_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=8, size_y=4,
                                        cuts=cuts, data=data_all, in_data_en=g10_en, out_data_en=g12_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(0, 2), init_data=inin_data)
        config.add_config(g11_config, core_offset=(8, 6))

    if g12_en:
        g12_config = gen_g12_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=8, size_y=4,
                                        cuts=cuts, data=data_all, in_data_en=g11_en, out_data_en=g12_ob_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(0, 2), init_data=inin_data)
        config.add_config(g12_config, core_offset=(8, 2))

    if g12_ob_en:
        g12_ob_config = gen_g12_ob_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                              in_data_en=g12_en, out_data_en=g13_ib_en, delay_l4=delay_l4_1,
                                              delay_l5=delay_l5_1, init_data=inin_data)
        config.add_config(g12_ob_config)

    if g13_ib_en:
        g13_ib_config = gen_g13_ib_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                              in_data_en=g12_ob_en, out_data_en=g13_en, delay_l4=delay_l4_1,
                                              delay_l5=delay_l5_1, chip=(1, 1), init_data=inin_data)
        config.add_config(g13_ib_config)

    if g13_en:
        g13_config = gen_g13_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=8, size_y=4,
                                        cuts=cuts, data=data_all, in_data_en=g13_ib_en, out_data_en=g14_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(1, 1), init_data=inin_data)
        config.add_config(g13_config, core_offset=(8, 5))

    if g14_en:
        g14_config = gen_g14_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=8, size_y=4,
                                        cuts=cuts, data=data_all, in_data_en=g13_en, out_data_en=g15_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(1, 1), init_data=inin_data)
        config.add_config(g14_config, core_offset=(0, 5))

    if g15_en:
        g15_config = gen_g15_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=16, size_y=4,
                                        cuts=cuts,
                                        data=data_all, in_data_en=g14_en, out_data_en=g15_ob_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(1, 1), init_data=inin_data)
        config.add_config(g15_config, core_offset=(0, 1))

    if g15_ob_en:
        g15_ob_config = gen_g15_ob_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                              in_data_en=g15_en, out_data_en=g16_ib_en,
                                              delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(1, 1),
                                              init_data=inin_data)
        config.add_config(g15_ob_config, core_offset=(0, 0))

    if g16_ib_en:
        g16_ib_config = gen_g16_ib_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, data=data_all,
                                              in_data_en=g15_ob_en, out_data_en=g16_en,
                                              delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(1, 0),
                                              init_data=inin_data)
        config.add_config(g16_ib_config, core_offset=(0, 0))

    if g16_en:
        g16_config = gen_g16_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=16, size_y=4,
                                        cuts=cuts,
                                        data=data_all, in_data_en=g16_ib_en, out_data_en=g17_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(1, 0), init_data=inin_data)
        config.add_config(g16_config, core_offset=(0, 5))

    if g17_en:
        g17_config = gen_g17_map_config(phase_en=deepcopy(phase), clock_in_phase=clock_in_phase, size_x=16, size_y=4,
                                        cuts=cuts,
                                        data=data_all, in_data_en=g16_en, out_data_en=g1_en,
                                        delay_l4=delay_l4_1, delay_l5=delay_l5_1, chip=(1, 0), init_data=inin_data)
        config.add_config(g17_config, core_offset=(0, 1))

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
        MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=None)

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
