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

def gen_map_config_for_input(phase_en, clock_in_phase, size_x, size_y, chip=(0, 0), data_input=False):
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
    # if phase_en[0]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         # print(len(data['conv0']['input'][(core_x, core_y)]))
    #         axon = None
    #         soma1 = None
    #         router = None
    #         soma2 = None
    #
    #         soma2 = p06(addr_in=0x7fff, addr_out=0x7fff >> 2, addr_ciso=0x8600, length_in=1,
    #                     num_in=1, length_ciso=1, num_ciso=1, length_out=1, num_out=1,
    #                     type_in=1, type_out=1, data_in=[])
    #
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})


    # 加载输入
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # print(len(data['conv0']['input'][(core_x, core_y)]))
            axon = None
            soma1 = None
            router = None
            soma2 = None

            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])

            soma1 = p06(addr_in=0x0000, addr_out=0x9000, addr_ciso=0x4000, length_in=input_neuron_grouped_by_16B,
                        num_in=1, length_ciso=1, num_ciso=1,
                        length_out=input_neuron_grouped_by_16B, num_out=1,
                        type_in=3, type_out=3,
                        data_in=None if data_input else [])

            router = p09(rhead_mode=1,
                         send_en=True,
                         receive_en=data_input,
                         send_num=input_bytes // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=input_bytes // 8 - 1,  # 输入时用p06+p09发送，p06一定是16B分组的
                         addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=input_bytes // 16 - 1,
                         soma_in_en=True)

            router.addRHead(S=0, T=1, P=0, Q=1,
                            X=1,
                            Y=0,
                            A=0,
                            pack_per_Rhead=input_bytes // 8 - 1,
                            A_offset=0, Const=0, EN=1)

            if data_input:
                soma2 = p06(addr_in=0x8400,
                            addr_out=0x4000,
                            addr_ciso=0x0000,
                            length_in=input_neuron_grouped_by_16B, num_in=1,
                            length_ciso=1, num_ciso=1,
                            length_out=input_neuron_grouped_by_16B, num_out=1,
                            type_in=3, type_out=3)

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # if phase_en[1]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         # print(len(data['conv0']['input'][(core_x, core_y)]))
    #         axon = None
    #         soma1 = None
    #         router = None
    #         soma2 = None
    #
    #         addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #
    #         soma1 = p26(addr_in=0x4000, addr_out=0x0000, addr_ciso=0x2000, length_in=input_neuron_grouped_by_16B,
    #                     num_in=1, length_ciso=1, num_ciso=1,
    #                     length_out=input_neuron_grouped_by_16B, num_out=1,
    #                     type_in=3, type_out=3,
    #                     data_in=None)
    #
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    return map_config


if __name__ == '__main__':

    case_file_name = 'shd_hidden'
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
    # phase[8] = 1
    # phase[9] = 1

    # phase[:] = 1

    clock_in_phase = 20_0000
    config_1 = gen_map_config_for_input(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=2, chip=chip)
    config.add_config(config_1, core_offset=(0, 0))

    MapConfigGen.add_router_info(map_config=config.map_config)

    # prim = {
    #     'axon': None, 'soma1': None, 'router': None,
    #     'soma2': p06(addr_in=0x7fff, addr_out=0x8400, addr_ciso=0x7fff >> 2, length_in=1,
    #                  num_in=1, length_ciso=1, num_ciso=1, length_out=1, num_out=1,
    #                  type_in=1, type_out=1, data_in=None)
    # }
    #
    # MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

    clock_0 = 100000
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
