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

input_bytes = 32
input_neuron_number = 64
input_neuron_grouped_by_16B = int(np.ceil(input_neuron_number / ternary_in_16B) * ternary_in_16B)


def gen_map_config_for_output(phase_en, clock_in_phase, size_x, size_y, chip=(0, 0), data_output=False):
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

    # 加载输入
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # print(len(data['conv0']['input'][(core_x, core_y)]))
            axon = None
            soma1 = None
            router = None
            soma2 = None

            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])

            router = p09(rhead_mode=1,
                         send_en=False,
                         receive_en=True,
                         send_num=0,
                         receive_num=1,
                         addr_din_base=0x400, addr_din_length=input_bytes // 8 - 1,  # 输入时用p06+p09发送，p06一定是16B分组的
                         addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0, addr_dout_length=0,
                         soma_in_en=False)

            soma2 = p06(addr_in=0x8400,
                        addr_out=0x0000,
                        addr_ciso=0x4000,
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
    #         if data_output:
    #             addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
    #
    #             soma1 = p06(addr_in=0x7000, addr_out=0x9000, addr_ciso=0x4000, length_in=32,
    #                         num_in=1, length_ciso=1, num_ciso=1,
    #                         length_out=32, num_out=1,
    #                         type_in=1, type_out=1,
    #                         data_in=None)
    #
    #             router = p09(rhead_mode=1,
    #                          send_en=True,
    #                          receive_en=False,
    #                          send_num=32 // 8 - 1,
    #                          receive_num=0,
    #                          addr_din_base=0x400, addr_din_length=input_bytes // 8 - 1,  # 输入时用p06+p09发送，p06一定是16B分组的
    #                          addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
    #                          addr_dout_base=0x1000, addr_dout_length=32 // 16 - 1,
    #                          soma_in_en=True)
    #
    #             router.addRHead(S=0, T=1, P=0, Q=0,
    #                             X=1,  # TODO
    #                             Y=-core_y-1,  # TODO
    #                             A=0,
    #                             pack_per_Rhead=32 // 8 - 1,
    #                             A_offset=0, Const=0, EN=1)
    #
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})


    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            # print(len(data['conv0']['input'][(core_x, core_y)]))
            axon = None
            soma1 = None
            router = None
            soma2 = None

            # W：192*32/4=0x600
            axon = p41(px=1, py=1, cin=64, cout=32, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0000, addr_inb=0x4000, addr_bias=0x2000, addr_out=0x5000,
                       ina_type=3, inb_type=1, load_bias=2,
                       pad_right=0, pad_top=0, pad_down=0, pad_left=0,
                       A2S2_mode=True,
                       data_w=[],
                       data_b=[])
            soma1 = p08(neuron_num=32, group_num=1, seed=0, Vth0=60000,
                        Vleaky_adpt_en=True, ref_len=0, row_ck_on=1, in_cut_start=11, in_row_max=1,
                        Vtheta_const_en=True, Vtheta_const=0,
                        Vleaky_alpha=128, Vleaky_beta=0,
                        addr_uin=0x5000, addr_v=0x6000, addr_s=0x7000, addr_vm=0x0120,
                        addr_vtheta=0x0130, addr_para=0x0400, Tw_en=True, Tw_len=1000, Tw_cnt=0,
                        fire_type=2,  # 树突上应该输出衰减前的膜电位
                        VR=1000,
                        data_v=np.random.random_integers(low=0, high=0x9999, size=(32, 1)).tolist()
                        )

            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

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
    config_1 = gen_map_config_for_output(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=2, chip=chip)
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
