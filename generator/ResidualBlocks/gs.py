import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.ResidualBlocks.g1_data_gen import generate_data
from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
from generator.ResidualBlocks.g1 import gen_1_map_config
from generator.ResidualBlocks.g0 import gen_0_map_config
from itertools import product


def gen_s_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, out_data_en=True, chip=(0, 0),
                     in_data_en=False):
    """
        Residual Block: Group source
        Data Source
        core_x * core_y: 8 * 5
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


    # ******** 数据收发 ********
    # 向后续的residual block发送
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=13, L5_num=20, A2S2_mode=True)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=13, L5_num=20, A2S2_mode=True)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=4, L5_num=46, A2S2_mode=True)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append(
                {'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # 接收或初始化输入
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            router = None
            soma2 = None
            if out_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                           addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                           L4_num=9, L5_num=223, A2S2_mode=True)
                soma1 = p06(addr_in=0x0000,
                            addr_out=0x9000,
                            addr_ciso=0x0000,
                            length_in=64, num_in=64,
                            length_ciso=1, num_ciso=64,
                            length_out=64, num_out=64,
                            type_in=1, type_out=1,
                            data_in=data['conv_in']['input'][(core_x, core_y)])
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=out_data_en, receive_en=False, send_num=16 * 256 // 8 - 1,
                             receive_num=0,
                             addr_din_base=0x400, addr_din_length=16 * 256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-8, Y=0, A=0, pack_per_Rhead=64 * 64 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})


    return map_config


def gen_d_map_config(phase_en, clock_in_phase, size_x, size_y, data=None, out_data_en=False, chip=(0, 0),
                     in_data_en=True):
    """
        Residual Block: Group dst
        Data Source
        core_x * core_y: 8 * 5
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


    # ******** 数据收发 ********
    # 向后续的residual block发送
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=13, L5_num=20, A2S2_mode=True)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=13, L5_num=20, A2S2_mode=True)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=4, L5_num=46, A2S2_mode=True)
            soma1 = None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append(
                {'axon': axon, 'soma1': soma1, 'router': router, 'soma2': soma2})

    # 接收或初始化输入
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                       addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                       L4_num=9, L5_num=223, A2S2_mode=True)
            soma1 = None
            soma2 = None
            router = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            router = None
            soma2 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=False, receive_en=in_data_en, send_num=0,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=16 * 420 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1)
            if in_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                           addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                           L4_num=68, L5_num=52, A2S2_mode=True)
                soma2 = p06(addr_in=0x8400,
                            addr_out=0x0000,
                            addr_ciso=0x0000,
                            length_in=16, num_in=420,
                            length_ciso=1, num_ciso=420,
                            length_out=16, num_out=420,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            router = None
            soma2 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=False, receive_en=in_data_en, send_num=0,
                         receive_num=0,
                         addr_din_base=0x400, addr_din_length=16 * 420 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1)
            if in_data_en:
                axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x7F00,
                           addr_inb=0x7F00, addr_bias=0x7F00, addr_out=0x7F00, axon_delay=True,
                           L4_num=68, L5_num=52, A2S2_mode=True)
                soma2 = p06(addr_in=0x8400,
                            addr_out=0x0000 + 0x960,
                            addr_ciso=0x0000,
                            length_in=16, num_in=420,
                            length_ciso=1, num_ciso=420,
                            length_out=16, num_out=420,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})


    return map_config



if __name__ == '__main__':

    case_file_name = 'residual_block_gs'
    chip_s = (1, 1)
    chip_0 = (0, 1)
    chip_d = (0, 0)

    handler = ModelDataHandler()
    data = generate_data(handler, 8, 5)

    # send_to_fpga = True

    config = MapConfigGen()

    phase = np.zeros(50).astype(int)

    phase[0] = 1
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1
    phase[4] = 1
    phase[5] = 1
    # phase[6] = 1
    # phase[7] = 1
    # phase[8] = 1
    # phase[9] = 1

    # phase[:] = 1

    clock_in_phase = 20_0000
    config_s = gen_s_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5,
                                in_data_en=False, out_data_en=True, chip=chip_s, data=data)

    config.add_config(config_s, core_offset=(0, 0))

    config_d = gen_d_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5,
                                in_data_en=True, out_data_en=False, chip=chip_d, data=data)

    config.add_config(config_d, core_offset=(8, 5))

    config_0 = gen_0_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=5,
                                in_data_en=True, out_data_en=False, chip=chip_0, data=data, last_layer_in_data_en=False, last_layer_out_data_en=True)

    config.add_config(config_0, core_offset=(8, 0))

    MapConfigGen.add_router_info(map_config=config.map_config)


    clock_0 = 40_0000 - 1
    clock_1 = 40_0000
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

