import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product


def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None, data=None,
                     in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False):
    """
        Obstacle: Group 1
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
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x2000 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=16,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=16, num_out=1,
                            type_in=1, type_out=1,
                            data_in=data['fc_cut3']['output'][(0, 0)] if init_data else None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=16 // 8 - 1,
                         receive_num=0,
                         addr_din_base=0x380, addr_din_length=32 * 28 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=16 // 8 - 1,
                            A_offset=0, Const=0, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x0000 >> 2, length_in=32,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=32, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    offset = 1
    # ******** 开始计算 ********
    # Conv1 Maxpool1
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p81(px=28, py=28, cin=1, cout=6, kx=5, ky=5, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x10320 >> 2, addr_out=0x1040 >> 2,
                       ina_type=1, inb_type=1, load_bias=2,
                       data_x=None if in_data_en else data['conv1']['input'][(core_x, core_y)],
                       data_w=data['conv1']['weight'][(core_x, core_y)],
                       data_b=data['conv1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1040 >> 2, addr_out=0x0400 >> 2,
                        cin=32, cout=6, px=24, py=24,
                        kx=2, ky=2, sx=2, sy=2, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['cut1'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=12, py=12, cin=6, cout=16, kx=5, ky=5, sx=1, sy=1,
                       addr_ina=0x400 >> 2, addr_inb=0x103A0 >> 2, addr_bias=0x11660 >> 2, addr_out=0x1040 >> 2,
                       pad_top=1, pad_down=1, pad_left=1, pad_right=1,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['conv2']['weight'][(core_x, core_y)],
                       data_b=data['conv2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1040 >> 2, addr_out=0x1A40 >> 2,
                        cin=32, cout=16, px=10, py=10,
                        kx=2, ky=2, sx=2, sy=2, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['cut2'], row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=400, cout=120, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x1A40 >> 2, addr_inb=0x116e0 >> 2, addr_bias=0x1dee0 >> 2, addr_out=0x400 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['fc1']['weight'][(core_x, core_y)],
                       data_b=data['fc1']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x400 >> 2, addr_out=0x1000 >> 2,
                        cin=128, cout=120, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['fc_cut1'], row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=120, cout=84, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x1000 >> 2, addr_inb=0x8000 >> 2, addr_bias=0xAD00 >> 2, addr_out=0x400 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['fc2']['weight'][(core_x, core_y)],
                       data_b=data['fc2']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x400 >> 2, addr_out=0x600 >> 2,
                        cin=96, cout=84, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['fc_cut2'], row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, cin=84, cout=10, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x600 >> 2, addr_inb=0x1e0e0 >> 2, addr_bias=0x1eb60 >> 2, addr_out=0x1000 >> 2,
                       pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       ina_type=1, inb_type=1, load_bias=2, data_x=None,
                       data_w=data['fc3']['weight'][(core_x, core_y)],
                       data_b=data['fc3']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x1000 >> 2, addr_out=0x2000 >> 2,
                        cin=32, cout=10, px=1, py=1,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=in_cut_start_dict['fc_cut3'], row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os
    from generator.LeNet.g0 import gen_0_map_config
    from generator.LeNet.g2 import gen_2_map_config
    from generator.LeNet.lenet_data import generate_data
    from generator.LeNet.lenet_model.lenet_data_handler import LeNetDataHandler
    from generator.LeNet.lenet_cut_config import QuantizationConfig

    case_file_name = 'LeNet_002'
    chip = (0, 0)
    send_to_fpga = True

    handler = LeNetDataHandler()
    data = generate_data(handler)
    cuts = QuantizationConfig()
    config = MapConfigGen()

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1
    phase[1] = 0
    phase[2] = 0
    phase[3] = 0
    phase[4] = 0
    phase[4] = 0

    phase[:] = 1

    clock_in_phase = 20_0000
    config_0 = gen_0_map_config(phase, clock_in_phase, size_x=1, size_y=1, data=data, out_data_en=True, chip=(0, 0),
                                in_data_en=not send_to_fpga)
    config.add_config(config_0, core_offset=(1, 0))

    config_1 = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, in_cut_start_dict=cuts,
                                in_data_en=True, out_data_en=True, chip=chip, data=data, init_data=True)
    config.add_config(config_1, core_offset=(0, 0))

    config_2 = gen_2_map_config(phase, clock_in_phase, size_x=8, size_y=1, data=data, in_data_en=True,
                                out_data_en=True, chip=(0, 0), init_data=True, send_to_fpga=send_to_fpga)
    config.add_config(config_2, core_offset=(0, 1))

    MapConfigGen.add_router_info(map_config=config.map_config)

    prim = {
        'axon': None, 'soma1': None, 'router': None,
        'soma2': p06(addr_in=0x0000 >> 2, addr_out=0x8400, addr_ciso=0x10000 >> 2, length_in=1024,
                     num_in=12, length_ciso=1, num_ciso=12, length_out=1024, num_out=12,
                     type_in=1, type_out=1, data_in=None)
    }

    MapConfigGen.add_prim_at_the_beginning(config.map_config, prim=prim)

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
