import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_2chip.prims import p06, p09, p41
from generator.mapping_utils.map_config_gen import MapConfigGen


def gen_g1_ib_map_config(phase_en, clock_in_phase, data=None,
                     in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0)):
    """
        ResNet-50 5-Chip Group1 Input Buffer
        core array : (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)
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
    for core_x, core_y in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]:
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    # 77 * 128
    if phase_en[0]:
        for core_x, core_y in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1c100 >> 2,
                       addr_inb=0x1c100 >> 2, addr_bias=0x1c100 >> 2, addr_out=0x1c100 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            soma1 = None
            din_length = 77 * 128 // 8 - 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x0, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=din_length + 1,
                        num_in=8, length_ciso=1, num_ciso=8, length_out=din_length + 1, num_out=8,
                        type_in=1, type_out=1)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[1]:
        for core_x, core_y in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1c100 >> 2,
                       addr_inb=0x1c100 >> 2, addr_bias=0x1c100 >> 2, addr_out=0x1c100 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            soma1 = None
            din_length = 76 * 128 // 8 - 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x0, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            soma2 = p06(addr_in=0x8380, addr_out=0x2680 >> 2, addr_ciso=0x10000 >> 2, length_in=din_length + 1,
                        num_in=8, length_ciso=1, num_ciso=8, length_out=din_length + 1, num_out=8,
                        type_in=1, type_out=1)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[2]:
        for core_x, core_y in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1c100 >> 2,
                       addr_inb=0x1c100 >> 2, addr_bias=0x1c100 >> 2, addr_out=0x1c100 >> 2, axon_delay=True,
                       L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            soma1 = None
            din_length = 76 * 128 // 8 - 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x0, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            soma2 = p06(addr_in=0x8380, addr_out=0x4c80 >> 2, addr_ciso=0x10000 >> 2, length_in=din_length + 1,
                        num_in=8, length_ciso=1, num_ciso=8, length_out=din_length + 1, num_out=8,
                        type_in=1, type_out=1)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    case_file_name = 'R00001IB'

    phase = np.zeros(50).astype(int)  # 39-49表示组件数据传输的Phase

    phase[0] = 1    # 接收
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1

    phase[4] = 1    # 搬运

    phase[5] = 1    # 发送
    phase[6] = 1
    phase[7] = 1
    phase[8] = 1

    config = gen_g1_ib_map_config(phase, clock_in_phase=100_000, in_data_en=False,
                                  out_data_en=False, chip=(1, 0))
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = 100_000

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

    tester = TestEngine(config, test_config)
    assert tester.run_test()
