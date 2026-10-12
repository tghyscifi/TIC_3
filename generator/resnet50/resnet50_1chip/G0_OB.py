import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.prims import p06, p09, p41
from generator.mapping_utils.map_config_gen import MapConfigGen


def gen_g0_ob_map_config(phase_en, clock_in_phase, data=None, send_to_fpga=False,
                         in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0)):
    """
        ResNet-50 5-Chip Group0 Output Buffer
        core array : (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)
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
    for core_x, core_y in [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)]:
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    # dst core (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)

    # ******** Send ********
    # 77 * 128   接收 32
    if phase_en[0]:
        for core_x, core_y in [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1c100 >> 2,
                       addr_inb=0x1c100 >> 2, addr_bias=0x1c100 >> 2, addr_out=0x1c100 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (10, 0):
                axon = None
            din_length = 77 * 128 // 8 - 1
            soma1 = p06(addr_in=0x0000 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=din_length + 1,
                        num_in=8, length_ciso=1, num_ciso=8, length_out=din_length + 1, num_out=8,
                        type_in=1, type_out=1, data_in=data['conv1a1_input2'][(core_x - 10, core_y)])
            if not out_data_en:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=(in_data_en and (not send_to_fpga)) and core_x == 10,
                         send_num=din_length, receive_num=0,
                         addr_din_base=0x380, addr_din_length=32//8-1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-10, Y=0, A=0, pack_per_Rhead=din_length,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 76 * 128
    if phase_en[1]:
        for core_x, core_y in [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1c100 >> 2,
                       addr_inb=0x1c100 >> 2, addr_bias=0x1c100 >> 2, addr_out=0x1c100 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (10, 0):
                axon = None
            din_length = 76 * 128 // 8 - 1
            soma1 = p06(addr_in=0x2680 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=din_length + 1,
                        num_in=8, length_ciso=1, num_ciso=8, length_out=din_length + 1, num_out=8,
                        type_in=1, type_out=1, data_in=data['conv1a1_input3'][(core_x - 10, core_y)])
            if not out_data_en:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=0, send_num=din_length, receive_num=0,
                         addr_din_base=0x380, addr_din_length=din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-10, Y=0, A=0, pack_per_Rhead=din_length,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 76 * 128
    if phase_en[2]:
        for core_x, core_y in [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1c100 >> 2,
                       addr_inb=0x1c100 >> 2, addr_bias=0x1c100 >> 2, addr_out=0x1c100 >> 2, axon_delay=True,
                       L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            axon = None
            din_length = 76 * 128 // 8 - 1
            soma1 = p06(addr_in=0x4c80 >> 2, addr_out=0x9000, addr_ciso=0x10000 >> 2, length_in=din_length + 1,
                        num_in=8, length_ciso=1, num_ciso=8, length_out=din_length + 1, num_out=8,
                        type_in=1, type_out=1, data_in=data['conv1a1_input4'][(core_x - 10, core_y)])
            if not out_data_en:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=0, send_num=din_length, receive_num=0,
                         addr_din_base=0x380, addr_din_length=din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=-10, Y=0, A=0, pack_per_Rhead=din_length,
                                A_offset=0, Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    case_file_name = 'R00000OB'

    phase = np.zeros(50).astype(int)  # 39-49表示组件数据传输的Phase

    delay_l4 = (28,) * 9
    delay_l5 = (28,) * 9

    phase[:] = 1

    config = gen_g0_ob_map_config(phase, clock_in_phase=100_000, data=None, in_data_en=0,
                                  out_data_en=1, chip=(0, 0), delay_l4=delay_l4, delay_l5=delay_l5)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = 200_000

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

    import pickle

    with open(case_file_name + '.map_config', 'wb') as f:
        pickle.dump(config, f)

    tester = TestEngine(config, test_config)
    assert tester.run_test()
