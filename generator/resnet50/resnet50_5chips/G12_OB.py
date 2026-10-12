import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.data_handler import ResNetDataHandler


def gen_g12_ob_map_config(phase_en, clock_in_phase, data=None,
                          in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, init_data=None):
    """
        ResNet-50 5-Chip Group-12 Output Buffer
        core array : chip(0, 2)   [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]
                     chip(0, 1)   [(12, 9), (13, 9), (14, 9), (15, 9)]
                     chip(1, 2)   [(9, 0), (11, 0), (13, 0), (15, 0)]
    """
    chip_0 = (0, 2)
    chip_1 = (0, 1)
    chip_2 = (1, 2)
    map_config = {
        'sim_clock': None,
        (chip_0, 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        },
        (chip_1, 0): {
            1: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        },
        (chip_2, 0): {
            2: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_x, core_y in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
        map_config[(chip_0, 0)][0][(chip_0, (core_x, core_y))] = {
            'prims': []
        }
    for core_x, core_y in [(12, 9), (13, 9), (14, 9), (15, 9)]:
        map_config[(chip_1, 0)][1][(chip_1, (core_x, core_y))] = {
            'prims': []
        }
    for core_x, core_y in [(9, 0), (11, 0), (13, 0), (15, 0)]:
        map_config[(chip_2, 0)][2][(chip_2, (core_x, core_y))] = {
            'prims': []
        }
    phase_group_0 = map_config[(chip_0, 0)][0]
    phase_group_1 = map_config[(chip_1, 0)][1]
    phase_group_2 = map_config[(chip_2, 0)][2]

    # ******** 数据交互 ********

    #  接收 7*7*256  发送 7*7*256
    if phase_en[0]:
        for core_x, core_y in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0000 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (12, 0):
                axon = None
            soma1 = None
            if out_data_en:
                soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=256, cout=256, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.3.cut5']['output3'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group_0[(chip_0, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=in_data_en, send_num=7 * 7 * 256 // 8 - 1, receive_num=3,
                         addr_din_base=0x380, addr_din_length=7 * 7 * 256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 1:
                    dst_x, dst_y = core_x, -1
                elif core_y == 0:
                    dst_x, dst_y = 25 + (core_x - 12) * 2, 0
                else:
                    raise ValueError
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                pack_per_Rhead=7 * 7 * 256 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0, length_in=256, length_out=256,
                        length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group_0[(chip_0, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

        for core_x, core_y in [(12, 9), (13, 9), (14, 9), (15, 9)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0000 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (12, 9):
                axon = None
            soma1 = None
            router = p09(rhead_mode=1, send_en=0,
                         receive_en=out_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=7 * 7 * 256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=out_data_en, relay_num=7 * 7 * 256 // 8 - 1, nx=0, ny=0, data_in=None)
            router.Nx = 12 + (core_x - 12) * 1
            soma2 = None
            phase_group_1[(chip_1, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
        for core_x, core_y in [(9, 0), (11, 0), (13, 0), (15, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0000 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (9, 0):
                axon = None
            soma1 = None
            router = p09(rhead_mode=1, send_en=0,
                         receive_en=out_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=7 * 7 * 256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=out_data_en, relay_num=7 * 7 * 256 // 8 - 1, nx=0, ny=-1, data_in=None)
            soma2 = None
            phase_group_2[(chip_2, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    #  接收 7*7*256   发送 7*7*256
    if phase_en[1]:
        for core_x, core_y in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0000 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            axon = None
            soma1 = None
            if out_data_en:
                soma1 = pX5(mode='max', addr_in=0x1c100 >> 2, addr_out=0x9000, cin=256, cout=256, px=7, py=7, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer3.3.cut5']['output4'][(core_x, core_y)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group_0[(chip_0, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=in_data_en, send_num=7 * 7 * 256 // 8 - 1, receive_num=3,
                         addr_din_base=0x380, addr_din_length=7 * 7 * 256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                if core_y == 1:
                    dst_x, dst_y = core_x, -1
                elif core_y == 0:
                    dst_x, dst_y = 25 + (core_x - 12) * 2, 0
                else:
                    raise ValueError
                router.addRHead(S=0, T=1, P=0, Q=1, X=dst_x - core_x, Y=dst_y - core_y, A=0,
                                pack_per_Rhead=7 * 7 * 256 // 8 - 1, A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x13100 >> 2, addr_ciso=0, length_in=256, length_out=256,
                        length_ciso=1, num_in=49, num_ciso=49, num_out=49, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group_0[(chip_0, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

        for core_x, core_y in [(12, 9), (13, 9), (14, 9), (15, 9)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0000 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            axon = None
            soma1 = None
            router = p09(rhead_mode=1, send_en=0,
                         receive_en=out_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=7 * 7 * 256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=out_data_en, relay_num=7 * 7 * 256 // 8 - 1, nx=0, ny=0, data_in=None)
            router.Nx = 12 + (core_x - 12) * 1
            soma2 = None
            phase_group_1[(chip_1, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
        for core_x, core_y in [(9, 0), (11, 0), (13, 0), (15, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0000 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            axon = None
            soma1 = None
            router = p09(rhead_mode=1, send_en=0,
                         receive_en=out_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=7 * 7 * 256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=out_data_en, relay_num=7 * 7 * 256 // 8 - 1, nx=0, ny=-1, data_in=None)
            soma2 = None
            phase_group_2[(chip_2, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    #  搬运
    if phase_en[2]:
        for core_x, core_y in [(12, 0), (13, 0), (14, 0), (15, 0), (12, 1), (13, 1), (14, 1), (15, 1)]:
            axon = None
            soma1 = pX5(mode='max', addr_in=0x10000 >> 2, addr_out=0x19000 >> 2, cin=256, cout=256, px=14, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                        in_row_max=1, data_in=None)
            router = None
            soma2 = None
            phase_group_0[(chip_0, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00016IB'
    chip = (1, 0)
    cuts = Resnet50Cuts()
    offset = 2
    delay = (28,) * 9

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1
    phase[1] = 1

    from generator.resnet50.resnet50_5chips.G16_data import generate_g16_data

    handler = ResNetDataHandler()
    data = generate_g16_data(handler, size_y=4, size_x=16)

    clock_in_phase = 100_000
    config = gen_g16_ib_map_config(phase, clock_in_phase=clock_in_phase, data=data, in_data_en=False,
                                   out_data_en=False, chip=chip)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[(chip, 0)][0][(chip, (0, 0))]['prims']) * clock_in_phase)

    from generator.test_engine import TestMode, TestEngine
    from generator.test_engine.test_config import HardwareDebugFileSwitch

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
        'debug_file_switch': HardwareDebugFileSwitch().close_all.dict,
        'test_group_phase': [(0, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
