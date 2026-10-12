import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.data_handler import ResNetDataHandler


def gen_g15_ob_map_config(phase_en, clock_in_phase, data=None,
                          in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0),
                          init_data=None):
    """
        ResNet-50 5-Chip Group-15 Output Buffer
        core array : [(0, 0), (4, 0), (8, 0), (12, 0)]
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
    for core_x, core_y in [(0, 0), (4, 0), (8, 0), (12, 0)]:
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group = map_config[(chip, 0)][0]

    # ******** 数据交互 ********

    #  接收24*512    发送 24*512
    if phase_en[0]:
        for core_x, core_y in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x0000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x0000 >> 2, addr_out=0x0000 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if out_data_en:
                soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x9000, cin=512, cout=512, px=12, py=2, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer4.1.conv1']['input1'][(0, core_x // 4)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=in_data_en, send_num=24 * 512 // 8 - 1, receive_num=7,
                         addr_din_base=0x380, addr_din_length=1535, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=24 * 512 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0, length_in=512, length_out=512,
                        length_ciso=1, num_in=24, num_ciso=24, num_out=24, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 接收25*512   发送 25*512
    if phase_en[1]:
        for core_x, core_y in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = None
            soma1 = None
            if out_data_en:
                soma1 = pX5(mode='max', addr_in=0x1c000 >> 2, addr_out=0x9000, cin=512, cout=512, px=5, py=5, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                            in_row_max=1,
                            data_in=data['layer4.1.conv1']['input3'][(0, core_x // 4)] if init_data else None)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=out_data_en,
                         receive_en=in_data_en, send_num=25 * 512 // 8 - 1, receive_num=7,
                         addr_din_base=0x380, addr_din_length=1599, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=0, nx=0, ny=0, data_in=None)
            if out_data_en:
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=25 * 512 // 8 - 1,
                                A_offset=0, Const=0, EN=1)
            soma2 = p06(addr_in=0x8380, addr_out=0x13000 >> 2, addr_ciso=0, length_in=512, length_out=512,
                        length_ciso=1, num_in=25, num_ciso=25, num_out=25, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 搬运
    if phase_en[2]:
        for core_x, core_y in [(0, 0), (4, 0), (8, 0), (12, 0)]:
            axon = None
            soma1 = pX5(mode='max', addr_in=0x10000 >> 2, addr_out=0x19000 >> 2, cin=512, cout=512, px=7, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0,
                        in_row_max=1, data_in=None)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00015OB'
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
    config = gen_g15_ob_map_config(phase, clock_in_phase=clock_in_phase, data=data, in_data_en=False,
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
