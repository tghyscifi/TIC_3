import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_2chip.prims import p04, p06, p26, p09, pX5, p41, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_2chip.G1_data import generate_g1_data
from generator.resnet50.resnet50_2chip.resnet50_2chip_data_handler import ResNetDataHandler
from itertools import product


def gen_g5_map_config(phase_en, clock_in_phase, size_x, size_y, cuts, static_data=None, g4_en=False, g0_en=False,
                      delay_l4=None, delay_l5=None, chip=(0, 0), init_data=True):
    """
        ResNet-50 5-Chip Group5
        core array : 2 * 16
    """
    offset = 1
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

    # 接收 Xfc = 256； --  发送 Yfc = 32；
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if g0_en and (core_x, core_y) == (0, 0):
                soma1 = p06(addr_in=0xffe0 >> 2, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=32,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=32, num_out=1, type_in=1, type_out=1,
                            data_in=static_data['fca2_output_all'][(core_x, core_y)] if init_data else None)

            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=(g0_en) and (core_x, core_y) == (0, 0), receive_en=g4_en,
                         send_num=32 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, nx=0, ny=0, relay_num=256 // 8 - 1, data_in=None)
            if core_y == 0:
                if core_x == 15:
                    router.CXY, router.Nx, router.Ny = 1, 0, 1
                else:
                    router.CXY, router.Nx, router.Ny = 1, 1, 0
            else:
                if core_x == 0:
                    router.CXY, router.Nx, router.Ny = 0, 0, 0
                else:
                    router.CXY, router.Nx, router.Ny = 1, -1, 0
            if (g0_en) and (core_x == 0 and core_y == 0):
                router.addRHead(S=0, T=1, P=0, Q=0, X=7 - core_x, Y=-10 - core_y, A=0,
                                pack_per_Rhead=32 // 8 - 1, A_offset=0, Const=0, EN=g0_en)
            if not g4_en:
                soma2 = None
            else:
                soma2 = p06(addr_in=0x8380, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2,
                            length_in=256 // 8,
                            num_in=8, length_ciso=1, num_ciso=8, length_out=256 // 8, num_out=8,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    in_data_en = g4_en

    # fc_0 relu，流水发送至core（0，0）·
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if static_data is None:
                data_x = []
                data_w = []
                data_b = []
            else:
                data_x = None if in_data_en else static_data['fc_0_input'][(core_x, core_y)]
                data_w = static_data['fc_0_weight'][(core_x, core_y)]
                data_b = static_data['fc_0_bias'][(core_x, core_y)]
            axon = p41(px=1, py=1, cin=256, cout=32, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x12000 >> 2,
                       addr_out=0x47c0 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=data_x, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x47C0 >> 2, addr_out=0x9000, cin=32, cout=32, px=1, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['fc_0_cut'],
                        row_ck_on=1, in_row_max=1)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=1, receive_en=1 if (core_x, core_y) == (0, 0) else 0,
                         send_num=32 // 8 - 1, receive_num=31,
                         addr_din_base=0x380, addr_din_length=1024 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=0 - core_x, Y=0 - core_y, A=(core_x + core_y * size_x) * (32 // 8),
                            pack_per_Rhead=32 // 8 - 1, A_offset=0, Const=0, EN=1)
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 多播
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if (core_x, core_y) == (0, 0):
                send_en, recv_en = 1, 0
            else:
                if core_x < 8 and core_y == 0:
                    send_en, recv_en = 0, 1
                else:
                    send_en, recv_en = 0, 0
            if core_y == 0:
                if core_x == 0:
                    cxy, nx, ny = 0, 0, 0
                elif core_x < 7:
                    cxy, nx, ny = 1, 1, 0
                else:
                    cxy, nx, ny = 0, 0, 0
            else:
                cxy, nx, ny = 0, 0, 0
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=send_en, receive_en=recv_en,
                         send_num=1024 // 8 - 1, receive_num=0,
                         addr_din_base=0x380, addr_din_length=1024 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x380, addr_dout_length=1024 // 16 - 1, soma_in_en=0,
                         cxy=cxy, nx=nx, ny=ny, relay_num=1024 // 8 - 1,
                         data_in=None)
            if core_x == 0 and core_y == 0:
                router.addRHead(S=0, T=1, P=0, Q=1, X=1, Y=0, A=0,
                                pack_per_Rhead=1024 // 8 - 1, A_offset=0, Const=0, EN=1)
            if core_x < 8 and core_y == 0:
                soma2 = pX5(mode='max', addr_in=0x8380, addr_out=0x0000 >> 2, cin=1024, cout=1024, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                            row_ck_on=0, in_row_max=0)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # fc_1 relu，流水发送至core（0，0）
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x < 8 and core_y == 0:
                axon = p41(px=1, py=1, cin=1024, cout=32, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=0x0000 >> 2, addr_inb=0x12080 >> 2, addr_bias=0x1a080 >> 2,
                           addr_out=0x47c0 >> 2, ina_type=1, inb_type=1,
                           load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None,
                           data_w=static_data['fc_1_weight'][(core_x, core_y)],
                           data_b=static_data['fc_1_bias'][(core_x, core_y)])
                soma1 = pX5(mode='max', addr_in=0x47C0 >> 2, addr_out=0x9000, cin=32, cout=32, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['fc_1_cut'],
                            row_ck_on=1, in_row_max=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1 if (core_x, core_y) == (0, 0) else 0,
                             send_num=32 // 8 - 1, receive_num=7,
                             addr_din_base=0x380, addr_din_length=256 // 8 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0 - core_x, Y=0 - core_y, A=core_x * (32 // 8),
                                pack_per_Rhead=32 // 8 - 1, A_offset=0, Const=0, EN=1)
            else:
                axon, soma1, router = None, None, None

            if (core_x, core_y) == (0, 0):
                soma2 = pX5(mode='max', addr_in=0x8380, addr_out=0x0000 >> 2, cin=256, cout=256, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                            row_ck_on=0, in_row_max=0)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # fc
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x, core_y) == (0, 0):
                axon = p41(px=1, py=1, cin=256, cout=32, kx=1, ky=1, sx=1, sy=1,
                           addr_ina=0x0000 >> 2, addr_inb=0x1a100 >> 2, addr_bias=0x1c100 >> 2,
                           addr_out=0x47c0 >> 2, ina_type=1, inb_type=1,
                           load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                           data_x=None,
                           data_w=static_data['fc_weight'][(core_x, core_y)],
                           data_b=static_data['fc_bias'][(core_x, core_y)])
                soma1 = pX5(mode='max', addr_in=0x47C0 >> 2, addr_out=0xffe0 >> 2, cin=32, cout=32, px=1, py=1, kx=1,
                            ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['fc_cut'],
                            row_ck_on=1, in_row_max=1)
            else:
                axon, soma1 = None, None
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    case_file_name = 'R00001'

    phase = np.zeros(50).astype(int)  # 39-49表示组件数据传输的Phase

    cuts = Resnet50Cuts()

    offset = 2

    phase[0] = 1
    phase[1] = 1

    # FCa2
    phase[offset + 0] = 1
    # Conv1a1数据整理
    phase[offset + 1] = 1
    phase[offset + 2] = 1
    phase[offset + 3] = 1
    # Conv1a1
    phase[offset + 4] = 1
    # Conv1a2 MaxPool数据整理
    phase[offset + 5] = 1
    phase[offset + 6] = 1
    phase[offset + 7] = 1
    # Conv1a2 MaxPool
    phase[offset + 8] = 1

    handler = ResNetDataHandler()
    static_data = generate_g1_data(handler, size_y=2, size_x=16)
    config = gen_g1_map_config(phase, clock_in_phase=200_000, size_x=16, size_y=2, cuts=cuts,
                               static_data=static_data, in_data_en=False, out_data_en=False, chip=(1, 0))
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

    tester = TestEngine(config, test_config)
    assert tester.run_test()
