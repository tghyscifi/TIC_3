import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.prims import p06, p26, p09, pX5, p41
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_5chips.G5_data import generate_g5_data
from generator.resnet50.data_handler import ResNetDataHandler
from itertools import product


def gen_g5_map_config1(phase_en, clock_in_phase, size_x, size_y, cuts, static_data=None, chip=(0, 0),
                       in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None):
    """
        ResNet-50 5-Chip Group5
        core array : 3 * 14
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

    # ********************* 组间数据传输 *******************************
    offset = 4

    # 接收 28*256
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1e400 >> 2,
                       addr_inb=0x1e400 >> 2, addr_bias=0x1e400 >> 2, addr_out=0x1e400 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=895, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None, cxy=0, nx=0, ny=0, relay_num=895)
            if in_data_en:
                if core_x // 7 == 1:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                elif core_x // 7 == 0:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                else:
                    raise ValueError
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x10000 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 接收 28*256
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1e400 >> 2,
                       addr_inb=0x1e400 >> 2, addr_bias=0x1e400 >> 2, addr_out=0x1e400 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=895, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None, cxy=0, nx=0, ny=0, relay_num=895)
            if in_data_en:
                if core_x // 7 == 1:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                elif core_x // 7 == 0:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                else:
                    raise ValueError
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x11c00 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 接收 28*256
    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1e400 >> 2,
                       addr_inb=0x1e400 >> 2, addr_bias=0x1e400 >> 2, addr_out=0x1e400 >> 2, axon_delay=True,
                       L4_num=delay_l4[2], L5_num=delay_l5[2], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=895, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None, cxy=0, nx=0, ny=0, relay_num=895)
            if in_data_en:
                if core_x // 7 == 1:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                elif core_x // 7 == 0:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                else:
                    raise ValueError
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x13800 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 接收 28*256
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x1e400 >> 2,
                       addr_inb=0x1e400 >> 2, addr_bias=0x1e400 >> 2, addr_out=0x1e400 >> 2, axon_delay=True,
                       L4_num=delay_l4[3], L5_num=delay_l5[3], A2S2_mode=True)
            # TODO 如果这个axon留着会即时原语那里c仿真器出错
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=895, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None, cxy=0, nx=0, ny=0, relay_num=895)
            if in_data_en:
                if core_x // 7 == 1:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                elif core_x // 7 == 0:
                    router.CXY, router.Nx, router.Ny = 1, 0, -1
                else:
                    raise ValueError
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x15400 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ************************ 计算部分 **************************

    # Conv3a3e + 截取
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if static_data is None:
                data_x = []
                data_w = []
                data_b = []
            else:
                data_x = None if in_data_en else static_data['conv3a3e_input'][(core_x, core_y)]
                data_w = static_data['conv3a3e_weight'][(core_x, core_y)]
                data_b = static_data['conv3a3e_bias'][(core_x, core_y)]
            axon = p41(px=4, py=28, cin=256, cout=256, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x17000 >> 2, addr_out=0x17400 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=data_x, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x17400 >> 2, addr_out=0x10000 >> 2, cin=256, cout=256, px=4, py=28, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer2.0.cut4'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 数据整理
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p26(addr_in=0x10000 >> 2, addr_out=0x17400 >> 2, addr_ciso=0x1AC00 >> 2, length_in=256,
                        num_in=4 * 28, length_ciso=128, num_ciso=4 * 28, length_out=128, num_out=4 * 28, type_in=1,
                        type_out=1)
            router = None
            soma2 = p06(addr_in=0x19000 >> 2, addr_out=0x1E400 >> 2, addr_ciso=0x0000 >> 2, length_in=128,
                        num_in=2 * 28, length_ciso=1, num_ciso=2 * 28, length_out=128, num_out=2 * 28,
                        type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x1AC00 >> 2, addr_out=0x19000 >> 2, addr_ciso=0x0000 >> 2, length_in=128,
                        num_in=2 * 28, length_ciso=1, num_ciso=2 * 28, length_out=128, num_out=2 * 28,
                        type_in=1, type_out=1)
            router = None
            soma2 = p06(addr_in=0x1E400 >> 2, addr_out=0x1AC00 >> 2, addr_ciso=0x0000 >> 2, length_in=128,
                        num_in=2 * 28, length_ciso=1, num_ciso=2 * 28, length_out=128, num_out=2 * 28,
                        type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    case_file_name = 'R00005_1'

    phase = np.zeros(50).astype(int)  # 39-49表示组件数据传输的Phase
    cuts = Resnet50Cuts()

    # Conv3a3e + 截取
    phase[0] = 1
    # 即时原语数据整理
    phase[1] = 1
    phase[2] = 1

    handler = ResNetDataHandler()
    static_data = generate_g5_data(handler, size_y=1, size_x=14)
    config = gen_g5_map_config1(phase, clock_in_phase=200_000, size_x=14, size_y=1, cuts=cuts,
                                static_data=static_data, in_data_en=False, out_data_en=False)
    MapConfigGen.add_router_info(map_config=config, group_idx_list=[0], chip_x_num=1, chip_y_num=1)

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
