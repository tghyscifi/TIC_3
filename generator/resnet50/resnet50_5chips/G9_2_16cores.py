import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler


def gen_g9_2_map_config(phase_en, clock_in_phase, size_x, size_y, cuts, data=None,
                        in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, chip=(0, 0)):
    """
        ResNet-50 5-Chip Group-9-2
        core array : 2 * 8
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

    # data in
    if data is None:
        data = {
            'layer3.0.downsample.0': {
                'input': {},
                'weight': {},
                'bias': {}
            }  # L23e
        }
        for core_y, core_x in product(range(size_y), range(size_x)):
            data['layer3.0.downsample.0']['input'][(core_x, core_y)] = []
            data['layer3.0.downsample.0']['weight'][(core_x, core_y)] = []
            data['layer3.0.downsample.0']['bias'][(core_x, core_y)] = []

    # ******** 数据交互 ********
    offset = 2

    # 接收 24*512
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x17a80 >> 2,
                       addr_inb=0x17a80 >> 2, addr_bias=0x17a80 >> 2, addr_out=0x17a80 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0,
                         receive_en=in_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=24*512//8-1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=24*512//8-1, nx=0, ny=0, data_in=None)
            if in_data_en:
                if core_x // 2 == 0:
                    if (core_x, core_y) == (0, 0):
                        router.CXY, router.Nx, router.Ny = 1, 0, 2
                    elif (core_x, core_y) == (1, 0):
                        router.CXY, router.Nx, router.Ny = 1, 0, 3
                    elif (core_x, core_y) == (0, 1):
                        router.CXY, router.Nx, router.Ny = 1, 2, 3
                    elif (core_x, core_y) == (1, 1):
                        router.CXY, router.Nx, router.Ny = 1, 2, 4
                    else:
                        raise ValueError
                else:
                    router.CXY, router.Nx, router.Ny = 1, -2, 0
            soma2 = p06(addr_in=0x8380, addr_out=0x10000 >> 2, addr_ciso=0, length_in=512, length_out=512,
                        length_ciso=1, num_in=24, num_ciso=24, num_out=24, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 接收 25*512
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x17a80 >> 2,
                       addr_inb=0x17a80 >> 2, addr_bias=0x17a80 >> 2, addr_out=0x17a80 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=0,
                         receive_en=in_data_en, send_num=0, receive_num=0,
                         addr_din_base=0x380, addr_din_length=25*512//8-1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, relay_num=25*512//8-1, nx=0, ny=0, data_in=None)
            if in_data_en:
                if core_x // 2 == 0:
                    if (core_x, core_y) == (0, 0):
                        router.CXY, router.Nx, router.Ny = 1, 0, 2
                    elif (core_x, core_y) == (1, 0):
                        router.CXY, router.Nx, router.Ny = 1, 0, 3
                    elif (core_x, core_y) == (0, 1):
                        router.CXY, router.Nx, router.Ny = 1, 2, 3
                    elif (core_x, core_y) == (1, 1):
                        router.CXY, router.Nx, router.Ny = 1, 2, 4
                    else:
                        raise ValueError
                else:
                    router.CXY, router.Nx, router.Ny = 1, -2, 0
            soma2 = p06(addr_in=0x8380, addr_out=0x13000 >> 2, addr_ciso=0, length_in=512, length_out=512,
                        length_ciso=1, num_in=25, num_ciso=25, num_out=25, type_in=1, type_out=1,
                        in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None, data_ciso=None)
            if not in_data_en:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ******** 开始计算 *******

    # L23e 卷积，流水截取 1/2
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x % 2 == 0:
                addr_out = 0x16200 >> 2
            else:
                addr_out = 0x17a80 >> 2
            axon = p41(px=7, py=7, cin=512, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x1fe00 >> 2, addr_out=0x19300 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None if in_data_en else data['layer3.0.downsample.0']['input'][(core_x, core_y)],
                       data_w=data['layer3.0.downsample.0']['weight'][(core_x, core_y)],
                       data_b=data['layer3.0.downsample.0']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19300 >> 2, addr_out=addr_out, cin=128, cout=128, px=7, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer3.0.cut4'],
                        row_ck_on=1, in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换输入数据 - 1/2 - 25*512
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = pX5(mode='max', addr_in=0x10000 >> 2, addr_out=0x9000, cin=512, cout=512, px=5, py=5, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1599, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1599, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=1 if core_x % 2 == 0 else -1, Y=0,
                            A=0, pack_per_Rhead=1599, A_offset=0, Const=0, EN=1)
            soma2 = pX5(mode='max', addr_in=0x8380, addr_out=0x10000 >> 2, cin=512, cout=512, px=5, py=5, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=3)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 交换输入数据 - 2/2 - 24*512
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = pX5(mode='max', addr_in=0x13200 >> 2, addr_out=0x9000, cin=512, cout=512, px=4, py=6, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=3)
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                 limit=0x380)
            router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=1535, receive_num=0, addr_din_base=0x380,
                         addr_din_length=1535, addr_rhead_base=addr_rhead_base, addr_rhead_length=0,
                         addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, cxy=0, relay_num=0, data_in=None)
            router.addRHead(S=0, T=1, P=0, Q=0, X=1 if core_x % 2 == 0 else -1, Y=0,
                            A=0, pack_per_Rhead=1535, A_offset=0, Const=0, EN=1)
            soma2 = pX5(mode='max', addr_in=0x8380, addr_out=0x13200 >> 2, cin=512, cout=512, px=4, py=6, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                        row_ck_on=0, in_row_max=3)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # L23e 卷积，流水截取 2/2
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x % 2 == 0:
                addr_out = 0x17a80 >> 2
            else:
                addr_out = 0x16200 >> 2
            axon = p41(px=7, py=7, cin=512, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x10000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x1fe00 >> 2, addr_out=0x19300 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None,
                       data_w=data['layer3.0.downsample.0']['weight'][(core_x, core_y)],
                       data_b=data['layer3.0.downsample.0']['bias'][(core_x, core_y)])
            soma1 = pX5(mode='max', addr_in=0x19300 >> 2, addr_out=addr_out, cin=128, cout=128, px=7, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['layer3.0.cut4'],
                        row_ck_on=1, in_row_max=3)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 数据搬运（方便check）
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = p06(addr_in=0x16200 >> 2, addr_out=0x10000 >> 2, addr_ciso=0x0, length_in=7 * 128,
                        length_out=7 * 128, length_ciso=1, num_in=14, num_ciso=14, num_out=14,
                        type_in=1, type_out=1, in_cut_start=0, in_row_max=0, row_ck_on=0, data_in=None,
                        data_ciso=None)
            router, soma2 = None, None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R000092'
    cuts = Resnet50Cuts()

    phase = np.zeros(50).astype(int)
    # 39~49 表示组件数据传输的Phase

    phase[0] = 1  # L23e 卷积，流水截取 1/2
    phase[1] = 1  # 交换输入数据 - 1/2 - 25*512
    phase[2] = 1  # 交换输入数据 - 2/2 - 24*512
    phase[3] = 1  # L23e 卷积，流水截取 2/2
    phase[4] = 1

    from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data

    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)

    clock_in_phase = 50_000
    config = gen_g9_2_map_config(phase, clock_in_phase=clock_in_phase, size_x=8, size_y=2, data=data, in_data_en=False,
                                 out_data_en=False, cuts=cuts)
    MapConfigGen.add_router_info(map_config=config)

    config['sim_clock'] = min(200_000, len(config[((0, 0), 0)][0][((0, 0), (0, 0))]['prims']) * clock_in_phase)

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
