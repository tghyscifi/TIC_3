import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.prims import p04, p06, p26, p09, pX5, p41, p81
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_5chips.G1_data import generate_g1_data
from generator.resnet50.data_handler import ResNetDataHandler
from itertools import product


def gen_g1_map_config(phase_en, clock_in_phase, size_x, size_y, cuts, static_data=None,
                      g1_ib_en=False, g2_en=False, delay_l4=None, delay_l5=None, chip=(0, 0),
                      g17_en=False, g0_en=False,
                      init_data=True):
    """
        ResNet-50 5-Chip Group1
        core array : 2 * 16
    """
    offset = 2
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

    # 接收 X1 = 17*128*3 （14*128*3， 16*128*3）； --  发送 Yfc = 32； Y1 = 4(3) * 28 * 64
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x84c0 >> 2,
                       addr_inb=0x84c0 >> 2, addr_bias=0x84c0 >> 2, addr_out=0x84c0 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            soma1 = None
            if core_x in [0]:
                h_in = 17
            elif core_x in [15]:
                h_in = 16
            else:
                h_in = 14
            if core_x % 2 == 0:
                addr_in = 0xe3e0 >> 2
                length_in = (4 * 28 * 64 + 32) // 8
            else:
                addr_in = 0xeae0 >> 2
                length_in = (3 * 28 * 64 + 32) // 8
            if g0_en or g2_en:
                soma1 = p06(addr_in=addr_in, addr_out=0x9000, addr_ciso=0x0000 >> 2, length_in=length_in * 4,
                            num_in=2, length_ciso=1, num_ciso=2, length_out=length_in * 4, num_out=2,
                            type_in=1, type_out=1,
                            data_in=static_data['mp_out_with_fc_out'][(core_x, core_y)] if init_data else None)
            recv_num = 0
            if (core_x, core_y) == (15, 1):
                if g17_en or g1_ib_en:
                    if g17_en:
                        recv_num += 16
                    if g1_ib_en:
                        recv_num += 3
                    recv_num -= 1
                addr_din_length = (h_in * 128 * 3 + 2048) // 8 - 1
            else:
                if g1_ib_en:
                    recv_num += 2
                addr_din_length = (h_in * 128 * 3) // 8 - 1
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=g0_en or g2_en, receive_en=g17_en or g1_ib_en, send_num=length_in - 1,
                         receive_num=recv_num,
                         addr_din_base=0x380, addr_din_length=addr_din_length, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1,
                         cxy=0, nx=0, ny=0, relay_num=0, data_in=None)
            if g0_en or g2_en:
                if core_x % 8 == 0:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=2 - core_x + 7 * (core_x // 8), Y=2, A=0,
                                    pack_per_Rhead=4 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                elif core_x % 8 == 1:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=3 - core_x + 7 * (core_x // 8), Y=2, A=0,
                                    pack_per_Rhead=3 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                elif core_x % 8 == 2:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=3 - core_x + 7 * (core_x // 8), Y=2, A=3 * 28 * 64 // 8,
                                    pack_per_Rhead=1 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=4 - core_x + 7 * (core_x // 8), Y=2, A=0,
                                    pack_per_Rhead=3 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                elif core_x % 8 == 3:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=4 - core_x + 7 * (core_x // 8), Y=2, A=3 * 28 * 64 // 8,
                                    pack_per_Rhead=1 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=5 - core_x + 7 * (core_x // 8), Y=2, A=0,
                                    pack_per_Rhead=2 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                elif core_x % 8 == 4:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=5 - core_x + 7 * (core_x // 8), Y=2, A=2 * 28 * 64 // 8,
                                    pack_per_Rhead=2 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=6 - core_x + 7 * (core_x // 8), Y=2, A=0,
                                    pack_per_Rhead=2 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                elif core_x % 8 == 5:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=6 - core_x + 7 * (core_x // 8), Y=2, A=2 * 28 * 64 // 8,
                                    pack_per_Rhead=2 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=7 - core_x + 7 * (core_x // 8), Y=2, A=0,
                                    pack_per_Rhead=1 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                elif core_x % 8 == 6:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=7 - core_x + 7 * (core_x // 8), Y=2, A=1 * 28 * 64 // 8,
                                    pack_per_Rhead=3 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=8 - core_x + 7 * (core_x // 8), Y=2, A=0,
                                    pack_per_Rhead=1 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                elif core_x % 8 == 7:
                    router.addRHead(S=0, T=1, P=0, Q=0, X=8 - core_x + 7 * (core_x // 8), Y=2, A=1 * 28 * 64 // 8,
                                    pack_per_Rhead=3 * 28 * 64 // 8 - 1,
                                    A_offset=0, Const=0, EN=g2_en)
                router.addRHead(S=0, T=1, P=0, Q=0, X=10 - core_x, Y=-1 - core_y, A=(core_x + core_y * 16) * 32 // 8,
                                pack_per_Rhead=32 // 8 - 1, A_offset=0, Const=0, EN=g0_en)
            if not (g0_en or g2_en):
                soma2 = None
            else:
                soma2 = p06(addr_in=0x8380, addr_out=0x0800 >> 2, addr_ciso=0x10000 >> 2,
                            length_in=(h_in * 128 * 3) // 8,
                            num_in=8, length_ciso=1, num_ciso=8, length_out=(h_in * 128 * 3) // 8, num_out=8,
                            type_in=1, type_out=1)
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 多播 Xfc = 2048
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            if (core_x, core_y) == (15, 1):
                send_en, recv_en = g17_en, 0
            else:
                send_en, recv_en = 0, g17_en
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=send_en, receive_en=recv_en, send_num=2048 // 8 - 1, receive_num=0,
                         addr_din_base=0x980, addr_din_length=2048 // 8 - 1, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x980, addr_dout_length=2048 // 16 - 1, soma_in_en=0,
                         cxy=0, nx=0, ny=0, relay_num=2048 // 8 - 1, data_in=None)
            if g17_en:
                if core_y == 1:
                    if core_x == 0:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    elif core_x == 15:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                        router.addRHead(S=0, T=1, P=0, Q=1, X=-1, Y=0, A=0, pack_per_Rhead=2048 // 8 - 1,
                                        A_offset=0, Const=0, EN=1)
                    else:
                        router.CXY, router.Nx, router.Ny = 1, -1, 0
                else:
                    if core_x == 15:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    else:
                        router.CXY, router.Nx, router.Ny = 1, 1, 0
            if g17_en:
                soma2 = p06(addr_in=0x8980, addr_out=0x0000 >> 2, addr_ciso=0x10000 >> 2, length_in=2048,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=2048, num_out=1,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    in_data_en = g17_en and g1_ib_en
    # FCa2 + 截取
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if static_data is None:
                data_x = []
                data_w = []
                data_b = []
            else:
                data_x = None if in_data_en else static_data['fca2_input'][(core_x, core_y)]
                data_w = static_data['fca2_weight'][(core_x, core_y)]
                data_b = static_data['fca2_bias'][(core_x, core_y)]
            axon = p04(cin=2048, cout=32, addr_ina=0x0000 >> 2, addr_inb=0x10000 >> 2, addr_bias=0x2180 >> 2,
                       addr_out=0x47C0 >> 2, ina_type=1, inb_type=1, load_bias=2,
                       data_x=data_x, data_w=data_w, data_b=data_b)
            axon = p41(px=1, py=1, cin=axon.cin, cout=axon.cout, kx=1, ky=1, sx=1, sy=1,
                       addr_ina=axon.Addr_InA_base, addr_inb=axon.Addr_InB_base, addr_bias=axon.Addr_Bias_base,
                       addr_out=axon.Addr_V_base, ina_type=axon.InA_type, inb_type=axon.InB_type,
                       load_bias=axon.Load_Bias, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=data_x, data_w=data_w, data_b=data_b)
            soma1 = None
            router = None
            soma2 = pX5(mode='max', addr_in=0x47C0 >> 2, addr_out=0xFFE0 >> 2, cin=32, cout=32, px=1, py=1, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1, in_cut_start=cuts['fc_cut'])
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    if phase_en[offset + 1]:
        if static_data is None:
            for core_y, core_x in product(range(size_y), range(size_x)):
                if core_x == 0:
                    length_in = 128 * 17
                elif core_x == size_x - 1:
                    length_in = 128 * 16
                else:
                    length_in = 128 * 14
                axon = None
                soma1 = p06(addr_in=0x0800 >> 2, addr_out=0x5000 >> 2, addr_ciso=0x0000 >> 2, length_in=length_in,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=1, num_out=3, type_in=1, type_out=1,
                            data_in=[], data_ciso=None)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
        else:
            for core_y, core_x in product(range(size_y), range(size_x)):
                if core_x == 0:
                    length_in = 128 * 17
                elif core_x == size_x - 1:
                    length_in = 128 * 16
                else:
                    length_in = 128 * 14
                data_in = None if in_data_en else static_data['conv1a1_input'][(core_x, core_y)]
                axon = None
                soma1 = p06(addr_in=0x0800 >> 2, addr_out=0x5000 >> 2, addr_ciso=0x0000 >> 2, length_in=length_in,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=1, num_out=3, type_in=1, type_out=1,
                            data_in=data_in, data_ciso=None)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

                # Conv1a1数据整理
    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x == 0:
                axon = None
                soma1 = p06(addr_in=0x0F00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 17,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=128 * 3, num_out=3, type_in=1, type_out=1,
                            data_in=None, data_ciso=None)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=144 - 1, receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=144 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x0800 >> 2, addr_out=0x5000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 17,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=128 * 17, num_out=3, type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            elif core_x == size_x - 1:
                axon = None
                soma1 = None
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=144 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x5000 >> 2, addr_ciso=0x0800 >> 2, length_in=128 * 3,
                            num_in=3, length_ciso=128 * 16, num_ciso=3, length_out=128 * 19, num_out=3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x0D80 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 14,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=128 * 3, num_out=3, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=144 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=144 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=144 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x5000 >> 2, addr_ciso=0x0800 >> 2, length_in=128 * 3,
                            num_in=3, length_ciso=128 * 14, num_ciso=3, length_out=128 * 17, num_out=3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x == size_x - 1:
                axon = None
                soma1 = p06(addr_in=0x0800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 16,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=128 * 2, num_out=3, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=96 - 1, receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=96 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x5000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 19,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=128 * 19, num_out=3, type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            elif core_x == 0:
                axon = None
                soma1 = None
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=96 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                soma2 = p06(addr_in=0x5000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x21000 >> 2, length_in=128 * 17,
                            num_in=3, length_ciso=128 * 2, num_ciso=3, length_out=128 * 19, num_out=3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x0800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 14,
                            num_in=3, length_ciso=1, num_ciso=3, length_out=128 * 2, num_out=3, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=96 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=96 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=96 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x5000 >> 2, addr_out=0x0000 >> 2, addr_ciso=0x21000 >> 2, length_in=128 * 17,
                            num_in=3, length_ciso=128 * 2, num_ciso=3, length_out=128 * 19, num_out=3,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # Conv1a1
    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            data_w = [] if static_data is None else static_data['conv1a1_weight'][(core_x, core_y)]
            data_b = [] if static_data is None else static_data['conv1a1_bias'][(core_x, core_y)]
            axon = p81(px=128, py=19, cin=3, cout=64, kx=7, ky=7, sx=2, sy=2, addr_ina=0x0000 >> 2,
                       addr_inb=0x2200 >> 2, addr_bias=0x46C0 >> 2, addr_out=0x47C0 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, pad_top=0, pad_down=0, pad_left=0, pad_right=0,
                       data_x=None, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x47C0 >> 2, addr_out=0x84C0 >> 2, cin=64, cout=64, px=61, py=7, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0x0, type_in=0, type_out=1, in_cut_start=cuts['conv1_cut'], row_ck_on=1,
                        in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Conv1a2 MaxPool数据整理
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 0:  # 前56列
                axon = None
                soma1 = p06(addr_in=0x84C0 >> 2, addr_out=0x55C0 >> 2, addr_ciso=0x0000 >> 2, length_in=61 * 64,
                            num_in=7, length_ciso=1, num_ciso=7, length_out=56 * 64, num_out=7, type_in=1, type_out=1)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:  # 后57列
                axon = None
                soma1 = p26(addr_in=0x84C0 >> 2, addr_out=0x21000 >> 2, addr_ciso=0x5600 >> 2, length_in=61 * 64,
                            num_in=7, length_ciso=57 * 64, num_ciso=7, length_out=4 * 64, num_out=7, type_in=1,
                            type_out=1)
                router = None
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 0:  # 前56列
                if core_x % 2 == 0:  # 偶数core
                    if core_x == 0:
                        axon = None
                        soma1 = None
                        router = None
                        soma2 = None
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})
                    else:
                        axon = None
                        soma1 = None
                        addr_rhead_base = MapConfigGen.get_router_rhead_base(
                            phase_group[(chip, (core_x, core_y))]['prims'])
                        router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                                     addr_din_base=0x1000 >> 2, addr_din_length=448 - 1,
                                     addr_rhead_base=addr_rhead_base, addr_rhead_length=0, addr_dout_base=0x0000 >> 2,
                                     addr_dout_length=0, soma_in_en=1, data_in=None)
                        soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x47C0 >> 2, addr_ciso=0x0000 >> 2,
                                    length_in=56 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=56 * 64,
                                    num_out=1, type_in=1, type_out=1)
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})
                else:  # 奇数core
                    if core_x == size_x - 1:
                        axon = None
                        soma1 = None
                        router = None
                        soma2 = None
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})
                    else:
                        axon = None
                        # merge最后一行
                        soma1 = p06(addr_in=0xA9C0 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2,
                                    length_in=56 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=56 * 64,
                                    num_out=1, type_in=1, type_out=1)
                        addr_rhead_base = MapConfigGen.get_router_rhead_base(
                            phase_group[(chip, (core_x, core_y))]['prims'])
                        router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=448 - 1, receive_num=0,
                                     addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                                     addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                                     data_in=None)
                        router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0,
                                        EN=1)
                        soma2 = None
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})

            else:  # 后57列
                if core_x % 2 == 0:  # 偶数core
                    if core_x == 0:
                        axon = None
                        soma1 = None
                        router = None
                        soma2 = None
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})
                    else:
                        axon = None
                        soma1 = None
                        addr_rhead_base = MapConfigGen.get_router_rhead_base(
                            phase_group[(chip, (core_x, core_y))]['prims'])
                        router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                                     addr_din_base=0x1000 >> 2, addr_din_length=456 - 1,
                                     addr_rhead_base=addr_rhead_base, addr_rhead_length=0, addr_dout_base=0x0000 >> 2,
                                     addr_dout_length=0, soma_in_en=1, data_in=None)
                        soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x47C0 >> 2, addr_ciso=0x0000 >> 2,
                                    length_in=57 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=57 * 64,
                                    num_out=1, type_in=1, type_out=1)
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})
                else:  # 奇数core
                    if core_x == size_x - 1:
                        axon = None
                        soma1 = None
                        router = None
                        soma2 = None
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})
                    else:
                        axon = None
                        # merge最后一行
                        soma1 = p06(addr_in=0xAB80 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2,
                                    length_in=57 * 64, num_in=1, length_ciso=1, num_ciso=1, length_out=57 * 64,
                                    num_out=1, type_in=1, type_out=1)
                        addr_rhead_base = MapConfigGen.get_router_rhead_base(
                            phase_group[(chip, (core_x, core_y))]['prims'])
                        router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=456 - 1, receive_num=0,
                                     addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                                     addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                                     data_in=None)
                        router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=456 - 1, A_offset=0, Const=0,
                                        EN=1)
                        soma2 = None
                        phase_group[(chip, (core_x, core_y))]['prims'].append(
                            {'axon': axon, 'soma1': soma1, 'router': router,
                             'soma2': soma2})

    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 0:  # 前56列
                if core_x % 2 == 0:  # 偶数core
                    axon = None
                    soma1 = None
                    addr_rhead_base = MapConfigGen.get_router_rhead_base(
                        phase_group[(chip, (core_x, core_y))]['prims'])
                    router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                                 addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                                 addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=1,
                                 data_in=None)
                    soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xB7C0 >> 2, addr_ciso=0x0000 >> 2, length_in=56 * 64,
                                num_in=1, length_ciso=1, num_ciso=1, length_out=56 * 64, num_out=1, type_in=1,
                                type_out=1)
                    phase_group[(chip, (core_x, core_y))]['prims'].append(
                        {'axon': axon, 'soma1': soma1, 'router': router,
                         'soma2': soma2})
                else:  # 奇数core
                    axon = None
                    # merge第一行
                    soma1 = p06(addr_in=0x55C0 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=56 * 64,
                                num_in=1, length_ciso=1, num_ciso=1, length_out=56 * 64, num_out=1, type_in=1,
                                type_out=1)
                    addr_rhead_base = MapConfigGen.get_router_rhead_base(
                        phase_group[(chip, (core_x, core_y))]['prims'])
                    router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=448 - 1, receive_num=0,
                                 addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                                 addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0,
                                 soma_in_en=1, data_in=None)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0,
                                    EN=1)
                    soma2 = None
                    phase_group[(chip, (core_x, core_y))]['prims'].append(
                        {'axon': axon, 'soma1': soma1, 'router': router,
                         'soma2': soma2})

            else:  # 后57列
                if core_x % 2 == 0:  # 偶数core
                    axon = None
                    soma1 = None
                    addr_rhead_base = MapConfigGen.get_router_rhead_base(
                        phase_group[(chip, (core_x, core_y))]['prims'])
                    router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                                 addr_din_base=0x1000 >> 2, addr_din_length=456 - 1, addr_rhead_base=addr_rhead_base,
                                 addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=1,
                                 data_in=None)
                    soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xB9C0 >> 2, addr_ciso=0x0000 >> 2, length_in=57 * 64,
                                num_in=1, length_ciso=1, num_ciso=1, length_out=57 * 64, num_out=1, type_in=1,
                                type_out=1)
                    phase_group[(chip, (core_x, core_y))]['prims'].append(
                        {'axon': axon, 'soma1': soma1, 'router': router,
                         'soma2': soma2})
                else:  # 奇数core
                    axon = None
                    # merge第一行
                    soma1 = p06(addr_in=0x5600 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=57 * 64,
                                num_in=1, length_ciso=1, num_ciso=1, length_out=57 * 64, num_out=1, type_in=1,
                                type_out=1)
                    addr_rhead_base = MapConfigGen.get_router_rhead_base(
                        phase_group[(chip, (core_x, core_y))]['prims'])
                    router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=456 - 1, receive_num=0,
                                 addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                                 addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                                 data_in=None)
                    router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=456 - 1, A_offset=0, Const=0,
                                    EN=1)
                    soma2 = None
                    phase_group[(chip, (core_x, core_y))]['prims'].append(
                        {'axon': axon, 'soma1': soma1, 'router': router,
                         'soma2': soma2})

    # Conv1a2 MaxPool
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            px = 56 if core_y == 0 else 57
            if core_x % 2 == 0:
                if core_x == 0:
                    py = 8
                else:
                    py = 9
            else:
                py = 7
            pad_top = 1 if core_x == 0 else 0
            pad_left = 1 if core_y == 0 else 0
            if core_y == 0:
                if core_x % 2 == 0:
                    if core_x == 0:
                        addr_in = 0x55C0 >> 2
                    else:
                        addr_in = 0x47C0 >> 2
                else:
                    addr_in = 0x55C0 >> 2
            else:
                if core_x % 2 == 0:
                    if core_x == 0:
                        addr_in = 0x5600 >> 2
                    else:
                        addr_in = 0x47C0 >> 2
                else:
                    addr_in = 0x5600 >> 2
            axon = None
            if core_x % 2 == 0:
                addr_out = 0xe3e0 >> 2
            else:
                addr_out = 0xeae0 >> 2
            soma1 = pX5(mode='max', addr_in=addr_in, addr_out=addr_out, cin=64, cout=64, px=px, py=py, kx=3,
                        ky=3, sx=2, sy=2, cmp_c=0x80808080, pad_top=pad_top, pad_down=0, pad_left=pad_left, pad_right=0,
                        type_in=1, type_out=1, in_cut_start=0, row_ck_on=0, in_row_max=0)
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
