import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.prims import p06, p09, pX5, p41, p02
from generator.mapping_utils.map_config_gen import MapConfigGen
from generator.resnet50.resnet50_5chips.G5_data import generate_g5_data
from generator.resnet50.data_handler import ResNetDataHandler
from itertools import product

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def gen_g5_map_config0(phase_en, clock_in_phase, size_x, size_y, cuts, static_data=None, chip=(0, 0),
                       in_data_en=False, out_data_en=False, delay_l4=None, delay_l5=None, init_data=None):
    """
        ResNet-50 5-Chip Group5
        core array : 2 * 14
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

    # 接收 28*256  发送 4*14*128 （每行的前半行）
    if phase_en[0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x9900 >> 2,
                       addr_inb=0x9900 >> 2, addr_bias=0x9900 >> 2, addr_out=0x9900 >> 2, axon_delay=True,
                       L4_num=delay_l4[0], L5_num=delay_l5[0], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x6100 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=14 * 128, num_out=4, type_in=1, type_out=1,
                            data_in=static_data['g5_add_output'][(core_x, core_y)] if init_data else None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=895, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None, cxy=0, nx=0, ny=0, relay_num=895)
            if in_data_en:
                if core_x // 7 == 1:
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                elif core_x // 7 == 0:
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                else:
                    raise ValueError
            # if core_y == 0:
            #     router.CXY, router.Receive_en = 0, 1
            # elif core_y == 1:
            #     router.CXY, router.Receive_en = 0, 0
            # else:
            #     raise ValueError
            if out_data_en:
                dst_id1 = (core_x % 7) * 4
                dst_x1, dst_y1 = dst_id1 % 14, dst_id1 // 14 + 3
                dst_x2, dst_y2 = (dst_id1 + 1) % 14, (dst_id1 + 1) // 14 + 3
                dst_x3, dst_y3 = (dst_id1 + 2) % 14, (dst_id1 + 2) // 14 + 3
                dst_x4, dst_y4 = (dst_id1 + 3) % 14, (dst_id1 + 3) // 14 + 3
                a = (core_x // 7 + core_y * 2) * 16
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x1 - core_x, Y=dst_y1 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x2 - core_x, Y=dst_y2 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x3 - core_x, Y=dst_y3 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x4 - core_x, Y=dst_y4 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x19000 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 接收 28*256  发送 4*14*128 （每行的后半行）
    if phase_en[1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x9900 >> 2,
                       addr_inb=0x9900 >> 2, addr_bias=0x9900 >> 2, addr_out=0x9900 >> 2, axon_delay=True,
                       L4_num=delay_l4[1], L5_num=delay_l5[1], A2S2_mode=True)
            # delay clock = 6 + 37 * (L4 + 1) * (L5 + 1)
            if (core_x, core_y) != (0, 0):
                axon = None
            if out_data_en:
                soma1 = p06(addr_in=0x6800 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 128,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=14 * 128, num_out=4, type_in=1, type_out=1,
                            data_in=static_data['g5_add_output1'][(core_x, core_y)] if init_data else None)
            else:
                soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=out_data_en, receive_en=in_data_en, send_num=895, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=1, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None, cxy=0, nx=0, ny=0, relay_num=895)
            if in_data_en:
                if core_x // 7 == 1:
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                elif core_x // 7 == 0:
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                else:
                    raise ValueError
            if out_data_en:
                dst_id1 = (core_x % 7) * 4
                dst_x1, dst_y1 = dst_id1 % 14, dst_id1 // 14 + 3
                dst_x2, dst_y2 = (dst_id1 + 1) % 14, (dst_id1 + 1) // 14 + 3
                dst_x3, dst_y3 = (dst_id1 + 2) % 14, (dst_id1 + 2) // 14 + 3
                dst_x4, dst_y4 = (dst_id1 + 3) % 14, (dst_id1 + 3) // 14 + 3
                a = (core_x // 7 + core_y * 2) * 16
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x1 - core_x, Y=dst_y1 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x2 - core_x, Y=dst_y2 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x3 - core_x, Y=dst_y3 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
                router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x4 - core_x, Y=dst_y4 - core_y, A=a,
                                pack_per_Rhead=223, A_offset=48, Const=15, EN=1)
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x1ac00 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # 接收 28*256
    if phase_en[2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p41(px=1, py=1, load_bias=0, cin=1, cout=1, kx=1, ky=1, sx=1, sy=1, addr_ina=0x9900 >> 2,
                       addr_inb=0x9900 >> 2, addr_bias=0x9900 >> 2, addr_out=0x9900 >> 2, axon_delay=True,
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
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                elif core_x // 7 == 0:
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                else:
                    raise ValueError
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x1c800 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})
    # # 接收 28*256
    if phase_en[3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = None
            soma1 = None
            addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
            router = p09(rhead_mode=1, send_en=0, receive_en=in_data_en, send_num=895, receive_num=3,
                         addr_din_base=0x1000 >> 2, addr_din_length=895, addr_rhead_base=addr_rhead_base,
                         addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                         data_in=None, cxy=0, nx=0, ny=0, relay_num=895)
            if in_data_en:
                if core_x // 7 == 1:
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                elif core_x // 7 == 0:
                    if core_y == 0:
                        router.CXY, router.Nx, router.Ny = 0, 0, 0
                    elif core_y == 1:
                        router.CXY, router.Nx, router.Ny = 1, 0, -1
                    else:
                        raise ValueError
                else:
                    raise ValueError
            if in_data_en:
                soma2 = p06(addr_in=0x8400, addr_out=0x1e400 >> 2, addr_ciso=0x0000 >> 2, length_in=256,
                            num_in=28, length_ciso=1, num_ciso=28, length_out=256, num_out=28,
                            type_in=1, type_out=1)
            else:
                soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # ************************ 计算部分 **************************

    # Conv3a1 + Relu 
    if phase_en[offset + 0]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if static_data is None:
                data_x = []
                data_w = []
                data_b = []
            else:
                data_x = None if in_data_en else static_data['conv3a1_input'][(core_x, core_y)]
                data_w = static_data['conv3a1_weight'][(core_x, core_y)]
                data_b = static_data['conv3a1_bias'][(core_x, core_y)]
            axon = p41(px=28, py=4, cin=256, cout=32, kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2,
                       addr_inb=0x0000 >> 2, addr_bias=0x2000 >> 2, addr_out=0x6100 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=data_x, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x6100 >> 2, addr_out=0x7D00 >> 2, cin=32, cout=32, px=28, py=4, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.0.cut1'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 通道方向整理
    if phase_en[offset + 1]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x // 7) % 2 == 0:
                axon = None
                soma1 = p06(addr_in=0x7D00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 32,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 32, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x7D00 >> 2, addr_out=0x8B00 >> 2, addr_ciso=0x21000 >> 2, length_in=32,
                            num_in=4 * 28, length_ciso=32, num_ciso=4 * 28, length_out=64, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x7D00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 32,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 32, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x8B00 >> 2, addr_ciso=0x7D00 >> 2, length_in=32,
                            num_in=4 * 28, length_ciso=32, num_ciso=4 * 28, length_out=64, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    if phase_en[offset + 2]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 0:
                axon = None
                soma1 = p06(addr_in=0x8B00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 64,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 64, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=896 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x8B00 >> 2, addr_out=0xA700 >> 2, addr_ciso=0x21000 >> 2, length_in=64,
                            num_in=4 * 28, length_ciso=64, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x8B00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 64,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 64, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=896 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xA700 >> 2, addr_ciso=0x8B00 >> 2, length_in=64,
                            num_in=4 * 28, length_ciso=64, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # 发送交叠数据
    if phase_en[offset + 3]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x % 7 == 0:
                axon = None
                soma1 = None
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=0,
                             data_in=None)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xDF00 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            elif (core_x + 1) % 7 == 0:
                axon = None
                soma1 = p06(addr_in=0xA700 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0xA700 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-1, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0xDF00 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            # 注意最大长度限制
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    if phase_en[offset + 4]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_x % 7 == 0:
                axon = None
                soma1 = p06(addr_in=0xD100 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x0000 >> 2, addr_din_length=0, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = None
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            elif (core_x + 1) % 7 == 0:
                axon = None
                soma1 = None
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x0000 >> 2, addr_dout_length=0, soma_in_en=0,
                             data_in=None)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x9900 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0xD100 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1,
                            type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=1, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x9900 >> 2, addr_ciso=0x0000 >> 2, length_in=128 * 28,
                            # 注意最大长度限制
                            num_in=1, length_ciso=1, num_ciso=1, length_out=128 * 28, num_out=1,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # Conv3a2 + Relu
    if phase_en[offset + 5]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            py = 5 if (core_x % 7 == 0 or (core_x + 1) % 7 == 0) else 6
            addr_ina = 0xA700 >> 2 if core_x % 7 == 0 else 0x9900 >> 2
            pad_top = 1 if core_x % 7 == 0 else 0
            pad_down = 1 if (core_x + 1) % 7 == 0 else 0
            data_w = static_data['conv3a2_weight'][(core_x, core_y)]
            data_b = static_data['conv3a2_bias'][(core_x, core_y)]
            axon = p41(px=28, py=py, cin=128, cout=32, kx=3, ky=3, sx=1, sy=1, addr_ina=addr_ina,
                       addr_inb=0x10000 >> 2, addr_bias=0x6080 >> 2, addr_out=0x6100 >> 2, ina_type=1, inb_type=1,
                       pad_top=pad_top, pad_down=pad_down, pad_left=1, pad_right=1,
                       load_bias=2, data_x=None, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x6100 >> 2, addr_out=0x7D00 >> 2, cin=32, cout=32, px=28, py=4, kx=1,
                        ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.0.cut2'],
                        row_ck_on=1, in_row_max=2)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # 通道方向整理
    if phase_en[offset + 6]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if (core_x // 7) % 2 == 0:
                axon = None
                soma1 = p06(addr_in=0x7D00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 32,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 32, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=7, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x7D00 >> 2, addr_out=0x8B00 >> 2, addr_ciso=0x21000 >> 2, length_in=32,
                            num_in=4 * 28, length_ciso=32, num_ciso=4 * 28, length_out=64, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x7D00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 32,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 32, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'])
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=448 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=448 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=-7, Y=0, A=0, pack_per_Rhead=448 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x8B00 >> 2, addr_ciso=0x7D00 >> 2, length_in=32,
                            num_in=4 * 28, length_ciso=32, num_ciso=4 * 28, length_out=64, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    if phase_en[offset + 7]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            if core_y == 0:
                axon = None
                soma1 = p06(addr_in=0x8B00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 64,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 64, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                     limit=0x34c)
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=1, A=0, pack_per_Rhead=896 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x8B00 >> 2, addr_out=0x19000 >> 2, addr_ciso=0x21000 >> 2, length_in=64,
                            num_in=4 * 28, length_ciso=64, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})
            else:
                axon = None
                soma1 = p06(addr_in=0x8B00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000 >> 2, length_in=28 * 64,
                            num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 64, num_out=4, type_in=1, type_out=1)
                addr_rhead_base = MapConfigGen.get_router_rhead_base(phase_group[(chip, (core_x, core_y))]['prims'],
                                                                     limit=0x34c)
                router = p09(rhead_mode=1, send_en=1, receive_en=1, send_num=896 - 1, receive_num=0,
                             addr_din_base=0x1000 >> 2, addr_din_length=896 - 1, addr_rhead_base=addr_rhead_base,
                             addr_rhead_length=0, addr_dout_base=0x4000 >> 2, addr_dout_length=0, soma_in_en=1,
                             data_in=None)
                router.addRHead(S=0, T=1, P=0, Q=0, X=0, Y=-1, A=0, pack_per_Rhead=896 - 1, A_offset=0, Const=0, EN=1)
                soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x19000 >> 2, addr_ciso=0x8B00 >> 2, length_in=64,
                            num_in=4 * 28, length_ciso=64, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
                            type_in=1, type_out=1)
                phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                       'soma2': soma2})

    # Conv3a3 + 截取
    if phase_en[offset + 8]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            data_w = static_data['conv3a3_weight'][(core_x, core_y)]
            data_b = static_data['conv3a3_bias'][(core_x, core_y)]
            axon = p41(px=28, py=4, cin=128, cout=128, kx=1, ky=1, sx=1, sy=1, addr_ina=0x19000 >> 2,
                       addr_inb=0x2080 >> 2, addr_bias=0xFE00 >> 2, addr_out=0x6100 >> 2, ina_type=1, inb_type=1,
                       load_bias=2, data_x=None, data_w=data_w, data_b=data_b)
            soma1 = pX5(mode='max', addr_in=0x6100 >> 2, addr_out=0x9900 >> 2, cin=128, cout=128, px=28, py=4,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=0, type_out=1,
                        in_cut_start=cuts['layer2.0.cut3'], row_ck_on=1, in_row_max=1)
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    # Add
    # if phase_en[offset + 9]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p06(addr_in=0x6100 >> 2, addr_out=0x19000 >> 2, addr_ciso=0x0000 >> 2, length_in=128,
    #                     num_in=4 * 28, length_ciso=1, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
    #                     type_in=1, type_out=1)
    #         router = None
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})

    # if phase_en[offset + 10]:
    #     for core_y, core_x in product(range(size_y), range(size_x)):
    #         axon = None
    #         soma1 = p06(addr_in=0x9900 >> 2, addr_out=0x19000 >> 2, addr_ciso=0x0000 >> 2, length_in=128,
    #                     num_in=4 * 28, length_ciso=1, num_ciso=4 * 28, length_out=128, num_out=4 * 28,
    #                     type_in=1, type_out=1)
    #         router = None
    #         soma2 = None
    #         phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
    #                                                                'soma2': soma2})
    
    if phase_en[offset + 9]:
        for core_y, core_x in product(range(size_y), range(size_x)):
            axon = p02(avg_pooling_en=False, ina_type=1, load_bias=0, kx=1, ky=2, sx=1, sy=1, cin=128, px=28, py=4,
                       addr_in=0x6100 >> 2, addr_bias=0x0000 >> 2, addr_out=0x19000 >> 2, data_x=None)
            # Axon计算完的结果均为INT32
            soma1 = pX5(mode='max', addr_in=0x19000 >> 2, addr_out=0x6100 >> 2, cin=128, cout=128, px=28, py=4,
                        kx=1, ky=1, sx=1, sy=1, cmp_c=0, type_in=0, type_out=1, in_cut_start=cuts['layer2.0.cut5'],
                        row_ck_on=1, in_row_max=1) 
            router = None
            soma2 = None
            phase_group[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                   'soma2': soma2})

    return map_config


if __name__ == '__main__':
    case_file_name = 'R00005_0'

    phase = np.zeros(50).astype(int)  # 39-49表示组件数据传输的Phase
    cuts = Resnet50Cuts()

    # Conv3a1
    phase[0] = 1
    # Conv3a2数据整理
    phase[1] = 1
    phase[2] = 1
    phase[3] = 1
    # Conv3a2
    phase[4] = 1
    # Conv3a3数据整理
    phase[5] = 1
    phase[6] = 1
    # Conv3a3
    phase[7] = 1
    # Add
    phase[8] = 1

    handler = ResNetDataHandler()
    static_data = generate_g5_data(handler, size_y=2, size_x=14)
    config = gen_g5_map_config0(phase, clock_in_phase=200_000, size_x=14, size_y=2, cuts=cuts,
                                static_data=static_data, in_data_en=False, out_data_en=False)
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
