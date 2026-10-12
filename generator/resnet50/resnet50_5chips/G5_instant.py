import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p06, p09
from itertools import product


def gen_g5_instant_map_config(clock_in_phase, chip=(0, 0), rhead_base=(0x350, 0x350),
                              start_instant_pi_num=0, receive_pi_addr_base=0x360):
    """
        ResNet-50 5-Chip Group-5-instant
        core array : 2 * 14 + 1 * 14
        只包含整个42cores阵列的即时原语部分
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {  # 2 * 14 阵列
                'clock': clock_in_phase,
                'mode': 1,
            },
            1: {  # 1 * 14 阵列
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(0, 2), range(14)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    for core_y, core_x in product(range(2, 3), range(14)):
        map_config[(chip, 0)][1][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group_0 = map_config[(chip, 0)][0]
    phase_group_1 = map_config[(chip, 0)][1]

    # 接收shortcut的数据  2 * 14阵列
    # 第一次即时原语请求
    for core_y, core_x in product(range(0, 2), range(14)):
        axon, soma1 = None, None
        src_y = 2
        if 0 <= core_x < 7 and core_y == 0:
            src_x = core_x
        elif 7 <= core_x < 14 and core_y == 0:
            src_x = core_x - 7
        elif 0 <= core_x < 7 and core_y == 1:
            src_x = core_x - 7
        else:
            src_x = core_x
        router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0, addr_din_base=0x1000 >> 2,
                     addr_din_length=896 - 1, addr_rhead_base=0x300, addr_rhead_length=0,  # 不发送普通数据包
                     addr_dout_base=0x0000, addr_dout_length=0, soma_in_en=0, data_in=None,
                     send_pi_en=0, back_sign_en=0, send_pi_num=0, receive_sign_num=0,
                     send_pi_addr_base=0, Q=0, receive_sign_en=1)
        router.send_destin_core_grp = []
        router.recv_source_core_grp = [
            {"core_id": (chip, (src_x, src_y)), "data_num": 896, "T_mode": 1, "Rhead_num": 1}]
        router.instant_request_back = []
        if (core_x, core_y) == (0, 1):
            router.Send_PI_en = 1
            router.Send_PI_addr_base = rhead_base[0] >> 2  # 16B寻址
            router.instant_prim_request = [((chip, (0, 2)), 0)]
            router.add_instant_pi(
                PI_addr_offset=0, A_valid=0, S1_valid=1, R_valid=1, S2_valid=0, X=0, Y=1, Q=1)
        soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x6100 >> 2, addr_ciso=0x0000 >> 2, length_in=128,
                    num_in=2 * 28, length_ciso=1, num_ciso=2 * 28, length_out=128, num_out=2 * 28,  # 只发送一半
                    type_in=1, type_out=1)
        phase_group_0[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                 'soma2': soma2})
        cxy = 1
        if core_y == 1:
            if core_x == 13:
                nx, ny = 0, -1
            else:
                nx, ny = 1, 0
        else:
            if core_x == 0:
                cxy, nx, ny = 0, 0, 0
            else:
                nx, ny = -1, 0
        phase_group_0[(chip, (core_x, core_y))]['registers'] = {
            "Receive_PI_addr_base": 0,
            "PI_CXY": 0,
            "PI_Nx": 0,
            "PI_Ny": 0,
            "PI_sign_CXY": cxy,
            "PI_sign_Nx": nx,
            "PI_sign_Ny": ny,
            "instant_PI_en": 0,
            "fixed_instant_PI": 0,
            "instant_PI_number": 0,
            "PI_loop_en": 0,
            "start_instant_PI_num": 0,
            "Addr_instant_PI_base": 0
        }
    # 第二次即时原语请求
    # 不需要再赋Core参数
    for core_y, core_x in product(range(0, 2), range(14)):
        axon, soma1 = None, None
        src_y = 2
        if 0 <= core_x < 7 and core_y == 0:
            src_x = core_x
        elif 7 <= core_x < 14 and core_y == 0:
            src_x = core_x - 7
        elif 0 <= core_x < 7 and core_y == 1:
            src_x = core_x - 7
        else:
            src_x = core_x
        router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0, addr_din_base=0x1000 >> 2,
                     addr_din_length=896 - 1, addr_rhead_base=0x300, addr_rhead_length=0,  # 不发送普通数据包
                     addr_dout_base=0x0000, addr_dout_length=0, soma_in_en=0, data_in=None,
                     send_pi_en=0, back_sign_en=0, send_pi_num=0, receive_sign_num=0,
                     send_pi_addr_base=0, Q=0, receive_sign_en=1)
        router.send_destin_core_grp = []
        router.recv_source_core_grp = [
            {"core_id": (chip, (src_x, src_y)), "data_num": 896, "T_mode": 1, "Rhead_num": 1}]
        router.instant_request_back = []
        if (core_x, core_y) == (0, 1):
            router.Send_PI_en = 1
            router.Send_PI_addr_base = (rhead_base[0] + 4) >> 2  # 16B寻址
            router.instant_prim_request = [((chip, (0, 2)), 0)]
            router.add_instant_pi(
                PI_addr_offset=2, A_valid=0, S1_valid=1, R_valid=1, S2_valid=0, X=0, Y=1, Q=1)
        soma2 = p06(addr_in=0x21000 >> 2, addr_out=0x7D00 >> 2, addr_ciso=0x0000 >> 2, length_in=128,
                    num_in=2 * 28, length_ciso=1, num_ciso=2 * 28, length_out=128, num_out=2 * 28,  # 只发送一半
                    type_in=1, type_out=1)
        phase_group_0[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                 'soma2': soma2})

    # 发送shortcut的数据  1 * 14阵列 *********************************************************************
    # 第一次即时原语
    for core_y, core_x in product(range(2, 3), range(14)):
        axon = None
        soma1 = p06(addr_in=0x17400 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000, length_in=28 * 128,
                    num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 128, num_out=4,
                    type_in=1, type_out=1, data_in=None)
        router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=1792 - 1, receive_num=0,
                     addr_din_base=0x0000, addr_din_length=0, addr_rhead_base=rhead_base[1], addr_rhead_length=0,
                     addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None,
                     send_pi_en=0, back_sign_en=0, send_pi_num=0, receive_sign_num=0,
                     send_pi_addr_base=0, Q=0, receive_sign_en=0)
        if 0 <= core_x < 7:
            dst_x0, dst_x1 = core_x, core_x + 7
            dst_y = 0
        else:
            dst_x0, dst_x1 = core_x - 7, core_x
            dst_y = 1
        router.send_destin_core_grp = [
            {"core_id": (chip, (dst_x0, dst_y)), "data_num": 896, "T_mode": 1, "Rhead_num": 1},
            {"core_id": (chip, (dst_x1, dst_y)), "data_num": 896, "T_mode": 1, "Rhead_num": 1}]
        router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x0 - core_x, Y=dst_y - core_y, A=0, pack_per_Rhead=896 - 1,
                        A_offset=0, Const=0, EN=1)
        router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x1 - core_x, Y=dst_y - core_y, A=0, pack_per_Rhead=896 - 1,
                        A_offset=0, Const=0, EN=1)
        router.recv_source_core_grp = []
        router.instant_prim_request = []
        if (core_x, core_y) == (13, 2):
            router.Back_sign_en = 1
            router.instant_request_back = [(chip, (i, j)) for i, j in product(range(14), range(2))]
            router.Q = 0
            cxy, nx, ny = 0, 0, 0
        else:
            cxy, nx, ny = 1, 1, 0
        soma2 = None
        phase_group_1[(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None,
                                                                 'soma2': None})
        phase_group_1[(chip, (core_x, core_y))]['instant_prims'] = [{'axon': axon, 'soma1': soma1, 'router': router,
                                                                     'soma2': soma2}]
        phase_group_1[(chip, (core_x, core_y))]['registers'] = {
            "Receive_PI_addr_base": receive_pi_addr_base >> 2,
            "PI_CXY": cxy,
            "PI_Nx": nx,
            "PI_Ny": ny,
            "PI_sign_CXY": 0,
            "PI_sign_Nx": 0,
            "PI_sign_Ny": 0,
            "instant_PI_en": 1,
            "fixed_instant_PI": 1,
            "instant_PI_number": 1,
            "PI_loop_en": 0,
            "start_instant_PI_num": start_instant_pi_num
        }

    # 第二次即时原语
    for core_y, core_x in product(range(2, 3), range(14)):
        axon = None
        soma1 = p06(addr_in=0x1AC00 >> 2, addr_out=0x24000 >> 2, addr_ciso=0x0000, length_in=28 * 128,
                    num_in=4, length_ciso=1, num_ciso=4, length_out=28 * 128, num_out=4,
                    type_in=1, type_out=1, data_in=None)
        router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=1792 - 1, receive_num=0,
                     addr_din_base=0x0000, addr_din_length=0, addr_rhead_base=(rhead_base[1] + 4), addr_rhead_length=0,
                     addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None,
                     send_pi_en=0, back_sign_en=0, send_pi_num=0, receive_sign_num=0,
                     send_pi_addr_base=0, Q=0, receive_sign_en=0)
        if 0 <= core_x < 7:
            dst_x0, dst_x1 = core_x, core_x + 7
            dst_y = 0
        else:
            dst_x0, dst_x1 = core_x - 7, core_x
            dst_y = 1
        router.send_destin_core_grp = [
            {"core_id": (chip, (dst_x0, dst_y)), "data_num": 896, "T_mode": 1, "Rhead_num": 1},
            {"core_id": (chip, (dst_x1, dst_y)), "data_num": 896, "T_mode": 1, "Rhead_num": 1}]
        router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x0 - core_x, Y=dst_y - core_y, A=0, pack_per_Rhead=896 - 1,
                        A_offset=0, Const=0, EN=1)
        router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x1 - core_x, Y=dst_y - core_y, A=0, pack_per_Rhead=896 - 1,
                        A_offset=0, Const=0, EN=1)
        router.recv_source_core_grp = []
        router.instant_prim_request = []
        if (core_x, core_y) == (13, 2):
            router.Back_sign_en = 1
            router.instant_request_back = [(chip, (i, j)) for i, j in product(range(14), range(2))]
            router.Q = 0
        soma2 = None
        phase_group_1[(chip, (core_x, core_y))]['prims'].append({'axon': None, 'soma1': None, 'router': None,
                                                                 'soma2': None})
        phase_group_1[(chip, (core_x, core_y))]['instant_prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                         'soma2': soma2})

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R00005_instant'

    clock_in_phase = 50_000
    config = gen_g5_instant_map_config(clock_in_phase, chip=(0, 0), rhead_base=(0x350, 0x350),
                                       start_instant_pi_num=0, receive_pi_addr_base=0x360)

    config['sim_clock'] = clock_in_phase

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
        'test_group_phase': [(0, 1), (1, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
