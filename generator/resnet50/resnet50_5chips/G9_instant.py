import sys
import os

sys.path.append(os.getcwd())

from generator.resnet50.resnet50_5chips.prims import p09, pX5
from itertools import product


def gen_g9_instant_map_config(clock_in_phase, chip=(0, 0), rhead_base=(0x350, 0x350),
                              start_instant_pi_num=0, receive_pi_addr_base=0x360, data=[]):
    """
        ResNet-50 5-Chip Group-9-instant
        core array : 4 * 8 + 2 * 8
        只包含整个48cores阵列的即时原语部分
    """
    map_config = {
        'sim_clock': None,
        (chip, 0): {
            0: {  # 4 * 8 阵列
                'clock': clock_in_phase,
                'mode': 1,
            },
            1: {  # 2 * 8 阵列
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for core_y, core_x in product(range(2, 6), range(8)):
        map_config[(chip, 0)][0][(chip, (core_x, core_y))] = {
            'prims': []
        }
    for core_y, core_x in product(range(2), range(8)):
        map_config[(chip, 0)][1][(chip, (core_x, core_y))] = {
            'prims': []
        }
    phase_group_0 = map_config[(chip, 0)][0]
    phase_group_1 = map_config[(chip, 0)][1]

    # 接收shortcut的数据 （4*8阵列）
    for core_y, core_x in product(range(2, 6), range(8)):
        axon, soma1 = None, None
        src_x = core_x
        src_y = 1 if core_y in [4, 5] else 0
        router = p09(rhead_mode=1, send_en=0, receive_en=1, send_num=0, receive_num=0, addr_din_base=0x380,
                     addr_din_length=783, addr_rhead_base=rhead_base[0], addr_rhead_length=0,
                     addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=0, data_in=None,
                     send_pi_en=0, back_sign_en=0, send_pi_num=0, receive_sign_num=0,
                     send_pi_addr_base=0, Q=0, receive_sign_en=1)
        router.send_destin_core_grp = []
        router.recv_source_core_grp = [
            {"core_id": (chip, (src_x, src_y)), "data_num": 784, "T_mode": 1, "Rhead_num": 1}]
        router.instant_request_back = []
        if (core_x, core_y) == (0, 2):
            router.Send_PI_en = 1
            router.Send_PI_addr_base = rhead_base[0] >> 2
            router.instant_prim_request = [((chip, (0, 1)), 0)]
            router.add_instant_pi(
                PI_addr_offset=0, A_valid=0, S1_valid=1, R_valid=1, S2_valid=0, X=0, Y=-1, Q=1)
        soma2 = pX5(mode='max', addr_in=0x8380, addr_out=0xab80 >> 2, cin=128, cout=128, px=7, py=7,
                    kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                    row_ck_on=0, in_row_max=2, data_in=None)
        phase_group_0[(chip, (core_x, core_y))]['prims'].append({'axon': axon, 'soma1': soma1, 'router': router,
                                                                 'soma2': soma2})
        cxy = 1
        if core_y in [2, 4]:
            if core_x == 7:
                nx, ny = 0, 1
            else:
                nx, ny = 1, 0
        else:
            if core_x == 0:
                nx, ny = 0, 1
            else:
                nx, ny = -1, 0
        if (core_x, core_y) == (0, 5):
            cxy, nx, ny = 0, 0, 0
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

    # 发送shortcut的数据 （2*8阵列）
    for core_y, core_x in product(range(2), range(8)):
        axon = None
        soma1 = pX5(mode='max', addr_in=0x16200 >> 2, addr_out=0x9000, cin=128, cout=128, px=14, py=7,
                    kx=1, ky=1, sx=1, sy=1, cmp_c=0x80808080, type_in=1, type_out=1, in_cut_start=0,
                    row_ck_on=0, in_row_max=2, data_in=data)
        router = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=1567, receive_num=0, addr_din_base=0x380,
                     addr_din_length=0, addr_rhead_base=rhead_base[1], addr_rhead_length=0,
                     addr_dout_base=0x1000, addr_dout_length=0, soma_in_en=1, data_in=None,
                     send_pi_en=0, back_sign_en=0, send_pi_num=0, receive_sign_num=0,
                     send_pi_addr_base=0, Q=0, receive_sign_en=0)
        if core_y == 0:
            dst_x = core_x
            dst_y_1, dst_y_2 = 2, 3
        else:
            dst_x = core_x
            dst_y_1, dst_y_2 = 4, 5
        router.send_destin_core_grp = [
            {"core_id": (chip, (dst_x, dst_y_1)), "data_num": 784, "T_mode": 1, "Rhead_num": 1},
            {"core_id": (chip, (dst_x, dst_y_2)), "data_num": 784, "T_mode": 1, "Rhead_num": 1}]
        router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y_1 - core_y, A=0, pack_per_Rhead=783, A_offset=0,
                        Const=0, EN=1)
        router.addRHead(S=0, T=1, P=0, Q=0, X=dst_x - core_x, Y=dst_y_2 - core_y, A=0, pack_per_Rhead=783, A_offset=0,
                        Const=0, EN=1)
        router.recv_source_core_grp = []
        router.instant_prim_request = []
        if (core_x, core_y) == (0, 0):
            router.Back_sign_en = 1
            router.instant_request_back = [(chip, (j, i)) for i, j in product(range(2, 6), range(8))]
            router.Q = 0
            cxy = 0
        else:
            cxy = 1
        if core_y == 1:
            if core_x == 7:
                nx, ny = 0, -1
            else:
                nx, ny = 1, 0
        else:
            if core_x == 0:
                nx, ny = 0, 0
            else:
                nx, ny = -1, 0
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
            "instant_PI_number": 0,
            "PI_loop_en": 0,
            "start_instant_PI_num": start_instant_pi_num
        }

    return map_config


if __name__ == '__main__':
    import os

    case_file_name = 'R000093'

    clock_in_phase = 50_000
    config = gen_g9_instant_map_config(clock_in_phase, chip=(0, 0), rhead_base=(0x350, 0x350),
                                       start_instant_pi_num=0, receive_pi_addr_base=0x360)

    config['sim_clock'] = clock_in_phase

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
        'test_group_phase': [(0, 1), (1, 1)]
    }

    tester = TestEngine(config, test_config)
    assert tester.run_test()
