# coding: utf-8

import os
import numpy as np
from primitive import Prim_09_Router
from generator.test_engine import TestMode, TestEngine
from generator.test.MC.MCprim import *


from primitive import Prim_02_Axon
from primitive import Prim_03_Axon
from primitive import Prim_04_Axon
from primitive import Prim_X5_Soma
from primitive import Prim_06_move_merge
from primitive import Prim_06_move_split
from primitive import Prim_07_LUT
from primitive import Prim_08_lif
from primitive import Prim_09_Router
from primitive import Prim_41_Axon
from primitive import Prim_43_Axon
from primitive import Prim_81_Axon
from primitive import Prim_83_Axon


# chip(0,0)core(0,0)路由发chip(0,0)core(1,1)，测试 同chip 跨斜对角core router
# 变量名没有对应修改
# 可以传过去，没有问题

def test_case():
    tb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]
    np.random.seed(0x9028)


    chip_0_0_soma1 = Prim_06_move_merge()
    chip_0_0_soma1.length_in = 256
    chip_0_0_soma1.length_out = 256
    chip_0_0_soma1.length_ciso = 0  # length可以随便设

    chip_0_0_soma1.num_in = 16
    chip_0_0_soma1.num_out = 16
    chip_0_0_soma1.num_ciso = 16  # 必须相等

    chip_0_0_soma1.type_in = 1
    chip_0_0_soma1.type_out = 1
    chip_0_0_soma1.in_cut_start = 0

    chip_0_0_soma1.Reset_Addr_in = 1
    chip_0_0_soma1.Reset_Addr_out = 1
    chip_0_0_soma1.Reset_Addr_ciso = 1

    chip_0_0_soma1.Row_ck_on = 0  # 不予Axon交互
    chip_0_0_soma1.Addr_Start_in = 0x4000
    chip_0_0_soma1.Addr_Start_out = 0x8400  # 输出到mem3先    out被拆出来
    chip_0_0_soma1.Addr_Start_ciso = 0x0000
    chip_0_0_soma1.in_row_max = 0
    prim_in = chip_0_0_soma1.init_data()
    chip_0_0_soma1.memory_blocks = [
        {'name': 'Input',  # x
            'start': chip_0_0_soma1.Addr_Start_in,
            'data': prim_in[0],
            'mode': 0,
            'initialize': True}, ]

    chip_0_0_router = Prim_09_Router()
    chip_0_0_router.Rhead_mode = 1
    chip_0_0_router.CXY = 0
    chip_0_0_router.Send_en = 1
    chip_0_0_router.Receive_en = 0
    chip_0_0_router.Dout_Mem_sel = 0

    chip_0_0_router.Addr_Dout_base = 0x400  # 4B寻址   i=0上半部分图发自己第29行给后半图的第一行 A80+400
    chip_0_0_router.Addr_Dout_length = 255  # 16B的个数，需要减1  28*4
    chip_0_0_router.Addr_Rhead_base = 0x300  # 4B寻址 20C00D
    chip_0_0_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
    chip_0_0_router.Addr_Din_base = 0x800  # 4B寻址接收地址最后一行的对于下一个core的接收地址起始地址  本core
    chip_0_0_router.Addr_Din_length = 511  # 8B寻址   6*64/8  ？？？？？30*6*64/8-1
    chip_0_0_router.Receive_number = 0
    chip_0_0_router.Send_number = 511  # 发送的所有包数，1个包 8B的倍数

    chip_0_0_router.Nx = 0
    chip_0_0_router.Ny = 0
    chip_0_0_router.Send_PI_en = 0
    chip_0_0_router.Back_sign_en = 0
    chip_0_0_router.Send_PI_num = 0
    chip_0_0_router.Receive_sign_num = 0
    chip_0_0_router.Send_PI_addr_base = 0
    chip_0_0_router.Relay_number = 0
    chip_0_0_router.Q = 0
    chip_0_0_router.T_mode = 1
    chip_0_0_router.Receive_sign_en = 0
    chip_0_0_router.Soma_in_en = 1       #只有在S1 的情况下才启用此参数
    chip_0_0_router.addRHead(S=0, T=1, P=0, Q=0, X=16, Y=0, A=0, pack_per_Rhead=511, A_offset=0, Const=1, EN=1) #28*6*64/8
    chip_0_0_router.send_destin_core_grp=[{'core_id': [((1, 0),(0, 0))], 'data_num': 512, 'T_mode': 1, 'Rhead_num': 1,'sync_en': 1, 'sync_phase_num': 0}]




    chip_1_0_router = Prim_09_Router()
    chip_1_0_router.Rhead_mode = 1
    chip_1_0_router.Send_en = 0
    chip_1_0_router.Receive_en = 1
    chip_1_0_router.Dout_Mem_sel = 0

    chip_1_0_router.Addr_Dout_base = 0x400  # 4B寻址   i=0上半部分图发自己第29行给后半图的第一行 A80+400
    chip_1_0_router.Addr_Dout_length = 0  # 只接收不发送
    chip_1_0_router.Addr_Rhead_base = 0x300  # 4B寻址 20C00D
    chip_1_0_router.Addr_Rhead_length = 0  # 16B的个数，需要减1
    chip_1_0_router.Addr_Din_base = 0x400  # 4B寻址接收地址最后一行的对于下一个core的接收地址起始地址  本core
    chip_1_0_router.Addr_Din_length = 511  # 8B寻址   6*64/8  ？？？？？30*6*64/8-1
    chip_1_0_router.Receive_number = 0
    chip_1_0_router.Send_number = 0  # 发送的所有包数，1个包 8B的倍数

    chip_1_0_router.CXY = 0
    chip_1_0_router.Nx = 0
    chip_1_0_router.Ny = 0
    chip_1_0_router.Send_PI_en = 0
    chip_1_0_router.Back_sign_en = 0
    chip_1_0_router.Send_PI_num = 0
    chip_1_0_router.Receive_sign_num = 0
    chip_1_0_router.Send_PI_addr_base = 0
    chip_1_0_router.Relay_number = 511
    chip_1_0_router.Q = 0
    chip_1_0_router.T_mode = 1
    chip_1_0_router.Receive_sign_en = 0
    chip_1_0_router.Soma_in_en = 0
    chip_1_0_router.recv_source_core_grp = [
        {'core_id': [((0, 0), (0, 0))], 'data_num': 512, 'T_mode': 1, 'Rhead_num': 1, 'sync_en': 1, 'sync_phase_num': 0}]




    map_config = {
        'sim_clock': 10000,
        'step_clock':{
            ((0, 0), 0): (10000, 10000),                # ((chip_x,chip_y),step_group_id):(clock0_in_step,clock1_in_step)
            ((1, 0), 0): (12000, 12000),
        },
        'chip_register':{                               # # #
            (0, 0): {                                   # chip_id
                'PCK_OUT_SEL':[0, 0, 1, 0],
                # 'GFINISH_MODE':[1, 1, 0, 0]
            },
            (1, 0): {
                'PCK_OUT_SEL':[1, 0, 0, 0],
                'GFINISH_MODE':[0, 0, 0, 0]
            }
        },
		'print_mem':{
			((0, 0), (0, 0), (0, 0)): (33792, 256),     # ((chip_id), (core_id), (step, phase)): (start, length)
            ((0, 0), (0, 0), (0, 0)): (33792, 32),
            ((1, 0), (0, 0), (0, 0)): (33792, 256),
		},
        ((0, 0), 0): {                                  # (chip_id, step_group)
            'step_exe_number': 1,
            0:{                                         # phase group
                'clock': 10000,
                'mode': 1,
                ((0, 0), (0, 0)):{
                    'prims':[{
                        # 'axon': chip_0_0_axon,
                        'axon': None,
                        'soma1': chip_0_0_soma1,
                        'router': chip_0_0_router,
                        'soma2': None
                    }]
                },
            }
        },
        ((1, 0), 0): {
            'step_exe_number':1,
            0:{
                'clock':12000,
                'mode':1,
                ((1, 0), (0, 0)):{
                    'prims':[{
                        # 'axon': chip_0_0_axon,
                        'axon': None,
                        'soma1': None,
                        'router': chip_1_0_router,
                        'soma2': None
                    }]
                }
            }
        }
    }

    
    from generator.test_engine.test_config import HardwareDebugFileSwitch

    test_config = {
        'tb_name': tb_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.burst_config_0.dict, # burst_config_0, burst_read_1
        'test_group_phase': [(0, 0), (0, 1)],
    }

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()

if __name__ =="__main__":


    test_case()


    # class Fib:
    #     def __init__(self, max):
    #         self.max = max
    #     def __iter__(self):
    #         for i in range(10):
    #             print('__iter__ called')
    #             self.a = 0
    #             self.b = 1
    #             # return self
    #             yield self.a, self.b
    #     def __next__(self):
    #         print('__next__ called')
    #         fib = self.a
    #         if fib > self.max:
    #             raise StopIteration
    #         self.a, self.b = self.b, self.a + self.b
    #         return fib

    # f = Fib(3)
    # print(f)
    # print("...............")
    # for i in f:
    #     print(i)
