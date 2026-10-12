# coding: utf-8

import os
import numpy as np
from generator.test_engine import TestMode, TestEngine
from generator.test_engine.test_config import HardwareDebugFileSwitch

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

"""
    test
    Lenet
    int2 精度的权重， int8 精度的输入
"""


def main():
    tb_name = str(os.path.basename(__file__)).split("_")[1].split(".")[0]
    # np.random.seed(0x9562)

    axon81_conv1 = Prim_81_Axon()
    axon81_conv1.PIC = 0x81
    axon81_conv1.InA_type = 1
    axon81_conv1.InB_type = 3
    axon81_conv1.Load_Bias = 0
    axon81_conv1.pad_on = False
    axon81_conv1.Input_fm_Px = 32
    axon81_conv1.Input_fm_Py = 32
    axon81_conv1.conv_Kx = 5
    axon81_conv1.conv_Ky = 5
    axon81_conv1.conv_Sx = 1
    axon81_conv1.conv_Sy = 1
    axon81_conv1.conv_Ex = 1
    axon81_conv1.conv_Ey = 1
    axon81_conv1.pad_up = 0
    axon81_conv1.pad_down = 0
    axon81_conv1.pad_left = 0
    axon81_conv1.pad_right = 0
    axon81_conv1.cin = 1
    axon81_conv1.cout = 6
    axon81_conv1.Reset_Addr_A = 1
    axon81_conv1.Reset_Addr_V = 1
    axon81_conv1.Addr_InA_base = 0x1290     # 0x0000 + 1200 + 2880 + 672
    # X: 32*32*1/4 = 256
    axon81_conv1.Addr_InB_base = 0x4000  # W: 5*5*1*32/4 = 200
    axon81_conv1.Addr_Bias_base = 0x0  # Bias: 32 = 32
    axon81_conv1.Addr_V_base = 0x1390   # 0x0000 + 1200 + 2880 + 672 + 256
    # Out: 28*28*32*4/4 = 25088
    data = axon81_conv1.init_data()
    axon81_conv1.memory_blocks = [
        {
            'name': "P81_input_X",
            'start': axon81_conv1.Addr_InA_base,
            'data': data[0],
            'mode': 0
        },
        {
            'name': "P81_weight",
            'start': axon81_conv1.Addr_InB_base,
            'data': data[1],
            'mode': 0
        }
    ]

    soma_relu1 = Prim_X5_Soma()
    soma_relu1.PIC_Mode = 0
    soma_relu1.pad_on = False
    soma_relu1.type_in = 0
    soma_relu1.type_out = 1
    soma_relu1.cin = 32
    soma_relu1.cout = 6
    soma_relu1.Input_fm_Px = 28
    soma_relu1.Input_fm_Py = 28
    soma_relu1.pad_top = 0
    soma_relu1.pad_down = 0
    soma_relu1.pad_left = 0
    soma_relu1.pad_right = 0
    soma_relu1.pooling_Kx = 2
    soma_relu1.pooling_Ky = 2
    soma_relu1.pooling_Sx = 2
    soma_relu1.pooling_Sy = 2
    soma_relu1.CMP_C = 0
    soma_relu1.in_cut_start = 0
    soma_relu1.reset_Addr_in = 1
    soma_relu1.reset_Addr_out = 1
    soma_relu1.Row_ck_on = 1
    soma_relu1.Addr_Start_in = axon81_conv1.Addr_V_base
    soma_relu1.Addr_Start_out = 0x72c8  # 0x4000 + 200 + 12800  # Out: 14*14*16/4 = 784
    soma_relu1.in_row_max = 2
    soma_relu1.mem_sel = 0

    axon41_conv2 = Prim_41_Axon()
    axon41_conv2.pad_on = False
    axon41_conv2.InA_type = 1
    axon41_conv2.InB_type = 3
    axon41_conv2.Load_Bias = 0
    axon41_conv2.cin = 6
    axon41_conv2.cout = 16
    axon41_conv2.Input_fm_Px = 14
    axon41_conv2.Input_fm_Py = 14
    axon41_conv2.pad_top = 0
    axon41_conv2.pad_down = 0
    axon41_conv2.pad_left = 0
    axon41_conv2.pad_right = 0
    axon41_conv2.conv_Kx = 5
    axon41_conv2.conv_Ky = 5
    axon41_conv2.conv_Sx = 1
    axon41_conv2.conv_Sy = 1
    axon41_conv2.conv_Ex = 1
    axon41_conv2.conv_Ey = 1
    axon41_conv2.Reset_Addr_A = 1
    axon41_conv2.Reset_Addr_V = 1
    axon41_conv2.Addr_InA_base = 0x72c8     # 0x4000 + 200 + 12800  # X: 14*14*16/4 = 784
    axon41_conv2.Addr_InB_base = 0x0000  # W: 5*5*6*32/4 = 1200
    axon41_conv2.Addr_Bias_base = 0x0000
    axon41_conv2.Addr_V_base = 0x75d8   # 0x4000 + 200 + 12800 + 784
    # Out: 10*10*32*4/4 = 3200
    data = axon41_conv2.init_data()
    axon41_conv2.memory_blocks = [
        {
            'name': "P41_weight",
            'start': axon41_conv2.Addr_InB_base,
            'data': data[1],
            'mode': 0
        }
    ]

    soma_relu2 = Prim_X5_Soma()
    soma_relu2.PIC_Mode = 0
    soma_relu2.pad_on = False
    soma_relu2.type_in = 0
    soma_relu2.type_out = 1
    soma_relu2.cin = 32
    soma_relu2.cout = 16
    soma_relu2.Input_fm_Px = 10
    soma_relu2.Input_fm_Py = 10
    soma_relu2.pad_top = 0
    soma_relu2.pad_down = 0
    soma_relu2.pad_left = 0
    soma_relu2.pad_right = 0
    soma_relu2.pooling_Kx = 2
    soma_relu2.pooling_Ky = 2
    soma_relu2.pooling_Sx = 2
    soma_relu2.pooling_Sy = 2
    soma_relu2.CMP_C = 0
    soma_relu2.in_cut_start = 0
    soma_relu2.reset_Addr_in = 1
    soma_relu2.reset_Addr_out = 1
    soma_relu2.Row_ck_on = 1
    soma_relu2.Addr_Start_in = axon41_conv2.Addr_V_base
    soma_relu2.Addr_Start_out = 0x1390  # 0x0000 + 1200 + 2880 + 672 + 256
    # Out: 5*5*16/4 = 100
    soma_relu2.in_row_max = 4
    soma_relu2.mem_sel = 0

    axon04_fc1 = Prim_04_Axon()
    axon04_fc1.PIC = 0x04
    axon04_fc1.InA_type = 1
    axon04_fc1.InB_type = 3
    axon04_fc1.Load_Bias = 0
    axon04_fc1.cin = 400        # 400
    axon04_fc1.cout = 120       # 120
    axon04_fc1.constant_b = 0
    axon04_fc1.Reset_Addr_A = 1
    axon04_fc1.Reset_Addr_V = 1
    axon04_fc1.Addr_InA_base = 0x1390   # 0x0000 + 1200 + 2880 + 672 + 256
    axon04_fc1.Addr_InB_base = 0x40c8   # 0x4000 + 200  # 400 * 128 / 4 = 12800
    axon04_fc1.Addr_Bias_base = 0x0000
    axon04_fc1.Addr_V_base = 0x0000 + 1200 + 2880 + 672 + 256 + 128
    # 128*4/4 = 128
    data = axon04_fc1.init_data()
    axon04_fc1.memory_blocks = [
        {
            'name': "P04_weight",
            'start': axon04_fc1.Addr_InB_base,
            'data': data[1],
            'mode': 0
        }
    ]

    soma_relu3 = Prim_X5_Soma()
    soma_relu3.PIC_Mode = 0
    soma_relu3.pad_on = False
    soma_relu3.type_in = 0
    soma_relu3.type_out = 1
    soma_relu3.cin = 120
    soma_relu3.cout = 120
    soma_relu3.Input_fm_Px = 1
    soma_relu3.Input_fm_Py = 1
    soma_relu3.pad_top = 0
    soma_relu3.pad_down = 0
    soma_relu3.pad_left = 0
    soma_relu3.pad_right = 0
    soma_relu3.pooling_Kx = 1
    soma_relu3.pooling_Ky = 1
    soma_relu3.pooling_Sx = 1
    soma_relu3.pooling_Sy = 1
    soma_relu3.CMP_C = 0
    soma_relu3.in_cut_start = 0
    soma_relu3.reset_Addr_in = 1
    soma_relu3.reset_Addr_out = 1
    soma_relu3.Row_ck_on = 0
    soma_relu3.Addr_Start_in = axon04_fc1.Addr_V_base
    soma_relu3.Addr_Start_out = 0x4000 + 200 + 12800 + 784
    # Out: 1*1*128/4 = 32
    soma_relu3.in_row_max = 1
    soma_relu3.mem_sel = 0

    axon04_fc2 = Prim_04_Axon()
    axon04_fc2.PIC = 0x04
    axon04_fc2.InA_type = 1
    axon04_fc2.InB_type = 3
    axon04_fc2.Load_Bias = 0
    axon04_fc2.cin = 120
    axon04_fc2.cout = 84
    axon04_fc2.constant_b = 0
    axon04_fc2.Reset_Addr_A = 1
    axon04_fc2.Reset_Addr_V = 1
    axon04_fc2.Addr_InA_base = 0x4000 + 200 + 12800 + 784
    axon04_fc2.Addr_InB_base = 0x0000 + 1200
    # 120 * 96 / 4 = 2880
    axon04_fc2.Addr_Bias_base = 0x0000
    axon04_fc2.Addr_V_base = 0x0000 + 1200 + 2880 + 672 + 256 + 128
    # 96*4/4 = 96
    data = axon04_fc2.init_data()
    axon04_fc2.memory_blocks = [
        {
            'name': "P04_weight",
            'start': axon04_fc2.Addr_InB_base,
            'data': data[1],
            'mode': 0
        }
    ]

    soma_relu4 = Prim_X5_Soma()
    soma_relu4.PIC_Mode = 0
    soma_relu4.pad_on = False
    soma_relu4.type_in = 0
    soma_relu4.type_out = 1
    soma_relu4.cin = 84
    soma_relu4.cout = 84
    soma_relu4.Input_fm_Px = 1
    soma_relu4.Input_fm_Py = 1
    soma_relu4.pad_top = 0
    soma_relu4.pad_down = 0
    soma_relu4.pad_left = 0
    soma_relu4.pad_right = 0
    soma_relu4.pooling_Kx = 1
    soma_relu4.pooling_Ky = 1
    soma_relu4.pooling_Sx = 1
    soma_relu4.pooling_Sy = 1
    soma_relu4.CMP_C = 0
    soma_relu4.in_cut_start = 0
    soma_relu4.reset_Addr_in = 1
    soma_relu4.reset_Addr_out = 1
    soma_relu4.Row_ck_on = 0
    soma_relu4.Addr_Start_in = axon04_fc2.Addr_V_base
    soma_relu4.Addr_Start_out = 0x4000 + 200 + 12800 + 784 + 32
    # Out: 1*1*96/4 = 24
    soma_relu4.in_row_max = 1
    soma_relu4.mem_sel = 0

    axon04_fc3 = Prim_04_Axon()
    axon04_fc3.PIC = 0x04
    axon04_fc3.InA_type = 1
    axon04_fc3.InB_type = 3
    axon04_fc3.Load_Bias = 0
    axon04_fc3.cin = 84
    axon04_fc3.cout = 10
    axon04_fc3.constant_b = 0
    axon04_fc3.Reset_Addr_A = 1
    axon04_fc3.Reset_Addr_V = 1
    axon04_fc3.Addr_InA_base = 0x4000 + 200 + 12800 + 784 + 32
    axon04_fc3.Addr_InB_base = 0x0000 + 1200 + 2880
    # 84 * 32 / 4 = 672
    axon04_fc3.Addr_Bias_base = 0x0000
    axon04_fc3.Addr_V_base = 0x0000 + 1200 + 2880 + 672 + 256 + 128
    # 32*4/4 = 32
    data = axon04_fc3.init_data()
    axon04_fc3.memory_blocks = [
        {
            'name': "P04_weight",
            'start': axon04_fc3.Addr_InB_base,
            'data': data[1],
            'mode': 0
        }
    ]

    soma_relu5 = Prim_X5_Soma()
    soma_relu5.PIC_Mode = 0
    soma_relu5.pad_on = False
    soma_relu5.type_in = 0
    soma_relu5.type_out = 1
    soma_relu5.cin = 16
    soma_relu5.cout = 10
    soma_relu5.Input_fm_Px = 1
    soma_relu5.Input_fm_Py = 1
    soma_relu5.pad_top = 0
    soma_relu5.pad_down = 0
    soma_relu5.pad_left = 0
    soma_relu5.pad_right = 0
    soma_relu5.pooling_Kx = 1
    soma_relu5.pooling_Ky = 1
    soma_relu5.pooling_Sx = 1
    soma_relu5.pooling_Sy = 1
    soma_relu5.CMP_C = 0x00000080
    soma_relu5.in_cut_start = 0
    soma_relu5.reset_Addr_in = 1
    soma_relu5.reset_Addr_out = 1
    soma_relu5.Row_ck_on = 0
    soma_relu5.Addr_Start_in = axon04_fc3.Addr_V_base
    soma_relu5.Addr_Start_out = 0x7800
    # Out: 1*1*16/4 = 4
    soma_relu5.in_row_max = 1
    soma_relu5.mem_sel = 0

    map_config = {
        'sim_clock': 30000,
        # 'step_clock': {
        #     ((0, 0), 0): (50000, 10000)
        # },
        ((0, 0), 0): {
            # 'step_exe_number': 1,
            0: {
                'clock': 15000,
                'mode': 1,
                ((0, 0), (0, 0)): {
                    'prims': [
                        {
                            'axon': axon81_conv1,
                            'soma1': soma_relu1,
                            'router': None,
                            'soma2': None
                        },
                        {
                            'axon': axon41_conv2,
                            'soma1': soma_relu2,
                            'router': None,
                            'soma2': None
                        },
                        {
                            'axon': axon04_fc1,
                            'soma1': None,
                            'router': None,
                            'soma2': soma_relu3
                        },
                        {
                            'axon': axon04_fc2,
                            'soma1': None,
                            'router': None,
                            'soma2': soma_relu4
                        },
                        {
                            'axon': axon04_fc3,
                            'soma1': None,
                            'router': None,
                            'soma2': soma_relu5
                        }
                    ]
                }
            }
        }
    }

    test_config = {
        'tb_name': tb_name,
        'test_mode': TestMode.MEMORY_STATE,
        # 'debug_file_switch': HardwareDebugFileSwitch().singla_chip.dict,
        'test_group_phase': [(0, 1)],
    }

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    main()
