# coding: utf-8

import os
import sys

sys.path.append(os.getcwd())
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


def case():
    tb_name = str(os.path.basename(__file__)).split("_")[1].split(".")[0]
    np.random.seed(0x9028)

    axon04 = Prim_04_Axon()
    axon04.PIC = 0x04

    axon04.InA_type = 1
    axon04.InB_type = 3
    axon04.Load_Bias = 0
    axon04.cin = 512
    axon04.cout = 512
    axon04.constant_b = 0
    axon04.Reset_Addr_A = 1
    axon04.Reset_Addr_V = 1
    axon04.Addr_InA_base = 0x0000
    axon04.Addr_InB_base = 0x4000
    axon04.Addr_Bias_base = 0x0000
    axon04.Addr_V_base = 0x2000

    a = axon04.init_data()
    axon04.memory_blocks = [{'name': "P04_input_X",
                             'start': axon04.Addr_InA_base,
                             'data': a[0],
                             'mode': 0},
                            {'name': "P04_weight",
                             'start': axon04.Addr_InB_base,
                             'data': a[1],
                             'mode': 0}]

    num_chips = 1
    num_cores = (1, 1)

    map_config = {
        'sim_clock': None,
        'step_clock': {
            ((0, 0), 0): (5000, 10000)
        },
        ((0, 0), 0): {
            'step_exe_number': 1,
            0: {
                'clock': 5000,
                'mode': 1,
                ((0, 0), (0, 0)): {
                    'prims': [{
                        'axon': axon04,
                        'soma1': None,
                        'router': None,
                        'soma2': None
                    }]
                }
            }
        }
    }

    step_clock_dict = map_config['step_clock']
    for i in range(num_chips):
        step_clock_dict[(0, i), 0] = (5000, 10000)
        map_config[(0, i), 0] = {}
        map_config[(0, i), 0]["step_exe_number"] = 1
        map_config[(0, i), 0][i] = {}
        map_config[(0, i), 0][i]['clock'] = 5000
        map_config[(0, i), 0][i]['mode'] = 1
        for j in range(num_cores[0]):
            for k in range(num_cores[1]):
                map_config[(0, i), 0][i][(0, i), (j, k)] = {'prims': [{
                    'axon': axon04,
                    'soma1': None,
                    'router': None,
                    'soma2': None
                }]}

    test_group_phase = [(i, 0) for i in range(num_chips)]

    test_config = {
        'tb_name': tb_name,
        'test_mode': TestMode.MEMORY_STATE,
        'debug_file_switch': HardwareDebugFileSwitch().close_all.multi_chip.dict,
        'test_group_phase': test_group_phase
    }

    tester = TestEngine(map_config, test_config)
    assert tester.run_test()


if __name__ == "__main__":
    case()
