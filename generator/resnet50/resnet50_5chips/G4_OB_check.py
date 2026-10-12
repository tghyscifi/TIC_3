import sys
import os

sys.path.append(os.getcwd())
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, empty_offset=0):
    data_1 = {}
    for i in range(32):
        data_1[i] = {}

    chip = (0, 0)
    for core_x, core_y in [(9, 9), (10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9)]:
        data_1[empty_offset + 0][chip, (core_x, core_y)] = [{
            'data': data['conv3a1_input1'][(core_x - 9, 0)],
            'addr_start': 0x0000 >> 2,
            'type': 1
        }]
        data_1[empty_offset + 1][chip, (core_x, core_y)] = [{
            'data': data['conv3a1_input2'][(core_x - 9, 0)],
            'addr_start': 0x1c00 >> 2,
            'type': 1
        }]
        data_1[empty_offset + 2][chip, (core_x, core_y)] = [{
            'data': data['conv3a1_input3'][(core_x - 9, 0)],
            'addr_start': 0x3800 >> 2,
            'type': 1
        }]
        data_1[empty_offset + 3][chip, (core_x, core_y)] = [{
            'data': data['conv3a1_input4'][(core_x - 9, 0)],
            'addr_start': 0x5400 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)
