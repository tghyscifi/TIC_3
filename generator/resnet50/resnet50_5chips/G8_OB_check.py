import sys
import os

sys.path.append(os.getcwd())
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, empty_offset=0):
    data_1 = {}
    for i in range(32):
        data_1[i] = {}

    chip = (0, 1)
    for core_x, core_y in [(8, 9), (9, 9), (10, 9), (11, 9)]:
        data_1[empty_offset + 0][chip, (core_x, core_y)] = [{
            'data': data['layer3.0.conv1']['input1'][(core_x, 0)],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        data_1[empty_offset + 1][chip, (core_x, core_y)] = [{
            'data': data['layer3.0.conv1']['input2'][(core_x, 0)],
            'addr_start': 0x13000 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)
