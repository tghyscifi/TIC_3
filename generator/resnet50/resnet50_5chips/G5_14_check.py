from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G5_data import generate_g5_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 4
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # recv 1/4
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        # recv 2/4
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x11c00 >> 2,
            'type': 1
        }]
        # recv 3/4
        data_1[empty_offset + 2][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x13800 >> 2,
            'type': 1
        }]
        # recv 4/4
        data_1[empty_offset + 3][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input4'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x15400 >> 2,
            'type': 1
        }]

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 0][chip, (core_x, core_y)] = [{
            'data': data['conv3a3e_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g5_data(handler, size_y=1, size_x=14)
    check(data, size_y=1, size_x=14)
