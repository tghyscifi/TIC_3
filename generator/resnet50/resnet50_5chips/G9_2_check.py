from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 2
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # recv 1/2
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.conv1']['input1'][
                ((core_x - size_x[0]) % 2 + (core_y - size_y[0]) % 2 * 2 + 8, 0)],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        # recv 2/2
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.conv1']['input2'][
                ((core_x - size_x[0]) % 2 + (core_y - size_y[0]) % 2 * 2 + 8, 0)],
            'addr_start': 0x13000 >> 2,
            'type': 1
        }]

        # L23e out
        data_1[empty_offset + offset + 4][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut4']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)
    check(data, size_y=2, size_x=8)
