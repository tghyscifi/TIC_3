from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 4
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # recv 1/2
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.conv1']['input1'][((core_y - size_y[0]) + 8, 0)],
            'addr_start': 0x9300 >> 2,
            'type': 1
        }]
        # recv 2/2
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.conv1']['input2'][((core_y - size_y[0]) + 8, 0)],
            'addr_start': 0xc300 >> 2,
            'type': 1
        }]

        # L24 in
        data_1[empty_offset + offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut1']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x18700 >> 2,
            'type': 1
        }]
        # L25 in
        data_1[empty_offset + offset + 5][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut2']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x9300 >> 2,
            'type': 1
        }]
        # L25 out 1 / 2
        data_1[empty_offset + offset + 8][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut3']['output1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xc400 >> 2,
            'type': 1
        }]
        # L25 out 2 / 2
        data_1[empty_offset + offset + 11][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut3']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xd080 >> 2,
            'type': 1
        }]
        # L25 out
        data_1[empty_offset + offset + 13][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut3']['output3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x9300 >> 2,
            'type': 1
        }]
        # shortcut in
        data_1[empty_offset + offset + 14][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut4']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xab80 >> 2,
            'type': 1
        }]
        # out
        data_1[empty_offset + offset + 15][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut5']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x18000 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)
    check(data, size_y=4, size_x=8)
