from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G11_data import generate_g11_data
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
            'data': data['layer3.2.conv1']['input5'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        # recv 2/4
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.conv1']['input6'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x13100 >> 2,
            'type': 1
        }]
        # recv 3/4
        data_1[empty_offset + 2][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.conv1']['input7'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x16200 >> 2,
            'type': 1
        }]
        # recv 4/4
        data_1[empty_offset + 3][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.conv1']['input8'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19300 >> 2,
            'type': 1
        }]

        # L26 out 1/2
        data_1[empty_offset + offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut1']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xe140 >> 2,
            'type': 1
        }]
        # L26 out 2/2
        data_1[empty_offset + offset + 3][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut1']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xe460 >> 2,
            'type': 1
        }, {
            'data': data['layer3.2.conv1']['input3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xe780 >> 2,
            'type': 1       # shortcut in
        }
        ]
        # L27 in
        data_1[empty_offset + offset + 5][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut1']['output3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        # L27 out 1/2
        data_1[empty_offset + offset + 7][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut2']['output1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1f9c0 >> 2,
            'type': 1
        }]
        # L27 out 2/2
        data_1[empty_offset + offset + 9][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut2']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1fce0 >> 2,
            'type': 1
        }]
        # L28 input 1/4 ~ 4/4
        data_1[empty_offset + offset + 11][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut2']['output3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        data_1[empty_offset + offset + 13][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut2']['output4'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x13200 >> 2,
            'type': 1
        }]
        data_1[empty_offset + offset + 15][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut2']['output5'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x16200 >> 2,
            'type': 1
        }]
        data_1[empty_offset + offset + 17][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut2']['output6'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19400 >> 2,
            'type': 1
        }]
        # L28 out
        data_1[empty_offset + offset + 18][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut3']['output1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xcf00 >> 2,
            'type': 1
        }]
        # out
        data_1[empty_offset + offset + 19][(chip, (core_x, core_y))] = [{
            'data': data['layer3.2.cut5']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1e780 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g11_data(handler, size_y=4, size_x=8)
    check(data, size_y=4, size_x=8, chip=(1, 2))
