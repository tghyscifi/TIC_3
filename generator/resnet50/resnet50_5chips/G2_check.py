from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G2_data import generate_g2_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 5
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # recv
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer1.0.conv1']['input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x5800 >> 2,
            'type': 1
        }]

        # 交换 + recv
        data_1[empty_offset + 4][(chip, (core_x, core_y))] = [{
            'data': data['layer1.0.conv1']['input'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x2000 >> 2,
            'type': 1
        }]

    # L2
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer1.0.cut1']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x6600 >> 2,
            'type': 1
        }]

    # L3
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 5][(chip, (core_x, core_y))] = [{
            'data': data['layer1.0.cut2']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x5800 >> 2,
            'type': 1
        }]

    # L4
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 6][(chip, (core_x, core_y))] = [{
            'data': data['layer1.0.cut3']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x9000 >> 2,
            'type': 1
        },
            {
                'data': data['layer1.0.downsample.0']['input'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x17500 >> 2,
                'type': 1
            }
        ]

    # L4e
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 7][(chip, (core_x, core_y))] = [{
            'data': data['layer1.0.cut4']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x2000 >> 2,
            'type': 1
        }]

    # Y
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 8][(chip, (core_x, core_y))] = [{
            'data': data['layer1.0.cut5']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x9000 >> 2,
            'type': 1
        }]
    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g2_data(handler, size_y=2, size_x=14)
    check(data, size_y=2, size_x=14, chip=(0, 1))
