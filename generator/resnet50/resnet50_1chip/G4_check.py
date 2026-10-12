from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G4_data import generate_g4_data
from generator.resnet50.data_handler import ResNetDataHandler


def check(data, compare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 4
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # recv 1/4
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer1.2.conv1']['input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x2000 >> 2,
            'type': 1
        }]
        # recv 2/4
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer1.2.conv1']['input2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x3c00 >> 2,
            'type': 1
        }]
        # recv 3/4
        data_1[empty_offset + 2][(chip, (core_x, core_y))] = [{
            'data': data['layer1.2.conv1']['input3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x5800 >> 2,
            'type': 1
        }]
        # recv 4/4
        data_1[empty_offset + 3][(chip, (core_x, core_y))] = [{
            'data': data['layer1.2.conv1']['input4'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x7400 >> 2,
            'type': 1
        }]
    # L2
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     addr_out = 0x9e00 >> 2 if core_y == 0 else 0xba00 >> 2
    #     data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
    #         'data': data['layer1.2.cut1']['output'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': addr_out,
    #         'type': 1
    #     }]
    #
    # # L6
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     data_1[empty_offset + 4][(chip, (core_x, core_y))] = [{
    #         'data': data['layer1.2.cut2']['output'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x1e400 >> 2,
    #         'type': 1
    #     }]
    #
    # # L7
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     data_1[empty_offset + 6][(chip, (core_x, core_y))] = [
    #         {
    #             'data': data['layer1.2.cut3']['output1'][(core_x - size_x[0], core_y - size_y[0])],
    #             'addr_start': 0x9000 >> 2,
    #             'type': 1
    #         },
    #     ]
    #
    # # ADD in L7
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     data_1[empty_offset + 9][(chip, (core_x, core_y))] = [{
    #         'data': data['layer1.2.cut3']['output2'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x9000 >> 2,
    #         'type': 1
    #     }]
    #

    # Y
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 8][(chip, (core_x, core_y))] = [{
            'data': data['layer1.2.cut5']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x9000 >> 2,
            'type': 1
        }]
        if (core_x - size_x[0], core_y - size_y[0]) == (0, 0):
            data_1[empty_offset + offset + 11][(chip, (core_x, core_y))] = [{
                'data': data['avgpool_cut']['output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x9000 >> 2,
                'type': 1
            }]
    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g4_data(handler, size_y=2, size_x=14)
    check(data, size_y=2, size_x=14, chip=(1, 0))
