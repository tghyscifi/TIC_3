from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G1_data import generate_g1_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 2

    data_1[empty_offset + 0] = {}
    data_1[empty_offset + 1] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + 0][chip, (core_x, core_y)] = [{
            'data': data['conv1a1_input'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0800 >> 2,
            'type': 1
        }]
        if (core_x, core_y) == (15, 2):
            data_1[empty_offset + 0][(chip, (core_x, core_y))].append(
                {
                    'data': [*data['conv1a1_input'][(core_x - size_x[0], core_y - size_y[0])],
                             *data['fca2_input'][(core_x - size_x[0], core_y - size_y[0])]],
                    'addr_start': 0x8380,
                    'type': 1
                }
            )
        data_1[empty_offset + 1][chip, (core_x, core_y)] = [{
            'data': data['fca2_input'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0000 >> 2,
            'type': 1
        }]

    data_1[empty_offset + offset + 0] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 0][chip, (core_x, core_y)] = [{
            'data': data['fca2_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xFFE0 >> 2,
            'type': 1
        }]
    # data_1[empty_offset + offset + 2] = {}
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     data_1[empty_offset + offset + 2][chip, (core_x, core_y)] = [{
    #         'data': data['conv1a1_collation0'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x5000 >> 2,
    #         'type': 1
    #     }]
    # data_1[empty_offset + offset + 3] = {}
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     data_1[empty_offset + offset + 3][chip, (core_x, core_y)] = [{
    #         'data': data['conv1a1_collation1'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x0000 >> 2,
    #         'type': 1
    #     }]
    data_1[empty_offset + offset + 4] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 4][chip, (core_x, core_y)] = [{
            'data': data['conv1a1_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x84C0 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 8] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 8][chip, (core_x, core_y)] = [{
            'data': data['maxpool1a2_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xe3e0 >> 2 if core_x % 2 == 0 else 0xeae0 >> 2,
            'type': 1
        }]
    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g1_data(handler, size_y=2, size_x=16)
    check(data, size_y=2, size_x=16, chip=(1, 0), data_file_name='R00001')
