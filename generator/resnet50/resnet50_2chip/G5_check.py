from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_2chip.G1_data import generate_g1_data
from generator.resnet50.resnet50_2chip.resnet50_2chip_data_handler import ResNetDataHandler


def check(data, compare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 1

    data_1[empty_offset + 0] = {}
    data_1[empty_offset + 1] = {}
    # receive
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [
            {
                'data': data['fc_0_input'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x8380,
                'type': 1
            },
            {
                'data': data['fc_0_input'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x0000 >> 2,
                'type': 1
            }
        ]

    data_1[empty_offset + offset + 0] = {}  # fc_0
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if (core_x - size_x[0], core_y - size_y[0]) == (0, 0):
            data_1[empty_offset + offset + 0][chip, (core_x, core_y)] = [{
                'data': data['fc_0_cut'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x8380,
                'type': 1
            }]

    data_1[empty_offset + offset + 1] = {}  # fc_0 out
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if (core_x - size_x[0]) < 8 and (core_y - size_y[0]) == 0:
            data_1[empty_offset + offset + 1][chip, (core_x, core_y)] = [{
                'data': data['fc_0_cut'][(0, 0)],
                'addr_start': 0x0000 >> 2,
                'type': 1
            }]

    # fc_1 relu
    data_1[empty_offset + offset + 2] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if (core_x - size_x[0], core_y - size_y[0]) == (0, 0):
            data_1[empty_offset + offset + 2][chip, (core_x, core_y)] = [{
                'data': data['fc_1_cut'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x0000 >> 2,
                'type': 1
            }]

    data_1[empty_offset + offset + 3] = {}  # fc out
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if (core_x - size_x[0], core_y - size_y[0]) == (0, 0):
            data_1[empty_offset + offset + 3][chip, (core_x, core_y)] = [{
                'data': data['fca2_output_all'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0xffe0 >> 2,
                'type': 1
            }]
    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g1_data(handler, size_y=2, size_x=16)
    check(data, size_y=2, size_x=16, chip=(1, 0), data_file_name='R00001')
