from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G6_data import generate_g6_data
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
            'data': data['conv3b1_input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19000 >> 2,
            'type': 1
        }]
        # recv 2/2
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['conv3b1_input2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1ac00 >> 2,
            'type': 1
        }]

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if core_x - size_x[0] < 7 and core_y - size_y[0] == 0:
            addr_start = 0x8400 >> 2
        elif core_x - size_x[0] >= 7 and core_y - size_y[0] == 0:
            addr_start = 0x8780 >> 2
        elif core_x - size_x[0] < 7 and core_y - size_y[0] == 1:
            addr_start = 0x8B00 >> 2
        else:
            addr_start = 0x8E80 >> 2

        data_1[empty_offset + offset + 0][chip, (core_x, core_y)] = [{
            'data': data['conv3b1_output_part1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': addr_start,
            'type': 1
        }]

    data_1[empty_offset + offset + 1] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 1][chip, (core_x, core_y)] = [{
            'data': data['conv3b1_weight1_1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x0000 >> 2,
            'type': 1
        }]

    data_1[empty_offset + offset + 3] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if core_x - size_x[0] < 7 and core_y - size_y[0] == 0:
            addr_start = 0x8E80 >> 2
        elif core_x - size_x[0] >= 7 and core_y - size_y[0] == 0:
            addr_start = 0x9200 >> 2
        elif core_x - size_x[0] < 7 and core_y - size_y[0] == 1:
            addr_start = 0x9200 >> 2
        else:
            addr_start = 0x9200 >> 2

        data_1[empty_offset + offset + 3][chip, (core_x, core_y)] = [{
            'data': data['conv3b1_output_part2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': addr_start,
            'type': 1
        }]

    data_1[empty_offset + offset + 9] = {}  # 第9个phase
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if core_y - size_y[0] == 1:
            data_1[empty_offset + offset + 9][chip, (core_x, core_y)] = [{
                'data': data['conv3b1_output'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x1C800 >> 2,
                'type': 1
            }]
        else:
            pass
    data_1[empty_offset + offset + 10] = {}  # 第10个phase
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if core_y - size_y[0] == 0:
            if core_x - size_x[0] == 0:
                data_1[empty_offset + offset + 10][chip, (core_x, core_y)] = [{
                    'data': data['conv3b1_output'][(core_x - size_x[0], core_y - size_y[0])],
                    'addr_start': 0xA380 >> 2,
                    'type': 1
                }]
            else:
                data_1[empty_offset + offset + 10][chip, (core_x, core_y)] = [{
                    'data': data['conv3b1_output'][(core_x - size_x[0], core_y - size_y[0])],
                    'addr_start': 0x1C800 >> 2,
                    'type': 1
                }]
        else:
            pass
    data_1[empty_offset + offset + 13] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if core_x - size_x[0] + (core_y - size_y[0]) * (size_x[1] - size_x[0]) == 0:
            data_1[empty_offset + offset + 13][chip, (core_x, core_y)] = [{
                'data': data['g6_phase13'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0xB180 >> 2,
                'type': 1
            }]
        else:
            pass
    data_1[empty_offset + offset + 14] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if 0 <= core_x - size_x[0] + (core_y - size_y[0]) * (size_x[1] - size_x[0]) <= 6:
            pass
        else:
            data_1[empty_offset + offset + 14][chip, (core_x, core_y)] = [{
                'data': data['conv3b2_input0'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0xA380 >> 2 if (core_x - size_x[0] + (core_y - size_y[0]) * (
                        size_x[1] - size_x[0])) % 7 == 0 else 0x9580 >> 2,
                'type': 1
            }]
    data_1[empty_offset + offset + 15] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if core_x - size_x[0] + (core_y - size_y[0]) * (size_x[1] - size_x[0]) == 0:
            data_1[empty_offset + offset + 15][chip, (core_x, core_y)] = [{
                'data': data['g6_phase15'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0xBF80 >> 2,
                'type': 1
            }]
        else:
            pass
    data_1[empty_offset + offset + 16] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        if 0 <= core_x - size_x[0] + (core_y - size_y[0]) * (size_x[1] - size_x[0]) <= 6:
            pass
        else:
            data_1[empty_offset + offset + 16][chip, (core_x, core_y)] = [{
                'data': data['conv3b2_input1'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0xBF80 >> 2,
                'type': 1
            }]
    data_1[empty_offset + offset + 17] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 17][chip, (core_x, core_y)] = [{
            'data': data['conv3b2_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1E400 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 18] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 18][chip, (core_x, core_y)] = [{
            'data': data['conv3b3_input0'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x8400 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 19] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 19][chip, (core_x, core_y)] = [{
            'data': data['conv3b3_input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1C800 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 20] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 20][chip, (core_x, core_y)] = [{
            'data': data['conv3b3_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xBC00 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 21] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 21][chip, (core_x, core_y)] = [{
            'data': data['conv3b3_output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1C800 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 22] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 22][chip, (core_x, core_y)] = [{
            'data': data['conv3b3_output3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1E400 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 23] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 23][chip, (core_x, core_y)] = [{
            'data': data['g6_add_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1c800 >> 2,
            'type': 1
        }]
    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g6_data(handler, size_y=2, size_x=14)
    check(data, size_y=2, size_x=14)
