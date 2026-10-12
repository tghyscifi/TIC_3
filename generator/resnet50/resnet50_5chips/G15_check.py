from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G15_data import generate_g15_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 2
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # L41 in
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.conv1']['input'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1cf00 >> 2,
            'type': 1
        }]

        # L41 out
        data_1[empty_offset + offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut1']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xfe60 >> 2,
            'type': 1
        }]
        # L41e out 1/3
        data_1[empty_offset + offset + 3][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut4']['output1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19000 >> 2,
            'type': 1
        }]
        # L41e out 2/3
        data_1[empty_offset + offset + 5][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut4']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19300 >> 2,
            'type': 1
        }]
        # L41e out 3/3
        if core_y == 0:
            data_1[empty_offset + offset + 7][(chip, (core_x, core_y))] = [{
                'data': data['layer4.0.cut4']['output3'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x19600 >> 2,
                'type': 1
            }]
        # L42 in
        data_1[empty_offset + offset + 9][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut1']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xe500 >> 2,
            'type': 1
        }]
        # L43 in
        data_1[empty_offset + offset + 12 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut2']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1e780 >> 2,
            'type': 1
        }]
        # L41e out 12, 12, 12, 13
        data_1[empty_offset + offset + 13 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut4']['output4'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xe500 >> 2,
            'type': 1
        }]
        # L43 out 1/3
        data_1[empty_offset + offset + 15 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut3']['output1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19000 >> 2,
            'type': 1
        }]
        # L43 out 2/3
        data_1[empty_offset + offset + 17 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut3']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19300 >> 2,
            'type': 1
        }]
        # L43 out 3/3
        if core_y == 0:
            data_1[empty_offset + offset + 19 + 1][(chip, (core_x, core_y))] = [{
                'data': data['layer4.0.cut3']['output3'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x19600 >> 2,
                'type': 1
            }]
        # L43 out 12, 12, 12, 13
        data_1[empty_offset + offset + 20 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut3']['output4'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xeb80 >> 2 if core_y - size_y[0] == 3 else 0xeb00 >> 2,
            'type': 1
        }]
        # Y out 12, 12, 12, 13
        data_1[empty_offset + offset + 21 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.0.cut5']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start':  0x19000 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g15_data(handler, size_y=4, size_x=16)
    check(data, size_y=4, size_x=16, chip=(1, 2))
