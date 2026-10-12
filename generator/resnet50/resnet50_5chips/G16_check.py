from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G16_data import generate_g16_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data: dict, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 2
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # receive 24*512
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.conv1']['input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19000 >> 2,
            'type': 1
        }]
        # receive 25*512
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.conv1']['input3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1c000 >> 2,
            'type': 1
        }]
        # L45 in
        data_1[empty_offset + offset + 2 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.cut1']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x8300 >> 2,
            'type': 1
        }]
        # shortcut in
        data_1[empty_offset + offset + 3 + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.conv1']['input2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1f980 >> 2,
            'type': 1
        }]
        # L46 in
        data_1[empty_offset + offset + 6 + 2][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.cut2']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19000 >> 2,
            'type': 1
        }]
        # L46 out 1/3
        data_1[empty_offset + offset + 8 + 2][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.cut3']['output1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x8300 >> 2,
            'type': 1
        }]
        # L46 out 2/3
        data_1[empty_offset + offset + 10 + 2][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.cut3']['output2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x8600 >> 2,
            'type': 1
        }]
        # L46 out 3/3
        if core_y == 0:
            data_1[empty_offset + offset + 12 + 2][(chip, (core_x, core_y))] = [{
                'data': data['layer4.1.cut3']['output3'][(core_x - size_x[0], core_y - size_y[0])],
                'addr_start': 0x8900 >> 2,
                'type': 1
            }]
        # # L46 out 12, 12, 12, 13
        # data_1[empty_offset + offset + 13][(chip, (core_x, core_y))] = [{
        #     'data': data['layer4.1.cut3']['output4'][(core_x - size_x[0], core_y - size_y[0])],
        #     'addr_start': 0x1f300 >> 2 if core_y == 3 else 0x1f380 >> 2,
        #     'type': 1
        # }]
        # Y out 12, 12, 12, 13
        data_1[empty_offset + offset + 14 + 2][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.cut5']['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start':  0x1f200 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g16_data(handler, size_y=4, size_x=16)
    check(data, size_y=4, size_x=16, chip=(1, 0), data_file_name='R00016')
