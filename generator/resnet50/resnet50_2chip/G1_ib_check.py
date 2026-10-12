import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_2chip.G1_data import generate_g1_data
from generator.resnet50.resnet50_2chip.resnet50_2chip_data_handler import ResNetDataHandler


def check(data, compare, empty_offset=0):
    data_1 = {}

    data_1[empty_offset + 0] = {}
    data_1[empty_offset + 1] = {}
    data_1[empty_offset + 2] = {}
    chip = (0, 0)
    for core_x, core_y in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]:
        data_1[empty_offset + 0][chip, (core_x, core_y)] = [{
            'data': data['conv1a1_input2'][(core_x, core_y)],
            'addr_start': 0x0000 >> 2,
            'type': 1
        }]
        data_1[empty_offset + 1][chip, (core_x, core_y)] = [{
            'data': data['conv1a1_input3'][(core_x, core_y)],
            'addr_start': 0x2680 >> 2,
            'type': 1
        }]
        data_1[empty_offset + 2][chip, (core_x, core_y)] = [{
            'data': data['conv1a1_input4'][(core_x, core_y)],
            'addr_start': 0x4c80 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g1_data(handler, size_y=2, size_x=16)
    check(data, size_y=2, size_x=16, chip=(1, 0), data_file_name='R00001')
