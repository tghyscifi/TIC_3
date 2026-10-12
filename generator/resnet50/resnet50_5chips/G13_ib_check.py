import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G12_data import generate_g12_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, chip=(0, 0), empty_offset=0):
    data_1 = {}
    for i in range(32):
        data_1[i] = {}

    for core_x, core_y in [(8, 9), (9, 9), (10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 9)]:
        # recv 1/2
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer3.4.conv1']['input5'][(core_x % 2, (core_x - 8) // 2)],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        # recv 2/2
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer3.4.conv1']['input6'][(core_x % 2, (core_x - 8) // 2)],
            'addr_start': 0x13100 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g12_data(handler, size_y=4, size_x=8)
    check(data, size_y=4, size_x=8, chip=(1, 2))
