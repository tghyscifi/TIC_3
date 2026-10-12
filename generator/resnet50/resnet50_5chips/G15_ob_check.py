import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G16_data import generate_g16_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, chip=(0, 0), empty_offset=0):
    data_1 = {}
    for i in range(32):
        data_1[i] = {}

    for core_x, core_y in [(0, 0), (4, 0), (8, 0), (12, 0)]:
        # receive 24*512
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.conv1']['input1'][(0, core_x // 4)],
            'addr_start': 0x10000 >> 2,
            'type': 1
        }]
        # receive 25*512
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['layer4.1.conv1']['input3'][(0, core_x // 4)],
            'addr_start': 0x13000 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g16_data(handler, size_y=4, size_x=16)
    check(data, chip=(1, 0), data_file_name='R00016')
