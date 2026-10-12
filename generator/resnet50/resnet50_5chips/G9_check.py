from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0)):
    data_1 = {}
    offset = 4
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(2, 2 + size_y), range(size_x)):
        # shortcut in
        data_1[offset + 14][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut4']['output2'][(core_x, core_y - 2)],
            'addr_start': 0xab80 >> 2,
            'type': 1
        }]
        # out
        data_1[offset + 15][(chip, (core_x, core_y))] = [{
            'data': data['layer3.0.cut5']['output'][(core_x, core_y - 2)],
            'addr_start': 0x18000 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)
    check(data, size_y=4, size_x=8, chip=(1, 0))
