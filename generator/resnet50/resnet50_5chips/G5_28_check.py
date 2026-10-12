from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G5_data import generate_g5_data
from generator.resnet50.data_handler import ResNetDataHandler
from generator.mapping_utils.result_compare import ResultCompare

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def check(data, compare: ResultCompare, size_y, size_x, chip=(0, 0), empty_offset=0):
    data_1 = {}
    offset = 4
    for i in range(32):
        data_1[i] = {}

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        # recv 1/4
        data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19000 >> 2,
            'type': 1
        }]
        # recv 2/4
        data_1[empty_offset + 1][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input2'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1ac00 >> 2,
            'type': 1
        }]
        # recv 3/4
        data_1[empty_offset + 2][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input3'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1c800 >> 2,
            'type': 1
        }]
        # recv 4/4
        data_1[empty_offset + 3][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_input4'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x1e400 >> 2,
            'type': 1
        }]

    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 0][(chip, (core_x, core_y))] = [{
            'data': data['conv3a1_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x7D00 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 2] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 2][(chip, (core_x, core_y))] = [{
            'data': data['conv3a2_collation0'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0xA700 >> 2,
            'type': 1
        }]
    # data_1[empty_offset + offset + 3] = {}
    # for core_y, core_x in product(range(*size_y), range(*size_x)):
    #     data_1[empty_offset + offset + 3][(chip, (core_x, core_y))] = [{
    #         'data': data['conv3a2_collation1'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0xDF00 >> 2 if core_x % 7 == 0 else 0x9900 >> 2,
    #         'type': 1
    #     }]
    data_1[empty_offset + offset + 5] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 5][(chip, (core_x, core_y))] = [{
            'data': data['conv3a2_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x7D00 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 7] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 7][(chip, (core_x, core_y))] = [{
            'data': data['conv3a3_input'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x19000 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 8] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 8][(chip, (core_x, core_y))] = [{
            'data': data['conv3a3_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x9900 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 9] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 9][(chip, (core_x, core_y))] = [{
            'data': data['instant_prim0'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x6100 >> 2,
            'type': 1
        }]
    data_1[empty_offset + offset + 10] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 10][(chip, (core_x, core_y))] = [{
            'data': data['instant_prim1'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x7D00 >> 2,
            'type': 1
        }]
    # data_1[empty_offset + offset + 11] = {}
    # for core_y, core_x in product(range(*size_y), range(*size_x)):
    #     data_1[empty_offset + offset + 11][(chip, (core_x, core_y))] = [{
    #         'data': data['test0'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x19000 >> 2,
    #         'type': 1
    #     }]
    # data_1[empty_offset + offset + 12] = {}
    # for core_y, core_x in product(range(*size_y), range(*size_x)):
    #     data_1[empty_offset + offset + 12][(chip, (core_x, core_y))] = [{
    #         'data': data['test1'][(core_x - size_x[0], core_y - size_y[0])],
    #         'addr_start': 0x19000 >> 2,
    #         'type': 1
    #     }]
    data_1[empty_offset + offset + 11] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data_1[empty_offset + offset + 11][(chip, (core_x, core_y))] = [{
            'data': data['g5_add_output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x6100 >> 2,
            'type': 1
        }]
    compare.add_ref_data(data_1)


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g5_data(handler, size_y=2, size_x=14)
    check(data, size_y=2, size_x=14, chip=(1, 0))
