from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.LeNet.lenet_data import generate_data
from generator.LeNet.lenet_model.lenet_data_handler import LeNetDataHandler
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator


def check(data, compare, empty_offset=0):
    data_1 = {}
    for i in range(32):
        data_1[i] = {}

    chip = (0, 0)

    # # recv
    # data_1[empty_offset + 0][(chip, (core_x, core_y))] = [{
    #     'data': data['layer1.0.conv1']['input1'][(core_x - size_x[0], core_y - size_y[0])],
    #     'addr_start': 0x5800 >> 2,
    #     'type': 1
    # }]

    # max pool 1
    data_1[empty_offset + 1][(chip, (0, 0))] = [{
        'data': data['max_pool1']['output'][(0, 0)],
        'addr_start': 0x0400 >> 2,
        'type': 1
    }]

    # max pool 2
    data_1[empty_offset + 2][(chip, (0, 0))] = [{
        'data': data['max_pool2']['output'][(0, 0)],
        'addr_start': 0x1A40 >> 2,
        'type': 1
    }]

    # fc cut3
    data_1[empty_offset + 5][(chip, (0, 0))] = [{
        'data': data['fc_cut3']['output'][(0, 0)],
        'addr_start': 0x2000 >> 2,
        'type': 1
    }]

    # g0
    chip = (0, 0)
    data_1[empty_offset + 0][(chip, (1, 0))] = [{
        'data': data['fc_cut3']['output'][(0, 0)],
        'addr_start': 0x8380,
        'type': 1
    }]

    # g2
    chip = (0, 0)
    data_1[empty_offset + 0][(chip, (0, 1))] = [
        {
            'data': data['fc_cut3']['output'][(0, 0)],
            'addr_start': 0x0000,
            'type': 1
        },
        {
            'data': data['fc_cut3']['output'][(0, 0)],
            'addr_start': 0x8380,
            'type': 1
        }
    ]

    compare.add_ref_data(data_1)


if __name__ == '__main__':
    case_file_name = 'LeNet_000'

    handler = LeNetDataHandler()
    data = generate_data(handler)
    compare = ResultCompareWithClockSpecificSimulator(data_file_name=case_file_name, save_ref_data_en=True,
                                                      phase_en=None, print_matched=True, step=0)
    check(data, compare=compare, empty_offset=1)
    compare.run()
    compare.show_result()
