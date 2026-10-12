from itertools import product
import sys
import os
from typing import Sequence

sys.path.append(os.getcwd())
from generator.SoundSNN.g1_data import generate_g1_data
from generator.SoundSNN.data_handler import SNNDataHandler
from generator.mapping_utils.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator


def check(ref_data, compare, size_y, size_x, chip=(0, 0), offset=0):
    data = {}
    offset += 3

    data[offset + 0] = {}
    for core_y, core_x in product(range(*size_y), range(*size_x)):
        data[offset + 0][chip, (core_x, core_y)] = [{
            'data': ref_data['fc1'][0]['output'][(core_x - size_x[0], core_y - size_y[0])],
            'addr_start': 0x6300 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data)


if __name__ == '__main__':
    handler = SNNDataHandler()
    data = generate_g1_data(handler, size_y=1, size_x=1, sequence_length=39)

    compare = ResultCompareWithClockSpecificSimulator(
        data_file_name='SNN_1', save_ref_data_en=True,
        print_matched=True, step=0)
    # compare = ResultCompare(data_2_file_name='Obstacle_1', save_data_1_en=True)
    check(data, compare, size_y=(0, 1), size_x=(0, 1), chip=(0, 0))
    compare.run()
    compare.show_result()
