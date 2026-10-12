import sys
import os

sys.path.append(os.getcwd())


def check(data, compare, empty_offset=0):
    data_1 = {}

    data_1[empty_offset + 0] = {}
    chip = (0, 0)
    for core_x, core_y in [(7, 0)]:
        data_1[empty_offset + 0][chip, (core_x, core_y)] = [{
            'data': data['fca2_output_all'][(core_x - 7, core_y)],
            'addr_start': 0x0000 >> 2,
            'type': 1
        }]

    compare.add_ref_data(data_1)
