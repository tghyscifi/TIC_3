import os
from generator.SoundSNN.data_handler import SNNDataHandler
from generator.SoundSNN.g1_data import generate_g1_data
import numpy as np
from generator.mapping_utils.map_config_gen import MapConfigGen
from itertools import product
from generator.SoundSNN.snn_config import SNNConfig
from generator.SoundSNN.g1_1core import gen_1_map_config


def main():
    chip = (0, 0)
    phase_offset = 0
    delay = (0,) * 9

    clock_in_phase = 150_000

    phase = np.zeros(50).astype(int)
    # 39~49 表示组间数据传输的Phase

    phase[phase_offset + 0] = 1

    handler = SNNDataHandler()
    data = generate_g1_data(handler, size_y=1, size_x=1, sequence_length=39)

    map_config = MapConfigGen()

    # Bias of ST Net
    empty_config = {
        'sim_clock': None,
        ((0, 0), 0): {
            0: {
                'clock': clock_in_phase,
                'mode': 1,
            }
        }
    }
    for i in range(18):
        map_config.add_config(empty_config, core_offset=(0, 0), clock_in_phase=None, phase_adaptive=True)

    snn_config = SNNConfig()
    config = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=1, size_y=1, data=data,
                              config=snn_config, in_data_en=False, out_data_en=False, chip=chip)
    MapConfigGen.add_router_info(map_config=config)

    map_config.add_config(config, core_offset=(1, 9))

    # delete empty map_config
    for i in range(18):
        map_config.map_config[((0, 0), 0)].pop(i)

    return map_config.map_config


