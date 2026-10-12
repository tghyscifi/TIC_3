#!/usr/bin/env python
# coding: utf-8

import json
from typing import Dict, Tuple

from generator.behavior_simulator.chip import Chip
from generator.util import Path


class ChipArray(object):
    def __init__(self):
        self.sim_clock = -1
        self._chips = {}   # type: Dict[Tuple, Chip]

    def simulate(self, tb_name):
        for chip in self._chips.values():
            chip.print_case_spec(tb_name)
            chip.simulate(tb_name)

    def add_chip(self, chip_id, chip):
        self._chips[chip_id] = chip

    def get_chip(self, chip_id):
        return self._chips.get(chip_id)

    def set_sim_clock(self, clock):
        if clock is None:
            return
        self.sim_clock = clock

    def get_chips(self):
        return self._chips

    def __contains__(self, chip_id):
        return chip_id in self._chips

    def config_json_output(self, tb_name: str):
        json_config = {
            "sim_clock": self.sim_clock,
            "ChipArray":
                []
        }

        for chip in self._chips.values():
            chip_jsons = chip.config_json_output(tb_name)
            json_config["ChipArray"].extend(chip_jsons)

        # json.dump(json_config, open(Path.json_config_path(tb_name), "w"),
        #           ensure_ascii=False, indent=4)
        return json_config
