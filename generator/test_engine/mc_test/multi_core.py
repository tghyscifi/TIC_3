#!/usr/bin/env python
# coding: utf-8

'''
MultiCore 类负责将RouterPattern映射为Multi Core结构，并准换为MapConfig
'''

from generator.code_generator import MapConfig

class MultiCore(object):
    def __init__(self, pattern):
        super().__init__()

        self._pattern = pattern
        self._core_condig = {}         # {core_id, prim_list}

        self._init_cores()

    def _init_cores(self):
        pass

    def get_map_config(self):
        return MapConfig({})

    def get_core_config(self, x, y):
        pass

    def set_core_config(self, x, y, core_config):
        pass
