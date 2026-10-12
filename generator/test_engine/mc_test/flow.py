#!/usr/bin/env python
# coding: utf-8

'''
Flow 类模式中继方案
'''

from enum import Enum


class FlowMode(Enum):
    Constant = 0


class Flow(object):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.constant_value = 1

    def inject_flow(self, packets):
        if self.mode == FlowMode.Constant:
            for packet in packets.values():
                packet.load = self.constant_value
