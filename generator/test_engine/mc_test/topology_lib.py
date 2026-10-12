#!/usr/bin/env python
# coding: utf-8

'''
TopologyLib 类包含各种各样的Premutation生成方式
'''

from generator.test_engine.mc_test.coder import Coder
from generator.test_engine.mc_test.topology import Topology


class TopologyLib(object):
    @staticmethod
    def bit_complement(coder: Coder):
        topology = Topology(coder)
        for sx in range(coder.x):
            for sy in range(coder.y):
                bits, bases = coder.bit(sx, sy)
                i = 1
                for i, bit in enumerate(bits):
                    bits[i] = (bit + 1) % bases[i]
                dx, dy = coder.x_y(bits)
                topology.add_path(sx, sy, dx, dy)
        return topology
