#!/usr/bin/env python
# coding: utf-8

'''
CircleBreakDecorator消除Topology里面的所有环
'''

from generator.test_engine.mc_test.topology import Topology
from generator.test_engine.mc_test.topology_decorator import TopologyDecorator


class CircleBreakDecorator(TopologyDecorator):
    def __init__(self, topology):
        super().__init__(topology)

        self.break_circle()

    def break_circle(self):
        pass
