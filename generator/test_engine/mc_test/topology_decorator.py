#!/usr/bin/env python
# coding: utf-8

'''
采用修饰器模式，为Topology增加新的功能
TopologyDecorator为所有修饰器的基类
'''

from generator.test_engine.mc_test.topology import Topology


class TopologyDecorator(Topology):
    def __init__(self, topology):
        self.coder = topology.coder
        self._graph = topology._graph
