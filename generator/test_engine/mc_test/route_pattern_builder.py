#!/usr/bin/env python
# coding: utf-8

'''
RoutePatternBuilder 利用建造者模式构造各种各样的RoutePattern
'''

from enum import Enum

from generator.test_engine.mc_test.coder import Coder
from generator.test_engine.mc_test.topology_lib import TopologyLib
from generator.test_engine.mc_test.relay import Relay
from generator.test_engine.mc_test.flow import Flow, FlowMode
from generator.test_engine.mc_test.route_pattern import RoutePattern
from generator.test_engine.mc_test.circle_break_decorator import CircleBreakDecorator


class TopologyRequirement(Enum):
    NO_CIRCLE = 0


class RoutePatternBuilder(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def build_coder(total_x, total_y, x_base, y_base, global_base):
        if x_base is None:
            x_base = total_x
        if y_base is None:
            y_base = total_y
        coder = Coder(total_x, total_y, x_base, y_base)
        if global_base is not None:
            coder.global_base = global_base
        return coder

    @staticmethod
    def build_bit_complement(total_x, total_y, x_base=None, y_base=None, global_base=None, requirements=None):
        coder = RoutePatternBuilder.build_coder(
            total_x, total_y, x_base, y_base, global_base)
        topology = TopologyLib.bit_complement(coder)
        topology = RoutePatternBuilder.add_topology_requirements(
            topology, requirements)
        relay = Relay()
        flow = Flow(FlowMode.Constant)

        route_pattern = RoutePattern(topology, relay, flow)
        route_pattern.cal_packages()
        return route_pattern

    @staticmethod
    def add_topology_requirements(topology, requirements):
        if requirements is None:
            return topology

        from collections import Iterable
        if not isinstance(requirements, Iterable):
            requirements = [requirements]
        for require in requirements:
            if require == TopologyRequirement.NO_CIRCLE:
                topology = CircleBreakDecorator(topology)

        return topology
