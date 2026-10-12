#!/usr/bin/env python
# coding: utf-8

from generator.test_engine.mc_test.route_pattern_builder import RoutePatternBuilder
from generator.test_engine.mc_test.multi_core import MultiCore


def test_mc():
    route_pattern = RoutePatternBuilder.build_bit_complement(3, 3)
    for source, packet in route_pattern:
        print(packet)

    multi_core = MultiCore(route_pattern)
    map_config = multi_core.get_map_config()

test_mc()
