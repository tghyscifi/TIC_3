#!/usr/bin/env python
# coding: utf-8

'''
RoutePattern 描述一个路由模式，包括三个部分：
    Topology: 路由拓扑
    Relay: 路由中继
    Flow: 流量
根据这些部分部分生成路由包: packets
'''

from typing import Dict, Tuple
from generator.test_engine.mc_test.topology import Topology
from generator.test_engine.mc_test.relay import Relay
from generator.test_engine.mc_test.flow import Flow

from .route_packet import RoutePacket


class RoutePattern(object):
    def __init__(self, topology, relay, flow):
        self._topology = topology        # type: Topology
        self._relay = relay        # type: Relay
        self._flow = flow       # type: Flow

        self.packets = {}   # type: Dict[Tuple, RoutePacket]

    def cal_packages(self):
        self.packets = self._relay.relay(self._topology)
        self._flow.inject_flow(self.packets)
    
    def get_packets(self, node_id):
        self.packets.get(node_id, None)

    # 尽量不要调用
    # def get_all_packets(self):
    #     return self.packets

    def __iter__(self):
        for source, packet in self.packets.items():
            yield source, packet
