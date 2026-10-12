#!/usr/bin/env python
# coding: utf-8

'''
Relay 类模式中继方案
'''

from .route_packet import RoutePacket

class Relay(object):
    def __init__(self):
        super().__init__()

    def relay(self, topology):
        packets = {}
        for source, destination in topology:
            packet = RoutePacket()
            packet.add_node(source)
            packet.add_node(destination)
            packets[source] = packet
        return packets
