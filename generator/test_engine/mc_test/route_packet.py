# !/usr/bin/env python
# coding: utf-8

'''
RoutePacket 描述一个路由包，期内部的path记录如下信息：
[node1_id, node2_id, ... , noden_id]
load 记录该路由包大小
根据这些部分部分生成路由包: packages
'''


class RoutePacket(object):
    __slots__ = ('path', 'load')

    def __init__(self):
        self.path = []
        self.load = 0

    @property
    def hop(self):
        return len(self.path)

    @property
    def source(self):
        return self.path[0] if len(self.path) > 0 else None

    @property
    def destination(self):
        return self.path[len(self.path) - 1] if len(self.path) > 0 else None

    def add_node(self, node_id):
        self.path.append(node_id)

    def __str__(self):
        return 'Path: ' + str(self.path) + ' Load: ' + str(self.load)