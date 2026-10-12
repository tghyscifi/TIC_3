#!/usr/bin/env python
# coding: utf-8

'''
Topology 类负责描述一个路由拓扑的映射
'''


class Topology(object):
    def __init__(self, coder):
        super().__init__()

        self.coder = coder
        self._graph = {}

    @property
    def x(self):
        return self.coder.x

    @property
    def y(self):
        return self.coder.y

    def add_path(self, s_x, s_y, d_x, d_y):
        if s_x >= self.x or s_y >= self.y or d_x >= self.x or d_y >= self.y:
            return

        if (s_x, s_y) in self._graph:
            self._graph.append((d_x, d_y))
        else:
            self._graph[(s_x, s_y)] = [(d_x, d_y)]

    def get_pathes_from(self, s_x, s_y):
        self._graph.get((s_x, s_y), [])

    def __iter__(self):
        for source, destination in self._graph.items():
            yield source, destination
