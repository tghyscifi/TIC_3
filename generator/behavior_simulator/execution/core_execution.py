#!/usr/bin/env python
# coding: utf-8

'''
CoreExecution类描述Core层面的执行模型
'''

from generator.behavior_simulator.execution.core_state import CoreState


class CoreExecution(object):
    def __init__(self):
        super().__init__()

        # 一个Phase core的执行步骤
        # [(func, func_args)]
        self._steps = []
        self._pointer = 0

        self.state = CoreState.READY
        self.locked = False      # TODO

    def clear(self):
        self._pointer = 0
        self._steps.clear()
        self.state = CoreState.READY

    def add_step(self, function, arguments=()):
        self._steps.append((function, arguments))

    def execute(self):
        if self.state == CoreState.LOCKED:
            return False
        while self._pointer < len(self._steps):
            func, args = self._steps[self._pointer]
            state = func(*args)
            if state is False:      # block
                self.state = CoreState.LOCKED
                self._pointer += 1
                self.locked = True
                return False
            self._pointer += 1
        self._pointer = 0
        self.locked = False
        self.state = CoreState.READY
        return True
