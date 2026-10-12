#!/usr/bin/env python
# coding: utf-8

'''
ChipExecution类描述chip层面的执行模型，其中
    lock_key_pool 管理全局执行锁
    state 表示当前core的状态
'''

import random

from generator.behavior_simulator.execution.lock_key_pool import LockKeyPool
from generator.behavior_simulator.execution.core_state import CoreState


class ChipExecution(object):
    class DeadLockException(RuntimeError):
        pass

    def __init__(self, cores):
        super().__init__()

        # self._state = {}  # 状态dict，包含所有core的状态
        self._execution_order = []  # 执行顺序list
        self._pointer = 0

        self._lock_key_pool = LockKeyPool()

        # TODO:先暂时这么做
        self._cores = cores

    def get_lock_key_pool(self):
        return self._lock_key_pool

    def init_core_state(self, core_ids):
        self._pointer = 0
        self._lock_key_pool.clear()
        self._execution_order = [id for id in core_ids]

    def add_core(self, core_id, state=CoreState.READY):
        if core_id not in self._execution_order:
            self._execution_order.append(core_id)

    def sequential_execute(self, start_core=None):
        self._deadlock_check()  # 死锁判断

        if start_core is not None:
            try:
                self._pointer = self._execution_order.index(
                    start_core)  # 执行顺序从定义的start_core开始，.index返回core的下标
            except ValueError:  # 如果出现ValueError则跳过
                pass

        current_core = self._execution_order[self._pointer]
        while self._cores[current_core].state != CoreState.READY:
            try:
                self._next_pointer()
            except StopIteration:
                return
            current_core = self._execution_order[self._pointer]
        while True:
            yield self._execution_order[self._pointer]
            try:
                self._next_pointer()
            except StopIteration:
                return

    def _next_pointer(self):
        start_pointer = self._pointer
        while True:
            self._pointer += 1
            if self._pointer >= len(self._execution_order):
                self._pointer = 0
            next_core = self._execution_order[self._pointer]
            # 全finish的情况的判断
            if self._pointer == start_pointer:
                self._deadlock_check()  # 死锁判断
                if self._cores[next_core].state == CoreState.FINISH:
                    raise StopIteration()
            # Find next pointer
            if self._cores[next_core].state == CoreState.READY:
                return

    def _deadlock_check(self):
        state_set = {core.state for core in self._cores.values()}
        locked_core = []
        for core in self._cores.values():
            if core.state == CoreState.LOCKED:
                locked_core.append(core.id)
        if CoreState.FINISH in state_set:
            state_set.remove(CoreState.FINISH)
        if len(state_set) == 1:
            for state in state_set:
                if state == CoreState.LOCKED:
                    raise ChipExecution.DeadLockException(
                        str("{0} dead lock! Please check router prim list of core{0},\n" +
                            "send_destin_core_grp,recv_source_core_grp,instant_prim_request,instant_request_back").format(locked_core[0]))

    def random_execute(self):
        self._deadlock_check()
        while True:
            ready_set = tuple(core for core in self._execution_order
                              if self._cores[core].state == CoreState.READY)
            if not ready_set:
                return
            yield ready_set[random.randint(0, len(ready_set) - 1)]

    # Lock管理相关：
    def add_lock(self, core_id, lock_id):
        if not self._lock_key_pool.add_lock(core_id, lock_id):
            self._cores[core_id].state = CoreState.LOCKED

    def add_key(self, core_id, key_id):
        if self._lock_key_pool.add_key(core_id, key_id):
            if self._cores[core_id].state == CoreState.LOCKED:
                self._cores[core_id].state = CoreState.READY

    def locked(self, core_id):
        return self._lock_key_pool.locked(core_id)
