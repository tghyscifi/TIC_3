#!/usr/bin/env python
# coding: utf-8

'''
LockKeyPool 类负责管理全局核程的锁与钥匙
'''


class LockKeyPool(object):
    def __init__(self):
        # {core_id: {lock_id: lock number}}
        self._lock = {}

        # {core_id: {key_id: key number}}
        self._key = {}

    def clear(self):
        self._lock.clear()
        self._key.clear()

    def add_lock(self, core_id, lock_id):
        if core_id in self._key and lock_id in self._key[core_id]:
            self.consume_key(core_id, lock_id)
            return not self.locked(core_id)

        if not self.locked(core_id):
            self._lock[core_id] = {lock_id: 1}
            return False

        self._lock[core_id][lock_id] = self._lock[core_id].get(lock_id, 0) + 1
        return False

    def consume_key(self, core_id, lock_id):
        key_num = self._key[core_id][lock_id]
        assert key_num > 0
        key_num -= 1
        if key_num == 0:
            self._key[core_id].pop(lock_id)
            if not self._key[core_id]:
                self._key.pop(core_id)
        else:
            self._key[core_id][lock_id] = key_num

    def add_key(self, core_id, key_id):
        if core_id in self._lock and key_id in self._lock[core_id]:
            self.consume_lock(core_id, key_id)
            return not self.locked(core_id)

        if core_id not in self._key:
            self._key[core_id] = {key_id: 1}
            return not self.locked(core_id)

        self._key[core_id][key_id] = self._key[core_id].get(key_id, 0) + 1
        return not self.locked(core_id)

    def consume_lock(self, core_id, key_id):
        lock_num = self._lock[core_id][key_id]
        assert lock_num > 0
        lock_num -= 1
        if lock_num == 0:
            self._lock[core_id].pop(key_id)
            if not self._lock[core_id]:
                self._lock.pop(core_id)
        else:
            self._lock[core_id][key_id] = lock_num

    def locked(self, core_id):
        return core_id in self._lock
