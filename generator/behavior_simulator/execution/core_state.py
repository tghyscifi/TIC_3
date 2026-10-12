#!/usr/bin/env python
# coding: utf-8

'''
CoreState 枚举不同的Core状态
'''

from enum import Enum


class CoreState(Enum):
    READY = 0
    LOCKED = 1
    FINISH = 2
