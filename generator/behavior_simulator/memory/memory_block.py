from functools import total_ordering
from enum import Enum

class MemroyType(Enum):
    AXON = 0
    SOMA1 = 1
    ROUTER = 2
    SOMA2 = 3

@total_ordering
class MemoryBlock(object):
    def __init__(self, start, length):
        self.start = start
        self.length = length
        self.memory_type = MemroyType.AXON
        self.data = []

    def is_overlap(self, other):
        if self < other:
            return self.length + self.start > other.start
        else:
            return other.length + other.start > self.start

    def __str__(self):
        return 'MemoryBlock(' + str(self.start) + ', ' + str(self.length) + ')'

    def __hash__(self):
        return hash(self.start)

    def __eq__(self, other):
        return (self.start, self.length) == (other.start, other.length)

    def __lt__(self, other):
        return (self.start, self.length) < (other.start, other.length)

    def __contain__(self, number):
        return number >= self.start and number < self.start + self.length
