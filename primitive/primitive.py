from enum import Enum


class PrimitiveType(Enum):
    Axon = 0,
    Soma1 = 1,
    Router = 2,
    Soma2 = 3


class Primitive(object):
    def __init__(self):
        # [{
        #   'name':
        #   'start':
        #   'mode':
        #   'data':
        # }]
        self._memory_blocks = []

    @property
    def memory_blocks(self):
        return self._memory_blocks

    @memory_blocks.setter
    def memory_blocks(self, blocks):
        for block in blocks:
            assert isinstance(block, dict)
        self._memory_blocks = blocks

    def init_data(self):
        '''
            初始化所有memory_blocks中的数据块
        '''
        self._memory_blocks = []
        return []
