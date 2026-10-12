import numpy as np

def random_array_32(size, min=-2147483648, max=2147483648):
    array_32 = np.random.randint(min, max, size, dtype='int32')
    return array_32

def random_array_28(size, min=-134217728, max=134217728):
    array_28 = np.random.randint(min, max, size, dtype='int32')
    return array_28

def random_array_8(size, min=-128, max=127):
    array_8 = np.random.randint(min, max, size, dtype='int8')
    return array_8

def random_array_u8(size, min=0, max=256):
    array_u8 = np.random.randint(min, max, size, dtype='uint8')
    return array_u8

def random_array_2(size, min=-1, max=2):
    array_2 = np.random.randint(min, max, size, dtype='int8')
    return array_2

def random_array_u4(size, min=0, max=16):
    array_u4 = np.random.randint(min, max, size, dtype='int8')
    return array_u4

def random_constant_32(min_val=-2147483648, max_val=2147483648):
    constant_32 = np.random.randint(min_val, max_val, size=None, dtype='int32')
    return constant_32

def random_constant_28(min_val=-134217728, max_val=134217728):
    constant_32 = np.random.randint(min_val, max_val, size=None, dtype='int32')
    return constant_32

def random_constant_u8(min_val=0, max_val=256):
    constant_u8 = np.random.randint(min_val, max_val, size=None, dtype='uint8')
    return constant_u8

def random_constant_8(min_val=-128, max_val=127):
    constant_8 = np.random.randint(min_val, max_val, size=None, dtype='int8')
    return constant_8

def random_constant_a(min_val=-256, max_val=256):
    constant_a = np.random.randint(min_val, max_val, size=None, dtype='int32')
    return constant_a
