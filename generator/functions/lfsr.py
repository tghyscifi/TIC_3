import numpy as np


class lfsr(object):
    def __init__(self, seed):
        self.lfsr = int(seed)
        if self.lfsr < 0:
            self.lfsr += 2**32

    def update(self):
        bit = ((self.lfsr >> 0) ^ (self.lfsr >> 1) ^ (self.lfsr >> 2) ^ (self.lfsr >> 3) ^ (self.lfsr >> 5) ^
              (self.lfsr >> 7)) & 1
        self.lfsr = (bit << 31) | (self.lfsr >> 1)
        # bit = ((self.lfsr >> 7) ^ (self.lfsr >> 2) ^ (self.lfsr >> 1) ^ self.lfsr) & 0x1
        # self.lfsr = (bit << 23) | (self.lfsr >> 1)
        if self.lfsr > 2**31 - 1:
            lfsr = self.lfsr - 2**32
        else:
            lfsr = self.lfsr
        return lfsr



if __name__ == '__main__':
    lfsr_num = lfsr(0)
    while  lfsr_num != 500:
        print(lfsr_num.update())


