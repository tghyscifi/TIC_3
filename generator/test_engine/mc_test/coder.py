#!/usr/bin/env python
# coding: utf-8

'''
Coder 类负责一个二维结点阵列的编码
'''


class Coder(object):
    def __init__(self, x, y, x_base, y_base):
        self._x = x
        self._y = y

        self._global_base = None
        self._total_len = 0

        self._x_base = x_base
        self._y_base = y_base
        self._x_digit_len = 0 if x_base is None else len(
            self.convert_to_base(x - 1, x_base))
        self._y_digit_len = 0 if y_base is None else len(
            self.convert_to_base(y - 1, y_base))

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def global_base(self):
        return self._global_base

    @global_base.setter
    def global_base(self, base):
        self._x_base, self._y_base = None, None
        self._global_base = base

        self._total_len = len(self.convert_to_base(self.x*self.y - 1, base))

    @property
    def x_base(self):
        return self._x_base

    @x_base.setter
    def x_base(self, base):
        self._global_base = None
        self._x_base = base

        self._x_digit_len = len(self.convert_to_base(self.x - 1, base))

    @property
    def y_base(self):
        return self._y_base

    @y_base.setter
    def y_base(self, base):
        self._global_base = None
        self._y_base = base

        self._y_digit_len = len(self.convert_to_base(self.y - 1, base))

    def bit(self, x, y):
        if self._global_base is not None:
            bits = self.convert_to_base(x * self._x + y, self._global_base)
            if len(bits) < self._total_len:
                bits = [0] * (self._total_len - len(bits)) + bits
            bit_bases = [self._global_base] * len(bits)
            return bits, bit_bases
        else:
            assert self._x_base is not None
            assert self._y_base is not None
            xs = self.convert_to_base(x, self._x_base)
            ys = self.convert_to_base(y, self._y_base)
            if len(xs) < self._x_digit_len:
                xs = [0] * (self._x_digit_len - len(xs)) + xs
            if len(ys) < self._y_digit_len:
                ys = [0] * (self._y_digit_len - len(ys)) + ys
            bit_bases = [self._y_base] * len(ys) + [self._x_base] * len(xs)
            return ys + xs, bit_bases

    def x_y(self, bits):
        if self._global_base is not None:
            num = self.convert_to_decimal(bits, self._global_base)
            return (num % self._x, num // self._x)
        else:
            assert self._x_base is not None
            assert self._y_base is not None
            assert len(bits) >= self._y_digit_len
            y_bits = bits[:self._y_digit_len]
            x_bits = bits[self._y_digit_len:]
            x = self.convert_to_decimal(x_bits, self._x_base)
            y = self.convert_to_decimal(y_bits, self._y_base)
            return (x, y)

    @staticmethod
    def convert_to_base(number, base):
        # Covert decimal number to base
        result = []
        if number == 0:
            result.append(0)
        while number > 0:
            result.insert(0, number % base)
            number = number // base
        return result

    @staticmethod
    def convert_to_decimal(bits, base):
        num = 0
        weight = 1
        for i in reversed(bits):
            num += i * weight
            weight *= base
        return num
