import numpy as np


def hex_to_string(number, width=2):
    number = np.bitwise_and(number, (2**4)**width-1)
    num_string = hex(number).replace('0x', '')
    zero_string = '0' * (width - len(num_string))
    num_string = zero_string + num_string
    return num_string


if __name__ == '__main__':
    x = 100
    print(hex_to_string(x))
