import os
import time

from generator.util import Path

def pack_cases(prefix, start, number):
    failed = []
    with open(Path.TEMP_DIR + 'failed.txt', 'r') as f:
        failed = [line.replace('\n', '') for line in f.readlines()]

    os.chdir(Path.hardware_out_dir())
    print(os.path.abspath(os.curdir))
    pre = str(prefix)

    for i in range(number):
        n = start + i
        zeros = max(3 - len(str(n)), 0) * '0'
        cur = pre + zeros + str(n)
        if cur in failed:
            print('Skip case: ', cur)
            continue
        print('Packing: ', cur)
        if not os.path.exists(cur):
            continue
        os.chdir(cur)
        # os.system("del "+cur+".rar")
        time.sleep(0.3)
        os.system("start winrar a " + cur + ".rar")
        os.chdir("..")

    for i in range(number):
        n = start + i
        zeros = max(3 - len(str(n)), 0) * '0'
        cur = pre + zeros + str(n)
        if cur in failed:
            print('Skip case: ', cur)
            continue
        if not os.path.exists(cur):
            continue
        print('Moving: ', cur)
        os.chdir(cur)
        os.system("move " + cur + ".rar ../" + prefix + "/")
        os.chdir("..")

pack_cases('C01', 5, 288)