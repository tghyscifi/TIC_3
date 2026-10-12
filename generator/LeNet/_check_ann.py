import os
base = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\simulator\Out_files'

for d, expected in [('LeNet5_0_00001', 0), ('LeNet5_2_23441', 2)]:
    f = os.path.join(base, d, 'output', 'output_1@7.txt')
    with open(f) as fp:
        lines = [l.strip() for l in fp.readlines() if l.strip()]
    
    scores = []
    for l in lines[4:14]:
        v = int(l, 16)
        if v >= 0x80000000:
            v -= 0x100000000
        scores.append(v)
    
    pred = max(range(10), key=lambda i: scores[i])
    ok = 'OK' if pred == expected else 'FAIL'
    print(f'{d}: scores={scores}, pred={pred}, expected={expected}, correct={ok}')
