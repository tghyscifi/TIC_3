import os
base = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\simulator\Out_files'
img_map = {
    '0':'00001','1':'00003','2':'00005','3':'00007','4':'00002',
    '5':'00000','6':'00013','7':'00015','8':'00017','9':'00004'
}
for label in ['0','1','2','3','4','5','6','7','8','9']:
    f = os.path.join(base, f'SNN_LeNet3_{label}_{img_map[label]}', 'output', 'output_2@2.txt')
    with open(f) as fp:
        lines = [l.strip() for l in fp.readlines() if l.strip()]
    raw = ''.join(lines)
    hex_fc3 = raw[32:52]
    scores = []
    for i in range(0, 20, 2):
        b = int(hex_fc3[i:i+2], 16)
        scores.append(b if b < 128 else b - 256)
    pred = scores.index(max(scores))
    print(f'label={label}: scores={scores}, pred={pred}, correct={"OK" if str(pred)==label else "FAIL"}')
