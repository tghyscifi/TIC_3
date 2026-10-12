import os
import sys
import re
import subprocess
import numpy as np

LENET_PATH = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\lenet.py'
IMG_DIR = r'E:\Tsinghua\tianji\package\package\mnist_train_jpg'
OUT_DIR = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\simulator\Out_files'

images = [
    ('0', '00001'),
    ('1', '00003'),
    ('2', '00005'),
    ('3', '00007'),
    ('4', '00002'),
    ('5', '00000'),
    ('6', '00013'),
    ('7', '00015'),
    ('8', '00017'),
    ('9', '00004'),
]

def parse_output(filepath):
    """Parse ANN LeNet output_1@7.txt: lines 5-14 are 10 int32 fc3 scores"""
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) < 14:
        return None, None, None, None
    scores = []
    for l in lines[4:14]:
        v = int(l, 16)
        if v >= 0x80000000:
            v -= 0x100000000
        scores.append(v)
    pred = int(np.argmax(scores))
    scores_arr = np.array(scores, dtype=np.float64)
    scores_arr -= scores_arr.max()
    exp_s = np.exp(scores_arr)
    softmax = exp_s / exp_s.sum()
    conf = softmax[pred] * 100
    return scores, pred, conf, softmax

def modify_and_run(label, img_id):
    case_name = f'LeNet5_{label}_{img_id}'
    img_path = f'E:\\\\Tsinghua\\\\tianji\\\\package\\\\package\\\\mnist_train_jpg\\\\{label}_{img_id}.jpg'

    with open(LENET_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(
        r"case_file_name\s*=\s*'[^']*'",
        f"case_file_name = '{case_name}'",
        content
    )
    content = re.sub(
        r"input_image_path\s*=\s*r'[^']*'",
        f"input_image_path=r'{img_path}'",
        content
    )

    with open(LENET_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'  Running simulator...', end=' ', flush=True)
    result = subprocess.run(
        [sys.executable, LENET_PATH],
        cwd=r'E:\Tsinghua\tianji\tianjic_x1_software_0814',
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f'FAILED')
        print(f'  stderr: {result.stderr[:500]}')
        return None

    out_file = os.path.join(OUT_DIR, case_name, 'output', 'output_1@7.txt')
    if not os.path.exists(out_file):
        print(f'Output not found')
        return None

    return parse_output(out_file)

if __name__ == '__main__':
    print('=' * 70)
    print('ANN LeNet Hardware Simulation: 10 images (0-9)')
    print('=' * 70)

    correct = 0
    for label, img_id in images:
        print(f'[{label}] {label}_{img_id}.jpg ...', end=' ', flush=True)
        result = modify_and_run(label, img_id)
        if result is None or result[0] is None:
            print('  SKIP (error)')
            continue
        scores, pred, conf, softmax = result
        symbol = 'OK' if str(pred) == label else 'FAIL'
        if str(pred) == label:
            correct += 1
        print(f'pred={pred} {symbol}, conf={conf:.1f}%')
        print(f'  scores: {scores}')
        probs = [f'{p:.1f}%' for p in softmax * 100]
        print(f'  probs:  {probs}')

    print('=' * 70)
    print(f'Total: {correct}/10 correct ({correct/10*100:.0f}%)')
    print('=' * 70)
