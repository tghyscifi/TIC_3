import os
import sys
import re
import subprocess
import numpy as np

SNNLENET_PATH = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\snnlenet.py'
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
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    raw_hex = ''.join(lines)
    scores = []
    idx = 0
    while len(scores) < 10:
        if idx + 8 > len(raw_hex):
            break
        word_hex = raw_hex[idx:idx+8]
        bytes_rev = bytes.fromhex(word_hex)
        for b in reversed(bytes_rev):
            val = b if b < 128 else b - 256
            scores.append(val)
        idx += 8
    scores = scores[:10]
    pred = int(np.argmax(scores))
    exp_s = np.exp(np.array(scores, dtype=np.float64))
    softmax = exp_s / exp_s.sum()
    conf = softmax[pred] * 100
    return scores, pred, conf, softmax

def modify_and_run(label, img_id):
    case_name = f'SNN_LeNet3_{label}_{img_id}'
    img_path = f'E:\\\\Tsinghua\\\\tianji\\\\package\\\\package\\\\mnist_train_jpg\\\\{label}_{img_id}.jpg'

    with open(SNNLENET_PATH, 'r', encoding='utf-8') as f:
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
    # 确保双反斜杠
    content = content.replace(f"r'{img_path}'", f"r'{img_path}'")

    with open(SNNLENET_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    result = subprocess.run(
        [sys.executable, SNNLENET_PATH],
        cwd=r'E:\Tsinghua\tianji\tianjic_x1_software_0814',
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f'  FAILED: {result.stderr[:500]}')
        return None

    out_file = os.path.join(OUT_DIR, case_name, 'output', 'output_1@7.txt')
    if not os.path.exists(out_file):
        print(f'  Output not found: {out_file}')
        return None

    return parse_output(out_file)


print('=' * 80)
print('SNN LeNet (LIF at fc2) Hardware Simulation: 10 images (0-9)')
print('=' * 80)

results = []
for label, img_id in images:
    img_name = f'{label}_{img_id}.jpg'
    print(f'\n[{label}] {img_name} ...', end=' ', flush=True)
    parsed = modify_and_run(label, img_id)
    if parsed is None:
        print('SKIPPED (error)')
        continue
    scores, pred, conf, softmax = parsed
    correct = '✓' if pred == int(label) else '✗'
    print(f'pred={pred} {correct}, conf={conf:.1f}%')
    print(f'  scores: {scores}')
    print(f'  probs:  {[f"{p*100:.1f}%" for p in softmax]}')
    results.append((int(label), pred, conf, scores))

print('\n' + '=' * 80)
print('Summary:')
print(f'{"Label":>5} {"Pred":>5} {"Conf":>8} {"Correct":>8}')
print('-' * 40)
correct_count = 0
for label, pred, conf, scores in results:
    ok = '✓' if label == pred else '✗'
    if label == pred:
        correct_count += 1
    print(f'{label:>5} {pred:>5} {conf:>7.1f}% {ok:>8}')
print('-' * 40)
print(f'Accuracy: {correct_count}/{len(results)} = {correct_count/len(results)*100:.1f}%')
