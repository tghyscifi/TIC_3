import sys, os, re, torch, numpy as np
from PIL import Image
sys.path.insert(0, '.')
from generator.LeNet.lenet_model.lenet import LeNet

# Load dequantized float model
model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=False, aware=False)
d = torch.load('generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth', map_location='cpu')
if d.get('state_dict') is not None: d = d['state_dict']
for k in list(d.keys()):
    if 'weight' in k: d[k] = d[k].float() / 128.0
    elif 'bias' in k: d[k] = d[k].float() / (128.0*128.0)
model.load_state_dict(d, strict=True)
model.eval()

# Test with all files in training set (up to 5000)
data_dir = r'E:\Tsinghua\tianji\package\package\mnist_train_jpg'
pattern = re.compile(r'^(\d+)_(\d+)\.jpg$')
samples = []
for fname in os.listdir(data_dir):
    m = pattern.match(fname)
    if m: samples.append((os.path.join(data_dir, fname), int(m.group(1))))
import random; random.Random(42).shuffle(samples)
samples = samples[:2000]

correct_mnist = 0
correct_m1 = 0
total = 0
for path, label in samples:
    arr = np.array(Image.open(path).convert('L').resize((28,28), Image.Resampling.LANCZOS), dtype=np.float32)
    t_m1 = torch.from_numpy(arr / 255.0 * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)
    t_mnist = torch.from_numpy((arr / 255.0 - 0.1307) / 0.3081).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        p_m1 = model(t_m1).argmax(1).item()
        p_mnist = model(t_mnist).argmax(1).item()
    if p_m1 == label: correct_m1 += 1
    if p_mnist == label: correct_mnist += 1
    total += 1
    if total % 500 == 0:
        print(f'  {total}/{len(samples)}: [-1,1]={100*correct_m1/total:.1f}%  MNIST={100*correct_mnist/total:.1f}%')

print(f'\nFinal ({total} samples):')
print(f'  [-1,1] norm: {100*correct_m1/total:.1f}%')
print(f'  MNIST norm:  {100*correct_mnist/total:.1f}%')
