import sys, os, re, torch, numpy as np
from PIL import Image
sys.path.insert(0, '.')
from generator.LeNet.lenet_model.lenet import LeNet

# Load model with dequantized weights
model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=False, aware=False)
d = torch.load('generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth', map_location='cpu')
if d.get('state_dict') is not None: d = d['state_dict']
for k in list(d.keys()):
    if 'weight' in k: d[k] = d[k].float() / 128.0
    elif 'bias' in k: d[k] = d[k].float() / (128.0*128.0)
model.load_state_dict(d, strict=True)
model.eval()

# Load reference model with cut_en=True
model_q = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)
dq = torch.load('generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth', map_location='cpu')
if dq.get('state_dict') is not None: dq = dq['state_dict']
model_q.load_state_dict(dq, strict=True)
model_q.eval()

# Test on random MNIST JPEGs
data_dir = r'E:\Tsinghua\tianji\package\package\mnist_train_jpg'
pattern = re.compile(r'^(\d+)_(\d+)\.jpg$')
samples = []
for fname in os.listdir(data_dir):
    m = pattern.match(fname)
    if m: samples.append((os.path.join(data_dir, fname), int(m.group(1))))

rng = np.random.RandomState(0)
rng.shuffle(samples)
samples = samples[:1000]

correct_float = 0
correct_quant = 0
total = 0

for path, label in samples:
    img = Image.open(path).convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    t = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0) / 255.0 * 2.0 - 1.0
    with torch.no_grad():
        y_f = model(t)
        y_q = model_q(t)
    if y_f.argmax(1).item() == label: correct_float += 1
    if y_q.argmax(1).item() == label: correct_quant += 1
    total += 1
    if total % 200 == 0:
        print(f'  {total}/1000: float_acc={100*correct_float/total:.1f}%  quant_acc={100*correct_quant/total:.1f}%')

print(f'\nFinal: float_acc={100*correct_float/total:.1f}%  quant_acc={100*correct_quant/total:.1f}%')
