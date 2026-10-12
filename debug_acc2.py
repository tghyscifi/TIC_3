import sys, os, re, torch, numpy as np
from PIL import Image
sys.path.insert(0, '.')
from generator.LeNet.lenet_model.lenet import LeNet

def load_img(path):
    img = Image.open(path).convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

# Float model (dequantized)
model_f = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=False, aware=False)
d = torch.load('generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth', map_location='cpu')
if d.get('state_dict') is not None: d = d['state_dict']
for k in list(d.keys()):
    if 'weight' in k: d[k] = d[k].float() / 128.0
    elif 'bias' in k: d[k] = d[k].float() / (128.0*128.0)
model_f.load_state_dict(d, strict=True)
model_f.eval()

# Quant model (hardware inference)
model_q = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)
dq = torch.load('generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth', map_location='cpu')
if dq.get('state_dict') is not None: dq = dq['state_dict']
model_q.load_state_dict(dq, strict=True)
model_q.eval()

# MNIST test JPEGs (standard MNIST test set in JPEG format)
test_dir = r'E:\Tsinghua\tianji\package\package\mnist_test_jpg'
if not os.path.exists(test_dir):
    # fallback to training set subset
    test_dir = r'E:\Tsinghua\tianji\package\package\mnist_train_jpg'

pattern = re.compile(r'^(\d+)_(\d+)\.jpg$')
samples = []
for fname in os.listdir(test_dir):
    m = pattern.match(fname)
    if m and int(m.group(2)) < 5:  # take first 5 of each class for speed
        samples.append((os.path.join(test_dir, fname), int(m.group(1))))

print(f'Testing on {len(samples)} samples...')
correct_f, correct_q = 0, 0
for path, label in samples:
    t = load_img(path)
    with torch.no_grad():
        y_f = model_f(t)
        y_q = model_q(t)
    if y_f.argmax(1).item() == label: correct_f += 1
    if y_q.argmax(1).item() == label: correct_q += 1

print(f'Float (dequant) model: {100*correct_f/len(samples):.1f}%')
print(f'Quant (cut_en=True) model: {100*correct_q/len(samples):.1f}%')

# Check if they agree
agree = 0
for path, label in samples:
    t = load_img(path)
    with torch.no_grad():
        p_f = model_f(t).argmax(1).item()
        p_q = model_q(t).argmax(1).item()
    if p_f == p_q: agree += 1
print(f'Models agree: {100*agree/len(samples):.1f}%')
