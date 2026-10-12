import sys, os, re, torch, numpy as np
from PIL import Image
sys.path.insert(0, '.')
from generator.LeNet.lenet_model.lenet import LeNet

# Use correct MNIST preprocessing: Resize(32) → ToTensor → Normalize(0.1307, 0.3081)
# But our LeNet expects 28x28 input, so use Resize(28) instead
def load_img_preproc(path):
    img = Image.open(path).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
    arr = (arr - 0.1307) / 0.3081  # MNIST normalization
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

def load_img_m1(path):
    img = Image.open(path).convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    t = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0) / 255.0 * 2.0 - 1.0
    return t

# Load dequantized model
model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=False, aware=False)
d = torch.load('generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth', map_location='cpu')
if d.get('state_dict') is not None: d = d['state_dict']
for k in list(d.keys()):
    if 'weight' in k: d[k] = d[k].float() / 128.0
    elif 'bias' in k: d[k] = d[k].float() / (128.0*128.0)
model.load_state_dict(d, strict=True)
model.eval()

data_dir = r'E:\Tsinghua\tianji\package\package\mnist_train_jpg'
pattern = re.compile(r'^(\d+)_(\d+)\.jpg$')
samples = []
for fname in os.listdir(data_dir):
    m = pattern.match(fname)
    if m: samples.append((os.path.join(data_dir, fname), int(m.group(1))))

rng = np.random.RandomState(0)
rng.shuffle(samples)
samples = samples[:500]

for label, fn in [('[-1,1] norm', load_img_m1), ('MNIST norm', load_img_preproc)]:
    correct = 0
    for path, lbl in samples:
        t = fn(path)
        with torch.no_grad():
            y = model(t)
        if y.argmax(1).item() == lbl: correct += 1
    print(f'{label}: {100*correct/len(samples):.1f}%')
