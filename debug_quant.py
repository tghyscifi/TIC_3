import sys, torch
sys.path.insert(0, '.')
from generator.LeNet.lenet_model.lenet import LeNet

x = torch.randn(1, 1, 28, 28).clamp(-1, 1)
print(f'Input range: [{x.min().item():.4f}, {x.max().item():.4f}]')

# cut_en=True model (hardware inference mode)
model_q = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)
d = torch.load('generator/LeNet/lenet_model/lenet_mnist_final_99.3.pth', map_location='cpu')
if d.get('state_dict') is not None: d = d['state_dict']
model_q.load_state_dict(d, strict=True)
model_q.eval()
with torch.no_grad():
    y_q = model_q(x)
print(f'cut_en=True  output: {y_q[0].tolist()}')
print(f'cut_en=True  argmax: {y_q.argmax(1).item()}')

# cut_en=False model with dequantized weights
model_f = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=False, aware=False)
d2 = {k: v.clone() for k, v in d.items()}
for k in list(d2.keys()):
    if 'weight' in k: d2[k] = d2[k].float() / 128.0
    elif 'bias' in k: d2[k] = d2[k].float() / (128.0*128.0)
missing, unexp = model_f.load_state_dict(d2, strict=True)
model_f.eval()
with torch.no_grad():
    y_f = model_f(x)
print(f'cut_en=False dequant output: {y_f[0].tolist()}')
print(f'cut_en=False dequant argmax: {y_f.argmax(1).item()}')

# Compare probability distributions
print()
print(f'Probs comparison:')
print(f'  quant: {[f"{p:.4f}" for p in torch.softmax(y_q,1)[0].tolist()]}')
print(f'  float: {[f"{p:.4f}" for p in torch.softmax(y_f,1)[0].tolist()]}')
print(f'  argmax match: {y_q.argmax(1).item() == y_f.argmax(1).item()}')
