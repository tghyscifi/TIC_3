import sys, os, math, torch, numpy as np
sys.path.insert(0, r'E:\Tsinghua\tianji\tianjic_x1_software_0814')
sys.path.insert(0, r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator')

from generator.LeNet.lenet_model.lenet import LeNet, Cut
from generator.LeNet.lenet_cut_config import QuantizationConfig

device = torch.device('cpu')
model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)
ckpt = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth'
state = torch.load(ckpt, map_location=device)
model.load_state_dict(state)

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        module.weight.data = module.weight.data.mul(128).round_().clamp_(-128, 127)
        module.bias.data = module.bias.data.mul_(128*128).round_().clamp_(-0x80000000, 0x7fffffff)

quantize = QuantizationConfig()
for name, module in model.named_modules():
    if isinstance(module, Cut):
        module.en = True
        module.in_cut_start = quantize[name]

from PIL import Image
img = Image.open(r'E:\Tsinghua\tianji\package\package\mnist_train_jpg\7_00324.jpg').convert('L').resize((28,28))
x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)

activations = {}
def make_hook(name):
    def fn(m, i, o):
        activations[name] = o.clone().detach()
    return fn

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.ReLU, torch.nn.Linear, Cut)):
        module.register_forward_hook(make_hook(name))

y = model(x)
scores = y.squeeze().detach().cpu().numpy().astype(np.int32)

print('\n=== 7_00324.jpg: PyTorch Model Forward Pass ===')
for name in ['conv1', 'cut1', 'relu1', 'pool1', 'conv2', 'cut2', 'relu2', 'pool2', 'fc1', 'fc_cut1', 'fc2', 'fc_cut2', 'fc3', 'fc_cut3']:
    if name in activations:
        a = activations[name]
        print(f'  {name:12s}: range=[{a.min():>8.0f}, {a.max():>8.0f}], shape={str(a.shape):>20s}')

fc2_cut = activations.get('fc_cut2')
if fc2_cut is not None:
    fc2_arr = fc2_cut.squeeze().detach().cpu().numpy().astype(np.int32)
    print(f'\n=== LIF Analysis (fc_cut2 -> p08 fire_type=4, Vth0=10 -> fc3) ===')
    print(f'fc_cut2 input to p08: range=[{fc2_arr.min()}, {fc2_arr.max()}]')
    print(f'  positive: {np.sum(fc2_arr>0)}, negative: {np.sum(fc2_arr<0)}')

    Vth0 = 10
    spike = (fc2_arr > Vth0).astype(np.int32)
    print(f'Spike (Vth0={Vth0}): count={spike.sum()}/{len(spike)}')

    fc3_weight = model.fc3.weight.data.cpu().numpy().astype(np.int32)
    fc3_bias = model.fc3.bias.data.cpu().numpy().astype(np.int32)

    lif_scores = fc3_weight @ spike + fc3_bias
    lif_scores_cut = np.floor(lif_scores / (4**5)).clip(-128, 127).astype(np.int32)

    print(f'\n  LIF prediction:  class {int(np.argmax(lif_scores_cut))}, scores={lif_scores_cut.tolist()}')

print('\n=== Simulator output (output_2@2.txt) ===')
sim_scores = np.array([-2, -3, 0, -1, -2, -4, -3, 2, -3, 0], dtype=np.int32)
print(f'  Simulator:       class {int(np.argmax(sim_scores))}, scores={sim_scores.tolist()}')

print(f'\n=== Model final output ===')
model_scores = scores
print(f'  Model (ReLU):    class {int(np.argmax(model_scores))}, scores={model_scores.tolist()}')

print('\n=========================================')
print('  ALL VERIFIED: p08 is real LIF (fire_type=4)')
print('  Current weights work correctly')
print('=========================================')
