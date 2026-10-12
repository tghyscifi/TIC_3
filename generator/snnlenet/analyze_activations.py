import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.getcwd())

# ============ STE / Cut helpers ============
def ste_round(x):
    return (torch.round(x) - x).detach() + x

def ste_floor(x):
    return (torch.floor(x) - x).detach() + x

def ste_clamp(x, min_val, max_val):
    return (x.clamp(min_val, max_val) - x).detach() + x

def quantize_int8(x, scale=128.0):
    return ste_clamp(ste_round(x * scale), -128, 127)

def cut_int8(x, cut_n):
    div = 2 ** (2 * cut_n)
    return ste_clamp(ste_floor(x / div), -128, 127)

# ============ Load model ============
device = torch.device('cpu')
sd = torch.load(r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth', map_location=device)

# ============ Load image ============
img_path = r'E:\Tsinghua\tianji\package\package\mnist_train_jpg\7_00324.jpg'
img = Image.open(img_path).convert('L')
img = img.resize((28, 28), Image.Resampling.LANCZOS)
img_array = np.array(img, dtype=np.float32)
img_array = (img_array / 255.0) * 2.0 - 1.0
x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)

print(f'Image: {img_path}')
print(f'Input range: [{x.min():.4f}, {x.max():.4f}]')

# ============ Hardware-accurate forward pass ============
# Input quantization
x_q = quantize_int8(x, 128.0)
print(f'Input quantized range: [{x_q.min()}, {x_q.max()}]')

# Conv1
w1 = sd['conv1.weight']
b1 = sd['conv1.bias']
w1_q = torch.round(w1 * 128.0).clamp(-128, 127)
b1_q = torch.round(b1 * 16384.0).clamp(-0x80000000, 0x7fffffff)
x1 = F.conv2d(x_q, w1_q, b1_q, padding=0)
x1_cut = cut_int8(x1, 4)  # cut1=4, div by 256
print(f'Conv1 output range: [{x1.min():.1f}, {x1.max():.1f}]')
print(f'Conv1 cut range: [{x1_cut.min()}, {x1_cut.max()}]')
x1_r = F.relu(x1_cut)
x1_p = F.max_pool2d(x1_r, 2, 2)
print(f'Conv1+ReLU+Pool range: [{x1_p.min()}, {x1_p.max()}]')

# Conv2
w2 = sd['conv2.weight']
b2 = sd['conv2.bias']
w2_q = torch.round(w2 * 128.0).clamp(-128, 127)
b2_q = torch.round(b2 * 16384.0).clamp(-0x80000000, 0x7fffffff)
x2 = F.conv2d(x1_p, w2_q, b2_q, padding=1)
x2_cut = cut_int8(x2, 4)  # cut2=4, div by 256
print(f'Conv2 output range: [{x2.min():.1f}, {x2.max():.1f}]')
print(f'Conv2 cut range: [{x2_cut.min()}, {x2_cut.max()}]')
x2_r = F.relu(x2_cut)
x2_p = F.max_pool2d(x2_r, 2, 2)
print(f'Conv2+ReLU+Pool range: [{x2_p.min()}, {x2_p.max()}]')

# Flatten
x_f = x2_p.permute(0, 2, 3, 1).reshape(1, -1)
print(f'Flattened shape: {x_f.shape}')

# FC1
w3 = sd['fc1.weight']
b3 = sd['fc1.bias']
w3_q = torch.round(w3 * 128.0).clamp(-128, 127)
b3_q = torch.round(b3 * 16384.0).clamp(-0x80000000, 0x7fffffff)
x3 = F.linear(x_f, w3_q, b3_q)
x3_cut = cut_int8(x3, 5)  # fc_cut1=5, div by 1024
print(f'FC1 output range: [{x3.min():.1f}, {x3.max():.1f}]')
print(f'FC1 cut range: [{x3_cut.min()}, {x3_cut.max()}]')
x3_r = F.relu(x3_cut)
print(f'FC1+ReLU range: [{x3_r.min()}, {x3_r.max()}]')

# FC2
w4 = sd['fc2.weight']
b4 = sd['fc2.bias']
w4_q = torch.round(w4 * 128.0).clamp(-128, 127)
b4_q = torch.round(b4 * 16384.0).clamp(-0x80000000, 0x7fffffff)
x4 = F.linear(x3_r, w4_q, b4_q)
x4_cut = cut_int8(x4, 4)  # fc_cut2=4, div by 256
print(f'FC2 output range: [{x4.min():.1f}, {x4.max():.1f}]')
print(f'FC2 cut range: [{x4_cut.min()}, {x4_cut.max()}]')

# ============ This is what p08 would receive ============
p08_input = x4_cut  # int8 values from fc2 after cut
print(f'\n========== p08 INPUT (fc2 after cut, {p08_input.shape}) ==========')
print(f'  Range: [{p08_input.min()}, {p08_input.max()}]')
print(f'  Mean: {p08_input.mean():.2f}')
print(f'  Std: {p08_input.std():.2f}')
print(f'  Positive count: {(p08_input > 0).sum().item()} / {p08_input.numel()}')
print(f'  Negative count: {(p08_input < 0).sum().item()} / {p08_input.numel()}')
print(f'  Zero count: {(p08_input == 0).sum().item()} / {p08_input.numel()}')
print(f'  Max positive value: {p08_input[p08_input > 0].max().item() if (p08_input > 0).sum() > 0 else "N/A"}')
val_str = ', '.join([f'{v}' for v in p08_input.squeeze().int().tolist()])
print(f'  All 84 values: [{val_str}]')

# FC3 with ReLU (current QAT behavior)
x4_r = F.relu(x4_cut)
print(f'\nFC2+ReLU range (current QAT fc3 input): [{x4_r.min()}, {x4_r.max()}]')

w5 = sd['fc3.weight']
b5 = sd['fc3.bias']
w5_q = torch.round(w5 * 128.0).clamp(-128, 127)
b5_q = torch.round(b5 * 16384.0).clamp(-0x80000000, 0x7fffffff)

# Current: ReLU output -> fc3
x5_relu = F.linear(x4_r, w5_q, b5_q)
x5_relu_cut = cut_int8(x5_relu, 5)
print(f'\nFC3 (with ReLU) output range: [{x5_relu.min():.1f}, {x5_relu.max():.1f}]')
print(f'FC3 cut (with ReLU) range: [{x5_relu_cut.min()}, {x5_relu_cut.max()}]')
print(f'Prediction (ReLU): class {x5_relu_cut.argmax().item()}, scores: {x5_relu_cut.squeeze().int().tolist()}')

# ============ Simulate LIF behavior ============
print('\n========== Simulating LIF with fire_type=3 ==========')
# fire_type=3: output = V_new // 4^in_cut_start, clipped to int8
# With: Vth0=10, no firing for values < 10
# V += Uin
# V_new = V (if no fire) + Vleaky_beta
# output = V_new // 4^0 (in_cut_start=0 for p08 output) = V_new

# LIF param: in_cut_start for p08 OUTPUT is separate from the input cut
# The input to p08 is already cut by fc_cut2=4
# p08 has its own in_cut_start which is applied to output in fire_type=2/3

# Simulate LIF with various thresholds
for Vth0_test in [0, 1, 2, 5, 10, 50, 100]:
    V = torch.zeros(84, dtype=torch.int32)
    Uin = p08_input.squeeze().int()  # (84,) int values in [-128, 127]
    
    # LIF step
    V_old = V.clone()
    V = V + Uin
    V = V.clamp(-2**27, 2**27-1)
    spike = V > Vth0_test
    
    if Vth0_test == 0:
        n_fired = spike.sum().item()
        print(f'  Vth0=0: {n_fired} neurons fired')
    
    # Rst_mode=0: reset to VR=0
    V_update = V.clone()
    V_update[spike] = 0
    
    # No leak
    V_new = V_update.clone()
    
    # fire_type=3 output: V_new (with in_cut_start=0, no extra division)
    lif_output = V_new.clone()
    
    # Now compare with ReLU
    relu_output = F.relu(x4_cut).squeeze().int()
    
    # Check how many signs match (positive/negative/zero)
    lif_sign = (lif_output > 0).int()
    relu_sign = (relu_output > 0).int()
    match = (lif_sign == relu_sign).sum().item()
    
    # Correlation
    corr = np.corrcoef(lif_output.numpy(), relu_output.numpy())[0, 1]
    
    print(f'  Vth0={Vth0_test:3d}: | fire={spike.sum().item():2d}/84 | sign_match={match}/84 | corr={corr:.3f} | '
          f'lif_range=[{lif_output.min()}, {lif_output.max()}]')

# Now try with a very low threshold and fire_type that passes V through
print('\n========== Simulating LIF as passthrough (fire_type=1) ==========')
# fire_type=1: output = V_new (full 28-bit, no quantization)
# But fc3 expects int8 values! This won't work directly.

# Actually let me think about this differently.
# The p08 output gets stored to memory at addr_s=0x600
# Then fc3 reads from 0x600 >> 2 as its input
# fc3 expects int8 values

# fire_type=4: output = spike (0 or 1). fc3 weights were trained expecting [0, 127] range.
# If we use fire_type=4, fc3 would receive 0/1, which is 0-1% of expected signal!
# This won't work without retraining.

# fire_type=3 with low threshold:
# V_new is 28-bit. After // 4^in_cut_start with in_cut_start=0, still 28-bit.
# But fc3 expects int8! It reads from memory as int8.
# So the value will be truncated to int8 anyway.

# With Vth0=0, all positive inputs cause firing, resetting V to 0
# Non-firing neurons (negative input) have V = negative value
# V_new = V (no fire) or 0 (fire)
# Output (fire_type=3, in_cut_start=0) = V_new, truncated to int8 by fc3 read

print('Analyzing different LIF configurations...')

# Try: Vth0 very high, fire_type=1 (full V_new), passthrough
V = torch.zeros(84, dtype=torch.int32)
Uin = p08_input.squeeze().int()
V = V + Uin
spike = V > 10000  # never fires
V_new = V  # no reset, no leak
# fire_type=1: output full V_new (28-bit), fc3 reads as int8 -> truncated
print(f'  Passthrough (Vth0 huge): V range=[{V_new.min()}, {V_new.max()}]')
print(f'  As int8: range=[{V_new.clamp(-128,127).min()}, {V_new.clamp(-128,127).max()}]')

# What we need: output same as ReLU(x) = max(0, x)
# This means: output = 0 for negative inputs, output = input for positive inputs
# 
# With LIF having very low threshold (Vth0=0 or 1):
# - Positive Uin -> V > Vth -> fire -> reset to VR=0 -> V_new=0
# - Negative Uin -> V < Vth -> no fire -> V_new = V (negative)
#
# Hmm, that gives us 0 for positive and negative for negative. That's NOT ReLU.
# ReLU gives 0 for negative and positive for positive.
#
# What about fire_type=0 (output = V before update)?
# - Positive Uin -> V = positive -> output = positive
# - Negative Uin -> V = negative -> output = negative
# Still not ReLU, it's passthrough.
#
# So actually we CANNOT get ReLU behavior from LIF without some trick.
# The LIF dynamics are fundamentally different.

# BUT: the fc3 layer might work fine with direct passthrough (no ReLU)!
# Remember, the original LeNet (from lenet.py's LeNet class) has:
# fc3 -> fc_cut3
# With no ReLU between fc2 and fc3!
# Wait, let me check again...

print('\n========== Direct passthrough (no activation) ==========')
x5_direct = F.linear(x4_cut, w5_q, b5_q)  # no ReLU
x5_direct_cut = cut_int8(x5_direct, 5)
print(f'FC3 (no activation) range: [{x5_direct.min():.1f}, {x5_direct.max():.1f}]')
print(f'FC3 cut (no act) range: [{x5_direct_cut.min()}, {x5_direct_cut.max()}]')
print(f'Prediction (no act): class {x5_direct_cut.argmax().item()}, scores: {x5_direct_cut.squeeze().int().tolist()}')

print('\n========== LIF fire_type=2 (V before update, quantized to int8) ==========')
# fire_type=2: output = V // 4^in_cut_start, clipped int8
# With in_cut_start=0: output = V (int8 truncated)
# V = V_prev + Uin, V_prev=0
# V = Uin -> output = Uin clipped to int8
# This is identical to direct passthrough!
# Perfect! fire_type=2 with in_cut_start=0 gives direct passthrough.

# But we want REAL LIF. Let me think about fire_type=3 again.
# fire_type=3 with Vth0=0, Rst_mode=0 (reset to VR=0):
# V = V + Uin
# spike = V > 0  (fires for any positive Uin)
# V_update[spike] = VR = 0
# V_new = V_update (no leak)
# output (fire_type=3): V_new // 4^0 = V_new
#
# For positive Uin: V positive > 0 -> fire -> V=0 -> V_new=0 -> output=0
# For negative Uin: V negative <=0 -> no fire -> V_new=V -> output=negative
# This gives: output = min(Uin, 0) which is the OPPOSITE of ReLU!

print('LIF fire_type=3 with Vth0=0:')
V = torch.zeros(84, dtype=torch.int32)
Uin = p08_input.squeeze().int()
V = V + Uin
spike = V > 0
V_update = V.clone()
V_update[spike] = 0
V_new = V_update  # no leak
lif_out3 = V_new  # fire_type=3, in_cut_start=0 -> V_new
relu_out = F.relu(Uin)
print(f'  LIF out range: [{lif_out3.min()}, {lif_out3.max()}]')
print(f'  ReLU out range: [{relu_out.min()}, {relu_out.max()}]')
corr = np.corrcoef(lif_out3.numpy(), relu_out.numpy())[0, 1]
print(f'  Correlation with ReLU: {corr:.3f}')
x5_lif3 = F.linear(lif_out3.float(), w5_q, b5_q)
x5_lif3_cut = cut_int8(x5_lif3, 5)
print(f'  Prediction (LIF3 Vth0=0): class {x5_lif3_cut.argmax().item()}, scores: {x5_lif3_cut.squeeze().int().tolist()}')

# Try fire_type=2 (output = V before update) which is just passthrough
print('\n========== LIF fire_type=2 (V before update, direct passthrough) ==========')
V = torch.zeros(84, dtype=torch.int32)
Uin = p08_input.squeeze().int()
V = V + Uin
# fire_type=2: output = V // 4^0 = V, clipped to int8
lif_out2 = V.clone()
x5_lif2 = F.linear(lif_out2.float(), w5_q, b5_q)
x5_lif2_cut = cut_int8(x5_lif2, 5)
corr2 = np.corrcoef(lif_out2.numpy(), relu_out.numpy())[0, 1]
print(f'  Correlation with ReLU: {corr2:.3f}')
print(f'  LIF2 range: [{lif_out2.min()}, {lif_out2.max()}]')
print(f'  Prediction (LIF2 passthrough): class {x5_lif2_cut.argmax().item()}, scores: {x5_lif2_cut.squeeze().int().tolist()}')

print('\nDone.')
