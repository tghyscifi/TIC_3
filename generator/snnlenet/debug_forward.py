import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Manually define classes to avoid running train.py's training code
class Cut(nn.Module):
    def __init__(self, in_cut_start=4):
        super().__init__()
        self.in_cut_start = in_cut_start
    def forward(self, x):
        scale = 2 ** (2 * self.in_cut_start)
        x_q = x.div(scale).floor().clamp(-128, 127)
        return (x_q - x.div(scale)).detach() + x.div(scale)

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        th = ctx.threshold
        grad_input = grad_output / (1 + 10 * (x - th).abs()).pow(2)
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, threshold=10):
        super().__init__()
        self.threshold = threshold
    def forward(self, x):
        return SurrogateSpike.apply(x, self.threshold)

class LeNetLIF(nn.Module):
    def __init__(self, lif_threshold=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.cut1 = Cut(in_cut_start=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=1)
        self.cut2 = Cut(in_cut_start=4)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400, 120)
        self.fc_cut1 = Cut(in_cut_start=5)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.fc_cut2 = Cut(in_cut_start=4)
        self.lif = LIFSpike(threshold=lif_threshold)
        self.fc3 = nn.Linear(84, 10)
        self.fc_cut3 = Cut(in_cut_start=5)

    def forward(self, x, use_lif=True):
        x = (torch.round(x * 128).clamp(-128, 127) - x).detach() + x
        w1 = torch.round(self.conv1.weight * 128).clamp(-128, 127)
        b1 = torch.round(self.conv1.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.conv2d(x, w1, b1)
        x = self.cut1(x); x = self.relu1(x); x = self.pool1(x)
        w2 = torch.round(self.conv2.weight * 128).clamp(-128, 127)
        b2 = torch.round(self.conv2.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.conv2d(x, w2, b2, padding=1)
        x = self.cut2(x); x = self.relu2(x); x = self.pool2(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        w3 = torch.round(self.fc1.weight * 128).clamp(-128, 127)
        b3 = torch.round(self.fc1.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.linear(x, w3, b3)
        x = self.fc_cut1(x); x = self.fc_relu1(x)
        w4 = torch.round(self.fc2.weight * 128).clamp(-128, 127)
        b4 = torch.round(self.fc2.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.linear(x, w4, b4)
        x = self.fc_cut2(x)
        if use_lif:
            x = self.lif(x)
        else:
            x = torch.relu(x)
        w5 = torch.round(self.fc3.weight * 128).clamp(-128, 127)
        b5 = torch.round(self.fc3.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.linear(x, w5, b5)
        x = self.fc_cut3(x)
        return x

model = LeNetLIF(lif_threshold=10)
state = torch.load(r'lenet_snn_sim.pth', map_location='cpu')
model.load_state_dict(state, strict=False)
model.eval()

img = Image.open(r'E:\Tsinghua\tianji\package\package\mnist_train_jpg\7_00324.jpg').convert('L').resize((28,28))
x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out_relu = model(x, use_lif=False)
    out_lif = model(x, use_lif=True)
    print(f'ReLU: scores={out_relu.squeeze().int().tolist()}, pred={out_relu.argmax().item()}')
    print(f'LIF:  scores={out_lif.squeeze().int().tolist()}, pred={out_lif.argmax().item()}')
