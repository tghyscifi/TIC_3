import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

class Cut(nn.Module):
    def __init__(self, in_cut_start=4):
        super().__init__()
        self.in_cut_start = in_cut_start
    def forward(self, x):
        scale = 2 ** (2 * self.in_cut_start)
        x_q = x.div(scale).floor().clamp(-128, 127)
        return (x_q - x).detach() + x

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
        grad_input = grad_output / (1 + (x - th).abs()).pow(2)
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

    def quantize_weight(self, w, scale=128):
        w_raw = w * scale
        w_q = torch.round(w_raw).clamp(-128, 127)
        return (w_q - w_raw).detach() + w_raw

    def quantize_bias(self, b, scale=16384):
        b_raw = b * scale
        b_q = torch.round(b_raw).clamp(-0x80000000, 0x7fffffff)
        return (b_q - b_raw).detach() + b_raw

    def forward(self, x, use_lif=True):
        x = (torch.round(x * 128).clamp(-128, 127) - x).detach() + x
        w1 = self.quantize_weight(self.conv1.weight, 128)
        b1 = self.quantize_bias(self.conv1.bias, 16384)
        x = F.conv2d(x, w1, b1)
        x = self.cut1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        w2 = self.quantize_weight(self.conv2.weight, 128)
        b2 = self.quantize_bias(self.conv2.bias, 16384)
        x = F.conv2d(x, w2, b2, padding=1)
        x = self.cut2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        w3 = self.quantize_weight(self.fc1.weight, 128)
        b3 = self.quantize_bias(self.fc1.bias, 16384)
        x = F.linear(x, w3, b3)
        x = self.fc_cut1(x)
        x = self.fc_relu1(x)
        w4 = self.quantize_weight(self.fc2.weight, 128)
        b4 = self.quantize_bias(self.fc2.bias, 16384)
        x = F.linear(x, w4, b4)
        x = self.fc_cut2(x)
        if use_lif:
            x = self.lif(x)
        else:
            x = torch.relu(x)
        w5 = self.quantize_weight(self.fc3.weight, 128)
        b5 = self.quantize_bias(self.fc3.bias, 16384)
        x = F.linear(x, w5, b5)
        x = self.fc_cut3(x)
        return x

device = torch.device("cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)
])
train_loader = DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

model = LeNetLIF(lif_threshold=10).to(device)
ckpt_path = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth'
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=False)

criterion = nn.CrossEntropyLoss()

images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)
print(f'Batch: {images.shape}, label distribution: {labels.bincount().tolist()}')

outputs = model(images, use_lif=True)
loss = criterion(outputs, labels)
print(f'Loss: {loss.item():.4f}')
loss.backward()

print(f'\nGradient norms (LIF mode):')
total_norm = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.data.norm().item()
        rel_norm = grad_norm / (param_norm + 1e-8)
        print(f'  {name:20s}: grad_norm={grad_norm:.8f}, param_norm={param_norm:.4f}, rel={rel_norm:.8f}')
        total_norm += grad_norm
    else:
        print(f'  {name:20s}: NO GRADIENT')
print(f'  {"Total":20s}: {total_norm:.8f}')
