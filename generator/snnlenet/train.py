import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# ===================== 1. Cut 层（匹配硬件量化，直通梯度） =====================

class Cut(nn.Module):
    def __init__(self, in_cut_start=4):
        super().__init__()
        self.in_cut_start = in_cut_start
    def forward(self, x):
        scale = 2 ** (2 * self.in_cut_start)
        x_q = x.div(scale).floor().clamp(-128, 127)
        # 直通梯度（STE），去掉 1/scale 衰减
        return (x_q - x).detach() + x

# ===================== 2. STE 代理梯度 LIF（匹配 p08 fire_type=4）=====================

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
        # 加强的代理梯度（去掉 10x 因子，让梯度更宽更强）
        grad_input = grad_output / (1 + (x - th).abs()).pow(2)
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, threshold=10):
        super().__init__()
        self.threshold = threshold
    def forward(self, x):
        return SurrogateSpike.apply(x, self.threshold)

# ===================== 3. LeNet + Cut + LIF =====================

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

# ===================== 4. 训练配置 =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)
])

train_loader = DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False
)

model = LeNetLIF(lif_threshold=10).to(device)

# ===================== 5. 加载预训练权重 =====================

ckpt_path = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth'
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded pretrained weights from {ckpt_path}')
    if missing:
        print(f'  Missing keys (new layers): {missing}')
    if unexpected:
        print(f'  Unexpected keys: {unexpected}')
else:
    print('No pretrained weights found, training from scratch')

for name, param in model.named_parameters():
    param.requires_grad = True
print(f'  Training all layers end-to-end')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ===================== 6. 验证初始准确率（ReLU vs LIF）=====================

def evaluate(model, loader, use_lif=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, use_lif=use_lif)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

print(f'\nInput: 28x28 MNIST, LIF threshold={model.lif.threshold}')
print(f'Initial ReLU accuracy: {evaluate(model, test_loader, use_lif=False):.2f}%')
print(f'Initial LIF  accuracy: {evaluate(model, test_loader, use_lif=True):.2f}%')

# ===================== 7. 训练 =====================

print(f'\nEnd-to-end training with LIF on {device}...')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, use_lif=True)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item()

    lif_acc = evaluate(model, test_loader, use_lif=True)
    relu_acc = evaluate(model, test_loader, use_lif=False)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, '
          f'LIF: {lif_acc:.2f}%, ReLU: {relu_acc:.2f}%')
    scheduler.step()

torch.save(model.state_dict(), 'lenet_lif_surrogate.pth')
print(f'\nModel saved to lenet_lif_surrogate.pth')
