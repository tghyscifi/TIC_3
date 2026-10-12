import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# ===================== STE 代理梯度 LIF =====================

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
        grad_input = grad_output / (1 + 5 * (x - th).abs()).pow(2)
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, threshold=10):
        super().__init__()
        self.threshold = threshold
    def forward(self, x):
        return SurrogateSpike.apply(x, self.threshold)

# ===================== 纯浮点 LeNet + LIF =====================

class LeNetLIF_Float(nn.Module):
    def __init__(self, lif_threshold=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400, 120)
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.lif = LIFSpike(threshold=lif_threshold)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, use_lif=True):
        # 没有量化，纯浮点
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        if use_lif:
            x = self.lif(x)
        else:
            x = torch.relu(x)
        x = self.fc3(x)
        return x

# ===================== 配置 =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
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

model = LeNetLIF_Float(lif_threshold=10).to(device)

# ===================== 加载预训练权重 =====================

ckpt_path = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth'
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    # 原始 checkpoint 的 key 名直接匹配 (conv1.weight, conv1.bias, ...)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded pretrained weights from {ckpt_path}')
    print(f'  Missing keys: {missing}')
    print(f'  Unexpected keys: {unexpected}')

for name, param in model.named_parameters():
    param.requires_grad = True
print('Training all layers end-to-end (float, no quantization)')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===================== 评估 =====================

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

print(f'\nInitial accuracy:')
print(f'  LIF:  {evaluate(model, test_loader, use_lif=True):.2f}%')
print(f'  ReLU: {evaluate(model, test_loader, use_lif=False):.2f}%')

# ===================== 训练 =====================

print(f'\nEnd-to-end float training with LIF on {device}...')
best_lif_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, use_lif=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    lif_acc = evaluate(model, test_loader, use_lif=True)
    relu_acc = evaluate(model, test_loader, use_lif=False)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, '
          f'LIF: {lif_acc:.2f}%, ReLU: {relu_acc:.2f}%')

    if lif_acc > best_lif_acc:
        best_lif_acc = lif_acc
        torch.save(model.state_dict(), 'lenet_lif_float_best.pth')

# ===================== 保存 =====================

torch.save(model.state_dict(), 'lenet_lif_float_final.pth')
print(f'\nBest LIF accuracy: {best_lif_acc:.2f}%')
print(f'Models saved: lenet_lif_float_best.pth, lenet_lif_float_final.pth')
