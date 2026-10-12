import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ==========================================
# 1. 配置与超参数
# ==========================================
batch_size = 128
data_path = './data/mnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_steps = 25      # 时间步长（SNN处理序列长度）
beta = 0.9          # 膜电位衰减率
learning_rate = 1e-3
num_epochs = 5
save_path = "best_snn_mnist.pth"

# ==========================================
# 2. 数据加载 (MNIST)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# ==========================================
# 3. 定义 SNN 网络架构
# ==========================================
# 代理梯度函数：解决阶跃函数不可导的问题
spike_grad = surrogate.fast_sigmoid(slope=25)

class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一层卷积：输入 1x28x28 -> 输出 12x24x24
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        # 第二层卷积：输入 12x12x12 -> 输出 32x8x8
        self.conv2 = nn.Conv2d(12, 32, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        # 全连接层：输入 32x4x4 -> 输出 10 (分类数)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        # 初始化每一层的膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # 用于记录输出层的脉冲和膜电位
        spk3_rec = []
        mem3_rec = []

        # 随时间迭代 (SNN 的核心逻辑)
        for step in range(num_steps):
            # 第一层
            cur1 = self.pool1(self.lif1(self.conv1(x), mem1)[0])
            _, mem1 = self.lif1(self.conv1(x), mem1) # 更新膜电位
            
            # 第二层
            cur2 = self.pool2(self.lif2(self.conv2(cur1), mem2)[0])
            _, mem2 = self.lif2(self.conv2(cur1), mem2)

            # 输出层
            cur3 = cur2.view(batch_size, -1)
            spk3, mem3 = self.lif3(self.fc1(cur3), mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

model = SNN().to(device)

# ==========================================
# 4. 优化器与损失函数
# ==========================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 使用脉冲计数损失：鼓励正确类别的神经元在时间窗口内发放更多脉冲
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

# ==========================================
# 5. 训练与测试逻辑
# ==========================================
def test_model(loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = model(data)
            
            total_loss += loss_fn(spk_rec, targets).item()
            # 预测值：脉冲累积总数最高的索引
            _, predicted = spk_rec.sum(dim=0).max(1)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100 * correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

# ---------------- 训练主循环 ----------------
best_acc = 0
print(f"开始训练 (设备: {device})...")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # 前向传播
        spk_rec, mem_rec = model(data)
        loss_val = loss_fn(spk_rec, targets)

        # 反向传播
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()

        if i % 100 == 0:
            print(f"Epoch {epoch+1} [{i}/{len(train_loader)}] Loss: {loss_val.item():.4f}")

    # 验证最佳模型
    val_loss, val_acc = test_model(test_loader)
    print(f"\n--- Epoch {epoch+1} 总结 ---")
    print(f"训练损失: {train_loss/len(train_loader):.4f}")
    print(f"测试准确率: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"发现更优模型，已保存至 {save_path}")
    print("-" * 30)

print(f"\n所有 Epoch 结束。最高准确率: {best_acc:.2f}%")
