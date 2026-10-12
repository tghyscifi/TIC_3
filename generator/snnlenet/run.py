import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
from pathlib import Path
import math   # 用于 LIF 中的指数计算

# ==========================================
# 1. 定义与训练时完全相同的模型结构
# ==========================================

class SurrogateLIF(torch.autograd.Function):
    """带代理梯度的 LIF 神经元（离散时间，硬重置）"""
    @staticmethod
    def forward(ctx, x, decay, threshold, alpha):
        batch, features, T = x.shape
        v = torch.zeros(batch, features, device=x.device)
        spikes = torch.zeros_like(x)
        for t in range(T):
            v = decay * v + x[:, :, t]
            spike = (v >= threshold).float()
            v = v - spike * threshold
            spikes[:, :, t] = spike
        ctx.save_for_backward(x, spikes)
        ctx.decay = decay
        ctx.threshold = threshold
        ctx.alpha = alpha
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        x, spikes = ctx.saved_tensors
        decay = ctx.decay
        threshold = ctx.threshold
        alpha = ctx.alpha
        batch, features, T = x.shape
        grad_x = torch.zeros_like(x)
        grad_v_next = torch.zeros(batch, features, device=x.device)
        for t in reversed(range(T)):
            v = torch.zeros(batch, features, device=x.device)
            for t2 in range(t + 1):
                v = decay * v + x[:, :, t2]
                if t2 < t:
                    spike_tmp = (v >= threshold).float()
                    v = v - spike_tmp * threshold
            v_before = v
            grad_spike = alpha * torch.sigmoid(alpha * (v_before - threshold)) * (1 - torch.sigmoid(alpha * (v_before - threshold)))
            grad_v_cur = grad_output[:, :, t] * grad_spike + grad_v_next
            grad_x[:, :, t] = grad_v_cur
            grad_v_next = grad_v_cur * decay
        return grad_x, None, None, None


class LIFSpikingLayer(nn.Module):
    def __init__(self, tau=2.0, dt=0.1, threshold=1.0, alpha=10.0):
        super().__init__()
        self.decay = math.exp(-dt / tau)
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, x):
        return SurrogateLIF.apply(x, self.decay, self.threshold, self.alpha)


class LeNetSNN(nn.Module):
    def __init__(self, T=8, tau=2.0, dt=0.1, threshold=1.0, alpha=10.0):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(400, 120)
        self.lif1 = LIFSpikingLayer(tau, dt, threshold, alpha)
        self.fc2 = nn.Linear(120, 84)
        self.lif2 = LIFSpikingLayer(tau, dt, threshold, alpha)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(batch_size, -1)               # (batch, 400)
        x = x.unsqueeze(-1).repeat(1, 1, self.T) # (batch, 400, T)

        x = x.transpose(1, 2)                    # (batch, T, 400)
        x = self.fc1(x)                          # (batch, T, 120)
        x = x.transpose(1, 2)                    # (batch, 120, T)
        x = self.lif1(x)                         # (batch, 120, T)

        x = x.transpose(1, 2)                    # (batch, T, 120)
        x = self.fc2(x)                          # (batch, T, 84)
        x = x.transpose(1, 2)                    # (batch, 84, T)
        x = self.lif2(x)                         # (batch, 84, T)

        x = x.transpose(1, 2)                    # (batch, T, 84)
        x = self.fc3(x)                          # (batch, T, 10)
        out = x.mean(dim=1)                      # (batch, 10)
        return out


# ==========================================
# 2. 图像预处理（与训练时完全一致）
# ==========================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),                # 训练时 Resize 到 32x32
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])


# ==========================================
# 3. 推理函数
# ==========================================
def predict_image(model, image_input, device):
    """
    参数:
        model: 已加载权重的 LeNetSNN 模型
        image_input: 图片路径(str) 或 PIL Image 对象
        device: torch.device
    返回:
        predicted_class: int, 0~9
        logits: numpy数组，长度为10
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('L')
    else:
        img = image_input.convert('L')

    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 1, 32, 32)

    with torch.no_grad():
        logits = model(img_tensor)   # (1, 10)
        probs = torch.softmax(logits, dim=1)
        predicted_class = int(logits.argmax(dim=1).item())
        logits_np = logits.squeeze(0).cpu().numpy()

    return predicted_class, logits_np


# ==========================================
# 4. 主函数：参数解析 + 模型加载 + 推理
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="LeNet-SNN MNIST 推理脚本")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="训练好的模型权重文件路径 (.pth)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--image", type=str, help="单张图片路径")
    group.add_argument("-f", "--folder", type=str, help="批量处理文件夹路径")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"], help="运行设备")
    parser.add_argument("--T", type=int, default=8,
                        help="时间步长（必须与训练时一致）")

    args = parser.parse_args()

    # 确定设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 加载模型（注意 T 必须与训练时相同）
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件 {args.model} 不存在")
    model = LeNetSNN(T=args.T).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"模型加载成功: {args.model} -> {device}, 时间步 T={args.T}")

    # 单张图片预测
    if args.image:
        if not os.path.exists(args.image):
            print(f"错误：图片文件 {args.image} 不存在")
            return
        pred, logits = predict_image(model, args.image, device)
        print(f"\n图片: {args.image}")
        print(f"预测类别: {pred}")
        print(f"各类 logits: {logits}")
        print(f"各类概率: {torch.softmax(torch.tensor(logits), dim=0).numpy()}")

    # 批量处理文件夹
    elif args.folder:
        folder = Path(args.folder)
        if not folder.exists() or not folder.is_dir():
            print(f"错误：文件夹 {folder} 不存在或不是目录")
            return
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pgm'}
        img_files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
        if not img_files:
            print(f"文件夹 {folder} 中没有找到图片文件")
            return
        print(f"\n批量处理文件夹 {folder}，共找到 {len(img_files)} 张图片\n")
        for img_file in sorted(img_files):
            pred, logits = predict_image(model, str(img_file), device)
            probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
            confidence = probs[pred]
            print(f"{img_file.name} -> 预测: {pred}, 置信度: {confidence:.2%}")
        print("\n批量处理完成")


if __name__ == "__main__":
    main()
