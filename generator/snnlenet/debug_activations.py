import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(r'E:\Tsinghua\tianji\tianjic_x1_software_0814')

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
        x = F.conv2d(x, w1, b1); x = self.cut1(x); x = self.relu1(x); x = self.pool1(x)
        w2 = torch.round(self.conv2.weight * 128).clamp(-128, 127)
        b2 = torch.round(self.conv2.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.conv2d(x, w2, b2, padding=1); x = self.cut2(x); x = self.relu2(x); x = self.pool2(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        w3 = torch.round(self.fc1.weight * 128).clamp(-128, 127)
        b3 = torch.round(self.fc1.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.linear(x, w3, b3); x = self.fc_cut1(x); x = self.fc_relu1(x)
        w4 = torch.round(self.fc2.weight * 128).clamp(-128, 127)
        b4 = torch.round(self.fc2.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.linear(x, w4, b4); x = self.fc_cut2(x)
        if use_lif:
            x = self.lif(x)
        else:
            x = torch.relu(x)
        w5 = torch.round(self.fc3.weight * 128).clamp(-128, 127)
        b5 = torch.round(self.fc3.bias * 16384).clamp(-0x80000000, 0x7fffffff)
        x = F.linear(x, w5, b5); x = self.fc_cut3(x)
        return x

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0)
])

test_loader = DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=1, shuffle=True
)

model = LeNetLIF(lif_threshold=10).to(device)
ckpt_path = r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth'
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

# Test on 5 random images
n_images = 5
total_good_lif = 0
total_good_relu = 0
for idx, (images, labels) in enumerate(test_loader):
    if idx >= n_images:
        break
    images, labels = images.to(device), labels.to(device)
    print(f'\n{"="*60}')
    print(f'Sample {idx+1}: Label = {labels.item()}')
    
    x = (torch.round(images * 128).clamp(-128, 127) - images).detach() + images

    w1 = torch.round(model.conv1.weight * 128).clamp(-128, 127)
    b1 = torch.round(model.conv1.bias * 16384).clamp(-0x80000000, 0x7fffffff)
    x = F.conv2d(x, w1, b1)
    print(f'  After conv1:    mean={x.mean().item():.2f}, std={x.std().item():.2f}, min={x.min().item():.2f}, max={x.max().item():.2f}')
    x = model.cut1(x)
    print(f'  After cut1:     mean={x.mean().item():.2f}, std={x.std().item():.2f}, min={x.min().item():.2f}, max={x.max().item():.2f}')
    x = model.relu1(x)
    x = model.pool1(x)

    w2 = torch.round(model.conv2.weight * 128).clamp(-128, 127)
    b2 = torch.round(model.conv2.bias * 16384).clamp(-0x80000000, 0x7fffffff)
    x = F.conv2d(x, w2, b2, padding=1)
    print(f'  After conv2:    mean={x.mean().item():.2f}, std={x.std().item():.2f}, min={x.min().item():.2f}, max={x.max().item():.2f}')
    x = model.cut2(x)
    print(f'  After cut2:     mean={x.mean().item():.2f}, std={x.std().item():.2f}, min={x.min().item():.2f}, max={x.max().item():.2f}')
    x = model.relu2(x)
    x = model.pool2(x)
    print(f'  After pool2:    mean={x.mean().item():.2f}, std={x.std().item():.2f}, min={x.min().item():.2f}, max={x.max().item():.2f}')

    x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    w3 = torch.round(model.fc1.weight * 128).clamp(-128, 127)
    b3 = torch.round(model.fc1.bias * 16384).clamp(-0x80000000, 0x7fffffff)
    x = F.linear(x, w3, b3)
    x = model.fc_cut1(x)
    x = model.fc_relu1(x)
    print(f'  After fc_relu1: mean={x.mean().item():.2f}, std={x.std().item():.2f}, min={x.min().item():.2f}, max={x.max().item():.2f}')
    
    # --- Branch: fc2 → fc_cut2 → LIF/ReLU → fc3 ---
    fc_relu1_out = x.clone()

    w4 = torch.round(model.fc2.weight * 128).clamp(-128, 127)
    b4 = torch.round(model.fc2.bias * 16384).clamp(-0x80000000, 0x7fffffff)
    x_fc2 = F.linear(fc_relu1_out, w4, b4)
    print(f'  After fc2(raw): mean={x_fc2.mean().item():.2f}, std={x_fc2.std().item():.2f}, min={x_fc2.min().item():.2f}, max={x_fc2.max().item():.2f}')
    
    x_cut = model.fc_cut2(x_fc2)
    print(f'  After fc_cut2:  mean={x_cut.mean().item():.2f}, std={x_cut.std().item():.2f}, min={x_cut.min().item():.2f}, max={x_cut.max().item():.2f}')
    
    fired = (x_cut >= 10).float()
    print(f'  LIF(th=10): {fired.sum().item():.0f}/{fired.numel()} fired ({100*fired.mean().item():.1f}%)')
    print(f'  fc_cut2 value histogram:')
    for v in range(-20, 21):
        cnt = (x_cut == v).sum().item()
        if cnt > 0:
            bar = '#' * cnt
            print(f'    value={v:3d}: [{cnt:3d}] {bar}')
    
    # LIF path
    x_lif = model.lif(x_cut)
    print(f'  LIF output spike sum: {x_lif.sum().item():.0f}/{x_lif.numel()}')
    
    w5 = torch.round(model.fc3.weight * 128).clamp(-128, 127)
    b5 = torch.round(model.fc3.bias * 16384).clamp(-0x80000000, 0x7fffffff)
    x_lif_out = F.linear(x_lif, w5, b5)
    lif_pred = x_lif_out.argmax().item()
    lif_correct = (lif_pred == labels.item())
    total_good_lif += 1 if lif_correct else 0
    print(f'  LIF  fc3 output: mean={x_lif_out.mean().item():.2f}, pred={lif_pred}, correct={lif_correct}')

    # ReLU path
    x_relu = torch.relu(x_cut)
    print(f'  After ReLU:     mean={x_relu.mean().item():.2f}, std={x_relu.std().item():.2f}, min={x_relu.min().item():.2f}, max={x_relu.max().item():.2f}')
    x_relu_out = F.linear(x_relu, w5, b5)
    relu_pred = x_relu_out.argmax().item()
    relu_correct = (relu_pred == labels.item())
    total_good_relu += 1 if relu_correct else 0
    print(f'  ReLU fc3 output: mean={x_relu_out.mean().item():.2f}, pred={relu_pred}, correct={relu_correct}')

print(f'\n{"="*60}')
print(f'Summary: LIF correct: {total_good_lif}/{n_images}, ReLU correct: {total_good_relu}/{n_images}')
