import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
from PIL import Image
import os
import math
import warnings


class Cut(nn.Module):
    def __init__(self, en: bool = False, in_cut_start: int = 4, type_out: int = 1):
        super(Cut, self).__init__()
        self.en = en
        self.in_cut_start = in_cut_start
        self.type_out = type_out
        if type_out == 0:
            self.max, self.min = 0x7fffffff, -0x80000000
        elif type_out == 1:
            self.max, self.min = 0x7f, -0x80
        else:
            raise ValueError

    def forward(self, x):
        if self.en:
            return x.div(2 ** (2 * self.in_cut_start)).floor().clamp(min=self.min, max=self.max)
        else:
            return x

    def __repr__(self):
        return '{}(en={}, in_cut_start={:d}, type_out={:d})'.format(
            self.__class__.__name__, self.en, self.in_cut_start, self.type_out)


class LIFNeuron(nn.Module):
    """基于 snntorch.Leaky 的 LIF 神经元（fire_type=4 模式，输出膜电位）

    配置 beta=0, threshold=0, reset_mechanism='zero'：
      beta=0  → 无时间累积，每次调用独立
      threshold=0 → 膜电位 >= 0 即点火
      reset_mechanism='zero' → 点火后重置为 0
      spk * mem → 输出膜电位（fire_type=4），与 ReLU(x) = max(x, 0) 等价
    """
    def __init__(self, threshold=0, reset_v=0, v_lower=-10, fire_type=4):
        super().__init__()
        self.v_lower = v_lower
        self._lif = snn.Leaky(
            beta=0.0,
            threshold=threshold,
            reset_mechanism='zero',
            init_hidden=True,
            output=True
        )

    def forward(self, x):
        V = torch.clamp(x, self.v_lower, None)
        spk, mem = self._lif(V)
        return spk * mem


class LeNet(torch.nn.Module):
    def __init__(self, batch_norm=True, clamp=False, aware_test=None,
                 cut_en=False, aware=False):
        super(LeNet, self).__init__()

        self.batch_norm = batch_norm
        self.clamp = clamp
        self.cut_en = cut_en
        self.aware = aware
        self.aware_test = aware_test

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), bias=not batch_norm)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(6)
        self.cut1 = Cut(en=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=not batch_norm)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(16)
        self.cut2 = Cut(en=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(400, 120, bias=not batch_norm)
        if self.batch_norm:
            self.fc_bn1 = nn.BatchNorm1d(120)
        self.fc_cut1 = Cut(en=True)
        self.fc_relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(120, 84, bias=not batch_norm)
        if self.batch_norm:
            self.fc_bn2 = nn.BatchNorm1d(84)
        self.fc_cut2 = Cut(en=True)
        self.fc_lif2 = LIFNeuron(threshold=0, reset_v=0, v_lower=-10, fire_type=4)
        self.fc3 = nn.Linear(84, 10, bias=not batch_norm)
        self.fc_cut3 = Cut(en=True)

    def forward(self, x):
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.cut_en:
            x = x.mul(128).round().clamp(-128, 127)

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        if self.cut_en:
            x = self.cut1(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.cut_en:
            x = self.cut2(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        if self.batch_norm:
            x = self.fc_bn1(x)
        if self.cut_en:
            x = self.fc_cut1(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        x = self.fc_relu1(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.fc_bn2(x)
        if self.cut_en:
            x = self.fc_cut2(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        x = self.fc_lif2(x)

        x = self.fc3(x)

        x = self.fc_cut3(x)

        return x


class QuantizationConfig:
    def __init__(self, name='in_cut_start'):
        self.__name = name

        self.__weight_parameters = {
            'conv1': 256,
            'conv2': 256,
            'fc1': 1024,
            'fc2': 256,
            'fc3': 256,
        }

        self.__cut_parameters = {
            'cut1': int(math.log(self.__weight_parameters['conv1'], 4)),
            'cut2': int(math.log(self.__weight_parameters['conv2'], 4)),
            'fc_cut1': int(math.log(self.__weight_parameters['fc1'], 4)),
            'fc_cut2': int(math.log(self.__weight_parameters['fc2'], 4)),
            'fc_cut3': int(math.log(self.__weight_parameters['fc3'] * 4, 4))
        }

    def __getitem__(self, item):
        if self.__name == 'in_cut_start':
            result = self.__cut_parameters.get(item)
        else:
            assert self.__name == 'weight'
            result = self.__weight_parameters.get(item)
        if result is None:
            warnings.warn('Module \'{}\' do not have in_cut_start parameter!'.format(item))
        return result


def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)

    saved_sd = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_sd, strict=False)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data.mul_(128).round_().clamp_(-128, 127)
            module.bias.data.mul_(128 * 128).round_().clamp_(-0x80000000, 0x7fffffff)

    cuts = QuantizationConfig()
    for name, module in model.named_modules():
        if isinstance(module, Cut):
            module.en = True
            module.in_cut_start = cuts[name]

    model.to(device)
    model.eval()
    return model, device


def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 255.0) * 2.0 - 1.0
    x = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return x


def predict(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    output_values = output.squeeze().cpu().tolist()
    return pred_class, output_values


def main():
    model_path = r'D:\tsinghua\tianjic\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth'
    img_path = r'D:\tsinghua\tianjic\package\mnist_train_jpg\0_00001.jpg'

    print(f"加载模型: {model_path}")
    model, device = load_model(model_path)
    print(f"模型加载完成，运行设备: {device}")

    if not os.path.exists(img_path):
        print(f"文件不存在: {img_path}")
        return

    true_label = int(os.path.basename(img_path).split('_')[0])

    img_tensor = preprocess_image(img_path)
    pred_class, output_values = predict(model, img_tensor, device)

    print(f"\n{'=' * 50}")
    print(f"图片文件: {os.path.basename(img_path)}")
    print(f"真实标签: {true_label}")
    print(f"预测结果: {pred_class}")
    print(f"{'正确' if pred_class == true_label else '错误'}")
    print(f"\n各类别输出值（10个评分）:")
    for i, val in enumerate(output_values):
        marker = " <-- 预测" if i == pred_class else ""
        print(f"  类别 {i}: {val:>12.4f}{marker}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
