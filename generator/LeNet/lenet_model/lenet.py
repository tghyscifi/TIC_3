import torch
import torch.nn as nn


# 用于量化权重和INT8激活
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_min, x_max, x_range):
        ctx.save_for_backward(x, torch.tensor(x_min), torch.tensor(x_max))
        x = x.mul(x_range).round().clamp(-128, 127).mul(1 / x_range)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask0 = torch.where(x < x_min, zeros, ones)
        mask1 = torch.where(x > x_max, zeros, ones)
        mask = mask0 * mask1
        grad = grad_output * mask
        return grad, None, None, None


# 用于量化偏置和INT32激活
class FakeQuantizeWithoutLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_range):
        ctx.save_for_backward(x)
        x = x.mul(x_range).round().mul(1 / x_range)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Cut(nn.Module):
    def __init__(self, en: bool = False, in_cut_start: int = 4, type_out: int = 1):
        """
        result = (x >> (2 * in_cut_start).clip()
        type_out: 0-int32, 1-int8
        """
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
            # return x.div(2 ** (2 * self.in_cut_start))
        else:
            return x

    def __repr__(self):
        return '{}(en={}, in_cut_start={:d}, type_out={:d})'.format(self.__class__.__name__, self.en,
                                                                    self.in_cut_start, self.type_out)


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
        self.fc_relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(84, 10, bias=not batch_norm)
        self.fc_cut3 = Cut(en=True)

    def forward(self, x):
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        if self.cut_en:
            x = x.mul(128).round().clamp(-128, 127)

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        if self.cut_en:
            x = self.cut1(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.cut_en:
            x = self.cut2(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
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
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.fc_relu1(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.fc_bn2(x)
        if self.cut_en:
            x = self.fc_cut2(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.fc_relu2(x)

        x = self.fc3(x)

        x = self.fc_cut3(x)

        return x


def lenet(pretrained=None, batch_norm=True, clamp=False, aware_test=None, cut_en=False, aware=False):
    model = LeNet(batch_norm=batch_norm, clamp=clamp, aware_test=aware_test, cut_en=cut_en, aware=aware)
    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained, map_location=device)
        if pretrained_dict.get('state_dict') is not None:
            pretrained_dict = pretrained_dict['state_dict']
        model.load_state_dict(pretrained_dict, strict=False)
    return model
