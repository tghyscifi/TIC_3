from pickle import NONE
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from generator.resnet50.resnet50_1chip.quantization_config_1chip import QuantizationConfig
import matplotlib.pyplot as plt


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

    def forward(self, x: Tensor) -> Tensor:
        if self.en:
            return x.div(2 ** (2 * self.in_cut_start)).floor().clamp(min=self.min, max=self.max)
            # return x.div(2 ** (2 * self.in_cut_start))
        else:
            return x

    def __repr__(self):
        return '{}(en={}, in_cut_start={:d}, type_out={:d})'.format(self.__class__.__name__, self.en,
                                                                    self.in_cut_start, self.type_out)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            batch_norm: bool = True,
            clamp: bool = True,
            aware_test=None,
            aware=False,
            cut_en=False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.batch_norm = batch_norm
        self.clamp = clamp
        self.cut_en = cut_en
        self.aware_test = aware_test
        self.aware = aware

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride=1, bias=not batch_norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.cut1 = Cut(en=self.cut_en)

        if self.batch_norm:
            self.bn1 = norm_layer(width)
            self.bn2 = norm_layer(width)
            self.bn3 = norm_layer(planes * self.expansion)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, bias=not batch_norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.cut2 = Cut(en=self.cut_en)

        self.conv3 = conv1x1(width, planes * self.expansion, bias=not batch_norm)
        self.cut3 = Cut(en=self.cut_en)

        self.downsample = downsample
        if downsample is not None:
            self.cut4 = Cut(en=self.cut_en)

        self.relu3 = nn.ReLU(inplace=True)
        self.cut5 = Cut(en=self.cut_en, in_cut_start=1)

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)

        identity = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        if self.cut_en:
            out = self.cut1(out)
        if self.clamp:
            out = out.clamp(-1, 1)
        if self.aware_test is not None:
            out = FakeQuantizeWithoutLimit.apply(out, self.aware_test)
        if self.aware:
            out = FakeQuantize.apply(out, -1, 1, 128)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        if self.cut_en:
            out = self.cut2(out)
        if self.clamp:
            out = out.clamp(-1, 1)
        if self.aware_test is not None:
            out = FakeQuantizeWithoutLimit.apply(out, self.aware_test)
        if self.aware:
            out = FakeQuantize.apply(out, -1, 1, 128)
        out = self.relu2(out)

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)
        if self.cut_en:
            out = self.cut3(out)
        if self.clamp:
            out = out.clamp(-1, 1)
        if self.aware_test is not None:
            out = FakeQuantizeWithoutLimit.apply(out, self.aware_test)
        if self.aware:
            out = FakeQuantize.apply(out, -1, 1, 128)

        if self.downsample is not None:
            identity = self.downsample(x)
            if self.cut_en:
                identity = self.cut4(identity)
            if self.clamp:
                identity = identity.clamp(-1, 1)
            if self.aware_test is not None:
                identity = FakeQuantizeWithoutLimit.apply(identity, self.aware_test)
            if self.aware:
                identity = FakeQuantize.apply(identity, -1, 1, 128)
        else:
            if self.stride == 2:
                identity = identity[:, :, ::2, ::2]

        out = out + identity
        if self.cut_en:
            out = self.cut5(out)
        if self.clamp:
            out = out.clamp(-1, 1)
        if self.aware_test is not None:
            out = FakeQuantizeWithoutLimit.apply(out, self.aware_test)
        if self.aware:
            out = FakeQuantize.apply(out, -1, 1, 128)
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[Bottleneck]],
            layers: List[int],
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            batch_norm: bool = True,
            clamp: bool = True,
            aware_test=None,
            aware=False,
            cut_en=False
    ) -> None:
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.clamp = clamp
        self.aware_test = aware_test
        self.cut_en = cut_en
        self.aware = aware

        self.inplanes = 64
        self.dilation = 1
        self.batch_norm = batch_norm
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=(3, 3),
                               bias=not batch_norm)
        if self.batch_norm:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_cut = Cut(en=self.cut_en)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))
        if self.batch_norm:
            downsample = nn.Sequential(
                conv1x1(64, 256, stride=1, bias=not self.batch_norm),
                norm_layer(256),
            )
        else:
            downsample = nn.Sequential(
                conv1x1(64, 256, stride=1, bias=not self.batch_norm)
            )
        self.layer1 = nn.Sequential(
            block(inplanes=64, planes=64, stride=1, downsample=downsample, base_width=64, clamp=self.clamp,
                  batch_norm=self.batch_norm, aware_test=self.aware_test, aware=self.aware, cut_en=self.cut_en),
            block(inplanes=256, planes=64, stride=1, downsample=None, base_width=64, clamp=self.clamp,
                  batch_norm=self.batch_norm, aware_test=self.aware_test, aware=self.aware, cut_en=self.cut_en),
            block(inplanes=256, planes=64, stride=2, downsample=None, base_width=64, clamp=self.clamp,
                  batch_norm=self.batch_norm, aware_test=self.aware_test, aware=self.aware, cut_en=self.cut_en),
        )
        self.maxpool_2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_cut = Cut(en=self.cut_en)
        self.drop_out = torch.nn.Dropout(0.5)
        self.fc_0 = nn.Linear(256, 1024)
        self.relu_0 = nn.ReLU(inplace=True)
        self.fc_0_cut = Cut(en=self.cut_en)
        self.fc_1 = nn.Linear(1024, 256)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc_1_cut = Cut(en=self.cut_en)
        self.fc = nn.Linear(256, num_classes)
        self.fc_cut = Cut(en=self.cut_en)

        if self.batch_norm:
            self.fc_bn_0 = nn.BatchNorm1d(1024)
            self.fc_bn_1 = nn.BatchNorm1d(256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data.uniform_(-1, 1)
                m.bias.data.uniform_(-1, 1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, int32_output: bool = True) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.batch_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, not self.batch_norm),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, not self.batch_norm)
                )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer, batch_norm=self.batch_norm, clamp=self.clamp,
                        aware_test=self.aware_test, cut_en=self.cut_en, aware=self.aware)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, batch_norm=self.batch_norm, clamp=self.clamp,
                                aware_test=self.aware_test, cut_en=self.cut_en, aware=self.aware))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

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
            x = self.conv1_cut(x)
        if self.clamp:
            x = x.clamp(-1, 1)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -1, 1, 128)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.maxpool_2(x)

        if self.cut_en:
            x = x.mul(49)

        x = self.avgpool(x)
        if self.cut_en:
            x = self.avgpool_cut(x)
        if self.clamp:
            x = x.clamp(-64 / 49, 64 / 49)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -64 / 49, 64 / 49, 49 * 2)

        x = torch.flatten(x, 1)

        x = self.fc_0(x)
        if self.batch_norm:
            x = self.fc_bn_0(x)
        x = self.relu_0(x)
        if self.cut_en:
            x = self.fc_0_cut(x)
        if self.clamp:
            x = x.clamp(-64 / 49, 64 / 49)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -64 / 49, 64 / 49, 98)

        x = self.fc_1(x)
        if self.batch_norm:
            x = self.fc_bn_1(x)
        x = self.relu_1(x)
        if self.cut_en:
            x = self.fc_1_cut(x)
        if self.clamp:
            x = x.clamp(-64 / 49, 64 / 49)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)
        if self.aware:
            x = FakeQuantize.apply(x, -64 / 49, 64 / 49, 98)

        x = self.fc(x)
        if self.aware_test is not None:
            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)

        if self.cut_en:
            x = self.fc_cut(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def aware_parameters(self, ckpt=0):
        weight_scale_dict = QuantizationConfig('weight', ckpt=ckpt)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                factor = weight_scale_dict[name]
                assert factor is not None
                module.weight.data = FakeQuantize.apply(module.weight.data, -128 / factor, 128 / factor, factor)
                if isinstance(module, nn.Linear):
                    module.bias.data = FakeQuantizeWithoutLimit.apply(module.bias.data, factor * 98)
                else:
                    module.bias.data = FakeQuantizeWithoutLimit.apply(module.bias.data, factor * 128)

    def quantize(self, ckpt=0):
        in_cut_start_dict = QuantizationConfig('in_cut_start', ckpt=ckpt)
        for name, module in self.named_modules():
            if isinstance(module, Cut):
                module.in_cut_start = in_cut_start_dict[name]

        # # 量化权重数据
        # weight_scale_dict = QuantizationConfig('weight', ckpt=0)
        # for name, module in self.named_modules():
        #     if isinstance(module, (nn.Conv2d, nn.Linear)):
        #         assert weight_scale_dict[name] is not None, name + ' cannot be found in the quantization config'
        #         module.weight.data = module.weight.data.mul(weight_scale_dict[name]).round().clamp(-128, 127)
        #         # module.weight.data = module.weight.data.mul(weight_scale_dict[name])
        #         if isinstance(module, nn.Linear):
        #             module.bias.data = module.bias.data.mul(weight_scale_dict[name] * 98).round().clamp(
        #                 -0x80000000, 0x7fffffff)
        #         else:
        #             module.bias.data = module.bias.data.mul(weight_scale_dict[name] * 128).round().clamp(
        #                 -0x80000000, 0x7fffffff)
        #         # module.bias.data = module.bias.data.mul(weight_scale_dict[name] * 128)


def _resnet(
        arch: str,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        pretrained: str,
        batch_norm: bool = True,
        clamp: bool = False,
        aware_test=None,
        aware=False,
        cut_en=False
) -> ResNet:
    model = ResNet(block, layers, batch_norm=batch_norm, clamp=clamp, aware_test=aware_test, cut_en=cut_en, aware=aware)
    if pretrained is not None:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained, map_location=device)
        if pretrained_dict.get('state_dict') is not None:
            pretrained_dict = pretrained_dict['state_dict']
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet50_1chip(pretrained: str = None, batch_norm: bool = True, clamp: bool = False,
             aware_test=None, cut_en=False, aware=False) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 0, 0, 0], pretrained,
                   batch_norm=batch_norm, clamp=clamp, aware_test=aware_test, cut_en=cut_en,
                   aware=aware)


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    FUSE_BN = False
    if FUSE_BN:
        pretrained_path = 'checkpoints/resnet50_fuse_bn.pth'
    else:
        pretrained_path = 'checkpoints/resnet50_75.388000.pth'
    x = torch.randn((1, 3, 224, 224))
    model = resnet50_1chip(pretrained=None, batch_norm=False, clamp=False)
    print(model)
    y = model(x)
    print(y.shape)
