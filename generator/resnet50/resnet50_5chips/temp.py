import torch
import warnings

"""
    ['conv1', 'relu', 'cut1', 'maxpool', 'layer1.0.conv1', 'layer1.0.relu1', 'layer1.0.cut1', 'layer1.0.conv2',
     'layer1.0.relu2', 'layer1.0.cut2', 'layer1.0.conv3', 'layer1.0.relu3', 'layer1.0.cut3', 'layer1.0.downsample.0',
     'layer1.0.cut4', 'layer1.0.cut5', 'layer1.1.conv1', 'layer1.1.relu1', 'layer1.1.cut1', 'layer1.1.conv2',
     'layer1.1.relu2', 'layer1.1.cut2', 'layer1.1.conv3', 'layer1.1.relu3', 'layer1.1.cut3', 'layer1.1.cut5',
     'layer1.2.conv1', 'layer1.2.relu1', 'layer1.2.cut1', 'layer1.2.conv2', 'layer1.2.relu2', 'layer1.2.cut2',
     'layer1.2.conv3', 'layer1.2.relu3', 'layer1.2.cut3', 'layer1.2.cut5', 'layer2.0.conv1', 'layer2.0.relu1',
     'layer2.0.cut1', 'layer2.0.conv2', 'layer2.0.relu2', 'layer2.0.cut2', 'layer2.0.conv3', 'layer2.0.relu3',
     'layer2.0.cut3', 'layer2.0.downsample.0', 'layer2.0.cut4', 'layer2.0.cut5', 'layer2.1.conv1', 'layer2.1.relu1',
     'layer2.1.cut1', 'layer2.1.conv2', 'layer2.1.relu2', 'layer2.1.cut2', 'layer2.1.conv3', 'layer2.1.relu3',
     'layer2.1.cut3', 'layer2.1.cut5', 'layer2.2.conv1', 'layer2.2.relu1', 'layer2.2.cut1', 'layer2.2.conv2',
     'layer2.2.relu2', 'layer2.2.cut2', 'layer2.2.conv3', 'layer2.2.relu3', 'layer2.2.cut3', 'layer2.2.cut5',
     'layer2.3.conv1', 'layer2.3.relu1', 'layer2.3.cut1', 'layer2.3.conv2', 'layer2.3.relu2', 'layer2.3.cut2',
     'layer2.3.conv3', 'layer2.3.relu3', 'layer2.3.cut3', 'layer2.3.cut5', 'layer3.0.conv1', 'layer3.0.relu1',
     'layer3.0.cut1', 'layer3.0.conv2', 'layer3.0.relu2', 'layer3.0.cut2', 'layer3.0.conv3', 'layer3.0.relu3',
     'layer3.0.cut3', 'layer3.0.downsample.0', 'layer3.0.cut4', 'layer3.0.cut5', 'layer3.1.conv1', 'layer3.1.relu1',
     'layer3.1.cut1', 'layer3.1.conv2', 'layer3.1.relu2', 'layer3.1.cut2', 'layer3.1.conv3', 'layer3.1.relu3',
     'layer3.1.cut3', 'layer3.1.cut5', 'layer3.2.conv1', 'layer3.2.relu1', 'layer3.2.cut1', 'layer3.2.conv2',
     'layer3.2.relu2', 'layer3.2.cut2', 'layer3.2.conv3', 'layer3.2.relu3', 'layer3.2.cut3', 'layer3.2.cut5',
     'layer3.3.conv1', 'layer3.3.relu1', 'layer3.3.cut1', 'layer3.3.conv2', 'layer3.3.relu2', 'layer3.3.cut2',
     'layer3.3.conv3', 'layer3.3.relu3', 'layer3.3.cut3', 'layer3.3.cut5', 'layer3.4.conv1', 'layer3.4.relu1',
     'layer3.4.cut1', 'layer3.4.conv2', 'layer3.4.relu2', 'layer3.4.cut2', 'layer3.4.conv3', 'layer3.4.relu3',
     'layer3.4.cut3', 'layer3.4.cut5', 'layer3.5.conv1', 'layer3.5.relu1', 'layer3.5.cut1', 'layer3.5.conv2',
     'layer3.5.relu2', 'layer3.5.cut2', 'layer3.5.conv3', 'layer3.5.relu3', 'layer3.5.cut3', 'layer3.5.cut5',
     'layer4.0.conv1', 'layer4.0.relu1', 'layer4.0.cut1', 'layer4.0.conv2', 'layer4.0.relu2', 'layer4.0.cut2',
     'layer4.0.conv3', 'layer4.0.relu3', 'layer4.0.cut3', 'layer4.0.downsample.0', 'layer4.0.cut4', 'layer4.0.cut5',
     'layer4.1.conv1', 'layer4.1.relu1', 'layer4.1.cut1', 'layer4.1.conv2', 'layer4.1.relu2', 'layer4.1.cut2',
     'layer4.1.conv3', 'layer4.1.relu3', 'layer4.1.cut3', 'layer4.1.cut5', 'layer4.2.conv1', 'layer4.2.relu1',
     'layer4.2.cut1', 'layer4.2.conv2', 'layer4.2.relu2', 'layer4.2.cut2', 'layer4.2.conv3', 'layer4.2.relu3',
     'layer4.2.cut3', 'layer4.2.cut5', 'avgpool', 'cut2', 'fc', 'cut3']
    """

kernel = (3, 3)
padding = (6, 0)
stride = (6, 1)
dilation = (1, 7)

a = torch.nn.Conv2d(3, 64, kernel_size=kernel, padding=padding, stride=stride, dilation=dilation).double()
a.weight.data.mul_(100).floor_()
a.bias.data.mul_(999).floor_()

b = torch.randn((1, 3, 224, 995)).double() * 1000
b.floor_()

c = a(b)

unfold = torch.nn.Unfold(kernel_size=kernel, dilation=dilation, padding=padding, stride=stride)
fold = torch.nn.Fold(output_size=(c.shape[2], c.shape[3]), kernel_size=(1, 1), dilation=1, padding=0,
                     stride=1)

m = unfold(b)
n = torch.einsum('ijk, kl -> ijkl', m.transpose(1, 2),
                 a.weight.data.view(a.weight.data.shape[0], -1).t())

result = a.bias.data.clone().repeat(n.shape[1], 1)

for i in range(n.shape[2]):
    result += n[0, :, i, :]
    up_over_flow = torch.where(result > 2147483647)
    result[up_over_flow] = 2147483647
    down_over_flow = torch.where(result < -2147483648)
    result[down_over_flow] = -2147483648
    if len(up_over_flow[0]) + len(down_over_flow[0]) > 0:
        warnings.warn('Overflow! result may not be as expected!')

result.unsqueeze_(0).transpose_(1, 2)

p = fold(result)

print(torch.sum(p - c))


w = 1
