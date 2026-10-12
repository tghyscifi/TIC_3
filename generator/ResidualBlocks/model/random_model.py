import torch
import torch.nn as nn
import os
import numpy as np
import random
from generator.ResidualBlocks.model.quantized_model import QuantizNet, QuantizationConfig
import cv2

cin = 32
cout = 32
pad = 1
feature_in_W = 90
feature_in_H = 280
kw = 3


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Cut(nn.Module):
    def __init__(self, en: bool = False, in_cut_start: int = 2, type_out: int = 1):
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


class Residual_Blocks(nn.Module):
    def __init__(self):
        super(Residual_Blocks, self).__init__()
        # 此处顺序要和网络中运行的顺序一样
        self.conv_in = nn.Conv2d(4, cout, kw, 1, pad, bias=True)
        self.cut_in = Cut(True, 4, 1)
        self.activation_in = nn.ReLU()

        self.conv0 = nn.Conv2d(cin, cout, kw, 1, pad, bias=True)
        self.cut0 = Cut(True, 3, 1)
        self.activation0 = nn.ReLU()
        self.conv1 = nn.Conv2d(cin, cout, kw, 1, pad, bias=True)
        self.cut1 = Cut(True, 3, 1)
        self.activation1 = nn.ReLU()
        self.cut_0 = Cut(True, 0, 1)

        self.conv2 = nn.Conv2d(cin, cout, kw, 1, pad, bias=True)
        self.cut2 = Cut(True, 4, 1)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv2d(cin, cout, kw, 1, pad, bias=True)
        self.cut3 = Cut(True, 4, 1)
        self.activation3 = nn.ReLU()
        self.cut_1 = Cut(True, 0, 1)

        self.conv4 = nn.Conv2d(cin, cout, kw, 1, pad, bias=True)
        self.cut4 = Cut(True, 4, 1)
        self.activation4 = nn.ReLU()
        self.conv5 = nn.Conv2d(cin, cout, kw, 1, pad, bias=True)
        self.cut5 = Cut(True, 3, 1)
        self.activation5 = nn.ReLU()
        self.cut_2 = Cut(True, 0, 1)

        self.conv_out = nn.Conv2d(cin, 6, kw, 1, pad, bias=True)
        self.cut_out = Cut(True, 5, 1)
        self.activation_out = nn.Tanh()

    def forward(self, x):
        # input layer
        x0 = self.conv_in(x)
        x0 = self.cut_in(x0)
        x0 = self.activation_in(x0)
        # print(x0)

        # residual block 0
        x1 = self.conv0(x0)
        x1 = self.cut0(x1)
        x1 = self.activation0(x1)
        x1 = self.conv1(x1)
        x1 = self.cut1(x1)
        x1 = self.activation1(x1)
        x1 = self.cut_0(x0 + x1)


        # residual block 1
        x2 = self.conv2(x1)
        x2 = self.cut2(x2)
        x2 = self.activation2(x2)
        x2 = self.conv3(x2)
        x2 = self.cut3(x2)
        x2 = self.activation3(x2)
        x2 = self.cut_1(x2 + x1)

        # residual block 2
        x3 = self.conv4(x2)
        x3 = self.cut4(x3)
        x3 = self.activation4(x3)
        x3 = self.conv5(x3)
        x3 = self.cut5(x3)
        x3 = self.activation5(x3)
        x3 = self.cut_2(x2 + x3)

        # output layer
        # x2 = torch.cat((x1, x), 1)
        x4 = x3
        x4 = self.conv_out(x4)
        x4 = self.cut_out(x4)
        x4 = x4 / 32
        x4 = self.activation_out(x4)
        x4 = torch.floor((x4 * 128).clamp(min=-128, max=127))

        return x4


def random_data_gen():
    seed_torch(7)
    input_rand = torch.randint(-16, 16, (1, 4, feature_in_H, feature_in_W))
    weight_input_rand = torch.randint(-16, 16, (cout, 4, kw, kw))

    weight_rand = torch.randint(-16, 16, (cout, cin, kw, kw))
    bias_rand = torch.randint(-0x0800, 0x0800, [cout])

    weight_output_rand = torch.randint(-16, 16, (6, cin, kw, kw))
    bias_output_rand = torch.randint(-0x0800, 0x0800, [6])

    seed_torch(42)
    weight_rand_1 = torch.randint(-16, 16, (cout, cin, kw, kw))
    bias_rand_1 = torch.randint(-0x0800, 0x0800, [cout])

    seed_torch(17)
    weight_rand_2 = torch.randint(-16, 16, (cout, cin, kw, kw))
    bias_rand_2 = torch.randint(-0x0800, 0x0800, [cout])

    seed_torch(7)

    rb = Residual_Blocks()
    rb.eval()

    with torch.no_grad():
        rb.state_dict()['conv_in.weight'].copy_(weight_input_rand.to(torch.float))
        rb.state_dict()['conv_in.bias'].copy_(bias_rand.to(torch.float))

        rb.state_dict()['conv0.weight'].copy_(weight_rand.to(torch.float))
        rb.state_dict()['conv0.bias'].copy_(bias_rand.to(torch.float))
        rb.state_dict()['conv1.weight'].copy_(weight_rand.to(torch.float))
        rb.state_dict()['conv1.bias'].copy_(bias_rand.to(torch.float))

        rb.state_dict()['conv2.weight'].copy_(weight_rand_1.to(torch.float))
        rb.state_dict()['conv2.bias'].copy_(bias_rand_1.to(torch.float))
        rb.state_dict()['conv3.weight'].copy_(weight_rand_1.to(torch.float))
        rb.state_dict()['conv3.bias'].copy_(bias_rand_1.to(torch.float))

        rb.state_dict()['conv4.weight'].copy_(weight_rand_2.to(torch.float))
        rb.state_dict()['conv4.bias'].copy_(bias_rand_2.to(torch.float))
        rb.state_dict()['conv5.weight'].copy_(weight_rand_2.to(torch.float))
        rb.state_dict()['conv5.bias'].copy_(bias_rand_2.to(torch.float))

        rb.state_dict()['conv_out.weight'].copy_(weight_output_rand.to(torch.float))
        rb.state_dict()['conv_out.bias'].copy_(bias_output_rand.to(torch.float))
        input_act = input_rand.to(torch.float)
        output_act = rb(input_act)

    return input_act, output_act, rb

def model_data_gen():
    seed_torch(7)
    # input_rand = torch.randint(-128, 127, (1, 4, feature_in_H, feature_in_W)).to(torch.float)
    input = np.load("generator/ResidualBlocks/data/rgbd_real.npy")[:,:,250:250+feature_in_H,880:880+feature_in_W] * 255
    # print(input)
    # amp = np.round(input[0,0:3,:,:]).astype(np.uint8)
    # cv2.imshow('amp', np.transpose(np.squeeze(amp), [1, 2, 0]))
    # cv2.waitKey()
    # exit()
    input_rand = torch.Tensor(input).to(torch.float) - 128
    qconfig = QuantizationConfig()
    net = QuantizNet(qconfig)

    net.load_state_dict(torch.load("generator/ResidualBlocks/model/pretrained_model/ckpt0", map_location='cpu'))
    net.quantize()

    rb = Residual_Blocks()
    rb.eval()

    with torch.no_grad():
        rb.state_dict()['conv_in.weight'].copy_(net.state_dict()['conv_in.weight'])
        rb.state_dict()['conv_in.bias'].copy_(net.state_dict()['conv_in.bias'])

        rb.state_dict()['conv0.weight'].copy_(net.state_dict()['res0.conv1.weight'])
        rb.state_dict()['conv0.bias'].copy_(net.state_dict()['res0.conv1.bias'])
        rb.state_dict()['conv1.weight'].copy_(net.state_dict()['res0.conv2.weight'])
        rb.state_dict()['conv1.bias'].copy_(net.state_dict()['res0.conv2.bias'])

        rb.state_dict()['conv2.weight'].copy_(net.state_dict()['res1.conv1.weight'])
        rb.state_dict()['conv2.bias'].copy_(net.state_dict()['res1.conv1.bias'])
        rb.state_dict()['conv3.weight'].copy_(net.state_dict()['res1.conv2.weight'])
        rb.state_dict()['conv3.bias'].copy_(net.state_dict()['res1.conv2.bias'])

        rb.state_dict()['conv4.weight'].copy_(net.state_dict()['res2.conv1.weight'])
        rb.state_dict()['conv4.bias'].copy_(net.state_dict()['res2.conv1.bias'])
        rb.state_dict()['conv5.weight'].copy_(net.state_dict()['res2.conv2.weight'])
        rb.state_dict()['conv5.bias'].copy_(net.state_dict()['res2.conv2.bias'])

        rb.state_dict()['conv_out.weight'].copy_(net.state_dict()['conv_out.weight'])
        rb.state_dict()['conv_out.bias'].copy_(net.state_dict()['conv_out.bias'])
        input_act = input_rand.to(torch.float)
        output_act = rb(input_act)
        # print(net(input_act))
        assert((output_act == net(input_act)).all())
        # output_act = net(input_rand)

    return input_rand, output_act, rb


if __name__ == "__main__":
    # i, o, model = random_data_gen()
    i, o, model = model_data_gen()

    amp = o[:,0:3,:,:]/128 * 0.5 ** 0.5 + 0.5 ** 0.5
    phs = o[:,3:6,:,:]/128 * 0.5 + 0.5

    phs = phs - torch.mean(phs, [2, 3], True) + 0.5

    amp2 = np.transpose(np.squeeze(amp[0, :, :, :].numpy()), [1, 2, 0])
    phs2 = np.transpose(np.squeeze(phs[0, :, :, :].numpy()), [1, 2, 0])

    cv2.imwrite("generator/ResidualBlocks/data/phs.png", phs2*255)
    cv2.imwrite("generator/ResidualBlocks/data/amp.png", amp2*255)


    # print(i)
    print(o)
