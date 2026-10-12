import torch

torch.manual_seed(1)
x = torch.randn((1, 16, 3, 3))  # n, c, h ,w

sh, sw = 1, 4
# 无pad
# conv = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(sh, sw), padding=(0, 0),
#                        groups=16, bias=False)
#
# y_f = conv(x)
#
# weight = conv.weight.data.detach()  # cout cin kh, kw
# y1 = x[:, :, 0 * sh::sh, 0 * sw::sw] * weight[:, :, 0, 0].reshape(1, 16, 1, 1)
# y2 = x[:, :, 0 * sh::sh, 1 * sw::sw] * weight[:, :, 0, 1].reshape(1, 16, 1, 1)
# y3 = x[:, :, 0 * sh::sh, 2 * sw::sw] * weight[:, :, 0, 2].reshape(1, 16, 1, 1)
# y4 = x[:, :, 1 * sh::sh, 0 * sw::sw] * weight[:, :, 1, 0].reshape(1, 16, 1, 1)
# y5 = x[:, :, 1 * sh::sh, 1 * sw::sw] * weight[:, :, 1, 1].reshape(1, 16, 1, 1)
# y6 = x[:, :, 1 * sh::sh, 2 * sw::sw] * weight[:, :, 1, 2].reshape(1, 16, 1, 1)
# y7 = x[:, :, 2 * sh::sh, 0 * sw::sw] * weight[:, :, 2, 0].reshape(1, 16, 1, 1)
# y8 = x[:, :, 2 * sh::sh, 1 * sw::sw] * weight[:, :, 2, 1].reshape(1, 16, 1, 1)
# y9 = x[:, :, 2 * sh::sh, 2 * sw::sw] * weight[:, :, 2, 2].reshape(1, 16, 1, 1)
#
# (_, _, oh, ow) = y9.shape
#
# y = y1[:, :, :oh, :ow] + y2[:, :, :oh, :ow] + y3[:, :, :oh, :ow] + y4[:, :, :oh, :ow] + y5[:, :, :oh, :ow] + \
#     y6[:, :, :oh, :ow] + y7[:, :, :oh, :ow] + y8[:, :, :oh, :ow] + y9[:, :, :oh, :ow]
#
# print(torch.sum(y - y_f))


conv = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(sh, sw), padding=(1, 1), groups=16,
                       bias=False)

y_f = conv(x)

weight = conv.weight.data.detach()  # cout cin kh, kw
y1 = x[:, :, 0 * sh::sh, 0 * sw::sw] * weight[:, :, 0, 0].reshape(1, 16, 1, 1)
y2 = x[:, :, 0 * sh::sh, 1 * sw::sw] * weight[:, :, 0, 1].reshape(1, 16, 1, 1)
y3 = x[:, :, 0 * sh::sh, 2 * sw::sw] * weight[:, :, 0, 2].reshape(1, 16, 1, 1)
y4 = x[:, :, 1 * sh::sh, 0 * sw::sw] * weight[:, :, 1, 0].reshape(1, 16, 1, 1)

y5 = x[:, :, 1 * sh::sh, 1 * sw::sw] * weight[:, :, 1, 1].reshape(1, 16, 1, 1)
y6 = x[:, :, 1 * sh::sh, 2 * sw::sw] * weight[:, :, 1, 2].reshape(1, 16, 1, 1)

y7 = x[:, :, 2 * sh::sh, 0 * sw::sw] * weight[:, :, 2, 0].reshape(1, 16, 1, 1)

y8 = x[:, :, 2 * sh::sh, 1 * sw::sw] * weight[:, :, 2, 1].reshape(1, 16, 1, 1)
y9 = x[:, :, 2 * sh::sh, 2 * sw::sw] * weight[:, :, 2, 2].reshape(1, 16, 1, 1)

(_, _, oh, ow) = y9.shape

y = y1[:, :, :oh, :ow] + y2[:, :, :oh, :ow] + y3[:, :, :oh, :ow] + y4[:, :, :oh, :ow] + y5[:, :, :oh, :ow] + \
    y6[:, :, :oh, :ow] + y7[:, :, :oh, :ow] + y8[:, :, :oh, :ow] + y9[:, :, :oh, :ow]

print(torch.sum(y - y_f))

xxx = 1
