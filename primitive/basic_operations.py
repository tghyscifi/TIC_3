import torch
import warnings
from typing import TypeVar, Union, Tuple, Optional
from math import floor
import numpy as np


class BasicOperations:
    @ staticmethod
    def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, kernel_size: Tuple[int, int],
               padding: Tuple[int, int, int, int], stride: Tuple[int, int], dilation: Tuple[int, int]):
        """
        x: [H, W, C]
        weight: ky, kx, r, f
        padding: top, left, down, right
        kernel_size: kh, kw (ky, kx)
        stride: sh, sw (sy, sx)
        dilation: eh, ew (ey, ex)
        """
        # when x input channel is aligned to 16, additional zeros are not needed!
        if x.shape[2] > weight.shape[2]:
            x = x[:, :, 0:weight.shape[2]]

        x_with_pad = BasicOperations.add_pad(x, padding)

        x_with_pad = torch.tensor(x_with_pad, dtype=torch.float64, requires_grad=False).permute(2, 0, 1)      # C, H, W
        weight = torch.tensor(weight, dtype=torch.float64, requires_grad=False).permute(3, 2, 0, 1)
        bias = torch.tensor(bias, dtype=torch.float64, requires_grad=False)

        x_with_pad = x_with_pad.unsqueeze(0)        # [N C H W]

        ouput_h = floor((x_with_pad.shape[2] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1
        ouput_w = floor((x_with_pad.shape[3] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1

        # [N, C, H, W] -> [N, C * Kh * Kw, h_out * w_out]
        unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
        # [N, C * Kh * Kw, h_out * w_out] -> [N, C, H, W]
        fold = torch.nn.Fold(output_size=(ouput_h, ouput_w), kernel_size=(1, 1), dilation=1, padding=0, stride=1)

        m = unfold(x_with_pad)
        n = torch.einsum('ijk, kl -> ijkl', m.transpose(1, 2), weight.reshape(weight.shape[0], -1).t())

        result = bias.clone().detach().repeat(n.shape[1], 1)

        for i in range(n.shape[2]):
            result += n[0, :, i, :]
            up_over_flow = torch.where(result > 0x7fffffff)
            result[up_over_flow] = 0x7fffffff
            down_over_flow = torch.where(result < -0x80000000)
            result[down_over_flow] = -0x80000000
            if len(up_over_flow[0]) + len(down_over_flow[0]) > 0:
                warnings.warn('Overflow! result may not be as expected!')

        result.unsqueeze_(0).transpose_(1, 2)

        result = fold(result)

        return result

    @staticmethod
    def add_pad(x: np.ndarray, padding: Tuple[int, int, int, int]):
        """
        x: [H, W, C]
        padding: top, left, down, right
        """
        if padding[0] + padding[1] + padding[2] + padding[3] > 0:
            x_with_pad = np.zeros((x.shape[0] + padding[0] + padding[2], x.shape[1] + padding[1] + padding[3],
                                   x.shape[2]), dtype=np.int64)
            x_with_pad[padding[0]: x_with_pad.shape[0]-padding[2], padding[1]: x_with_pad.shape[1]-padding[3], :] = x
        else:
            x_with_pad = x
        return x_with_pad

    @staticmethod
    def convert_type(x: np.ndarray, type_in: int, type_out: int, in_cut_start: int):
        assert (0 <= in_cut_start <= 15)
        x = x >> (in_cut_start * 2)
        if type_in == 0 and type_out == 1:
            np.clip(x, a_min=-0x80, a_max=0x7f, out=x)
        elif type_in == 0 and type_out == 3:
            np.clip(x, a_min=-1, a_max=1, out=x)
        elif type_in == 1 and type_out == 3:
            np.clip(x, a_min=-1, a_max=1, out=x)
        else:
            raise ValueError
        return x

    @staticmethod
    def max_min_pool(pic_mode: int, x: np.ndarray, type_in: int, type_out: int, cmp_c: int,
                     kernel_size: Tuple[int, int], stride: Tuple[int, int]):
        """
        x: [H, W, C]
        kernel_size: kh, kw (ky, kx)
        stride: sh, sw (sy, sx)
        """
        if pic_mode == 0:
            m_fn, mm_fn = np.max, np.maximum
        elif pic_mode == 1:
            m_fn, mm_fn = np.min, np.minimum
        else:
            raise ValueError
        (ih, iw, ic) = x.shape
        oh = (ih - kernel_size[0]) // stride[0] + 1
        ow = (iw - kernel_size[1]) // stride[1] + 1
        result = np.zeros((oh, ow, ic), dtype=np.int64)
        if (type_in == 0 and type_out == 1) or (type_in == 1 and type_out == 1):
            cmp_array = np.array([(cmp_c >> (i * 8)) & 0xff for i in range(4)], dtype=np.int8)
        elif (type_in == 0 and type_out == 3) or (type_in == 1 and type_out == 3) or (type_in == 3 and type_out == 3):
            cmp_array = np.array([(cmp_c >> (i * 2)) & 0x3 for i in range(16)], dtype=np.int8)
            cmp_array[np.where(cmp_array > 1)] = -1
        elif type_in == 0 and type_out == 0:
            cmp_c = cmp_c - 0x100000000 if cmp_c > 0x7fffffff else cmp_c
            cmp_array = np.array([cmp_c], dtype=np.int32)
        elif type_in == 3 and type_out == 1:
            cmp_array = np.array([(cmp_c >> (i * 2)) & 0x3 for i in range(16)], dtype=np.int8)
            cmp_array[np.where(cmp_array > 1)] = -1
        else:
            raise ValueError
        assert (ic % len(cmp_array) == 0)
        cmp_array = np.tile(cmp_array, ic // len(cmp_array))
        for h_idx in range(oh):
            for w_idx in range(ow):
                result[h_idx, w_idx, :] = mm_fn(
                    cmp_array, m_fn(m_fn(x[h_idx * stride[0]: h_idx * stride[0] + kernel_size[0],
                                         w_idx * stride[1]: w_idx * stride[1] + kernel_size[1], :], axis=0), axis=0))
        return result

    @staticmethod
    def avg_pool(x: np.ndarray, kernel_size: Tuple[int, int], stride: Tuple[int, int], bias: Union[np.ndarray, int]):
        """
        x: [H, W, C]
        kernel_size: kh, kw (ky, kx)
        stride: sh, sw (sy, sx)
        """
        (ih, iw, ic) = x.shape
        oh = (ih - kernel_size[0]) // stride[0] + 1
        ow = (iw - kernel_size[1]) // stride[1] + 1
        if type(bias) == int:
            result = np.ones((oh, ow, ic), dtype=np.int64) * bias
        elif type(bias) == np.ndarray:
            assert (len(bias.shape) == 1 and ic % len(bias) == 0)
            result = np.ones((oh, ow, ic), dtype=np.int64) * np.tile(bias, ic // len(bias))
        else:
            raise TypeError
        for kh_idx in range(kernel_size[0]):
            for kw_idx in range(kernel_size[1]):
                result += x[kh_idx:ih-(
                        kernel_size[0]-1-kh_idx):stride[0], kw_idx:iw-(kernel_size[1]-1-kw_idx):stride[1], :]
                up_over_flow = np.where(result > 0x7fffffff)
                result[up_over_flow] = 0x7fffffff
                down_over_flow = np.where(result < -0x80000000)
                result[down_over_flow] = -0x80000000
                if len(up_over_flow[0]) + len(down_over_flow[0]) > 0:
                    warnings.warn('Overflow! result may not be as expected!')
        return result

    @staticmethod
    def tensor_sum(x: np.ndarray, bias: Union[np.ndarray, int]):
        """
        x: [N, H, W, C]
        """
        (n, ih, iw, ic) = x.shape
        if type(bias) == int:
            result = np.ones((ih, iw, ic), dtype=np.int64) * bias
        elif type(bias) == np.ndarray:
            assert (len(bias.shape) == 1 and ic % len(bias) == 0)
            if ic != len(bias):
                raise NotImplementedError('this situation is not supported by the original python simulator!')
            result = np.ones((ih, iw, ic), dtype=np.int64) * np.tile(bias, ic // len(bias))
        else:
            raise TypeError
        for n_idx in range(n):
            result += x[n_idx]
            up_over_flow = np.where(result > 0x7fffffff)
            result[up_over_flow] = 0x7fffffff
            down_over_flow = np.where(result < -0x80000000)
            result[down_over_flow] = -0x80000000
            if len(up_over_flow[0]) + len(down_over_flow[0]) > 0:
                warnings.warn('Overflow! result may not be as expected!')
        return result

    @staticmethod
    def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
        """
        x: [cin]
        weight: [cin, cout]
        bias: [cout]
        """
        if x.shape[0] > weight.shape[0]:        # when x is aligned to 16B, additional zeros are not needed!
            x = x[0:weight.shape[0]]
        n = weight * x.reshape(-1, 1).repeat(weight.shape[1], axis=1)
        result = bias.reshape((1, -1)).copy()
        for i in range(n.shape[0]):
            result += n[i, :]
            up_over_flow = np.where(result > 0x7fffffff)
            result[up_over_flow] = 0x7fffffff
            down_over_flow = np.where(result < -0x80000000)
            result[down_over_flow] = -0x80000000
            if len(up_over_flow[0]) + len(down_over_flow[0]) > 0:
                warnings.warn('Overflow! result may not be as expected!')
        result = result.ravel()
        return result


if __name__ == '__main__':
    kernel = (7, 7)
    pad = (3, 3)
    ss = (2, 2)
    dd = (1, 1)

    a = torch.nn.Conv2d(3, 64, kernel_size=kernel, padding=pad, stride=ss, dilation=dd).double()
    a.weight.data.mul_(100).floor_()
    a.bias.data.mul_(999).floor_()

    b = torch.randn((1, 3, 224, 224)).double() * 1000
    b.floor_()

    c = a(b)

    p = BasicOperations.conv2d(b.squeeze(0).permute(1, 2, 0), a.weight.data, a.bias.data, kernel,
                               padding=(pad[0], pad[1], pad[0], pad[1]), stride=ss, dilation=dd)

    print(torch.sum(p - c))

    # avg pool test
    # avp_x = torch.randn((1, 64, 200, 400)).double() * 10000
    # avp_x.floor_()
    #
    # kk = (56, 7)
    # ss = (3, 2)
    #
    # avg = torch.nn.AvgPool2d(kernel_size=kk, stride=ss).double()
    #
    # ref_y = np.array(avg(avp_x)).squeeze(0).transpose(1, 2, 0)
    #
    # y = BasicOperations.avg_pool(np.array(avp_x, dtype=np.int64).squeeze(0).transpose(1, 2, 0),
    #                              kernel_size=kk, stride=ss, bias=np.arange(16))
    #
    # print(np.max(np.abs(ref_y - y)))
