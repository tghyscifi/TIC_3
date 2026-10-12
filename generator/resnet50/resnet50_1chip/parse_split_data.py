import os
import numpy as np
import matplotlib.pyplot as plt
from generator.resnet50.resnet50_1chip.resnet50_1chip_data_handler import ResNetDataHandler
import torch


def str2list(str: str, type=1):
    if type == 1:
        assert len(str) % 2 == 0
        length = len(str) // 2
        result = []
        for i in range(length):
            result.append(int(str[(length - i) * 2 - 2:(length - i) * 2], 16))
        # print(str, np.array(result, dtype=np.int8))
        return np.array(result, dtype=np.int8)
    else:
        raise NotImplemented


def list2str(x: np.ndarray, type=1):
    if type == 1:
        result = ['{:0>2x}'.format(i) for i in x]
        return result
    else:
        raise NotImplemented


def parse_each_data(file_name, shape=(229, 128)):
    result = np.empty(shape, dtype=np.int8)
    with open(file_name, 'r') as rf:
        idx = 0
        for line in rf.readlines():
            if line == '\n':
                continue
            h, w = idx // 16, idx % 16
            sub_list = str2list(line[4:20])
            result[h][w * 8: w * 8 + 8] = sub_list
            idx += 1
    assert idx == 3664
    return result


def recover_image(file_names):
    r0 = parse_each_data(file_names[0])
    r1 = parse_each_data(file_names[1])
    g0 = parse_each_data(file_names[2])
    g1 = parse_each_data(file_names[3])
    b0 = parse_each_data(file_names[4])
    b1 = parse_each_data(file_names[5])

    if not (r0[:, 102:] == r1[:, :26]).all():
        print('r error')
    if not (g0[:, 102:] == g1[:, :26]).all():
        print('g error')
    if not (b0[:, 102:] == b1[:, :26]).all():
        print('b error')
    img = np.zeros((3, 229, 230)).astype(np.int8)
    img[0, :, 0:128] = r0
    img[0, :, 128:] = r1[:, 26:]
    img[1, :, 0:128] = g0
    img[1, :, 128:] = g1[:, 26:]
    img[2, :, 0:128] = b0
    img[2, :, 128:] = b1[:, 26:]
    return img


def show_img(img):
    # img = np.random.randint(0, 255, (600, 400, 3), dtype=np.uint8)
    img = (img + 128).astype(np.uint8).transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()


def inference(img):
    """
        img: [C, H, W]
    """
    handler = ResNetDataHandler(ckpt=0)
    assert img.shape == (3, 229, 230)
    assert (img[:, :3, :] == 0).all()
    assert (img[:, -2:, :] == 0).all()
    assert (img[:, :, :3] == 0).all()
    assert (img[:, :, -2:] == 0).all()
    x = torch.tensor(img, dtype=torch.float).div(128)
    x = x.unsqueeze(0)
    y = handler.inference(x[:, :, 3:-2, 3:-3])
    return y


if __name__ == '__main__':
    # 'n02086079': 0, 狮子狗
    # 'n02123159': 1, 山猫, 虎猫
    # 'n02391049': 2, 斑马
    # 'n02403003': 3, 牛
    # 'n02917067': 4, 动车, 子弹头列车
    # 'n03594945': 5, 吉普车
    # 'n03673027': 6, 远洋班轮
    # 'n03791053': 7, 摩托车
    # 'n12144580': 8, 玉米
    class_name = '船'
    dir_names_0 = [
        'temp/resnet/' + class_name + 'R_l_pkg.txt',
        'temp/resnet/' + class_name + 'R_r_pkg.txt',
        'temp/resnet/' + class_name + 'G_l_pkg.txt',
        'temp/resnet/' + class_name + 'G_r_pkg.txt',
        'temp/resnet/' + class_name + 'B_l_pkg.txt',
        'temp/resnet/' + class_name + 'B_r_pkg.txt',
    ]
    image = recover_image(dir_names_0)
    # show_img(image)
    res = inference(image)
    print(res[0].detach())
    print(list2str(res.detach().numpy()[0].astype(np.uint8)))
