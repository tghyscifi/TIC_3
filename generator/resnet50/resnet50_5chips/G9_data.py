from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler
import numpy as np


def generate_g9_data(handler, size_y, size_x):
    data = {
        'layer3.0.conv1': {},  # L2
        'layer3.0.relu1': {},
        'layer3.0.cut1': {},
        'layer3.0.conv2': {},  # l24
        'layer3.0.relu2': {},
        'layer3.0.cut2': {},
        'layer3.0.conv3': {},  # L4
        'layer3.0.cut3': {},
        'layer3.0.downsample.0': {},  # L4e
        'layer3.0.cut4': {},
        'layer3.0.relu3': {},
        'layer3.0.cut5': {}
    }

    # L23 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l23_input_split_dict, l23_weight_split_dict, l23_bias_split_dict, l23_output_split_dict = {}, {}, {}, {}
    l23_output_split_dict2, l23_output_split_dict3 = {}, {}
    l23_input_split_dict1, l23_input_split_dict2 = {}, {}
    l23_input_raw_data = handler.parameters['layer3.0.conv1']['input'].astype(np.int8)[:, ::2, ::2].reshape(512, 28, 7)
    # [512, 14, 14] --> [512, 28, 7]
    l23_weight_raw_data = handler.parameters['layer3.0.conv1']['weight'].astype(np.int8)
    l23_bias_raw_data = handler.parameters['layer3.0.conv1']['bias'].astype(np.int32)
    l23_output_raw_data = handler.parameters['layer3.0.cut1']['output'].astype(np.int8).reshape(256, 28, 7)
    (l23_ic, l23_ih, l23_iw) = l23_input_raw_data.shape
    (_, l23_wci, l23_wkh, l23_wkw) = l23_weight_raw_data.shape
    (l23_oc, l23_oh, l23_ow) = l23_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l23_input_split_dict[(core_x, core_y)] = ((0, l23_ic), (core_y * 7, core_y * 7 + 7), (0, l23_iw))
        l23_weight_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32), (0, l23_wci), (0, l23_wkh),
                                                   (0, l23_wkw))
        l23_bias_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),)
        l23_output_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32), (core_y * 7, core_y * 7 + 7),
                                                   (0, l23_ow))
        l23_output_split_dict2[(core_x, core_y)] = (((core_y % 2) * 128, (core_y % 2) * 128 + 128),
                                                    (core_y // 2 * 14, core_y // 2 * 14 + 14),
                                                    (0, l23_ow))
        l23_output_split_dict3[(core_x, core_y)] = (((core_y % 2) * 128, (core_y % 2) * 128 + 128),
                                                    (core_y // 2 * 12, core_y // 2 * 14 + 16),
                                                    (0, l23_ow))  # L24 input
    for core_x, core_y in [(8, 0), (9, 0), (10, 0), (11, 0)]:
        l23_input_split_dict1[(core_x, core_y)] = ((0, l23_ic), ((core_x - 8) * 49, (core_x - 8) * 49 + 24),
                                                   (0, 1))
        l23_input_split_dict2[(core_x, core_y)] = ((0, l23_ic), ((core_x - 8) * 49 + 24, (core_x - 8) * 49 + 49),
                                                   (0, 1))

    data['layer3.0.conv1']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l23_input_raw_data, split_dict=l23_input_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    data['layer3.0.conv1']['input1'] = ResNetDataHandler.tensor_split(
        raw_data=l23_input_raw_data.reshape((512, 28 * 7, 1)), split_dict=l23_input_split_dict1,
        data_type=1, alignment=(16, None, None), dims=(0, 2, 1))
    data['layer3.0.conv1']['input2'] = ResNetDataHandler.tensor_split(
        raw_data=l23_input_raw_data.reshape((512, 28 * 7, 1)), split_dict=l23_input_split_dict2,
        data_type=1, alignment=(16, None, None), dims=(0, 2, 1))

    data['layer3.0.conv1']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l23_weight_raw_data, split_dict=l23_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer3.0.conv1']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l23_bias_raw_data, split_dict=l23_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer3.0.cut1']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l23_output_raw_data, split_dict=l23_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.0.cut1']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l23_output_raw_data, split_dict=l23_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.0.cut1']['output3'] = ResNetDataHandler.tensor_split(
        raw_data=l23_output_raw_data, split_dict=l23_output_split_dict3, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # L24 weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l24_weight_split_dict, l24_bias_split_dict, l24_output_split_dict, l24_output_split_dict2 = {}, {}, {}, {}
    l24_weight_raw_data = handler.parameters['layer3.0.conv2']['weight'].astype(np.int8)
    l24_bias_raw_data = np.append(handler.parameters['layer3.0.conv2']['bias'].astype(np.int32),
                                  np.zeros_like(handler.parameters['layer3.0.conv2']['bias'], dtype=np.int32))
    l24_output_raw_data = handler.parameters['layer3.0.cut2']['output'].astype(np.int8).reshape(256, 28, 7)
    (l24_wco, l24_wci, l24_wkh, l24_wkw) = l24_weight_raw_data.shape
    (l24_oc, l24_oh, l24_ow) = l24_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l24_weight_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),
                                                   ((core_y % 2) * 128, (core_y % 2) * 128 + 128),
                                                   (0, l24_wkh),
                                                   (0, l24_wkw))
        if core_y in [0, 2]:
            l24_bias_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),)
        else:
            l24_bias_split_dict[(core_x, core_y)] = ((core_x * 32 + l24_wco, core_x * 32 + 32 + l24_wco),)
        l24_output_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),
                                                   (core_y * 7, core_y * 7 + 7), (0, l24_ow))
        l24_output_split_dict2[(core_x, core_y)] = (((core_y % 2) * 128, (core_y % 2) * 128 + 128),
                                                    (core_y // 2 * 14, core_y // 2 * 14 + 14),
                                                    (0, l24_ow))  # L25 input
    data['layer3.0.conv2']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l24_weight_raw_data, split_dict=l24_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer3.0.conv2']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l24_bias_raw_data, split_dict=l24_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer3.0.cut2']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l24_output_raw_data, split_dict=l24_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.0.cut2']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l24_output_raw_data, split_dict=l24_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # l25 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l25_input_split_dict, l25_weight_split_dict, l25_bias_split_dict, l25_out_32_split, l25_output_split_dict1 = \
        {}, {}, {}, {}, {}
    l25_output_split_dict2, l25_output_split_dict3 = {}, {}
    l25_input_raw_data = handler.parameters['layer3.0.conv3']['input'].astype(np.int8).reshape(256, 28, 7)
    l25_weight_raw_data = handler.parameters['layer3.0.conv3']['weight'].astype(np.int8)
    l25_bias_raw_data = np.append(handler.parameters['layer3.0.conv3']['bias'].astype(np.int32),
                                  np.zeros_like(handler.parameters['layer3.0.conv3']['bias'], dtype=np.int32))
    l25_output_raw_data = handler.parameters['layer3.0.cut3']['output'].astype(np.int8).reshape(1024, 14 * 14, 1)
    (l25_ic, l25_ih, l25_iw) = l25_input_raw_data.shape
    (l25_wco, l25_wci, l25_wkh, l25_wkw) = l25_weight_raw_data.shape
    (l25_oc, l25_oh, l25_ow) = l25_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l25_input_split_dict[(core_x, core_y)] = (((core_y % 2) * 128, (core_y % 2) * 128 + 128),
                                                  (core_y // 2 * 14, core_y // 2 * 14 + 7),
                                                  (0, l25_iw))
        l25_weight_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                   ((core_y % 2) * 128, (core_y % 2) * 128 + 128),
                                                   (0, l25_wkh), (0, l25_wkw))
        if core_y in [0, 2]:
            l25_bias_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),)
        else:
            l25_bias_split_dict[(core_x, core_y)] = ((core_x * 128 + l25_wco, core_x * 128 + 128 + l25_wco),)
        l25_output_split_dict1[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                    (core_y // 2 * 98 + core_y % 2 * 24,
                                                     core_y // 2 * 98 + core_y % 2 * 24 + 24 + core_y % 2),
                                                    (0, l25_ow))
        l25_output_split_dict2[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                    (core_y // 2 * 98 + core_y % 2 * 24 + 49,
                                                     core_y // 2 * 98 + core_y % 2 * 24 + 24 + core_y % 2 + 49),
                                                    (0, l25_ow))
        l25_output_split_dict3[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                    (core_y * 49, core_y * 49 + 49),
                                                    (0, l25_ow))
    data['layer3.0.conv3']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l25_input_raw_data, split_dict=l25_input_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.0.conv3']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l25_weight_raw_data, split_dict=l25_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer3.0.conv3']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l25_bias_raw_data, split_dict=l25_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer3.0.cut3']['output1'] = ResNetDataHandler.tensor_split(
        raw_data=l25_output_raw_data, split_dict=l25_output_split_dict1, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))  # conv 1/2
    data['layer3.0.cut3']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l25_output_raw_data, split_dict=l25_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))  # conv 2/2
    data['layer3.0.cut3']['output3'] = ResNetDataHandler.tensor_split(
        raw_data=l25_output_raw_data, split_dict=l25_output_split_dict3, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))  # conv

    # l23e input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l23e_input_split_dict, l23e_weight_split_dict, l23e_bias_split_dict, l23e_output_split_dict = {}, {}, {}, {}
    l23e_output_split_dict2 = {}
    l23e_input_raw_data = handler.parameters['layer3.0.downsample.0']['input'].astype(np.int8)[:, ::2, ::2].reshape(
        (512, 28, 7))
    l23e_weight_raw_data = handler.parameters['layer3.0.downsample.0']['weight'].astype(np.int8)
    l23e_bias_raw_data = handler.parameters['layer3.0.downsample.0']['bias'].astype(np.int32)
    l23e_output_raw_data = handler.parameters['layer3.0.cut4']['output'].astype(np.int8).reshape(1024, 28, 7)
    (l23e_ic, l23e_ih, l23e_iw) = l23e_input_raw_data.shape
    (_, l23e_wci, l23e_wkh, l23e_wkw) = l23e_weight_raw_data.shape
    (l23e_oc, l23e_oh, l23e_ow) = l23e_output_raw_data.shape
    for core_y, core_x in product(range(2), range(8)):  # 2 * 8 array
        if core_x % 2 == 0:
            l23e_input_split_dict[(core_x, core_y)] = ((0, l23e_ic), (core_y * 14, core_y * 14 + 7), (0, l23e_iw))
        else:
            l23e_input_split_dict[(core_x, core_y)] = ((0, l23e_ic), (core_y * 14 + 7, core_y * 14 + 14), (0, l23e_iw))
        l23e_weight_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128), (0, l23e_wci), (0, l23e_wkh),
                                                    (0, l23e_wkw))
        l23e_bias_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),)
        l23e_output_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128), (core_y * 14, core_y * 14 + 14),
                                                    (0, l23e_ow))
    for core_y, core_x in product(range(4), range(8)):  # 4 * 8 array
        l23e_output_split_dict2[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                     (core_y * 7, core_y * 7 + 7),
                                                     (0, l23e_ow))
    data['layer3.0.downsample.0']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l23e_input_raw_data, split_dict=l23e_input_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.0.downsample.0']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l23e_weight_raw_data, split_dict=l23e_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer3.0.downsample.0']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l23e_bias_raw_data, split_dict=l23e_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer3.0.cut4']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l23e_output_raw_data, split_dict=l23e_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.0.cut4']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l23e_output_raw_data, split_dict=l23e_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # ReLU(L4 + L4e) input -- [C, H, W]; output -- [C, H, W]
    y_output_split_dict = {}
    y_output_split_dict1, y_output_split_dict2, y_output_split_dict3, y_output_split_dict4 = {}, {}, {}, {}
    y_output_raw_data = handler.parameters['layer3.0.cut5']['output'].astype(np.int8).reshape(1024, 28, 7)
    (_, _, y_ow) = y_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        y_output_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128), (core_y * 7, core_y * 7 + 7),
                                                 (0, y_ow))
    data['layer3.0.cut5']['output'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data, split_dict=y_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    return data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)
    print(handler.names)
