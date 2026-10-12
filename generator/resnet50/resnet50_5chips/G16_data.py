from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler
import numpy as np


def generate_g16_data(handler, size_y, size_x):
    data = {
        'layer4.1.conv1': {},
        'layer4.1.relu1': {},
        'layer4.1.cut1': {},

        'layer4.1.conv2': {},
        'layer4.1.relu2': {},
        'layer4.1.cut2': {},

        'layer4.1.conv3': {},
        'layer4.1.cut3': {},

        'layer4.1.relu3': {},
        'layer4.1.cut5': {}
    }
    psum = {
        0: (0, 13),
        1: (13, 25),
        2: (25, 37),
        3: (37, 49)
    }

    # L41 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l1_input_split_dict1, l1_weight_split_dict, l1_bias_split_dict, l1_output_split_dict = {}, {}, {}, {}
    l1_output_split_dict2, l1_output_split_dict3 = {}, {}
    l1_input_split_dict3 = {}
    l1_input_split_dict2 = {}
    l1_input_split_dict = {}
    l1_input_raw_data = handler.parameters['layer4.1.conv1']['input'].astype(np.int8)  # 2048, 7, 7
    l1_weight_raw_data = handler.parameters['layer4.1.conv1']['weight'].astype(np.int8)
    l1_bias_raw_data = np.append(
        handler.parameters['layer4.1.conv1']['bias'].astype(np.int32),
        np.zeros_like(handler.parameters['layer4.1.conv1']['bias']).astype(np.int32))
    l1_output_raw_data = handler.parameters['layer4.1.cut1']['output'].astype(np.int8).reshape(512, -1, 1)  # 512, 49, 1
    (l1_ic, l1_ih, l1_iw) = l1_input_raw_data.shape
    (l1_wco, l1_wci, l1_wkh, l1_wkw) = l1_weight_raw_data.shape
    (l1_oc, l1_oh, l1_ow) = l1_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l1_input_split_dict[(core_x, core_y)] = ((core_y * 512, (core_y + 1) * 512), (0, 49), (0, l1_iw))
        l1_input_split_dict1[(core_x, core_y)] = ((core_y * 512, (core_y + 1) * 512), (0, 24), (0, l1_iw))
        l1_input_split_dict3[(core_x, core_y)] = ((core_y * 512, (core_y + 1) * 512), (24, 49), (0, l1_iw))
        if core_y == 3:
            l1_input_split_dict2[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                      (36, 49), (0, 1))
        else:
            l1_input_split_dict2[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                      (core_y * 12, (core_y + 1) * 12), (0, 1))
        l1_weight_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32), (core_y * 512, (core_y + 1) * 512),
                                                  (0, l1_wkh), (0, l1_wkw))
        if core_y == 0:
            l1_bias_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),)
        else:
            l1_bias_split_dict[(core_x, core_y)] = ((core_x * 32 + l1_wco, core_x * 32 + 32 + l1_wco),)
        # l1_output_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128), psum[core_y], (0, l1_ow))
        l1_output_split_dict2[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                   (0, l1_oh), (0, l1_ow))  # L42 input
    data['layer4.1.conv1']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape(2048, -1, 1), split_dict=l1_input_split_dict, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer4.1.conv1']['input1'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape(2048, -1, 1), split_dict=l1_input_split_dict1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer4.1.conv1']['input3'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape(2048, -1, 1), split_dict=l1_input_split_dict3, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer4.1.conv1']['input2'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape(2048, -1, 1), split_dict=l1_input_split_dict2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))     # shortcut
    data['layer4.1.conv1']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l1_weight_raw_data, split_dict=l1_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer4.1.conv1']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l1_bias_raw_data, split_dict=l1_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    # data['layer4.1.cut1']['output'] = ResNetDataHandler.tensor_split(
    #     raw_data=l1_output_raw_data, split_dict=l1_output_split_dict, data_type=1, alignment=(16, None, None),
    #     dims=(0, 2, 1))
    data['layer4.1.cut1']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l1_output_raw_data, split_dict=l1_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # L45 weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l2_weight_split_dict, l2_bias_split_dict, l2_output_split_dict, l2_output_split_dict2 = {}, {}, {}, {}
    l2_weight_raw_data = handler.parameters['layer4.1.conv2']['weight'].astype(np.int8)
    l2_bias_raw_data = np.append(handler.parameters['layer4.1.conv2']['bias'].astype(np.int32),
                                 np.zeros_like(handler.parameters['layer4.1.conv2']['bias'], dtype=np.int32))
    l2_output_raw_data = handler.parameters['layer4.1.cut2']['output'].astype(np.int8)
    (l2_wco, l2_wci, l2_wkh, l2_wkw) = l2_weight_raw_data.shape
    (l2_oc, l2_oh, l2_ow) = l2_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l2_weight_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),
                                                  (core_y * 128, core_y * 128 + 128),
                                                  (0, l2_wkh),
                                                  (0, l2_wkw))
        if core_y in [0]:
            l2_bias_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),)
        else:
            l2_bias_split_dict[(core_x, core_y)] = ((core_x * 32 + l2_wco, core_x * 32 + 32 + l2_wco),)
        # l2_output_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),
        #                                           (core_y * 7, core_y * 7 + 7), (0, l2_ow))
        l2_output_split_dict2[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                   (0, l2_oh),
                                                   (0, l2_ow))  # L25 input
    data['layer4.1.conv2']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l2_weight_raw_data, split_dict=l2_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer4.1.conv2']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l2_bias_raw_data, split_dict=l2_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    # data['layer4.1.cut2']['output'] = ResNetDataHandler.tensor_split(
    #     raw_data=l2_output_raw_data, split_dict=l2_output_split_dict, data_type=1, alignment=(16, None, None),
    #     dims=(0, 2, 1))
    data['layer4.1.cut2']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # l46 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l3_input_split_dict, l3_weight_split_dict, l3_bias_split_dict, l3_out_32_split, l3_output_split_dict1 = \
        {}, {}, {}, {}, {}
    l3_output_split_dict2, l3_output_split_dict3, l3_output_split_dict4 = {}, {}, {}
    l3_input_raw_data = handler.parameters['layer4.1.conv3']['input'].astype(np.int8)
    l3_weight_raw_data = handler.parameters['layer4.1.conv3']['weight'].astype(np.int8)
    l3_bias_raw_data = np.append(handler.parameters['layer4.1.conv3']['bias'].astype(np.int32),
                                 np.zeros_like(handler.parameters['layer4.1.conv3']['bias'], dtype=np.int32))
    l3_output_raw_data = handler.parameters['layer4.1.cut3']['output'].astype(np.int8).reshape((2048, -1, 1))
    (l3_ic, l3_ih, l3_iw) = l3_input_raw_data.shape
    (l3_wco, l3_wci, l3_wkh, l3_wkw) = l3_weight_raw_data.shape
    (l3_oc, l3_oh, l3_ow) = l3_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        # l3_input_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
        #                                          (0, l3_ih),
        #                                          (0, l3_iw))
        l3_weight_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                  (core_y * 128, core_y * 128 + 128),
                                                  (0, l3_wkh), (0, l3_wkw))
        if core_y in [0]:
            l3_bias_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),)
        else:
            l3_bias_split_dict[(core_x, core_y)] = ((core_x * 128 + l3_wco, core_x * 128 + 128 + l3_wco),)
        l3_output_split_dict1[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128), (core_y * 6, core_y * 6 + 6),
                                                   (0, l3_ow))
        l3_output_split_dict2[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                   (core_y * 6 + 24, core_y * 6 + 6 + 24),
                                                   (0, l3_ow))
        if core_y == 0:
            l3_output_split_dict3[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                       (core_y * 6 + 48, core_y * 6 + 1 + 48),
                                                       (0, l3_ow))
        else:
            l3_output_split_dict3[(core_x, core_y)] = ((0, 0), (0, 0), (0, 0))
        if core_y == 3:
            l3_output_split_dict4[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                       (core_y * 12, core_y * 12 + 13),
                                                       (0, l3_ow))
        else:
            l3_output_split_dict4[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                       (core_y * 12, core_y * 12 + 12),
                                                       (0, l3_ow))
    # data['layer4.1.conv3']['input'] = ResNetDataHandler.tensor_split(
    #     raw_data=l3_input_raw_data, split_dict=l3_input_split_dict, data_type=1, alignment=(16, None, None),
    #     dims=(0, 2, 1))
    data['layer4.1.conv3']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l3_weight_raw_data, split_dict=l3_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer4.1.conv3']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l3_bias_raw_data, split_dict=l3_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer4.1.cut3']['output1'] = ResNetDataHandler.tensor_split(
        raw_data=l3_output_raw_data, split_dict=l3_output_split_dict1, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer4.1.cut3']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l3_output_raw_data, split_dict=l3_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer4.1.cut3']['output3'] = ResNetDataHandler.tensor_split(
        raw_data=l3_output_raw_data, split_dict=l3_output_split_dict3, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer4.1.cut3']['output4'] = ResNetDataHandler.tensor_split(
        raw_data=l3_output_raw_data, split_dict=l3_output_split_dict4, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # ReLU(L4 + L4e) input -- [C, H, W]; output -- [C, H, W]
    y_output_split_dict = {}
    y_output_raw_data = handler.parameters['layer4.1.cut5']['output'].astype(np.int8).reshape((2048, -1, 1))
    (_, _, y_ow) = y_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 3:
            y_output_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                     (core_y * 12, core_y * 12 + 13),
                                                     (0, y_ow))
        else:
            y_output_split_dict[(core_x, core_y)] = ((core_x * 128, core_x * 128 + 128),
                                                     (core_y * 12, core_y * 12 + 12),
                                                     (0, y_ow))
    data['layer4.1.cut5']['output'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data, split_dict=y_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    return data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g16_data(handler, size_y=4, size_x=16)
    print(handler.names)
