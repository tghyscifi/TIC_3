from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler
import numpy as np


def generate_g4_data(handler, size_y, size_x):
    data = {
        'layer1.2.conv1': {},       # L5
        'layer1.2.relu1': {},
        'layer1.2.cut1': {},
        'layer1.2.conv2': {},       # L6
        'layer1.2.relu2': {},
        'layer1.2.cut2': {},
        'layer1.2.conv3': {},       # L7
        'layer1.2.cut3': {},
        'layer1.2.relu3': {},
        'layer1.2.cut5': {}
    }

    # L5 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l5_input_split_dict, l5_weight_split_dict, l5_bias_split_dict, l5_output_split_dict = {}, {}, {}, {}
    l5_input_split_dict1, l5_input_split_dict2, l5_input_split_dict3, l5_input_split_dict4 = {}, {}, {}, {}
    l5_input_raw_data = handler.parameters['layer1.2.conv1']['input'].astype(np.int8)
    l5_weight_raw_data = handler.parameters['layer1.2.conv1']['weight'].astype(np.int8)
    l5_bias_raw_data = handler.parameters['layer1.2.conv1']['bias'].astype(np.int32)
    l5_output_raw_data = handler.parameters['layer1.2.cut1']['output'].astype(np.int8)
    (l5_ic, l5_ih, l5_iw) = l5_input_raw_data.shape
    (l5_wco, l5_wci, l5_wkh, l5_wkw) = l5_weight_raw_data.shape
    (l5_oc, l5_oh, l5_ow) = l5_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l5_input_split_dict[(core_x, core_y)] = ((0, l5_ic), (core_y * 2 + core_x * 4, core_y * 2 + core_x * 4 + 2),
                                                 (0, l5_iw))

        l5_input_split_dict1[(core_x, core_y)] = ((0, l5_ic),
                                                  ((core_x * 2 + core_y) * 112, (core_x * 2 + core_y) * 112 + 28),
                                                  (0, 1))
        l5_input_split_dict2[(core_x, core_y)] = ((0, l5_ic),
                                                  ((core_x * 2 + core_y) * 112 + 28, (core_x * 2 + core_y) * 112 + 56),
                                                  (0, 1))
        l5_input_split_dict3[(core_x, core_y)] = ((0, l5_ic),
                                                  ((core_x * 2 + core_y) * 112 + 56, (core_x * 2 + core_y) * 112 + 84),
                                                  (0, 1))
        l5_input_split_dict4[(core_x, core_y)] = ((0, l5_ic),
                                                  ((core_x * 2 + core_y) * 112 + 84, (core_x * 2 + core_y) * 112 + 112),
                                                  (0, 1))

        l5_weight_split_dict[(core_x, core_y)] = ((0, l5_wco), (0, l5_wci), (0, l5_wkh), (0, l5_wkw))
        l5_bias_split_dict[(core_x, core_y)] = ((0, l5_wco),)
        l5_output_split_dict[(core_x, core_y)] = ((0, l5_oc), (core_y * 2 + core_x * 4, core_y * 2 + core_x * 4 + 2),
                                                  (0, l5_ow))
    data['layer1.2.conv1']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l5_input_raw_data, split_dict=l5_input_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    data['layer1.2.conv1']['input1'] = ResNetDataHandler.tensor_split(
        raw_data=l5_input_raw_data.reshape((256, -1, 1)), split_dict=l5_input_split_dict1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer1.2.conv1']['input2'] = ResNetDataHandler.tensor_split(
        raw_data=l5_input_raw_data.reshape((256, -1, 1)), split_dict=l5_input_split_dict2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer1.2.conv1']['input3'] = ResNetDataHandler.tensor_split(
        raw_data=l5_input_raw_data.reshape((256, -1, 1)), split_dict=l5_input_split_dict3, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer1.2.conv1']['input4'] = ResNetDataHandler.tensor_split(
        raw_data=l5_input_raw_data.reshape((256, -1, 1)), split_dict=l5_input_split_dict4, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    data['layer1.2.conv1']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l5_weight_raw_data, split_dict=l5_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer1.2.conv1']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l5_bias_raw_data, split_dict=l5_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer1.2.cut1']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l5_output_raw_data, split_dict=l5_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # L6 weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l6_weight_split_dict, l6_bias_split_dict, l6_output_split_dict = {}, {}, {}
    l6_weight_raw_data = handler.parameters['layer1.2.conv2']['weight'].astype(np.int8)
    l6_bias_raw_data = handler.parameters['layer1.2.conv2']['bias'].astype(np.int32)
    l6_output_raw_data = handler.parameters['layer1.2.cut2']['output'].astype(np.int8)
    (l6_wco, l6_wci, l6_wkh, l6_wkw) = l6_weight_raw_data.shape
    (l6_oc, l6_oh, l6_ow) = l6_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l6_weight_split_dict[(core_x, core_y)] = ((core_y * 32, core_y * 32 + 32), (0, l6_wci), (0, l6_wkh),
                                                  (0, l6_wkw))
        l6_bias_split_dict[(core_x, core_y)] = ((core_y * 32, core_y * 32 + 32),)
        l6_output_split_dict[(core_x, core_y)] = ((core_y * 32, core_y * 32 + 32), (core_x * 4, core_x * 4 + 4),
                                                  (0, l6_ow))
    data['layer1.2.conv2']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l6_weight_raw_data, split_dict=l6_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer1.2.conv2']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l6_bias_raw_data, split_dict=l6_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer1.2.cut2']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l6_output_raw_data, split_dict=l6_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # L7 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l7_input_split_dict, l7_weight_split_dict, l7_bias_split_dict, l7_out_32_split, l7_output_split_dict = \
        {}, {}, {}, {}, {}
    l7_input_raw_data = handler.parameters['layer1.2.conv3']['input'].astype(np.int8)
    l7_weight_raw_data = handler.parameters['layer1.2.conv3']['weight'].astype(np.int8)
    l7_bias_raw_data = handler.parameters['layer1.2.conv3']['bias'].astype(np.int32)
    l7_output_raw_data = handler.parameters['layer1.2.cut3']['output'].astype(np.int8)
    l7_out_32_raw_data = handler.parameters['layer1.2.conv3']['output'].astype(np.int32)
    (l7_ic, l7_ih, l7_iw) = l7_input_raw_data.shape
    (_, l7_wci, l7_wkh, l7_wkw) = l7_weight_raw_data.shape
    (l7_oc, l7_oh, l7_ow) = l7_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        # l7_input_split_dict[(core_x, core_y)] = ((0, l7_ic), (core_x * 4, core_x * 4 + 4), (0, l7_iw))
        l7_weight_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (0, l7_wci), (0, l7_wkh),
                                                  (0, l7_wkw))
        l7_bias_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),)
        l7_output_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (core_x * 4, core_x * 4 + 4),
                                                  (0, l7_ow))
        # l7_out_32_split[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (core_x * 4, core_x * 4 + 4),
        #                                      (0, l7_iw))
    # data['layer1.2.conv3']['input'] = ResNetDataHandler.tensor_split(
    #     raw_data=l7_input_raw_data, split_dict=l7_input_split_dict, data_type=1, alignment=(16, None, None),
    #     dims=(0, 2, 1))
    data['layer1.2.conv3']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l7_weight_raw_data, split_dict=l7_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer1.2.conv3']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l7_bias_raw_data, split_dict=l7_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer1.2.cut3']['output1'] = ResNetDataHandler.tensor_split(
        raw_data=l7_output_raw_data, split_dict=l7_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    # data['layer1.2.conv3']['output'] = ResNetDataHandler.tensor_split(
    #     raw_data=l7_out_32_raw_data, split_dict=l7_out_32_split, data_type=0, alignment=(32, None, None),
    #     dims=(0, 2, 1))

    # ReLU(L7 + X5) input -- [C, H, W]; output -- [C, H, W]
    y_c3_split_dict, y_output_split_dict = {}, {}
    y_c3_raw_data = handler.parameters['layer1.2.cut3']['output'].astype(np.int8)
    y_output_raw_data = handler.parameters['layer1.2.cut5']['output'].astype(np.int8)[:, ::2, ::2]
    (y_c3_oc, y_c3_oh, y_c3_ow) = y_c3_raw_data.shape
    (y_oc, y_oh, y_ow) = y_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        y_c3_split_dict[(core_x, core_y)] = ((0, y_oc), (core_y * 2 + core_x * 4, core_y * 2 + core_x * 4 + 2),
                                             (0, y_ow))
        y_output_split_dict[(core_x, core_y)] = ((0, y_oc), (core_y + core_x * 2, core_y + core_x * 2 + 1),
                                                 (0, y_ow))
    data['layer1.2.cut3']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=y_c3_raw_data, split_dict=y_c3_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer1.2.cut5']['output'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data, split_dict=y_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    return data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g4_data(handler, size_y=2, size_x=14)
    print(handler.names)
