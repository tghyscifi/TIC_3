from itertools import product
from generator.resnet50.resnet50_2chip.resnet50_2chip_data_handler import ResNetDataHandler
import numpy as np


def generate_g2_data(handler, size_y, size_x):
    g2_data = {
        'layer1.0.conv1': {},  # L2
        'layer1.0.relu1': {},
        'layer1.0.cut1': {},
        'layer1.0.conv2': {},  # L3
        'layer1.0.relu2': {},
        'layer1.0.cut2': {},
        'layer1.0.conv3': {},  # L4
        'layer1.0.cut3': {},
        'layer1.0.downsample.0': {},  # L4e
        'layer1.0.cut4': {},
        'layer1.0.relu3': {},
        'layer1.0.cut5': {}
    }

    # L2 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l2_input_split_dict, l2_weight_split_dict, l2_bias_split_dict, l2_output_split_dict = {}, {}, {}, {}
    l2_input_split_dict1 = {}
    l2_input_raw_data = handler.parameters['layer1.0.conv1']['input'].astype(np.int8)
    l2_weight_raw_data = handler.parameters['layer1.0.conv1']['weight'].astype(np.int8)
    l2_bias_raw_data = handler.parameters['layer1.0.conv1']['bias'].astype(np.int32)
    l2_output_raw_data = handler.parameters['layer1.0.cut1']['output'].astype(np.int8)
    (l2_ic, l2_ih, l2_iw) = l2_input_raw_data.shape
    (_, l2_wci, l2_wkh, l2_wkw) = l2_weight_raw_data.shape
    (l2_oc, l2_oh, l2_ow) = l2_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l2_input_split_dict[(core_x, core_y)] = ((0, l2_ic), (core_x * 4, core_x * 4 + 4), (0, l2_iw))
        if core_y == 0:
            l2_input_split_dict1[(core_x, core_y)] = ((0, l2_ic), (core_x * 4, core_x * 4 + 4), (0, 28))
        else:
            l2_input_split_dict1[(core_x, core_y)] = ((0, l2_ic), (core_x * 4, core_x * 4 + 4), (28, 56))
        l2_weight_split_dict[(core_x, core_y)] = ((core_y * 32, core_y * 32 + 32), (0, l2_wci), (0, l2_wkh),
                                                  (0, l2_wkw))
        l2_bias_split_dict[(core_x, core_y)] = ((core_y * 32, core_y * 32 + 32),)
        l2_output_split_dict[(core_x, core_y)] = ((0, l2_oc), (core_x * 4, core_x * 4 + 4), (0, l2_ow))
    g2_data['layer1.0.conv1']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l2_input_raw_data, split_dict=l2_input_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    g2_data['layer1.0.conv1']['input1'] = ResNetDataHandler.tensor_split(
        raw_data=l2_input_raw_data, split_dict=l2_input_split_dict1, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    g2_data['layer1.0.conv1']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l2_weight_raw_data, split_dict=l2_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    g2_data['layer1.0.conv1']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l2_bias_raw_data, split_dict=l2_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    g2_data['layer1.0.cut1']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # L3 weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l3_weight_split_dict, l3_bias_split_dict, l3_output_split_dict = {}, {}, {}
    l3_weight_raw_data = handler.parameters['layer1.0.conv2']['weight'].astype(np.int8)
    l3_bias_raw_data = handler.parameters['layer1.0.conv2']['bias'].astype(np.int32)
    l3_output_raw_data = handler.parameters['layer1.0.cut2']['output'].astype(np.int8)
    (_, l3_wci, l3_wkh, l3_wkw) = l3_weight_raw_data.shape
    (l3_oc, l3_oh, l3_ow) = l3_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l3_weight_split_dict[(core_x, core_y)] = ((core_y * 32, core_y * 32 + 32), (0, l3_wci), (0, l3_wkh),
                                                  (0, l3_wkw))
        l3_bias_split_dict[(core_x, core_y)] = ((core_y * 32, core_y * 32 + 32),)
        l3_output_split_dict[(core_x, core_y)] = ((0, l3_oc), (core_x * 4, core_x * 4 + 4), (0, l3_ow))
    g2_data['layer1.0.conv2']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l3_weight_raw_data, split_dict=l3_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    g2_data['layer1.0.conv2']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l3_bias_raw_data, split_dict=l3_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    g2_data['layer1.0.cut2']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l3_output_raw_data, split_dict=l3_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # L4 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l4_input_split_dict, l4_weight_split_dict, l4_bias_split_dict, l4_out_32_split, l4_output_split_dict = \
        {}, {}, {}, {}, {}
    l4_input_raw_data = handler.parameters['layer1.0.conv3']['input'].astype(np.int8)
    l4_weight_raw_data = handler.parameters['layer1.0.conv3']['weight'].astype(np.int8)
    l4_bias_raw_data = handler.parameters['layer1.0.conv3']['bias'].astype(np.int32)
    l4_output_raw_data = handler.parameters['layer1.0.cut3']['output'].astype(np.int8)
    l4_out_32_raw_data = handler.parameters['layer1.0.conv3']['output'].astype(np.int32)
    (l4_ic, l4_ih, l4_iw) = l4_input_raw_data.shape
    (_, l4_wci, l4_wkh, l4_wkw) = l4_weight_raw_data.shape
    (l4_oc, l4_oh, l4_ow) = l4_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l4_input_split_dict[(core_x, core_y)] = ((0, l4_ic), (core_x * 4, core_x * 4 + 4), (0, l4_iw))
        l4_weight_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (0, l4_wci), (0, l4_wkh),
                                                  (0, l4_wkw))
        l4_bias_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),)
        l4_output_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (core_x * 4, core_x * 4 + 4),
                                                  (0, l4_ow))
        l4_out_32_split[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (core_x * 4, core_x * 4 + 4),
                                             (0, l4_iw))
    g2_data['layer1.0.conv3']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l4_input_raw_data, split_dict=l4_input_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    g2_data['layer1.0.conv3']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l4_weight_raw_data, split_dict=l4_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    g2_data['layer1.0.conv3']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l4_bias_raw_data, split_dict=l4_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    g2_data['layer1.0.cut3']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l4_output_raw_data, split_dict=l4_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    g2_data['layer1.0.conv3']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l4_out_32_raw_data, split_dict=l4_out_32_split, data_type=0, alignment=(32, None, None),
        dims=(0, 2, 1))

    # L4e input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l4e_input_split_dict, l4e_weight_split_dict, l4e_bias_split_dict, l4e_output_split_dict = {}, {}, {}, {}
    l4e_input_raw_data = handler.parameters['layer1.0.downsample.0']['input'].astype(np.int8)
    l4e_weight_raw_data = handler.parameters['layer1.0.downsample.0']['weight'].astype(np.int8)
    l4e_bias_raw_data = handler.parameters['layer1.0.downsample.0']['bias'].astype(np.int32)
    l4e_output_raw_data = handler.parameters['layer1.0.cut4']['output'].astype(np.int8)
    (l4e_ic, l4e_ih, l4e_iw) = l4e_input_raw_data.shape
    (_, l4e_wci, l4e_wkh, l4e_wkw) = l4e_weight_raw_data.shape
    (l4e_oc, l4e_oh, l4e_ow) = l4e_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l4e_input_split_dict[(core_x, core_y)] = ((0, l4e_ic), (core_x * 4, core_x * 4 + 4), (0, l4e_iw))
        l4e_weight_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (0, l4e_wci), (0, l4e_wkh),
                                                   (0, l4e_wkw))
        l4e_bias_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),)
        l4e_output_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (core_x * 4, core_x * 4 + 4),
                                                   (0, l4e_ow))
    g2_data['layer1.0.downsample.0']['input'] = ResNetDataHandler.tensor_split(
        raw_data=l4e_input_raw_data, split_dict=l4e_input_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    g2_data['layer1.0.downsample.0']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l4e_weight_raw_data, split_dict=l4e_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    g2_data['layer1.0.downsample.0']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l4e_bias_raw_data, split_dict=l4e_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    g2_data['layer1.0.cut4']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l4e_output_raw_data, split_dict=l4e_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # ReLU(L4 + L4e) input -- [C, H, W]; output -- [C, H, W]
    y_output_split_dict = {}
    y_output_split_dict_in_1, y_output_split_dict_ciso_1 = {}, {}
    y_output_split_dict_in_2, y_output_split_dict_ciso_2 = {}, {}
    y_output_split_dict_in_3, y_output_split_dict_ciso_3 = {}, {}
    y_output_split_dict_in_4, y_output_split_dict_ciso_4 = {}, {}
    y_output_raw_data = handler.parameters['layer1.0.cut5']['output'].astype(np.int8)
    (_, _, y_ow) = y_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        y_output_split_dict[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128), (core_x * 4, core_x * 4 + 4),
                                                 (0, y_ow))

        y_output_split_dict_in_1[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                      (core_x * 224, core_x * 224 + 28),
                                                      (0, 1))
        y_output_split_dict_ciso_1[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                        (core_x * 224 + 112, core_x * 224 + 112 + 28),
                                                        (0, 1))

        y_output_split_dict_in_2[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                      (core_x * 224 + 28, core_x * 224 + 56),
                                                      (0, 1))
        y_output_split_dict_ciso_2[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                        (core_x * 224 + 112 + 28, core_x * 224 + 112 + 56),
                                                        (0, 1))

        y_output_split_dict_in_3[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                      (core_x * 224 + 56, core_x * 224 + 84),
                                                      (0, 1))
        y_output_split_dict_ciso_3[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                        (core_x * 224 + 112 + 56, core_x * 224 + 112 + 84),
                                                        (0, 1))

        y_output_split_dict_in_4[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                      (core_x * 224 + 84, core_x * 224 + 112),
                                                      (0, 1))
        y_output_split_dict_ciso_4[(core_x, core_y)] = ((core_y * 128, core_y * 128 + 128),
                                                        (core_x * 224 + 112 + 84, core_x * 224 + 112 + 112),
                                                        (0, 1))

    g2_data['layer1.0.cut5']['output'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data, split_dict=y_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    g2_data['layer1.0.cut5']['in_1'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_in_1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    g2_data['layer1.0.cut5']['ciso_1'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_ciso_1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    g2_data['layer1.0.cut5']['in_2'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_in_2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    g2_data['layer1.0.cut5']['ciso_2'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_ciso_2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    g2_data['layer1.0.cut5']['in_3'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_in_3, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    g2_data['layer1.0.cut5']['ciso_3'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_ciso_3, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    g2_data['layer1.0.cut5']['in_4'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_in_4, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    g2_data['layer1.0.cut5']['ciso_4'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((256, -1, 1)), split_dict=y_output_split_dict_ciso_4, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    return g2_data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g2_data(handler, size_y=2, size_x=14)
    print(handler.names)
