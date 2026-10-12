from itertools import product
from generator.resnet50.data_handler import ResNetDataHandler
import numpy as np


def generate_g11_data(handler, size_y, size_x):
    data = {
        'layer3.2.conv1': {},  # L26
        'layer3.2.relu1': {},
        'layer3.2.cut1': {},
        'layer3.2.conv2': {},  # L27
        'layer3.2.relu2': {},
        'layer3.2.cut2': {},
        'layer3.2.conv3': {},  # L28
        'layer3.2.relu3': {},
        'layer3.2.cut3': {},
        'layer3.2.cut5': {}
    }
    psum = {
        0: (0, 25),
        1: (25, 50),
        2: (50, 74),
        3: (74, 98)
    }

    # L26 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l1_input_split_dict1, l1_weight_split_dict, l1_bias_split_dict, l1_output_split_dict = {}, {}, {}, {}
    l1_input_split_dict2, l1_input_split_dict3 = {}, {}
    l1_input_split_dict5, l1_input_split_dict6, l1_input_split_dict7, l1_input_split_dict8 = {}, {}, {}, {}
    l1_output_split_dict2, l1_output_split_dict3 = {}, {}
    l1_input_raw_data = handler.parameters['layer3.2.conv1']['input'].astype(np.int8)
    l1_weight_raw_data = handler.parameters['layer3.2.conv1']['weight'].astype(np.int8)
    l1_bias_raw_data = np.append(handler.parameters['layer3.2.conv1']['bias'].astype(np.int32),
                                 np.zeros_like(handler.parameters['layer3.2.conv1']['bias']).astype(np.int32))
    l1_output_raw_data = handler.parameters['layer3.2.cut1']['output'].astype(np.int8).reshape((256, -1, 1))
    (l1_ic, l1_ih, l1_iw) = l1_input_raw_data.shape
    (l1_wco, l1_wci, l1_wkh, l1_wkw) = l1_weight_raw_data.shape
    (l1_oc, l1_oh, l1_ow) = l1_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l1_input_split_dict1[(core_x, core_y)] = ((core_y * 256, core_y * 256 + 256), (0, 7), (0, l1_iw))
        l1_input_split_dict2[(core_x, core_y)] = ((core_y * 256, core_y * 256 + 256), (7, 14), (0, l1_iw))

        l1_input_split_dict5[(core_x, core_y)] = ((core_y * 256, core_y * 256 + 256), (0, 7), (0, 7))
        l1_input_split_dict6[(core_x, core_y)] = ((core_y * 256, core_y * 256 + 256), (7, 14), (0, 7))
        l1_input_split_dict7[(core_x, core_y)] = ((core_y * 256, core_y * 256 + 256), (14, 21), (0, 7))
        l1_input_split_dict8[(core_x, core_y)] = ((core_y * 256, core_y * 256 + 256), (21, 28), (0, 7))

        l1_weight_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32), (core_y * 256, core_y * 256 + 256),
                                                  (0, l1_wkh), (0, l1_wkw))
        if core_y == 0:
            l1_bias_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),)
        else:
            l1_bias_split_dict[(core_x, core_y)] = ((core_x * 32 + l1_wco, core_x * 32 + 32 + l1_wco),)
        l1_output_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32), psum[core_y],
                                                  (0, l1_ow))
        l1_output_split_dict2[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),
                                                   (psum[core_y][0] + 98, psum[core_y][1] + 98),
                                                   (0, l1_ow))
        l1_output_split_dict3[(core_x, core_y)] = ((core_y * 64, core_y * 64 + 64),
                                                   (0, 14 * 14),
                                                   (0, l1_ow))
        l1_input_split_dict3[(core_x, core_y)] = ((core_x // 2 * 64 + core_y * 256,
                                                   core_x // 2 * 64 + core_y * 256 + 64),
                                                  (core_x % 2 * 7, core_x % 2 * 7 + 7),
                                                  (0, l1_iw))  # ADD input
    data['layer3.2.conv1']['input1'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict1, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.conv1']['input2'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.conv1']['input3'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data, split_dict=l1_input_split_dict3, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    data['layer3.2.conv1']['input5'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape((1024, 28, 7)), split_dict=l1_input_split_dict5, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer3.2.conv1']['input6'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape((1024, 28, 7)), split_dict=l1_input_split_dict6, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer3.2.conv1']['input7'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape((1024, 28, 7)), split_dict=l1_input_split_dict7, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer3.2.conv1']['input8'] = ResNetDataHandler.tensor_split(
        raw_data=l1_input_raw_data.reshape((1024, 28, 7)), split_dict=l1_input_split_dict8, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    data['layer3.2.conv1']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l1_weight_raw_data, split_dict=l1_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer3.2.conv1']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l1_bias_raw_data, split_dict=l1_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer3.2.cut1']['output'] = ResNetDataHandler.tensor_split(
        raw_data=l1_output_raw_data, split_dict=l1_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.cut1']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l1_output_raw_data, split_dict=l1_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.cut1']['output3'] = ResNetDataHandler.tensor_split(
        raw_data=l1_output_raw_data, split_dict=l1_output_split_dict3, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # L27 weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l2_weight_split_dict, l2_bias_split_dict, l2_output_split_dict1, l2_output_split_dict2 = {}, {}, {}, {}
    l2_output_split_dict3, l2_output_split_dict4, l2_output_split_dict5, l2_output_split_dict6 = {}, {}, {}, {}
    l2_weight_raw_data = handler.parameters['layer3.2.conv2']['weight'].astype(np.int8)
    l2_bias_raw_data = np.append(handler.parameters['layer3.2.conv2']['bias'].astype(np.int32),
                                 np.zeros_like(handler.parameters['layer3.2.conv2']['bias'], dtype=np.int32))
    l2_output_raw_data = handler.parameters['layer3.2.cut2']['output'].astype(np.int8).reshape((256, -1, 1))
    (l2_wco, l2_wci, l2_wkh, l2_wkw) = l2_weight_raw_data.shape
    (l2_oc, l2_oh, l2_ow) = l2_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l2_weight_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),
                                                  (core_y * 64, core_y * 64 + 64),
                                                  (0, l2_wkh),
                                                  (0, l2_wkw))
        if core_y in [0]:
            l2_bias_split_dict[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),)
        else:
            l2_bias_split_dict[(core_x, core_y)] = ((core_x * 32 + l2_wco, core_x * 32 + 32 + l2_wco),)
        l2_output_split_dict1[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32), psum[core_y],
                                                   (0, l2_ow))
        l2_output_split_dict2[(core_x, core_y)] = ((core_x * 32, core_x * 32 + 32),
                                                   (psum[core_y][0] + 98, psum[core_y][1] + 98),
                                                   (0, l2_ow))

        l2_output_split_dict3[(core_x, core_y)] = ((0, l2_oc), (0, 50), (0, l2_ow))  # L3 input-1/4
        l2_output_split_dict4[(core_x, core_y)] = ((0, l2_oc), (50, 98), (0, l2_ow))  # L3 input-2/4
        l2_output_split_dict5[(core_x, core_y)] = ((0, l2_oc), (98, 148), (0, l2_ow))  # L3 input-3/4
        l2_output_split_dict6[(core_x, core_y)] = ((0, l2_oc), (148, 196), (0, l2_ow))  # L3 input-4/4
    data['layer3.2.conv2']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l2_weight_raw_data, split_dict=l2_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer3.2.conv2']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l2_bias_raw_data, split_dict=l2_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer3.2.cut2']['output1'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict1, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.cut2']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict2, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    # L3 input
    data['layer3.2.cut2']['output3'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict3, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.cut2']['output4'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict4, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.cut2']['output5'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict5, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.cut2']['output6'] = ResNetDataHandler.tensor_split(
        raw_data=l2_output_raw_data, split_dict=l2_output_split_dict6, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # l28 input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    l3_input_split_dict, l3_weight_split_dict, l3_bias_split_dict, l3_out_32_split, l3_output_split_dict1 = \
        {}, {}, {}, {}, {}
    l3_output_split_dict2, l3_output_split_dict3 = {}, {}
    l3_input_raw_data = handler.parameters['layer3.2.conv3']['input'].astype(np.int8).reshape(256, 28, 7)
    l3_weight_raw_data = handler.parameters['layer3.2.conv3']['weight'].astype(np.int8)
    l3_bias_raw_data = np.append(handler.parameters['layer3.2.conv3']['bias'].astype(np.int32),
                                 np.zeros_like(handler.parameters['layer3.2.conv3']['bias'], dtype=np.int32))
    l3_output_raw_data = handler.parameters['layer3.2.cut3']['output'].astype(np.int8)
    (l3_ic, l3_ih, l3_iw) = l3_input_raw_data.shape
    (l3_wco, l3_wci, l3_wkh, l3_wkw) = l3_weight_raw_data.shape
    (l3_oc, l3_oh, l3_ow) = l3_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        l3_weight_split_dict[(core_x, core_y)] = ((core_x * 32 + core_y * 32 * 8, core_x * 32 + core_y * 32 * 8 + 32),
                                                  (0, l3_wci), (0, l3_wkh), (0, l3_wkw))
        l3_bias_split_dict[(core_x, core_y)] = ((core_x * 32 + core_y * 32 * 8, core_x * 32 + core_y * 32 * 8 + 32),)
        l3_output_split_dict1[(core_x, core_y)] = ((core_x // 2 * 64 + core_y * 256,
                                                    core_x // 2 * 64 + core_y * 256 + 64),
                                                   (core_x % 2 * 7, core_x % 2 * 7 + 7),
                                                   (0, l3_ow))  # L3 output
    data['layer3.2.conv3']['weight'] = ResNetDataHandler.tensor_split(
        raw_data=l3_weight_raw_data, split_dict=l3_weight_split_dict, data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    data['layer3.2.conv3']['bias'] = ResNetDataHandler.tensor_split(
        raw_data=l3_bias_raw_data, split_dict=l3_bias_split_dict, data_type=0, alignment=(32,), dims=(0,))
    data['layer3.2.cut3']['output1'] = ResNetDataHandler.tensor_split(
        raw_data=l3_output_raw_data, split_dict=l3_output_split_dict1, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    # ReLU(L3 + Le) input -- [C, H, W]; output -- [C, H, W]
    y_output_split_dict = {}
    y_output_split_dict1, y_output_split_dict2 = {}, {}
    y_output_raw_data = handler.parameters['layer3.2.cut5']['output'].astype(np.int8)
    (_, _, y_ow) = y_output_raw_data.shape
    for core_y, core_x in product(range(size_y), range(size_x)):
        y_output_split_dict[(core_x, core_y)] = ((core_x // 2 * 64 + core_y * 256,
                                                  core_x // 2 * 64 + core_y * 256 + 64),
                                                 (core_x % 2 * 7, core_x % 2 * 7 + 7),
                                                 (0, l3_ow))
        y_output_split_dict1[(core_x, core_y)] = ((core_x // 2 * 64 + core_y * 256,
                                                   core_x // 2 * 64 + core_y * 256 + 64),
                                                  (core_x % 2 * 14, core_x % 2 * 14 + 7),
                                                  (0, 7))
        y_output_split_dict2[(core_x, core_y)] = ((core_x // 2 * 64 + core_y * 256,
                                                   core_x // 2 * 64 + core_y * 256 + 64),
                                                  (core_x % 2 * 14 + 7, core_x % 2 * 14 + 14),
                                                  (0, 7))
    data['layer3.2.cut5']['output'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data, split_dict=y_output_split_dict, data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    data['layer3.2.cut5']['output1'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((1024, 28, 7)), split_dict=y_output_split_dict1, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))
    data['layer3.2.cut5']['output2'] = ResNetDataHandler.tensor_split(
        raw_data=y_output_raw_data.reshape((1024, 28, 7)), split_dict=y_output_split_dict2, data_type=1,
        alignment=(16, None, None), dims=(0, 2, 1))

    return data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g11_data(handler, size_y=4, size_x=8)
    print(handler.names)
