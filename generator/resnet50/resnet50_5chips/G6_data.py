from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.data_handler import ResNetDataHandler
import numpy as np


def generate_g6_data(handler, size_y, size_x):
    # ['conv1', 'relu', 'cut1', 'maxpool', 'layer1.0.conv1', 'layer1.0.relu1', 'layer1.0.cut1', 'layer1.0.conv2', 'layer1.0.relu2', 'layer1.0.cut2', 'layer1.0.conv3', 'layer1.0.cut3', 'layer1.0.downsample.0', 'layer1.0.cut4', 'layer1.0.relu3', 'layer1.0.cut5', 'layer1.1.conv1', 'layer1.1.relu1', 'layer1.1.cut1', 'layer1.1.conv2', 'layer1.1.relu2', 'layer1.1.cut2', 'layer1.1.conv3', 'layer1.1.cut3', 'layer1.1.relu3', 'layer1.1.cut5', 'layer1.2.conv1', 'layer1.2.relu1', 'layer1.2.cut1', 'layer1.2.conv2', 'layer1.2.relu2', 'layer1.2.cut2', 'layer1.2.conv3', 'layer1.2.cut3', 'layer1.2.relu3', 'layer1.2.cut5', 'layer2.0.conv1', 'layer2.0.relu1', 'layer2.0.cut1', 'layer2.0.conv2', 'layer2.0.relu2', 'layer2.0.cut2', 'layer2.0.conv3', 'layer2.0.cut3', 'layer2.0.downsample.0', 'layer2.0.cut4', 'layer2.0.relu3', 'layer2.0.cut5', 'layer2.1.conv1', 'layer2.1.relu1', 'layer2.1.cut1', 'layer2.1.conv2', 'layer2.1.relu2', 'layer2.1.cut2', 'layer2.1.conv3', 'layer2.1.cut3', 'layer2.1.relu3', 'layer2.1.cut5', 'layer2.2.conv1', 'layer2.2.relu1', 'layer2.2.cut1', 'layer2.2.conv2', 'layer2.2.relu2', 'layer2.2.cut2', 'layer2.2.conv3', 'layer2.2.cut3', 'layer2.2.relu3', 'layer2.2.cut5', 'layer2.3.conv1', 'layer2.3.relu1', 'layer2.3.cut1', 'layer2.3.conv2', 'layer2.3.relu2', 'layer2.3.cut2', 'layer2.3.conv3', 'layer2.3.cut3', 'layer2.3.relu3', 'layer2.3.cut5', 'layer3.0.conv1', 'layer3.0.relu1', 'layer3.0.cut1', 'layer3.0.conv2', 'layer3.0.relu2', 'layer3.0.cut2', 'layer3.0.conv3', 'layer3.0.cut3', 'layer3.0.downsample.0', 'layer3.0.cut4', 'layer3.0.relu3', 'layer3.0.cut5', 'layer3.1.conv1', 'layer3.1.relu1', 'layer3.1.cut1', 'layer3.1.conv2', 'layer3.1.relu2', 'layer3.1.cut2', 'layer3.1.conv3', 'layer3.1.cut3', 'layer3.1.relu3', 'layer3.1.cut5', 'layer3.2.conv1', 'layer3.2.relu1', 'layer3.2.cut1', 'layer3.2.conv2', 'layer3.2.relu2', 'layer3.2.cut2', 'layer3.2.conv3', 'layer3.2.cut3', 'layer3.2.relu3', 'layer3.2.cut5', 'layer3.3.conv1', 'layer3.3.relu1', 'layer3.3.cut1', 'layer3.3.conv2', 'layer3.3.relu2', 'layer3.3.cut2', 'layer3.3.conv3', 'layer3.3.cut3', 'layer3.3.relu3', 'layer3.3.cut5', 'layer3.4.conv1', 'layer3.4.relu1', 'layer3.4.cut1', 'layer3.4.conv2', 'layer3.4.relu2', 'layer3.4.cut2', 'layer3.4.conv3', 'layer3.4.cut3', 'layer3.4.relu3', 'layer3.4.cut5', 'layer3.5.conv1', 'layer3.5.relu1', 'layer3.5.cut1', 'layer3.5.conv2', 'layer3.5.relu2', 'layer3.5.cut2', 'layer3.5.conv3', 'layer3.5.cut3', 'layer3.5.relu3', 'layer3.5.cut5', 'layer4.0.conv1', 'layer4.0.relu1', 'layer4.0.cut1', 'layer4.0.conv2', 'layer4.0.relu2', 'layer4.0.cut2', 'layer4.0.conv3', 'layer4.0.cut3', 'layer4.0.downsample.0', 'layer4.0.cut4', 'layer4.0.relu3', 'layer4.0.cut5', 'layer4.1.conv1', 'layer4.1.relu1', 'layer4.1.cut1', 'layer4.1.conv2', 'layer4.1.relu2', 'layer4.1.cut2', 'layer4.1.conv3', 'layer4.1.cut3', 'layer4.1.relu3', 'layer4.1.cut5', 'layer4.2.conv1', 'layer4.2.relu1', 'layer4.2.cut1', 'layer4.2.conv2', 'layer4.2.relu2', 'layer4.2.cut2', 'layer4.2.conv3', 'layer4.2.cut3', 'layer4.2.relu3', 'layer4.2.cut5', 'avgpool', 'cut2', 'fc', 'cut3']

    data = {}

    # Conv3b1 input
    conv3b1_input_raw_data = handler.parameters['layer2.1.conv1']['input'].astype(np.int8)
    conv3b1_input_split_dict = {}
    conv3b1_input_split_dict1, conv3b1_input_split_dict2 = {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b1_input_split_dict[(core_x, core_y)] = ((0, conv3b1_input_raw_data.shape[0]),
                                                      (core_x + size_x * core_y, core_x + size_x * core_y + 1),
                                                      (0, conv3b1_input_raw_data.shape[2]))
        conv3b1_input_split_dict1[(core_x, core_y)] = ((0, conv3b1_input_raw_data.shape[0]),
                                                       (core_x + size_x * core_y, core_x + size_x * core_y + 1),
                                                       (0, 14))
        conv3b1_input_split_dict2[(core_x, core_y)] = ((0, conv3b1_input_raw_data.shape[0]),
                                                       (core_x + size_x * core_y, core_x + size_x * core_y + 1),
                                                       (14, 28))
    data['conv3b1_input'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_input_raw_data,
                                                           split_dict=conv3b1_input_split_dict,
                                                           data_type=1, alignment=(32, None, None), dims=[0, 2, 1])
    data['conv3b1_input1'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_input_raw_data,
                                                            split_dict=conv3b1_input_split_dict1,
                                                            data_type=1, alignment=(32, None, None), dims=[0, 2, 1])
    data['conv3b1_input2'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_input_raw_data,
                                                            split_dict=conv3b1_input_split_dict2,
                                                            data_type=1, alignment=(32, None, None), dims=[0, 2, 1])

    # Conv3b1 weight
    conv3b1_weight_raw_data = handler.parameters['layer2.1.conv1']['weight'].astype(np.int8)
    conv3b1_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b1_weight_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                        ((core_x + core_y * size_x) // 7 + 1) * 32),
                                                       (0, conv3b1_weight_raw_data.shape[1]),
                                                       (0, conv3b1_weight_raw_data.shape[2]),
                                                       (0, conv3b1_weight_raw_data.shape[3]))
    data['conv3b1_weight'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_weight_raw_data,
                                                            split_dict=conv3b1_weight_split_dict,
                                                            data_type=1, alignment=(32, None, None, None),
                                                            dims=[0, 1, 3, 2], is_weight=True)

    # Conv3b1 bias
    conv3b1_bias_raw_data = handler.parameters['layer2.1.conv1']['bias'].astype(np.int32)
    conv3b1_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b1_bias_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                      ((core_x + core_y * size_x) // 7 + 1) * 32),)
    data['conv3b1_bias'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_bias_raw_data,
                                                          split_dict=conv3b1_bias_split_dict,
                                                          data_type=0, alignment=(32,), dims=[0])

    # Conv3b1 output
    conv3b1_output_raw_data = handler.parameters['layer2.1.cut1']['output'].astype(np.int8)
    conv3b1_output_part1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b1_output_part1_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                              ((core_x + core_y * size_x) // 7 + 1) * 32),
                                                             (core_x + size_x * core_y, core_x + size_x * core_y + 1),
                                                             (0, conv3b1_output_raw_data.shape[2]))
    data['conv3b1_output_part1'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_output_raw_data,
                                                                  split_dict=conv3b1_output_part1_split_dict,
                                                                  data_type=1, alignment=(32, None, None),
                                                                  dims=[0, 2, 1])

    # Conv3b1 weight 1_1
    conv3b1_weight1_1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if (0 <= core_x + core_y * size_x and core_x + core_y * size_x < 7):
            conv3b1_weight1_1_split_dict[(core_x, core_y)] = ((96, 128),
                                                              (0, conv3b1_weight_raw_data.shape[1] // 2),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
        elif (7 <= core_x + core_y * size_x and core_x + core_y * size_x < 14):
            conv3b1_weight1_1_split_dict[(core_x, core_y)] = ((0, 32),
                                                              (0, conv3b1_weight_raw_data.shape[1] // 2),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
        elif (14 <= core_x + core_y * size_x and core_x + core_y * size_x < 21):
            conv3b1_weight1_1_split_dict[(core_x, core_y)] = ((32, 64),
                                                              (0, conv3b1_weight_raw_data.shape[1] // 2),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
        elif (21 <= core_x + core_y * size_x and core_x + core_y * size_x < 28):
            conv3b1_weight1_1_split_dict[(core_x, core_y)] = ((64, 96),
                                                              (0, conv3b1_weight_raw_data.shape[1] // 2),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
    data['conv3b1_weight1_1'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_weight_raw_data,
                                                               split_dict=conv3b1_weight1_1_split_dict,
                                                               data_type=1, alignment=(32, None, None, None),
                                                               dims=[0, 1, 3, 2], is_weight=True)

    # Conv3b1 weight 1_2
    conv3b1_weight1_2_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if (0 <= core_x + core_y * size_x and core_x + core_y * size_x < 7):
            conv3b1_weight1_2_split_dict[(core_x, core_y)] = ((96, 128),
                                                              (conv3b1_weight_raw_data.shape[1] // 2,
                                                               conv3b1_weight_raw_data.shape[1]),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
        elif (7 <= core_x + core_y * size_x and core_x + core_y * size_x < 14):
            conv3b1_weight1_2_split_dict[(core_x, core_y)] = ((0, 32),
                                                              (conv3b1_weight_raw_data.shape[1] // 2,
                                                               conv3b1_weight_raw_data.shape[1]),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
        elif (14 <= core_x + core_y * size_x and core_x + core_y * size_x < 21):
            conv3b1_weight1_2_split_dict[(core_x, core_y)] = ((32, 64),
                                                              (conv3b1_weight_raw_data.shape[1] // 2,
                                                               conv3b1_weight_raw_data.shape[1]),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
        elif (21 <= core_x + core_y * size_x and core_x + core_y * size_x < 28):
            conv3b1_weight1_2_split_dict[(core_x, core_y)] = ((64, 96),
                                                              (conv3b1_weight_raw_data.shape[1] // 2,
                                                               conv3b1_weight_raw_data.shape[1]),
                                                              (0, conv3b1_weight_raw_data.shape[2]),
                                                              (0, conv3b1_weight_raw_data.shape[3]))
    data['conv3b1_weight1_2'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_weight_raw_data,
                                                               split_dict=conv3b1_weight1_2_split_dict,
                                                               data_type=1, alignment=(32, None, None, None),
                                                               dims=[0, 1, 3, 2], is_weight=True)

    # Conv3b1 output part2
    conv3b1_output_part2_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if (0 <= core_x + core_y * size_x and core_x + core_y * size_x < 7):
            conv3b1_output_part2_split_dict[(core_x, core_y)] = ((96, 128),
                                                                 (core_x + size_x * core_y,
                                                                  core_x + size_x * core_y + 1),
                                                                 (0, conv3b1_output_raw_data.shape[2]))
        elif (7 <= core_x + core_y * size_x and core_x + core_y * size_x < 14):
            conv3b1_output_part2_split_dict[(core_x, core_y)] = ((0, 64),
                                                                 (core_x + size_x * core_y,
                                                                  core_x + size_x * core_y + 1),
                                                                 (0, conv3b1_output_raw_data.shape[2]))
        elif (14 <= core_x + core_y * size_x and core_x + core_y * size_x < 21):
            conv3b1_output_part2_split_dict[(core_x, core_y)] = ((32, 96),
                                                                 (core_x + size_x * core_y,
                                                                  core_x + size_x * core_y + 1),
                                                                 (0, conv3b1_output_raw_data.shape[2]))
        elif (21 <= core_x + core_y * size_x and core_x + core_y * size_x < 28):
            conv3b1_output_part2_split_dict[(core_x, core_y)] = ((64, 128),
                                                                 (core_x + size_x * core_y,
                                                                  core_x + size_x * core_y + 1),
                                                                 (0, conv3b1_output_raw_data.shape[2]))
    data['conv3b1_output_part2'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_output_raw_data,
                                                                  split_dict=conv3b1_output_part2_split_dict,
                                                                  data_type=1, alignment=(32, None, None),
                                                                  dims=[0, 2, 1])

    conv3b1_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b1_output_split_dict[(core_x, core_y)] = ((0, conv3b1_output_raw_data.shape[0]),
                                                       (core_x + size_x * core_y, core_x + size_x * core_y + 1),
                                                       (0, conv3b1_output_raw_data.shape[2]))
    data['conv3b1_output'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_output_raw_data,
                                                            split_dict=conv3b1_output_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    phase13_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        phase13_split_dict[(core_x, core_y)] = ((0, conv3b1_output_raw_data.shape[0]),
                                                (1, 2),
                                                (0, conv3b1_output_raw_data.shape[2]))
    data['g6_phase13'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_output_raw_data,
                                                        split_dict=phase13_split_dict,
                                                        data_type=1, alignment=(32, None, None),
                                                        dims=[0, 2, 1])

    conv3b2_input0_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x % 7 == 0:
            H_tuple = (0, 2)
        elif core_x % 7 == 1:
            H_tuple = (3, 6)
        elif core_x % 7 == 2:
            H_tuple = (7, 10)
        elif core_x % 7 == 3:
            H_tuple = (11, 14)
        elif core_x % 7 == 4:
            H_tuple = (15, 18)
        elif core_x % 7 == 5:
            H_tuple = (19, 22)
        elif core_x % 7 == 6:
            H_tuple = (23, 26)
        else:
            raise ValueError
        conv3b2_input0_split_dict[(core_x, core_y)] = ((0, conv3b1_output_raw_data.shape[0]),
                                                       H_tuple,
                                                       (0, conv3b1_output_raw_data.shape[2]))
    data['conv3b2_input0'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_output_raw_data,
                                                            split_dict=conv3b2_input0_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    phase15_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        phase15_split_dict[(core_x, core_y)] = ((0, conv3b1_output_raw_data.shape[0]),
                                                (2, 5),
                                                (0, conv3b1_output_raw_data.shape[2]))
    data['g6_phase15'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_output_raw_data,
                                                        split_dict=phase15_split_dict,
                                                        data_type=1, alignment=(32, None, None),
                                                        dims=[0, 2, 1])

    conv3b2_input1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x % 7 == 0:
            H_tuple = (2, 5)
        elif core_x % 7 == 1:
            H_tuple = (6, 9)
        elif core_x % 7 == 2:
            H_tuple = (10, 13)
        elif core_x % 7 == 3:
            H_tuple = (14, 17)
        elif core_x % 7 == 4:
            H_tuple = (18, 21)
        elif core_x % 7 == 5:
            H_tuple = (22, 25)
        elif core_x % 7 == 6:
            H_tuple = (26, 28)
        else:
            raise ValueError
        conv3b2_input1_split_dict[(core_x, core_y)] = ((0, conv3b1_output_raw_data.shape[0]),
                                                       H_tuple,
                                                       (0, conv3b1_output_raw_data.shape[2]))
    data['conv3b2_input1'] = ResNetDataHandler.tensor_split(raw_data=conv3b1_output_raw_data,
                                                            split_dict=conv3b2_input1_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    conv3b2_weight_raw_data = handler.parameters['layer2.1.conv2']['weight'].astype(np.int8)
    conv3b2_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b2_weight_split_dict[(core_x, core_y)] = (((core_x + size_x * core_y) // 7 * 32,
                                                        ((core_x + size_x * core_y) // 7 + 1) * 32),
                                                       (0, conv3b2_weight_raw_data.shape[1]),
                                                       (0, conv3b2_weight_raw_data.shape[2]),
                                                       (0, conv3b2_weight_raw_data.shape[3]))
    data['conv3b2_weight'] = ResNetDataHandler.tensor_split(raw_data=conv3b2_weight_raw_data,
                                                            split_dict=conv3b2_weight_split_dict,
                                                            data_type=1, alignment=(32, None, None, None),
                                                            dims=[0, 1, 3, 2], is_weight=True)

    conv3b2_bias_raw_data = handler.parameters['layer2.1.conv2']['bias'].astype(np.int32)
    conv3b2_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b2_bias_split_dict[(core_x, core_y)] = (((core_x + size_x * core_y) // 7 * 32,
                                                      ((core_x + size_x * core_y) // 7 + 1) * 32),)
    data['conv3b2_bias'] = ResNetDataHandler.tensor_split(raw_data=conv3b2_bias_raw_data,
                                                          split_dict=conv3b2_bias_split_dict,
                                                          data_type=0, alignment=(32,),
                                                          dims=[0])

    conv3b2_output_raw_data = handler.parameters['layer2.1.cut2']['output'].astype(np.int8)
    conv3b2_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b2_output_split_dict[(core_x, core_y)] = (((core_x + size_x * core_y) // 7 * 32,
                                                        ((core_x + size_x * core_y) // 7 + 1) * 32),
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3b2_output_raw_data.shape[2]))
    data['conv3b2_output'] = ResNetDataHandler.tensor_split(raw_data=conv3b2_output_raw_data,
                                                            split_dict=conv3b2_output_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    conv3b3_input_raw_data = handler.parameters['layer2.1.conv3']['input'].astype(np.int8)
    conv3b3_input0_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if 0 <= (core_x + size_x * core_y) <= 13:
            C_tuple = (0, 64)
        else:
            C_tuple = (64, 128)
        conv3b3_input0_split_dict[(core_x, core_y)] = (C_tuple,
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3b3_input_raw_data.shape[2]))
    data['conv3b3_input0'] = ResNetDataHandler.tensor_split(raw_data=conv3b3_input_raw_data,
                                                            split_dict=conv3b3_input0_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    conv3b3_input1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b3_input1_split_dict[(core_x, core_y)] = ((0, conv3b3_input_raw_data.shape[0]),
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3b3_input_raw_data.shape[2]))
    data['conv3b3_input1'] = ResNetDataHandler.tensor_split(raw_data=conv3b3_input_raw_data,
                                                            split_dict=conv3b3_input1_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    conv3b3_weight_raw_data = handler.parameters['layer2.1.conv3']['weight'].astype(np.int8)
    conv3b3_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b3_weight_split_dict[(core_x, core_y)] = (((core_x + size_x * core_y) // 7 * 128,
                                                        ((core_x + size_x * core_y) // 7 + 1) * 128),
                                                       (0, conv3b3_weight_raw_data.shape[1]),
                                                       (0, conv3b3_weight_raw_data.shape[2]),
                                                       (0, conv3b3_weight_raw_data.shape[3]))
    data['conv3b3_weight'] = ResNetDataHandler.tensor_split(raw_data=conv3b3_weight_raw_data,
                                                            split_dict=conv3b3_weight_split_dict,
                                                            data_type=1, alignment=(32, None, None, None),
                                                            dims=[0, 1, 3, 2], is_weight=True)

    conv3b3_bias_raw_data = handler.parameters['layer2.1.conv3']['bias'].astype(np.int32)
    conv3b3_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b3_bias_split_dict[(core_x, core_y)] = (((core_x + size_x * core_y) // 7 * 128,
                                                      ((core_x + size_x * core_y) // 7 + 1) * 128),)
    data['conv3b3_bias'] = ResNetDataHandler.tensor_split(raw_data=conv3b3_bias_raw_data,
                                                          split_dict=conv3b3_bias_split_dict,
                                                          data_type=0, alignment=(32,),
                                                          dims=[0])

    conv3b3_output_raw_data = handler.parameters['layer2.1.cut3']['output'].astype(np.int8)
    conv3b3_output_split_dict, conv3b3_output_split_dict2, conv3b3_output_split_dict3 = {}, {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3b3_output_split_dict[(core_x, core_y)] = (((core_x + size_x * core_y) // 7 * 128,
                                                        ((core_x + size_x * core_y) // 7 + 1) * 128),
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3b3_output_raw_data.shape[2]))
        conv3b3_output_split_dict2[(core_x, core_y)] = ((0, 512),
                                                        (core_x + core_y * size_x, core_x + core_y * size_x + 1),
                                                        (0, 14))
        conv3b3_output_split_dict3[(core_x, core_y)] = ((0, 512),
                                                        (core_x + core_y * size_x, core_x + core_y * size_x + 1),
                                                        (14, 28))
    data['conv3b3_output'] = ResNetDataHandler.tensor_split(raw_data=conv3b3_output_raw_data,
                                                            split_dict=conv3b3_output_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])
    data['conv3b3_output2'] = ResNetDataHandler.tensor_split(raw_data=conv3b3_output_raw_data,
                                                             split_dict=conv3b3_output_split_dict2,
                                                             data_type=1, alignment=(32, None, None),
                                                             dims=[0, 2, 1])
    data['conv3b3_output3'] = ResNetDataHandler.tensor_split(raw_data=conv3b3_output_raw_data,
                                                             split_dict=conv3b3_output_split_dict3,
                                                             data_type=1, alignment=(32, None, None),
                                                             dims=[0, 2, 1])

    add_output_raw_data = handler.parameters['layer2.1.cut5']['output'].astype(np.int8)
    add_output_split_dict = {}
    add_output_split_dict1, add_output_split_dict2 = {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        add_output_split_dict[(core_x, core_y)] = ((0, add_output_raw_data.shape[0]),
                                                   (core_x + core_y * size_x, core_x + core_y * size_x + 1),
                                                   (0, add_output_raw_data.shape[2]))
        add_output_split_dict1[(core_x, core_y)] = ((0, add_output_raw_data.shape[0]),
                                                    (core_x + core_y * size_x, core_x + core_y * size_x + 1),
                                                    (0, 14))
        add_output_split_dict2[(core_x, core_y)] = ((0, add_output_raw_data.shape[0]),
                                                    (core_x + core_y * size_x, core_x + core_y * size_x + 1),
                                                    (14, 28))
    data['g6_add_output'] = ResNetDataHandler.tensor_split(raw_data=add_output_raw_data,
                                                           split_dict=add_output_split_dict,
                                                           data_type=1, alignment=(32, None, None),
                                                           dims=[0, 2, 1])
    data['g6_add_output1'] = ResNetDataHandler.tensor_split(raw_data=add_output_raw_data,
                                                            split_dict=add_output_split_dict1,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])
    data['g6_add_output2'] = ResNetDataHandler.tensor_split(raw_data=add_output_raw_data,
                                                            split_dict=add_output_split_dict2,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    return data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g6_data(handler, size_y=2, size_x=14)
