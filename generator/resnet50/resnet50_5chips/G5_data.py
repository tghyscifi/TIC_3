from itertools import product
import sys
import os

sys.path.append(os.getcwd())
from generator.resnet50.data_handler import ResNetDataHandler
import numpy as np


def generate_g5_data(handler, size_y, size_x):
    data = {}

    # Conv3a1 input
    conv3a1_input_raw_data = handler.parameters['layer2.0.conv1']['input'].astype(np.int8)[:, ::2, ::2]
    conv3a1_input_split_dict = {}
    conv3a1_input_split_dict1, conv3a1_input_split_dict2, conv3a1_input_split_dict3, conv3a1_input_split_dict4 = \
        {}, {}, {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a1_input_split_dict[(core_x, core_y)] = ((0, conv3a1_input_raw_data.shape[0]),
                                                      (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                      (0, conv3a1_input_raw_data.shape[2]))

        conv3a1_input_split_dict1[(core_x, core_y)] = ((0, conv3a1_input_raw_data.shape[0]),
                                                       (core_x % 7 * 4, (core_x % 7) * 4 + 1),
                                                       (0, conv3a1_input_raw_data.shape[2]))
        conv3a1_input_split_dict2[(core_x, core_y)] = ((0, conv3a1_input_raw_data.shape[0]),
                                                       (core_x % 7 * 4 + 1, (core_x % 7) * 4 + 2),
                                                       (0, conv3a1_input_raw_data.shape[2]))
        conv3a1_input_split_dict3[(core_x, core_y)] = ((0, conv3a1_input_raw_data.shape[0]),
                                                       (core_x % 7 * 4 + 2, (core_x % 7) * 4 + 3),
                                                       (0, conv3a1_input_raw_data.shape[2]))
        conv3a1_input_split_dict4[(core_x, core_y)] = ((0, conv3a1_input_raw_data.shape[0]),
                                                       (core_x % 7 * 4 + 3, (core_x % 7) * 4 + 4),
                                                       (0, conv3a1_input_raw_data.shape[2]))

    data['conv3a1_input'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_input_raw_data,
                                                           split_dict=conv3a1_input_split_dict,
                                                           data_type=1, alignment=(32, None, None), dims=[0, 2, 1])

    data['conv3a1_input1'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_input_raw_data,
                                                            split_dict=conv3a1_input_split_dict1,
                                                            data_type=1, alignment=(32, None, None), dims=[0, 2, 1])
    data['conv3a1_input2'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_input_raw_data,
                                                            split_dict=conv3a1_input_split_dict2,
                                                            data_type=1, alignment=(32, None, None), dims=[0, 2, 1])
    data['conv3a1_input3'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_input_raw_data,
                                                            split_dict=conv3a1_input_split_dict3,
                                                            data_type=1, alignment=(32, None, None), dims=[0, 2, 1])
    data['conv3a1_input4'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_input_raw_data,
                                                            split_dict=conv3a1_input_split_dict4,
                                                            data_type=1, alignment=(32, None, None), dims=[0, 2, 1])

    # Conv3a1 weight
    conv3a1_weight_raw_data = handler.parameters['layer2.0.conv1']['weight'].astype(np.int8)
    conv3a1_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a1_weight_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                        ((core_x + core_y * size_x) // 7 + 1) * 32),
                                                       (0, conv3a1_weight_raw_data.shape[1]),
                                                       (0, conv3a1_weight_raw_data.shape[2]),
                                                       (0, conv3a1_weight_raw_data.shape[3]))
    data['conv3a1_weight'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_weight_raw_data,
                                                            split_dict=conv3a1_weight_split_dict,
                                                            data_type=1, alignment=(32, None, None, None),
                                                            dims=[0, 1, 3, 2], is_weight=True)

    # Conv3a1 bias
    conv3a1_bias_raw_data = handler.parameters['layer2.0.conv1']['bias'].astype(np.int32)
    conv3a1_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a1_bias_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                      ((core_x + core_y * size_x) // 7 + 1) * 32),)
    data['conv3a1_bias'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_bias_raw_data,
                                                          split_dict=conv3a1_bias_split_dict,
                                                          data_type=0, alignment=(32,), dims=[0])

    # Conv3a1 output
    conv3a1_output_raw_data = handler.parameters['layer2.0.cut1']['output'].astype(np.int8)
    conv3a1_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a1_output_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                        ((core_x + core_y * size_x) // 7 + 1) * 32),
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3a1_output_raw_data.shape[2]))
    data['conv3a1_output'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_output_raw_data,
                                                            split_dict=conv3a1_output_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    # Conv3a2 collation0
    conv3a2_collation0_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a2_collation0_split_dict[(core_x, core_y)] = ((0, conv3a1_output_raw_data.shape[0]),
                                                           (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                           (0, conv3a1_output_raw_data.shape[2]))
    data['conv3a2_collation0'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_output_raw_data,
                                                                split_dict=conv3a2_collation0_split_dict,
                                                                data_type=1, alignment=(32, None, None),
                                                                dims=[0, 2, 1])

    # Conv3a2 collation1
    conv3a2_collation1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x % 7 == 0:
            conv3a2_collation1_split_dict[(core_x, core_y)] = ((0, conv3a1_output_raw_data.shape[0]),
                                                               (4, 5),
                                                               (0, conv3a1_output_raw_data.shape[2]))
        elif (core_x + 1) % 7 == 0:
            conv3a2_collation1_split_dict[(core_x, core_y)] = ((0, conv3a1_output_raw_data.shape[0]),
                                                               (23, 24),
                                                               (0, conv3a1_output_raw_data.shape[2]))
        else:
            conv3a2_collation1_split_dict[(core_x, core_y)] = ((0, conv3a1_output_raw_data.shape[0]),
                                                               (core_x % 7 * 4 - 1, core_x % 7 * 4),
                                                               (0, conv3a1_output_raw_data.shape[2]))
    data['conv3a2_collation1'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_output_raw_data,
                                                                split_dict=conv3a2_collation1_split_dict,
                                                                data_type=1, alignment=(32, None, None),
                                                                dims=[0, 2, 1])

    # Conv3a2 input
    conv3a2_input_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x % 7 == 0:
            conv3a2_input_split_dict[(core_x, core_y)] = ((0, conv3a1_output_raw_data.shape[0]),
                                                          (0, 5),
                                                          (0, conv3a1_output_raw_data.shape[2]))
        elif (core_x + 1) % 7 == 0:
            conv3a2_input_split_dict[(core_x, core_y)] = ((0, conv3a1_output_raw_data.shape[0]),
                                                          (23, 28),
                                                          (0, conv3a1_output_raw_data.shape[2]))
        else:
            conv3a2_input_split_dict[(core_x, core_y)] = ((0, conv3a1_output_raw_data.shape[0]),
                                                          (core_x % 7 * 4 - 1, (core_x % 7 + 1) * 4 + 1),
                                                          (0, conv3a1_output_raw_data.shape[2]))

    data['conv3a2_input'] = ResNetDataHandler.tensor_split(raw_data=conv3a1_output_raw_data,
                                                           split_dict=conv3a2_input_split_dict,
                                                           data_type=1, alignment=(32, None, None), dims=[0, 2, 1])

    # Conv3a2 weight
    conv3a2_weight_raw_data = handler.parameters['layer2.0.conv2']['weight'].astype(np.int8)
    conv3a2_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a2_weight_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                        ((core_x + core_y * size_x) // 7 + 1) * 32),
                                                       (0, conv3a2_weight_raw_data.shape[1]),
                                                       (0, conv3a2_weight_raw_data.shape[2]),
                                                       (0, conv3a2_weight_raw_data.shape[3]))
    data['conv3a2_weight'] = ResNetDataHandler.tensor_split(raw_data=conv3a2_weight_raw_data,
                                                            split_dict=conv3a2_weight_split_dict,
                                                            data_type=1, alignment=(32, None, None, None),
                                                            dims=[0, 1, 3, 2], is_weight=True)

    # Conv3a2 bias
    conv3a2_bias_raw_data = handler.parameters['layer2.0.conv2']['bias'].astype(np.int32)
    conv3a2_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a2_bias_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                      ((core_x + core_y * size_x) // 7 + 1) * 32),)
    data['conv3a2_bias'] = ResNetDataHandler.tensor_split(raw_data=conv3a2_bias_raw_data,
                                                          split_dict=conv3a2_bias_split_dict,
                                                          data_type=0, alignment=(32,), dims=[0])

    # Conv3a2 output
    conv3a2_output_raw_data = handler.parameters['layer2.0.cut2']['output'].astype(np.int8)
    conv3a2_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a2_output_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 32,
                                                        ((core_x + core_y * size_x) // 7 + 1) * 32),
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3a2_output_raw_data.shape[2]))
    data['conv3a2_output'] = ResNetDataHandler.tensor_split(raw_data=conv3a2_output_raw_data,
                                                            split_dict=conv3a2_output_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    # Conv3a3 input
    conv3a3_input_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a3_input_split_dict[(core_x, core_y)] = ((0, conv3a2_output_raw_data.shape[0]),
                                                      (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                      (0, conv3a2_output_raw_data.shape[2]))
    data['conv3a3_input'] = ResNetDataHandler.tensor_split(raw_data=conv3a2_output_raw_data,
                                                           split_dict=conv3a3_input_split_dict,
                                                           data_type=1, alignment=(32, None, None),
                                                           dims=[0, 2, 1])

    # Conv3a3 weight
    conv3a3_weight_raw_data = handler.parameters['layer2.0.conv3']['weight'].astype(np.int8)
    conv3a3_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a3_weight_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                                        ((core_x + core_y * size_x) // 7 + 1) * 128),
                                                       (0, conv3a3_weight_raw_data.shape[1]),
                                                       (0, conv3a3_weight_raw_data.shape[2]),
                                                       (0, conv3a3_weight_raw_data.shape[3]))
    data['conv3a3_weight'] = ResNetDataHandler.tensor_split(raw_data=conv3a3_weight_raw_data,
                                                            split_dict=conv3a3_weight_split_dict,
                                                            data_type=1, alignment=(32, None, None, None),
                                                            dims=[0, 1, 3, 2], is_weight=True)

    # Conv3a3 bias
    conv3a3_bias_raw_data = handler.parameters['layer2.0.conv3']['bias'].astype(np.int32)
    conv3a3_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a3_bias_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                                      ((core_x + core_y * size_x) // 7 + 1) * 128),)
    data['conv3a3_bias'] = ResNetDataHandler.tensor_split(raw_data=conv3a3_bias_raw_data,
                                                          split_dict=conv3a3_bias_split_dict,
                                                          data_type=0, alignment=(32,), dims=[0])

    # Conv3a3 output
    conv3a3_output_raw_data = handler.parameters['layer2.0.cut3']['output'].astype(np.int8)
    conv3a3_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a3_output_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                                        ((core_x + core_y * size_x) // 7 + 1) * 128),
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3a3_output_raw_data.shape[2]))
    data['conv3a3_output'] = ResNetDataHandler.tensor_split(raw_data=conv3a3_output_raw_data,
                                                            split_dict=conv3a3_output_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    # Conv3a3e input
    conv3a3e_input_raw_data = handler.parameters['layer2.0.downsample.0']['input'].astype(np.int8)[:, ::2, ::2]
    conv3a3e_input_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv3a3e_input_split_dict[(core_x, core_y)] = ((0, conv3a3e_input_raw_data.shape[0]),
                                                       (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                       (0, conv3a3e_input_raw_data.shape[2]))
    data['conv3a3e_input'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_input_raw_data,
                                                            split_dict=conv3a3e_input_split_dict,
                                                            data_type=1, alignment=(32, None, None),
                                                            dims=[0, 2, 1])

    # Conv3a3e weight
    conv3a3e_weight_raw_data = handler.parameters['layer2.0.downsample.0']['weight'].astype(np.int8)
    conv3a3e_weight_split_dict = {}
    for core_y, core_x in product(range(1), range(size_x)):
        conv3a3e_weight_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 256,
                                                         ((core_x + core_y * size_x) // 7 + 1) * 256),
                                                        (0, conv3a3e_weight_raw_data.shape[1]),
                                                        (0, conv3a3e_weight_raw_data.shape[2]),
                                                        (0, conv3a3e_weight_raw_data.shape[3]))
    data['conv3a3e_weight'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_weight_raw_data,
                                                             split_dict=conv3a3e_weight_split_dict,
                                                             data_type=1, alignment=(32, None, None, None),
                                                             dims=[0, 1, 3, 2], is_weight=True)

    # Conv3a3e bias
    conv3a3e_bias_raw_data = handler.parameters['layer2.0.downsample.0']['bias'].astype(np.int32)
    conv3a3e_bias_split_dict = {}
    for core_y, core_x in product(range(1), range(size_x)):
        conv3a3e_bias_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 256,
                                                       ((core_x + core_y * size_x) // 7 + 1) * 256),)
    data['conv3a3e_bias'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_bias_raw_data,
                                                           split_dict=conv3a3e_bias_split_dict,
                                                           data_type=0, alignment=(32,), dims=[0])

    # Conv3a3e output
    conv3a3e_output_raw_data = handler.parameters['layer2.0.cut4']['output'].astype(np.int8)
    conv3a3e_output_split_dict = {}
    for core_y, core_x in product(range(1), range(size_x)):
        conv3a3e_output_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 256,
                                                         ((core_x + core_y * size_x) // 7 + 1) * 256),
                                                        (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                        (0, conv3a3e_output_raw_data.shape[2]))
    data['conv3a3e_output'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_output_raw_data,
                                                             split_dict=conv3a3e_output_split_dict,
                                                             data_type=1, alignment=(32, None, None),
                                                             dims=[0, 2, 1])

    # Instant prim
    instant_prim0_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        instant_prim0_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                                       ((core_x + core_y * size_x) // 7 + 1) * 128),
                                                      (core_x % 7 * 4, core_x % 7 * 4 + 2),
                                                      (0, conv3a3e_output_raw_data.shape[2]))
    data['instant_prim0'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_output_raw_data,
                                                           split_dict=instant_prim0_split_dict,
                                                           data_type=1, alignment=(32, None, None),
                                                           dims=[0, 2, 1])

    instant_prim1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        instant_prim1_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                                       ((core_x + core_y * size_x) // 7 + 1) * 128),
                                                      (core_x % 7 * 4 + 2, core_x % 7 * 4 + 4),
                                                      (0, conv3a3e_output_raw_data.shape[2]))
    data['instant_prim1'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_output_raw_data,
                                                           split_dict=instant_prim1_split_dict,
                                                           data_type=1, alignment=(32, None, None),
                                                           dims=[0, 2, 1])

    # # Add input
    # add_input1_split_dict = {}
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     add_input1_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128, 
    #                                                ((core_x + core_y * size_x) // 7 + 1) * 128),
    #                                                (core_x % 7 * 4, (core_x % 7 + 1) * 4),
    #                                                (0, conv3a3e_output_raw_data.shape[2]))
    # data['add_input1'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_output_raw_data,
    #                                                     split_dict=add_input1_split_dict,
    #                                                     data_type=1, alignment=(32, None, None), 
    #                                                     dims=[0, 2, 1])
    # data['add_input'] = {}
    # for core_y, core_x in product(range(size_y), range(size_x)):
    #     data['add_input'][(core_x, core_y)] = data['conv3a3_output'][(core_x, core_y)].extend(
    #     data['add_input1'][core_x, core_y])

    # test
    test0_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        test0_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                               ((core_x + core_y * size_x) // 7 + 1) * 128),
                                              (core_x % 7 * 4, core_x % 7 * 4 + 4),
                                              (0, conv3a3e_output_raw_data.shape[2]))
    data['test0'] = ResNetDataHandler.tensor_split(raw_data=conv3a3e_output_raw_data,
                                                   split_dict=test0_split_dict,
                                                   data_type=1, alignment=(32, None, None),
                                                   dims=[0, 2, 1])

    test1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        test1_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                               ((core_x + core_y * size_x) // 7 + 1) * 128),
                                              (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                              (0, conv3a3_output_raw_data.shape[2]))
    data['test1'] = ResNetDataHandler.tensor_split(raw_data=conv3a3_output_raw_data,
                                                   split_dict=test1_split_dict,
                                                   data_type=1, alignment=(32, None, None),
                                                   dims=[0, 2, 1])

    # Add output
    add_output_raw_data = handler.parameters['layer2.0.cut5']['output'].astype(np.int8)
    add_output_split_dict, add_output_split_dict1 = {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        add_output_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) // 7 * 128,
                                                    ((core_x + core_y * size_x) // 7 + 1) * 128),
                                                   (core_x % 7 * 4, (core_x % 7 + 1) * 4),
                                                   (0, add_output_raw_data.shape[2]))
    data['g5_add_output'] = ResNetDataHandler.tensor_split(raw_data=add_output_raw_data,
                                                           split_dict=add_output_split_dict,
                                                           data_type=1, alignment=(32, None, None),
                                                           dims=[0, 2, 1])
    data['g5_add_output1'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        data['g5_add_output1'][(core_x, core_y)] = [*data['g5_add_output'][(core_x, core_y)][448:],
                                                    *data['g5_add_output'][(core_x, core_y)][:448]]

    return data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    data = generate_g5_data(handler, size_y=2, size_x=14)
