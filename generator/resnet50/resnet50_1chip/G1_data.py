from itertools import product
import sys
import os
import torch
from torch.nn import ZeroPad2d

sys.path.append(os.getcwd())
from generator.resnet50.data_handler import ResNetDataHandler
import numpy as np


def generate_g1_data(handler, size_y, size_x):
    g1_data = {}

    # FCa2 input
    fca2_input_raw_data = handler.parameters['fc_0']['input'].astype(np.int8)
    fca2_input_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        fca2_input_split_dict[(core_x, core_y)] = ((0, fca2_input_raw_data.shape[0]),)

    g1_data['fc_0_input'] = ResNetDataHandler.tensor_split(raw_data=fca2_input_raw_data,
                                                           split_dict=fca2_input_split_dict,
                                                           data_type=1, alignment=(32,), dims=[0])

    # FC_0 weight
    fc_0_weight_raw_data = handler.parameters['fc_0']['weight'].astype(np.int8)
    fc_1_weight_raw_data = handler.parameters['fc_1']['weight'].astype(np.int8)
    fc_weight_raw_data = handler.parameters['fc']['weight'].astype(np.int8)
    fc_0_weight_split_dict, fc_1_weight_split_dict, fc_weight_split_dict = {}, {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        fc_0_weight_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) * 32,
                                                     (core_x + core_y * size_x + 1) * 32),
                                                    (0, fc_0_weight_raw_data.shape[1]))
        if core_x < 8 and core_y == 0:
            fc_1_weight_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) * 32,
                                                         (core_x + core_y * size_x + 1) * 32),
                                                        (0, fc_1_weight_raw_data.shape[1]))
    fc_weight_split_dict[(0, 0)] = ((0, 10), (0, 256))
    g1_data['fc_0_weight'] = ResNetDataHandler.tensor_split(raw_data=fc_0_weight_raw_data,
                                                            split_dict=fc_0_weight_split_dict,
                                                            data_type=1, alignment=(32, None), dims=[0, 1],
                                                            is_weight=True)
    g1_data['fc_1_weight'] = ResNetDataHandler.tensor_split(raw_data=fc_1_weight_raw_data,
                                                            split_dict=fc_1_weight_split_dict,
                                                            data_type=1, alignment=(32, None), dims=[0, 1],
                                                            is_weight=True)
    g1_data['fc_weight'] = ResNetDataHandler.tensor_split(raw_data=fc_weight_raw_data,
                                                          split_dict=fc_weight_split_dict,
                                                          data_type=1, alignment=(32, None), dims=[0, 1],
                                                          is_weight=True)

    # fc_0 bias
    fc_0_bias_raw_data = handler.parameters['fc_0']['bias'].astype(np.int32)
    fc_1_bias_raw_data = handler.parameters['fc_1']['bias'].astype(np.int32)
    fc_bias_raw_data = handler.parameters['fc']['bias'].astype(np.int32)
    fc_0_bias_split_dict, fc_1_bias_split_dict, fc_bias_split_dict = {}, {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        fc_0_bias_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) * 32,
                                                   (core_x + core_y * size_x + 1) * 32),)
        if core_x < 8 and core_y == 0:
            fc_1_bias_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) * 32,
                                                       (core_x + core_y * size_x + 1) * 32),)
    fc_bias_split_dict[(0, 0)] = ((0, 10),)

    g1_data['fc_0_bias'] = ResNetDataHandler.tensor_split(raw_data=fc_0_bias_raw_data,
                                                          split_dict=fc_0_bias_split_dict,
                                                          data_type=0, alignment=(32,), dims=[0])
    g1_data['fc_1_bias'] = ResNetDataHandler.tensor_split(raw_data=fc_1_bias_raw_data,
                                                          split_dict=fc_1_bias_split_dict,
                                                          data_type=0, alignment=(32,), dims=[0])
    g1_data['fc_bias'] = ResNetDataHandler.tensor_split(raw_data=fc_bias_raw_data,
                                                        split_dict=fc_bias_split_dict,
                                                        data_type=0, alignment=(32,), dims=[0])

    # FCa2 output
    fca2_output_raw_data = handler.parameters['fc_cut']['output'].astype(np.int8)
    fca2_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x == size_x - 1 and core_y == size_y - 1:
            fca2_output_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) * 32,
                                                         fca2_output_raw_data.shape[0]),)
        else:
            fca2_output_split_dict[(core_x, core_y)] = (((core_x + core_y * size_x) * 32,
                                                         (core_x + core_y * size_x + 1) * 32),)

    g1_data['fca2_output'] = ResNetDataHandler.tensor_split(raw_data=fca2_output_raw_data,
                                                            split_dict=fca2_output_split_dict,
                                                            data_type=1, alignment=(32,), dims=[0])
    g1_data['fca2_output_all'] = ResNetDataHandler.tensor_split(raw_data=fca2_output_raw_data,
                                                                split_dict={
                                                                    (0, 0): ((0, 32),)
                                                                },
                                                                data_type=1, alignment=(32,), dims=[0])
    g1_data['fc_0_cut'] = ResNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc_0_cut']['output'].astype(np.int8),
        split_dict={
            (0, 0): ((0, 1024),)
        },
        data_type=1, alignment=(32,), dims=[0])
    g1_data['fc_1_cut_0'] = ResNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc_1_cut']['output'].astype(np.int8),
        split_dict=fc_1_bias_split_dict,
        data_type=1, alignment=(32,), dims=[0])
    g1_data['fc_1_cut'] = ResNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc_1_cut']['output'].astype(np.int8),
        split_dict={
            (0, 0): ((0, 256),)
        },
        data_type=1, alignment=(32,), dims=[0])

    # Conv1a1 input
    conv1a1_input_raw_data = handler.parameters['conv1']['input'].astype(np.int8)
    conv1a1_input_C = conv1a1_input_raw_data.shape[0]
    conv1a1_input_H = conv1a1_input_raw_data.shape[1]
    conv1a1_input_W = conv1a1_input_raw_data.shape[2]
    temp = torch.tensor(conv1a1_input_raw_data)
    pad = ZeroPad2d((3, 0, 3, 2))
    temp = pad(temp)
    conv1a1_input_raw_data = np.array(temp).astype(np.int8)
    conv1a1_input_split_dict = {}
    conv1a1_input_split_dict1 = {}
    conv1a1_input_split_dict2, conv1a1_input_split_dict3, conv1a1_input_split_dict4 = {}, {}, {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            if core_x == 0:
                conv1a1_input_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                              (0, 17),
                                                              (0, 128))
            elif core_x == size_x - 1:
                conv1a1_input_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                              (213, 229),
                                                              (0, 128))
            else:
                conv1a1_input_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                              (3 + core_x * conv1a1_input_W // 16,
                                                               3 + (core_x + 1) * conv1a1_input_W // 16),
                                                              (0, 128))
        else:
            if core_x == 0:
                conv1a1_input_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                              (0, 17),
                                                              (102, 227))
            elif core_x == size_x - 1:
                conv1a1_input_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                              (213, 229),
                                                              (102, 227))
            else:
                conv1a1_input_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                              (3 + core_x * conv1a1_input_W // 16,
                                                               3 + (core_x + 1) * conv1a1_input_W // 16),
                                                              (102, 227))
    for core_x, core_y in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]:
        if core_x % 2 == 0:
            conv1a1_input_split_dict1[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (0, 229), (0, 128))

            conv1a1_input_split_dict2[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (0, 77), (0, 128))
            conv1a1_input_split_dict3[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (77, 153), (0, 128))
            conv1a1_input_split_dict4[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (153, 229), (0, 128))
        else:
            conv1a1_input_split_dict1[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (0, 229), (102, 227))

            conv1a1_input_split_dict2[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (0, 77), (102, 227))
            conv1a1_input_split_dict3[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (77, 153), (102, 227))
            conv1a1_input_split_dict4[(core_x, core_y)] = ((core_x // 2, core_x // 2 + 1), (153, 229), (102, 227))

    g1_data['conv1a1_input'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_input_raw_data,
                                                              split_dict=conv1a1_input_split_dict,
                                                              data_type=1, alignment=(None, None, 16),
                                                              dims=[2, 1, 0])
    g1_data['conv1a1_input1'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_input_raw_data,
                                                               split_dict=conv1a1_input_split_dict1,
                                                               data_type=1, alignment=(None, None, 16),
                                                               dims=[2, 1, 0])

    g1_data['conv1a1_input2'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_input_raw_data,
                                                               split_dict=conv1a1_input_split_dict2,
                                                               data_type=1, alignment=(None, None, 16),
                                                               dims=[2, 1, 0])
    g1_data['conv1a1_input3'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_input_raw_data,
                                                               split_dict=conv1a1_input_split_dict3,
                                                               data_type=1, alignment=(None, None, 16),
                                                               dims=[2, 1, 0])
    g1_data['conv1a1_input4'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_input_raw_data,
                                                               split_dict=conv1a1_input_split_dict4,
                                                               data_type=1, alignment=(None, None, 16),
                                                               dims=[2, 1, 0])

    # Conv1a1 weight
    conv1a1_weight_raw_data = handler.parameters['conv1']['weight'].astype(np.int8)
    conv1a1_weight_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv1a1_weight_split_dict[(core_x, core_y)] = ((0, conv1a1_weight_raw_data.shape[0]),
                                                       (0, conv1a1_weight_raw_data.shape[1]),
                                                       (0, conv1a1_weight_raw_data.shape[2]),
                                                       (0, conv1a1_weight_raw_data.shape[3]))

    g1_data['conv1a1_weight'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_weight_raw_data,
                                                               split_dict=conv1a1_weight_split_dict,
                                                               data_type=1, alignment=(32, None, None, None),
                                                               dims=[0, 3, 2, 1], is_weight=True)

    # Conv1a1 bias
    conv1a1_bias_raw_data = handler.parameters['conv1']['bias'].astype(np.int32)
    conv1a1_bias_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        conv1a1_bias_split_dict[(core_x, core_y)] = ((0, conv1a1_bias_raw_data.shape[0]),)

    g1_data['conv1a1_bias'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_bias_raw_data,
                                                             split_dict=conv1a1_bias_split_dict,
                                                             data_type=0, alignment=(32,), dims=[0])

    # Conv1a1 collation0
    conv1a1_collation0_raw_data = handler.parameters['conv1']['input'].astype(np.int8)
    conv1a1_collation0_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            if core_x == 0:
                conv1a1_collation0_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (0, 14),
                                                                   (0, 128))
            elif core_x == size_x - 1:
                conv1a1_collation0_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (207, 224),
                                                                   (0, 128))
            else:
                conv1a1_collation0_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (core_x * conv1a1_input_W // 16 - 3,
                                                                    (core_x + 1) * conv1a1_input_W // 16),
                                                                   (0, 128))
        else:
            if core_x == 0:
                conv1a1_collation0_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (0, 14),
                                                                   (conv1a1_input_H - 128, conv1a1_input_H))
            elif core_x == size_x - 1:
                conv1a1_collation0_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (207, 224),
                                                                   (conv1a1_input_H - 128, conv1a1_input_H))
            else:
                conv1a1_collation0_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (core_x * conv1a1_input_W // 16 - 3,
                                                                    (core_x + 1) * conv1a1_input_W // 16),
                                                                   (conv1a1_input_H - 128, conv1a1_input_H))

    g1_data['conv1a1_collation0'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_collation0_raw_data,
                                                                   split_dict=conv1a1_collation0_split_dict,
                                                                   data_type=1, alignment=(None, None, 32),
                                                                   dims=[2, 1, 0])

    # Conv1a1 collation1
    conv1a1_collation1_raw_data = handler.parameters['conv1']['input'].astype(np.int8)
    conv1a1_collation1_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            if core_x == 0:
                conv1a1_collation1_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (0, 16),
                                                                   (0, 128))
            elif core_x == size_x - 1:
                conv1a1_collation1_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (207, 224),
                                                                   (0, 128))
            else:
                conv1a1_collation1_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (core_x * conv1a1_input_W // 16 - 3,
                                                                    (core_x + 1) * conv1a1_input_W // 16 + 2),
                                                                   (0, 128))
        else:
            if core_x == 0:
                conv1a1_collation1_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (0, 16),
                                                                   (conv1a1_input_H - 128, conv1a1_input_H))
            elif core_x == size_x - 1:
                conv1a1_collation1_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (207, 224),
                                                                   (conv1a1_input_H - 128, conv1a1_input_H))
            else:
                conv1a1_collation1_split_dict[(core_x, core_y)] = ((0, conv1a1_input_C),
                                                                   (core_x * conv1a1_input_W // 16 - 3,
                                                                    (core_x + 1) * conv1a1_input_W // 16 + 2),
                                                                   (conv1a1_input_H - 128, conv1a1_input_H))

    g1_data['conv1a1_collation1'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_collation1_raw_data,
                                                                   split_dict=conv1a1_collation1_split_dict,
                                                                   data_type=1, alignment=(None, None, 32),
                                                                   dims=[2, 1, 0])

    # Conv1a1 output
    conv1a1_output_raw_data = handler.parameters['conv1_cut']['output'].astype(np.int8)
    conv1a1_output_split_dict = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_y == 0:
            conv1a1_output_split_dict[(core_x, core_y)] = ((0, conv1a1_output_raw_data.shape[0]),
                                                           (core_x * 7, (core_x + 1) * 7),
                                                           (0, 61))
        else:
            conv1a1_output_split_dict[(core_x, core_y)] = ((0, conv1a1_output_raw_data.shape[0]),
                                                           (core_x * 7, (core_x + 1) * 7),
                                                           (112 - 61, 112))
    g1_data['conv1a1_output'] = ResNetDataHandler.tensor_split(raw_data=conv1a1_output_raw_data,
                                                               split_dict=conv1a1_output_split_dict,
                                                               data_type=1, alignment=(32, None, None), dims=[0, 2, 1])

    # MaxPool1a2 output
    maxpool1a2_output_raw_data = handler.parameters['maxpool']['output'].astype(np.int8)
    maxpool1a2_output_split_dict = {}
    maxpool1a2_output_C = maxpool1a2_output_raw_data.shape[0]
    maxpool1a2_output_H = maxpool1a2_output_raw_data.shape[1]
    maxpool1a2_output_W = maxpool1a2_output_raw_data.shape[2]
    for core_y, core_x in product(range(size_y), range(size_x)):
        if core_x % 2 == 0:
            maxpool1a2_output_split_dict[(core_x, core_y)] = ((0, maxpool1a2_output_C),
                                                              (core_x // 2 * maxpool1a2_output_H // 8,
                                                               core_x // 2 * maxpool1a2_output_H // 8 + 4),
                                                              (core_y * maxpool1a2_output_W // 2,
                                                               (core_y + 1) * maxpool1a2_output_W // 2))
        else:
            maxpool1a2_output_split_dict[(core_x, core_y)] = ((0, maxpool1a2_output_C),
                                                              (core_x // 2 * maxpool1a2_output_H // 8 + 4,
                                                               (core_x + 1) // 2 * maxpool1a2_output_H // 8),
                                                              (core_y * maxpool1a2_output_W // 2,
                                                               (core_y + 1) * maxpool1a2_output_W // 2))

    g1_data['maxpool1a2_output'] = ResNetDataHandler.tensor_split(raw_data=maxpool1a2_output_raw_data,
                                                                  split_dict=maxpool1a2_output_split_dict,
                                                                  data_type=1, alignment=(16, None, None),
                                                                  dims=[0, 2, 1])

    g1_data['mp_out_with_fc_out'] = {}
    for core_y, core_x in product(range(size_y), range(size_x)):
        g1_data['mp_out_with_fc_out'][(core_x, core_y)] = [
            *g1_data['maxpool1a2_output'][(core_x, core_y)],
            *g1_data['fca2_output'][(core_x, core_y)]
        ]

    return g1_data


if __name__ == '__main__':
    handler = ResNetDataHandler()
    g1_data = generate_g1_data(handler, size_y=2, size_x=16)
