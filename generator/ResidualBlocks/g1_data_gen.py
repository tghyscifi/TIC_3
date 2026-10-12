from itertools import product
from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
import numpy as np

Sm = 1
Sn = 1
Sj = 1
Si = 40


def generate_data(handler, size_x, size_y):
    g_data = {
        'conv_in': {},
        'activation_in': {},
        'cut_in': {},

        'conv0': {},
        'activation0': {},
        'cut0': {},
        'activation1': {},
        'conv1': {},
        'cut_0': {},

        'conv2': {},
        'activation2': {},
        'cut2': {},
        'activation3': {},
        'conv3': {},
        'cut_1': {},

        'conv4': {},
        'activation4': {},
        'cut4': {},
        'activation5': {},
        'conv5': {},
        'cut_2': {},

        'conv_out': {},
        'activation_out': {},
        'cut_out': {}
    }

    # input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    
    ###############################################################################################################
    # conv_in input
    conv_in_input_raw = handler.parameters['conv_in']['input'].astype(np.int8)
    conv_in_input_padded = np.pad(conv_in_input_raw, ((0, 0), (1, 1), (1, 1)), 'constant')
    conv_in_input_split_dict = {}
    line_split_step = int(conv_in_input_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                conv_in_input_split_dict[(core_x, core_y)] = ((0, conv_in_input_padded.shape[0]),
                                                              (core_idx * line_split_step,
                                                               core_idx * line_split_step + line_split_step + 2),
                                                              (0, conv_in_input_padded.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                conv_in_input_split_dict[(core_x, core_y)] = ((0, conv_in_input_padded.shape[0]),
                                                              (core_idx * line_split_step,
                                                               core_idx * line_split_step + line_split_step + 2),
                                                              (0, conv_in_input_padded.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['conv_in']['input'] = ModelDataHandler.tensor_split(
        raw_data=conv_in_input_padded,
        split_dict=conv_in_input_split_dict,
        data_type=1,
        alignment=(None, None, 16),
        dims=(2, 1, 0)
    )


    # conv_in weight
    conv_in_weight_raw = handler.parameters['conv_in']['weight'].astype(np.int8)
    conv_in_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv_in_weight_split_dict[(core_x, core_y)] = (
            (0, conv_in_weight_raw.shape[0]), (0, conv_in_weight_raw.shape[1]), (0, conv_in_weight_raw.shape[2]))

    g_data['conv_in']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv_in_weight_raw,
        split_dict=conv_in_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 3, 2, 1),
        is_weight=True
    )

    # conv_in bias
    conv_in_bias_raw = handler.parameters['conv_in']['bias'].astype(np.int32)
    conv_in_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv_in_bias_split_dict[(core_x, core_y)] = ((0, conv_in_bias_raw.shape[0]),)

    g_data['conv_in']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv_in_bias_raw,
        split_dict=conv_in_bias_split_dict,
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )

    # cut_in output
    cut_in_output_raw = handler.parameters['cut_in']['output'].astype(np.int8)
    cut_in_output_split_dict = {}
    line_split_step = int(cut_in_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut_in_output_split_dict[(core_x, core_y)] = ((0, cut_in_output_raw.shape[0]),
                                                              (core_idx * line_split_step,
                                                               core_idx * line_split_step + line_split_step),
                                                              (0, cut_in_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut_in_output_split_dict[(core_x, core_y)] = ((0, cut_in_output_raw.shape[0]),
                                                              (core_idx * line_split_step,
                                                               core_idx * line_split_step + line_split_step),
                                                              (0, cut_in_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut_in']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut_in_output_raw,
        split_dict=cut_in_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    ###############################################################################################################
    
    # activation_in output
    # conv0 input
    conv0_input_raw = handler.parameters['conv0']['input'].astype(np.int8)
    conv0_input_split_dict = {}
    line_split_step = int(conv0_input_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                conv0_input_split_dict[(core_x, core_y)] = ((0, conv0_input_raw.shape[0]),
                                                            (core_idx * line_split_step, core_idx * line_split_step + line_split_step),
                                                            (0, conv0_input_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                conv0_input_split_dict[(core_x, core_y)] = ((0, conv0_input_raw.shape[0]),
                                                            (core_idx * line_split_step, core_idx * line_split_step + line_split_step),
                                                            (0, conv0_input_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['conv0']['input'] = ModelDataHandler.tensor_split(
        raw_data=conv0_input_raw,
        split_dict=conv0_input_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )




    # conv0 weight
    conv0_weight_raw = handler.parameters['conv0']['weight'].astype(np.int8)
    conv0_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv0_weight_split_dict[(core_x, core_y)] = ((0, conv0_weight_raw.shape[0]), (0, conv0_weight_raw.shape[1]), (0, conv0_weight_raw.shape[2]))

    g_data['conv0']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv0_weight_raw,
        split_dict=conv0_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )

    # conv0 bias
    conv0_bias_raw = handler.parameters['conv0']['bias'].astype(np.int32)
    conv0_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv0_bias_split_dict[(core_x, core_y)] = ((0, conv0_bias_raw.shape[0]),)

    g_data['conv0']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv0_bias_raw,
        split_dict=conv0_bias_split_dict,
        data_type=0,
        alignment=(32, ),
        dims=(0, )
    )

    # cut0 output
    cut0_output_raw = handler.parameters['cut0']['output'].astype(np.int8)
    cut0_output_split_dict = {}
    line_split_step = int(cut0_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut0_output_split_dict[(core_x, core_y)] = ((0, cut0_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut0_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut0_output_split_dict[(core_x, core_y)] = ((0, cut0_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut0_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut0']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut0_output_raw,
        split_dict=cut0_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # activation0
    act0_output_raw = handler.parameters['activation0']['output'].astype(np.int8)
    act0_output_split_dict = {}
    line_split_step = int(act0_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                act0_output_split_dict[(core_x, core_y)] = ((0, act0_output_raw.shape[0]),
                                                            (core_idx * line_split_step, core_idx * line_split_step + line_split_step),
                                                            (0, act0_output_raw.shape[2]))
                core_idx = core_idx + 1
        else:
            for core_x in range(size_x - 1, -1, -1):
                act0_output_split_dict[(core_x, core_y)] = ((0, act0_output_raw.shape[0]),
                                                            (core_idx * line_split_step, core_idx * line_split_step + line_split_step),
                                                            (0, act0_output_raw.shape[2]))
                core_idx = core_idx + 1

    g_data['activation0']['output'] = ModelDataHandler.tensor_split(
        raw_data=act0_output_raw,
        split_dict=act0_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # conv1 weight
    conv1_weight_raw = handler.parameters['conv1']['weight'].astype(np.int8)
    conv1_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv1_weight_split_dict[(core_x, core_y)] = (
            (0, conv1_weight_raw.shape[0]), (0, conv1_weight_raw.shape[1]), (0, conv1_weight_raw.shape[2]))

    g_data['conv1']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv1_weight_raw,
        split_dict=conv1_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )

    # conv1 bias
    conv1_bias_raw = handler.parameters['conv1']['bias'].astype(np.int32)
    conv1_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv1_bias_split_dict[(core_x, core_y)] = ((0, conv1_bias_raw.shape[0]),)

    g_data['conv1']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv1_bias_raw,
        split_dict=conv1_bias_split_dict,
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )

    # activation1
    act1_output_raw = handler.parameters['activation1']['output'].astype(np.int8)
    act1_output_split_dict = {}
    line_split_step = int(act1_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                act1_output_split_dict[(core_x, core_y)] = ((0, act1_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act1_output_raw.shape[2]))
                core_idx = core_idx + 1
        else:
            for core_x in range(size_x - 1, -1, -1):
                act1_output_split_dict[(core_x, core_y)] = ((0, act1_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act1_output_raw.shape[2]))
                core_idx = core_idx + 1

    g_data['activation1']['output'] = ModelDataHandler.tensor_split(
        raw_data=act1_output_raw,
        split_dict=act1_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # cut_0 output
    cut_0_output_raw = handler.parameters['cut_0']['output'].astype(np.int8)
    cut_0_output_split_dict = {}
    line_split_step = int(cut_0_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut_0_output_split_dict[(core_x, core_y)] = ((0, cut_0_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut_0_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut_0_output_split_dict[(core_x, core_y)] = ((0, cut_0_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut_0_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut_0']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut_0_output_raw,
        split_dict=cut_0_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    g_data['conv2']['input'] = g_data['cut_0']['output']

    ###############################################################################################################
    # conv2 weight
    conv2_weight_raw = handler.parameters['conv2']['weight'].astype(np.int8)
    conv2_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv2_weight_split_dict[(core_x, core_y)] = (
            (0, conv2_weight_raw.shape[0]), (0, conv2_weight_raw.shape[1]), (0, conv2_weight_raw.shape[2]))

    g_data['conv2']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv2_weight_raw,
        split_dict=conv2_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )

    # conv2 bias
    conv2_bias_raw = handler.parameters['conv2']['bias'].astype(np.int32)
    conv2_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv2_bias_split_dict[(core_x, core_y)] = ((0, conv2_bias_raw.shape[0]),)

    g_data['conv2']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv2_bias_raw,
        split_dict=conv2_bias_split_dict,
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )

    # cut2 output
    cut2_output_raw = handler.parameters['cut2']['output'].astype(np.int8)
    cut2_output_split_dict = {}
    line_split_step = int(cut2_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut2_output_split_dict[(core_x, core_y)] = ((0, cut2_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut2_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut2_output_split_dict[(core_x, core_y)] = ((0, cut2_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut2_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut2']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut2_output_raw,
        split_dict=cut2_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # activation2
    act2_output_raw = handler.parameters['activation2']['output'].astype(np.int8)
    act2_output_split_dict = {}
    line_split_step = int(act2_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                act2_output_split_dict[(core_x, core_y)] = ((0, act2_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act2_output_raw.shape[2]))
                core_idx = core_idx + 1
        else:
            for core_x in range(size_x - 1, -1, -1):
                act2_output_split_dict[(core_x, core_y)] = ((0, act2_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act2_output_raw.shape[2]))
                core_idx = core_idx + 1

    g_data['activation2']['output'] = ModelDataHandler.tensor_split(
        raw_data=act2_output_raw,
        split_dict=act2_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # conv3 weight
    conv3_weight_raw = handler.parameters['conv3']['weight'].astype(np.int8)
    conv3_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv3_weight_split_dict[(core_x, core_y)] = (
                (0, conv3_weight_raw.shape[0]), (0, conv3_weight_raw.shape[1]), (0, conv3_weight_raw.shape[2]))

    g_data['conv3']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv3_weight_raw,
        split_dict=conv3_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )

    # conv3 bias
    conv3_bias_raw = handler.parameters['conv3']['bias'].astype(np.int32)
    conv3_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv3_bias_split_dict[(core_x, core_y)] = ((0, conv3_bias_raw.shape[0]),)

    g_data['conv3']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv3_bias_raw,
        split_dict=conv3_bias_split_dict,
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )

    # activation3
    act3_output_raw = handler.parameters['activation3']['output'].astype(np.int8)
    act3_output_split_dict = {}
    line_split_step = int(act3_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                act3_output_split_dict[(core_x, core_y)] = ((0, act3_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act3_output_raw.shape[2]))
                core_idx = core_idx + 1
        else:
            for core_x in range(size_x - 1, -1, -1):
                act3_output_split_dict[(core_x, core_y)] = ((0, act3_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act3_output_raw.shape[2]))
                core_idx = core_idx + 1

    g_data['activation3']['output'] = ModelDataHandler.tensor_split(
        raw_data=act3_output_raw,
        split_dict=act3_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # cut_1 output
    cut_1_output_raw = handler.parameters['cut_1']['output'].astype(np.int8)
    cut_1_output_split_dict = {}
    line_split_step = int(cut_1_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut_1_output_split_dict[(core_x, core_y)] = ((0, cut_1_output_raw.shape[0]),
                                                             (core_idx * line_split_step,
                                                              core_idx * line_split_step + line_split_step),
                                                             (0, cut_1_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut_1_output_split_dict[(core_x, core_y)] = ((0, cut_1_output_raw.shape[0]),
                                                             (core_idx * line_split_step,
                                                              core_idx * line_split_step + line_split_step),
                                                             (0, cut_1_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut_1']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut_1_output_raw,
        split_dict=cut_1_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    g_data['conv4']['input'] = g_data['cut_1']['output']
    ###############################################################################################################
    # conv4 weight
    conv4_weight_raw = handler.parameters['conv4']['weight'].astype(np.int8)
    conv4_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv4_weight_split_dict[(core_x, core_y)] = (
                (0, conv4_weight_raw.shape[0]), (0, conv4_weight_raw.shape[1]), (0, conv4_weight_raw.shape[2]))

    g_data['conv4']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv4_weight_raw,
        split_dict=conv4_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )

    # conv4 bias
    conv4_bias_raw = handler.parameters['conv4']['bias'].astype(np.int32)
    conv4_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv4_bias_split_dict[(core_x, core_y)] = ((0, conv4_bias_raw.shape[0]),)

    g_data['conv4']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv4_bias_raw,
        split_dict=conv4_bias_split_dict,
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )

    # cut4 output
    cut4_output_raw = handler.parameters['cut4']['output'].astype(np.int8)
    cut4_output_split_dict = {}
    line_split_step = int(cut4_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut4_output_split_dict[(core_x, core_y)] = ((0, cut4_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut4_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut4_output_split_dict[(core_x, core_y)] = ((0, cut4_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, cut4_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut4']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut4_output_raw,
        split_dict=cut4_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # activation4
    act4_output_raw = handler.parameters['activation4']['output'].astype(np.int8)
    act4_output_split_dict = {}
    line_split_step = int(act4_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                act4_output_split_dict[(core_x, core_y)] = ((0, act4_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act4_output_raw.shape[2]))
                core_idx = core_idx + 1
        else:
            for core_x in range(size_x - 1, -1, -1):
                act4_output_split_dict[(core_x, core_y)] = ((0, act4_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act4_output_raw.shape[2]))
                core_idx = core_idx + 1

    g_data['activation4']['output'] = ModelDataHandler.tensor_split(
        raw_data=act4_output_raw,
        split_dict=act4_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # conv5 weight
    conv5_weight_raw = handler.parameters['conv5']['weight'].astype(np.int8)
    conv5_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv5_weight_split_dict[(core_x, core_y)] = (
                (0, conv5_weight_raw.shape[0]), (0, conv5_weight_raw.shape[1]), (0, conv5_weight_raw.shape[2]))

    g_data['conv5']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv5_weight_raw,
        split_dict=conv5_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )

    # conv5 bias
    conv5_bias_raw = handler.parameters['conv5']['bias'].astype(np.int32)
    conv5_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv5_bias_split_dict[(core_x, core_y)] = ((0, conv5_bias_raw.shape[0]),)

    g_data['conv5']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv5_bias_raw,
        split_dict=conv5_bias_split_dict,
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )

    # activation5
    act5_output_raw = handler.parameters['activation5']['output'].astype(np.int8)
    act5_output_split_dict = {}
    line_split_step = int(act5_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                act5_output_split_dict[(core_x, core_y)] = ((0, act5_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act5_output_raw.shape[2]))
                core_idx = core_idx + 1
        else:
            for core_x in range(size_x - 1, -1, -1):
                act5_output_split_dict[(core_x, core_y)] = ((0, act5_output_raw.shape[0]),
                                                            (core_idx * line_split_step,
                                                             core_idx * line_split_step + line_split_step),
                                                            (0, act5_output_raw.shape[2]))
                core_idx = core_idx + 1

    g_data['activation5']['output'] = ModelDataHandler.tensor_split(
        raw_data=act5_output_raw,
        split_dict=act5_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # cut_2 output
    cut_2_output_raw = handler.parameters['cut_2']['output'].astype(np.int8)
    cut_2_output_split_dict = {}
    line_split_step = int(cut_2_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut_2_output_split_dict[(core_x, core_y)] = ((0, cut_2_output_raw.shape[0]),
                                                             (core_idx * line_split_step,
                                                              core_idx * line_split_step + line_split_step),
                                                             (0, cut_2_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut_2_output_split_dict[(core_x, core_y)] = ((0, cut_2_output_raw.shape[0]),
                                                             (core_idx * line_split_step,
                                                              core_idx * line_split_step + line_split_step),
                                                             (0, cut_2_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut_2']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut_2_output_raw,
        split_dict=cut_2_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    ###############################################################################################################
    # conv_out weight
    conv_out_weight_raw = handler.parameters['conv_out']['weight'].astype(np.int8)
    conv_out_weight_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv_out_weight_split_dict[(core_x, core_y)] = (
                (0, conv_out_weight_raw.shape[0]), (0, conv_out_weight_raw.shape[1]), (0, conv_out_weight_raw.shape[2]))

    g_data['conv_out']['weight'] = ModelDataHandler.tensor_split(
        raw_data=conv_out_weight_raw,
        split_dict=conv_out_weight_split_dict,
        data_type=1,
        alignment=(32, None, None, None),
        dims=(0, 1, 3, 2),
        is_weight=True
    )

    # conv_out bias
    conv_out_bias_raw = handler.parameters['conv_out']['bias'].astype(np.int32)
    conv_out_bias_split_dict = {}
    core_idx = 0
    for core_y in range(size_y):
        for core_x in range(size_x):
            conv_out_bias_split_dict[(core_x, core_y)] = ((0, conv_out_bias_raw.shape[0]),)

    g_data['conv_out']['bias'] = ModelDataHandler.tensor_split(
        raw_data=conv_out_bias_raw,
        split_dict=conv_out_bias_split_dict,
        data_type=0,
        alignment=(32,),
        dims=(0,)
    )

    # cut_out output
    cut_out_output_raw = handler.parameters['cut_out']['output'].astype(np.int8)
    cut_out_output_split_dict = {}
    line_split_step = int(cut_out_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                cut_out_output_split_dict[(core_x, core_y)] = ((0, cut_out_output_raw.shape[0]),
                                                               (core_idx * line_split_step,
                                                                core_idx * line_split_step + line_split_step),
                                                               (0, cut_out_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)
        else:
            for core_x in range(size_x - 1, -1, -1):
                cut_out_output_split_dict[(core_x, core_y)] = ((0, cut_out_output_raw.shape[0]),
                                                               (core_idx * line_split_step,
                                                                core_idx * line_split_step + line_split_step),
                                                               (0, cut_out_output_raw.shape[2]))
                core_idx = core_idx + 1
                # print(core_idx, core_x, core_y)

    g_data['cut_out']['output'] = ModelDataHandler.tensor_split(
        raw_data=cut_out_output_raw,
        split_dict=cut_out_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    g_data['activation_out']['lut'] = ModelDataHandler.lut_gen(
        func="tanh",
        max_abs_float_input=4,
        input_width=8,
        input_q_factor=128,
        output_width=8,
        output_q_factor=128
    )

    # activation_out
    act_out_output_raw = handler.parameters['output'].astype(np.int8)
    act_out_output_split_dict = {}
    line_split_step = int(act_out_output_raw.shape[1] / Si)
    core_idx = 0
    for core_y in range(size_y):
        if (core_y % 2) == 0:
            for core_x in range(size_x):
                act_out_output_split_dict[(core_x, core_y)] = ((0, act_out_output_raw.shape[0]),
                                                               (core_idx * line_split_step,
                                                                core_idx * line_split_step + line_split_step),
                                                               (0, act_out_output_raw.shape[2]))
                core_idx = core_idx + 1
        else:
            for core_x in range(size_x - 1, -1, -1):
                act_out_output_split_dict[(core_x, core_y)] = ((0, act_out_output_raw.shape[0]),
                                                               (core_idx * line_split_step,
                                                                core_idx * line_split_step + line_split_step),
                                                               (0, act_out_output_raw.shape[2]))
                core_idx = core_idx + 1

    g_data['activation_out']['output'] = ModelDataHandler.tensor_split(
        raw_data=act_out_output_raw,
        split_dict=act_out_output_split_dict,
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )



    return g_data


if __name__ == '__main__':
    handler = ModelDataHandler(model='real')
    data = generate_data(handler, 8, 5)
    # print(data['conv_in'].keys())
    print(len(data['conv_in']['input'][(0, 0)]))
    # print(len(data['conv_in']['weight'][(0, 0)]))
    # print(len(data['conv_in']['bias'][(0, 0)]))
    print(len(data['activation_out']['lut']))
    # exit()
    example_x = 5
    example_y = 3
    layer_name = 'activation_out'
    param_name = 'output'
    with open("generator/ResidualBlocks/data/cmp.txt", "w") as f:
        for i in range(len(data[layer_name][param_name][(example_x, example_y)])):
            # print(data['activation0']['output'][(example_x, example_y)][i])
            str_input = ""
            for j in range(3, -1, -1):
                if (data[layer_name][param_name][(example_x, example_y)][i][j]) > 0x7f:
                    a = 0x7f
                elif (data[layer_name][param_name][(example_x, example_y)][i][j]) < -0x80:
                    a = -0x80
                else:
                    a = (data[layer_name][param_name][(example_x, example_y)][i][j])
                if a < 0:
                    a = int(a) & 0xff
                str_input += "{:0>2x}".format(int(a))
            f.write(str_input+"\n")

    startline = 3 # 5765
    length = 3360  # 6720
    for xx in range(8):
        for yy in range(5):
            alg_out = ""
            count = 0
            data[layer_name][param_name][(xx, yy)] = data[layer_name][param_name][(xx, yy)][:]
            for i in range(len(data[layer_name][param_name][(xx, yy)])):
                str_input = ""
                for j in range(3, -1, -1):
                    if (data[layer_name][param_name][(xx, yy)][i][j]) > 0x7f:
                        a = 0x7f
                    elif (data[layer_name][param_name][(xx, yy)][i][j]) < -0x80:
                        a = -0x80
                    else:
                        a = (data[layer_name][param_name][(xx, yy)][i][j])
                    if a < 0:
                        a = int(a) & 0xff
                    str_input += "{:0>2x}".format(int(a))
                alg_out += (str_input + "\n")
                count += 1
                if count >= length:
                    break

            simulator_out = ""
            with open("simulator/Out_files/residual_blocks/cmp_out/cmp_out_0_0_0_"+str(xx+8)+"_"+str(yy+0)+"@4_9.txt", "r") as f:
                all_lines = f.readlines()
                for ii in range(length):
                    simulator_out += all_lines[startline + ii - 1]

            # if xx == 5 and yy == 3:
            #     print("alg", alg_out)
            #     print("sim", simulator_out)
            if alg_out == simulator_out:
                print("The output of core:", xx, yy, "is right")
            else:
                print("The output of core:", xx, yy, "is wrong")

