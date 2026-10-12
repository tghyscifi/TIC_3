from itertools import product
from generator.LeNet.lenet_model.lenet_data_handler import LeNetDataHandler
import numpy as np


def generate_data(handler):
    g_data = {
        'conv1': {},
        'cut1': {},
        'max_pool1': {},
        'conv2': {},
        'max_pool2': {},
        'fc1': {},
        'fc_relu1': {},
        'fc2': {},
        'fc_relu2': {},
        'fc3': {},
        'fc_cut3': {}
    }

    # input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    g_data['conv1']['input'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv1']['input'].astype(np.int8),
        split_dict={(0, 0): ((0, 1), (0, 28), (0, 28))},
        data_type=1, alignment=(None, None, 16),
        dims=(2, 1, 0))
    g_data['conv1']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv1']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 6), (0, 1), (0, 5), (0, 5))},
        data_type=1, alignment=(32, None, None, None),
        dims=(0, 3, 2, 1), is_weight=True)
    g_data['conv1']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv1']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 6), )},
        data_type=0, alignment=(32,), dims=(0,))
    g_data['cut1']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['cut1']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 6), (0, 24), (0, 24))},
        data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))
    g_data['max_pool1']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['max_pool1']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 6), (0, 12), (0, 12))},
        data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    g_data['conv2']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv2']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 16), (0, 6), (0, 5), (0, 5))},
        data_type=1, alignment=(32, None, None, None),
        dims=(0, 1, 3, 2), is_weight=True)
    g_data['conv2']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['conv2']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 16),)},
        data_type=0, alignment=(32,), dims=(0,))
    g_data['max_pool2']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['max_pool2']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 16), (0, 5), (0, 5))},
        data_type=1, alignment=(16, None, None),
        dims=(0, 2, 1))

    g_data['fc1']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc1']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 120), (0, 400))},
        data_type=1, alignment=(32, None),
        dims=(0, 1), is_weight=True)
    g_data['fc1']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc1']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 120),)},
        data_type=0, alignment=(32,), dims=(0,))

    g_data['fc2']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc2']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 84), (0, 120))},
        data_type=1, alignment=(32, None),
        dims=(0, 1), is_weight=True)
    g_data['fc2']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc2']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 84),)},
        data_type=0, alignment=(32,), dims=(0,))

    g_data['fc3']['weight'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc3']['weight'].astype(np.int8),
        split_dict={(0, 0): ((0, 10), (0, 84))},
        data_type=1, alignment=(32, None),
        dims=(0, 1), is_weight=True)
    g_data['fc3']['bias'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc3']['bias'].astype(np.int32),
        split_dict={(0, 0): ((0, 10),)},
        data_type=0, alignment=(32,), dims=(0,))

    g_data['fc_cut3']['output'] = LeNetDataHandler.tensor_split(
        raw_data=handler.parameters['fc_cut3']['output'].astype(np.int8),
        split_dict={(0, 0): ((0, 10), )},
        data_type=1, alignment=(16,),
        dims=(0,))

    return g_data


if __name__ == '__main__':
    handler = LeNetDataHandler()
    data = generate_data(handler)
    print(handler.names)
