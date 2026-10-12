import warnings
import math


class QuantizationConfig:
    def __init__(self, name='in_cut_start'):
        self.__name = name

        self.__weight_parameters = {
            'conv1': 256,
            'conv2': 256,
            'fc1': 1024,
            'fc2': 256,
            'fc3': 256,
        }

        self.__cut_parameters = {
            'cut1': int(math.log(self.__weight_parameters['conv1'], 4)),
            'cut2': int(math.log(self.__weight_parameters['conv2'], 4)),
            'fc_cut1': int(math.log(self.__weight_parameters['fc1'], 4)),
            'fc_cut2': int(math.log(self.__weight_parameters['fc2'], 4)),
            'fc_cut3': int(math.log(self.__weight_parameters['fc3'] * 4, 4))
        }

    def __getitem__(self, item):
        if self.__name == 'in_cut_start':
            result = self.__cut_parameters.get(item)
        else:
            assert self.__name == 'weight'
            result = self.__weight_parameters.get(item)
        if result is None:
            warnings.warn('Module \'{}\' do not have in_cut_start parameter!'.format(item))
        return result


if __name__ == '__main__':
    q_config = QuantizationConfig()