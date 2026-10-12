import warnings
import math


class QuantizationConfig:
    def __init__(self, name='in_cut_start', ckpt=None):
        self.__name = name
        self.ckpt = ckpt
        assert ckpt is not None
        # self.__resnet50_0_weight_parameters = {
        #     'conv1': 256,
        #
        #     'layer1.0.conv1': 64,
        #     'layer1.0.conv2': 256,
        #     'layer1.0.conv3': 64,
        #     'layer1.0.downsample.0': 64,
        #
        #     'layer1.1.conv1': 256,
        #     'layer1.1.conv2': 256,
        #     'layer1.1.conv3': 64,
        #
        #     'layer1.2.conv1': 256,
        #     'layer1.2.conv2': 256,
        #     'layer1.2.conv3': 64,
        #
        #     'fc_0': 1024,
        #     'fc_1': 1024,
        #     'fc': 256
        # }
        self.__resnet50_0_weight_parameters = {
            'conv1': 1024,

            'layer1.0.conv1': 256,
            'layer1.0.conv2': 1024,
            'layer1.0.conv3': 256,
            'layer1.0.downsample.0': 256,

            'layer1.1.conv1': 256,
            'layer1.1.conv2': 1024,
            'layer1.1.conv3': 256,

            'layer1.2.conv1': 256,
            'layer1.2.conv2': 1024,
            'layer1.2.conv3': 256,

            'fc_0': 256,
            'fc_1': 1024,
            'fc': 256
        }
        self.__resnet50_1_weight_parameters = {
            'conv1': 1024,

            'layer1.0.conv1': 256,
            'layer1.0.conv2': 1024,
            'layer1.0.conv3': 256,
            'layer1.0.downsample.0': 256,

            'layer1.1.conv1': 256,
            'layer1.1.conv2': 1024,
            'layer1.1.conv3': 256,

            'layer1.2.conv1': 256,
            'layer1.2.conv2': 1024,
            'layer1.2.conv3': 256,

            'fc_0': 256,
            'fc_1': 1024,
            'fc': 256
        }
        self.__resnet50_2_weight_parameters = {
            'conv1': 1024,

            'layer1.0.conv1': 256,
            'layer1.0.conv2': 1024,
            'layer1.0.conv3': 256,
            'layer1.0.downsample.0': 256,

            'layer1.1.conv1': 256,
            'layer1.1.conv2': 1024,
            'layer1.1.conv3': 256,

            'layer1.2.conv1': 256,
            'layer1.2.conv2': 1024,
            'layer1.2.conv3': 256,

            'fc_0': 256,
            'fc_1': 1024,
            'fc': 256
        }
        self.__resnet50_3_weight_parameters = {
            'conv1': 1024,

            'layer1.0.conv1': 256,
            'layer1.0.conv2': 1024,
            'layer1.0.conv3': 256,
            'layer1.0.downsample.0': 256,

            'layer1.1.conv1': 256,
            'layer1.1.conv2': 1024,
            'layer1.1.conv3': 256,

            'layer1.2.conv1': 256,
            'layer1.2.conv2': 1024,
            'layer1.2.conv3': 256,

            'fc_0': 256,
            'fc_1': 1024,
            'fc': 256
        }

        self.__resnet50_0_cut_parameters = {
            'conv1_cut': int(math.log(self.__resnet50_0_weight_parameters['conv1'], 4)),

            'layer1.0.cut1': int(math.log(self.__resnet50_0_weight_parameters['layer1.0.conv1'], 4)),
            'layer1.0.cut2': int(math.log(self.__resnet50_0_weight_parameters['layer1.0.conv2'], 4)),
            'layer1.0.cut3': int(math.log(self.__resnet50_0_weight_parameters['layer1.0.conv3'], 4)),
            'layer1.0.cut4': int(math.log(self.__resnet50_0_weight_parameters['layer1.0.downsample.0'], 4)),
            'layer1.0.cut5': 0,

            'layer1.1.cut1': int(math.log(self.__resnet50_0_weight_parameters['layer1.1.conv1'], 4)),
            'layer1.1.cut2': int(math.log(self.__resnet50_0_weight_parameters['layer1.1.conv2'], 4)),
            'layer1.1.cut3': int(math.log(self.__resnet50_0_weight_parameters['layer1.1.conv3'], 4)),
            'layer1.1.cut5': 0,

            'layer1.2.cut1': int(math.log(self.__resnet50_0_weight_parameters['layer1.2.conv1'], 4)),
            'layer1.2.cut2': int(math.log(self.__resnet50_0_weight_parameters['layer1.2.conv2'], 4)),
            'layer1.2.cut3': int(math.log(self.__resnet50_0_weight_parameters['layer1.2.conv3'], 4)),
            'layer1.2.cut5': 0,

            'avgpool_cut': int(math.log(64, 4)),
            'fc_0_cut': int(math.log(self.__resnet50_0_weight_parameters['fc_0'], 4)),
            'fc_1_cut': int(math.log(self.__resnet50_0_weight_parameters['fc_1'], 4)),
            'fc_cut': int(math.log(self.__resnet50_0_weight_parameters['fc'] * 16, 4))
        }
        self.__resnet50_1_cut_parameters = {
            'conv1_cut': int(math.log(self.__resnet50_1_weight_parameters['conv1'], 4)),

            'layer1.0.cut1': int(math.log(self.__resnet50_1_weight_parameters['layer1.0.conv1'], 4)),
            'layer1.0.cut2': int(math.log(self.__resnet50_1_weight_parameters['layer1.0.conv2'], 4)),
            'layer1.0.cut3': int(math.log(self.__resnet50_1_weight_parameters['layer1.0.conv3'], 4)),
            'layer1.0.cut4': int(math.log(self.__resnet50_1_weight_parameters['layer1.0.downsample.0'], 4)),
            'layer1.0.cut5': 0,

            'layer1.1.cut1': int(math.log(self.__resnet50_1_weight_parameters['layer1.1.conv1'], 4)),
            'layer1.1.cut2': int(math.log(self.__resnet50_1_weight_parameters['layer1.1.conv2'], 4)),
            'layer1.1.cut3': int(math.log(self.__resnet50_1_weight_parameters['layer1.1.conv3'], 4)),
            'layer1.1.cut5': 0,

            'layer1.2.cut1': int(math.log(self.__resnet50_1_weight_parameters['layer1.2.conv1'], 4)),
            'layer1.2.cut2': int(math.log(self.__resnet50_1_weight_parameters['layer1.2.conv2'], 4)),
            'layer1.2.cut3': int(math.log(self.__resnet50_1_weight_parameters['layer1.2.conv3'], 4)),
            'layer1.2.cut5': 0,

            'avgpool_cut': int(math.log(64, 4)),
            'fc_0_cut': int(math.log(self.__resnet50_1_weight_parameters['fc_0'], 4)),
            'fc_1_cut': int(math.log(self.__resnet50_1_weight_parameters['fc_1'], 4)),
            'fc_cut': int(math.log(self.__resnet50_1_weight_parameters['fc'] * 16, 4))
        }
        self.__resnet50_2_cut_parameters = {
            'conv1_cut': int(math.log(self.__resnet50_2_weight_parameters['conv1'], 4)),

            'layer1.0.cut1': int(math.log(self.__resnet50_2_weight_parameters['layer1.0.conv1'], 4)),
            'layer1.0.cut2': int(math.log(self.__resnet50_2_weight_parameters['layer1.0.conv2'], 4)),
            'layer1.0.cut3': int(math.log(self.__resnet50_2_weight_parameters['layer1.0.conv3'], 4)),
            'layer1.0.cut4': int(math.log(self.__resnet50_2_weight_parameters['layer1.0.downsample.0'], 4)),
            'layer1.0.cut5': 0,

            'layer1.1.cut1': int(math.log(self.__resnet50_2_weight_parameters['layer1.1.conv1'], 4)),
            'layer1.1.cut2': int(math.log(self.__resnet50_2_weight_parameters['layer1.1.conv2'], 4)),
            'layer1.1.cut3': int(math.log(self.__resnet50_2_weight_parameters['layer1.1.conv3'], 4)),
            'layer1.1.cut5': 0,

            'layer1.2.cut1': int(math.log(self.__resnet50_2_weight_parameters['layer1.2.conv1'], 4)),
            'layer1.2.cut2': int(math.log(self.__resnet50_2_weight_parameters['layer1.2.conv2'], 4)),
            'layer1.2.cut3': int(math.log(self.__resnet50_2_weight_parameters['layer1.2.conv3'], 4)),
            'layer1.2.cut5': 0,

            'avgpool_cut': int(math.log(64, 4)),
            'fc_0_cut': int(math.log(self.__resnet50_2_weight_parameters['fc_0'], 4)),
            'fc_1_cut': int(math.log(self.__resnet50_2_weight_parameters['fc_1'], 4)),
            'fc_cut': int(math.log(self.__resnet50_2_weight_parameters['fc'] * 16, 4))
        }
        self.__resnet50_3_cut_parameters = {
            'conv1_cut': int(math.log(self.__resnet50_3_weight_parameters['conv1'], 4)),

            'layer1.0.cut1': int(math.log(self.__resnet50_3_weight_parameters['layer1.0.conv1'], 4)),
            'layer1.0.cut2': int(math.log(self.__resnet50_3_weight_parameters['layer1.0.conv2'], 4)),
            'layer1.0.cut3': int(math.log(self.__resnet50_3_weight_parameters['layer1.0.conv3'], 4)),
            'layer1.0.cut4': int(math.log(self.__resnet50_3_weight_parameters['layer1.0.downsample.0'], 4)),
            'layer1.0.cut5': 0,

            'layer1.1.cut1': int(math.log(self.__resnet50_3_weight_parameters['layer1.1.conv1'], 4)),
            'layer1.1.cut2': int(math.log(self.__resnet50_3_weight_parameters['layer1.1.conv2'], 4)),
            'layer1.1.cut3': int(math.log(self.__resnet50_3_weight_parameters['layer1.1.conv3'], 4)),
            'layer1.1.cut5': 0,

            'layer1.2.cut1': int(math.log(self.__resnet50_3_weight_parameters['layer1.2.conv1'], 4)),
            'layer1.2.cut2': int(math.log(self.__resnet50_3_weight_parameters['layer1.2.conv2'], 4)),
            'layer1.2.cut3': int(math.log(self.__resnet50_3_weight_parameters['layer1.2.conv3'], 4)),
            'layer1.2.cut5': 0,

            'avgpool_cut': int(math.log(64, 4)),
            'fc_0_cut': int(math.log(self.__resnet50_3_weight_parameters['fc_0'], 4)),
            'fc_1_cut': int(math.log(self.__resnet50_3_weight_parameters['fc_1'], 4)),
            'fc_cut': int(math.log(self.__resnet50_3_weight_parameters['fc'] * 16, 4))
        }

    def __getitem__(self, item):
        if self.ckpt == 0:
            if self.__name == 'in_cut_start':
                result = self.__resnet50_0_cut_parameters.get(item)
            else:
                assert self.__name == 'weight'
                result = self.__resnet50_0_weight_parameters.get(item)
        elif self.ckpt == 1:
            if self.__name == 'in_cut_start':
                result = self.__resnet50_1_cut_parameters.get(item)
            else:
                assert self.__name == 'weight'
                result = self.__resnet50_1_weight_parameters.get(item)
        elif self.ckpt == 2:
            if self.__name == 'in_cut_start':
                result = self.__resnet50_2_cut_parameters.get(item)
            else:
                assert self.__name == 'weight'
                result = self.__resnet50_2_weight_parameters.get(item)
        elif self.ckpt == 3:
            if self.__name == 'in_cut_start':
                result = self.__resnet50_3_cut_parameters.get(item)
            else:
                assert self.__name == 'weight'
                result = self.__resnet50_3_weight_parameters.get(item)
        else:
            raise ValueError
        if result is None:
            warnings.warn('Module \'{}\' do not have in_cut_start parameter!'.format(item))
        return result


if __name__ == '__main__':
    q_config = QuantizationConfig()