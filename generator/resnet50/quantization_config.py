import warnings
import math


class QuantizationConfig:
    def __init__(self, name='in_cut_start'):
        self.__name = name
        self.__resnet50_cut_parameters = {
            'conv1_cut': 4,

            'layer1.0.cut1': 4,
            'layer1.0.cut2': 4,
            'layer1.0.cut3': 4,
            'layer1.0.cut4': 4,
            'layer1.0.cut5': 1,

            'layer1.1.cut1': 4,
            'layer1.1.cut2': 4,
            'layer1.1.cut3': 4,
            'layer1.1.cut5': 1,

            'layer1.2.cut1': 4,
            'layer1.2.cut2': 4,
            'layer1.2.cut3': 4,
            'layer1.2.cut5': 1,

            'layer2.0.cut1': 4,
            'layer2.0.cut2': 4,
            'layer2.0.cut3': 4,
            'layer2.0.cut4': 4,
            'layer2.0.cut5': 1,

            'layer2.1.cut1': 4,
            'layer2.1.cut2': 4,
            'layer2.1.cut3': 4,
            'layer2.1.cut5': 1,

            'layer2.2.cut1': 4,
            'layer2.2.cut2': 4,
            'layer2.2.cut3': 4,
            'layer2.2.cut5': 1,

            'layer2.3.cut1': 4,
            'layer2.3.cut2': 4,
            'layer2.3.cut3': 4,
            'layer2.3.cut5': 1,

            'layer3.0.cut1': 4,
            'layer3.0.cut2': 4,
            'layer3.0.cut3': 4,
            'layer3.0.cut4': 4,
            'layer3.0.cut5': 1,

            'layer3.1.cut1': 4,
            'layer3.1.cut2': 4,
            'layer3.1.cut3': 4,
            'layer3.1.cut5': 1,

            'layer3.2.cut1': 4,
            'layer3.2.cut2': 4,
            'layer3.2.cut3': 4,
            'layer3.2.cut5': 1,

            'layer3.3.cut1': 4,
            'layer3.3.cut2': 4,
            'layer3.3.cut3': 4,
            'layer3.3.cut5': 1,

            'layer3.4.cut1': 4,
            'layer3.4.cut2': 4,
            'layer3.4.cut3': 4,
            'layer3.4.cut5': 1,

            'layer3.5.cut1': 4,
            'layer3.5.cut2': 4,
            'layer3.5.cut3': 4,
            'layer3.5.cut5': 1,

            'layer4.0.cut1': 4,
            'layer4.0.cut2': 4,
            'layer4.0.cut3': 4,
            'layer4.0.cut4': 4,
            'layer4.0.cut5': 1,

            'layer4.1.cut1': 4,
            'layer4.1.cut2': 4,
            'layer4.1.cut3': 4,
            'layer4.1.cut5': 1,

            'layer4.2.cut1': 4,
            'layer4.2.cut2': 4,
            'layer4.2.cut3': 4,
            'layer4.2.cut5': 1,

            'avgpool_cut': 3,
            'fc_cut': 4
        }

        self.__resnet50_weight_parameters = {
            'conv1': 1024,

            'layer1.0.conv1': 64,
            'layer1.0.conv2': 1024,
            'layer1.0.conv3': 64,
            'layer1.0.downsample.0': 64,

            'layer1.1.conv1': 1024,
            'layer1.1.conv2': 1024,
            'layer1.1.conv3': 256,

            'layer1.2.conv1': 1024,
            'layer1.2.conv2': 256,
            'layer1.2.conv3': 256,

            'layer2.0.conv1': 256,
            'layer2.0.conv2': 1024,
            'layer2.0.conv3': 64,
            'layer2.0.downsample.0': 1024,

            'layer2.1.conv1': 256,
            'layer2.1.conv2': 1024,
            'layer2.1.conv3': 256,

            'layer2.2.conv1': 1024,
            'layer2.2.conv2': 1024,
            'layer2.2.conv3': 256,

            'layer2.3.conv1': 1024,
            'layer2.3.conv2': 1024,
            'layer2.3.conv3': 256,

            'layer3.0.conv1': 1024,
            'layer3.0.conv2': 1024,
            'layer3.0.conv3': 256,
            'layer3.0.downsample.0': 1024,

            'layer3.1.conv1': 1024,
            'layer3.1.conv2': 1024,
            'layer3.1.conv3': 256,

            'layer3.2.conv1': 1024,
            'layer3.2.conv2': 1024,
            'layer3.2.conv3': 256,

            'layer3.3.conv1': 1024,
            'layer3.3.conv2': 1024,
            'layer3.3.conv3': 256,

            'layer3.4.conv1': 1024,
            'layer3.4.conv2': 1024,
            'layer3.4.conv3': 256,

            'layer3.5.conv1': 1024,
            'layer3.5.conv2': 1024,
            'layer3.5.conv3': 256,

            'layer4.0.conv1': 1024,
            'layer4.0.conv2': 1024,
            'layer4.0.conv3': 256,
            'layer4.0.downsample.0': 256,

            'layer4.1.conv1': 1024,
            'layer4.1.conv2': 1024,
            'layer4.1.conv3': 256,

            'layer4.2.conv1': 256,
            'layer4.2.conv2': 1024,
            'layer4.2.conv3': 64,

            'fc': 1024
        }
        
        self.__resnet50_cut_parameters = {
            'conv1_cut': int(math.log(self.__resnet50_weight_parameters['conv1'], 4)),

            'layer1.0.cut1': int(math.log(self.__resnet50_weight_parameters['layer1.0.conv1'], 4)),
            'layer1.0.cut2': int(math.log(self.__resnet50_weight_parameters['layer1.0.conv2'], 4)),
            'layer1.0.cut3': int(math.log(self.__resnet50_weight_parameters['layer1.0.conv3'], 4)),
            'layer1.0.cut4': int(math.log(self.__resnet50_weight_parameters['layer1.0.downsample.0'], 4)),
            'layer1.0.cut5': 0,

            'layer1.1.cut1': int(math.log(self.__resnet50_weight_parameters['layer1.1.conv1'], 4)),
            'layer1.1.cut2': int(math.log(self.__resnet50_weight_parameters['layer1.1.conv2'], 4)),
            'layer1.1.cut3': int(math.log(self.__resnet50_weight_parameters['layer1.1.conv3'], 4)),
            'layer1.1.cut5': 0,

            'layer1.2.cut1': int(math.log(self.__resnet50_weight_parameters['layer1.2.conv1'], 4)),
            'layer1.2.cut2': int(math.log(self.__resnet50_weight_parameters['layer1.2.conv2'], 4)),
            'layer1.2.cut3': int(math.log(self.__resnet50_weight_parameters['layer1.2.conv3'], 4)),
            'layer1.2.cut5': 0,

            'layer2.0.cut1': int(math.log(self.__resnet50_weight_parameters['layer2.0.conv1'], 4)),
            'layer2.0.cut2': int(math.log(self.__resnet50_weight_parameters['layer2.0.conv2'], 4)),
            'layer2.0.cut3': int(math.log(self.__resnet50_weight_parameters['layer2.0.conv3'], 4)),
            'layer2.0.cut4': int(math.log(self.__resnet50_weight_parameters['layer2.0.downsample.0'], 4)),
            'layer2.0.cut5': 0,

            'layer2.1.cut1': int(math.log(self.__resnet50_weight_parameters['layer2.1.conv1'], 4)),
            'layer2.1.cut2': int(math.log(self.__resnet50_weight_parameters['layer2.1.conv2'], 4)),
            'layer2.1.cut3': int(math.log(self.__resnet50_weight_parameters['layer2.1.conv3'], 4)),
            'layer2.1.cut5': 0,

            'layer2.2.cut1': int(math.log(self.__resnet50_weight_parameters['layer2.2.conv1'], 4)),
            'layer2.2.cut2': int(math.log(self.__resnet50_weight_parameters['layer2.2.conv2'], 4)),
            'layer2.2.cut3': int(math.log(self.__resnet50_weight_parameters['layer2.2.conv3'], 4)),
            'layer2.2.cut5': 0,

            'layer2.3.cut1': int(math.log(self.__resnet50_weight_parameters['layer2.3.conv1'], 4)),
            'layer2.3.cut2': int(math.log(self.__resnet50_weight_parameters['layer2.3.conv2'], 4)),
            'layer2.3.cut3': int(math.log(self.__resnet50_weight_parameters['layer2.3.conv3'], 4)),
            'layer2.3.cut5': 0,

            'layer3.0.cut1': int(math.log(self.__resnet50_weight_parameters['layer3.0.conv1'], 4)),
            'layer3.0.cut2': int(math.log(self.__resnet50_weight_parameters['layer3.0.conv2'], 4)),
            'layer3.0.cut3': int(math.log(self.__resnet50_weight_parameters['layer3.0.conv3'], 4)),
            'layer3.0.cut4': int(math.log(self.__resnet50_weight_parameters['layer3.0.downsample.0'], 4)),
            'layer3.0.cut5': 0,

            'layer3.1.cut1': int(math.log(self.__resnet50_weight_parameters['layer3.1.conv1'], 4)),
            'layer3.1.cut2': int(math.log(self.__resnet50_weight_parameters['layer3.1.conv2'], 4)),
            'layer3.1.cut3': int(math.log(self.__resnet50_weight_parameters['layer3.1.conv3'], 4)),
            'layer3.1.cut5': 0,

            'layer3.2.cut1': int(math.log(self.__resnet50_weight_parameters['layer3.2.conv1'], 4)),
            'layer3.2.cut2': int(math.log(self.__resnet50_weight_parameters['layer3.2.conv2'], 4)),
            'layer3.2.cut3': int(math.log(self.__resnet50_weight_parameters['layer3.2.conv3'], 4)),
            'layer3.2.cut5': 0,

            'layer3.3.cut1': int(math.log(self.__resnet50_weight_parameters['layer3.3.conv1'], 4)),
            'layer3.3.cut2': int(math.log(self.__resnet50_weight_parameters['layer3.3.conv2'], 4)),
            'layer3.3.cut3': int(math.log(self.__resnet50_weight_parameters['layer3.3.conv3'], 4)),
            'layer3.3.cut5': 0,

            'layer3.4.cut1': int(math.log(self.__resnet50_weight_parameters['layer3.4.conv1'], 4)),
            'layer3.4.cut2': int(math.log(self.__resnet50_weight_parameters['layer3.4.conv2'], 4)),
            'layer3.4.cut3': int(math.log(self.__resnet50_weight_parameters['layer3.4.conv3'], 4)),
            'layer3.4.cut5': 0,

            'layer3.5.cut1': int(math.log(self.__resnet50_weight_parameters['layer3.5.conv1'], 4)),
            'layer3.5.cut2': int(math.log(self.__resnet50_weight_parameters['layer3.5.conv2'], 4)),
            'layer3.5.cut3': int(math.log(self.__resnet50_weight_parameters['layer3.5.conv3'], 4)),
            'layer3.5.cut5': 0,

            'layer4.0.cut1': int(math.log(self.__resnet50_weight_parameters['layer4.0.conv1'], 4)),
            'layer4.0.cut2': int(math.log(self.__resnet50_weight_parameters['layer4.0.conv2'], 4)),
            'layer4.0.cut3': int(math.log(self.__resnet50_weight_parameters['layer4.0.conv3'], 4)),
            'layer4.0.cut4': int(math.log(self.__resnet50_weight_parameters['layer4.0.downsample.0'], 4)),
            'layer4.0.cut5': 0,

            'layer4.1.cut1': int(math.log(self.__resnet50_weight_parameters['layer4.1.conv1'], 4)),
            'layer4.1.cut2': int(math.log(self.__resnet50_weight_parameters['layer4.1.conv2'], 4)),
            'layer4.1.cut3': int(math.log(self.__resnet50_weight_parameters['layer4.1.conv3'], 4)),
            'layer4.1.cut5': 0,

            'layer4.2.cut1': int(math.log(self.__resnet50_weight_parameters['layer4.2.conv1'], 4)),
            'layer4.2.cut2': int(math.log(self.__resnet50_weight_parameters['layer4.2.conv2'], 4)),
            'layer4.2.cut3': int(math.log(self.__resnet50_weight_parameters['layer4.2.conv3'], 4)),
            'layer4.2.cut5': 0,

            # 'avgpool_cut': int(math.log(self.__resnet50_weight_parameters['layer4.2.conv3'] / 64, 4)),
            'avgpool_cut': int(math.log(64, 4)),
            'fc_cut': int(math.log(self.__resnet50_weight_parameters['fc'] * 64, 4))
        }
        

            
    def __getitem__(self, item):
        if self.__name == 'in_cut_start':
            result = self.__resnet50_cut_parameters.get(item)
        else:
            assert self.__name == 'weight'
            result = self.__resnet50_weight_parameters.get(item)
        if result is None:
            warnings.warn('Module \'{}\' do not have in_cut_start parameter!'.format(item))
        return result


if __name__ == '__main__':
    q_config = QuantizationConfig()