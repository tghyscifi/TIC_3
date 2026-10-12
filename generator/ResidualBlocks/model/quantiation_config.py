import warnings
from math import log

BITS = 8
INT_MAX = 2 ** (BITS - 1) - 1
INT_MIN = - 2 ** (BITS - 1)
TANH_LIMIT = 4

# 根据权重分布得出量化参数
class QuantizationConfig:
    def __init__(self, weights=[]):
        if len(weights) == 0:
        # Original
            self.__quantization_config = {
                'weight':
                {
                    'conv_in': 256,
                    'res0': {
                        'conv1': 64,
                        'conv2': 64
                    },
                    'res1': {
                        'conv1': 256,
                        'conv2': 256
                    },
                    'res2': {
                        'conv1': 256,
                        'conv2': 64
                    },
                    'conv_out': 256               
                }
            }
            # 16 bits
            # self.__quantization_config = {
            #     'weight':
            #     {
            #         'conv_in': 16384,
            #         'res0': {
            #             'conv1': 4096,
            #             'conv2': 16384
            #         },
            #         'res1': {
            #             'conv1': 16384,
            #             'conv2': 16384
            #         },
            #         'res2': {
            #             'conv1': 16384,
            #             'conv2': 4096
            #         },
            #         'conv_out': 16384               
            #     }
            # }
        else:
            # Test
            self.__quantization_config = {
                'weight':
                {
                    'conv_in': weights[0],
                    'res0': {
                        'conv1': weights[1],
                        'conv2': weights[2]
                    },
                    'res1': {
                        'conv1': weights[3],
                        'conv2': weights[4]
                    },
                    'res2': {
                        'conv1': weights[5],
                        'conv2': weights[6]
                    },
                    'conv_out': weights[7]               
                }
            }


        self.__quantization_config['in_cut_start'] = {
            'conv_in': int(log(self.__quantization_config['weight']['conv_in'], 4)),
            'res0': {
                'conv1': int(log(self.__quantization_config['weight']['res0']['conv1'], 4)),
                'conv2': int(log(self.__quantization_config['weight']['res0']['conv2'], 4)),
                'add': 0
            },
            'res1': {
                'conv1': int(log(self.__quantization_config['weight']['res1']['conv1'], 4)),
                'conv2': int(log(self.__quantization_config['weight']['res1']['conv2'], 4)),
                'add': 0
            },
            'res2': {
                'conv1': int(log(self.__quantization_config['weight']['res2']['conv1'], 4)),
                'conv2': int(log(self.__quantization_config['weight']['res2']['conv2'], 4)),
                'add': 0
            },
            'conv_out': int(log(self.__quantization_config['weight']['conv_out'] * TANH_LIMIT, 4))                    
        }

    def __getitem__(self, item):
        result = self.__quantization_config.get(item)
        if result is None:
            warnings.warn('\'{}\' is not a quantization parameter!'.format(item))
        return result


if __name__ == '__main__': 
    qconfig = QuantizationConfig()
    print(qconfig['weight'])
    print(qconfig['in_cut_start'])