import warnings


class SNNConfig:
    def __init__(self):
        self.__snn_config = {
            'fc1': {
                'thresh': 64,
                'decay': 128,
                'beta': 0
            },
            'fc2': {
                'thresh': 64,
                'decay': 128,
                'beta': 0
            },
            'fc3': {
                'thresh': 64,
                'decay': 128,
                'beta': 0
            },
            'fc4': {
                'thresh': 64,
                'decay': 128,
                'beta': 0
            }
        }

    def __getitem__(self, item):
        result = self.__snn_config.get(item)
        if result is None:
            warnings.warn('\'{}\' is not an SNN parameter!'.format(item))
        return result
