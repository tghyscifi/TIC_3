from generator.resnet50.resnet50_2chip.resnet50_2chip_data_handler import ResNetDataHandler
from generator.resnet50.resnet50_2chip.G1_data import generate_g1_data
from generator.resnet50.resnet50_2chip.G2_data import generate_g2_data
from generator.resnet50.resnet50_2chip.G3_data import generate_g3_data
from generator.resnet50.resnet50_2chip.G4_data import generate_g4_data


def merge_dict(a: dict, b: dict):
    """
    merge B to A
    """
    for key, value in b.items():
        if a.get(key) is not None:
            raise KeyError('key " {} " is already in dictionary!'.format(key))
        a[key] = value


def resnet50_data(ckpt=None):
    handler = ResNetDataHandler(ckpt=ckpt)
    g1_data = generate_g1_data(handler, size_y=2, size_x=16)
    g2_data = generate_g2_data(handler, size_y=2, size_x=14)
    g3_data = generate_g3_data(handler, size_y=2, size_x=14)
    g4_data = generate_g4_data(handler, size_y=2, size_x=14)

    data_all = {}
    merge_dict(data_all, g1_data)
    merge_dict(data_all, g2_data)
    merge_dict(data_all, g3_data)
    merge_dict(data_all, g4_data)

    return data_all


if __name__ == '__main__':
    data = resnet50_data()
    xx = 1
