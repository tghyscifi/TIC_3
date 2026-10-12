from generator.resnet50.data_handler import ResNetDataHandler
from generator.resnet50.resnet50_5chips.G1_data import generate_g1_data
from generator.resnet50.resnet50_5chips.G2_data import generate_g2_data
from generator.resnet50.resnet50_5chips.G3_data import generate_g3_data
from generator.resnet50.resnet50_5chips.G4_data import generate_g4_data
from generator.resnet50.resnet50_5chips.G5_data import generate_g5_data
from generator.resnet50.resnet50_5chips.G6_data import generate_g6_data
from generator.resnet50.resnet50_5chips.G7_data import generate_g7_data
from generator.resnet50.resnet50_5chips.G8_data import generate_g8_data
from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data
from generator.resnet50.resnet50_5chips.G10_data import generate_g10_data
from generator.resnet50.resnet50_5chips.G11_data import generate_g11_data
from generator.resnet50.resnet50_5chips.G12_data import generate_g12_data
from generator.resnet50.resnet50_5chips.G13_data import generate_g13_data
from generator.resnet50.resnet50_5chips.G14_data import generate_g14_data
from generator.resnet50.resnet50_5chips.G15_data import generate_g15_data
from generator.resnet50.resnet50_5chips.G16_data import generate_g16_data
from generator.resnet50.resnet50_5chips.G17_data import generate_g17_data


def merge_dict(a: dict, b: dict):
    """
    merge B to A
    """
    for key, value in b.items():
        if a.get(key) is not None:
            raise KeyError('key " {} " is already in dictionary!'.format(key))
        a[key] = value


def resnet50_data():
    handler = ResNetDataHandler()
    g1_data = generate_g1_data(handler, size_y=2, size_x=16)
    g2_data = generate_g2_data(handler, size_y=2, size_x=14)
    g3_data = generate_g3_data(handler, size_y=2, size_x=14)
    g4_data = generate_g4_data(handler, size_y=2, size_x=14)
    g5_data = generate_g5_data(handler, size_y=2, size_x=14)
    g6_data = generate_g6_data(handler, size_y=2, size_x=14)
    g7_data = generate_g7_data(handler, size_y=2, size_x=14)
    g8_data = generate_g8_data(handler, size_y=2, size_x=14)
    g9_data = generate_g9_data(handler, size_y=4, size_x=8)
    g10_data = generate_g10_data(handler, size_y=4, size_x=8)
    g11_data = generate_g11_data(handler, size_y=4, size_x=8)
    g12_data = generate_g12_data(handler, size_y=4, size_x=8)
    g13_data = generate_g13_data(handler, size_y=4, size_x=8)
    g14_data = generate_g14_data(handler, size_y=4, size_x=8)
    g15_data = generate_g15_data(handler, size_y=4, size_x=16)
    g16_data = generate_g16_data(handler, size_y=4, size_x=16)
    g17_data = generate_g17_data(handler, size_y=4, size_x=16)

    data_all = {}
    merge_dict(data_all, g1_data)
    merge_dict(data_all, g2_data)
    merge_dict(data_all, g3_data)
    merge_dict(data_all, g4_data)
    merge_dict(data_all, g5_data)
    merge_dict(data_all, g6_data)
    merge_dict(data_all, g7_data)
    merge_dict(data_all, g8_data)
    merge_dict(data_all, g9_data)
    merge_dict(data_all, g10_data)
    merge_dict(data_all, g11_data)
    merge_dict(data_all, g12_data)
    merge_dict(data_all, g13_data)
    merge_dict(data_all, g14_data)
    merge_dict(data_all, g15_data)
    merge_dict(data_all, g16_data)
    merge_dict(data_all, g17_data)

    return data_all


if __name__ == '__main__':
    data = resnet50_data()
    xx = 1
