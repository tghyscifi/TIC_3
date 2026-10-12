from itertools import product

import torch

from generator.ResidualBlocks.model.random_data_handler import ModelDataHandler
import numpy as np

Sm = 1
Sn = 1
Sj = 1
Si = 40


def generate_test_data():
    g_data = {
        'nonoverlapping': {},
        'overlapping': {},
        'test_out1': {},
        'test_out2': {},
        'activation_out': {}
    }

    # input -- [C, H, W]; weight -- [C_out, C_in, Ky, Kx]; bias -- [cout]; output -- [C, H, W]
    
    ###############################################################################################################
    # nonoverlapping
    test_tensor = torch.zeros((32, 15, 16)) + 1
    test_tensor[:, :, 0] = 2
    test_tensor[:, :, 15] = 2
    # print(test_tensor)
    g_data['nonoverlapping'] = ModelDataHandler.tensor_split(
        raw_data=test_tensor.numpy().astype(np.int8),
        split_dict={(0, 0): ((0, 32), (0, 15), (0, 16))},
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    # overlapping
    test_overlapping = np.pad(test_tensor.numpy(), ((0, 0), (0, 0), (1, 1)), 'constant', constant_values=(2, 2))
    # print(test_overlapping)
    g_data['overlapping'] = ModelDataHandler.tensor_split(
        raw_data=test_overlapping.astype(np.int8),
        split_dict={(0, 0): ((0, 32), (0, 15), (0, 18))},
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )

    test_out1 = torch.zeros((16, 15, 16)) + 1
    test_out2 = test_out1 + 1
    g_data['test_out1'] = ModelDataHandler.tensor_split(
        raw_data=test_out1.numpy().astype(np.int8),
        split_dict={(0, 0): ((0, 16), (0, 15), (0, 16))},
        data_type=1,
        alignment=(16, None, None),
        dims=(0, 2, 1)
    )
    g_data['test_out2'] = ModelDataHandler.tensor_split(
        raw_data=test_out2.numpy().astype(np.int8),
        split_dict={(0, 0): ((0, 16), (0, 15), (0, 16))},
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


    return g_data


if __name__ == '__main__':
    handler = ModelDataHandler()
    data = generate_test_data()
    print(data['nonoverlapping'])
    exit()


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

