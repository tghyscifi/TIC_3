import numpy as np
import math
from primitive.basic_operations import BasicOperations

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_81_Axon(Primitive):
    def __init__(self):
        super().__init__()

        self.InA_type = 1  # 可配置更改：[00]int32[01]int8[10]uint8 [11]Tenary
        self.InB_type = 1
        self.Load_Bias = 0  # 可配置更改：2,3为Bias 0,1为常数b
        self.pad_on = False
        self.Input_fm_Px = 0
        self.Input_fm_Py = 0
        self.conv_Kx = 0
        self.conv_Ky = 0
        self.conv_Sx = 1
        self.conv_Sy = 1
        self.conv_Ex = 1
        self.conv_Ey = 1
        self.pad_up = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.cin = 0
        self.cout = 0
        self.Bias_length = 0
        self.PIC = 0x81
        self.Reset_Addr_A = 1
        self.Reset_Addr_V = 1
        self.Addr_InA_base = 0x00
        self.Addr_InB_base = 0x00
        self.Addr_Bias_base = 0x00
        self.Addr_V_base = 0x00
        self.Addr_InA_end = 0x00
        self.Addr_V_end = 0x00
        self.A2S2_mode = 0

        self.total_core_num = 1
        self.areas_num_in_core = 1

    def __str__(self):
        return "81(CNN1)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)
        _retLsit = []
        self.cal_para()
        InputX_size = (self.Input_fm_Py, self.Input_fm_Px, self.cin)

        num_in_4B = 4
        if self.InA_type == 2:
            InputX_array = data_generator.random_array_u8(InputX_size).astype(np.int64)
        elif self.InA_type == 3:
            InputX_array = data_generator.random_array_2(InputX_size).astype(np.int64)
            num_in_4B = 16
        else:
            InputX_array = data_generator.random_array_8(InputX_size).astype(np.int64)

        if self.pad_on:
            InputX_fm_size_real = (self.Py_real, self.Px_real, self.cin)
            Input_fm_real = np.zeros(InputX_fm_size_real).astype(np.int64)

            for NOY_cnt in range((self.pad_up - 1), (self.pad_up + self.Input_fm_Py - 1)):
                for NOX_cnt in range((self.pad_left - 1), (self.pad_left + self.Input_fm_Px - 1)):
                    for CIN_cnt in range(self.cin):
                        Input_fm_real[NOY_cnt + 1][NOX_cnt + 1][CIN_cnt] = \
                            InputX_array[NOY_cnt + 1 - self.pad_up][NOX_cnt + 1 - self.pad_left][CIN_cnt]
                    pass
        else:
            Input_fm_real = InputX_array
        pass

        if self.InA_type == 3:  # int2 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 64)
            self.nix_wr_real = self.nix_grp_real * 64
        else:  # int8 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 16)
            self.nix_wr_real = self.nix_grp_real * 16

        InputX_size_equal = (self.Py_real, self.nix_wr_real, self.cin)
        self.Input_fm_real = np.zeros(InputX_size_equal).astype(np.int64)

        self.Input_fm_real[:, :self.Px_real, :] = Input_fm_real[:, :, :]

        _X = []
        for cin in range(self.cin):
            for py in range(self.Py_real):
                for i in range(self.nix_wr_real // num_in_4B):
                    _tmp = []
                    for j in range(num_in_4B):
                        _tmp.append(int(self.Input_fm_real[py][num_in_4B * i + j][cin]))
                        pass
                    _X.append(_tmp)
                    pass
        _retLsit.append(_X)
        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32

        weight_size = (self.conv_Ky, self.conv_Kx, self.cin, self.cout)
        weight_size_equal = (self.conv_Ky, self.conv_Kx, self.cin, self.cout_wr_real)

        self.weight = np.zeros(weight_size_equal).astype(np.int64)

        if self.InB_type == 2:
            weight = data_generator.random_array_u8(weight_size).astype(np.int64)
        elif self.InB_type == 3:
            weight = data_generator.random_array_2(weight_size).astype(np.int64)
        else:
            weight = data_generator.random_array_8(weight_size).astype(np.int64)

        self.weight[:, :, :, :self.cout] = weight[:, :, :, :]
        if self.InB_type == 3:
            num_in_4B = 16
        else:
            num_in_4B = 4

        _weight = []
        for w_grp_cnt in range(self.w_grp_num):
            for cin in range(self.cin):
                for ky in range(self.conv_Ky):
                    for kx in range(self.conv_Kx):
                        for i in range(32 // num_in_4B):
                            _tmp = []
                            for j in range(num_in_4B):
                                _tmp.append(int(self.weight[ky][kx][cin][num_in_4B * i + j + 32 * w_grp_cnt]))
                                pass
                            _weight.append(_tmp)
                            pass
                        pass
                    pass
                pass
            pass
        _retLsit.append(_weight)

        if self.Load_Bias == 2 or self.Load_Bias == 3:
            Bias_size = self.cout_wr_real
            self.Bias_array = data_generator.random_array_32(Bias_size)
            _bias = []
            for i in range(Bias_size):
                _tmp = []
                _tmp.append(int(self.Bias_array[i]))
                _bias.append(_tmp)
                pass
            _retLsit.append(_bias)
        else:
            pass
        # _retLsit.append(self)
        blocks = [{'name': "P81_input_X",
                   'start': self.Addr_InA_base,
                   'data': _retLsit[0],
                   'mode': 0},
                  {'name': "P81_weight",
                   'start': self.Addr_InB_base,
                   'data': _retLsit[1],
                   'mode': 0}]
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            blocks.append({'name': "P81_bias",
                           'start': self.Addr_Bias_base,
                           'data': _retLsit[2],
                           'mode': 0})
        self.memory_blocks = blocks
        return _retLsit
        pass

    def getInfoList(self) -> list:
        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32
        self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        self.Py_real = self.Input_fm_Py + self.pad_up + self.pad_down
        self.Kx_real = (self.conv_Kx - 1) * self.conv_Ex + 1
        self.Ky_real = (self.conv_Ky - 1) * self.conv_Ey + 1
        self.Output_fm_Ox = int((self.Px_real - self.Kx_real) / self.conv_Sx) + 1
        self.Output_fm_Oy = int((self.Py_real - self.Ky_real) / self.conv_Sy) + 1

        if self.InA_type == 3:  # int2 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 64)
            self.nix_wr_real = self.nix_grp_real * 64
        else:  # int8 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 16)
            self.nix_wr_real = self.nix_grp_real * 16

        if (self.InA_type == 3):
            self.Read_X_length = int(self.nix_wr_real * self.Py_real * self.cin / 16)
        else:
            self.Read_X_length = int(self.nix_wr_real * self.Py_real * self.cin / 4)

        if self.InB_type == 3:
            self.Read_weight_length = int(
                np.ceil((self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 2) / 8) * 8)
        else:
            self.Read_weight_length = int(self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 8)

        self.Read_Bias_length = self.cout_wr_real

        _infoList = []
        _X = {
            "start": self.Addr_InA_base,
            "length": self.Read_X_length,
            "type": self.InA_type
        }
        _infoList.append(_X)
        _W = {
            "start": self.Addr_InB_base,
            "length": self.Read_weight_length,
            "type": self.InB_type
        }
        _infoList.append(_W)
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            _B = {
                "start": self.Addr_Bias_base,
                "length": self.Read_Bias_length,
                "type": 0
            }
            _infoList.append(_B)
            pass
        return _infoList
        pass  # func getInfoList

    def convertPrim2Mem(self, inputList: list) -> list:
        _Vresult = []
        for NOY_cnt in range(self.Output_fm_Oy):
            for NOX_cnt in range(self.Output_fm_Ox):
                for i in range(self.cout_wr_real):
                    _tmp = []
                    _tmp.append(self.result[NOY_cnt, NOX_cnt, i])
                    _Vresult.append(_tmp)
                    pass  # for i in range(self.cout_wr_real)
                pass  # for NOX_cnt in range(self.Output_fm_Ox)
            pass  # for NOY_cnt in range(self.Output_fm_Oy)
        return _Vresult
        pass  # func convertPrim2Mem

    def convertMem2Prim(self, inputList: list):

        if self.InA_type == 3:  # int2 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 64)
            self.nix_wr_real = self.nix_grp_real * 64
            num_in_4B = 16
        else:  # int8 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 16)
            self.nix_wr_real = self.nix_grp_real * 16
            num_in_4B = 4

        InputX_wr_size_real = (self.Py_real, self.nix_wr_real, self.cin)
        self.Input_fm_real = np.zeros(InputX_wr_size_real).astype(np.int64)

        perPy_in_4B = self.nix_grp_real * (16 / 4)
        percin_in_4B = self.Py_real * perPy_in_4B

        for cin in range(self.cin):
            for py in range(self.Py_real):
                for i in range(self.nix_wr_real // num_in_4B):
                    for j in range(num_in_4B):
                        self.Input_fm_real[py][num_in_4B * i + j][cin] = inputList[0][int(py * perPy_in_4B + cin * percin_in_4B + i)][j]
                    pass

        # weight转化：四维数组
        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32
        weight_size_equal = (self.conv_Ky, self.conv_Kx, self.cin, self.cout_wr_real)
        self.weight = np.zeros(weight_size_equal).astype(np.int64)
        if self.InB_type == 3:
            num_in_4B = 16
            per32cout_in_bytes = 8  # 32个cout的weight占几个1B
        else:
            num_in_4B = 4
            per32cout_in_bytes = 32  # 32个cout的weight占几个1B

        per32cout_in_4B = per32cout_in_bytes / 4  # 32个cout的weight占几个4B

        # perKx_in_4B = self.cin * per32cout_in_4B  # kernel中一个点的每32个weight占的4B数
        perKy_in_4B = self.conv_Kx * per32cout_in_4B  # kernel中的一整行的所有点的32个weight占的4B数
        percin_in_4B = self.conv_Ky * perKy_in_4B
        perGrp_in_4B = self.cin * percin_in_4B  # 每个grp的所有weight占的4B数

        for w_grp_cnt in range(self.w_grp_num):
            for cin in range(self.cin):
                for ky in range(self.conv_Ky):
                    for kx in range(self.conv_Kx):
                        for i in range(32 // num_in_4B):
                            for j in range(num_in_4B):
                                self.weight[ky][kx][cin][num_in_4B * i + j + 32 * w_grp_cnt] = \
                                    inputList[1][int(w_grp_cnt * perGrp_in_4B + cin * percin_in_4B + ky * perKy_in_4B + kx * per32cout_in_4B + i)][j]

        # Bias转化：一位数组
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            self.Bias_size = self.cout_wr_real
            self.Bias_array = np.zeros(self.Bias_size).astype(np.int64)
            for i in range(self.Bias_size):
                self.Bias_array[i] = int(inputList[2][i][0])
        else:
            pass
        pass  # func convertMem2Prim

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)

        if self.Load_Bias == 0 or self.Load_Bias == 1:
            bias = np.zeros(self.Bias_size).astype(np.int64)   # 不支持常数bias
        else:
            bias = self.Bias_array

        result = BasicOperations.conv2d(x=self.Input_fm_real, weight=self.weight, bias=bias,
                                        kernel_size=(self.conv_Ky, self.conv_Kx),
                                        padding=(0, 0, 0, 0),
                                        stride=(self.conv_Sy, self.conv_Sx), dilation=(self.conv_Ey, self.conv_Ex))

        self.result = np.array(result, dtype=np.int64).squeeze(0).transpose(1, 2, 0)

        # self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        # self.Py_real = self.Input_fm_Py + self.pad_up + self.pad_down
        # self.Kx_real = (self.conv_Kx - 1) * self.conv_Ex + 1
        # self.Ky_real = (self.conv_Ky - 1) * self.conv_Ey + 1
        # self.Output_fm_Ox = int((self.Px_real - self.Kx_real) / self.conv_Sx) + 1
        # self.Output_fm_Oy = int((self.Py_real - self.Ky_real) / self.conv_Sy) + 1
        #
        # output_fm_size = (self.Output_fm_Oy, self.Output_fm_Ox, self.cout_wr_real)
        # self.result = np.zeros(output_fm_size).astype(np.int64)
        # if self.Load_Bias == 0 or self.Load_Bias == 1:
        #     for y in range(self.Output_fm_Oy):
        #         for x in range(self.Output_fm_Ox):
        #             for f in range(self.cout):
        #                 for r in range(self.cin):
        #                     for ky in range(self.conv_Ky):
        #                         for kx in range(self.conv_Kx):
        #                             self.result[y][x][f] += self.Input_fm_real[ky * self.conv_Ey + y * self.conv_Sy][
        #                                                         kx * self.conv_Ex + x * self.conv_Sx][r] * \
        #                                                     self.weight[ky][kx][r][f]
        #                             if self.result[y][x][f] >= 0x7fffffff:
        #                                 self.result[y][x][f] = 0x7fffffff
        #                             elif self.result[y][x][f] <= -0x80000000:
        #                                 self.result[y][x][f] = -0x80000000
        #                         pass
        # else:
        #     for y in range(self.Output_fm_Oy):
        #         for x in range(self.Output_fm_Ox):
        #             for f in range(self.cout):
        #                 self.result[y][x][f] = self.Bias_array[f]
        #                 for r in range(self.cin):
        #                     for ky in range(self.conv_Ky):
        #                         for kx in range(self.conv_Kx):
        #                             self.result[y][x][f] += self.Input_fm_real[ky * self.conv_Ey + y * self.conv_Sy][
        #                                                         kx * self.conv_Ex + x * self.conv_Sx][r] * \
        #                                                     self.weight[ky][kx][r][f]
        #                             if self.result[y][x][f] >= 0x7fffffff:
        #                                 self.result[y][x][f] = 0x7fffffff
        #                             elif self.result[y][x][f] <= -0x80000000:
        #                                 self.result[y][x][f] = -0x80000000
        #                         pass
        # pass

        self.Write_V_length = self.Output_fm_Oy * self.Output_fm_Ox * self.cout_wr_real
        # --------------------------------------------------example start----------------------------------------------
        _resultList = []
        _Vresult = self.convertPrim2Mem([self.result])  # 多维数组转化到Mem中的存储数组的4Bytes格式
        # print(_Vresult)
        _V = {
            "Model": "Axon",
            "data": _Vresult,
            "startInMemory": self.Addr_V_base,
            "lengthInMemory": self.Write_V_length
        }
        _resultList.append(_V)
        return _resultList

        # --------------------------------------------------example   end-----------------------------------------------

    def cal_para(self):

        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32
        self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        self.Py_real = self.Input_fm_Py + self.pad_up + self.pad_down
        self.Kx_real = (self.conv_Kx - 1) * self.conv_Ex + 1
        self.Ky_real = (self.conv_Ky - 1) * self.conv_Ey + 1
        self.Output_fm_Ox = int((self.Px_real - self.Kx_real) / self.conv_Sx) + 1
        self.Output_fm_Oy = int((self.Py_real - self.Ky_real) / self.conv_Sy) + 1
        Row_in_mem = math.ceil(self.Px_real / 16)
        oneFM_in_mem = Row_in_mem * self.Py_real

        if self.InA_type == 3:  # int2 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 64)
            self.nix_wr_real = self.nix_grp_real * 64
        else:  # int8 按16B补零
            self.nix_grp_real = math.ceil(self.Px_real / 16)
            self.nix_wr_real = self.nix_grp_real * 16

        self.L0_num = self.conv_Kx - 1
        self.L1_num = self.conv_Ky - 1
        self.L2_num = self.cin - 1
        self.L3_num = self.w_grp_num - 1
        self.L4_num = math.ceil(self.Output_fm_Ox / 4) - 1
        self.L5_num = self.Output_fm_Oy - 1
        self.MAC_grp_num_last = self.Output_fm_Ox - self.L4_num * 4 - 1
        self.Addr_InA_L1_step = Row_in_mem * self.conv_Ey
        self.Addr_InA_L2_step = oneFM_in_mem - Row_in_mem * (self.Ky_real - 1)
        self.Addr_InA_L3_step = 0 - (self.cin - 1) * oneFM_in_mem - Row_in_mem * (self.Ky_real - 1)
        self.Addr_InA_L4_step = self.Addr_InA_L3_step

        Px_last_cnt = self.L4_num * 4 * self.conv_Sx
        Addr_L5_step_base = 0 - (
                (self.cin - 1) * oneFM_in_mem + Row_in_mem * self.Ky_real - 1) + Row_in_mem * self.conv_Sy

        self.nix_wr_use_real = (self.Output_fm_Ox - 1) * self.conv_Sx + self.Kx_real
        nix_use_wr_real = math.ceil(self.nix_wr_use_real / 16) * 16
        # 把self.nix_wr_real全部换成self.nix_wr_use_real
        if self.nix_wr_use_real <= 16:
            self.Addr_InA_L5_step = Addr_L5_step_base
        elif self.nix_wr_use_real <= 32 and Px_last_cnt < 16:
            self.Addr_InA_L5_step = Addr_L5_step_base + 1
        elif self.nix_wr_use_real <= 32 and Px_last_cnt >= 16:
            self.Addr_InA_L5_step = Addr_L5_step_base
        elif self.nix_wr_use_real > 32 and Px_last_cnt < (nix_use_wr_real - 32):  # (self.nix_wr_real - 32)
            self.Addr_InA_L5_step = Addr_L5_step_base + 2
        elif self.nix_wr_use_real > 32 and Px_last_cnt < (nix_use_wr_real - 16):  # (self.nix_wr_real - 16)
            self.Addr_InA_L5_step = Addr_L5_step_base + 1
        else:
            self.Addr_InA_L5_step = Addr_L5_step_base

        # 有用数据判断
        if (self.nix_wr_real - self.nix_wr_use_real > 15):
            self.Addr_InA_L5_step = self.Addr_InA_L5_step + 1
        else:
            pass

        if (self.InA_type == 3):
            self.Read_X_length = int(self.nix_wr_real * self.Py_real * self.cin / 16)
        else:
            self.Read_X_length = int(self.nix_wr_real * self.Py_real * self.cin / 4)

        if self.InB_type == 3:
            self.Read_weight_length = int(
                np.ceil((self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 2) / 8) * 8)
        else:
            self.Read_weight_length = int(self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 8)

        self.Read_Bias_length = self.cout_wr_real
        self.Write_V_length = self.Output_fm_Oy * self.Output_fm_Ox * self.cout_wr_real

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('Reset_Addr_A = ' + str(self.Reset_Addr_A) + '\n')
            f.write('Reset_Addr_V = ' + str(self.Reset_Addr_V) + '\n')
            f.write('MAC_grp_num_last = ' + str(self.MAC_grp_num_last) + '\n')
            f.write('Addr_InA_base = ' + str(hex(self.Addr_InA_base >> 2)) + '\n')
            f.write('InA_type = ' + str(self.InA_type) + '\n')
            f.write('Addr_InB_base = ' + str(hex(self.Addr_InB_base >> 2)) + '\n')
            f.write('InB_type = ' + str(self.InB_type) + '\n')
            f.write('PI_A.Addr_Bias_base = ' + str(hex(self.Addr_Bias_base >> 2)) + ';\n')
            f.write('Load_Bias = ' + str(self.Load_Bias) + '\n')
            f.write('Addr_V_base = ' + str(hex(self.Addr_V_base >> 2)) + '\n')
            f.write('PI_A.Addr_InA_end = ' + str(
                hex(((self.Addr_InA_base + self.Read_X_length + 8) >> 2) - 1) + '\n'))  # 保留修改
            f.write('PI_A.Addr_V_end = ' + str(hex((((self.Addr_V_base + self.Write_V_length) >> 3) - 1) << 1) + '\n'))
            f.write('Load_Bias = ' + str(self.Load_Bias) + '\n')
            f.write('L0_num = ' + str(self.L0_num) + '\n')
            f.write('L1_num = ' + str(self.L1_num) + '\n')
            f.write('L2_num = ' + str(self.L2_num) + '\n')
            f.write('L3_num = ' + str(self.L3_num) + '\n')
            f.write('L4_num = ' + str(self.L4_num) + '\n')
            f.write('L5_num = ' + str(self.L5_num) + '\n')
            f.write('Addr_InA_L1_step = ' + str(self.Addr_InA_L1_step) + '\n')
            f.write('Addr_InA_L2_step = ' + str(self.Addr_InA_L2_step) + '\n')
            f.write('Addr_InA_L3_step = ' + str(self.Addr_InA_L3_step) + '\n')
            f.write('Addr_InA_L4_step = ' + str(self.Addr_InA_L4_step) + '\n')
            f.write('Addr_InA_L5_step = ' + str(self.Addr_InA_L5_step) + '\n')
            f.write('Sx = ' + str(self.conv_Sx) + '\n')
            f.write('Sy = ' + str(self.conv_Sy) + '\n')
            f.write('Ex = ' + str(self.conv_Ex - 1) + '\n')
            f.write('Ey = ' + str(self.conv_Ey - 1) + '\n')
            f.write('pad_up = ' + str(self.pad_up) + '\n')
            f.write('pad_down = ' + str(self.pad_down) + '\n')
            f.write('pad_left = ' + str(self.pad_left) + '\n')
            f.write('pad_right = ' + str(self.pad_right) + '\n')
            f.write('******************************' + '\n')
            f.write('Input_fm_Px = ' + str(self.Input_fm_Px) + '\n')
            f.write('Input_fm_Py = ' + str(self.Input_fm_Py) + '\n')
            f.write('Output_fm_Ox = ' + str(self.Output_fm_Ox) + '\n')
            f.write('Output_fm_Oy = ' + str(self.Output_fm_Oy) + '\n')
            f.write('Read_X_length = ' + str(self.Read_X_length) + '\n')
            f.write('Read_Bias_length = ' + str(self.Read_Bias_length) + '\n')
            f.write('Read_weight_length = ' + str(self.Read_weight_length) + '\n')
            f.write('Write_V_length = ' + str(self.Write_V_length) + '\n')
            pass

    def save_results(self, SIMPATH, TBNAME, t):

        path = SIMPATH + TBNAME

        with open(path + '/Data_VOUT.txt', 'w') as f:

            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for i in range(self.cout_wr_real):
                        final_string = hex_to_string(self.result[NOY_cnt, NOX_cnt, i], width=8)
                        f.write(final_string)
                        f.write('\n')
                        pass

        with open(path + '/Data_VOUT_dec.txt', 'w') as f:
            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for CIN_cnt in range(self.cout_wr_real):
                        final_value = self.result[NOY_cnt, NOX_cnt, CIN_cnt]
                        f.write(str(final_value))
                        f.write('\n')
                        pass
