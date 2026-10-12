import numpy as np
import math
import copy

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_83_Axon(Primitive):
    def __init__(self):
        super().__init__()

        self.tensor_en = False
        self.InA_type = 1
        self.Load_Bias = 0
        self.Bias_length = 0
        self.X_array_num = 0
        self.Px = 0
        self.Py = 0
        self.stride_x = 1
        self.stride_y = 1
        self.cin = 0
        # self.constant_b = 0
        self.PIC = 0x83
        self.Reset_Addr_A = 1
        self.Reset_Addr_V = 1
        self.Addr_InA_base = 0x00
        self.Addr_Bias_base = 0x00
        self.Addr_V_base = 0x00
        self.Addr_InA_end = 0x00
        self.Addr_V_end = 0x00
        self.A2S2_mode = 0

        self.total_core_num = 1
        self.areas_num_in_core = 1

    def __str__(self):
        return "83(aX+Bias)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)
        _retLsit = []

        self.constant_a = data_generator.random_constant_a()

        if self.Load_Bias == 0 or self.Load_Bias == 1:
            self.constant_b = data_generator.random_constant_32()
            self.cin_wr_real = math.ceil(self.cin / 32) * 32
        else:
            self.Bias_size = self.Bias_length
            self.Bias_size_in_equal = (np.ceil(self.Bias_length / 32).astype(int)) * 32
            self.Bias_array = np.zeros(self.Bias_size_in_equal).astype(np.int64)
            Bias_array = data_generator.random_array_32(self.Bias_size).astype(np.int64)
            self.Bias_array[:self.Bias_length] = Bias_array[:]
            self.cin_wr_real = self.Bias_size_in_equal

        if self.tensor_en:
            InputX_size = (self.Py, self.Px, self.cin)
            InputX_size_equal = (self.Py, self.Px, self.cin_wr_real)
        else:
            InputX_size = (self.X_array_num, self.cin)
            InputX_size_equal = (self.X_array_num, self.cin_wr_real)

        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)

        if self.InA_type == 2:
            InputX_array = data_generator.random_array_u8(InputX_size).astype(np.int64)
        else:
            InputX_array = data_generator.random_array_8(InputX_size).astype(np.int64)

        if self.tensor_en:
            self.InputX_array[:, :, :self.cin] = InputX_array[:, :, :]
        else:
            self.InputX_array[:, :self.cin] = InputX_array[:, :]
        num_in_4B = 4
        _data = []
        if self.tensor_en:
            for py in range(self.Py):
                for px in range(self.Px):
                    for i in range(self.cin_wr_real // num_in_4B):
                        _tmp = []
                        for j in range(num_in_4B):
                            _tmp.append(int(self.InputX_array[py][px][num_in_4B * i + j]))
                            pass
                        _data.append(_tmp)
                        pass
        else:
            for px in range(self.X_array_num):
                for i in range(self.cin_wr_real // num_in_4B):
                    _tmp = []
                    for j in range(num_in_4B):
                        _tmp.append(int(self.InputX_array[px][num_in_4B * i + j]))
                        pass
                    _data.append(_tmp)
                    pass
        _retLsit.append(_data)
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            _data = []
            for i in range(self.Bias_size_in_equal):
                _tmp = []
                _tmp.append(int(self.Bias_array[i]))
                _data.append(_tmp)
                pass
            _retLsit.append(_data)
        # _retLsit.append(self)

        blocks = [{'name': "P83_input_X",
                   'start': self.Addr_InA_base,
                   'data': _retLsit[0],
                   'mode': 0}]
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            blocks.append({'name': "P83_bias",
                           'start': self.Addr_Bias_base,
                           'data': _retLsit[1],
                           'mode': 0})
        self.memory_blocks = blocks
        return _retLsit

    def getInfoList(self) -> list:

        self.cin_wr_real = math.ceil(self.cin / 32) * 32

        if self.tensor_en:
            self.Read_X_length = self.Py * self.Px * (math.ceil(self.cin_wr_real / 32) * 8)
        else:
            self.Read_X_length = self.X_array_num * (math.ceil(self.cin_wr_real / 32) * 8)

        self.Read_Bias_length = math.ceil(self.Bias_length / 32) * 32
        _infoList = []
        _X = {
            "start": self.Addr_InA_base,
            "length": self.Read_X_length,
            "type": self.InA_type
        }
        _infoList.append(_X)
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
        if self.tensor_en:
            for oy_cnt in range(self.Oy):
                for ox_cnt in range(self.Ox):
                    for i in range(self.cin_wr_real):
                        _tmp = []
                        _tmp.append(self.result[oy_cnt][ox_cnt][i])
                        _Vresult.append(_tmp)
                        pass  # for i in range(self.cout_wr_real)
                    pass  # for NOX_cnt in range(self.Output_fm_Ox)
                pass  # for NOY_cnt in range(self.Output_fm_Oy)
        else:
            for ox_cnt in range(self.X_array_num):
                for i in range(self.cin_wr_real):
                    _tmp = []
                    _tmp.append(self.result[ox_cnt][i])
                    _Vresult.append(_tmp)
                    pass  # for i in range(self.cout_wr_real)
                pass  # for NOX_cnt in range(self.Output_fm_Ox)
        return _Vresult
        pass  # func convertPrim2Mem

    def convertMem2Prim(self, inputList: list):
        self.cin_wr_real = math.ceil(self.cin / 32) * 32
        num_in_4B = 4
        if self.tensor_en:
            InputX_size_equal = (self.Py, self.Px, self.cin_wr_real)
        else:
            InputX_size_equal = (self.X_array_num, self.cin_wr_real)

        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)
        perPx_in_4B = math.ceil(self.cin / 32) * (32 / 4)  # 每个点的所有cin占多少的4bytes
        perPy_in_4B = self.Px * perPx_in_4B  # 输入图像的每一行的所有cin占多少的4bytes

        if self.tensor_en:
            for py in range(self.Py):
                for px in range(self.Px):
                    for i in range(self.cin_wr_real // num_in_4B):
                        for j in range(num_in_4B):
                            self.InputX_array[py][px][num_in_4B * i + j] = \
                                inputList[0][int(py * perPy_in_4B + px * perPx_in_4B + i)][j]
                        pass
        else:
            for px in range(self.X_array_num):
                for i in range(self.cin_wr_real // num_in_4B):
                    for j in range(num_in_4B):
                        self.InputX_array[px][num_in_4B * i + j] = \
                            inputList[0][int(px * perPx_in_4B + i)][j]
                    pass

        # Bias转化：一位数组
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            self.Bias_size = self.Bias_length
            self.Bias_size_in_equal = math.ceil(self.Bias_length / 32) * 32
            self.Bias_array = np.zeros(self.Bias_size_in_equal).astype(np.int64)
            for i in range(self.Bias_size):
                self.Bias_array[i] = int(inputList[1][i][0])
        else:
            pass
        pass  # func convertMem2Prim

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)
        if self.Load_Bias == 0 or self.Load_Bias == 1:
            self.cin_wr_real = math.ceil(self.cin / 32) * 32
        else:
            self.Bias_size_in_equal = (np.ceil(self.Bias_length / 32).astype(int)) * 32
            self.cin_wr_real = self.Bias_size_in_equal

        self.Oy = int((self.Py - 1) / self.stride_y) + 1
        self.Ox = int((self.Px - 1) / self.stride_x) + 1

        if self.tensor_en:
            output_size = (self.Oy, self.Ox, self.cin_wr_real)
            self.result = np.zeros(output_size).astype(np.int64)
            if self.Load_Bias == 0 or self.Load_Bias == 1:
                for oy_cnt in range(self.Oy):
                    for ox_cnt in range(self.Ox):
                        for cin_cnt in range(self.cin_wr_real):
                            self.result[oy_cnt][ox_cnt][cin_cnt] = self.constant_b + self.constant_a * self.InputX_array[oy_cnt * self.stride_y][ox_cnt * self.stride_x][cin_cnt]
                            if self.result[oy_cnt][ox_cnt][cin_cnt] >= 0x7fffffff:
                                self.result[oy_cnt][ox_cnt][cin_cnt] = 0x7fffffff
                            elif self.result[oy_cnt][ox_cnt][cin_cnt] <= -0x80000000:
                                self.result[oy_cnt][ox_cnt][cin_cnt] = -0x80000000
                        pass
            else:
                for oy_cnt in range(self.Oy):
                    for ox_cnt in range(self.Ox):
                        for cin_cnt in range(self.cin_wr_real):
                            self.result[oy_cnt][ox_cnt][cin_cnt] = self.Bias_array[cin_cnt] + self.constant_a * self.InputX_array[oy_cnt * self.stride_y][ox_cnt * self.stride_x][
                                cin_cnt]
                            if self.result[oy_cnt][ox_cnt][cin_cnt] >= 0x7fffffff:
                                self.result[oy_cnt][ox_cnt][cin_cnt] = 0x7fffffff
                            elif self.result[oy_cnt][ox_cnt][cin_cnt] <= -0x80000000:
                                self.result[oy_cnt][ox_cnt][cin_cnt] = -0x80000000
                        pass
        else:
            output_size = (self.X_array_num, self.cin_wr_real)
            self.result = np.zeros(output_size).astype(np.int64)
            if self.Load_Bias == 0 or self.Load_Bias == 1:
                for i in range(self.X_array_num):
                    for cin_cnt in range(self.cin_wr_real):
                        self.result[i][cin_cnt] = self.constant_b + self.constant_a * \
                                                  self.InputX_array[i][cin_cnt]
                        if self.result[i][cin_cnt] >= 0x7fffffff:
                            self.result[i][cin_cnt] = 0x7fffffff
                        elif self.result[i][cin_cnt] <= -0x80000000:
                            self.result[i][cin_cnt] = -0x80000000
                    pass
            else:
                for i in range(self.X_array_num):
                    for cin_cnt in range(self.cin_wr_real):
                        self.result[i][cin_cnt] = self.Bias_array[cin_cnt] + self.constant_a * \
                                                  self.InputX_array[i][cin_cnt]
                        if self.result[i][cin_cnt] >= 0x7fffffff:
                            self.result[i][cin_cnt] = 0x7fffffff
                        elif self.result[i][cin_cnt] <= -0x80000000:
                            self.result[i][cin_cnt] = -0x80000000
                    pass

        if self.tensor_en:
            self.Write_V_length = self.Oy * self.Ox * self.cin_wr_real
        else:
            self.Write_V_length = self.X_array_num * self.cin_wr_real

        _resultList = []
        _Vresult = self.convertPrim2Mem([self.result])  # 多维数组转化到Mem中的存储数组的4Bytes格式

        _V = {
            "Model": "Axon",
            "data": _Vresult,
            "startInMemory": self.Addr_V_base,
            "lengthInMemory": self.Write_V_length
        }
        _resultList.append(_V)
        return _resultList

    def cal_para(self):
        self.cin_wr_real = math.ceil(self.cin / 32) * 32
        self.Oy = int((self.Py - 1) / self.stride_y) + 1
        self.Ox = int((self.Px - 1) / self.stride_x) + 1
        if self.Load_Bias == 0 or self.Load_Bias == 1:
            self.L3_num = math.ceil(self.cin / 32) - 1
        else:
            self.L3_num = math.ceil(self.Bias_length / 32) - 1
            self.constant_b = 0

        if self.tensor_en:
            self.L4_num = self.Ox - 1
            self.L5_num = self.Oy - 1
            self.Addr_InA_L3_step = 2
            self.Addr_InA_L4_step = 2 * (self.stride_x - 1) * (self.L3_num + 1) + 2
            self.Addr_InA_L5_step = 2 * (self.stride_y - 1) * self.Px * (self.L3_num + 1) + 2 * (
                    self.Px - 1 - (self.Ox - 1) * self.stride_x) * (self.L3_num + 1) + 2
        else:
            self.L4_num = 0
            self.L5_num = self.X_array_num - 1
            self.Addr_InA_L3_step = 2
            self.Addr_InA_L4_step = 2
            self.Addr_InA_L5_step = 2

        self.Read_Bias_length = math.ceil(self.Bias_length / 32) * 32

        if self.tensor_en:
            self.Read_X_length = self.Py * self.Px * (math.ceil(self.cin_wr_real / 32) * 8)
            self.Write_V_length = self.Oy * self.Ox * self.cin_wr_real
        else:
            self.Read_X_length = self.X_array_num * (math.ceil(self.cin_wr_real / 32) * 8)
            self.Write_V_length = self.X_array_num * self.cin_wr_real

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('Reset_Addr_A = ' + str(self.Reset_Addr_A) + '\n')
            f.write('Reset_Addr_V = ' + str(self.Reset_Addr_V) + '\n')
            f.write('Addr_InA_base = ' + str(hex(self.Addr_InA_base >> 2)) + '\n')
            f.write('InA_type = ' + str(self.InA_type) + '\n')
            f.write('PI_A.Addr_Bias_base = ' + str(hex(self.Addr_Bias_base >> 2)) + ';\n')
            f.write('Load_Bias = ' + str(self.Load_Bias) + '\n')
            f.write('Addr_V_base = ' + str(hex(self.Addr_V_base >> 2)) + '\n')
            f.write('PI_A.Addr_InA_end = ' + str(
                hex(((self.Addr_InA_base + self.Read_X_length) >> 2) - 1) + '\n'))
            f.write('PI_A.Addr_V_end = ' + str(hex((((self.Addr_V_base + self.Write_V_length) >> 3) - 1) << 1) + '\n'))
            f.write('L3_num = ' + str(self.L3_num) + '\n')
            f.write('L4_num = ' + str(self.L4_num) + '\n')
            f.write('L5_num = ' + str(self.L5_num) + '\n')
            f.write('Addr_InA_L3_step = ' + str(self.Addr_InA_L3_step) + '\n')
            f.write('Addr_InA_L4_step = ' + str(self.Addr_InA_L4_step) + '\n')
            f.write('Addr_InA_L5_step = ' + str(self.Addr_InA_L5_step) + '\n')
            f.write('constant_a = ' + str(self.constant_a) + '\n')
            f.write('constant_b = ' + str(self.constant_b) + '\n')
            f.write('******************************' + '\n')
            f.write('Px = ' + str(self.Px) + '\n')
            f.write('Py = ' + str(self.Py) + '\n')
            f.write('Ox = ' + str(self.Ox) + '\n')
            f.write('Oy = ' + str(self.Oy) + '\n')
            f.write('Read_X_length = ' + str(self.Read_X_length) + '\n')
            f.write('Read_Bias_length = ' + str(self.Read_Bias_length) + '\n')
            f.write('Write_V_length = ' + str(self.Write_V_length) + '\n')

    def save_results(self, SIMPATH, TBNAME, t):

        path = SIMPATH + TBNAME

        with open(path + '/Data_VOUT.txt', 'w') as f:

            if self.tensor_en:
                for oy_cnt in range(self.Oy):
                    for ox_cnt in range(self.Ox):
                        for i in range(self.cin_wr_real):
                            final_string = hex_to_string(self.result[oy_cnt][ox_cnt][i], width=8)
                            f.write(final_string)
                            f.write('\n')
                            pass
            else:
                for ox_cnt in range(self.X_array_num):
                    for i in range(self.cin_wr_real):
                        final_string = hex_to_string(self.result[ox_cnt][i], width=8)
                        f.write(final_string)
                        f.write('\n')
                        pass

        with open(path + '/Data_VOUT_dec.txt', 'w') as f:
            if self.tensor_en:
                for oy_cnt in range(self.Oy):
                    for ox_cnt in range(self.Ox):
                        for i in range(self.cin_wr_real):
                            final_value = self.result[oy_cnt][ox_cnt][i]
                            f.write(str(final_value))
                            f.write('\n')
            else:
                for ox_cnt in range(self.X_array_num):
                    for i in range(self.cin_wr_real):
                        final_value = self.result[ox_cnt][i]
                        f.write(str(final_value))
                        f.write('\n')
