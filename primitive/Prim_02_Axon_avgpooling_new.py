import numpy as np
import math
from primitive.basic_operations import BasicOperations
from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_02_Axon(Primitive):
    def __init__(self):
        super().__init__()

        self.InA_type = 1  # 可配置更改：[00]int32[01]int8[10]uint8 [11]Tenary
        self.Load_Bias = 0  # 可配置更改：2,3为Bias 0,1为常数b
        self.Bias_length = 0
        self.pad_on = False
        self.avg_pooling_en = True
        self.Input_fm_Px = 0
        self.Input_fm_Py = 0
        self.pooling_Kx = 0
        self.pooling_Ky = 0
        self.pooling_Sx = 1
        self.pooling_Sy = 1
        self.pad_top = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.cin = 0
        self.cout = self.cin
        self.PIC = 0x02
        self.Reset_Addr_A = 1
        self.Reset_Addr_V = 1
        self.Addr_InA_base = 0x00
        self.Addr_Bias_base = 0x00
        self.Addr_V_base = 0x00
        self.Addr_InA_end = 0x00
        self.Addr_V_end = 0x00
        self.constant_b = 0
        self.A2S2_mode = 0

        self.total_core_num = 1
        self.areas_num_in_core = 1

    def __str__(self):
        return "02(X1+X2)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)

        _retLsit = []
        if self.InA_type == 0:
            self.Km_num = np.ceil(self.cin / 8).astype(int)
            self.cin_wr_real = self.Km_num * 8
            num_in_4B = 1
        elif self.InA_type == 3:
            self.Km_num = np.ceil(self.cin / 128).astype(int)
            self.cin_wr_real = self.Km_num * 128
            num_in_4B = 16
        else:
            self.Km_num = np.ceil(self.cin / 32).astype(int)
            self.cin_wr_real = self.Km_num * 32
            num_in_4B = 4

        self.array_num = self.pooling_Kx * self.pooling_Ky

        if self.avg_pooling_en:
            InputX_size = (self.Input_fm_Py, self.Input_fm_Px, self.cin)
            InputX_size_equal = (self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)
        else:
            InputX_size = (self.array_num, self.Input_fm_Py, self.Input_fm_Px, self.cin)
            InputX_size_equal = (self.array_num, self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)

        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)

        if self.InA_type == 0:
            InputX_array = data_generator.random_array_32(InputX_size).astype(np.int64)
        elif self.InA_type == 2:
            InputX_array = data_generator.random_array_u8(InputX_size).astype(np.int64)
        elif self.InA_type == 3:
            InputX_array = data_generator.random_array_2(InputX_size).astype(np.int64)
        else:
            InputX_array = data_generator.random_array_8(InputX_size).astype(np.int64)

        if self.avg_pooling_en:
            self.InputX_array[:, :, :self.cin] = InputX_array[:, :, :]
            _data = []
            for py in range(self.Input_fm_Py):
                for px in range(self.Input_fm_Px):
                    for i in range(self.cin_wr_real // num_in_4B):
                        _tmp = []
                        for j in range(num_in_4B):
                            _tmp.append(int(self.InputX_array[py][px][num_in_4B * i + j]))
                            pass
                        _data.append(_tmp)
                        pass
                    pass
            _retLsit.append(_data)
        else:
            self.InputX_array[:, :, :, :self.cin] = InputX_array[:, :, :, :]
            _data = []
            for P_cnt in range(self.array_num):
                for py in range(self.Input_fm_Py):
                    for px in range(self.Input_fm_Px):
                        for i in range(self.cin_wr_real // num_in_4B):
                            _tmp = []
                            for j in range(num_in_4B):
                                _tmp.append(int(self.InputX_array[P_cnt][py][px][num_in_4B * i + j]))
                                pass
                            _data.append(_tmp)
                            pass
                        pass
            _retLsit.append(_data)

        if self.Load_Bias == 0 or self.Load_Bias == 1:
            # self.constant_b = data_generator.random_constant_32()
            pass
        else:
            Bias_size = self.Bias_length
            self.Bias_array = data_generator.random_array_32(Bias_size)
            _data = []
            for i in range(Bias_size):
                _tmp = []
                _tmp.append(int(self.Bias_array[i]))
                _data.append(_tmp)
                pass
            _retLsit.append(_data)

        # _retLsit.append(self)
        blocks = [{'name': "P02_input_X",
                   'start': self.Addr_InA_base,
                   'data': _retLsit[0],
                   'mode': 0}]
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            blocks.append({'name': "P02_bias",
                           'start': self.Addr_Bias_base,
                           'data': _retLsit[1],
                           'mode': 0})
        self.memory_blocks = blocks
        return _retLsit
        pass

    def getInfoList(self) -> list:
        if self.InA_type == 0:
            self.Km_num = np.ceil(self.cin / 8).astype(int)
            self.cin_wr_real = self.Km_num * 8
        elif self.InA_type == 3:
            self.Km_num = np.ceil(self.cin / 128).astype(int)
            self.cin_wr_real = self.Km_num * 128
        else:
            self.Km_num = np.ceil(self.cin / 32).astype(int)
            self.cin_wr_real = self.Km_num * 32

        if self.avg_pooling_en:
            self.Read_X_length = self.Input_fm_Px * self.Input_fm_Py * self.Km_num * 8
            # self.Write_V_length = self.Output_fm_Oy * self.Output_fm_Ox * self.cin_wr_real
        else:
            self.Read_X_length = self.array_num * self.Input_fm_Px * self.Input_fm_Py * self.Km_num * 8
            # self.Write_V_length = self.Input_fm_Px * self.Input_fm_Py * self.cin_wr_real

        self.Read_Bias_length = self.Bias_length

        self.array_num = self.pooling_Kx * self.pooling_Ky

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
        if self.avg_pooling_en:
            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for i in range(self.cin_wr_real):
                        _tmp = []
                        _tmp.append(self.result[NOY_cnt, NOX_cnt, i])
                        _Vresult.append(_tmp)
                        pass  # for i in range(self.cout_wr_real)
                    pass  # for NOX_cnt in range(self.Output_fm_Ox)
                pass  # for NOY_cnt in range(self.Output_fm_Oy)
        else:
            for py_cnt in range(self.Input_fm_Py):
                for px_cnt in range(self.Input_fm_Px):
                    for i in range(self.cin_wr_real):
                        _tmp = []
                        _tmp.append(self.result[py_cnt, px_cnt, i])
                        _Vresult.append(_tmp)
                        pass  # for i in range(self.cout_wr_real)
                    pass  # for NOX_cnt in range(self.Output_fm_Ox)
                pass  # for NOY_cnt in range(self.Output_fm_Oy)
        return _Vresult
        pass  # func convertPrim2Mem

    def convertMem2Prim(self, inputList: list):
        if self.InA_type == 0:
            self.Km_num = math.ceil(self.cin / 8)
            self.cin_wr_real = self.Km_num * 8
            num_in_4B = 1
        elif self.InA_type == 3:
            self.Km_num = math.ceil(self.cin / 128)
            self.cin_wr_real = self.Km_num * 128
            num_in_4B = 16
        else:
            self.Km_num = math.ceil(self.cin / 32)
            self.cin_wr_real = self.Km_num * 32
            num_in_4B = 4

        self.array_num = self.pooling_Kx * self.pooling_Ky

        if self.avg_pooling_en:
            InputX_size_equal = (self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)
        else:
            InputX_size_equal = (self.array_num, self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)

        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)

        perPx_in_4B = self.Km_num * (32 / 4)  # 每个点的所有cin占多少的4bytes
        perPy_in_4B = self.Input_fm_Px * perPx_in_4B  # 输入图像的每一行的所有cin占多少的4bytes
        perarray_in_4B = self.Input_fm_Py * perPy_in_4B

        # print(type(np.int64(inputList[0][0][0])))
        # InputX转化：三维数组
        if self.avg_pooling_en:
            for py in range(self.Input_fm_Py):
                for px in range(self.Input_fm_Px):
                    for i in range(self.cin_wr_real // num_in_4B):
                        for j in range(num_in_4B):
                            self.InputX_array[py][px][num_in_4B * i + j] = inputList[0][int(py * perPy_in_4B + px * perPx_in_4B + i)][j]
                        pass
        else:
            for P_cnt in range(self.array_num):
                for py in range(self.Input_fm_Py):
                    for px in range(self.Input_fm_Px):
                        for i in range(self.cin_wr_real // num_in_4B):
                            for j in range(num_in_4B):
                                self.InputX_array[P_cnt][py][px][num_in_4B * i + j] = inputList[0][int(P_cnt * perarray_in_4B + py * perPy_in_4B + px * perPx_in_4B + i)][j]
                            pass

        # Bias转化：一位数组
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            self.Bias_size = self.Bias_length
            self.Bias_array = np.zeros(self.Bias_size).astype(np.int64)
            for i in range(self.Bias_size):
                self.Bias_array[i] = int(inputList[1][i][0])
        else:
            pass
        pass  # func convertMem2Prim

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)
        # 先对输入 padding 补零
        self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        self.Py_real = self.Input_fm_Py + self.pad_top + self.pad_down
        self.Bias_grp = math.ceil(self.Bias_length / 32)
        self.array_num = self.pooling_Kx * self.pooling_Ky
        self.Output_fm_Ox = int((self.Px_real - self.pooling_Kx) / self.pooling_Sx) + 1
        self.Output_fm_Oy = int((self.Py_real - self.pooling_Ky) / self.pooling_Sy) + 1

        if self.avg_pooling_en and self.pad_on:

            self.Input_fm_real = BasicOperations.add_pad(
                self.InputX_array, padding=(self.pad_top, self.pad_left, self.pad_down, self.pad_right))

            # InputX_fm_size_real = (self.Py_real, self.Px_real, self.cin_wr_real)
            # self.Input_fm_real = np.zeros(InputX_fm_size_real).astype(np.int64)
            #
            # for NOY_cnt in range((self.pad_top - 1), (self.pad_top + self.Input_fm_Py - 1)):
            #     for NOX_cnt in range((self.pad_left - 1), (self.pad_left + self.Input_fm_Px - 1)):
            #         for CIN_cnt in range(self.cin_wr_real):
            #             self.Input_fm_real[NOY_cnt + 1][NOX_cnt + 1][CIN_cnt] = \
            #                 self.InputX_array[NOY_cnt + 1 - self.pad_top][NOX_cnt + 1 - self.pad_left][CIN_cnt]
            #         pass
        else:
            self.Input_fm_real = self.InputX_array

        if self.Load_Bias == 0 or self.Load_Bias == 1:
            bias = self.constant_b
        else:
            bias = np.array(self.Bias_array, dtype=np.int64).reshape(-1)
        if self.avg_pooling_en:
            self.result = BasicOperations.avg_pool(x=self.Input_fm_real, kernel_size=(self.pooling_Ky, self.pooling_Kx),
                                                   stride=(self.pooling_Sy, self.pooling_Sx), bias=bias)
            # output_fm_size = (self.Output_fm_Oy, self.Output_fm_Ox, self.cin_wr_real)
            # self.result = np.zeros(output_fm_size).astype(np.int64)
            # if self.Load_Bias == 0 or self.Load_Bias == 1:
            #     for f in range(self.cin):
            #         for oy in range(self.Output_fm_Oy):
            #             for ox in range(self.Output_fm_Ox):
            #                 sum_aver = self.constant_b  # np.array(0).astype(np.int64)
            #                 for oky in range(self.pooling_Ky):
            #                     for okx in range(self.pooling_Kx):
            #                         sum_aver += \
            #                             self.Input_fm_real[oky + oy * self.pooling_Sy][okx + ox * self.pooling_Sx][f]
            #                         if sum_aver >= 0x7fffffff:
            #                             sum_aver = 0x7fffffff
            #                         elif sum_aver <= -0x80000000:
            #                             sum_aver = -0x80000000
            #                 self.result[oy][ox][f] = sum_aver
            # else:
            #     for f in range(self.cin):
            #         for oy in range(self.Output_fm_Oy):
            #             for ox in range(self.Output_fm_Ox):
            #                 sum_aver = self.Bias_array[f]  # np.array(0).astype(np.int64)
            #                 for oky in range(self.pooling_Ky):
            #                     for okx in range(self.pooling_Kx):
            #                         sum_aver += \
            #                             self.Input_fm_real[oky + oy * self.pooling_Sy][okx + ox * self.pooling_Sx][
            #                                 f]
            #                         if sum_aver >= 0x7fffffff:
            #                             sum_aver = 0x7fffffff
            #                         elif sum_aver <= -0x80000000:
            #                             sum_aver = -0x80000000
            #                 self.result[oy][ox][f] = sum_aver
        else:
            self.result = BasicOperations.tensor_sum(x=self.InputX_array, bias=bias)
            # output_array_size = (self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)
            # self.result = np.zeros(output_array_size).astype(np.int64)
            # if self.Load_Bias == 0 or self.Load_Bias == 1:
            #     for py_cnt in range(self.Input_fm_Py):
            #         for px_cnt in range(self.Input_fm_Px):
            #             for f in range(self.cin):
            #                 sum_array = self.constant_b  # 可能需要修改
            #                 for cnt in range(self.array_num):
            #                     sum_array += self.InputX_array[cnt][py_cnt][px_cnt][f]
            #                     if sum_array >= 0x7fffffff:
            #                         sum_array = 0x7fffffff
            #                     elif sum_array <= -0x80000000:
            #                         sum_array = -0x80000000
            #                 self.result[py_cnt][px_cnt][f] = sum_array
            #                 pass
            # else:
            #     self.InputX_grp = int(self.cin_wr_real / self.Bias_length)
            #     for py_cnt in range(self.Input_fm_Py):
            #         for px_cnt in range(self.Input_fm_Px):
            #             for i in range(self.InputX_grp):
            #                 for f in range(self.Bias_length):
            #                     self.result[py_cnt][px_cnt][f] = self.Bias_array[f]     # ？？？左边的f对吗
            #                     for cnt in range(self.array_num):
            #                         self.result[py_cnt][px_cnt][f] += self.InputX_array[cnt][py_cnt][px_cnt][self.Bias_length * i + f]
            #                         if self.result[py_cnt][px_cnt][f] >= 0x7fffffff:
            #                             self.result[py_cnt][px_cnt][f] = 0x7fffffff
            #                         elif self.result[py_cnt][px_cnt][f] <= -0x80000000:
            #                             self.result[py_cnt][px_cnt][f] = -0x80000000
            #                     pass

        if self.avg_pooling_en:
            self.Write_V_length = self.Output_fm_Oy * self.Output_fm_Ox * self.cin_wr_real
        else:
            self.Write_V_length = self.Input_fm_Px * self.Input_fm_Py * self.cin_wr_real
        # --------------------------------------------------example start--------------------------------------------------
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

        # --------------------------------------------------example   end--------------------------------------------------
        pass

    def cal_para(self):
        self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        self.Py_real = self.Input_fm_Py + self.pad_top + self.pad_down
        self.Bias_grp = math.ceil(self.Bias_length / 32)
        self.array_num = self.pooling_Kx * self.pooling_Ky
        self.Output_fm_Ox = int((self.Px_real - self.pooling_Kx) / self.pooling_Sx) + 1
        self.Output_fm_Oy = int((self.Py_real - self.pooling_Ky) / self.pooling_Sy) + 1

        if self.InA_type == 0:
            self.Km_num = math.ceil(self.cin / 8)
            self.cin_wr_real = self.Km_num * 8
        elif self.InA_type == 3:
            self.Km_num = math.ceil(self.cin / 128)
            self.cin_wr_real = self.Km_num * 128
        else:
            self.Km_num = math.ceil(self.cin / 32)
            self.cin_wr_real = self.Km_num * 32

        if self.avg_pooling_en:
            # self.Load_Bias = 0
            self.array_num = 0
            self.L1_num = self.pooling_Kx - 1
            self.L2_num = self.pooling_Ky - 1
            self.L3_num = self.Km_num - 1
            self.L4_num = self.Output_fm_Ox - 1
            self.L5_num = self.Output_fm_Oy - 1
        else:
            self.L1_num = self.pooling_Kx - 1
            self.L2_num = self.pooling_Ky - 1
            self.L3_num = self.Km_num - 1
            self.L4_num = self.Input_fm_Px - 1
            self.L5_num = self.Input_fm_Py - 1
            self.Output_fm_Ox = self.Input_fm_Px
            self.Output_fm_Oy = self.Input_fm_Py

        pass

        if self.Load_Bias == 2 or self.Load_Bias == 3:
            self.constant_b = 0
        else:
            pass

        if self.avg_pooling_en:
            one_row_in_mem = self.Input_fm_Px * (self.Km_num * 2)
            self.Addr_InA_L1_step = self.Km_num * 2
            self.Addr_InA_L2_step = -(self.pooling_Kx - 1) * (self.Km_num * 2) + one_row_in_mem
            self.Addr_InA_L3_step = -(self.pooling_Kx - 1) * (self.Km_num * 2) - (self.pooling_Ky - 1) * one_row_in_mem + 2
            self.Addr_InA_L4_step = self.Addr_InA_L3_step + (self.pooling_Sx - 1) * (self.Km_num * 2)
            self.Addr_InA_L5_step = self.Addr_InA_L3_step - (self.Km_num * 2) - (
                    self.Output_fm_Ox - 1) * self.pooling_Sx * (self.Km_num * 2) + self.pooling_Sy * one_row_in_mem
        else:
            self.Addr_InA_L1_step = self.Input_fm_Px * self.Input_fm_Py * (self.Km_num * 2)
            self.Addr_InA_L2_step = self.Input_fm_Px * self.Input_fm_Py * (self.Km_num * 2)
            self.Addr_InA_L3_step = -self.Km_num * 2 * (self.array_num - 1) * self.Input_fm_Px * self.Input_fm_Py + 2
            self.Addr_InA_L4_step = self.Addr_InA_L3_step
            self.Addr_InA_L5_step = self.Addr_InA_L3_step

        if self.avg_pooling_en and self.pad_on:
            self.Addr_start_offset = self.pad_left * self.Addr_InA_L1_step + self.pad_top * (self.pooling_Kx - 1) * self.Addr_InA_L1_step + self.pad_top * self.Addr_InA_L2_step
        else:
            self.Addr_start_offset = 0

        pass

        if self.avg_pooling_en:
            self.Read_X_length = self.Input_fm_Px * self.Input_fm_Py * self.Km_num * 8
            self.Write_V_length = self.Output_fm_Oy * self.Output_fm_Ox * self.cin_wr_real
        else:
            self.Read_X_length = self.array_num * self.Input_fm_Px * self.Input_fm_Py * self.Km_num * 8
            self.Write_V_length = self.Input_fm_Px * self.Input_fm_Py * self.cin_wr_real

        self.Read_Bias_length = self.Bias_length

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PI_A.PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('PI_A.Reset_Addr_A = ' + str(self.Reset_Addr_A) + '\n')
            f.write('PI_A.Reset_Addr_V = ' + str(self.Reset_Addr_V) + '\n')
            if self.pad_on:
                _addr = ((self.Addr_InA_base >> 2) - self.Addr_start_offset)
                if _addr < 0:
                    _addr += 0x8000
                    pass
                f.write('PI_A.Addr_InA_base = ' + str(hex(_addr)) + '\n')
            else:
                f.write('PI_A.Addr_InA_base = ' + str(hex(self.Addr_InA_base >> 2) + '\n'))
            f.write('PI_A.InA_type = ' + str(self.InA_type) + '\n')
            f.write('PI_A.Load_Bias = ' + str(self.Load_Bias) + '\n')
            f.write('PI_A.Addr_V_base = ' + str(hex(self.Addr_V_base >> 2)) + '\n')
            f.write('PI_A.Addr_InA_end = ' + str(hex(((self.Addr_InA_base + self.Read_X_length) >> 2) + self.Addr_start_offset - 1) + '\n'))
            f.write('PI_A.Addr_V_end = ' + str(hex((((self.Addr_V_base + self.Write_V_length) >> 3) - 1) << 1) + '\n'))
            f.write('PI_A.Addr_Bias_base = ' + str(hex(self.Addr_Bias_base >> 2)) + ';\n')
            f.write('PI_A.Load_Bias = ' + str(self.Load_Bias) + '\n')
            f.write('PI_A.L1_num = ' + str(self.L1_num) + '\n')
            f.write('PI_A.L2_num = ' + str(self.L2_num) + '\n')
            f.write('PI_A.L3_num = ' + str(self.L3_num) + '\n')
            f.write('PI_A.L4_num = ' + str(self.L4_num) + '\n')
            f.write('PI_A.L5_num = ' + str(self.L5_num) + '\n')
            f.write('PI_A.Addr_InA_L1_step = ' + str(self.Addr_InA_L1_step) + '\n')
            f.write('PI_A.Addr_InA_L2_step = ' + str(self.Addr_InA_L2_step) + '\n')
            f.write('PI_A.Addr_InA_L3_step = ' + str(self.Addr_InA_L3_step) + '\n')
            f.write('PI_A.Addr_InA_L4_step = ' + str(self.Addr_InA_L4_step) + '\n')
            f.write('PI_A.Addr_InA_L5_step = ' + str(self.Addr_InA_L5_step) + '\n')
            f.write('PI_A.Sx = ' + str(self.pooling_Sx) + '\n')
            f.write('PI_A.Sy = ' + str(self.pooling_Sy) + '\n')
            f.write('PI_A.pad_top = ' + str(self.pad_top) + '\n')
            f.write('PI_A.pad_down = ' + str(self.pad_down) + '\n')
            f.write('PI_A.pad_left = ' + str(self.pad_left) + '\n')
            f.write('PI_A.pad_right = ' + str(self.pad_right) + '\n')
            f.write('PI_A.constant_b = ' + str(self.constant_b) + '\n')
            f.write('******************************' + '\n')
            f.write('Input_fm_Px = ' + str(self.Input_fm_Px) + '\n')
            f.write('Input_fm_Py = ' + str(self.Input_fm_Py) + '\n')
            f.write('Output_fm_Ox = ' + str(self.Output_fm_Ox) + '\n')
            f.write('Output_fm_Oy = ' + str(self.Output_fm_Oy) + '\n')
            f.write('array_num = ' + str(self.array_num) + '\n')
            f.write('Read_X_length = ' + str(self.Read_X_length) + '\n')
            f.write('Read_Bias_length = ' + str(self.Read_Bias_length) + '\n')
            f.write('Write_V_length = ' + str(self.Write_V_length) + '\n')

    def save_results(self, SIMPATH, TBNAME, t):
        path = SIMPATH + TBNAME

        if self.avg_pooling_en and self.pad_on:
            with open(path + '/InputX_padding.txt', 'w') as f:
                if self.InA_type == 0:
                    for NOY_cnt in range(self.Py_real):
                        for NOX_cnt in range(self.Px_real):
                            for i in range(self.cin_wr_real):
                                data = []
                                final_string = ''
                                final_string = hex_to_string(self.Input_fm_real[NOY_cnt, NOX_cnt, i],
                                                             width=8) + final_string
                                f.write(final_string)
                                f.write('\n')
                                pass
                elif self.InA_type == 3:
                    for NOY_cnt in range(self.Py_real):
                        for NOX_cnt in range(self.Px_real):
                            for j in range(self.cin_wr_real // 16):
                                final_string = ''
                                for k in range(8):
                                    data_4b = ((self.Input_fm_real[NOY_cnt][NOX_cnt][j * 16 + k * 2 + 1] & 0x3) << 2) | (
                                            self.Input_fm_real[NOY_cnt][NOX_cnt][j * 16 + k * 2] & 0x3)
                                    final_string = hex_to_string(data_4b, width=1) + final_string
                                f.write(final_string)
                                f.write('\n')
                                pass
                else:
                    for NOY_cnt in range(self.Py_real):
                        for NOX_cnt in range(self.Px_real):
                            for i in range(self.cin_wr_real // 4):
                                data = []
                                final_string = ''
                                for j in range(4):
                                    final_string = hex_to_string(
                                        self.Input_fm_real[NOY_cnt, NOX_cnt, 4 * i + j]) + final_string
                                f.write(final_string)
                                f.write('\n')
                                pass
        else:
            pass

        with open(path + '/Data_VOUT.txt', 'w') as f:
            if self.avg_pooling_en:
                for NOY_cnt in range(self.Output_fm_Oy):
                    for NOX_cnt in range(self.Output_fm_Ox):
                        for i in range(self.cin_wr_real):
                            final_string = hex_to_string(self.result[NOY_cnt, NOX_cnt, i], width=8)
                            f.write(final_string)
                            f.write('\n')
                            pass
            else:
                for py_cnt in range(self.Input_fm_Py):
                    for px_cnt in range(self.Input_fm_Px):
                        for i in range(self.cin_wr_real):
                            final_string = hex_to_string(self.result[py_cnt, px_cnt, i], width=8)
                            f.write(final_string)
                            f.write('\n')
                            pass
            pass

        with open(path + '/Data_VOUT_dec.txt', 'w') as f:
            if self.avg_pooling_en:
                for NOY_cnt in range(self.Output_fm_Oy):
                    for NOX_cnt in range(self.Output_fm_Ox):
                        for CIN_cnt in range(self.cin_wr_real):
                            final_value = self.result[NOY_cnt, NOX_cnt, CIN_cnt]
                            f.write(str(final_value))
                            f.write('\n')
                            pass
            else:
                for py_cnt in range(self.Input_fm_Py):
                    for px_cnt in range(self.Input_fm_Px):
                        for i in range(self.cin_wr_real):
                            final_value = self.result[py_cnt, px_cnt, i]
                            f.write(str(final_value))
                            f.write('\n')
                            pass
