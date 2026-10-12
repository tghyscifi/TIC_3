import numpy as np
import copy

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_07_LUT(Primitive):
    def __init__(self):
        super().__init__()

        self.neuron_real_num = 0
        self.group_num = 0
        self.PIC = 0x07
        self.PIC_Mode = 0x00
        self.reset_Addr_X = 1
        self.reset_Addr_Y = 1
        self.Row_ck_on = 0
        self.Addr_X_Start = 0x00
        self.X_type = 1  # 0:int32 1/2/3 int8
        self.Addr_X_end = 0x00
        self.Addr_Start_out = 0x00
        self.Y_type = 1
        self.Addr_Y_end = 0x00
        self.Addr_LUT_Start = 0x00
        self.LUT_DW = 0  # 0:4b   1:8b    2:12b   3:16b
        self.X_cut_start = 0
        self.in_row_max = 0

        self.total_core_num = 1
        self.areas_num_in_core = 1
        self.mem_sel = 0

    def __str__(self):
        return "07(LUT)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)
        _retLsit = []
        if self.X_type == 0:
            self.neu_num = np.ceil(self.neuron_real_num / 4).astype(int)
            self.neuron_real_num_wr = self.neu_num * 4
            self.neuron_length = self.neuron_real_num_wr * self.group_num
            num_in_4B = 1
        else:
            self.neu_num = np.ceil(self.neuron_real_num / 16).astype(int)
            self.neuron_real_num_wr = self.neu_num * 16
            self.neuron_length = self.neuron_real_num_wr * self.group_num
            num_in_4B = 4

        X_size = (self.group_num, self.neuron_real_num)
        X_size_equal = (self.group_num, self.neuron_real_num_wr)
        self.X_array = np.zeros(X_size_equal).astype(np.int64)

        if self.X_type == 0:
            X_array = data_generator.random_array_32(X_size).astype(np.int64)
        else:
            X_array = data_generator.random_array_8(X_size).astype(np.int64)

        self.X_array[:, :self.neuron_real_num] = X_array[:, :]

        _data = []
        for k in range(self.group_num):
            for i in range(self.neuron_real_num_wr // num_in_4B):
                _tmp = []
                for j in range(num_in_4B):
                    _tmp.append(int(self.X_array[k][num_in_4B * i + j]))
                    pass
                _data.append(_tmp)
                pass

        _retLsit.append(_data)
        if self.LUT_DW == 0:  # 4b地址位宽
            self.LUT_length = 16
        elif self.LUT_DW == 1:
            self.LUT_length = 256
        elif self.LUT_DW == 2:
            self.LUT_length = 2 ** 12
        elif self.LUT_DW == 3:
            self.LUT_length = 2 ** 16
        else:
            self.LUT_length = 256

        if self.Y_type == 0:
            self.LUT_array = data_generator.random_array_32(self.LUT_length).astype(np.int64)
        else:
            self.LUT_array = data_generator.random_array_8(self.LUT_length).astype(np.int64)
        if self.Y_type == 0:
            num_in_4B = 1
        else:
            num_in_4B = 4

        _data = []
        for i in range(self.LUT_length // num_in_4B):
            _tmp = []
            for j in range(num_in_4B):
                _tmp.append(int(self.LUT_array[num_in_4B * i + j]))
                pass
            _data.append(_tmp)
            pass
        _retLsit.append(_data)
        # _retLsit.append(self)
        soma2_mark = ''
        if hasattr(self, 'soma2') and self.soma2 is True:
            soma2_mark = '_2'
        self.memory_blocks = [
            {'name': 'P07' + soma2_mark + '_input_x1',
             'start': self.Addr_X_Start,
             'data': _retLsit[0],
             'mode': 0},
            {'name': 'P07' + soma2_mark + '_LUT',
             'start': self.Addr_LUT_Start,
             'data': _retLsit[1],
             'mode': 0}
        ]
        return _retLsit

    def getInfoList(self) -> list:
        if self.X_type == 0:
            self.neu_num = np.ceil(self.neuron_real_num / 4).astype(int)
            self.neuron_real_num_wr = self.neu_num * 4
            self.neuron_length = self.neuron_real_num_wr * self.group_num
        else:
            self.neu_num = np.ceil(self.neuron_real_num / 16).astype(int)
            self.neuron_real_num_wr = self.neu_num * 16
            self.neuron_length = self.neuron_real_num_wr * self.group_num

        self.Read_X_length = self.neu_num * self.group_num * 4

        if self.LUT_DW == 0:  # 4b地址位宽
            self.LUT_length = 16
        elif self.LUT_DW == 1:
            self.LUT_length = 256
        elif self.LUT_DW == 2:
            self.LUT_length = 2 ** 12
        elif self.LUT_DW == 3:
            self.LUT_length = 2 ** 16
        else:
            self.LUT_length = 256

        if self.Y_type == 0:
            self.Read_LUT_length = self.LUT_length
        else:
            self.Read_LUT_length = self.LUT_length // 4

        _infoList = []
        _X = {
            "start": self.Addr_X_Start,
            "length": self.Read_X_length,
            "type": self.X_type
        }
        _infoList.append(_X)
        _LUT = {
            "start": self.Addr_LUT_Start,
            "length": self.Read_LUT_length,
            "type": self.Y_type
        }
        _infoList.append(_LUT)
        return _infoList
        pass  # func getInfoList

    def convertPrim2Mem(self, inputList: list) -> list:
        _Vresult = []
        if self.X_type == 0 and self.Y_type == 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 4).astype(int)) * 4
        else:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16

        if self.Y_type == 0:
            for i in range(self.group_num):
                for j in range(self.Y_neuron_num_wr):
                    # for j in range(self.neu_num * 4):
                    _tmp = []
                    _tmp.append(self.result[i, j])
                    _Vresult.append(_tmp)
                    pass
                pass
        else:
            for k in range(self.group_num):
                for i in range(self.Y_neuron_num_wr // 4):
                    _tmp = []
                    for j in range(4):
                        _tmp.append(self.result[k, 4 * i + j])
                        pass
                    _Vresult.append(_tmp)
                    pass
                pass
            pass
        return _Vresult
        pass  # func convertPrim2Mem

    def convertMem2Prim(self, inputList: list):
        if self.X_type == 0:
            self.neu_num = np.ceil(self.neuron_real_num / 4).astype(int)
            self.neuron_real_num_wr = self.neu_num * 4
            self.neuron_length = self.neuron_real_num_wr * self.group_num
            num_in_4B = 1
        else:
            self.neu_num = np.ceil(self.neuron_real_num / 16).astype(int)
            self.neuron_real_num_wr = self.neu_num * 16
            self.neuron_length = self.neuron_real_num_wr * self.group_num
            num_in_4B = 4

        self.X_size = (self.group_num, self.neuron_real_num_wr)
        self.X_array = np.zeros(self.X_size).astype(np.int64)

        pergrp_in_4B = self.neu_num * 4

        # print(inputList[0])
        for k in range(self.group_num):
            for i in range(self.neuron_real_num_wr // num_in_4B):
                for j in range(num_in_4B):
                    self.X_array[k][num_in_4B * i + j] = inputList[0][int(k * pergrp_in_4B + i)][j]
                    pass

        if self.LUT_DW == 0:  # 4b地址位宽
            self.LUT_length = 16
        elif self.LUT_DW == 1:
            self.LUT_length = 256
        elif self.LUT_DW == 2:
            self.LUT_length = 2 ** 12
        elif self.LUT_DW == 3:
            self.LUT_length = 2 ** 16
        else:
            self.LUT_length = 256

        self.LUT_array = np.zeros(self.LUT_length).astype(np.int64)

        if self.Y_type == 0:
            num_in_4B = 1
        else:
            num_in_4B = 4

        for i in range(self.LUT_length // num_in_4B):
            for j in range(num_in_4B):
                self.LUT_array[num_in_4B * i + j] = inputList[1][i][j]
                pass

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)

        self.X_cut_start_real = self.X_cut_start * 2

        if self.X_type == 0:
            self.neu_num = np.ceil(self.neuron_real_num / 4).astype(int)
            self.neuron_real_num_wr = self.neu_num * 4
            self.neuron_length = self.neuron_real_num_wr * self.group_num
        else:
            self.neu_num = np.ceil(self.neuron_real_num / 16).astype(int)
            self.neuron_real_num_wr = self.neu_num * 16
            self.neuron_length = self.neuron_real_num_wr * self.group_num

        self.convert_addr_size = (self.group_num, self.neuron_real_num_wr)
        self.convert_addr = np.zeros(self.convert_addr_size).astype(np.int64)
        if self.LUT_DW == 3:
            if self.X_type == 0:
                for i in range(self.group_num):
                    for j in range(self.neuron_real_num_wr):
                        self.convert_addr[i][j] = np.floor(self.X_array[i][j] / (2 ** self.X_cut_start_real))
                        if self.convert_addr[i][j] > 0x7fff:
                            self.convert_addr[i][j] = 0x7fff
                        elif self.convert_addr[i][j] < -0x8000:
                            self.convert_addr[i][j] = -0x8000
                        else:
                            pass
            else:
                pass
        elif self.LUT_DW == 2:
            if self.X_type == 0:
                for i in range(self.group_num):
                    for j in range(self.neuron_real_num_wr):
                        self.convert_addr[i][j] = np.floor(self.X_array[i][j] / (2 ** self.X_cut_start_real))
                        if self.convert_addr[i][j] > 2047:
                            self.convert_addr[i][j] = 2047
                        elif self.convert_addr[i][j] < -2048:
                            self.convert_addr[i][j] = -2048
                        else:
                            pass
            else:
                pass
        elif self.LUT_DW == 1:
            if self.X_type == 0:
                for i in range(self.group_num):
                    for j in range(self.neuron_real_num_wr):
                        self.convert_addr[i][j] = np.floor(self.X_array[i][j] / (2 ** self.X_cut_start_real))
                        if self.convert_addr[i][j] > 127:
                            self.convert_addr[i][j] = 127
                        elif self.convert_addr[i][j] < -128:
                            self.convert_addr[i][j] = -128
                        else:
                            pass
            elif self.X_type == 1 or self.X_type == 2 or self.X_type == 3:
                self.convert_addr = self.X_array
            else:
                pass
        elif self.LUT_DW == 0:
            if self.X_type == 0 or self.X_type == 1 or self.X_type == 2 or self.X_type == 3:
                for i in range(self.group_num):
                    for j in range(self.neuron_real_num_wr):
                        self.convert_addr[i][j] = np.floor(self.X_array[i][j] / (2 ** self.X_cut_start_real))
                        # print(self.X_array[i][j], self.convert_addr[i][j])
                        if self.convert_addr[i][j] > 7:
                            self.convert_addr[i][j] = 7
                        elif self.convert_addr[i][j] < -8:
                            self.convert_addr[i][j] = -8
                        else:
                            pass
            else:
                pass
        else:
            pass

        if self.X_type == 0 and self.Y_type == 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 4).astype(int)) * 4
            self.Write_Y_length = self.Y_neuron_num_wr * self.group_num
        elif self.X_type == 0 and self.Y_type != 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16
            self.Write_Y_length = (self.Y_neuron_num_wr // 4) * self.group_num
        elif self.X_type != 0 and self.Y_type != 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16
            self.Write_Y_length = (self.Y_neuron_num_wr // 4) * self.group_num
        elif self.X_type != 0 and self.Y_type == 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16
            self.Write_Y_length = self.Y_neuron_num_wr * self.group_num

        self.result = np.zeros((self.group_num, self.Y_neuron_num_wr)).astype(np.int64)

        for i in range(self.group_num):
            for j in range(self.neuron_real_num_wr):
                tmp = self.convert_addr[i][j]
                # print(i, j, tmp)
                self.result[i][j] = self.LUT_array[tmp]
                pass

        # if self.Y_type == 0:
        #     self.Write_Y_length = self.Y_neuron_num_wr * self.group_num
        # else:
        #     self.Write_Y_length = (self.Y_neuron_num_wr // 4) * self.group_num

        _resultList = []
        _Vresult = self.convertPrim2Mem([self.result])  # 多维数组转化到Mem中的存储数组的4Bytes格式

        if self.Row_ck_on:
            inputRealLength = self.neu_num * self.in_row_max * 4  # 待确定
        else:
            inputRealLength = self.neu_num * self.group_num * 4  # 没行流水

        # print("inputRealLength =", inputRealLength)
        _V = {
            "Model": "Soma",
            "data": _Vresult,
            "startInMemory": self.Addr_Start_out,
            "lengthInMemory": self.Write_Y_length,
            "inputAddr": self.Addr_X_Start,
            "inputRealLength": inputRealLength,
            "mem_sel": self.mem_sel
        }
        # print(_V)
        _resultList.append(_V)
        return _resultList

        pass

    def cal_para(self):
        if self.X_type == 0:
            self.neu_num = np.ceil(self.neuron_real_num / 4).astype(int)
            self.neuron_real_num_wr = self.neu_num * 4
            self.neuron_length = self.neuron_real_num_wr * self.group_num
        else:
            self.neu_num = np.ceil(self.neuron_real_num / 16).astype(int)
            self.neuron_real_num_wr = self.neu_num * 16
            self.neuron_length = self.neuron_real_num_wr * self.group_num

        self.Read_X_length = self.neu_num * self.group_num * 4

        if self.LUT_DW == 0:  # 4b地址位宽
            self.LUT_length = 16
        elif self.LUT_DW == 1:
            self.LUT_length = 256
        elif self.LUT_DW == 2:
            self.LUT_length = 2 ** 12
        elif self.LUT_DW == 3:
            self.LUT_length = 2 ** 16
        else:
            self.LUT_length = 256

        if self.Y_type == 0:
            self.Read_LUT_length = self.LUT_length
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 4).astype(int)) * 4
        else:
            self.Read_LUT_length = self.LUT_length // 4
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16

        if self.X_type == 0 and self.Y_type == 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 4).astype(int)) * 4
            self.Write_Y_length = self.Y_neuron_num_wr * self.group_num
        elif self.X_type == 0 and self.Y_type != 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16
            self.Write_Y_length = (self.Y_neuron_num_wr // 4) * self.group_num
        elif self.X_type != 0 and self.Y_type != 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16
            self.Write_Y_length = (self.Y_neuron_num_wr // 4) * self.group_num
        elif self.X_type != 0 and self.Y_type == 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16
            self.Write_Y_length = self.Y_neuron_num_wr * self.group_num

        if self.Row_ck_on:
            self.Addr_X_end = self.Addr_X_Start + self.neu_num * self.in_row_max * 4
            pass
        else:
            self.Addr_X_end = self.Addr_X_Start + self.neu_num * self.group_num * 4
            pass

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PIS.PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('PIS.PIC_Mode = ' + str(hex(self.PIC_Mode)) + '\n')
            f.write('PIS.reset_Addr_X = ' + str(self.reset_Addr_X) + '\n')
            f.write('PIS.reset_Addr_Y = ' + str(self.reset_Addr_Y) + '\n')
            f.write('PIS.Row_ck_on = ' + str(self.Row_ck_on) + '\n')
            f.write('PIS.Addr_X_Start = ' + str(hex(self.Addr_X_Start >> 2) + '\n'))
            f.write('PIS.X_type = ' + str(self.X_type) + '\n')
            f.write('PIS.Addr_X_end = ' + str(hex(((self.Addr_X_Start + self.Read_X_length) >> 2) - 1) + '\n'))
            f.write('PIS.Addr_Y_Start = ' + str(hex(self.Addr_Start_out >> 2) + '\n'))
            f.write('PIS.Y_type = ' + str(self.Y_type) + '\n')
            f.write('PIS.Addr_Y_end = ' + str(hex(((self.Addr_Start_out + self.Write_Y_length) >> 2) - 1) + '\n'))
            f.write('PIS.neu_num = ' + str(self.neu_num - 1) + '\n')
            f.write('PIS.Y_num = ' + str(self.group_num - 1) + '\n')
            f.write('PIS.Addr_LUT_Start = ' + str(hex(self.Addr_LUT_Start >> 2) + '\n'))
            f.write('PIS.LUT_DW = ' + str(self.LUT_DW) + '\n')
            f.write('PIS.X_cut_start = ' + str(self.X_cut_start) + '\n')
            f.write('PIS.Row_length = ' + str(self.in_row_max) + '\n')
            f.write('******************************' + '\n')
            f.write('Read_X_length = ' + str(self.Read_X_length) + '\n')
            f.write('Read_LUT_length = ' + str(self.Read_LUT_length) + '\n')
            f.write('Write_Y_length = ' + str(self.Write_Y_length) + '\n')

        # with open(path + '/InputX.txt', 'w') as f:
        #     if self.X_type == 0:
        #         for i in range(self.group_num):
        #             for j in range(self.neu_num * 4):
        #                 data = []
        #                 final_string = ''
        #                 final_string = hex_to_string(self.X_array[i,j], width=8) + final_string
        #                 f.write(final_string)
        #                 f.write('\n')
        #                 pass
        #     else:
        #         for k in range(self.group_num):
        #             for i in range(self.neu_num * 4):
        #                 data = []
        #                 final_string = ''
        #                 for j in range(4):
        #                     final_string = hex_to_string(self.X_array[k, 4 * i + j]) + final_string
        #                 f.write(final_string)
        #                 f.write('\n')
        #                 pass
        #
        #     pass
        #
        # with open(path + '/InputX_dec.txt', 'w') as f:
        #     for i in range(self.group_num):
        #         for j in range(self.neuron_real_num):
        #             final_value = self.X_array[i, j]
        #             f.write(str(final_value))
        #             f.write('\n')
        #             pass
        #
        # with open(path + '/LUT.txt', 'w') as f:
        #     if self.Y_type == 0:
        #         for j in range(self.LUT_length):
        #             data = []
        #             final_string = ''
        #             final_string = hex_to_string(self.LUT_array[j], width=8) + final_string
        #             f.write(final_string)
        #             f.write('\n')
        #             pass
        #     else:
        #         for i in range(self.LUT_length // 4):
        #             data = []
        #             final_string = ''
        #             for j in range(4):
        #                 final_string = hex_to_string(self.LUT_array[4 * i + j]) + final_string
        #             f.write(final_string)
        #             f.write('\n')
        #             pass
        #
        #     pass
        #
        # with open(path + '/LUT_dec.txt', 'w') as f:
        #     for j in range(self.LUT_length):
        #         final_value = self.LUT_array[j]
        #         f.write(str(final_value))
        #         f.write('\n')
        #         pass

    def save_results(self, SIMPATH, TBNAME, t):
        if self.Y_type == 0:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 4).astype(int)) * 4
        else:
            self.Y_neuron_num_wr = (np.ceil(self.neuron_real_num / 16).astype(int)) * 16

        path = SIMPATH + TBNAME
        with open(path + '/convert_addr_dec.txt', 'w') as f:
            for i in range(self.group_num):
                for j in range(self.neuron_real_num_wr):
                    final_value = self.convert_addr[i, j]
                    f.write(str(final_value))
                    f.write('\n')
                    pass

        with open(path + '/Data_VOUT.txt', 'w') as f:
            if self.Y_type == 0:
                for i in range(self.group_num):
                    for j in range(self.Y_neuron_num_wr):
                        data = []
                        final_string = ''
                        final_string = hex_to_string(self.result[i, j], width=8) + final_string
                        f.write(final_string)
                        f.write('\n')
                        pass
            else:
                for k in range(self.group_num):
                    for i in range(self.Y_neuron_num_wr // 4):
                        data = []
                        final_string = ''
                        for j in range(4):
                            final_string = hex_to_string(self.result[k, 4 * i + j]) + final_string
                        f.write(final_string)
                        f.write('\n')
                        pass

            pass

        with open(path + '/Data_VOUT_dec.txt', 'w') as f:
            for i in range(self.group_num):
                for j in range(self.Y_neuron_num_wr):
                    final_value = self.result[i, j]
                    f.write(str(final_value))
                    f.write('\n')
                    pass
