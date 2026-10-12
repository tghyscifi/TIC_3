import numpy as np
import math
from primitive.basic_operations import BasicOperations

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_04_Axon(Primitive):
    def __init__(self):
        super().__init__()

        self.InA_type = 1
        self.InB_type = 1
        self.Load_Bias = 0
        self.cin = 0
        self.cout = 0
        # self.constant_b = 0
        self.PIC = 0x04
        self.Reset_Addr_A = 1
        self.Reset_Addr_V = 1
        self.Addr_InA_base = 0x00
        self.Addr_InB_base = 0x0000
        self.Addr_Bias_base = 0x00
        self.Addr_V_base = 0x00
        self.Addr_InA_end = 0x00
        self.Addr_V_end = 0x00
        self.A2S2_mode = 0

        self.Km_num = None

        self.total_core_num = 1
        self.areas_num_in_core = 1

    def __str__(self):
        return "04(MLP)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)
        _retLsit = []
        if self.InA_type == 1 or self.InA_type == 2:
            self.Km_num = np.ceil(self.cin / 16).astype(int)
            self.length_in_equal = self.Km_num * 16
            num_in_4B = 4
        else:
            self.Km_num = np.ceil(self.cin / 64).astype(int)
            self.length_in_equal = self.Km_num * 64
            num_in_4B = 16
            pass

        InputX_size = self.cin
        InputX_size_equal = self.length_in_equal

        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)

        if self.InA_type == 2:
            InputX_array = data_generator.random_array_u8(InputX_size).astype(np.int64)
        elif self.InA_type == 3:
            InputX_array = data_generator.random_array_2(InputX_size).astype(np.int64)
        else:
            InputX_array = data_generator.random_array_8(InputX_size).astype(np.int64)

        self.InputX_array[:self.cin] = InputX_array[:]

        _data = []
        for i in range(self.length_in_equal // num_in_4B):
            _tmp = []
            for j in range(num_in_4B):
                _tmp.append(int(self.InputX_array[num_in_4B * i + j]))
                pass
            _data.append(_tmp)
            pass
        _retLsit.append(_data)

        if self.InB_type == 1 or self.InB_type == 2:
            self.w_grp_num = np.ceil(self.cout / 32).astype(int)
            self.w_wr_real = self.w_grp_num * 32
        else:
            self.w_grp_num = np.ceil(self.cout / 128).astype(int)
            self.w_wr_real = self.w_grp_num * 128
            pass

        weight_size = (self.cin, self.cout)
        weight_size_equal = (self.cin, self.w_wr_real)

        self.weight = np.zeros(weight_size_equal).astype(np.int64)

        if self.InB_type == 2:
            weight = data_generator.random_array_u8(weight_size).astype(np.int64)
            num_in_4B = 4
        elif self.InB_type == 3:
            weight = data_generator.random_array_2(weight_size).astype(np.int64)
            num_in_4B = 16
        else:
            weight = data_generator.random_array_8(weight_size).astype(np.int64)
            num_in_4B = 4

        self.weight[:, :self.cout] = weight[:, :]
        percin_in_4B = 8
        _data = []
        for w_grp_cnt in range(self.w_grp_num):
            for cin in range(self.cin):
                for i in range(8):
                    _tmp = []
                    for j in range(num_in_4B):
                        _tmp.append(int(self.weight[cin][num_in_4B * i + j + percin_in_4B * num_in_4B * w_grp_cnt]))
                        pass
                    _data.append(_tmp)
                    pass
        _retLsit.append(_data)
        self.Bias_length = self.w_wr_real

        if self.Load_Bias == 0 or self.Load_Bias == 1:
            # self.constant_b = data_generator.random_constant_32()
            self.constant_b = 0
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
            pass
        # _retLsit.append(self)
        blocks = [{'name': "P04_input_X",
                   'start': self.Addr_InA_base,
                   'data': _retLsit[0],
                   'mode': 0},
                  {'name': "P04_weight",
                   'start': self.Addr_InB_base,
                   'data': _retLsit[1],
                   'mode': 0}]
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            blocks.append({'name': "P04_bias",
                           'start': self.Addr_Bias_base,
                           'data': _retLsit[2],
                           'mode': 0})
        self.memory_blocks = blocks
        return _retLsit
        pass

    def getInfoList(self) -> list:
        if self.InA_type == 1 or self.InA_type == 2:
            self.Km_num = np.ceil(self.cin / 16).astype(int)
            self.length_in_equal = self.Km_num * 16
        else:
            self.Km_num = np.ceil(self.cin / 64).astype(int)
            self.length_in_equal = self.Km_num * 64
            pass

        if self.InB_type == 1 or self.InB_type == 2:
            self.w_grp_num = np.ceil(self.cout / 32).astype(int)
            self.w_wr_real = self.w_grp_num * 32
        else:
            self.w_grp_num = np.ceil(self.cout / 128).astype(int)
            self.w_wr_real = self.w_grp_num * 128
            pass

        self.Read_X_length = self.Km_num * 4
        self.Read_weight_length = self.cin * self.w_grp_num * 8
        self.Read_Bias_length = self.w_wr_real

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
        for i in range(self.w_wr_real):
            _tmp = []
            _tmp.append(self.result[i])
            _Vresult.append(_tmp)
            pass  # for i in range(self.cout_wr_real)
        return _Vresult
        pass  # func convertPrim2Mem

    def convertMem2Prim(self, inputList: list):
        if self.InA_type == 1 or self.InA_type == 2:
            self.Km_num = np.ceil(self.cin / 16).astype(int)
            self.length_in_equal = self.Km_num * 16
            num_in_4B = 4
        else:
            self.Km_num = np.ceil(self.cin / 64).astype(int)
            self.length_in_equal = self.Km_num * 64
            num_in_4B = 16
            pass

        InputX_size_equal = self.length_in_equal
        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)

        for i in range(self.length_in_equal // num_in_4B):
            for j in range(num_in_4B):
                self.InputX_array[num_in_4B * i + j] = inputList[0][i][j]
            pass

        if self.InB_type == 1 or self.InB_type == 2:
            self.w_grp_num = np.ceil(self.cout / 32).astype(int)
            self.w_wr_real = self.w_grp_num * 32
            cout_in_grp = 32
            num_in_4B = 4

        else:
            self.w_grp_num = np.ceil(self.cout / 128).astype(int)
            self.w_wr_real = self.w_grp_num * 128
            cout_in_grp = 128
            num_in_4B = 16
            pass

        per32cout_in_4B = cout_in_grp / num_in_4B
        perGrp_in_4B = self.cin * per32cout_in_4B

        weight_size_equal = (self.cin, self.w_wr_real)

        self.weight = np.zeros(weight_size_equal).astype(np.int64)

        for w_grp_cnt in range(self.w_grp_num):
            for cin in range(self.cin):
                for i in range(cout_in_grp // num_in_4B):
                    for j in range(num_in_4B):
                        self.weight[cin][num_in_4B * i + j + w_grp_cnt * cout_in_grp] = int(inputList[1][int(w_grp_cnt * perGrp_in_4B + cin * per32cout_in_4B + i)][j])

        self.Bias_size = self.w_wr_real
        self.Bias_array = np.zeros(self.Bias_size).astype(np.int64)
        # Bias转化：一位数组
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            for i in range(self.Bias_size):
                self.Bias_array[i] = int(inputList[2][i][0])
        else:
            pass
        pass  # func convertMem2Prim

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)

        if self.Load_Bias == 0 or self.Load_Bias == 1:
            bias = np.ones(self.Bias_size).astype(np.int64) * self.constant_b
        else:
            bias = self.Bias_array

        self.result = BasicOperations.linear(x=self.InputX_array, weight=self.weight, bias=bias)

        # self.result = np.zeros(self.w_wr_real).astype(np.int64)
        #
        # if self.Load_Bias == 0 or self.Load_Bias == 1:
        #     for r in range(self.w_wr_real):
        #         self.result[r] = self.constant_b
        #         for i in range(self.cin):
        #             self.result[r] += self.weight[i][r] * self.InputX_array[i]
        #             if self.result[r] >= 0x7fffffff:
        #                 self.result[r] = 0x7fffffff
        #             elif self.result[r] <= -0x80000000:
        #                 self.result[r] = -0x80000000
        #
        # else:
        #     for r in range(self.w_wr_real):
        #         self.result[r] = self.Bias_array[r]
        #         for i in range(self.cin):
        #             self.result[r] += self.weight[i][r] * self.InputX_array[i]
        #             if self.result[r] >= 0x7fffffff:
        #                 self.result[r] = 0x7fffffff
        #             elif self.result[r] <= -0x80000000:
        #                 self.result[r] = -0x80000000

        self.Write_V_length = self.w_wr_real
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

    def cal_para(self):

        if self.InA_type != 3 and self.InB_type == 1:
            self.L0_num = math.ceil(self.cin / 16) - 1
            self.L3_num = math.ceil(self.cout / 32) - 1
            self.L0_num_in_last_row = self.cin - self.L0_num * 16 - 1
        elif self.InA_type != 3 and self.InB_type == 3:
            self.L0_num = math.ceil(self.cin / 16) - 1
            self.L3_num = math.ceil(self.cout / 128) - 1
            self.L0_num_in_last_row = self.cin - self.L0_num * 16 - 1
        elif self.InA_type == 3 and self.InB_type != 3:
            self.L0_num = math.ceil(self.cin / 64) - 1
            self.L3_num = math.ceil(self.cout / 32) - 1
            self.L0_num_in_last_row = self.cin - self.L0_num * 64 - 1
        elif self.InA_type == 3 and self.InB_type == 3:
            self.L0_num = math.ceil(self.cin / 64) - 1
            self.L3_num = math.ceil(self.cout / 128) - 1
            self.L0_num_in_last_row = self.cin - self.L0_num * 64 - 1
        else:  # 默认按照输入X和输入W都是int8来计算
            self.L0_num = math.ceil(self.cin / 16) - 1
            self.L3_num = math.ceil(self.cout / 32) - 1
            self.L0_num_in_last_row = self.cin - self.L0_num * 16 - 1

        self.Addr_InA_L3_step = 0 - self.L0_num

        self.Read_X_length = self.Km_num * 4
        self.Read_weight_length = self.cin * self.w_grp_num * 8
        self.Read_Bias_length = self.Bias_length
        self.Write_V_length = self.w_wr_real
        if self.Load_Bias == 0 or self.Load_Bias == 1:
            self.Read_Bias_length = 0
        else:
            self.Read_Bias_length = self.Bias_length
            self.constant_b = 0

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PI_A.PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('PI_A.Reset_Addr_A = ' + str(self.Reset_Addr_A) + '\n')
            f.write('PI_A.Reset_Addr_V = ' + str(self.Reset_Addr_V) + '\n')
            f.write('PI_A.Addr_InA_base = ' + str(hex(self.Addr_InA_base >> 2) + '\n'))
            f.write('PI_A.InA_type = ' + str(self.InA_type) + '\n')
            f.write('PI_A.Addr_InB_base = ' + str(hex(self.Addr_InB_base >> 2)) + '\n')
            f.write('PI_A.InB_type = ' + str(self.InB_type) + '\n')
            f.write('PI_A.Addr_Bias_base = ' + str(hex(self.Addr_Bias_base >> 2)) + ';\n')
            f.write('PI_A.Load_Bias = ' + str(self.Load_Bias) + '\n')
            f.write('PI_A.Addr_V_base = ' + str(hex(self.Addr_V_base >> 2)) + '\n')
            f.write('PI_A.Addr_InA_end = ' + str(
                hex(((self.Addr_InA_base + self.Read_X_length) >> 2) - 1) + '\n'))
            f.write('PI_A.Addr_V_end = ' + str(hex((((self.Addr_V_base + self.Write_V_length) >> 3) - 1) << 1) + '\n'))
            f.write('PI_A.Load_Bias = ' + str(self.Load_Bias) + '\n')
            f.write('PI_A.L0_num = ' + str(self.L0_num) + '\n')
            f.write('PI_A.L3_num = ' + str(self.L3_num) + '\n')
            f.write('PI_A.L0_num_in_last_row = ' + str(self.L0_num_in_last_row) + '\n')
            f.write('PI_A.Addr_InA_L3_step = ' + str(self.Addr_InA_L3_step) + '\n')
            f.write('constant_b = ' + str(self.constant_b) + '\n')
            f.write('******************************' + '\n')
            f.write('Read_X_length = ' + str(self.Read_X_length) + '\n')
            f.write('Read_Bias_length = ' + str(self.Read_Bias_length) + '\n')
            f.write('Read_weight_length = ' + str(self.Read_weight_length) + '\n')
            f.write('Write_V_length = ' + str(self.Write_V_length) + '\n')

    def save_results(self, SIMPATH, TBNAME, t):
        path = SIMPATH + TBNAME

        with open(path + '/Data_VOUT.txt', 'w') as f:
            for i in range(self.w_wr_real):
                final_string = hex_to_string(self.result[i], width=8)
                f.write(final_string)
                f.write('\n')
                pass
        with open(path + '/Data_VOUT_dec.txt', 'w') as f:
            for i in range(self.w_wr_real):
                final_value = self.result[i]
                f.write(str(final_value))
                f.write('\n')
                pass
