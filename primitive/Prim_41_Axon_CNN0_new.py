import numpy as np
import math
from primitive.basic_operations import BasicOperations

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_41_Axon(Primitive):
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
        self.pad_top = 0
        self.pad_down = 0
        self.pad_left = 0
        self.pad_right = 0
        self.cin = 0
        self.cout = 0
        self.Bias_length = 0
        self.PIC = 0x41
        self.Reset_Addr_A = 1
        self.Reset_Addr_V = 1
        self.Addr_InA_base = 0x00
        self.Addr_InB_base = 0x0000
        self.Addr_Bias_base = 0x00
        self.Addr_V_base = 0x00
        self.Addr_InA_end = 0x00
        self.Addr_V_end = 0x00
        self.A2S2_mode = 0
        self.L4_num = 0
        self.L5_num = 0

        self.total_core_num = 1
        self.areas_num_in_core = 1

        # 得暂时加上这个了
        self.input = None

        # axon_delay=True,需要配置L4_num,L5_num,Addr_InA_base,Addr_InB_base,Addr_V_base
        self.axon_delay = False

    # _retLsit return List
    def __str__(self):
        return "41(CNN0)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)
        _retLsit = []
        if self.InA_type == 3:
            self.Km_num = np.ceil(self.cin / 64).astype(int)
            self.cin_wr_real = self.Km_num * 64
            num_in_4B = 16
        else:
            self.Km_num = np.ceil(self.cin / 16).astype(int)
            self.cin_wr_real = self.Km_num * 16
            num_in_4B = 4

        InputX_size = (self.Input_fm_Py, self.Input_fm_Px, self.cin)
        InputX_size_equal = (
            self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)

        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)

        if self.InA_type == 2:
            InputX_array = data_generator.random_array_u8(
                InputX_size).astype(np.int64)
        elif self.InA_type == 3:
            InputX_array = data_generator.random_array_2(
                InputX_size).astype(np.int64)
        else:
            InputX_array = data_generator.random_array_8(
                InputX_size).astype(np.int64)

        self.InputX_array[:, :, :self.cin] = InputX_array[:, :, :]

        _X = []
        for py in range(self.Input_fm_Py):
            for px in range(self.Input_fm_Px):
                for i in range(self.cin_wr_real // num_in_4B):
                    _tmp = []
                    for j in range(num_in_4B):
                        _tmp.append(
                            int(self.InputX_array[py][px][num_in_4B * i + j]))
                        # = inputList[0][int(py * perPy_in_4B + px * perPx_in_4B + i)][j]
                    pass
                    _X.append(_tmp)
                    pass
                pass
            pass
        _retLsit.append(_X)

        # weight->Mem
        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32

        weight_size = (self.conv_Ky, self.conv_Kx, self.cin, self.cout)
        weight_size_equal = (self.conv_Ky, self.conv_Kx,
                             self.cin, self.cout_wr_real)

        self.weight = np.zeros(weight_size_equal).astype(np.int64)

        if self.InB_type == 2:
            weight = data_generator.random_array_u8(
                weight_size).astype(np.int64)
        elif self.InB_type == 3:
            weight = data_generator.random_array_2(
                weight_size).astype(np.int64)
        else:
            weight = data_generator.random_array_8(
                weight_size).astype(np.int64)

        self.weight[:, :, :, :self.cout] = weight[:, :, :, :]
        if self.InB_type == 3:
            num_in_4B = 16
        else:
            num_in_4B = 4

        _weight = []
        for w_grp_cnt in range(self.w_grp_num):
            for ky in range(self.conv_Ky):
                for kx in range(self.conv_Kx):
                    for cin in range(self.cin):
                        for i in range(32 // num_in_4B):
                            _tmp = []
                            for j in range(num_in_4B):
                                # print(int(w_grp_cnt * perGrp_in_4B + ky * perKy_in_4B + kx * perKx_in_4B + cin * per32cout_in_4B + i))
                                _tmp.append(
                                    int(self.weight[ky][kx][cin][num_in_4B * i + j + 32 * w_grp_cnt]))
                                # inputList[1][int(w_grp_cnt * perGrp_in_4B + ky * perKy_in_4B + kx * perKx_in_4B + cin * per32cout_in_4B + i)][j] = \
                                pass
                            _weight.append(_tmp)
                            pass
                        pass
                    pass
                pass
            pass
        _retLsit.append(_weight)

        # bias->Mem
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            Bias_size = self.cout_wr_real
            self.Bias_array = data_generator.random_array_32(Bias_size)
            _bias = []
            # print(self.Bias_array.__len__())
            for i in range(self.Bias_array.__len__()):
                _tmp = []
                _tmp.append(int(self.Bias_array[i]))
                _bias.append(_tmp)
                pass
            _retLsit.append(_bias)
        else:
            pass

        # _retLsit.append(self)

        blocks = [{'name': "P41_input_X",
                   'start': self.Addr_InA_base,
                   'data': _retLsit[0],
                   'mode': 0},
                  {'name': "P41_weight",
                   'start': self.Addr_InB_base,
                   'data': _retLsit[1],
                   'mode': 0}]
        if self.Load_Bias == 2 or self.Load_Bias == 3:
            blocks.append({'name': "P41_bias",
                           'start': self.Addr_Bias_base,
                           'data': _retLsit[2],
                           'mode': 0})
        self.memory_blocks = blocks
        return _retLsit

    def getInfoList(self) -> list:
        if self.InA_type == 3:
            self.Km_num = np.ceil(self.cin / 64).astype(int)
            self.cin_wr_real = self.Km_num * 64
        else:
            self.Km_num = np.ceil(self.cin / 16).astype(int)
            self.cin_wr_real = self.Km_num * 16

        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32

        self.Read_X_length = self.Input_fm_Px * self.Input_fm_Py * self.Km_num * 4
        if self.InB_type == 3:
            self.Read_weight_length = int(
                np.ceil((self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 2) / 8) * 8)
        else:
            self.Read_weight_length = int(
                self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 8)

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
        if self.InA_type == 3:
            self.Km_num = np.ceil(self.cin / 64).astype(int)
            self.cin_wr_real = self.Km_num * 64
            num_in_4B = 16
        else:
            self.Km_num = np.ceil(self.cin / 16).astype(int)
            self.cin_wr_real = self.Km_num * 16
            num_in_4B = 4

        InputX_size_equal = (
            self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)

        self.InputX_array = np.zeros(InputX_size_equal).astype(np.int64)

        perPx_in_4B = self.Km_num * (16 / 4)  # 每个点的所有cin占多少的4bytes
        perPy_in_4B = self.Input_fm_Px * perPx_in_4B  # 输入图像的每一行的所有cin占多少的4bytes

        # print(type(np.int64(inputList[0][0][0])))
        # InputX转化：三维数组
        for py in range(self.Input_fm_Py):
            for px in range(self.Input_fm_Px):
                for i in range(self.cin_wr_real // num_in_4B):
                    for j in range(num_in_4B):
                        self.InputX_array[py][px][num_in_4B * i + j] = inputList[0][int(
                            py * perPy_in_4B + px * perPx_in_4B + i)][j]
                    pass

        # weight转化：四维数组
        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32
        weight_size_equal = (self.conv_Ky, self.conv_Kx,
                             self.cin, self.cout_wr_real)
        self.weight = np.zeros(weight_size_equal).astype(np.int64)
        if self.InB_type == 3:
            num_in_4B = 16
            per32cout_in_bytes = 8  # 32个cout的weight占几个1B
        else:
            num_in_4B = 4
            per32cout_in_bytes = 32  # 32个cout的weight占几个1B

        per32cout_in_4B = per32cout_in_bytes / 4  # 32个cout的weight占几个4B

        perKx_in_4B = self.cin * per32cout_in_4B  # kernel中一个点的每32个weight占的4B数
        perKy_in_4B = self.conv_Kx * perKx_in_4B  # kernel中的一整行的所有点的32个weight占的4B数
        perGrp_in_4B = self.conv_Ky * perKy_in_4B  # 每个grp的所有weight占的4B数

        for w_grp_cnt in range(self.w_grp_num):
            for ky in range(self.conv_Ky):
                for kx in range(self.conv_Kx):
                    for cin in range(self.cin):
                        for i in range(32 // num_in_4B):
                            for j in range(num_in_4B):
                                self.weight[ky][kx][cin][num_in_4B * i + j + 32 * w_grp_cnt] = \
                                    inputList[1][int(
                                        w_grp_cnt * perGrp_in_4B + ky * perKy_in_4B + kx * perKx_in_4B + cin * per32cout_in_4B + i)][j]

        # Bias转化：一位数组
        self.Bias_size = self.cout_wr_real
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
            bias = np.zeros(self.Bias_size).astype(np.int64)   # TODO 不支持常数bias ?
        else:
            bias = self.Bias_array

        result = BasicOperations.conv2d(x=self.InputX_array, weight=self.weight, bias=bias,
                                        kernel_size=(self.conv_Ky, self.conv_Kx),
                                        padding=(self.pad_top, self.pad_left, self.pad_down, self.pad_right),
                                        stride=(self.conv_Sy, self.conv_Sx), dilation=(self.conv_Ey, self.conv_Ex))

        self.result = np.array(result, dtype=np.int64).squeeze(0).transpose(1, 2, 0)

        # # 先对输入 padding 补零
        # self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        # self.Py_real = self.Input_fm_Py + self.pad_top + self.pad_down
        # self.Kx_real = (self.conv_Kx - 1) * self.conv_Ex + 1
        # self.Ky_real = (self.conv_Ky - 1) * self.conv_Ey + 1
        # self.Output_fm_Ox = int(
        #     (self.Px_real - self.Kx_real) / self.conv_Sx) + 1
        # self.Output_fm_Oy = int(
        #     (self.Py_real - self.Ky_real) / self.conv_Sy) + 1
        #
        # if self.pad_on:
        #     InputX_fm_size_real = (
        #         self.Py_real, self.Px_real, self.cin_wr_real)
        #     self.Input_fm_real = np.zeros(InputX_fm_size_real).astype(np.int64)
        #
        #     for NOY_cnt in range((self.pad_top - 1), (self.pad_top + self.Input_fm_Py - 1)):
        #         for NOX_cnt in range((self.pad_left - 1), (self.pad_left + self.Input_fm_Px - 1)):
        #             for CIN_cnt in range(self.cin_wr_real):
        #                 self.Input_fm_real[NOY_cnt + 1][NOX_cnt + 1][CIN_cnt] = \
        #                     self.InputX_array[NOY_cnt + 1 -
        #                                       self.pad_top][NOX_cnt + 1 - self.pad_left][CIN_cnt]     # 5, 56, 64
        #             pass
        # else:
        #     self.Input_fm_real = self.InputX_array
        # pass
        #
        # output_fm_size = (self.Output_fm_Oy,
        #                   self.Output_fm_Ox, self.cout_wr_real)
        # self.result = np.zeros(output_fm_size).astype(np.int64)
        # if self.Load_Bias == 0 or self.Load_Bias == 1:
        #     for y in range(self.Output_fm_Oy):
        #         for x in range(self.Output_fm_Ox):
        #             for f in range(self.cout):
        #                 for r in range(self.cin):
        #                     for ky in range(self.conv_Ky):
        #                         for kx in range(self.conv_Kx):
        #                             self.result[y][x][f] += self.Input_fm_real[ky * self.conv_Ey + y * self.conv_Sy][
        #                                 kx * self.conv_Ex + x * self.conv_Sx][r] * \
        #                                 self.weight[ky][kx][r][f]
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
        #                                 kx * self.conv_Ex + x * self.conv_Sx][r] * \
        #                                 self.weight[ky][kx][r][f]
        #                             if self.result[y][x][f] >= 0x7fffffff:
        #                                 self.result[y][x][f] = 0x7fffffff
        #                             elif self.result[y][x][f] <= -0x80000000:
        #                                 self.result[y][x][f] = -0x80000000
        #                         pass
        # result 4, 56 ,32
        self.Write_V_length = self.Output_fm_Oy * self.Output_fm_Ox * self.cout_wr_real
        # --------------------------------------------------example start--------------------------------------------------
        _resultList = []
        _Vresult = self.convertPrim2Mem(
            [self.result])  # 多维数组转化到Mem中的存储数组的4Bytes格式
        # print(_Vresult)
        _V = {
            "Model": "Axon",
            "startInMemory": self.Addr_V_base,
            "lengthInMemory": self.Write_V_length,
            "data": _Vresult
        }
        # print(_V)
        _resultList.append(_V)
        return _resultList

    # --------------------------------------------------example   end--------------------------------------------------

    def cal_para(self):
        if hasattr(self, "axon_delay") and self.axon_delay:
            self.Reset_Addr_A = 1
            self.Reset_Addr_V = 1
            self.MAC_grp_num_last = 3
            self.InA_type = 1
            self.InB_type = 1
            self.Load_Bias = 0
            # self.Addr_Bias_base = 0
            self.Addr_start_offset = 0
            self.L0_num = 0
            self.L1_num = 0
            self.L2_num = 0
            self.L3_num = 0
            self.Addr_InA_L1_step = 0
            self.Addr_InA_L2_step = 0
            self.Addr_InA_L3_step = 0
            self.Addr_InA_L4_step = 0
            self.Addr_InA_L5_step = 0
            self.L0_num_in_last_row = 15
            self.Addr_InA_MAC_step = 0
            self.conv_Sx = 1
            self.conv_Sy = 1
            self.conv_Ex = 1
            self.conv_Ey = 1
            self.pad_top = 0
            self.pad_down = 0
            self.pad_left = 0
            self.pad_right = 0
            self.Read_X_length = 4  # InA_base=InA_end
            self.Write_V_length = 4  # v_base=v_end
            # 确保json输出处所有变量有定义
            self.Output_fm_Ox = 0
            self.Output_fm_Oy = 0
            self.Read_Bias_length = 0
            self.Read_weight_length = 0
            return
        if self.InA_type == 3:
            self.Km_num = np.ceil(self.cin / 64).astype(int)
            self.cin_wr_real = self.Km_num * 64
        else:
            self.Km_num = np.ceil(self.cin / 16).astype(int)
            self.cin_wr_real = self.Km_num * 16

        self.w_grp_num = np.ceil(self.cout / 32).astype(int)
        self.cout_wr_real = self.w_grp_num * 32

        self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        self.Py_real = self.Input_fm_Py + self.pad_top + self.pad_down
        self.Kx_real = (self.conv_Kx - 1) * self.conv_Ex + 1
        self.Ky_real = (self.conv_Ky - 1) * self.conv_Ey + 1
        self.Output_fm_Ox = int(
            (self.Px_real - self.Kx_real) / self.conv_Sx) + 1
        self.Output_fm_Oy = int(
            (self.Py_real - self.Ky_real) / self.conv_Sy) + 1
        self.L0_num = self.Km_num - 1
        self.L1_num = self.conv_Kx - 1
        self.L2_num = self.conv_Ky - 1
        self.L3_num = self.w_grp_num - 1
        self.L4_num = math.ceil(self.Output_fm_Ox / 4) - 1
        self.L5_num = self.Output_fm_Oy - 1
        self.MAC_grp_num_last = self.Output_fm_Ox - self.L4_num * 4 - 1
        self.Addr_InA_MAC_step = self.Km_num * self.conv_Sx

        if self.pad_on:
            self.Addr_InA_L1_step = (self.conv_Ex - 1) * self.Km_num + 1
            self.Addr_InA_L2_step = self.Input_fm_Px * self.conv_Ey * \
                self.Km_num - self.Kx_real * self.Km_num + 1
            self.Addr_InA_L3_step = -(
                (self.conv_Ky - 1) * self.conv_Ey * self.Input_fm_Px + self.Kx_real) * self.Km_num + 1
            self.Addr_InA_L4_step = (4 * self.conv_Sx - self.Input_fm_Px * self.conv_Ey * (
                self.conv_Ky - 1) - self.Kx_real) * self.Km_num + 1
            self.Addr_InA_L5_step = - self.Input_fm_Px * self.Km_num * (self.conv_Ey * (
                self.conv_Ky - 1) - self.conv_Sy) - self.L4_num * 4 * self.conv_Sx * self.Km_num - self.Kx_real * self.Km_num + 1
        else:
            self.Addr_InA_L1_step = (self.conv_Ex - 1) * self.Km_num + 1
            self.Addr_InA_L2_step = self.Input_fm_Px * self.conv_Ey * \
                self.Km_num - self.Kx_real * self.Km_num + 1
            self.Addr_InA_L3_step = -(
                (self.conv_Ky - 1) * self.conv_Ey * self.Input_fm_Px + self.Kx_real) * self.Km_num + 1
            self.Addr_InA_L4_step = (4 * self.conv_Sx - self.Input_fm_Px * self.conv_Ey * (
                self.conv_Ky - 1) - self.Kx_real) * self.Km_num + 1
            self.Addr_InA_L5_step = - self.Input_fm_Px * self.Km_num * (self.conv_Ey * (
                self.conv_Ky - 1) - self.conv_Sy) - self.L4_num * 4 * self.conv_Sx * self.Km_num - self.Kx_real * self.Km_num + 1

        if self.InA_type == 3:
            self.L0_num_in_last_row = self.cin - self.L0_num * 64 - 1
        else:
            self.L0_num_in_last_row = self.cin - self.L0_num * 16 - 1

        L2_step_equal = self.Input_fm_Px * self.Km_num - self.Kx_real * self.Km_num + 1

        if self.pad_on:
            self.Addr_start_offset = self.pad_left + self.pad_top * (
                self.Kx_real - 1) + self.pad_top * L2_step_equal + (
                self.pad_top * self.Kx_real + self.pad_left) * (self.Km_num - 1)
        else:
            self.Addr_start_offset = 0

        self.Read_X_length = self.Input_fm_Px * self.Input_fm_Py * self.Km_num * 4
        if self.InB_type == 3:
            self.Read_weight_length = int(
                np.ceil((self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 2) / 8) * 8)
        else:
            self.Read_weight_length = int(
                self.conv_Kx * self.conv_Ky * self.cin * self.w_grp_num * 8)

        self.Read_Bias_length = self.cout_wr_real
        self.Write_V_length = self.Output_fm_Oy * self.Output_fm_Ox * self.cout_wr_real
        pass

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PI_A.PIC = ' + str(hex(self.PIC)) + ';\n')
            f.write('PI_A.Reset_Addr_A = ' + str(self.Reset_Addr_A) + ';\n')
            f.write('PI_A.Reset_Addr_V = ' + str(self.Reset_Addr_V) + ';\n')
            f.write('PI_A.MAC_grp_num_last = ' +
                    str(self.MAC_grp_num_last) + ';\n')
            if self.pad_on:
                _addr = ((self.Addr_InA_base >> 2) - self.Addr_start_offset)
                if _addr < 0:
                    _addr += 0x8000
                    pass
                f.write('PI_A.Addr_InA_base = ' + str(hex(_addr)) + '\n')
            else:
                f.write('PI_A.Addr_InA_base = ' +
                        str(hex(self.Addr_InA_base >> 2) + '\n'))

            f.write('PI_A.self.Addr_start_offset = ' +
                    str(0 - self.Addr_start_offset) + ';\n')
            f.write('PI_A.InA_type = ' + str(self.InA_type) + ';\n')
            f.write('PI_A.Addr_InB_base = ' +
                    str(hex(self.Addr_InB_base >> 2)) + ';\n')
            f.write('PI_A.InB_type = ' + str(self.InB_type) + ';\n')
            f.write('PI_A.Addr_Bias_base = ' +
                    str(hex(self.Addr_Bias_base >> 2)) + ';\n')
            f.write('PI_A.Load_Bias = ' + str(self.Load_Bias) + ';\n')
            f.write('PI_A.Addr_V_base = ' +
                    str(hex(self.Addr_V_base >> 2)) + ';\n')
            f.write('PI_A.Addr_InA_end = ' + str(
                hex(((self.Addr_InA_base + self.Read_X_length) >> 2) + self.Addr_start_offset - 1) + ';\n'))
            f.write('PI_A.Addr_V_end = ' + str(hex((((self.Addr_V_base +
                                                      self.Write_V_length) >> 3) - 1) << 1) + ';\n'))
            f.write('PI_A.Load_Bias = ' + str(self.Load_Bias) + ';\n')
            f.write('PI_A.L0_num = ' + str(self.L0_num) + ';\n')
            f.write('PI_A.L1_num = ' + str(self.L1_num) + ';\n')
            f.write('PI_A.L2_num = ' + str(self.L2_num) + ';\n')
            f.write('PI_A.L3_num = ' + str(self.L3_num) + ';\n')
            f.write('PI_A.L4_num = ' + str(self.L4_num) + ';\n')
            f.write('PI_A.L5_num = ' + str(self.L5_num) + ';\n')
            f.write('PI_A.L0_num_in_last_row = ' +
                    str(self.L0_num_in_last_row) + ';\n')
            f.write('PI_A.Addr_InA_L1_step = ' +
                    str(self.Addr_InA_L1_step) + ';\n')
            f.write('PI_A.Addr_InA_L2_step = ' +
                    str(self.Addr_InA_L2_step) + ';\n')
            f.write('PI_A.Addr_InA_L3_step = ' +
                    str(self.Addr_InA_L3_step) + ';\n')
            f.write('PI_A.Addr_InA_L4_step = ' +
                    str(self.Addr_InA_L4_step) + ';\n')
            f.write('PI_A.Addr_InA_L5_step = ' +
                    str(self.Addr_InA_L5_step) + ';\n')
            f.write('PI_A.Addr_InA_MAC_step = ' +
                    str(self.Addr_InA_MAC_step) + ';\n')
            f.write('PI_A.Ex = ' + str(self.conv_Ex - 1) + ';\n')
            f.write('PI_A.Ey = ' + str(self.conv_Ey - 1) + ';\n')
            f.write('PI_A.Sx = ' + str(self.conv_Sx) + ';\n')
            f.write('PI_A.Sy = ' + str(self.conv_Sy) + ';\n')
            f.write('PI_A.Pad_top = ' + str(self.pad_top) + ';\n')
            f.write('PI_A.Pad_down = ' + str(self.pad_down) + ';\n')
            f.write('PI_A.Pad_left = ' + str(self.pad_left) + ';\n')
            f.write('PI_A.Pad_right = ' + str(self.pad_right) + ';\n')
            f.write('******************************' + '\n')
            f.write('Input_fm_Px = ' + str(self.Input_fm_Px) + '\n')
            f.write('Input_fm_Py = ' + str(self.Input_fm_Py) + '\n')
            f.write('Output_fm_Ox = ' + str(self.Output_fm_Ox) + '\n')
            f.write('Output_fm_Oy = ' + str(self.Output_fm_Oy) + '\n')
            f.write('Read_X_length = ' + str(self.Read_X_length) + '\n')
            f.write('Read_Bias_length = ' + str(self.Read_Bias_length) + '\n')
            f.write('Read_weight_length = ' +
                    str(self.Read_weight_length) + '\n')
            f.write('Write_V_length = ' + str(self.Write_V_length) + '\n')
            pass
        pass

    def save_results(self, SIMPATH, TBNAME, t):
        path = SIMPATH + TBNAME

        if self.pad_on:
            with open(path + '/InputX_padding.txt', 'w') as f:
                if self.InA_type == 3:
                    for NOY_cnt in range(self.Py_real):
                        for NOX_cnt in range(self.Px_real):
                            for j in range(self.cin_wr_real // 16):
                                final_string = ''
                                for k in range(8):
                                    data_4b = ((self.Input_fm_real[NOY_cnt][NOX_cnt][
                                        j * 16 + k * 2 + 1] & 0x3) << 2) | (
                                        self.Input_fm_real[NOY_cnt][NOX_cnt][j * 16 + k * 2] & 0x3)
                                    final_string = hex_to_string(
                                        data_4b, width=1) + final_string
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
            # string_1 = hex_to_string(self.total_core_num, width=8)
            # f.write(string_1)
            # f.write('\n')
            #
            # string_2 = hex_to_string(self.areas_num_in_core, width=8)
            # f.write(string_2)
            # f.write('\n')
            #
            # string_3 = hex_to_string(self.Addr_V_base, width=8)
            # f.write(string_3)
            # f.write('\n')
            #
            # string_4 = hex_to_string(self.Write_V_length, width=8)
            # f.write(string_4)
            # f.write('\n')

            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for i in range(self.cout_wr_real):
                        final_string = hex_to_string(
                            self.result[NOY_cnt, NOX_cnt, i], width=8)
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
