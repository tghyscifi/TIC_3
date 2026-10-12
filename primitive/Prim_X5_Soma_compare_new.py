import numpy as np
import math
from primitive.basic_operations import BasicOperations
from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_X5_Soma(Primitive):
    def __init__(self):
        super().__init__()

        self.PIC = 0x05
        self.PIC_Mode = 0x00  # 0x01是min
        self.type_in = 1  # 可配置更改：[00]int32[01]int8 [11]Tenary
        self.type_out = 1
        # self.Km_num_in = 1
        # self.Km_num_out = 1
        self.cin = 0
        self.cout = 0
        self.pad_on = False
        # self.CMP_C_en = True
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
        self.CMP_C = 0
        self.in_cut_start = 0

        self.reset_Addr_in = 0
        self.reset_Addr_out = 0
        self.Row_ck_on = False
        self.Addr_Start_in = 0x00
        self.Addr_end_in = 0x00
        self.Addr_Start_out = 0x00
        self.Addr_end_out = 0x00
        self.in_row_max = 0

        self.total_core_num = 1
        self.areas_num_in_core = 1

        self.mem_sel = 0

    def __str__(self):
        if self.PIC_Mode:
            return "20(max pooling)"
        else:
            return "25(min pooling)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)
        _retLsit = []
        if self.type_in == 0:
            self.Km_num_in = math.ceil(self.cin / 4)
            self.cin_wr_real = self.Km_num_in * 4
            num_in_4B = 1
        elif self.type_in == 3:
            self.Km_num_in = math.ceil(self.cin / 64)
            self.cin_wr_real = self.Km_num_in * 64
            num_in_4B = 16
        else:
            self.Km_num_in = math.ceil(self.cin / 16)
            self.cin_wr_real = self.Km_num_in * 16
            num_in_4B = 4

        X_size = (self.Input_fm_Py, self.Input_fm_Px, self.cin)
        X_size_equal = (self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)

        self.X_array = np.zeros(X_size_equal).astype(np.int64)

        if self.type_in == 0:
            X_array = data_generator.random_array_32(X_size).astype(np.int64)
        elif self.type_in == 3:
            X_array = data_generator.random_array_2(X_size).astype(np.int64)
        else:
            X_array = data_generator.random_array_8(X_size).astype(np.int64)

        self.X_array[:, :, :self.cin] = X_array[:, :, :]  # 后面的维度小

        _X = []
        for py in range(self.Input_fm_Py):
            for px in range(self.Input_fm_Px):
                for i in range(self.cin_wr_real // num_in_4B):
                    _tmp = []
                    for j in range(num_in_4B):
                        _tmp.append(int(self.X_array[py][px][num_in_4B * i + j]))
                        pass
                    _X.append(_tmp)
                    pass
        _retLsit.append(_X)
        # _retLsit.append(self)
        if self.Row_ck_on:
            self.Read_X_length = self.Input_fm_Px * \
                                 self.in_row_max * self.Km_num_in * 4
        else:
            self.Read_X_length = self.Input_fm_Px * \
                                 self.Input_fm_Py * self.Km_num_in * 4
        soma2_mark = ''
        if hasattr(self, 'soma2') and self.soma2 is True:
            soma2_mark = '_2'
        prefix = 'P05'
        if self.PIC_Mode == 1:
            prefix = 'P25'
        self.memory_blocks = [
            {'name': prefix + soma2_mark + '_intput_X',
             'start': self.Addr_Start_in,
             'length': self.Read_X_length,
             'data': _retLsit[0],
             'mode': 0}
        ]
        return _retLsit

    def getInfoList(self) -> list:
        if self.type_in == 0:
            self.Km_num_in = math.ceil(self.cin / 4)
            self.cin_wr_real = self.Km_num_in * 4
        elif self.type_in == 3:
            self.Km_num_in = math.ceil(self.cin / 64)
            self.cin_wr_real = self.Km_num_in * 64
        else:
            self.Km_num_in = math.ceil(self.cin / 16)
            self.cin_wr_real = self.Km_num_in * 16

        Read_X_length = self.Input_fm_Px * self.Input_fm_Py * self.Km_num_in * 4  # 没行流水

        _infoList = []
        _X = {
            "start": self.Addr_Start_in,
            "length": Read_X_length,
            "type": self.type_in
        }
        _infoList.append(_X)

        return _infoList
        pass

    def convertPrim2Mem(self, inputList: list) -> list:
        # if self.type_out == 0:
        #     self.cout = self.Km_num_out * 4
        # elif self.type_out == 3:
        #     self.cout = self.Km_num_out * 64
        # else:
        #     self.cout = self.Km_num_out * 16

        if self.type_out == 0:
            self.Km_num_out = np.ceil(self.cout / 4).astype(int)
            self.cout_real = self.Km_num_out * 4
        elif self.type_out == 1:
            self.Km_num_out = np.ceil(self.cout / 16).astype(int)
            self.cout_real = self.Km_num_out * 16
        else:
            self.Km_num_out = np.ceil(self.cout / 64).astype(int)
            self.cout_real = self.Km_num_out * 64

        _Vresult = []
        if self.type_out == 0:
            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for i in range(self.cout_real):
                        _tmp = []
                        _tmp.append(self.pooling_result[NOY_cnt, NOX_cnt, i])
                        _Vresult.append(_tmp)
                        pass  # for i in range(self.cout_wr_real)
                    pass  # for NOX_cnt in range(self.Output_fm_Ox)
                pass  # for NOY_cnt in range(self.Output_fm_Oy)
            pass
        elif self.type_out == 3:
            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for i in range(self.cout_real // 16):
                        _tmp = []
                        for j in range(16):
                            _tmp.append(self.pooling_result[NOY_cnt, NOX_cnt, 16 * i + j])
                            pass
                        _Vresult.append(_tmp)
                        pass
                    pass
                pass
            pass
        else:
            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for i in range(self.cout_real // 4):
                        _tmp = []
                        for j in range(4):
                            _tmp.append(self.pooling_result[NOY_cnt, NOX_cnt, 4 * i + j])
                            pass
                        _Vresult.append(_tmp)
                        pass
                    pass
                pass
            pass
        return _Vresult
        pass  # func convertPrim2Mem

    def convertMem2Prim(self, inputList: list):
        if self.type_in == 0:
            self.Km_num_in = math.ceil(self.cin / 4)
            self.cin_wr_real = self.Km_num_in * 4
            num_in_4B = 1
        elif self.type_in == 3:
            self.Km_num_in = math.ceil(self.cin / 64)
            self.cin_wr_real = self.Km_num_in * 64
            num_in_4B = 16
        else:
            self.Km_num_in = math.ceil(self.cin / 16)
            self.cin_wr_real = self.Km_num_in * 16
            num_in_4B = 4

        X_size_equal = (self.Input_fm_Py, self.Input_fm_Px, self.cin_wr_real)
        self.X_array = np.zeros(X_size_equal).astype(np.int64)
        perPx_in_4B = self.Km_num_in * (16 / 4)  # 每个点的所有cin占多少的4bytes
        perPy_in_4B = self.Input_fm_Px * perPx_in_4B  # 输入图像的每一行的所有cin占多少的4bytes

        # InputX转化：三维数组
        for py in range(self.Input_fm_Py):
            for px in range(self.Input_fm_Px):
                for i in range(self.cin_wr_real // num_in_4B):
                    for j in range(num_in_4B):
                        self.X_array[py][px][num_in_4B * i + j] = inputList[0][int(py * perPy_in_4B + px * perPx_in_4B + i)][j]
                    pass
        pass

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)
        # 先对输入 padding 补零
        self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        self.Py_real = self.Input_fm_Py + self.pad_top + self.pad_down
        self.Output_fm_Ox = int((self.Px_real - self.pooling_Kx) / self.pooling_Sx) + 1
        self.Output_fm_Oy = int((self.Py_real - self.pooling_Ky) / self.pooling_Sy) + 1

        inputx_fm_real = BasicOperations.add_pad(x=self.X_array, padding=(self.pad_top, self.pad_left,
                                                                          self.pad_down, self.pad_right))

        # 数据转换
        if self.type_out == 0:
            self.Km_num_out = np.ceil(self.cout / 4).astype(int)
            self.cout_real = self.Km_num_out * 4
        elif self.type_out == 1:
            self.Km_num_out = np.ceil(self.cout / 16).astype(int)
            self.cout_real = self.Km_num_out * 16
        else:
            self.Km_num_out = np.ceil(self.cout / 64).astype(int)
            self.cout_real = self.Km_num_out * 64

        self.cut_start = self.in_cut_start * 2

        X_size_trans = (self.Py_real, self.Px_real, max(self.cin_wr_real, self.cout_real))  # self.cout_real

        if self.type_in >= self.type_out:
            x_array_trans = np.zeros(X_size_trans).astype(np.int64)
            if self.cin_wr_real <= self.cout_real:
                x_array_trans[:, :, :self.cin_wr_real] = inputx_fm_real[:, :, :]  # 后面的维度小
            else:
                x_array_trans[:, :, :self.cin_wr_real] = inputx_fm_real[:, :, :]      # ????
        else:
            x_array_trans = BasicOperations.convert_type(inputx_fm_real, type_in=self.type_in,
                                                         type_out=self.type_out, in_cut_start=self.in_cut_start)

        # speed up when using X5 to ReLU or Move
        if self.type_out == 0:  # int32
            cmp_const = self.CMP_C - 0x100000000 if self.CMP_C > 0x7fffffff else self.CMP_C
            cmp_const = np.array([cmp_const], dtype=np.int32)
            pooling_en = False
        elif self.type_out == 1:    # int8
            cmp_const = np.array([(self.CMP_C >> (i * 8)) & 0xff for i in range(4)], dtype=np.int8)
            if cmp_const[0] == cmp_const[1] and cmp_const[1] == cmp_const[2] and (
                    cmp_const[2] == cmp_const[3]) and cmp_const[3] == cmp_const[0]:
                pooling_en = False
            else:
                pooling_en = True
        else:
            pooling_en = True   # not support
        if ((self.pooling_Kx + self.pooling_Ky + self.pooling_Sx + self.pooling_Sy) == 0) and (
                (not self.pad_on) or (
                self.pad_top + self.pad_down + self.pad_left + self.pad_right) == 0) and (not pooling_en):
            if self.PIC_Mode == 0:
                self.pooling_result = x_array_trans.clip(min=cmp_const[0], max=math.inf)
            else:
                self.pooling_result = x_array_trans.clip(max=cmp_const[0], min=-math.inf)
        else:
            self.pooling_result = BasicOperations.max_min_pool(pic_mode=self.PIC_Mode, x=x_array_trans,
                                                               type_in=self.type_in, type_out=self.type_out,
                                                               cmp_c=self.CMP_C,
                                                               kernel_size=(self.pooling_Ky, self.pooling_Kx),
                                                               stride=(self.pooling_Sy, self.pooling_Sx))
        _resultList = []
        _Vresult = self.convertPrim2Mem([self.pooling_result])  # 多维数组转化到Mem中的存储数组的4Bytes格式

        if self.Row_ck_on:
            inputRealLength = self.Input_fm_Px * self.in_row_max * self.Km_num_in * 4
        else:
            inputRealLength = self.Input_fm_Px * self.Input_fm_Py * self.Km_num_in * 4  # 没行流水

        # print("inputRealLength =", inputRealLength)

        _V = {
            "Model": "Soma",
            "startInMemory": self.Addr_Start_out,
            "lengthInMemory": self.Write_Y_length,
            "inputAddr": self.Addr_Start_in,
            "inputRealLength": inputRealLength,
            "data": _Vresult,
            "mem_sel": self.mem_sel
        }
        # print(_V)
        _resultList.append(_V)
        return _resultList

        pass

    def cal_para(self):
        self.Px_real = self.Input_fm_Px + self.pad_left + self.pad_right
        self.Py_real = self.Input_fm_Py + self.pad_top + self.pad_down
        self.Output_fm_Ox = int((self.Px_real - self.pooling_Kx) / self.pooling_Sx) + 1
        self.Output_fm_Oy = int((self.Py_real - self.pooling_Ky) / self.pooling_Sy) + 1

        if self.type_in == 0:
            self.Km_num_in = math.ceil(self.cin / 4)
            self.cin_wr_real = self.Km_num_in * 4
        elif self.type_in == 3:
            self.Km_num_in = math.ceil(self.cin / 64)
            self.cin_wr_real = self.Km_num_in * 64
        else:
            self.Km_num_in = math.ceil(self.cin / 16)
            self.cin_wr_real = self.Km_num_in * 16

        if self.type_out == 0:
            self.Km_num_out = np.ceil(self.cout / 4).astype(int)
            self.cout_real = self.Km_num_out * 4
        elif self.type_out == 1:
            self.Km_num_out = np.ceil(self.cout / 16).astype(int)
            self.cout_real = self.Km_num_out * 16
        else:
            self.Km_num_out = np.ceil(self.cout / 64).astype(int)
            self.cout_real = self.Km_num_out * 64

        if (self.type_in == 0 and self.type_out == 1) or (self.type_in == 1 and self.type_out == 3):
            trans_cnt = 4
        elif self.type_in == 0 and self.type_out == 3:
            trans_cnt = 16
        elif self.type_in == 1 and self.type_out == 3:
            trans_cnt = 4  # 不确定
        else:
            trans_cnt = 1

        one_row_in_mem = self.Input_fm_Px * self.Km_num_in

        if self.type_in >= self.type_out:
            self.Kx_step = self.Km_num_in
            self.Ky_step = -(self.pooling_Kx - 1) * self.Km_num_in + one_row_in_mem
        else:
            self.Kx_step = self.Km_num_in - trans_cnt + 1
            self.Ky_step = -(self.pooling_Kx - 1) * self.Km_num_in + one_row_in_mem - trans_cnt + 1

        self.Km_step = -(self.pooling_Kx - 1) * self.Km_num_in - (self.pooling_Ky - 1) * one_row_in_mem + 1

        self.Ox_step = self.Km_step + (self.pooling_Sx - 1) * self.Km_num_in
        self.Oy_step = self.Km_step - self.Km_num_in - (self.Output_fm_Ox - 1) * self.pooling_Sx * self.Km_num_in + self.pooling_Sy * one_row_in_mem

        if self.pad_on:
            if self.type_in >= self.type_out:
                self.Addr_start_offset = self.pad_left * self.Kx_step + self.pad_top * (self.pooling_Kx - 1) * self.Kx_step + self.pad_top * self.Ky_step
            else:
                self.Addr_start_offset = (self.pad_left + self.pad_top * (self.pooling_Kx - 1)) * (self.Kx_step + trans_cnt - 1) + self.pad_top * (
                        self.Ky_step + trans_cnt - 1)  # 需要验证
        else:
            self.Addr_start_offset = 0

        ###以下为新加的代码###
        if self.type_in == self.type_out:
            self.Ox_step = self.Ox_step - (self.Km_num_out - self.Km_num_in)
            self.Oy_step = self.Oy_step - (self.Km_num_out - self.Km_num_in)
        elif (self.type_in == 0 and self.type_out == 1) or (self.type_in == 1 and self.type_out == 3):
            self.Ox_step = self.Ox_step - (4 * self.Km_num_out - self.Km_num_in)
            self.Oy_step = self.Oy_step - (4 * self.Km_num_out - self.Km_num_in)
        elif self.type_in == 0 and self.type_out == 3:
            self.Ox_step = self.Ox_step - (16 * self.Km_num_out - self.Km_num_in)
            self.Oy_step = self.Oy_step - (16 * self.Km_num_out - self.Km_num_in)
        elif (self.type_in == 3 and self.type_out == 1):
            self.Ox_step = self.Ox_step - (math.ceil(self.Km_num_out / 4) - self.Km_num_in)
            self.Oy_step = self.Oy_step - (math.ceil(self.Km_num_out / 4) - self.Km_num_in)
        else:
            pass

        if self.Row_ck_on:
            self.Read_X_length = self.Input_fm_Px * self.in_row_max * self.Km_num_in * 4
        else:
            self.Read_X_length = self.Input_fm_Px * self.Input_fm_Py * self.Km_num_in * 4  # 没行流水

        self.Write_Y_length = self.Output_fm_Oy * self.Output_fm_Ox * self.Km_num_out * 4

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PI_S.PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('PI_S.PIC_Mode = ' + str(hex(self.PIC_Mode)) + '\n')
            f.write('PI_S.reset_Addr_in = ' + str(self.reset_Addr_in) + '\n')
            f.write('PI_S.reset_Addr_out = ' + str(self.reset_Addr_out) + '\n')
            f.write('PI_S.Row_ck_on = ' + str(self.Row_ck_on) + '\n')
            if self.pad_on:
                _addr = ((self.Addr_Start_in >> 2) - self.Addr_start_offset)
                if _addr < 0:
                    _addr += 0x8000
                    pass
                f.write('PI_S.Addr_Start_in = ' + str(hex(_addr)) + '\n')
            else:
                f.write('PI_S.Addr_Start_in = ' + str(hex(self.Addr_Start_in >> 2) + '\n'))
            f.write('Addr_start_offset = ' + str(0 - self.Addr_start_offset) + ';\n')
            f.write('PI_S.type_in = ' + str(self.type_in) + '\n')
            f.write('PI_S.Addr_end_in = ' + str(hex(((self.Addr_Start_in + self.Read_X_length) >> 2) + self.Addr_start_offset - 1) + '\n'))
            f.write('PI_S.Addr_Start_out = ' + str(hex(self.Addr_Start_out >> 2)) + '\n')
            f.write('PI_S.type_out = ' + str(self.type_out) + '\n')
            f.write('PI_S.Addr_end_out = ' + str(hex(((self.Addr_Start_out + self.Write_Y_length) >> 2) - 1) + '\n'))
            f.write('PI_S.Km_num_in = ' + str(self.Km_num_in - 1) + '\n')
            f.write('PI_S.Kx_num = ' + str(self.pooling_Kx - 1) + '\n')
            f.write('PI_S.Ky_num = ' + str(self.pooling_Ky - 1) + '\n')
            f.write('PI_S.Km_num_out = ' + str(self.Km_num_out - 1) + '\n')
            f.write('PI_S.Ox_num = ' + str(self.Output_fm_Ox - 1) + '\n')
            f.write('PI_S.Oy_num = ' + str(self.Output_fm_Oy - 1) + '\n')
            f.write('PI_S.Kx_step = ' + str(self.Kx_step) + '\n')
            f.write('PI_S.Ky_step = ' + str(self.Ky_step) + '\n')
            f.write('PI_S.Km_step = ' + str(self.Km_step) + '\n')
            f.write('PI_S.Ox_step = ' + str(self.Ox_step) + '\n')
            f.write('PI_S.Oy_step = ' + str(self.Oy_step) + '\n')
            f.write('PI_S.Sx = ' + str(self.pooling_Sx) + '\n')
            f.write('PI_S.Sy = ' + str(self.pooling_Sy) + '\n')
            f.write('PI_S.pad_top = ' + str(self.pad_top) + '\n')
            f.write('PI_S.pad_down = ' + str(self.pad_down) + '\n')
            f.write('PI_S.pad_left = ' + str(self.pad_left) + '\n')
            f.write('PI_S.pad_right = ' + str(self.pad_right) + '\n')
            f.write('PI_S.CMP_C = ' + str(hex(self.CMP_C)) + '\n')
            f.write('PI_S.in_cut_start = ' + str(self.in_cut_start) + '\n')
            f.write('PI_S.in_row_max = ' + str(self.in_row_max) + '\n')
            f.write('PI_S.mem_sel = ' + str(self.mem_sel) + '\n')
            f.write('******************************' + '\n')
            f.write('Input_fm_Px = ' + str(self.Input_fm_Px) + '\n')
            f.write('Input_fm_Py = ' + str(self.Input_fm_Py) + '\n')
            f.write('Output_fm_Ox = ' + str(self.Output_fm_Ox) + '\n')
            f.write('Output_fm_Oy = ' + str(self.Output_fm_Oy) + '\n')
            f.write('Read_X_length = ' + str(self.Read_X_length) + '\n')
            f.write('Write_Y_length = ' + str(self.Write_Y_length) + '\n')

    def save_results(self, SIMPATH, TBNAME, t):

        path = SIMPATH + TBNAME

        if self.pad_on:
            with open(path + '/InputX_padding.txt', 'w') as f:
                if self.type_in == 0:
                    for NOY_cnt in range(self.Py_real):
                        for NOX_cnt in range(self.Px_real):
                            for i in range(self.cin_wr_real):
                                data = []
                                final_string = ''
                                final_string = hex_to_string(self.InputX_fm_real[NOY_cnt, NOX_cnt, i],
                                                             width=8) + final_string
                                f.write(final_string)
                                f.write('\n')
                                pass
                elif self.type_in == 3:
                    for NOY_cnt in range(self.Py_real):
                        for NOX_cnt in range(self.Px_real):
                            for j in range(self.cin_wr_real // 16):
                                final_string = ''
                                for k in range(8):
                                    data_4b = ((self.InputX_fm_real[NOY_cnt][NOX_cnt][j * 16 + k * 2 + 1] & 0x3) << 2) | (
                                            self.InputX_fm_real[NOY_cnt][NOX_cnt][j * 16 + k * 2] & 0x3)
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
                                        self.InputX_fm_real[NOY_cnt, NOX_cnt, 4 * i + j]) + final_string
                                f.write(final_string)
                                f.write('\n')
                                pass
        else:
            pass

        with open(path + '/Data_VOUT.txt', 'w') as f:
            if self.type_out == 0:
                for NOY_cnt in range(self.Output_fm_Oy):
                    for NOX_cnt in range(self.Output_fm_Ox):
                        for i in range(self.cout_real):
                            final_string = hex_to_string(self.pooling_result[NOY_cnt, NOX_cnt, i], width=8)
                            f.write(final_string)
                            f.write('\n')
                            pass
            elif self.type_out == 3:
                for NOY_cnt in range(self.Output_fm_Oy):
                    for NOX_cnt in range(self.Output_fm_Ox):
                        for j in range(self.cout_real // 16):
                            final_string = ''
                            for k in range(8):
                                data_4b = ((self.pooling_result[NOY_cnt][NOX_cnt][j * 16 + k * 2 + 1] & 0x3) << 2) | (
                                        self.pooling_result[NOY_cnt][NOX_cnt][j * 16 + k * 2] & 0x3)
                                final_string = hex_to_string(data_4b, width=1) + final_string
                            f.write(final_string)
                            f.write('\n')
                            pass
            else:
                for NOY_cnt in range(self.Output_fm_Oy):
                    for NOX_cnt in range(self.Output_fm_Ox):
                        for i in range(self.cout_real // 4):
                            data = []
                            final_string = ''
                            for j in range(4):
                                final_string = hex_to_string(
                                    self.pooling_result[NOY_cnt, NOX_cnt, 4 * i + j]) + final_string
                            f.write(final_string)
                            f.write('\n')
                            pass

        with open(path + '/Data_VOUT_dec.txt', 'w') as f:
            for NOY_cnt in range(self.Output_fm_Oy):
                for NOX_cnt in range(self.Output_fm_Ox):
                    for COUT_cnt in range(self.cout_real):
                        final_value = self.pooling_result[NOY_cnt, NOX_cnt, COUT_cnt]
                        f.write(str(final_value))
                        f.write('\n')
                        pass

        # with open(path + '/Data_cmp_result_dec.txt', 'w') as f:
        #     for NOY_cnt in range(self.Output_fm_Oy):
        #         for NOX_cnt in range(self.Output_fm_Ox):
        #             for cin_cnt in range(self.cin):
        #                 final_value = self.pooling_result[NOY_cnt, NOX_cnt, cin_cnt]
        #                 f.write(str(final_value))
        #                 f.write('\n')
        #                 pass
        #
        # with open(path + '/Data_cmp_result.txt', 'w') as f:
        #     if self.type_in == 0:
        #         for NOY_cnt in range(self.Output_fm_Oy):
        #             for NOX_cnt in range(self.Output_fm_Ox):
        #                 for cin_cnt in range(self.cin):
        #                     final_string = hex_to_string(self.pooling_result[NOY_cnt, NOX_cnt, cin_cnt], width=8)
        #                     f.write(final_string)
        #                     f.write('\n')
        #                     pass
        #     elif self.type_in == 3:
        #         for NOY_cnt in range(self.Output_fm_Oy):
        #             for NOX_cnt in range(self.Output_fm_Ox):
        #                 for cin_cnt in range(self.cin):
        #                     final_string = hex_to_string(self.pooling_result[NOY_cnt, NOX_cnt, cin_cnt], width=1)
        #                     f.write(final_string)
        #                     f.write('\n')
        #                     pass
        #     else:
        #         for NOY_cnt in range(self.Output_fm_Oy):
        #             for NOX_cnt in range(self.Output_fm_Ox):
        #                 for cin_cnt in range(self.cin):
        #                     final_string = hex_to_string(self.pooling_result[NOY_cnt, NOX_cnt, cin_cnt])
        #                     f.write(final_string)
        #                     f.write('\n')
        #                     pass
