import numpy as np
import os
import copy

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_06_move_merge(Primitive):
    def __init__(self):
        super().__init__()

        self.length_in = 0
        self.length_out = 0
        self.length_ciso = 0

        self.num_in = 0
        self.num_out = 0
        self.num_ciso = 0

        self.type_in = 0
        self.type_out = 0

        self.in_cut_start = 0
        self.PIC = 0x06
        self.PIC_Mode = 0x00
        self.Reset_Addr_in = 1
        self.Reset_Addr_out = 1
        self.Reset_Addr_ciso = 1
        self.Row_ck_on = 0
        self.Addr_Start_in = 0x0000
        self.Addr_end_in = 0x0000
        self.Addr_Start_out = 0x0000
        self.Addr_end_out = 0x0000
        self.Addr_Start_ciso = 0x0000
        self.Addr_end_ciso = 0x0000
        self.in_row_max = 0
        self.mem_sel = 0
        self.in_ciso_pipe_sel = 0
        
        self.real_length_in_en=False
        self.real_num_in=0


    def __str__(self):
        return "06(move_merge)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)

        if self.real_length_in_en==True:
            tmp_num_in=self.real_num_in
        else:
            tmp_num_in=self.num_in


        _retLsit = []
        if self.type_in == 0:
            self.Km_num_in = np.ceil(self.length_in / 4).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 4).astype(int)
            self.length_in_equal = self.Km_num_in * 4
            self.length_ciso_equal = self.Km_num_ciso * 4
        elif self.type_in == 3:
            self.Km_num_in = np.ceil(self.length_in / 64).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 64).astype(int)
            self.length_in_equal = self.Km_num_in * 64
            self.length_ciso_equal = self.Km_num_ciso * 64
        else:
            self.Km_num_in = np.ceil(self.length_in / 16).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 16).astype(int)
            self.length_in_equal = self.Km_num_in * 16
            self.length_ciso_equal = self.Km_num_ciso * 16

        in_size = (tmp_num_in, self.length_in)
        ciso_size = (self.num_ciso, self.length_ciso)

        in_size_equal = (tmp_num_in, self.length_in_equal)
        ciso_size_equal = (self.num_ciso, self.length_ciso_equal)
        # print(tmp_num_in)
        # print(self.num_ciso)

        if self.type_out == 0:
            self.Km_num_out = np.ceil(self.length_out / 4).astype(int)
            self.length_out_equal = self.Km_num_out * 4
        elif self.type_out == 3:
            self.Km_num_out = np.ceil(self.length_out / 64).astype(int)
            self.length_out_equal = self.Km_num_out * 64
        else:
            self.Km_num_out = np.ceil(self.length_out / 16).astype(int)
            self.length_out_equal = self.Km_num_out * 16

        self.array_in = np.zeros(in_size_equal).astype(int)
        self.array_ciso = np.zeros(ciso_size_equal).astype(int)

        if self.type_in == 0:
            array_in = data_generator.random_array_32(in_size).astype(int)
            array_ciso = data_generator.random_array_32(ciso_size).astype(int)
            num_in_4B = 1
        elif self.type_in == 3:
            array_in = data_generator.random_array_2(in_size).astype(int)
            array_ciso = data_generator.random_array_2(ciso_size).astype(int)
            num_in_4B = 16
        else:
            array_in = data_generator.random_array_8(in_size).astype(int)
            array_ciso = data_generator.random_array_8(ciso_size).astype(int)
            num_in_4B = 4

        self.array_in[:, :self.length_in] = array_in[:, :]
        self.array_ciso[:, :self.length_ciso] = array_ciso[:, :]

        pernum_in_in_4B = int(self.Km_num_in * (16 / 4))
        pernum_ciso_in_4B = int(self.Km_num_ciso * (16 / 4))

        # print(num_in_4B)
        _in = []
        for cnt in range(tmp_num_in):
            for i in range(self.length_in_equal // num_in_4B):
                _tmp = []
                for j in range(num_in_4B):
                    _tmp.append(int(self.array_in[cnt][int(num_in_4B * i + j)]))
                    pass
                _in.append(_tmp)
                pass
            pass

        _retLsit.append(_in)

        _ciso = []
        for cnt in range(self.num_ciso):
            for i in range(self.length_ciso_equal // num_in_4B):
                _tmp = []
                for j in range(num_in_4B):
                    _tmp.append(int(self.array_ciso[cnt][num_in_4B * i + j]))
                    pass
                _ciso.append(_tmp)
                pass
            pass
        _retLsit.append(_ciso)
        # _retLsit.append(self)
        soma2_mark = ''
        if hasattr(self, 'soma2') and self.soma2 is True:
            soma2_mark = '_2'

        self.memory_blocks = [
            {'name': 'P06' + soma2_mark + '_input_x1',
             'start': self.Addr_Start_in,
             'data': _retLsit[0],
             'mode': 0},
            {'name': 'P06' + soma2_mark + '_input_x2',
             'start': self.Addr_Start_ciso,
             'data': _retLsit[1],
             'mode': 0}
        ]
        return _retLsit
        pass

    def getInfoList(self) -> list:
        if self.real_length_in_en==True:
            tmp_num_in=self.real_num_in
        else:
            tmp_num_in=self.num_in
        if self.type_in == 0:
            self.Km_num_in = np.ceil(self.length_in / 4).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 4).astype(int)
            self.length_in_equal = self.Km_num_in * 4
            self.length_ciso_equal = self.Km_num_ciso * 4
        elif self.type_in == 3:
            self.Km_num_in = np.ceil(self.length_in / 64).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 64).astype(int)
            self.length_in_equal = self.Km_num_in * 64
            self.length_ciso_equal = self.Km_num_ciso * 64
        else:
            self.Km_num_in = np.ceil(self.length_in / 16).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 16).astype(int)
            self.length_in_equal = self.Km_num_in * 16
            self.length_ciso_equal = self.Km_num_ciso * 16

        Read_in_length = self.Km_num_in * tmp_num_in * 4
        Read_ciso_length = self.Km_num_ciso * self.num_ciso * 4

        _infoList = []
        _In = {
            "start": self.Addr_Start_in,
            "length": Read_in_length,
            "type": self.type_in
        }
        _infoList.append(_In)
        _Ciso = {
            "start": self.Addr_Start_ciso,
            "length": Read_ciso_length,
            "type": self.type_in
        }
        _infoList.append(_Ciso)

        return _infoList
        pass

    def convertPrim2Mem(self, inputList: list) -> list:
        _Vresult = []
        if self.type_out == 0:
            for i in range(self.num_out):
                for j in range(self.length_out_equal):
                    _tmp = []
                    _tmp.append(int(self.array_out[i][j]))
                    _Vresult.append(_tmp)
                    pass  # for i in range(self.cout_wr_real)
                pass  # for NOX_cnt in range(self.Output_fm_Ox)
            pass  # for NOY_cnt in range(self.Output_fm_Oy)
        elif self.type_out == 3:
            for i in range(self.num_out):
                for j in range(self.length_out_equal // 16):
                    _tmp = []
                    for k in range(16):
                        _tmp.append(int(self.array_out[i][k + j * 16]))
                        pass
                    _Vresult.append(_tmp)
                    pass
                pass
            pass
        else:
            for i in range(self.num_out):
                for j in range(self.length_out_equal // 4):
                    _tmp = []
                    for k in range(4):
                        _tmp.append(int(self.array_out[i, j * 4 + k]))
                        pass
                    _Vresult.append(_tmp)
                    pass
                pass
            pass
        return _Vresult
        pass  # func convertPrim2Mem

    def convertMem2Prim(self, inputList: list):
        if self.type_in == 0:
            self.Km_num_in = np.ceil(self.length_in / 4).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 4).astype(int)
            self.length_in_equal = self.Km_num_in * 4
            self.length_ciso_equal = self.Km_num_ciso * 4
            num_in_4B = 1
        elif self.type_in == 3:
            self.Km_num_in = np.ceil(self.length_in / 64).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 64).astype(int)
            self.length_in_equal = self.Km_num_in * 64
            self.length_ciso_equal = self.Km_num_ciso * 64
            num_in_4B = 16
        else:
            self.Km_num_in = np.ceil(self.length_in / 16).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 16).astype(int)
            self.length_in_equal = self.Km_num_in * 16
            self.length_ciso_equal = self.Km_num_ciso * 16
            num_in_4B = 4

        in_size_equal = (self.num_in, self.length_in_equal)
        ciso_size_equal = (self.num_ciso, self.length_ciso_equal)
        self.array_in = np.zeros(in_size_equal).astype(int)
        self.array_ciso = np.zeros(ciso_size_equal).astype(int)

        pernum_in_in_4B = int(self.Km_num_in * (16 / 4))
        pernum_ciso_in_4B = int(self.Km_num_ciso * (16 / 4))

        tmp_inputlist=[]
        if self.real_length_in_en==True:
            for i in range(self.num_in//self.real_num_in):
                tmp_inputlist+=copy.deepcopy(inputList[0])
            if self.num_in%self.real_num_in!=0:
                tmp_inputlist+=copy.deepcopy(inputList[0][0:int(self.length_in_equal*(self.num_in%self.real_num_in))])
        else:
            tmp_inputlist=inputList[0]
        for cnt in range(self.num_in):
            for i in range(self.length_in_equal // num_in_4B):
                for j in range(num_in_4B):
                    self.array_in[cnt][num_in_4B * i + j] = tmp_inputlist[int(cnt * pernum_in_in_4B + i)][j]

        for cnt in range(self.num_ciso):
            for i in range(self.length_ciso_equal // num_in_4B):
                for j in range(num_in_4B):
                    self.array_ciso[cnt][num_in_4B * i + j] = inputList[1][int(cnt * pernum_ciso_in_4B + i)][j]
        
        

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)
        out_size = (self.num_out, self.length_out_equal)
        self.array_out = np.zeros(out_size).astype(int)

        num_in_real = min(self.num_in, self.num_out)
        num_ciso_real = min(self.num_ciso, self.num_out)
        num_out_real = max(num_in_real, num_ciso_real)

        length_out_real = min((self.length_in_equal + self.length_ciso_equal), self.length_out_equal)

        trans_array_in = np.zeros_like(self.array_in)
        trans_array_ciso = np.zeros_like(self.array_ciso)

        if self.type_out <= self.type_in:
            trans_array_in[:, :] = self.array_in[:, :]
            trans_array_ciso[:, :] = self.array_ciso[:, :]
        else:
            trans_array_in[:, :] = self.array_in // 2 ** (self.in_cut_start * 2)
            trans_array_ciso[:, :] = self.array_ciso // 2 ** (self.in_cut_start * 2)

        if self.type_out == 1:
            trans_array_in = trans_array_in.clip(-128, 127)
            trans_array_ciso = trans_array_ciso.clip(-128, 127)
        elif self.type_out == 3:
            trans_array_in = trans_array_in.clip(-1, 1)
            trans_array_ciso = trans_array_ciso.clip(-1, 1)

        for i in range(num_out_real):
            for j in range(length_out_real):
                if j in range(self.length_in_equal):
                    if i in range(num_in_real):
                        self.array_out[i, j] = trans_array_in[i, j]

                elif j in range(self.length_in_equal, length_out_real):
                    if i in range(num_ciso_real):
                        self.array_out[i, j] = trans_array_ciso[i, j - self.length_in_equal]

        if self.Row_ck_on:
            if self.in_ciso_pipe_sel:
                inputRealLength = self.length_ciso_equal * self.in_row_max
                start_addr = self.Addr_Start_ciso
            else:
                inputRealLength = self.length_in_equal * self.in_row_max
                start_addr = self.Addr_Start_in
            pass
        else:
            if self.in_ciso_pipe_sel:
                inputRealLength = self.length_ciso_equal * self.num_in  # 没行流水
                start_addr = self.Addr_Start_ciso
            else:
                inputRealLength = self.length_in_equal * self.num_in  # 没行流水
                start_addr = self.Addr_Start_in
            pass
        # print(inputRealLength)
        # print(self.array_out[0])
        # print(self.array_out.shape)
        _resultList = []
        _Vresult = self.convertPrim2Mem([self.array_out])  # 多维数组转化到Mem中的存储数组的4Bytes格式
        # print(_Vresult)
        # print(_Vresult.__len__())
        _V = {
            "Model": "Soma",
            "data": _Vresult,
            "startInMemory": self.Addr_Start_out,
            "lengthInMemory": self.Write_Y_length,
            "inputAddr": start_addr,
            "inputRealLength": inputRealLength,
            "mem_sel": self.mem_sel
        }
        _resultList.append(_V)
        return _resultList

    def cal_para(self):
        if self.type_in == 0:
            self.Km_num_in = np.ceil(self.length_in / 4).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 4).astype(int)
            self.length_in_equal = self.Km_num_in * 4
            self.length_ciso_equal = self.Km_num_ciso * 4
        elif self.type_in == 3:
            self.Km_num_in = np.ceil(self.length_in / 64).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 64).astype(int)
            self.length_in_equal = self.Km_num_in * 64
            self.length_ciso_equal = self.Km_num_ciso * 64
        else:
            self.Km_num_in = np.ceil(self.length_in / 16).astype(int)
            self.Km_num_ciso = np.ceil(self.length_ciso / 16).astype(int)
            self.length_in_equal = self.Km_num_in * 16
            self.length_ciso_equal = self.Km_num_ciso * 16

        if self.type_out == 0:
            self.Km_num_out = np.ceil(self.length_out / 4).astype(int)
            self.length_out_equal = self.Km_num_out * 4
        elif self.type_out == 3:
            self.Km_num_out = np.ceil(self.length_out / 64).astype(int)
            self.length_out_equal = self.Km_num_out * 64
        else:
            self.Km_num_out = np.ceil(self.length_out / 16).astype(int)
            self.length_out_equal = self.Km_num_out * 16

        if self.real_length_in_en==True:
            tmp_num_in=self.real_num_in
        else:
            tmp_num_in=self.num_in
        self.Read_in_length = self.Km_num_in * tmp_num_in * 4
        self.Read_ciso_length = self.Km_num_ciso * self.num_ciso * 4
        self.Write_Y_length = self.Km_num_out * self.num_out * 4

        if self.Row_ck_on:
            if self.in_ciso_pipe_sel:
                self.Addr_end_ciso = self.Addr_Start_ciso + self.Km_num_ciso * 4 * self.in_row_max
                self.Addr_end_in = self.Addr_Start_in + self.Km_num_in * 4 * tmp_num_in
            else:
                self.Addr_end_in = self.Addr_Start_in + self.Km_num_in * 4 * self.in_row_max
                self.Addr_end_ciso = self.Addr_Start_ciso + self.Km_num_ciso * 4 * self.num_ciso
        else:
            self.Addr_end_ciso = self.Addr_Start_ciso + self.Km_num_ciso * 4 * self.num_ciso
            self.Addr_end_in = self.Addr_Start_in + self.Km_num_in * 4 * tmp_num_in
        pass  # func cal_para

    def save_para(self, path: str):
        with open(path, 'w') as f:
            f.write('PI_S.PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('PI_S.PIC_Mode = ' + str(hex(self.PIC_Mode)) + '\n')
            f.write('PI_S.Reset_Addr_in = ' + str(self.Reset_Addr_in) + '\n')
            f.write('PI_S.Reset_Addr_out = ' + str(self.Reset_Addr_out) + '\n')
            f.write('PI_S.Reset_Addr_ciso = ' + str(self.Reset_Addr_ciso) + '\n')
            f.write('PI_S.Row_ck_on = ' + str(self.Row_ck_on) + '\n')
            f.write('PI_S.Addr_Start_in = ' + str(hex(self.Addr_Start_in >> 2)) + '\n')
            # f.write('PI_S.Addr_end_in = ' + str(hex(((self.Addr_Start_in + self.Read_in_length) >> 2) - 1)) + '\n')
            f.write('PI_S.Addr_end_in = ' + str(hex((self.Addr_end_in >> 2) - 1)) + '\n')
            f.write('PI_S.Addr_Start_out = ' + str(hex(self.Addr_Start_out >> 2)) + '\n')
            f.write('PI_S.Addr_end_out = ' + str(hex(((self.Addr_Start_out + self.Write_Y_length) >> 2) - 1)) + '\n')
            f.write('PI_S.Addr_Start_ciso = ' + str(hex(self.Addr_Start_ciso >> 2)) + '\n')
            f.write('PI_S.Addr_end_ciso = ' + str(hex(((self.Addr_Start_ciso + self.Read_ciso_length) >> 2) - 1)) + '\n')
            f.write('PI_S.in_row_max = ' + str(self.in_row_max) + '\n')
            f.write('PI_S.Km_num_in = ' + str(self.Km_num_in - 1) + '\n')
            f.write('PI_S.Km_num_ciso = ' + str(self.Km_num_ciso - 1) + '\n')
            f.write('PI_S.Km_num_out = ' + str(self.Km_num_out - 1) + '\n')
            f.write('PI_S.num_in = ' + str(self.num_in - 1) + '\n')
            f.write('PI_S.num_ciso = ' + str(self.num_ciso - 1) + '\n')
            f.write('PI_S.num_out = ' + str(self.num_out - 1) + '\n')
            f.write('PI_S.type_in = ' + str(self.type_in) + '\n')
            f.write('PI_S.type_out = ' + str(self.type_out) + '\n')
            f.write('PI_S.in_cut_start = ' + str(self.in_cut_start) + '\n')
            f.write('******************************' + '\n')
            f.write('Read_in_length = ' + str(self.Read_in_length) + '\n')
            f.write('Read_ciso_length = ' + str(self.Read_ciso_length) + '\n')
            f.write('Write_out_length = ' + str(self.Write_Y_length) + '\n')
        pass

    def save_results(self, SIMPATH, TBNAME, t):
        path = SIMPATH + TBNAME

        with open(path + '/Data_OUT.txt', 'w') as f:
            if self.type_out == 0:
                for i in range(self.num_out):
                    for j in range(self.length_out_equal):
                        final_string = hex_to_string(self.array_out[i, j], width=8)
                        f.write(final_string)
                        f.write('\n')

            elif self.type_out == 3:
                for i in range(self.num_out):
                    for j in range(self.length_out_equal // 16):
                        final_string = ''
                        for k in range(8):
                            data_4b = ((self.array_out[i][j * 16 + k * 2 + 1] & 0x3) << 2) | (self.array_out[i][j * 16 + k * 2] & 0x3)
                            final_string = hex_to_string(data_4b, width=1) + final_string
                        f.write(final_string)
                        f.write('\n')

            else:
                for i in range(self.num_out):
                    for j in range(self.length_out_equal // 4):
                        final_string = ''
                        for k in range(4):
                            final_string = hex_to_string(self.array_out[i][j * 4 + k]) + final_string
                        f.write(final_string)
                        f.write('\n')
        pass
