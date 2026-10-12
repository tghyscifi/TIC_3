import numpy as np
import os
import copy
import math

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.lfsr import lfsr
from generator.functions.util import hex_to_string


class Prim_08_lif(Primitive):
    def __init__(self):
        super().__init__()

        self.neu_num = 0
        self.group_num = 0
        self.Row_ck_on = 0

        self.Seed = 0
        self.Vth0 = 0
        self.Vth_adpt_en = False
        self.Vth_alpha = 0
        self.Vth_beta = 0
        self.Vth_Incre = 0
        self.VR = 0
        self.VL = 0
        self.Vleaky_adpt_en = False
        self.Vleaky_alpha = 0
        self.Vleaky_beta = 0
        self.dV = 0
        self.Ref_len = 0
        self.Tw_cnt = 0
        self.Vinit = 0
        self.Tw_len = 0
        self.Tw_en = True

        self.VM_const_en = True
        self.VM_const = 0
        self.VM_len = 4
        self.VM_para = np.array([])

        self.Vtheta_const_en = True
        self.Vtheta_const = 0
        self.Vtheta_len = 4
        self.Vtheta_para = np.array([])

        self.ref_cnt_const_en = True
        self.ref_cnt_const = 0

        self.Rst_mode = 0
        self.fire_type = 0
        self.in_cut_start = 0

        self.PIC = 0x08
        self.PIC_Mode = 0x00
        self.reset_Addr_Uin = 1
        self.reset_Addr_V = 1
        self.reset_Addr_S = 1
        self.reset_Addr_VM = 1
        self.reset_Addr_Vtheta = 1
        self.Addr_Uin_start = 0x0000
        self.Addr_Uin_end = 0x0000
        self.Addr_S_Start = 0x0000
        self.Addr_S_end = 0x0000
        self.Addr_V_start = 0x0000
        self.Addr_V_end = 0x0000
        self.Addr_VM_start = 0x0000
        self.Addr_VM_end = 0x0000
        self.Addr_Vtheta_start = 0x0000
        self.Addr_Vtheta_end = 0x0000
        self.Addr_para = 0x0000
        self.in_cut_start = 0  # 有疑问
        self.in_row_max = 0

        self.mem_sel = 0

    def __str__(self):
        return "08(LIF)"

    def init_data(self) -> list:
        # self.attributes = copy.deepcopy(self.__dict__)

        # uin VM Vtheta V
        _retLsit = []
        self.neuron_length = self.neu_num * self.group_num

        self.Uin = data_generator.random_array_28(self.neuron_length)
        self.V = data_generator.random_array_28(self.neuron_length).astype(np.int64)

        if self.ref_cnt_const_en is True:
            self.ref_cnt = np.ones(self.neuron_length).astype('int8') * self.ref_cnt_const
        else:
            self.ref_cnt = data_generator.random_array_u4(self.neuron_length, max=self.Ref_len + 1)

        if self.VM_const_en is True:
            self.VM_para = np.ones(self.VM_len).astype('int32') * self.VM_const
        else:
            self.VM_para = data_generator.random_array_32(self.VM_len)

        if self.Vtheta_const_en is True:
            self.Vtheta_para = np.ones(self.Vtheta_len).astype('int32') * self.Vtheta_const
        else:
            self.Vtheta_para = data_generator.random_array_28(self.Vtheta_len)

        _Uin = []
        for i in range(self.neuron_length):
            _tmp = []
            _tmp.append(int(self.Uin[i]))
            _Uin.append(_tmp)
            pass
        _retLsit.append(_Uin)

        _VMpara = []
        for i in range(self.VM_len):
            _tmp = []
            _tmp.append(int(self.VM_para[i]))
            _VMpara.append(_tmp)
            pass
        _retLsit.append(_VMpara)
        # print(self.VM_para)
        _Vthetapara = []
        for i in range(self.Vtheta_len):
            _tmp = []
            _tmp.append(int(self.Vtheta_para[i]))
            _Vthetapara.append(_tmp)
            pass
        _retLsit.append(_Vthetapara)

        # print(self.Vtheta_para)
        V_ref_cnt = (self.ref_cnt.astype('int32') << 28) | (self.V & (2 ** 28 - 1))  # 这个是否是应该放在这里
        _V = []
        for i in range(self.neuron_length):
            _tmp = []
            _tmp.append(int(V_ref_cnt[i]))
            _V.append(_tmp)
            pass
        _retLsit.append(_V)
        # _retLsit.append(self)
        # print(V_ref_cnt)
        soma2_mark = ''
        if hasattr(self, 'soma2') and self.soma2 is True:
            soma2_mark = '_2'
        self.memory_blocks = [
            {'name': 'P08' + soma2_mark + '_Uin',
             'start': self.Addr_Uin_start,
             'length': len(_retLsit[0]),
             'data': _retLsit[0],
             'mode': 0},
            {'name': 'P08' + soma2_mark + '_VM',
             'start': self.Addr_VM_start,
             'length': len(_retLsit[1]),
             'data': _retLsit[1],
             'mode': 0},
            {'name': 'P08' + soma2_mark + '_Vtheta',
             'start': self.Addr_Vtheta_start,
             'length': len(_retLsit[2]),
             'data': _retLsit[2],
             'mode': 0},
            {'name': 'P08' + soma2_mark + '_V',
             'start': self.Addr_V_start,
             'length': len(_retLsit[3]),
             'data': _retLsit[3],
             'mode': 0},
        ]
        return _retLsit

    def getInfoList(self) -> list:
        self.Read_Uin_length = self.neuron_length
        self.Read_V_length = self.neuron_length
        self.Read_VM_length = self.VM_len
        self.Read_Vtheta_length = self.Vtheta_len

        _infoList = []
        _Uin = {
            "start": self.Addr_Uin_start,
            "length": self.Read_Uin_length,
            "type": 0
        }
        _infoList.append(_Uin)
        _VMpara = {
            "start": self.Addr_VM_start,
            "length": self.Read_VM_length,
            "type": 0
        }
        _infoList.append(_VMpara)
        _Vthetapara = {
            "start": self.Addr_Vtheta_start,
            "length": self.Read_Vtheta_length,
            "type": 0
        }
        _infoList.append(_Vthetapara)
        _V = {
            "start": self.Addr_V_start,
            "length": self.Read_V_length,
            "type": 0
        }
        _infoList.append(_V)
        # print(_infoList)
        # os.system("pause")
        return _infoList

    def convertPrim2Mem(self, inputList: list) -> list:
        _Vresult = []
        out_shape = self.out_reg.shape

        _out_reg = []
        # print(self.out_reg)
        if self.fire_type in (0, 1, 6):
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    _out_reg.append([self.out_reg[i][j]])
        elif self.fire_type in (2, 3, 4, 7):
            for i in range(out_shape[0]):
                for j in range(out_shape[1] // 4):
                    tmp = []
                    for k in range(4):
                        tmp.append(self.out_reg[i][j * 4 + k])
                    _out_reg.append(tmp)
        elif self.fire_type == 5:
            for i in range(out_shape[0]):
                for j in range(out_shape[1] // 16):
                    tmp = []
                    for k in range(8):
                        tmp.append(self.out_reg[i][j * 16 + k * 2] & 0x3)
                        tmp.append(self.out_reg[i][j * 16 + k * 2 + 1] & 0x3)
                    _out_reg.append(tmp)
        _Vresult.append(_out_reg)

        V_ref_cnt = (self.ref_cnt.astype('int32') << 28) | (self.V & (2 ** 28 - 1))
        # print(V_ref_cnt[0:8])
        _V_ref_cnt = []
        for i in range(self.neuron_length):
            _V_ref_cnt.append([V_ref_cnt[i]])
        _Vresult.append(_V_ref_cnt)
        # print(self.Vtheta)
        _Vtheta = []
        for i in range(self.Vtheta_len):
            _Vtheta.append([self.Vtheta[i]])
        _Vresult.append(_Vtheta)

        # para
        _para = []
        _para.append([self.seed])
        data = ((self.Vth0 & 0xffffff) << 8) | (self.Tw_cnt & 0xff)
        _para.append([data])
        data = ((self.Vth_beta & 0xfffff) << 12) | ((self.Vth_alpha & 0xff) << 4) | (self.Vth0 >> 24) & 0xf
        _para.append([data])
        data = ((self.Vth_Incre & 0xffffff) << 8) | ((self.Vth_beta >> 20) & 0xff)
        _para.append([data])
        _Vresult.append(_para)

        # os.system("pause")
        return _Vresult

    def convertMem2Prim(self, inputList: list):

        self.neuron_length = self.neu_num * self.group_num
        self.Uin = np.zeros(self.neuron_length).astype(np.int64)
        self.V = np.zeros(self.neuron_length).astype(np.int64)
        self.ref_cnt = np.zeros(self.neuron_length).astype(np.int64)
        VM_repeat = self.neuron_length // self.VM_len
        Vtheta_repeat = self.neuron_length // self.Vtheta_len

        for i in range(self.neuron_length):
            self.Uin[i] = int(inputList[0][i][0])

        self.VM_para = np.zeros(self.VM_len).astype(np.int64)
        for i in range(self.VM_len):
            self.VM_para[i] = int(inputList[1][i][0])
        self.VM = np.tile(self.VM_para, VM_repeat)

        self.Vtheta_para = np.zeros(self.Vtheta_len).astype(np.int64)
        for i in range(self.Vtheta_len):
            self.Vtheta_para[i] = int(inputList[2][i][0])
        self.Vtheta = np.tile(self.Vtheta_para, Vtheta_repeat)
        # print(self.Vtheta)
        for i in range(self.neuron_length):
            v_ref_cnt = int(inputList[3][i][0])
            self.V[i] = v_ref_cnt & (2 ** 28 - 1)
            if self.V[i] > 0x7ffffff:
                self.V[i] = (self.V[i] | 0xf0000000) - 0xffffffff - 1
            self.ref_cnt[i] = (v_ref_cnt & 0xffffffff) >> 28
        # for i in range(int(len(self.V) / 8)):
        #     print(self.V[i * 8:(i + 1) * 8])

    def prim_execution(self, inputList: list) -> list:
        self.convertMem2Prim(inputList)
        # 输出时间窗判断是否结束
        Tw_finish = self.Tw_cnt >= self.Tw_len

        # 生成32bit随机向量
        self.lfsr_vec = np.zeros(self.neuron_length).astype('int32')
        for i in range(self.neuron_length):
            self.lfsr_vec[i] = self.lfsr.update()

        # 对随机向量做掩膜，限定随机数范围
        lfsr_mask = np.zeros(self.neuron_length).astype('int32')
        for i in range(self.neuron_length):
            # print(self.neuron_length)
            if self.lfsr_vec[i] < 0:
                lfsr_mask[i] = ~((~self.lfsr_vec[i]) & self.VM[i]) + 1
            else:
                lfsr_mask[i] = self.lfsr_vec[i] & self.VM[i]

        # 32bit转成有符号28bit数
        lfsr_mask_cut = lfsr_mask & 0xfffffff
        lfsr_mask_cut[lfsr_mask_cut > 0x7ffffff] = lfsr_mask_cut[lfsr_mask_cut > 0x7ffffff] - 0x10000000

        # 更新Vth: 阈值Vth由三个部分组成，固定值，自适应，随机数
        self.Vth = self.Vth0 + self.Vtheta + lfsr_mask_cut
        self.Vth.clip(-2 ** 27, 2 ** 27 - 1)  # 28bit饱和截取
        self.Vth[self.Vth < self.VL] = self.VL  # 下限饱和

        # 更新V: 不应期期间不响应外部计数，ref_cnt是不应期时间计数，减计数
        self.V[self.ref_cnt == 0] = self.V[self.ref_cnt == 0] + self.Uin[self.ref_cnt == 0]
        self.V = self.V.clip(-2 ** 27, 2 ** 27 - 1)

        # V和Vth做比较 发放spike
        spike = self.V > self.Vth

        # 根据规则进行一系列更新
        # 更新V
        V_update = np.zeros(self.neuron_length).astype('int32')
        V_update[:] = self.V[:]
        if self.Rst_mode == 0:
            V_update[spike == 1] = self.VR
        elif self.Rst_mode == 1:
            V_update[spike == 1] = V_update[spike == 1] - self.Vth[spike == 1]
        elif self.Rst_mode == 2:
            V_update[spike == 1] = V_update[spike == 1] - self.dV
        else:
            V_update = V_update

        # 更新后膜电位衰减
        V_new = np.zeros(self.neuron_length).astype('int32')
        if Tw_finish is False:
            if self.Vleaky_adpt_en is True:  # 自适应开启
                V_new[:] = (V_update.astype('int64') * self.Vleaky_alpha // 256) + self.Vleaky_beta
            else:  # 常值变化
                V_new[:] = V_update + self.Vleaky_beta
        else:  # 时间窗结束，复位成初始值
            V_new[:] = self.Vinit

        V_new = V_new.clip(-2 ** 27, 2 ** 27 - 1)

        V_new[V_new < self.VL] = self.VL

        # 自适应阈值衰减
        if self.Vth_adpt_en is True:
            Vtheta_updata = (self.Vtheta.astype('int64') * self.Vth_alpha // 256) + self.Vth_beta
        else:
            Vtheta_updata = self.Vtheta + self.Vth_beta
        # spike后有增量
        Vtheta_updata[spike == 1] = Vtheta_updata[spike == 1] + self.Vth_Incre

        Vtheta_updata = Vtheta_updata.clip(-2 ** 27, 2 ** 27 - 1)

        # 根据不同的fire模式选择输出数据
        if self.fire_type == 0:
            self.out_reg[:] = self.V[:]
        elif self.fire_type == 1:
            self.out_reg[:] = V_new[:]
        elif self.fire_type == 2:
            self.out_reg[:] = self.V[:] // 2 ** (self.in_cut_start * 2)
            self.out_reg = self.out_reg.clip(-128, 127)
        elif self.fire_type == 3:
            self.out_reg[:] = V_new[:] // 2 ** (self.in_cut_start * 2)
            self.out_reg = self.out_reg.clip(-128, 127)
        elif self.fire_type == 4 or self.fire_type == 5:
            self.out_reg[:] = spike
        elif self.fire_type == 6:
            self.out_reg[:] = lfsr_mask
        elif self.fire_type == 7:
            self.out_reg[:] = np.bitwise_and(lfsr_mask[:], 255)
            self.out_reg[self.out_reg > 127] = self.out_reg[self.out_reg > 127] - 256

        # 每32bit,高4bit代表不应期，低28bit代表膜电位
        # 原语执行完一次会更新一次神经元的不应期，每个神经元的不应期不同
        for i in range(self.neuron_length):
            if spike[i] == 1:
                self.ref_cnt[i] = self.Ref_len
            elif self.ref_cnt[i] > 0:
                self.ref_cnt[i] = self.ref_cnt[i] - 1
            else:
                self.ref_cnt[i] = 0

        if self.Tw_en is True:
            if self.Tw_cnt == self.Tw_len:
                self.Tw_cnt = 0
            else:
                self.Tw_cnt = self.Tw_cnt + 1

        # 除了Uin，几乎都要写回更新  下面输出写回mem
        self.V = V_new
        self.Vtheta = Vtheta_updata

        self.out_reg = self.out_reg.reshape(self.group_num, self.neu_num)
        out_shape = self.out_reg.shape
        out_shape_new = list(out_shape)

        if self.fire_type in (2, 3, 4, 7):
            out_shape_new[1] = np.ceil(out_shape_new[1] / 16).astype(int) * 16
        elif self.fire_type == 5:
            out_shape_new[1] = np.ceil(out_shape_new[1] / 64).astype(int) * 64

        out_reg = np.zeros(out_shape_new).astype('int32')
        out_reg[:out_shape[0], :out_shape[1]] = self.out_reg[:, :]

        self.out_reg = out_reg

        # 以同样一个随机数种子连续生成随机数
        self.seed = self.lfsr_vec[-1]

        # 多输出需要校核-----------------------------------------------------------------------------
        _resultList = []
        _Vresult = self.convertPrim2Mem([])  # 多维数组转化成MEM中的存储数组的4Bytes格式

        if self.Row_ck_on:
            inputRealLength = self.neu_num * self.in_row_max  # 待确定
        else:
            inputRealLength = self.neu_num * self.group_num  # 没行流水
            pass
        # print(self.Addr_S_Start, self.S1_out_length)
        # print(self.Addr_V_start, self.Read_V_length)
        # print(self.Addr_Vtheta_start, self.Read_Vtheta_length)
        # print(self.Addr_para, self.S1_para_length)
        # os.system("pause")
        _out_reg = {
            "Model": "Soma",
            "data": _Vresult[0],
            "startInMemory": self.Addr_S_Start,
            "lengthInMemory": self.S1_out_length,
            "inputAddr": self.Addr_Uin_start,
            "inputRealLength": inputRealLength,
            "mem_sel": self.mem_sel
        }
        _resultList.append(_out_reg)
        _V = {
            "Model": "Soma",
            "data": _Vresult[1],
            "startInMemory": self.Addr_V_start,
            "lengthInMemory": self.Read_V_length,
            "inputAddr": self.Addr_Uin_start,
            "inputRealLength": inputRealLength,
            "mem_sel": 0
        }
        _resultList.append(_V)
        _Vtheta = {
            "Model": "Soma",
            "data": _Vresult[2],
            "startInMemory": self.Addr_Vtheta_start,
            "lengthInMemory": self.Read_Vtheta_length,
            "inputAddr": self.Addr_Uin_start,
            "inputRealLength": inputRealLength,
            "mem_sel": 0
        }
        _resultList.append(_Vtheta)
        _para = {
            "Model": "Soma",
            "data": _Vresult[3],
            "startInMemory": self.Addr_para,
            "lengthInMemory": self.S1_para_length,
            "inputAddr": self.Addr_Uin_start,
            "inputRealLength": inputRealLength,
            "mem_sel": 0
        }
        _resultList.append(_para)
        return _resultList
        # --------------------------------------------------------------------------------------

    def cal_para(self):
        self.lfsr = lfsr(self.Seed)
        self.neuron_length = self.neu_num * self.group_num
        self.out_reg = np.zeros(self.neuron_length)
        self.out_reg1 = self.out_reg.reshape(self.group_num, self.neu_num)
        out_shape1 = self.out_reg1.shape
        out_shape_new1 = list(out_shape1)

        self.Read_Uin_length = self.neuron_length
        self.Read_V_length = self.neuron_length
        self.Read_VM_length = self.VM_len
        self.Read_Vtheta_length = self.Vtheta_len
        # self.S1_V_length = self.neuron_length
        # self.S1_Vtheta_length = self.Vtheta_len
        self.S1_para_length = 4

        if self.fire_type in (0, 1, 6):
            self.S1_out_length = out_shape_new1[0] * out_shape_new1[1]
        elif self.fire_type in (2, 3, 4, 7):
            self.S1_out_length = out_shape_new1[0] * math.ceil(out_shape_new1[1] / 16)*4
        elif self.fire_type == 5:
            self.S1_out_length = out_shape_new1[0] * math.ceil(out_shape_new1[1] / 64)*4
        else:
            pass
        if self.Row_ck_on:
            self.Addr_Uin_end = self.Addr_Uin_start + int(self.Read_Uin_length / self.group_num * self.in_row_max)
        else:
            self.Addr_Uin_end = self.Addr_Uin_start + self.Read_Uin_length
        pass

    def save_para(self, SIMPATH, TBNAME):

        path = SIMPATH + TBNAME
        isExists = os.path.exists(path)

        if not isExists:
            os.mkdir(path)

        with open(path + '/para.txt', 'w') as f:
            f.write('PI_S.PIC = ' + str(hex(self.PIC)) + '\n')
            f.write('PI_S.PIC_Mode = ' + str(hex(self.PIC_Mode)) + '\n')
            f.write('PI_S.reset_Addr_Uin = ' + str(self.reset_Addr_Uin) + '\n')
            f.write('PI_S.reset_Addr_V = ' + str(self.reset_Addr_V) + '\n')
            f.write('PI_S.reset_Addr_S = ' + str(self.reset_Addr_S) + '\n')
            f.write('PI_S.reset_Addr_VM = ' + str(self.reset_Addr_VM) + '\n')
            f.write('PI_S.reset_Addr_Vtheta = ' + str(self.reset_Addr_Vtheta) + '\n')
            f.write('PI_S.Tw_en = ' + str(int(self.Tw_en)) + '\n')
            f.write('PI_S.Addr_Uin_start = ' + str(hex(self.Addr_Uin_start >> 2)) + '\n')
            f.write('PI_S.Addr_Uin_end = ' + str(hex(((self.Addr_Uin_start + self.Read_Uin_length) >> 2) - 1)) + '\n')
            f.write('PI_S.Addr_S_Start = ' + str(hex(self.Addr_S_Start >> 2)) + '\n')
            f.write('PI_S.Addr_S_end = ' + str(hex(((self.Addr_S_Start + self.S1_out_length) >> 2) - 1)) + '\n')
            f.write('PI_S.Addr_V_start = ' + str(hex(self.Addr_V_start >> 2)) + '\n')
            f.write('PI_S.Addr_V_end = ' + str(hex(((self.Addr_V_start + self.Read_V_length) >> 2) - 1)) + '\n')
            f.write('PI_S.neu_num = ' + str((self.neu_num // 4) - 1) + '\n')
            f.write('PI_S.Tw_len = ' + str(self.Tw_len) + '\n')
            f.write('PI_S.Y_num = ' + str(self.group_num - 1) + '\n')
            f.write('PI_S.Addr_VM_start = ' + str(hex(self.Addr_VM_start >> 2)) + '\n')
            f.write('PI_S.Addr_VM_end = ' + str(hex(((self.Addr_VM_start + self.Read_VM_length) >> 2) - 1)) + '\n')
            f.write('PI_S.Addr_Vtheta_start = ' + str(hex(self.Addr_Vtheta_start >> 2)) + '\n')
            f.write('PI_S.Addr_Vtheta_end = ' + str(hex(((self.Addr_Vtheta_start + self.Read_Vtheta_length) >> 2) - 1)) + '\n')
            f.write('PI_S.Vinit = ' + str(self.Vinit) + '\n')
            f.write('PI_S.Rst_mode = ' + str(self.Rst_mode) + '\n')
            f.write('PI_S.fire_type = ' + str(self.fire_type) + '\n')
            f.write('PI_S.Vth_adpt_en = ' + str(int(self.Vth_adpt_en)) + '\n')
            f.write('PI_S.Vleaky_adpt_en = ' + str(int(self.Vleaky_adpt_en)) + '\n')
            f.write('PI_S.Addr_para = ' + str(hex(self.Addr_para >> 2)) + '\n')
            f.write('PI_S.in_cut_start = ' + str(self.in_cut_start) + '\n')
            f.write('PI_S.in_row_max = ' + str(self.in_row_max) + '\n')
            f.write('******************************' + '\n')
            f.write('P_LIF.Seed = ' + str(self.Seed) + '\n')
            f.write('P_LIF.Vth0 = ' + str(self.Vth0) + '\n')
            f.write('P_LIF.Vth_alpha = ' + str(self.Vth_alpha) + '\n')
            f.write('P_LIF.Vth_beta = ' + str(self.Vth_beta) + '\n')
            f.write('P_LIF.Vth_Incre = ' + str(self.Vth_Incre) + '\n')
            f.write('P_LIF.VR = ' + str(self.VR) + '\n')
            f.write('P_LIF.VL = ' + str(self.VL) + '\n')
            f.write('P_LIF.Vleaky_alpha = ' + str(self.Vleaky_alpha) + '\n')
            f.write('P_LIF.Vleaky_beta = ' + str(self.Vleaky_beta) + '\n')
            f.write('P_LIF.dV = ' + str(self.dV) + '\n')
            f.write('P_LIF.Ref_len = ' + str(self.Ref_len) + '\n')
            f.write('P_LIF.Tw_cnt = ' + str(self.Tw_cnt) + '\n')
            f.write('******************************' + '\n')
            f.write('Read_Uin_length = ' + str(self.Read_Uin_length) + '\n')
            f.write('Read_V_length = ' + str(self.Read_V_length) + '\n')
            f.write('Read_VM_length = ' + str(self.Read_VM_length) + '\n')
            f.write('Read_Vtheta_length = ' + str(self.Read_Vtheta_length) + '\n')
            f.write('S1_out_length = ' + str(self.S1_out_length) + '\n')
            f.write('S1_V_length = ' + str(self.Read_V_length) + '\n')
            f.write('S1_Vtheta_length = ' + str(self.Read_Vtheta_length) + '\n')
            f.write('S1_para_length = ' + str(self.S1_para_length) + '\n')

        with open(path + '/VM.txt', 'w') as f:
            for i in range(self.VM_len):
                final_string = hex_to_string(self.VM_para[i], width=8)
                f.write(final_string)
                f.write('\n')
                pass

        with open(path + '/VM_d.txt', 'w') as f:
            for i in range(self.VM_len):
                f.write(str(self.VM_para[i]))
                f.write('\n')
                pass

        with open(path + '/Vtheta.txt', 'w') as f:
            for i in range(self.Vtheta_len):
                final_string = hex_to_string(self.Vtheta_para[i], width=8)
                f.write(final_string)
                f.write('\n')
                pass

        with open(path + '/Vtheta_d.txt', 'w') as f:
            for i in range(self.Vtheta_len):
                f.write(str(self.Vtheta_para[i]))
                f.write('\n')
                pass

        with open(path + '/Uin.txt', 'w') as f:
            for i in range(self.neuron_length):
                final_string = hex_to_string(self.Uin[i], width=8)
                f.write(final_string)
                f.write('\n')
                pass

        with open(path + '/Uin_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.Uin[i]))
                f.write('\n')
                pass

        V_ref_cnt = (self.ref_cnt.astype('int32') << 28) | (self.V & (2 ** 28 - 1))
        with open(path + '/V.txt', 'w') as f:
            for i in range(self.neuron_length):
                final_string = hex_to_string(V_ref_cnt[i], width=8)
                f.write(final_string)
                f.write('\n')
                pass

        with open(path + '/V_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.V[i]))
                f.write('\n')
                pass

        with open(path + '/ref_cnt_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.ref_cnt[i]))
                f.write('\n')
                pass

    def save_results(self, SIMPATH, TBNAME, t):
        path = SIMPATH + TBNAME

        out_shape = self.out_reg.shape

        V_ref_cnt = (self.ref_cnt.astype('int32') << 28) | (self.V & (2 ** 28 - 1))

        with open(path + '/out_file@' + str(t) + '.txt', 'w') as f:
            # Out_put
            if self.fire_type in (0, 1, 6):
                for i in range(out_shape[0]):
                    for j in range(out_shape[1]):
                        final_string = hex_to_string(self.out_reg[i][j], width=8)
                        f.write(final_string)
                        f.write('\n')

            elif self.fire_type in (2, 3, 4, 7):
                for i in range(out_shape[0]):
                    for j in range(out_shape[1] // 4):
                        final_string = ''
                        for k in range(4):
                            final_string = hex_to_string(self.out_reg[i][j * 4 + k]) + final_string
                        f.write(final_string)
                        f.write('\n')

            elif self.fire_type == 5:
                for i in range(out_shape[0]):
                    for j in range(out_shape[1] // 16):
                        final_string = ''
                        for k in range(8):
                            data_4b = ((self.out_reg[i][j * 16 + k * 2 + 1] & 0x3) << 2) | (self.out_reg[i][j * 16 + k * 2] & 0x3)
                            final_string = hex_to_string(data_4b, width=1) + final_string
                        f.write(final_string)
                        f.write('\n')
            # V
            for i in range(self.neuron_length):
                final_string = hex_to_string(V_ref_cnt[i], width=8)
                f.write(final_string)
                f.write('\n')

            # Vtheta
            for i in range(self.Vtheta_len):
                final_string = hex_to_string(self.Vtheta[i], width=8)
                f.write(final_string)
                f.write('\n')

            # para
            final_string = hex_to_string(self.seed, width=8)
            f.write(final_string)
            f.write('\n')
            data = ((self.Vth0 & 0xffffff) << 8) | (self.Tw_cnt & 0xff)
            final_string = hex_to_string(data, width=8)
            f.write(final_string)
            f.write('\n')
            data = ((self.Vth_beta & 0xfffff) << 12) | ((self.Vth_alpha & 0xff) << 4) | (self.Vth0 >> 24) & 0xf
            final_string = hex_to_string(data, width=8)
            f.write(final_string)
            f.write('\n')
            data = ((self.Vth_Incre & 0xffffff) << 8) | ((self.Vth_beta >> 20) & 0xff)
            final_string = hex_to_string(data, width=8)
            f.write(final_string)
            f.write('\n')

        with open(path + '/out_file@' + str(t) + '_d.txt', 'w') as f:
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    f.write(str(self.out_reg[i][j]))
                    f.write('\n')

        with open(path + '/V@' + str(t) + '_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.V[i]))
                f.write('\n')
                pass

        with open(path + '/ref_cnt@' + str(t) + '_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.ref_cnt[i]))
                f.write('\n')
                pass

        with open(path + '/Vtheta@' + str(t) + '_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.Vtheta[i]))
                f.write('\n')
                pass

        with open(path + '/lfsr_vec@' + str(t) + '_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.lfsr_vec[i]))
                f.write('\n')
                pass

        with open(path + '/Vth@' + str(t) + '_d.txt', 'w') as f:
            for i in range(self.neuron_length):
                f.write(str(self.Vth[i]))
                f.write('\n')
