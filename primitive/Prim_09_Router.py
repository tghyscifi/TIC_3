import numpy as np
import copy
import math

from primitive.primitive import Primitive
import generator.functions.data_generator as data_generator
from generator.functions.util import hex_to_string


class Prim_09_Router(Primitive):

    def __init__(self):
        super().__init__()

        self.PIC = 0x09
        self.Rhead_mode = 0
        self.CXY = 0b00
        self.Send_en = 0
        self.Receive_en = 0
        self.Addr_Dout_base = 0
        self.Dout_Mem_sel = 0
        self.Addr_Dout_length = 0
        self.Send_number = 0
        self.Addr_Rhead_base = 0
        self.Addr_Rhead_length = 0
        self.Addr_Din_base = 0
        self.Addr_Din_length = 0
        self.Receive_number = 0
        self.Nx = 0
        self.Ny = 0
        self.Send_PI_en = 0
        self.Back_sign_en = 0
        self.Send_PI_num = 0
        self.Receive_sign_num = 0
        self.Send_PI_addr_base = 0
        self.Relay_number = 0
        self.Q = 0
        self.Receive_sign_en = 0
        self.T_mode = 0
        self.Soma_in_en = 0

        self.send_destin_core_grp = []
        self.recv_source_core_grp = []
        self.instant_prim_request = []
        self.instant_request_back = []

        self.RHeadList = []
        self.router_desc = {}
        self.instant_pi = []
        self.instant_pi_in_memory = []

    def __str__(self):
        return "09(Router)"

    def add_instant_pi(self, PI_addr_offset, A_valid, S1_valid, R_valid, S2_valid, X, Y, Q):
        """
        PI_addr_offset(7bit):0-128
        A_valid(1bit):0/1
        S1_valid(1bit):0/1
        R_valid(1bit):0/1
        S2_valid(1bit):0/1
        X(8bit):-128-127
        Y(8bit):-128-127
        Q(1bit):0/1
        """
        self.instant_pi.append({"PI_addr_offset": PI_addr_offset, "A_valid": A_valid,
                                "S1_valid": S1_valid, "R_valid": R_valid, "S2_valid": S2_valid, "X": X, "Y": Y, "Q": Q})
        self.instant_pi_in_memory.append([((PI_addr_offset & 0b1111111) << 25 | (A_valid & 0b1) << 24 | (S1_valid & 0b1) << 23 | (
            R_valid & 0b1) << 22 | (S2_valid & 0b1) << 21 | (X & 0xff) << 13 | (Y & 0xff) << 5 | (Q & 0b1) << 4) & 0xfffffff0])

    def init_data(self):
        # self.attributes = copy.deepcopy(self.__dict__)
        Router_Dout = []
        if self.Soma_in_en == 0:
            for i in range((self.Addr_Dout_length + 1) * 4):
                tmp = []
                for j in range(4):
                    tmp.append(np.random.randint(200))
                Router_Dout.append(tmp)
            self.memory_blocks = [
                {'name': 'Router_Dout',
                 'start': self.Addr_Dout_base + 0x8000,
                 'length': (self.Addr_Dout_length + 1) * 4,
                 'data': Router_Dout,
                 'mode': 0},
            ]
        return Router_Dout

    def RHead2Mem(self) -> list:
        _retList = []
        for i in range(self.RHeadList.__len__()):
            _RHead = self.RHeadList[i]
            if self.Rhead_mode == 1:  # 一个包头发多次
                _Byte4 = (_RHead["pack_per_Rhead"] & 0xfff) << 20 | (
                    _RHead["A_offset"] & 0xfff) << 8 | (_RHead["Const"] & 0x7f) << 1 | (_RHead["EN"] & 1)
                _tmp = [_Byte4]
                _retList.append(_tmp)
                pass  # self.Rhead_mode==1

            _Byte4 = (_RHead["S"] & 1) << 31 | (_RHead["T"] & 1) << 30 | (_RHead["P"] & 1) << 29 | (_RHead["Q"] & 1) << 28 | (_RHead["X"] & 0xff) << 20 | (
                _RHead["Y"] & 0xff) << 12 | (_RHead["A"] & 0xfff)
            _tmp = [_Byte4]
            _retList.append(_tmp)
            pass  # for i in range(self.RHeadList.__len__())
        return _retList
        pass  # func RHead2Mem

    def addRHead(self, S: int, T: int, P: int, Q: int, X: int, Y: int, A: int, pack_per_Rhead: int = None, A_offset: int = None, Const: int = None, EN: int = None):
        """
        添加一个路由包头。包头的存储顺序与addRHead函数调用顺序一致。
        :param S:
        :param T:
        :param P:
        :param Q:
        :param X:
        :param Y:
        :param A:
        :param pack_per_Rhead:
        :param A_offset:
        :param Const:
        :param EN:
        :return:
        """
        if pack_per_Rhead is None and A_offset is None and Const is None and EN is None:
            _head = {"S": S, "T": T, "P": 1, "Q": Q, "X": X, "Y": Y, "A": A}
            if self.router_desc.get((X, Y)):
                self.router_desc[(X, Y)]["num"] += 1
            else:
                self.router_desc[(X, Y)] = {"num": 1, "T": T, "RHead": 0}
        else:
            _head = {"S": S, "T": T, "P": 0, "Q": Q, "X": X, "Y": Y, "A": A,
                     "pack_per_Rhead": pack_per_Rhead, "A_offset": A_offset, "Const": Const, "EN": EN}
            if self.router_desc.get((X, Y)):
                self.router_desc[(X, Y)]["num"] += pack_per_Rhead+1
            else:
                self.router_desc[(X, Y)] = {
                    "num": pack_per_Rhead+1, "T": T, "RHead": 1}
        self.RHeadList.append(_head)

    def routerSend(self, selfCoreHandle):
        send_data = {}

        if self.Send_en == 1:
            if len(selfCoreHandle.coreMemory.mem3) != 0 and self.Dout_Mem_sel != 1:
                raise Exception("soma1 输出到mem3, 但router的Dout_Mem_sel ！= 1")
            _RHeadIndex = 0
            _count = 0  # 仅用于记录一个包头发送了多少包
            _curPackPerRhead = 0
            _preA = 0
            for _sendNum in range(self.Send_number + 1):
                _rawhead = self.RHeadList[_RHeadIndex]

                if self.Rhead_mode == 1:  # 一个包头发多次

                    _curPackPerRhead = _rawhead["pack_per_Rhead"] + 1
                    _curA_offset = _rawhead["A_offset"] + 1
                    _curConst = _rawhead["Const"]
                    _curEN = _rawhead["EN"]

                    if _rawhead["EN"] == 0:
                        _count += 1
                        if _count == _curPackPerRhead:
                            _RHeadIndex += 1
                            _count = 0
                        continue

                    _rawA = _rawhead["A"]
                    if (_count != 0) and (_count % (_curConst + 1) == 0):
                        _A = _preA + _curA_offset
                        _preA = _A
                    elif _count == 0:
                        _preA = _rawA
                        _A = _preA
                    else:
                        _A = _preA + 1
                        _preA = _A
                    _P = 0
                    if _count + 1 == _curPackPerRhead:
                        _P = 1
                    _head = {"S": _rawhead["S"], "T": _rawhead["T"], "P": _P,
                             "Q": _rawhead["Q"], "X": _rawhead["X"], "Y": _rawhead["Y"], "A": _A}
                else:  # 一个包头发一次
                    _curPackPerRhead = 1
                    _head = _rawhead

                # print(self.Addr_Dout_length)
                _data = []
                if self.Dout_Mem_sel == 0:  # mem2
                    if self.T_mode == 0:  # 单包
                        _byteNumin4B = _sendNum % 4
                        if self.Soma_in_en == 1:
                            _DoutAddr = self.Addr_Dout_base + \
                                int(math.floor(_sendNum / 4)) + \
                                0x8000  # 相对于mem2的地址转化为绝对地址
                        else:
                            _DoutAddr = self.Addr_Dout_base + int(math.floor(_sendNum / 4) % (
                                (self.Addr_Dout_length + 1) << 2)) + 0x8000  # 相对于mem2的地址转化为绝对地址
                        _data4B = copy.deepcopy(
                            selfCoreHandle.coreMemory.readMemSection(_DoutAddr, 1))
                        _data = _data4B[0][_byteNumin4B]
                    elif self.T_mode == 1:  # 多包
                        if self.Soma_in_en == 1:
                            _DoutAddr = self.Addr_Dout_base + \
                                int(int((_sendNum * 8 / 4))) + \
                                0x8000  # 相对于mem2的地址转化为绝对地址
                        else:
                            _DoutAddr = self.Addr_Dout_base + int(int((_sendNum * 8 / 4)) % (
                                (self.Addr_Dout_length + 1) << 2)) + 0x8000  # 相对于mem2的地址转化为绝对地址
                        _data = copy.deepcopy(
                            selfCoreHandle.coreMemory.readMemSection(_DoutAddr, 2))
                else:  # mem3
                    if self.T_mode == 0:  # 单包
                        _byteNumin4B = _sendNum % 4
                        _data4B = copy.deepcopy(
                            selfCoreHandle.coreMemory.mem3[0][int(_sendNum / 4)])
                        _data = _data4B[_byteNumin4B]
                    elif self.T_mode == 1:  # 多包
                        _data = copy.deepcopy([selfCoreHandle.coreMemory.mem3[0][int(
                            _sendNum * 8 / 4)], selfCoreHandle.coreMemory.mem3[0][int(_sendNum * 8 / 4) + 1]])

                _sendPack = {}
                _sendPack.update(_head)
                _sendPack.update({"data": _data})

                _dx = _rawhead["X"]
                _dy = _rawhead["Y"]
                destination_abs_x = selfCoreHandle.x + \
                    selfCoreHandle.chip_id[0]*16 + _dx
                destination_abs_y = selfCoreHandle.y + \
                    selfCoreHandle.chip_id[1]*10 + _dy

                dest_core_id = (((destination_abs_x//16), (destination_abs_y//10)),
                                ((destination_abs_x % 16), (destination_abs_y % 10)))
                if dest_core_id in send_data:
                    send_data[dest_core_id].append(
                        copy.deepcopy(_sendPack))
                else:
                    send_data[dest_core_id] = [
                        copy.deepcopy(_sendPack)]

                _count += 1
                if _count == _curPackPerRhead:
                    _RHeadIndex += 1
                    _count = 0

            if (self.Soma_in_en == 1) and (self.Dout_Mem_sel == 0):
                info = {
                    "Model": "Router",
                    "row_ck_on": 1,
                    "inputAddr": int(0x8000 + self.Addr_Dout_base),
                    "inputRealLength": int((self.Addr_Dout_length + 1) * 4),
                    "data": [[0]]
                }
                selfCoreHandle.coreMemory.writeMem([info])
            if self.Dout_Mem_sel == 1:
                selfCoreHandle.coreMemory.mem3.clear()
        if self.CXY == 0b01 or self.CXY == 0b10:
            destination_abs_x = selfCoreHandle.x + \
                selfCoreHandle.chip_id[0]*16 + self.Nx
            destination_abs_y = selfCoreHandle.y + \
                selfCoreHandle.chip_id[1]*10 + self.Ny
            dest_chip_x = destination_abs_x//16
            dest_chip_y = destination_abs_y//16
            dest_core_in_dest_chip_dx = destination_abs_x % 16
            dest_core_in_dest_chip_dy = destination_abs_y % 16
            send_data["multicast_relay"] = (
                self.CXY, self.Relay_number+1, ((dest_chip_x, dest_chip_y), (dest_core_in_dest_chip_dx, dest_core_in_dest_chip_dy)))
        send_data["multicast"] = []
        for send_info in self.send_destin_core_grp:
            core_id = send_info['core_id']
            if not isinstance(core_id, list):
                continue
            for i in range(len(core_id)-1):
                send_data["multicast"].append(
                    (core_id[i], 1, send_info['data_num'], core_id[i+1]))
        return send_data

    def get_instant_packets(self, selfCoreHandle):
        instant_index = None
        if self.Send_PI_en:
            instant_index = {selfCoreHandle.id: self.instant_prim_request}
        return instant_index

    def routerRecv(self, selfCoreHandle, packets):
        if self.Receive_en == 1:
            _recvNum = 0
            if packets == None:
                raise Exception(str("{0}) has no received any packets").format(
                    selfCoreHandle.id))
            for source in packets:
                for i in source:
                    _recvPack = copy.deepcopy(i)
                    _T_mode = _recvPack["T"]
                    if _recvPack["P"] == 1:
                        _recvNum += 1

                    if _T_mode == 1:  # 收到的是多包
                        _A = _recvPack["A"]
                        _absInAddr = int(self.Addr_Din_base + (_A * 8 / 4) %
                                         ((self.Addr_Din_length + 1)*2) + 0x8000)
                        _data = _recvPack["data"]
                        selfCoreHandle.coreMemory.writeMem(
                            [{"Model": "Router", "startInMemory": _absInAddr, "lengthInMemory": 2, "data": _data}])
                    else:  # 收到的是单包
                        _A = _recvPack["A"]
                        _byteNumin4B = _A % 4
                        _absInAddr = int(self.Addr_Din_base + (_A / 4) %
                                         ((self.Addr_Din_length + 1)*2) + 0x8000)
                        _data = _recvPack["data"]
                        selfCoreHandle.coreMemory.modifyByte(
                            _absInAddr, _byteNumin4B, _data)

                    if _recvNum == self.Receive_number + 1:
                        break
            if _recvNum != self.Receive_number + 1:
                raise Exception(str("{0}) 设置的receive_number与实际接收到的包头数不匹配").format(
                    selfCoreHandle.id))

    def cal_para(self):
        pass  # func cal_para

    def save_para(self, path: str):

        with open(path, 'w') as f:
            f.write("PIC = " + str(int(self.PIC)) + "\n")
            f.write("Rhead_mode = " + str(int(self.Rhead_mode)) + "\n")
            f.write("CXY = " + str(int(self.CXY)) + "\n")
            f.write("Send_en = " + str(int(self.Send_en)) + "\n")
            f.write("Receive_en = " + str(int(self.Receive_en)) + "\n")
            f.write("Dout_Mem_sel = " + str(int(self.Dout_Mem_sel)) + "\n")
            f.write("Addr_Dout_base = " +
                    str(hex(int(self.Addr_Dout_base >> 2))) + "\n")
            f.write("Addr_Dout_length = " +
                    str(int(self.Addr_Dout_length)) + "\n")
            f.write("Addr_Rhead_base = " +
                    str(hex(int(self.Addr_Rhead_base >> 2))) + "\n")
            f.write("Addr_Rhead_length = " +
                    str(int(self.Addr_Rhead_length)) + "\n")
            f.write("Addr_Din_base = " +
                    str(hex(int(self.Addr_Din_base >> 1))) + "\n")
            f.write("Addr_Din_length = " +
                    str(int(self.Addr_Din_length)) + "\n")
            f.write("Send_number = " + str(int(self.Send_number)) + "\n")
            f.write("Receive_number = " + str(int(self.Receive_number)) + "\n")
            f.write("Nx = " + str(int(self.Nx)) + "\n")
            f.write("Ny = " + str(int(self.Ny)) + "\n")
            f.write("Send_PI_en = " + str(int(self.Send_PI_en)) + "\n")
            f.write("Back_Sign_en = " + str(int(self.Back_sign_en)) + "\n")
            f.write("Send_PI_num = " + str(int(self.Send_PI_num)) + "\n")
            f.write("Receive_sign_num = " +
                    str(int(self.Receive_sign_num)) + "\n")
            f.write("Send_PI_addr_base = " +
                    str(int(self.Send_PI_addr_base)) + "\n")
            f.write("Relay_number = " + str(int(self.Relay_number)) + "\n")
            f.write("Q = " + str(int(self.Q)) + "\n")
            f.write("T_mode = " + str(int(self.T_mode)) + "\n")
            f.write("Receive_sign_en = " +
                    str(int(self.Receive_sign_en)) + "\n")
            f.write("Soma_in_en = " + str(int(self.Soma_in_en)) + "\n")
        pass  # func save_para

    pass  # class Prim_09_Router(object)

# r1 = Prim_09_Router()
# r1.addRHead(0, 1, 1, 1, 0, -1, 0)
# print(r1.RHead2Mem())
#
# print((-1) & 0xff)

# print(math.floor(5 / 4))
# print(0 % 0)
