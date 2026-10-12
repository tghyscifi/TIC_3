from generator.functions.util import hex_to_string
import copy
import os


class Memory(object):
    def __init__(self):
        self.memory = []
        '''
        [
            {
                "data": Data,
                "start":start,
                "length":length
            }
        ]
        '''
        self.outputBuffer = []
        '''
        # Axon返回的数据格式
        [
            {
                "Model":"Axon",
                "data": Data,
                "startInMemory":startAddr,
                "lengthInMemory":length
                "inputAddr":inputAddr,
                "inputRealLength":realLength
            }
        ]
        # Soma返回的数据格式
        [
            {
                "Model":"Soma",
                "data": Data,
                "startInMemory":startAddr,
                "lengthInMemory":length,
                "inputAddr":inputAddr,
                "inputRealLength":realLength
            }
        ]
        '''
        _zero = []
        for i in range(0x9000):
            tmp = [0, 0, 0, 0]
            _zero.append(tmp)
            pass

        self.memory.append({"start": 0, "length": 0x9000, "data": _zero})

        self.mem3 = []

        pass

    @staticmethod
    def MemFormat(data: list, numPer4B: int, is_int8=False) -> list:
        _retList = []
        _formatMem = Memory.format_4B(data)
        for i in range(_formatMem.__len__()):
            if numPer4B == 1:
                tmp = 0
                for j in range(4):
                    tmp |= int((_formatMem[i][j] & 0xff) << (j * 8))
                    pass
                if tmp > 0x7fffffff:
                    tmp -= 0x100000000
                    pass
                _retList.append([int(tmp)])
                pass  # if _formatMem[i].__len__() == 1
            elif numPer4B == 4:
                if not is_int8:
                    _retList.append(_formatMem[i])
                else:
                    _tmp = []
                    for k in range(4):
                        tmp = _formatMem[i][k]
                        if _formatMem[i][k] > 127:
                            tmp -= 256
                        _tmp.append(tmp)
                    _retList.append(_tmp)
                pass  # elif _formatMem[i].__len__() == 16
            elif numPer4B == 16:
                tmp = []
                for j in range(4):
                    for k in range(4):
                        _tmp = int((_formatMem[i][j] >> (k * 2)) & 0b11)
                        if _tmp == 3:
                            _tmp = -1
                            pass
                        tmp.append(_tmp)
                        pass
                    pass
                _retList.append(tmp)
                pass  # elif _formatMem[i].__len__() == 16
            pass  # for i in range(_formatMem.__len__())
        return copy.deepcopy(_retList)
        pass  # func readMemFormat

    @staticmethod
    def format_4B(data: list) -> list:
        _retList = []
        for i in range(data.__len__()):
            if data[i].__len__() == 1:
                tmp = []
                for j in range(4):
                    tmp.append(int((data[i][0] >> (j * 8)) & 0xff))
                    pass
                _retList.append(tmp)
                pass  # if data[i].__len__() == 1
            elif data[i].__len__() == 4:
                _retList.append(data[i])
                pass  # data[i].__len__() == 4
            elif data[i].__len__() == 16:
                tmp = []
                for j in range(4):
                    _tmp = 0
                    for k in range(4):
                        _tmp |= int((data[i][j * 4 + k] & 0b11) << (k * 2))
                        pass
                    tmp.append(int(_tmp))
                    pass
                _retList.append(tmp)
                pass  # elif data[i].__len__() == 16
            pass  # for i in range(data.__len__())
        return copy.deepcopy(_retList)
        pass  # func Format_4B

    @staticmethod
    def Memto4BStringList(memData: list) -> list:
        _stringList = []

        for k in range(memData.__len__()):
            if memData[k].__len__() == 1:
                _tmp = ""
                _tmp = hex_to_string(memData[k][0], width=8) + "\n" + _tmp
                _stringList.append(_tmp)
                pass
            elif memData[k].__len__() == 4:
                _tmp = ""
                for n in range(4):
                    _tmp = hex_to_string(memData[k][n], width=2) + _tmp
                    pass
                _tmp += "\n"
                _stringList.append(_tmp)
                pass
            elif memData[k].__len__() == 16:
                _tmp = ""
                for n in range(8):
                    _tmp = hex_to_string(
                        ((memData[k][n * 2 + 1] & 0x3) << 2) | (memData[k][n * 2] & 0x3), width=1) + _tmp
                    pass
                _tmp += "\n"
                _stringList.append(_tmp)
                pass
            pass
        return _stringList
        pass  # func Mem2StringList

    def addMemoryBlock(self, argStart: int, argLength: int, argData: list) -> bool:
        """
        :description addMemoryBlock:初始化数据，添加数据块
        :param argStart:起始地址，位宽4Bytes
        :param argLength:长度
        :param argData:数据数组，应满足argData.__len__()==argLength
        :return:true/false:返回是否添加成功
        """
        # _argData = copy.deepcopy(argData)
        # _block = {
        #     "start": argStart,
        #     "length": argLength,
        #     "data": _argData
        # }
        # _newIndex = 0
        # for i in range(self.memory.__len__()):
        #
        #     if argStart < self.memory[_newIndex]["start"]:
        #         break
        #         pass  # if argStart < self.memory[j]["start"]
        #     _newIndex = _newIndex + 1
        #     pass  # for i in range(self.memory.__len__())
        # self.memory.insert(_newIndex, _block)
        # self.__mergeMemory__()

        _argData = copy.deepcopy(argData)
        _block = {
            "startInMemory": argStart,
            "lengthInMemory": argLength,
            "data": _argData
        }
        # print(_block)
        # os.system("pause")
        self.outputBuffer.append(_block)
        self.updateMemory()
        # print(argData)
        # _data = Memory.format_4B(argData)
        # print(_data)
        # print(self.memory[0])
        #
        # print(argStart, argLength)
        # print(self.outputBuffer.__len__())
        # if self.outputBuffer.__len__()==1:
        #     print(self.outputBuffer[0])
        #     pass
        # os.system("pause")
        # for i in range(argLength):
        #     for j in range(4):
        #         self.memory[0]["data"][argStart + i][j] = _data[i][j]
        #         pass
        #     pass

        return True
        pass  # func addMemoryBlock

    def readMemSection(self, start: int, length: int) -> list:
        return self.read_memory([{"start": start, "length": length}])[0]
        pass  # func readMemSection

    def read_memory(self, argSectionList: list) -> list:
        """
        :description read_memory:读mem，通过参数sectionList索引地址，通过返回值返回数据
        :param argSectionList:格式说明
                        [
                            {
                                "start":start,
                                "length":length
                            }
                        ]
        :return:返回从参数argSectionList描述的各地址处读取的数据，type=list，
                数据顺序与argSectionList顺序相同，即returnList[i]等于argSectionList[i]["start"]处的数据，
                长度returnList[i].__len__()==argSectionList[i]["length"]
        """
        _outData = []
        for i in range(argSectionList.__len__()):
            _addr = argSectionList[i]["start"]
            _len = argSectionList[i]["length"]  # 需要读取的数据长度
            _data = []
            # _data = self.__readData__(_addr, _len)

            if argSectionList[i].get("type") is None:
                _data = self.__readData__(_addr, _len)
                pass
            else:
                if argSectionList[i]["type"] == 0:
                    _data = Memory.MemFormat(self.__readData__(_addr, _len), 1)
                    pass
                elif argSectionList[i]["type"] == 1:
                    _data = Memory.MemFormat(self.__readData__(_addr, _len), 4, True)
                    pass
                elif argSectionList[i]["type"] == 2:
                    _data = Memory.MemFormat(self.__readData__(_addr, _len), 4)
                    pass
                elif argSectionList[i]["type"] == 3:
                    _data = Memory.MemFormat(self.__readData__(_addr, _len), 16)
                    pass
                pass
            # print(_data)
            _outData.append(copy.deepcopy(_data))
            pass  # for i in range(argSectionList.__len__())
        # os.system("pause")
        return _outData
        pass  # func read_memory

    def readMemList(self, argSectionList: list):
        return self.read_memory(argSectionList)

    def writeMem(self, SectionList: list):
        """
        :description writeMem:写数据至outputBuffer中(Axon和Soma单独处理，格式不同)，
                            根据Soma的信息，更新Axon输出数据的地址长度
        :param argSectionList:原语输出的数据列表，格式参考self.outputBuffer元素格式
        :return: None
        """

        argSectionList = copy.deepcopy(SectionList)
        for i in range(argSectionList.__len__()):
            argSectionList[i]["data"] = Memory.format_4B(SectionList[i]["data"])
            # print(SectionList[i]["data"])
            # print(Memory.format_4B(SectionList[i]["data"]))
            # os.system("pause")
            pass

        for i in range(argSectionList.__len__()):
            # 如果这个包不是router输出的（因为router输出不带数据，所以不会添加route的data到outputbuffer）
            if argSectionList[i].get("row_ck_on") == None:
                # 如果是soma输出到mem3的，则不添加到outputbuffer中
                if (argSectionList[i]["Model"] == "Soma") and (argSectionList[i]["mem_sel"] == 1):
                    self.mem3.append(argSectionList[i]["data"])
                    pass
                else:
                    _newIndex = 0
                    for j in range(self.outputBuffer.__len__()):
                        if argSectionList[i]["startInMemory"] < self.outputBuffer[_newIndex]["startInMemory"]:
                            break
                            pass  # if argStart < self.memory[j]["start"]
                        _newIndex += 1
                        pass  # for j in range(self.memory.__len__())
                    self.outputBuffer.insert(_newIndex, argSectionList[i])
                    pass
                pass

            if argSectionList[i]["Model"] == "Soma":
                # print("Soma")
                _realLen = argSectionList[i]["inputRealLength"]
                for j in range(self.outputBuffer.__len__()):
                    # print("self.outputBuffer[j][\"startInMemory\"] =", self.outputBuffer[j]["startInMemory"])
                    # print("   argSectionList[i][\"inputAddr\"]     =", argSectionList[i]["inputAddr"])
                    if self.outputBuffer[j]["startInMemory"] == argSectionList[i]["inputAddr"]:
                        # print(_realLen, self.outputBuffer[j])
                        # os.system("pause")
                        self.outputBuffer[j]["lengthInMemory"] = _realLen
                        # print("_realLen =", _realLen)
                        pass
                    pass  # for j in range(self.outputBuffer.__len__())
                pass  # if argSectionList[i]["Model"]=="Soma"

            # 如果有router输出的结构，说明有mem2的流水，mem3交互流水并不输出结构
            if argSectionList[i]["Model"] == "Router":
                # print("Router")
                if argSectionList[i].get("row_ck_on") != None:
                    _realLen = argSectionList[i]["inputRealLength"]
                    # print(_realLen,argSectionList[i]["inputAddr"])
                    for j in range(self.outputBuffer.__len__()):
                        # print("self.outputBuffer[j][\"startInMemory\"] =", self.outputBuffer[j]["startInMemory"])
                        # print("self.outputBuffer[j][\"lengthInMemory\"]     =", self.outputBuffer[j]["lengthInMemory"])
                        if self.outputBuffer[j]["startInMemory"] == argSectionList[i]["inputAddr"]:
                            self.outputBuffer[j]["lengthInMemory"] = _realLen
                            # print("_realLen =", _realLen)
                            pass
                        pass  # for j in range(self.outputBuffer.__len__())
                    pass
                # os.system("pause")
                pass  # if argSectionList[i]["Model"]=="Router"

            pass  # for i in range(argSectionList.__len__())
        pass  # func writeMem

    def modifyByte(self, address: int, _byteNumin4B: int, value: int):
        _findNum = -1
        for i in range(self.memory.__len__()):
            if (address >= self.memory[i]["start"]) and (address < self.memory[i]["start"] + self.memory[i]["length"]):
                _index = address - self.memory[i]["start"]
                if self.memory[i]["data"][_index].__len__() == 4:
                    self.memory[i]["data"][_index][_byteNumin4B] = value
                    _findNum += 1
                    pass
                pass
            pass
        for i in range(self.outputBuffer.__len__()):
            if (address >= self.outputBuffer[i]["startInMemory"]) and (address < self.outputBuffer[i]["startInMemory"] + self.outputBuffer[i]["lengthInMemory"]):
                _index = address - self.outputBuffer[i]["startInMemory"]
                if self.outputBuffer[i]["data"][_index].__len__() == 4:
                    self.outputBuffer[i]["data"][_index][_byteNumin4B] = value
                    _findNum += 1
                    pass
                pass
            pass

        if _findNum == -1:
            _tmp = [[0, 0, 0, 0]]
            _tmp[0][_byteNumin4B] = value
            self.addMemoryBlock(address, 1, _tmp)
            pass
        pass  # func modifyByte

    def updateMemory(self):
        """
        :description updateMemory:在phase中所有原语执行完成后(待定)，
                                将self.outputBuffer中的数据更新到self.memory中
        :return: None
        """
        for index in range(self.outputBuffer.__len__()):
            # print(self.outputBuffer.__len__())
            _rawData = self.outputBuffer[index]["data"]
            _start = self.outputBuffer[index]["startInMemory"]
            _length = self.outputBuffer[index]["lengthInMemory"]
            _finalData = []
            for j in range(_rawData.__len__()):
                if _finalData.__len__() < _length:
                    _finalData.append(_rawData[j])
                    pass
                else:
                    _finalData[j % _finalData.__len__()] = _rawData[j]
                    pass
                pass

            # 36863
            self.__writeMemory__(_start, _length, _finalData)

            pass  # for index in range(self.outputBuffer.__len__())
        self.outputBuffer.clear()
        pass  # func updateMemory

    def __readData__(self, Start: int, Length: int) -> list:
        _curLen = 0  # 当前读取到的长度
        _tmpBuffer = []
        _start = Start
        _len = Length
        while _curLen != _len:
            _searchResult = self.__searchOutputBuffer__(_start, _len)
            # print(_searchResult["shape"])
            if _searchResult["index"] != -1:
                _mem = self.outputBuffer[_searchResult["index"]]
                if _searchResult["shape"] == 0:
                    _tmpBuffer = _tmpBuffer + _mem["data"]
                    _curLen += _mem["lengthInMemory"]
                    pass
                elif _searchResult["shape"] == 1:
                    if _searchResult["isBegin"]:
                        _tmpBuffer = _tmpBuffer + _mem["data"][0:_len]
                        _curLen += _len

                        # _insertIndex = _searchResult["index"]
                        # _insertData = _tmpBuffer
                        # _insertAddr = _start
                        # _insertLen = _len
                        # self.outputBuffer.insert(_insertIndex,
                        #                          {"data": _insertData, "startInMemory": _insertAddr,
                        #                           "lengthInMemory": _insertLen})
                        #
                        # _mem["data"] = _mem["data"][_len:_mem["data"].__len__()]
                        # _mem["startInMemory"] = _mem["startInMemory"] + _len
                        # _mem["lengthInMemory"] = _mem["lengthInMemory"] - _len
                        pass
                    else:
                        if _start + _len == _mem["startInMemory"] + _mem["lengthInMemory"]:
                            _tmpBuffer = _tmpBuffer + \
                                _mem["data"][(-_len): _mem["lengthInMemory"]]
                            _curLen += _len

                            # _insertIndex = _searchResult["index"] + 1
                            # _insertData = _tmpBuffer
                            # _insertAddr = _start
                            # _insertLen = _len
                            # self.outputBuffer.insert(_insertIndex,
                            #                          {"data": _insertData, "startInMemory": _insertAddr,
                            #                           "lengthInMemory": _insertLen})
                            # _mem["data"] = _mem["data"][0:(_mem["lengthInMemory"] - _len)]
                            # _mem["lengthInMemory"] = _mem["lengthInMemory"] - _len
                            pass
                        else:
                            _tmpBuffer = _tmpBuffer + _mem["data"][(_start - _mem["startInMemory"]): (
                                _start - _mem["startInMemory"] + _len)]
                            _curLen += _len

                            # _insertIndex = _searchResult["index"] + 1
                            # _insertData = _tmpBuffer
                            # _insertAddr = _start
                            # _insertLen = _len
                            # self.outputBuffer.insert(_insertIndex,
                            #                          {"data": _insertData, "startInMemory": _insertAddr,
                            #                           "lengthInMemory": _insertLen})
                            #
                            # _insertIndex = _searchResult["index"] + 2
                            # _insertData = _mem["data"][(_start + _len - _mem["startInMemory"]):_mem["lengthInMemory"]]
                            # _insertAddr = _start + _len
                            # _insertLen = _mem["lengthInMemory"] - (_start + _len - _mem["startInMemory"])
                            # self.outputBuffer.insert(_insertIndex,
                            #                          {"data": _insertData, "startInMemory": _insertAddr,
                            #                           "lengthInMemory": _insertLen})
                            #
                            # _mem["data"] = _mem["data"][0:(_start - _mem["startInMemory"])]
                            # _mem["lengthInMemory"] = _mem["lengthInMemory"] - _len

                            pass
                        pass
                    pass  # elif _searchResult["shape"]==1
                elif _searchResult["shape"] == 2:
                    _tmpBuffer = _tmpBuffer + _mem["data"]
                    _curLen += _mem["lengthInMemory"]
                    _start = _start + _mem["lengthInMemory"]
                    _newlen = Length - _mem["lengthInMemory"]
                    _tmpBuffer += self.__readData__(_start, _newlen)
                    _curLen += _newlen
                    pass  # elif _searchResult["shape"]==2
                elif _searchResult["shape"] == 3:
                    # print(_start - _mem["startInMemory"])

                    _newbuffer = _mem["data"][(
                        _start - _mem["startInMemory"]): _mem["lengthInMemory"]]
                    _tmpBuffer = _tmpBuffer + _newbuffer
                    _curLen += _mem["lengthInMemory"] - \
                        (_start - _mem["startInMemory"])

                    # print("_curLen = ", _curLen)
                    # _insertIndex = _searchResult["index"] + 1
                    # _insertData = _newbuffer
                    # _insertAddr = _start
                    # _insertLen = _mem["lengthInMemory"] - (_start - _mem["startInMemory"])
                    # self.outputBuffer.insert(_insertIndex,
                    #                          {"data": _insertData, "startInMemory": _insertAddr,
                    #                           "lengthInMemory": _insertLen})
                    # _mem["data"] = _mem["data"][0:(_mem["lengthInMemory"] - _insertLen)]
                    # _mem["lengthInMemory"] = _mem["lengthInMemory"] - _insertLen

                    _start = _start + _curLen
                    _newlen = Length - _curLen
                    _tmpBuffer += self.__readData__(_start, _newlen)
                    _curLen += _newlen
                    pass  # elif _searchResult["shape"]==3
                pass  # if _searchResult["index"]!=-1
            else:
                _searchResult = self.__searchMemory__(_start, _len)

                # print("index :", (_searchResult["index"]), "shape :", (_searchResult["shape"]))
                if _searchResult["index"] != -1:
                    _mem = self.memory[_searchResult["index"]]
                    if _searchResult["shape"] == 0:
                        _tmpBuffer = _tmpBuffer + _mem["data"]
                        _curLen += _mem["length"]
                        pass
                    elif _searchResult["shape"] == 1:
                        # print("isBegin =", _searchResult["isBegin"])
                        if _searchResult["isBegin"]:
                            _tmpBuffer = _tmpBuffer + _mem["data"][0:_len]
                            _curLen += _len

                            # _insertIndex = _searchResult["index"]
                            # _insertData = _tmpBuffer
                            # _insertAddr = _start
                            # _insertLen = _len
                            # self.memory.insert(_insertIndex, {"start": _insertAddr, "length": _insertLen, "data": _insertData})
                            #
                            # _mem["data"] = _mem["data"][_len:_mem["data"].__len__()]
                            # _mem["start"] = _mem["start"] + _len
                            # _mem["length"] = _mem["length"] - _len
                            pass
                        else:
                            # print("_start =", _start, "_len =", _len, "_memstart =", _mem["start"], "_memlen =", _mem["length"])
                            if _start + _len == _mem["start"] + _mem["length"]:

                                _tmpBuffer = _tmpBuffer + \
                                    _mem["data"][(-_len): _mem["length"]]
                                _curLen += _len

                                # _insertIndex = _searchResult["index"] + 1
                                # _insertData = _tmpBuffer
                                # _insertAddr = _start
                                # _insertLen = _len
                                # self.memory.insert(_insertIndex, {"start": _insertAddr, "length": _insertLen, "data": _insertData})
                                # _mem["data"] = _mem["data"][0:(_mem["length"] - _len)]
                                # _mem["length"] = _mem["length"] - _len
                                pass
                            else:
                                _tmpBuffer = _tmpBuffer + \
                                    _mem["data"][(_start - _mem["start"])
                                                  : (_start - _mem["start"] + _len)]
                                _curLen += _len
                                # print(_tmpBuffer)
                                # _insertIndex = _searchResult["index"] + 1
                                # _insertData = _tmpBuffer
                                # _insertAddr = _start
                                # _insertLen = _len
                                # self.memory.insert(_insertIndex, {"start": _insertAddr, "length": _insertLen, "data": _insertData})
                                #
                                # _insertIndex = _searchResult["index"] + 2
                                # _insertData = _mem["data"][(_start + _len - _mem["start"]):_mem["length"]]
                                # _insertAddr = _start + _len
                                # _insertLen = _mem["length"] - (_start + _len - _mem["start"])
                                # self.memory.insert(_insertIndex, {"start": _insertAddr, "length": _insertLen, "data": _insertData})
                                #
                                # _mem["data"] = _mem["data"][0:(_start - _mem["start"])]
                                # _mem["length"] = _mem["length"] - _len - _insertLen

                                pass
                            pass
                        pass
                    elif _searchResult["shape"] == 2:
                        _tmpBuffer = _tmpBuffer + _mem["data"]
                        _curLen += _mem["length"]
                        _start = _start + _mem["length"]
                        _newlen = Length - _mem["length"]
                        _tmpBuffer += self.__readData__(_start, _newlen)
                        _curLen += _newlen
                        pass
                    elif _searchResult["shape"] == 3:
                        # print(_start - _mem["start"])

                        _newbuffer = _mem["data"][(
                            _start - _mem["start"]): _mem["length"]]
                        _tmpBuffer = _tmpBuffer + _newbuffer
                        _curLen += _mem["length"] - (_start - _mem["start"])

                        # print("_curLen = ", _curLen)
                        # _insertIndex = _searchResult["index"] + 1
                        # _insertData = _newbuffer
                        # _insertAddr = _start
                        # _insertLen = _mem["length"] - (_start - _mem["start"])
                        # self.outputBuffer.insert(_insertIndex,
                        #                          {"data": _insertData, "start": _insertAddr,
                        #                           "length": _insertLen})
                        # _mem["data"] = _mem["data"][0:(_mem["length"] - _insertLen)]
                        # _mem["length"] = _mem["length"] - _insertLen

                        _start = _start + _curLen
                        _newlen = Length - _curLen
                        _tmpBuffer += self.__readData__(_start, _newlen)
                        _curLen += _newlen
                        pass
                    pass
                else:
                    # 都没有找到
                    break
                    pass
                pass
            pass  # while _curLen!=_len

        return _tmpBuffer
        pass  # func __readData__

    def __writeMemory__(self, argStart: int, argLength: int, argData: list):
        _searchResult = self.__searchMemory__(argStart, argLength)
        # print("_searchResult[\"shape\"] =",_searchResult["shape"])
        if _searchResult["index"] == -1:  # 未找到，直接添加
            raise Exception("a memory section end address over 0x9000")
            self.addMemoryBlock(argStart, argLength, argData)
            pass  # if _searchResult["index"]==-1
        else:
            if _searchResult["shape"] == 0:
                # print("shape=0")
                self.memory[_searchResult["index"]]["data"] = argData
                pass
            elif _searchResult["shape"] == 1:
                # print("shape=1")
                # print("argStart  = ", argStart)
                # print("argLength = ", argLength)
                self.memory[_searchResult["index"]]["data"][
                    (argStart - self.memory[_searchResult["index"]]["start"]):(
                        argStart - self.memory[_searchResult["index"]]["start"] + argLength)] = argData
                pass
            elif _searchResult["shape"] == 2:
                # print("shape=2")
                self.memory[_searchResult["index"]
                            ]["data"] = argData[0:self.memory[_searchResult["index"]]["length"]]
                self.__writeMemory__(
                    (self.memory[_searchResult["index"]]["start"] +
                     self.memory[_searchResult["index"]]["length"]),
                    (argLength -
                     self.memory[_searchResult["index"]]["length"]),
                    argData[self.memory[_searchResult["index"]]["length"]:argLength])
                pass
            elif _searchResult["shape"] == 3:
                # print("shape=3")
                _curLen = (self.memory[_searchResult["index"]]["length"] - (
                    argStart - self.memory[_searchResult["index"]]["start"]))
                self.memory[_searchResult["index"]]["data"][
                    (argStart - self.memory[_searchResult["index"]]["start"]):(
                        self.memory[_searchResult["index"]]["length"])] = argData[0:_curLen]
                self.__writeMemory__(
                    (self.memory[_searchResult["index"]]["start"] +
                     self.memory[_searchResult["index"]]["length"]),
                    argLength - _curLen, argData[_curLen:argLength])
                pass
            pass
        pass  # func __writeMemory__

    def __searchMemory__(self, startAddr: int, length: int) -> dict:
        """
        :description searchMemory:
        :param length:
        :param startAddr:
        :return:
        {
            "index":,
            "isBegin",
            "shape"
        }
        shape=
        {
            0：完全包含在内，不需要split和merge
            1：完全包含在内，需要split
            2：未完全包含在内，需要merge
            3：未完全包含在内，需要split和merge
        }
        """
        _index = 0
        _pos = False
        _shape = 0
        for j in range(self.memory.__len__()):
            _addr = self.memory[_index]["start"]
            _len = self.memory[_index]["length"]
            _end = _addr + _len - 1
            if (startAddr >= _addr) and (startAddr <= _end):  # 找到了，下标为_index，返回

                _searchEnd = startAddr + length - 1
                if _addr == startAddr:
                    _pos = True
                    if _searchEnd < _end:
                        _shape = 1
                        pass
                    elif _searchEnd == _end:
                        _shape = 0
                        pass
                    else:
                        _shape = 2
                        pass
                    pass
                else:
                    _pos = False
                    if _searchEnd <= _end:
                        _shape = 1
                        pass
                    else:
                        _shape = 3
                        pass
                    pass

                break
                pass  # if startAddr <= _end
            else:
                _index = _index + 1
                pass  # else
            pass  # for j in range(self.memory.__len__())

        if _index == self.memory.__len__():  # 如果未找到，返回-1
            _index = -1
            pass
        _info = {
            "index": _index,
            "isBegin": _pos,
            "shape": _shape
        }
        return _info
        pass  # func searchMemory

    def __searchOutputBuffer__(self, startAddr: int, length: int) -> dict:
        """
        :description searchOutputBuffer:
        :param length:
        :param startAddr:
        :return:
        {
            "index":,
            "isBegin",
            "shape"
        }
        shape=
        {
            0：完全包含在内，不需要split和merge
            1：完全包含在内，需要split
            2：未完全包含在内，需要merge
            3：未完全包含在内，需要split和merge
        }
        """
        _index = 0
        _pos = False
        _shape = 0
        for j in range(self.outputBuffer.__len__()):
            _addr = self.outputBuffer[_index]["startInMemory"]
            _len = self.outputBuffer[_index]["lengthInMemory"]
            _end = _addr + _len - 1
            if (startAddr >= _addr) and (startAddr <= _end):  # 找到了，下标为_index，返回

                _searchEnd = startAddr + length - 1
                if _addr == startAddr:
                    _pos = True
                    if _searchEnd < _end:
                        _shape = 1
                        pass
                    elif _searchEnd == _end:
                        _shape = 0
                        pass
                    else:
                        _shape = 2
                        pass
                    pass
                else:
                    _pos = False
                    if _searchEnd <= _end:
                        _shape = 1
                        pass
                    else:
                        _shape = 3
                        pass
                    pass

                break
                pass  # if startAddr <= _end
            else:
                _index = _index + 1
                pass  # else
            pass  # for j in range(self.outputBuffer.__len__())

        if _index == self.outputBuffer.__len__():  # 如果未找到，返回-1
            _index = -1
            pass
        _info = {
            "index": _index,
            "isBegin": _pos,
            "shape": _shape
        }
        return _info
        pass  # func searchOutputBuffer

    def __mergeMemory__(self):
        if self.memory.__len__() >= 2:
            _popNum = 0
            _rawLen = self.memory.__len__()
            for i in range(1, _rawLen):
                _cur = i - _popNum
                _curStart = self.memory[_cur]["start"]
                _preStart = self.memory[_cur - 1]["start"]
                _prelen = self.memory[_cur - 1]["length"]
                if _preStart + _prelen == _curStart:
                    self.memory[_cur - 1]["data"] += self.memory[_cur]["data"]
                    self.memory[_cur -
                                1]["length"] += self.memory[_cur]["length"]
                    self.memory.remove(self.memory[_cur])
                    _popNum += 1
                    pass  # if _preStart + _prelen == _curStart
                pass  # for i in range(1,self.memory.__len__())
            pass  # if self.memory.__len__() >= 2
        pass  # func __mergeMemory__

    pass

# a = [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
# b = [[2], [2], [2], [2], [2]]
# print(a)
# a[1:3] = b
# print(a)

# m1 = Memory()
# m1.outputBuffer.append(
#     {
#         "Model": "Axon",
#         "data": [[1, 22, 37, 4], [9, 6, 9, 8], [41, 42, 43, 44], [301, 311, 321, 331]],
#         "startInMemory": 503,
#         "lengthInMemory": 4,
#     }
# )
# m1.Memory.append(
#     {
#         "data": [[51, 52, 53, 54], [95, 96, 97, 98], [74, 75, 76, 77], [30, 31, 32, 33]],
#         "start": 500,
#         "length": 4,
#     }
# )
# m1.writeMem([
#     {
#         "Model":"Axon",
#         "data": [1, 2, 3, 4, 5, 6],
#         "startInMemory": 503,
#         "lengthInMemory": 6,
#     }
# ])
# m1.writeMem([
#     {
#         "Model":"Axon",
#         "data": [1, 2, 3, 4, 5, 6],
#         "startInMemory": 501,
#         "lengthInMemory": 6,
#     }
# ])
# m1.writeMem([
#     {
#         "Model":"Axon",
#         "data": [1, 2, 3, 4, 5, 6],
#         "startInMemory": 0,
#         "lengthInMemory": 6,
#     }
# ])
# print("memory       : ", m1.Memory)
# print("outputBuffer : ", m1.outputBuffer)
# m1.modifyByte(503, 0, 3)
# m1.modifyByte(428, 3, 7)
# m1.updateMemory()
# print("memory       : ", m1.Memory)
# print("outputBuffer : ", m1.outputBuffer)
# m1.updateMemory()
# print("memory       : ", m1.Memory)
# print("outputBuffer : ", m1.outputBuffer)

# m1 = Memory()
# m2 = Memory()
# m1.Memory.append(
#     {
#         "start": 500,
#         "length": 4,
#         "data": [[51, 52, 53, 54], [95, 96, 97, 98], [74, 75, 76, 77], [30, 31, 32, 33]]
#     }
# )
# m1.Memory.append(
#     {
#         "start": 504,
#         "length": 4,
#         "data": [[51, 52, 53, 54], [95, 96, 97, 98], [74, 75, 76, 77], [30, 31, 32, 33]]
#     }
# )
# m2.addMemoryBlock(500, 4, [[51, 52, 53, 54], [95, 96, 97, 98], [74, 75, 76, 77], [30, 31, 32, 33]])
# m2.addMemoryBlock(504, 4, [[51, 52, 53, 54], [95, 96, 97, 98], [74, 75, 76, 77], [30, 31, 32, 33]])
# print("memory1       : ", m1.Memory)
# print("memory2       : ", m2.Memory)
# m1.__mergeMemory__()
# m2.__mergeMemory__()
# print("memory1       : ", m1.Memory)
# print("memory2       : ", m2.Memory)

# a = {"data": 1}
# print(a.get("data"))
