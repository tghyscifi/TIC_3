#!/usr/bin/env python
# coding: utf-8

from generator.code_generator import PrimitiveJson
from primitive.primitive import PrimitiveType

from primitive.Prim_41_Axon_CNN0_new import Prim_41_Axon
from primitive.Prim_81_Axon_CNN1_new import Prim_81_Axon
from primitive.Prim_02_Axon_avgpooling_new import Prim_02_Axon
from primitive.Prim_03_Axon_new import Prim_03_Axon
from primitive.Prim_43_Axon_new import Prim_43_Axon
from primitive.Prim_83_Axon_new import Prim_83_Axon
from primitive.Prim_04_Axon_MLP_new import Prim_04_Axon
from primitive.Prim_X5_Soma_compare_new import Prim_X5_Soma
from primitive.Prim_06_move_merge_new import Prim_06_move_merge
from primitive.Prim_06_move_split_new import Prim_06_move_split
from primitive.Prim_07_LUT_new import Prim_07_LUT
from primitive.Prim_08_lif_new import Prim_08_lif


class PIGroup:
    """
    对一组原语的抽象类，可添加Axon/Soma1/Router/Soma2四条原语，同时作为一个phase添加到core中。
    """

    def __init__(self):
        self.axon = None
        self.soma1 = None
        self.router = None
        self.soma2 = None

        self.outputList = []
        """
        [
            #first section
            {
                "start":num,
                "length":num
            }
            #second section
            {
                "start":num,
                "length":num
            }
        ]
        """

        self.para_table = {
            "axon": None,
            "soma1": None,
            "router": None,
            "soma2": None
        }

        # 醉了
        self.json = {
            "A_valid": False,
            "S1_valid": False,
            "R_valid": False,
            "S2_valid": False,
            "PI_parameter":
                [{}, {}, {}, {}],
            "Additional":
                [{}, {}, {}, {}],
            "Addr":
                [{}, {}, {}, {}],
            "Data":
                [{}, {}, {}, {}]
        }

        self.routerHeadList = []  # mem section格式,仅当R_valid=true时有内容

        self.router_desc = {}

    def add_output_section(self, address: int, length: int):
        cnt = 0
        for i in range(self.outputList.__len__()):
            if self.outputList[i]["start"] > address:
                break
            else:
                cnt = cnt + 1
        _section = {"start": int(address), "length": int(length)}
        self.outputList.insert(cnt, _section)

    def add_phase_output_print(self, address: int, length: int):
        """
        添加一个需要打印到output_x@x.txt中的段落信息
        :param address:打印内存段落的起始地址
        :param length: 打印内存段落的长度
        :return: 无
        """
        if address < 0x4000:
            if address + length > 0x8000:
                self.add_output_section(address, 0x4000 - address)
                self.add_output_section(0x4000, 0x8000 - 0x4000)
                self.add_output_section(0x8000, (address + length) - 0x8000)
            elif address + length > 0x4000:
                self.add_output_section(address, 0x4000 - address)
                self.add_output_section(0x4000, (address + length) - 0x4000)
            else:
                self.add_output_section(address, length)
        elif address < 0x8000:
            if address + length > 0x8000:
                self.add_output_section(address, 0x8000 - address)
                self.add_output_section(0x8000, (address + length) - 0x8000)
            else:
                self.add_output_section(address, length)
        else:
            self.add_output_section(address, length)

    def add_PI(self, primitive, primitive_type: PrimitiveType):
        if primitive is None:
            return
        primitive.cal_para()
        if primitive_type == PrimitiveType.Axon:
            PrimitiveJson.set_primitive_json(primitive, self.json)
            self.axon = primitive
        elif primitive_type == PrimitiveType.Soma1:
            PrimitiveJson.set_primitive_json(primitive, self.json, index=1)
            self.soma1 = primitive
        elif primitive_type == PrimitiveType.Soma2:
            PrimitiveJson.set_primitive_json(primitive, self.json, index=2)
            self.soma2 = primitive
        elif primitive_type == PrimitiveType.Router:
            PrimitiveJson.set_primitive_json(primitive, self.json)
            self.router = primitive
            if primitive.Send_en:
                RHead_length = int((primitive.Addr_Rhead_length + 1) << 2)
                RHead = primitive.RHead2Mem()
                _len = min(256 * 4, max(RHead_length, len(RHead)))
                primitive.Addr_Rhead_length = (_len >> 2) - 1
                data = []
                if len(RHead) > _len:
                    data = RHead[0:_len]
                elif len(RHead) < _len:
                    tmp = []
                    for i in range(_len - len(RHead)):
                        tmp.append([0])
                    data = RHead + tmp
                else:
                    data = RHead
                self.routerHeadList.append(
                    {"start": int(primitive.Addr_Rhead_base + 0x8000),
                     "length": RHead_length,
                     "data": data})
            self.router_desc.update(primitive.router_desc)
            if primitive.Send_PI_en:
                self.routerHeadList.append(
                    {"start": int(primitive.Send_PI_addr_base*4 + 0x8000),
                     "length": len(primitive.instant_pi_in_memory),
                     "data": primitive.instant_pi_in_memory})

    def print_para(self):
        self.config_json_output()
        flag = [prim is not None for prim in [
            self.axon, self.soma1, self.router, self.soma2]]
        for k in range(4):
            if flag[k]:
                para_str_list = []
                for i in self.json["PI_parameter"][k].keys():
                    para_str_list.append(
                        i + " = " + str(self.json["PI_parameter"][k][i]) + "\n")
                para_str_list.append("************************\n")
                for i in self.json["Additional"][k].keys():
                    para_str_list.append(
                        i + " = " + str(self.json["Additional"][k][i]) + "\n")
                self.para_table[list(self.para_table.keys())[
                    k]] = para_str_list

                para_str_list.append("************************\n")
                for i in self.json["Addr"][k].keys():
                    para_str_list.append(
                        i + " = " + str(hex(self.json["Addr"][k][i])) + "\n")
                self.para_table[list(self.para_table.keys())[
                    k]] = para_str_list

    def config_json_output(self):
        json_config = {
            "A_valid": False,
            "S1_valid": False,
            "R_valid": False,
            "S2_valid": False,
            "PI_parameter":
                [{}, {}, {}, {}],
            "Additional":
                [{}, {}, {}, {}],
            "Additional":
                [{}, {}, {}, {}],
            "Data":
                [{}, {}, {}, {}]
        }

        # TODO: 执行会改变原语参数，比如self.Tw_cnt = self.Tw_cnt + 1
        # PrimitiveJson.set_primitive_json(self.axon, json_config)
        # PrimitiveJson.set_primitive_json(self.soma1, json_config, 1)
        # PrimitiveJson.set_primitive_json(self.router, json_config)
        # PrimitiveJson.set_primitive_json(self.soma2, json_config, 2)

        if self.json["S1_valid"] == True:
            if self.json["PI_parameter"][1]["Row_ck_on"] == 1:
                if self.json["A_valid"] == True:
                    PIC = self.json["PI_parameter"][1]["PIC"]
                    if PIC == 0x05:
                        end = self.json["PI_parameter"][1]["Addr_X_End"]
                    elif PIC == 0x06:
                        if self.json["PI_parameter"][1]["in_ciso_pipe_sel"] == 0:
                            end = self.json["PI_parameter"][1]["Addr_end_in"]
                        else:
                            end = self.json["PI_parameter"][1]["Addr_end_ciso"]
                    elif PIC == 0x07:
                        end = self.json["PI_parameter"][1]["Addr_X_End"]
                    elif PIC == 0x08:
                        end = self.json["PI_parameter"][1]["PI_S.Addr_Uin_end"]
                        pass
                    self.json["PI_parameter"][0]["Addr_V_end"] = end
        # Soma-Router
        if self.json["R_valid"] == True:
            if self.json["PI_parameter"][2]["Soma_in_en"] == 1:
                if self.json["S1_valid"] == True:
                    end = ((0x8000 + self.json["PI_parameter"][2]["Addr_Dout_base"] * 4 + (
                            self.json["PI_parameter"][2]["Addr_Dout_length"] + 1) * 4) >> 2) - 1

                    PIC = self.json["PI_parameter"][1]["PIC"]
                    if PIC == 0x05:
                        self.json["PI_parameter"][1]["Addr_Y_End"] = int(end)
                    elif PIC == 0x06:
                        if self.json["PI_parameter"][1]["PIC_Mode"] == 1 and \
                                self.json["PI_parameter"][1]["out_ciso_sel"] == 1:
                            self.json["PI_parameter"][1]["Addr_end_ciso"] = int(
                                end)
                        else:
                            self.json["PI_parameter"][1]["Addr_end_out"] = int(
                                end)
                    elif PIC == 0x07:
                        self.json["PI_parameter"][1]["Addr_Y_End"] = int(end)
                    elif PIC == 0x08:
                        self.json["PI_parameter"][1]["PI_S.Addr_S_end"] = int(
                            end)
                        pass

        return self.json
