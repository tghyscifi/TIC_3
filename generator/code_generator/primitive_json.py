import numpy as np

from primitive import Prim_02_Axon, Prim_03_Axon, Prim_04_Axon, \
    Prim_06_move_merge, Prim_06_move_split, Prim_07_LUT, Prim_08_lif, \
    Prim_09_Router, Prim_41_Axon, Prim_43_Axon, Prim_81_Axon, \
    Prim_83_Axon, Prim_X5_Soma


class PrimitiveJson(object):
    @staticmethod
    def set_primitive_json(primitive, json_config, index=1):
        if (isinstance(primitive, Prim_02_Axon)):
            PrimitiveJson.set_prim_02_axon(primitive, json_config)
        elif (isinstance(primitive, Prim_03_Axon)):
            PrimitiveJson.set_prim_03_axon(primitive, json_config)
        elif (isinstance(primitive, Prim_04_Axon)):
            PrimitiveJson.set_prim_04_axon(primitive, json_config)
        elif (isinstance(primitive, Prim_06_move_merge)):
            PrimitiveJson.set_prim_06_move_merge(
                primitive, json_config, index)
        elif (isinstance(primitive, Prim_06_move_split)):
            PrimitiveJson.set_prim_06_move_split(
                primitive, json_config, index)
        elif (isinstance(primitive, Prim_07_LUT)):
            PrimitiveJson.set_prim_07_lut(
                primitive, json_config, index)
        elif (isinstance(primitive, Prim_08_lif)):
            PrimitiveJson.set_prim_08_lif(
                primitive, json_config, index)
        elif (isinstance(primitive, Prim_09_Router)):
            PrimitiveJson.set_prim_09_router(primitive, json_config)
        elif (isinstance(primitive, Prim_41_Axon)):
            PrimitiveJson.set_prim_41_axon(primitive, json_config)
        elif (isinstance(primitive, Prim_43_Axon)):
            PrimitiveJson.set_prim_43_axon(primitive, json_config)
        elif (isinstance(primitive, Prim_81_Axon)):
            PrimitiveJson.set_prim_81_axon(primitive, json_config)
        elif (isinstance(primitive, Prim_83_Axon)):
            PrimitiveJson.set_prim_83_axon(primitive, json_config)
        elif (isinstance(primitive, Prim_X5_Soma)):
            PrimitiveJson.set_prim_x5_soma(
                primitive, json_config, index)
        else:
            return

    @staticmethod
    def set_prim_02_axon(primitive, json_config):
        if primitive.avg_pooling_en and primitive.pad_on:
            _Addr_InA_base = (primitive.Addr_InA_base >> 2) - \
                primitive.Addr_start_offset + 0x8000
        else:
            _Addr_InA_base = (primitive.Addr_InA_base >> 2)
        json_config["PI_parameter"][0] = {
            "PIC": int(primitive.PIC),
            "Reset_Addr_A": int(primitive.Reset_Addr_A),
            "Reset_Addr_V": int(primitive.Reset_Addr_V),
            "InA_type": int(primitive.InA_type),
            "Load_Bias": int(primitive.Load_Bias),
            "Addr_Bias_base": int(primitive.Addr_Bias_base >> 2),
            "Addr_InA_base": int(_Addr_InA_base),
            "Addr_InA_end": int(((primitive.Addr_InA_base + primitive.Read_X_length) >> 2) + primitive.Addr_start_offset - 1),
            "Addr_V_base": int(primitive.Addr_V_base >> 2),
            "Addr_V_end": int((((primitive.Addr_V_base + primitive.Write_V_length) >> 2) - 1)),
            "L1_num": int(primitive.L1_num),
            "L2_num": int(primitive.L2_num),
            "L3_num": int(primitive.L3_num),
            "L4_num": int(primitive.L4_num),
            "L5_num": int(primitive.L5_num),
            "Addr_InA_L1_step": int(primitive.Addr_InA_L1_step),
            "Addr_InA_L2_step": int(primitive.Addr_InA_L2_step),
            "Addr_InA_L3_step": int(primitive.Addr_InA_L3_step),
            "Addr_InA_L4_step": int(primitive.Addr_InA_L4_step),
            "Addr_InA_L5_step": int(primitive.Addr_InA_L5_step),
            "Sx": int(primitive.pooling_Sx),
            "Sy": int(primitive.pooling_Sy),
            "pad_top": int(primitive.pad_top),
            "pad_down": int(primitive.pad_down),
            "pad_left": int(primitive.pad_left),
            "pad_right": int(primitive.pad_right),
            "constant_b": int(primitive.constant_b),
            "A2S2_mode": int(primitive.A2S2_mode)
        }
        json_config["Additional"][0] = {
            "Input_fm_Px": int(primitive.Input_fm_Px),
            "Input_fm_Py": int(primitive.Input_fm_Py),
            "Output_fm_Ox": int(primitive.Output_fm_Ox),
            "Output_fm_Oy": int(primitive.Output_fm_Oy),
            "array_num": int(primitive.array_num),
            "Read_X_length": int(primitive.Read_X_length),
            "Read_Bias_length": int(primitive.Read_Bias_length),
            "Write_V_length": int(primitive.Write_V_length),
        }
        json_config["Addr"][0] = {
            "Addr_InA_base": int(_Addr_InA_base << 2),
            "Addr_InA_end": int(((((primitive.Addr_InA_base + primitive.Read_X_length) >> 2) + primitive.Addr_start_offset) << 2) - 4),
            "Addr_Bias_base": int(primitive.Addr_Bias_base),
            "Addr_V_base": int(primitive.Addr_V_base),
            "Addr_V_end": int((((primitive.Addr_V_base + primitive.Write_V_length))) - 4),
        }
        json_config["A_valid"] = True

        _PI = json_config["PI_parameter"][0]
        _Additional = json_config["Additional"][0]

    @staticmethod
    def set_prim_03_axon(primitive, json_config):
        json_config["PI_parameter"][0] = {
            "PIC": int(primitive.PIC),
            "Reset_Addr_A": int(primitive.Reset_Addr_A),
            "Reset_Addr_V": int(primitive.Reset_Addr_V),
            "InA_type": int(primitive.InA_type),
            "InB_type": int(primitive.InA_type),
            "Load_Bias": int(primitive.Load_Bias),
            "Addr_Bias_base": int(primitive.Addr_Bias_base >> 2),
            "Addr_InA_base": int(primitive.Addr_InA_base >> 2),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X1_length) >> 2) - 1), int(primitive.Addr_InA_base >> 2)),
            "Addr_InB_base": int(primitive.Addr_InB_base >> 2),
            "Addr_V_base": int(primitive.Addr_V_base >> 2),
            "Addr_V_end": max(int((((primitive.Addr_V_base + primitive.Write_V_length) >> 2) - 1)), int(primitive.Addr_V_base >> 2)),
            "L3_num": int(primitive.L3_num),
            "L4_num": int(primitive.L4_num),
            "L5_num": int(primitive.L5_num),
            "Addr_InA_L3_step": int(primitive.Addr_InA_L3_step),
            "Addr_InA_L4_step": int(primitive.Addr_InA_L4_step),
            "Addr_InA_L5_step": int(primitive.Addr_InA_L5_step),
            "constant_b": int(primitive.constant_b),
            "A2S2_mode": int(primitive.A2S2_mode)
        }
        json_config["Additional"][0] = {
            "Px": int(primitive.Px),
            "Py": int(primitive.Py),
            "Ox": int(primitive.Ox),
            "Oy": int(primitive.Oy),
            "Read_X1_length": int(primitive.Read_X1_length),
            "Read_X2_length": int(primitive.Read_X2_length),
            "Read_Bias_length": int(primitive.Read_Bias_length),
            "Write_V_length": int(primitive.Write_V_length),
        }
        json_config["Addr"][0] = {
            "Addr_InA_base": int(primitive.Addr_InA_base),
            "Addr_InA_end": int((primitive.Addr_InA_base + primitive.Read_X1_length) - 4),
            "Addr_InB_base": int(primitive.Addr_InB_base),
            "Addr_Bias_base": int(primitive.Addr_Bias_base),
            "Addr_V_base": int(primitive.Addr_V_base),
            "Addr_V_end": int((primitive.Addr_V_base + primitive.Write_V_length) - 4),
        }
        json_config["A_valid"] = True

        _PI = json_config["PI_parameter"][0]
        _Additional = json_config["Additional"][0]

    @staticmethod
    def set_prim_04_axon(primitive, json_config):
        json_config["PI_parameter"][0] = {
            "PIC": int(primitive.PIC),
            "Reset_Addr_A": int(primitive.Reset_Addr_A),
            "Reset_Addr_V": int(primitive.Reset_Addr_V),
            "InA_type": int(primitive.InA_type),
            "InB_type": int(primitive.InB_type),
            "Load_Bias": int(primitive.Load_Bias),
            "Addr_Bias_base": int(primitive.Addr_Bias_base >> 2),
            "Addr_InA_base": int(primitive.Addr_InA_base >> 2),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length) >> 2) - 1), int(primitive.Addr_InA_base >> 2)),
            "Addr_InB_base": int(primitive.Addr_InB_base >> 2),
            "Addr_V_base": int(primitive.Addr_V_base >> 2),
            "Addr_V_end": max(int((((primitive.Addr_V_base + primitive.Write_V_length) >> 2) - 1)), int(primitive.Addr_V_base >> 2)),
            "L0_num": int(primitive.L0_num),
            "L3_num": int(primitive.L3_num),
            "L0_num_in_last_row": int(primitive.L0_num_in_last_row),
            "Addr_InA_L3_step": int(primitive.Addr_InA_L3_step),
            "constant_b": int(primitive.constant_b),
            "A2S2_mode": int(primitive.A2S2_mode)
        }
        json_config["Additional"][0] = {
            "Read_X_length": int(primitive.Read_X_length),
            "Read_Bias_length": int(primitive.Read_Bias_length),
            "Read_weight_length": int(primitive.Read_weight_length),
            "Write_V_length": int(primitive.Write_V_length),
        }
        json_config["Addr"][0] = {
            "Addr_InA_base": int(primitive.Addr_InA_base),
            "Addr_InA_end": int(((primitive.Addr_InA_base + primitive.Read_X_length)) - 4),
            "Addr_InB_base": int(primitive.Addr_InB_base),
            "Addr_Bias_base": int(primitive.Addr_Bias_base),
            "Addr_V_base": int(primitive.Addr_V_base),
            "Addr_V_end": int((((primitive.Addr_V_base + primitive.Write_V_length))) - 4),
        }
        json_config["A_valid"] = True

    @staticmethod
    def set_prim_06_move_merge(primitive, json_config, index):
        json_config["PI_parameter"][2 * index - 1] = {
            "PIC": int((primitive.PIC)),
            "PIC_Mode": int((primitive.PIC_Mode)),
            "Reset_Addr_in": int(primitive.Reset_Addr_in),
            "Reset_Addr_out": int(primitive.Reset_Addr_out),
            "Reset_Addr_ciso": int(primitive.Reset_Addr_ciso),
            "Row_ck_on": int(primitive.Row_ck_on),
            "Addr_Start_in": int((primitive.Addr_Start_in >> 2)),
            "Addr_end_in": max(int(((primitive.Addr_end_in >> 2) - 1)), int((primitive.Addr_Start_in >> 2))),
            "Addr_Start_out": int((primitive.Addr_Start_out >> 2)),
            "Addr_end_out": max(int(
                (((primitive.Addr_Start_out + primitive.Write_Y_length) >> 2) - 1)), int((primitive.Addr_Start_out >> 2))),
            "Addr_Start_ciso": int((primitive.Addr_Start_ciso >> 2)),
            "Addr_end_ciso": max(int(((primitive.Addr_end_ciso >> 2) - 1)), int((primitive.Addr_Start_ciso >> 2))),
            "in_row_max": int(max(primitive.in_row_max - 1, 0)),
            "Km_num_in": int(max(primitive.Km_num_in - 1, 0)),
            "Km_num_ciso": int(max(primitive.Km_num_ciso - 1, 0)),
            "Km_num_out": int(max(primitive.Km_num_out - 1, 0)),
            "num_in": int(max(primitive.num_in - 1, 0)),
            "num_ciso": int(max(primitive.num_ciso - 1, 0)),
            "num_out": int(max(primitive.num_out - 1, 0)),
            "type_in": int(primitive.type_in),
            "type_out": int(primitive.type_out),
            "in_cut_start": int(primitive.in_cut_start),
            "mem_sel": int(primitive.mem_sel),
            "in_ciso_pipe_sel": int(primitive.in_ciso_pipe_sel)
        }
        json_config["Additional"][2 * index - 1] = {
            "Read_in_length": int(primitive.Read_in_length),
            "Read_ciso_length": int(primitive.Read_ciso_length),
            "Write_out_length": int(primitive.Write_Y_length)
        }
        json_config["Addr"][2 * index - 1] = {
            "Addr_Start_in": int((primitive.Addr_Start_in)),
            "Addr_end_in": int(((primitive.Addr_end_in) - 4)),
            "Addr_Start_out": int((primitive.Addr_Start_out)),
            "Addr_end_out": max(int((((primitive.Addr_Start_out + primitive.Write_Y_length))-4)), int((primitive.Addr_Start_out))),
            "Addr_Start_ciso": int((primitive.Addr_Start_ciso)),
            "Addr_end_ciso": max(int(((primitive.Addr_end_ciso) - 4)), int((primitive.Addr_Start_ciso))),
        }
        json_config["S" + str(index) + "_valid"] = True

        _PI = json_config["PI_parameter"][2 * index - 1]
        _Additional = json_config["Additional"][2 * index - 1]

    @staticmethod
    def set_prim_06_move_split(primitive, json_config, index):
        Addr_end_ciso = int(
            (((primitive.Addr_Start_ciso + primitive.Write_ciso_length) >> 2) - 1))
        Addr_end_out = int(
            (((primitive.Addr_Start_out + primitive.Write_Y_length) >> 2) - 1))

        if hasattr(primitive, "set_ciso_addr_end"):
            if primitive.set_ciso_addr_end:
                Addr_end_ciso = int(primitive.Addr_end_ciso >> 2)
            if primitive.set_out_addr_end:
                Addr_end_out = int(primitive.Addr_end_out >> 2)

        json_config["PI_parameter"][2 * index - 1] = {
            "PIC": int((primitive.PIC)),
            "PIC_Mode": int((primitive.PIC_Mode)),
            "Reset_Addr_in": int(primitive.Reset_Addr_in),
            "Reset_Addr_out": int(primitive.Reset_Addr_out),
            "Reset_Addr_ciso": int(primitive.Reset_Addr_ciso),
            "Row_ck_on": int(primitive.Row_ck_on),
            "Addr_Start_in": int((primitive.Addr_Start_in >> 2)),
            "Addr_end_in": int(max(((primitive.Addr_end_in >> 2) - 1), primitive.Addr_Start_in >> 2)),
            "Addr_Start_out": int((primitive.Addr_Start_out >> 2)),
            "Addr_end_out": max(Addr_end_out, int((primitive.Addr_Start_out >> 2))),
            "Addr_Start_ciso": int((primitive.Addr_Start_ciso >> 2)),
            "Addr_end_ciso": max(Addr_end_ciso, int((primitive.Addr_Start_ciso >> 2))),
            "in_row_max": int(max(primitive.in_row_max - 1, 0)),
            "Km_num_in": int(max(primitive.Km_num_in - 1, 0)),
            "Km_num_ciso": int(max(primitive.Km_num_ciso - 1, 0)),
            "Km_num_out": int(max(primitive.Km_num_out - 1, 0)),
            "num_in": int(max(primitive.num_in - 1, 0)),
            "num_ciso": int(max(primitive.num_ciso - 1, 0)),
            "num_out": int(max(primitive.num_out - 1, 0)),
            "type_in": int(primitive.type_in),
            "type_out": int(primitive.type_out),
            "in_cut_start": int(primitive.in_cut_start),
            "mem_sel": int(primitive.mem_sel),
            "out_ciso_sel": int(primitive.out_ciso_sel),
            "in_ciso_pipe_sel": 0
        }
        json_config["Additional"][2 * index - 1] = {
            "Read_in_length": int(primitive.Read_in_length),
            "Write_ciso_length": int(primitive.Write_ciso_length),
            "Write_out_length": int(primitive.Write_Y_length)
        }
        json_config["Addr"][2 * index - 1] = {
            "Addr_Start_in": int((primitive.Addr_Start_in)),
            "Addr_end_in": int(max(((primitive.Addr_end_in) - 4), primitive.Addr_Start_in)),
            "Addr_Start_out": int((primitive.Addr_Start_out)),
            "Addr_end_out": max(int((((primitive.Addr_Start_out + primitive.Write_Y_length)) - 4)), int((primitive.Addr_Start_out))),
            "Addr_Start_ciso": int((primitive.Addr_Start_ciso)),
            "Addr_end_ciso": max(int((((primitive.Addr_Start_ciso + primitive.Write_ciso_length)) - 4)), int((primitive.Addr_Start_ciso))),
        }
        json_config["S" + str(index) + "_valid"] = True

        _PI = json_config["PI_parameter"][2 * index - 1]
        _Additional = json_config["Additional"][2 * index - 1]

    @staticmethod
    def set_prim_07_lut(primitive, json_config, index):
        x_len = int(np.ceil(primitive.Read_X_length / 4) * 4)
        y_len = int(np.ceil(primitive.Write_Y_length / 4) * 4)
        json_config["PI_parameter"][2 * index - 1] = {
            "PIC": int((primitive.PIC)),
            "PIC_Mode": int((primitive.PIC_Mode)),
            "Reset_Addr_X": int(primitive.reset_Addr_X),
            "Reset_Addr_Y": int(primitive.reset_Addr_Y),
            "Row_ck_on": int(primitive.Row_ck_on),
            "X_type": int(primitive.X_type),
            "Y_type": int(primitive.Y_type),
            "Addr_X_Start": int(primitive.Addr_X_Start >> 2),
            "Addr_X_End": max(int((primitive.Addr_X_end >> 2) - 1), int(primitive.Addr_X_Start >> 2)),
            "Addr_Y_Start": int((primitive.Addr_Start_out >> 2)),
            "Addr_Y_End": max(int(((primitive.Addr_Start_out + y_len) >> 2) - 1), int((primitive.Addr_Start_out >> 2))),
            "neu_num": int(primitive.neu_num - 1),
            "Y_num": int(primitive.group_num - 1),
            "Addr_LUT_start": int(primitive.Addr_LUT_Start >> 2),
            "LUT_DW": int(primitive.LUT_DW),
            "X_cut_start": int(primitive.X_cut_start),
            "in_row_max": int(max(primitive.in_row_max - 1, 0)),
            "mem_sel": int(primitive.mem_sel)
        }
        json_config["Additional"][2 * index - 1] = {
            "Read_X_length": int(primitive.Read_X_length),
            "Read_LUT_length": int(primitive.Read_LUT_length),
            "Write_Y_length": int(primitive.Write_Y_length)
        }
        json_config["Addr"][2 * index - 1] = {
            "Addr_X_Start": int(primitive.Addr_X_Start),
            "Addr_X_End": int((primitive.Addr_X_end) - 4),
            "Addr_LUT_start": int(primitive.Addr_LUT_Start),
            "Addr_Y_Start": int((primitive.Addr_Start_out)),
            "Addr_Y_End": int(((primitive.Addr_Start_out + y_len)) - 4),
        }
        json_config["S" + str(index) + "_valid"] = True

        _PI = json_config["PI_parameter"][2 * index - 1]
        _Additional = json_config["Additional"][2 * index - 1]

    @staticmethod
    def set_prim_08_lif(primitive, json_config, index):
        json_config["PI_parameter"][2 * index - 1] = {
            "PIC": int(primitive.PIC),
            "PI_S.PIC_Mode": int(primitive.PIC_Mode),
            "PI_S.reset_Addr_Uin": int(primitive.reset_Addr_Uin),
            "PI_S.reset_Addr_V": int(primitive.reset_Addr_V),
            "PI_S.reset_Addr_S": int(primitive.reset_Addr_S),
            "PI_S.reset_Addr_VM": int(primitive.reset_Addr_VM),
            "PI_S.reset_Addr_Vtheta": int(primitive.reset_Addr_Vtheta),
            "PI_S.Tw_en": int(int(primitive.Tw_en)),
            "PI_S.Addr_Uin_start": int((primitive.Addr_Uin_start >> 2)),
            "PI_S.Addr_Uin_end": max(int(((max(primitive.Addr_Uin_end, primitive.Addr_Uin_start) >> 2) - 1)), int((primitive.Addr_Uin_start >> 2))),
            "PI_S.Addr_S_Start": int((primitive.Addr_S_Start >> 2)),
            "PI_S.Addr_S_end": max(int((((primitive.Addr_S_Start + primitive.S1_out_length) >> 2) - 1)), int((primitive.Addr_S_Start >> 2))),
            "PI_S.Addr_V_start": int((primitive.Addr_V_start >> 2)),
            "PI_S.Addr_V_end": max(int((((primitive.Addr_V_start + primitive.Read_V_length) >> 2) - 1)), int((primitive.Addr_V_start >> 2))),
            "PI_S.Addr_VM_start": int((primitive.Addr_VM_start >> 2)),
            "PI_S.Addr_VM_end": max(int((((primitive.Addr_VM_start + primitive.Read_VM_length) >> 2) - 1)), int((primitive.Addr_VM_start >> 2))),
            "PI_S.Addr_Vtheta_start": int((primitive.Addr_Vtheta_start >> 2)),
            "PI_S.Addr_Vtheta_end": max(int(
                (((primitive.Addr_Vtheta_start + primitive.Read_Vtheta_length) >> 2) - 1)), int((primitive.Addr_Vtheta_start >> 2))),
            "PI_S.Addr_para": int((primitive.Addr_para >> 2)),
            "PI_S.neu_num": int((primitive.neu_num // 4) - 1),
            "PI_S.Tw_len": int(primitive.Tw_len),
            "PI_S.Y_num": int(primitive.group_num - 1),
            "PI_S.Vinit": int(primitive.Vinit),
            "PI_S.Rst_mode": int(primitive.Rst_mode),
            "PI_S.fire_type": int(primitive.fire_type),
            "PI_S.Vth_adpt_en": int(int(primitive.Vth_adpt_en)),
            "PI_S.Vleaky_adpt_en": int(int(primitive.Vleaky_adpt_en)),
            "PI_S.in_cut_start": int(primitive.in_cut_start),
            "PI_S.in_row_max": int(max(primitive.in_row_max - 1, 0)),
            "Row_ck_on": int(primitive.Row_ck_on),
            "PI_S.mem_sel": int(primitive.mem_sel),

            "P_LIF.Seed": int(primitive.Seed),
            "P_LIF.Vth0": int(primitive.Vth0),
            "P_LIF.Vth_alpha": int(primitive.Vth_alpha),
            "P_LIF.Vth_beta": int(primitive.Vth_beta),
            "P_LIF.Vth_Incre": int(primitive.Vth_Incre),
            "P_LIF.VR": int(primitive.VR),
            "P_LIF.VL": int(primitive.VL),
            "P_LIF.Vleaky_alpha": int(primitive.Vleaky_alpha),
            "P_LIF.Vleaky_beta": int(primitive.Vleaky_beta),
            "P_LIF.dV": int(primitive.dV),
            "P_LIF.Ref_len": int(primitive.Ref_len),
            "P_LIF.Tw_cnt": int(primitive.Tw_cnt)
        }
        json_config["Additional"][2 * index - 1] = {
            "Read_Uin_length": int(primitive.Read_Uin_length),
            "Read_V_length": int(primitive.Read_V_length),
            "Read_VM_length": int(primitive.Read_VM_length),
            "Read_Vtheta_length": int(primitive.Read_Vtheta_length),
            "S1_V_out_length": int(primitive.S1_out_length),
            "S1_V_length": int(primitive.Read_V_length),
            "S1_Vtheta_length": int(primitive.Read_Vtheta_length),
            "S1_para_length": int(primitive.S1_para_length),
        }
        json_config["Addr"][2 * index - 1] = {
            "Addr_Uin_start": int((primitive.Addr_Uin_start)),
            "Addr_Uin_end": max(int(((max(primitive.Addr_Uin_end, primitive.Addr_Uin_start)) - 4)), int((primitive.Addr_Uin_start))),
            "Addr_S_Start": int((primitive.Addr_S_Start)),
            "Addr_S_end": max(int((((primitive.Addr_S_Start + primitive.S1_out_length)) - 4)), int((primitive.Addr_S_Start))),
            "Addr_V_start": int((primitive.Addr_V_start)),
            "Addr_V_end": max(int((((primitive.Addr_V_start + primitive.Read_V_length)) - 4)), int((primitive.Addr_V_start))),
            "Addr_VM_start": int((primitive.Addr_VM_start)),
            "Addr_VM_end": max(int((((primitive.Addr_VM_start + primitive.Read_VM_length)) - 4)), int((primitive.Addr_VM_start))),
            "Addr_Vtheta_start": int((primitive.Addr_Vtheta_start)),
            "Addr_Vtheta_end": max(int((((primitive.Addr_Vtheta_start + primitive.Read_Vtheta_length)) - 4)), int((primitive.Addr_Vtheta_start))),
            "Addr_para": int((primitive.Addr_para)),
        }
        json_config["S" + str(index) + "_valid"] = True

    @staticmethod
    def set_prim_09_router(primitive, json_config):
        json_config["PI_parameter"][2] = {
            "PIC": int(primitive.PIC),
            "Rhead_mode": int(primitive.Rhead_mode),
            "CXY": int(primitive.CXY),
            "Send_en": int(primitive.Send_en),
            "Receive_en": int(primitive.Receive_en),
            "Dout_Mem_sel": int(primitive.Dout_Mem_sel),
            "Addr_Dout_base": int(primitive.Addr_Dout_base >> 2),
            "Addr_Dout_length": int(primitive.Addr_Dout_length),
            "Addr_Rhead_base": int(primitive.Addr_Rhead_base >> 2),
            "Addr_Rhead_length": int(primitive.Addr_Rhead_length),
            "Addr_Din_base": int(primitive.Addr_Din_base >> 1),
            "Addr_Din_length": int(primitive.Addr_Din_length),
            "Send_number": int(primitive.Send_number),
            "Receive_number": int(primitive.Receive_number),
            "Nx": int(primitive.Nx),
            "Ny": int(primitive.Ny),
            "Send_PI_en": int(primitive.Send_PI_en),
            "Back_Sign_en": int(primitive.Back_sign_en),
            "Send_PI_num": int(primitive.Send_PI_num),
            "Receive_sign_num": int(primitive.Receive_sign_num),
            "Send_PI_addr_base": int(primitive.Send_PI_addr_base),
            "Relay_number": int(primitive.Relay_number),
            "Q": int(primitive.Q),
            "T_mode": int(primitive.T_mode),
            "Receive_sign_en": int(primitive.Receive_sign_en),
            "Soma_in_en": int(primitive.Soma_in_en)
        }
        json_config["Addr"][2] = {
            "Addr_Dout_base": int(0x8000+primitive.Addr_Dout_base),
            "Addr_Dout_end": int(0x8000+(primitive.Addr_Dout_base)+(primitive.Addr_Dout_length+1)*4-4),
            "Addr_Rhead_base": int(0x8000+primitive.Addr_Rhead_base),
            "Addr_Rhead_end": int(0x8000+(primitive.Addr_Rhead_base)+(primitive.Addr_Rhead_length+1)*4-4),
            "Addr_Din_base": int(0x8000+primitive.Addr_Din_base),
            "Addr_Din_end": int(0x8000+(primitive.Addr_Din_base)+(primitive.Addr_Din_length+1)*2-2),
        }
        json_config["R_valid"] = True

        _PI = json_config["PI_parameter"][2]
        _Additional = json_config["Additional"][2]

    @staticmethod
    def set_prim_41_axon(primitive, json_config):
        if primitive.pad_on:
            _Addr_InA_base = ((primitive.Addr_InA_base >> 2) -
                              primitive.Addr_start_offset)
            if _Addr_InA_base < 0:
                _Addr_InA_base += 0x8000
        else:
            _Addr_InA_base = (primitive.Addr_InA_base >> 2)
        json_config["PI_parameter"][0] = {
            "PIC": int(primitive.PIC),
            "Reset_Addr_A": int(primitive.Reset_Addr_A),
            "Reset_Addr_V": int(primitive.Reset_Addr_V),
            "MAC_grp_num_last": int(primitive.MAC_grp_num_last),
            "Load_Bias": int(primitive.Load_Bias),
            "Addr_Bias_base": int(primitive.Addr_Bias_base >> 2),
            "Addr_InA_base": int(_Addr_InA_base),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length) >> 2) + primitive.Addr_start_offset - 1), int(_Addr_InA_base)),
            "Addr_InB_base": int(primitive.Addr_InB_base >> 2),
            "Addr_V_base": int(primitive.Addr_V_base >> 2),
            "Addr_V_end": max(int(int((((primitive.Addr_V_base + primitive.Write_V_length) >> 2) - 1))), int(primitive.Addr_V_base >> 2)),
            "InA_type": int(primitive.InA_type),
            "InB_type": int(primitive.InB_type),
            "L0_num": int(primitive.L0_num),
            "L1_num": int(primitive.L1_num),
            "L2_num": int(primitive.L2_num),
            "L3_num": int(primitive.L3_num),
            "L4_num": int(primitive.L4_num),
            "L5_num": int(primitive.L5_num),
            "L0_num_in_last_row": int(primitive.L0_num_in_last_row),
            "Addr_InA_L1_step": int(primitive.Addr_InA_L1_step),
            "Addr_InA_L2_step": int(primitive.Addr_InA_L2_step),
            "Addr_InA_L3_step": int(primitive.Addr_InA_L3_step),
            "Addr_InA_L4_step": int(primitive.Addr_InA_L4_step),
            "Addr_InA_L5_step": int(primitive.Addr_InA_L5_step),
            "Addr_InA_MAC_step": int(primitive.Addr_InA_MAC_step),
            "Sx": int(primitive.conv_Sx),
            "Sy": int(primitive.conv_Sy),
            "Ex": int(primitive.conv_Ex - 1),
            "Ey": int(primitive.conv_Ey - 1),
            "pad_top": int(primitive.pad_top),
            "pad_down": int(primitive.pad_down),
            "pad_left": int(primitive.pad_left),
            "pad_right": int(primitive.pad_right),
            "A2S2_mode": int(primitive.A2S2_mode)
        }
        json_config["Additional"][0] = {
            "Input_fm_Px": int(primitive.Input_fm_Px),
            "Input_fm_Py": int(primitive.Input_fm_Py),
            "Output_fm_Ox": int(primitive.Output_fm_Ox),
            "Output_fm_Oy": int(primitive.Output_fm_Oy),
            "Read_X_length": int(primitive.Read_X_length),
            "Read_Bias_length": int(primitive.Read_Bias_length),
            "Read_weight_length": int(primitive.Read_weight_length),
            "Write_V_length": int(primitive.Write_V_length),
        }
        json_config["Addr"][0] = {
            "Addr_Bias_base": int(primitive.Addr_Bias_base),
            "Addr_InA_base": int(_Addr_InA_base << 2),
            "Addr_InA_end": max(int(((((primitive.Addr_InA_base + primitive.Read_X_length) >> 2) + primitive.Addr_start_offset) << 2)-4), int(_Addr_InA_base << 2)),
            "Addr_InB_base": int(primitive.Addr_InB_base),
            "Addr_V_base": int(primitive.Addr_V_base),
            "Addr_V_end": max(int(int((((primitive.Addr_V_base + primitive.Write_V_length)) - 4))), int(primitive.Addr_V_base)),
        }
        json_config["A_valid"] = True

        _PI = json_config["PI_parameter"][0]
        _Additional = json_config["Additional"][0]

    @staticmethod
    def set_prim_43_axon(primitive, json_config):
        json_config["PI_parameter"][0] = {
            "PIC": int(primitive.PIC),
            "Reset_Addr_A": int(primitive.Reset_Addr_A),
            "Reset_Addr_V": int(primitive.Reset_Addr_V),
            "InA_type": int(primitive.InA_type),
            "InB_type": int(primitive.InB_type),
            "Load_Bias": int(primitive.Load_Bias),
            "Addr_Bias_base": int(primitive.Addr_Bias_base >> 2),
            "Addr_InA_base": int(primitive.Addr_InA_base >> 2),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length) >> 2) - 1), int(primitive.Addr_InA_base >> 2)),
            "Addr_InB_base": int(primitive.Addr_InB_base >> 2),
            "Addr_V_base": int((primitive.Addr_V_base >> 2)),
            "Addr_V_end": max(int((((primitive.Addr_V_base + primitive.Write_V_length) >> 2) - 1)), int((primitive.Addr_V_base >> 2))),
            "L3_num": int(primitive.L3_num),
            "L4_num": int(primitive.L4_num),
            "L5_num": int(primitive.L5_num),
            "Addr_InA_L3_step": int(primitive.Addr_InA_L3_step),
            "Addr_InA_L4_step": int(primitive.Addr_InA_L4_step),
            "Addr_InA_L5_step": int(primitive.Addr_InA_L5_step),
            "constant_b": int(primitive.constant_b),
            "A2S2_mode": int(primitive.A2S2_mode)
        }
        json_config["Additional"][0] = {
            "Px": int(primitive.Px),
            "Py": int(primitive.Py),
            "Ox": int(primitive.Ox),
            "Oy": int(primitive.Oy),
            "Read_X_length": int(primitive.Read_X_length),
            "Read_A_length": int(primitive.Read_A_length),
            "Read_Bias_length": int(primitive.Read_Bias_length),
            "Write_V_length": int(primitive.Write_V_length),
        }
        json_config["Addr"][0] = {
            "Addr_Bias_base": int(primitive.Addr_Bias_base),
            "Addr_InA_base": int(primitive.Addr_InA_base),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length)) - 4), int(primitive.Addr_InA_base)),
            "Addr_InB_base": int(primitive.Addr_InB_base),
            "Addr_V_base": int((primitive.Addr_V_base)),
            "Addr_V_end": max(int((((primitive.Addr_V_base + primitive.Write_V_length)) - 4)), int((primitive.Addr_V_base))),
        }
        json_config["A_valid"] = True

        _PI = json_config["PI_parameter"][0]
        _Additional = json_config["Additional"][0]

    @staticmethod
    def set_prim_81_axon(primitive, json_config):
        json_config["PI_parameter"][0] = {
            "PIC": int((primitive.PIC)),
            "Reset_Addr_A": int(primitive.Reset_Addr_A),
            "Reset_Addr_V": int(primitive.Reset_Addr_V),
            "MAC_grp_num_last": int(primitive.MAC_grp_num_last),
            "InA_type": int(primitive.InA_type),
            "InB_type": int(primitive.InB_type),
            "Load_Bias": int(primitive.Load_Bias),
            "Addr_Bias_base": int((primitive.Addr_Bias_base >> 2)),
            "Addr_InA_base": int((primitive.Addr_InA_base >> 2)),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length + 8) >> 2) - 1), int((primitive.Addr_InA_base >> 2))),
            "Addr_InB_base": int((primitive.Addr_InB_base >> 2)),
            "Addr_V_base": int((primitive.Addr_V_base >> 2)),
            "Addr_V_end": max(int(((((primitive.Addr_V_base + primitive.Write_V_length) >> 2) - 1))), int((primitive.Addr_V_base >> 2))),
            "L0_num": int(primitive.L0_num),
            "L1_num": int(primitive.L1_num),
            "L2_num": int(primitive.L2_num),
            "L3_num": int(primitive.L3_num),
            "L4_num": int(primitive.L4_num),
            "L5_num": int(primitive.L5_num),
            "Addr_InA_L1_step": int(primitive.Addr_InA_L1_step),
            "Addr_InA_L2_step": int(primitive.Addr_InA_L2_step),
            "Addr_InA_L3_step": int(primitive.Addr_InA_L3_step),
            "Addr_InA_L4_step": int(primitive.Addr_InA_L4_step),
            "Addr_InA_L5_step": int(primitive.Addr_InA_L5_step),
            "Sx": int(primitive.conv_Sx),
            "Sy": int(primitive.conv_Sy),
            "Ex": int(primitive.conv_Ex - 1),
            "Ey": int(primitive.conv_Ey - 1),
            "pad_up": int(primitive.pad_up),
            "pad_down": int(primitive.pad_down),
            "pad_left": int(primitive.pad_left),
            "pad_right": int(primitive.pad_right),
            "A2S2_mode": int(primitive.A2S2_mode)
        }
        json_config["Additional"][0] = {
            "Input_fm_Px": int(primitive.Input_fm_Px),
            "Input_fm_Py": int(primitive.Input_fm_Py),
            "Output_fm_Ox": int(primitive.Output_fm_Ox),
            "Output_fm_Oy": int(primitive.Output_fm_Oy),
            "Read_X_length": int(primitive.Read_X_length),
            "Read_Bias_length": int(primitive.Read_Bias_length),
            "Read_weight_length": int(primitive.Read_weight_length),
            "Write_V_length": int(primitive.Write_V_length)
        }
        json_config["Addr"][0] = {
            "Addr_Bias_base": int((primitive.Addr_Bias_base)),
            "Addr_InA_base": int((primitive.Addr_InA_base)),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length + 8)) - 4), int((primitive.Addr_InA_base))),
            "Addr_InB_base": int((primitive.Addr_InB_base)),
            "Addr_V_base": int((primitive.Addr_V_base)),
            "Addr_V_end": max(int(((((primitive.Addr_V_base + primitive.Write_V_length)) - 4))), int((primitive.Addr_V_base))),
        }
        json_config["A_valid"] = True

        _PI = json_config["PI_parameter"][0]
        _Additional = json_config["Additional"][0]

    @staticmethod
    def set_prim_83_axon(primitive, json_config):
        json_config["PI_parameter"][0] = {
            "PIC": int(primitive.PIC),
            "Reset_Addr_A": int(primitive.Reset_Addr_A),
            "Reset_Addr_V": int(primitive.Reset_Addr_V),
            "InA_type": int(primitive.InA_type),
            "Load_Bias": int(primitive.Load_Bias),
            "Addr_Bias_base": int(primitive.Addr_Bias_base >> 2),
            "Addr_InA_base": int(primitive.Addr_InA_base >> 2),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length) >> 2) - 1), int(primitive.Addr_InA_base >> 2)),
            "Addr_V_base": int(primitive.Addr_V_base >> 2),
            "Addr_V_end": max(int((((primitive.Addr_V_base + primitive.Write_V_length) >> 2)) - 1), int(primitive.Addr_V_base >> 2)),
            "L3_num": int(primitive.L3_num),
            "L4_num": int(primitive.L4_num),
            "L5_num": int(primitive.L5_num),
            "Addr_InA_L3_step": int(primitive.Addr_InA_L3_step),
            "Addr_InA_L4_step": int(primitive.Addr_InA_L4_step),
            "Addr_InA_L5_step": int(primitive.Addr_InA_L5_step),
            "constant_a": int(primitive.constant_a),
            "constant_b": int(primitive.constant_b),
            "A2S2_mode": int(primitive.A2S2_mode)
        }
        json_config["Additional"][0] = {
            "Px": int(primitive.Px),
            "Py": int(primitive.Py),
            "Ox": int(primitive.Ox),
            "Oy": int(primitive.Oy),
            "Read_X_length": int(primitive.Read_X_length),
            "Read_Bias_length": int(primitive.Read_Bias_length),
            "Write_V_length": int(primitive.Write_V_length)
        }
        json_config["Addr"][0] = {
            "Addr_Bias_base": int(primitive.Addr_Bias_base),
            "Addr_InA_base": int(primitive.Addr_InA_base),
            "Addr_InA_end": max(int(((primitive.Addr_InA_base + primitive.Read_X_length)) - 4), int(primitive.Addr_InA_base)),
            "Addr_V_base": int(primitive.Addr_V_base),
            "Addr_V_end": max(int((((primitive.Addr_V_base + primitive.Write_V_length))) - 4), int(primitive.Addr_V_base)),
        }
        json_config["A_valid"] = True

        _PI = json_config["PI_parameter"][0]
        _Additional = json_config["Additional"][0]

    @staticmethod
    def set_prim_x5_soma(primitive, json_config, index):
        if (primitive.type_in == 0 and primitive.type_out == 1) or (
                primitive.type_in == 1 and primitive.type_out == 3):
            trans_cnt = 4
        elif primitive.type_in == 0 and primitive.type_out == 3:
            trans_cnt = 16
        elif primitive.type_in == 1 and primitive.type_out == 3:
            trans_cnt = 4  # 不确定
        else:
            trans_cnt = 1

        if primitive.pad_on:
            _Addr_X_Start = ((primitive.Addr_Start_in >> 2) -
                             primitive.Addr_start_offset)
            if _Addr_X_Start < 0:
                _Addr_X_Start += 0x8000
        else:
            _Addr_X_Start = primitive.Addr_Start_in >> 2

        json_config["PI_parameter"][2 * index - 1] = {
            "PIC": int(primitive.PIC),
            "PIC_Mode": int(primitive.PIC_Mode),
            "Reset_Addr_X": int(primitive.reset_Addr_in),
            "Reset_Addr_Y": int(primitive.reset_Addr_out),
            "Row_ck_on": int(primitive.Row_ck_on),
            "X_type": int(primitive.type_in),
            "Y_type": int(primitive.type_out),
            "Addr_X_Start": int(_Addr_X_Start),
            "Addr_X_End": max(int(((primitive.Addr_Start_in + primitive.Read_X_length) >> 2) - 1), int(_Addr_X_Start)),
            "Addr_Y_Start": int(primitive.Addr_Start_out >> 2),
            "Addr_Y_End": max(int(((primitive.Addr_Start_out + primitive.Write_Y_length) >> 2) - 1), int(primitive.Addr_Start_out >> 2)),
            "X_Km_num": int(primitive.Km_num_in - 1),
            "Kx_num": int(primitive.pooling_Kx - 1),
            "Ky_num": int(primitive.pooling_Ky - 1),
            "Y_Km_num": int(primitive.Km_num_out - 1),
            "Ox_num": int(primitive.Output_fm_Ox - 1),
            "Oy_num": int(primitive.Output_fm_Oy - 1),
            "Kx_step": int(primitive.Kx_step),
            "Ky_step": int(primitive.Ky_step),
            "Km_step": int(primitive.Km_step),
            "Ox_step": int(primitive.Ox_step),
            "Oy_step": int(primitive.Oy_step),
            "Sx": int(primitive.pooling_Sx),
            "Sy": int(primitive.pooling_Sy),
            "pad_top": int(primitive.pad_top),
            "pad_down": int(primitive.pad_down),
            "pad_left": int(primitive.pad_left),
            "pad_right": int(primitive.pad_right),
            "CMP_C": int(primitive.CMP_C),
            "X_cut_start": int(primitive.in_cut_start),
            "in_row_max": int(max(primitive.in_row_max - 1, 0)),
            "mem_sel": int(primitive.mem_sel)
        }
        json_config["Additional"][2 * index - 1] = {
            "Input_fm_Px": int(primitive.Input_fm_Px),
            "Input_fm_Py": int(primitive.Input_fm_Py),
            "Output_fm_Ox": int(primitive.Output_fm_Ox),
            "Output_fm_Oy": int(primitive.Output_fm_Oy),
            "Read_X_length": int(primitive.Read_X_length),
            "Write_Y_length": int(primitive.Write_Y_length),
        }
        json_config["Addr"][2 * index - 1] = {
            "Addr_X_Start": int(_Addr_X_Start << 2),
            "Addr_X_End": max(int(((primitive.Addr_Start_in + primitive.Read_X_length)) - 4), int(_Addr_X_Start << 2)),
            "Addr_Y_Start": int(primitive.Addr_Start_out),
            "Addr_Y_End": max(int(((primitive.Addr_Start_out + primitive.Write_Y_length)) - 4), int(primitive.Addr_Start_out)),
        }
        json_config["S" + str(index) + "_valid"] = True

        _PI = json_config["PI_parameter"][2 * index - 1]
        _Additional = json_config["Additional"][2 * index - 1]
