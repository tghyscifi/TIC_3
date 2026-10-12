from primitive.Prim_41_Axon_CNN0_new import Prim_41_Axon
from primitive.Prim_81_Axon_CNN1_new import Prim_81_Axon
from primitive.Prim_06_move_merge_new import Prim_06_move_merge
from primitive.Prim_06_move_split_new import Prim_06_move_split
from primitive.Prim_09_Router import Prim_09_Router
from primitive.Prim_X5_Soma_compare_new import Prim_X5_Soma
from primitive.Prim_02_Axon_avgpooling_new import Prim_02_Axon
from primitive.Prim_04_Axon_MLP_new import Prim_04_Axon


def p06(addr_in, addr_out, addr_ciso, length_in, num_in, length_ciso, num_ciso, length_out, num_out, type_in,
        type_out, in_cut_start=0, row_ck_on=0, in_row_max=0, data_in=None, data_ciso=None,
        reset_addr_in=1, reset_addr_ciso=1, reset_addr_out=1):
    soma1 = Prim_06_move_merge()
    soma1.length_in = length_in
    soma1.length_ciso = length_ciso
    soma1.num_in = num_in
    soma1.num_ciso = num_ciso
    soma1.length_out = length_out
    soma1.num_out = num_out
    soma1.type_in = type_in
    soma1.type_out = type_out
    soma1.in_cut_start = in_cut_start
    soma1.Reset_Addr_in = reset_addr_in
    soma1.Reset_Addr_out = reset_addr_out
    soma1.Reset_Addr_ciso = reset_addr_ciso
    soma1.Row_ck_on = row_ck_on
    soma1.Addr_Start_in = addr_in
    soma1.Addr_Start_ciso = addr_ciso
    soma1.Addr_Start_out = addr_out
    soma1.in_row_max = in_row_max
    soma1.mem_sel = 1 if addr_out == 0x9000 else 0
    if (data_in is not None) or (data_ciso is not None):
        data = soma1.init_data()
        soma1.memory_blocks = []
        if data_in is not None:
            if len(data_in) > 0:
                check_data_shape(data[0], data_in, type_in)
                data[0] = data_in
            soma1.memory_blocks.append(
                {'name': 'P06 data in',
                 'start': soma1.Addr_Start_in,
                 'data': data[0],
                 'mode': 0}
            )
        if data_ciso is not None:
            if len(data_ciso) > 0:
                check_data_shape(data[1], data_ciso, type_in)
                data[1] = data_ciso
            soma1.memory_blocks.append(
                {'name': 'P06 data ciso',
                 'start': soma1.Addr_Start_ciso,
                 'data': data[1],
                 'mode': 0}
            )
    return soma1


def p26(addr_in, addr_out, addr_ciso, length_in, num_in, length_ciso, num_ciso, length_out, num_out, type_in,
        type_out, in_cut_start=0, row_ck_on=0, in_row_max=0, data_in=None,
        reset_addr_in=1, reset_addr_ciso=1, reset_addr_out=1):
    soma1 = Prim_06_move_split()
    soma1.length_in = length_in
    soma1.length_ciso = length_ciso
    soma1.num_in = num_in
    soma1.num_ciso = num_ciso
    soma1.length_out = length_out
    soma1.num_out = num_out
    soma1.type_in = type_in
    soma1.type_out = type_out
    soma1.in_cut_start = in_cut_start
    soma1.Reset_Addr_in = reset_addr_in
    soma1.Reset_Addr_out = reset_addr_out
    soma1.Reset_Addr_ciso = reset_addr_ciso
    soma1.Row_ck_on = row_ck_on
    soma1.Addr_Start_in = addr_in
    soma1.Addr_Start_ciso = addr_ciso
    soma1.Addr_Start_out = addr_out
    soma1.in_row_max = in_row_max
    soma1.mem_sel = 1 if addr_out == 0x9000 else 0
    if data_in is not None:
        data = soma1.init_data()
        soma1.memory_blocks = []
        if len(data_in) > 0:
            check_data_shape(data[0], data_in, type_in)
            data[0] = data_in
        soma1.memory_blocks.append(
            {'name': 'P06 data in',
             'start': soma1.Addr_Start_in,
             'data': data[0],
             'mode': 0}
        )
    return soma1


def pX5(mode, addr_in, addr_out, cin, cout, px, py, kx, ky, sx, sy, cmp_c, pad_top=0, pad_down=0, pad_left=0,
        pad_right=0, type_in=1, type_out=1, in_cut_start=0, row_ck_on=0, in_row_max=0, data_in=None,
        reset_addr_in=1, reset_addr_out=1):
    soma1 = Prim_X5_Soma()
    soma1.PIC_Mode = 0 if mode == 'max' else 1
    soma1.pad_on = False if pad_top + pad_left + pad_down + pad_right == 0 else True
    soma1.type_in = type_in
    soma1.type_out = type_out
    soma1.cin = cin
    soma1.cout = cout
    soma1.Input_fm_Px = px
    soma1.Input_fm_Py = py
    soma1.pad_top = pad_top
    soma1.pad_down = pad_down
    soma1.pad_left = pad_left
    soma1.pad_right = pad_right
    soma1.pooling_Kx = kx
    soma1.pooling_Ky = ky
    soma1.pooling_Sx = sx
    soma1.pooling_Sy = sy
    soma1.CMP_C = cmp_c
    soma1.in_cut_start = in_cut_start
    soma1.reset_Addr_in = reset_addr_in
    soma1.reset_Addr_out = reset_addr_out
    soma1.Row_ck_on = row_ck_on
    soma1.Addr_Start_in = addr_in
    soma1.Addr_Start_out = addr_out
    soma1.in_row_max = in_row_max
    soma1.mem_sel = 1 if addr_out == 0x9000 else 0
    if data_in is not None:
        data = soma1.init_data()
        soma1.memory_blocks = []
        if len(data_in) > 0:
            check_data_shape(data[0], data_in, type_in)
            data[0] = data_in
        soma1.memory_blocks.append(
            {'name': 'PX5 data in',
             'start': soma1.Addr_Start_in,
             'length': soma1.Read_X_length,
             'data': data[0],
             'mode': 0}
        )
    return soma1


def p09(rhead_mode, send_en, receive_en, send_num, addr_din_length, addr_rhead_base, receive_num, addr_rhead_length,
        addr_dout_base=0x1000, addr_dout_length=0, addr_din_base=0x400, data_in=None, t_mode=1,
        soma_in_en=1, cxy=0, nx=0, ny=0, relay_num=0, send_pi_en=0, back_sign_en=0, send_pi_num=0, receive_sign_num=0,
        send_pi_addr_base=0, Q=0, receive_sign_en=0):
    """
    需要手动添加路由表
    """
    router = Prim_09_Router()
    router.Rhead_mode = rhead_mode
    router.CXY = cxy
    router.Send_en = send_en
    router.Receive_en = receive_en
    router.Addr_Dout_base = addr_dout_base
    router.Dout_Mem_sel = 1 if addr_dout_base == 0x1000 else 0
    router.Addr_Dout_length = 0 if addr_dout_base == 0x1000 else addr_dout_length
    router.Send_number = send_num
    router.Addr_Rhead_base = addr_rhead_base
    router.Addr_Rhead_length = addr_rhead_length
    router.Addr_Din_base = addr_din_base
    router.Addr_Din_length = addr_din_length
    router.Receive_number = receive_num
    router.Nx = nx
    router.Ny = ny
    router.Send_PI_en = send_pi_en
    router.Back_sign_en = back_sign_en
    router.Send_PI_num = send_pi_num
    router.Receive_sign_num = receive_sign_num
    router.Send_PI_addr_base = send_pi_addr_base    # 16B对齐
    router.Relay_number = relay_num
    router.Q = Q
    router.Recevie_sign_en = receive_sign_en
    router.T_mode = t_mode
    router.Soma_in_en = soma_in_en
    if data_in is not None:
        assert (soma_in_en == 0)
        data = router.init_data()
        router.memory_blocks = []
        if len(data_in) > 0:
            check_data_shape(data, data_in, data_type=1)
            data = data_in
        router.memory_blocks.append(
            {'name': 'Router_Dout',
             'start': router.Addr_Dout_base + 0x8000,
             'length': (router.Addr_Dout_length + 1) * 4,
             'data': data,
             'mode': 0},
        )
    return router


def p41(px, py, cin, cout, kx, ky, sx, sy, addr_ina, addr_inb, addr_bias, addr_out, ex=1, ey=1, ina_type=1,
        inb_type=1, load_bias=0, pad_top=0, pad_down=0, pad_right=0, pad_left=0, data_x=None,
        data_w=None, data_b=None, reset_addr_a=1, reset_addr_v=1, axon_delay=False, L5_num=None, L4_num=None,
        A2S2_mode=False):
    axon = Prim_41_Axon()
    axon.pad_on = False if pad_top + pad_left + pad_down + pad_right == 0 else True
    axon.InA_type = ina_type
    axon.InB_type = inb_type
    axon.Load_Bias = load_bias
    axon.cin = cin
    axon.cout = cout
    axon.Input_fm_Px = px
    axon.Input_fm_Py = py
    axon.pad_top = pad_top
    axon.pad_down = pad_down
    axon.pad_left = pad_left
    axon.pad_right = pad_right
    axon.conv_Kx = kx
    axon.conv_Ky = ky
    axon.conv_Sx = sx
    axon.conv_Sy = sy
    axon.conv_Ex = ex
    axon.conv_Ey = ey
    axon.Reset_Addr_A = reset_addr_a
    axon.Reset_Addr_V = reset_addr_v
    axon.Addr_InA_base = addr_ina
    axon.Addr_InB_base = addr_inb
    axon.Addr_Bias_base = addr_bias
    axon.Addr_V_base = addr_out
    axon.axon_delay = axon_delay
    if axon_delay:
        axon.L5_num = L5_num
        axon.L4_num = L4_num
    axon.A2S2_mode = A2S2_mode
    if (data_x is not None) or (data_w is not None) or (data_b is not None):
        data = axon.init_data()
        axon.memory_blocks = []
        if data_x is not None:
            if len(data_x) > 0:
                check_data_shape(data[0], data_x, ina_type)
                data[0] = data_x
            axon.memory_blocks.append(
                {'name': "P41_input_X",
                 'start': addr_ina,
                 'data': data[0],
                 'mode': 0},
            )
        if data_w is not None:
            if len(data_w) > 0:
                check_data_shape(data[1], data_w, inb_type)
                data[1] = data_w
            axon.memory_blocks.append(
                {'name': "P41_weight",
                 'start': addr_inb,
                 'data': data[1],
                 'mode': 0}
            )
        if data_b is not None:
            assert (load_bias == 2 or load_bias == 3)
            if len(data_b) > 0:
                check_data_shape(data[2], data_b, 0)
                data[2] = data_b
            axon.memory_blocks.append(
                {'name': "P41_bias",
                 'start': addr_bias,
                 'data': data[2],
                 'mode': 0}
            )
    return axon


def p02(avg_pooling_en, ina_type, load_bias, kx, ky, sx, sy, cin, px, py, addr_in, addr_bias, addr_out,
        pad_top=0, pad_down=0, pad_left=0, pad_right=0, bias_length=0, data_x=None, data_b=None, constant_b=0,
        reset_addr_a=1, reset_addr_v=1, ):
    axon = Prim_02_Axon()
    axon.avg_pooling_en = avg_pooling_en
    axon.pad_on = False if pad_top + pad_left + pad_down + pad_right == 0 else True
    axon.InA_type = ina_type
    axon.Load_Bias = load_bias
    axon.Bias_length = bias_length
    axon.cin = cin
    axon.Input_fm_Px = px
    axon.Input_fm_Py = py
    axon.pad_top = pad_top
    axon.pad_down = pad_down
    axon.pad_left = pad_left
    axon.pad_right = pad_right
    axon.pooling_Kx = kx
    axon.pooling_Ky = ky
    axon.pooling_Sx = sx
    axon.pooling_Sy = sy
    axon.Reset_Addr_A = reset_addr_a
    axon.Reset_Addr_V = reset_addr_v
    axon.Addr_InA_base = addr_in
    axon.Addr_Bias_base = addr_bias
    axon.Addr_V_base = addr_out
    axon.constant_b = constant_b
    if (data_x is not None) or (data_b is not None):
        data = axon.init_data()
        axon.memory_blocks = []
        if data_x is not None:
            if len(data_x) > 0:
                check_data_shape(data[0], data_x, ina_type)
                data[0] = data_x
            axon.memory_blocks.append(
                {'name': "P02_input_X",
                 'start': axon.Addr_InA_base,
                 'data': data[0],
                 'mode': 0}
            )
        if data_b is not None:
            assert (load_bias == 2 or load_bias == 3)
            if len(data_b) > 0:
                check_data_shape(data[1], data_b, 0)
                data[1] = data_b
            axon.memory_blocks.append(
                {'name': "P02_bias",
                 'start': axon.Addr_Bias_base,
                 'data': data[1],
                 'mode': 0}
            )
    return axon


def p81(px, py, cin, cout, kx, ky, sx, sy, addr_ina, addr_inb, addr_bias, addr_out, ex=1, ey=1, ina_type=1,
        inb_type=1, load_bias=0, pad_top=0, pad_down=0, pad_right=0, pad_left=0, data_x=None,
        data_w=None, data_b=None, reset_addr_a=1, reset_addr_v=1):
    axon = Prim_81_Axon()
    axon.InA_type = ina_type
    axon.InB_type = inb_type
    axon.Load_Bias = load_bias
    axon.pad_on = False if pad_top + pad_left + pad_down + pad_right == 0 else True
    axon.Input_fm_Px = px
    axon.Input_fm_Py = py
    axon.conv_Kx = kx
    axon.conv_Ky = ky
    axon.conv_Sx = sx
    axon.conv_Sy = sy
    axon.conv_Ex = ex
    axon.conv_Ey = ey
    axon.pad_up = pad_top
    axon.pad_down = pad_down
    axon.pad_left = pad_left
    axon.pad_right = pad_right
    axon.cin = cin
    axon.cout = cout
    axon.Reset_Addr_A = reset_addr_a
    axon.Reset_Addr_V = reset_addr_v
    axon.Addr_InA_base = addr_ina
    axon.Addr_InB_base = addr_inb
    axon.Addr_Bias_base = addr_bias
    axon.Addr_V_base = addr_out
    if (data_x is not None) or (data_w is not None) or (data_b is not None):
        data = axon.init_data()
        axon.memory_blocks = []
        if data_x is not None:
            if len(data_x) > 0:
                check_data_shape(data[0], data_x, ina_type)
                data[0] = data_x
            axon.memory_blocks.append(
                {'name': "P81_input_X",
                 'start': axon.Addr_InA_base,
                 'data': data[0],
                 'mode': 0},
            )
        if data_w is not None:
            if len(data_w) > 0:
                check_data_shape(data[1], data_w, inb_type)
                data[1] = data_w
            axon.memory_blocks.append(
                {'name': "P81_weight",
                 'start': axon.Addr_InB_base,
                 'data': data[1],
                 'mode': 0}
            )
        if data_b is not None:
            assert (load_bias == 2 or load_bias == 3)
            if len(data_b) > 0:
                check_data_shape(data[2], data_b, 0)
                data[2] = data_b
            axon.memory_blocks.append(
                {'name': "P81_bias",
                 'start': axon.Addr_Bias_base,
                 'data': data[2],
                 'mode': 0}
            )
    return axon


def p04(cin, cout, addr_ina, addr_inb, addr_bias, addr_out, ina_type=1, inb_type=1, 
        load_bias=0, data_x=None, data_w=None, data_b=None, reset_addr_a=1, reset_addr_v=1):
    axon = Prim_04_Axon()
    axon.PIC = 0x04
    axon.InA_type = ina_type
    axon.InB_type = inb_type
    axon.Load_Bias = load_bias
    axon.cin = cin
    axon.cout = cout
    axon.constant_b = 0
    axon.Reset_Addr_A = reset_addr_a
    axon.Reset_Addr_V = reset_addr_v
    axon.Addr_InA_base = addr_ina
    axon.Addr_InB_base = addr_inb
    axon.Addr_Bias_base = addr_bias
    axon.Addr_V_base = addr_out

    if (data_x is not None) or (data_w is not None) or (data_b is not None):
        data = axon.init_data()
        axon.memory_blocks = []
        if data_x is not None:
            if len(data_x) > 0:
                check_data_shape(data[0], data_x, ina_type)
                data[0] = data_x
            axon.memory_blocks.append(
                {'name': "P04_input_X",
                 'start': axon.Addr_InA_base,
                 'data': data[0],
                 'mode': 0},
            )
        if data_w is not None:
            if len(data_w) > 0:
                check_data_shape(data[1], data_w, inb_type)
                data[1] = data_w
            axon.memory_blocks.append(
                {'name': "P04_weight",
                 'start': axon.Addr_InB_base,
                 'data': data[1],
                 'mode': 0}
            )
        if data_b is not None:
            assert (load_bias == 2 or load_bias == 3)
            if len(data_b) > 0:
                check_data_shape(data[2], data_b, 0)
                data[2] = data_b
            axon.memory_blocks.append(
                {'name': "P04_bias",
                 'start': axon.Addr_Bias_base,
                 'data': data[2],
                 'mode': 0}
            )
    return axon


def check_data_shape(x, y, data_type):
    assert (len(x) == len(y))
    if data_type == 0:  # int 32
        for item_x, item_y in zip(x, y):
            assert (len(item_x) == 1 and len(item_y) == 1)
    elif data_type == 1:  # int 8
        for item_x, item_y in zip(x, y):
            assert (len(item_x) == 4 and len(item_y) == 4)
    else:
        raise NotImplemented


if __name__ == '__main__':
    s = p26(addr_in=0x0000, addr_out=0x4000, addr_ciso=0x2000, length_in=17, num_in=3, length_ciso=1, num_ciso=3,
            length_out=16, num_out=3, type_in=0, type_out=1, in_cut_start=0, row_ck_on=0, in_row_max=0,
            data_in=None)
    r = p09(rhead_mode=1, send_en=1, receive_en=0, send_num=3, addr_din_length=0, addr_rhead_base=0x300,
            receive_num=0, addr_rhead_length=0, addr_dout_base=0x1000, addr_dout_length=1, addr_din_base=0x400,
            data_in=None, t_mode=1, soma_in_en=0, cxy=0, nx=None, ny=None, relay_num=None)
