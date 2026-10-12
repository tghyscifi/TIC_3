import numpy as np


def get_dst(src_chip_x, src_chip_y, src_core_x, src_core_y, x, y):
    dst_chip_x = src_chip_x
    dst_chip_y = src_chip_y
    dst_core_x = src_core_x + x
    dst_core_y = src_core_y + y
    while dst_core_x > 15:
        dst_core_x -= 16
        dst_chip_x += 1
    while dst_core_x < 0:
        dst_core_x += 16
        dst_chip_x -= 1
    while dst_core_y > 9:
        dst_core_y -= 10
        dst_chip_y += 1
    while dst_core_y < 0:
        dst_core_y += 10
        dst_chip_y -= 1
    assert 0 <= dst_core_x <= 15 and 0 <= dst_core_y <= 9 and 0 <= dst_chip_x <= 2 and 0 <= dst_chip_y <= 2
    return dst_chip_x, dst_chip_y, dst_core_x, dst_core_y


def go_deeper(router_static, map_config, group_idx_list, src_core_x, src_core_y, phase_idx, pack_num, src_chip_x=0,
              src_chip_y=0):
    """
    遇到多播包的时候，逐层深入
    """
    if map_config.get(((src_chip_x, src_chip_y), 0)) is None:
        print('dst chip {} {} is not exist'.format(src_chip_x, src_chip_y))
        exit(44)
    for group_idx in group_idx_list:
        this_core_idx = src_chip_x * 16 + src_core_x + (src_chip_y * 10 + src_core_y) * (16 * 3)
        if map_config[((src_chip_x, src_chip_y), 0)].get(group_idx) is None:
            continue
        if map_config[((src_chip_x, src_chip_y), 0)][group_idx].get(((src_chip_x, src_chip_y), (src_core_x, src_core_y))) is None:
            continue
        prims = map_config[((src_chip_x, src_chip_y), 0)][group_idx][((src_chip_x, src_chip_y), (src_core_x, src_core_y))]['prims']
        if len(prims) <= phase_idx:
            return
        if prims[phase_idx].get('router') is None:
            return
        if isinstance(prims[phase_idx]['router'], list):
            router = prims[phase_idx]['router'][0]
        else:
            router = prims[phase_idx]['router']
        if router is None:
            continue
        if router.CXY == 0:
            return
        dst_chip_x, dst_chip_y, dst_core_x, dst_core_y = get_dst(src_chip_x, src_chip_y, src_core_x, src_core_y,
                                                                 router.Nx, router.Ny)
        dst_core_idx = dst_chip_x * 16 + dst_core_x + (dst_chip_y * 10 + dst_core_y) * (16 * 3)
        router_static[1, phase_idx, this_core_idx, dst_core_idx] = pack_num  # 多播
        go_deeper(router_static, map_config, group_idx_list, dst_core_x, dst_core_y, phase_idx, pack_num)
        go_deeper(router_static, map_config, group_idx_list, dst_core_x, dst_core_y, phase_idx,
                  pack_num, src_chip_x=dst_chip_x, src_chip_y=dst_chip_y)



def get_router_info(map_config):
    """
    input: map_config 映射的IR
    output: 路由信息
    * 只支持单chip，core_x_max_num=16, core_y_max_num=10

    路由表格式：

    静态原语路由包
    格式： mode * phase * 160 * 160
    - 维度1 ： 单播还是多播  0 - 单播   1 - 多播
    - 维度2 ： 数据传输的phase
    - 维度3 ： 发送core
    - 维度4 ： 接收core
    * core(0,0), core(1,0), ... , core(15,0), core(0,1), ... ,core(15,9) 依次展开，序号为 0, 1, ... , 159
    * 矩阵中(mode, phase, x, y) 代表模式为（单播\多播）,第phase个相位，第x个core发送给第y个core的路由包个数

    即时原语路由包

    """
    group_idx_list = range(0, 100)

    router_static = np.zeros([2, 32, 160*9, 160*9], dtype=int)  # 静态路由包

    for chip_x_idx in range(3):
        for chip_y_idx in range(3):
            if map_config.get(((chip_x_idx, chip_y_idx), 0)) is None:
                continue
            for core_x_idx in range(16):
                for core_y_idx in range(10):
                    for group_idx in group_idx_list:
                        this_core_idx = chip_x_idx * 16 + core_x_idx + (chip_y_idx * 10 + core_y_idx) * (16 * 3)
                        if map_config[((chip_x_idx, chip_y_idx), 0)].get(group_idx) is None:
                            continue
                        if map_config[((chip_x_idx, chip_y_idx), 0)][group_idx].get(((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))) is None:
                            continue
                        prims = map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['prims']
                        if map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))].get('instant_prims') is not None:
                            if len(map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['instant_prims']) > 0:
                                print('there are instant prim in ')
                                exit(22)
                        # instant_prims = map_config[((chip_x_idx, chip_y_idx), 0)][group_idx][((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['instant_prims']
                        # 静态原语路由
                        for phase_idx in range(len(prims)):
                            if prims[phase_idx].get('router') is None:
                                continue
                            if isinstance(prims[phase_idx]['router'], list):
                                router = prims[phase_idx]['router'][0]
                            else:
                                router = prims[phase_idx]['router']
                            if router is None:
                                continue
                            if router.Send_en == 0:
                                continue
                            for rhead_idx in range(len(router.RHeadList)):
                                rhead = router.RHeadList[rhead_idx]
                                if rhead['EN'] == 0:
                                    continue
                                dst_chip_x, dst_chip_y, dst_core_x, dst_core_y = get_dst(chip_x_idx, chip_y_idx, core_x_idx, core_y_idx, rhead['X'], rhead['Y'])
                                dst_core_idx = dst_chip_x * 16 + dst_core_x + (dst_chip_y * 10 + dst_core_y) * (16 * 3)
                                if rhead['Q']:
                                    router_static[1, phase_idx, this_core_idx, dst_core_idx] = rhead['pack_per_Rhead'] + 1  # 多播
                                    # go deeper
                                    go_deeper(router_static, map_config, group_idx_list, dst_core_x, dst_core_y, phase_idx,
                                              rhead['pack_per_Rhead'] + 1, src_chip_x=dst_chip_x, src_chip_y=dst_chip_y)
                                else:
                                    router_static[0, phase_idx, this_core_idx, dst_core_idx] = rhead['pack_per_Rhead'] + 1  # 单播
                        # 即时原语路由

                        break
    return router_static


if __name__ == "__main__":
    import scipy.io as scio
    import pickle

    map_config_name = 'mapping_ir_merged.mapping'

    with open(map_config_name, 'rb') as f:
        map_config = pickle.load(f)

    router_static = get_router_info(map_config)
    print('total router pack number is {}'.format(sum(sum(sum(sum(router_static))))))
    scio.savemat('./resnet50_9chip_LUT.mat', {'router_static': router_static})


