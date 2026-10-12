import warnings
from itertools import product
import copy


class MapConfigGen:
    def __init__(self):
        self.map_config = {
            'sim_clock': None,
            # 'step_clock': {
            #     ((0, 0), 0): (50000, 10000)
            # },
            # ((0, 0), 0): {
            #     # 'step_exe_number': 1,
            #     0: {
            #         'clock': 15000,
            #         'mode': 1,
            #         ((0, 0), (0, 0)): {
            #             'prims': []
            #         }
            #     }
            # }
        }
        self.phase_group_count = {}
        # {
        #     (0, 0): 0     # chip phase group number
        # }

    def _add_phase_group(self, chip: tuple):
        if self.phase_group_count.get(chip) is None:
            self.phase_group_count[chip] = 0
        old_phase_id = self.phase_group_count[chip]
        self.phase_group_count[chip] += 1
        return old_phase_id

    def add_config(self, map_config, core_offset=(0, 0), clock_in_phase=None, phase_adaptive=True):
        """
        core_offset: 所有core的物理位置会加上offset
        """
        (x_offset, y_offset) = core_offset
        assert ((0 <= x_offset <= 15) and (0 <= y_offset <= 9))
        phase_group_dict = {}  # 传入的map_config和新的map_config中phase group的对应关系
        for key in map_config.keys():
            if isinstance(key, tuple):
                if self.map_config.get(key) is None:
                    self.map_config[key] = {}
                for phase_id in map_config[key].keys():
                    if isinstance(phase_id, int):
                        if phase_group_dict.get(phase_id) is None:
                            phase_group_dict[phase_id] = self._add_phase_group(key[0])
                        new_phase_id = phase_group_dict[phase_id]
                        if self.map_config[key].get(new_phase_id) is None:
                            self.map_config[key][new_phase_id] = {
                                'clock': map_config[key][phase_id][
                                    'clock'] if clock_in_phase is None else clock_in_phase,
                                'mode': 1 if phase_adaptive else 0
                            }
                        for location in map_config[key][phase_id].keys():
                            if isinstance(location, tuple):
                                new_x, new_y = location[1][0] + x_offset, location[1][1] + y_offset
                                assert ((0 <= new_x <= 15) and (0 <= new_y <= 9))
                                new_location = (location[0], (new_x, new_y))
                                assert (self.map_config[key][new_phase_id].get(new_location) is None)
                                self.map_config[key][new_phase_id][new_location] = map_config[key][phase_id][location]

    @staticmethod
    def _get_real_core_idx(dst_x, dst_y, dst_chip_x, dst_chip_y):
        while dst_x > 15:
            dst_chip_x += 1
            dst_x -= 16
        while dst_x < 0:
            dst_chip_x -= 1
            dst_x += 16
        while dst_y > 9:
            dst_chip_y += 1
            dst_y -= 10
        while dst_y < 0:
            dst_chip_y -= 1
            dst_y += 10
        return dst_x, dst_y, dst_chip_x, dst_chip_y

    @staticmethod
    def _get_used_chips(map_config: dict):
        """
        return all used chips and correspond phases
        """
        chip_phase_dict = {}
        for chip, phase_groups in map_config.items():
            if isinstance(chip, tuple):
                if chip[1] != 0:
                    raise ValueError('Only support step 0')
                if chip in chip_phase_dict:
                    raise ValueError('There are two chips have the same position!')
                else:
                    chip_phase_dict[chip] = []
                    for phase in phase_groups.keys():
                        if isinstance(phase, int):
                            if phase in chip_phase_dict[chip]:
                                raise ValueError('There are two phase groups have the same position!')
                            chip_phase_dict[chip].append(phase)
        return chip_phase_dict

    @staticmethod
    def add_router_info(map_config: dict, group_idx_list=None, chip_x_num=None, chip_y_num=None,
                        core_x_num=16, core_y_num=10):
        """
        为map_config中的路由增加路由信息
        chip_x_num, chip_y_num  --  chip array
        core_x_num, core_y_num  --  core array
        涉及到组间同步的时候，只能自动完成两组在同一个phase数的时候组间传输的情况
        只能处理同一个step的情况
        """
        if (group_idx_list is not None) or (chip_x_num is not None) or (chip_y_num is not None):
            warnings.warn('\'group_idx_list, chip_x_num, chip_y_num\' are not needed any more!')
        chip_phase_dict = MapConfigGen._get_used_chips(map_config)
        for ((chip_x_idx, chip_y_idx), step) in chip_phase_dict.keys():
            for group_idx in chip_phase_dict[((chip_x_idx, chip_y_idx), step)]:
                if map_config[((chip_x_idx, chip_y_idx), 0)].get(group_idx) is None:
                    continue
                phase_group = map_config[((chip_x_idx, chip_y_idx), 0)][group_idx]
                for core_x_idx, core_y_idx in product(range(core_x_num), range(core_y_num)):
                    if phase_group.get(((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))) is None:
                        continue
                    prims = phase_group[((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['prims']
                    for phase_idx in range(len(prims)):
                        router = prims[phase_idx]['router']
                        if router is None:
                            continue
                        if not (router.Send_en == 1 and router.Receive_sign_en != 1):
                            continue
                        for rhead in router.RHeadList:
                            if rhead['EN'] != 1:
                                continue
                            send_dst = {
                                'data_num': rhead['pack_per_Rhead'] + 1,
                                'T_mode': rhead['T'],
                                'Rhead_num': 1,
                                'sync_en': 0
                            }
                            dst_core_list = []
                            dst_x, dst_y = core_x_idx + rhead['X'], core_y_idx + rhead['Y']
                            dst_x, dst_y, dst_chip_x, dst_chip_y = MapConfigGen._get_real_core_idx(
                                dst_x, dst_y, chip_x_idx, chip_y_idx)
                            dst_core_list.append(((dst_chip_x, dst_chip_y), (dst_x, dst_y)))
                            src_chip_x, src_chip_y, src_x, src_y = chip_x_idx, chip_y_idx, core_x_idx, core_y_idx
                            src_group = group_idx
                            if rhead['Q'] == 1:  # 多播包
                                info_str = 'phase {:d}, src core: (({:d}, {:d}), ({:d}, {:d})), dst core: (({:d}, ' \
                                           '{:d}), ({:d}, {:d})) '.format(phase_idx, src_chip_x, src_chip_y, src_x,
                                                                          src_y, dst_chip_x, dst_chip_y, dst_x, dst_y)
                                flag_0 = 0
                                if chip_phase_dict.get(((dst_chip_x, dst_chip_y), 0)) is None:
                                    raise ValueError(info_str + 'dst chip not exist!')
                                dst_chip_phase_group_list = chip_phase_dict[((dst_chip_x, dst_chip_y), 0)]
                                for dst_group_idx in dst_chip_phase_group_list:
                                    if map_config[((dst_chip_x, dst_chip_y), 0)].get(dst_group_idx) is None:
                                        raise ValueError(info_str + 'dst chip do not have phase group of {:d}!'.format(
                                            dst_group_idx))
                                    if map_config[((dst_chip_x, dst_chip_y), 0)][dst_group_idx].get(
                                            ((dst_chip_x, dst_chip_y), (dst_x, dst_y))) is None:
                                        continue
                                    dst_prims = map_config[((dst_chip_x, dst_chip_y), 0)][dst_group_idx][
                                        ((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['prims']
                                    if phase_idx >= len(dst_prims):
                                        raise ValueError(info_str + 'dst core not have enough phase num!')
                                    dst_router = dst_prims[phase_idx]['router']
                                    if dst_router is None or dst_router.Receive_en == 0:
                                        raise ValueError(info_str + 'dst router prim wrong!')
                                    flag_0 = 1
                                    if dst_group_idx != src_group:
                                        dst_router.recv_source_core_grp.append(
                                            {
                                                'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                'data_num': rhead['pack_per_Rhead'] + 1,
                                                'T_mode': rhead['T'],
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase_idx
                                            }
                                        )
                                        send_dst['sync_en'] = 1
                                        send_dst['sync_phase_num'] = phase_idx
                                    else:
                                        dst_router.recv_source_core_grp.append(
                                            {
                                                'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                'data_num': rhead['pack_per_Rhead'] + 1,
                                                'T_mode': rhead['T'],
                                                'Rhead_num': 1,
                                                'sync_en': 0
                                            }
                                        )
                                    while dst_router.CXY == 1:
                                        dst_x = dst_x + dst_router.Nx
                                        dst_y = dst_y + dst_router.Ny
                                        dst_x, dst_y, dst_chip_x, dst_chip_y = MapConfigGen._get_real_core_idx(
                                            dst_x, dst_y, dst_chip_x, dst_chip_y)
                                        dst_core_list.append(((dst_chip_x, dst_chip_y), (dst_x, dst_y)))
                                        info_str = 'phase {:d}, src core: (({:d}, {:d}), ({:d}, {:d})), dst core: ' \
                                                   '(({:d}, {:d}), ({:d}, {:d})) '.format(phase_idx, src_chip_x,
                                                                                          src_chip_y,
                                                                                          src_x, src_y, dst_chip_x,
                                                                                          dst_chip_y, dst_x, dst_y)
                                        flag_1 = 0
                                        if chip_phase_dict.get(((dst_chip_x, dst_chip_y), 0)) is None:
                                            raise ValueError(info_str + 'dst chip not exist!')
                                        dst_chip_phase_group_list = chip_phase_dict[((dst_chip_x, dst_chip_y), 0)]
                                        for dst_group_idx in dst_chip_phase_group_list:
                                            if map_config[((dst_chip_x, dst_chip_y), 0)].get(dst_group_idx) is None:
                                                raise ValueError(info_str +
                                                                 'dst chip do not have phase group of {:d}!'.format(
                                                                     dst_group_idx))
                                            if map_config[((dst_chip_x, dst_chip_y), 0)][dst_group_idx].get(
                                                    ((dst_chip_x, dst_chip_y), (dst_x, dst_y))) is None:
                                                continue
                                            dst_prims = map_config[((dst_chip_x, dst_chip_y), 0)][dst_group_idx][
                                                ((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['prims']
                                            if phase_idx >= len(dst_prims):
                                                raise ValueError(info_str + 'dst core not have enough phase num!')
                                            dst_router = dst_prims[phase_idx]['router']
                                            if dst_router is None or dst_router.Receive_en == 0:
                                                raise ValueError(info_str + 'dst router prim wrong!')
                                            flag_1 = 1
                                            if dst_group_idx != group_idx:
                                                dst_router.recv_source_core_grp.append(
                                                    {
                                                        'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                        'data_num': rhead['pack_per_Rhead'] + 1,
                                                        'T_mode': rhead['T'],
                                                        'Rhead_num': 1,
                                                        'sync_en': 1,
                                                        'sync_phase_num': phase_idx
                                                    }
                                                )
                                            else:
                                                dst_router.recv_source_core_grp.append(
                                                    {
                                                        'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                        'data_num': rhead['pack_per_Rhead'] + 1,
                                                        'T_mode': rhead['T'],
                                                        'Rhead_num': 1,
                                                        'sync_en': 0
                                                    }
                                                )
                                            break
                                        if flag_1 == 0:
                                            raise ValueError('can not find dst core of : phase {:d}, src core: (({:d}, '
                                                             '{:d}), ({:d}, {:d}))'.format(phase_idx, src_chip_x,
                                                                                           src_chip_y, src_x, src_y))
                                    break
                                if flag_0 == 0:
                                    raise ValueError('can not find dst core of : phase {:d}, src core: (({:d}, {:d}), ('
                                                     '{:d}, {:d}))'.format(phase_idx, src_chip_x, src_chip_y, src_x,
                                                                           src_y))
                            else:  # 普通路由包
                                info_str = 'phase {:d}, src core: (({:d}, {:d}), ({:d}, {:d})), dst core: (({:d}, ' \
                                           '{:d}), ({:d}, {:d})) '.format(phase_idx, src_chip_x, src_chip_y, src_x,
                                                                          src_y, dst_chip_x, dst_chip_y, dst_x, dst_y)
                                flag_2 = 0
                                if chip_phase_dict.get(((dst_chip_x, dst_chip_y), 0)) is None:
                                    warnings.warn(info_str + 'dst chip not exist!')
                                    continue
                                    raise ValueError(info_str + 'dst chip not exist!')
                                dst_chip_phase_group_list = chip_phase_dict[((dst_chip_x, dst_chip_y), 0)]
                                for dst_group_idx in dst_chip_phase_group_list:
                                    if map_config[((dst_chip_x, dst_chip_y), 0)].get(dst_group_idx) is None:
                                        raise ValueError(info_str +
                                                         'dst chip do not have phase group of {:d}!'.format(
                                                             dst_group_idx))
                                    if map_config[((dst_chip_x, dst_chip_y), 0)][dst_group_idx].get(
                                            ((dst_chip_x, dst_chip_y), (dst_x, dst_y))) is None:
                                        continue
                                    dst_prims = map_config[((dst_chip_x, dst_chip_y), 0)][dst_group_idx][
                                        ((dst_chip_x, dst_chip_y), (dst_x, dst_y))]['prims']
                                    if phase_idx >= len(dst_prims):
                                        raise ValueError(info_str + 'dst core not have enough phase num!')
                                    dst_router = dst_prims[phase_idx]['router']
                                    if dst_router is None or dst_router.Receive_en == 0:
                                        raise ValueError(info_str + 'dst router prim wrong!')
                                    flag_2 = 1
                                    if dst_group_idx != src_group:
                                        dst_router.recv_source_core_grp.append(
                                            {
                                                'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                'data_num': rhead['pack_per_Rhead'] + 1,
                                                'T_mode': rhead['T'],
                                                'Rhead_num': 1,
                                                'sync_en': 1,
                                                'sync_phase_num': phase_idx
                                            }
                                        )
                                        send_dst['sync_en'] = 1
                                        send_dst['sync_phase_num'] = phase_idx
                                    else:
                                        dst_router.recv_source_core_grp.append(
                                            {
                                                'core_id': [((src_chip_x, src_chip_y), (src_x, src_y))],
                                                'data_num': rhead['pack_per_Rhead'] + 1,
                                                'T_mode': rhead['T'],
                                                'Rhead_num': 1,
                                                'sync_en': 0
                                            }
                                        )
                                    break
                                if flag_2 == 0:
                                    raise ValueError('can not find dst core of : phase {:d}, src core: (({:d}, {:d}), ('
                                                     '{:d}, {:d}))'.format(phase_idx, src_chip_x, src_chip_y, src_x,
                                                                           src_y))
                            send_dst['core_id'] = dst_core_list
                            router.send_destin_core_grp.append(send_dst)

    @staticmethod
    def clean_router_info(map_config: dict, group_idx_list: list, chip_x_num: int, chip_y_num: int,
                          core_x_num=16, core_y_num=10):
        """
        清除router附加的额外信息
        先不用
        """
        for chip_x_idx, chip_y_idx in product(range(chip_x_num), range(chip_y_num)):
            if map_config.get(((chip_x_idx, chip_y_idx), 0)) is None:
                continue
            for group_idx in group_idx_list:
                if map_config[((chip_x_idx, chip_y_idx), 0)].get(group_idx) is None:
                    continue
                phase_group = map_config[((chip_x_idx, chip_y_idx), 0)][group_idx]
                for core_x_idx, core_y_idx in product(range(core_x_num), range(core_y_num)):
                    if phase_group.get(((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))) is None:
                        continue
                    prims = phase_group[((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['prims']
                    for phase_idx in range(len(prims)):
                        router = prims[phase_idx]['router']
                        if router is None:
                            continue
                        router.send_destin_core_grp = []
                        router.recv_source_core_grp = []

    @staticmethod
    def get_router_rhead_base(prims: dict, index=-1, rhead_base=0x300, limit=0x400):
        """
        根据前一个router原语，计算已经存在的路由表长度，返回新的路由表应该存放的基地址（4B寻址）
        """
        if index < 0:
            index += len(prims)
        assert (index < len(prims))
        while index >= 0:
            if prims[index]['router'] is not None:
                router = prims[index]['router']
                rhead_base = router.Addr_Rhead_base + router.Send_en * (router.Addr_Rhead_length + 1) * 4
                break
            index -= 1
        if rhead_base >= limit:
            raise ValueError('Router head {:X} has exceeded limit ({:X})'.format(rhead_base, limit))
        return rhead_base

    @staticmethod
    def get_max_router_rhead_base(config: dict, index=-1, rhead_base=0x300, limit=0x400):
        """
        返回一个map_config文件中所有core都未被占用的最小路由表区域地址
        """
        max_rhead_base = 0
        for chip_step_key, chip_step_value in config.items():
            if not isinstance(chip_step_key, tuple):
                continue
            for phase_group_key, phase_group_value in chip_step_value.items():
                assert isinstance(phase_group_key, int)
                for core_key, core_value in phase_group_value.items():
                    if not isinstance(core_key, tuple):
                        continue
                    max_rhead_base = max(
                        max_rhead_base, MapConfigGen.get_router_rhead_base(core_value['prims'],
                                                                           rhead_base=rhead_base,
                                                                           limit=limit))
        return max_rhead_base

    @staticmethod
    def check(map_config: dict):
        pass

    @staticmethod
    def _check_static_prim_end_addr():
        pass

    @staticmethod
    def conut_prim_number(map_config: dict):
        max_prim_num = 0
        prim_count_dict = {}
        for chip_step, step_group in map_config.items():
            if not isinstance(chip_step, tuple):
                continue
            assert prim_count_dict.get(chip_step) is None
            prim_count_dict[chip_step] = {}
            for phase_group_idx, phase_group in step_group.items():
                assert prim_count_dict[chip_step].get(phase_group_idx) is None
                prim_count_dict[chip_step][phase_group_idx] = {}
                for chip_core, core_phases in phase_group.items():
                    if not isinstance(chip_core, tuple):
                        continue
                    assert prim_count_dict[chip_step][phase_group_idx].get(chip_core) is None
                    prims = core_phases['prims']
                    core_prim_num = 0
                    for phase in prims:
                        for prim in phase.values():
                            if prim is not None:
                                core_prim_num += 1
                    max_prim_num = max(max_prim_num, core_prim_num)
                    prim_count_dict[chip_step][phase_group_idx][chip_core] = core_prim_num
        return max_prim_num, prim_count_dict

    @property
    def sim_clock(self):
        return self.map_config['sim_clock']

    @sim_clock.setter
    def sim_clock(self, value):
        assert (type(value) is int)
        assert (value > 0)
        self.map_config['sim_clock'] = value

    @staticmethod
    def set_step_clock(map_config, clock_0, clock_1):
        map_config['step_clock'] = {}
        chip_phase_dict = MapConfigGen._get_used_chips(map_config)
        for ((chip_x_idx, chip_y_idx), step) in chip_phase_dict.keys():
            map_config['step_clock'][((chip_x_idx, chip_y_idx), step)] = (clock_0, clock_1)

    @staticmethod
    def set_step_exe_number(map_config: dict, step_exe_number: int):
        chip_phase_dict = MapConfigGen._get_used_chips(map_config)
        for ((chip_x_idx, chip_y_idx), step) in chip_phase_dict.keys():
            map_config[((chip_x_idx, chip_y_idx), step)]['step_exe_number'] = step_exe_number

    @property
    def gen_dict_pure(self):
        return self.map_config

    @property
    def gen_dict(self):
        map_config_copy = copy.deepcopy(self.map_config)
        MapConfigGen.add_router_info(map_config_copy)
        return map_config_copy

    @staticmethod
    def add_prim_at_the_beginning(map_config: dict, prim: dict, core_x_num=16, core_y_num=10):
        """
        add prim at the beginning of every core
        """
        if prim is None:
            prim = {'axon': None, 'soma1': None, 'router': None, 'soma2': None}
            warnings.warn('the phase inserted is not specific, an all None phase will be used!')
        chip_phase_dict = MapConfigGen._get_used_chips(map_config)
        for ((chip_x_idx, chip_y_idx), step) in chip_phase_dict.keys():
            for group_idx in chip_phase_dict[((chip_x_idx, chip_y_idx), step)]:
                if map_config[((chip_x_idx, chip_y_idx), 0)].get(group_idx) is None:
                    continue
                inserted = False
                phase_group = map_config[((chip_x_idx, chip_y_idx), 0)][group_idx]
                for core_x_idx, core_y_idx in product(range(core_x_num), range(core_y_num)):
                    if phase_group.get(((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))) is None:
                        continue
                    if not inserted:
                        inserted = True
                        phase_group[((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['prims'].insert(0, prim)
                    else:
                        phase_group[((chip_x_idx, chip_y_idx), (core_x_idx, core_y_idx))]['prims'].insert(
                            0, {'axon': None, 'soma1': None, 'router': None, 'soma2': None})


if __name__ == '__main__':
    from generator.resnet50.resnet50_5chips.G9_48cores import gen_g9_map_config
    import numpy as np
    from generator.resnet50.data_handler import ResNetDataHandler
    from generator.resnet50.resnet50_5chips.G9_data import generate_g9_data

    handler = ResNetDataHandler()
    data = generate_g9_data(handler, size_y=4, size_x=8)

    map = gen_g9_map_config(np.ones(32), 50000, chip=(1, 2), data=data)
    rhead = MapConfigGen.get_max_router_rhead_base(map)
    print(rhead)
