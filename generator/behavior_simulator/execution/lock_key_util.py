#!/usr/bin/env python
# coding: utf-8

from typing import List

from primitive import Prim_09_Router


class LockKeyUtil(object):
    @staticmethod
    def generate_router_lock(router_prim: Prim_09_Router, phase_num, group_to_core, src_group_id) -> List[str]:
        # 返回 lock_id
        # 普通路由接收锁： r1, r2, ...
        # 即时原语应答锁： i1, i2, ...
        locks = []
        if router_prim.Receive_en:
            receive_info = router_prim.recv_source_core_grp
            for info in receive_info:
                sync = info.get('sync_en', False)
                if sync:
                    sync_phase = info['sync_phase_num']
                    locks.append(str(sync_phase) + 'r')
                else:
                    core_ids = info['core_id']
                    if not isinstance(core_ids, list):
                        core_ids = [core_ids]
                    for core_id in core_ids:
                        is_find = False
                        for group_core in group_to_core[src_group_id]:
                            if core_id == group_core.id:
                                locks.append(str(phase_num)+'r')
                                is_find = True
                                break
                        if not is_find:
                            locks.append('r')

        return locks

    @staticmethod
    def generate_router_key(router_prim: Prim_09_Router, phase_num, group_to_core, src_group_id) -> List[str]:
        # 返回 [(core_id, key_id)]
        # 命名规则与lock相同
        keys = []
        if router_prim.Send_en:
            for info in router_prim.send_destin_core_grp:  # 遍历list中每个dict
                core_ids = info['core_id']
                # 如果core_id这个key的值是个list，代表存在多播或中继
                # 不存在多播中继，就是一个core获得钥匙
                if not isinstance(core_ids, list):
                    core_ids = [core_ids]
                sync = info.get('sync_en', False)
                for core_id in core_ids:        # 这些core都要获得钥匙
                    if sync:
                        sync_phase = info['sync_phase_num']
                        keys.append((core_id, str(sync_phase) + 'r'))
                    else:
                        is_find = False
                        for group_core in group_to_core[src_group_id]:
                            if core_id == group_core.id:
                                keys.append((core_id, str(phase_num)+'r'))
                                is_find = True
                                break
                        if not is_find:
                            keys.append((core_id, 'r'))

        return keys

    @staticmethod
    def generate_instant_lock(router_prim: Prim_09_Router, phase_num) -> List[str]:
        # 返回 lock_id
        # 普通路由接收锁： r1, r2, ...
        # 即时原语应答锁： i1, i2, ...
        locks = []
        if router_prim.Receive_sign_en:
            locks = ['i'] * (router_prim.Receive_sign_num + 1)
        return locks

    @staticmethod
    def generate_instant_key(router_prim: Prim_09_Router, phase_num) -> List[str]:
        # 返回 [(core_id, key_id)]
        # 命名规则与lock相同
        keys = []

        if router_prim.Back_sign_en:
            # core_id = instant_request_back[0]
            # # 找到这个core所在的phase_grp，然后给这个grp里的所有core都返回一个钥匙?是否需要？
            # # 索引group_id
            # isfined = 0
            # group_id_current = 0
            # group = chip.get_group_to_cores()
            # for group_id, cores in group.items():
            #     for core in cores:
            #         if core_id == (core.x, core.y):
            #             isfined = 1
            #             break
            #     if isfined == 1:
            #         group_id_current = group_id
            #         break

            # for core in group[group_id_current]:
            #     core_id = (core.x, core.y)
            #     key_id = 'i'
            #     keys.append((core_id, key_id))
            for index in range(len(router_prim.instant_request_back)):
                core_id = router_prim.instant_request_back[index]
                key_id = 'i'
                keys.append((core_id, key_id))
        return keys

    # (普通core间路由)发送的目的core的信息(都是实际数量，不用减1)
    # send_destin_core_grp[
        # {core_id :[(0,1),(2,1)], data_num :10， T_mode :0/1， Rhead_num :1},
        #  {core_id :[(0,2)], data_num :10， T_mode :0/1， Rhead_num :1}]

    # (普通core间路由)接收的来源core的信息(发送core的信息，数据包的个数，单包/多包，表头的个数（与发送那边对应）)
    # recv_source_core_grp[{core_id: (0,0), data_num:10, T_mode: 0/1, Rhead_num: 1}]

    # (即时原语信息)发送
    # instant_prim_request[(core_id, index), (core_id, index)]

    # (即时原语返回应答)Back_sign_en = 1时,就是这个应答会返回给哪个phase_grp的所有core_id
    # instant_request_back[(0,0),(0,1),(0,2),(0,3)]
