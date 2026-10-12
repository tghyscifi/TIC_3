#!/usr/bin/env python
# coding: utf-8

'''
Router 类负责整体的路由逻辑
'''

from enum import Enum
import copy


class MulticastRelay(Enum):
    MULTICAST = 1
    RELAY = 2


class Router(object):
    class RouteCoreNotFoundError(RuntimeError):
        pass

    def __init__(self, group_to_core: dict):
        super().__init__()

        self.packet_pool = {}
        self.multicast_list = {}
        self.relay_list = {}
        self._group_to_cores = group_to_core

    # destination_core：((chip.x,chip.y),(dest.x,dest.y))
    # source_core_group_id((chip.x,chip.y),group_id)
    def add_route(self, destination_core, packet, phase_num, source_core_group_id, instant_index=False):
        if not instant_index:
            is_find = False
            is_same_group = False
            for group_id, cores in self._group_to_cores.items():
                for core in cores:
                    if core.id == destination_core:
                        is_same_group = True if group_id == source_core_group_id else False
                        is_find = True
                        break
                if is_find:
                    break
            if not is_same_group:
                phase_num = "instant"
            if phase_num not in self.packet_pool:
                self.packet_pool[phase_num] = {}

            if destination_core in self.packet_pool[phase_num]:
                self.packet_pool[phase_num][destination_core].append(packet)
            else:
                self.packet_pool[phase_num][destination_core] = [packet]
        else:
            source_id = destination_core
            for (receive_one_core, index) in packet:
                receive_group_id = 0
                is_find = False
                for group_id, cores in self._group_to_cores.items():
                    for core in cores:
                        if core.id == receive_one_core:
                            receive_group_id = core.group_id
                            is_find = True
                            break
                    if is_find:
                        break
                if not is_find:
                    raise Exception(receive_one_core, "has no configuration!")
                for receive_core in self._group_to_cores[receive_group_id]:
                    source_abs_x = source_id[0][0]*16+source_id[1][0]
                    source_abs_y = source_id[0][1]*10+source_id[1][1]
                    recv_abs_x = receive_core.id[0][0]*16+receive_core.id[1][0]
                    recv_abs_y = receive_core.id[0][1]*10+receive_core.id[1][1]
                    receive_core.receive_instant_index(
                        source_abs_x-recv_abs_x, source_abs_y-recv_abs_y, index)
                    pass

    def get_packets(self, destination_core, phase_num):
        normal_packets = None
        instant_packets = None

        if phase_num in self.packet_pool:
            if destination_core in self.packet_pool[phase_num]:
                normal_packets = self.packet_pool[phase_num].pop(
                    destination_core)

        if "instant" in self.packet_pool:
            if destination_core in self.packet_pool["instant"]:
                instant_packets = self.packet_pool["instant"].pop(
                    destination_core)

        if normal_packets != None and instant_packets != None:
            return normal_packets+instant_packets
        elif normal_packets != None:
            return normal_packets
        elif instant_packets != None:
            return instant_packets

    def clear_packets(self):
        self.packet_pool.clear()

    def add_multicast_relay(self, multicast_relay, packets_num, source_core, destination_core, phase_num):
        if phase_num not in self.multicast_list:
            self.multicast_list[phase_num] = {}
        if phase_num not in self.relay_list:
            self.relay_list[phase_num] = {}
        if multicast_relay == 1:
            if source_core in self.multicast_list[phase_num]:
                # self.multicast_list[phase_num][source_core] = [
                #     destination_core, packets_num, self.multicast_list[phase_num][source_core][2]]
                assert destination_core == self.multicast_list[phase_num][source_core][0]
                self.multicast_list[phase_num][source_core] = [
                    destination_core, packets_num + self.multicast_list[phase_num][source_core][1],
                    self.multicast_list[phase_num][source_core][2]]
            else:
                self.multicast_list[phase_num][source_core] = [
                    destination_core, packets_num, 0]
        elif multicast_relay == 2:
            if source_core in self.relay_list[phase_num]:
                self.relay_list[phase_num][source_core] = [
                    destination_core, packets_num, self.relay_list[phase_num][source_core][2]]
            else:
                self.relay_list[phase_num][source_core] = [
                    destination_core, packets_num, 0]

    def route(self, phase_num, source_group_id):
        if self.multicast_list.get(phase_num):
            for source, [dest, packets_num, current_core_index] in self.multicast_list[phase_num].items():
                key = phase_num
                is_find = False
                for group_id, cores in self._group_to_cores.items():
                    for core in cores:
                        if core.id == source:
                            if group_id != source_group_id:
                                key = 'instant'
                                is_find = True
                        if is_find:
                            break
                    if is_find:
                        break
                if packets_num != 0:
                    multicast_packet = []
                    i = 0
                    for packets_received_one_core in self.packet_pool[key][source]:
                        if i < current_core_index:  # 当前core发过来的数据已经处理过了,跳过处理过多播的core
                            i += 1
                            continue
                        for data in packets_received_one_core:
                            if data["Q"]:
                                # 如果还有需要多播的包数,packets_num！=0
                                if self.multicast_list[phase_num][source][1] != 0:
                                    multicast_packet.append(data)
                                    # packets_num-1
                                    self.multicast_list[phase_num][source][1] -= 1
                        # current_core_index+1，表示已经处理的core又增加了1，下一次再进行处理时，跳过该core
                        self.multicast_list[phase_num][source][2] += 1

                    is_find = False
                    find_group_id = 0
                    for group_id, cores in self._group_to_cores.items():
                        for core in cores:
                            if core.id == dest:
                                find_group_id = group_id
                                is_find = True
                                break
                        if is_find:
                            break
                    if not is_find:
                        raise Exception("core"+str(dest)+" is not find")
                    if key == 'instant':
                        self.add_route(dest, multicast_packet,
                                       phase_num, (find_group_id[0], find_group_id[1]+1))
                    else:
                        self.add_route(dest, multicast_packet,
                                       phase_num, find_group_id)
        if self.relay_list.get(phase_num):
            for source, [dest, packets_num, current_core_index] in self.relay_list[phase_num].items():
                key = phase_num
                for group_id, cores in self._group_to_cores.items():
                    for core in cores:
                        if core.id == source:
                            if group_id != source_group_id:
                                key = 'instant'
                if packets_num != 0:
                    relay_packet = []
                    delete_Q_index = []
                    i = 0
                    for packets_received_one_core in self.packet_pool[phase_num][source]:
                        if i < current_core_index:
                            i += 1
                            continue
                        Q_index = 0
                        for data in packets_received_one_core:
                            if data["Q"]:
                                if self.relay_list[phase_num][source][1] != 0:
                                    relay_packet.append(data)
                                    self.relay_list[phase_num][source][1] -= 1
                                delete_Q_index.append(Q_index)
                            Q_index += 1
                        self.relay_list[phase_num][source][2] += 1
                        # 中继core删除packet_pool中接收到的包头中Q=1的包
                        delete_num = 0
                        for i in delete_Q_index:
                            packets_received_one_core.pop(i-delete_num)
                            delete_num += 1
                        delete_Q_index.clear()
                    is_find = False
                    find_group_id = 0
                    for group_id, cores in self._group_to_cores.items():
                        for core in cores:
                            if core.id == source:
                                find_group_id = group_id
                                is_find = True
                                break
                        if is_find:
                            break
                    if not is_find:
                        raise Exception("core"+str(dest)+" is not find")
                    if key == 'instant':
                        self.add_route(dest, relay_packet,
                                       phase_num, (find_group_id[0], find_group_id[1]+1))
                    else:
                        self.add_route(dest, relay_packet,
                                       phase_num, find_group_id)
