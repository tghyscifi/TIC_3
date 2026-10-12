#!/usr/bin/env python
# coding: utf-8

import copy
from typing import List

from generator.behavior_simulator.memory import Memory
from generator.util import Path, NameGenerator
from generator.behavior_simulator.execution import CoreExecution,\
    CoreState, LockKeyUtil
from generator.behavior_simulator.PI_group import PIGroup
import numpy


class Core(object):
    def __init__(self, chip_id, core_id, group_id=0):
        """
        实例化一个需要配置的core对象，创建时需指定该core的坐标。
        :param x:
        :param y:
        :param group_id:范围：0~31
        """
        self.x, self.y = core_id
        self.group_id = (chip_id, group_id)
        self.chip_id = chip_id

        self._memory_entry = []
        self.MemoryOutputList = []
        """
        [
            #first phase
            [],
            #second phase
            [],
            #third phase
            []
        ]
        """
        self.coreMemory = Memory()  # 144K数据

        self.inputList = []
        """
        {
            "name": Name, 
            "start": Start, 
            "length": Length
        }
        """

        self.para_table = []  # 啥呀
        self._router = None

        self._context = None
        self._execution = CoreExecution()
        self.phase_pointer = 0
        self._PI_groups = []   # type: List[PIGroup]

        self.instant_pointer = 0
        self._instant_PI_groups = []   # type: List[PIGroup]
        self._instant_request_num = 0
        self._is_instant = False

        self.registers = {
            "Receive_PI_addr_base": 0,
            "PI_CXY": 0,
            "PI_Nx": 0,
            "PI_Ny": 0,
            "PI_sign_CXY": 0,
            "PI_sign_Nx": 0,
            "PI_sign_Ny": 0,
            "instant_PI_en": 0,
            "fixed_instant_PI": 0,
            "instant_PI_number": 0,
            "PI_loop_en": 0,
            "PI_loop_num": 0,
            "start_instant_PI_num": 0,
            "Addr_instant_PI_base": 0
        }

        self._cycles_number = 1
        self.PI_loop_num = 1
        self.PI_loop_en = False

        self._group_to_core = None

    @property
    def start_instant_num(self):
        return self.registers['start_instant_PI_num']

    @property
    def cycles_number(self):
        return self._cycles_number

    @cycles_number.setter
    def cycles_number(self, number):
        self._cycles_number = number

    @property
    def id(self):
        return (self.chip_id, (self.x, self.y))

    @property
    def phase_num(self):
        return len(self._PI_groups)

    @property
    def instant_phase_num(self):
        return len(self._instant_PI_groups)

    @property
    def current_phase(self):
        if self._is_instant:
            return self.instant_pointer
        else:
            return self.phase_pointer

    @property
    def is_instant(self):
        return self._is_instant

    @property
    def state(self):
        return self._execution.state

    @state.setter
    def state(self, new_state):
        self._execution.state = new_state

    def extend_exec_list(self):
        self._PI_groups = list(numpy.tile(
            self._PI_groups, self._cycles_number))
        self._instant_PI_groups = list(numpy.tile(
            self._instant_PI_groups, self._cycles_number))

    def add_router(self, router_point):
        self._router = router_point

    def add_context(self, context):
        self._context = context

    def simulate_axon(self, phase_num: int, instant=False):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        handle = pi_groups[phase_num].axon

        if not hasattr(handle, "axon_delay") or not handle.axon_delay:
            info_list = handle.getInfoList()
            data = self.coreMemory.read_memory(info_list)
            self.coreMemory.writeMem(handle.prim_execution(data))

    def simulate_soma1(self, phase_num: int, instant=False):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        handle = pi_groups[phase_num].soma1
        result = handle.prim_execution(
            self.coreMemory.read_memory(handle.getInfoList()))
        self.coreMemory.writeMem(result)

    def simulate_router_send(self, phase_num: int, instant=False):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        packets = {}
        packets = pi_groups[phase_num].router.routerSend(self)
        for core, data in packets.items():
            if core == "multicast_relay":
                # self._router.add_multicast_relay(
                #     data[0], data[1], self.id, data[2], phase_num)
                continue
            if core == "multicast":
                for source_core, multicast_relay, packets_num,  destination_core in data:
                    self._router.add_multicast_relay(
                        multicast_relay, packets_num, source_core, destination_core, phase_num)
                continue
            self._router.add_route(core, data, phase_num, self.group_id)
        self._router.route(phase_num, self.group_id)

    def simulate_instant_request(self, phase_num: int, instant=False):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        instant_index = pi_groups[phase_num].router.get_instant_packets(self)
        if instant_index is not None:
            for core, data in instant_index.items():
                self._router.add_route(
                    core, data, phase_num, self.group_id, True)

    def simulate_router_recv(self, phase_num: int, instant=False):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        packets = self._router.get_packets(self.id, phase_num)
        pi_groups[phase_num].router.routerRecv(self, packets)

    def simulate_soma2(self, phase_num: int, instant=False):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        handle = pi_groups[phase_num].soma2
        self.coreMemory.writeMem(handle.prim_execution(
            self.coreMemory.read_memory(handle.getInfoList())))

    def simulate(self, tb_name: str, group_to_core):
        # if self.state == CoreState.READY and self._instant_request_num > 0:
        #     self._is_instant = True
        self._group_to_core = group_to_core
        if not self._execution.locked:  # 上一个phase执行完了
            self._create_execution_steps()

        self._execution.execute()

        self._state_transfer()

    def receive_instant_index(self, dx, dy, index=None):
        self._instant_request_num += 1

        if self.state == CoreState.FINISH \
                and self.phase_pointer > self.start_instant_num:
            self.state = CoreState.READY
            self._is_instant = True
        # write dx dy to memory.
        init_x = 256+dx if dx < 0 else dx
        init_y = 256+dy if dy < 0 else dy
        data = (((init_x & 0xff) << 8) | (init_y & 0xff)) & 0xffff
        self.coreMemory.writeMem(
            [{"Model": "", "startInMemory": self.registers["Receive_PI_addr_base"]*4+0x8000, "lengthInMemory":2, "data":[[data], [0]]}])

    def execution_state(self, phase_num, instant):
        print("core", self.id, "execution phase", phase_num,
              ("instant" if instant else "static"))

    def _create_execution_steps(self):
        instant = self._is_instant
        phase_num = self.instant_pointer if instant else self.phase_pointer
        PI_groups = self._instant_PI_groups if instant else self._PI_groups
        self._execution.clear()
        self.execution_state(phase_num, instant)
        if PI_groups[phase_num].axon is not None:
            self._execution.add_step(self.simulate_axon, (phase_num, instant))
        if PI_groups[phase_num].soma1 is not None:
            self._execution.add_step(self.simulate_soma1, (phase_num, instant))
        if PI_groups[phase_num].router is not None:
            # send_PI_request add_instant_request simulate_instant_request
            self._execution.add_step(
                self.simulate_instant_request, (phase_num, instant))
            self._execution.add_step(self.lock_instant, (instant,))
            self._execution.add_step(
                self.simulate_router_send, (phase_num, instant))
            self._execution.add_step(self.lock_router, (instant,))
            self._execution.add_step(
                self.simulate_router_recv, (phase_num, instant))
        self._execution.add_step(self.coreMemory.updateMemory)
        if PI_groups[phase_num].soma2 is not None:
            self._execution.add_step(self.simulate_soma2, (phase_num, instant))
            self._execution.add_step(self.coreMemory.updateMemory)

    def _state_transfer(self):
        if self._execution.state == CoreState.LOCKED:
            return

        if self._execution.state == CoreState.READY:
            if self._is_instant is True:
                self._instant_request_num -= 1
                if self._instant_request_num == 0:
                    self._is_instant = False
                self.instant_pointer += 1
            else:
                self.phase_pointer += 1
                if self._instant_request_num > 0 \
                        and self.phase_pointer > self.start_instant_num:
                    self._is_instant = True

        if self.phase_pointer == self.phase_num and \
                self._instant_request_num == 0:
            self.state = CoreState.FINISH

    def lock_instant(self, instant):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        phase_num = self.instant_pointer if instant else self.phase_pointer
        router_prim = pi_groups[phase_num].router
        locks = LockKeyUtil.generate_instant_lock(router_prim, phase_num)
        keys = LockKeyUtil.generate_instant_key(router_prim, phase_num)

        for lock in locks:
            self._context.add_lock(self.id, lock)
        for core_id, key in keys:
            self._context.add_key(core_id, key)

        return not self._context.locked(self.id)

    def lock_router(self, instant):
        pi_groups = self._instant_PI_groups if instant else self._PI_groups
        phase_num = self.instant_pointer if instant else self.phase_pointer
        router_prim = pi_groups[phase_num].router
        locks = LockKeyUtil.generate_router_lock(
            router_prim, phase_num, self._group_to_core, self.group_id)
        keys = LockKeyUtil.generate_router_key(
            router_prim, phase_num, self._group_to_core, self.group_id)

        for lock in locks:
            self._context.add_lock(self.id, lock)
        for core_id, key in keys:
            self._context.add_key(core_id, key)

        return not self._context.locked(self.id)

    def save_prim_para(self, phase_num: int, path: str, tb_name: str, spec_fp):

        is_static = True
        pi_group_point = self._PI_groups
        _phase_num = phase_num
        if phase_num >= len(self._PI_groups):
            pi_group_point = self._instant_PI_groups
            _phase_num = phase_num-len(self._PI_groups)
            is_static = False

        if is_static:
            one_step_static_len = self.phase_num//self.cycles_number
            step_num = phase_num // one_step_static_len
            phase_num_in_step = phase_num % one_step_static_len
        else:
            instant_num = phase_num-self.phase_num
            one_step_instant_len = self.instant_phase_num//self.cycles_number
            step_num = instant_num // one_step_instant_len
            phase_num_in_step = instant_num % one_step_instant_len

        pi_group = self.para_table[int(phase_num % len(self.para_table))]
        phase_num_in_step += 1
        import os
        Path.create_simulation_case_dir(tb_name)
        if not os.path.exists(path):
            os.mkdir(path)

        core_str = "chip" + \
            str(self.id[0]).replace(' ', '')+",core" + \
            str(self.id[1]).replace(' ', '')+" 第"+str(step_num+1)+"个step"

        prim_state = (str("静态原语") if is_static else str("即时原语"))+"的"
        if pi_group["axon"] is not None:
            fp = open(path + str(phase_num) + "_Axon_" +
                      str(hex(pi_group_point[_phase_num].axon.PIC)) + ".txt", "w")
            spec_fp.write("\n" + core_str + "," + prim_state+"第"+str(
                phase_num_in_step)+"个phase的Axon原语参数:\n")
            for i in pi_group["axon"]:
                fp.write(i)
                spec_fp.write(i)
            fp.close()
        if pi_group["soma1"] is not None:
            fp = open(path + str(phase_num) + "_Soma1_" +
                      str(hex(pi_group_point[_phase_num].soma1.PIC)) + ".txt", "w")
            spec_fp.write("\n" + core_str + "," + prim_state+" 第"+str(
                phase_num_in_step)+"个phase的Soma1原语参数:\n")
            for i in pi_group["soma1"]:
                fp.write(i)
                spec_fp.write(i)
            fp.close()
        if pi_group["router"] is not None:
            fp = open(path + str(phase_num) + "_Router_" +
                      str(hex(pi_group_point[_phase_num].router.PIC)) + ".txt", "w")
            spec_fp.write("\n" + core_str + "," + prim_state+" 第"+str(
                phase_num_in_step)+"个phase的Router原语参数:\n")
            for i in pi_group["router"]:
                fp.write(i)
                spec_fp.write(i)
            fp.close()
        if pi_group["soma2"] is not None:
            fp = open(path + str(phase_num) + "_Soma2_" +
                      str(hex(pi_group_point[_phase_num].soma2.PIC)) + ".txt", "w")
            spec_fp.write("\n" + core_str + "," + prim_state+" 第"+str(
                phase_num_in_step)+"个phase的Soma2原语参数:\n")
            for i in pi_group["soma2"]:
                fp.write(i)
                spec_fp.write(i)
            fp.close()

    def print_phase_output_to_string(self, phase_num: int) -> list:
        _printList = self._get_phase_output(phase_num)
        _retList = []
        _retList += Memory.Memto4BStringList([[_printList.__len__()]])
        for i in range(_printList.__len__()):
            _retList += Memory.Memto4BStringList([[_printList[i]["start"]]])
            _retList += Memory.Memto4BStringList([[_printList[i]["length"]]])
            _retList += Memory.Memto4BStringList(_printList[i]["data"])
        # self.MemoryOutputList.clear()
        return _retList

    def _get_phase_output(self, phase_num: int) -> list:
        """
        :param phase_num:
        :return: [{"start": _address, "length": _length, "data": _data}]
        """
        _retList = []
        _outputList = self.MemoryOutputList[phase_num]
        for i in range(_outputList.__len__()):
            _address = _outputList[i]["start"]
            _length = _outputList[i]["length"]
            data = self.coreMemory.read_memory(
                [{"start": _address, "length": _length}])
            _retList.append(
                {"start": _address, "length": _length, "data": data[0]})
        return _retList

    def init_memory(self, name: str, start: int, length: int, data: list):
        """
        初始化一段内存数据，起始地址为参数start，长度为参数length，数据为参数Data
        """
        self.coreMemory.addMemoryBlock(start, length, data)
        self._memory_entry.append((name, start, length, data))
        self.inputList.append(
            {"name": name, "start": int(start), "length": int(length)})
        print({"name": name, "start": int(start), "length": int(length)})

    def add_PI_group(self, PI_group):
        self._PI_groups.append(PI_group)
        self._add_PI_group(PI_group)

    def add_instant_PI_group(self, PI_group):
        self._instant_PI_groups.append(PI_group)
        self._add_PI_group(PI_group)

    def _add_PI_group(self, PI_group):
        PI_group.print_para()
        self.MemoryOutputList.append(PI_group.outputList)
        self.para_table.append(PI_group.para_table)
        if PI_group.router is not None:  # mem写入路由表
            for routerHeadList in PI_group.routerHeadList:
                _start = routerHeadList["start"]
                _length = routerHeadList["length"]
                data = copy.deepcopy(routerHeadList["data"])
                # 名字暂时没用
                memory_name = NameGenerator.memory_name(
                    self.x, self.y, len(self._PI_groups))
                self.init_memory(memory_name, _start, _length, data)

    def router_desc(self):
        router_desc = []
        for pi_group in self._PI_groups:
            tmp = copy.deepcopy(pi_group.router_desc)
            _tmp = {}
            for x, y in pi_group.router_desc:
                _tmp[(self.x+x, self.y+y)] = tmp.pop((x, y))
            router_desc.append(_tmp)

        for pi_group in self._instant_PI_groups:
            tmp = copy.deepcopy(pi_group.router_desc)
            _tmp = {}
            for x, y in pi_group.router_desc:
                _tmp[(self.x+x, self.y+y)] = tmp.pop((x, y))
            router_desc.append(_tmp)
        return router_desc

    def config_json_output(self):
        json_config = {
            "CoreInfo":
                {
                    "x": self.x,  # Used_cores_in_row
                    "y": self.y,  # Used_cores_in_column
                    "CoreGroup": self.group_id[1],  # phase group
                    "static_PI_base_Addr": 0,
                    "registers": self.registers
                },
            "PI":
                [],
            "initData":
                [],
            "MemoryOutput":
                []
        }

        for _, start, length, data in self._memory_entry:
            json_config["initData"].append(
                {"start": int(start), "length": int(length), "data": data})
        raw_pi_group = self._PI_groups[0:(
            len(self._PI_groups)//self.cycles_number)]
        if self.PI_loop_en:
            raw_pi_group = raw_pi_group[0:(
                len(raw_pi_group)//self.PI_loop_num)]
        for PI_group in raw_pi_group:
            json_config["PI"].append(PI_group.config_json_output())
        raw_instant_pi_group = self._instant_PI_groups[0:(
            len(self._instant_PI_groups)//self.cycles_number)]
        for PI_group in raw_instant_pi_group:
            json_config["PI"].append(PI_group.config_json_output())
        json_config["MemoryOutput"] = self.MemoryOutputList

        from generator.mapping_utils.print_memory import print_memory, print_mem_map
        if print_memory:
            core_unique_id = (self.chip_id[0], self.chip_id[1], self.x, self.y)
            if core_unique_id in print_mem_map:
                json_config['printmem'] = print_mem_map[core_unique_id]
        return json_config

    def set_registers(self, registers: dict):
        for register in self.registers:
            self.registers[register] = registers.get(register, 0)

    def set_pi_loop(self):
        if self.registers['PI_loop_en'] and self.registers['PI_loop_num'] > 1:
            self.PI_loop_num = self.registers['PI_loop_num']
            self.PI_loop_en = True
            # self._PI_groups = list(numpy.tile(
            #     self._PI_groups, self.PI_loop_num))
