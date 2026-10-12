#!/usr/bin/env python
# coding: utf-8

from typing import Dict, List

from generator.behavior_simulator.core import Core
from generator.functions.util import hex_to_string
from generator.util import Path, NameGenerator
from generator.behavior_simulator.router import Router
from generator.behavior_simulator.execution import ChipExecution, CoreState


class Chip:
    def __init__(self):
        # {core_id: Core}
        self._cores = {}    # type: Dict[str: Core]
        # {group_id: [cores]}
        self._group_to_cores = {}  # type: Dict[str: List[Core]]
        # {chip_id: {group_id:(phase_clock, mode, clock0_in_step, clock1_in_step, trigger)}}
        self._phase_clock = {}

        # {chip_id: [cores]}
        self._chip_to_cores = {}

        self._router = Router(self._group_to_cores)

        self._execution = ChipExecution(self._cores)

        self._clock_in_step = {}

        self.step_clock = {}

        self._step_exe_num = {}

    def set_step_exe_num(self, chip_id, step_num):
        if chip_id in self._step_exe_num:
            assert step_num == self._step_exe_num[chip_id]
        else:
            self._step_exe_num[chip_id] = step_num

    def step_exe_num(self, chip_id):
        if chip_id not in self._step_exe_num:
            return 1
        return self._step_exe_num[chip_id]

    def set_trigger_clock(self, chip_id, trigger, clock0_in_step, clock1_in_step):
        if chip_id in self.step_clock:
            if trigger in self.step_clock[chip_id]:
                self.step_clock[chip_id][trigger]['clock0_in_step'] = clock0_in_step
                self.step_clock[chip_id][trigger]['clock1_in_step'] = clock1_in_step
            else:
                self.step_clock[chip_id][trigger] = {}
                self.step_clock[chip_id][trigger]['clock0_in_step'] = clock0_in_step
                self.step_clock[chip_id][trigger]['clock1_in_step'] = clock1_in_step
        else:
            self.step_clock[chip_id] = {}
            self.step_clock[chip_id][trigger] = {}
            self.step_clock[chip_id][trigger]['clock0_in_step'] = clock0_in_step
            self.step_clock[chip_id][trigger]['clock1_in_step'] = clock1_in_step

    # def set_clock_in_step(self, chip_id, step_id, clock0_in_step, clock1_in_step):
        # if chip_id not in self._clock_in_step:
        #     self._clock_in_step[chip_id] = {}

        # if clock0_in_step is None or clock1_in_step is None:
        #     return

        # if str(step_id) in self._clock_in_step[chip_id]:
        #     self._clock_in_step[chip_id][str(
        #         step_id)]["clock0_in_step"] = clock0_in_step
        #     self._clock_in_step[chip_id][str(
        #         step_id)]["clock1_in_step"] = clock1_in_step
        # else:
        #     self._clock_in_step[chip_id][str(step_id)] = {}
        #     self._clock_in_step[chip_id][str(
        #         step_id)]["clock0_in_step"] = clock0_in_step
        #     self._clock_in_step[chip_id][str(
        #         step_id)]["clock1_in_step"] = clock1_in_step

    def get_group_to_cores(self):
        return self._group_to_cores

    def add_core(self, core_id, core: Core):
        self._cores[core_id] = core
        group_id = core.group_id
        if group_id in self._group_to_cores:
            self._group_to_cores[group_id].append(core)
        else:
            self._group_to_cores[group_id] = [core]

        chip_id = core_id[0]
        if chip_id in self._chip_to_cores:
            self._chip_to_cores[chip_id].append(core)
        else:
            self._chip_to_cores[chip_id] = [core]

        self._execution.add_core(core_id)
        core.add_router(self._router)
        core.add_context(self._execution)

    def get_core(self, chip_id, x: int, y: int) -> Core:
        return self._cores.get((chip_id, (x, y)), None)

    # TODO
    def print_case_spec(self, tb_name: str):
        # 打印测试用例说明
        spec_path = Path.hardware_out_dir(
            tb_name) + tb_name + "_description.txt"
        spec_fp = open(spec_path, "w", encoding="UTF-8")
        spec_fp.write(tb_name + "测试用例说明:\n")

        for chip_id, cores in self._chip_to_cores.items():
            group_num = 0
            for group_id in self._group_to_cores:
                if group_id[0] == chip_id:
                    group_num += 1
            spec_fp.write("\nchip "+str(chip_id) + "执行" + str(self.step_exe_num(chip_id))+"个step, " +
                          str(group_num)+"个phase group\n")
            group_to_core = sorted(self._group_to_cores.items(
            ), key=lambda items: items[1][0].group_id[1])
            for group_id, cores in group_to_core:
                if group_id[0] == chip_id:
                    _cores = sorted(cores, key=lambda core: core.x)
                    _cores = sorted(cores, key=lambda core: core.y)
                    spec_fp.write("\nphase group "+str(group_id[1])+" 执行"+str(len(_cores))+"个core,执行"+str(
                        _cores[0].phase_num) + "个静态原语phase,执行"+str(
                        _cores[0].instant_phase_num) + "个即时原语phase,每个phase执行的时钟数为" + str(self._phase_clock[chip_id][group_id][0]) + "个clock\n")

        spec_fp.write(
            "\n\n******************************************* separator *******************************************\n\n")
        for chip_id, cores in self._chip_to_cores.items():
            group_num = 0
            for group_id in self._group_to_cores:
                if group_id[0] == chip_id:
                    group_num += 1
            spec_fp.write("\nchip "+str(chip_id) + "执行" + str(self.step_exe_num(chip_id))+"个step, " +
                          str(group_num)+"个phase group\n")
            group_to_core = sorted(self._group_to_cores.items(
            ), key=lambda items: items[1][0].group_id[1])
            for group_id, cores in group_to_core:
                if group_id[0] == chip_id:
                    _cores = sorted(cores, key=lambda core: core.x)
                    _cores = sorted(cores, key=lambda core: core.y)
                    spec_fp.write("\nphase group "+str(group_id[1])+" 执行"+str(len(_cores))+"个core,执行"+str(
                        _cores[0].phase_num) + "个静态原语phase,执行"+str(
                        _cores[0].instant_phase_num) + "个即时原语phase,每个phase执行的时钟数为" + str(self._phase_clock[chip_id][group_id][0]) + "个clock\n")
                    for core in _cores:
                        for i in range(core.phase_num):
                            core.save_prim_para(i, Path.simulation_input_dir(
                                tb_name, core.x, core.y), tb_name, spec_fp)
                        for i in range(core.instant_phase_num):
                            core.save_prim_para(core.phase_num+i, Path.simulation_input_dir(
                                tb_name, core.x, core.y), tb_name, spec_fp)
        spec_fp.close()

    def simulate(self, tb_name: str):

        # self._execution.init_core_state(cores)
        # core_sequence = self._execution.sequential_execute()
        # cores = [next(core_sequence) for i in range(len(cores))]
        # # 执行
        # for i in range(phase_num):
        #     # FIXME  遍历Core执行可能造成路由问题
        #     for core in cores:
        #         core.phase_pointer = i
        #         core.simulate(tb_name)

        #     self.print_phase_output(group_id, i, tb_name)

        for core_id in self._execution.sequential_execute():
            core = self._cores[core_id]   # type: Core

            phase_num = core.phase_pointer
            is_instant = core.is_instant
            if core.is_instant:
                phase_num = core.phase_num+core.current_phase

            core.simulate(tb_name, self._group_to_cores)
            if core.state == CoreState.FINISH or core.state == CoreState.READY:
                # if core.is_instant or is_instant:
                # if core_id == ((0, 0), (1, 0)):
                # if core.registers["instant_PI_en"]:
                #     core.MemoryOutputList[phase_num % (len(core.MemoryOutputList))].append(
                #         {"start": core.registers["Receive_PI_addr_base"]*4+0x8000, "length": 2})
                #     core.MemoryOutputList[phase_num % (len(core.MemoryOutputList))] = sorted(
                #         core.MemoryOutputList[phase_num % (len(core.MemoryOutputList))], key=lambda section: section["start"])
                self.print_core_phase_output(
                    core_id, phase_num, tb_name, is_instant)

    def print_phase_output(self, group_id: int, phase_num: int, tb_name: str):
        file_name = NameGenerator.out_file_name(group_id, phase_num + 1)
        file_path = Path.create_simulation_out_dir(tb_name) + file_name
        out_list = []
        cores = self._group_to_cores[group_id]
        cores = sorted(cores, key=lambda core: core.x)
        cores = sorted(cores, key=lambda core: core.y)
        for core in cores:
            out_list += core.print_phase_output_to_string(
                phase_num % (len(core.MemoryOutputList)))

        with open(file_path, "w") as f:
            f.write(hex_to_string(len(cores), width=8) + "\n")
            for line in out_list:
                f.write(line)

    def print_core_phase_output(self, core_id, phase_num, tb_name, is_instant=False):
        core = self._cores[core_id]
        step_num = phase_num_in_step = 0
        if is_instant:
            instant_num = phase_num-core.phase_num
            one_step_instant_len = core.instant_phase_num//core.cycles_number
            step_num = instant_num // one_step_instant_len
            phase_num_in_step = instant_num % one_step_instant_len
        else:
            one_step_static_len = core.phase_num//core.cycles_number
            step_num = phase_num // one_step_static_len
            phase_num_in_step = phase_num % one_step_static_len
        out_list = core.print_phase_output_to_string(
            phase_num % (len(core.MemoryOutputList)))
        file_name = NameGenerator.out_core_file_name(core_id, phase_num + 1)
        file_name = file_name.replace(' ', '')
        file_path = Path.create_simulation_out_dir(tb_name) + file_name

        with open(file_path, "w") as f:
            for line in out_list:
                f.write(line)

        if is_instant:
            file_name = NameGenerator.instant_pi_phase_out_core_file_name(
                core.group_id, core_id, phase_num_in_step, 0, step_num)
            file_path = Path.create_simulation_cmp_out_dir(tb_name) + file_name
            with open(file_path, "w") as f:
                for line in out_list[1:]:
                    f.write(line)
        else:
            file_name = NameGenerator.static_pi_phase_out_core_file_name(
                core.group_id, core_id, phase_num_in_step, step_num)
            file_path = Path.create_simulation_cmp_out_dir(tb_name) + file_name
            with open(file_path, "w") as f:
                for line in out_list[1:]:
                    f.write(line)

    def set_phase_clock(self, chip_id, group_id: int, clock: int, mode: int, clock0_in_step: int, clock1_in_step: int, trigger: int = 0):
        """
        设置phase组（0~31）的时钟数，并设置该组使用的trigger（0~3），每个phase的时钟数相同，
        :param group_id:要配置的phase组号，范围：0~31
        :param clock:设置“group_id”组的每个phase执行的clock数
        :param trigger:设置“group_id”组使用的trigger，范围：0~3
        :return:
        """
        if chip_id not in self._phase_clock:
            self._phase_clock[chip_id] = {}
        group_id = (chip_id, group_id)
        self._phase_clock[chip_id][group_id] = (
            clock, mode, clock0_in_step, clock1_in_step, trigger)

    def config_json_output(self, tb_name: str):
        json_configs = []
        for chip_id, cores in self._chip_to_cores.items():
            json_config = {
                "ChipID": {
                    "cx": chip_id[0],
                    "cy": chip_id[1],
                },
                "coreXMax": 0,
                "coreYMax": 0,
                "trigger": {},
                "ChipConfig": [],
                "CoreConfig": []
            }
            already_add_group = set()
            for core in cores:
                group_id = core.group_id[1]
                json_config["coreXMax"] = max(json_config["coreXMax"], core.x)
                json_config["coreYMax"] = max(json_config["coreYMax"], core.y)
                json_config["CoreConfig"].append(core.config_json_output())

                clock, mode, clock0_in_step, clock1_in_step, trigger = self._phase_clock[chip_id].get(
                    (chip_id, group_id), (1000, 0, None, None, 0))
                if group_id not in already_add_group:
                    config = {"PhaseGroup": group_id, "Sim_clock": clock,
                              "trigger": trigger, "P_adpt": mode}
                    json_config["ChipConfig"].append(config)
                already_add_group.add(group_id)
            step_clock = []
            for i in range(4):
                step_clock.append(self.step_clock.get(chip_id, {}).get(i, {}))
            json_config["step_clock"] = step_clock
            clock0_in_step_in_trigger_0 = step_clock[0]['clock0_in_step']
            start = [0, 0, 0, 0]
            high_level = [0, 0, 0, 0]
            low_level = [0, 0, 0, 0]
            if clock0_in_step_in_trigger_0 != None:
                start[0] = 1
                high_level[0] = clock0_in_step_in_trigger_0 * \
                    self._step_exe_num[chip_id]
                low_level[0] = 1
            json_config["trigger"]['start'] = start
            json_config["trigger"]['high'] = high_level
            json_config["trigger"]['low'] = low_level

            json_configs.append(json_config)

        return json_configs
