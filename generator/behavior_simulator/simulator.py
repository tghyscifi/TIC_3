#!/usr/bin/env python
# coding: utf-8
from generator.code_generator import MapConfig, GroupConfig
from generator.behavior_simulator.PI_group import PIGroup
from generator.behavior_simulator.core import Core
from generator.behavior_simulator.chip import Chip
from generator.behavior_simulator.chip_array import ChipArray
from primitive import PrimitiveType
from primitive import Prim_08_lif, Prim_06_move_split
from generator.functions.output_file_gen import OutputFileGenerator


class Simulator(object):
    def __init__(self, map_config: MapConfig, target_group_phase):
        self._map_config = map_config
        self._target_group_phase = target_group_phase

        self._chip_array = ChipArray()
        self._chip_array.set_sim_clock(self._map_config.sim_clock)

    def map(self):
        for step_id, group_id, config in self._map_config:
            if not isinstance(config, GroupConfig):
                continue
            step_raw_id = step_id[0]
            step_id = step_id[1]
            for id in config.core_list:
                chip_id, core_id = id
                if (0, 0) not in self._chip_array:
                    self._chip_array.add_chip((0, 0), Chip())
                chip = self._chip_array.get_chip((0, 0))
                core = Core(chip_id, core_id, group_id)
                self._add_PI_groups(id, core, config)
                chip.add_core(id, core)
                if step_raw_id != chip_id:
                    raise Exception("core"+str(id)+" should be config to step group" +
                                    str((chip_id, 0))+",but "+str((step_raw_id, 0)))
                chip.set_phase_clock(chip_id, group_id, config.clock, config.mode,
                                     config.clock0_in_step, config.clock1_in_step, step_id)

                # TODO Step_id也要包含chip_id信息
                # chip.set_clock_in_step(chip_id, step_id,
                #                        self._map_config.get_clock0_in_step(
                #                            step_id), self._map_config.get_clock1_in_step(step_id))
            for y in range(4):          # FIXME 只支持3*3阵列
                for x in range(4):
                    for trigger in range(4):
                        clock0, clock1 = self._map_config.get_trigger_clock(
                            ((x, y), trigger))
                        chip = self._chip_array.get_chip((0, 0))
                        chip.set_trigger_clock((x, y), trigger, clock0, clock1)

        for step_id, group_id, config in self._map_config:
            if isinstance(step_id, tuple):
                cycles_number = self._map_config.get_cycles_number(step_id)
            else:
                continue
            if cycles_number == 1:
                chip = self._chip_array.get_chip((0, 0))
                chip.set_step_exe_num(step_id[0], cycles_number)
                continue
            if not isinstance(config, GroupConfig):
                continue
            for id in config.core_list:
                chip_id, core_id = id
                chip = self._chip_array.get_chip((0, 0))
                chip.set_step_exe_num(chip_id, cycles_number)
                core = chip.get_core(chip_id, *core_id)
                core.cycles_number = cycles_number
                core.extend_exec_list()

    @ staticmethod
    def _add_PI_groups(id, core, group_config: GroupConfig):

        core.set_registers(group_config.get_registers(id))

        axons = group_config.axon_list(id)
        soma1s = group_config.soma1_list(id)
        routers = group_config.router_list(id)
        soma2s = group_config.soma2_list(id)

        import numpy
        if core.registers['PI_loop_en'] and core.registers['PI_loop_num'] > 1:
            axons = list(numpy.tile(axons, core.registers['PI_loop_num']))
            soma1s = list(numpy.tile(soma1s, core.registers['PI_loop_num']))
            routers = list(numpy.tile(routers, core.registers['PI_loop_num']))
            soma2s = list(numpy.tile(soma2s, core.registers['PI_loop_num']))

        for i in range(len(axons)):
            pi = PIGroup()
            pi.add_PI(axons[i], PrimitiveType.Axon)
            pi.add_PI(soma1s[i], PrimitiveType.Soma1)
            pi.add_PI(routers[i], PrimitiveType.Router)
            pi.add_PI(soma2s[i], PrimitiveType.Soma2)

            Simulator._add_phase_output_print(
                pi, axons[i], soma1s[i], routers[i], soma2s[i])
            Simulator._init_memory(
                core, axons[i], soma1s[i], routers[i], soma2s[i])

            core.add_PI_group(pi)

        instant_pi_group = group_config.get_instant_pi_list(id)
        for pi_group in instant_pi_group:
            pi = PIGroup()
            pi.add_PI(pi_group["axon"], PrimitiveType.Axon)
            pi.add_PI(pi_group["soma1"], PrimitiveType.Soma1)
            pi.add_PI(pi_group["router"], PrimitiveType.Router)
            pi.add_PI(pi_group["soma2"], PrimitiveType.Soma2)

            Simulator._add_phase_output_print(
                pi, pi_group["axon"], pi_group["soma1"], pi_group["router"], pi_group["soma2"])
            Simulator._init_memory(
                core, pi_group["axon"], pi_group["soma1"], pi_group["router"], pi_group["soma2"])

            core.add_instant_PI_group(pi)

        core.set_pi_loop()

    @ staticmethod
    def _add_phase_output_print(pi, axon, soma1, router, soma2):
        line_buffer_ratio = 1
        if soma1 is not None and soma1.Row_ck_on == 1:
            if hasattr(soma1, 'Input_fm_Py'):
                line_buffer_ratio = soma1.Input_fm_Py / \
                    soma1.in_row_max
            elif hasattr(soma1, 'num_in'):
                line_buffer_ratio = soma1.num_in / soma1.in_row_max
            else:  # hasattr(soma1, 'group_num'):
                line_buffer_ratio = soma1.group_num / soma1.in_row_max

        line_buffer_ratio = max(1, line_buffer_ratio)
        if axon is not None:
            if not hasattr(axon, "axon_delay") or not axon.axon_delay:
                pi.add_phase_output_print(
                    axon.Addr_V_base, axon.Write_V_length / line_buffer_ratio)
        if soma1 is not None:
            if soma1.mem_sel == 0:
                length = 0
                if hasattr(soma1, 'S1_out_length'):
                    length = soma1.S1_out_length
                else:
                    if isinstance(soma1, Prim_06_move_split) and \
                            soma1.out_ciso_sel == 1:
                        length = soma1.Write_ciso_length
                    else:
                        length = soma1.Write_Y_length

                if router is not None and router.Soma_in_en == 1:
                    length = (router.Addr_Dout_length + 1) * 4
                if isinstance(soma1, Prim_08_lif):
                    pi.add_phase_output_print(soma1.Addr_S_Start, length)
                    pi.add_phase_output_print(
                        soma1.Addr_V_start, soma1.Read_V_length)
                    pi.add_phase_output_print(
                        soma1.Addr_Vtheta_start, soma1.Read_Vtheta_length)
                    pi.add_phase_output_print(
                        soma1.Addr_para, soma1.S1_para_length)
                elif isinstance(soma1, Prim_06_move_split):
                    if soma1.out_ciso_sel == 0:
                        pi.add_phase_output_print(
                            soma1.Addr_Start_ciso, soma1.Write_ciso_length)
                        pi.add_phase_output_print(soma1.Addr_Start_out, length)
                    else:
                        pi.add_phase_output_print(
                            soma1.Addr_Start_ciso, length)
                        pi.add_phase_output_print(
                            soma1.Addr_Start_out, soma1.Write_Y_length)
                else:
                    pi.add_phase_output_print(
                        soma1.Addr_Start_out, length)
            else:
                if isinstance(soma1, Prim_06_move_split):
                    if soma1.out_ciso_sel == 0:
                        pi.add_phase_output_print(
                            soma1.Addr_Start_ciso, soma1.Write_ciso_length)
                    else:
                        pi.add_phase_output_print(
                            soma1.Addr_Start_out, soma1.Write_Y_length)
        if router is not None:
            if router.Receive_en:
                pi.add_phase_output_print(
                    0x8000 + router.Addr_Din_base, (router.Addr_Din_length + 1) * 2)
        if soma2 is not None:
            if isinstance(soma2, Prim_08_lif):
                pi.add_phase_output_print(
                    soma2.Addr_S_Start, soma2.S1_out_length)
                pi.add_phase_output_print(
                    soma2.Addr_V_start, soma2.Read_V_length)
                pi.add_phase_output_print(
                    soma2.Addr_Vtheta_start, soma2.Read_Vtheta_length)
                pi.add_phase_output_print(
                    soma2.Addr_para, soma2.S1_para_length)
            elif isinstance(soma2, Prim_06_move_split):
                pi.add_phase_output_print(
                    soma2.Addr_Start_ciso, soma2.Write_ciso_length)
                pi.add_phase_output_print(
                    soma2.Addr_Start_out, soma2.Write_Y_length)
            else:
                pi.add_phase_output_print(
                    soma2.Addr_Start_out, soma2.Write_Y_length)

    @ staticmethod
    def _init_memory(core, axon, soma1, router, soma2):
        for one_pi in (axon, soma1, router, soma2):
            if one_pi is not None:
                Simulator._init_memory_for_one_pi(core, one_pi)

    @ staticmethod
    def _init_memory_for_one_pi(core, one_pi):
        for init_config in one_pi.memory_blocks:
            name = init_config['name']
            start = init_config['start']
            data = init_config['data']
            length = init_config.get('length', len(data))
            if 'initialize' in init_config and not init_config['initialize']:
                continue
            core.init_memory(name, start, length, data)

    def simulate(self, tb_name):
        group_max = {}
        for group, phase in self._target_group_phase:
            if group in group_max:
                group_max[group] = max(phase, group_max[group])
            else:
                group_max[group] = phase
        self._chip_array.simulate(tb_name)
        # self._chip_array.simulate_all(tb_name)
        json_config = self._chip_array.config_json_output(tb_name)

        from generator.util.path import Path
        for chip in self._chip_array.get_chips().values():
            OutputFileGenerator.output(chip,
                                       Path.create_simulation_out_dir(tb_name))
        return json_config
