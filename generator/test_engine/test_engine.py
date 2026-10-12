from enum import Enum
import json
from generator.behavior_simulator import Simulator
from generator.test_engine.hardwar_proxy import HardwareProxy
from generator.code_generator import MapConfig
from generator.test_engine.test_config import TestConfig
from generator.util import Path, Command, NameGenerator
from generator.functions.output_file_gen import OutputFileGenerator
import os
import platform


class TestMode(Enum):
    MEMORY_STATE = 0
    PRIM_OUTPUT = 1


class TestEngine(object):
    def __init__(self, map_config, test_config):
        self._map_config = MapConfig(map_config)
        self._test_config = TestConfig(test_config)
        self._test_mode = self._test_config.test_mode

    def run_test(self, exe_name='TianjicX1_SIM.exe'):
        simulator = self.construct_simulator()
        # simulation_config = self.construct_simulation_config()
        simulator.map()
        Path.delete_all(Path.simulation_case_dir(self._test_config.tb_name))
        Path.delete_all(Path.hardware_out_dir(self._test_config.tb_name))
        Path.delete_all(Path.hardware_debug_message_dir())
        json_config = simulator.simulate(self._test_config.tb_name)
        json_config.update(
            {'debug_file_switch': self._test_config.debug_file_switch()})
        json.dump(json_config, open(Path.json_config_path(self._test_config.tb_name), "w"),
                  ensure_ascii=False, indent=4)
        # 运行硬件，严格说这个函数不能放在Simulator里
        # HardwareProxy.clear_output_dirtory(self._test_config.tb_name)

        if platform.system() == 'Windows':
            HardwareProxy.execute(self._test_config.tb_name, exe_name=exe_name)
            return self.compare_cmp_out_dir_result()
        else:
            return True

        # return self.compare_result()

    def construct_simulator(self):
        if self._test_mode == TestMode.MEMORY_STATE:
            return Simulator(self._map_config, self._test_config.test_group_phase)
        else:
            return Simulator(self._map_config, self._test_config.test_group_phase)

    def construct_simulation_config(self):
        return {}

    def compare_result(self):
        result = True
        for group_num, phase_num in self._test_config.test_group_phase:
            result = self.compare_result_for_one_group_phase(
                group_num, phase_num) and result
        return result

    def compare_result_for_one_group_phase(self, group_num, phase_num):
        tb_name = self._test_config.tb_name

        compare_name = NameGenerator.compare_file_name(
            tb_name, group_num, phase_num)
        compare_path = Path.create_compare_result_dir() + compare_name

        output_file_name = NameGenerator.out_file_name(group_num, phase_num)
        simulate_out = Path.simulation_out_dir(tb_name) + output_file_name
        hardware_out = Path.hardware_out_dir(tb_name) + output_file_name

        # 打印当前时间至输出结果对比文件中
        Command.output_time_to_file(compare_path)
        # 对比文件
        not_match = Command.compare_files(
            hardware_out, simulate_out, compare_path)
        # 打印对比返回值，0：完全匹配 1：不匹配
        print(output_file_name, "isMatching = ", not not_match)
        # 打开对比输出文本
        if not_match:
            # Command.open_and_show_file(compare_path)
            return False
        else:
            return True

    def compare_cmp_out_dir_result(self):
        result = True
        test_groups = {g for g, p in self._test_config.test_group_phase}
        test_phases = {p for g, p in self._test_config.test_group_phase}
        for file_name in os.listdir(Path.simulation_cmp_out_dir(self._test_config.tb_name)):
            parts = file_name.replace('.txt', '').split('@')
            name_prefix = parts[0].split('_')
            suffix = parts[1].split('_')
            if len(name_prefix) >= 6 and len(suffix) >= 2:
                file_group = int(name_prefix[3])
                file_phase = int(suffix[-1])
                if file_group not in test_groups or file_phase not in test_phases:
                    continue
            result = self.compare_cmp_out_dir_result_for_one_file(file_name, Path.simulation_cmp_out_dir(
                self._test_config.tb_name), Path.hardware_cmp_out_dir(self._test_config.tb_name)) and result
        OutputFileGenerator.check_time_out(
            Path.hardware_cmp_out_dir(self._test_config.tb_name))
        return result

    def compare_cmp_out_dir_result_for_one_file(self, file_name: str, simulator_path: str, hardware_path: str):

        not_match = Command.compare_files(
            simulator_path+file_name, hardware_path+file_name)
        print(file_name, "isMatching = ", not not_match)
        # if not_match:
        #     print(file_name, "isMatching = ", not not_match)
        return False if not_match else True
