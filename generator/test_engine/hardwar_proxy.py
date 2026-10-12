#!/usr/bin/env python
# coding: utf-8

import os
from generator.util import Path, Command


class HardwareProxy(object):
    class HardwareExecutionError(RuntimeError):
        pass

    @staticmethod
    def execute(tb_name, exe_name='TianjicX1_SIM.exe'):
        # 启动c++仿真器执行脚本
        os.chdir(Path.HARDWARE_DIR)
        execute_result = Command.execute_hardware(tb_name, executable=True, exe_name=exe_name)
        os.chdir(Path.WORK_DIR)

        if execute_result != 0:
            raise HardwareProxy.HardwareExecutionError()

    @staticmethod
    def clear_output_dirtory(tb_name):
        path = Path.hardware_out_dir(tb_name)
        # files = os.listdir(path)
        for file in os.listdir(path):
            full_path = os.path.abspath(path)+"/"+file
            if os.path.isdir(full_path):
                for sub_dir_file in os.listdir(full_path):
                    full_sub_path = full_path+"/"+sub_dir_file
                    os.remove(full_sub_path)
                continue
            os.remove(full_path)
