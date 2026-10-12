import os
import time

from generator.util.path import Path
from typing import overload


class Command(object):
    @staticmethod
    def execute_hardware(tb_name, executable=True, exe_name='TianjicX1_SIM999.exe'):
        if executable:
            return os.system(exe_name + " " + Path.JSON_CONFIG_ABSOLUTE_PATH + " " + tb_name)
        else:
            return os.system("top.bat " + tb_name + " " + Path.JSON_CONFIG_ABSOLUTE_PATH)

    @staticmethod
    def output_time_to_file(file_path):
        os.system("echo " + time.strftime('%y-%m-%d %H:%M:%S',
                                          time.localtime()) + "> " + file_path)

    @staticmethod
    def compare_files(file1, file2, result_file=None):
        if result_file != None:
            return os.system("fc " + file1 + " " + file2 + " /N >> " + result_file)
        else:
            return os.system("fc " + file1 + " " + file2 + " /N >> nul")

    @staticmethod
    def open_and_show_file(file_path):
        os.system(file_path)
