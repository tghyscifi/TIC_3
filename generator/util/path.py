import os
import platform

from generator.util.name_generator import NameGenerator


class Path(object):
    WORK_DIR = os.path.abspath(os.curdir)

    MAPPER_DIR = 'mapper/'
    GENERATOR_DIR = 'generator/'
    HARDWARE_DIR = 'simulator/'
    TEMP_DIR = 'temp/'

    COMPARE_RESULT_DIR = TEMP_DIR + 'compare_result/'
    SIMULATION_LOG_DIR = TEMP_DIR + 'out_files/'
    JSON_CONFID_DIR = TEMP_DIR + 'config/'

    # TODO: May cause problem
    JSON_CONFIG_ABSOLUTE_PATH = ""

    @staticmethod
    def json_config_path(tb_name):
        if not os.path.exists(Path.JSON_CONFID_DIR):
            os.mkdir(Path.JSON_CONFID_DIR)
        path = Path.JSON_CONFID_DIR + tb_name + 'Config.json'
        Path.JSON_CONFIG_ABSOLUTE_PATH = os.path.abspath(path)
        return path

    @staticmethod
    def json_config_absolute_path(tb_name):
        return os.path.abspath(Path.json_config_path(tb_name))

    @staticmethod
    def simulation_case_dir(tb_name):
        return Path.SIMULATION_LOG_DIR + tb_name + "/"

    @staticmethod
    def simulation_out_dir(tb_name):
        return Path.simulation_case_dir(tb_name) + "output/"

    @staticmethod
    def simulation_input_dir(tb_name, x, y):
        input_dir = NameGenerator.input_dir_name(x, y)
        return Path.simulation_case_dir(tb_name) + input_dir + "/"

    @staticmethod
    def simulation_cmp_out_dir(tb_name):
        if platform.system() == 'Windows':
            return Path.simulation_case_dir(tb_name) + "cmp_out" + "\\"
        else:
            return Path.simulation_case_dir(tb_name) + "cmp_out" + "/"

    @staticmethod
    def hardware_cmp_out_dir(tb_name):
        return Path.hardware_out_dir(tb_name) + "cmp_out" + "/"

    @staticmethod
    def hardware_out_dir(tb_name=''):
        dir = Path.HARDWARE_DIR + "Out_files/" + tb_name + "/"
        if tb_name == '':
            dir = Path.HARDWARE_DIR + "Out_files/"
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    @staticmethod
    def hardware_debug_message_dir():
        return Path.HARDWARE_DIR + "Debug_files/"

    @staticmethod
    def create_compare_result_dir():
        if not os.path.exists(Path.COMPARE_RESULT_DIR):
            os.mkdir(Path.COMPARE_RESULT_DIR)
        return Path.COMPARE_RESULT_DIR

    @staticmethod
    def create_simulation_case_dir(tb_name):
        if not os.path.exists(Path.SIMULATION_LOG_DIR):
            os.mkdir(Path.SIMULATION_LOG_DIR)
        if not os.path.exists(Path.simulation_case_dir(tb_name)):
            os.mkdir(Path.simulation_case_dir(tb_name))
        return Path.simulation_case_dir(tb_name)

    @staticmethod
    def create_simulation_input_dir(tb_name, x, y):
        Path.create_simulation_case_dir(tb_name)
        dir = Path.simulation_input_dir(tb_name, x, y)
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    @staticmethod
    def create_simulation_out_dir(tb_name):
        Path.create_simulation_case_dir(tb_name)
        dir = Path.simulation_out_dir(tb_name)
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    @staticmethod
    def create_simulation_cmp_out_dir(tb_name):
        Path.create_simulation_case_dir(tb_name)
        dir = Path.simulation_cmp_out_dir(tb_name)
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    @staticmethod
    def delete_all(path):
        if not os.path.exists(path):
            return
        if os.path.isdir(path):
            for name in os.listdir(path):
                sub_path = os.path.join(path, name)
                if os.path.isdir(sub_path):
                    Path.delete_all(sub_path)
                    os.rmdir(sub_path)
                else:
                    os.remove(sub_path)
        else:
            os.remove(path)
