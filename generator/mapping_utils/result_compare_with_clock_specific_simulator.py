import warnings
from math import inf
import os
import numpy as np


class ResultCompareWithClockSpecificSimulator:
    def __init__(self, data_file_name: str, save_ref_data_en: bool = False, phase_en=None, print_matched=True, step=0):
        """
            data_1: {
                0: {  # Phase
                    ((0, 0), (0, 0)): [
                        {
                            data: [],
                            addr_start: 0x1000,  # int
                            type: 1  # int - data type
                        }
                    ]
                }
            }
            data_2_file_name: 'R00001'
            save_data_1: save data 1 to compare
            phase_en: <= phase_en will be compared
        """
        self.data_path = './simulator/Out_files/' + data_file_name + '/cmp_out/'
        self.data_ref_path = './simulator/Out_files/' + data_file_name + '/cmp_out_ref/'
        self.step = step
        if phase_en is None:
            self.phase_en = inf
        else:
            self.phase_en = phase_en
        self.print_matched = print_matched
        self.match = True
        self.save_ref_data_en = save_ref_data_en
        self.data_ref_list = []
        self.phase_group_by_core = {}

    def add_ref_data(self, data: dict):
        self.data_ref_list.append(data)

    def run(self):
        self._update_phase_group_by_core()
        for data_1 in self.data_ref_list:
            for phase, data_phase in data_1.items():
                if phase > self.phase_en:
                    continue
                for ((chip_x, chip_y), (core_x, core_y)), data_core in data_phase.items():
                    if self.phase_group_by_core.get(((chip_x, chip_y), (core_x, core_y))) is None:
                        warnings.warn('Core(({:}, {:}), ({:}, {:})) does not exist!'.format(
                            chip_x, chip_y, core_x, core_y))
                        continue
                    phase_group = self.phase_group_by_core[((chip_x, chip_y), (core_x, core_y))]
                    file_name = 'cmp_out_{:d}_{:d}_{:d}_{:d}_{:d}@{:d}_{:d}.txt'.format(
                        chip_x, chip_y, phase_group, core_x, core_y, self.step, phase)
                    self.compare_with_file(data_core, file_name)
                    if self.save_ref_data_en:
                        self.save_data(data_core, file_name)

    def _update_phase_group_by_core(self):
        for file in os.listdir(self.data_path):
            if file[0:7] == 'cmp_out':
                _, _, chip_x, chip_y, phase_group, core_x, core_y = file.split('@')[0].split('_')
                chip_x, chip_y, phase_group, core_x, core_y = int(chip_x), int(chip_y), int(phase_group), int(
                    core_x), int(core_y)
                if self.phase_group_by_core.get(((chip_x, chip_y), (core_x, core_y))) is not None:
                    if self.phase_group_by_core[((chip_x, chip_y), (core_x, core_y))] != phase_group:
                        raise ValueError(
                            'core (({:}, {:}), ({:}, {:})) can not belong to two different phase groups!'.format(
                                chip_x, chip_y, core_x, core_y))
                self.phase_group_by_core[((chip_x, chip_y), (core_x, core_y))] = phase_group

    def save_data(self, data_core, file_name, print_mem_block_num: bool = False):
        if not os.path.exists(self.data_ref_path):
            os.mkdir(self.data_ref_path)
        if print_mem_block_num:
            file_list = ['{:0>8x}\n'.format(len(data_core))]
        else:
            file_list = []
        data_core = sorted(data_core, key=lambda x: x['addr_start'])
        for mem_block in data_core:
            file_list.append('{:0>8x}\n'.format(mem_block['addr_start']))
            file_list.append('{:0>8x}\n'.format(len(mem_block['data'])))
            file_list.extend(ResultCompareWithClockSpecificSimulator.list2file(mem_block['data'], mem_block['type']))
        with open(self.data_ref_path + file_name, 'w') as f:
            f.writelines(file_list)

    def show_result(self):
        print('phase > {:} are not compared!!'.format(self.phase_en))
        if not self.match:
            print('******************************** Not Matched *******************************')
            warnings.warn('Not Matched')
        else:
            print('********************************** Matched **********************************')

    def compare_with_file(self, ref_data_core, file_name):
        not_match_addr = []
        file_mem_blocks, analyse_success = ResultCompareWithClockSpecificSimulator.analyse_one_file(
            self.data_path + file_name)
        if not analyse_success:
            self.match = False
        for ref_mem_block in ref_data_core:
            if file_mem_blocks.get(ref_mem_block['addr_start']) is None:
                not_match_addr.append('{:0>8x}'.format(ref_mem_block['addr_start']))
                continue
            if file_mem_blocks[ref_mem_block['addr_start']].get(len(ref_mem_block['data'])) is None:
                warnings.warn('length do not match in {:0>8x}; length_ref = {:d}, but lenght in the file is '.format(
                    ref_mem_block['addr_start'], len(ref_mem_block['data'])) + str(
                    file_mem_blocks[ref_mem_block['addr_start']].keys()))
                not_match_addr.append('{:0>8x}'.format(ref_mem_block['addr_start']))
            if file_mem_blocks[ref_mem_block['addr_start']].get(len(ref_mem_block['data'])) is None:
                raise ValueError(
                    'Data length error! ref data length is {}, while length of data in address {:x} is {}'.format(
                        len(ref_mem_block['data']), ref_mem_block['addr_start'],
                        [length for length in file_mem_blocks[ref_mem_block['addr_start']].keys()]))
            file_mem_block_data = ResultCompareWithClockSpecificSimulator.file2list(
                file_mem_blocks[ref_mem_block['addr_start']][len(ref_mem_block['data'])],
                data_type=ref_mem_block['type'])
            if not file_mem_block_data == ref_mem_block['data']:
                if len(file_mem_block_data) != len(ref_mem_block['data']):
                    warnings.warn('length do not match in {:0>8x}; length = {:d}, length_ref = {:d}!'.format(
                        ref_mem_block['addr_start'], len(file_mem_block_data), len(ref_mem_block['data'])))
                not_match_addr.append('{:0>8x}'.format(ref_mem_block['addr_start']))
        if len(not_match_addr) == 0:
            if self.print_matched:
                print(file_name, 'matched = True')
        else:
            print(file_name, ', '.join(not_match_addr), 'matched = False')
            self.match = False

    @staticmethod
    def analyse_one_file(file_path: str):
        file_mem_blocks = {}
        success = True
        if not os.path.exists(file_path):
            warnings.warn('File ' + file_path + ' do not exist')
            success = False
        else:
            with open(file_path, 'r') as f:
                file_lines = f.readlines()
                line_num = len(file_lines)
                start_line = 0
                while start_line < line_num:
                    addr_start, addr_length = int(file_lines[start_line], 16), int(file_lines[start_line + 1], 16)
                    if file_mem_blocks.get(addr_start) is not None:
                        if file_mem_blocks[addr_start].get(addr_length) is not None:
                            if file_mem_blocks[addr_start][addr_length] != \
                                    file_lines[start_line + 2: start_line + 2 + addr_length]:
                                warnings.warn(
                                    'Addr {:x} can not have different values in same Core {}'.format(addr_start,
                                                                                                     file_path))
                        else:
                            file_mem_blocks[addr_start][addr_length] = file_lines[
                                                                       start_line + 2: start_line + 2 + addr_length]
                    else:
                        file_mem_blocks[addr_start] = {}
                        file_mem_blocks[addr_start][addr_length] = file_lines[
                                                                   start_line + 2: start_line + 2 + addr_length]
                    start_line += addr_length + 2
                if start_line != line_num:
                    raise ValueError('Error occur when analyse file \'{}\''.format(file_path))
        return file_mem_blocks, success

    @staticmethod
    def file2list(x: list, data_type: int):
        if data_type == 0:
            result = np.array([[int(i, 16)] for i in x]).astype(np.int32).tolist()
        elif data_type == 1:
            result = np.array([[int(i[6:8], 16), int(i[4:6], 16), int(i[2:4], 16), int(i[0:2], 16)] for i in x],
                              dtype=np.int8).tolist()
        else:
            raise ValueError
        return result

    @staticmethod
    def list2file(x: list, data_type: int):
        if data_type == 0:
            x = np.array(x, dtype=np.uint32).tolist()
            result = ['{:0>8x}\n'.format(i[0]) for i in x]
        elif data_type == 1:
            x = np.array(x, dtype=np.uint8).tolist()
            result = ['{:0>8x}\n'.format((i[3] << 24) + (i[2] << 16) + (i[1] << 8) + i[0]) for i in x]
        else:
            raise ValueError
        return result


if __name__ == '__main__':
    c = ResultCompareWithClockSpecificSimulator(data_file_name='Q00001', save_ref_data_en=True, phase_en=0,
                                                print_matched=True, step=0)
    c.analyse_one_file('./simulator/Out_files/Q00001/cmp_out/cmp_out_2_0_2_7_7@0_15.txt')
    xx = 1
