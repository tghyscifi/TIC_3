import math
import numpy as np
import os
import warnings


class ResultCompare:
    def __init__(self, data_2_file_name: str, save_data_1_en: bool = False, phase_en=None, print_matched=True):
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
        self.data_2_path = './temp/out_files/' + data_2_file_name + '/output/'
        self.data_1_path = './temp/out_files/' + data_2_file_name + '/output_ref/'
        if phase_en is None:
            self.phase_en = math.inf
        else:
            self.phase_en = phase_en
        self.print_matched = print_matched
        self.match = True
        self.save_data_1_en = save_data_1_en
        self.data_1_list = []

    def add_ref_data(self, data: dict):
        self.data_1_list.append(data)

    def show_result(self):
        print('phase > {:} are not compared!!'.format(self.phase_en))
        if not self.match:
            print('******************************** Not Matched *******************************')
            warnings.warn('Not Matched')
        else:
            print('********************************** Matched **********************************')

    def run(self):
        for data_1 in self.data_1_list:
            for phase, data_phase in data_1.items():
                if phase > self.phase_en:
                    continue
                for ((chip_x, chip_y), (core_x, core_y)), data_core in data_phase.items():
                    file_name = 'output_(({:d},{:d}),({:d},{:d}))@{:d}.txt'.format(chip_x, chip_y, core_x, core_y,
                                                                                   phase + 1)
                    self.compare_with_file(data_core, file_name)
                    if self.save_data_1_en:
                        self.save_data(data_core, file_name)

    def compare_with_file(self, data_core, file_name):
        not_match_addr = []
        f_mem_block = []
        if not os.path.exists(self.data_2_path + file_name):
            warnings.warn('File ' + self.data_2_path + file_name + ' do not exist')
            self.match = False
        else:
            with open(self.data_2_path + file_name, 'r') as f:
                f_data = f.readlines()
                f_mem_block_num = int(f_data[0])
                start_line = 1
                while f_mem_block_num > 0:
                    f_mem_block.append(
                        {
                            'addr_start': int(f_data[start_line], 16),
                            'start_line': start_line + 2,
                            'length': int(f_data[start_line + 1], 16)
                        }
                    )
                    start_line += int(f_data[start_line + 1], 16) + 2
                    f_mem_block_num -= 1
                for mb in data_core:
                    find = False
                    for f_mb in f_mem_block:
                        if mb['addr_start'] == f_mb['addr_start']:
                            find = True
                            f_mb_data = ResultCompare.file2list(
                                f_data[f_mb['start_line']: f_mb['start_line'] + f_mb['length']], data_type=mb['type'])
                            if not f_mb_data == mb['data']:
                                if len(f_mb_data) != len(mb['data']):
                                    warnings.warn(
                                        'length do not match in {:0>8x}; length = {:d}, length_ref = {:d}!'.format(
                                            mb['addr_start'], len(f_mb_data), len(mb['data'])))
                                not_match_addr.append('{:0>8x}'.format(mb['addr_start']))
                            break
                    if not find:
                        not_match_addr.append('{:0>8x}'.format(mb['addr_start']))
                        print('not find address: {:x}'.format(mb['addr_start']))
                if len(not_match_addr) == 0:
                    if self.print_matched:
                        print(file_name, 'matched = True')
                else:
                    print(file_name, ', '.join(not_match_addr), 'matched = False')
                    self.match = False

    def save_data(self, data_core, file_name):
        if not os.path.exists(self.data_1_path):
            os.mkdir(self.data_1_path)
        file_list = ['{:0>8x}\n'.format(len(data_core))]
        data_core = sorted(data_core, key=lambda x: x['addr_start'])
        for mem_block in data_core:
            file_list.append('{:0>8x}\n'.format(mem_block['addr_start']))
            file_list.append('{:0>8x}\n'.format(len(mem_block['data'])))
            file_list.extend(ResultCompare.list2file(mem_block['data'], mem_block['type']))
        with open(self.data_1_path + file_name, 'w') as f:
            f.writelines(file_list)

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
    yyy = ResultCompare.list2file([[-21356], [12546], [0], [-1]], 0)
    xxx = 0
