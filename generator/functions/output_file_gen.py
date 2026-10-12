
import os
from generator.functions.util import hex_to_string
from generator.behavior_simulator.chip import Chip


class OutputFileGenerator(object):
    @staticmethod
    def output(chip: Chip, path: str):
        path = os.path.abspath(path) + '/'
        group_phase_num = {}
        group = chip.get_group_to_cores()
        for group_id, cores in group.items():
            cores = sorted(cores, key=lambda core: core.x)
            cores = sorted(cores, key=lambda core: core.y)
            for core in cores:
                file_list = os.listdir(path)
                for name in file_list:
                    if name.find(str((core.id)).replace(' ', '')) >= 0:
                        if group_id in group_phase_num:
                            group_phase_num[group_id] = max(group_phase_num[group_id], int(
                                name.split("@")[1].split(".")[0], 10))
                        else:
                            group_phase_num[group_id] = int(
                                name.split("@")[1].split(".")[0], 10)

        for group_id, phase_num in group_phase_num.items():
            for i in range(phase_num):  # output_0@1.txt
                cores = group[group_id]
                cores = sorted(cores, key=lambda core: core.x)
                cores = sorted(cores, key=lambda core: core.y)
                fp = open(path+"output_"+str(group_id[1]) +
                          "@"+str(i + 1)+".txt", "w")
                fp.write(hex_to_string(len(cores), width=8) + "\n")
                for core in cores:
                    name = path+"output_" + \
                        str(core.id).replace(' ', '') + \
                        "@" + str(i + 1) + ".txt"
                    if not os.path.exists(name):
                        fp.write(hex_to_string(0, width=8) + "\n")
                        continue
                    core_output_fp = open(name, "r")
                    data = core_output_fp.readlines()
                    fp.writelines(data)
                    core_output_fp.close()
                fp.close()

    @staticmethod
    def check_time_out(path: str):
        # print(os.listdir(path))
        for name in os.listdir(path):
            if name.find("timeout") != -1:
                # name:timeout_Chipx_Chipt_PhaseGroup_StepNum_Phase_Num
                id = name.split("timeout-")[1].split("-")
                print("chip:", ((int(id[0].split("_")[0]), int(id[0].split("_")[1]))), ",", "phase group:",
                      id[1], ",", "step", id[2], ",", "phase", id[3], ",", "time out phase")
