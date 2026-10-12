class NameGenerator(object):
    @staticmethod
    def compare_file_name(tb_name, group_num, phase_num):
        return tb_name + "CompareResult_" + str(group_num) + "@" + str(phase_num) + ".txt"

    @staticmethod
    def out_file_name(group_num, phase_num):
        return "output_" + str(group_num) + "@" + str(phase_num) + ".txt"

    @staticmethod
    def out_core_file_name(core_id, phase_num):
        return "output_" + str(core_id).replace(' ', '') + "@" + str(phase_num) + ".txt"

    @staticmethod
    def static_pi_phase_out_core_file_name(core_phase_group_id, core_id, phase_num, step_num=0):
        return "cmp_out_" + str(core_phase_group_id[0][0]).replace(' ', '') + "_" + \
            str(core_phase_group_id[0][1]).replace(' ', '') + "_" + str(core_phase_group_id[1]) + "_" + \
            str(core_id[1][0]).replace(' ', '') + "_" + \
            str(core_id[1][1]).replace(' ', '') + "@" + str(step_num) + \
            "_" + str(phase_num) + ".txt"

    @staticmethod
    def instant_pi_phase_out_core_file_name(core_phase_group_id, core_id, index, perform_times=0, step_num=0):
        return "cmp_out_" + str(core_phase_group_id[0][0]).replace(' ', '') + "_" + \
            str(core_phase_group_id[0][1]).replace(' ', '') + "_" + str(core_phase_group_id[1]) + "_" + \
            str(core_id[1][0]).replace(' ', '') + "_" + str(core_id[1][1]).replace(' ', '') + \
            "@" + str(step_num) + "_" + str(index) + \
            "_" + str(perform_times) + ".txt"

    @staticmethod
    def input_dir_name(x, y):
        return str(x) + "_" + str(y) + "_input"

    @staticmethod
    def memory_name(x, y, phase_num_in_step):
        return str(x) + "_" + str(y) + "_phase" + str(phase_num_in_step) + "_RHead_in"

    @staticmethod
    def axon_input_file_name(phase_num, PIC):
        return str(phase_num) + "_Axon_" + str(hex(PIC)) + ".txt"

    @staticmethod
    def soma1_input_file_name(phase_num, PIC):
        return str(phase_num) + "_Soma1_" + str(hex(PIC)) + ".txt"

    @staticmethod
    def router_input_file_name(phase_num, PIC):
        return str(phase_num) + "_Router_" + str(hex(PIC)) + ".txt"

    @staticmethod
    def soma2_input_file_name(phase_num, PIC):
        return str(phase_num) + "_Soma2_" + str(hex(PIC)) + ".txt"
