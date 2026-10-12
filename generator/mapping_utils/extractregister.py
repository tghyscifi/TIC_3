
import os


def extractcoremem(read_file_name, case_name, path_name):
    
    filepath = path_name + case_name + '_debug_rdCoreMem'

    if not os.path.exists(filepath):
        os.makedirs(filepath) 

    file_with_comment = {}
    file_without_comment = {}

    with open(read_file_name, 'r') as fr:
        for line in fr.readlines():
            if 'Core' not in line:
                continue
            
            core_id = int(line.split(' ')[5]), int(line.split(' ')[8])
            if core_id not in file_with_comment:
                core_file_name_1 = filepath + '/' + 'cRDcoreMem' + str(core_id) + '.txt'
                core_file_name_2 = filepath + '/' + 'RDcoreMem' + str(core_id) + '.txt'
                file_with_comment[core_id] = open(core_file_name_1, 'w')
                file_without_comment[core_id] = open(core_file_name_2, 'w')

            file_with_comment[core_id].write(line)
            file_without_comment[core_id].write(line.split('	')[0] + '\n')
    
    for key, file in file_with_comment.items():
        file.close()
        file_without_comment[key].close()



def extractAddr(read_file_name, write_cfile_name, write_file_name):
    with open(read_file_name, 'r') as fr:
        with open(write_cfile_name, 'w') as fw:    
            for line in fr.readlines():
                bank_name = line.split(' ')[2]
                if bank_name == 'Chip':
                    fw.writelines([line])
                addr_str = line.split('Start_Addr')[1].split(' ')[2]
                if addr_str[0] == '9' and len(addr_str) == 4:
                    fw.writelines([line])
    with open(write_cfile_name, 'r') as fr:
        with open(write_file_name, 'w') as fw:
            for line in fr.readlines():
                fw.writelines(line.split('	')[0])
                fw.write('\n')

case_select = 1
case_name = 'ST_041'
path_name = 'simulator/Out_files/' + case_name + '/'
read_file_name = path_name + 'chwcfg_ck_0_0.txt'

if case_select == 0:
    write_cfile_name = path_name + case_name + '_' + 'cRdregister_chwconfig_0_0.txt'
    write_file_name = path_name + case_name + '_' +'Rdregister_chwconfig_0_0.txt'
    extractAddr(read_file_name, write_cfile_name, write_file_name)
elif case_select == 1:
    extractcoremem(read_file_name, case_name, path_name)