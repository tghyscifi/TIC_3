import os
import codecs
import csv
import time
# import pandas as pd
import numpy as np
# import win32api, win32con
import time


#print (os.getcwd())
# C++ 仿真器输出文件目录
CSIM_PATH = "simulator/Out_files/"

# Python 仿真器输出文件目录
PSIM_PATH = "temp/out_files/"

# 输出case汇总信息csv文件目录
CSV_PATH = "simulator/Out_files/C05/"

# # Testcase输出目录
# TESTCASE_PATH = "../../TestCase/P02/"

def saveCSV(tb_name, Somaname, Axonname, pipeline):

    info_str = 'Soma{}'.format(Somaname)
    info_str +='与Router{}'.format(Axonname)
    info_str +='联调'
    if pipeline:
        info_str+=' ,行流水'
    else:
        info_str += ' ,无行流水'

    with open(CSV_PATH + 'caseInfo.csv', "a+", newline='') as csv_file:
        csv_file = csv.writer(csv_file)
        data = [tb_name]
        data.append(info_str)
        data.append('')
        data.append(time.strftime("%Y.%m.%d", time.localtime()))
        csv_file.writerow(data)

def saveInfoTxt(tb_name, Somaname, Axonname, clock):
    with codecs.open(CSIM_PATH + tb_name + "/" + tb_name + '测试用例说明.txt', 'w', encoding='utf-8') as f:
        f.write('Axon+Soma1测试用例_：' + str(tb_name) + '\n')

        f.write('\n')
        f.write('运行1个phase，时钟数被配置为 {} 个clock\n'.format(clock))
        f.write('\n')
        f.write('Axon原语参数：\n'.format(clock))
        with open(PSIM_PATH+tb_name+ "/" +"0_0_input"+"/"+"0_Axon_"+Axonname+".txt",'r') as paraFile:
            para = paraFile.readlines()
            f.writelines(para[0:38])
        f.write('\n')
        f.write('Soma原语参数：\n'.format(clock))
        with open(PSIM_PATH + tb_name + "/" + "0_0_input" + "/" + "0_Soma1_"+Somaname+".txt", 'r') as paraFile:
            para = paraFile.readlines()
            f.writelines(para[0:50])



def makerar(tb_name):
    rar_command = r'"D:\Program Files\WinRaR\WinRaR.exe"' + " a -r -ep1 " + CSIM_PATH + tb_name + "/" + tb_name + ".rar " + CSIM_PATH + tb_name + "/"
    print(rar_command)
    os.system(rar_command)
    copy_command = "copy " + CSIM_PATH + tb_name + "/" + tb_name + ".rar " + CSIM_PATH + "caseToday" + "/"
    print(copy_command)
    os.system(copy_command)

