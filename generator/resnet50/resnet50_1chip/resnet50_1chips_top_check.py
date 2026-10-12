import os
import sys

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_1chip.G1_check import check as g1_check
from generator.resnet50.resnet50_1chip.G1_ib_check import check as g1_ib_check
from generator.resnet50.resnet50_1chip.G0_OB_check import check as g0_ob_check
from generator.resnet50.resnet50_1chip.G2_check import check as g2_check
from generator.resnet50.resnet50_1chip.G3_check import check as g3_check
from generator.resnet50.resnet50_1chip.G4_check import check as g4_check
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from generator.resnet50.resnet50_1chip.data_gen import resnet50_data

case_file_name = 'Q00731'

# **** Group En ****
g0_en = 1

g1_ib_en = 1
g1_en = 1
g2_en = 1
g3_en = 1
g4_en = 1


empty_offset = 1

phase = None
step = 0
data_all = resnet50_data(ckpt=0)
compare = ResultCompareWithClockSpecificSimulator(data_file_name=case_file_name, save_ref_data_en=True, phase_en=phase,
                                                  print_matched=True, step=step)

if g0_en:
    g0_ob_check(data=data_all, compare=compare, empty_offset=empty_offset)

if g1_ib_en:
    g1_ib_check(data=data_all, compare=compare, empty_offset=empty_offset)

if g1_en:
    g1_check(data=data_all, compare=compare, size_y=(1, 3), size_x=(0, 16), chip=(0, 0), empty_offset=empty_offset)

if g2_en:
    g2_check(data=data_all, compare=compare, size_y=(3, 5), size_x=(2, 16), chip=(0, 0), empty_offset=empty_offset)

if g3_en:
    g3_check(data=data_all, compare=compare, size_y=(5, 7), size_x=(2, 16), chip=(0, 0), empty_offset=empty_offset)

if g4_en:
    g4_check(data=data_all, compare=compare, size_y=(7, 9), size_x=(2, 16), chip=(0, 0), empty_offset=empty_offset)

compare.run()
compare.show_result()
