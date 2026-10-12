import os
import sys

sys.path.append(os.getcwd())
from generator.resnet50.resnet50_5chips.G17_check import check as g17_check
from generator.resnet50.resnet50_5chips.G16_check import check as g16_check
from generator.resnet50.resnet50_5chips.G15_ob_check import check as g15_ob_check
from generator.resnet50.resnet50_5chips.G15_check import check as g15_check
from generator.resnet50.resnet50_5chips.G14_check import check as g14_check
from generator.resnet50.resnet50_5chips.G13_check import check as g13_check
from generator.resnet50.resnet50_5chips.G12_ob_check import check as g12_ob_check
from generator.resnet50.resnet50_5chips.G12_check import check as g12_check
from generator.resnet50.resnet50_5chips.G11_check import check as g11_check
from generator.resnet50.resnet50_5chips.G10_check import check as g10_check
from generator.resnet50.resnet50_5chips.G9_1_check import check as g9_1_check
from generator.resnet50.resnet50_5chips.G9_2_check import check as g9_2_check
from generator.resnet50.resnet50_5chips.G1_check import check as g1_check
from generator.resnet50.resnet50_5chips.G1_ib_check import check as g1_ib_check
from generator.resnet50.resnet50_5chips.G0_OB_check import check as g0_ob_check
from generator.resnet50.resnet50_5chips.G2_check import check as g2_check
from generator.resnet50.resnet50_5chips.G3_check import check as g3_check
from generator.resnet50.resnet50_5chips.G4_check import check as g4_check
from generator.resnet50.resnet50_5chips.G4_OB_check import check as g4_ob_check
from generator.resnet50.resnet50_5chips.G5_14_check import check as g5_1_check
from generator.resnet50.resnet50_5chips.G5_28_check import check as g5_0_check
from generator.resnet50.resnet50_5chips.G6_check import check as g6_check
from generator.resnet50.resnet50_5chips.G7_check import check as g7_check
from generator.resnet50.resnet50_5chips.G8_check import check as g8_check
from generator.resnet50.resnet50_5chips.G8_OB_check import check as g8_ob_check
from generator.resnet50.resnet50_5chips.G16_ib_check import check as g16_ib_check
from generator.resnet50.resnet50_5chips.G13_ib_check import check as g13_ib_check
from generator.resnet50.resnet50_5chips.G9_IB_check import check as g9_ib_check
from generator.resnet50.resnet50_5chips.G5_IB_check import check as g5_ib_check
# from generator.resnet50.result_compare import ResultCompare
from generator.mapping_utils.result_compare_with_clock_specific_simulator import ResultCompareWithClockSpecificSimulator
from generator.resnet50.resnet50_5chips.data_gen import resnet50_data

case_file_name = 'Q00425'

# **** Group En ****
g0_en = 1

g1_ib_en = 1
g1_en = 1
g2_en = 1
g3_en = 1
g4_en = 1
g4_ob_en = 1

g5_ib_en = 1
g5_en = 1
g6_en = 1
g7_en = 1
g8_en = 1
g8_ob_en = 1

g9_ib_en = 1
g9_en = 1
g10_en = 1
g11_en = 1
g12_en = 1
g12_ob_en = 1

g13_ib_en = 1
g13_en = 1
g14_en = 1
g15_en = 1
g15_ob_en = 1

g16_ib_en = 1
g16_en = 1
g17_en = 1


empty_offset = 1

phase = None
step = 0
data_all = resnet50_data()
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

if g4_ob_en:
    g4_ob_check(data=data_all, compare=compare, empty_offset=empty_offset)

if g5_ib_en:
    g5_ib_check(data=data_all, compare=compare, empty_offset=empty_offset)

if g5_en:
    g5_0_check(data=data_all, compare=compare, size_x=(0, 14), size_y=(0, 2), chip=(0, 1), empty_offset=empty_offset)
    g5_1_check(data=data_all, compare=compare, size_x=(0, 14), size_y=(2, 3), chip=(0, 1), empty_offset=empty_offset)

if g6_en:
    g6_check(data=data_all, compare=compare, size_x=(0, 14), size_y=(3, 5), chip=(0, 1), empty_offset=empty_offset)

if g7_en:
    g7_check(data=data_all, compare=compare, size_x=(0, 14), size_y=(5, 7), chip=(0, 1), empty_offset=empty_offset)

if g8_en:
    g8_check(data=data_all, compare=compare, size_x=(0, 14), size_y=(7, 9), chip=(0, 1), empty_offset=empty_offset)

if g8_ob_en:
    g8_ob_check(data=data_all, compare=compare, empty_offset=empty_offset)

if g9_ib_en:
    g9_ib_check(data=data_all, compare=compare, empty_offset=empty_offset)

if g9_en:
    g9_1_check(data=data_all, compare=compare, size_y=(2, 6), size_x=(0, 8), chip=(0, 2), empty_offset=empty_offset)
    g9_2_check(data=data_all, compare=compare, size_y=(0, 2), size_x=(0, 8), chip=(0, 2), empty_offset=empty_offset)

if g10_en:
    g10_check(data=data_all, compare=compare, size_y=(6, 10), size_x=(0, 8), chip=(0, 2), empty_offset=empty_offset)

if g11_en:
    g11_check(data=data_all, compare=compare, size_y=(6, 10), size_x=(8, 16), chip=(0, 2), empty_offset=empty_offset)

if g12_en:
    g12_check(data=data_all, compare=compare, size_y=(2, 6), size_x=(8, 16), chip=(0, 2), empty_offset=empty_offset)

if g12_ob_en:
    g12_ob_check(data=data_all, compare=compare, chip=(0, 2), empty_offset=empty_offset)

if g13_ib_en:
    g13_ib_check(data=data_all, compare=compare, chip=(1, 1), empty_offset=empty_offset)

if g13_en:
    g13_check(data=data_all, compare=compare, size_y=(5, 9), size_x=(8, 16), chip=(1, 1), empty_offset=empty_offset)

if g14_en:
    g14_check(data=data_all, compare=compare, size_y=(5, 9), size_x=(0, 8), chip=(1, 1), empty_offset=empty_offset)

if g15_en:
    g15_check(data=data_all, compare=compare, size_y=(1, 5), size_x=(0, 16), chip=(1, 1), empty_offset=empty_offset)

if g15_ob_en:
    g15_ob_check(data=data_all, compare=compare, chip=(1, 1), empty_offset=empty_offset)

if g16_ib_en:
    g16_ib_check(data=data_all, compare=compare, chip=(1, 0), empty_offset=empty_offset)

if g16_en:
    g16_check(data=data_all, compare=compare, size_y=(5, 9), size_x=(0, 16), chip=(1, 0), empty_offset=empty_offset)

if g17_en:
    g17_check(data=data_all, compare=compare, size_y=(1, 5), size_x=(0, 16), chip=(1, 0), empty_offset=empty_offset)

compare.run()
compare.show_result()
