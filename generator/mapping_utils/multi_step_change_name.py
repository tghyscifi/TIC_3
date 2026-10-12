import os
import shutil


def main(phase_num_by_group, file_dir=''):
    """
    [4, 30, 1, None]
    """
    print('Phase group number is : {:d}'.format(len(phase_num_by_group)))
    file_name_list = []
    for root, dirs, files in os.walk(file_dir, topdown=False):
        for name in files:
            if name[:7] == 'output_' and 'dbg' not in name:
                file_name_list.append(name)
    if not os.path.exists(file_dir + '/output_bak'):
        os.mkdir(file_dir + '/output_bak')
    for name in file_name_list:
        if os.path.exists(file_dir + '/output_bak/' + name):
            os.remove(file_dir + '/output_bak/' + name)
        shutil.move(file_dir + '/' + name, file_dir + '/output_bak')
    for name in file_name_list:
        _, group_phase, chip = name.split('_')
        group, phase = group_phase.split('@')
        chip = chip.split('.')[0]
        group, phase, chip = int(group), int(phase), int(chip)
        assert group < len(phase_num_by_group), 'group: {:d} is out of the index!'
        new_phase = (phase - 1) // phase_num_by_group[group] + phase
        new_name = 'output_{:d}@{:d}_{:d}.txt'.format(group, new_phase, chip)
        if os.path.exists(file_dir + '/' + new_name):
            os.remove(file_dir + '/' + new_name)
        shutil.copy(file_dir + '/output_bak/' + name, file_dir + '/' + new_name)


if __name__ == '__main__':
    phase_num_by_group = [12, 30, 3]
    file_dir = './simulator/Out_files/ST_045'
    main(phase_num_by_group=phase_num_by_group, file_dir=file_dir)
