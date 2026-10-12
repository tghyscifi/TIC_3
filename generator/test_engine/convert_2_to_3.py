import os
from bidict import bidict
import numpy
from collections import OrderedDict

prim_to_value = {}
clocks = []
tb_names = []

def is_test_case(path, prefix):
    pathes = path.split('/')
    case_file = pathes[len(pathes) - 1]
    if case_file.startswith(prefix) and case_file.endswith('.py'):
        return True
    return False


def get_all_files(dir, prefix):
    files = []
    file_list = os.listdir(dir)
    for i in range(0, len(file_list)):
        path = os.path.join(dir, file_list[i])
        if os.path.isdir(path):
            files.extend(get_all_files(path, prefix))
        if os.path.isfile(path):
            if is_test_case(path, prefix):
                files.append(path)
    return files

def format_convert(dir):
    files = get_all_files(dir, prefix='Testcase_')

    # add def test_case
    for file_name in files:
        code = ''

        with open(file_name, 'r', encoding='utf-8') as f:
            code = f.read()
        code = code.replace('Source.Prim', 'primitive')
        code = code.replace('Source.Functions', 'generator.functions')
        code = code.replace('..//..//Out_files//', 'temp//out_files//')
        code = code.replace('TBNAME', 'tb_name')
        code = code.replace('Testcase_', 'test_')
        code = code.replace('\'..//..//Config.json\'', '\'temp//config//\'+tb_name+\'Config.json\'')
        code = code.replace('\"..//..//Config.json\"', '\'temp//config//\'+tb_name+\'Config.json\'')
        code = code.replace('\"../../Config.json\"', '\'temp//config//\'+tb_name+\'Config.json\'')
        code = code.replace('ConfigJson', 'generator.ConfigJson')
        code = code.replace('init_data_gen', 'init_data')
        code = code.replace(', a[0])', ', a[0], a)')
        code = code.replace(', a[1])', ', a[1], a)')
        code = code.replace(', a[2])', ', a[2], a)')
        code = code.replace(', a[3])', ', a[3], a)')
        code = code.replace(', b[0])', ', b[0], b)')
        code = code.replace(', b[1])', ', b[1], b)')
        code = code.replace('simluator.compareResult(0, 1)',
                            'assert simluator.compareResult(0, 1)')
        code = code.replace('simluator.compareResult(0,1)',
                            'assert simluator.compareResult(0, 1)')
        code = code.replace('assert assert', 'assert')
        
        if 'def test_case()' in code:
            code = code.replace('assert simluator.compareResult(0, 1)',
                     'assert simluator.compareResult(0, 1)\n    return chip1')
            continue

        lines =  ['    ' + line + '\n' for line in code.split('\n')]
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write('# coding: utf-8\n\n')
            f.write('def test_case():\n')
            f.writelines(lines)
            f.write('    return chip1\n')
            f.write('\nif __name__ =="__main__":\n')
            f.write('    test_case()\n')

def add_to_prims(new_prims, prims):
    num = len(prims)
    for prim in new_prims:
        if prim is not None and prim not in prims:
            prims[prim] = 'prim' + str(num)
            num += 1


def convert_all_cases(dir, prefix):
    for file in get_all_files(dir, prefix):
        convert_one_case(file)
    #print_case_param()


def convert_one_case(path: str):
    case_path = path.replace('/', '.')
    case_path = case_path.replace('.py', '')
    imports = {}
    exec('from ' + case_path + ' import test_case', imports)
    test_case = imports['test_case']
    chip = test_case()
    if chip is None:
        return

    print('Converting case: ', path)

    original_code = ''
    with open(path, 'r', encoding='utf-8') as f:
        original_code = f.read()
    original_code = original_code.split('init_data')[0]

    prims = bidict({None: None})
    step_config = {
        'clock': None
    }

    clock = 0
    for group in chip.phase_groups:
        clock = chip.group_clock[group][0]
        group_config = {
            'clock': clock,
            'trigger': chip.group_clock[group][1]
        }

        cores_dict = chip.phase_group_to_cores(group)
        for id, core in cores_dict.items():
            new_prims = []
            new_prims.extend(core.axon_list)
            new_prims.extend(core.soma1_list)
            new_prims.extend(core.router_list)
            new_prims.extend(core.soma2_list)
            add_to_prims(new_prims, prims)
            core_config = {
                'axon': [prims[p] for p in core.axon_list],
                'soma1': [prims[p] for p in core.soma1_list],
                'router': [prims[p] for p in core.router_list],
                'soma2': [prims[p] for p in core.soma2_list]
            }

            group_config[id] = core_config

        step_config[group] = group_config

    map_config = {0: step_config}

    tb_name = chip.tb_name

    from generator.test_engine import TestMode
    test_config = {
        'tb_name': tb_name,
        'test_mode': TestMode.MEMORY_STATE,
        'test_group_phase': [(0, 1)]    # (phase_group, phase_num)
    }
    code = case_to_string(prims, map_config, test_config, tb_name, original_code)
    code = code.replace('\t', '    ')
    print_code_to_new_file(code, path, tb_name)
    clocks.append(clock)
    tb_names.append(tb_name)


def case_to_string(prims, map_config, test_config, tb_name, original_code):
    code = '# coding: utf-8\n\n'
    code += 'import os\n'
    code += 'import numpy as np\n'
    code += 'from generator.test_engine import TestMode, TestEngine\n\n'

    code += 'def test_case():\n'
    code += '\ttb_name = str(os.path.basename(__file__)).split("test_")[1].split(".")[0]\n'
    code += '\tnp.random.seed(sum(ord(c) for c in tb_name))\n\n'
    code += prim_to_string(prims, original_code)
    code += map_config_to_string(map_config, prims)
    code += test_config_to_string(test_config, tb_name)

    code += '\ttester = TestEngine(map_config, test_config)\n'
    code += '\tassert tester.run_test()\n\n'
    code += 'if __name__ =="__main__":\n'
    code += '\ttest_case()\n'

    return code


def prim_to_string(prims, original_code):
    prims_type = [p.__class__.__name__ for p in prims.keys() if p is not None]
    code = '\tfrom primitive import '
    for i, prim_name in enumerate(prims_type):
        code += prim_name
        if i < len(prims_type) - 1:
            code += ', '

    code += '\n'

    for prim, name in prims.items():
        if prim is None:
            continue
        code += '\t' + name + ' = ' + prim.__class__.__name__ + '()\n'
        for key, value in prim.attributes.items():
            if key in ['InputX_array', '_memory_blocks', 'Bias_array', 'A_array',
                       'X_array', 'LUT_array', 'array_in']:
                continue
            if isinstance(value, list) or isinstance(value, numpy.ndarray):
                print('Also ingnore: ', key)
                continue
            if key not in original_code:
                continue
            
            code += '\t' + name + '.' + key + ' = ' + str(value) + '\n'

            prim_dict = OrderedDict()
            if prim.__class__ in prim_to_value:
                prim_dict = prim_to_value[prim.__class__]
            else:
                prim_to_value[prim.__class__] = prim_dict
            
            if key in prim_dict:
                prim_dict[key].append(value)
            else:
                prim_dict[key] = []
                prim_dict[key].append(value)

        code += '\t' + name + '_in = ' + name + '.init_data()\n'
        code += '\t' + name + '.memory_blocks = [\n'
        for block in prim._memory_blocks:
            code += '\t\t{\'name\': \'' + str(block['name']) + '\',\n'
            code += '\t\t \'start\': ' + str(block['start']) + ',\n'
            code += '\t\t \'length\': ' + str(block['length']) + ',\n'
            code += '\t\t \'data\': ' + name + \
                '_in[' + str(block['data']) + '],\n'
            code += '\t\t \'mode\': ' + str(block['mode']) + '},\n'
        code += '\t]\n'
        code += '\n'
    return code


def map_config_to_string(map_config, prims):
    map_string = get_indent_string(map_config)
    for prim, name in prims.items():
        if prim is not None:
            map_string = map_string.replace('\'' + name + '\'', name)
    code = '\tmap_config = ' + map_string
    code += '\n\n'
    return code


def test_config_to_string(test_config, tb_name):
    test_string = get_indent_string(test_config)
    test_string = test_string.replace(
        '<TestMode.MEMORY_STATE: 0>', 'TestMode.MEMORY_STATE')
    test_string = test_string.replace('\'' + tb_name + '\'', 'tb_name')
    code = '\ttest_config = ' + test_string
    code += '\n\n'
    return code


def get_indent_string(config_dict, indent=1):
    indent_one = '{\n'
    indent += 1
    for key, value in config_dict.items():
        value_string = str(value)
        if isinstance(value, dict):
            value_string = get_indent_string(value, indent)
        if isinstance(value, str):
            value_string = '\'' + value_string + '\''
        key_string = str(key)
        if isinstance(key, str):
            key_string = '\'' + key + '\''
        indent_one += '\t'*indent + key_string + ': ' + value_string + ',\n'
    indent -= 1
    indent_one += indent * '\t' + '}'
    return indent_one


def print_code_to_new_file(code, path, tb_name):
    new_file_path = path
    dir = path.replace('test_' + tb_name + '.py', '')
    old_file_path = dir + 'test2_' + tb_name + '.py'

    os.rename(path, old_file_path)

    with open(new_file_path, 'w') as f:
        f.write(code)

def print_case_param():
    para_list = []
    keys = ['tb_name']
    if len(prim_to_value) > 1: # 多原语
        return
    
    for i, tb_name in enumerate(tb_names):
        para_list.append([])
        para_list[i].append(tb_name)
        # para_list[i].append(clocks[i])


    for prim, prim_dict in prim_to_value.items():
        for key, attributes in prim_dict.items():
            if key in ('Vth0', 'Seed', 'Vth_alpha', 'Vth_beta', 'VM_len', 'Vtheta_len',
            'Vth_Incre', 'Vleaky_alpha', 'Vleaky_beta', 'ref_cnt_const'):
                continue
            no_repeat_value = set(attributes)
            if len(no_repeat_value) == 1:
                continue
            keys.append(key)
            for i, value in enumerate(attributes):
                if key in ('VR', 'Vinit', 'dV'):
                    value = 1 if value != 0 else 0
                if key is 'VL':
                    value = 1 if value != -2**27 else 0
                if key in ('Addr_Uin_start', 'Addr_S_Start', 'Addr_V_start',
                           'Addr_VM_start', 'Addr_Vtheta_start', 'Addr_para', 'VM_const'):
                    value = hex(value)
                para_list[i].append(value)

    with open('test_para.py', 'w') as f:
        f.write(str(keys))
        f.write('\n')
        for line in para_list:
            f.write(str(line))
            f.write('\n')
    
def rename_test_file(dir):
    prefix = 'Testcase_'
    files = get_all_files(dir, prefix)
    for file_name in files:
        new_name = file_name.replace(prefix, 'test_')
        os.rename(file_name, new_name)


def convert():
    dir = 'generator/test/MP'
    # format_convert(dir)
    # rename_test_file(dir)
    prefix='test_'
    convert_all_cases(dir, prefix)


convert()

