"""Conservative single-core stateful SRAM plan; state is never an aliasable tensor."""
from dataclasses import dataclass, asdict
import math
from .ir import ConvLayer
from .layout import align_ceil
from .diagnostics import CompilerError

def physical_shape(shape):
    return (align_ceil(shape[0], 32),) + tuple(shape[1:])

def neuron_words(shape):
    return math.prod(physical_shape(shape))

@dataclass
class SNNPlan:
    input_addr: int
    input_words: int
    outputs: dict
    states: dict
    parameters: dict
    scratch_addr: int
    scratch_words: int
    phases: int
    regions: list

def _allocate(model):
    model.validate()
    cursor = 0; regions = []
    def mem0(words, label):
        nonlocal cursor
        start = align_ceil(cursor, 4); cursor = start + words
        if cursor > 0x4000:
            raise CompilerError('SNN state/activations exceed Mem0; partitioning required', code='MEMORY')
        regions.append((start, cursor, label)); return start
    input_words = neuron_words(model.input_shape)//4
    input_addr = mem0(input_words, 'input')
    outputs = {}; states = {}; parameters = {}
    for b in model.blocks:
        shape = tuple(b.affine.output_shape); n = neuron_words(shape)
        outputs[b.output] = {'address':mem0(n//4, b.output), 'words':n//4, 'shape':shape}
        states[b.name] = {'membrane_addr':mem0(n, b.name+'.membrane'), 'membrane_words':n,
                          'vm_addr':mem0(4, b.name+'.VM'), 'vm_words':4,
                          'vtheta_addr':mem0(4, b.name+'.Vtheta'), 'vtheta_words':4,
                          'para_addr':mem0(4, b.name+'.para'), 'para_words':4}
    scratch_words = max(neuron_words(b.affine.output_shape) for b in model.blocks)
    scratch_addr = mem0(scratch_words, 'current.scratch')
    mem1 = 0x4000
    shape = model.input_shape
    for b in model.blocks:
        l = b.affine
        cin = (physical_shape(shape)[0] if isinstance(l, ConvLayer) else neuron_words(shape))
        kernel = l.kx*l.ky if isinstance(l, ConvLayer) else 1
        bw = align_ceil(l.cout, 32); ww = bw*cin*kernel//4
        start = align_ceil(mem1, 4)
        if start+ww+bw <= 0x8000:
            mem1 = start+ww+bw
            regions.append((start, mem1, b.name+'.parameters'))
        else:
            start = mem0(ww+bw, b.name+'.parameters')
        parameters[b.name] = {'weight_addr':start, 'weight_words':ww,
                              'bias_addr':start+ww, 'bias_words':bw, 'physical_cin':cin}
        shape = l.output_shape
    return SNNPlan(input_addr,input_words,outputs,states,parameters,scratch_addr,
                   scratch_words,len(model.blocks)+1,regions)

def validate_snn_plan(model, plan):
    """Recompute sizes/coverage/regions; no caller-provided metadata is trusted."""
    expected = _allocate(model)
    if asdict(plan) != asdict(expected):
        raise CompilerError('SNN plan differs from independently recomputed state/parameter allocation', code='ADDRESS')
    ordered = sorted(plan.regions)
    previous_end = 0
    for start,end,name in ordered:
        if type(start) is not int or start%4 or end <= start or start < previous_end:
            raise CompilerError('SNN unaligned/overlapping SRAM region '+name, code='ADDRESS')
        if not (0 <= start < end <= 0x4000 or 0x4000 <= start < end <= 0x8000):
            raise CompilerError('SNN region crosses SRAM bank '+name, code='ADDRESS')
        previous_end = end
    return plan

def plan_snn(model):
    return validate_snn_plan(model, _allocate(model))
