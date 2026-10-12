"""Conservative prototype checks. This is not a full X1 hardware verifier."""
import numpy as np
from .diagnostics import CompilerError, UnsupportedModelError
from .ir import ConvLayer, FCLayer
from .scheduler import plan_blocks

def validate_sequence(seq):
    if len(seq.input_shape) != 3 or any(not isinstance(n, int) or n <= 0 for n in seq.input_shape):
        raise CompilerError('Input must be positive static CHW')
    blocks = plan_blocks(seq)
    if not blocks or len(blocks)+1 > 32:
        raise UnsupportedModelError('Expected 1..31 compute blocks')
    if not isinstance(blocks[0].layer,ConvLayer) or blocks[0].layer.cin > 16:
        raise UnsupportedModelError('Prototype entry requires a p81 convolution with cin <= 16')
    previous = seq.input_shape
    names = set()
    for block in blocks:
        l = block.layer
        if l.name in names:
            raise CompilerError('Layer names must be unique', node=l.name)
        names.add(l.name)
        if isinstance(l, ConvLayer):
            dimensions=(l.cin,l.cout,l.px,l.py,l.kx,l.ky,l.sx,l.sy)
            if any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<=0 for n in dimensions):
                raise CompilerError('Convolution dimensions must be positive', node=l.name)
            pads=(l.pad_top,l.pad_down,l.pad_left,l.pad_right)
            if any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<0 for n in pads):
                raise CompilerError('Convolution padding must be nonnegative integers',node=l.name)
            if l.groups != 1 or l.ex != 1 or l.ey != 1:
                raise UnsupportedModelError('Grouped/dilated convolution is not supported', node=l.name)
            if l.input_shape != previous:
                raise CompilerError('Convolution input shape breaks sequence', node=l.name)
            if block.index == 0 and any((l.pad_top,l.pad_down,l.pad_left,l.pad_right)):
                raise UnsupportedModelError('First p81 padded input is not verified', node=l.name)
            expected = (l.cout,l.cin,l.ky,l.kx)
        elif isinstance(l, FCLayer):
            if any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<=0 for n in (l.cin,l.cout)) or l.cin != int(np.prod(previous)):
                raise CompilerError('FC input shape breaks sequence', node=l.name)
            if l.input_layout not in ('VECTOR','CHW','HWC'):
                raise UnsupportedModelError('Unknown flatten layout', node=l.name)
            if len(previous) == 3:
                if l.feature_shape != previous:
                    raise CompilerError('FC needs explicit preflatten feature_shape', node=l.name)
                if block.index > 0 and previous[0] % 16:
                    raise UnsupportedModelError('FC flatten with channel padding requires compaction',node=l.name)
            expected = (l.cout,l.cin)
        else:
            raise UnsupportedModelError('Unknown affine layer', node=l.name)
        if np.asarray(l.weights).shape != expected or (l.bias is not None and np.asarray(l.bias).shape != (l.cout,)):
            raise CompilerError(f'Expected weights {expected} and bias {(l.cout,)}',node=l.name)
        if any(n <= 0 for n in l.output_shape):
            raise CompilerError('Empty layer output', node=l.name)
        if block.pool is not None:
            p = block.pool
            if p.input_shape != l.output_shape:
                raise CompilerError('Pool shape mismatch', node=p.name)
            if (p.kx,p.ky,p.sx,p.sy) != (2,2,2,2) or any((p.pad_top,p.pad_down,p.pad_left,p.pad_right)):
                raise UnsupportedModelError('Only unpadded 2x2 stride2 fused maxpool is verified',node=p.name)
        previous = block.output_shape
    return blocks

def validate_addresses(seq, assignments):
    blocks = validate_sequence(seq)
    if len(assignments) != len(blocks):
        raise CompilerError('Allocation count mismatch',code='ADDRESS')
    from .address_allocator import allocate_addresses
    required,_,_ = allocate_addresses(seq)
    for actual,expected in zip(assignments,required):
        for field in ('input_words','weight_words','bias_words','scratch_words','output_words'):
            if getattr(actual,field) != getattr(expected,field):
                raise CompilerError(f'{field} disagrees with padded physical footprint',code='ADDRESS')
    permanent, activations, scratch = [], [], []
    def region(start, words, role, mem0=False):
        if not isinstance(start, int) or not isinstance(words,int) or words <= 0 or start % 4:
            raise CompilerError(f'{role}: positive word length and four-word-aligned address required',code='ADDRESS')
        end = start+words
        if not (0 <= start < end <= 0x4000 or (not mem0 and 0x4000 <= start < end <= 0x8000)):
            raise CompilerError(f'{role}: range crosses data SRAM bank boundary',code='ADDRESS')
        return (start,end,role)
    for i,a in enumerate(assignments):
        if a.block_index != i or (i and a.input_addr != assignments[i-1].output_addr):
            raise CompilerError('Allocation chain/index mismatch',code='ADDRESS')
        permanent += [region(a.weight_addr,a.weight_words,f'weight{i}'),region(a.bias_addr,a.bias_words,f'bias{i}')]
        if i == 0: activations.append(region(a.input_addr,a.input_words,'input',True))
        activations.append(region(a.output_addr,a.output_words,f'output{i}',True))
        scratch.append(region(a.scratch_addr,a.scratch_words,f'scratch{i}',True))
    def overlap(a,b): return a[0]<b[1] and b[0]<a[1]
    # This allocator keeps all activations live; only scratch/scratch alias is allowed.
    stable = permanent+activations
    for i,a in enumerate(stable):
        for b in stable[i+1:]:
            if overlap(a,b): raise CompilerError(f'Overlap: {a[2]} / {b[2]}',code='ADDRESS')
        for b in scratch:
            if overlap(a,b): raise CompilerError(f'Overlap: {a[2]} / {b[2]}',code='ADDRESS')
    return True
