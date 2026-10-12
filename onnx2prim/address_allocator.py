"""Conservative all-live activation allocation with shared INT32 scratch."""
from dataclasses import dataclass
import math
from .diagnostics import CompilerError, UnsupportedModelError
from .ir import ConvLayer, FCLayer
from .layout import align_ceil
from .scheduler import plan_blocks


@dataclass(frozen=True)
class AddrAssign:
    block_index: int
    input_addr: int
    weight_addr: int
    bias_addr: int
    scratch_addr: int
    output_addr: int
    input_words: int
    weight_words: int
    bias_words: int
    scratch_words: int
    output_words: int


def _activation_words(shape):
    if len(shape) == 1:
        count = align_ceil(shape[0], 16)
    elif len(shape) == 3:
        c, h, w = shape
        count = align_ceil(c, 16)*h*w
    else:
        raise UnsupportedModelError('Expected CHW or vector activation')
    if any(type(n) is not int or n <= 0 for n in shape):
        raise CompilerError('Positive static dimensions required')
    return align_ceil(count, 4)//4


def allocate_addresses(seq):
    blocks = plan_blocks(seq)
    if not blocks:
        raise UnsupportedModelError('Empty sequence cannot be allocated')
    first = blocks[0].layer
    if isinstance(first, ConvLayer):
        if tuple(seq.input_shape) != first.input_shape:
            raise CompilerError('First convolution input shape mismatch')
        if any(type(n) is not int or n <= 0 for n in first.input_shape):
            raise CompilerError('Positive static input dimensions required')
        if any((first.pad_top,first.pad_down,first.pad_left,first.pad_right)):
            raise UnsupportedModelError('First p81 padded input is not verified', node=first.name)
        input_words = align_ceil(first.cin*first.py*align_ceil(first.px, 16), 4)//4
    else:
        input_words = _activation_words(seq.input_shape)
    cursor, input_addr, previous = align_ceil(input_words, 4), 0, tuple(seq.input_shape)
    rows = []
    for b in blocks:
        l = b.layer
        if isinstance(l, ConvLayer):
            if any(type(n) is not int or n <= 0 for n in (l.cin,l.cout,l.px,l.py,l.kx,l.ky,l.sx,l.sy)):
                raise CompilerError('Positive static convolution dimensions required', node=l.name)
            if l.input_shape != previous:
                raise CompilerError('Convolution input shape breaks sequence', node=l.name)
            if l.groups != 1 or l.ex != 1 or l.ey != 1:
                raise UnsupportedModelError('Grouped/dilated convolution is unsupported', node=l.name)
            weight_words = align_ceil(align_ceil(l.cout, 32)*l.cin*l.ky*l.kx, 4)//4
            scratch_h = l.py_out
            if b.pool is not None:
                p = b.pool
                if p.input_shape != l.output_shape or (p.kx,p.ky,p.sx,p.sy) != (2,2,2,2) or any((p.pad_top,p.pad_down,p.pad_left,p.pad_right)):
                    raise UnsupportedModelError('Only matching unpadded 2x2 stride2 pooling is supported')
                scratch_h = min(2, l.py_out)
            scratch_words = l.px_out*scratch_h*align_ceil(l.cout, 32)
        elif isinstance(l, FCLayer):
            if l.cin != math.prod(previous):
                raise CompilerError('FC input shape mismatch', node=l.name)
            if len(previous) == 3:
                if l.feature_shape != previous or l.input_layout not in ('CHW','HWC'):
                    raise UnsupportedModelError('Spatial FC requires matching feature_shape and explicit layout')
                if previous[0] % 16:
                    raise UnsupportedModelError('FC flatten with channel padding requires compaction')
            elif l.input_layout != 'VECTOR':
                raise UnsupportedModelError('Vector FC requires VECTOR layout')
            weight_words = align_ceil(align_ceil(l.cout, 32)*l.cin, 4)//4
            scratch_words = align_ceil(l.cout, 32)
        else:
            raise UnsupportedModelError('Unsupported compute layer')
        if min(l.cin,l.cout,weight_words,scratch_words) <= 0:
            raise CompilerError('Invalid compute dimensions')
        output_words = _activation_words(b.output_shape)
        rows.append((b.index,input_addr,cursor,input_words,weight_words,align_ceil(l.cout,32),scratch_words,output_words))
        input_addr, input_words = cursor, output_words
        cursor = align_ceil(cursor+output_words, 4)
        previous = b.output_shape
    scratch_addr = cursor
    scratch_end = scratch_addr + max(r[6] for r in rows)
    if scratch_end > 0x4000:
        raise CompilerError('Activation and scratch exceed Mem0', code='MEMORY')
    spill, mem1, assignments = align_ceil(max(0x2000,scratch_end),4), 0x4000, []
    for index,ina,out,iw,ww,bw,sw,ow in rows:
        start = align_ceil(mem1,4)
        bias = align_ceil(start+ww,4)
        if bias+bw <= 0x8000:
            mem1 = bias+bw
        else:
            start = align_ceil(spill,4)
            bias = align_ceil(start+ww,4)
            if bias+bw > 0x4000:
                raise CompilerError('Whole weight/bias pair cannot fit SRAM banks', code='MEMORY')
            spill = bias+bw
        assignments.append(AddrAssign(index,ina,start,bias,scratch_addr,out,iw,ww,bw,sw,ow))
    weight_end = max(a.bias_addr+a.bias_words for a in assignments)
    return assignments, weight_end, assignments[-1].output_addr


def allocate_addresses_for_stage(seq, stage, input_shape=None):
    expected = tuple(b.index for b in plan_blocks(seq))
    indices = tuple(stage.block_indices) if hasattr(stage, 'block_indices') else tuple(stage)
    if input_shape is not None and tuple(input_shape) != tuple(seq.input_shape):
        raise UnsupportedModelError('Stage input reshaping is unimplemented')
    if indices != expected or (hasattr(stage, 'block_indices') and
            (stage.size_x != 1 or stage.size_y != 1 or stage.core_offset != (0,0))):
        raise UnsupportedModelError('Partial or multicore stages are unimplemented')
    return allocate_addresses(seq)
