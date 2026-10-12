"""Single-core compute backend. IO/router and legacy data are unsupported.

Imports of the original (Torch-dependent, NumPy<2) primitives are lazy.
Addresses are already in words; packed arrays may be flat or word rows.
"""
import numpy as np
from ..diagnostics import CompilerError, UnsupportedModelError, DependencyError
from ..ir import ConvLayer
from ..scheduler import plan_blocks


def _align_ceil(value, alignment=32):
    return ((value + alignment - 1) // alignment) * alignment


def _integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise CompilerError(f'{label} must be an integer >= {minimum}')
    return int(value)


def _rows(raw, width, words, label):
    a = np.asarray(raw)
    if a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] != width):
        raise CompilerError(f'{label} must be flat or rows of {width} values')
    if a.size != words * width or not np.issubdtype(a.dtype, np.integer):
        raise CompilerError(f'{label} must contain {words * width} integer values')
    lo, hi = (-128, 127) if width == 4 else (-(2**31), 2**31-1)
    if np.any(a < lo) or np.any(a > hi):
        raise CompilerError(f'{label} values exceed target range')
    return a.reshape(-1, width).astype(np.int64).tolist()


def compile_compute(seq, assignments, cuts, packed_data, phase_en=None, clock_in_phase=200000):
    """Build legacy map_config with phase zero empty and one phase per block.

    phase_en[0] enables unimplemented IO and therefore raises Unsupported.
    Disabled compute phases retain empty slots. No random initialization occurs.
    """
    blocks = plan_blocks(seq)
    if not blocks:
        raise UnsupportedModelError('Empty compute sequence')
    if len(assignments) != len(blocks) or len(packed_data) != len(blocks):
        raise CompilerError('One assignment and packed data dictionary required per block')
    clock = _integer(clock_in_phase, 'clock_in_phase', 1)
    flags = [False] + [True] * len(blocks) if phase_en is None else list(phase_en)
    if len(flags) < len(blocks) + 1 or any(v not in (0, 1) for v in flags):
        raise CompilerError('phase_en must contain binary flags for IO and every block')
    if flags[0] or any(flags[len(blocks)+1:]):
        raise UnsupportedModelError('IO/router phases are not implemented by the compute prototype')
    if set(cuts) != {b.cut_key for b in blocks}:
        raise CompilerError('Cuts must exactly match scheduled block keys')
    prepared = []
    for b, a, data in zip(blocks, assignments, packed_data):
        if _integer(a.block_index, 'block_index') != b.index:
            raise CompilerError('Assignments must follow block order')
        for role in ('input', 'weight', 'bias', 'scratch', 'output'):
            _integer(getattr(a, role + '_addr'), role + '_addr')
            _integer(getattr(a, role + '_words'), role + '_words', 1)
        cut = _integer(cuts[b.cut_key], b.cut_key)
        if cut > 15:
            raise CompilerError('Cut must lie in [0,15]')
        l, pool = b.layer, b.pool
        if isinstance(l, ConvLayer) and l.groups != 1:
            raise UnsupportedModelError('Grouped convolution is unsupported', node=l.name)
        if isinstance(l, ConvLayer) and (l.ex != 1 or l.ey != 1):
            raise UnsupportedModelError('Dilated convolution packing is unverified', node=l.name)
        if pool is not None and pool.input_shape != l.output_shape:
            raise CompilerError('Fused pool input shape must match convolution output', node=pool.name)
        if pool is not None and (pool.kx, pool.ky, pool.sx, pool.sy,
                pool.pad_top, pool.pad_down, pool.pad_left, pool.pad_right) != (2, 2, 2, 2, 0, 0, 0, 0):
            raise UnsupportedModelError('Only unpadded 2x2 stride-2 fused maxpool is verified', node=pool.name)
        rows = {'w': _rows(data['w'], 4, a.weight_words, 'w'),
                'b': _rows(data['b'], 1, a.bias_words, 'b')}
        if b.index == 0:
            rows['x'] = _rows(data['x'], 4, a.input_words, 'x')
        elif data.get('x') is not None:
            raise CompilerError('Only the first block may initialize input')
        prepared.append((b, a, cut, rows))
    try:
        from primitive.Prim_81_Axon_CNN1_new import Prim_81_Axon
        from primitive.Prim_41_Axon_CNN0_new import Prim_41_Axon
        from primitive.Prim_X5_Soma_compare_new import Prim_X5_Soma
    except ImportError as exc:
        raise DependencyError('Compute backend requires original primitives and Torch; use the Tianjic runtime') from exc
    if int(np.__version__.split('.')[0]) >= 2:
        raise DependencyError('Original primitives require NumPy < 2')
    prims = [dict(axon=None, soma1=None, router=None, soma2=None)]
    for b, a, cut, rows in prepared:
        if not flags[b.index+1]:
            prims.append(dict(axon=None, soma1=None, router=None, soma2=None))
            continue
        l = b.layer
        conv = isinstance(l, ConvLayer)
        first81 = b.index == 0 and conv and l.cin <= 16
        axon = Prim_81_Axon() if first81 else Prim_41_Axon()
        axon.InA_type = axon.InB_type = 1
        axon.Load_Bias = 2
        axon.cin, axon.cout = l.cin, l.cout
        for attr, value in [('Input_fm_Px', l.px if conv else 1),
                ('Input_fm_Py', l.py if conv else 1), ('conv_Kx', l.kx if conv else 1),
                ('conv_Ky', l.ky if conv else 1), ('conv_Sx', l.sx if conv else 1),
                ('conv_Sy', l.sy if conv else 1), ('conv_Ex', l.ex if conv else 1),
                ('conv_Ey', l.ey if conv else 1)]:
            setattr(axon, attr, value)
        pads = [getattr(l, key, 0) for key in ('pad_top', 'pad_down', 'pad_left', 'pad_right')]
        axon.pad_on = any(pads)
        for key, value in zip(('pad_up' if first81 else 'pad_top', 'pad_down', 'pad_left', 'pad_right'), pads):
            setattr(axon, key, value)
        axon.Addr_InA_base, axon.Addr_InB_base = a.input_addr, a.weight_addr
        axon.Addr_Bias_base, axon.Addr_V_base = a.bias_addr, a.scratch_addr
        axon.memory_blocks = [dict(name=f'{axon.PIC:02X}_{key}', start=addr, data=rows[key], mode=0)
            for key, addr in [('x', a.input_addr), ('w', a.weight_addr), ('b', a.bias_addr)] if key in rows]
        soma = Prim_X5_Soma()
        soma.PIC_Mode = 0
        soma.type_in, soma.type_out = 0, 1
        soma.cin, soma.cout = _align_ceil(l.cout, 32), l.cout
        soma.Input_fm_Px, soma.Input_fm_Py = (l.px_out, l.py_out) if conv else (1, 1)
        soma.pooling_Kx = soma.pooling_Ky = 2 if b.pool else 1
        soma.pooling_Sx = soma.pooling_Sy = 2 if b.pool else 1
        soma.CMP_C = 0 if b.has_relu else 0x80808080
        soma.in_cut_start = cut
        soma.Row_ck_on, soma.in_row_max = (1, 2) if b.pool else ((0,0) if conv else (1,1))
        soma.reset_Addr_in = soma.reset_Addr_out = 1
        soma.Addr_Start_in, soma.Addr_Start_out = a.scratch_addr, a.output_addr
        # Same preparation performed by PIGroup.add_PI in the original mapper.
        axon.cal_para()
        soma.cal_para()
        prims.append(dict(axon=axon, soma1=soma, router=None, soma2=None))
    return {'sim_clock': None, ((0, 0), 0): {0: {'clock': clock, 'mode': 1,
            ((0, 0), (0, 0)): {'prims': prims}}}}
