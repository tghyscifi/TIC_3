"""Typed compute compilation is primary; unverified routing fails closed."""
from .runtime import compile_compute, _align_ceil
from ..diagnostics import UnsupportedModelError


def build_cut_config(seq):
    from ..quantization import build_cut_config as build
    return build(seq)


def generate_g1_1core(model_name, seq, assignments, cut_config, final_out_addr=None):
    """Emit a compute-only adapter with the old public function signature.

    Data keys are IR layer names with input/weight/bias dictionaries keyed (0,0).
    Routing requests fail explicitly. Source names are quoted data, never code.
    """
    from dataclasses import fields, asdict
    from ..validate import validate_addresses
    from ..scheduler import plan_blocks
    validate_addresses(seq,assignments)
    if final_out_addr is not None and final_out_addr != assignments[-1].output_addr:
        from ..diagnostics import CompilerError
        raise CompilerError('Final address disagrees with allocation',code='ADDRESS')
    layers = []
    for layer,relu in seq.layers:
        args = ', '.join(f'{f.name}={getattr(layer,f.name)!r}' for f in fields(layer) if f.name not in ('weights','bias'))
        layers.append(f'({type(layer).__name__}({args}), {relu!r})')
    sequence = f'LayerSeq({tuple(seq.input_shape)!r}, [{", ".join(layers)}], model_name={str(model_name)!r})'
    allocation = '['+', '.join(f'AddrAssign(**{asdict(a)!r})' for a in assignments)+']'
    from ..quantization import QuantizationConfig
    default_cuts = {b.cut_key:cut_config[b.cut_key] for b in plan_blocks(seq)}
    QuantizationConfig(default_cuts).validate(seq)
    default_cuts = {key:int(value) for key,value in default_cuts.items()}
    return f'''# Generated compute adapter; no IO/router or multi-core support.
from onnx2prim.ir import ConvLayer, FCLayer, MaxPoolLayer, LayerSeq
from onnx2prim.address_allocator import AddrAssign
from onnx2prim.codegen.runtime import compile_compute
from onnx2prim.scheduler import plan_blocks
from onnx2prim.diagnostics import CompilerError, UnsupportedModelError

def gen_1_map_config(phase_en, clock_in_phase, size_x, size_y, in_cut_start_dict=None,
                     data=None, in_data_en=False, out_data_en=False, chip=(0,0), init_data=False):
    if (size_x,size_y) != (1,1) or chip != (0,0) or in_data_en or out_data_en:
        raise UnsupportedModelError('Generated adapter is single-core compute-only')
    seq = {sequence}
    assignments = {allocation}
    cuts = {default_cuts!r} if in_cut_start_dict is None else in_cut_start_dict
    packed = []
    try:
        for block in plan_blocks(seq):
            d = data[block.layer.name]
            packed.append(dict(x=d['input'][(0,0)] if block.index == 0 else None,
                               w=d['weight'][(0,0)], b=d['bias'][(0,0)]))
    except (KeyError,TypeError) as exc:
        raise CompilerError('Expected packed data keyed by IR layer names') from exc
    flags = list(phase_en)
    if flags: flags[0] = 0
    return compile_compute(seq,assignments,cuts,packed,flags,clock_in_phase)
'''


def _unsupported(*args, **kwargs):
    raise UnsupportedModelError('IO/router or top-level source generation is not verified; use compile_compute')


generate_g0 = _unsupported
generate_g2 = _unsupported
generate_gi_compute = _unsupported
generate_g_out = _unsupported
generate_top_py = _unsupported

__all__ = ['compile_compute', 'build_cut_config', '_align_ceil', 'generate_g1_1core',
           'generate_g0', 'generate_g2', 'generate_gi_compute', 'generate_g_out', 'generate_top_py']
