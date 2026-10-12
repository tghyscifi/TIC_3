"""Explicit deterministic p08 contract, not an implicit snntorch conversion.

Each block is an int8 affine operation producing int32 current followed by
LIF. The LIF cut affects readout only, never the incoming current. State is
initialized at a sample boundary and persists between externally fed frames.
"""
from dataclasses import dataclass, field, fields
import math
import numpy as np
from .ir import ConvLayer, FCLayer, StateSpec
from .diagnostics import CompilerError, UnsupportedModelError

@dataclass(frozen=True)
class LIFSpec:
    threshold: int = 64
    leak_enabled: bool = True
    leak_alpha: int = 128
    leak_beta: int = 0
    reset_mode: int = 0
    reset_value: int = 0
    reset_decrement: int = 0
    floor: int = -(1 << 27)
    initial_v: int = 0
    refractory: int = 0
    window_enabled: bool = False
    window_length: int = 1
    window_initial: int = 0
    fire_type: int = 4
    readout_cut: int = 0

    def validate(self):
        for f in fields(self):
            value = getattr(self, f.name)
            expected = bool if f.name in ('leak_enabled', 'window_enabled') else int
            if type(value) is not expected:
                raise CompilerError(f'LIF {f.name} must be {expected.__name__}', code='SNN_CONTRACT')
        for name in ('threshold', 'leak_beta', 'reset_value', 'reset_decrement', 'floor', 'initial_v'):
            if not -(1 << 27) <= getattr(self, name) < (1 << 27):
                raise CompilerError(f'LIF {name} exceeds signed 28-bit range', code='SNN_CONTRACT')
        if self.reset_mode not in (0, 1, 2) or self.fire_type not in (2, 3, 4):
            raise UnsupportedModelError('Only reset modes 0/1/2 and int8 fire types 2/3/4 are supported')
        if not 0 <= self.leak_alpha <= 255 or not 0 <= self.refractory <= 15:
            raise CompilerError('LIF leak alpha must be 0..255 and refractory 0..15', code='SNN_CONTRACT')
        # Legacy p08 packs Tw_cnt into only eight bits in Addr_para.
        # Refuse longer windows rather than export a silently truncated state.
        if not 0 <= self.window_initial <= self.window_length <= 255 or not 0 <= self.readout_cut <= 15:
            raise CompilerError('Invalid LIF time window or readout cut', code='SNN_CONTRACT')
        if self.initial_v < self.floor:
            raise CompilerError('Initial membrane must not be below LIF floor', code='SNN_CONTRACT')

@dataclass
class SNNBlock:
    name: str
    affine: object
    lif: LIFSpec
    output: str

@dataclass
class SNNModel:
    input_shape: tuple
    blocks: list
    dt: float = 1.0
    input_encoding: str = 'int8-current'
    model_name: str = 'snn'
    source_path: str = ''
    metadata: dict = field(default_factory=dict)

    @property
    def states(self):
        result = []
        for block in self.blocks:
            shape = block.affine.output_shape
            result.extend((StateSpec(block.name+'.membrane', shape, 'int28', block.lif.initial_v, 'sample'),
                           StateSpec(block.name+'.refractory', shape, 'uint4', 0, 'sample'),
                           StateSpec(block.name+'.window', (), 'uint8', block.lif.window_initial, 'sample')))
        return result

    def validate(self):
        if (not isinstance(self.dt, (int, float)) or isinstance(self.dt, bool)
                or not math.isfinite(self.dt) or self.dt <= 0):
            raise CompilerError('SNN dt must be positive finite; it is descriptive, not a chip clock', code='SNN_CONTRACT')
        if self.input_encoding not in ('int8-current', 'binary-spikes'):
            raise UnsupportedModelError('Unsupported SNN input encoding')
        if not self.blocks or len(self.blocks) > 31:
            raise UnsupportedModelError('SNN needs 1..31 affine/LIF blocks for single-core static PI')
        shape = tuple(self.input_shape)
        if len(shape) not in (1, 3) or any(type(x) is not int or x <= 0 for x in shape):
            raise CompilerError('SNN input is a positive vector or CHW shape', code='SNN_CONTRACT')
        names = set(); outputs = set(); affine_names = set()
        for b in self.blocks:
            if (not isinstance(b.name,str) or not b.name or b.name in names
                    or not isinstance(b.output,str) or not b.output or b.output in outputs
                    or not isinstance(getattr(b.affine,'name',None),str)
                    or not b.affine.name or b.affine.name in affine_names):
                raise CompilerError('SNN block, affine and output names must be unique', code='SNN_CONTRACT')
            names.add(b.name); outputs.add(b.output); affine_names.add(b.affine.name)
            b.lif.validate(); l = b.affine
            if not isinstance(l,(ConvLayer,FCLayer)):
                raise UnsupportedModelError('Only Conv/FC followed by LIF supported',node=b.name)
            if any(type(getattr(l,key)) is not int or getattr(l,key) <= 0 for key in ('cin','cout')):
                raise CompilerError('SNN affine channels must be positive integers',node=b.name)
            if isinstance(l, ConvLayer):
                if tuple(l.input_shape) != shape or l.groups != 1 or l.ex != 1 or l.ey != 1:
                    raise UnsupportedModelError('SNN convolution shape/groups/dilation unsupported', node=b.name)
                if (any(type(getattr(l,key)) is not int or getattr(l,key) <= 0 for key in ('kx','ky','sx','sy','px','py'))
                        or any(type(getattr(l,key)) is not int or getattr(l,key) < 0 for key in ('pad_top','pad_down','pad_left','pad_right'))):
                    raise CompilerError('Invalid SNN convolution geometry', node=b.name)
            elif isinstance(l, FCLayer):
                if len(shape) == 3:
                    if l.feature_shape != shape or l.input_layout != 'CHW' or l.cin != math.prod(shape):
                        raise UnsupportedModelError('Spatial SNN FC requires explicit CHW Flatten provenance', node=b.name)
                elif l.cin != shape[0] or l.input_layout != 'VECTOR' or l.feature_shape is not None:
                    raise CompilerError('SNN FC input shape/layout mismatch', node=b.name)
            else:
                raise UnsupportedModelError('Only Conv/FC followed by LIF supported', node=b.name)
            if any(x <= 0 for x in l.output_shape):
                raise CompilerError('Empty SNN affine output', node=b.name)
            shape = tuple(l.output_shape)
        trace_names = {b.name+suffix for b in self.blocks for suffix in ('.membrane','.refractory','.window')}
        if outputs & trace_names:
            raise CompilerError('SNN output names collide with reserved state trace names',code='SNN_CONTRACT')

def validate_input(model, frames):
    model.validate()
    x = np.asarray(frames)
    if (x.ndim != len(model.input_shape)+1 or tuple(x.shape[1:]) != tuple(model.input_shape)
            or x.shape[0] <= 0 or x.dtype.kind not in 'iu' or np.any(x < -128) or np.any(x > 127)):
        raise CompilerError('SNN input must be T x input_shape integers in int8 range', code='INPUT')
    if model.input_encoding == 'binary-spikes' and np.any((x != 0) & (x != 1)):
        raise CompilerError('binary-spikes input must contain only 0 and 1', code='INPUT')
    return x.astype(np.int64, copy=True)
