"""Explicit integer compilation contract, not an automatic PTQ algorithm."""
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
from .diagnostics import CompilerError
from .scheduler import plan_blocks

@dataclass(frozen=True)
class QuantizationConfig(Mapping):
    cuts: dict
    parameter_mode: str = 'integer'
    weight_scale: float = 128.0
    bias_scale: float = 16384.0
    def __getitem__(self, key): return self.cuts[key]
    def __iter__(self): return iter(self.cuts)
    def __len__(self): return len(self.cuts)

    def validate(self, seq):
        expected = {b.cut_key for b in plan_blocks(seq)}
        if set(self.cuts) != expected:
            raise CompilerError(f'Cut keys must be exactly {sorted(expected)}', code='QUANTIZATION')
        for key, value in self.cuts.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 15:
                raise CompilerError(f'{key} must be an integer in [0,15]', code='QUANTIZATION')
        if self.parameter_mode not in ('integer', 'scaled'):
            raise CompilerError('parameter_mode must be integer or scaled', code='QUANTIZATION')
        if not all(np.isfinite(v) and v > 0 for v in (self.weight_scale, self.bias_scale)):
            raise CompilerError('Parameter scales must be finite positive numbers', code='QUANTIZATION')
        return self

def build_cut_config(seq):
    """Compatibility API requires explicit metadata; never guess cuts from shape."""
    if 'cut_config' not in seq.metadata:
        raise CompilerError('Provide an explicit cut_config; automatic calibration is not implemented', code='QUANTIZATION')
    return QuantizationConfig(dict(seq.metadata['cut_config'])).validate(seq)

def integer_parameters(layer, config):
    result = []
    for name, raw, lo, hi, scale in (
        ('weights', layer.weights, -128, 127, config.weight_scale),
        ('bias', layer.bias if layer.bias is not None else np.zeros(layer.cout),
         -(2**31), 2**31-1, config.bias_scale)):
        a = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise CompilerError(f'{name} must be finite', node=layer.name, code='QUANTIZATION')
        if config.parameter_mode == 'scaled':
            a = np.rint(a * scale)
        if np.any(a != np.rint(a)) or np.any(a < lo) or np.any(a > hi):
            raise CompilerError(f'{name} is not representable as target integer; no silent clipping', node=layer.name, code='QUANTIZATION')
        result.append(a.astype(np.int64))
    return tuple(result)

def cut(values, amount, relu=False):
    a = np.floor_divide(np.asarray(values, dtype=np.int64), 2**(2*int(amount)))
    return np.clip(a, 0 if relu else -128, 127).astype(np.int64)
