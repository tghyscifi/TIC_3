"""Integer tensor layouts. Returned arrays contain elements, not packed words."""
import math
import operator
import numpy as np
from .diagnostics import CompilerError, UnsupportedModelError
from .ir import ConvLayer, FCLayer


def align_ceil(value, alignment=32):
    value, alignment = operator.index(value), operator.index(alignment)
    if value < 0 or alignment <= 0:
        raise ValueError('Nonnegative value and positive alignment required')
    return (value + alignment - 1) // alignment * alignment


def _integer(data, dtype):
    a = np.asarray(data)
    dt = np.dtype(dtype)
    if a.dtype.kind not in 'iu' or dt.kind not in 'iu':
        raise CompilerError('Packing requires already integer arrays')
    limits = np.iinfo(dt)
    if a.size and (np.any(a < limits.min) or np.any(a > limits.max)):
        raise CompilerError('Integer tensor exceeds target dtype range')
    return a.astype(dt, copy=False)


def pack_tensor(data, dims=None, alignment=16, is_weight=False, dtype=np.int8):
    """Pad axis zero; weights use separate output-channel groups, ravel F."""
    a = _integer(data, dtype)
    if dims is not None:
        dims = tuple(operator.index(n) for n in dims)
        if any(n <= 0 for n in dims) or math.prod(dims) != a.size:
            raise CompilerError('Tensor dimensions do not match data')
        a = a.reshape(dims)
    if a.ndim == 0 or any(n <= 0 for n in a.shape):
        raise CompilerError('Nonempty tensor required')
    padded = np.zeros((align_ceil(a.shape[0], alignment),) + a.shape[1:], dtype=a.dtype)
    padded[:a.shape[0]] = a
    if is_weight:
        return np.concatenate([padded[i:i+alignment].ravel(order='F')
                               for i in range(0, padded.shape[0], alignment)])
    return padded.ravel(order='F')


def pack_activation(data, dims=None, *, first_input=False, dtype=np.int8):
    """CHW source becomes C-fast HWC; p81 input instead pads W to 16."""
    a = _integer(data, dtype)
    if dims is not None:
        a = a.reshape(tuple(dims))
    if a.ndim == 1:
        if first_input:
            raise CompilerError('p81 first input requires CHW')
        return pack_tensor(a, alignment=16, dtype=dtype)
    if a.ndim != 3 or any(n <= 0 for n in a.shape):
        raise CompilerError('Activation must be a positive CHW tensor or vector')
    if first_input:
        c, h, w = a.shape
        padded = np.zeros((c, h, align_ceil(w, 16)), dtype=a.dtype)
        padded[:, :, :w] = a
        return padded.ravel()
    return pack_tensor(a.transpose(0, 2, 1), alignment=16, dtype=dtype)


def pack_weights(layer, *, first_conv=False, dtype=np.int8):
    """Conv source OIHW; FC source OI with explicit flatten provenance."""
    a = _integer(layer.weights, dtype)
    if isinstance(layer, ConvLayer):
        if layer.groups != 1 or layer.ex != 1 or layer.ey != 1:
            raise UnsupportedModelError('Grouped/dilated convolution packing is unsupported')
        if a.shape != (layer.cout, layer.cin, layer.ky, layer.kx):
            raise CompilerError('Convolution weight shape mismatch')
        a = a.transpose(0, 3, 2, 1) if first_conv else a.transpose(0, 1, 3, 2)
    elif isinstance(layer, FCLayer):
        if a.shape != (layer.cout, layer.cin):
            raise CompilerError('FC weight shape mismatch')
        if layer.input_layout not in ('VECTOR', 'CHW', 'HWC'):
            raise UnsupportedModelError('Unknown FC input layout')
        if layer.feature_shape is not None:
            c, h, w = layer.feature_shape
            if min(c, h, w) <= 0 or c*h*w != layer.cin:
                raise CompilerError('FC feature shape mismatch')
            if c % 16:
                raise UnsupportedModelError('Padded FC flatten requires compaction')
            if layer.input_layout == 'CHW':
                a = a.reshape(layer.cout, c, h, w).transpose(0, 2, 3, 1).reshape(layer.cout, layer.cin)
            elif layer.input_layout != 'HWC':
                raise UnsupportedModelError('Spatial FC requires explicit CHW/HWC layout')
        elif layer.input_layout != 'VECTOR':
            raise CompilerError('Spatial FC needs feature_shape')
    else:
        raise UnsupportedModelError('Unknown weight layer')
    return pack_tensor(a, alignment=32, is_weight=True, dtype=dtype)


def pack_first_input(data):
    return pack_activation(data, first_input=True)


def pack_bias(bias, cout, dtype=np.int32):
    if cout <= 0:
        raise CompilerError('Positive cout required')
    a = np.zeros(cout, dtype=dtype) if bias is None else _integer(bias, dtype)
    if a.shape != (cout,):
        raise CompilerError('Bias shape mismatch')
    return pack_tensor(a, alignment=32, dtype=dtype)
