"""Logical tensor oracle independent of hardware packing and primitives."""
import numpy as np
from .ir import ConvLayer
from .quantization import integer_parameters, cut
from .scheduler import plan_blocks
from .diagnostics import CompilerError

def checked_int32(a, name):
    if np.any(a < -(2**31)) or np.any(a > 2**31-1):
        raise CompilerError('Accumulator overflow; wrapping/saturation not modeled', node=name, code='OVERFLOW')
    return a

def run_reference(seq, config, input_data):
    config.validate(seq)
    x = np.asarray(input_data, dtype=np.int64)
    outputs = []
    for block in plan_blocks(seq):
        layer = block.layer
        w, bias = integer_parameters(layer, config)
        bound = int(np.max(np.abs(x))) * np.sum(np.abs(w).reshape(layer.cout,-1),axis=1) + np.abs(bias)
        if np.any(bound > 2**31-1):
            raise CompilerError('Conservative accumulator bound exceeds int32',node=layer.name,code='OVERFLOW')
        if isinstance(layer, ConvLayer):
            padded = np.pad(x, ((0,0),(layer.pad_top,layer.pad_down),(layer.pad_left,layer.pad_right)))
            y = np.empty(layer.output_shape, dtype=np.int64)
            for h in range(layer.py_out):
                for v in range(layer.px_out):
                    patch = padded[:,h*layer.sy:h*layer.sy+layer.ky,v*layer.sx:v*layer.sx+layer.kx]
                    y[:,h,v] = np.sum(w*patch[None], axis=(1,2,3)) + bias
        else:
            if layer.input_layout == 'HWC' and x.ndim == 3:
                flat = x.transpose(1,2,0).reshape(-1)
            else:
                flat = x.reshape(-1)
            y = w @ flat + bias
        checked_int32(y, layer.name)
        if block.pool is not None:
            pool = block.pool
            yp = np.pad(y, ((0,0),(pool.pad_top,pool.pad_down),(pool.pad_left,pool.pad_right)), constant_values=-(2**31))
            pooled = np.empty(pool.output_shape, dtype=np.int64)
            for h in range(pool.py):
                for v in range(pool.px):
                    pooled[:,h,v] = np.max(yp[:,h*pool.sy:h*pool.sy+pool.ky,v*pool.sx:v*pool.sx+pool.kx],axis=(1,2))
            y = pooled
        x = cut(y, config[block.cut_key], block.has_relu)
        outputs.append(x.copy())
    return outputs
