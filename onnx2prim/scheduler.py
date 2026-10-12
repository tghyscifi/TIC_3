"""Conservative sequence-to-block planning; no source time loop guessing."""
from dataclasses import dataclass
from .ir import ConvLayer, FCLayer, MaxPoolLayer
from .diagnostics import UnsupportedModelError

@dataclass(frozen=True)
class ComputeBlock:
    index: int
    layer_index: int
    layer: object
    has_relu: bool
    pool: object = None
    cut_key: str = ''
    @property
    def output_shape(self):
        return self.pool.output_shape if self.pool is not None else self.layer.output_shape

def plan_blocks(seq):
    result, i, conv_count, fc_count = [], 0, 0, 0
    while i < len(seq.layers):
        layer, relu = seq.layers[i]
        pool = None
        if isinstance(layer, ConvLayer):
            conv_count += 1
            key = f'cut{conv_count}'
            if i+1 < len(seq.layers) and isinstance(seq.layers[i+1][0], MaxPoolLayer):
                pool, pool_relu = seq.layers[i+1]
                relu = relu or pool_relu
        elif isinstance(layer, FCLayer):
            fc_count += 1
            key = f'fc_cut{fc_count}'
        else:
            raise UnsupportedModelError('Standalone pool or unknown layer cannot form an axon/soma block', node=layer.name)
        result.append(ComputeBlock(len(result), i, layer, relu, pool, key))
        i += 2 if pool is not None else 1
    return result

