"""Deterministic miniature ResidualBlocks; not a trained accuracy benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np


def make_fixture(directory, seed=7, height=6, width=8):
    """Save an opset-11 model, CHW integer input and explicit quantization."""
    if isinstance(height, bool) or isinstance(width, bool) or not all(
        isinstance(v, (int, np.integer)) and v > 0 for v in (height, width)
    ):
        raise ValueError('height and width must be positive integers')
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    nodes, initializers, cuts = [], [], {}

    def conv(name, source, cin, cout):
        weight = rng.integers(-4, 5, size=(cout, cin, 3, 3)).astype(np.float32)
        bias = rng.integers(-64, 65, size=cout).astype(np.float32)
        initializers.extend([nh.from_array(weight, name + '.weight'),
                             nh.from_array(bias, name + '.bias')])
        nodes.append(h.make_node('Conv', [source, name + '.weight', name + '.bias'],
                                 [name], name=name, kernel_shape=[3, 3],
                                 pads=[1, 1, 1, 1], strides=[1, 1]))
        cuts[name] = 2
        return name

    def relu(name, source):
        nodes.append(h.make_node('Relu', [source], [name], name=name))
        return name

    current = relu('relu_in', conv('conv_in', 'input', 4, 32))
    for index in range(3):
        prefix = f'res{index}'
        skip = current
        current = relu(prefix + '_relu1', conv(prefix + '_conv1', current, 32, 32))
        current = relu(prefix + '_relu2', conv(prefix + '_conv2', current, 32, 32))
        name = prefix + '_add'
        nodes.append(h.make_node('Add', [skip, current], [name], name=name))
        cuts[name] = 0
        current = name
    current = conv('conv_out', current, 32, 6)
    nodes.append(h.make_node('Tanh', [current], ['tanh'], name='tanh'))
    graph = h.make_graph(nodes, 'prototype_residual',
                         [h.make_tensor_value_info('input', T.FLOAT, [1, 4, height, width])],
                         [h.make_tensor_value_info('tanh', T.FLOAT, [1, 6, height, width])],
                         initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, str(directory / 'residual.onnx'))
    np.save(directory / 'input.npy',
            rng.integers(-128, 128, size=(4, height, width), dtype=np.int64))
    config = {'cuts': cuts, 'parameter_mode': 'integer',
              'tanh_input_scales': {'tanh': 32.0},
              'tanh_output_scales': {'tanh': 128.0}}
    (directory / 'quantization.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return directory


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='A NEW directory; existing paths are rejected')
    parser.add_argument('--height', type=positive_int, default=6)
    parser.add_argument('--width', type=positive_int, default=8)
    args = parser.parse_args()
    print(make_fixture(args.directory, height=args.height, width=args.width).resolve())
