"""Logical CHW integer graph oracle, independent of target packing and LUTs."""
from dataclasses import dataclass, field
import numpy as np
from .diagnostics import CompilerError
from .quantization import QuantizationConfig, integer_parameters, cut


def _fail(message, node=None):
    raise CompilerError(message, node=node, code='QUANTIZATION')


def _scales(mapping, expected, label):
    if set(mapping) != expected:
        _fail(f'{label} keys must be exactly {sorted(expected)}')
    for key, value in mapping.items():
        try:
            valid = not isinstance(value, (bool, np.bool_)) and np.ndim(value) == 0 and np.isfinite(value) and value > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            _fail(f'{label}[{key}] must be finite and positive')


@dataclass(frozen=True)
class GraphQuantizationConfig:
    cuts: dict
    parameter_mode: str = 'integer'
    weight_scales: dict = field(default_factory=dict)
    bias_scales: dict = field(default_factory=dict)
    tanh_input_scales: dict = field(default_factory=dict)
    tanh_output_scales: dict = field(default_factory=dict)
    activation_scales: dict = None

    def validate(self, graph):
        nodes = graph.topological_nodes()
        expected = {n.name for n in nodes if n.op in ('Conv', 'Add')}
        if set(self.cuts) != expected:
            _fail(f'Cut keys must be exactly {sorted(expected)}')
        for key, amount in self.cuts.items():
            if isinstance(amount, (bool, np.bool_)) or not isinstance(amount, (int, np.integer)) or not 0 <= amount <= 15:
                _fail(f'{key} cut must be an integer in [0,15]')
        if self.parameter_mode not in ('integer', 'scaled'):
            _fail('parameter_mode must be integer or scaled')
        conv = {n.name for n in nodes if n.op == 'Conv'}
        for label in ('weight_scales', 'bias_scales'):
            _scales(getattr(self, label), conv if self.parameter_mode == 'scaled' else set(), label)
        tanh = {n.name for n in nodes if n.op == 'Tanh'}
        _scales(self.tanh_input_scales, tanh, 'tanh_input_scales')
        _scales(self.tanh_output_scales, tanh, 'tanh_output_scales')
        if self.activation_scales is None:
            if self.parameter_mode == 'scaled':
                _fail('scaled mode requires explicit activation_scales')
            scales = {graph.input_name: 1, **{n.output: 1 for n in nodes}}
        else:
            scales = self.activation_scales
            _scales(scales, {graph.input_name} | {n.output for n in nodes}, 'activation_scales')
        for node in nodes:
            if node.op == 'Add' and (len(node.inputs) != 2 or scales[node.inputs[0]] != scales[node.inputs[1]]):
                _fail('Add inputs must have identical activation scales; automatic rescale is unsupported', node.name)
        return self

    def parameter_config(self, node):
        return QuantizationConfig({'cut1': self.cuts[node.name]}, self.parameter_mode,
                                  self.weight_scales[node.name] if self.parameter_mode == 'scaled' else 1,
                                  self.bias_scales[node.name] if self.parameter_mode == 'scaled' else 1)


def make_tanh_table(config, node):
    """Unsigned indices: 0..127 map to themselves, 128..255 to -128..-1."""
    _scales({node.name: config.tanh_input_scales[node.name]}, {node.name}, 'tanh_input_scales')
    _scales({node.name: config.tanh_output_scales[node.name]}, {node.name}, 'tanh_output_scales')
    signed = np.arange(256, dtype=np.int64)
    signed[128:] -= 256
    return np.floor(np.clip(np.tanh(signed / config.tanh_input_scales[node.name]) *
                            config.tanh_output_scales[node.name], -128, 127)).astype(np.int8)


def _int32(value, name):
    if np.any(value < -(2**31)) or np.any(value > 2**31 - 1):
        raise CompilerError('Accumulator overflow; wrapping is unsupported', node=name, code='OVERFLOW')
    return value


def run_graph_reference(graph, config, input_data):
    config.validate(graph)
    raw = np.asarray(input_data)
    try:
        valid = raw.shape == tuple(graph.input_shape) and np.all(np.isfinite(raw)) and np.all(raw == np.floor(raw)) and np.all((raw >= -128) & (raw <= 127))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        _fail('Input must be finite signed int8 integers with the declared CHW shape')
    values = {graph.input_name: raw.astype(np.int64)}
    for node in graph.topological_nodes():
        args = [values[key] for key in node.inputs]
        if len(args) != (2 if node.op == 'Add' else 1):
            _fail('Invalid operator input count', node.name)
        x = args[0]
        if node.op == 'Conv':
            layer = node.layer
            if layer is None or layer.groups != 1 or x.shape != layer.input_shape:
                _fail('Conv requires a compatible ungrouped ConvLayer', node.name)
            w, bias = integer_parameters(layer, config.parameter_config(node))
            if w.shape != (layer.cout, layer.cin, layer.ky, layer.kx) or bias.shape != (layer.cout,):
                _fail('Invalid Conv parameter shape', node.name)
            bound = np.abs(bias) + int(np.max(np.abs(x))) * np.sum(np.abs(w), axis=(1, 2, 3))
            if np.any(bound > 2**31 - 1):
                raise CompilerError('Conservative accumulator bound exceeds int32', node=node.name, code='OVERFLOW')
            padded = np.pad(x, ((0, 0), (layer.pad_top, layer.pad_down), (layer.pad_left, layer.pad_right)))
            y = np.empty(layer.output_shape, dtype=np.int64)
            for h, v in np.ndindex(y.shape[1:]):
                patch = padded[:, h*layer.sy:h*layer.sy+(layer.ky-1)*layer.ey+1:layer.ey,
                               v*layer.sx:v*layer.sx+(layer.kx-1)*layer.ex+1:layer.ex]
                y[:, h, v] = np.sum(w * patch[None], axis=(1, 2, 3)) + bias
            _int32(y, node.name)
            y = cut(y, config.cuts[node.name])
        elif node.op == 'Add':
            if x.shape != args[1].shape:
                _fail('Add requires equal shapes; broadcasting is unsupported', node.name)
            y = cut(_int32(x + args[1], node.name), config.cuts[node.name])
        elif node.op == 'Relu':
            y = np.clip(x, 0, 127)
        elif node.op == 'MaxPool':
            a = node.attrs
            padded = np.pad(x, ((0, 0), (a.get('pad_top', 0), a.get('pad_down', 0)),
                                (a.get('pad_left', 0), a.get('pad_right', 0))), constant_values=-128)
            y = np.empty(node.shape, dtype=np.int64)
            for c, h, v in np.ndindex(y.shape):
                y[c, h, v] = np.max(padded[c, h*a.get('sy', 1):h*a.get('sy', 1)+a['ky'],
                                             v*a.get('sx', 1):v*a.get('sx', 1)+a['kx']])
        elif node.op == 'Tanh':
            y = np.floor(np.clip(np.tanh(x / config.tanh_input_scales[node.name]) *
                                 config.tanh_output_scales[node.name], -128, 127)).astype(np.int64)
        else:
            raise CompilerError(f'Unsupported graph operator {node.op}', node=node.name, code='UNSUPPORTED')
        if y.shape != tuple(node.shape):
            _fail('Output shape differs from declared graph shape', node.name)
        values[node.output] = y
    return values
