"""Fail-closed static NCHW ANN DAG parser, independent of sequence parsing."""
import numpy as np

from .diagnostics import DependencyError, UnsupportedModelError
from .ir import ConvLayer, GraphIR, GraphNode


def parse_onnx_graph(path):
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise DependencyError('parse_onnx_graph requires ONNX') from exc

    def fail(message, node=None):
        raise UnsupportedModelError(message, node=node)

    try:
        model = onnx.load(str(path))
    except Exception as exc:
        fail('Invalid ONNX: ' + str(exc))
    if model.functions or any(o.domain not in ('', 'ai.onnx') for o in model.opset_import):
        fail('Custom domains/functions are unsupported')
    if any(n.domain not in ('', 'ai.onnx') for n in model.graph.node):
        fail('Custom node domain')
    # Inspect original declarations before inference can resolve symbolic shapes.
    def declared_shapes(graph):
        result = {}
        for value in list(graph.input) + list(graph.value_info) + list(graph.output):
            tensor = value.type.tensor_type
            if not tensor.HasField('shape') or any(
                    d.dim_param or not d.HasField('dim_value') or d.dim_value <= 0
                    for d in tensor.shape.dim):
                fail('Dynamic or nonpositive tensor shape: ' + value.name)
            result[value.name] = tuple(d.dim_value for d in tensor.shape.dim)
        return result

    declared_shapes(model.graph)
    try:
        onnx.checker.check_model(model)
        graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception as exc:
        fail('Invalid ONNX: ' + str(exc))
    shapes = declared_shapes(graph)
    constants = {t.name: numpy_helper.to_array(t).copy() for t in graph.initializer}
    inputs = [v for v in graph.input if v.name not in constants]
    if len(inputs) != 1 or len(graph.output) != 1:
        fail('Exactly one activation input and output required')
    input_name, output_name = inputs[0].name, graph.output[0].name
    input_shape = shapes[input_name]
    if len(input_shape) != 4 or input_shape[0] != 1:
        fail('Input must be static batch-one NCHW')
    if inputs[0].type.tensor_type.elem_type not in (1, 10, 11):
        fail('Floating activation input required')
    allowed = {
        'Conv': {'kernel_shape', 'strides', 'pads', 'dilations', 'group', 'auto_pad'},
        'MaxPool': {'kernel_shape', 'strides', 'pads', 'dilations', 'ceil_mode',
                    'storage_order', 'auto_pad'},
        'Add': set(), 'Relu': set(), 'Tanh': set(), 'Constant': {'value'},
    }
    values = {input_name: input_shape[1:]}
    nodes, names, parameter_uses, constant_nodes = [], set(), set(), {}

    def parameter(key, name):
        if key not in constants:
            fail('Conv parameter must be constant: ' + key, name)
        value = constants[key]
        if value.dtype.kind != 'f' or not np.isfinite(value).all():
            fail('Finite floating parameter required: ' + key, name)
        parameter_uses.add(key)
        return value.copy()

    for raw in graph.node:
        name = raw.name or (raw.output[0] if raw.output else '')
        if not name or name in names:
            fail('Graph node names must be unique', name)
        names.add(name)
        op = raw.op_type
        if op not in allowed or any(a.name not in allowed[op] for a in raw.attribute):
            fail('Unsupported operator or attribute: ' + op, name)
        if len(raw.output) != 1 or not raw.output[0]:
            fail('Exactly one node output required', name)
        output = raw.output[0]
        if output in values or output in constants:
            fail('Graph values must be unique', name)
        attrs = {a.name: helper.get_attribute_value(a) for a in raw.attribute}
        if op == 'Constant':
            if raw.input or set(attrs) != {'value'} or not isinstance(attrs['value'], onnx.TensorProto):
                fail('Only tensor parameter Constant supported', name)
            constants[output] = numpy_helper.to_array(attrs['value']).copy()
            constant_nodes[output] = name
            continue
        activation_inputs = tuple(raw.input[:1] if op == 'Conv' else raw.input)
        if not activation_inputs or any(v not in values for v in activation_inputs):
            fail('Activation input must be a preceding activation', name)
        shape = values[activation_inputs[0]]
        layer, normalized = None, {}
        if op in ('Conv', 'MaxPool'):
            if attrs.get('auto_pad', b'NOTSET') != b'NOTSET' or attrs.get('group', 1) != 1 or list(attrs.get('dilations', [1, 1])) != [1, 1]:
                fail('Only explicit pads, group one and dilation one supported', name)
            strides, pads = list(attrs.get('strides', [1, 1])), list(attrs.get('pads', [0, 0, 0, 0]))
            if len(strides) != 2 or min(strides) <= 0 or len(pads) != 4 or min(pads) < 0:
                fail('Invalid spatial attributes', name)
            if op == 'Conv':
                if len(raw.input) not in (2, 3):
                    fail('Invalid Conv inputs', name)
                weights = parameter(raw.input[1], name)
                if weights.ndim != 4 or weights.shape[1] != shape[0] or min(weights.shape) <= 0:
                    fail('Conv weight dimensions', name)
                kernel = list(attrs.get('kernel_shape', weights.shape[2:]))
                if kernel != list(weights.shape[2:]):
                    fail('Conv kernel mismatch', name)
                bias = parameter(raw.input[2], name) if len(raw.input) == 3 and raw.input[2] else np.zeros(weights.shape[0], dtype=weights.dtype)
                if bias.shape != (weights.shape[0],):
                    fail('Conv bias dimensions', name)
                cout = weights.shape[0]
            else:
                if len(raw.input) != 1 or attrs.get('ceil_mode', 0) or attrs.get('storage_order', 0):
                    fail('Only single-output floor MaxPool supported', name)
                kernel = list(attrs.get('kernel_shape', []))
                if len(kernel) != 2 or min(kernel) <= 0 or any(pads[j] >= kernel[j % 2] for j in range(4)):
                    fail('Invalid pool kernel/pads', name)
                cout = shape[0]
            normalized = dict(kx=kernel[1], ky=kernel[0], sx=strides[1], sy=strides[0],
                              pad_top=pads[0], pad_down=pads[2], pad_left=pads[1], pad_right=pads[3])
            out_shape = (cout, (shape[1] + pads[0] + pads[2] - kernel[0]) // strides[0] + 1,
                         (shape[2] + pads[1] + pads[3] - kernel[1]) // strides[1] + 1)
            if op == 'Conv':
                layer = ConvLayer(name, shape[0], cout, shape[2], shape[1],
                                  weights=weights, bias=bias, source_name=name, **normalized)
        else:
            if op == 'Add':
                if len(raw.input) != 2 or values[raw.input[1]] != shape:
                    fail('Add requires exactly two equal-shape activations; no broadcast', name)
            elif len(raw.input) != 1:
                fail('Unary activation requires one input', name)
            out_shape = shape
        if min(out_shape) <= 0 or output in shapes and shapes[output] != (1,) + out_shape:
            fail('Output dimensions disagree with operator', name)
        values[output] = out_shape
        nodes.append(GraphNode(name, op, activation_inputs, output, out_shape, normalized, layer))
    if not nodes or output_name not in values:
        fail('Incomplete or empty activation graph')
    live = {output_name}
    for node in reversed(nodes):
        if node.output not in live:
            fail('Dead activation node', node.name)
        live.update(node.inputs)
    for value, name in constant_nodes.items():
        if value not in parameter_uses:
            fail('Dead or nonparameter Constant', name)
    result = GraphIR(input_name, input_shape[1:], output_name, nodes,
                     graph.name or 'graph', str(path),
                     {'opset': next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0)})
    result.topological_nodes()
    return result
