"""Fail-closed static ONNX frontend for explicit temporal DAG contracts.

ONNX floating declarations describe source tensors; IR dtypes describe integer
execution. Flatten is an alias with CHW provenance, never a compute primitive.
"""
from dataclasses import fields
import math
import numpy as np

from .diagnostics import DependencyError, UnsupportedModelError
from .ir import ConvLayer, FCLayer
from .temporal_ir import P08Contract, TemporalGraph, TemporalNode


def parse_onnx_temporal(path, config):
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise DependencyError('Temporal frontend requires ONNX') from exc

    def fail(message, node=None):
        raise UnsupportedModelError(message, node=node)

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:
        fail('Invalid temporal ONNX: ' + str(exc))
    imports = {o.domain: o.version for o in model.opset_import}
    if (len(imports) != len(model.opset_import) or model.functions
            or any(d not in ('', 'ai.onnx', 'com.tianjic') for d in imports)
            or '' in imports and 'ai.onnx' in imports
            or not 11 <= imports.get('', imports.get('ai.onnx', 0)) <= 18
            or imports.get('com.tianjic', 1) != 1):
        fail('Require ONNX opset 11..18, custom version 1 and no functions')
    meta = {p.key: p.value for p in model.metadata_props}
    if len(meta) != len(model.metadata_props) or meta.get('snn_profile') not in (
            'tianjic-temporal-v1', 'tianjic-p08-v1'):
        fail('Explicit temporal snn_profile metadata required')
    try:
        dt = float(meta['snn_dt'])
    except (KeyError, ValueError):
        fail('Explicit snn_dt required')
    readout = meta.get('snn_readout', 'spikes')
    if not math.isfinite(dt) or dt <= 0 or readout not in ('spikes', 'sum', 'mean','last'):
        fail('Invalid snn_dt/readout')
    input_dtype=meta.get('snn_input_dtype','int8')
    if input_dtype not in ('int8','int2'):fail('Invalid snn_input_dtype')
    cuts = getattr(config, 'cuts', None)
    if not isinstance(cuts, dict) or any(not isinstance(k, str) or type(v) is not int
                                       or not 0 <= v <= 15 for k, v in cuts.items()):
        fail('config.cuts must map named affine/Add nodes to integer cuts 0..15')
    g = model.graph
    if g.sparse_initializer:
        fail('Sparse initializers unsupported')
    constants = {t.name: numpy_helper.to_array(t).copy() for t in g.initializer}
    if len(constants) != len(g.initializer):
        fail('Duplicate initializers')
    inputs = [v for v in g.input if v.name not in constants]
    if len(inputs) != 1 or len(g.output) != 1:
        fail('Exactly one activation input and output required')
    shapes = {}
    for v in list(g.input) + list(g.value_info) + list(g.output):
        t = v.type.tensor_type
        if (not t.HasField('shape') or any(d.dim_param or not d.HasField('dim_value')
                                         or d.dim_value <= 0 for d in t.shape.dim)):
            fail('Dynamic or nonpositive tensor shape: ' + v.name)
        shape = tuple(int(d.dim_value) for d in t.shape.dim)
        if v.name in shapes and shapes[v.name] != shape:
            fail('Conflicting shape declarations: ' + v.name)
        if v.name not in constants and (len(shape) not in (2, 4) or shape[0] != 1
                or t.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16,
                                      onnx.TensorProto.DOUBLE)):
            fail('Require static batch-one floating vector/CHW declarations')
        shapes[v.name] = shape
    input_name = inputs[0].name
    values = {input_name: (shapes[input_name][1:], input_dtype, input_name, None)}
    names, outputs, used, configured, aliases = set(), {input_name} | set(constants), set(), set(), {}
    nodes = []

    def declared(output, shape):
        if output in shapes and shapes[output] != (1,) + tuple(shape):
            fail('Declared shape mismatch: ' + output)

    def parameter(key, name):
        if key not in constants:
            fail('Affine parameters must be static initializers', name)
        a = constants[key]
        if a.dtype.kind != 'f' or not np.isfinite(a).all():
            fail('Finite floating parameters required', name)
        used.add(key)
        return a.copy()

    allowed = {
        'Conv': {'kernel_shape', 'strides', 'pads', 'dilations', 'group', 'auto_pad'},
        'Gemm': {'alpha', 'beta', 'transA', 'transB'}, 'Add': set(),
        'Concat': {'axis'}, 'Relu': set(), 'Flatten': {'axis'},
        'MaxPool': {'kernel_shape', 'strides', 'pads', 'dilations', 'ceil_mode',
                    'storage_order', 'auto_pad'}, 'Cut': {'cut', 'relu'},
        'Cast': set(),
        'LIF': {f.name for f in fields(P08Contract)} | {'profile', 'beta_numerator',
            'beta_denominator', 'invocation_count', 'invocation_state', 'leak_offset'},
    }
    raw_nodes = list(g.node)
    for raw in raw_nodes:
        name = raw.name or (raw.output[0] if raw.output else '')
        if not name or name in names or len(raw.output) != 1 or not raw.output[0] or raw.output[0] in outputs:
            fail('Unique node names and single unique outputs required', name)
        names.add(name)
        outputs.add(raw.output[0])
    pending = list(raw_nodes)
    while pending:
        ready = [r for r in pending if r.input and r.input[0] in values
                 and all(x in values or x in constants or not x for x in r.input)]
        if not ready:
            fail('Cycle, undefined input or unsupported constant activation')
        for raw in ready:
            pending.remove(raw)
            name, op, output = raw.name or raw.output[0], raw.op_type, raw.output[0]
            if (op not in allowed or raw.domain not in ('', 'ai.onnx', 'com.tianjic')
                    or (raw.domain == 'com.tianjic') != (op in ('LIF', 'Cut','Cast'))
                    or raw.domain == 'com.tianjic' and 'com.tianjic' not in imports
                    or len({a.name for a in raw.attribute}) != len(raw.attribute)
                    or any(a.name not in allowed[op] or a.ref_attr_name for a in raw.attribute)):
                fail('Unsupported operator/domain/attributes: ' + op, name)
            a = {x.name: helper.get_attribute_value(x) for x in raw.attribute}
            # Validate attribute types before interpreting geometry or contracts.
            for key, val in a.items():
                expected = (bytes if key in ('auto_pad', 'profile', 'invocation_state') else
                            float if key in ('alpha', 'beta') else
                            list if key in ('kernel_shape', 'strides', 'pads', 'dilations') or key=='initial_v' and isinstance(val,list) else
                            bytes if key == 'reset_mode' and a.get('profile', b'p08') != b'p08' else int)
                if type(val) is not expected or expected is list and any(type(x) is not int for x in val):
                    fail('Invalid attribute type: ' + key, name)
            shape, dtype, source, feature = values[raw.input[0]]
            layer = neuron = None
            normalized = {}
            activation = raw.input[:1] if op in ('Conv', 'Gemm') else raw.input
            if any(x not in values for x in activation):
                fail('Only activation inputs supported', name)
            ins = tuple(values[x][2] for x in activation)
            if op not in ('Add', 'Concat', 'Conv', 'Gemm') and len(raw.input) != 1:
                fail('Unary operator requires one input', name)
            if op == 'Flatten':
                if a.get('axis', 1) != 1:
                    fail('Flatten requires axis 1', name)
                out_shape = (math.prod(shape),)
                declared(output, out_shape)
                values[output] = (out_shape, dtype, source, feature or (shape if len(shape) == 3 else None))
                aliases[output] = raw.input[0]
                continue
            if feature is not None and op != 'Gemm':
                fail('CHW Flatten alias may only feed Gemm', name)
            if op in ('Conv', 'Gemm'):
                if len(raw.input) not in (2, 3) or dtype not in ('int8', 'int2'):
                    fail('Affine requires int8/int2 input; supply an explicit cut', name)
                w = parameter(raw.input[1], name)
                if op == 'Conv':
                    if len(shape) != 3 or w.ndim != 4 or min(w.shape) <= 0 or w.shape[1] != shape[0]:
                        fail('Conv weight/input shape mismatch', name)
                else:
                    if (len(shape) != 1 or a.get('transA', 0) != 0 or a.get('transB', 0) not in (0, 1)
                            or a.get('alpha', 1) != 1 or a.get('beta', 1) != 1):
                        fail('Gemm requires vector, transA=0 and alpha=beta=1', name)
                    w = w if a.get('transB', 0) else w.T
                    if w.ndim != 2 or min(w.shape) <= 0 or w.shape[1] != shape[0]:
                        fail('Gemm weight shape mismatch', name)
                b = parameter(raw.input[2], name) if len(raw.input) == 3 and raw.input[2] else np.zeros(w.shape[0])
                if b.shape != (w.shape[0],):
                    fail('Affine bias must be cout vector', name)
                if op == 'Gemm':
                    layer = FCLayer(name, shape[0], w.shape[0], w, b,
                                    'CHW' if feature else 'VECTOR', feature, name)
                    op = 'FC'
            if op in ('Conv', 'MaxPool'):
                if (len(shape) != 3 or a.get('auto_pad', b'NOTSET') not in (b'NOTSET', b'VALID')
                        or a.get('auto_pad') == b'VALID' and 'pads' in a
                        or a.get('group', 1) != 1 or a.get('dilations', [1, 1]) != [1, 1]):
                    fail('Only CHW, explicit/VALID pads and unit group/dilation supported', name)
                kernel = a.get('kernel_shape', list(w.shape[2:]) if op == 'Conv' else [])
                strides, pads = a.get('strides', [1, 1]), a.get('pads', [0]*4)
                if (len(kernel) != 2 or min(kernel) <= 0 or len(strides) != 2 or min(strides) <= 0
                        or len(pads) != 4 or min(pads) < 0):
                    fail('Invalid spatial geometry', name)
                normalized = dict(kx=kernel[1], ky=kernel[0], sx=strides[1], sy=strides[0],
                                  pad_top=pads[0], pad_down=pads[2], pad_left=pads[1], pad_right=pads[3])
                if op == 'Conv':
                    if kernel != list(w.shape[2:]):
                        fail('Conv kernel/weight mismatch', name)
                    layer = ConvLayer(name, shape[0], w.shape[0], shape[2], shape[1],
                                      weights=w, bias=b, source_name=name, **normalized)
                    out_shape = layer.output_shape
                else:
                    if dtype != 'int8' or any(pads) or a.get('ceil_mode', 0) or a.get('storage_order', 0):
                        fail('MaxPool requires int8, no padding and floor mode', name)
                    out_shape = (shape[0], (shape[1]-kernel[0])//strides[0]+1,
                                 (shape[2]-kernel[1])//strides[1]+1)
            elif op == 'FC':
                out_shape = layer.output_shape
            else:
                out_shape = shape
                if op == 'Add':
                    if len(raw.input) != 2 or any(values[x][0] != shape for x in raw.input):
                        fail('Add requires two same-shape activations without broadcast', name)
                elif op == 'Concat':
                    if len(raw.input) < 2 or a.get('axis') != 1 or any(
                            values[x][0][1:] != shape[1:] or len(values[x][0]) != len(shape)
                            or values[x][1] != dtype or values[x][3] is not None for x in raw.input):
                        fail('Concat requires same dtype and channel axis 1', name)
                    out_shape = (sum(values[x][0][0] for x in raw.input),) + shape[1:]
                    normalized = {'axis': 1}
                elif op == 'LIF':
                    if dtype != 'int32':
                        fail('LIF requires int32 current', name)
                    try:
                        profile = a.pop('profile', b'p08').decode('utf-8')
                    except UnicodeDecodeError:
                        fail('Invalid neuron profile encoding', name)
                    if profile == 'p08':
                        if any(k not in {f.name for f in fields(P08Contract)} for k in a):
                            fail('Source neuron attributes on p08', name)
                        for k in ('leak_enabled', 'window_enabled'):
                            if k in a:
                                if a[k] not in (0, 1):
                                    fail('Boolean neuron fields require 0 or 1', name)
                                a[k] = bool(a[k])
                        if isinstance(a.get('initial_v'), list):
                            initial = np.asarray(a['initial_v'], dtype=np.int64)
                            if initial.size != math.prod(out_shape):
                                fail('Per-neuron initial_v must match output size', name)
                            a['initial_v'] = initial.reshape(out_shape)
                        neuron = P08Contract(**a)
                        neuron.validate()
                    else:
                        try:
                            from .temporal_neurons import NeuronContract
                            if any(k not in {f.name for f in fields(NeuronContract)} - {'lif_spec', 'profile'} for k in a):
                                fail('Unsupported source neuron attributes', name)
                            for k in ('reset_mode', 'invocation_state'):
                                if k in a:
                                    a[k] = a[k].decode('utf-8')
                            neuron = NeuronContract(profile=profile, **a)
                        except (ImportError, ValueError, TypeError) as exc:
                            fail('Invalid source neuron contract: ' + str(exc), name)
                elif op == 'Cut':
                    if 'cut' not in a or not 0 <= a['cut'] <= 15 or a.get('relu', 0) not in (0, 1):
                        fail('Cut requires explicit cut 0..15 and boolean relu', name)
                    normalized = {'cut': a['cut'], 'relu': bool(a.get('relu', 0))}
                elif op == 'Relu':
                    if dtype != 'int8':
                        fail('Relu requires int8; supply explicit Cut/config.cuts', name)
                    normalized = {'cut': 0, 'relu': True}
                elif op == 'Cast':
                    if dtype not in ('int8','int2'):fail('Cast only widens int8/int2 to int32',name)
            if min(out_shape) <= 0:
                fail('Nonpositive output shape', name)
            declared(output, out_shape)
            out_dtype = ('int32' if op in ('Conv', 'FC', 'Add','Cast') else
                         'int2' if op == 'LIF' and isinstance(neuron, P08Contract) and neuron.fire_type == 5 else
                         'int8' if op in ('LIF', 'Cut', 'Relu') else dtype)
            nodes.append(TemporalNode(name, op, ins, output, tuple(out_shape), out_dtype,
                                      normalized, layer, neuron))
            values[output] = (tuple(out_shape), out_dtype, output, None)
            if name in cuts:
                if op not in ('Conv', 'FC', 'Add'):
                    fail('Configured cut must target affine/Add', name)
                configured.add(name)
                cut_name, cut_output = name + '__cut', output + '__cut'
                if cut_name in names or cut_output in outputs:
                    fail('Inserted cut identity collision', name)
                names.add(cut_name)
                outputs.add(cut_output)
                nodes.append(TemporalNode(cut_name, 'Cut', (output,), cut_output, tuple(out_shape),
                                          'int8', {'cut': cuts[name], 'relu': False}))
                values[output] = (tuple(out_shape), 'int8', cut_output, None)
    if configured != set(cuts) or used != set(constants):
        fail('Unknown configured cuts or dead parameters')
    # Include aliases in source-level liveness so dead Flatten nodes cannot vanish.
    live = {g.output[0].name}
    pending_live = list(raw_nodes)
    while pending_live:
        found = [r for r in pending_live if r.output[0] in live]
        if not found:
            fail('Dead temporal node')
        for r in found:
            live.update(x for x in r.input if x in values)
            pending_live.remove(r)
    output_name = g.output[0].name
    if output_name not in values or values[output_name][3] is not None or output_name in aliases:
        fail('Temporal output must be a compute value, not Flatten alias')
    result = TemporalGraph(input_name, shapes[input_name][1:], values[output_name][2], nodes,
                           input_dtype=input_dtype,dt=dt, readout=readout, model_name=g.name or 'temporal',
                           source_path=str(path), metadata=meta)
    result.ordered()
    return result
