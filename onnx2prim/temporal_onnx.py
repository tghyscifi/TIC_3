"""Export the explicit integer one-frame contract, not raw training graphs.

Custom LIF/Cut/Cast nodes deliberately keep neuron and quantization semantics.
DOUBLE tensor declarations preserve every int32 bias exactly; target execution
still follows declared integer semantics. No checkpoint loading or NIR guessing.
"""
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from .diagnostics import CompilerError, DependencyError
from .temporal_ir import P08Contract


def save_temporal_onnx(graph, path):
    try:
        import onnx
        from onnx import helper as h, numpy_helper as nh
    except ImportError as exc:
        raise DependencyError('Explicit temporal ONNX export requires ONNX') from exc
    path = Path(path)
    if path.exists() or not path.parent.is_dir():
        raise CompilerError('ONNX output must be a new file in an existing directory', code='OUTPUT')
    ordered = graph.ordered()
    occupied = {graph.input_name, graph.output_name} | {n.name for n in ordered} | {n.output for n in ordered}
    counter = 0

    def ident(label):
        nonlocal counter
        while True:
            value = f'__onnx2prim_{label}_{counter:06d}'
            counter += 1
            if value not in occupied:
                occupied.add(value)
                return value

    def info(name, shape):
        return h.make_tensor_value_info(name, onnx.TensorProto.DOUBLE, [1] + list(shape))

    initializers, nodes, value_info = [], [], []
    shapes = {graph.input_name: graph.input_shape}
    for n in ordered:
        ins, op, attrs, domain = list(n.inputs), n.op, {}, ''
        if op in ('Conv', 'FC'):
            l = n.layer
            if op == 'FC' and len(shapes[ins[0]]) == 3:
                flat = ident('flatten')
                nodes.append(h.make_node('Flatten', ins, [flat], name=ident('flatten_node'), axis=1))
                value_info.append(info(flat, (math.prod(shapes[ins[0]]),)))
                ins = [flat]
            for label, raw in (('weight', l.weights), ('bias', l.bias)):
                if label == 'bias' and raw is None:
                    continue
                a = np.asarray(raw, dtype=np.float64)
                if not np.isfinite(a).all() or np.any(a != np.rint(a)):
                    raise CompilerError('Explicit integer ONNX exporter requires integer-valued parameters', node=n.name)
                key = ident(label)
                initializers.append(nh.from_array(a.copy(), key))
                ins.append(key)
            if op == 'Conv':
                attrs = {'kernel_shape':[l.ky,l.kx], 'strides':[l.sy,l.sx],
                         'pads':[l.pad_top,l.pad_left,l.pad_down,l.pad_right]}
            else:
                op, attrs = 'Gemm', {'transB':1}
        elif op == 'LIF':
            domain = 'com.tianjic'
            attrs = asdict(n.neuron)
            if isinstance(n.neuron, P08Contract):
                attrs['profile'] = 'p08'
                initial = np.asarray(attrs['initial_v'], dtype=np.int64)
                attrs['initial_v'] = int(initial) if initial.ndim == 0 else initial.reshape(-1).tolist()
                for key in ('leak_enabled','window_enabled'):
                    attrs[key] = int(attrs[key])
            else:
                if attrs.pop('lif_spec') is not None or attrs['profile'] == 'p08':
                    raise CompilerError('Use P08Contract for temporal p08 ONNX export', node=n.name)
        elif op == 'Add' and len(ins) > 2:
            left = ins[0]
            for right in ins[1:-1]:
                intermediate = ident('add_value')
                nodes.append(h.make_node('Add',[left,right],[intermediate],name=ident('add_node')))
                value_info.append(info(intermediate, n.shape))
                left = intermediate
            ins = [left, ins[-1]]
        elif op == 'Concat':
            attrs = {'axis':1}
        elif op == 'MaxPool':
            attrs = {'kernel_shape':[n.attrs['ky'],n.attrs['kx']],
                     'strides':[n.attrs['sy'],n.attrs['sx']]}
        elif op == 'Cut':
            domain, attrs = 'com.tianjic', {'cut':n.attrs.get('cut',0), 'relu':int(n.attrs.get('relu',False))}
        elif op == 'Relu':
            # A typed Relu may also request a cut. Preserve it explicitly.
            if n.attrs.get('cut',0):
                op, domain, attrs = 'Cut', 'com.tianjic', {'cut':n.attrs['cut'],'relu':1}
        elif op == 'Cast':
            domain = 'com.tianjic'
        elif op != 'Add':
            raise CompilerError('Unsupported explicit temporal ONNX operator', node=n.name)
        nodes.append(h.make_node(op,ins,[n.output],name=n.name,domain=domain,**attrs))
        shapes[n.output] = n.shape
        if n.output != graph.output_name:
            value_info.append(info(n.output,n.shape))
    g = h.make_graph(nodes, graph.model_name, [info(graph.input_name,graph.input_shape)],
                     [info(graph.output_name,shapes[graph.output_name])], initializers, value_info=value_info)
    model = h.make_model(g, opset_imports=[h.make_opsetid('',13),h.make_opsetid('com.tianjic',1)])
    h.set_model_props(model, {'snn_profile':'tianjic-temporal-v1','snn_dt':str(graph.dt),
        'snn_input_dtype':graph.input_dtype,'snn_readout':graph.readout,
        'onnx2prim_integer_contract':'explicit; not lossless raw training conversion',
        'source_model_id':graph.metadata.get('model_id',graph.model_name)})
    onnx.checker.check_model(model)
    encoded = model.SerializeToString()
    # Exclusive creation preserves existing user artifacts even on races.
    with path.open('xb') as stream:
        stream.write(encoded)
    return path.resolve()
