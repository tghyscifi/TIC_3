"""Fail-closed custom-domain p08 frontend. Plain ONNX does not identify LIF state."""
from dataclasses import fields
import math
import numpy as np
from .diagnostics import UnsupportedModelError, DependencyError
from .ir import ConvLayer,FCLayer
from .snn_ir import SNNModel,SNNBlock,LIFSpec

def parse_onnx_snn(path):
    try:
        import onnx
        from onnx import helper,numpy_helper
    except ImportError as exc:
        raise DependencyError('SNN ONNX frontend requires ONNX') from exc
    def fail(message,node=None): raise UnsupportedModelError(message,node=node)
    try:
        model = onnx.load(str(path)); onnx.checker.check_model(model)
    except Exception as exc:
        fail('Invalid SNN ONNX: '+str(exc))
    imports = {o.domain:o.version for o in model.opset_import}
    if (len(imports) != len(model.opset_import) or imports.get('com.tianjic') != 1
            or any(k not in ('','ai.onnx','com.tianjic') for k in imports)
            or not 11 <= imports.get('',imports.get('ai.onnx',0)) <= 18 or model.functions):
        fail('Require ONNX opset 11..18 and com.tianjic version 1 without functions')
    if '' in imports and 'ai.onnx' in imports: fail('Ambiguous ONNX domain imports')
    meta = {p.key:p.value for p in model.metadata_props}
    if len(meta) != len(model.metadata_props) or meta.get('snn_profile') != 'tianjic-p08-v1':
        fail('SNN must explicitly declare snn_profile=tianjic-p08-v1')
    try: dt = float(meta['snn_dt']); encoding = meta['snn_encoding']
    except (KeyError,ValueError): fail('SNN requires snn_dt and snn_encoding metadata')
    graph = model.graph
    if graph.sparse_initializer: fail('Sparse SNN initializers unsupported')
    constants = {v.name:numpy_helper.to_array(v).copy() for v in graph.initializer}
    if len(constants) != len(graph.initializer): fail('Duplicate SNN initializers')
    if len(graph.input) != 1 or len(graph.output) != 1 or graph.input[0].name in constants:
        fail('Exactly one external frame input and one output; no explicit state inputs')
    shapes = {}
    for value in list(graph.input)+list(graph.value_info)+list(graph.output):
        t = value.type.tensor_type
        if (not t.HasField('shape') or any(d.dim_param or not d.HasField('dim_value') or d.dim_value <= 0 for d in t.shape.dim)
                or t.elem_type not in (onnx.TensorProto.FLOAT,onnx.TensorProto.FLOAT16,onnx.TensorProto.DOUBLE)):
            fail('Require static positive floating activation declaration '+value.name)
        shape = tuple(int(d.dim_value) for d in t.shape.dim)
        if value.name in shapes and shapes[value.name] != shape: fail('Conflicting declared shapes')
        shapes[value.name] = shape
    fullshape = shapes[graph.input[0].name]
    if len(fullshape) not in (2,4) or fullshape[0] != 1: fail('Input must be batch-one vector or NCHW')
    used = set(); allnames = set(); alloutputs = {graph.input[0].name}|set(constants)
    for node in graph.node:
        name = node.name or (node.output[0] if node.output else '')
        if not name or name in allnames or len(node.output) != 1 or not node.output[0] or node.output[0] in alloutputs:
            fail('Nodes must have unique names and single unique output',name)
        allnames.add(name); alloutputs.add(node.output[0])
        if len({a.name for a in node.attribute}) != len(node.attribute): fail('Duplicate node attributes',name)
    def attrs(node,allowed):
        if any(a.name not in allowed for a in node.attribute): fail('Unsupported attributes',node.name)
        return {a.name:helper.get_attribute_value(a) for a in node.attribute}
    def declared(output,shape):
        if output in shapes and shapes[output] != (1,)+tuple(shape): fail('Declared shape mismatch '+output)
    def param(key,name):
        if key not in constants: fail('Affine parameters must be static initializers',name)
        a = constants[key]
        if a.dtype.kind != 'f' or not np.all(np.isfinite(a)): fail('Finite floating parameters required',name)
        used.add(key); return a
    nodes = list(graph.node); index = 0; current = graph.input[0].name
    shape = fullshape[1:]; blocks = []
    while index < len(nodes):
        flatten_shape = None
        node = nodes[index]
        if node.op_type == 'Flatten' and node.domain in ('','ai.onnx'):
            a = attrs(node,{'axis'})
            if not blocks or len(shape) != 3 or list(node.input) != [current] or a.get('axis',1) != 1:
                fail('Only CHW Flatten axis=1 between LIF and FC is supported',node.name)
            flatten_shape = shape; shape = (math.prod(shape),); current = node.output[0]
            declared(current,shape); index += 1
            if index == len(nodes): fail('SNN cannot terminate at Flatten')
            node = nodes[index]
        name = node.name or node.output[0]
        if node.domain not in ('','ai.onnx') or not node.input or node.input[0] != current:
            fail('Only a linear affine -> LIF chain is supported',name)
        if node.op_type == 'Conv':
            if flatten_shape is not None or len(shape) != 3 or len(node.input) not in (2,3): fail('Invalid SNN Conv inputs',name)
            a = attrs(node,{'kernel_shape','strides','pads','dilations','group','auto_pad'})
            auto = a.get('auto_pad',b'NOTSET')
            if (auto not in (b'NOTSET',b'VALID') or a.get('group',1) != 1
                    or list(a.get('dilations',[1,1])) != [1,1] or auto == b'VALID' and 'pads' in a):
                fail('Conv requires explicit/VALID padding, group/dilation one',name)
            w = param(node.input[1],name)
            if w.ndim != 4 or min(w.shape) <= 0 or w.shape[1] != shape[0]: fail('Conv weight shape mismatch',name)
            kernel = list(a.get('kernel_shape',w.shape[2:])); strides = list(a.get('strides',[1,1])); pads = list(a.get('pads',[0]*4))
            if (kernel != list(w.shape[2:]) or len(strides) != 2 or min(strides) <= 0 or len(pads) != 4 or min(pads) < 0):
                fail('Invalid Conv geometry',name)
            bias = param(node.input[2],name) if len(node.input) == 3 and node.input[2] else np.zeros(w.shape[0])
            layer = ConvLayer(name,int(shape[0]),int(w.shape[0]),int(shape[2]),int(shape[1]),int(kernel[1]),int(kernel[0]),
                int(strides[1]),int(strides[0]),int(pads[0]),int(pads[2]),int(pads[1]),int(pads[3]),weights=w,bias=bias)
        elif node.op_type == 'Gemm':
            if len(shape) != 1 or len(node.input) not in (2,3): fail('Gemm requires vector or CHW Flatten',name)
            a = attrs(node,{'alpha','beta','transA','transB'})
            if a.get('alpha',1) != 1 or a.get('beta',1) != 1 or a.get('transA',0) != 0 or a.get('transB',0) != 1:
                fail('Gemm requires transB=1, transA=0, alpha=beta=1',name)
            w = param(node.input[1],name)
            if w.ndim != 2 or min(w.shape) <= 0 or w.shape[1] != shape[0]: fail('FC weight shape mismatch',name)
            bias = param(node.input[2],name) if len(node.input) == 3 and node.input[2] else np.zeros(w.shape[0])
            layer = FCLayer(name,int(shape[0]),int(w.shape[0]),w,bias,
                           'CHW' if flatten_shape is not None else 'VECTOR',flatten_shape)
        else: fail('Unsupported SNN affine op '+node.op_type,name)
        if bias.shape != (layer.cout,): fail('Affine bias must be vector of cout',name)
        declared(node.output[0],layer.output_shape)
        index += 1
        if index == len(nodes): fail('Every SNN affine must be followed by explicit LIF',name)
        lif = nodes[index]; lname = lif.name or lif.output[0]
        if lif.domain != 'com.tianjic' or lif.op_type != 'LIF' or list(lif.input) != [node.output[0]]:
            fail('Expected com.tianjic::LIF after affine; no inferred neuron semantics',lname)
        a = attrs(lif,{f.name for f in fields(LIFSpec)})
        if not {'threshold','leak_enabled','leak_alpha'} <= set(a): fail('LIF must explicitly declare threshold/leak_enabled/leak_alpha',lname)
        for f in fields(LIFSpec):
            if f.name not in a: continue
            if type(a[f.name]) is not int: fail('LIF fields must be integer ONNX attributes',lname)
            if f.name in ('leak_enabled','window_enabled'):
                if a[f.name] not in (0,1): fail('LIF boolean attributes must be 0 or 1',lname)
                a[f.name] = bool(a[f.name])
        spec = LIFSpec(**a); spec.validate()
        shape = tuple(layer.output_shape); current = lif.output[0]; declared(current,shape)
        blocks.append(SNNBlock(lname,layer,spec,current)); index += 1
    if not blocks or current != graph.output[0].name or used != set(constants):
        fail('Dead parameters/nodes or final output outside supported SNN chain')
    result = SNNModel(tuple(fullshape[1:]),blocks,dt,encoding,graph.name or 'snn',str(path),meta)
    result.validate()
    return result
