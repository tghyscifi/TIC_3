"""Fail-closed parser for static, batch-one, linear ONNX networks."""
import math
import numpy as np
from .ir import (ConvLayer, MaxPoolLayer, FCLayer, LayerSeq, TensorSpec,
                 StateSpec, TimeContract, ModelIR)
from .diagnostics import UnsupportedModelError, DependencyError


def parse_onnx(path):
    try:
        import onnx
        from onnx import numpy_helper, helper
    except ImportError as exc:
        raise DependencyError('parse_onnx requires ONNX') from exc
    def fail(message, node=None):
        raise UnsupportedModelError(message, node=node)
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        fail('Invalid ONNX: ' + str(exc))
    g = inferred.graph
    if any(o.domain not in ('', 'ai.onnx') for o in model.opset_import):
        fail('Custom domains are unsupported')
    constants = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    inputs = [v for v in g.input if v.name not in constants]
    if len(inputs) != 1 or len(g.output) != 1:
        fail('Exactly one activation input and output required')
    shapes = {}
    for v in list(g.input) + list(g.value_info) + list(g.output):
        dims = v.type.tensor_type.shape.dim
        if not v.type.tensor_type.HasField('shape') or any(d.dim_param or not d.HasField('dim_value') or d.dim_value <= 0 for d in dims):
            fail('Dynamic or nonpositive tensor shape: ' + v.name)
        shapes[v.name] = tuple(d.dim_value for d in dims)
    shape = shapes[inputs[0].name]
    if len(shape) not in (2, 4) or shape[0] != 1:
        fail('Input must be batch-one NC or NCHW')
    if inputs[0].type.tensor_type.elem_type not in (1, 10, 11):
        fail('Floating activation input required')
    nodes = []
    for n in g.node:
        if n.domain not in ('', 'ai.onnx'):
            fail('Custom node domain', n.name)
        if n.op_type == 'Constant':
            if len(n.input) or len(n.output) != 1 or len(n.attribute) != 1 or n.attribute[0].name != 'value':
                fail('Only tensor Constant supported', n.name)
            constants[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
        else:
            nodes.append(n)
    consumers = {}
    for n in nodes:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    seq = LayerSeq(shape[1:], model_name=g.name or 'model', source_path=str(path),
                   opset=next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0),
                   input_name=inputs[0].name, output_name=g.output[0].name)
    current = inputs[0].name
    feature, layout, pending_transpose = None, 'VECTOR', False
    def tensor(name):
        if name not in constants:
            fail('Parameter must be constant: ' + name)
        a = constants[name]
        if a.dtype.kind != 'f' or not np.isfinite(a).all():
            fail('Finite floating parameter required: ' + name)
        return a.astype(np.float64, copy=True)
    for i, n in enumerate(nodes):
        op, name = n.op_type, n.name or n.output[0]
        allowed = {'Conv': {'kernel_shape','strides','pads','dilations','group','auto_pad'},
                   'Relu': set(), 'MaxPool': {'kernel_shape','strides','pads','dilations','ceil_mode','storage_order','auto_pad'},
                   'Flatten': {'axis'}, 'Reshape': {'allowzero'}, 'Transpose': {'perm'},
                   'Gemm': {'transA','transB','alpha','beta'}, 'MatMul': set(), 'Add': set()}
        if op not in allowed or any(a.name not in allowed[op] for a in n.attribute):
            fail('Unsupported operator or attribute: ' + op, name)
        a = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if len(n.output) != 1 or current not in n.input or len(consumers.get(current, [])) != 1:
            fail('Graph must be an exact linear chain without branches', name)
        if current == seq.output_name:
            fail('Graph output consumed by another node', name)
        if any(x != current and x not in constants for x in n.input):
            fail('Nonconstant side input', name)
        if pending_transpose and op not in ('Flatten', 'Reshape'):
            fail('Transpose must immediately precede flatten', name)
        out = None
        if op in ('Conv', 'MaxPool'):
            if len(shape) != 4 or layout == 'HWC': fail('Spatial operator requires NCHW', name)
            if a.get('auto_pad', b'NOTSET') != b'NOTSET' or a.get('group', 1) != 1 or list(a.get('dilations', [1,1])) != [1,1]:
                fail('Unsupported spatial mode', name)
            if a.get('ceil_mode', 0) or a.get('storage_order', 0): fail('Unsupported pool mode', name)
            stride, pads = list(a.get('strides', [1,1])), list(a.get('pads', [0,0,0,0]))
            if len(stride)!=2 or len(pads)!=4 or min(stride)<=0 or min(pads)<0: fail('Invalid spatial attributes', name)
            if op == 'Conv':
                if len(n.input) not in (2,3) or n.input[0]!=current: fail('Invalid Conv inputs', name)
                w = tensor(n.input[1])
                if w.ndim!=4 or w.shape[1]!=shape[1]: fail('Conv weight dimensions', name)
                kernel = list(a.get('kernel_shape', w.shape[2:]))
                if kernel != list(w.shape[2:]): fail('Conv kernel mismatch', name)
                b = tensor(n.input[2]) if len(n.input)==3 else np.zeros(w.shape[0])
                if b.shape!=(w.shape[0],): fail('Conv bias dimensions', name)
                layer = ConvLayer(name, shape[1], w.shape[0], shape[3], shape[2], kernel[1], kernel[0], stride[1], stride[0], pad_top=pads[0], pad_down=pads[2], pad_left=pads[1], pad_right=pads[3], weights=w, bias=b, source_name=name)
            else:
                if len(n.input)!=1: fail('Invalid pool inputs', name)
                kernel = list(a.get('kernel_shape', []))
                if len(kernel)!=2 or min(kernel)<=0 or any(pads[j]>=kernel[j%2] for j in range(4)): fail('Invalid pool kernel/pads', name)
                layer = MaxPoolLayer(name, shape[1], shape[3], shape[2], kernel[1], kernel[0], stride[1], stride[0], pad_top=pads[0], pad_down=pads[2], pad_left=pads[1], pad_right=pads[3], source_name=name)
            out = (1,) + layer.output_shape
            seq.layers.append((layer, False))
            feature, layout = None, 'VECTOR'
        elif op == 'Relu':
            if len(n.input)!=1 or not seq.layers or seq.layers[-1][1] or pending_transpose or len(shape)==2 and feature is not None:
                fail('Relu cannot be represented at this position', name)
            seq.layers[-1] = (seq.layers[-1][0], True)
            out = shape
        elif op == 'Transpose':
            if len(n.input)!=1 or len(shape)!=4 or list(a.get('perm', [])) != [0,2,3,1]: fail('Only NCHW to NHWC supported', name)
            feature, layout, pending_transpose = shape[1:], 'HWC', True
            out = (1,shape[2],shape[3],shape[1])
        elif op in ('Flatten','Reshape'):
            out = (1, math.prod(shape[1:]))
            if op=='Flatten':
                if len(n.input)!=1 or a.get('axis',1)!=1: fail('Only batch-preserving flatten supported', name)
            else:
                if len(n.input)!=2 or n.input[0]!=current or a.get('allowzero',0): fail('Invalid Reshape', name)
                target = constants[n.input[1]]
                if target.dtype.kind not in 'iu' or target.shape!=(2,): fail('Reshape requires constant two-axis shape', name)
                resolved = [shape[j] if int(d)==0 else int(d) for j,d in enumerate(target)]
                if resolved.count(-1)>1 or any(d < -1 for d in resolved): fail('Invalid reshape dimensions', name)
                if -1 in resolved:
                    j=resolved.index(-1); other=resolved[1-j]
                    if other<=0 or math.prod(shape)%other: fail('Invalid reshape product', name)
                    resolved[j]=math.prod(shape)//other
                if tuple(resolved)!=out: fail('Reshape must flatten with batch one', name)
            if len(shape)==4 and feature is None: feature, layout = shape[1:], 'CHW'
            pending_transpose = False
        elif op in ('Gemm','MatMul'):
            if len(shape)!=2 or n.input[0]!=current or len(n.input) not in ((2,3) if op=='Gemm' else (2,)) or a.get('transA',0): fail('Invalid FC inputs', name)
            if a.get('transB',0) not in (0,1): fail('Invalid transB', name)
            w=tensor(n.input[1])
            if w.ndim!=2: fail('FC weight rank', name)
            w = w if a.get('transB',0) else w.T
            w = w * float(a.get('alpha',1))
            if w.shape[1]!=shape[1]: fail('FC weight dimensions', name)
            b=np.zeros(w.shape[0])
            if len(n.input)==3:
                raw=tensor(n.input[2])
                if raw.shape not in ((),(1,),(w.shape[0],),(1,w.shape[0])): fail('FC bias dimensions', name)
                b=np.broadcast_to(raw,(1,w.shape[0])).reshape(-1).copy()*float(a.get('beta',1))
            if not np.isfinite(w).all() or not np.isfinite(b).all(): fail('Nonfinite FC scaling', name)
            seq.layers.append((FCLayer(name,shape[1],w.shape[0],w,b,layout,feature,name),False))
            out=(1,w.shape[0]); feature,layout=None,'VECTOR'
        else:
            if len(n.input)!=2 or i==0 or nodes[i-1].op_type not in ('MatMul','Gemm') or not isinstance(seq.layers[-1][0],FCLayer): fail('Add only supported as immediate FC bias', name)
            raw=tensor(n.input[1] if n.input[0]==current else n.input[0])
            if raw.shape not in ((shape[1],),(1,shape[1])): fail('Add bias dimensions',name)
            seq.layers[-1][0].bias += raw.reshape(-1)
            if not np.isfinite(seq.layers[-1][0].bias).all(): fail('Nonfinite accumulated FC bias',name)
            out=shape
        if any(d<=0 for d in out) or n.output[0] in shapes and shapes[n.output[0]]!=out:
            fail('Output dimensions disagree with operator', name)
        shape,current=out,n.output[0]
    if current!=seq.output_name or pending_transpose or not seq.layers:
        fail('Incomplete or empty compute chain')
    # Constants are allowed only for shape targets or actual operator parameters.
    for n in g.node:
        if n.op_type=='Constant':
            uses=consumers.get(n.output[0], [])
            if not uses or any(u.op_type!='Reshape' or len(u.input)!=2 or u.input[1]!=n.output[0] for u in uses):
                fail('Constant nodes are supported only as reshape shapes',n.name)
    return seq
