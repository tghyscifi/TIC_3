"""Canonical static batch-one sequence plus explicit future state contracts.

Spatial names: px/kx/sx = width, py/ky/sy = height. Shapes are CHW.
MaxPool.px/py are OUTPUT dimensions for compatibility with the old consumer.
No target addresses or inferred quantization are stored in the source IR.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

Shape = Tuple[int, ...]

@dataclass
class ConvLayer:
    name: str
    cin: int
    cout: int
    px: int
    py: int
    kx: int
    ky: int
    sx: int = 1
    sy: int = 1
    pad_top: int = 0
    pad_down: int = 0
    pad_left: int = 0
    pad_right: int = 0
    ex: int = 1
    ey: int = 1
    groups: int = 1
    weights: Any = field(default=None, repr=False)
    bias: Any = field(default=None, repr=False)
    source_name: str = ''
    @property
    def px_out(self):
        return (self.px + self.pad_left + self.pad_right - (self.kx-1)*self.ex - 1)//self.sx + 1
    @property
    def py_out(self):
        return (self.py + self.pad_top + self.pad_down - (self.ky-1)*self.ey - 1)//self.sy + 1
    @property
    def input_shape(self):
        return (self.cin, self.py, self.px)
    @property
    def output_shape(self):
        return (self.cout, self.py_out, self.px_out)

@dataclass
class MaxPoolLayer:
    name: str
    cin: int
    input_px: int
    input_py: int
    kx: int
    ky: int
    sx: int = 1
    sy: int = 1
    pad_top: int = 0
    pad_down: int = 0
    pad_left: int = 0
    pad_right: int = 0
    source_name: str = ''
    @property
    def cout(self): return self.cin
    @property
    def px(self): return (self.input_px+self.pad_left+self.pad_right-self.kx)//self.sx+1
    @property
    def py(self): return (self.input_py+self.pad_top+self.pad_down-self.ky)//self.sy+1
    @property
    def px_out(self): return self.px
    @property
    def py_out(self): return self.py
    @property
    def input_shape(self): return (self.cin, self.input_py, self.input_px)
    @property
    def output_shape(self): return (self.cin, self.py, self.px)

@dataclass
class FCLayer:
    name: str
    cin: int
    cout: int
    weights: Any = field(default=None, repr=False)
    bias: Any = field(default=None, repr=False)
    input_layout: str = 'VECTOR'
    feature_shape: Optional[Shape] = None
    source_name: str = ''
    @property
    def input_shape(self): return (self.cin,)
    @property
    def output_shape(self): return (self.cout,)

@dataclass
class LayerSeq:
    input_shape: Shape
    layers: List[Tuple[Any, bool]] = field(default_factory=list)
    model_name: str = 'model'
    source_path: str = ''
    opset: int = 0
    input_name: str = 'input'
    output_name: str = 'output'
    metadata: dict = field(default_factory=dict)
    @property
    def output_size(self):
        import math
        return math.prod(self.layers[-1][0].output_shape if self.layers else self.input_shape)

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Shape
    dtype: str
    layout: str
    role: str = 'activation'

@dataclass(frozen=True)
class StateSpec:
    name: str
    shape: Shape
    dtype: str
    initial_value: Any
    reset_boundary: str

@dataclass(frozen=True)
class TimeContract:
    dt: float
    input_encoding: str
    update_order: Tuple[str, ...]
    threshold_comparison: str
    reset_mode: str

@dataclass
class ModelIR:
    sequence: LayerSeq
    tensors: List[TensorSpec] = field(default_factory=list)
    states: List[StateSpec] = field(default_factory=list)
    time: Optional[TimeContract] = None

@dataclass
class GraphNode:
    """Named single-output node; shape excludes batch, no implicit quantization."""
    name: str
    op: str
    inputs: Tuple[str, ...]
    output: str
    shape: Shape
    attrs: dict = field(default_factory=dict)
    layer: Any = field(default=None, repr=False)

@dataclass
class GraphIR:
    input_name: str
    input_shape: Shape
    output_name: str
    nodes: List[GraphNode]
    model_name: str = 'graph'
    source_path: str = ''
    metadata: dict = field(default_factory=dict)

    def topological_nodes(self):
        from .diagnostics import CompilerError
        values = {self.input_name}
        names, outputs = set(), set()
        for node in self.nodes:
            if node.name in names or node.output in outputs or node.output == self.input_name:
                raise CompilerError('Graph node names and values must be unique',node=node.name)
            names.add(node.name); outputs.add(node.output)
        if self.output_name not in values | outputs:
            raise CompilerError('Graph output is undefined')
        pending, ordered = list(self.nodes), []
        while pending:
            ready = [n for n in pending if all(v in values for v in n.inputs)]
            if not ready:
                raise CompilerError('Graph contains a cycle or undefined input')
            for node in ready:
                ordered.append(node); values.add(node.output); pending.remove(node)
        return ordered

def sequence_to_graph(seq, cuts):
    """Keep affine Cut before activation; old compiler retains fused pooling."""
    from .scheduler import plan_blocks
    nodes, current = [], seq.input_name
    for block in plan_blocks(seq):
        layer=block.layer;output=layer.name+'__affine'
        nodes.append(GraphNode(layer.name,'Conv' if isinstance(layer,ConvLayer) else 'FC',
                               (current,),output,layer.output_shape,{'cut':cuts[block.cut_key]},layer))
        current=output
        if block.has_relu:
            output=layer.name+'__relu'
            nodes.append(GraphNode(output,'Relu',(current,),output,layer.output_shape));current=output
        if block.pool is not None:
            pool=block.pool;output=pool.name+'__pool'
            attrs={key:getattr(pool,key) for key in ('kx','ky','sx','sy','pad_top','pad_down','pad_left','pad_right')}
            nodes.append(GraphNode(pool.name,'MaxPool',(current,),output,pool.output_shape,attrs));current=output
    return GraphIR(seq.input_name,seq.input_shape,current,nodes,seq.model_name,seq.source_path)

