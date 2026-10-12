"""DAG lifetimes and conservative single-core C32/HWC physical layout."""
from dataclasses import dataclass,field
import math
import numpy as np
from .ir import ConvLayer
from .layout import align_ceil
from .diagnostics import CompilerError,UnsupportedModelError

def physical_channels(shape): return align_ceil(shape[0],32)
def tensor_words(shape): return physical_channels(shape)*shape[1]*shape[2]//4

def validate_graph(graph):
    ordered=graph.topological_nodes()
    if not ordered: raise UnsupportedModelError('Empty graph')
    shapes={graph.input_name:tuple(graph.input_shape)}
    for shape in [graph.input_shape]+[n.shape for n in ordered]:
        if len(shape)!=3 or any(type(d) is not int or d<=0 for d in shape):
            raise CompilerError('Graph backend requires positive static CHW shapes')
    for n in ordered:
        if n.op not in ('Conv','Add','Relu','MaxPool','Tanh'):
            raise UnsupportedModelError('Unsupported graph operator '+n.op,node=n.name)
        if len(n.inputs)!=(2 if n.op=='Add' else 1):
            raise CompilerError('Wrong graph input count',node=n.name)
        source=shapes[n.inputs[0]]
        if n.op=='Conv':
            l=n.layer
            if not isinstance(l,ConvLayer) or l.groups!=1 or l.ex!=1 or l.ey!=1:
                raise UnsupportedModelError('Only ordinary Conv is verified',node=n.name)
            dims=(l.cin,l.cout,l.kx,l.ky,l.sx,l.sy,l.px,l.py)
            pads=(l.pad_top,l.pad_down,l.pad_left,l.pad_right)
            if any(type(d) is not int or d<=0 for d in dims) or any(type(d) is not int or d<0 for d in pads):
                raise CompilerError('Invalid convolution dimensions',node=n.name)
            if l.input_shape!=source or l.output_shape!=tuple(n.shape):
                raise CompilerError('Conv shape mismatch',node=n.name)
            if np.asarray(l.weights).shape!=(l.cout,l.cin,l.ky,l.kx) or (l.bias is not None and np.asarray(l.bias).shape!=(l.cout,)):
                raise CompilerError('Conv parameter shape mismatch',node=n.name)
        elif n.op=='MaxPool':
            a=n.attrs
            if any(a.get(k,0) for k in ('pad_top','pad_down','pad_left','pad_right')):
                raise UnsupportedModelError('Padded maxpool needs verified negative-infinity boundary handling',node=n.name)
            dims=[a.get(k) for k in ('kx','ky','sx','sy')]
            if any(type(d) is not int or d<=0 for d in dims):
                raise CompilerError('Invalid pool dimensions',node=n.name)
            kx,ky,sx,sy=dims
            if tuple(n.shape)!=(source[0],(source[1]-ky)//sy+1,(source[2]-kx)//sx+1):
                raise CompilerError('Pool shape mismatch',node=n.name)
        else:
            if tuple(n.shape)!=source or (n.op=='Add' and shapes[n.inputs[1]]!=source):
                raise UnsupportedModelError('Broadcasting or shape-changing elementwise operation',node=n.name)
        shapes[n.output]=tuple(n.shape)
    # Every node must contribute to the declared result; dead computation rejected.
    reachable={graph.output_name}
    for n in reversed(ordered):
        if n.output in reachable: reachable.update(n.inputs)
    if any(n.output not in reachable for n in ordered):
        raise CompilerError('Dead/disconnected graph nodes are not supported')
    phases=1+sum(3 if n.op=='Add' else 1 for n in ordered)
    if phases>32:
        raise UnsupportedModelError(f'{phases} phases exceed conservative single-core PI limit; partitioning required')
    return ordered,shapes,phases

@dataclass(frozen=True)
class TensorAllocation:
    address: int
    words: int
    shape: tuple
    born: int
    last_use: int

@dataclass
class GraphPlan:
    tensors: dict
    parameters: dict
    activation_end: int
    concat_addr: int
    concat_words: int
    scratch_addr: int
    scratch_words: int
    phases: int
    persistent_regions: list=field(default_factory=list)

def plan_graph(graph):
    nodes,shapes,phases=validate_graph(graph)
    last={graph.input_name:-1}
    for i,n in enumerate(nodes):
        last.setdefault(n.output,i)
        for v in n.inputs: last[v]=max(last.get(v,-1),i)
    last[graph.output_name]=len(nodes)
    tensors={}; active={}; peak=0
    def allocate(value,born):
        nonlocal peak
        size=tensor_words(shapes[value]); cursor=0
        for _,(start,length) in sorted(active.items(),key=lambda entry:entry[1][0]):
            if cursor+size<=start: break
            cursor=align_ceil(max(cursor,start+length),4)
        active[value]=(cursor,size)
        tensors[value]=TensorAllocation(cursor,size,shapes[value],born,last[value])
        peak=max(peak,cursor+size)
    allocate(graph.input_name,-1)
    for i,n in enumerate(nodes):
        active={v:r for v,r in active.items() if last[v]>=i}
        allocate(n.output,i)
    concat_words=max([0]+[2*tensor_words(n.shape) for n in nodes if n.op=='Add'])
    scratch_words=max([0]+[physical_channels(n.shape)*n.shape[1]*n.shape[2] for n in nodes if n.op in ('Conv','Add')])
    concat_addr=align_ceil(peak,4); scratch_addr=concat_addr+concat_words
    end=scratch_addr+scratch_words
    if end>0x4000: raise CompilerError('Live activation and temporary buffers exceed Mem0; spatial partitioning required',code='MEMORY')
    spill,mem1=align_ceil(max(0x2000,end),4),0x4000
    parameters={}; regions=[]
    def persistent(count,label):
        nonlocal spill,mem1
        start=align_ceil(mem1,4)
        if start+count<=0x8000: mem1=start+count
        else:
            start=align_ceil(spill,4)
            if start+count>0x4000: raise CompilerError('Graph parameters cannot fit data SRAM banks',code='MEMORY')
            spill=start+count
        regions.append((start,start+count,label));return start
    for n in nodes:
        if n.op=='Conv':
            l=n.layer;ww=physical_channels(n.shape)*align_ceil(l.cin,32)*l.kx*l.ky//4;bw=physical_channels(n.shape)
            start=persistent(ww+bw,n.name)
            parameters[n.name]={'weight_addr':start,'weight_words':ww,'bias_addr':start+ww,'bias_words':bw}
        elif n.op=='Tanh':
            parameters[n.name]={'lut_addr':persistent(64,n.name+'.LUT'),'lut_words':64}
    plan=GraphPlan(tensors,parameters,peak,concat_addr,concat_words,scratch_addr,scratch_words,phases,regions)
    validate_graph_plan(graph,plan)
    return plan

def validate_graph_plan(graph,plan):
    nodes,shapes,phases=validate_graph(graph)
    if plan.phases!=phases or set(plan.tensors)!=set(shapes): raise CompilerError('Graph plan coverage mismatch',code='ADDRESS')
    # Recompute facts from the graph, never trust supplied lifetime/length metadata.
    born={graph.input_name:-1};last={graph.input_name:-1}
    for i,n in enumerate(nodes):
        born[n.output]=i;last.setdefault(n.output,i)
        for value in n.inputs: last[value]=max(last.get(value,-1),i)
    last[graph.output_name]=len(nodes)
    required_concat=max([0]+[2*tensor_words(n.shape) for n in nodes if n.op=='Add'])
    required_scratch=max([0]+[physical_channels(n.shape)*n.shape[1]*n.shape[2] for n in nodes if n.op in ('Conv','Add')])
    if any(type(v) is not int for v in (plan.activation_end,plan.concat_addr,plan.concat_words,plan.scratch_addr,plan.scratch_words)) or plan.concat_words<required_concat or plan.scratch_words<required_scratch or min(plan.concat_words,plan.scratch_words)<0 or not 0<plan.activation_end<=0x4000:
        raise CompilerError('Temporary footprint disagrees with graph requirements',code='ADDRESS')
    if set(plan.parameters)!={n.name for n in nodes if n.op in ('Conv','Tanh')}:
        raise CompilerError('Persistent parameter coverage mismatch',code='ADDRESS')
    expected_regions=[]
    for n in nodes:
        if n.op not in ('Conv','Tanh'): continue
        p=plan.parameters[n.name]
        keys={'weight_addr','weight_words','bias_addr','bias_words'} if n.op=='Conv' else {'lut_addr','lut_words'}
        if set(p)!=keys or any(type(value) is not int for value in p.values()):
            raise CompilerError('Invalid parameter record',node=n.name,code='ADDRESS')
        if n.op=='Conv':
            ww=physical_channels(n.shape)*align_ceil(n.layer.cin,32)*n.layer.kx*n.layer.ky//4
            bw=physical_channels(n.shape)
            if p['weight_words']!=ww or p['bias_words']!=bw or p['bias_addr']!=p['weight_addr']+ww:
                raise CompilerError('Conv parameter footprint mismatch',node=n.name,code='ADDRESS')
            expected_regions.append((p['weight_addr'],p['bias_addr']+bw,n.name))
        else:
            if p['lut_words']!=64: raise CompilerError('LUT footprint mismatch',node=n.name,code='ADDRESS')
            expected_regions.append((p['lut_addr'],p['lut_addr']+64,n.name+'.LUT'))
    if sorted(plan.persistent_regions)!=sorted(expected_regions):
        raise CompilerError('Persistent region registry disagrees with parameter records',code='ADDRESS')
    regions=[]
    for value,a in plan.tensors.items():
        if any(type(v) is not int for v in (a.address,a.words,a.born,a.last_use)) or a.born!=born[value] or a.last_use!=last[value] or a.words!=tensor_words(shapes[value]) or a.shape!=shapes[value] or a.address%4 or not 0<=a.address<a.address+a.words<=plan.activation_end:
            raise CompilerError('Invalid activation allocation '+value,code='ADDRESS')
    alloc=list(plan.tensors.items())
    for i,(va,a) in enumerate(alloc):
        for vb,b in alloc[i+1:]:
            if a.born<=b.last_use and b.born<=a.last_use and a.address<b.address+b.words and b.address<a.address+a.words:
                raise CompilerError('Live tensors alias: '+va+' / '+vb,code='ADDRESS')
    for start,length in ((plan.concat_addr,plan.concat_words),(plan.scratch_addr,plan.scratch_words)):
        if length:
            if start%4 or start<plan.activation_end or start+length>0x4000: raise CompilerError('Invalid temporary region',code='ADDRESS')
            regions.append((start,start+length,'temporary'))
    regions.append((0,plan.activation_end,'activation'))
    for a,b,name in plan.persistent_regions:
        if a%4 or not (0<=a<b<=0x4000 or 0x4000<=a<b<=0x8000): raise CompilerError('Persistent bank violation',code='ADDRESS')
        regions.append((a,b,name))
    for i,a in enumerate(regions):
        for b in regions[i+1:]:
            if a[0]<b[1] and b[0]<a[1]: raise CompilerError('Persistent/temporary overlap',code='ADDRESS')
    return True
