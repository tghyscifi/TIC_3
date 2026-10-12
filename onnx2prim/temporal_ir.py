"""Stateful DAG contract for staged full-size compute (not implicit routing)."""
from dataclasses import dataclass,field
import math
from .diagnostics import CompilerError,UnsupportedModelError
from .ir import ConvLayer,FCLayer

@dataclass(frozen=True)
class P08Contract:
    threshold:int=64
    leak_enabled:bool=True
    leak_alpha:int=128
    leak_beta:int=0
    reset_mode:int=0
    reset_value:int=0
    reset_decrement:int=0
    floor:int=-(1<<27)
    initial_v:object=0
    refractory:int=0
    window_enabled:bool=False
    window_length:int=1
    window_initial:int=0
    fire_type:int=4
    readout_cut:int=0

    def validate(self):
        # Wide windows are an explicit staged-controller operation. The old
        # single-frame backend and its <=255 contract are deliberately unchanged.
        from dataclasses import fields
        for f in fields(self):
            if f.name=='initial_v':continue
            expected=bool if f.name in ('leak_enabled','window_enabled') else int
            if type(getattr(self,f.name)) is not expected:
                raise CompilerError('Invalid p08 field '+f.name,code='TEMPORAL')
        if self.fire_type not in (2,3,4,5) or self.reset_mode not in (0,1,2):
            raise UnsupportedModelError('Unsupported deterministic p08 reset/readout')
        if not 0<=self.leak_alpha<=255 or not 0<=self.refractory<=15 or not 0<=self.readout_cut<=15:
            raise CompilerError('Invalid p08 alpha/refractory/cut',code='TEMPORAL')
        if not 0<=self.window_initial<=self.window_length<=65535:
            raise CompilerError('Invalid explicit temporal window',code='TEMPORAL')
        for k in ('threshold','leak_beta','reset_value','reset_decrement','floor'):
            if not -(1<<27)<=getattr(self,k)<(1<<27):raise CompilerError('p08 28-bit field overflow '+k)
        import numpy as np
        initial=np.asarray(self.initial_v)
        if initial.dtype.kind not in 'iu' or np.any(initial<-(1<<27)) or np.any(initial>=(1<<27)):
            raise CompilerError('p08 initial states must be signed28 integers',code='TEMPORAL')

@dataclass
class TemporalNode:
    name:str
    op:str
    inputs:tuple
    output:str
    shape:tuple
    dtype:str='int8'
    attrs:dict=field(default_factory=dict)
    layer:object=None
    neuron:object=None

@dataclass
class TemporalGraph:
    input_name:str
    input_shape:tuple
    output_name:str
    nodes:list
    input_dtype:str='int8'
    dt:float=1.0
    readout:str='spikes'
    model_name:str='temporal'
    source_path:str=''
    metadata:dict=field(default_factory=dict)

    def ordered(self):
        if self.readout not in ('spikes','sum','mean','last') or not math.isfinite(self.dt) or self.dt<=0:
            raise CompilerError('Invalid temporal dt/readout')
        if self.input_dtype not in ('int8','int2'):raise CompilerError('Input must be int8 or signed int2')
        def shape_ok(shape):return len(shape) in (1,3) and all(type(d) is int and d>0 for d in shape)
        if not shape_ok(self.input_shape):raise CompilerError('Positive vector or CHW input required')
        values={self.input_name:(tuple(self.input_shape),self.input_dtype)};names=set();remaining=list(self.nodes);ordered=[]
        if len({n.output for n in remaining})!=len(remaining):raise CompilerError('Duplicate temporal outputs')
        while remaining:
            ready=[n for n in remaining if all(v in values for v in n.inputs)]
            if not ready:raise CompilerError('Temporal graph has cycle or undefined input')
            for n in ready:
                if not n.name or n.name in names or n.output in values or not n.inputs or not shape_ok(n.shape):
                    raise CompilerError('Invalid temporal node identity/shape',node=n.name)
                names.add(n.name)
                if n.dtype not in ('int8','int32','int2'):raise CompilerError('Unsupported temporal dtype',node=n.name)
                src=[values[v] for v in n.inputs]
                if n.op in ('Conv','FC'):
                    l=n.layer
                    if len(src)!=1 or src[0][1] not in ('int8','int2') or n.dtype!='int32':raise CompilerError('Affine input/output dtype',node=n.name)
                    if n.op=='Conv':
                        if not isinstance(l,ConvLayer) or l.input_shape!=src[0][0] or l.groups!=1 or l.ex!=1 or l.ey!=1:raise UnsupportedModelError('Invalid temporal Conv',node=n.name)
                    else:
                        if not isinstance(l,FCLayer) or math.prod(src[0][0])!=l.cin:raise CompilerError('Temporal FC input shape',node=n.name)
                        if len(src[0][0])==3 and (l.feature_shape!=src[0][0] or l.input_layout!='CHW'):raise CompilerError('FC requires CHW flatten provenance',node=n.name)
                    if tuple(n.shape)!=tuple(l.output_shape):raise CompilerError('Affine output shape mismatch',node=n.name)
                elif n.op=='LIF':
                    if len(src)!=1 or src[0]!=(tuple(n.shape),'int32') or n.neuron is None:raise CompilerError('LIF requires int32 current',node=n.name)
                    n.neuron.validate()
                    dtype='int2' if isinstance(n.neuron,P08Contract) and n.neuron.fire_type==5 else 'int8'
                    if n.dtype!=dtype:raise CompilerError('LIF readout dtype mismatch',node=n.name)
                elif n.op in ('Cut','Relu'):
                    if len(src)!=1 or src[0][0]!=tuple(n.shape) or n.dtype!='int8':raise CompilerError('Cut/Relu shape/dtype',node=n.name)
                    amount=n.attrs.get('cut',0)
                    if type(amount) is not int or not 0<=amount<=15:raise CompilerError('Explicit cut must be 0..15',node=n.name)
                elif n.op=='Cast':
                    if len(src)!=1 or src[0][0]!=tuple(n.shape) or n.dtype!='int32' or src[0][1] not in ('int8','int2'):
                        raise CompilerError('Cast only widens integer current',node=n.name)
                elif n.op=='MaxPool':
                    if len(src)!=1 or len(src[0][0])!=3 or src[0][1]!='int8' or n.dtype!='int8':raise CompilerError('Pool requires CHW int8',node=n.name)
                    a=n.attrs
                    if any(type(a.get(k)) is not int or a[k]<=0 for k in ('kx','ky','sx','sy')):raise CompilerError('Pool geometry required',node=n.name)
                    c,h,w=src[0][0]
                    if tuple(n.shape)!=(c,(h-a['ky'])//a['sy']+1,(w-a['kx'])//a['sx']+1):raise CompilerError('Pool shape mismatch',node=n.name)
                    if any(a.get(k,0) for k in ('pad_top','pad_down','pad_left','pad_right')):raise UnsupportedModelError('Padded temporal pooling not verified',node=n.name)
                elif n.op=='Add':
                    if len(src)<2 or any(s[0]!=tuple(n.shape) for s in src) or n.dtype!='int32':raise CompilerError('Add requires same-shaped inputs, int32 output',node=n.name)
                elif n.op=='Concat':
                    if len(src)<2 or any(len(s[0])!=len(n.shape) or s[0][1:]!=tuple(n.shape[1:]) or s[1]!=n.dtype for s in src):raise CompilerError('Concat must be channel/vector axis',node=n.name)
                    if n.shape[0]!=sum(s[0][0] for s in src):raise CompilerError('Concat size mismatch',node=n.name)
                else:raise UnsupportedModelError('Unsupported temporal op '+n.op,node=n.name)
                values[n.output]=(tuple(n.shape),n.dtype);ordered.append(n);remaining.remove(n)
        if not ordered or self.output_name not in values:raise CompilerError('Temporal output missing')
        reserved={'readout'}|{n.name+suffix for n in ordered if n.op=='LIF'
            for suffix in ('.membrane','.refractory','.window','.previous_spike')}
        if any(n.output in reserved for n in ordered):raise CompilerError('Temporal output collides with state/readout trace names')
        live={self.output_name}
        for n in reversed(ordered):
            if n.output not in live:raise CompilerError('Dead temporal node',node=n.name)
            live.update(n.inputs)
        return ordered
