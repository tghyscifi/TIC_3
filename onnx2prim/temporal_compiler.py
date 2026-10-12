"""Full-size tiled temporal DAG compute with explicit host-staged transfers.

This is not an autonomous routed-chip compiler. State remains in per-tile
Memory across frames; stage gathers, wide windows and time readout are owned
by the Python controller and are recorded as such in exported manifests.
"""
from dataclasses import dataclass,asdict
from copy import deepcopy
from pathlib import Path
import json,math
import numpy as np
from .quantization import integer_parameters
from .diagnostics import CompilerError
from .temporal_ir import P08Contract
from .temporal_reference import run_temporal_reference,validate_frames
from .temporal_kernels import affine_tiles,sum_tile,point_tile,pool_tiles,lif_tile,pack,unpack,padded

@dataclass
class TemporalCompilation:
    graph:object
    quantization:object
    input_data:object
    kernels:dict
    reductions:dict
    reference_outputs:dict
    behavior_outputs:object=None

    def manifest(self):
        return {'schema_version':4,'scope':'full-size temporal DAG primitive compute, host-staged',
                'model_name':self.graph.model_name,'time_steps':len(self.input_data),'dt':self.graph.dt,
                'input_shape':list(self.graph.input_shape),'input_dtype':self.graph.input_dtype,
                'address_unit':'4-byte word','primitive_behavior_verification':'passed' if self.behavior_outputs is not None else 'not run',
                'state_reset_boundary':'sample','host_staging_required':True,'autonomous_chip_deployment':'NOT implemented',
                'router_timing_or_chip_verification':'not run','time_readout':self.graph.readout,
                'readout_host_aggregation':self.graph.readout!='spikes',
                'state_trace_contract':{'membrane':'signed28 target state',
                    'refractory':'actual packed high4 target counter; source-profile macros keep zero',
                    'previous_spike':'source-profile logical spike history, not refractory counter',
                    'window':'logical host counter, not proof of native wide counter support'},
                'metadata':self.graph.metadata,'nodes':[{'name':n.name,'op':n.op,'shape':list(n.shape),
                    'dtype':n.dtype,'inputs':list(n.inputs),'output':n.output,
                    'neuron':asdict(n.neuron) if n.neuron is not None else None,
                    'tiles':[{'id':t.id,'core':list(t.core),'shape':list(t.shape),
                              'PICs':[op.PIC for op in t.operations],'attributes':t.attrs} for t in self.kernels[n.name]]}
                    for n in self.graph.ordered()]}

def compile_temporal(graph,config,frames):
    from .snn_compiler import validate_quantization
    # Temporal frontend owns explicit cut nodes; do not send these keys to the
    # old linear quantization validator which would interpret them as LIF cuts.
    validate_quantization(type(config)({},config.parameter_mode,config.weight_scale,config.bias_scale))
    x=validate_frames(graph,frames);reference=run_temporal_reference(graph,config,x)
    shapes={graph.input_name:(graph.input_shape,graph.input_dtype)};kernels={};reductions={};statecore=0
    for n in graph.ordered():
        src=shapes[n.inputs[0]]
        if n.op in ('Conv','FC'):
            w,b=integer_parameters(n.layer,config)
            tiles=affine_tiles(n,src[0],src[1],w,b)
            groups={}
            for tile in tiles:
                key=(tile.attrs['co'],tile.attrs['h'],tile.attrs['w'])
                groups.setdefault(key,[]).append(tile)
            for key,members in groups.items():
                if len(members)>1:reductions[(n.name,key)]=sum_tile(n.name+'.partial_sum',members[0].shape,len(members))
        elif n.op=='LIF':
            if not isinstance(n.neuron,P08Contract):
                if n.neuron.profile=='hnn' and n.neuron.threshold<0:raise CompilerError('HNN target macro requires nonnegative threshold',node=n.name)
                if n.neuron.profile=='snntorch' and n.neuron.reset_mode=='subtract' and not -255<=n.neuron.threshold<=256:
                    raise CompilerError('Delayed reset multiplier exceeds target signed9 constant',node=n.name)
            tiles=[]
            for start in range(0,math.prod(n.shape),512):
                if statecore>=159:raise CompilerError('Persistent neuron tiles exhaust chip cores',code='MEMORY')
                tiles.append(lif_tile(n,start,min(512,math.prod(n.shape)-start),(statecore%16,statecore//16)))
                statecore+=1
        elif n.op=='MaxPool':tiles=pool_tiles(n,src[0])
        elif n.op=='Add':
            tiles=[]
            for start in range(0,math.prod(n.shape),512):
                length=min(512,math.prod(n.shape)-start);t=sum_tile(f'{n.name}:{start}',(length,),len(n.inputs))
                t.node=n.name;t.attrs={'start':start,'length':length};tiles.append(t)
        else:
            sizes=[math.prod(shapes[v][0]) for v in n.inputs] if n.op=='Concat' else [math.prod(n.shape)]
            tiles=[]
            for operand,size in enumerate(sizes):
                for start in range(0,size,512):
                    t=point_tile(n,start,min(512,size-start),src[1]);t.attrs['operand']=operand
                    t.id+=f':{operand}';tiles.append(t)
        for t in tiles:
            if t.kind!='lif':t.core=(15,9)
        kernels[n.name]=tiles;shapes[n.output]=(n.shape,n.dtype)
    return TemporalCompilation(graph,config,x,kernels,reductions,reference)

def compile_onnx_temporal(path,config,frames):
    from .temporal_parser import parse_onnx_temporal
    return compile_temporal(parse_onnx_temporal(path,config),config,frames)

def _execute(memory,op):
    outputs=op.prim_execution(memory.read_memory(op.getInfoList()))
    for r in outputs:
        start,length=int(r['startInMemory']),int(r['lengthInMemory'])
        if start<0 or length<0 or start+length>0x8000 or start<0x4000<start+length:
            raise CompilerError('Primitive tile writes across SRAM bank',code='ADDRESS')
        if length!=len(r['data']):raise CompilerError('Primitive tile length/data mismatch',code='ADDRESS')
    memory.writeMem(outputs);memory.updateMemory()

def _load(memory,address,data,dtype):
    rows=pack(data,dtype)
    if address+len(rows)>0x3000:raise CompilerError('Dynamic transfer would overwrite neuron state',code='ADDRESS')
    memory.addMemoryBlock(address,len(rows),rows)

def verify_temporal_behavior(compilation):
    from generator.behavior_simulator.memory import Memory
    c=compilation;c.behavior_outputs=None;work=Memory();tiles=deepcopy(c.kernels);reductions=deepcopy(c.reductions)
    state_mem={};counters={};traces={}
    for n in c.graph.ordered():
        if n.op=='LIF':
            counters[n.name]=n.neuron.window_initial if isinstance(n.neuron,P08Contract) else 0
            for t in tiles[n.name]:
                m=Memory()
                for op in t.operations:
                    for r in op.memory_blocks:m.addMemoryBlock(r['start'],len(r['data']),r['data'])
                state_mem[t.id]=m
    def run(tile,arrays,memory=work):
        for op in tile.operations:
            # LIF state initialization is done only at sample start.
            if tile.kind!='lif':
                for r in op.memory_blocks:memory.addMemoryBlock(r['start'],len(r['data']),r['data'])
        for (address,dtype,shape),a in zip(tile.inputs,arrays):_load(memory,address,a,dtype)
        for op in tile.operations:_execute(memory,op)
        return unpack(memory,tile.output_addr,tile.shape,tile.dtype)
    for time,frame in enumerate(c.input_data):
        values={c.graph.input_name:frame}
        for n in c.graph.ordered():
            args=[values[v] for v in n.inputs];out=np.empty(n.shape,dtype=np.int64)
            if n.op in ('Conv','FC'):
                groups={};l=n.layer
                for t in tiles[n.name]:
                    a=t.attrs;ci,ic=a['ci'],a['ic'];co,oc=a['co'],a['oc']
                    if n.op=='Conv':
                        ph,pw=a['input_shape'][1:];rh=a['h']*l.sy-l.pad_top;rw=a['w']*l.sx-l.pad_left
                        patch=np.zeros(a['input_shape'],dtype=np.int64)
                        h0,h1=max(rh,0),min(rh+ph,l.py);w0,w1=max(rw,0),min(rw+pw,l.px)
                        if h1>h0 and w1>w0:patch[:,h0-rh:h1-rh,w0-rw:w1-rw]=args[0][ci:ci+ic,h0:h1,w0:w1]
                    else:patch=args[0].reshape(-1)[ci:ci+ic]
                    actual=run(t,[patch]);key=(co,a['h'],a['w']);groups.setdefault(key,[]).append(actual)
                for key,partials in groups.items():
                    result=partials[0] if len(partials)==1 else run(reductions[(n.name,key)],partials)
                    co,h,w=key
                    if n.op=='Conv':out[co:co+result.shape[0],h:h+result.shape[1],w:w+result.shape[2]]=result
                    else:out[co:co+len(result)]=result
            elif n.op=='LIF':
                vfull=np.empty(n.shape,dtype=np.int64);rfull=np.empty(n.shape,dtype=np.int64)
                for t in tiles[n.name]:
                    m=state_mem[t.id];a=t.attrs;s=a['start'];length=a['length'];current=args[0].reshape(-1)[s:s+length]
                    if isinstance(n.neuron,P08Contract):
                        p=t.operations[0];p.Tw_cnt=0;p.Tw_len=0 if counters[n.name]>=n.neuron.window_length else 1
                    else:
                        # Zero current is consumed by pre-decay / comparison
                        # stages; the real current remains at address 0.
                        _load(m,0x2800,np.zeros((length,),dtype=np.int64),'int32')
                        if n.neuron.profile=='snntorch' and n.neuron.reset_mode=='subtract':
                            # Place unmodified current after the reset correction
                            # buffer; p83 and p02 do the arithmetic on target.
                            _load(m,0x1800+a['neurons'],current,'int32')
                    result=run(t,[current],m);out.reshape(-1)[s:s+length]=result
                    raw=np.asarray(m.read_memory([{'start':0x3000,'length':a['neurons'],'type':0}])[0],dtype=np.int64).reshape(-1)
                    v=raw&((1<<28)-1);v=np.where(v>=(1<<27),v-(1<<28),v)
                    vfull.reshape(-1)[s:s+length]=v[:length]
                    rfull.reshape(-1)[s:s+length]=((raw&0xffffffff)>>28)[:length]
                if isinstance(n.neuron,P08Contract) and n.neuron.window_enabled:
                    counters[n.name]=0 if counters[n.name]==n.neuron.window_length else counters[n.name]+1
                state_traces=[(n.name+'.membrane',vfull),(n.name+'.refractory',rfull),(n.name+'.window',np.asarray(counters[n.name]))]
                if not isinstance(n.neuron,P08Contract):state_traces.append((n.name+'.previous_spike',out))
                for key,value in state_traces:
                    expected=c.reference_outputs[key][time]
                    if not np.array_equal(value,expected):raise CompilerError(f'Temporal state mismatch t={time}, {key}',code='VERIFICATION')
                    traces.setdefault(key,[]).append(value.copy())
            elif n.op=='MaxPool':
                a=n.attrs
                for t in tiles[n.name]:
                    p=t.attrs;row,col=p['h']*a['sy'],p['w']*a['sx']
                    patch=args[0][p['co']:p['co']+p['oc'],row:row+p['ph'],col:col+p['pw']]
                    result=run(t,[patch]);out[p['co']:p['co']+p['oc'],p['h']:p['h']+p['oh'],p['w']:p['w']+p['ow']]=result
            elif n.op=='Concat':
                copied=[np.empty_like(a) for a in args]
                for t in tiles[n.name]:
                    a=t.attrs;s=a['start'];l=a['length'];i=a['operand']
                    copied[i].reshape(-1)[s:s+l]=run(t,[args[i].reshape(-1)[s:s+l]])
                out=np.concatenate(copied,axis=0)
            else:
                for t in tiles[n.name]:
                    s=t.attrs['start'];l=t.attrs['length']
                    out.reshape(-1)[s:s+l]=run(t,[a.reshape(-1)[s:s+l] for a in args])
            expected=c.reference_outputs[n.output][time]
            if not np.array_equal(out,expected):
                raise CompilerError(f'Temporal node mismatch t={time}, max_abs_diff={int(np.abs(out-expected).max())}',node=n.name,code='VERIFICATION')
            values[n.output]=out;traces.setdefault(n.output,[]).append(out.copy())
    result={k:np.stack(a) for k,a in traces.items()};out=result[c.graph.output_name]
    result['readout']=out.sum(axis=0,dtype=np.int64) if c.graph.readout=='sum' else out.mean(axis=0) if c.graph.readout=='mean' else out[-1].copy() if c.graph.readout=='last' else out.copy()
    if not np.array_equal(result['readout'],c.reference_outputs['readout']):raise CompilerError('Temporal time readout mismatch',code='VERIFICATION')
    c.behavior_outputs=result;return result
