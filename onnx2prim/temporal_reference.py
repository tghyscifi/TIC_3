"""Independent logical DAG and temporal oracle. No primitive/layout imports."""
import numpy as np
from .snn_reference import affine_current
from .temporal_ir import P08Contract
from .temporal_neurons import step
from .diagnostics import CompilerError

def validate_frames(graph,frames):
    graph.ordered();x=np.asarray(frames)
    lo,hi=(-1,1) if graph.input_dtype=='int2' else (-128,127)
    if x.ndim!=len(graph.input_shape)+1 or tuple(x.shape[1:])!=tuple(graph.input_shape) or not len(x) or x.dtype.kind not in 'iu' or np.any(x<lo) or np.any(x>hi):
        raise CompilerError('Temporal frames must be T x shape target integers',code='INPUT')
    return x.astype(np.int64,copy=True)

def p08_step(u,v,ref,window,n):
    integrated=np.clip(v+np.where(ref==0,u,0),-(1<<27),(1<<27)-1)
    firing=integrated>max(n.threshold,n.floor)
    reset=integrated.copy()
    reset[firing]=(n.reset_value if n.reset_mode==0 else integrated[firing]-max(n.threshold,n.floor)
                   if n.reset_mode==1 else integrated[firing]-n.reset_decrement)
    new=(reset*n.leak_alpha//256 if n.leak_enabled else reset)+n.leak_beta
    if window>=n.window_length:
        new=np.full_like(reset,int(n.initial_v) if np.ndim(n.initial_v)==0 else 0)
    new=np.maximum(np.clip(new,-(1<<27),(1<<27)-1),n.floor)
    output=firing.astype(np.int64) if n.fire_type in (4,5) else np.clip(
        (integrated if n.fire_type==2 else new)//(1<<(2*n.readout_cut)),-128,127)
    refs=np.where(firing,n.refractory,np.maximum(ref-1,0))
    counter=(0 if window==n.window_length else window+1) if n.window_enabled else window
    return output,new,refs,counter

def run_temporal_reference(graph,config,frames):
    x=validate_frames(graph,frames);ordered=graph.ordered();states={};traces={}
    for n in ordered:
        if n.op=='LIF':
            initial=np.asarray(n.neuron.initial_v,dtype=np.int64) if isinstance(n.neuron,P08Contract) else np.asarray(0)
            if initial.ndim and initial.shape!=tuple(n.shape):raise CompilerError('Initial neuron state shape',node=n.name)
            v=np.full(n.shape,int(initial),dtype=np.int64) if initial.ndim==0 else initial.copy()
            states[n.name]=(v,np.zeros(n.shape,dtype=np.int64),n.neuron.window_initial if isinstance(n.neuron,P08Contract) else 0)
    for frame in x:
        values={graph.input_name:frame}
        for n in ordered:
            args=[values[k] for k in n.inputs]
            if n.op in ('Conv','FC'):y=affine_current(n.layer,config,args[0])
            elif n.op in ('Cut','Relu'):
                y=np.clip(args[0]//(1<<(2*n.attrs.get('cut',0))),0 if n.op=='Relu' or n.attrs.get('relu',False) else -128,127)
            elif n.op=='MaxPool':
                a=n.attrs;y=np.empty(n.shape,dtype=np.int64)
                for h,w in np.ndindex(n.shape[1:]):
                    y[:,h,w]=args[0][:,h*a['sy']:h*a['sy']+a['ky'],w*a['sx']:w*a['sx']+a['kx']].max(axis=(1,2))
            elif n.op=='Add':
                bound=np.sum([np.abs(a) for a in args],axis=0)
                if np.any(bound>(1<<31)-1):raise CompilerError('Reduction conservative bound exceeds int32',code='OVERFLOW',node=n.name)
                y=np.sum(args,axis=0,dtype=np.int64)
            elif n.op=='Concat':y=np.concatenate(args,axis=0)
            elif n.op=='Cast':y=args[0].copy()
            elif n.op=='LIF':
                v,ref,window=states[n.name]
                if isinstance(n.neuron,P08Contract):y,v,ref,window=p08_step(args[0],v,ref,window,n.neuron)
                else:
                    # Source profiles have no silent hardware saturation.
                    # Fail before target execution if any intermediate could
                    # require a different state representation.
                    if np.any(np.abs(args[0])+np.abs(v)>=(1<<27)):
                        raise CompilerError('Source neuron exceeds conservative signed28 state bound',code='OVERFLOW',node=n.name)
                    try:y,v=step(args[0],v,ref,n.neuron)
                    except OverflowError as exc:raise CompilerError(str(exc),code='OVERFLOW',node=n.name) from exc
                    ref=y.copy()
                states[n.name]=(v,ref,window)
                state_traces=[(n.name+'.membrane',v),(n.name+'.refractory',ref if isinstance(n.neuron,P08Contract) else np.zeros_like(ref)),(n.name+'.window',np.asarray(window))]
                if not isinstance(n.neuron,P08Contract):state_traces.append((n.name+'.previous_spike',ref))
                for key,a in state_traces:
                    traces.setdefault(key,[]).append(a.copy())
            else:raise CompilerError('Unknown temporal oracle op',node=n.name)
            if y.shape!=tuple(n.shape):raise CompilerError('Temporal oracle output mismatch',node=n.name)
            values[n.output]=y;traces.setdefault(n.output,[]).append(y.copy())
    result={k:np.stack(a) for k,a in traces.items()}
    out=result[graph.output_name]
    result['readout']=out.sum(axis=0,dtype=np.int64) if graph.readout=='sum' else out.mean(axis=0) if graph.readout=='mean' else out[-1].copy() if graph.readout=='last' else out.copy()
    return result
