"""Logical integer LIF oracle, independent of primitives and hardware packing."""
import numpy as np
from .ir import ConvLayer
from .quantization import integer_parameters
from .snn_ir import validate_input
from .diagnostics import CompilerError

LOW = -(1 << 27)
HIGH = (1 << 27)-1

def lif_step(current, membrane, refractory, window, spec):
    spec.validate()
    current = np.asarray(current,dtype=np.int64)
    membrane = np.asarray(membrane,dtype=np.int64)
    refractory = np.asarray(refractory,dtype=np.int64)
    if current.shape != membrane.shape or current.shape != refractory.shape:
        raise CompilerError('LIF current and state shapes differ',code='SNN_CONTRACT')
    integrated = np.clip(membrane + np.where(refractory == 0,current,0),LOW,HIGH)
    threshold = max(spec.threshold,spec.floor)
    firing = integrated > threshold
    reset = integrated.copy()
    if spec.reset_mode == 0: reset[firing] = spec.reset_value
    elif spec.reset_mode == 1: reset[firing] -= threshold
    else: reset[firing] -= spec.reset_decrement
    if window >= spec.window_length:
        new_v = np.full_like(reset,spec.initial_v)
    else:
        new_v = ((reset*spec.leak_alpha)//256 if spec.leak_enabled else reset) + spec.leak_beta
    new_v = np.maximum(np.clip(new_v,LOW,HIGH),spec.floor)
    if spec.fire_type == 4: output = firing.astype(np.int64)
    else:
        readout = integrated if spec.fire_type == 2 else new_v
        output = np.clip(readout//(1 << (2*spec.readout_cut)),-128,127)
    new_ref = np.where(firing,spec.refractory,np.maximum(refractory-1,0))
    new_window = (0 if window == spec.window_length else window+1) if spec.window_enabled else window
    return output,new_v,new_ref,new_window

def affine_current(layer, config, x):
    w,bias = integer_parameters(layer,config)
    expected = (layer.cout,layer.cin,layer.ky,layer.kx) if isinstance(layer,ConvLayer) else (layer.cout,layer.cin)
    if w.shape != expected or bias.shape != (layer.cout,):
        raise CompilerError('SNN affine parameter shape mismatch',node=layer.name)
    axes = tuple(range(1,w.ndim))
    bound = np.abs(bias) + int(np.max(np.abs(x))) * np.sum(np.abs(w),axis=axes)
    if np.any(bound > (1 << 31)-1):
        raise CompilerError('Conservative SNN current accumulator bound exceeds int32',code='OVERFLOW',node=layer.name)
    if isinstance(layer,ConvLayer):
        padded = np.pad(x,((0,0),(layer.pad_top,layer.pad_down),(layer.pad_left,layer.pad_right)))
        y = np.empty(layer.output_shape,dtype=np.int64)
        for h,v in np.ndindex(y.shape[1:]):
            patch = padded[:,h*layer.sy:h*layer.sy+layer.ky,v*layer.sx:v*layer.sx+layer.kx]
            y[:,h,v] = np.sum(w*patch[None],axis=(1,2,3)) + bias
    else:
        # Declared Flatten axis=1 uses logical CHW, not physical padded HWC.
        y = w @ x.reshape(-1) + bias
    return y

def run_snn_reference(model, config, frames):
    x = validate_input(model,frames)
    states = {b.name:(np.full(b.affine.output_shape,b.lif.initial_v,dtype=np.int64),
                       np.zeros(b.affine.output_shape,dtype=np.int64),b.lif.window_initial) for b in model.blocks}
    traces = {}
    for frame in x:
        current = frame
        for b in model.blocks:
            u = affine_current(b.affine,config,current)
            current,v,ref,window = lif_step(u,*states[b.name],b.lif)
            states[b.name] = (v,ref,window)
            for key,value in ((b.output,current),(b.name+'.membrane',v),
                              (b.name+'.refractory',ref),(b.name+'.window',np.asarray(window))):
                traces.setdefault(key,[]).append(value.copy())
    return {key:np.stack(values) for key,values in traces.items()}
