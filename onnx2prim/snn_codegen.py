"""Affine(p41) -> deterministic LIF(p08), with persistent packed membrane/ref state."""
from dataclasses import replace
import numpy as np
from .ir import ConvLayer, FCLayer
from .layout import pack_weights, pack_bias
from .quantization import integer_parameters
from .snn_planner import physical_shape, neuron_words, validate_snn_plan
from .diagnostics import DependencyError, CompilerError

def pack_snn_activation(x):
    x = np.asarray(x, dtype=np.int64)
    padded = np.zeros(physical_shape(x.shape), dtype=np.int64)
    slices = (slice(x.shape[0]),) + (slice(None),)*(x.ndim-1)
    padded[slices] = x
    return (padded.transpose(1,2,0) if x.ndim == 3 else padded).reshape(-1,4).tolist()

def unpack_snn(data, shape):
    raw = np.asarray(data, dtype=np.int64)
    if len(shape) == 1:
        return raw.reshape(-1)[:shape[0]].copy()
    c,h,w = shape
    return raw.reshape(h,w,physical_shape(shape)[0])[:,:,:c].transpose(2,0,1).copy()

def physical_affine(layer, source_shape, config):
    weights,bias = integer_parameters(layer,config)
    if isinstance(layer, ConvLayer):
        cin = physical_shape(source_shape)[0]
        padded = np.zeros((layer.cout,cin,layer.ky,layer.kx),dtype=np.int64)
        padded[:,:layer.cin] = weights
        return replace(layer,cin=cin,weights=padded,bias=bias)
    cin = neuron_words(source_shape)
    if len(source_shape) == 3:
        c,h,w = source_shape
        padded = np.zeros((layer.cout,h,w,physical_shape(source_shape)[0]),dtype=np.int64)
        padded[:,:,:,:c] = weights.reshape(layer.cout,c,h,w).transpose(0,2,3,1)
        padded = padded.reshape(layer.cout,cin)
    else:
        padded = np.zeros((layer.cout,cin),dtype=np.int64)
        padded[:,:layer.cin] = weights
    return FCLayer(layer.name,cin,layer.cout,padded,bias)

def build_snn_mapping(model, config, plan, initial_input):
    validate_snn_plan(model,plan)
    if int(np.__version__.split('.')[0]) >= 2:
        raise DependencyError('Original behavior primitives require NumPy < 2')
    try:
        from generator.mapping_utils.prims import p41,p08
    except ImportError as exc:
        raise DependencyError('SNN backend requires original primitives and PyTorch') from exc
    prims = [dict(axon=None,soma1=None,router=None,soma2=None)]
    schedule = []; source_shape = model.input_shape; source_addr = plan.input_addr
    for phase,b in enumerate(model.blocks,1):
        logical = b.affine; l = physical_affine(logical,source_shape,config)
        p = plan.parameters[b.name]; s = plan.states[b.name]; out = plan.outputs[b.output]
        geometry = (l.px,l.py,l.kx,l.ky,l.sx,l.sy) if isinstance(l,ConvLayer) else (1,1,1,1,1,1)
        px,py,kx,ky,sx,sy = geometry
        pads = {key:getattr(l,key) for key in ('pad_top','pad_down','pad_left','pad_right')} if isinstance(l,ConvLayer) else {}
        axon = p41(px,py,l.cin,l.cout,kx,ky,sx,sy,source_addr,p['weight_addr'],p['bias_addr'],
                   plan.scratch_addr,load_bias=2,**pads)
        axon.cal_para()
        w = pack_weights(l).reshape(-1,4).astype(np.int64).tolist()
        bias = pack_bias(l.bias,l.cout).reshape(-1,1).astype(np.int64).tolist()
        if len(w) != p['weight_words'] or len(bias) != p['bias_words']:
            raise CompilerError('SNN physical parameter size mismatch',node=b.name,code='ADDRESS')
        axon.memory_blocks = [{'name':b.name+'.weight','start':p['weight_addr'],'data':w},
                              {'name':b.name+'.bias','start':p['bias_addr'],'data':bias}]
        shape = logical.output_shape; f = b.lif
        # Row pipeline is off: flatten spatial HWC neurons into one group.
        # The original p08 reshapes out_reg in-place and cannot replay a
        # multi-group vector on its second call. Single group preserves layout
        # and avoids modifying the legacy primitive implementation.
        soma = p08(neuron_num=neuron_words(shape),group_num=1,row_ck_on=0,seed=0,
                   Vth0=f.threshold,Vleaky_adpt_en=f.leak_enabled,Vleaky_alpha=f.leak_alpha,
                   Vleaky_beta=f.leak_beta,ref_len=f.refractory,addr_uin=plan.scratch_addr,
                   addr_v=s['membrane_addr'],addr_s=out['address'],addr_vm=s['vm_addr'],
                   addr_vtheta=s['vtheta_addr'],addr_para=s['para_addr'],in_cut_start=f.readout_cut,
                   in_row_max=0,VR=f.reset_value,VL=f.floor,dV=f.reset_decrement,Vinit=f.initial_v,
                   Tw_en=f.window_enabled,Tw_len=f.window_length,Tw_cnt=f.window_initial,
                   Rst_mode=f.reset_mode,fire_type=f.fire_type)
        soma.cal_para()
        soma.memory_blocks = [
            {'name':b.name+'.membrane','start':s['membrane_addr'],
             'data':[[f.initial_v & ((1<<28)-1)]]*s['membrane_words']},
            {'name':b.name+'.VM','start':s['vm_addr'],'data':[[0]]*4},
            {'name':b.name+'.Vtheta','start':s['vtheta_addr'],'data':[[0]]*4},
            {'name':b.name+'.para','start':s['para_addr'],'data':[[0]]*4}]
        if phase == 1:
            axon.memory_blocks.append({'name':'snn.input','start':plan.input_addr,
                                       'data':pack_snn_activation(initial_input)})
        prims.append(dict(axon=axon,soma1=soma,router=None,soma2=None))
        schedule.append({'phase':phase,'block':b.name,'output':b.output,'state':s.copy()})
        source_shape = shape; source_addr = out['address']
    return {'sim_clock':100000,'step_clock':{((0,0),0):(99999,100000)},
            ((0,0),0):{0:{'clock':200000,'mode':1,((0,0),(0,0)):{'prims':prims}}}},schedule
