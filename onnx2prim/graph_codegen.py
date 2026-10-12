"""Single-core DAG lowering; Add uses p06 copies + p02 sum + pX5 cut."""
from dataclasses import replace
import numpy as np
from .layout import pack_weights,pack_bias
from .graph_planner import physical_channels
from .quantization import integer_parameters
from .diagnostics import DependencyError,CompilerError

def pack_graph_activation(x):
    x=np.asarray(x)
    c,h,w=x.shape
    padded=np.zeros((physical_channels(x.shape),h,w),dtype=np.int64)
    padded[:c]=x
    return padded.transpose(1,2,0).reshape(-1,4).tolist()

def build_graph_mapping(graph,config,plan,input_data):
    if int(np.__version__.split('.')[0])>=2:
        raise DependencyError('Original behavior primitives require NumPy < 2')
    try:
        from generator.mapping_utils.prims import p41,pX5,p02,p06,p07
    except ImportError as exc:
        raise DependencyError('Graph backend requires original Tianjic primitives and PyTorch') from exc
    from .graph_reference import make_tanh_table
    prims=[dict(axon=None,soma1=None,router=None,soma2=None)]
    schedule=[]
    def add_phase(axon=None,soma1=None,soma2=None):
        for op in (axon,soma1,soma2):
            if op is not None: op.cal_para()
        prims.append(dict(axon=axon,soma1=soma1,router=None,soma2=soma2))
    for n in graph.topological_nodes():
        before=len(prims)
        src=[plan.tensors[v] for v in n.inputs];out=plan.tensors[n.output]
        c,h,w=n.shape;pc=physical_channels(n.shape)
        if n.op=='Conv':
            l=n.layer;params=plan.parameters[n.name]
            weights,bias=integer_parameters(l,config.parameter_config(n))
            # Uniform C32 storage, including first padded convolution.
            physical_cin=physical_channels(src[0].shape)
            padded=np.zeros((l.cout,physical_cin,l.ky,l.kx),dtype=np.int64)
            padded[:,:l.cin]=weights
            physical=replace(l,cin=physical_cin,weights=padded,bias=bias)
            axon=p41(l.px,l.py,physical_cin,l.cout,l.kx,l.ky,l.sx,l.sy,
                     src[0].address,params['weight_addr'],params['bias_addr'],plan.scratch_addr,
                     load_bias=2,pad_top=l.pad_top,pad_down=l.pad_down,pad_left=l.pad_left,pad_right=l.pad_right)
            axon.memory_blocks=[{'name':n.name+'.weight','start':params['weight_addr'],
                'data':pack_weights(physical).reshape(-1,4).astype(np.int64).tolist()},
                {'name':n.name+'.bias','start':params['bias_addr'],
                 'data':pack_bias(bias,l.cout).reshape(-1,1).astype(np.int64).tolist()}]
            soma=pX5('max',plan.scratch_addr,out.address,pc,pc,w,h,1,1,1,1,
                      0x80808080,type_in=0,type_out=1,in_cut_start=config.cuts[n.name])
            add_phase(axon,soma)
        elif n.op=='Add':
            # Read both branches before writing result; never alias the live skip.
            for i,a in enumerate(src):
                length=a.words*4
                mover=p06(a.address,plan.concat_addr+i*a.words,0,
                           length,1,1,1,length,1,1,1)
                add_phase(soma1=mover)
            axon=p02(False,1,0,1,2,1,1,pc,w,h,plan.concat_addr,0,plan.scratch_addr,constant_b=0)
            soma=pX5('max',plan.scratch_addr,out.address,pc,pc,w,h,1,1,1,1,
                      0x80808080,type_in=0,type_out=1,in_cut_start=config.cuts[n.name])
            add_phase(axon,soma)
        elif n.op in ('Relu','MaxPool'):
            a=n.attrs;source_c,source_h,source_w=src[0].shape
            kx,ky,sx,sy=(a[k] for k in ('kx','ky','sx','sy')) if n.op=='MaxPool' else (1,1,1,1)
            soma=pX5('max',src[0].address,out.address,pc,pc,source_w,source_h,
                     kx,ky,sx,sy,0 if n.op=='Relu' else 0x80808080,type_in=1,type_out=1,in_cut_start=0)
            add_phase(soma1=soma)
        elif n.op=='Tanh':
            params=plan.parameters[n.name]
            table=np.asarray(make_tanh_table(config,n))
            if table.shape!=(256,) or np.any(table<-128) or np.any(table>127):
                raise CompilerError('Invalid int8 Tanh table',node=n.name)
            soma=p07(src[0].address,out.address,params['lut_addr'],h*w,pc,1,type_in=1,type_out=1)
            soma.memory_blocks=[{'name':n.name+'.LUT','start':params['lut_addr'],
                                 'data':table.reshape(-1,4).astype(np.int64).tolist()}]
            add_phase(soma1=soma)
        schedule.append({'node':n.name,'output':n.output,'start_phase':before,'end_phase':len(prims)-1})
    if len(prims)!=plan.phases: raise CompilerError('Lowering phase count disagrees with planner')
    first=next(op for p in prims[1:] for op in (p['axon'],p['soma1'],p['soma2']) if op is not None)
    first.memory_blocks.append({'name':'graph.input','start':plan.tensors[graph.input_name].address,
                                'data':pack_graph_activation(input_data)})
    mapping={'sim_clock':100000,'step_clock':{((0,0),0):(99999,100000)},
             ((0,0),0):{0:{'clock':200000,'mode':1,((0,0),(0,0)):{'prims':prims}}}}
    return mapping,schedule
