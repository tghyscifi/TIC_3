"""SRAM-bounded primitive kernels for explicit host-staged temporal execution.

Host gathering/halo/flatten actions are transfers, NOT implemented routers.
Every arithmetic kernel below uses the original Memory and primitives.
"""
from dataclasses import dataclass,field,replace
import math
import numpy as np
from .ir import ConvLayer,FCLayer
from .layout import align_ceil,pack_weights,pack_bias
from .diagnostics import CompilerError
from .temporal_ir import P08Contract

TYPES={'int32':0,'int8':1,'int2':3}
def padded(shape,dtype):return (align_ceil(shape[0],128 if dtype=='int2' else 32),)+tuple(shape[1:])
def pack(data,dtype):
    x=np.asarray(data,dtype=np.int64);p=np.zeros(padded(x.shape,dtype),dtype=np.int64);p[:x.shape[0]]=x
    a=(p.transpose(1,2,0) if p.ndim==3 else p).reshape(-1)
    return a.reshape(-1,1 if dtype=='int32' else 16 if dtype=='int2' else 4).tolist()
def unpack(memory,address,shape,dtype):
    ps=padded(shape,dtype);factor=1 if dtype=='int32' else 16 if dtype=='int2' else 4
    rows=memory.read_memory([{'start':address,'length':math.prod(ps)//factor,'type':TYPES[dtype]}])[0]
    x=np.asarray(rows,dtype=np.int64)
    if len(shape)==3:return x.reshape(shape[1],shape[2],ps[0])[:,:,:shape[0]].transpose(2,0,1).copy()
    return x.reshape(-1)[:shape[0]].copy()
def initblock(name,start,data):return {'name':name,'start':start,'data':data}
def operation(memory,op):
    memory.writeMem(op.prim_execution(memory.read_memory(op.getInfoList())))
    memory.updateMemory()

@dataclass
class TileKernel:
    id:str
    node:str
    kind:str
    shape:tuple
    dtype:str
    output_addr:int
    operations:list
    inputs:list=field(default_factory=list)
    attrs:dict=field(default_factory=dict)
    core:tuple=(0,0)

    def validate(self):
        # Verify real primitive read lengths and initial regions, not estimated
        # logical sizes. All dynamic buffers must stay outside persistent state.
        regions=[]
        for op in self.operations:
            op.cal_para()
            for r in op.getInfoList():
                start,length=r['start'],int(r['length'])
                if start<0 or length<0 or start+length>0x8000 or start<0x4000<start+length:
                    raise CompilerError('Tile primitive read crosses SRAM bank',code='ADDRESS',node=self.id)
            for r in op.memory_blocks:
                start=r['start'];end=start+len(r['data'])
                if start%4 or end>0x8000 or start<0x4000<end:
                    raise CompilerError('Tile init region invalid',code='ADDRESS',node=self.id)
                if any(start<b and a<end for a,b in regions):
                    raise CompilerError('Tile init regions overlap',code='ADDRESS',node=self.id)
                regions.append((start,end))
        # Writes are checked after every execution as well (including p08's
        # four outputs), because legacy primitives derive lengths internally.
        return self

    def mapping(self):
        prims=[dict(axon=None,soma1=None,router=None,soma2=None)]
        for op in self.operations:
            prims.append(dict(axon=op if op.PIC in (2,4,0x41,0x81,0x83) else None,
                              soma1=op if op.PIC in (5,6,7,8) else None,router=None,soma2=None))
        return {'sim_clock':100000,'step_clock':{((0,0),0):(99999,100000)},
                ((0,0),0):{0:{'clock':200000,'mode':1,((0,0),self.core):{'prims':prims}}}}

def _imports():
    if int(np.__version__.split('.')[0])>=2:raise CompilerError('Legacy kernels require NumPy <2')
    from generator.mapping_utils.prims import p41,p02,p08,pX5,p83,p06
    return p41,p02,p08,pX5,p83,p06

def affine_tiles(node,source_shape,source_dtype,weights,bias,spatial=4,cout_tile=32):
    p41,*_=_imports();l=node.layer;tiles=[]
    isconv=isinstance(l,ConvLayer);kernel=l.kx*l.ky if isconv else 1
    ci_align=128 if source_dtype=='int2' else 32
    maxci=align_ceil(1,ci_align)*((0x4000-32)*4//(32*kernel*ci_align))
    maxci=min(512,maxci)
    if maxci<ci_align:raise CompilerError('Minimum affine tile cannot fit weight bank',code='MEMORY',node=node.name)
    positions=[(0,0,1,1)] if not isconv else [(h,w,min(spatial,l.py_out-h),min(spatial,l.px_out-w))
                for h in range(0,l.py_out,spatial) for w in range(0,l.px_out,spatial)]
    for co in range(0,l.cout,cout_tile):
        oc=min(cout_tile,l.cout-co)
        for h,w,oh,ow in positions:
            for ci in range(0,l.cin,maxci):
                ic=min(maxci,l.cin-ci);pc=align_ceil(ic,ci_align)
                b=bias[co:co+oc] if ci==0 else np.zeros(oc,dtype=np.int64)
                if isconv:
                    ph=(oh-1)*l.sy+l.ky;pw=(ow-1)*l.sx+l.kx
                    raw=np.zeros((oc,pc,l.ky,l.kx),dtype=np.int64);raw[:,:ic]=weights[co:co+oc,ci:ci+ic]
                    physical=ConvLayer(node.name,pc,oc,pw,ph,l.kx,l.ky,l.sx,l.sy,weights=raw,bias=b)
                    shape=(oc,oh,ow);input_shape=(ic,ph,pw)
                    geo=(pw,ph,l.kx,l.ky,l.sx,l.sy)
                else:
                    raw=np.zeros((oc,pc),dtype=np.int64);raw[:,:ic]=weights[co:co+oc,ci:ci+ic]
                    physical=FCLayer(node.name,pc,oc,raw,b);shape=(oc,);input_shape=(ic,);geo=(1,1,1,1,1,1)
                ww=pack_weights(physical).reshape(-1,4).astype(np.int64).tolist()
                bb=pack_bias(b,oc).reshape(-1,1).astype(np.int64).tolist()
                input_data=pack(np.zeros(input_shape,dtype=np.int64),source_dtype)
                outaddr=align_ceil(len(input_data),4);outwords=math.prod(padded(shape,'int32'))
                if outaddr+outwords>=0x3000 or len(ww)+len(bb)>0x4000:
                    raise CompilerError('Affine tile SRAM footprint exceeded',code='MEMORY',node=node.name)
                op=p41(geo[0],geo[1],pc,oc,*geo[2:],0,0x4000,0x4000+len(ww),outaddr,ina_type=TYPES[source_dtype],load_bias=2)
                op.memory_blocks=[initblock('weight',0x4000,ww),initblock('bias',0x4000+len(ww),bb)]
                attrs=dict(co=co,oc=oc,ci=ci,ic=ic,h=h,w=w,oh=oh,ow=ow,
                           input_shape=input_shape,bias_once=ci==0,source_shape=tuple(source_shape))
                tile=TileKernel(f'{node.name}:{co}:{h}:{w}:{ci}',node.name,'affine',shape,'int32',outaddr,[op],
                                [(0,source_dtype,input_shape)],attrs)
                tiles.append(tile.validate())
    return tiles

def sum_tile(name,shape,count):
    _,p02,*_=_imports();c=align_ceil(shape[0],32);h,w=shape[1:] if len(shape)==3 else (1,1)
    words=c*h*w
    if count*words>=0x2000 or words>0x1000:raise CompilerError('Reduction tile exceeds SRAM',code='MEMORY')
    op=p02(False,0,0,1,count,1,1,c,w,h,0,0,0x2000,constant_b=0)
    return TileKernel(name,name,'sum',shape,'int32',0x2000,[op],
                       [(i*words,'int32',shape) for i in range(count)]).validate()

def point_tile(node,start,length,source_dtype):
    _,_,_,pX5,_,p06=_imports();count=align_ceil(length,128 if source_dtype=='int2' else 32)
    dtype=node.dtype
    if node.op in ('Concat','Cast'):
        op=p06(0,0x1000,0x2ff0,count,1,1,1,count,1,TYPES[source_dtype],TYPES[dtype])
    else:
        op=pX5('max',0,0x1000,count,count,1,1,1,1,1,1,
                0 if node.op=='Relu' or node.attrs.get('relu',False) else 0x80808080,
                type_in=TYPES[source_dtype],type_out=1,in_cut_start=node.attrs.get('cut',0))
    return TileKernel(f'{node.name}:{start}',node.name,'point',(length,),dtype,0x1000,[op],
                       [(0,source_dtype,(length,))],{'start':start,'length':length}).validate()

def pool_tiles(node,source_shape,spatial=4):
    _,_,_,pX5,*_=_imports();c,h,w=node.shape;a=node.attrs;tiles=[]
    for co in range(0,c,32):
        oc=min(32,c-co);pc=align_ceil(oc,32)
        for row in range(0,h,spatial):
            for col in range(0,w,spatial):
                oh,ow=min(spatial,h-row),min(spatial,w-col)
                ph,pw=(oh-1)*a['sy']+a['ky'],(ow-1)*a['sx']+a['kx']
                op=pX5('max',0,0x2000,pc,pc,pw,ph,a['kx'],a['ky'],a['sx'],a['sy'],0x80808080,type_in=1,type_out=1)
                tiles.append(TileKernel(f'{node.name}:{co}:{row}:{col}',node.name,'pool',(oc,oh,ow),'int8',0x2000,[op],
                    [(0,'int8',(oc,ph,pw))],dict(co=co,oc=oc,h=row,w=col,oh=oh,ow=ow,ph=ph,pw=pw)).validate())
    return tiles

def lif_tile(node,start,length,core):
    _,_,p08,_,p83,_=_imports();n=node.neuron
    count=align_ceil(length,128 if node.dtype=='int2' else 32)
    def make(threshold,alpha=256,reset=3,fire=4,offset=0):
        op=p08(count,1,0,0,threshold,alpha!=256,min(alpha,255),offset,0,
               0,0x3000,0x1000,0x3fe0,0x3fe4,0x3fe8,0,0,
               VL=-(1<<27),Tw_en=False,Tw_len=1,Rst_mode=2 if reset==3 else reset,fire_type=fire,dV=0)
        return op
    if isinstance(n,P08Contract):
        op=make(n.threshold,n.leak_alpha if n.leak_enabled else 256,n.reset_mode,n.fire_type,n.leak_beta)
        op.VR=n.reset_value;op.VL=n.floor;op.dV=n.reset_decrement;op.Ref_len=n.refractory;op.in_cut_start=n.readout_cut
        op.Tw_len=0 if n.window_initial>=n.window_length else 1
        op.Vinit=int(n.initial_v) if np.ndim(n.initial_v)==0 else 0
        initial=np.asarray(n.initial_v,dtype=np.int64)
        initial=np.full(length,int(initial)) if initial.ndim==0 else initial.reshape(-1)[start:start+length]
        if initial.size!=length or np.any(initial<-(1<<27)) or np.any(initial>=(1<<27)):
            raise CompilerError('Per-neuron initial state shape/range mismatch',node=node.name)
        operations=[op]
    else:
        if n.profile=='p08':n=n.p08_spec();raise CompilerError('Use P08Contract for temporal p08',node=node.name)
        if n.profile=='sound':operations=[make(n.threshold,n.beta_numerator,0,4,n.leak_offset)]
        else:
            reset=0 if n.profile=='hnn' or n.reset_mode=='zero' else 3
            threshold=n.threshold if reset==0 else (1<<27)-1
            pre=make(threshold,n.beta_numerator,reset,1)
            pre.Addr_Uin_start=0x2800
            operations=[pre]
            if n.profile=='snntorch' and n.reset_mode=='subtract':
                compare=make(n.threshold,256,3,4)
                compare.Addr_Uin_start=0x2800;compare.Addr_S_Start=0x0800
                operations.insert(0,compare)
                scale=p83(False,1,count,1,1,1,1,0x0800,0,0x1800,-n.threshold,0,1,0,0)
                operations.append(scale)
                # State threshold test is recomputed from the old membrane,
                # matching snntorch (not assumed to equal prior output).
                red=sum_tile(node.name+'.reset_sum',(length,),2)
                operations.append(red.operations[0])
                operations[-1].Addr_InA_base=0x1800
                operations[-1].Addr_V_base=0
            operations.append(make(n.threshold,256,3,4))
        initial=np.zeros(length,dtype=np.int64)
    first=operations[0]
    iv=np.zeros(count,dtype=np.int64);iv[:length]=initial
    first.memory_blocks=[initblock('membrane',0x3000,((iv & ((1<<28)-1)).reshape(-1,1)).tolist()),
        initblock('VM',0x3fe0,[[0]]*4),initblock('Vtheta',0x3fe4,[[0]]*4),initblock('para',0x3fe8,[[0]]*4)]
    tile=TileKernel(f'{node.name}:{start}',node.name,'lif',(length,),node.dtype,0x1000,operations,
                    [(0,'int32',(length,))],{'start':start,'length':length,'neurons':count,
                    'software_window_counter':isinstance(n,P08Contract) and n.window_length>255,
                    'host_window_control_required':isinstance(n,P08Contract)},core)
    return tile.validate()
