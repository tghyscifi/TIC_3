"""Tiny explicit p08 FC and spatial Conv->Flatten->FC fixtures; no pretrained claims."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
from onnx2prim.snn_ir import LIFSpec

def lif(name, source, output, **kwargs):
    attrs = asdict(LIFSpec(**kwargs))
    attrs = {k:int(v) if isinstance(v,bool) else v for k,v in attrs.items()}
    return h.make_node('LIF',[source],[output],name=name,domain='com.tianjic',**attrs)

def make_fixture(directory, spatial=False):
    path = Path(directory)
    if path.exists(): raise ValueError('Choose a new fixture directory')
    rng = np.random.default_rng(19)
    if spatial:
        shape = (2,3,5); outshape = (3,)
        weights1 = rng.integers(-2,3,(3,2,3,3)).astype(np.float32)
        weights2 = rng.integers(-2,3,(3,45)).astype(np.float32)
        nodes = [h.make_node('Conv',['input','w1','b1'],['u1'],name='conv',pads=[1,1,1,1]),
                 lif('lif1','u1','s1',threshold=5,leak_alpha=192,refractory=1,reset_mode=1),
                 h.make_node('Flatten',['s1'],['flat'],name='flatten',axis=1),
                 h.make_node('Gemm',['flat','w2','b2'],['u2'],name='fc2',transB=1),
                 lif('lif2','u2','output',threshold=2,leak_alpha=128,window_enabled=True,window_length=3)]
        frames = rng.integers(0,2,(6,)+shape,dtype=np.int8)
        encoding = 'binary-spikes'
    else:
        shape = (4,); outshape = (2,)
        weights1 = np.array([[1,2,-1,0],[-2,1,1,1],[0,0,2,-1]],dtype=np.float32)
        weights2 = np.array([[2,-1,1],[-1,2,1]],dtype=np.float32)
        nodes = [h.make_node('Gemm',['input','w1','b1'],['u1'],name='fc1',transB=1),
                 lif('lif1','u1','s1',threshold=3,leak_alpha=192,reset_mode=1,refractory=1),
                 h.make_node('Gemm',['s1','w2','b2'],['u2'],name='fc2',transB=1),
                 lif('lif2','u2','output',threshold=2,leak_alpha=128,window_enabled=True,window_length=3)]
        frames = np.array([[1,1,0,1],[1,0,1,1],[0,1,1,0],[1,1,1,1],[-1,1,0,1],[1,0,0,1]],dtype=np.int8)
        encoding = 'int8-current'
    inits = [nh.from_array(weights1,'w1'),nh.from_array(weights2,'w2'),
             nh.from_array(np.zeros(weights1.shape[0],dtype=np.float32),'b1'),
             nh.from_array(np.zeros(weights2.shape[0],dtype=np.float32),'b2')]
    graph = h.make_graph(nodes,'tiny_snn_spatial' if spatial else 'tiny_snn_fc',
                         [h.make_tensor_value_info('input',TP.FLOAT,[1]+list(shape))],
                         [h.make_tensor_value_info('output',TP.FLOAT,[1]+list(outshape))],inits)
    model = h.make_model(graph,opset_imports=[h.make_opsetid('',11),h.make_opsetid('com.tianjic',1)])
    h.set_model_props(model,{'snn_profile':'tianjic-p08-v1','snn_dt':'1.0','snn_encoding':encoding})
    onnx.checker.check_model(model)
    path.mkdir(parents=True,exist_ok=False)
    onnx.save(model,str(path/'model.onnx'))
    np.save(path/'input.npy',frames)
    (path/'quantization.json').write_text(json.dumps({'cuts':{},'parameter_mode':'integer'},indent=2),encoding='utf-8')
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--spatial',action='store_true')
    args = parser.parse_args()
    print(make_fixture(args.output,args.spatial))
