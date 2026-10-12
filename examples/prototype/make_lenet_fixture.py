"""Deterministic synthetic LeNet topology, NOT a trained accuracy benchmark."""
from pathlib import Path
import argparse
import json
import numpy as np

def make_fixture(directory, seed=7):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    initializers, nodes = [], []
    def affine(name, input_name, shape, op, **attrs):
        w = rng.integers(-8,9,size=shape).astype(np.float32)
        b = rng.integers(-256,257,size=shape[0]).astype(np.float32)
        initializers.extend([nh.from_array(w,name+'.weight'),nh.from_array(b,name+'.bias')])
        nodes.append(h.make_node(op,[input_name,name+'.weight',name+'.bias'],[name],name=name,**attrs))
        return name
    def relu_pool(name, input_name):
        nodes.append(h.make_node('Relu',[input_name],[name+'relu'],name=name+'relu'))
        nodes.append(h.make_node('MaxPool',[name+'relu'],[name+'pool'],name=name+'pool',kernel_shape=[2,2],strides=[2,2]))
        return name+'pool'
    x = affine('conv1','input',(6,1,5,5),'Conv')
    x = relu_pool('c1',x)
    x = affine('conv2',x,(16,6,5,5),'Conv',pads=[1,1,1,1])
    x = relu_pool('c2',x)
    nodes.append(h.make_node('Transpose',[x],['hwc'],name='to_hwc',perm=[0,2,3,1]))
    nodes.append(h.make_node('Flatten',['hwc'],['flat'],name='flatten',axis=1))
    x = affine('fc1','flat',(120,400),'Gemm',transB=1)
    nodes.append(h.make_node('Relu',[x],['r1'],name='r1'))
    x = affine('fc2','r1',(84,120),'Gemm',transB=1)
    nodes.append(h.make_node('Relu',[x],['r2'],name='r2'))
    affine('fc3','r2',(10,84),'Gemm',transB=1)
    graph = h.make_graph(nodes,'prototype_lenet',
        [h.make_tensor_value_info('input',T.FLOAT,[1,1,28,28])],
        [h.make_tensor_value_info('fc3',T.FLOAT,[1,10])], initializers)
    model = h.make_model(graph,opset_imports=[h.make_opsetid('',11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model,str(directory/'lenet.onnx'))
    np.save(directory/'input.npy',rng.integers(0,128,size=(1,28,28),dtype=np.int64))
    config = {'cuts':{'cut1':4,'cut2':4,'fc_cut1':5,'fc_cut2':4,'fc_cut3':5},'parameter_mode':'integer'}
    (directory/'quantization.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    return directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',help='A NEW directory, never overwrite old fixtures')
    args = parser.parse_args()
    print(make_fixture(args.directory).resolve())
