import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnx2prim.onnx_parser import parse_onnx, FCLayer
from onnx2prim.diagnostics import UnsupportedModelError


class ParserTests(unittest.TestCase):
    def parse(self, nodes, shape, output, weights=()):
        graph=h.make_graph(nodes,'synthetic',[h.make_tensor_value_info('x',T.FLOAT,shape)],
                           [h.make_tensor_value_info('y',T.FLOAT,output)],list(weights))
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',13)])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'model.onnx'
            onnx.save(model,path)
            return parse_onnx(path)

    def test_nonsquare_conv_pool(self):
        seq=self.parse([h.make_node('Conv',['x','w'],['c'],strides=[2,1]),
                        h.make_node('Relu',['c'],['r']),
                        h.make_node('MaxPool',['r'],['y'],kernel_shape=[2,3],strides=[1,2])],
                       [1,1,7,9],[1,2,2,3],
                       [nh.from_array(np.ones((2,1,3,2),np.float32),'w')])
        conv,relu=seq.layers[0]; pool,_=seq.layers[1]
        self.assertEqual((conv.px,conv.py,conv.kx,conv.ky,conv.sx,conv.sy),(9,7,2,3,1,2))
        self.assertTrue(relu)
        self.assertEqual((pool.px,pool.py),(3,2))

    def test_transpose_flatten_fc(self):
        w=np.arange(24,dtype=np.float32).reshape(2,12)
        seq=self.parse([h.make_node('Transpose',['x'],['t'],perm=[0,2,3,1]),
                        h.make_node('Flatten',['t'],['f']),
                        h.make_node('Gemm',['f','w','b'],['y'],transB=1,alpha=2.,beta=3.)],
                       [1,2,2,3],[1,2],[nh.from_array(w,'w'),nh.from_array(np.array([1,2],np.float32),'b')])
        fc=seq.layers[0][0]
        self.assertIsInstance(fc,FCLayer)
        self.assertEqual((fc.input_layout,fc.feature_shape),('HWC',(2,2,3)))
        np.testing.assert_array_equal(fc.weights,w*2)
        np.testing.assert_array_equal(fc.bias,[3,6])

    def test_matmul_add(self):
        w=np.arange(6,dtype=np.float32).reshape(3,2)
        seq=self.parse([h.make_node('MatMul',['x','w'],['f']),h.make_node('Add',['b','f'],['y'])],
                       [1,3],[1,2],[nh.from_array(w,'w'),nh.from_array(np.array([2,3],np.float32),'b')])
        np.testing.assert_array_equal(seq.layers[0][0].weights,w.T)
        np.testing.assert_array_equal(seq.layers[0][0].bias,[2,3])

    def test_constant_reshape(self):
        seq=self.parse([h.make_node('Constant',[],['s'],value=nh.from_array(np.array([1,-1],np.int64))),
                        h.make_node('Reshape',['x','s'],['f']),h.make_node('MatMul',['f','w'],['y'])],
                       [1,1,2,3],[1,2],[nh.from_array(np.ones((6,2),np.float32),'w')])
        self.assertEqual(seq.layers[0][0].feature_shape,(1,2,3))

    def test_rejections(self):
        cases=[([h.make_node('Sigmoid',['x'],['y'])],[1,3],[1,3],[]),
               ([h.make_node('Relu',['x'],['a']),h.make_node('Relu',['x'],['y'])],[1,3],[1,3],[]),
               ([h.make_node('MatMul',['x','w'],['y'])],[1,'dynamic'],[1,2],
                [nh.from_array(np.ones((3,2),np.float32),'w')]),
               ([h.make_node('MaxPool',['x'],['y'],kernel_shape=[2,2],ceil_mode=1)],[1,1,4,4],[1,1,2,2],[]),
               ([h.make_node('BatchNormalization',['x','s','b','m','v'],['y'])],[1,1,2,2],[1,1,2,2],
                [nh.from_array(np.ones(1,np.float32),n) for n in ['s','b','m','v']])]
        for nodes,shape,out,weights in cases:
            with self.subTest(op=nodes[0].op_type),self.assertRaises(UnsupportedModelError):
                self.parse(nodes,shape,out,weights)

    def test_custom_domain_and_transpose_position(self):
        for nodes in ([h.make_node('Relu',['x'],['y'],domain='custom')],
                      [h.make_node('Transpose',['x'],['t'],perm=[0,2,3,1]),
                       h.make_node('Relu',['t'],['y'])]):
            with self.assertRaises(UnsupportedModelError):
                self.parse(nodes,[1,1,2,3],[1,2,3,1])

    def test_gemm_default_transpose(self):
        w=np.arange(6,dtype=np.float32).reshape(3,2)
        seq=self.parse([h.make_node('Gemm',['x','w'],['y'])],[1,3],[1,2],
                       [nh.from_array(w,'w')])
        np.testing.assert_array_equal(seq.layers[0][0].weights,w.T)
        self.assertEqual(seq.metadata,{})


if __name__=='__main__':
    unittest.main()
