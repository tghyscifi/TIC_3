import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from onnx2prim.diagnostics import UnsupportedModelError
from onnx2prim.graph_parser import parse_onnx_graph


class GraphParserTests(unittest.TestCase):
    def parse(self, nodes, shape=(1, 1, 4, 5), output=None, weights=(), inputs=None, outputs=None):
        graph = h.make_graph(nodes, 'residual', inputs or [h.make_tensor_value_info('x', T.FLOAT, shape)],
                             outputs or [h.make_tensor_value_info('y', T.FLOAT, output or shape)], list(weights))
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 13)])
        root = Path(__file__).resolve().parents[2] / 'prototype_runs'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path = Path(tmp) / 'graph.onnx'
            onnx.save(model, path)
            return parse_onnx_graph(path)

    def test_residual_fork_join(self):
        weights = np.array([[[[0.25]]]], dtype=np.float32)
        bias = np.array([-0.5], dtype=np.float32)
        graph = self.parse([h.make_node('Conv', ['x', 'w', 'b'], ['c'], name='branch'),
                            h.make_node('Relu', ['c'], ['r']),
                            h.make_node('Add', ['x', 'r'], ['a']),
                            h.make_node('Tanh', ['a'], ['y'])],
                           weights=[nh.from_array(weights, 'w'), nh.from_array(bias, 'b')])
        self.assertEqual(graph.input_shape, (1, 4, 5))
        self.assertEqual([n.op for n in graph.topological_nodes()], ['Conv', 'Relu', 'Add', 'Tanh'])
        self.assertEqual(graph.nodes[2].inputs, ('x', 'r'))
        np.testing.assert_array_equal(graph.nodes[0].layer.weights, weights)
        np.testing.assert_array_equal(graph.nodes[0].layer.bias, bias)
        self.assertEqual(graph.nodes[0].layer.weights.dtype, np.float32)

    def test_nonsquare_padded_conv_pool(self):
        graph = self.parse([h.make_node('Conv', ['x', 'w'], ['c'], strides=[2, 1], pads=[1, 2, 0, 1]),
                            h.make_node('MaxPool', ['c'], ['y'], kernel_shape=[2, 3], strides=[1, 2], pads=[0, 1, 1, 0])],
                           shape=[1, 1, 7, 9], output=[1, 2, 3, 5],
                           weights=[nh.from_array(np.ones((2, 1, 3, 2), np.float32), 'w')])
        conv = graph.nodes[0].layer
        self.assertEqual((conv.px, conv.py, conv.kx, conv.ky, conv.sx, conv.sy), (9, 7, 2, 3, 1, 2))
        self.assertEqual(conv.output_shape, (2, 3, 11))
        self.assertEqual(graph.nodes[1].attrs, dict(kx=3, ky=2, sx=2, sy=1,
                                                  pad_top=0, pad_down=1, pad_left=1, pad_right=0))

    def test_constant_conv_parameter(self):
        graph = self.parse([h.make_node('Constant', [], ['w'], value=nh.from_array(np.ones((1, 1, 1, 1), np.float32))),
                            h.make_node('Conv', ['x', 'w'], ['y'])])
        self.assertEqual(len(graph.nodes), 1)

    def test_rejections(self):
        cases = [
            ([h.make_node('MaxPool', ['x'], ['p'], kernel_shape=[4, 5]), h.make_node('Add', ['x', 'p'], ['y'])], {}),
            ([h.make_node('Add', ['x', 'b'], ['y'])], {'weights': [nh.from_array(np.ones((1, 1, 4, 5), np.float32), 'b')]}),
            ([h.make_node('Relu', ['x'], ['y'])], {'shape': [1, 1, 'height', 5]}),
            ([h.make_node('Relu', ['x'], ['y'], domain='custom')], {}),
        ]
        cases += [([h.make_node(op, ['x'], ['y'])], {}) for op in ('Sigmoid', 'Floor', 'Clip', 'Div', 'Gemm', 'BatchNormalization')]
        cases += [
            ([h.make_node('Relu', ['x'], ['dead']), h.make_node('Tanh', ['x'], ['y'])], {}),
            ([h.make_node('Relu', ['x'], ['r'], name='same'), h.make_node('Tanh', ['r'], ['y'], name='same')], {}),
            ([h.make_node('MaxPool', ['x'], ['y'], kernel_shape=[2, 2], ceil_mode=1)], {'output': [1, 1, 2, 3]}),
            ([h.make_node('Relu', ['x'], ['y'])], {'inputs': [h.make_tensor_value_info(n, T.FLOAT, [1, 1, 4, 5]) for n in ('x', 'z')]}),
            ([h.make_node('Relu', ['x'], ['y'])], {'output': [1, 1, 4, 6]}),
            ([h.make_node('Relu', ['missing'], ['y'])], {}),
            ([h.make_node('Relu', ['x'], ['y'])], {'shape': [2, 1, 4, 5]}),
            ([h.make_node('Relu', ['x'], ['y'])], {'outputs': [h.make_tensor_value_info(n, T.FLOAT, [1, 1, 4, 5]) for n in ('x', 'y')]}),
        ]
        for nodes, kwargs in cases:
            with self.subTest(nodes=[n.op_type for n in nodes]), self.assertRaises(UnsupportedModelError):
                self.parse(nodes, **kwargs)

    def test_unsupported_conv_modes(self):
        for attrs in ({'group': 2}, {'dilations': [2, 1]}, {'auto_pad': 'SAME_UPPER'}):
            with self.subTest(attrs=attrs), self.assertRaises(UnsupportedModelError):
                self.parse([h.make_node('Conv', ['x', 'w'], ['y'], **attrs)],
                           weights=[nh.from_array(np.ones((1, 1, 1, 1), np.float32), 'w')])


if __name__ == '__main__':
    unittest.main()
