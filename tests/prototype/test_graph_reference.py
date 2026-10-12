import unittest
import numpy as np
from onnx2prim.ir import GraphIR, GraphNode, ConvLayer
from onnx2prim.diagnostics import CompilerError
from onnx2prim.graph_reference import GraphQuantizationConfig, run_graph_reference, make_tanh_table


class GraphReferenceTests(unittest.TestCase):
    def residual(self):
        shape = (1, 1, 3)
        layer = ConvLayer('different_layer_name', 1, 1, 3, 1, 1, 1,
                          weights=np.array([[[[2]]]]), bias=np.array([0]))
        return GraphIR('x', shape, 'y', [
            GraphNode('conv', 'Conv', ('x',), 'c', shape, layer=layer),
            GraphNode('add', 'Add', ('c', 'x'), 'y', shape)])

    def test_residual_cuts_floor_and_saturation(self):
        graph = self.residual()
        result = run_graph_reference(graph, GraphQuantizationConfig({'conv': 0, 'add': 1}), [[[-3, 100, -100]]])
        np.testing.assert_array_equal(result['c'], [[[-6, 127, -128]]])
        np.testing.assert_array_equal(result['y'], [[[-3, 56, -57]]])
        self.assertEqual(set(result), {'x', 'c', 'y'})
        result = run_graph_reference(graph, GraphQuantizationConfig({'conv': 1, 'add': 0}), [[[-3, 100, -100]]])
        np.testing.assert_array_equal(result['c'], [[[-2, 50, -50]]])
        np.testing.assert_array_equal(result['y'], [[[-5, 127, -128]]])

    def test_scaled_parameters_by_node_name(self):
        graph = self.residual()
        graph.nodes[0].layer.weights = np.array([[[[0.5]]]])
        graph.nodes[0].layer.bias = np.array([0.25])
        cfg = GraphQuantizationConfig({'conv': 0, 'add': 0}, 'scaled',
                                     {'conv': 4}, {'conv': 8}, activation_scales={'x': 1, 'c': 1, 'y': 1})
        np.testing.assert_array_equal(run_graph_reference(graph, cfg, [[[1, 2, 3]]])['c'], [[[4, 6, 8]]])
        self.assertEqual(cfg.parameter_config(graph.nodes[0]).weight_scale, 4)

    def test_configuration_rejections(self):
        graph = self.residual()
        configs = [GraphQuantizationConfig(c) for c in ({}, {'conv': 0}, {'conv': 0, 'add': 0, 'extra': 0},
                   {'conv': -1, 'add': 0}, {'conv': 16, 'add': 0}, {'conv': True, 'add': 0})]
        configs += [GraphQuantizationConfig({'conv': 0, 'add': 0}, 'scaled'),
                    GraphQuantizationConfig({'conv': 0, 'add': 0}, weight_scales={'conv': 1}),
                    GraphQuantizationConfig({'conv': 0, 'add': 0}, activation_scales={'x': 1}),
                    GraphQuantizationConfig({'conv': 0, 'add': 0}, activation_scales={'x': 1, 'c': 2, 'y': 1})]
        for cfg in configs:
            with self.subTest(cfg=cfg), self.assertRaises(CompilerError):
                cfg.validate(graph)
        for bad in (0, -1, np.inf, np.nan):
            with self.subTest(scale=bad), self.assertRaises(CompilerError):
                GraphQuantizationConfig({'conv': 0, 'add': 0}, activation_scales={'x': bad, 'c': bad, 'y': 1}).validate(graph)
            with self.assertRaises(CompilerError):
                GraphQuantizationConfig({'conv': 0, 'add': 0}, 'scaled', {'conv': bad}, {'conv': 1},
                                        activation_scales={'x': 1, 'c': 1, 'y': 1}).validate(graph)

    def test_tanh_formula_and_unsigned_table(self):
        shape = (1, 1, 256)
        nodes = [GraphNode('t1', 'Tanh', ('x',), 'a', shape), GraphNode('t2', 'Tanh', ('a',), 'b', shape)]
        graph = GraphIR('x', shape, 'b', nodes)
        cfg = GraphQuantizationConfig({}, tanh_input_scales={'t1': 32, 't2': 64},
                                     tanh_output_scales={'t1': 150, 't2': 40})
        x = np.concatenate((np.arange(128), np.arange(-128, 0))).reshape(shape)
        result = run_graph_reference(graph, cfg, x)
        expected = np.floor(np.clip(np.tanh(x / 32) * 150, -128, 127))
        np.testing.assert_array_equal(result['a'], expected)
        np.testing.assert_array_equal(result['b'], np.floor(np.clip(np.tanh(expected / 64)*40, -128, 127)))
        table = make_tanh_table(cfg, nodes[0])
        self.assertEqual(table.shape, (256,))
        self.assertEqual(table.dtype, np.int8)
        np.testing.assert_array_equal(table, expected.reshape(-1))
        for scales in ({}, {'t1': 1, 't2': np.nan}, {'t1': 1, 't2': 1, 'extra': 1}):
            with self.assertRaises(CompilerError):
                GraphQuantizationConfig({}, tanh_input_scales=scales, tanh_output_scales=cfg.tanh_output_scales).validate(graph)

    def test_relu_pool_negative_padding(self):
        graph = GraphIR('x', (1, 1, 1), 'r', [
            GraphNode('pool', 'MaxPool', ('x',), 'p', (1, 1, 2),
                      dict(kx=2, ky=1, pad_left=1, pad_right=1)),
            GraphNode('relu', 'Relu', ('p',), 'r', (1, 1, 2))])
        result = run_graph_reference(graph, GraphQuantizationConfig({}), [[[-9]]])
        np.testing.assert_array_equal(result['p'], [[[-9, -9]]])
        np.testing.assert_array_equal(result['r'], [[[0, 0]]])

    def test_parameter_input_range_and_overflow(self):
        graph = self.residual()
        cfg = GraphQuantizationConfig({'conv': 0, 'add': 0})
        for bad in (np.nan, 128, 0.5):
            with self.assertRaises(CompilerError):
                run_graph_reference(graph, cfg, [[[bad, 0, 0]]])
        for bad in (np.nan, 128, 0.5):
            graph.nodes[0].layer.weights = np.array([[[[bad]]]])
            with self.assertRaises(CompilerError):
                run_graph_reference(graph, cfg, [[[1, 1, 1]]])
        graph.nodes[0].layer.weights = np.array([[[[1]]]])
        graph.nodes[0].layer.bias = np.array([2**31-1])
        with self.assertRaises(CompilerError) as error:
            run_graph_reference(graph, cfg, [[[1, 1, 1]]])
        self.assertEqual(error.exception.code, 'OVERFLOW')


if __name__ == '__main__':
    unittest.main()
