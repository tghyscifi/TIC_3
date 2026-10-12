"""Temporal frontend tests use generated ONNX only; no simulator or training."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from onnx2prim.diagnostics import CompilerError
from onnx2prim.quantization import QuantizationConfig
from onnx2prim.temporal_ir import P08Contract
from onnx2prim.temporal_parser import parse_onnx_temporal


def node(op, inputs, output, name=None, **attrs):
    return h.make_node(op, inputs, [output], name=name or output,
                       domain='com.tianjic' if op in ('LIF', 'Cut', 'Cast') else '', **attrs)


def weight(name, shape):
    return nh.from_array(np.ones(shape, dtype=np.float32), name)


class TemporalParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='temporal_parser_test_', dir=Path(__file__).resolve().parents[2])
        self.path = Path(self.temp.name) / 'model.onnx'

    def tearDown(self):
        self.temp.cleanup()

    def model(self, nodes=None, parameters=None, input_shape=(1, 4), output_shape=(1, 3)):
        nodes = nodes if nodes is not None else [
            node('Gemm', ['x', 'w'], 'u', name='fc', transB=1),
            node('LIF', ['u'], 'y')]
        parameters = parameters if parameters is not None else [weight('w', (3, 4))]
        graph = h.make_graph(nodes, 'fixture',
            [h.make_tensor_value_info('x', onnx.TensorProto.FLOAT, input_shape)],
            [h.make_tensor_value_info(nodes[-1].output[0], onnx.TensorProto.FLOAT, output_shape)],
            parameters)
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 13), h.make_opsetid('com.tianjic', 1)])
        h.set_model_props(model, {'snn_profile': 'tianjic-temporal-v1', 'snn_dt': '1'})
        return model

    def parse(self, model, cuts=None):
        onnx.save(model, str(self.path))
        return parse_onnx_temporal(self.path, QuantizationConfig(cuts or {}))

    def rejected(self, model, cuts=None):
        with self.assertRaises(CompilerError):
            self.parse(model, cuts)

    def test_default_p08_and_raw_current(self):
        graph = self.parse(self.model())
        self.assertEqual([n.op for n in graph.ordered()], ['FC', 'LIF'])
        self.assertEqual(graph.nodes[0].dtype, 'int32')
        self.assertEqual(graph.nodes[1].neuron, P08Contract())
        self.assertEqual(graph.input_shape, (4,))

    def test_long_window_and_fire5(self):
        m = self.model()
        m.graph.node[1].attribute.extend([h.make_attribute('window_enabled', 1),
            h.make_attribute('window_length', 1000), h.make_attribute('fire_type', 5)])
        g = self.parse(m)
        self.assertEqual(g.nodes[1].dtype, 'int2')
        self.assertEqual(g.nodes[1].neuron.window_length, 1000)

    def test_metadata_readouts_and_legacy_profile(self):
        for readout in ('spikes', 'sum', 'mean', 'last'):
            m = self.model()
            h.set_model_props(m, {'snn_profile': 'tianjic-p08-v1', 'snn_dt': '.25', 'snn_readout': readout})
            g = self.parse(m)
            self.assertEqual((g.dt, g.readout), (.25, readout))

    def test_hybrid_conv_relu_lif_pool_flatten_fc(self):
        ns = [node('Conv', ['x', 'c1'], 'a', name='conv'),
              node('Relu', ['a'], 'r'), node('Conv', ['r', 'c2'], 'u'),
              node('LIF', ['u'], 's'), node('MaxPool', ['s'], 'p', kernel_shape=[2, 2], strides=[2, 2]),
              node('Flatten', ['p'], 'f', axis=1), node('Gemm', ['f', 'w'], 'z', transB=1),
              node('LIF', ['z'], 'y')]
        ps = [weight('c1', (2, 1, 1, 1)), weight('c2', (3, 2, 1, 1)), weight('w', (5, 12))]
        m = self.model(ns, ps, (1, 1, 4, 4), (1, 5))
        self.rejected(m)
        g = self.parse(m, {'conv': 3})
        self.assertEqual([n.op for n in g.nodes], ['Conv', 'Cut', 'Relu', 'Conv', 'LIF', 'MaxPool', 'FC', 'LIF'])
        fc = g.nodes[-2]
        self.assertEqual(fc.inputs, ('p',))
        self.assertEqual(fc.layer.feature_shape, (3, 2, 2))
        self.assertEqual(fc.layer.input_layout, 'CHW')

    def test_shd_four_forks_concat_add_lif(self):
        ns, ps = [], []
        for i in range(4):
            ns += [node('Gemm', ['x', 'w'+str(i)], 'u'+str(i), transB=1),
                   node('LIF', ['u'+str(i)], 's'+str(i))]
            ps.append(weight('w'+str(i), (3, 4)))
        ns += [node('Concat', ['s0', 's1', 's2', 's3'], 'cat', axis=1),
               node('Add', ['cat', 'cat'], 'sum'), node('LIF', ['sum'], 'y')]
        g = self.parse(self.model(ns, ps, output_shape=(1, 12)))
        self.assertEqual(g.nodes[-2].dtype, 'int32')
        self.assertEqual(g.nodes[-3].shape, (12,))

    def test_add_accepts_chw_int32_and_mixed_dtypes(self):
        ns = [node('Conv', ['x', 'w'], 'u'), node('Add', ['u', 'x'], 'a'), node('LIF', ['a'], 'y')]
        g = self.parse(self.model(ns, [weight('w', (1, 1, 1, 1))], (1, 1, 2, 3), (1, 1, 2, 3)))
        self.assertEqual(g.nodes[1].shape, (1, 2, 3))
        self.assertEqual(g.nodes[1].dtype, 'int32')

    def test_explicit_cut_attributes(self):
        ns = [node('Gemm', ['x', 'w'], 'u', transB=1), node('Cut', ['u'], 'y', cut=4, relu=1)]
        g = self.parse(self.model(ns))
        self.assertEqual(g.nodes[-1].attrs, {'cut': 4, 'relu': True})
        self.assertEqual(g.nodes[-1].dtype, 'int8')

    def test_gemm_default_weight_orientation(self):
        ns = [node('Gemm', ['x', 'w'], 'u'), node('LIF', ['u'], 'y')]
        g = self.parse(self.model(ns, [weight('w', (4, 3))]))
        self.assertEqual(g.nodes[0].layer.weights.shape, (3, 4))

    def test_config_cut_on_add_for_downstream_fc(self):
        ns = [node('Add', ['x', 'x'], 'a', name='add'),
              node('Gemm', ['a', 'w'], 'u', transB=1), node('LIF', ['u'], 'y')]
        m = self.model(ns)
        self.rejected(m)
        g = self.parse(m, {'add': 2})
        self.assertEqual(g.nodes[2].inputs, ('a__cut',))
        self.assertEqual(g.nodes[0].dtype, 'int32')

    def test_source_profiles_preserve_contracts(self):
        for profile in ('sound', 'snntorch', 'hnn'):
            m = self.model()
            attrs = dict(profile=profile, threshold=17, beta_numerator=230, beta_denominator=256,
                         reset_mode='zero')
            if profile == 'snntorch':
                attrs.update(invocation_count=2, invocation_state='same_incoming_membrane')
            m.graph.node[1].attribute.extend(h.make_attribute(k, v) for k, v in attrs.items())
            g = self.parse(m)
            self.assertEqual(g.nodes[1].neuron.profile, profile)
            self.assertFalse(g.nodes[1].neuron.p08_mapping_supported)
            g.ordered()

    def test_reject_bad_metadata(self):
        for props in ({}, {'snn_profile': 'other', 'snn_dt': '1'},
                      {'snn_profile': 'tianjic-temporal-v1', 'snn_dt': 'nan'},
                      {'snn_profile': 'tianjic-temporal-v1', 'snn_dt': '0'},
                      {'snn_profile': 'tianjic-temporal-v1', 'snn_dt': '1', 'snn_readout': 'unknown'}):
            m = self.model(); h.set_model_props(m, props); self.rejected(m)

    def test_reject_shapes_dynamic_batch_and_mismatch(self):
        for shape in ((2, 4), (1, 'N'), (1, 0), (1, 1, 4)):
            self.rejected(self.model(input_shape=shape))
        self.rejected(self.model(output_shape=(1, 2)))
        m = self.model()
        m.graph.value_info.append(h.make_tensor_value_info('u', onnx.TensorProto.FLOAT, [1, 'C']))
        self.rejected(m)

    def test_reject_dead_nodes_parameters_and_aliases(self):
        m = self.model(); m.graph.node.insert(0, node('Relu', ['x'], 'dead')); self.rejected(m)
        m = self.model(); m.graph.initializer.append(weight('unused', (1,))); self.rejected(m)
        m = self.model(); m.graph.node.insert(0, node('Flatten', ['x'], 'dead')); self.rejected(m)

    def test_reject_unknown_domain_functions_and_attributes(self):
        m = self.model(); m.graph.node[1].domain = 'unknown'; self.rejected(m)
        m = self.model(); m.graph.node[1].attribute.append(h.make_attribute('adaptation', 1)); self.rejected(m)
        m = self.model(); m.graph.node[1].attribute.extend([h.make_attribute('threshold', 1)]*2); self.rejected(m)
        m = self.model(); m.functions.append(h.make_function('com.tianjic', 'Fn', ['a'], ['b'],
            [h.make_node('Identity', ['a'], ['b'])], [h.make_opsetid('', 13)])); self.rejected(m)

    def test_reject_neuron_fields_and_unknown_profiles(self):
        for key, val in [('leak_enabled', 2), ('threshold', 1.5), ('fire_type', 0),
                         ('window_length', 65536), ('profile', 'unknown')]:
            m = self.model(); m.graph.node[1].attribute.append(h.make_attribute(key, val)); self.rejected(m)

    def test_reject_broadcast_add_concat_axis_and_pool_modes(self):
        ns = [node('Gemm', ['x', 'w'], 'u', transB=1), node('Add', ['x', 'u'], 'y')]
        self.rejected(self.model(ns, output_shape=(1, 4)))
        for axis in (0, -1, 2):
            self.rejected(self.model([node('Concat', ['x', 'x'], 'y', axis=axis)], [], output_shape=(1, 8)))
        for attrs in ({'pads': [1, 1, 1, 1]}, {'ceil_mode': 1}, {'dilations': [2, 2]}):
            self.rejected(self.model([node('MaxPool', ['x'], 'y', kernel_shape=[2, 2], **attrs)], [],
                                    (1, 1, 4, 4), (1, 1, 2, 2)))

    def test_reject_unknown_cuts_and_cut_before_lif(self):
        for cuts in ({'missing': 1}, {'fc': 16}, {'fc': True}, {'y': 1}, {'fc': 1}):
            self.rejected(self.model(), cuts)

    def test_reject_invalid_parameters_and_affine_attrs(self):
        for params in ([weight('w', (3, 5))], [nh.from_array(np.full((3, 4), np.nan, np.float32), 'w')]):
            self.rejected(self.model(parameters=params))
        m = self.model(); m.graph.node[0].attribute.append(h.make_attribute('alpha', 2.0)); self.rejected(m)

    def test_int2_can_feed_affine_but_not_pool(self):
        ns = [node('Gemm', ['x', 'w'], 'u', transB=1), node('LIF', ['u'], 's', fire_type=5),
              node('Gemm', ['s', 'v'], 'z', transB=1), node('LIF', ['z'], 'y')]
        g = self.parse(self.model(ns, [weight('w', (3, 4)), weight('v', (3, 3))]))
        self.assertEqual(g.nodes[1].dtype, 'int2')
        ns = [node('Conv', ['x', 'w'], 'u'), node('LIF', ['u'], 's', fire_type=5),
              node('MaxPool', ['s'], 'y', kernel_shape=[2, 2])]
        self.rejected(self.model(ns, [weight('w', (1, 1, 1, 1))], (1, 1, 2, 2), (1, 1, 1, 1)))

    def test_flatten_provenance_can_fork_to_two_fc_nodes(self):
        ns = [node('Flatten', ['x'], 'f', axis=1),
              node('Gemm', ['f', 'a'], 'u', transB=1),
              node('Gemm', ['f', 'b'], 'v', transB=1),
              node('Add', ['u', 'v'], 'z'), node('LIF', ['z'], 'y')]
        g = self.parse(self.model(ns, [weight('a', (3, 12)), weight('b', (3, 12))],
                                 (1, 2, 2, 3), (1, 3)))
        for fc in g.nodes[:2]:
            self.assertEqual(fc.inputs, ('x',))
            self.assertEqual(fc.layer.feature_shape, (2, 2, 3))

    def test_reject_flatten_axis_output_and_spatial_consumers(self):
        self.rejected(self.model([node('Flatten', ['x'], 'y', axis=0)], [], output_shape=(1, 4)))
        self.rejected(self.model([node('Flatten', ['x'], 'y', axis=1)], [], output_shape=(1, 4)))
        ns = [node('Flatten', ['x'], 'f'), node('Relu', ['f'], 'y')]
        self.rejected(self.model(ns, [], (1, 1, 2, 2), (1, 4)))

    def test_reject_conv_geometry_and_malformed_cut(self):
        for attrs in ({'group': 2}, {'dilations': [2, 1]}, {'strides': [0, 1]},
                      {'kernel_shape': [2, 2]}, {'auto_pad': 'SAME_UPPER'}):
            ns = [node('Conv', ['x', 'w'], 'u', **attrs), node('LIF', ['u'], 'y')]
            self.rejected(self.model(ns, [weight('w', (1, 1, 1, 1))],
                                    (1, 1, 2, 2), (1, 1, 2, 2)))
        for attrs in ({}, {'cut': -1}, {'cut': 16}, {'cut': 2.0}, {'cut': 1, 'relu': 2}):
            ns = [node('Gemm', ['x', 'w'], 'u', transB=1), node('Cut', ['u'], 'y', **attrs)]
            self.rejected(self.model(ns))

    def test_reject_undefined_cycle_duplicate_and_multiple_outputs(self):
        m = self.model(); m.graph.node[0].input[0] = 'missing'; self.rejected(m)
        m = self.model(); m.graph.node[0].input[0] = 'y'; self.rejected(m)
        m = self.model(); m.graph.node[1].name = 'fc'; self.rejected(m)
        m = self.model(); m.graph.node[1].output.append('extra'); self.rejected(m)

    def test_affine_terminal_is_int32_without_implicit_cut(self):
        g = self.parse(self.model([node('Gemm', ['x', 'w'], 'y', transB=1)]))
        self.assertEqual(len(g.nodes), 1)
        self.assertEqual(g.nodes[0].dtype, 'int32')
        g = self.parse(self.model([node('Add', ['x', 'x'], 'y')], [], output_shape=(1, 4)))
        self.assertEqual(g.nodes[0].dtype, 'int32')

    def test_int2_input_and_explicit_widening(self):
        m = self.model([node('Cast', ['x'], 'u'), node('LIF', ['u'], 'y')], [], output_shape=(1,4))
        h.set_model_props(m, {'snn_profile':'tianjic-temporal-v1','snn_dt':'1','snn_input_dtype':'int2'})
        g = self.parse(m)
        self.assertEqual(g.input_dtype, 'int2')
        self.assertEqual(g.nodes[0].dtype, 'int32')
        for dtype in ('int4','float'):
            h.set_model_props(m, {'snn_profile':'tianjic-temporal-v1','snn_dt':'1','snn_input_dtype':dtype})
            self.rejected(m)

    def test_per_neuron_initial_states_in_spatial_onnx(self):
        ns = [node('Conv',['x','w'],'u'), node('LIF',['u'],'y',initial_v=list(range(6)))]
        m = self.model(ns, [weight('w',(1,1,1,1))], (1,1,2,3), (1,1,2,3))
        g = self.parse(m)
        np.testing.assert_array_equal(g.nodes[1].neuron.initial_v, np.arange(6).reshape(1,2,3))
        m.graph.node[1].attribute[0].ints.pop()
        self.rejected(m)


if __name__ == '__main__':
    unittest.main()
