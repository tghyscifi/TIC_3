"""Bounded source-preset checks; no training, checkpoints or TestEngine."""
import builtins
import unittest
from unittest.mock import patch
import numpy as np

from onnx2prim.diagnostics import CompilerError, UnsupportedModelError
from onnx2prim.project_snn_models import (
    MODEL_IDS, FLOAT_QUANTIZATION_POLICY, build_project_graph,
)
from onnx2prim.temporal_ir import P08Contract
from onnx2prim.temporal_neurons import NeuronContract


def build(name, **kwargs):
    if name == 'snnlenet/train_lif_float':
        kwargs.setdefault('quantization_policy', FLOAT_QUANTIZATION_POLICY)
    return build_project_graph(name, **kwargs)


def affines(graph):
    return [n for n in graph.nodes if n.op in ('Conv', 'FC')]


def supplied(graph):
    """Recover API source layout from a non-SHD graph, without file writes."""
    result = {}
    for node in affines(graph):
        w = node.layer.weights.copy()
        if graph.metadata['parameter_layers'][node.name].get('column_conversion'):
            c, h, width = node.layer.feature_shape
            w = w.reshape(w.shape[0], c, h, width).transpose(0, 2, 3, 1).reshape(w.shape[0], -1)
        result[node.name] = {'weight': w.copy(), 'bias': node.layer.bias.copy()}
    return result


class ProjectSNNModelTests(unittest.TestCase):
    def test_all_eleven_ids_and_full_mapping_validate(self):
        self.assertEqual(set(MODEL_IDS), {'SoundSNN', 'snn', 'snn_mnist_s', 'full', 'snnlenet',
                                        'HNN', 'SNN_forward', 'shd', 'snnlenet/train',
                                        'snnlenet/train_lif_float', 'snnlenet/snnpredict'})
        for name in (*MODEL_IDS, 'full/mapping'):
            with self.subTest(name=name):
                graph = build(name)
                self.assertEqual(len(graph.ordered()), len(graph.nodes))
                self.assertTrue(graph.metadata['coverage']['full_size_compute_dag'])
                self.assertFalse(graph.metadata['checkpoint_loaded'])
                self.assertFalse(graph.metadata['lossless_float_conversion'])
                self.assertEqual(graph.metadata['coverage']['affine_count'], len(affines(graph)))

    def test_deterministic_integer_synthetic_parameters(self):
        for name in MODEL_IDS:
            with self.subTest(name=name):
                a, b = build(name, seed=17), build(name, seed=17)
                for left, right in zip(affines(a), affines(b)):
                    np.testing.assert_array_equal(left.layer.weights, right.layer.weights)
                    np.testing.assert_array_equal(left.layer.bias, right.layer.bias)
                    self.assertEqual(left.layer.weights.dtype.kind, 'i')
                    # Folded SHD duplicate columns may exceed the source -2..2 range.
                    if name != 'shd':
                        self.assertLessEqual(np.max(np.abs(left.layer.weights)), 2)
                self.assertEqual(a.metadata, b.metadata)
        a, b = build('SoundSNN', seed=1), build('SoundSNN', seed=2)
        self.assertFalse(np.array_equal(a.nodes[0].layer.weights, b.nodes[0].layer.weights))

    def test_sound_full_sizes_profile_and_mean(self):
        g = build('SoundSNN')
        self.assertEqual(g.input_shape, (784,))
        self.assertEqual([(n.layer.cin, n.layer.cout) for n in affines(g)],
                         [(784, 128), (128, 128), (128, 128), (128, 10)])
        self.assertEqual(g.readout, 'mean')
        for n in g.nodes:
            if n.op == 'LIF':
                self.assertIsInstance(n.neuron, NeuronContract)
                self.assertEqual((n.neuron.profile, n.neuron.beta_numerator, n.neuron.threshold), ('sound', 256, 64))
        self.assertIn('threshold=1.0', g.metadata['approximations'][0])

    def test_root_pooled_geometry_and_duplicate_calls(self):
        g = build('snn')
        self.assertEqual([(n.op, n.shape) for n in g.nodes], [
            ('Conv', (12, 24, 24)), ('LIF', (12, 24, 24)), ('MaxPool', (12, 12, 12)),
            ('Conv', (32, 8, 8)), ('LIF', (32, 8, 8)), ('MaxPool', (32, 4, 4)),
            ('FC', (10,)), ('LIF', (10,))])
        neurons = [n.neuron for n in g.nodes if n.op == 'LIF']
        self.assertEqual([n.invocation_count for n in neurons], [2, 2, 1])
        self.assertTrue(all(n.invocation_state == 'same_incoming_membrane' for n in neurons))
        self.assertTrue(all(n.beta_numerator == 230 and n.threshold == 64 for n in neurons))
        self.assertEqual(affines(g)[-1].layer.feature_shape, (32, 4, 4))

    def test_stride_presets_and_full_model_mapping_conflict(self):
        for name in ('snn_mnist_s', 'full', 'full/mapping'):
            g = build(name)
            self.assertEqual([n.shape for n in affines(g)], [(12, 14, 14), (32, 7, 7), (10,)])
            self.assertEqual(affines(g)[-1].layer.cin, 1568)
            self.assertEqual(affines(g)[-1].layer.feature_shape, (32, 7, 7))
            conv = affines(g)[0].layer
            self.assertEqual((conv.sx, conv.sy, conv.pad_top, conv.pad_left), (2, 2, 2, 2))
            for node in g.nodes:
                if node.op != 'LIF':
                    continue
                if name == 'full/mapping':
                    self.assertIsInstance(node.neuron, P08Contract)
                    self.assertEqual((node.neuron.threshold, node.neuron.floor, node.neuron.window_length), (1, -10, 0))
                    self.assertFalse(node.neuron.leak_enabled)
                else:
                    self.assertEqual(node.neuron.profile, 'snntorch')
                    self.assertEqual(node.neuron.invocation_count, 1)
            if name.startswith('full'):
                self.assertTrue(g.metadata['coverage']['source_model_vs_mapping_conflict'])

    def test_hybrid_lenet_geometry_cuts_and_neuron(self):
        g = build('snnlenet')
        self.assertEqual([n.shape for n in affines(g)], [(6, 24, 24), (16, 10, 10), (120,), (84,), (10,)])
        self.assertEqual([n.shape for n in g.nodes if n.op == 'MaxPool'], [(6, 12, 12), (16, 5, 5)])
        self.assertEqual([n.attrs['cut'] for n in g.nodes if n.op == 'Cut'], [4, 4, 5, 5])
        neuron = next(n.neuron for n in g.nodes if n.op == 'LIF')
        self.assertEqual((neuron.threshold, neuron.floor, neuron.window_length), (10, -10, 0))
        self.assertFalse(neuron.leak_enabled)
        self.assertEqual(affines(g)[2].layer.feature_shape, (16, 5, 5))

    def test_hwc_parameter_conversion_preserves_dense_dot(self):
        g = build('snnlenet/train', seed=9)
        params = supplied(g)
        weights = np.arange(120 * 400).reshape(120, 400) % 5 - 2
        params['fc1']['weight'] = weights.copy()
        rebuilt = build('snnlenet/train', parameters=params)
        fc = next(n.layer for n in affines(rebuilt) if n.name == 'fc1')
        chw = np.arange(400).reshape(16, 5, 5) % 7
        hwc = chw.transpose(1, 2, 0).reshape(-1)
        np.testing.assert_array_equal(fc.weights @ chw.reshape(-1), weights @ hwc)
        np.testing.assert_array_equal(params['fc1']['weight'], weights)
        self.assertEqual((fc.input_layout, fc.feature_shape), ('CHW', (16, 5, 5)))

    def test_training_threshold_and_cast_are_stateless(self):
        for name in ('snnlenet/train', 'snnlenet/train_lif_float'):
            g = build(name)
            self.assertEqual([n.attrs['cut'] for n in g.nodes if n.op == 'Cut'], [4, 4, 5, 4, 5])
            idx = next(i for i, n in enumerate(g.nodes) if n.op == 'LIF')
            self.assertEqual(g.nodes[idx - 1].op, 'Cast')
            self.assertEqual(g.nodes[idx - 1].dtype, 'int32')
            n = g.nodes[idx].neuron
            self.assertEqual((n.threshold, n.window_length, n.floor), (9, 0, -(1 << 27)))
            self.assertFalse(n.leak_enabled)
            x = np.arange(-128, 128)
            np.testing.assert_array_equal(x >= 10, x > n.threshold)

    def test_float_requires_explicit_approximation_policy(self):
        with self.assertRaisesRegex(UnsupportedModelError, 'explicit quantization_policy'):
            build_project_graph('snnlenet/train_lif_float')
        with self.assertRaises(UnsupportedModelError):
            build_project_graph('snnlenet/train_lif_float', quantization_policy='lossless')
        g = build('snnlenet/train_lif_float')
        self.assertEqual(g.metadata['quantization_policy'], FLOAT_QUANTIZATION_POLICY)
        self.assertIn('Float source has no Cut', g.metadata['approximations'][0])

    def test_snnpredict_is_analog_relu_not_binary_lif(self):
        g = build('snnlenet/snnpredict')
        self.assertFalse(any(n.op == 'LIF' for n in g.nodes))
        self.assertEqual(next(n for n in g.nodes if n.name == 'fc_lif2_relu').op, 'Relu')
        x = np.arange(-128, 128)
        v = np.maximum(x, -10)
        np.testing.assert_array_equal((v > 0) * v, np.maximum(x, 0))
        self.assertFalse(g.metadata['batch_norm'])
        self.assertTrue(g.metadata['source_default_batch_norm'])

    def test_beijing_sizes_ignored_bias_and_readout(self):
        for name, dims, readout in (
            ('HNN', [256, 512, 512, 10], 'sum'),
            ('SNN_forward', [4, 128, 128, 128, 128, 2], 'last'),
        ):
            g = build(name)
            self.assertEqual([(n.layer.cin, n.layer.cout) for n in affines(g)], list(zip(dims, dims[1:])))
            self.assertEqual(g.readout, readout)
            for n in g.nodes:
                if n.op == 'LIF':
                    self.assertEqual((n.neuron.profile, n.neuron.beta_numerator, n.neuron.threshold), ('hnn', 128, 64))
            params = supplied(g)
            for entry in params.values():
                entry['bias'][:] = 37
            rebuilt = build(name, parameters=params)
            self.assertTrue(all(np.all(n.layer.bias == 0) for n in affines(rebuilt)))
            self.assertTrue(all(v['bias_ignored'] for v in rebuilt.metadata['parameter_layers'].values()))

    def test_shd_full_dendrite_states_and_unverified_selectors(self):
        g = build('shd', seed=6)
        self.assertEqual(g.input_shape, (700,))
        self.assertEqual(g.input_dtype, 'int2')
        self.assertEqual([(n.layer.cin, n.layer.cout) for n in affines(g)], [(700, 64)] * 4 + [(64, 32)])
        neurons = [n for n in g.nodes if n.op == 'LIF']
        self.assertEqual([n.neuron.fire_type for n in neurons], [2, 2, 2, 2, 5, 2])
        self.assertEqual([n.neuron.readout_cut for n in neurons], [2, 2, 2, 2, 6, 11])
        self.assertTrue(all(n.neuron.window_enabled and n.neuron.window_length == 1000 for n in neurons))
        self.assertTrue(all(n.neuron.floor == 0 for n in neurons))
        self.assertEqual(neurons[-2].dtype, 'int2')
        self.assertEqual(neurons[-1].neuron.reset_value, 1000)
        self.assertEqual(neurons[-1].neuron.threshold, 60000)
        self.assertEqual(neurons[-1].neuron.initial_v.shape, (32,))
        self.assertTrue(np.all(neurons[-1].neuron.initial_v <= 0x9999))
        self.assertEqual(g.metadata['legacy_router_equivalence'], 'not_verified')
        self.assertEqual(g.metadata['selector_provenance'], 'declared_synthetic_selectors')
        self.assertEqual(np.asarray(g.metadata['selector_indices']).shape, (4, 2, 192))
        self.assertFalse(any(n.op == 'Slice' for n in g.nodes))

    def test_shd_folded_weights_duplicate_padding_and_core_lookup(self):
        selectors = np.full((4, 2, 192), -1, dtype=np.int64)
        selectors[:, 0, :3] = [3, 3, 10]
        selectors[:, 1, :3] = [99, 20, 99]
        params = {'shd_selectors': selectors.copy()}
        native = np.zeros((64, 192), dtype=np.int64)
        native[:, :3] = [1, -2, 2]
        for i in range(4):
            params[f'branch{i + 1}'] = {'weight': native.copy(), 'bias': np.zeros(64, dtype=np.int64)}
        params['fc_out'] = {'weight': np.zeros((32, 64), dtype=np.int64), 'bias': np.zeros(32, dtype=np.int64)}
        g = build('shd', parameters=params)
        x = np.arange(700) % 3 - 1
        for branch in affines(g)[:4]:
            for core in range(2):
                indices = selectors[0, core]
                lookup = np.zeros(192, dtype=np.int64)
                keep = indices >= 0
                lookup[keep] = x[indices[keep]]
                rows = slice(core * 32, (core + 1) * 32)
                np.testing.assert_array_equal(branch.layer.weights[rows] @ x, native[rows] @ lookup)
        self.assertEqual(g.metadata['legacy_router_equivalence'], 'not_verified')
        with self.assertRaisesRegex(UnsupportedModelError, 'explicit shd_selectors'):
            build('shd', parameters={k: v for k, v in params.items() if k != 'shd_selectors'})
        params['shd_selectors'][0, 0, 0] = 700
        with self.assertRaises(CompilerError):
            build('shd', parameters=params)

    def test_parameters_shape_range_dtype_completeness_and_copy(self):
        g = build('SoundSNN')
        params = supplied(g)
        rebuilt = build('SoundSNN', parameters=params)
        params['fc1']['weight'][0, 0] = 127
        self.assertNotEqual(rebuilt.nodes[0].layer.weights[0, 0], 127)
        for mutation in ('shape', 'float', 'range', 'missing', 'extra', 'empty'):
            p = supplied(g)
            if mutation == 'shape':
                p['fc1']['weight'] = np.zeros((128, 16), dtype=np.int64)
            elif mutation == 'float':
                p['fc1']['weight'] = p['fc1']['weight'].astype(float)
            elif mutation == 'range':
                p['fc1']['weight'][0, 0] = 128
            elif mutation == 'missing':
                del p['fc4']
            elif mutation == 'extra':
                p['invented_layer'] = p['fc4']
            else:
                p = {}
            with self.subTest(mutation=mutation), self.assertRaises(CompilerError):
                build('SoundSNN', parameters=p)

    def test_batch_norm_checkpoint_parameters_rejected(self):
        for name in ('snnlenet', 'snnlenet/snnpredict'):
            p = supplied(build(name))
            p['bn1'] = {'weight': np.ones(6, dtype=np.int64), 'bias': np.zeros(6, dtype=np.int64)}
            with self.assertRaisesRegex(UnsupportedModelError, 'Unfused BatchNorm'):
                build(name, parameters=p)

    def test_no_project_or_training_imports(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('generator', 'torch', 'torchvision', 'snntorch', 'snn', 'snn_mnist_s'):
                raise AssertionError('Training/source import forbidden: ' + name)
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            for name in MODEL_IDS:
                build(name)

    def test_unknown_id_and_invalid_seed_rejected(self):
        with self.assertRaises(UnsupportedModelError):
            build('invented_three_layer_shd')
        for seed in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                build('snn', seed=seed)


if __name__ == '__main__':
    unittest.main()
