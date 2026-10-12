"""Portable temporal artifact integration tests; no TestEngine invocation."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import shutil
import tempfile
import unittest

import numpy as np

from onnx2prim.diagnostics import CompilerError
from onnx2prim.ir import ConvLayer, FCLayer
from onnx2prim.quantization import QuantizationConfig
from onnx2prim.temporal_compiler import compile_temporal, verify_temporal_behavior
from onnx2prim.temporal_export import export_temporal_artifacts, load_temporal_graph, save_temporal_graph
from onnx2prim.temporal_ir import P08Contract, TemporalGraph, TemporalNode
from onnx2prim.temporal_neurons import NeuronContract


class TemporalExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='temporal_export_test_',
            dir=Path(__file__).resolve().parents[2])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def graph(self, width=4, neuron=None, name='fc', readout='spikes'):
        layer = FCLayer(name, width, 3, np.ones((3, width), dtype=np.float32),
                        np.array([1, 2, 3], dtype=np.float32))
        return TemporalGraph('x', (width,), 'y', [
            TemporalNode(name, 'FC', ('x',), 'u', (3,), 'int32', layer=layer),
            TemporalNode('lif', 'LIF', ('u',), 'y', (3,), neuron=neuron or P08Contract(threshold=3))],
            readout=readout, source_path='missing-original-model.onnx',
            metadata={'snn_profile': 'tianjic-temporal-v1'})

    def compilation(self, graph=None):
        graph = graph or self.graph()
        frames = np.ones((3,) + graph.input_shape, dtype=np.int64)
        return compile_temporal(graph, QuantizationConfig({}), frames)

    def manifest(self, path):
        return json.loads((path / 'manifest.json').read_text(encoding='utf-8'))

    def traces(self, path, manifest, label):
        spec = manifest['output_archives'][label]
        with np.load(path / spec['file'], allow_pickle=False) as archive:
            return {name: archive[key].copy() for key, name in spec['keys'].items()}

    def test_tiny_fc_lif_portable_roundtrip(self):
        original = self.compilation()
        verify_temporal_behavior(original)
        out = export_temporal_artifacts(original, self.root / 'export')
        moved = self.root / 'moved'
        shutil.move(str(out), moved)
        graph = load_temporal_graph(moved / 'model.json', moved / 'weights.npz')
        self.assertEqual(graph.source_path, '')
        np.testing.assert_array_equal(graph.nodes[0].layer.bias, [1, 2, 3])
        namespace = {'__file__': str(moved / 'mapping.py')}
        exec(compile((moved / 'mapping.py').read_text(encoding='utf-8'), '<mapping>', 'exec'), namespace)
        rebuilt = namespace['compilation']
        verify_temporal_behavior(rebuilt)
        manifest = self.manifest(moved)
        self.assertTrue(namespace['host_staging_required'])
        self.assertEqual(len(namespace['stage_map_configs']), len(manifest['stage_index']))
        for label in ('reference_outputs', 'behavior_outputs'):
            saved = self.traces(moved, manifest, label)
            for key, expected in getattr(original, label).items():
                np.testing.assert_array_equal(saved[key], expected)
                np.testing.assert_array_equal(getattr(rebuilt, label)[key], expected)
        self.assertEqual(manifest['primitive_behavior_verification'], 'passed')

    def test_stage_configs_and_state_boundaries(self):
        c = self.compilation()
        out = export_temporal_artifacts(c, self.root / 'export')
        m = self.manifest(out)
        self.assertTrue(m['host_staging_required'])
        self.assertNotIn('behavior_outputs', m['output_archives'])
        self.assertFalse((out / 'behavior_outputs.npz').exists())
        for i, stage in enumerate(m['stage_index']):
            self.assertEqual(stage['id'], f'stage{i:06d}')
            self.assertFalse(stage['standalone_meaningful_execution'])
            self.assertEqual(stage['transfer']['dynamic_input_initialization'], 'zero')
            self.assertTrue(stage['transfer']['host_load_required'])
            hardware = json.loads((out / stage['hardware_config']).read_text(encoding='utf-8'))
            self.assertIn('ChipArray', hardware)
            self.assertEqual(stage['transfer']['source_shapes'], [[4]] if stage['node'] == 'fc' else [[3]])
            if stage['kind'] == 'lif':
                self.assertEqual(stage['state_core'], stage['core'])
                self.assertEqual(stage['initial_memory_load_boundary'], 'sample_start')
                self.assertTrue(stage['host_window_control_required'])
            else:
                self.assertEqual(stage['initial_memory_load_boundary'], 'per_stage')
                self.assertTrue(stage['transfer']['bias_once'])

    def test_fresh_directory_only(self):
        c = self.compilation()
        out = export_temporal_artifacts(c, self.root / 'export')
        before = (out / 'manifest.json').read_bytes()
        with self.assertRaises(CompilerError):
            export_temporal_artifacts(c, out)
        self.assertEqual((out / 'manifest.json').read_bytes(), before)

    def test_untrusted_names_are_only_data(self):
        name = "../../escape'; raise RuntimeError('injected') #"
        c = self.compilation(self.graph(name=name))
        # Trace names can also contain path separators; archive members use safe IDs.
        c.reference_outputs['../../trace'] = np.zeros(3, dtype=np.int64)
        out = export_temporal_artifacts(c, self.root / 'export')
        source = (out / 'mapping.py').read_text(encoding='utf-8')
        self.assertNotIn(name, source)
        with np.load(out / 'weights.npz', allow_pickle=False) as weights:
            self.assertEqual(weights.files, ['p000000', 'p000001'])
        with np.load(out / 'reference_outputs.npz', allow_pickle=False) as traces:
            self.assertTrue(all(k.startswith('a') and '/' not in k for k in traces.files))
        self.assertEqual(load_temporal_graph(out / 'model.json', out / 'weights.npz').nodes[0].name, name)
        self.assertEqual(sorted(p.name for p in (out / 'stage_configs').iterdir()),
                         ['stage000000.json', 'stage000001.json'])

    def test_partial_reductions_have_linked_stages(self):
        c = self.compilation(self.graph(width=600))
        self.assertTrue(c.reductions)
        out = export_temporal_artifacts(c, self.root / 'export')
        m = self.manifest(out)
        reductions = [s for s in m['stage_index'] if s['transfer']['reduction_group'] is not None]
        self.assertEqual(len(reductions), len(c.reductions))
        lookup = {s['id']: s for s in m['stage_index']}
        for reduction in reductions:
            members = [lookup[k] for k in reduction['transfer']['partial_source_stages']]
            self.assertEqual(sum(s['transfer']['bias_once'] for s in members), 1)
            self.assertEqual(len(members), len(reduction['transfer']['inputs']))
            self.assertEqual(reduction['transfer']['source_shapes'], [s['shape'] for s in members])
            self.assertTrue((out / reduction['hardware_config']).is_file())

    def test_all_p08_windows_require_host_control(self):
        for length in (1, 16, 1000):
            c = self.compilation(self.graph(neuron=P08Contract(threshold=3, window_enabled=True,
                                                            window_length=length, window_initial=0)))
            out = export_temporal_artifacts(c, self.root / f'export{length}')
            m = self.manifest(out)
            self.assertTrue(m['host_window_control_required'])
            lif = m['nodes'][1]
            self.assertTrue(lif['host_window_control_required'])
            self.assertEqual(lif['software_extended_window'], length > 255)
            g = load_temporal_graph(out / 'model.json', out / 'weights.npz')
            self.assertEqual(g.nodes[1].neuron.window_length, length)

    def test_source_profiles_snapshot_and_last_readout(self):
        for profile in ('sound', 'hnn', 'snntorch'):
            g = self.graph(neuron=NeuronContract(profile=profile, threshold=3), readout='last')
            c = self.compilation(g)
            out = export_temporal_artifacts(c, self.root / profile)
            restored = load_temporal_graph(out / 'model.json', out / 'weights.npz')
            self.assertEqual(restored.nodes[1].neuron, g.nodes[1].neuron)
            self.assertEqual(restored.readout, 'last')

    def test_conv_geometry_and_chw_feature_provenance_snapshot(self):
        l = ConvLayer('conv', 1, 2, 3, 2, 1, 1, weights=np.ones((2, 1, 1, 1)), bias=np.zeros(2))
        f = FCLayer('fc', 12, 3, np.ones((3, 12)), np.zeros(3), 'CHW', (2, 2, 3))
        g = TemporalGraph('x', (1, 2, 3), 'y', [
            TemporalNode('conv', 'Conv', ('x',), 'u', (2, 2, 3), 'int32', layer=l),
            TemporalNode('cut', 'Cut', ('u',), 's', (2, 2, 3), attrs={'cut': 0}),
            TemporalNode('fc', 'FC', ('s',), 'v', (3,), 'int32', layer=f),
            TemporalNode('lif', 'LIF', ('v',), 'y', (3,), neuron=P08Contract())])
        c = self.compilation(g)
        out = export_temporal_artifacts(c, self.root / 'export')
        restored = load_temporal_graph(out / 'model.json', out / 'weights.npz')
        self.assertEqual(restored.nodes[0].layer.input_shape, (1, 2, 3))
        self.assertEqual(restored.nodes[2].layer.feature_shape, (2, 2, 3))

    def test_loader_rejects_unsafe_parameter_ids_and_unknown_types(self):
        out = export_temporal_artifacts(self.compilation(), self.root / 'export')
        original = json.loads((out / 'model.json').read_text(encoding='utf-8'))
        for key, value in (('weights', '../../escape'), ('type', 'eval')):
            data = json.loads(json.dumps(original))
            data['nodes'][0]['layer'][key] = value
            (out / 'model.json').write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(CompilerError):
                load_temporal_graph(out / 'model.json', out / 'weights.npz')

    def test_reject_object_arrays_before_creating_directory(self):
        c = self.compilation()
        c.reference_outputs['bad'] = np.array([object()], dtype=object)
        target = self.root / 'export'
        with self.assertRaises(CompilerError):
            export_temporal_artifacts(c, target)
        self.assertFalse(target.exists())

    def test_cast_snapshot_and_int2_input_dtype(self):
        g = TemporalGraph('x', (3,), 'y', [
            TemporalNode('cast', 'Cast', ('x',), 'u', (3,), 'int32'),
            TemporalNode('lif', 'LIF', ('u',), 'y', (3,), neuron=P08Contract(threshold=9))],
            input_dtype='int2', readout='last', metadata={'surrogate_threshold_ge': 10})
        c = self.compilation(g)
        out = export_temporal_artifacts(c, self.root / 'export')
        restored = load_temporal_graph(out / 'model.json', out / 'weights.npz')
        self.assertEqual(restored.input_dtype, 'int2')
        self.assertEqual(restored.nodes[0].op, 'Cast')
        self.assertEqual(restored.nodes[1].neuron.threshold, 9)
        self.assertEqual(restored.metadata['surrogate_threshold_ge'], 10)

    def test_zero_input_serialization_does_not_mutate_compilation(self):
        c = self.compilation()
        before = [[len(op.memory_blocks) for op in t.operations]
                  for tiles in c.kernels.values() for t in tiles]
        export_temporal_artifacts(c, self.root / 'export')
        after = [[len(op.memory_blocks) for op in t.operations]
                 for tiles in c.kernels.values() for t in tiles]
        self.assertEqual(before, after)

    def test_public_save_graph_fixture_roundtrip_and_fresh_directory(self):
        original = self.graph()
        out = save_temporal_graph(original, self.root / 'fixture')
        restored = load_temporal_graph(out / 'model.json', out / 'weights.npz')
        self.assertEqual(restored.nodes[1].neuron, original.nodes[1].neuron)
        np.testing.assert_array_equal(restored.nodes[0].layer.weights, original.nodes[0].layer.weights)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['model.json', 'weights.npz'])
        before = (out / 'model.json').read_bytes()
        with self.assertRaises(CompilerError):
            save_temporal_graph(original, out)
        self.assertEqual((out / 'model.json').read_bytes(), before)

    def test_export_suppresses_original_serializer_init_prints(self):
        c = self.compilation()
        output = io.StringIO()
        with redirect_stdout(output):
            export_temporal_artifacts(c, self.root / 'export')
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
