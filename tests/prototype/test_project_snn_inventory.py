"""Source-only inventory tests. No model, simulator or training imports."""
import ast
import builtins
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from onnx2prim.project_snn_inventory import MODEL_SOURCES, inspect_project


ROOT = Path(__file__).resolve().parents[2]


class ProjectSNNInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {r['model_id']: r for r in inspect_project(ROOT)}

    def test_all_actual_model_ids(self):
        self.assertEqual(set(self.models), {
            'shd', 'SoundSNN', 'snnlenet', 'snnlenet/snnpredict',
            'snnlenet/train', 'snnlenet/train_lif_float', 'full', 'snn',
            'snn_mnist_s', 'HNN', 'SNN_forward'})
        for record in self.models.values():
            self.assertTrue(record['source_paths'])
            self.assertTrue(all((ROOT / p).is_file() for p in record['source_paths']))
            self.assertTrue(record['topology'])
            self.assertTrue(record['semantics']['methods'] or record['semantics']['neuron_calls'])
            self.assertTrue(record['requirements'])
        json.dumps(list(self.models.values()))

    def test_inspection_never_imports_or_executes_sources(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'generator', 'torch', 'snntorch', 'torchvision', 'snn', 'snn_mnist_s'}:
                raise AssertionError('unsafe project/dependency import: ' + name)
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded), patch('builtins.exec', side_effect=AssertionError('exec forbidden')):
            self.assertEqual(inspect_project(ROOT), list(self.models.values()))

    def test_root_double_lif_is_same_explicit_state_not_chained(self):
        record = self.models['snn']
        self.assertEqual(record['semantics']['lif_calls_per_step'], {'lif1': 2, 'lif2': 2, 'lif3': 1})
        issue = next(e for e in record['source_errors'] if e['code'] == 'double_lif_call')
        self.assertIn('same explicit old mem', issue['message'])
        self.assertIn('mutate twice', issue['message'])
        self.assertIn('not two chained', issue['message'])
        self.assertEqual(record['dimensions']['spatial_chain'][-1]['observed_input'], 512)

    def test_current_no_pool_shape_is_observed_not_assumed_broken(self):
        record = self.models['snn_mnist_s']
        shapes = record['dimensions']['spatial_chain']
        self.assertEqual(shapes[0]['observed_output'], [12, 14, 14])
        self.assertEqual(shapes[1]['observed_output'], [32, 7, 7])
        self.assertEqual(shapes[-1]['observed_input'], 1568)
        # Independently read the actual FC declaration: other workers may change it.
        tree = ast.parse((ROOT / 'snn_mnist_s.py').read_text(encoding='utf-8-sig'))
        fc = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Attribute) and t.attr == 'fc1' for t in n.targets))
        declared = ast.literal_eval(fc.args[0]) if isinstance(fc.args[0], ast.Constant) else 32 * 7 * 7
        self.assertEqual(shapes[-1]['intended_input'], declared)
        self.assertEqual(shapes[-1]['matches'], declared == 1568)

    def test_mismatch_detected_from_modified_source_without_repair(self):
        source = (ROOT / 'snn_mnist_s.py').read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'fc1' for t in n.targets):
                n.value.args[0] = ast.Constant(value=512)
        with patch.object(Path, 'is_file', lambda p: p.name == 'snn_mnist_s.py'), patch.object(Path, 'read_text', return_value=ast.unparse(tree)):
            records = inspect_project('virtual-project')
        self.assertEqual([r['model_id'] for r in records], ['snn_mnist_s'])
        issue = next(e for e in records[0]['source_errors'] if e['code'] == 'fc_shape_mismatch')
        self.assertEqual((issue['observed_input'], issue['intended_input']), (1568, 512))

    def test_sound_source_dimensions_and_incomplete_mapping(self):
        record = self.models['SoundSNN']
        self.assertEqual(record['dimensions']['fc_dimensions'], [[784, 128], [128, 128], [128, 128], [128, 10]])
        issue = next(e for e in record['source_errors'] if e['code'] == 'sound_input_dimension_mismatch')
        self.assertEqual((issue['observed_input'], issue['intended_input']), (16, 784))
        self.assertIn('incomplete_mapping', {e['code'] for e in record['source_errors']})
        self.assertIn('mem*(decay/255)+beta', record['semantics']['state_update'])
        self.assertIn('mem > thresh', record['semantics']['state_update'])

    def test_shd_internal_dendrites_soma_and_time_window(self):
        record = self.models['shd']
        states = record['semantics']['internal_states']
        self.assertEqual(len(states), 5)
        self.assertEqual([s['fire_type'] for s in states], [2, 2, 2, 2, 5])
        self.assertEqual(len({s['addr_v'] for s in states}), 5)
        self.assertTrue(all(s['Tw_en'] and s['Tw_len'] == 1000 for s in states))
        self.assertTrue(all(s['Vleaky_alpha'] == 128 for s in states))
        self.assertEqual(record['dimensions']['input_neurons'], 700)
        self.assertEqual(record['dimensions']['hidden_neurons'], 64)
        self.assertEqual(len(record['dimensions']['branch_fc']), 4)

    def test_hybrid_is_not_invented_temporal_python_model(self):
        record = self.models['snnlenet']
        self.assertIn('hybrid_reference_mismatch', {e['code'] for e in record['source_errors']})
        neuron = next(n for n in record['semantics']['neuron_calls'] if n['operation'] == 'p08')
        self.assertEqual(neuron['kwargs']['Vth0'], 10)
        self.assertEqual(neuron['kwargs']['fire_type'], 4)
        self.assertFalse(neuron['kwargs']['Tw_en'])
        for name in ('snnlenet/train', 'snnlenet/train_lif_float'):
            self.assertIn('Stateless', self.models[name]['semantics']['state_update'])

    def test_beijing_topology_bias_and_reduction(self):
        self.assertEqual(self.models['HNN']['dimensions']['layer_sizes'], [[256, 512, 512, 10]])
        self.assertEqual(self.models['SNN_forward']['dimensions']['layer_sizes'], [[4, 128, 128, 128, 128, 2]])
        self.assertIn('sum', self.models['HNN']['semantics']['output'])
        self.assertIn('last timestep', self.models['SNN_forward']['semantics']['output'])
        for name in ('HNN', 'SNN_forward'):
            self.assertIn('b argument unused', self.models[name]['semantics']['state_update'])

    def test_missing_and_malformed_sources_are_reported_no_invented_models(self):
        with patch.object(Path, 'is_file', return_value=False):
            self.assertEqual(inspect_project('virtual-project'), [])
        def read_source(path, **kwargs):
            if path.name == 'shd_top.py':
                return 'def broken(:\n'
            raise FileNotFoundError(str(path))
        with patch.object(Path, 'is_file', lambda p: p.name == 'shd_top.py'), patch.object(Path, 'read_text', read_source):
            records = inspect_project('virtual-project')
        self.assertEqual([r['model_id'] for r in records], ['shd'])
        self.assertEqual(len(records[0]['source_errors']), len(MODEL_SOURCES['shd']))
        self.assertTrue(all(e['code'] == 'source_unreadable' for e in records[0]['source_errors']))


if __name__ == '__main__':
    unittest.main()
