"""No training scripts imported, hardware engine invoked, or files written."""
import unittest
from pathlib import Path
import ast

import numpy as np

from onnx2prim.temporal_neurons import NeuronContract, step, native_float_step
from onnx2prim.snn_ir import LIFSpec
from onnx2prim.snn_reference import lif_step, LOW, HIGH


class TemporalNeuronTests(unittest.TestCase):
    def assert_step(self, contract, current, membrane, previous, spike, new):
        result = step(current, membrane, previous, contract)
        np.testing.assert_array_equal(result[0], spike)
        np.testing.assert_array_equal(result[1], new)
        self.assertEqual(result[1].dtype, np.dtype('int64'))

    def test_orders_and_threshold_equality(self):
        for profile in ('p08', 'sound', 'snntorch', 'hnn'):
            c = NeuronContract(profile, threshold=4, beta_numerator=128)
            expected = {'p08': ([0, 1], [2, 0]), 'sound': ([0, 1], [2, 0]),
                        'snntorch': ([0, 1], [4, 5]), 'hnn': ([0, 1], [4, 5])}[profile]
            self.assert_step(c, [4, 5], [0, 0], [0, 0], *expected)

    def test_hnn_prior_spike_gates_only_old_membrane(self):
        self.assert_step(NeuronContract('hnn', 4), [3, 3], [10, 10], [0, 1], [1, 0], [8, 3])

    def test_snntorch_reset_recomputed_not_previous_spike(self):
        for mode, expected in [('subtract', [1, 5]), ('zero', [2, 5]), ('none', [5, 5])]:
            c = NeuronContract('snntorch', 4, reset_mode=mode)
            self.assert_step(c, [2, 3], [6, 4], [0, 1], [1 if mode == 'none' else 0, 1], expected)

    def test_negative_floor_and_offset(self):
        for profile in ('p08', 'sound', 'snntorch', 'hnn'):
            self.assert_step(NeuronContract(profile, 10), [0], [-3], [0], [0], [-2])
        self.assert_step(NeuronContract('sound', 4, leak_offset=2), [5], [0], [0], [1], [2])

    def test_beta_endpoints_and_no_input_mutation(self):
        for beta in (0, 256):
            u, v, p = np.array([1]), np.array([2]), np.array([0])
            self.assert_step(NeuronContract('hnn', 10, beta), u, v, p, [0], [1 if beta == 0 else 3])
            np.testing.assert_array_equal(v, [2])
        self.assert_step(NeuronContract('sound', 10, 256), [1], [2], [0], [0], [3])

    def test_p08_delegation_and_saturation(self):
        rng = np.random.default_rng(21)
        for mode in ('zero', 'subtract', 'none'):
            spec = LIFSpec(threshold=4, reset_mode={'zero': 0, 'subtract': 1, 'none': 2}[mode],
                           leak_alpha=128, reset_value=-2, floor=-100)
            c = NeuronContract('p08', 4, reset_mode=mode, lif_spec=spec)
            v = np.array([HIGH, LOW, 3, -3])
            for _ in range(20):
                u = rng.integers(-20, 21, size=4)
                expected = lif_step(u, v, np.zeros_like(v), 0, spec)
                actual = step(u, v, np.zeros_like(v), c)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                v = actual[1]

    def test_reject_uncarried_p08_state_and_conflicts(self):
        for spec in (LIFSpec(refractory=1), LIFSpec(window_enabled=True), LIFSpec(fire_type=2),
                     LIFSpec(threshold=5), LIFSpec(window_length=0)):
            with self.assertRaises(ValueError):
                NeuronContract(lif_spec=spec)

    def test_invalid_contracts(self):
        for kwargs in ({'profile': 'other'}, {'beta_numerator': -1}, {'beta_numerator': 257},
                       {'beta_denominator': 255}, {'threshold': 0.3}, {'threshold': True},
                       {'profile': 'hnn', 'reset_mode': 'subtract'}, {'invocation_count': 2},
                       {'profile': 'snntorch', 'invocation_count': 3}):
            with self.assertRaises(ValueError):
                NeuronContract(**kwargs)

    def test_invalid_states(self):
        for args in (([1.2], [0], [0]), ([1], [0, 0], [0]), ([1], [0], [2]),
                     ([1 << 60], [0], [0])):
            with self.assertRaises(ValueError):
                step(*args, NeuronContract('sound'))

    def test_native_float_and_quantized_difference(self):
        c = NeuronContract('snntorch', threshold=256, beta_numerator=230)
        _, native = native_float_step([0], [100], [0], c, beta=0.9)
        _, integer = step([0], [100], [0], c)
        self.assertEqual(native[0], 90)
        self.assertEqual(integer[0], 89)
        h = NeuronContract('hnn', threshold=3)
        s, v = native_float_step([0.3, 0.4], [0, 0], [0, 0], h, beta=0.5, threshold=0.3)
        np.testing.assert_array_equal(s, [0, 1])
        np.testing.assert_allclose(v, [0.3, 0.4])
        with self.assertRaises(ValueError):
            native_float_step([0], [0], [0], NeuronContract(), beta=0.9)

    def test_mapping_is_explicit(self):
        for profile in ('sound', 'snntorch', 'hnn'):
            c = NeuronContract(profile)
            self.assertFalse(c.p08_mapping_supported)
            with self.assertRaises(ValueError):
                c.p08_spec()
        self.assertTrue(NeuronContract().p08_mapping_supported)

    def test_sound_and_hnn_native_trajectories(self):
        for profile, beta, threshold in [('sound', 200 / 255, 1.0), ('hnn', 0.5, 0.3)]:
            c = NeuronContract(profile, threshold=3)
            v = np.zeros(3)
            previous = np.zeros(3, dtype=np.int64)
            for u in (np.array([1.0, 0.4, -0.5]), np.array([0.2, 0.2, 0.1]),
                      np.array([1.1, -0.1, 0.5]), np.zeros(3)):
                if profile == 'sound':
                    integrated = v + u
                    expected_spike = (integrated > threshold).astype(np.int64)
                    expected_v = integrated * (1 - expected_spike) * beta
                else:
                    expected_v = v * beta * (1 - previous) + u
                    expected_spike = (expected_v > threshold).astype(np.int64)
                spike, new = native_float_step(u, v, previous, c, beta=beta, threshold=threshold)
                np.testing.assert_array_equal(spike, expected_spike)
                np.testing.assert_allclose(new, expected_v)
                v, previous = new, spike

    def test_duplicate_metadata_does_not_advance_state_twice(self):
        c = NeuronContract('snntorch', 4, 128, invocation_count=2)
        self.assert_step(c, [3], [4], [0], [1], [5])
        # Chaining the first result would instead produce floor(5/2)+3-4=1.
        self.assert_step(c, [3], [5], [1], [0], [1])

    def test_validate_alias_and_parameter_bounds(self):
        for profile in ('p08', 'sound', 'hnn', 'snntorch'):
            NeuronContract(profile).validate()
            for threshold in (LOW - 1, HIGH + 1):
                with self.assertRaises(ValueError):
                    NeuronContract(profile, threshold=threshold)
        for offset in (LOW - 1, HIGH + 1):
            with self.assertRaises(ValueError):
                NeuronContract('sound', leak_offset=offset)
        for threshold in (-256, 257):
            with self.assertRaises(ValueError):
                NeuronContract('snntorch', threshold=threshold)
        for threshold in (-255, 256):
            NeuronContract('snntorch', threshold=threshold).validate()
        for profile in ('sound', 'hnn', 'snntorch'):
            NeuronContract(profile, threshold=HIGH, reset_mode='zero').validate()

    def test_source_membrane_overflow_is_signalled(self):
        for profile in ('sound', 'hnn', 'snntorch'):
            c = NeuronContract(profile, threshold=HIGH, beta_numerator=256, reset_mode='zero')
            for current, membrane in (([1], [HIGH]), ([-1], [LOW]), ([0], [HIGH + 1])):
                with self.assertRaisesRegex(OverflowError, 'OVERFLOW'):
                    step(current, membrane, [0], c)
            self.assert_step(c, [0], [HIGH], [0], [0], [HIGH])
        # Even a subsequent reset cannot hide overflowing Sound integration.
        with self.assertRaisesRegex(OverflowError, 'OVERFLOW'):
            step([HIGH], [1], [0], NeuronContract('sound', threshold=4))
        with self.assertRaisesRegex(OverflowError, 'OVERFLOW'):
            step([0], [HIGH], [0], NeuronContract('sound', threshold=HIGH,
                                              beta_numerator=256, leak_offset=1))

    def test_source_call_counts_without_import(self):
        root = Path(__file__).resolve().parents[2]
        for filename, counts in [('snn.py', [2, 2, 1]), ('snn_mnist_s.py', [1, 1, 1])]:
            tree = ast.parse((root / filename).read_text(encoding='utf-8-sig'))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SNN')
            forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
            calls = [n.func.attr for n in ast.walk(forward) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)]
            self.assertEqual([calls.count('lif' + str(i)) for i in (1, 2, 3)], counts)

    def test_installed_snntorch_094_native_and_duplicate_calls(self):
        import torch
        import snntorch
        self.assertEqual(snntorch.__version__, '0.9.4')
        for mode in ('subtract', 'zero', 'none'):
            neuron = snntorch.Leaky(beta=0.9, threshold=1.0, reset_mechanism=mode)
            c = NeuronContract('snntorch', 1, 230, reset_mode=mode, invocation_count=2)
            mem = torch.zeros(3, dtype=torch.float64)
            previous = np.zeros(3, dtype=np.int64)
            for current in ([1, 2, -0.3], [0.4, 0, 1.5], [0, -1, 0.5]):
                u = torch.tensor(current, dtype=torch.float64)
                # Both script calls receive the same explicitly passed old state.
                s1, v1 = neuron(u, mem)
                s2, v2 = neuron(u, mem)
                np.testing.assert_array_equal(s1.numpy(), s2.numpy())
                np.testing.assert_array_equal(v1.numpy(), v2.numpy())
                s, v = native_float_step(current, mem.numpy(), previous, c,
                                         beta=float(neuron.beta), threshold=1.0)
                np.testing.assert_array_equal(s, s1.numpy())
                np.testing.assert_allclose(v, v2.numpy(), rtol=1e-12, atol=1e-12)
                mem, previous = v2, s
            self.assertEqual(c.invocation_count, 2)


if __name__ == '__main__':
    unittest.main()
