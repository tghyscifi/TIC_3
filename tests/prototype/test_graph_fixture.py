"""Synthetic residual fixture contract; no trained-accuracy assertions."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from examples.prototype.make_residual_fixture import make_fixture, positive_int


class ResidualFixtureTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2] / 'prototype_runs'
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='graph_fixture_', dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_structure_and_quantization(self):
        directory = make_fixture(self.root / 'fixture')
        model = onnx.load(directory / 'residual.onnx')
        onnx.checker.check_model(model)
        self.assertEqual(model.opset_import[0].version, 11)
        self.assertEqual(Counter(n.op_type for n in model.graph.node),
                         {'Conv': 8, 'Relu': 7, 'Add': 3, 'Tanh': 1})
        nodes = {n.name: n for n in model.graph.node}
        for i, skip in enumerate(('relu_in', 'res0_add', 'res1_add')):
            self.assertEqual(list(nodes[f'res{i}_add'].input), [skip, f'res{i}_relu2'])
            self.assertEqual(nodes[f'res{i}_conv1'].input[0], skip)
            self.assertEqual(sum(skip in n.input for n in model.graph.node), 2)
        for tensor in model.graph.initializer:
            values = numpy_helper.to_array(tensor)
            bound = 4 if tensor.name.endswith('.weight') else 64
            self.assertEqual(values.dtype, np.float32)
            self.assertTrue(np.all(values == np.round(values)))
            self.assertTrue(np.all(np.abs(values) <= bound))
        data = np.load(directory / 'input.npy')
        self.assertEqual(data.shape, (4, 6, 8))
        self.assertTrue(np.issubdtype(data.dtype, np.integer))
        self.assertTrue(np.all((data >= -128) & (data <= 127)))
        config = json.loads((directory / 'quantization.json').read_text())
        expected = {n.name: (0 if n.op_type == 'Add' else 2)
                    for n in model.graph.node if n.op_type in ('Conv', 'Add')}
        self.assertEqual(config, {'cuts': expected, 'parameter_mode': 'integer',
                                 'tanh_input_scales': {'tanh': 32.0},
                                 'tanh_output_scales': {'tanh': 128.0}})

    def test_reproducible_and_no_overwrite(self):
        first = make_fixture(self.root / 'first', seed=11)
        second = make_fixture(self.root / 'second', seed=11)
        before = {p.name: p.read_bytes() for p in first.iterdir()}
        self.assertEqual(before, {p.name: p.read_bytes() for p in second.iterdir()})
        with self.assertRaises(FileExistsError):
            make_fixture(first, seed=12)
        self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
        third = make_fixture(self.root / 'third', seed=12)
        self.assertNotEqual(before['residual.onnx'], (third / 'residual.onnx').read_bytes())

    def test_dimensions(self):
        directory = make_fixture(self.root / 'custom', height=3, width=5)
        self.assertEqual(np.load(directory / 'input.npy').shape, (4, 3, 5))
        model = onnx.load(directory / 'residual.onnx')
        self.assertEqual([d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim],
                         [1, 6, 3, 5])
        for height, width in ((0, 8), (6, -1), (1.5, 8)):
            with self.assertRaises(ValueError):
                make_fixture(self.root / 'invalid', height=height, width=width)
        self.assertFalse((self.root / 'invalid').exists())
        self.assertEqual(positive_int('3'), 3)
        with self.assertRaises(ValueError):
            positive_int('abc')


if __name__ == '__main__':
    unittest.main()
