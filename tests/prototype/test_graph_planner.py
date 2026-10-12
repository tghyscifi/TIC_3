"""Safety contracts for the conservative single-core DAG memory planner."""
import copy
from dataclasses import replace
import unittest

import numpy as np

from onnx2prim.diagnostics import CompilerError
from onnx2prim.graph_planner import plan_graph, validate_graph_plan
from onnx2prim.ir import ConvLayer, GraphIR, GraphNode


def conv(name, source, shape, channels=None, kernel=1):
    c, h, w = shape
    out = c if channels is None else channels
    layer = ConvLayer(name, c, out, w, h, kernel, kernel,
                      pad_top=kernel // 2, pad_down=kernel // 2,
                      pad_left=kernel // 2, pad_right=kernel // 2,
                      weights=np.ones((out, c, kernel, kernel), dtype=np.int8),
                      bias=np.zeros(out, dtype=np.int32))
    return GraphNode(name, 'Conv', (source,), name, (out, h, w),
                     {'cut': 0}, layer)


def residual(shape=(3, 4, 5)):
    # stem survives both branch computations until the residual merge.
    return GraphIR('input', shape, 'output', [
        conv('stem', 'input', shape),
        GraphNode('relu', 'Relu', ('stem',), 'relu', shape),
        conv('branch', 'relu', shape),
        GraphNode('merge', 'Add', ('stem', 'branch'), 'output', shape),
    ])


class GraphPlannerTests(unittest.TestCase):
    def setUp(self):
        self.graph = residual()
        self.plan = plan_graph(self.graph)

    def reject(self, plan):
        with self.assertRaises(CompilerError):
            validate_graph_plan(self.graph, plan)

    def test_residual_lifetimes_padding_and_reuse(self):
        p = self.plan
        self.assertTrue(validate_graph_plan(self.graph, p))
        self.assertEqual(p.phases, 7)
        self.assertEqual(p.tensors['stem'].last_use, 3)
        self.assertEqual(p.tensors['output'].last_use, 4)
        for allocation in p.tensors.values():
            self.assertEqual(allocation.shape, (3, 4, 5))
            self.assertEqual(allocation.words, 32 * 4 * 5 // 4)
            self.assertEqual(allocation.address % 4, 0)
        allocations = list(p.tensors.values())
        reused = False
        for i, a in enumerate(allocations):
            for b in allocations[i + 1:]:
                overlaps = a.address < b.address + b.words and b.address < a.address + a.words
                live = a.born <= b.last_use and b.born <= a.last_use
                self.assertFalse(overlaps and live)
                reused |= overlaps and not live
        self.assertTrue(reused, 'Disjoint lifetimes should permit buffer reuse')
        self.assertEqual(p.concat_words, 2 * 160)
        self.assertEqual(p.scratch_words, 32 * 4 * 5)

    def test_live_alias_rejected(self):
        p = copy.deepcopy(self.plan)
        p.tensors['branch'] = replace(p.tensors['branch'], address=p.tensors['stem'].address)
        self.reject(p)

    def test_forged_lifetime_cannot_hide_live_alias(self):
        p = copy.deepcopy(self.plan)
        p.tensors['stem'] = replace(p.tensors['stem'], last_use=0)
        p.tensors['branch'] = replace(p.tensors['branch'], address=p.tensors['stem'].address)
        self.reject(p)

    def test_invalid_addresses_and_coverage(self):
        for address in (-4, 1, self.plan.activation_end):
            with self.subTest(address=address):
                p = copy.deepcopy(self.plan)
                p.tensors['stem'] = replace(p.tensors['stem'], address=address)
                self.reject(p)
        p = copy.deepcopy(self.plan)
        del p.tensors['branch']
        self.reject(p)

    def test_temporary_sizes_must_cover_operations(self):
        for field in ('concat_words', 'scratch_words'):
            for size in (0, 4, -4):
                with self.subTest(field=field, size=size):
                    p = copy.deepcopy(self.plan)
                    setattr(p, field, size)
                    self.reject(p)

    def test_temporary_overlap_rejected(self):
        p = copy.deepcopy(self.plan)
        p.scratch_addr = p.concat_addr
        self.reject(p)

    def test_parameter_coverage_and_addresses(self):
        for mutation in ('missing', 'address'):
            with self.subTest(mutation=mutation):
                p = copy.deepcopy(self.plan)
                if mutation == 'missing':
                    del p.parameters['stem']
                else:
                    p.parameters['stem']['weight_addr'] = 0
                self.reject(p)

    def test_persistent_regions_required(self):
        p = copy.deepcopy(self.plan)
        p.persistent_regions = []
        self.reject(p)

    def test_persistent_overlap_and_bank_crossing(self):
        for region in ((0, 4, 'bad'), (0x7ffc, 0x8004, 'bad')):
            with self.subTest(region=region):
                p = copy.deepcopy(self.plan)
                p.persistent_regions.append(region)
                self.reject(p)

    def test_invalid_graphs(self):
        dead = copy.deepcopy(self.graph)
        dead.nodes.append(GraphNode('dead', 'Relu', ('input',), 'dead', dead.input_shape))
        cycle = GraphIR('input', (3, 4, 5), 'a', [
            GraphNode('a', 'Relu', ('b',), 'a', (3, 4, 5)),
            GraphNode('b', 'Relu', ('a',), 'b', (3, 4, 5))])
        unknown = copy.deepcopy(self.graph)
        unknown.nodes[0].inputs = ('missing',)
        # Use a standalone Add to isolate broadcast validation from Conv shapes.
        broadcast = GraphIR('input', (3, 4, 5), 'out', [
            GraphNode('pool', 'MaxPool', ('input',), 'small', (3, 1, 1),
                      {'kx': 5, 'ky': 4, 'sx': 1, 'sy': 1}),
            GraphNode('add', 'Add', ('input', 'small'), 'out', (3, 4, 5))])
        for graph in (dead, cycle, unknown, broadcast):
            with self.subTest(graph=graph):
                with self.assertRaises(CompilerError):
                    plan_graph(graph)

    def test_pi_capacity_fails_explicitly(self):
        nodes = []
        value = 'input'
        for i in range(32):
            output = 'relu' + str(i)
            nodes.append(GraphNode(output, 'Relu', (value,), output, (3, 1, 1)))
            value = output
        with self.assertRaisesRegex(CompilerError, 'PI limit'):
            plan_graph(GraphIR('input', (3, 1, 1), value, nodes))

    def test_realistic_spatial_size_fails_mem0(self):
        with self.assertRaisesRegex(CompilerError, 'Mem0') as error:
            plan_graph(residual((3, 560, 60)))
        self.assertEqual(error.exception.code, 'MEMORY')

    def test_persistent_spill_to_mem0_is_disjoint(self):
        shape = (32, 1, 1)
        nodes = []
        value = 'input'
        for i in range(8):
            node = conv('conv' + str(i), value, shape, kernel=3)
            nodes.append(node)
            value = node.output
        graph = GraphIR('input', shape, value, nodes)
        p = plan_graph(graph)
        self.assertTrue(validate_graph_plan(graph, p))
        self.assertTrue(any(a >= 0x4000 for a, _, _ in p.persistent_regions))
        self.assertTrue(any(a < 0x4000 for a, _, _ in p.persistent_regions))
        for a, b, _ in p.persistent_regions:
            self.assertTrue(0 <= a < b <= 0x4000 or 0x4000 <= a < b <= 0x8000)
            if a < 0x4000:
                self.assertGreaterEqual(a, p.scratch_addr + p.scratch_words)

    def test_persistent_capacity_failure(self):
        shape = (32, 1, 1)
        node = conv('huge', 'input', shape, channels=256, kernel=7)
        with self.assertRaisesRegex(CompilerError, 'SRAM banks'):
            plan_graph(GraphIR('input', shape, 'huge', [node]))


if __name__ == '__main__':
    unittest.main()
