import unittest
from types import SimpleNamespace
import numpy as np
from onnx2prim.ir import ConvLayer, FCLayer, LayerSeq, MaxPoolLayer
from onnx2prim.codegen import compile_compute, generate_g1_1core, generate_g0
from onnx2prim.diagnostics import CompilerError, UnsupportedModelError


def assignment(index=0, input_words=40, weight_words=48):
    return SimpleNamespace(block_index=index, input_addr=7, weight_addr=1000,
        bias_addr=2000, scratch_addr=3000, output_addr=4000,
        input_words=input_words, weight_words=weight_words, bias_words=32,
        scratch_words=960, output_words=120)


def data(a, first=True, nested=False):
    values = dict(w=np.zeros(a.weight_words*4, dtype=np.int8),
                  b=np.zeros(a.bias_words, dtype=np.int32))
    if first:
        values['x'] = np.zeros(a.input_words*4, dtype=np.int8)
    if nested:
        values = {k: v.reshape(-1, 1 if k == 'b' else 4) for k, v in values.items()}
    return values


def prims(config):
    return config[((0, 0), 0)][0][((0, 0), (0, 0))]['prims']


class CodegenTests(unittest.TestCase):
    def test_nonsquare_conv_fields_and_word_addresses(self):
        l = ConvLayer('conv', 2, 3, 7, 5, 3, 2, pad_top=1, pad_right=2)
        seq = LayerSeq((2, 5, 7), [(l, False)])
        a = assignment()
        p = prims(compile_compute(seq, [a], {'cut1': 3}, [data(a)]))
        self.assertIsNone(p[0]['axon'])
        ax, soma = p[1]['axon'], p[1]['soma1']
        self.assertEqual((ax.PIC, ax.Input_fm_Px, ax.Input_fm_Py), (0x81, 7, 5))
        self.assertEqual((ax.conv_Kx, ax.conv_Ky, ax.pad_up, ax.pad_right), (3, 2, 1, 2))
        self.assertEqual((ax.Addr_InA_base, ax.Addr_V_base), (7, 3000))
        self.assertEqual((soma.Input_fm_Px, soma.Input_fm_Py), (7, 5))
        self.assertEqual((soma.CMP_C, soma.in_cut_start, soma.cin), (0x80808080, 3, 32))
        self.assertEqual((soma.Row_ck_on, soma.in_row_max), (0, 0))
        self.assertEqual(len(ax.memory_blocks[0]['data'][0]), 4)

    def test_pool_relu_and_second_conv_p41(self):
        l = ConvLayer('conv', 2, 3, 8, 6, 1, 1)
        pool = MaxPoolLayer('pool', 3, 8, 6, 2, 2, 2, 2)
        second = ConvLayer('second', 3, 4, 4, 3, 1, 1, pad_top=1)
        seq = LayerSeq((2, 6, 8), [(l, True), (pool, False), (second, False)])
        a, b = assignment(), assignment(1)
        p = prims(compile_compute(seq, [a, b], {'cut1': 1, 'cut2': 2}, [data(a, nested=True), data(b, False)]))
        self.assertEqual((p[1]['soma1'].Row_ck_on, p[1]['soma1'].in_row_max, p[1]['soma1'].CMP_C), (1, 2, 0))
        self.assertEqual((p[2]['axon'].PIC, p[2]['axon'].pad_top), (0x41, 1))
        self.assertEqual(len(p[2]['axon'].memory_blocks), 2)

    def test_fc_and_large_first_conv_use_p41(self):
        for l in [FCLayer('fc', 16, 3), ConvLayer('conv', 17, 3, 2, 2, 1, 1)]:
            a = assignment()
            key = 'fc_cut1' if isinstance(l, FCLayer) else 'cut1'
            seq = LayerSeq(l.input_shape, [(l, True)])
            ax = prims(compile_compute(seq, [a], {key: 0}, [data(a)]))[1]['axon']
            self.assertEqual((ax.PIC, ax.Load_Bias), (0x41, 2))

    def test_fail_closed_and_source(self):
        seq = LayerSeq((2, 5, 7), [(ConvLayer('evil\nname', 2, 3, 7, 5, 3, 2), False)])
        a = assignment()
        with self.assertRaises(UnsupportedModelError):
            compile_compute(seq, [a], {'cut1': 0}, [data(a)], [1, 1])
        with self.assertRaises(CompilerError):
            compile_compute(seq, [a], {'cut1': 16}, [data(a)])
        with self.assertRaises(CompilerError):
            compile_compute(seq, [a], {'cut1': 0}, [dict(x=[1], w=[1], b=[1])])
        with self.assertRaises(UnsupportedModelError):
            generate_g0(seq)
        from onnx2prim.address_allocator import allocate_addresses
        seq.layers[0][0].weights=np.zeros((3,2,2,3),dtype=np.int8)
        allocated,_,final=allocate_addresses(seq)
        source = generate_g1_1core('quoted\nmodel',seq,allocated,{'cut1':0},final)
        namespace={}
        exec(compile(source, '<generated>', 'exec'),namespace)
        supplied={'evil\nname':{'input':{(0,0):np.zeros(allocated[0].input_words*4,dtype=np.int8)},
                               'weight':{(0,0):np.zeros(allocated[0].weight_words*4,dtype=np.int8)},
                               'bias':{(0,0):np.zeros(allocated[0].bias_words,dtype=np.int32)}}}
        generated=namespace['gen_1_map_config']([1,1],200000,1,1,data=supplied)
        self.assertEqual(prims(generated)[1]['axon'].cin,2)


if __name__ == '__main__':
    unittest.main()
