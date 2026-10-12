import unittest
import numpy as np
from onnx2prim.ir import ConvLayer, FCLayer, LayerSeq, MaxPoolLayer
from onnx2prim.layout import align_ceil, pack_activation, pack_first_input, pack_weights, pack_bias
from onnx2prim.address_allocator import allocate_addresses, allocate_addresses_for_stage
from onnx2prim.memory_planner import Stage, plan_stages
from onnx2prim.hardware_config import HardwareConfig
from onnx2prim.diagnostics import CompilerError, UnsupportedModelError
from onnx2prim.validate import validate_addresses


class MemoryTests(unittest.TestCase):
    def fcseq(self, cin=400, cout=160):
        return LayerSeq((cin,), [(FCLayer('fc',cin,cout),False)])

    def test_alignment(self):
        self.assertEqual(align_ceil(33,32),64)
        self.assertEqual(align_ceil(0,4),0)
        for args in [(-1,4),(1,0)]:
            with self.assertRaises(ValueError): align_ceil(*args)

    def test_spill_and_no_overlap(self):
        seq = self.fcseq(400,160)
        seq.layers.append((FCLayer('next',160,32),False))
        aa,end,out = allocate_addresses(seq)
        self.assertGreaterEqual(aa[0].weight_addr,0x4000)
        self.assertGreaterEqual(aa[1].weight_addr,0x2000)
        self.assertLess(aa[1].weight_addr,0x4000)
        self.assertGreaterEqual(aa[1].weight_addr,aa[0].scratch_addr+aa[0].scratch_words)
        self.assertEqual(aa[1].input_addr,aa[0].output_addr)
        self.assertEqual(aa[1].scratch_addr,aa[0].scratch_addr)
        self.assertEqual(end,max(a.bias_addr+a.bias_words for a in aa))
        self.assertEqual(out,aa[-1].output_addr)
        for a in aa:
            for addr in (a.input_addr,a.output_addr,a.scratch_addr,a.weight_addr,a.bias_addr):
                self.assertEqual(addr%4,0)

    def test_out_of_space(self):
        with self.assertRaises(CompilerError): allocate_addresses(self.fcseq(400,1000))
        with self.assertRaises(CompilerError): allocate_addresses(self.fcseq(70000,16))

    def test_stages(self):
        seq = self.fcseq()
        stage = plan_stages(seq)[0]
        self.assertEqual(stage.block_indices,(0,))
        self.assertEqual(allocate_addresses_for_stage(seq,stage),allocate_addresses(seq))
        self.assertEqual(allocate_addresses_for_stage(seq,(0,),seq.input_shape),allocate_addresses(seq))
        for stage in [Stage(()), Stage((0,),size_x=2)]:
            with self.assertRaises(UnsupportedModelError): allocate_addresses_for_stage(seq,stage)
        with self.assertRaises(UnsupportedModelError): allocate_addresses_for_stage(seq,(0,),(1,))
        with self.assertRaises(CompilerError): plan_stages(seq,HardwareConfig(16))

    def test_activation_roundtrip(self):
        x = np.arange(3*2*5,dtype=np.int8).reshape(3,2,5)
        packed = pack_activation(x).reshape(2,5,16)
        np.testing.assert_array_equal(packed[:,:,:3].transpose(2,0,1),x)
        self.assertTrue(np.all(packed[:,:,3:]==0))
        first = pack_first_input(x).reshape(3,2,16)
        np.testing.assert_array_equal(first[:,:,:5],x)
        self.assertTrue(np.all(first[:,:,5:]==0))
        self.assertEqual(pack_activation(np.arange(17,dtype=np.int8)).size,32)

    def test_conv_weight_roundtrip(self):
        l = ConvLayer('conv',2,35,4,3,2,3)
        l.weights = (np.arange(35*2*3*2)%101).astype(np.int8).reshape(35,2,3,2)
        for first,axes,shape in [(True,(0,3,2,1),(32,2,3,2)),(False,(0,1,3,2),(32,2,2,3))]:
            p = pack_weights(l,first_conv=first)
            groups = [p[i:i+np.prod(shape)].reshape(shape,order='F') for i in range(0,p.size,np.prod(shape))]
            recovered = np.concatenate(groups)[:35].transpose(tuple(np.argsort(axes)))
            np.testing.assert_array_equal(recovered,l.weights)
            self.assertTrue(np.all(groups[-1][3:]==0))

    def test_fc_physical_dot_product(self):
        shape=(16,2,3)
        x=(np.arange(96)%7).astype(np.int8).reshape(shape)
        l=FCLayer('fc',96,3,input_layout='CHW',feature_shape=shape)
        l.weights=(np.arange(288)%5).astype(np.int8).reshape(3,96)
        physical_w=pack_weights(l).reshape(32,96,order='F')[:3].astype(np.int64)
        actual=physical_w @ pack_activation(x).astype(np.int64)
        expected=l.weights.astype(np.int64) @ x.ravel().astype(np.int64)
        np.testing.assert_array_equal(actual,expected)
        bias=pack_bias(np.array([-2,3,4],dtype=np.int32),3)
        self.assertEqual(bias.size,32)
        np.testing.assert_array_equal(bias[:3],[-2,3,4])
        self.assertTrue(np.all(bias[3:]==0))

    def test_pool_scratch_and_pad_rejection(self):
        conv=ConvLayer('conv',1,16,8,6,1,1)
        pool=MaxPoolLayer('pool',16,8,6,2,2,2,2)
        fc=FCLayer('fc',16*3*4,10,input_layout='CHW',feature_shape=(16,3,4))
        seq=LayerSeq((1,6,8),[(conv,False),(pool,False),(fc,False)])
        aa,_,_=allocate_addresses(seq)
        self.assertEqual(aa[0].scratch_words,8*2*32)
        self.assertEqual(aa[0].input_words,1*6*16//4)
        conv.cout=3
        pool.cin=3
        fc.cin=3*3*4
        fc.feature_shape=(3,3,4)
        with self.assertRaises(UnsupportedModelError): allocate_addresses(seq)

    def test_fail_closed_packing(self):
        with self.assertRaises(CompilerError): pack_activation(np.ones((1,2,2)))
        with self.assertRaises(CompilerError): pack_activation(np.array([128],dtype=np.int32))
        with self.assertRaises(CompilerError): pack_bias(np.array([2]),2)

    def test_strict_address_validator(self):
        conv=ConvLayer('conv',1,16,8,6,1,1,weights=np.zeros((16,1,1,1),dtype=np.int8))
        pool=MaxPoolLayer('pool',16,8,6,2,2,2,2)
        fc=FCLayer('fc',192,10,weights=np.zeros((10,192),dtype=np.int8),input_layout='CHW',feature_shape=(16,3,4))
        seq=LayerSeq((1,6,8),[(conv,False),(pool,False),(fc,False)])
        self.assertTrue(validate_addresses(seq,allocate_addresses(seq)[0]))
        conv.pad_left=1
        with self.assertRaises(UnsupportedModelError): allocate_addresses(seq)


if __name__ == '__main__':
    unittest.main()
