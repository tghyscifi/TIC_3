import unittest
import numpy as np
from onnx2prim.ir import FCLayer, LayerSeq
from onnx2prim.quantization import QuantizationConfig, integer_parameters, cut, build_cut_config
from onnx2prim.reference import run_reference
from onnx2prim.diagnostics import CompilerError

class QuantizationTests(unittest.TestCase):
    def make_seq(self):
        return LayerSeq((1,1,2),[(FCLayer('fc1',2,2,weights=np.array([[1,2],[-1,-2]]),bias=np.zeros(2),feature_shape=(1,1,2),input_layout='CHW'),False)])
    def test_negative_floor_and_saturation(self):
        np.testing.assert_array_equal(cut([-5,-4,-1,512],1),[-2,-1,-1,127])
        np.testing.assert_array_equal(cut([-5,512],1,True),[0,127])
    def test_no_implicit_cut_or_clipping(self):
        seq=self.make_seq()
        with self.assertRaises(CompilerError): build_cut_config(seq)
        cfg=QuantizationConfig({'fc_cut1':0})
        seq.layers[0][0].weights[0,0]=128
        with self.assertRaises(CompilerError): integer_parameters(seq.layers[0][0],cfg)
    def test_reference_signed_fc(self):
        out=run_reference(self.make_seq(),QuantizationConfig({'fc_cut1':0}),np.array([[[3,4]]]))
        np.testing.assert_array_equal(out[0],[11,-11])
    def test_invalid_cut_rejected(self):
        for cuts in ({}, {'fc_cut1':False},{'fc_cut1':1.5},{'fc_cut1':16}):
            with self.assertRaises(CompilerError): QuantizationConfig(cuts).validate(self.make_seq())

if __name__ == '__main__': unittest.main()
