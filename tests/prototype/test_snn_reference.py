import unittest
from dataclasses import replace
import numpy as np
from onnx2prim.snn_ir import LIFSpec
from onnx2prim.snn_reference import lif_step,LOW,HIGH
from onnx2prim.diagnostics import CompilerError

class SNNReferenceTests(unittest.TestCase):
    def step(self,current,v=None,ref=None,window=0,spec=None):
        x=np.asarray(current,dtype=np.int64)
        return lif_step(x,np.zeros_like(x) if v is None else np.asarray(v),
                        np.zeros_like(x) if ref is None else np.asarray(ref),window,spec or LIFSpec())

    def test_strict_threshold_and_leak_after_reset(self):
        out,v,ref,w=self.step([4,5],spec=LIFSpec(threshold=4,reset_value=2,leak_alpha=128))
        np.testing.assert_array_equal(out,[0,1]);np.testing.assert_array_equal(v,[2,1])

    def test_negative_floor_division_and_lower_floor(self):
        _,v,_,_=self.step([-3,-100],spec=LIFSpec(threshold=10,leak_alpha=128,floor=-7))
        np.testing.assert_array_equal(v,[-2,-7])

    def test_saturates_before_threshold(self):
        out,v,_,_=self.step([100,-100],v=[HIGH-1,LOW+1],spec=LIFSpec(threshold=HIGH,leak_enabled=False))
        np.testing.assert_array_equal(out,[0,0]);np.testing.assert_array_equal(v,[HIGH,LOW])

    def test_all_reset_modes(self):
        for mode,expected in ((0,2),(1,3),(2,4)):
            with self.subTest(mode=mode):
                out,v,_,_=self.step([7],spec=LIFSpec(threshold=4,leak_enabled=False,
                    reset_mode=mode,reset_value=2,reset_decrement=3))
                self.assertEqual(int(out[0]),1);self.assertEqual(int(v[0]),expected)

    def test_refractory_gates_input_not_threshold(self):
        out,v,ref,_=self.step([100,100],v=[1,5],ref=[2,2],
            spec=LIFSpec(threshold=4,leak_enabled=False,refractory=3))
        np.testing.assert_array_equal(out,[0,1]);np.testing.assert_array_equal(v,[1,0]);np.testing.assert_array_equal(ref,[1,3])

    def test_disabled_zero_window_still_resets(self):
        out,v,_,w=self.step([3],spec=LIFSpec(threshold=4,leak_enabled=False,window_enabled=False,window_length=0))
        self.assertEqual(int(out[0]),0);self.assertEqual(int(v[0]),0);self.assertEqual(w,0)

    def test_enabled_window_reset_boundary(self):
        spec=LIFSpec(threshold=100,leak_enabled=False,window_enabled=True,window_length=1)
        _,v,ref,w=self.step([3],spec=spec)
        self.assertEqual(w,1);self.assertEqual(int(v[0]),3)
        _,v,ref,w=self.step([3],v=v,ref=ref,window=w,spec=spec)
        self.assertEqual(w,0);self.assertEqual(int(v[0]),0)

    def test_readouts_and_floor_cut(self):
        spec=LIFSpec(threshold=4,leak_alpha=128,readout_cut=1)
        a=self.step([7,-3],spec=replace(spec,fire_type=2))[0]
        b=self.step([7,-3],spec=replace(spec,fire_type=3))[0]
        np.testing.assert_array_equal(a,[1,-1]);np.testing.assert_array_equal(b,[0,-1])

    def test_reject_unrepresentable_contract(self):
        for spec in (LIFSpec(fire_type=5),LIFSpec(leak_alpha=256),LIFSpec(refractory=16),
                     LIFSpec(window_initial=2,window_length=1),LIFSpec(window_length=256),
                     LIFSpec(threshold=1.5),LIFSpec(initial_v=-2,floor=0)):
            with self.subTest(spec=spec):
                with self.assertRaises(CompilerError): spec.validate()

if __name__=='__main__':unittest.main()
