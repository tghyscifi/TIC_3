import unittest
from copy import deepcopy
import numpy as np
from onnx2prim.ir import FCLayer, ConvLayer
from onnx2prim.snn_ir import SNNModel, SNNBlock, LIFSpec
from onnx2prim.snn_planner import plan_snn, validate_snn_plan
from onnx2prim.snn_compiler import compile_snn, verify_snn_behavior
from onnx2prim.quantization import QuantizationConfig
from onnx2prim.diagnostics import CompilerError

class SNNBackendTests(unittest.TestCase):
    def single(self, spec=None, weights=None, bias=None):
        weights = np.eye(3,dtype=np.int64) if weights is None else weights
        layer = FCLayer('fc',weights.shape[1],weights.shape[0],weights,bias)
        return SNNModel((weights.shape[1],),[SNNBlock('lif',layer,spec or LIFSpec(), 'spikes')])

    def compare(self, model, frames):
        comp = compile_snn(model,QuantizationConfig({}),np.asarray(frames,dtype=np.int64))
        actual = verify_snn_behavior(comp)
        for key,value in comp.reference_outputs.items(): np.testing.assert_array_equal(actual[key],value)
        return comp

    def test_fc_state_persists_and_verification_resets_at_sample(self):
        model = self.single(LIFSpec(threshold=3,leak_enabled=False))
        comp = self.compare(model,[[2,1,0],[2,1,0],[2,1,0],[2,1,0]])
        np.testing.assert_array_equal(comp.behavior_outputs['spikes'][:,0],[0,1,0,1])
        first = {k:v.copy() for k,v in comp.behavior_outputs.items()}
        verify_snn_behavior(comp)
        for key in first: np.testing.assert_array_equal(first[key],comp.behavior_outputs[key])
        self.assertEqual(comp.map_config[((0,0),0)][0][((0,0),(0,0))]['prims'][1]['soma1'].Tw_cnt,0)

    def test_threshold_negative_leak_refractory_and_reset_modes(self):
        frames = [[3,-2,5],[1,1,1],[-2,-1,0],[2,4,3],[0,0,-1]]
        for mode in (0,1,2):
            with self.subTest(mode=mode):
                self.compare(self.single(LIFSpec(threshold=3,reset_mode=mode,reset_value=-1,
                    reset_decrement=2,leak_alpha=127,leak_beta=-1,floor=-7,refractory=2)),frames)

    def test_window_and_both_membrane_readouts(self):
        for fire in (2,3,4):
            for length in (0,2):
                with self.subTest(fire=fire,length=length):
                    self.compare(self.single(LIFSpec(threshold=3,window_enabled=True,
                        window_length=length,fire_type=fire,readout_cut=1)),[[5,-5,1]]*5)

    def test_readout_cut_does_not_scale_spike_current(self):
        for amount in (0,7):
            comp = self.compare(self.single(LIFSpec(threshold=3,leak_enabled=False,readout_cut=amount)),[[4,0,0]])
            self.assertEqual(comp.behavior_outputs['spikes'][0,0],1)

    def test_two_fc_blocks(self):
        a = FCLayer('a',3,5,np.ones((5,3),dtype=np.int64),np.arange(5))
        b = FCLayer('b',5,2,np.array([[1,-1,2,0,1],[-1,1,0,2,1]]),np.array([0,1]))
        model = SNNModel((3,),[SNNBlock('l1',a,LIFSpec(threshold=3,leak_alpha=200),'s1'),
                               SNNBlock('l2',b,LIFSpec(threshold=2,leak_alpha=128),'out')])
        self.compare(model,[[1,2,0],[0,1,1],[1,1,1],[0,0,0]])

    def test_non_square_padded_conv_flatten_fc(self):
        rng = np.random.default_rng(42)
        a = ConvLayer('c',2,3,5,3,3,3,pad_top=1,pad_down=1,pad_left=1,pad_right=1,
                      weights=rng.integers(-2,3,(3,2,3,3)),bias=np.zeros(3,dtype=np.int64))
        b = FCLayer('f',45,2,rng.integers(-2,3,(2,45)),np.zeros(2,dtype=np.int64),'CHW',(3,3,5))
        model = SNNModel((2,3,5),[SNNBlock('lc',a,LIFSpec(threshold=2),'s'),SNNBlock('lf',b,LIFSpec(threshold=2),'out')])
        self.compare(model,rng.integers(0,2,(4,2,3,5)))

    def test_membrane_saturation(self):
        self.compare(self.single(LIFSpec(threshold=(1<<27)-1,initial_v=(1<<27)-2,
                                        leak_enabled=False)),[[127,127,127]]*2)

    def test_plan_rejects_forged_state_and_missing_regions(self):
        model = self.single(); plan = plan_snn(model)
        bad = deepcopy(plan); bad.states['lif']['membrane_addr'] = bad.input_addr
        with self.assertRaises(CompilerError): validate_snn_plan(model,bad)
        bad = deepcopy(plan); bad.regions.clear()
        with self.assertRaises(CompilerError): validate_snn_plan(model,bad)

    def test_state_capacity_rejected(self):
        model = self.single(weights=np.zeros((20000,3),dtype=np.int8))
        with self.assertRaisesRegex(CompilerError,'Mem0'): plan_snn(model)

    def test_ambiguous_cut_and_bad_input_rejected(self):
        with self.assertRaises(CompilerError): compile_snn(self.single(),QuantizationConfig({'cut1':2}),np.ones((2,3),dtype=np.int8))
        model = self.single(); model.input_encoding = 'binary-spikes'
        with self.assertRaises(CompilerError): compile_snn(model,QuantizationConfig({}),np.array([[2,0,1]]))

    def test_scaled_parameters_and_typed_dispatch(self):
        from onnx2prim.compiler import compile_ir
        layer = FCLayer('fc',2,2,np.array([[.5,-.5],[1.,.5]]),np.array([.25,0.]))
        model = SNNModel((2,),[SNNBlock('lif',layer,LIFSpec(threshold=3,leak_enabled=False),'spikes')])
        comp = compile_ir(model,QuantizationConfig({},'scaled',2.,4.),np.array([[2,1],[2,1]]))
        verify_snn_behavior(comp)
        np.testing.assert_array_equal(comp.behavior_outputs['spikes'],[[0,1],[1,1]])

    def test_shd_style_pre_leak_readout_and_int8_inputs(self):
        # SHD-like parameter profile with a representable 200-step window.
        # The literal legacy 1000-step window is intentionally rejected below.
        rng=np.random.default_rng(77)
        spec=LIFSpec(threshold=64,leak_alpha=128,floor=0,fire_type=2,readout_cut=2,
                     window_enabled=True,window_length=200)
        self.compare(self.single(spec,weights=rng.integers(-3,4,(32,192)),bias=np.arange(32)),
                     rng.integers(-128,128,(4,192)))
        with self.assertRaises(CompilerError): LIFSpec(window_enabled=True,window_length=1000).validate()

    def test_parameter_spill_and_four_fc_lif_layers(self):
        rng=np.random.default_rng(90);blocks=[];cin=256
        for i,cout in enumerate((128,128,128,10)):
            layer=FCLayer('fc'+str(i),cin,cout,rng.integers(-2,3,(cout,cin)),np.zeros(cout,dtype=np.int64))
            blocks.append(SNNBlock('l'+str(i),layer,LIFSpec(threshold=64,leak_alpha=128,floor=0),'s'+str(i)))
            cin=cout
        comp=self.compare(SNNModel((256,),blocks),rng.integers(0,2,(3,256)))
        self.assertTrue(any(p['weight_addr'] < 0x4000 for p in comp.plan.parameters.values()))

    def test_reserved_trace_names_rejected(self):
        model=self.single();model.blocks[0].output='lif.membrane'
        with self.assertRaises(CompilerError):model.validate()

    def test_conv_to_conv_temporal_chain(self):
        rng=np.random.default_rng(29)
        a=ConvLayer('c1',1,3,5,3,3,3,pad_top=1,pad_down=1,pad_left=1,pad_right=1,
                    weights=rng.integers(-2,3,(3,1,3,3)),bias=np.zeros(3,dtype=np.int64))
        b=ConvLayer('c2',3,2,5,3,1,1,weights=rng.integers(-2,3,(2,3,1,1)),bias=np.zeros(2,dtype=np.int64))
        model=SNNModel((1,3,5),[SNNBlock('l1',a,LIFSpec(threshold=2),'s1'),SNNBlock('l2',b,LIFSpec(threshold=1),'s2')])
        self.compare(model,rng.integers(0,2,(3,1,3,5)))

    def test_project_sound_neuron_integer_no_leak_profile(self):
        """Third oracle: real LIFFCNeuron, integer parameters, decay=255/255."""
        import torch
        from generator.SoundSNN.sound_snn import LIFFCNeuron
        rng=np.random.default_rng(41)
        w=rng.integers(-2,3,(5,3));bias=np.arange(5)-2
        frames=rng.integers(-2,4,(8,3))
        neuron=LIFFCNeuron(3,5,{'thresh':4,'decay':255,'beta':0}).eval()
        with torch.no_grad():
            neuron.kernel.weight.copy_(torch.tensor(w,dtype=torch.float32))
            neuron.kernel.bias.copy_(torch.tensor(bias,dtype=torch.float32))
            expected=neuron(torch.tensor(frames[None],dtype=torch.float32),{}).numpy()[0]
        comp=self.compare(self.single(LIFSpec(threshold=4,leak_enabled=False),w,bias),frames)
        np.testing.assert_array_equal(comp.behavior_outputs['spikes'],expected)

if __name__ == '__main__': unittest.main()
