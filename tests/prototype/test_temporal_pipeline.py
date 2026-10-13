import unittest
from dataclasses import replace
import numpy as np
from onnx2prim.ir import ConvLayer,FCLayer
from onnx2prim.temporal_ir import TemporalGraph,TemporalNode,P08Contract
from onnx2prim.temporal_neurons import NeuronContract
from onnx2prim.temporal_compiler import compile_temporal,verify_temporal_behavior
from onnx2prim.quantization import QuantizationConfig
from onnx2prim.diagnostics import CompilerError

class TemporalPipelineTests(unittest.TestCase):
    def verify(self,g,x):
        c=compile_temporal(g,QuantizationConfig({}),np.asarray(x,dtype=np.int64))
        result=verify_temporal_behavior(c)
        for k in c.reference_outputs:np.testing.assert_array_equal(result[k],c.reference_outputs[k])
        return c
    def graph(self,profile,cin=3,cout=5,**kwargs):
        rng=np.random.default_rng(8)
        f=FCLayer('fc',cin,cout,rng.integers(-2,3,(cout,cin)),np.arange(cout)-2)
        dtype='int2' if isinstance(profile,P08Contract) and profile.fire_type==5 else 'int8'
        return TemporalGraph('x',(cin,),'y',[TemporalNode('fc','FC',('x',),'u',(cout,),'int32',layer=f),
               TemporalNode('lif','LIF',('u',),'y',(cout,),dtype,neuron=profile)],**kwargs)
    def test_all_source_neuron_macros(self):
        x=np.array([[3,-2,1],[-1,4,0],[1,1,1],[2,0,-1],[0,-3,2]])
        for profile in ('sound','hnn','snntorch'):
            for beta in (0,128,230,256):
                with self.subTest(profile=profile,beta=beta):
                    self.verify(self.graph(NeuronContract(profile,threshold=3,beta_numerator=beta)),x)
    def test_snntorch_zero_none_and_threshold256(self):
        for mode in ('zero','none','subtract'):
            self.verify(self.graph(NeuronContract('snntorch',threshold=256,beta_numerator=230,reset_mode=mode)),[[127,127,127]]*5)
    def test_packed_2bit_700_fc_and_readout(self):
        g=self.graph(P08Contract(threshold=3,fire_type=5),700,64,input_dtype='int2',readout='mean')
        rng=np.random.default_rng(2);c=self.verify(g,rng.integers(-1,2,(3,700)))
        self.assertGreater(len(c.kernels['fc']),1)
    def test_sound_full_first_layer_and_bias_once(self):
        g=self.graph(NeuronContract('sound',threshold=64,beta_numerator=256),784,128)
        c=self.verify(g,np.random.default_rng(3).integers(0,2,(2,784)))
        self.assertEqual(len(c.kernels['fc']),8)
        self.assertEqual(sum(t.attrs['bias_once'] for t in c.kernels['fc']),4)
    def test_spatial_halo_stride_and_state(self):
        rng=np.random.default_rng(5)
        l=ConvLayer('c',2,7,9,7,5,3,2,2,pad_top=1,pad_down=1,pad_left=2,pad_right=2,
                    weights=rng.integers(-2,3,(7,2,3,5)),bias=np.arange(7))
        g=TemporalGraph('x',(2,7,9),'y',[TemporalNode('c','Conv',('x',),'u',l.output_shape,'int32',layer=l),
            TemporalNode('lif','LIF',('u',),'y',l.output_shape,'int8',neuron=P08Contract(threshold=3))])
        self.verify(g,rng.integers(-3,4,(3,2,7,9)))
    def test_branch_add_concat_pool_and_cast(self):
        rng=np.random.default_rng(13);shape=(2,5,7)
        l=ConvLayer('c',2,3,7,5,1,1,weights=rng.integers(-2,3,(3,2,1,1)))
        nodes=[TemporalNode('c','Conv',('x',),'u',l.output_shape,'int32',layer=l),
            TemporalNode('l1','LIF',('u',),'a',l.output_shape,'int8',neuron=P08Contract(threshold=2)),
            TemporalNode('l2','LIF',('u',),'b',l.output_shape,'int8',neuron=P08Contract(threshold=3)),
            TemporalNode('add','Add',('a','b'),'sum',l.output_shape,'int32'),
            TemporalNode('cut','Cut',('sum',),'i8',l.output_shape,'int8',{'cut':0}),
            TemporalNode('join','Concat',('a','i8'),'cat',(6,5,7),'int8'),
            TemporalNode('pool','MaxPool',('cat',),'p',(6,2,3),'int8',dict(kx=2,ky=2,sx=2,sy=2)),
            TemporalNode('cast','Cast',('p',),'y',(6,2,3),'int32')]
        self.verify(TemporalGraph('x',shape,'y',nodes),rng.integers(-2,3,(3,)+shape))
    def test_wide_window_and_per_neuron_initial_state(self):
        g=self.graph(P08Contract(threshold=10,leak_enabled=False,window_enabled=True,window_length=1000,
                                window_initial=999,initial_v=[2,3,4,5,6]),3,5)
        c=self.verify(g,[[1,0,0]]*4)
        self.assertTrue(c.kernels['lif'][0].attrs['software_window_counter'])
        np.testing.assert_array_equal(c.behavior_outputs['lif.window'],[1000,0,1,2])
    def test_repeated_samples_reset_and_export_initial_counter(self):
        c=self.verify(self.graph(P08Contract(threshold=3,window_enabled=True,window_length=2)),[[2,1,0]]*4)
        before={k:v.copy() for k,v in c.behavior_outputs.items()};verify_temporal_behavior(c)
        for k in before:np.testing.assert_array_equal(before[k],c.behavior_outputs[k])
        self.assertEqual(c.kernels['lif'][0].operations[0].Tw_cnt,0)
    def test_invalid_target_input(self):
        g=self.graph(P08Contract())
        with self.assertRaises(CompilerError):compile_temporal(g,QuantizationConfig({}),np.ones((2,3),dtype=float))

    def test_config_cuts_are_never_silently_ignored(self):
        g=self.graph(P08Contract())
        for cuts in ({'missing':0},{'fc':1},{'fc':True}):
            with self.subTest(cuts=cuts), self.assertRaises(CompilerError):
                compile_temporal(g,QuantizationConfig(cuts),np.ones((2,3),dtype=np.int64))

    def test_matching_frontend_cut_configuration(self):
        layer=FCLayer('fc',3,4,np.ones((4,3),dtype=np.int64),np.zeros(4,dtype=np.int64))
        g=TemporalGraph('x',(3,),'y',[
            TemporalNode('fc','FC',('x',),'u',(4,),'int32',layer=layer),
            TemporalNode('fc__cut','Cut',('u',),'y',(4,),'int8',{'cut':1})])
        c=compile_temporal(g,QuantizationConfig({'fc':1}),np.array([[4,4,4],[-4,-4,-4]]))
        actual=verify_temporal_behavior(c)
        np.testing.assert_array_equal(actual['y'],[[3]*4,[-3]*4])
        with self.assertRaises(CompilerError):compile_temporal(g,QuantizationConfig({'fc':2}),c.input_data)

if __name__=='__main__':unittest.main()
