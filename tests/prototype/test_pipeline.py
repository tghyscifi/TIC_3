import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from onnx2prim.compiler import compile_sequence,compile_onnx,compile_ir,verify_behavior,export_artifacts
from onnx2prim.quantization import QuantizationConfig,integer_parameters
from onnx2prim.ir import ConvLayer,FCLayer,LayerSeq
from onnx2prim.layout import pack_weights,pack_bias,pack_activation
from onnx2prim.validate import validate_addresses
from onnx2prim.diagnostics import CompilerError,UnsupportedModelError

class PipelineTests(unittest.TestCase):
    def test_nonsquare_signed_chw_flatten(self):
        rng=np.random.default_rng(32)
        conv=ConvLayer('conv1',1,16,8,6,3,2,weights=rng.integers(-3,4,(16,1,2,3)),bias=rng.integers(-64,65,16))
        fc=FCLayer('fc1',480,7,weights=rng.integers(-2,3,(7,480)),bias=rng.integers(-64,65,7),feature_shape=(16,5,6),input_layout='CHW')
        seq=LayerSeq((1,6,8),[(conv,False),(fc,False)])
        compilation=compile_sequence(seq,QuantizationConfig({'cut1':1,'fc_cut1':2}),rng.integers(-20,21,seq.input_shape))
        verify_behavior(compilation)
        self.assertTrue(np.any(compilation.reference_outputs[0]<0))
        np.testing.assert_array_equal(compilation.behavior_outputs[-1],compilation.reference_outputs[-1])
        with self.assertRaises(CompilerError):
            validate_addresses(seq,[replace(compilation.assignments[0],weight_words=1),compilation.assignments[1]])
        with self.assertRaises(CompilerError):
            validate_addresses(seq,[replace(compilation.assignments[0],output_addr=compilation.assignments[0].input_addr),compilation.assignments[1]])

    def test_lenet_roundtrip_and_handwritten_oracle(self):
        from examples.prototype.make_lenet_fixture import make_fixture
        from onnx2prim.scheduler import plan_blocks
        from generator.LeNet.g1_1core import gen_1_map_config
        temporary_root=Path(__file__).resolve().parents[2]/'prototype_runs'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root=Path(directory)
            fixture=make_fixture(root/'fixture',seed=19)
            cfg=QuantizationConfig(**json.loads((fixture/'quantization.json').read_text()))
            comp=compile_onnx(fixture/'lenet.onnx',cfg,np.load(fixture/'input.npy'))
            verify_behavior(comp)
            data={}
            for block in plan_blocks(comp.sequence):
                w,b=integer_parameters(block.layer,cfg)
                l=replace(block.layer,weights=w,bias=b)
                data[l.name]={'weight':{(0,0):pack_weights(l,first_conv=block.index==0).reshape(-1,4).tolist()},
                              'bias':{(0,0):pack_bias(b,l.cout).reshape(-1,1).tolist()}}
                if block.index==0:
                    data[l.name]['input']={(0,0):pack_activation(comp.input_data,first_input=True).reshape(-1,4).tolist()}
            hand=gen_1_map_config([0,1,1,1,1,1],200000,1,1,in_cut_start_dict=cfg,data=data)
            manual=hand[((0,0),0)][0][((0,0),(0,0))]['prims']
            generated=comp.map_config[((0,0),0)][0][((0,0),(0,0))]['prims'][1:]
            ax_fields=('PIC','cin','cout','Input_fm_Px','Input_fm_Py','conv_Kx','conv_Ky','conv_Sx','conv_Sy','InA_type','InB_type','Load_Bias','pad_on','Reset_Addr_A','Reset_Addr_V')
            soma_fields=('PIC','PIC_Mode','cin','cout','Input_fm_Px','Input_fm_Py','pooling_Kx','pooling_Ky','pooling_Sx','pooling_Sy','CMP_C','in_cut_start','Row_ck_on','in_row_max','type_in','type_out')
            for hp,gp in zip(manual,generated):
                for slot,fields in (('axon',ax_fields),('soma1',soma_fields)):
                    for field in fields:
                        self.assertEqual(getattr(hp[slot],field),getattr(gp[slot],field),(slot,field))
                # Run the original hand-addressed compute blocks against same oracle.
                hp['axon'].cal_para(); hp['soma1'].cal_para()
            hand_assign=[]
            for a,hp in zip(comp.assignments,manual):
                ax,sm=hp['axon'],hp['soma1']
                hand_assign.append(replace(a,input_addr=ax.Addr_InA_base,weight_addr=ax.Addr_InB_base,
                    bias_addr=ax.Addr_Bias_base,scratch_addr=ax.Addr_V_base,output_addr=sm.Addr_Start_out))
            # Hand allocator deliberately reuses buffers; don't run all-live validator.
            hand[((0,0),0)][0][((0,0),(0,0))]['prims'].insert(0,dict(axon=None,soma1=None,router=None,soma2=None))
            handcomp=replace(comp,map_config=hand,assignments=hand_assign,behavior_outputs=None)
            verify_behavior(handcomp)
            target=export_artifacts(comp,root/'export')
            report=json.loads((target/'manifest.json').read_text())
            self.assertEqual(report['primitive_behavior_verification'],'passed')
            config=json.loads((target/'hardware_config.json').read_text())
            self.assertEqual(len(config['ChipArray'][0]['CoreConfig'][0]['PI']),6)
            with self.assertRaises(CompilerError): export_artifacts(comp,target)
            namespace={'__file__':str(target/'mapping.py')}
            exec(compile((target/'mapping.py').read_text(),'<mapping>','exec'),namespace)
            verify_behavior(namespace['compilation'])

    def test_fail_closed_primary(self):
        conv=ConvLayer('conv1',1,1,2,2,1,1,weights=np.ones((1,1,1,1)),bias=np.zeros(1))
        seq=LayerSeq((1,2,2),[(conv,False)])
        config=QuantizationConfig({'cut1':0})
        for bad_input in (np.ones(seq.input_shape,dtype=float),np.full(seq.input_shape,128)):
            with self.assertRaises(CompilerError): compile_sequence(seq,config,bad_input)
        conv.groups=2
        with self.assertRaises(UnsupportedModelError): compile_sequence(seq,config,np.ones(seq.input_shape,dtype=int))

    def test_snn_state_not_dropped(self):
        from onnx2prim.ir import ModelIR,StateSpec
        model=ModelIR(LayerSeq((1,2,2)),states=[StateSpec('v',(4,),'int28',0,'sample')])
        with self.assertRaises(UnsupportedModelError): compile_ir(model,QuantizationConfig({}),np.zeros((1,2,2),dtype=int))

    def test_overflow_fails_before_backend(self):
        conv=ConvLayer('conv1',1,1,2,2,1,1,weights=np.ones((1,1,1,1)),bias=np.full(1,2**31-1))
        with self.assertRaises(CompilerError):
            compile_sequence(LayerSeq((1,2,2),[(conv,False)]),QuantizationConfig({'cut1':0}),np.ones((1,2,2),dtype=int))

if __name__=='__main__': unittest.main()
