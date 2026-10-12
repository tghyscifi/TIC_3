import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h
from examples.prototype.make_snn_fixture import make_fixture
from onnx2prim.snn_parser import parse_onnx_snn
from onnx2prim.snn_compiler import compile_onnx_snn,verify_snn_behavior,export_snn_artifacts
from onnx2prim.quantization import QuantizationConfig
from onnx2prim.diagnostics import CompilerError

class SNNParserTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[2]/'prototype_runs';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        self.fixture=make_fixture(self.root/'fixture')
        self.path=self.fixture/'model.onnx'
    def tearDown(self):self.temp.cleanup()
    def mutate(self, callback):
        model=onnx.load(self.path);callback(model);onnx.save(model,str(self.root/'modified.onnx'))
        return self.root/'modified.onnx'
    def test_import_explicit_fc_chain(self):
        model=parse_onnx_snn(self.path)
        self.assertEqual(model.input_shape,(4,));self.assertEqual(len(model.blocks),2)
        self.assertEqual(len(model.states),6);self.assertEqual(model.blocks[0].lif.refractory,1)
    def test_missing_metadata_rejected(self):
        path=self.mutate(lambda m:m.ClearField('metadata_props'))
        with self.assertRaises(CompilerError):parse_onnx_snn(path)
    def test_missing_neuron_contract_rejected(self):
        def change(m):
            node=m.graph.node[1];attrs=[a for a in node.attribute if a.name!='threshold']
            node.ClearField('attribute');node.attribute.extend(attrs)
        with self.assertRaises(CompilerError):parse_onnx_snn(self.mutate(change))
    def test_unknown_neuron_and_adaptation_rejected(self):
        path=self.mutate(lambda m:m.graph.node[1].attribute.append(h.make_attribute('Vth_adpt_en',1)))
        with self.assertRaises(CompilerError):parse_onnx_snn(path)
    def test_plain_relu_not_inferred_as_lif(self):
        def change(m):
            m.graph.node[1].op_type='Relu';m.graph.node[1].domain='';m.graph.node[1].ClearField('attribute')
        with self.assertRaises(CompilerError):parse_onnx_snn(self.mutate(change))
    def test_custom_version_rejected(self):
        def change(m):m.opset_import[1].version=2
        with self.assertRaises(CompilerError):parse_onnx_snn(self.mutate(change))
    def test_dynamic_shape_rejected(self):
        def change(m):
            dim=m.graph.input[0].type.tensor_type.shape.dim[1];dim.ClearField('dim_value');dim.dim_param='N'
        with self.assertRaises(CompilerError):parse_onnx_snn(self.mutate(change))
    def test_final_shape_mismatch_rejected(self):
        def change(m):m.graph.output[0].type.tensor_type.shape.dim[1].dim_value=3
        with self.assertRaises(CompilerError):parse_onnx_snn(self.mutate(change))
    def test_boolean_attribute_requires_zero_or_one(self):
        def change(m):
            for a in m.graph.node[1].attribute:
                if a.name=='leak_enabled':a.i=2
        with self.assertRaises(CompilerError):parse_onnx_snn(self.mutate(change))
    def test_spatial_fixture_full_roundtrip(self):
        fixture=make_fixture(self.root/'spatial',True)
        model=parse_onnx_snn(fixture/'model.onnx')
        self.assertEqual(model.blocks[1].affine.feature_shape,(3,3,5))
        comp=compile_onnx_snn(fixture/'model.onnx',QuantizationConfig({}),np.load(fixture/'input.npy'))
        verify_snn_behavior(comp)
        output=export_snn_artifacts(comp,self.root/'result')
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['primitive_behavior_verification'],'passed')
        self.assertEqual(manifest['time_steps'],6)
        hw=json.loads((output/'hardware_config.json').read_text())
        self.assertEqual(hw['ChipArray'][0]['CoreConfig'][0]['PI'][1]['PI_parameter'][1]['PIC'],8)
        self.assertEqual(hw['ChipArray'][0]['CoreConfig'][0]['PI'][2]['PI_parameter'][1]['P_LIF.Tw_cnt'],0)
        namespace={'__file__':str(output/'mapping.py')}
        exec(compile((output/'mapping.py').read_text(),'<SNN generated>','exec'),namespace)
        verify_snn_behavior(namespace['compilation'])
        for key in comp.behavior_outputs:np.testing.assert_array_equal(comp.behavior_outputs[key],namespace['compilation'].behavior_outputs[key])
        with self.assertRaises(CompilerError):export_snn_artifacts(comp,output)

if __name__=='__main__':unittest.main()
