import tempfile
import unittest
from pathlib import Path
import numpy as np

from onnx2prim.diagnostics import CompilerError
from onnx2prim.project_snn_models import MODEL_IDS,build_project_graph,FLOAT_QUANTIZATION_POLICY
from onnx2prim.quantization import QuantizationConfig
from onnx2prim.temporal_onnx import save_temporal_onnx
from onnx2prim.temporal_parser import parse_onnx_temporal
from onnx2prim.temporal_reference import run_temporal_reference
from onnx2prim.temporal_compiler import compile_onnx_temporal,verify_temporal_behavior

TEST_ROOT=Path(__file__).resolve().parents[2]/'prototype_runs'


class TemporalONNXTests(unittest.TestCase):
    def test_all_full_size_integer_presets_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix='explicit_onnx_',dir=TEST_ROOT) as directory:
            for i,model in enumerate(MODEL_IDS):
                with self.subTest(model=model):
                    options={'quantization_policy':FLOAT_QUANTIZATION_POLICY} if model=='snnlenet/train_lif_float' else {}
                    graph=build_project_graph(model,seed=20+i,**options)
                    path=Path(directory)/f'model{i}.onnx'
                    save_temporal_onnx(graph,path)
                    restored=parse_onnx_temporal(path,QuantizationConfig({}))
                    self.assertEqual((graph.input_shape,graph.readout,graph.input_dtype),
                                     (restored.input_shape,restored.readout,restored.input_dtype))
                    x=np.random.default_rng(i).integers(-1,2,(2,)+graph.input_shape)
                    expected=run_temporal_reference(graph,QuantizationConfig({}),x)
                    actual=run_temporal_reference(restored,QuantizationConfig({}),x)
                    for key,value in expected.items():np.testing.assert_array_equal(actual[key],value)

    def test_onnx_to_original_primitives(self):
        with tempfile.TemporaryDirectory(prefix='explicit_onnx_',dir=TEST_ROOT) as directory:
            graph=build_project_graph('SNN_forward')
            path=save_temporal_onnx(graph,Path(directory)/'model.onnx')
            frames=np.array([[7,-8,6,-3],[1,2,3,4]],dtype=np.int64)
            c=compile_onnx_temporal(path,QuantizationConfig({}),frames)
            actual=verify_temporal_behavior(c)
            expected=run_temporal_reference(graph,QuantizationConfig({}),frames)
            for key,value in expected.items():np.testing.assert_array_equal(actual[key],value)

    def test_export_does_not_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='explicit_onnx_',dir=TEST_ROOT) as directory:
            path=save_temporal_onnx(build_project_graph('SoundSNN'),Path(directory)/'model.onnx')
            before=path.read_bytes()
            with self.assertRaises(CompilerError):save_temporal_onnx(build_project_graph('HNN'),path)
            self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
