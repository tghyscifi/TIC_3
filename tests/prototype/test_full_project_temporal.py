"""Full original geometry, explicitly quantized synthetic model contracts."""
import unittest
import numpy as np
from onnx2prim.project_snn_models import MODEL_IDS,build_project_graph,FLOAT_QUANTIZATION_POLICY
from onnx2prim.temporal_compiler import compile_temporal,verify_temporal_behavior
from onnx2prim.quantization import QuantizationConfig

class FullProjectTemporalTests(unittest.TestCase):
    def test_all_inventory_model_families_at_full_size(self):
        from onnx2prim.project_snn_inventory import MODEL_SOURCES
        self.assertEqual(set(MODEL_IDS),set(MODEL_SOURCES))
        for i,model_id in enumerate(MODEL_IDS):
            with self.subTest(model=model_id):
                options={'quantization_policy':FLOAT_QUANTIZATION_POLICY} if model_id=='snnlenet/train_lif_float' else {}
                graph=build_project_graph(model_id,seed=101+i,**options)
                rng=np.random.default_rng(103+i)
                frames=rng.integers(-1,2,(3,)+graph.input_shape) if graph.input_dtype=='int2' else rng.integers(-8,9,(3,)+graph.input_shape)
                c=compile_temporal(graph,QuantizationConfig({}),frames)
                actual=verify_temporal_behavior(c)
                for k,expected in c.reference_outputs.items():np.testing.assert_array_equal(actual[k],expected)
                self.assertTrue(c.manifest()['host_staging_required'])
                self.assertEqual(c.manifest()['autonomous_chip_deployment'],'NOT implemented')

if __name__=='__main__':unittest.main()
