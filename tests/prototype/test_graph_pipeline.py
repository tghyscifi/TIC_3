import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from onnx2prim.ir import GraphIR,GraphNode,ConvLayer,LayerSeq,sequence_to_graph
from onnx2prim.graph_compiler import compile_graph,compile_onnx_graph,verify_graph_behavior,export_graph_artifacts
from onnx2prim.graph_reference import GraphQuantizationConfig
from onnx2prim.graph_planner import plan_graph,validate_graph_plan
from onnx2prim.diagnostics import CompilerError,UnsupportedModelError

class GraphPipelineTests(unittest.TestCase):
    def test_micro_residual_signed_saturation_and_per_node_scaling(self):
        shape=(3,2,4)
        w=np.zeros((3,3,1,1),dtype=np.int64)
        for i in range(3): w[i,i,0,0]=2
        conv=ConvLayer('affine',3,3,4,2,1,1,weights=w,bias=np.array([1,-1,0]))
        graph=GraphIR('x',shape,'y',[
            GraphNode('affine','Conv',('x',),'main',shape,layer=conv),
            GraphNode('join','Add',('x','main'),'y',shape)])
        x=np.array([-128,-100,-3,-1,0,2,80,127]*3,dtype=np.int64).reshape(shape)
        cfg=GraphQuantizationConfig({'affine':1,'join':0})
        comp=compile_graph(graph,cfg,x);verify_graph_behavior(comp)
        expected=np.clip(x+np.clip((2*x+conv.bias[:,None,None])//4,-128,127),-128,127)
        np.testing.assert_array_equal(comp.behavior_outputs['y'],expected)
        self.assertEqual(comp.plan.tensors['x'].last_use,1)
        codes=[p['axon'].PIC for p in comp.map_config[((0,0),0)][0][((0,0),(0,0))]['prims'] if p['axon'] is not None]
        self.assertEqual(codes,[0x41,0x02])
        # Distinct branch scales must not be silently added.
        with self.assertRaises(CompilerError):
            compile_graph(graph,replace(cfg,activation_scales={'x':1,'main':2,'y':1}),x)

    def test_all_signed_tanh_indices(self):
        shape=(1,1,256)
        node=GraphNode('tanh','Tanh',('x',),'y',shape)
        graph=GraphIR('x',shape,'y',[node])
        cfg=GraphQuantizationConfig({},tanh_input_scales={'tanh':32},tanh_output_scales={'tanh':128})
        comp=compile_graph(graph,cfg,np.arange(-128,128,dtype=np.int64).reshape(shape))
        verify_graph_behavior(comp)
        import torch
        mathematical=torch.floor((torch.tanh(torch.arange(-128,128,dtype=torch.float64)/32)*128).clamp(-128,127)).numpy()
        np.testing.assert_array_equal(comp.behavior_outputs['y'].reshape(-1),mathematical)

    def test_unpadded_pool_non_square_negative_values(self):
        shape=(3,5,7);out=(3,2,3)
        node=GraphNode('pool','MaxPool',('x',),'y',out,dict(kx=3,ky=2,sx=2,sy=2))
        graph=GraphIR('x',shape,'y',[node])
        x=-np.arange(1,106,dtype=np.int64).reshape(shape)
        comp=compile_graph(graph,GraphQuantizationConfig({}),x);verify_graph_behavior(comp)
        self.assertTrue(np.all(comp.behavior_outputs['y']<0))
        node.attrs['pad_top']=1
        with self.assertRaises(UnsupportedModelError): compile_graph(graph,GraphQuantizationConfig({}),x)

    def test_three_block_residual_onnx_and_export(self):
        from examples.prototype.make_residual_fixture import make_fixture
        base=Path(__file__).resolve().parents[2]/'prototype_runs';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            fixture=make_fixture(Path(tmp)/'fixture',seed=29,height=4,width=6)
            cfg=GraphQuantizationConfig(**json.loads((fixture/'quantization.json').read_text()))
            comp=compile_onnx_graph(fixture/'residual.onnx',cfg,np.load(fixture/'input.npy'))
            verify_graph_behavior(comp)
            self.assertEqual(len(comp.behavior_outputs),19)
            self.assertEqual(comp.plan.phases,26)
            self.assertTrue(any(a.address==b.address and a.born>b.last_use for a in comp.plan.tensors.values() for b in comp.plan.tensors.values()))
            target=export_graph_artifacts(comp,Path(tmp)/'result')
            config=json.loads((target/'hardware_config.json').read_text())
            self.assertEqual(len(config['ChipArray'][0]['CoreConfig'][0]['PI']),26)
            self.assertEqual(json.loads((target/'manifest.json').read_text())['primitive_behavior_verification'],'passed')
            ns={'__file__':str(target/'mapping.py')}
            exec(compile((target/'mapping.py').read_text(),'<generated DAG>','exec'),ns)
            verify_graph_behavior(ns['compilation'])
            with self.assertRaises(CompilerError): export_graph_artifacts(comp,target)

    def test_topological_order_and_sequence_adapter(self):
        shape=(1,2,2)
        graph=GraphIR('x',shape,'y',[GraphNode('join','Add',('x','r'),'y',shape),GraphNode('r','Relu',('x',),'r',shape)])
        comp=compile_graph(graph,GraphQuantizationConfig({'join':0}),np.array([[[-3,-1],[2,4]]]))
        verify_graph_behavior(comp)
        np.testing.assert_array_equal(comp.behavior_outputs['y'],[[[-3,-1],[4,8]]])
        conv=ConvLayer('conv1',1,1,2,2,1,1,weights=np.ones((1,1,1,1),dtype=int))
        adapted=sequence_to_graph(LayerSeq(shape,[(conv,True)]),{'cut1':0})
        self.assertEqual([n.op for n in adapted.topological_nodes()],['Conv','Relu'])

    def test_original_quantiznet_semantics_at_small_size(self):
        """Third oracle: actual project model, random quantized params, no checkpoint."""
        import torch
        from generator.ResidualBlocks.model.quantized_model import QuantizNet
        from generator.ResidualBlocks.model.quantiation_config import QuantizationConfig as ProjectConfig
        torch.manual_seed(71)
        project_config=ProjectConfig()
        model=QuantizNet(project_config).eval();model.quantize()
        nodes=[];cuts={};current='input';shape=(4,4,6)
        def append_conv(name,module,source,source_shape,amount):
            layer=ConvLayer(name,module.in_channels,module.out_channels,source_shape[2],source_shape[1],
                module.kernel_size[1],module.kernel_size[0],module.stride[1],module.stride[0],
                pad_top=module.padding[0],pad_down=module.padding[0],
                pad_left=module.padding[1],pad_right=module.padding[1],
                weights=module.weight.detach().numpy(),bias=module.bias.detach().numpy())
            nodes.append(GraphNode(name,'Conv',(source,),name,layer.output_shape,layer=layer))
            cuts[name]=amount
            return name,layer.output_shape
        def append_relu(name,source,source_shape):
            nodes.append(GraphNode(name,'Relu',(source,),name,source_shape));return name
        current,shape=append_conv('conv_in',model.conv_in,current,shape,model.in_cut_start_dict['conv_in'])
        current=append_relu('relu_in',current,shape)
        for index in range(3):
            block=getattr(model,f'res{index}');skip=current
            current,shape=append_conv(f'res{index}_conv1',block.conv1,current,shape,block.in_cut_start_dict['conv1'])
            current=append_relu(f'res{index}_relu1',current,shape)
            current,shape=append_conv(f'res{index}_conv2',block.conv2,current,shape,block.in_cut_start_dict['conv2'])
            current=append_relu(f'res{index}_relu2',current,shape)
            name=f'res{index}_add'
            nodes.append(GraphNode(name,'Add',(skip,current),name,shape));cuts[name]=block.in_cut_start_dict['add'];current=name
        current,shape=append_conv('conv_out',model.conv_out,current,shape,model.in_cut_start_dict['conv_out'])
        nodes.append(GraphNode('tanh','Tanh',(current,),'output',shape))
        graph=GraphIR('input',(4,4,6),'output',nodes)
        config=GraphQuantizationConfig(cuts,tanh_input_scales={'tanh':32},tanh_output_scales={'tanh':128})
        x=np.random.default_rng(71).integers(-128,128,graph.input_shape,dtype=np.int64)
        relu_results=[]
        hooks=[m.register_forward_hook(lambda module,args,result:relu_results.append(result.detach().numpy()[0].copy()))
               for m in model.modules() if isinstance(m,torch.nn.ReLU)]
        try:
            with torch.no_grad(): expected=model(torch.tensor(x[None],dtype=torch.float32)).numpy()[0]
        finally:
            for hook in hooks: hook.remove()
        compilation=compile_graph(graph,config,x);verify_graph_behavior(compilation)
        np.testing.assert_array_equal(compilation.behavior_outputs['output'],expected)
        relu_nodes=[n for n in graph.topological_nodes() if n.op=='Relu']
        self.assertEqual(len(relu_results),len(relu_nodes))
        for node,actual_project in zip(relu_nodes,relu_results):
            np.testing.assert_array_equal(compilation.behavior_outputs[node.output],actual_project)

if __name__=='__main__': unittest.main()
