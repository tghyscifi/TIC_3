"""Verified DAG compute pipeline. Routing and spatial partitioning are separate."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import numpy as np
from .diagnostics import CompilerError
from .graph_planner import plan_graph,physical_channels

@dataclass
class GraphCompilation:
    graph: object
    quantization: object
    plan: object
    input_data: object
    map_config: dict
    schedule: list
    reference_outputs: dict
    behavior_outputs: object=None
    def manifest(self):
        return {'schema_version':2,'scope':'ANN single-core DAG compute only',
                'model_name':self.graph.model_name,'address_unit':'4-byte word','layout':'HWC C32',
                'input_shape':list(self.graph.input_shape),'static_validation':'passed',
                'primitive_behavior_verification':'passed' if self.behavior_outputs is not None else 'not run',
                'router_timing_or_chip_verification':'not run','phases':self.plan.phases,
                'tensors':{name:asdict(a) for name,a in self.plan.tensors.items()},
                'parameters':self.plan.parameters,'schedule':self.schedule,
                'temporary_buffers':{'concat_addr':self.plan.concat_addr,'concat_words':self.plan.concat_words,
                                    'scratch_addr':self.plan.scratch_addr,'scratch_words':self.plan.scratch_words}}

def compile_graph(graph,config,input_data):
    from .graph_reference import run_graph_reference
    from .graph_codegen import build_graph_mapping
    plan=plan_graph(graph);config.validate(graph)
    x=np.asarray(input_data)
    if x.shape!=tuple(graph.input_shape) or x.dtype.kind not in 'iu' or np.any(x<-128) or np.any(x>127):
        raise CompilerError('Graph input must be matching CHW integers in int8 range',code='INPUT')
    reference=run_graph_reference(graph,config,x)
    mapping,schedule=build_graph_mapping(graph,config,plan,x)
    return GraphCompilation(graph,config,plan,x.copy(),mapping,schedule,reference)

def compile_onnx_graph(path,config,input_data):
    from .graph_parser import parse_onnx_graph
    return compile_graph(parse_onnx_graph(path),config,input_data)

def verify_graph_behavior(compilation):
    from generator.behavior_simulator.memory import Memory
    compilation.behavior_outputs=None
    memory=Memory()
    prims=compilation.map_config[((0,0),0)][0][((0,0),(0,0))]['prims']
    for p in prims:
        for slot in ('axon','soma1','soma2'):
            op=p.get(slot)
            if op is not None:
                for block in op.memory_blocks:
                    if block.get('initialize',True):
                        memory.addMemoryBlock(block['start'],len(block['data']),block['data'])
    boundaries={s['end_phase']:s for s in compilation.schedule}
    outputs={}
    for phase,p in enumerate(prims[1:],1):
        for slot in ('axon','soma1','soma2'):
            op=p.get(slot)
            if op is not None:
                memory.writeMem(op.prim_execution(memory.read_memory(op.getInfoList())))
        memory.updateMemory()
        if phase in boundaries:
            spec=boundaries[phase];value=spec['output'];a=compilation.plan.tensors[value]
            c,h,w=a.shape
            raw=np.asarray(memory.read_memory([{'start':a.address,'length':a.words,'type':1}])[0],dtype=np.int64)
            actual=raw.reshape(h,w,physical_channels(a.shape))[:,:,:c].transpose(2,0,1)
            expected=compilation.reference_outputs[value]
            if not np.array_equal(actual,expected):
                raise CompilerError(f"Graph behavior mismatch, max_abs_diff={int(np.abs(actual-expected).max())}",node=spec['node'],code='VERIFICATION')
            outputs[value]=actual.copy()
    if set(outputs)!={n.output for n in compilation.graph.nodes}:
        raise CompilerError('Verification did not cover every graph node',code='VERIFICATION')
    compilation.behavior_outputs=outputs
    return outputs

def export_graph_artifacts(compilation,directory):
    import onnx
    from .compiler import hardware_json
    directory=Path(directory)
    if directory.exists(): raise CompilerError('Choose a new output directory',code='OUTPUT')
    encoded=json.dumps(hardware_json(compilation),indent=2,ensure_ascii=False)
    model=onnx.load(compilation.graph.source_path)
    directory.mkdir(parents=True,exist_ok=False)
    onnx.save(model,str(directory/'model.onnx'))
    np.save(directory/'input.npy',compilation.input_data)
    (directory/'quantization.json').write_text(json.dumps(asdict(compilation.quantization),indent=2),encoding='utf-8')
    (directory/'manifest.json').write_text(json.dumps(compilation.manifest(),indent=2),encoding='utf-8')
    (directory/'hardware_config.json').write_text(encoded,encoding='utf-8')
    np.savez(directory/'reference_outputs.npz',**compilation.reference_outputs)
    if compilation.behavior_outputs is not None: np.savez(directory/'behavior_outputs.npz',**compilation.behavior_outputs)
    source='''# Single-core DAG compute only. No router/timing deployment claim.
from pathlib import Path
import json
import numpy as np
from onnx2prim.graph_compiler import compile_onnx_graph
from onnx2prim.graph_reference import GraphQuantizationConfig
here=Path(__file__).resolve().parent
config=GraphQuantizationConfig(**json.loads((here/'quantization.json').read_text(encoding='utf-8')))
compilation=compile_onnx_graph(here/'model.onnx',config,np.load(here/'input.npy',allow_pickle=False))
map_config=compilation.map_config
'''
    (directory/'mapping.py').write_text(source,encoding='utf-8')
    return directory.resolve()
