"""Explicit temporal compute template, verified against original Memory/p08.

Frames are fed by this Python verifier, NOT a generated chip/router input driver.
Exported static PI computes one frame. Hardware time-step replay is not claimed.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from .diagnostics import CompilerError
from .snn_ir import validate_input
from .snn_planner import plan_snn, validate_snn_plan

def validate_quantization(config):
    from .quantization import QuantizationConfig
    if not isinstance(config,QuantizationConfig) or not isinstance(config.cuts,dict):
        raise CompilerError('SNN requires QuantizationConfig with a cuts dictionary',code='QUANTIZATION')
    if config.cuts:
        raise CompilerError('SNN cuts belong to LIF readout_cut; affine current is uncut int32. Use cuts={}',code='QUANTIZATION')
    try:
        scales_valid = all(not isinstance(v,(bool,np.bool_)) and np.ndim(v) == 0 and np.isfinite(v) and v > 0
                           for v in (config.weight_scale,config.bias_scale))
    except (TypeError,ValueError): scales_valid = False
    if config.parameter_mode not in ('integer','scaled') or not scales_valid:
        raise CompilerError('Invalid explicit SNN parameter quantization',code='QUANTIZATION')

@dataclass
class SNNCompilation:
    model: object
    quantization: object
    plan: object
    input_data: object
    map_config: dict
    schedule: list
    reference_outputs: dict
    behavior_outputs: object = None

    def manifest(self):
        return {'schema_version':3,'scope':'SNN single-core deterministic temporal compute template',
                'model_name':self.model.model_name,'input_shape':list(self.model.input_shape),
                'time_steps':len(self.input_data),'dt':self.model.dt,'input_encoding':self.model.input_encoding,
                'address_unit':'4-byte word','layout':'C32 vectors / HWC C32 spatial tensors',
                'threshold_comparison':'>','update_order':['integrate','threshold','reset','leak','refractory','window'],
                'state_reset_boundary':'sample','hardware_temporal_driver':'NOT implemented; PI template is one frame',
                'static_validation':'passed','primitive_behavior_verification':'passed' if self.behavior_outputs is not None else 'not run',
                'router_timing_or_chip_verification':'not run','allocation':asdict(self.plan),
                'lif_contracts':{b.name:asdict(b.lif) for b in self.model.blocks},
                'states':[asdict(s) for s in self.model.states], 'schedule':self.schedule}

def compile_snn(model, config, frames):
    from .snn_reference import run_snn_reference
    from .snn_codegen import build_snn_mapping
    x = validate_input(model,frames); validate_quantization(config)
    plan = plan_snn(model)
    reference = run_snn_reference(model,config,x)
    mapping,schedule = build_snn_mapping(model,config,plan,x[0])
    return SNNCompilation(model,config,plan,x,mapping,schedule,reference)

def compile_onnx_snn(path, config, frames):
    from .snn_parser import parse_onnx_snn
    return compile_snn(parse_onnx_snn(path),config,frames)

def verify_snn_behavior(compilation):
    from generator.behavior_simulator.memory import Memory
    from .snn_codegen import pack_snn_activation, unpack_snn
    validate_snn_plan(compilation.model,compilation.plan)
    compilation.behavior_outputs = None
    # p08 mutates window/LFSR counters; never contaminate the exported initial PI.
    prims = deepcopy(compilation.map_config[((0,0),0)][0][((0,0),(0,0))]['prims'])
    memory = Memory()
    for p in prims[1:]:
        for slot in ('axon','soma1'):
            for init in p[slot].memory_blocks:
                memory.addMemoryBlock(init['start'],len(init['data']),init['data'])
    traces = {key:[] for key in compilation.reference_outputs}
    for t,frame in enumerate(compilation.input_data):
        packed = pack_snn_activation(frame)
        memory.addMemoryBlock(compilation.plan.input_addr,len(packed),packed)
        for b,p in zip(compilation.model.blocks,prims[1:]):
            for slot in ('axon','soma1'):
                op = p[slot]
                memory.writeMem(op.prim_execution(memory.read_memory(op.getInfoList())))
            memory.updateMemory()
            a = compilation.plan.outputs[b.output]; s = compilation.plan.states[b.name]
            raw = memory.read_memory([{'start':a['address'],'length':a['words'],'type':1}])[0]
            vpacked = np.asarray(memory.read_memory([{'start':s['membrane_addr'],
                      'length':s['membrane_words'],'type':0}])[0],dtype=np.int64).reshape(-1)
            v = vpacked & ((1<<28)-1); v = np.where(v >= (1<<27),v-(1<<28),v)
            ref = (vpacked & 0xffffffff) >> 28
            shape = b.affine.output_shape
            actual = {b.output:unpack_snn(raw,shape), b.name+'.membrane':unpack_snn(v,shape),
                      b.name+'.refractory':unpack_snn(ref,shape), b.name+'.window':np.asarray(p['soma1'].Tw_cnt)}
            for key,value in actual.items():
                expected = compilation.reference_outputs[key][t]
                if not np.array_equal(value,expected):
                    raise CompilerError(f'SNN behavior mismatch at t={t}, trace={key}, max_abs_diff={int(np.abs(value-expected).max())}',
                                        code='VERIFICATION',node=b.name)
                traces[key].append(value.copy())
    result = {key:np.stack(values) for key,values in traces.items()}
    compilation.behavior_outputs = result
    return result

def export_snn_artifacts(compilation, directory):
    import onnx
    from .compiler import hardware_json
    directory = Path(directory)
    if directory.exists():
        raise CompilerError('Choose a new SNN output directory',code='OUTPUT')
    # Serializer is map-only; does not run TestEngine, exe or cleanup.
    encoded = json.dumps(hardware_json(compilation),indent=2,ensure_ascii=False)
    model = onnx.load(compilation.model.source_path)
    directory.mkdir(parents=True,exist_ok=False)
    onnx.save(model,str(directory/'model.onnx'))
    np.save(directory/'input.npy',compilation.input_data)
    (directory/'quantization.json').write_text(json.dumps(asdict(compilation.quantization),indent=2),encoding='utf-8')
    (directory/'manifest.json').write_text(json.dumps(compilation.manifest(),indent=2),encoding='utf-8')
    (directory/'hardware_config.json').write_text(encoded,encoding='utf-8')
    np.savez(directory/'reference_outputs.npz',**compilation.reference_outputs)
    if compilation.behavior_outputs is not None:
        np.savez(directory/'behavior_outputs.npz',**compilation.behavior_outputs)
    source = '''# One-frame PI template; Python verifier feeds frames and preserves state.
from pathlib import Path
import json
import numpy as np
from onnx2prim.snn_compiler import compile_onnx_snn
from onnx2prim.quantization import QuantizationConfig
here=Path(__file__).resolve().parent
q=QuantizationConfig(**json.loads((here/'quantization.json').read_text(encoding='utf-8')))
compilation=compile_onnx_snn(here/'model.onnx',q,np.load(here/'input.npy',allow_pickle=False))
map_config=compilation.map_config
'''
    (directory/'mapping.py').write_text(source,encoding='utf-8')
    return directory.resolve()
