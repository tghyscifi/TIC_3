"""Typed, conservative ONNX -> single-core compute primitive pipeline."""
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import json
import numpy as np
from .diagnostics import CompilerError, UnsupportedModelError
from .ir import ConvLayer
from .quantization import QuantizationConfig, integer_parameters
from .reference import run_reference
from .scheduler import plan_blocks
from .validate import validate_sequence, validate_addresses

@dataclass
class Compilation:
    sequence: object
    quantization: QuantizationConfig
    assignments: list
    input_data: object
    map_config: dict
    reference_outputs: list
    behavior_outputs: object = None

    def manifest(self):
        return {'schema_version':1,'scope':'ANN single-core compute only',
                'model_name':self.sequence.model_name,'input_shape':list(self.sequence.input_shape),
                'parameter_mode':self.quantization.parameter_mode,
                'cuts':dict(self.quantization),'address_unit':'4-byte word',
                'static_validation':'passed',
                'primitive_behavior_verification':'passed' if self.behavior_outputs is not None else 'not run',
                'chip_or_exe_verification':'not run',
                'allocations':[asdict(a) for a in self.assignments]}

def compile_sequence(seq, config, input_data):
    """Input is already int8 CHW. Float-input calibration is out of scope."""
    from .layout import pack_activation, pack_weights, pack_bias
    from .address_allocator import allocate_addresses
    from .codegen.runtime import compile_compute
    blocks = validate_sequence(seq)
    config.validate(seq)
    x = np.asarray(input_data)
    if x.shape != tuple(seq.input_shape) or x.dtype.kind not in 'iu' or np.any(x < -128) or np.any(x > 127):
        raise CompilerError('Input must be integer int8-range CHW matching model shape',code='INPUT')
    assignments, _, _ = allocate_addresses(seq)
    validate_addresses(seq, assignments)
    packed = []
    for b in blocks:
        w, bias = integer_parameters(b.layer, config)
        layer = replace(b.layer, weights=w, bias=bias)
        packed.append({'x':pack_activation(x,first_input=True) if b.index == 0 else None,
                       'w':pack_weights(layer,first_conv=b.index == 0),
                       'b':pack_bias(bias,layer.cout)})
    # Check logical accumulator overflow before executing target operations.
    reference = run_reference(seq, config, x)
    mapping = compile_compute(seq,assignments,config,packed)
    mapping['sim_clock'] = 100000
    mapping['step_clock'] = {((0,0),0):(99999,100000)}
    return Compilation(seq,config,assignments,x.copy(),mapping,reference)

def compile_onnx(path, config, input_data):
    from .onnx_parser import parse_onnx
    return compile_sequence(parse_onnx(path),config,input_data)

def compile_ir(model, config, input_data):
    """Dispatch explicit SNNModel; arbitrary legacy states/time are not dropped."""
    from .snn_ir import SNNModel
    if isinstance(model,SNNModel):
        from .snn_compiler import compile_snn
        return compile_snn(model,config,input_data)
    from .ir import ModelIR
    if not isinstance(model,ModelIR):
        raise CompilerError('compile_ir expects ModelIR')
    if model.states or model.time is not None:
        raise UnsupportedModelError('Use explicit SNNModel/p08 contract; arbitrary stateful ModelIR cannot be lowered')
    return compile_sequence(model.sequence,config,input_data)

def verify_behavior(compilation):
    """Execute original Axon+Soma+Memory implementation without deleting outputs.

    This verifies phase-level primitive behavior, not router/timing or hardware.
    """
    from generator.behavior_simulator.memory import Memory
    compilation.behavior_outputs = None
    memory = Memory()
    prims = compilation.map_config[((0,0),0)][0][((0,0),(0,0))]['prims']
    for prim in prims:
        for slot in ('axon','soma1'):
            operation = prim.get(slot)
            if operation is not None:
                for init in operation.memory_blocks:
                    if init.get('initialize',True):
                        data = init['data']
                        memory.addMemoryBlock(init['start'],len(data),data)
    result = []
    for b,a,prim,expected in zip(plan_blocks(compilation.sequence),compilation.assignments,prims[1:],compilation.reference_outputs):
        for slot in ('axon','soma1'):
            op = prim[slot]
            memory.writeMem(op.prim_execution(memory.read_memory(op.getInfoList())))
        memory.updateMemory()
        raw = np.asarray(memory.read_memory([{'start':a.output_addr,'length':a.output_words,'type':1}])[0],dtype=np.int64).reshape(-1)
        if len(b.output_shape) == 3:
            c,h,w = b.output_shape
            padded_c = (c+15)//16*16
            actual = raw.reshape(h,w,padded_c)[:,:,:c].transpose(2,0,1)
        else:
            actual = raw[:b.output_shape[0]]
        if not np.array_equal(actual,expected):
            delta = np.abs(actual-expected)
            raise CompilerError(f'Behavior mismatch at block {b.index}, max_abs_diff={int(delta.max())}',node=b.layer.name,code='VERIFICATION')
        result.append(actual.copy())
    compilation.behavior_outputs = result
    return result

def hardware_json(compilation):
    """Serialize through original serializer; no exe execution or file cleanup."""
    from generator.behavior_simulator import Simulator
    from generator.code_generator import MapConfig
    from generator.test_engine.test_config import HardwareDebugFileSwitch
    simulator = Simulator(MapConfig(compilation.map_config),[])
    simulator.map()
    # Existing Simulator has no public map-only serializer facade.
    config = simulator._chip_array.config_json_output('onnx2prim_prototype')
    config['debug_file_switch'] = HardwareDebugFileSwitch().close_all.dict
    return config

def export_artifacts(compilation, directory):
    """Create a fresh self-contained run directory; never overwrite prior runs."""
    import onnx
    directory = Path(directory)
    if directory.exists():
        raise CompilerError('Output directory already exists; choose a new run name',code='OUTPUT')
    # Precompute JSON before creating any output if serialization fails.
    encoded = json.dumps(hardware_json(compilation),indent=2,ensure_ascii=False)
    model = onnx.load(compilation.sequence.source_path)
    directory.mkdir(parents=True,exist_ok=False)
    onnx.save(model,str(directory/'model.onnx'))
    np.save(directory/'input.npy',compilation.input_data)
    config = {'cuts':dict(compilation.quantization),
              'parameter_mode':compilation.quantization.parameter_mode,
              'weight_scale':compilation.quantization.weight_scale,
              'bias_scale':compilation.quantization.bias_scale}
    (directory/'quantization.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (directory/'hardware_config.json').write_text(encoded,encoding='utf-8')
    (directory/'manifest.json').write_text(json.dumps(compilation.manifest(),indent=2),encoding='utf-8')
    np.savez(directory/'reference_outputs.npz',**{f'block{i}':x for i,x in enumerate(compilation.reference_outputs)})
    if compilation.behavior_outputs is not None:
        np.savez(directory/'behavior_outputs.npz',**{f'block{i}':x for i,x in enumerate(compilation.behavior_outputs)})
    source = '''# Generated typed-builder entry. Compute-only, not a deployment script.
from pathlib import Path
import json
import numpy as np
from onnx2prim.compiler import compile_onnx
from onnx2prim.quantization import QuantizationConfig
here = Path(__file__).resolve().parent
q = json.loads((here / 'quantization.json').read_text(encoding='utf-8'))
compilation = compile_onnx(here / 'model.onnx', QuantizationConfig(**q), np.load(here / 'input.npy', allow_pickle=False))
map_config = compilation.map_config
'''
    (directory/'mapping.py').write_text(source,encoding='utf-8')
    return directory.resolve()
