"""Portable data snapshots and map-only templates for host-staged execution.

Stage configurations contain zero dynamic inputs. They need the temporal host
controller for gathers, state persistence, reductions and window counters.
"""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict, fields
import io
from pathlib import Path
import json
import re
from types import SimpleNamespace

import numpy as np

from .diagnostics import CompilerError
from .ir import ConvLayer, FCLayer
from .temporal_ir import P08Contract, TemporalGraph, TemporalNode


def _json_data(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise CompilerError('Only numeric arrays may be exported', code='OUTPUT')
        return _json_data(value.tolist())
    if isinstance(value, np.generic):
        return _json_data(value.item())
    if isinstance(value, (tuple, list)):
        return [_json_data(v) for v in value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise CompilerError('JSON data keys must be strings', code='OUTPUT')
        return {k: _json_data(v) for k, v in value.items()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise CompilerError('Unsupported export data type: ' + type(value).__name__, code='OUTPUT')


def _encoded(value):
    return json.dumps(_json_data(value), indent=2, ensure_ascii=False, allow_nan=False)


def _numeric(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'biuf' or not np.isfinite(a).all():
        raise CompilerError('Artifacts require finite numeric arrays without pickle', code='OUTPUT')
    return a.copy()


def _snapshot(graph):
    weights, records = {}, []
    for n in graph.ordered():
        layer = None
        if n.layer is not None:
            if not isinstance(n.layer, (ConvLayer, FCLayer)):
                raise CompilerError('Unsupported temporal layer snapshot', node=n.name)
            layer = {'type': 'ConvLayer' if isinstance(n.layer, ConvLayer) else 'FCLayer',
                     'fields': {f.name: getattr(n.layer, f.name) for f in fields(n.layer)
                                if f.name not in ('weights', 'bias')}}
            for key in ('weights', 'bias'):
                raw = getattr(n.layer, key)
                ident = None
                if raw is not None:
                    ident = f'p{len(weights):06d}'
                    weights[ident] = _numeric(raw)
                layer[key] = ident
        neuron = None
        if n.neuron is not None:
            from .temporal_neurons import NeuronContract
            if not isinstance(n.neuron, (P08Contract, NeuronContract)):
                raise CompilerError('Unsupported temporal neuron snapshot', node=n.name)
            neuron = {'type': 'P08Contract' if isinstance(n.neuron, P08Contract) else 'NeuronContract',
                      'fields': asdict(n.neuron)}
        records.append({'name': n.name, 'op': n.op, 'inputs': n.inputs, 'output': n.output,
                        'shape': n.shape, 'dtype': n.dtype, 'attrs': n.attrs,
                        'layer': layer, 'neuron': neuron})
    return {'schema_version': 1, 'type': 'TemporalGraph', 'input_name': graph.input_name,
            'input_shape': graph.input_shape, 'input_dtype': graph.input_dtype,
            'output_name': graph.output_name, 'dt': graph.dt, 'readout': graph.readout,
            'model_name': graph.model_name, 'metadata': graph.metadata, 'nodes': records}, weights


def load_temporal_graph(model_path, weights_path):
    """Load a whitelisted typed graph; names are data and never paths or code."""
    try:
        data = json.loads(Path(model_path).read_text(encoding='utf-8'))
        if data.pop('schema_version') != 1 or data.pop('type') != 'TemporalGraph':
            raise ValueError('Unknown graph snapshot schema')
        used, nodes = set(), []
        with np.load(weights_path, allow_pickle=False) as archive:
            def param(ident):
                if ident is None:
                    return None
                if not isinstance(ident, str) or not re.fullmatch(r'p[0-9]{6}', ident):
                    raise ValueError('Unsafe parameter ID')
                used.add(ident)
                return _numeric(archive[ident])

            for record in data.pop('nodes'):
                record = dict(record)
                layer = record.pop('layer')
                if layer is not None:
                    layer = dict(layer)
                    kind = layer.pop('type')
                    cls = {'ConvLayer': ConvLayer, 'FCLayer': FCLayer}.get(kind)
                    if cls is None:
                        raise ValueError('Unknown layer snapshot type')
                    kwargs = dict(layer.pop('fields'))
                    if 'feature_shape' in kwargs and kwargs['feature_shape'] is not None:
                        kwargs['feature_shape'] = tuple(kwargs['feature_shape'])
                    kwargs['weights'], kwargs['bias'] = param(layer.pop('weights')), param(layer.pop('bias'))
                    if layer:
                        raise ValueError('Unknown layer snapshot fields')
                    layer = cls(**kwargs)
                neuron = record.pop('neuron')
                if neuron is not None:
                    if set(neuron) != {'type', 'fields'}:
                        raise ValueError('Unknown neuron snapshot fields')
                    kwargs = dict(neuron['fields'])
                    if neuron['type'] == 'P08Contract':
                        neuron = P08Contract(**kwargs)
                    elif neuron['type'] == 'NeuronContract':
                        from .temporal_neurons import NeuronContract
                        from .snn_ir import LIFSpec
                        if kwargs.get('lif_spec') is not None:
                            kwargs['lif_spec'] = LIFSpec(**kwargs['lif_spec'])
                        neuron = NeuronContract(**kwargs)
                    else:
                        raise ValueError('Unknown neuron snapshot type')
                    neuron.validate()
                record['inputs'], record['shape'] = tuple(record['inputs']), tuple(record['shape'])
                nodes.append(TemporalNode(**record, layer=layer, neuron=neuron))
            if set(archive.files) != used or len(archive.files) != len(used):
                raise ValueError('Unused or duplicate parameter archive entries')
        data['input_shape'] = tuple(data['input_shape'])
        graph = TemporalGraph(**data, nodes=nodes)
        graph.ordered()
        return graph
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CompilerError('Invalid temporal snapshot: ' + str(exc), code='INPUT') from exc


def save_temporal_graph(graph, new_dir):
    """Save model.json and weights.npz in a fresh directory for portable fixtures."""
    directory = Path(new_dir)
    if directory.exists():
        raise CompilerError('Choose a fresh temporal snapshot directory', code='OUTPUT')
    snapshot, weights = _snapshot(graph)
    encoded = _encoded(snapshot)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'model.json').write_text(encoded, encoding='utf-8')
    np.savez(directory / 'weights.npz', **weights)
    return directory.resolve()


def _stages(compilation):
    """Tuple-keyed reductions become ordered records, never JSON dict keys."""
    stages = []
    known = set()
    for node in compilation.graph.ordered():
        members = []
        for tile in compilation.kernels[node.name]:
            ident = f'stage{len(stages):06d}'
            stages.append((ident, node, tile, None, []))
            members.append((ident, tile))
        reductions = [(key, tile) for (name, key), tile in compilation.reductions.items() if name == node.name]
        for key, tile in sorted(reductions, key=lambda item: item[0]):
            known.add((node.name, key))
            sources = [ident for ident, t in members
                       if (t.attrs.get('co'), t.attrs.get('h'), t.attrs.get('w')) == key]
            if len(sources) != len(tile.inputs):
                raise CompilerError('Partial reduction inputs do not match affine stages', node=node.name)
            stages.append((f'stage{len(stages):06d}', node, tile, key, sources))
    if known != set(compilation.reductions):
        raise CompilerError('Reduction references unknown temporal node', code='OUTPUT')
    return stages


def _zero_mapping(tile):
    from .temporal_kernels import pack
    template = deepcopy(tile)
    first = template.operations[0]
    first.memory_blocks = list(first.memory_blocks)
    for index, (address, dtype, shape) in enumerate(template.inputs):
        first.memory_blocks.append({'name': f'dynamic_input{index:06d}', 'start': address,
                                    'data': pack(np.zeros(shape, dtype=np.int64), dtype)})
    return template.mapping()


_MAPPING_SOURCE = '''# Portable typed builder; host-staged compute, not chip deployment.
from pathlib import Path
import json
import numpy as np
from onnx2prim.temporal_export import load_temporal_graph, _stages, _zero_mapping
from onnx2prim.temporal_compiler import compile_temporal
from onnx2prim.quantization import QuantizationConfig
here = Path(__file__).resolve().parent
graph = load_temporal_graph(here / 'model.json', here / 'weights.npz')
quant = QuantizationConfig(**json.loads((here / 'quant.json').read_text(encoding='utf-8')))
compilation = compile_temporal(graph, quant, np.load(here / 'input.npy', allow_pickle=False))
stage_map_configs = {ident: _zero_mapping(tile) for ident, node, tile, group, sources in _stages(compilation)}
host_staging_required = True
'''


def export_temporal_artifacts(compilation, directory):
    """Export into a fresh directory without accessing original source files."""
    from .compiler import hardware_json
    directory = Path(directory)
    if directory.exists():
        raise CompilerError('Choose a fresh temporal output directory', code='OUTPUT')
    snapshot, weights = _snapshot(compilation.graph)
    manifest = compilation.manifest()
    stages = []
    encoded_stages = {}
    shapes = {compilation.graph.input_name: compilation.graph.input_shape}
    shapes.update({n.output: n.shape for n in compilation.graph.ordered()})
    for ident, node, tile, group, sources in _stages(compilation):
        filename = 'stage_configs/' + ident + '.json'
        # The original map-only serializer prints every initial memory region.
        # Keep that diagnostic chatter out of normal artifact export output.
        with redirect_stdout(io.StringIO()):
            hardware = hardware_json(SimpleNamespace(map_config=_zero_mapping(tile)))
        encoded_stages[filename] = _encoded(hardware)
        source_stages = {s['id']: s for s in stages}
        source_shapes = [source_stages[s]['shape'] for s in sources] if sources else [shapes[v] for v in node.inputs]
        stages.append({'id': ident, 'node': node.name, 'kernel_id': tile.id, 'kind': tile.kind,
            'hardware_config': filename, 'core': tile.core,
            'state_core': tile.core if tile.kind == 'lif' else None,
            'shape': tile.shape, 'dtype': tile.dtype, 'output_address': tile.output_addr,
            'PICs': [op.PIC for op in tile.operations], 'attributes': tile.attrs,
            'initial_memory_load_boundary': 'sample_start' if tile.kind == 'lif' else 'per_stage',
            'host_window_control_required': isinstance(node.neuron, P08Contract),
            'software_extended_window': isinstance(node.neuron, P08Contract) and node.neuron.window_length > 255,
            'window_control': {'owner': 'host', 'window_enabled': node.neuron.window_enabled,
                'window_length': node.neuron.window_length, 'window_initial': node.neuron.window_initial,
                'Tw_cnt': 0, 'Tw_len': 'host selects 0 or 1 before every frame'}
                if isinstance(node.neuron, P08Contract) else None,
            'transfer': {'source_values': sources or node.inputs, 'source_shapes': source_shapes,
                'source_kind': 'partial_stage_outputs' if sources else 'graph_values',
                'node_inputs': node.inputs, 'full_source_shapes': [shapes[v] for v in node.inputs],
                'gather_attributes': tile.attrs, 'operator_attributes': node.attrs,
                'inputs': [{'address': address, 'dtype': dtype, 'shape': shape}
                           for address, dtype, shape in tile.inputs],
                'partial_source_stages': sources, 'reduction_group': group,
                'bias_once': tile.attrs.get('bias_once', False),
                'host_load_required': True,
                'dynamic_input_initialization': 'zero', 'host_staging_required': True},
            'standalone_meaningful_execution': False})
    manifest.update({'host_staging_required': True, 'autonomous_chip_deployment': 'NOT implemented',
                     'stage_configs_are_zero_input_templates': True,
                     'stage_index': stages, 'snapshot': {'model': 'model.json', 'weights': 'weights.npz',
                     'input': 'input.npy', 'quantization': 'quant.json'},
                     'software_window_controller_required': any(
                         isinstance(n.neuron, P08Contract) for n in compilation.graph.nodes),
                     'host_window_control_required': any(
                         isinstance(n.neuron, P08Contract) for n in compilation.graph.nodes)})
    for n in manifest['nodes']:
        n['stage_ids'] = [s['id'] for s in stages if s['node'] == n['name']]
        neuron = next(node.neuron for node in compilation.graph.nodes if node.name == n['name'])
        n['host_window_control_required'] = isinstance(neuron, P08Contract)
        n['software_extended_window'] = isinstance(neuron, P08Contract) and neuron.window_length > 255
    arrays = {}
    manifest['output_archives'] = {}
    for label, traces in (('reference_outputs', compilation.reference_outputs),
                          ('behavior_outputs', compilation.behavior_outputs)):
        if traces is None:
            continue
        arrays[label] = {f'a{i:06d}': _numeric(value) for i, value in enumerate(traces.values())}
        manifest['output_archives'][label] = {'file': label + '.npz',
            'keys': {f'a{i:06d}': name for i, name in enumerate(traces)}}
    input_data = _numeric(compilation.input_data)
    # Validate and serialize before creating the output directory.
    documents = {'model.json': _encoded(snapshot), 'quant.json': _encoded(asdict(compilation.quantization)),
                 'manifest.json': _encoded(manifest), **encoded_stages}
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'stage_configs').mkdir()
    for filename, encoded in documents.items():
        (directory / filename).write_text(encoded, encoding='utf-8')
    np.savez(directory / 'weights.npz', **weights)
    np.save(directory / 'input.npy', input_data, allow_pickle=False)
    for label, archive in arrays.items():
        np.savez(directory / (label + '.npz'), **archive)
    (directory / 'mapping.py').write_text(_MAPPING_SOURCE, encoding='utf-8')
    return directory.resolve()
