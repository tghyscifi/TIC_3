"""Full-size, explicit integer project DAG presets, never training imports.

Parameters use OIHW convolution and OI dense arrays. LeNet fc1 columns are
source HWC and are converted to CHW. HNN callers transpose native I/O matrices
to this API's O/I convention. Float checkpoints and unfused BN are rejected.
SHD defaults are a synthetic selector fixture, NOT verified legacy routing.
"""
from collections.abc import Mapping
import math
import numpy as np

from .diagnostics import CompilerError, UnsupportedModelError
from .ir import ConvLayer, FCLayer
from .temporal_ir import P08Contract, TemporalGraph, TemporalNode
from .temporal_neurons import NeuronContract


MODEL_IDS = ('SoundSNN', 'snn', 'snn_mnist_s', 'full', 'snnlenet', 'HNN',
             'SNN_forward', 'shd', 'snnlenet/train', 'snnlenet/train_lif_float',
             'snnlenet/snnpredict')
FLOAT_QUANTIZATION_POLICY = 'integer_cuts_4_4_5_4_5'
_SOURCES = {
    'SoundSNN': 'generator/SoundSNN/sound_snn.py', 'snn': 'snn.py',
    'snn_mnist_s': 'snn_mnist_s.py', 'full': 'generator/full.py',
    'full/mapping': 'generator/full.py', 'snnlenet': 'generator/snnlenet/snnlenet.py',
    'HNN': 'generator/test/beijingshi/HNN.py',
    'SNN_forward': 'generator/test/beijingshi/SNN_forward.py',
    'shd': 'generator/shd/shd_hidden.py',
    'snnlenet/train': 'generator/snnlenet/train.py',
    'snnlenet/train_lif_float': 'generator/snnlenet/train_lif_float.py',
    'snnlenet/snnpredict': 'generator/snnlenet/snnpredict.py',
}


def _integer_array(value, shape, name, low, high):
    a = np.asarray(value)
    if a.shape != tuple(shape):
        raise CompilerError(f'{name} shape must be {tuple(shape)}, got {a.shape}', code='PARAMETERS')
    if a.dtype.kind not in 'iu' or np.any(a < low) or np.any(a > high):
        raise CompilerError(f'{name} requires declared integers in [{low},{high}]', code='PARAMETERS')
    return a.astype(np.int64, copy=True)


class _Builder:
    def __init__(self, model_id, shape, seed, parameters):
        self.rng = np.random.default_rng(seed)
        self.parameters = {} if parameters is None else parameters
        self.supplied = parameters is not None
        self.used = set()
        self.nodes = []
        self.current, self.shape = 'input', tuple(shape)
        self.graph = TemporalGraph('input', tuple(shape), '', self.nodes,
                                   model_name=model_id, source_path=_SOURCES[model_id])
        self.graph.metadata = {
            'model_id': model_id, 'seed': seed,
            'parameter_provenance': 'deterministic_synthetic_integer_weights' if parameters is None else 'caller_supplied_integer_parameters',
            'checkpoint_loaded': False, 'lossless_float_conversion': False,
            'parameter_layout': 'Conv OIHW; FC OI; LeNet fc1 source HWC columns converted to CHW',
            'synthetic_weight_range': [-2, 2], 'synthetic_bias_range': [-2, 2],
            'coverage': {'full_size_compute_dag': True, 'legacy_routing_verified': False,
                         'source_model_vs_mapping_conflict': False},
            'approximations': [], 'parameter_layers': {},
        }

    def parameter(self, name, shape, *, ignored_bias=False):
        self.used.add(name)
        if name in self.parameters:
            entry = self.parameters[name]
            if not isinstance(entry, Mapping) or set(entry) != {'weight', 'bias'}:
                raise CompilerError(f'{name} requires exactly weight and bias', code='PARAMETERS')
            weight = _integer_array(entry['weight'], shape, name + '.weight', -128, 127)
            bias = _integer_array(entry['bias'], (shape[0],), name + '.bias', -(1 << 31), (1 << 31) - 1)
            provenance = 'caller_supplied'
        elif self.supplied:
            raise CompilerError(f'Missing parameter layer {name}', code='PARAMETERS')
        else:
            weight = self.rng.integers(-2, 3, shape, dtype=np.int64)
            bias = self.rng.integers(-2, 3, shape[0], dtype=np.int64)
            provenance = 'synthetic'
        if ignored_bias:
            bias = np.zeros(shape[0], dtype=np.int64)
        self.graph.metadata['parameter_layers'][name] = {
            'provenance': provenance, 'source_weight_shape': list(shape),
            'bias_ignored': ignored_bias,
        }
        return weight, bias

    def node(self, name, op, shape=None, dtype='int8', *, inputs=None, attrs=None, layer=None, neuron=None):
        shape = self.shape if shape is None else tuple(shape)
        output = name + '__out'
        self.nodes.append(TemporalNode(name, op, tuple(inputs or (self.current,)), output,
                                       tuple(shape), dtype, attrs or {}, layer, neuron))
        self.current, self.shape = output, tuple(shape)
        return output

    def conv(self, name, cout, *, stride=1, padding=0):
        cin, h, w = self.shape
        weight, bias = self.parameter(name, (cout, cin, 5, 5))
        layer = ConvLayer(name, cin, cout, w, h, 5, 5, sx=stride, sy=stride,
                          pad_top=padding, pad_down=padding, pad_left=padding,
                          pad_right=padding, weights=weight, bias=bias,
                          source_name=self.graph.source_path)
        self.node(name, 'Conv', layer.output_shape, 'int32', layer=layer)

    def fc(self, name, cout, *, hwc=False, ignored_bias=False):
        shape = self.shape
        weight, bias = self.parameter(name, (cout, math.prod(shape)), ignored_bias=ignored_bias)
        if hwc:
            c, h, w = shape
            weight = weight.reshape(cout, h, w, c).transpose(0, 3, 1, 2).reshape(cout, -1).copy()
            self.graph.metadata['parameter_layers'][name]['column_conversion'] = 'source HWC -> target CHW'
        layer = FCLayer(name, math.prod(shape), cout, weights=weight, bias=bias,
                        feature_shape=shape if len(shape) == 3 else None,
                        input_layout='CHW' if len(shape) == 3 else 'VECTOR',
                        source_name=self.graph.source_path)
        self.node(name, 'FC', layer.output_shape, 'int32', layer=layer)

    def lif(self, name, contract):
        dtype = 'int2' if isinstance(contract, P08Contract) and contract.fire_type == 5 else 'int8'
        self.node(name, 'LIF', dtype=dtype, neuron=contract)

    def cut_relu(self, cut_name, amount, relu_name=None):
        self.node(cut_name, 'Cut', attrs={'cut': amount})
        if relu_name:
            self.node(relu_name, 'Relu', attrs={'cut': 0})

    def pool(self, name):
        c, h, w = self.shape
        self.node(name, 'MaxPool', (c, h // 2, w // 2), attrs={'kx': 2, 'ky': 2, 'sx': 2, 'sy': 2})

    def finish(self):
        unknown = set(self.parameters) - self.used - ({'shd_selectors'} if self.graph.model_name == 'shd' else set())
        if unknown:
            raise CompilerError(f'Unknown parameter layers: {sorted(unknown)}', code='PARAMETERS')
        self.graph.output_name = self.current
        self.graph.ordered()
        self.graph.metadata['coverage'].update(
            input_shape=list(self.graph.input_shape), output_shape=list(self.shape),
            node_count=len(self.nodes), affine_count=sum(n.op in ('Conv', 'FC') for n in self.nodes),
            neuron_count=sum(n.op == 'LIF' for n in self.nodes))
        return self.graph


def _reject_bn(parameters):
    if parameters is None:
        return
    for name in parameters:
        lower = name.lower()
        entry = parameters[name]
        if ('bn' in lower or 'batch_norm' in lower or 'running_' in lower
                or isinstance(entry, Mapping) and any('running_' in str(k) for k in entry)):
            raise UnsupportedModelError('Unfused BatchNorm/checkpoint BN parameters require explicit folding; only batch_norm=False integer variant is supported')


def _shd(b):
    """Four paired-core branches; native (64,192) weights fold into (64,700).

    -1 selector denotes zero padding. Duplicate indices accumulate weights.
    No implicit slice and no claim that these selectors reproduce p09 routing.
    """
    supplied = b.parameters.get('shd_selectors')
    if supplied is None:
        if b.parameters:
            raise UnsupportedModelError('SHD needs inspection: supplied parameters require explicit shd_selectors (4,2,192); legacy router lookup is not verified')
        # Explicit synthetic fixture only, not contiguous slices inferred from routing.
        selectors = b.rng.integers(0, 700, (4, 2, 192), dtype=np.int64)
        selectors[:, :, -16:] = -1
        selector_origin = 'declared_synthetic_selectors'
    else:
        selectors = _integer_array(supplied, (4, 2, 192), 'shd_selectors', -1, 699)
        selector_origin = 'caller_supplied_selectors_not_source_verified'
    b.graph.input_dtype = 'int2'
    b.graph.metadata.update(
        legacy_router_equivalence='not_verified', selector_provenance=selector_origin,
        selector_indices=selectors.tolist(), zero_padding_index=-1,
        source_paths=['generator/shd/shd_top.py', 'generator/shd/shd_hidden.py', 'generator/shd/shd_output.py'],
        selector_inspection={
            'status': 'needs_inspection',
            'source_facts': 'shd_hidden: shuffle addr_diff=0x8, p06 length=892 after router A=175 append; branch starts 0x0+core_y*0x8, 0x100, 0x200, 0x40+core_y*0x8; p06 transfer starts 0x8/0x14+core_y*0x8, receive 0x8404; each p41 cin=192 cout=32',
            'unresolved': 'p09 t_mode=1 lookup, packed ternary address units and router/p06 transfer equivalence are not proven',
        })
    outputs = []
    for branch in range(4):
        name = f'branch{branch + 1}'
        weight, bias = b.parameter(name, (64, 192))
        folded = np.zeros((64, 700), dtype=np.int64)
        for core in range(2):
            indices = selectors[branch, core]
            keep = indices >= 0
            for row in range(core * 32, (core + 1) * 32):
                np.add.at(folded[row], indices[keep], weight[row, keep])
        if np.any(folded < -128) or np.any(folded > 127):
            raise CompilerError('Folded SHD selector weight exceeds int8; duplicates must not silently clip', code='PARAMETERS')
        layer = FCLayer(name, 700, 64, weights=folded, bias=bias, source_name=b.graph.source_path)
        b.node(name, 'FC', (64,), 'int32', inputs=('input',), layer=layer)
        b.lif(name + '_lif', P08Contract(threshold=64, leak_alpha=128, floor=0,
                                        fire_type=2, readout_cut=2,
                                        window_enabled=True, window_length=1000))
        outputs.append(b.current)
    b.node('branch_sum', 'Add', (64,), 'int32', inputs=tuple(outputs))
    b.lif('soma', P08Contract(threshold=64, leak_alpha=128, floor=0, fire_type=5,
                             readout_cut=6, window_enabled=True, window_length=1000))
    b.fc('fc_out', 32)
    initial = b.rng.integers(0, 0x999A, (32,), dtype=np.int64)
    b.lif('output_lif', P08Contract(threshold=60000, leak_alpha=128, floor=0,
                                   reset_value=1000, initial_v=initial,
                                   fire_type=2, readout_cut=11,
                                   window_enabled=True, window_length=1000))
    b.graph.metadata['initial_v_provenance'] = 'seeded_synthetic_uniform_0_through_0x9999'


def build_project_graph(model_id, seed=0, parameters=None, *, quantization_policy=None) -> TemporalGraph:
    """Build an integer compute preset with full source dimensions.

    ``parameters`` is a complete layer-name -> {weight, bias} mapping (or None
    for deterministic synthetic integers). SHD additionally accepts
    ``shd_selectors`` of shape (4,2,192), entries -1 or source columns 0..699;
    branch weights are (64,192), folding to (64,700). These indices are caller
    declarations, not certified legacy-router lookup. Partial params rejected.

    Float LeNet requires explicit FLOAT_QUANTIZATION_POLICY acknowledgement.
    No automatic checkpoint loading, BN folding, float quantization or training.
    """
    if model_id not in _SOURCES:
        raise UnsupportedModelError(f'Unknown project model_id {model_id!r}')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if parameters is not None and not isinstance(parameters, Mapping):
        raise TypeError('parameters must be a layer mapping')
    if model_id == 'snnlenet/train_lif_float':
        if quantization_policy != FLOAT_QUANTIZATION_POLICY:
            raise UnsupportedModelError('Float LeNet requires explicit quantization_policy=' + FLOAT_QUANTIZATION_POLICY + '; declared integer approximation, not lossless float conversion')
    elif quantization_policy is not None:
        raise ValueError('quantization_policy applies only to snnlenet/train_lif_float')
    if model_id.startswith('snnlenet'):
        _reject_bn(parameters)
    shape = (784,) if model_id == 'SoundSNN' else (256,) if model_id == 'HNN' else (4,) if model_id == 'SNN_forward' else (700,) if model_id == 'shd' else (1, 28, 28)
    b = _Builder(model_id, shape, seed, parameters)
    meta = b.graph.metadata
    if model_id == 'shd':
        _shd(b)
    elif model_id == 'SoundSNN':
        b.graph.readout = 'mean'
        meta['approximations'].append('Native training threshold=1.0 is declared integer threshold=64; no lossless scaling claim. Native decay=255/255 maps to alpha=256/256 exactly.')
        for i, cout in enumerate((128, 128, 128, 10), 1):
            b.fc(f'fc{i}', cout)
            b.lif(f'lif{i}', NeuronContract(profile='sound', threshold=64, beta_numerator=256))
    elif model_id in ('HNN', 'SNN_forward'):
        b.graph.readout = 'sum' if model_id == 'HNN' else 'last'
        meta['approximations'].append('Native threshold=0.3 is declared integer threshold=64; beta=0.5 is exactly 128/256. Bias argument is ignored by source neuron_block.')
        meta['native_weight_layout'] = 'I,O; caller must transpose to API O,I'
        for i, cout in enumerate((512, 512, 10) if model_id == 'HNN' else (128, 128, 128, 128, 2), 1):
            b.fc(f'fc{i}', cout, ignored_bias=True)
            b.lif(f'lif{i}', NeuronContract(profile='hnn', threshold=64, beta_numerator=128))
    elif model_id in ('snn', 'snn_mnist_s', 'full', 'full/mapping'):
        mapped = model_id == 'full/mapping'
        if not mapped:
            meta['approximations'].append('Native beta=0.9 approximated by alpha=230/256 with floor arithmetic; threshold=1.0 mapped to declared integer 64, not lossless.')
        if model_id.startswith('full'):
            meta['coverage']['source_model_vs_mapping_conflict'] = True
            meta['source_model_vs_mapping_conflict'] = 'Source SNNLeNet uses snntorch.Leaky beta=0.9; legacy mapping uses p08 threshold1/no leak/window0 and lacks conv1 padding. Mapping preset repairs spatial shape only.'
            meta['preset_role'] = 'repaired_legacy_mapping' if mapped else 'source_model_integer_approximation'
        for i, cout in enumerate((12, 32), 1):
            b.conv(f'conv{i}', cout, stride=1 if model_id == 'snn' else 2, padding=0 if model_id == 'snn' else 2)
            neuron = P08Contract(threshold=1, leak_enabled=False, floor=-10,
                                  window_enabled=False, window_length=0) if mapped else NeuronContract(
                                      profile='snntorch', threshold=64, beta_numerator=230,
                                      invocation_count=2 if model_id == 'snn' else 1)
            b.lif(f'lif{i}', neuron)
            if model_id == 'snn':
                b.pool(f'pool{i}')
        b.fc('fc1', 10)
        b.lif('lif3', P08Contract(threshold=1, leak_enabled=False, floor=-10,
                                 window_enabled=False, window_length=0) if mapped else NeuronContract(
                                     profile='snntorch', threshold=64, beta_numerator=230))
    else:
        meta.update(batch_norm=False, source_default_batch_norm=True if model_id in ('snnlenet', 'snnlenet/snnpredict') else False,
                    supported_variant='batch_norm=False, integer parameters, quantized input frames, use_lif=True',
                    source_fc1_layout='HWC', target_fc1_layout='CHW')
        if model_id == 'snnlenet/train_lif_float':
            meta['quantization_policy'] = quantization_policy
            meta['approximations'].append('Float source has no Cut nodes. Explicit policy adds cuts 4,4,5,4,5 and uses caller integer parameters; this is an approximation, not a float importer.')
        meta['input_policy'] = 'caller supplies already quantized integer frames; no automatic image scaling'
        for i, cout in enumerate((6, 16), 1):
            b.conv(f'conv{i}', cout, padding=0 if i == 1 else 1)
            b.cut_relu(f'cut{i}', 4, f'relu{i}')
            b.pool(f'pool{i}')
        b.fc('fc1', 120, hwc=True)
        b.cut_relu('fc_cut1', 5, 'fc_relu1')
        b.fc('fc2', 84)
        if model_id == 'snnlenet':
            meta['coverage']['source_model_vs_mapping_conflict'] = True
            meta['source_model_vs_mapping_conflict'] = 'Python LeNet fc_relu2 is ReLU; this hybrid preset represents the legacy fc2 p08 threshold10/fire4 mapping.'
            b.lif('fc_lif2', P08Contract(threshold=10, leak_enabled=False, floor=-10,
                                       window_enabled=False, window_length=0))
        elif model_id == 'snnlenet/snnpredict':
            b.cut_relu('fc_cut2', 4, 'fc_lif2_relu')
            meta['neuron_simplification'] = 'Exact on integer fc_cut2 inputs: beta0, threshold0, delayed zero reset, v_lower=-10 returns spk*mem=max(input,0); not binary spikes.'
        else:
            b.cut_relu('fc_cut2', 4)
            b.node('fc2_cast', 'Cast', dtype='int32', attrs={'to': 'int32'})
            b.lif('fc_lif2', P08Contract(threshold=9, leak_enabled=False, floor=-(1 << 27),
                                       window_enabled=False, window_length=0))
            meta['threshold_contract'] = 'Source integer x>=10 represented exactly by strict p08 x>9; window0 keeps each frame stateless.'
        b.fc('fc3', 10)
        b.cut_relu('fc_cut3', 5)
    return b.finish()
