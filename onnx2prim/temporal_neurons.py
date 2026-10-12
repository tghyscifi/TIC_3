"""Explicit neuron oracles, not a hardware lowering or training-model importer.

Integer beta is an explicitly chosen approximation with floor division by 256.
In particular 230/256 is NOT lossless conversion of snntorch beta=0.9.
Current already includes the affine transform (HNN's unused bias is excluded).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .snn_ir import LIFSpec
from .snn_reference import lif_step, LOW, HIGH


@dataclass(frozen=True)
class NeuronContract:
    """One invocation; thresholds/current/state share caller-selected integer units.

    invocation_count describes calls per frame, not iterations inside ``step``.
    For snn.py LIF1/2 use 2 and 'same_incoming_membrane': both calls receive
    the same old mem; the first spike is consumed and the second mem retained.
    For snn_mnist_s.py and snn.py LIF3 use the default count of 1.
    """
    profile: str = 'p08'
    threshold: int = 64
    beta_numerator: int = 128
    beta_denominator: int = 256
    reset_mode: Optional[str] = None
    invocation_count: int = 1
    invocation_state: str = 'same_incoming_membrane'
    leak_offset: int = 0
    lif_spec: Optional[LIFSpec] = None

    def __post_init__(self):
        if self.reset_mode is None:
            object.__setattr__(self, 'reset_mode', 'subtract' if self.profile == 'snntorch' else 'zero')
        if self.profile not in ('p08', 'sound', 'snntorch', 'hnn'):
            raise ValueError('Unknown neuron profile')
        for name in ('threshold', 'beta_numerator', 'beta_denominator',
                     'invocation_count', 'leak_offset'):
            if type(getattr(self, name)) is not int:
                raise ValueError(name + ' must be an integer')
        for name in ('threshold', 'leak_offset'):
            if not LOW <= getattr(self, name) <= HIGH:
                raise ValueError(name + ' must fit signed 28 bits')
        if self.profile == 'snntorch' and self.reset_mode == 'subtract' and not -255 <= self.threshold <= 256:
            raise ValueError('snntorch subtract requires -threshold to fit p83 signed9 (-256..255)')
        if self.beta_denominator != 256 or not 0 <= self.beta_numerator <= 256:
            raise ValueError('beta must be numerator 0..256 over fixed 256')
        if self.invocation_count not in (1, 2) or self.invocation_state != 'same_incoming_membrane':
            raise ValueError('Only one call or duplicate calls with the same incoming membrane are supported')
        if self.invocation_count == 2 and self.profile != 'snntorch':
            raise ValueError('Duplicate invocation metadata is specific to snn.py')
        if self.reset_mode not in ('zero', 'subtract', 'none'):
            raise ValueError('Unknown reset mode')
        if self.profile in ('sound', 'hnn') and self.reset_mode != 'zero':
            raise ValueError('Source profile requires reset-to-zero')
        if self.profile in ('hnn', 'snntorch') and self.leak_offset:
            raise ValueError('Source profile has no neuron leak offset')
        if self.lif_spec is not None and self.profile != 'p08':
            raise ValueError('LIFSpec is only valid for p08')
        if self.profile == 'p08':
            self.p08_spec().validate()

    def validate(self):
        """Validation entry point used by TemporalGraph.ordered."""
        self.__post_init__()

    @property
    def semantic_order(self):
        return {
            'p08': ('integrate_saturate28', 'strict_threshold', 'reset', 'decay_floor256', 'offset_clip_floor'),
            'sound': ('integrate', 'strict_threshold', 'reset', 'decay_floor256', 'offset'),
            'snntorch': ('reset_from_incoming_membrane', 'decay_floor256', 'integrate', 'delayed_reset', 'strict_threshold'),
            'hnn': ('gate_old_membrane_by_previous_spike', 'decay_floor256', 'integrate', 'strict_threshold'),
        }[self.profile]

    @property
    def prior_spike_state(self):
        return 'required_binary_state' if self.profile == 'hnn' else (
            'ignored_reset_recomputed_from_incoming_membrane' if self.profile == 'snntorch' else 'ignored')

    @property
    def source_semantics(self):
        return {'p08': 'snn_reference.lif_step; restricted two-state contract',
                'sound': 'SoundSNN LIFFCNeuron; native decay=config decay/255',
                'hnn': 'HNN.py and SNN_forward.py; native decay=0.5, threshold=0.3; bias unused',
                'snntorch': 'snntorch 0.9.4 Leaky; reset_delay=True, no state_quant/inhibition, unit spikes'}[self.profile]

    @property
    def p08_mapping_supported(self):
        """Only p08 is certified here; no implicit source-profile lowering."""
        return self.profile == 'p08'

    def p08_spec(self):
        if self.profile != 'p08':
            raise ValueError('Source profile requires explicit semantic lowering; not certified as p08')
        if self.lif_spec is not None:
            s = self.lif_spec
            effective_beta = s.leak_alpha if s.leak_enabled else 256
            if (s.threshold != self.threshold or effective_beta != self.beta_numerator
                    or s.leak_beta != self.leak_offset
                    or s.reset_mode != {'zero': 0, 'subtract': 1, 'none': 2}[self.reset_mode]):
                raise ValueError('Contract and LIFSpec disagree')
            if s.refractory or s.window_enabled or s.window_length < 1 or s.window_initial or s.fire_type != 4:
                raise ValueError('Two-state step cannot carry refractory/window or membrane readout')
            if self.reset_mode == 'none' and s.reset_decrement:
                raise ValueError('none reset requires zero decrement')
            return s
        return LIFSpec(threshold=self.threshold, leak_enabled=self.beta_numerator != 256,
                       leak_alpha=min(self.beta_numerator, 255), leak_beta=self.leak_offset,
                       reset_mode={'zero': 0, 'subtract': 1, 'none': 2}[self.reset_mode])


def _arrays(current, membrane, previous_spike, integer):
    values = [np.asarray(x) for x in (current, membrane, previous_spike)]
    if len({x.shape for x in values}) != 1:
        raise ValueError('Current, membrane and previous_spike shapes must match')
    if integer and any(x.dtype.kind not in 'iub' for x in values):
        raise ValueError('Quantized reference requires integer inputs and states')
    if any(np.any(~np.isfinite(x)) for x in values):
        raise ValueError('Inputs and states must be finite')
    if np.any((values[2] != 0) & (values[2] != 1)):
        raise ValueError('previous_spike must be binary')
    # Conservative bound prevents int64 overflow in integrate and multiply.
    if integer and any(np.any(x > (1 << 52)) or np.any(x < -(1 << 52)) for x in values[:2]):
        raise ValueError('Integer oracle input exceeds safe arithmetic range')
    return [x.astype(np.int64 if integer else np.float64, copy=True) for x in values]


def _source_step(u, v, previous, contract, decay, check=lambda x: x):
    if contract.profile == 'hnn':
        v = check(check(decay(v * (1 - previous))) + u)
    elif contract.profile == 'snntorch':
        reset = v > contract.threshold
        if contract.reset_mode == 'zero':
            v = check(check(decay(v * (1 - reset))) + u)
        else:
            pre = check(decay(v))
            corrected_current = u
            if contract.reset_mode == 'subtract':
                corrected_current = u - reset * contract.threshold
            v = check(pre + corrected_current)
    else:  # sound: reset before decay; spike comes from pre-reset membrane
        integrated = check(v + u)
        spike = (integrated > contract.threshold).astype(np.int64)
        return spike, check(decay(integrated * (1 - spike)) + contract.leak_offset)
    return (v > contract.threshold).astype(np.int64), v


def step(current, membrane, previous_spike, contract):
    """Return (binary spike, new integer membrane) for ONE invocation.

    Source profiles raise OVERFLOW for out-of-range signed28 membrane stages;
    they never silently clip. Floor quantization is an explicit approximation.
    HNN's threshold-based pre-stage hardware reset additionally requires a
    consistent incoming previous_spike == (membrane > threshold). snntorch's
    p83 reset multiplier has the same requirement; native reset is recomputed.
    """
    u, v, previous = _arrays(current, membrane, previous_spike, True)
    if contract.profile == 'p08':
        spike, v, _, _ = lif_step(u, v, np.zeros_like(v), 0, contract.p08_spec())
        return spike, v
    _check_membrane(v)
    return _source_step(u, v, previous, contract,
                        lambda x: (x * contract.beta_numerator) // 256, _check_membrane)


def _check_membrane(value):
    if np.any(value < LOW) or np.any(value > HIGH):
        raise OverflowError('OVERFLOW: source-profile membrane stage exceeds signed28')
    return value


def native_float_step(current, membrane, previous_spike, contract, *, beta, threshold=None):
    """Separate float64 equation oracle; explicitly supply native decay.

    Sound uses config['decay']/255, snntorch scripts use 0.9, HNN uses 0.5
    and threshold 0.3. This is not bitwise PyTorch float32 reproduction.
    Only default delayed-reset snntorch Leaky semantics are represented.
    """
    if contract.profile == 'p08':
        raise ValueError('p08 has integer semantics only')
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('Native beta must be finite in 0..1')
    from dataclasses import replace
    # Threshold may be fractional in the native reference without weakening
    # validation of the integer contract.
    native = replace(contract)
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError('Native threshold must be finite')
        object.__setattr__(native, 'threshold', float(threshold))
    u, v, previous = _arrays(current, membrane, previous_spike, False)
    return _source_step(u, v, previous, native, lambda x: x * beta)
