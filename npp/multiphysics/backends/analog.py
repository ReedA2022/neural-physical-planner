"""Bounded, explicitly hypothetical differential-conductance matrix tile.

A signed value is a bipolar voltage. A signed weight is the difference of two
nonnegative conductances. Differential readout is divided by the declared input
and weight scales. This is a local reduced-order model, not a circuit solver.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from npp.models import InputValidationError

BACKEND_ID = 'analog.differential_conductance'
BACKEND_VERSION = '1.0'
MAX_DIMENSION = 64
MAX_BATCH = 128


def _invalid(path: str, message: str, code: str = 'invalid_input'):
    raise InputValidationError([{'path': path, 'message': message, 'code': code}])


def _number(value, path, *, positive=False, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(path, 'expected a finite real number')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        _invalid(path, 'expected a finite positive number' if positive else 'expected a finite nonnegative number')
    if maximum is not None and value > maximum:
        _invalid(path, f'value exceeds the supported maximum {maximum}', 'envelope_violation')
    return value


def _integer(value, path, low, high):
    if type(value) is not int or not low <= value <= high:
        _invalid(path, f'expected an integer in [{low}, {high}]')
    return value


def _array(value, path, ndim):
    def check(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.integer, np.floating)):
            _invalid(path, 'expected finite real numeric array without booleans')
    check(value.tolist() if isinstance(value, np.ndarray) else value)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _invalid(path, 'expected a rectangular real numeric array')
    if result.ndim != ndim or not all(result.shape) or not np.isfinite(result).all():
        _invalid(path, f'expected a nonempty finite array of rank {ndim}')
    return result.copy()


# Physical parameters and coefficients must be provided; only reference_pack
# supplies defaults. Costs use the common Quantity contract through catalog.
PHYSICAL_FIELDS = {
    'mode', 'g_min_s', 'g_max_s', 'input_scale_v', 'weight_scale',
    'input_limit', 'output_current_limit_a', 'program_bits', 'dac_bits',
    'adc_bits', 'programming_noise_relative', 'drift_per_ns',
    'read_disturbance_relative', 'line_resistance_ohm', 'integration_ns',
    'programming_ns_per_cell', 'dac_ns', 'adc_ns', 'control_ns',
    'temperature_min_c', 'temperature_max_c',
}
COST_FIELDS = {
    'dac_energy_pj', 'adc_energy_pj', 'programming_energy_pj_per_cell',
    'readout_energy_pj_per_output', 'control_energy_pj_per_read',
    'reference_energy_pj_per_read', 'bias_add_energy_pj_per_output',
    'static_power_mw', 'cell_area_um2', 'converter_area_um2',
}


def validate_parameters(parameters):
    if not isinstance(parameters, dict):
        _invalid('parameters', 'expected an object')
    expected = PHYSICAL_FIELDS | COST_FIELDS
    if set(parameters) != expected:
        _invalid('parameters', f'missing fields {sorted(expected-set(parameters))}; unknown fields {sorted(set(parameters)-expected)}')
    p = copy.deepcopy(parameters)
    if p['mode'] not in ('finite', 'ideal_diagnostic'):
        _invalid('parameters.mode', 'supported modes are finite and ideal_diagnostic')
    for name in PHYSICAL_FIELDS - {'mode', 'program_bits', 'dac_bits', 'adc_bits', 'temperature_min_c', 'temperature_max_c'}:
        p[name] = _number(p[name], 'parameters.' + name, positive=name in {
            'g_max_s', 'input_scale_v', 'weight_scale', 'input_limit',
            'output_current_limit_a', 'integration_ns'}, maximum=1e15)
    if p['g_min_s'] >= p['g_max_s']:
        _invalid('parameters.g_min_s', 'g_min_s must be smaller than g_max_s')
    if p['g_max_s'] - p['g_min_s'] < 1e-15:
        _invalid('parameters', 'conductance span below supported numeric envelope 1e-15 S', 'envelope_violation')
    for name in ('input_scale_v', 'weight_scale', 'input_limit', 'output_current_limit_a', 'integration_ns'):
        if p[name] < 1e-15:
            _invalid('parameters.'+name, 'below supported numeric envelope 1e-15', 'envelope_violation')
    for name in ('program_bits', 'dac_bits', 'adc_bits'):
        p[name] = _integer(p[name], 'parameters.' + name, 2, 24)
    for name in ('temperature_min_c', 'temperature_max_c'):
        value = p[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < -273.15:
            _invalid('parameters.' + name, 'expected finite temperature >= -273.15 degrees C')
    if p['temperature_min_c'] > p['temperature_max_c']:
        _invalid('parameters.temperature_min_c', 'temperature envelope is reversed')
    if p['mode'] == 'ideal_diagnostic' and any(p[k] != 0 for k in (
            'programming_noise_relative', 'drift_per_ns', 'read_disturbance_relative', 'line_resistance_ohm')):
        _invalid('parameters.mode', 'ideal_diagnostic requires zero programming noise, drift, read disturbance and line resistance')
    # Quantity validation is shared, not a separate unknown-as-zero convention.
    from . import cost_quantity, coefficient_unit
    for name in COST_FIELDS:
        unit=coefficient_unit(name)
        p[name]=cost_quantity(p[name],unit,'parameters.'+name)
    return p


def _context(context):
    c = copy.deepcopy(context or {})
    allowed = {'seed', 'stream_id', 'age_ns', 'read_count', 'reuse_count', 'temperature_c', 'owner_id', 'environment', 'geometry', 'schedule', 'fidelity', 'seed_policy'}
    if set(c) - allowed:
        _invalid('context', f'unknown context fields {sorted(set(c)-allowed)}')
    c.setdefault('seed', 0)
    c.setdefault('stream_id', 'physical.analog')
    c.setdefault('age_ns', 0.0)
    c.setdefault('read_count', 0)
    c.setdefault('reuse_count', 1)
    c.setdefault('temperature_c', 25.0)
    c.setdefault('owner_id', 'analog_tile')
    _integer(c['seed'], 'context.seed', 0, 2**32-1)
    _integer(c['read_count'], 'context.read_count', 0, 10**9)
    _integer(c['reuse_count'], 'context.reuse_count', 1, 10**6)
    c['age_ns'] = _number(c['age_ns'], 'context.age_ns', maximum=1e15)
    if not isinstance(c['stream_id'], str) or not c['stream_id'] or not isinstance(c['owner_id'], str) or not c['owner_id']:
        _invalid('context', 'owner_id and stream_id must be nonempty strings')
    if isinstance(c['temperature_c'], bool) or not isinstance(c['temperature_c'], (int, float)) or not math.isfinite(c['temperature_c']):
        _invalid('context.temperature_c', 'expected a finite temperature')
    return c


def _quantize_signed(values, maximum, bits):
    # Symmetric mid-tread signed quantizer: 2**bits-1 levels, zero represented.
    levels = 2**(bits-1) - 1
    return np.rint(values / maximum * levels) / levels * maximum


def evaluate_linear(weights, inputs, bias, parameters, context=None):
    """Evaluate one programmed tile on a batch; all state/history is explicit.

    Reuse amortizes the one-time programming charge. It repeats this evaluation
    at the same explicit age/read_count, not an unmodeled drift trajectory.
    ``advance`` must be requested separately for changed state age/read count.
    """
    p, c = validate_parameters(parameters), _context(context)
    w = _array(weights, 'weights', 2)
    try:
        raw_x = np.asarray(inputs)
    except (TypeError, ValueError):
        _invalid('inputs', 'expected a rectangular input array')
    if raw_x.ndim not in (1, 2):
        _invalid('inputs', 'expected a vector or batch matrix')
    x = _array(inputs, 'inputs', raw_x.ndim)
    vector = x.ndim == 1
    if vector:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != w.shape[1]:
        _invalid('inputs', 'input last dimension must match matrix columns')
    if max(w.shape) > MAX_DIMENSION or x.shape[0] > MAX_BATCH:
        _invalid('weights', f'one tile is limited to {MAX_DIMENSION} rows/columns and batch {MAX_BATCH}', 'envelope_violation')
    b = np.zeros(w.shape[0]) if bias is None else _array(bias, 'bias', 1)
    if b.shape != (w.shape[0],):
        _invalid('bias', 'bias size must match matrix rows')
    if np.max(np.abs(w)) > p['weight_scale']:
        _invalid('weights', 'weight magnitude exceeds declared weight_scale', 'envelope_violation')
    if np.max(np.abs(x)) > p['input_limit']:
        _invalid('inputs', 'input magnitude exceeds declared input_limit; clipping is not implicit', 'envelope_violation')
    if not p['temperature_min_c'] <= c['temperature_c'] <= p['temperature_max_c']:
        _invalid('context.temperature_c', 'temperature outside device envelope', 'envelope_violation')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='raise'):
            ideal = x @ w.T + b
            span = p['g_max_s'] - p['g_min_s']
            gp = p['g_min_s'] + np.maximum(w, 0) / p['weight_scale'] * span
            gn = p['g_min_s'] + np.maximum(-w, 0) / p['weight_scale'] * span
            voltage = x * p['input_scale_v']
            if p['mode'] == 'finite':
                steps = 2**p['program_bits'] - 1
                gp = p['g_min_s'] + np.rint((gp - p['g_min_s']) / span * steps) / steps * span
                gn = p['g_min_s'] + np.rint((gn - p['g_min_s']) / span * steps) / steps * span
                # Program noise is independently sampled on each physical rail;
                # deterministic stream derivation does not touch global RNG state.
                import hashlib
                salt = int.from_bytes(hashlib.sha256(c['stream_id'].encode()).digest()[:4], 'big')
                rng = np.random.default_rng(np.random.SeedSequence([c['seed'], salt]))
                sigma = p['programming_noise_relative'] * span
                if sigma:
                    gp += rng.normal(0, sigma, gp.shape)
                    gn += rng.normal(0, sigma, gn.shape)
                attenuation = math.exp(-p['drift_per_ns'] * c['age_ns'])
                attenuation *= math.exp(-p['read_disturbance_relative'] * c['read_count'])
                # Saturation represents the explicit physical conductance bounds.
                gp = np.clip(p['g_min_s'] + (gp-p['g_min_s']) * attenuation, p['g_min_s'], p['g_max_s'])
                gn = np.clip(p['g_min_s'] + (gn-p['g_min_s']) * attenuation, p['g_min_s'], p['g_max_s'])
                voltage = _quantize_signed(voltage, p['input_limit']*p['input_scale_v'], p['dac_bits'])
            # Reduced-order column loading: one series resistance per input,
            # loading by all positive and negative conductance branches.
            loaded_voltage = voltage / (1 + p['line_resistance_ohm'] * np.sum(gp+gn, axis=0))
            current_p, current_n = loaded_voltage @ gp.T, loaded_voltage @ gn.T
            current = current_p - current_n
            limit = p['output_current_limit_a']
            if np.max(np.abs(current_p)) > limit or np.max(np.abs(current_n)) > limit or np.max(np.abs(current)) > limit:
                _invalid('outputs', 'rail or differential current exceeds readout envelope', 'envelope_violation')
            digitized = current if p['mode'] == 'ideal_diagnostic' else _quantize_signed(current, limit, p['adc_bits'])
            decoded = digitized * p['weight_scale'] / (span * p['input_scale_v'])
            actual = decoded + b
            if not all(np.isfinite(a).all() for a in (ideal, actual, loaded_voltage, gp, gn)):
                raise FloatingPointError('nonfinite local device arithmetic')
            # Dissipation includes both rails, including baseline conductance.
            read_energy_pj = float(np.sum(loaded_voltage**2 * np.sum(gp+gn, axis=0))) * p['integration_ns'] * 1e3
            line_energy_pj = float(np.sum((voltage-loaded_voltage)*loaded_voltage*np.sum(gp+gn,axis=0))) * p['integration_ns'] * 1e3
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        _invalid('evaluation', f'numerical failure: {exc}', 'numerical_failure')
    from . import build_ledger, error_record, evaluation_identity
    repetitions = c['reuse_count']
    count = x.shape[0] * repetitions
    cells = 2*w.size
    duration = (p['dac_ns'] + p['integration_ns'] + p['adc_ns'] + p['control_ns']) * count + p['programming_ns_per_cell'] * cells
    entries = [
        ('programming', 'weight_programming', p['programming_energy_pj_per_cell'], cells),
        ('conversion', 'input_dac', p['dac_energy_pj'], x.shape[1]*count),
        ('conversion', 'output_adc', p['adc_energy_pj'], w.shape[0]*count),
        ('compute', 'current_readout', p['readout_energy_pj_per_output'], w.shape[0]*count),
        ('compute', 'bias_addition', p['bias_add_energy_pj_per_output'], (w.shape[0]*count if bias is not None else 0)),
        ('control', 'read_control', p['control_energy_pj_per_read'], count),
        ('support', 'reference_supply', p['reference_energy_pj_per_read'], count),
        ('support', 'static_energy', p['static_power_mw'], duration),
        ('compute', 'conductance_dissipation', {'state':'known','value':read_energy_pj,'unit':'pJ'}, repetitions),
        ('movement', 'series_line_dissipation', {'state':'known','value':line_energy_pj,'unit':'pJ'}, repetitions),
    ]
    ledger = build_ledger(c['owner_id'], entries, duration, [
        ('cells', p['cell_area_um2'], cells), ('converters', p['converter_area_um2'], x.shape[1]+w.shape[0])])
    outputs = lambda a: a[0].tolist() if vector else a.tolist()
    return {
        'backend_id': BACKEND_ID, 'backend_version': BACKEND_VERSION,
        'status': 'conditional' if not ledger['complete'] else 'model_feasible',
        'ideal_output': outputs(ideal), 'actual_output': outputs(actual),
        'error': error_record(ideal, actual, mode=p['mode']), 'ledger': ledger,
        'local_duration_ns': duration, 'schedule_status': 'local_serial_model_not_system_schedule',
        'resources': {'tile':1,'dac_lanes':x.shape[1],'adc_lanes':w.shape[0], 'programmed_cells':cells},
        'physical': {'voltage_v':loaded_voltage.tolist(),'dac_voltage_v':voltage.tolist(),'loaded_voltage_v':loaded_voltage.tolist(),
                     'readout_before_bias':outputs(decoded),'bias':b.tolist(),'bias_present':bias is not None,'conductance_positive_s':gp.tolist(),
                     'conductance_negative_s':gn.tolist(),'current_positive_a':current_p.tolist(),
                     'current_negative_a':current_n.tolist(),'differential_current_a':current.tolist()},
        'state': {'owner_id':c['owner_id'],'version':0,'weight_identity':evaluation_identity(BACKEND_ID,BACKEND_VERSION,{'weights':w.tolist()},p,{'seed':c['seed'],'stream_id':c['stream_id']})['cache_key'],'age_ns':c['age_ns'],'read_count':c['read_count'],
                  'reads_evaluated':count,'programmed_once':True,'reuse_count':repetitions},
        'dependencies': evaluation_identity(BACKEND_ID, BACKEND_VERSION, {'weights':w.tolist(),'inputs':x.tolist(),'bias':None if bias is None else b.tolist()}, p, c),
        'assumptions': ['hypothetical coefficients; no measured device calibration',
                        'bipolar electrical voltage inputs and two nonnegative conductance rails per weight',
                        'one series resistance per input column; no distributed wire/circuit solve',
                        'temperature is an operating guard only; drift is the declared reduced-order exponential law',
                        'bias addition is digital after differential ADC; bias is added once',
                        'local serial duration; no system resource contention or inter-tile reduction modeled',
                        'reuse repeats the same explicit age/read count; advance state explicitly for time-dependent trajectories',
                        'ideal_diagnostic bypasses all quantizers and is a mathematical diagnostic, not a finite-precision physical device' if p['mode']=='ideal_diagnostic' else 'finite quantizers and bounded conductance clipping are active'],
    }
