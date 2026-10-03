"""Directional, bounded classical interfaces and explicit alignment choices.

All reference coefficients are hypothetical. Optical payloads contain field
amplitudes in sqrt(mW), never signed intensity. These local interfaces do not
claim fabrication calibration, quantum conversion, or a complete chip model.
"""
from __future__ import annotations

import copy
import functools
import math
import re

import numpy as np

from .backends import fail, quantity_scale, evaluation_identity
from .contracts import AccountingOwner, CostEntry, CostLedger, Quantity, known


PARAMETER_UNITS = {
    'input_limit': '1', 'gate_scale': '1', 'source_power_mw': 'mW',
    'lo_power_mw': 'mW', 'wallplug_efficiency': '1', 'dac_bits': 'bit',
    'adc_bits': 'bit', 'gate_bits': 'bit', 'symbol_ns': 'ns', 'dac_ns': 'ns',
    'adc_ns': 'ns', 'driver_ns': 'ns', 'detector_ns': 'ns', 'gate_ns': 'ns',
    'control_ns': 'ns', 'input_scale_v': 'V', 'voltage_limit_v': 'V',
    'field_limit_sqrt_mw': 'sqrt(mW)', 'lo_phase_rad': 'rad',
    'optical_loss_db_per_ns': 'dB/ns', 'max_optical_delay_ns': 'ns',
    'buffer_capacity_bits': 'bit', 'buffer_retention_ns': 'ns',
    'buffer_word_bits': 'bit', 'buffer_access_ns': 'ns',
    'dac_energy_pj': 'pJ/sample', 'adc_energy_pj': 'pJ/sample',
    'driver_energy_pj': 'pJ/sample', 'detector_energy_pj': 'pJ/sample',
    'gate_energy_pj': 'pJ/sample', 'control_energy_pj': 'pJ/invocation',
    'static_power_mw': 'mW', 'converter_area_um2': 'um^2/lane',
    'optical_area_um2': 'um^2/lane', 'delay_energy_pj_per_lane_ns': 'pJ/(lane ns)',
    'delay_area_um2_per_lane_ns': 'um^2/(lane ns)',
    'buffer_write_energy_pj_per_byte': 'pJ/byte',
    'buffer_read_energy_pj_per_byte': 'pJ/byte',
    'buffer_hold_energy_pj_per_bit_ns': 'pJ/(bit ns)',
    'buffer_area_um2_per_bit': 'um^2/bit',
}
COST_FIELDS = {
    'dac_energy_pj', 'adc_energy_pj', 'driver_energy_pj', 'detector_energy_pj',
    'gate_energy_pj', 'control_energy_pj', 'static_power_mw', 'converter_area_um2',
    'optical_area_um2', 'delay_energy_pj_per_lane_ns', 'delay_area_um2_per_lane_ns',
    'buffer_write_energy_pj_per_byte', 'buffer_read_energy_pj_per_byte',
    'buffer_hold_energy_pj_per_bit_ns', 'buffer_area_um2_per_bit',
}


def reference_parameters():
    """Independent illustrative coefficients, not measured device data."""
    p = dict(input_limit=2., gate_scale=2., source_power_mw=1., lo_power_mw=1.,
             wallplug_efficiency=.2, dac_bits=16, adc_bits=20, gate_bits=16,
             symbol_ns=1., dac_ns=1., adc_ns=1., driver_ns=.2, detector_ns=.2,
             gate_ns=.2, control_ns=.1, input_scale_v=.1, voltage_limit_v=.2,
             field_limit_sqrt_mw=8., lo_phase_rad=0., optical_loss_db_per_ns=.05,
             max_optical_delay_ns=100., buffer_capacity_bits=1_048_576,
             buffer_retention_ns=1e6, buffer_word_bits=64, buffer_access_ns=1.)
    costs = dict(dac_energy_pj=.1, adc_energy_pj=.2, driver_energy_pj=.05,
                 detector_energy_pj=.05, gate_energy_pj=.05, control_energy_pj=.01,
                 static_power_mw=.01, converter_area_um2=10., optical_area_um2=20.,
                 delay_energy_pj_per_lane_ns=.001, delay_area_um2_per_lane_ns=2e5,
                 buffer_write_energy_pj_per_byte=.01, buffer_read_energy_pj_per_byte=.01,
                 buffer_hold_energy_pj_per_bit_ns=1e-6, buffer_area_um2_per_bit=.1)
    p.update({k: known(v, PARAMETER_UNITS[k]).model_dump(mode='json', exclude_none=True)
              for k, v in costs.items()})
    return p


def _number(value, path, *, positive=False, maximum=1e15):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        fail(path, 'expected a finite real number')
    try:
        v = float(value)
    except OverflowError:
        fail(path, 'number exceeds finite range', 'numerical_failure')
    if not math.isfinite(v) or v < 0 or (positive and v == 0):
        fail(path, 'expected finite positive value' if positive else 'expected finite nonnegative value')
    if v > maximum or (positive and v < 1e-15):
        fail(path, 'outside supported numeric envelope', 'envelope_violation')
    return v


def validate_parameters(parameters=None):
    p = copy.deepcopy(reference_parameters() if parameters is None else parameters)
    if not isinstance(p, dict) or set(p) != set(PARAMETER_UNITS):
        fail('interface.parameters', 'requires the exact reference parameter fields')
    integers = {'dac_bits': (2, 24), 'adc_bits': (2, 24), 'gate_bits': (2, 24),
                'buffer_word_bits': (2, 64), 'buffer_capacity_bits': (1, 1_000_000_000)}
    positive = {'input_limit', 'gate_scale', 'source_power_mw', 'lo_power_mw',
                'wallplug_efficiency', 'symbol_ns', 'input_scale_v', 'voltage_limit_v',
                'field_limit_sqrt_mw', 'buffer_retention_ns'}
    for name, value in p.items():
        path = 'interface.parameters.' + name
        if name in COST_FIELDS:
            try:
                q = Quantity.model_validate(value)
            except (ValueError, TypeError) as exc:
                fail(path, str(exc))
            if q.unit != PARAMETER_UNITS[name] or any(v is not None and v < 0 for v in (q.value, q.lower, q.upper)):
                fail(path, 'coefficient must be nonnegative and use ' + PARAMETER_UNITS[name])
            p[name] = q.model_dump(mode='json', exclude_none=True)
        elif name in integers:
            low, high = integers[name]
            if type(value) is not int or not low <= value <= high:
                fail(path, f'expected an integer in [{low}, {high}]')
        elif name == 'lo_phase_rad':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > math.pi:
                fail(path, 'LO phase must be finite radians in [-pi, pi]')
        else:
            p[name] = _number(value, path, positive=name in positive)
    if p['wallplug_efficiency'] > 1:
        fail('interface.parameters.wallplug_efficiency', 'efficiency cannot exceed one')
    if p['buffer_word_bits'] not in (32, 64):
        fail('interface.parameters.buffer_word_bits', 'buffer stores explicit IEEE binary32 or binary64 words')
    return p


def _context(context, operation, extras=()):
    c = copy.deepcopy({} if context is None else context)
    if not isinstance(c, dict) or set(c) - {'owner_id', 'reference_id', 'mode', *extras}:
        fail('interface.context', 'unknown or malformed interface context')
    c.setdefault('owner_id', 'interface.' + operation)
    c.setdefault('reference_id', 'laser.reference')
    c.setdefault('mode', 'finite')
    for key in ('owner_id', 'reference_id'):
        if not isinstance(c[key], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.:-]{0,159}', c[key]):
            fail('interface.context.' + key, 'expected a registered-style identifier')
    if c['mode'] not in ('finite', 'ideal_diagnostic'):
        fail('interface.context.mode', 'expected finite or ideal_diagnostic')
    return c


def _array(values, path='values'):
    # Inspect before NumPy coercion: bool mixed with numbers is not a real input.
    def visit(v, depth=0):
        if depth > 2:
            fail(path, 'expected a vector or batch matrix')
        if isinstance(v, (list, tuple)):
            if len(v) > 128:
                fail(path, 'axis exceeds supported 128 values', 'envelope_violation')
            for child in v:
                visit(child, depth+1)
        elif isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            fail(path, 'expected finite real values without booleans')
    if isinstance(values, np.ndarray):
        if values.ndim not in (1, 2) or values.size > 8192:
            fail(path, 'expected bounded vector or batch matrix', 'envelope_violation')
        values = values.tolist()
    visit(values)
    try:
        result = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        fail(path, 'expected a rectangular real array')
    if result.ndim not in (1, 2) or not result.size or not np.isfinite(result).all():
        fail(path, 'expected a nonempty finite vector or batch matrix')
    if result.shape[-1] > 64 or (result.ndim == 2 and result.shape[0] > 128):
        fail(path, 'interface supports at most 64 lanes and 128 symbols', 'envelope_violation')
    return result.copy()


def _finite_array(value, path):
    if not np.isfinite(value).all():
        fail(path, 'nonfinite interface arithmetic', 'numerical_failure')
    return value


def _field(payload, p):
    fields = {'carrier', 'regime', 'encoding', 'real', 'imag', 'decode_scale', 'reference_id'}
    if not isinstance(payload, dict) or set(payload) != fields:
        fail('field', 'requires explicit coherent field payload')
    if (payload['carrier'], payload['regime'], payload['encoding']) != ('optical', 'classical_analog', 'signed_coherent_field'):
        fail('field', 'interface requires classical signed coherent field, not intensity or quantum state', 'domain_mismatch')
    real, imag = _array(payload['real'], 'field.real'), _array(payload['imag'], 'field.imag')
    if real.shape != imag.shape:
        fail('field', 'real and imaginary shapes differ')
    scale = _number(payload['decode_scale'], 'field.decode_scale', positive=True)
    reference = payload['reference_id']
    if not isinstance(reference, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.:-]{0,159}', reference):
        fail('field.reference_id', 'expected explicit valid phase reference identifier')
    if np.max(np.hypot(real, imag)) > p['field_limit_sqrt_mw']:
        fail('field', 'field amplitude exceeds declared input envelope', 'envelope_violation')
    return real, imag, scale


def _payload(real, imag, scale, reference):
    _finite_array(real, 'field.real'); _finite_array(imag, 'field.imag')
    _number(scale, 'field.decode_scale', positive=True)
    return dict(carrier='optical', regime='classical_analog', encoding='signed_coherent_field',
                real=real.tolist(), imag=imag.tolist(), decode_scale=scale, reference_id=reference)


def validate_field(payload, parameters=None):
    """Return an owned normalized coherent-field snapshot after envelope checks."""
    real, imag, scale = _field(payload, validate_parameters(parameters))
    return _payload(real, imag, scale, payload['reference_id'])


def _quantize(values, fullscale, bits, mode):
    if np.max(np.abs(values)) > fullscale:
        fail('conversion.values', 'clipping is not implicit; signal exceeds converter range', 'envelope_violation')
    if mode == 'ideal_diagnostic':
        return values.copy()
    levels = 2**(bits-1)-1
    return np.rint(values/fullscale*levels)/levels*fullscale


def _ledger(owner, entries, duration, areas=()):
    costs, records, area_records = [], [], []
    for category, effect, coefficient, multiplier in entries:
        quantity = quantity_scale(coefficient, multiplier, 'pJ')
        costs.append(CostEntry(contribution_id=effect, owner=owner, boundary='local_interface',
                               category=category, metric='energy_pj', quantity=Quantity.model_validate(quantity),
                               basis='declared coefficient times explicit count, power-time, or hold duration'))
        records.append(dict(owner_id=owner, effect_id=effect, category=category,
                            coefficient=copy.deepcopy(coefficient), multiplier=multiplier, energy_pj=quantity))
    for effect, coefficient, multiplier in areas:
        quantity = quantity_scale(coefficient, multiplier, 'um^2')
        costs.append(CostEntry(contribution_id='area.'+effect, owner=owner, boundary='local_interface',
                               category='support', metric='area_um2', quantity=Quantity.model_validate(quantity),
                               basis='installed capacity; not amortized per symbol'))
        area_records.append(dict(owner_id=owner, effect_id=effect, area_um2=quantity))
    if not areas:
        costs.append(CostEntry(contribution_id='area.external', owner=owner, boundary='local_interface',
                               category='support', metric='area_um2', quantity=known(0., 'um^2'),
                               basis='no additional installed device; caller owns upstream device capacity'))
    ledger = CostLedger(boundary='local_interface', owners=(AccountingOwner(id=owner, mode='leaf'),),
                        entries=tuple(costs), expected_contributions=tuple(e.contribution_id for e in costs))
    energy, area = ledger.aggregate('energy_pj', 'pJ'), ledger.aggregate('area_um2', 'um^2')
    complete = energy.complete and area.complete
    coverage = {}
    reasons = {
        'compute': 'interface performs conversion, transport or storage; neural arithmetic is charged by its own component',
        'movement': 'no external link movement inside this local interface boundary',
        'storage': 'stateless streaming interface has no persistent data storage',
        'programming': 'preconfigured interface boundary; no trainable or programmed weights',
        'calibration': 'calibration procedure is outside this hypothetical configured interface; not free end-to-end calibration',
        'source': 'this interface does not create a fresh optical carrier; upstream source owns incident photons',
        'reset': 'no retained mutable device state in the declared local model',
        'conversion': 'no domain conversion in this alignment operation',
        'control': 'control owned upstream', 'support': 'support charged by installed-area entries',
    }
    for category, reason in reasons.items():
        ids = [e.contribution_id for e in costs if e.category == category]
        coverage[category] = dict(status='charged' if ids else 'excluded', contribution_ids=ids,
                                  reason='enumerated exactly once in typed ledger' if ids else reason)
    return dict(contract=ledger.model_dump(mode='json'), boundary='inclusive_local_interface', entries=records,
                area_entries=area_records, category_coverage=coverage, energy_pj=energy.quantity.model_dump(mode='json', exclude_none=True),
                area_um2=area.quantity.model_dump(mode='json', exclude_none=True), complete=complete,
                duration_ns=known(duration, 'ns').model_dump(mode='json', exclude_none=True),
                obligations=[] if complete else ['resolve unknown or bounded interface costs before point-valued comparison'])


def _finish(operation, p, c, shape, duration, entries, areas=(), **result):
    duration = _number(duration, 'interface.duration_ns')
    entries = list(entries) + [('control', 'control', p['control_energy_pj'], 1),
                               ('support', 'active_static', p['static_power_mw'], duration)]
    ledger = _ledger(c['owner_id'], entries, duration, areas)
    result.update(status='model_feasible' if ledger['complete'] else 'conditional',
                  operation=operation, ledger=ledger, duration_ns=duration,
                  duration_breakdown=dict(programming_ns=0., calibration_ns=0., read_ns=duration, reset_ns=0.),
                  resources=dict(lanes=shape[-1], symbols=shape[0] if len(shape) == 2 else 1),
                  assumptions=['All reference coefficients are hypothetical, with epistemic uncertainty.',
                               'Local configured interface only; no fabrication, startup or calibration claim.',
                               'No stochastic noise model; finite mode represents converter quantization only.'],
                  dependencies=evaluation_identity('interface.'+operation, '1.0', copy.deepcopy(result), p, c))
    if c['mode'] == 'ideal_diagnostic':
        result['assumptions'].append('Ideal diagnostic bypasses quantization; it is not finite precision hardware validation.')
    return result


def encode(values, parameters=None, context=None):
    """Digital words -> driven coherent field; source photons are charged."""
    p = validate_parameters(parameters); c = _context(context, 'encode', ('input_scale',))
    x = _array(values)
    scale = _number(c.get('input_scale', p['input_limit']), 'context.input_scale', positive=True)
    if scale > p['input_limit']:
        fail('context.input_scale', 'scale exceeds supported semantic input range', 'envelope_violation')
    q = _quantize(x, scale, p['dac_bits'], c['mode'])
    real = q/scale*math.sqrt(p['source_power_mw'])
    if np.max(np.abs(real)) > p['field_limit_sqrt_mw']:
        fail('encode.field', 'source exceeds field amplitude envelope', 'envelope_violation')
    photons = float(np.sum(real**2))*p['symbol_ns']/p['wallplug_efficiency']
    duration = (p['dac_ns']+p['driver_ns']+p['symbol_ns']+p['control_ns'])*(x.shape[0] if x.ndim == 2 else 1)
    return _finish('encode', p, c, x.shape, duration,
                   [('conversion', 'input_dac', p['dac_energy_pj'], x.size),
                    ('conversion', 'electro_optic_driver', p['driver_energy_pj'], x.size),
                    ('source', 'injected_carrier', known(1., 'pJ').model_dump(), photons)],
                   [('converter', p['converter_area_um2'], x.shape[-1]), ('encoder', p['optical_area_um2'], x.shape[-1])],
                   field=_payload(real, np.zeros_like(real), scale/math.sqrt(p['source_power_mw']), c['reference_id']),
                   physical=dict(injected_optical_energy_pj=photons*p['wallplug_efficiency'],
                                 equation='E = quantize(x)/input_scale * sqrt(source_power_mw)'))


def readout(field, parameters=None, context=None):
    """Balanced homodyne + ADC. LO power is paid per lane and symbol."""
    p = validate_parameters(parameters); c = _context(context, 'readout')
    real, imag, scale = _field(field, p)
    if field['reference_id'] != c['reference_id']:
        fail('readout.reference_id', 'LO and input must share an explicitly declared phase reference', 'domain_mismatch')
    # Difference photocurrent after normalizing responsivity is 2 sqrt(P_LO)
    # times the field quadrature. ADC fullscale is declared, not inferred.
    quadrature = real*math.cos(p['lo_phase_rad']) + imag*math.sin(p['lo_phase_rad'])
    difference = 2*math.sqrt(p['lo_power_mw'])*quadrature
    fullscale = 2*math.sqrt(p['lo_power_mw'])*p['field_limit_sqrt_mw']
    digitized = _quantize(difference, fullscale, p['adc_bits'], c['mode'])
    values = _finite_array(digitized/(2*math.sqrt(p['lo_power_mw']))*scale, 'readout.values')
    duration = (p['symbol_ns']+p['detector_ns']+p['adc_ns']+p['control_ns'])*(real.shape[0] if real.ndim == 2 else 1)
    lo = p['lo_power_mw']*p['symbol_ns']/p['wallplug_efficiency']*real.size
    return _finish('readout', p, c, real.shape, duration,
                   [('conversion', 'balanced_detector', p['detector_energy_pj'], 2*real.size),
                    ('conversion', 'output_adc', p['adc_energy_pj'], real.size),
                    ('source', 'local_oscillator', known(1., 'pJ').model_dump(), lo)],
                   [('converter', p['converter_area_um2'], real.shape[-1]), ('readout', p['optical_area_um2'], 2*real.shape[-1])],
                   values=values.tolist(), physical=dict(balanced_difference=difference.tolist(),
                           lo_energy_pj=lo, lo_phase_rad=p['lo_phase_rad'],
                           equation='difference = 2 sqrt(P_LO) Re(E exp(-i phi_LO)); x = ADC(difference)/(2 sqrt(P_LO)) * decode_scale'))


def signed_gate(field, gate, parameters=None, context=None):
    """Fused electrical control of field amplitude and 0/pi sign phase.

    No homodyne or ADC readout occurs here. A negative control is a phase flip,
    not a negative optical power. The scale update preserves semantic units.
    """
    p = validate_parameters(parameters); c = _context(context, 'signed_gate', ('g_scale',))
    real, imag, scale = _field(field, p); g = _array(gate, 'gate')
    if field['reference_id'] != c['reference_id']:
        fail('gate.reference_id', 'gate phase frame and field phase reference disagree', 'domain_mismatch')
    if g.shape != real.shape:
        fail('gate', 'gate shape must exactly match field; no implicit broadcasting')
    gs = _number(c.get('g_scale', p['gate_scale']), 'context.g_scale', positive=True)
    if gs > p['gate_scale']:
        fail('context.g_scale', 'exceeds supported gate range', 'envelope_violation')
    gq = _quantize(g, gs, p['gate_bits'], c['mode'])
    ratio = gq/gs
    new_scale = scale*gs
    duration = (p['dac_ns']+p['gate_ns']+p['control_ns'])*(real.shape[0] if real.ndim == 2 else 1)
    return _finish('signed_gate', p, c, real.shape, duration,
                   [('conversion', 'gate_dac', p['dac_energy_pj'], real.size),
                    ('compute', 'signed_electro_optic_gate', p['gate_energy_pj'], real.size)],
                   [('converter', p['converter_area_um2'], real.shape[-1]), ('gate', p['optical_area_um2'], real.shape[-1])],
                   field=_payload(real*ratio, imag*ratio, new_scale, field['reference_id']),
                   physical=dict(amplitude_ratio=ratio.tolist(), sign_phase_rad=np.where(gq < 0, math.pi, 0.).tolist(),
                                 equation='E_out = (quantize(g)/g_scale) E_in; decode_scale_out = decode_scale_in * g_scale',
                                 hidden_optical_readout=False))


def attenuation_gate(intensity, gate, parameters=None, context=None):
    """Unsigned intensity attenuator; signed gates are explicitly illegal."""
    p = validate_parameters(parameters); c = _context(context, 'attenuation_gate')
    if not isinstance(intensity, dict) or set(intensity) != {'carrier', 'regime', 'encoding', 'power_mw'} or (
            intensity['carrier'], intensity['regime'], intensity['encoding']) != ('optical', 'classical_analog', 'intensity'):
        fail('intensity', 'requires an unsigned intensity payload', 'domain_mismatch')
    power, g = _array(intensity['power_mw']), _array(gate, 'gate')
    if power.shape != g.shape or np.any(power < 0) or np.any(g < 0) or np.any(g > 1):
        fail('attenuation_gate', 'intensity and gate need matching shapes; power >= 0 and 0 <= gate <= 1', 'domain_mismatch')
    if np.max(power) > p['field_limit_sqrt_mw']**2:
        fail('intensity', 'power exceeds interface envelope', 'envelope_violation')
    out = power*g
    return _finish('attenuation_gate', p, c, power.shape, (p['gate_ns']+p['control_ns'])*(power.shape[0] if power.ndim == 2 else 1),
                   [('compute', 'intensity_attenuator', p['gate_energy_pj'], power.size)],
                   [('attenuator', p['optical_area_um2'], power.shape[-1])],
                   intensity=dict(carrier='optical', regime='classical_analog', encoding='intensity', power_mw=out.tolist()))


def dac(values, parameters=None, context=None):
    p = validate_parameters(parameters); c = _context(context, 'dac')
    x = _array(values)
    q = _quantize(x, p['input_limit'], p['dac_bits'], c['mode'])
    volts = q*p['input_scale_v']
    if np.max(np.abs(volts)) > p['voltage_limit_v']:
        fail('dac', 'output voltage exceeds declared envelope', 'envelope_violation')
    return _finish('dac', p, c, x.shape, (p['dac_ns']+p['control_ns'])*(x.shape[0] if x.ndim == 2 else 1),
                   [('conversion', 'dac', p['dac_energy_pj'], x.size)],
                   [('converter', p['converter_area_um2'], x.shape[-1])],
                   analog=dict(carrier='electrical', regime='classical_analog', encoding='signed_voltage',
                               voltage_v=volts.tolist(), decode_scale=1/p['input_scale_v']))


def adc(analog, parameters=None, context=None):
    p = validate_parameters(parameters); c = _context(context, 'adc')
    if not isinstance(analog, dict) or set(analog) != {'carrier', 'regime', 'encoding', 'voltage_v', 'decode_scale'} or (
            analog['carrier'], analog['regime'], analog['encoding']) != ('electrical', 'classical_analog', 'signed_voltage'):
        fail('adc', 'requires an electrical signed-voltage payload', 'domain_mismatch')
    voltage = _array(analog['voltage_v']); scale = _number(analog['decode_scale'], 'adc.decode_scale', positive=True)
    values = _quantize(voltage, p['voltage_limit_v'], p['adc_bits'], c['mode'])*scale
    _finite_array(values, 'adc.values')
    return _finish('adc', p, c, voltage.shape, (p['adc_ns']+p['control_ns'])*(voltage.shape[0] if voltage.ndim == 2 else 1),
                   [('conversion', 'adc', p['adc_energy_pj'], voltage.size)],
                   [('converter', p['converter_area_um2'], voltage.shape[-1])], values=values.tolist())


def optical_delay(field, delay_ns, parameters=None, context=None):
    p = validate_parameters(parameters); c = _context(context, 'optical_delay')
    real, imag, scale = _field(field, p); delay = _number(delay_ns, 'delay_ns')
    if delay > p['max_optical_delay_ns']:
        fail('delay_ns', 'optical delay exceeds supported retention/routing envelope', 'envelope_violation')
    attenuation = 10**(-p['optical_loss_db_per_ns']*delay/20)
    if attenuation == 0:
        fail('optical_delay', 'positive transmission underflowed', 'numerical_failure')
    return _finish('optical_delay', p, c, real.shape, delay,
                   [('movement', 'optical_hold', p['delay_energy_pj_per_lane_ns'], real.size*delay)],
                   [('delay_line', p['delay_area_um2_per_lane_ns'], real.shape[-1]*delay)],
                   field=_payload(real*attenuation, imag*attenuation, scale, field['reference_id']),
                   physical=dict(amplitude_transmission=attenuation, power_transmission=attenuation**2,
                                 requested_delay_ns=delay, equation='E_out = E_in * 10^(-loss_db_per_ns * delay_ns / 20)'))


def digital_buffer(values, delay_ns, parameters=None, context=None):
    p = validate_parameters(parameters); c = _context(context, 'digital_buffer')
    x = _array(values); delay = _number(delay_ns, 'delay_ns')
    bits = x.size*p['buffer_word_bits']
    if bits > p['buffer_capacity_bits'] or delay > p['buffer_retention_ns']:
        fail('buffer', 'capacity or retention bound exceeded', 'resource_infeasibility')
    stored = x.astype(np.float32).astype(np.float64) if p['buffer_word_bits'] == 32 and c['mode'] == 'finite' else x
    _finite_array(stored, 'buffer.values')
    duration = delay+2*p['buffer_access_ns']+p['control_ns']
    return _finish('digital_buffer', p, c, x.shape, duration,
                   [('storage', 'buffer_write', p['buffer_write_energy_pj_per_byte'], bits/8),
                    ('storage', 'buffer_read', p['buffer_read_energy_pj_per_byte'], bits/8),
                    ('storage', 'buffer_hold', p['buffer_hold_energy_pj_per_bit_ns'], bits*delay)],
                   [('buffer', p['buffer_area_um2_per_bit'], bits)], values=stored.tolist(),
                   alignment=dict(requested_hold_ns=delay, access_ns=2*p['buffer_access_ns'], stored_bits=bits,
                                  representation='IEEE binary'+str(p['buffer_word_bits'])))


def recompute_alignment(values, delay_ns, upstream_energy_pj, upstream_duration_ns, parameters=None, context=None):
    """Explicit repeat-compute alternative; never a free generic wait.

    The caller supplies a deterministic repeatable upstream result and its full
    invocation energy/time. The helper accounts repetitions; it does not certify
    repeatability of mutable, destructive-read or stochastic upstream devices.
    """
    p = validate_parameters(parameters); c = _context(context, 'recompute_alignment')
    x = _array(values); delay = _number(delay_ns, 'delay_ns')
    time = _number(upstream_duration_ns, 'upstream_duration_ns', positive=True)
    try:
        energy = Quantity.model_validate(upstream_energy_pj)
    except (ValueError, TypeError) as exc:
        fail('upstream_energy_pj', str(exc))
    if energy.unit != 'pJ' or any(v is not None and v < 0 for v in (energy.value, energy.lower, energy.upper)):
        fail('upstream_energy_pj', 'requires nonnegative energy quantity in pJ')
    repetitions = math.ceil(delay/time)
    if repetitions > 1_000_000:
        fail('recompute_alignment', 'recomputation count exceeds bounded support', 'resource_infeasibility')
    result = _finish('recompute_alignment', p, c, x.shape, repetitions*time,
                     [('compute', 'recomputed_upstream', energy.model_dump(), repetitions)], values=x.tolist(),
                     alignment=dict(requested_delay_ns=delay, repetitions=repetitions, upstream_duration_ns=time))
    result['assumptions'].append('Caller must establish upstream repeatability; upstream area remains owned by its existing installed component.')
    return result


def _numerical_guard(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        try:
            with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            fail('interface.'+function.__name__, 'unsupported numerical range: '+str(exc), 'numerical_failure')
    return guarded


encode = _numerical_guard(encode)
readout = _numerical_guard(readout)
signed_gate = _numerical_guard(signed_gate)
attenuation_gate = _numerical_guard(attenuation_gate)
dac = _numerical_guard(dac)
adc = _numerical_guard(adc)
optical_delay = _numerical_guard(optical_delay)
digital_buffer = _numerical_guard(digital_buffer)
recompute_alignment = _numerical_guard(recompute_alignment)


_REGISTRY = {
    'digital_to_voltage': (dac, 'electrical.classical_digital', 'electrical.signed_voltage'),
    'voltage_to_digital': (adc, 'electrical.signed_voltage', 'electrical.classical_digital'),
    'digital_to_field': (encode, 'electrical.classical_digital', 'optical.signed_coherent_field'),
    'field_to_digital': (readout, 'optical.signed_coherent_field', 'electrical.classical_digital'),
    'signed_electro_optic_gate': (signed_gate, 'optical.signed_coherent_field+electrical.classical_digital', 'optical.signed_coherent_field'),
    'intensity_attenuator': (attenuation_gate, 'optical.intensity+electrical.classical_digital', 'optical.intensity'),
    'optical_delay': (optical_delay, 'optical.signed_coherent_field', 'optical.signed_coherent_field'),
    'digital_buffer': (digital_buffer, 'electrical.classical_digital', 'electrical.classical_digital'),
}


def registered_interfaces():
    return {name: dict(source=source, target=target, evaluator_version='1.0', evidence_kind='hypothetical')
            for name, (_, source, target) in _REGISTRY.items()}


def interface_catalog(parameters=None):
    """Versioned directional port templates with parameter-level evidence.

    Ports describe one lane; instantiation binds vector/batch shape explicitly.
    This is a separate catalog, not an unrecognized TechnologyPack extension.
    Evaluators are application-owned Python functions, never submitted formulas.
    """
    from .contracts import (SignalContract, PortContract, Parameter, Evidence, Uncertainty,
                            Interval, ClockContract, LifetimeContract, canonical_hash)
    p = validate_parameters(parameters)
    evidence = Evidence(kind='hypothetical', source='NPP illustrative interface coefficients',
                        location='npp.multiphysics.interfaces.reference_parameters', version='1.0',
                        scope='bounded local mathematical model; not independently calibrated or measured hardware')
    uncertainty = Uncertainty(kind='epistemic', description='real-device parameter uncertainty is uncharacterized')
    params = tuple(Parameter(name=name, quantity=Quantity.model_validate(value) if name in COST_FIELDS else known(float(value), PARAMETER_UNITS[name]),
                             evidence=evidence, uncertainty=uncertainty, instance_group='interface.reference.v1')
                   for name, value in sorted(p.items()))
    def signal(kind):
        field = kind == 'optical.signed_coherent_field'
        intensity = kind == 'optical.intensity'
        voltage = kind == 'electrical.signed_voltage'
        digital = kind == 'electrical.classical_digital'
        limit = p['field_limit_sqrt_mw'] if field else p['field_limit_sqrt_mw']**2 if intensity else p['voltage_limit_v'] if voltage else p['input_limit']
        return SignalContract(
            payload='tensor', carrier='electrical' if digital or voltage else 'optical',
            regime='classical_digital' if digital else 'classical_analog', shape=(1,), axis_order=('lane',),
            encoding='signed_numeric' if digital else 'signed_voltage' if voltage else 'intensity' if intensity else 'signed_coherent_field',
            physical_variable='digital_word' if digital else 'voltage' if voltage else 'power' if intensity else 'field_amplitude',
            signed=not intensity, scale=1., unit='1' if digital else 'V' if voltage else 'mW' if intensity else 'sqrt(mW)',
            value_range=Interval(minimum=0. if intensity else -limit, maximum=limit),
            bandwidth_hz=known(1e9/p['symbol_ns'], 'Hz'),
            clock=ClockContract(domain='local_interface', mode='asynchronous'),
            lifetime=LifetimeContract(arrival_ns=Interval(minimum=0., maximum=0.), integration_ns=p['symbol_ns'],
                                      retention_ns=known(0., 'ns'), reset_required=False),
            required_services=('power_supply', 'phase_reference') if field else ('power_supply',),
            phase_reference='laser.reference' if field else None)
    rows = []
    for name, (_, source, target) in _REGISTRY.items():
        ports = [PortContract(name='input' if i == 0 else 'control', direction='in', signal=signal(kind))
                 for i, kind in enumerate(source.split('+'))]
        ports.append(PortContract(name='output', direction='out', signal=signal(target)))
        rows.append(dict(id=name, evaluator_id='interface.'+name, evaluator_version='1.0',
                         direction=dict(source=source, target=target),
                         ports=[v.model_dump(mode='json') for v in ports],
                         parameters=[v.model_dump(mode='json') for v in params],
                         accounting_boundary='local_interface',
                         limitations=['scalar-lane port templates; bound shapes and scales validated by evaluator',
                                      'classical interfaces only; no arbitrary quantum-state conversion',
                                      'no detector noise or fabrication calibration model']))
    result = dict(schema_version='npp-multiphysics-interfaces-1', version='1.0', interfaces=rows,
                  assumptions=['hypothetical coefficients; configured local devices',
                               'interface directions are explicit; amplitude and intensity are distinct'],
                  parameter_hash=canonical_hash(p))
    return result


def evaluate_interface(kind, payload, parameters=None, context=None, *, gate=None, delay_ns=None):
    if kind not in _REGISTRY:
        fail('interface.kind', 'unregistered directional interface', 'unavailable_model')
    fn = _REGISTRY[kind][0]
    if fn in (signed_gate, attenuation_gate):
        if gate is None or delay_ns is not None:
            fail('interface', 'gate interface requires gate and forbids delay_ns')
        return fn(payload, gate, parameters, context)
    if fn in (optical_delay, digital_buffer):
        if delay_ns is None or gate is not None:
            fail('interface', 'alignment interface requires delay_ns and forbids gate')
        return fn(payload, delay_ns, parameters, context)
    if gate is not None or delay_ns is not None:
        fail('interface', 'unexpected gate or delay for conversion')
    return fn(payload, parameters, context)
