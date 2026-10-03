"""Explicit system support, transport and storage assumptions for mixed graphs.

These are lumped hypothetical scenario coefficients, not measured package data.
Time is scheduled by the orchestrator; this module does not hide an evaluator.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, DecimalException, localcontext
import math
import re

from .contracts import Parameter, canonical_json


# (canonical unit, illustrative value). A value is never inferred from a desired
# speedup. Cost coefficients may be replaced by explicit bounded/unknown values.
FIELDS = {
    'calibration_ns': ('ns', 5.),
    'calibration_energy_pj_per_component': ('pJ/component', 2.),
    'clock_startup_ns': ('ns', 10.),
    'clock_startup_energy_pj': ('pJ', 10.),
    'clock_static_power_mw': ('mW', .01),
    'electrical_route_ns_per_um': ('ns/um', .005),
    'electrical_route_energy_pj_per_byte_um': ('pJ/(byte*um)', .0001),
    'electrical_link_bytes_per_ns': ('byte/ns', 64.),
    'electrical_route_area_um2_per_um': ('um^2/um', .1),
    'electronic_buffer_retention_ns': ('ns', 1e6),
    'electronic_buffer_capacity_bytes': ('byte', 1048576.),
    'electronic_buffer_read_energy_pj_per_byte': ('pJ/byte', .04),
    'electronic_buffer_write_energy_pj_per_byte': ('pJ/byte', .05),
    'electronic_buffer_static_power_mw_per_byte': ('mW/byte', .000001),
    'electronic_buffer_area_um2_per_byte': ('um^2/byte', .1),
    'electronic_buffer_read_bytes_per_ns': ('byte/ns', 64.),
    'electronic_buffer_write_bytes_per_ns': ('byte/ns', 64.),
    'package_startup_ns': ('ns', 20.),
    'package_startup_energy_pj': ('pJ', 50.),
    'package_static_power_mw': ('mW', .05),
    'support_area_um2': ('um^2', 100.),
}
COST_FIELDS = {name for name in FIELDS if any(word in name for word in ('energy', 'power', 'area'))}
POSITIVE_FIELDS = {'electrical_link_bytes_per_ns', 'electronic_buffer_retention_ns',
                   'electronic_buffer_capacity_bytes', 'electronic_buffer_read_bytes_per_ns',
                   'electronic_buffer_write_bytes_per_ns'}


def _parameter(name, value):
    return {
        'name': name, 'quantity': {'state': 'known', 'value': float(value), 'unit': FIELDS[name][0]},
        'evidence': {'kind': 'hypothetical', 'source': 'Explicit NPP system-support scenario',
                     'location': 'npp.multiphysics.infrastructure.FIELDS', 'version': '1.0',
                     'scope': 'Lumped active package/clock/storage/link accounting, not measured hardware'},
        'uncertainty': {'kind': 'epistemic', 'description': 'Illustrative scenario point; physical uncertainty uncharacterized'},
        'instance_group': 'system.reference.v1', 'dependencies': [],
    }


def default_system():
    return normalize_system({
        'schema_version': 'npp-system-1', 'id': 'hypothetical_system', 'version': '1.0',
        'parameters': [_parameter(name, value) for name, (_, value) in sorted(FIELDS.items())],
        'boundary': 'Active modeled package: explicit startup, clock, calibration, routes and electronic storage; workload acquisition and fabrication excluded.',
        'assumptions': [
            'All default coefficients are hypothetical scenario values, never calibration evidence.',
            'Package/clock background energy is power times the complete scheduled elapsed time.',
            'Local component static energy already owned by each device must not be charged here again.',
            'Calibration is charged per installed component; periodic recalibration needs explicit additional tasks.',
            'Electronic routes are lumped links; their length is not a photonic/acoustic propagation model.',
        ],
    })


def _friendly_value(value, unit, name):
    if type(value) in (int, float):
        value = float(value)
    elif isinstance(value, str):
        match = re.fullmatch(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S+)\s*', value)
        if not match:
            raise ValueError(f'system.{name}: expected a number or explicit {unit} quantity')
        factor = '1'
        supplied = match[2]
        if supplied != unit:
            conversions = {'fJ': ('pJ', '.001'), 'nJ': ('pJ', '1000'),
                           'uJ': ('pJ', '1000000'), 'ps': ('ns', '.001'),
                           'us': ('ns', '1000'), 'ms': ('ns', '1000000'),
                           'uW': ('mW', '.001'), 'W': ('mW', '1000')}
            allowed = False
            for prefix, (canonical, multiplier) in conversions.items():
                if supplied.startswith(prefix) and canonical + supplied[len(prefix):] == unit:
                    factor, allowed = multiplier, True
                    break
            if not allowed:
                raise ValueError(f'system.{name}: expected canonical unit {unit}')
        try:
            with localcontext() as context:
                context.prec = max(32, len(match[1]) + 16)
                raw = Decimal(match[1])
                scaled = raw * Decimal(factor)
                value = float(scaled)
                if raw != 0 and (scaled == 0 or value == 0):
                    raise ValueError(f'system.{name}: nonzero quantity underflows')
        except (DecimalException, OverflowError) as exc:
            raise ValueError(f'system.{name}: quantity outside supported range') from exc
    else:
        raise ValueError(f'system.{name}: expected a quantity, not a boolean or arbitrary object')
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'system.{name}: expected finite nonnegative coefficient')
    return value


def normalize_system(raw):
    if not isinstance(raw, dict):
        raise ValueError('system must be a versioned parameter record or a named preset')
    if 'preset' in raw:
        if set(raw) - {'preset', 'overrides'} or raw['preset'] != 'hypothetical_system':
            raise ValueError('only the explicitly hypothetical_system preset is supported')
        result = default_system()
        overrides = raw.get('overrides', {})
        if not isinstance(overrides, dict) or set(overrides) - set(FIELDS):
            raise ValueError('unknown infrastructure coefficient override')
        for parameter in result['parameters']:
            name = parameter['name']
            if name in overrides:
                value = overrides[name]
                parameter['quantity'] = (deepcopy(value) if isinstance(value, dict) else
                    {'state': 'known', 'value': _friendly_value(value, FIELDS[name][0], name), 'unit': FIELDS[name][0]})
                parameter['evidence']['source'] = 'User-selected explicit hypothetical scenario override'
                parameter['evidence']['location'] = 'implementation.system.overrides.' + name
        return normalize_system(result)
    required = {'schema_version', 'id', 'version', 'parameters', 'boundary', 'assumptions'}
    if set(raw) != required or raw['schema_version'] != 'npp-system-1':
        raise ValueError('unsupported system schema or missing/unknown fields')
    if any(not isinstance(raw[k], str) or not raw[k].strip() for k in ('id', 'version', 'boundary')):
        raise ValueError('system requires a nonempty identity, version and accounting boundary')
    if not isinstance(raw['assumptions'], list) or not raw['assumptions'] or any(not isinstance(v, str) or not v.strip() for v in raw['assumptions']):
        raise ValueError('system assumptions must be nonempty explicit strings')
    if not isinstance(raw['parameters'], list) or len(raw['parameters']) != len(FIELDS):
        raise ValueError('system must declare every required coefficient exactly once')
    parameters = [Parameter.model_validate_json(canonical_json(p)) for p in raw['parameters']]
    by_name = {p.name: p for p in parameters}
    if len(by_name) != len(parameters) or set(by_name) != set(FIELDS):
        raise ValueError('system coefficients are missing, duplicated or unrecognized')
    for name, parameter in by_name.items():
        q = parameter.quantity
        if q.unit != FIELDS[name][0] or any(v is not None and v < 0 for v in (q.value, q.lower, q.upper)):
            raise ValueError(f'system.{name}: incorrect unit or negative value')
        if name not in COST_FIELDS:
            if q.state != 'known':
                raise ValueError(f'system.{name}: unresolved scheduling parameter; supply a known scenario value')
            if name in POSITIVE_FIELDS and q.value <= 0:
                raise ValueError(f'system.{name}: must be positive')
            if name == 'electronic_buffer_capacity_bytes' and (not q.value.is_integer() or q.value > 2**40):
                raise ValueError('system buffer capacity must be an integer <= 1 TiB')
        if any(dep not in by_name for dep in parameter.dependencies):
            raise ValueError(f'system.{name}: dependency references an undeclared parameter')
        if any(by_name[dep].instance_group != parameter.instance_group for dep in parameter.dependencies):
            raise ValueError(f'system.{name}: incompatible correlated instance assumptions')
    return {**deepcopy(raw), 'parameters': [by_name[name].model_dump(mode='json') for name in sorted(by_name)]}


def system_parameters(raw):
    system = normalize_system(raw)
    return {p['name']: deepcopy(p['quantity']) if p['name'] in COST_FIELDS else
            int(p['quantity']['value']) if p['name'] == 'electronic_buffer_capacity_bytes' else p['quantity']['value']
            for p in system['parameters']}
