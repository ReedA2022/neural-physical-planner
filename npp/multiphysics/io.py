"""Portable, explicit multiphysics projects; legacy input meaning is unchanged."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, DecimalException, localcontext
import json
import math
from pathlib import Path
import re

from npp.config import read_spec
from npp.weights import load_weights

SCHEMA = "npp-multiphysics-1"
MAX_PROJECT_BYTES = 16 * 1024 * 1024


def keys(value, allowed, required=(), label="object"):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{label}: expected a mapping with string keys")
    if set(value) - set(allowed):
        raise ValueError(f"{label}: unknown fields {sorted(set(value) - set(allowed))}")
    if set(required) - set(value):
        raise ValueError(f"{label}: missing fields {sorted(set(required) - set(value))}")


def number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a real number, not a boolean or string")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label}: outside finite numerical range") from exc
    if not math.isfinite(result) or minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise ValueError(f"{label}: outside the declared finite range")
    return result


def integer(value, label, minimum=0, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label}: expected an integer in [{minimum}, {maximum}]")
    return value


_UNITS = {
    "pJ": {"fJ": "0.001", "pJ": "1", "nJ": "1000", "uJ": "1e6", "J": "1e12"},
    "ns": {"ps": "0.001", "ns": "1", "us": "1000", "ms": "1e6", "s": "1e9"},
    "um": {"nm": "0.001", "um": "1", "mm": "1000", "m": "1e6"},
    "bytes": {"B": "1", "bytes": "1", "KiB": "1024", "MiB": "1048576"},
}


def quantity(value, unit, label):
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]+)\s*", value)
        if not match or match[2] not in _UNITS[unit]:
            raise ValueError(f"{label}: expected a quantity convertible to {unit}")
        try:
            with localcontext() as context:
                context.prec = max(32, len(match[1]) + 16)
                raw = Decimal(match[1])
                scaled = raw * Decimal(_UNITS[unit][match[2]])
                result = float(scaled)
                if raw != 0 and (scaled == 0 or result == 0):
                    raise ValueError(f"{label}: nonzero quantity underflows binary64")
        except (DecimalException, OverflowError) as exc:
            raise ValueError(f"{label}: invalid quantity") from exc
        return number(result, label, minimum=0)
    return number(value, label, minimum=0)


def _read(path):
    path = Path(path)
    if path.stat().st_size > MAX_PROJECT_BYTES:
        raise ValueError(f"{path}: exceeds the 16 MiB specification limit; use external weights")
    return read_spec(path)


def _resolve(value, base, label):
    if isinstance(value, str):
        path = (base / value).resolve()
        return _read(path), path.parent
    if isinstance(value, dict):
        return deepcopy(value), base
    raise ValueError(f"{label}: expected a mapping or a relative file path")


def normalize_model(raw, base=Path('.')):
    """Resolve named tensors with an explicit architecture and matrix orientation."""
    from .semantic import FFNWorkload
    raw = deepcopy(raw)
    if not isinstance(raw, dict):
        raise ValueError("model: expected a mapping")
    if 'weights' in raw:
        filename = raw.pop('weights')
        if not isinstance(filename, str):
            raise ValueError("model.weights: expected a file path")
        state_key = raw.pop('state_dict_key', None)
        tensors = load_weights(Path(base) / filename, state_dict_key=state_key)
        for name in ('W_up', 'W_gate', 'W_down', 'b_up', 'b_gate', 'b_down'):
            value = raw.get(name)
            if isinstance(value, str):
                if value not in tensors:
                    raise ValueError(f"model.{name}: tensor {value!r} not found")
                raw[name] = tensors[value].tolist()
    elif 'state_dict_key' in raw:
        raise ValueError("model.state_dict_key requires a weights file")
    layout = raw.pop('layout', 'out_in')
    if layout not in ('out_in', 'in_out'):
        raise ValueError("model.layout must be out_in or in_out")
    if layout == 'in_out':
        for name in ('W_up', 'W_gate', 'W_down'):
            matrix = raw.get(name)
            if not isinstance(matrix, list) or not matrix or any(not isinstance(row, list) for row in matrix) or len({len(row) for row in matrix}) != 1:
                raise ValueError(f"model.{name}: expected a rectangular matrix before transposing")
            raw[name] = [list(row) for row in zip(*matrix)]
    model = FFNWorkload.from_dict(raw)
    return model.to_dict()


def normalize_workload(raw):
    from .semantic import NumericPolicy
    keys(raw, {'inputs', 'semantic_track', 'numeric_policy', 'reuse_count', 'seed',
               'weight_residency', 'quality', 'evidence_policy'},
         {'inputs', 'semantic_track', 'numeric_policy', 'reuse_count', 'seed',
          'weight_residency', 'quality', 'evidence_policy'}, 'workload')
    if raw['semantic_track'] not in ('E', 'A', 'R'):
        raise ValueError("workload.semantic_track must be E, A or R")
    if raw['weight_residency'] not in ('sram', 'dram'):
        raise ValueError("workload.weight_residency must be sram or dram")
    integer(raw['reuse_count'], 'workload.reuse_count', 1)
    integer(raw['seed'], 'workload.seed', 0, 2**32 - 1)
    inputs = raw['inputs']
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 64 or any(not isinstance(row, list) or not 1 <= len(row) <= 256 for row in inputs):
        raise ValueError("workload.inputs must contain 1–64 rows of 1–256 real values")
    if len({len(row) for row in inputs}) != 1:
        raise ValueError("workload.inputs must be rectangular")
    inputs = [[number(v, 'workload.inputs') for v in row] for row in inputs]
    keys(raw['quality'], {'max_rms_error'}, {'max_rms_error'}, 'workload.quality')
    quality = {'max_rms_error': number(raw['quality']['max_rms_error'], 'workload.quality.max_rms_error', minimum=0)}
    evidence = raw['evidence_policy']
    supported = {'hypothetical', 'source_parameterized', 'independently_simulated', 'characterized', 'measured_instance'}
    if not isinstance(evidence, list) or not evidence or any(v not in supported for v in evidence) or len(evidence) != len(set(evidence)):
        raise ValueError("workload.evidence_policy must list distinct permitted evidence categories")
    policy = NumericPolicy.from_dict(raw['numeric_policy'])
    return {**deepcopy(raw), 'inputs': inputs, 'numeric_policy': policy.to_dict(), 'quality': quality}


def normalize_constraints(raw):
    keys(raw, {'resources', 'max_energy_pj', 'max_latency_ns', 'max_memory_bytes'},
         {'resources', 'max_energy_pj', 'max_latency_ns', 'max_memory_bytes'}, 'constraints')
    resources = raw['resources']
    if not isinstance(resources, dict) or not resources or any(not isinstance(k, str) or not k for k in resources):
        raise ValueError("constraints.resources must explicitly name positive resource capacities")
    return {'resources': {k: integer(v, f'resource.{k}', 1, 1024) for k, v in resources.items()},
            'max_energy_pj': quantity(raw['max_energy_pj'], 'pJ', 'max_energy_pj'),
            'max_latency_ns': quantity(raw['max_latency_ns'], 'ns', 'max_latency_ns'),
            'max_memory_bytes': quantity(raw['max_memory_bytes'], 'bytes', 'max_memory_bytes')}


def load_project(path):
    path = Path(path).resolve()
    return normalize_project(_read(path), path.parent)


def load_implementation(path, model):
    """Resolve separate physical-choice files beside their declaring spec.

    Only this input boundary reads files. Replay receives embedded snapshots and
    must never resolve paths taken from a saved record.
    """
    from .graph import normalize_implementation
    path = Path(path).resolve()
    raw = _read(path)
    if not isinstance(raw, dict):
        raise ValueError('implementation must be a mapping')
    for field in ('geometry', 'system', 'interfaces'):
        if isinstance(raw.get(field), str):
            raw[field], _ = _resolve(raw[field], path.parent, field)
    return normalize_implementation(raw, model)


def normalize_project(project, base=Path('.')):
    from .catalog import normalize_pack, reference_pack
    from .contracts import canonical_hash
    base = Path(base)
    keys(project, {'schema_version', 'name', 'model', 'workload', 'technology', 'constraints', 'environment', 'exploration'},
         {'schema_version', 'name', 'model', 'workload', 'technology', 'constraints', 'environment'}, 'project')
    if project['schema_version'] != SCHEMA or not isinstance(project['name'], str) or not project['name'].strip():
        raise ValueError(f"project requires schema_version={SCHEMA!r} and a nonempty name")
    resolved = {name: _resolve(project[name], base, name) for name in ('model', 'workload', 'technology', 'constraints', 'environment')}
    technology = resolved['technology'][0]
    if technology == {'preset': 'hypothetical_reference'}:
        technology = reference_pack()
    environment = resolved['environment'][0]
    keys(environment, {'temperature_c', 'region', 'substrate', 'services'},
         {'temperature_c', 'region', 'substrate', 'services'}, 'environment')
    number(environment['temperature_c'], 'environment.temperature_c', minimum=-273.15)
    if any(not isinstance(environment[k], str) or not environment[k] for k in ('region', 'substrate')):
        raise ValueError("environment requires named region and substrate")
    if not isinstance(environment['services'], list) or any(not isinstance(s, str) or not s for s in environment['services']):
        raise ValueError("environment.services must explicitly list available controls and supplies")
    result = {'schema_version': SCHEMA, 'name': project['name'],
              'model': normalize_model(*resolved['model']),
              'workload': normalize_workload(resolved['workload'][0]),
              'technology': normalize_pack(technology),
              'constraints': normalize_constraints(resolved['constraints'][0]),
              'environment': environment}
    if 'exploration' in project:
        exploration, _ = _resolve(project['exploration'], base, 'exploration')
        # M1 does not execute design searches. Later gates own this strict schema.
        if exploration:
            raise ValueError("exploration settings require a later implemented search contract")
        result['exploration'] = {}
    if result['model'].get('semantic_track', 'E') != result['workload']['semantic_track']:
        raise ValueError("model and workload semantic tracks disagree")
    from .semantic import evaluate_ffn, NumericPolicy
    evaluate_ffn(result['model'], result['workload']['inputs'], policy=NumericPolicy.from_dict(result['workload']['numeric_policy']))
    result['input_hash'] = canonical_hash(result)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def init_project(directory, overwrite=False):
    """Create all files after a collision preflight, with an explicit synthetic pack."""
    import numpy as np
    directory = Path(directory).resolve()
    names = ('project.yaml', 'model.yaml', 'workload.yaml', 'technology.yaml', 'constraints.yaml', 'environment.yaml', 'implementation.yaml', 'weights.npz')
    if directory.exists() and not directory.is_dir():
        raise ValueError("project destination is not a directory")
    for name in names:
        target = directory / name
        if target.exists() and (not overwrite or not target.is_file()):
            raise ValueError(f"output already exists: {target}")
    texts = {
        'project.yaml': '# Opt-in multiphysics project. All device coefficients are hypothetical.\n'
                        f'schema_version: {SCHEMA}\nname: Small SwiGLU study\nmodel: model.yaml\nworkload: workload.yaml\n'
                        'technology: technology.yaml\nconstraints: constraints.yaml\nenvironment: environment.yaml\n',
        'model.yaml': '# The architecture is explicit; weights alone cannot define it.\n'
                      'name: small_ffn\nsemantic_track: E\nweights: weights.npz\nlayout: out_in\n'
                      'W_up: up\nW_gate: gate\nW_down: down\nb_up: up_bias\nb_gate: gate_bias\nb_down: down_bias\n',
        'workload.yaml': '# Each row is one input vector. E preserves the ideal function.\n'
                         'inputs: [[0.25, -0.5], [-0.2, 0.4]]\nsemantic_track: E\n'
                         'numeric_policy: {format: float64, accumulation: float64, rounding: nearest_even, overflow: reject, underflow: gradual}\n'
                         'reuse_count: 1\nseed: 7\nweight_residency: sram\nquality: {max_rms_error: 0.1}\n'
                         'evidence_policy: [hypothetical]\n',
        'technology.yaml': '# Explicitly selected illustrative coefficients; no device calibration.\npreset: hypothetical_reference\n',
        'constraints.yaml': '# Capacities are simultaneous reservations, not advertised throughput.\n'
                            'resources: {digital: 1, analog: 1, photonic: 1, converter: 1, memory: 1, link: 1, control: 1, source: 1, buffer: 1}\n'
                            'max_energy_pj: 10 uJ\nmax_latency_ns: 1 ms\nmax_memory_bytes: 1 MiB\n',
        'environment.yaml': 'temperature_c: 25\nregion: room_temperature\nsubstrate: hypothetical_chiplet\n'
                            'services: [power_supply, clock, reference, cooling, laser, phase_reference, control]\n',
        'implementation.yaml': '# Compile with: npp mp compile --project project.yaml --implementation implementation.yaml --out-dir run\n'
                               '# Choices: digital, analog, photonic. Optional input_cuts/output_cuts partition each matrix axis.\n'
                               'schema_version: npp-implementation-1\nname: Editable mixed FFN example\n'
                               'operators:\n  up: {family: photonic}\n  gate: {family: analog}\n  down: {family: digital}\n'
                               '# Unfused explicitly reads out and multiplies the branches.\nfusion: unfused\n'
                               '# Explicit illustrative package, clock, storage and route costs. Exported records pin every coefficient.\n'
                               'system: {preset: hypothetical_system}\n',
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (directory / name).write_text(text, encoding='utf-8')
    np.savez(directory / 'weights.npz', up=[[.3, -.2], [.1, .4]], gate=[[.2, .1], [-.3, .2]],
             down=[[.5, -.4], [.2, .3]], up_bias=[.02, -.01], gate_bias=[0., .01], down_bias=[0., 0.])
    return {'status': 'created', 'project': str(directory / 'project.yaml'), 'files': list(names),
            'scope': 'Synthetic small FFN; physical coefficients are explicitly hypothetical.'}
