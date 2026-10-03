"""Presentation-only, offline inspection of an external optical comparison.

A report displays the stored comparison. It does not execute SAX, authenticate
artifacts, or establish calibrated device, noise, timing, or whole-NN accuracy.
"""
from __future__ import annotations

import html
import json
import math
import re
from typing import Any

from .implementation import _hash as _realization_hash
from .models import InputValidationError
from .optical_simulation import (ADAPTER_VERSION, BACKEND, MAX_ATOL_MW,
                                 MAX_MATRIX_ELEMENTS, MAX_RTOL, MAX_SEGMENTS,
                                 MAX_SEGMENT_INSTANCES, MAX_WAVELENGTHS,
                                 MODEL_DEFINITIONS)
from .technology import validate_technology


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _value(value: Any) -> str:
    if value is None:
        return '<span class="muted">Not recorded</span>'
    if isinstance(value, (dict, list)):
        return '<pre>' + _escape(json.dumps(value, indent=2, ensure_ascii=False)) + '</pre>'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return _escape(f'{value:.9g}')
    return _escape(value)


def _details(value: Any) -> str:
    value = _mapping(value)
    if not value:
        return '<p class="muted">No fields recorded.</p>'
    return '<div class="scroll"><table><tbody>' + ''.join(
        '<tr><th scope="row">' + _escape(key) + '</th><td>' + _value(item) + '</td></tr>'
        for key, item in value.items()) + '</tbody></table></div>'


def _table(rows: list[dict], *, table_id: str, preferred: tuple[str, ...] = ()) -> str:
    if not rows:
        return '<p class="muted">No rows recorded.</p>'
    columns = list(dict.fromkeys([key for key in preferred if any(key in row for row in rows)]
                                + [key for row in rows for key in row]))
    return (f'<div class="scroll"><table id="{_escape(table_id)}"><thead><tr>'
            + ''.join('<th scope="col">' + _escape(key) + '</th>' for key in columns)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td>' + _value(row.get(key)) + '</td>' for key in columns) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _number(value: Any, *, minimum: float = 0.0) -> bool:
    if type(value) not in (float, int):
        return False
    try:
        return math.isfinite(value) and value >= minimum
    except (OverflowError, ValueError):
        return False


_STYLE = '''
body{margin:0;background:#f4f7fa;color:#183044;font:15px/1.55 system-ui,sans-serif}
main{max-width:1240px;margin:auto;padding:30px 24px 60px}h1{font-size:30px;line-height:1.2}h2{font-size:21px;margin-top:0}h3{font-size:17px}p{max-width:100ch}.eyebrow{color:#386e8b;font-weight:700;text-transform:uppercase;letter-spacing:.07em;font-size:12px}.muted{color:#5e7384;font-size:13px}.banner{padding:16px 20px;background:#fff2d8;border-left:4px solid #c88e23}.panel{background:white;border:1px solid #dce4eb;border-radius:10px;padding:21px;margin-top:22px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce4eb;padding:10px 9px;overflow-wrap:anywhere}th{font-weight:650;color:#436073}pre{font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;margin:0}details{margin-top:16px}summary{cursor:pointer;font-weight:650}.status{display:inline-block;font-weight:650;padding:5px 10px;border-radius:6px;background:#e9eef4}.warning{background:#fff0db;color:#704e15}.bad{background:#fde9e7;color:#892b25}@media(max-width:650px){main{padding:20px 12px}.panel{padding:14px}h1{font-size:25px}}@media print{body{background:white}.panel{break-inside:avoid}.scroll{overflow:visible}pre{max-height:none}}
'''



def _export_structure_issues(exported: dict) -> list[str]:
    """Inspect supported portable structures; no graph or physics evaluation."""
    issues = []
    if exported.get('schema_version') != '0.3':
        issues.append('export: unsupported schema version')
    if exported.get('recipe') not in ('passive', 'regenerate'):
        issues.append('export: unsupported optical recipe')
    hashes = _mapping(exported.get('realization_hashes'))
    if set(hashes) != {'network', 'technology', 'spec', 'graph'} or any(
            not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None for value in hashes.values()):
        issues.append('export: all four SHA-256 realization hash fields are required; their authenticity is not checked here')
    technology = _mapping(exported.get('technology'))
    try:
        if _realization_hash(technology) != hashes.get('technology'):
            issues.append('export: embedded technology content does not match its recorded SHA-256 hash')
    except (TypeError, ValueError, OverflowError, RecursionError):
        issues.append('export: embedded technology cannot be encoded as canonical finite JSON')
    try:
        canonical = validate_technology(technology)
    except (InputValidationError, TypeError, ValueError, OverflowError):
        canonical = {}
        issues.append('export: technology snapshot is missing or violates its structural contract')
    if canonical:
        status = {key: canonical[key] for key in ('id', 'version', 'calibration_status', 'review')}
        if exported.get('technology_status') != status:
            issues.append('export: technology status disagrees with the supplied technology snapshot')
    components = {component['id']: component for component in canonical.get('components', [])}
    if exported.get('model_definitions') != MODEL_DEFINITIONS:
        issues.append('export: complete supported optical model definitions are required')
    operating = _mapping(exported.get('operating'))
    if (set(operating) != {'wavelength_nm', 'temperature_c', 'symbol_duration_ns'}
            or not _number(operating.get('wavelength_nm')) or operating['wavelength_nm'] <= 0
            or not _number(operating.get('symbol_duration_ns')) or operating['symbol_duration_ns'] <= 0
            or not _number(operating.get('temperature_c'), minimum=-273.15)):
        issues.append('export: operating point fields are missing or malformed')
    segments = _records(exported.get('segments'))
    if not 1 <= len(segments) <= MAX_SEGMENTS:
        issues.append('export: segment count is outside the supported adapter range')
    all_mapped, matrix_elements = set(), 0
    model_kinds = {'npp_splitter_v1': 'splitter', 'npp_waveguide_v1': 'waveguide',
                   'npp_modulator_full_scale_v1': 'modulator'}
    settings_keys = {'npp_splitter_v1': {'ratio', 'loss_db'},
                     'npp_waveguide_v1': {'length_um', 'loss_db_per_um', 'neff', 'ng', 'reference_wavelength_um'},
                     'npp_modulator_full_scale_v1': {'loss_db'}}
    for segment in segments:
        identifier = str(segment.get('id', 'unidentified segment'))
        netlist = _mapping(segment.get('netlist'))
        if set(netlist) != {'instances', 'connections', 'ports'} or any(not isinstance(netlist.get(key), dict)
                                                                       for key in ('instances', 'connections', 'ports')):
            issues.append(f'{identifier}: complete netlist instances, connections and ports are required')
        instances = _mapping(netlist.get('instances'))
        connections = _mapping(netlist.get('connections'))
        external_ports = _mapping(netlist.get('ports'))
        mappings = _mapping(segment.get('instance_mapping'))
        if not 1 <= len(instances) <= MAX_SEGMENT_INSTANCES or set(mappings) != set(instances):
            issues.append(f'{identifier}: instance mapping must cover every supported netlist instance exactly once')
        declared_ports = set()
        for local, instance in instances.items():
            instance = _mapping(instance)
            model = instance.get('component')
            if not isinstance(local, str) or not local.isidentifier() or not isinstance(model, str) or model not in MODEL_DEFINITIONS:
                issues.append(f'{identifier}: unsupported local identifier or optical model')
                continue
            declared_ports.update(f'{local},{port}' for port in MODEL_DEFINITIONS[model]['ports'])
            settings = _mapping(instance.get('settings'))
            if set(settings) != settings_keys[model] or any(not _number(value) for value in settings.values()):
                issues.append(f'{identifier}: optical model settings are missing or malformed')
            elif ((model == 'npp_splitter_v1' and not 0 < settings['ratio'] < 1)
                  or (model == 'npp_waveguide_v1' and any(settings[key] <= 0 for key in ('neff', 'ng', 'reference_wavelength_um')))):
                issues.append(f'{identifier}: optical model settings violate the supported domain')
            mapping = _mapping(mappings.get(local))
            physical_id, component_id = mapping.get('instance'), mapping.get('technology_component')
            if not isinstance(physical_id, str) or not physical_id or physical_id in all_mapped:
                issues.append(f'{identifier}: physical instance mapping is missing or duplicated')
            else:
                all_mapped.add(physical_id)
            component = components.get(component_id) if isinstance(component_id, str) else None
            if component is None or component['kind'] != model_kinds[model]:
                issues.append(f'{identifier}: technology component binding is unresolved or has a different kind')
        outputs = _records(segment.get('outputs'))
        source = _mapping(segment.get('source'))
        named_outputs = [output.get('sax_port') for output in outputs]
        if (source.get('sax_port') != 'input' or any(not isinstance(port, str) for port in named_outputs)
                or set(external_ports) != {'input', *[port for port in named_outputs if isinstance(port, str)]}):
            issues.append(f'{identifier}: external netlist ports must match the source and every output boundary')
        used_ports = list(external_ports.values()) + [port for pair in connections.items() for port in pair]
        if (any(not isinstance(port, str) or port not in declared_ports for port in used_ports)
                or len(used_ports) != len(declared_ports)
                or len({port for port in used_ports if isinstance(port, str)}) != len(used_ports)):
            issues.append(f'{identifier}: every declared model port must be connected or exposed exactly once')
        matrix_elements += len(external_ports) ** 2
    if matrix_elements > MAX_MATRIX_ELEMENTS:
        issues.append('export: external scattering matrix exceeds the supported adapter size')
    excluded = exported.get('excluded_electrical_connections')
    if not isinstance(excluded, list) or any(not isinstance(edge, dict) or set(edge) != {'source', 'target'} for edge in excluded):
        issues.append('export: excluded electrical connections must be explicitly recorded')
    return issues


def _presentation_issues(result: dict) -> list[str]:
    """Check stored coverage and arithmetic, never propagate an optical signal."""
    issues = []
    def rows(parent, key):
        value = parent.get(key)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            issues.append(f'{key}: expected a complete list of objects')
            return []
        return value
    def endpoint(value):
        return (isinstance(value, dict) and set(value) == {'instance', 'port'}
                and all(isinstance(value[key], str) and bool(value[key]) for key in ('instance', 'port')))
    def close(value, expected):
        return _number(value) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15)
    def same(value, expected):
        # JSON comparison preserves boolean/integer distinctions in NN bindings.
        return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)

    if result.get('schema_version') != '0.3':
        issues.append('result: unsupported comparison schema version')
    if result.get('kind') != 'sax_optical_comparison':
        issues.append('result: unexpected comparison record kind')
    engine = _mapping(result.get('engine'))
    if engine.get('name') != 'SAX' or any(not isinstance(engine.get(key), str) or not engine[key].strip()
                                        for key in ('version', 'backend', 'jax_version', 'adapter_version')):
        issues.append('engine: SAX identity and actual version fields must be recorded')
    tolerance = _mapping(result.get('tolerances'))
    rtol, atol = tolerance.get('rtol'), tolerance.get('atol_mw')
    tolerance_valid = _number(rtol) and _number(atol) and rtol <= MAX_RTOL and atol <= MAX_ATOL_MW
    if not tolerance_valid:
        issues.append(f'tolerances: supported bounds require 0 <= rtol <= {MAX_RTOL:g} and 0 <= atol_mw <= {MAX_ATOL_MW:g}, with finite real numbers')
    if tolerance.get('rule') != 'absolute_error_mw <= atol_mw + rtol*abs(expected_power_mw)':
        issues.append('tolerances: the recorded comparison rule is missing or unsupported')
    scope = _mapping(result.get('scope'))
    if scope.get('coverage') != 'partial_optical_only':
        issues.append('scope: the comparison must retain its partial optical coverage')
    for key in ('checked', 'unchecked'):
        if not isinstance(scope.get(key), list) or not scope[key] or any(not isinstance(x, str) for x in scope[key]):
            issues.append(f'scope: {key} must be explicitly recorded')
    exported = _mapping(result.get('export'))
    if not same(scope, exported.get('scope')):
        issues.append('scope: result and exported scope declarations disagree')
    if exported.get('kind') != 'sax_optical_export' or exported.get('baseline_feasible') is not True:
        issues.append('export: a feasible optical export is required')
    issues.extend(_export_structure_issues(exported))
    if engine.get('backend') != BACKEND or engine.get('adapter_version') != ADAPTER_VERSION:
        issues.append('engine: unsupported backend or adapter contract version')
    if exported.get('adapter_version') != engine.get('adapter_version'):
        issues.append('export: adapter versions disagree')
    if not _mapping(exported.get('technology_status')):
        issues.append('export: technology calibration and review status are absent')
    if not _mapping(exported.get('realization_hashes')):
        issues.append('export: source realization hashes are absent')
    segments = rows(exported, 'segments')
    expected = {}
    segment_ids = set()
    if not segments:
        issues.append('segments: no optical segments are recorded')
    for segment in segments:
        identifier = segment.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in segment_ids:
            issues.append('segments: identifiers must be unique nonempty strings')
            continue
        segment_ids.add(identifier)
        source = _mapping(segment.get('source'))
        if not endpoint(source.get('endpoint')) or not _number(source.get('power_mw')):
            issues.append(f'{identifier}: source boundary or power is malformed')
        outputs = rows(segment, 'outputs')
        if not outputs:
            issues.append(f'{identifier}: no output boundaries recorded')
        for output in outputs:
            port = output.get('sax_port')
            if not isinstance(port, str) or not port or (identifier, port) in expected:
                issues.append(f'{identifier}: output ports must be unique nonempty strings')
                continue
            if not endpoint(output.get('endpoint')) or output.get('kind') not in ('receiver_input', 'regeneration_detector_input', 'matched_termination'):
                issues.append(f'{identifier}: malformed output endpoint or boundary kind')
            expected[identifier, port] = {'source': source.get('endpoint'), **output}

    samples = rows(result, 'samples')
    sample_flags = []
    wavelengths = set()
    all_comparisons = []
    if not 1 <= len(samples) <= MAX_WAVELENGTHS:
        issues.append('samples: wavelength count is outside the supported adapter range')
    for sample in samples:
        wavelength = sample.get('wavelength_nm')
        if not _number(wavelength) or wavelength <= 0 or wavelength in wavelengths:
            issues.append('samples: wavelengths must be distinct finite positive numbers')
        else:
            wavelengths.add(wavelength)
        comparisons = rows(sample, 'comparisons')
        all_comparisons.extend(comparisons)
        observed, flags = set(), []
        for comparison in comparisons:
            segment, port = comparison.get('segment'), comparison.get('sax_port')
            key = (segment, port) if isinstance(segment, str) and isinstance(port, str) else None
            if key not in expected or key in observed:
                issues.append('comparisons: duplicate or undeclared boundary')
            else:
                observed.add(key)
                if any(not same(comparison.get(field), expected[key].get(field))
                       for field in ('source', 'endpoint', 'kind', 'binding')):
                    issues.append('comparisons: boundary or NN binding differs from the optical export')
            numbers = ('expected_power_mw', 'sax_power_mw', 'absolute_error_mw', 'allowed_error_mw')
            values_valid = all(_number(comparison.get(field)) for field in numbers)
            flag = comparison.get('passed')
            if type(flag) is not bool:
                issues.append('comparisons: passed must be a boolean')
            flags.append(flag)
            if not values_valid:
                issues.append('comparisons: powers and errors must be finite nonnegative numbers')
                continue
            reference, actual = comparison['expected_power_mw'], comparison['sax_power_mw']
            absolute = abs(actual - reference)
            if not close(comparison['absolute_error_mw'], absolute):
                issues.append('comparisons: recorded absolute error disagrees with stored powers')
            relative = comparison.get('relative_error')
            if (reference == 0 and relative is not None) or (reference != 0 and not close(relative, absolute / reference)):
                issues.append('comparisons: recorded relative error disagrees with stored powers')
            if tolerance_valid:
                allowed = atol + rtol * abs(reference)
                if not math.isfinite(allowed) or not close(comparison['allowed_error_mw'], allowed):
                    issues.append('comparisons: recorded allowed error disagrees with tolerances')
                if type(flag) is not bool or flag != (absolute <= allowed):
                    issues.append('comparisons: agreement claim disagrees with stored numbers and tolerances')
        if observed != set(expected) or len(comparisons) != len(expected):
            issues.append('comparisons: every exported boundary must occur exactly once at each wavelength')
        passed = sample.get('passed')
        if type(passed) is not bool or passed != (bool(flags) and all(value is True for value in flags)):
            issues.append('samples: aggregate agreement flag disagrees with comparison flags')
        sample_flags.append(passed)
    passed = result.get('passed')
    if (type(passed) is not bool or passed != (bool(sample_flags) and all(value is True for value in sample_flags))
            or result.get('status') != ('passed' if passed is True else 'failed')):
        issues.append('result: aggregate agreement status disagrees with sample flags')
    diagnostics = rows(result, 'diagnostics')
    if passed is True and diagnostics:
        issues.append('result: agreement claimed despite recorded diagnostics')
    summary = _mapping(result.get('summary'))
    for key, count in (('segment_count', len(segments)), ('wavelength_count', len(samples)),
                       ('comparison_count', len(all_comparisons))):
        if type(summary.get(key)) is not int or summary[key] != count:
            issues.append(f'summary: {key} disagrees with recorded coverage')
    absolute_errors = [row['absolute_error_mw'] for row in all_comparisons if _number(row.get('absolute_error_mw'))]
    relative_errors = [row['relative_error'] for row in all_comparisons if _number(row.get('relative_error'))]
    if not absolute_errors or not close(summary.get('max_absolute_error_mw'), max(absolute_errors)):
        issues.append('summary: maximum absolute error disagrees with comparison rows')
    if not relative_errors or not close(summary.get('max_relative_error'), max(relative_errors)):
        issues.append('summary: maximum relative error disagrees with comparison rows')
    return list(dict.fromkeys(issues))


def render_optical_simulation(result: dict) -> str:
    """Render a stored external-solver comparison without running any solver."""
    result = _mapping(result)
    exported = _mapping(result.get('export'))
    issues = _presentation_issues(result)
    if issues:
        status, style = 'Incomplete or inconsistent comparison record', 'warning'
    elif result.get('passed') is True:
        status, style = 'Recorded optical-power agreement within tolerance', 'status'
    else:
        status, style = 'Recorded optical-power disagreement', 'bad'
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NPP optical transfer comparison</title><style>{_STYLE}</style></head><body><main>',
             '<p class="eyebrow">Neural Physical Planner · external optical solver comparison</p>',
             '<h1>Optical power comparison</h1>', f'<p><span class="status {style}">{status}</span></p>',
             '<p class="muted">Stored comparison status: ' + _value(result.get('status'))
             + '. Stored agreement claim: ' + _value(result.get('passed'))
             + '. Report generation does not execute the external engine or authenticate the source artifacts.</p>',
             '<div class="banner"><strong>Partial optical composition check.</strong> Agreement concerns full-scale optical powers under the same declared component assumptions. It does not validate physical calibration, noise, timing, energy, whole-network inference or task accuracy. An external circuit solver supplies a composition check, not experimental ground truth.</div>',
             '<section class="panel"><h2>External engine and recorded versions</h2>', _details(result.get('engine')),
             '<p class="muted">Versions are copied from the simulation result. Rendering this document does not install or invoke these packages.</p></section>',
             '<section class="panel"><h2>Coverage, boundary conditions and exclusions</h2>', _details(result.get('scope')), '</section>',
             '<section class="panel"><h2>Technology, review and model evidence</h2>', _details(exported.get('technology_status')),
             '<h3>Exported model definitions</h3>', _details(exported.get('model_definitions')),
             '<h3>Technology parameter evidence</h3>',
             _table(_records(_mapping(exported.get('technology')).get('sources')), table_id='technology-evidence',
                    preferred=('id', 'kind', 'citation', 'version', 'coverage', 'limitations')),
             '<details><summary>Complete exported technology snapshot</summary>', _value(exported.get('technology')), '</details>',
             '<p class="muted">Technology provenance and parameter evidence are copied from the exported snapshot. Solver agreement does not change the recorded calibration or expert-review status.</p></section>',
             '<section class="panel"><h2>Comparison tolerances</h2>', _details(result.get('tolerances')),
             '<p class="muted">Absolute errors are in mW. The relative tolerance is dimensionless. Each boundary is tested separately at each wavelength.</p></section>',
             '<section class="panel"><h2>Presentation consistency</h2>']
    if issues:
        parts += ['<p>Stored claims are not presented as established agreement because the record is incomplete or internally inconsistent.</p><ul>',
                  ''.join('<li>' + _escape(issue) + '</li>' for issue in issues), '</ul>']
    else:
        parts.append('<p>Recorded boundary coverage, field types and comparison arithmetic are internally consistent. This does not authenticate the record or independently reproduce its optical calculations.</p>')
    parts += ['</section>', '<section class="panel"><h2>Recorded comparison summary</h2>']
    if issues:
        parts.append('<p class="warning">No complete consistent summary is available. Raw fields below retain the stored claims for inspection.</p>')
    parts += [_details(result.get('summary')), '</section>',
              '<section class="panel"><h2>Power at every optical boundary</h2><p class="muted">Expected powers come from the planner model; SAX powers come from the stored external-solver run. Boundary kinds distinguish receiver inputs, early regeneration-detector inputs and unused matched optical terminations.</p>']
    samples = _records(result.get('samples'))
    if not samples:
        parts.append('<p class="muted">No wavelength samples recorded.</p>')
    for index, sample in enumerate(samples):
        parts.append('<h3>Wavelength: ' + _value(sample.get('wavelength_nm')) + ' nm</h3>')
        parts.append('<p class="muted">Stored sample agreement claim: ' + _value(sample.get('passed')) + '.</p>')
        parts.append(_table(_records(sample.get('comparisons')), table_id=f'comparisons-{index}',
                            preferred=('segment', 'sax_port', 'endpoint', 'kind', 'expected_power_mw', 'sax_power_mw',
                                       'absolute_error_mw', 'allowed_error_mw', 'relative_error', 'passed', 'source', 'binding')))
    parts += ['</section>', '<section class="panel"><h2>Recorded diagnostics</h2>',
              _table(_records(result.get('diagnostics')), table_id='diagnostics', preferred=('code', 'path', 'message')), '</section>',
              '<section class="panel"><h2>Optical segments and excluded electrical connections</h2>',
              _table(_records(exported.get('segments')), table_id='segments', preferred=('id', 'source', 'outputs', 'instance_mapping', 'netlist')),
              '<h3>Electrical connections excluded from this optical solve</h3>',
              _table(_records(exported.get('excluded_electrical_connections')), table_id='excluded-connections', preferred=('source', 'target')), '</section>',
              '<section class="panel"><h2>Source provenance and reproducibility</h2>', _details(exported.get('realization_hashes')),
              '<p class="muted">The embedded technology snapshot is checked against its recorded SHA-256 value for content consistency. This is not an authorship signature or attestation. Network, specification and graph hashes are displayed without recomputation because their bodies are not included in this export.</p>',
              '<h3>Recorded operating point</h3>', _details(exported.get('operating')),
              '<details><summary>Complete stored comparison record</summary>', _value(result), '</details></section>',
              '<p class="muted">Self-contained document with no scripts, remote assets or network dependencies. The JSON artifacts and simulation adapter provide the reproducible workflow.</p></main></body></html>']
    return ''.join(parts)
