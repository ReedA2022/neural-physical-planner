"""Offline comparison of physical designs, gated by bounded nominal replay.

The replay verifies the submitted finite design search under its declared model.
It does not invoke SAX, attest authorship, or validate calibrated hardware.
"""
from __future__ import annotations

import html
import json
import math
import re
from typing import Any


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
        return '<pre>' + _escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return _escape(f'{value:.8g}')
    return _escape(value)


def _details(value: Any) -> str:
    value = _mapping(value)
    if not value:
        return '<p class="muted">No fields recorded.</p>'
    return '<div class="scroll"><table><tbody>' + ''.join(
        '<tr><th scope="row">' + _escape(key) + '</th><td>' + _value(item) + '</td></tr>'
        for key, item in value.items()) + '</tbody></table></div>'


def _table(rows: list[dict], *, table_id: str, columns: tuple[str, ...] = ()) -> str:
    if not rows:
        return '<p class="muted">No entries recorded.</p>'
    keys = list(columns) if columns else list(dict.fromkeys(key for row in rows for key in row))
    return (f'<div class="scroll"><table id="{_escape(table_id)}"><thead><tr>'
            + ''.join('<th scope="col">' + _escape(key) + '</th>' for key in keys)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td>' + _value(row.get(key)) + '</td>' for key in keys) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _local_plan_link(identifier: Any) -> str | None:
    # Only supported candidate directory names may become relative hyperlinks.
    if not isinstance(identifier, str) or re.fullmatch(r'candidate-[0-9]{4,}', identifier) is None:
        return None
    return f'plans/{identifier}/report.html'


_AXES = ('energy_pj', 'latency_ns', 'area_um2', 'noise_rms_estimate')


def _scatter(candidates: list[dict]) -> str:
    rows = [row for row in candidates if row.get('status') == 'feasible']
    if not rows:
        return '<p class="muted">No feasible candidates are available for this projection.</p>'
    metrics = [_mapping(row.get('metrics')) for row in rows]
    for values in metrics:
        for key in ('energy_pj', 'latency_ns'):
            value = values.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                return '<p class="muted">Projection unavailable because metric fields are malformed.</p>'
    maximum_x = max(values['energy_pj'] for values in metrics) or 1.0
    maximum_y = max(values['latency_ns'] for values in metrics) or 1.0
    pieces = ['<svg class="tradeoff" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 330" role="img" aria-label="Feasible candidate energy versus latency; all four metrics are shown in the table">',
              '<path d="M80 24 V270 H690" stroke="#a8b8c7" fill="none"/>']
    for index in range(5):
        proportion = index / 4
        x, y = 80 + 610 * proportion, 270 - 246 * proportion
        pieces.append(f'<path d="M80 {y:g} H690" stroke="#e3ebf1"/><text x="{x:g}" y="289" text-anchor="middle">{maximum_x*proportion:.4g}</text><text x="72" y="{y+4:g}" text-anchor="end">{maximum_y*proportion:.4g}</text>')
    for row, values in zip(rows, metrics):
        x, y = 80 + 610 * (values['energy_pj'] / maximum_x), 270 - 246 * (values['latency_ns'] / maximum_y)
        colour = '#096e92' if row.get('pareto') else '#afbac5'
        label = f"{row.get('id')}: {values['energy_pj']:.8g} pJ; {values['latency_ns']:.8g} ns; " + ('retained Pareto candidate' if row.get('pareto') else 'feasible candidate')
        pieces.append(f'<circle cx="{x:g}" cy="{y:g}" r="5" stroke="white" fill="{colour}"><title>{_escape(label)}</title></circle>')
    pieces += ['<text x="385" y="321" text-anchor="middle">Full-scale energy per symbol (pJ)</text>',
               '<text x="17" y="146" transform="rotate(-90 17 146)" text-anchor="middle">Complete-symbol latency (ns)</text></svg>',
               '<p class="muted">Blue: retained Pareto candidates. Gray: other feasible candidates. This two-axis projection omits area and noise, which remain visible in the comparison table; overlapping points can represent different circuits.</p>']
    return ''.join(pieces)


_STYLE = '''
body{margin:0;background:#f4f7fa;color:#183044;font:15px/1.55 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:30px 24px 60px}h1{font-size:30px;line-height:1.2}h2{font-size:21px;margin-top:0}h3{font-size:17px}p{max-width:100ch}.eyebrow{color:#386e8b;font-weight:700;text-transform:uppercase;letter-spacing:.07em;font-size:12px}.muted{color:#5e7384;font-size:13px}.banner{padding:16px 20px;background:#fff2d8;border-left:4px solid #c88e23}.panel{background:white;border:1px solid #dce4eb;border-radius:10px;padding:21px;margin-top:22px}.scroll{overflow:auto}.comparison-scroll{max-height:520px}.comparison-scroll th{position:sticky;top:0;background:white}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce4eb;padding:10px 9px;overflow-wrap:anywhere}th{font-weight:650;color:#436073}pre{font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:340px;overflow:auto;margin:0}details{margin-top:16px}summary{cursor:pointer;font-weight:650}.status{display:inline-block;font-weight:650;padding:5px 10px;border-radius:6px;background:#e9eef4}.warning{background:#fff0db;color:#704e15}.bad{background:#fde9e7;color:#892b25}.facts{display:flex;flex-wrap:wrap;gap:14px}.fact{border:1px solid #dce4eb;border-radius:8px;padding:14px;min-width:140px}.fact strong{display:block;font-size:23px}a{color:#096e92}.tradeoff{display:block;width:100%;max-width:850px;height:auto;margin:auto}.tradeoff text{font:12px system-ui,sans-serif;fill:#355367}@media(max-width:650px){main{padding:20px 12px}.panel{padding:14px}h1{font-size:25px}}@media print{body{background:white}.panel{break-inside:avoid}.scroll{overflow:visible}pre{max-height:none}}
'''


def render_physical_design(result: dict) -> str:
    """Render a design result after the core's bounded nominal-search replay.

    Unlike a simple formatter this repeats ``check_physical_design`` before
    presenting feasibility or Pareto claims. The checker enforces its input and
    work limits before evaluation. No SAX solve or hardware calibration occurs.
    Invalid inputs remain inspectable and cannot produce candidate file links.
    """
    original = result
    result = _mapping(result)
    try:
        from .designer import check_physical_design
        check = _mapping(check_physical_design(result))
    except Exception as exc:
        check = {'valid': False, 'diagnostics': [{'code': 'design_report_replay_failed', 'path': 'result',
                                                'message': f'Bounded nominal replay could not complete ({type(exc).__name__}): {exc}'}]}
    valid = check.get('valid') is True
    inputs = _mapping(result.get('inputs'))
    network = _mapping(inputs.get('network'))
    technology = _mapping(inputs.get('technology'))
    design = _mapping(inputs.get('design'))
    search = _mapping(result.get('search'))
    candidates = _records(result.get('candidates'))
    plans = _records(result.get('plans'))
    plan_ids = {plan['id'] for plan in plans if isinstance(plan.get('id'), str)}
    title = network.get('name', 'Physical design alternatives')
    if not valid:
        headline, css = 'Incomplete, inconsistent or unsupported design record', 'warning'
        qualification = 'Stored candidate fields below are unverified claims. No feasibility or Pareto conclusion is established.'
    elif search.get('feasible_count') == 0:
        headline, css = 'No feasible candidate in the evaluated search', 'warning'
        qualification = 'Nominal replay confirms the recorded rejections. This does not establish that every possible hardware implementation is infeasible.'
    else:
        headline, css = 'Nominal design search replay passed', 'status'
        qualification = 'Recorded feasibility and Pareto membership agree with bounded replay of the submitted design request under its declared physical model.'
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{_escape(title)} — NPP designer</title><style>{_STYLE}</style></head><body><main>',
             '<p class="eyebrow">Neural Physical Planner · designer alternatives</p>', f'<h1>{_escape(title)}</h1>',
             f'<p><span class="status {css}">{headline}</span></p><p>{qualification}</p>',
             '<div class="banner"><strong>Nominal fanout-subcircuit design assistance.</strong> Generating this report repeats the core’s bounded finite-search replay. It does not run an external SAX simulation, calibrate devices, certify a fabrication layout, or establish whole-network task accuracy. Energy is a full-scale per-symbol quantity; noise is a model estimate. Component provenance and exclusions remain decisive.</div>',
             '<section class="panel"><h2>Search coverage and limits</h2>']
    if valid:
        parts.append('<p><strong>' + ('Complete for the submitted finite grid.' if search.get('complete') is True
                                      else 'Incomplete search of the submitted finite grid.') + '</strong>')
        parts.append(' Pareto membership is relative to the evaluated feasible candidates and the requested objectives; this is not a global optimum over all hardware designs.</p><div class="facts">')
        for key, label in [('submitted_grid_size', 'Submitted candidates'), ('evaluated_count', 'Evaluated'),
                           ('excluded_count', 'Excluded by choices or locks'), ('not_evaluated_count', 'Not evaluated'),
                           ('feasible_count', 'Feasible'), ('pareto_count', 'Retained Pareto candidates')]:
            parts.append('<div class="fact"><strong>' + _value(search.get(key)) + '</strong>' + label + '</div>')
        parts.append('</div>')
    else:
        parts.append('<p class="muted">Search counters are raw stored fields because replay did not validate this record.</p>')
    parts += [_details(search), '</section>',
              '<section class="panel"><h2>Objectives, hard limits and allowed design choices</h2>', _details(design),
              '<h3>Resolved locks</h3>', _details(result.get('resolved_locks')),
              '<p class="muted">Locks and forbidden choices constrain candidate generation. Hard limits reject evaluated candidates; they are not additional optimization objectives.</p></section>',
              '<section class="panel"><h2>Replay result</h2>', _details(check), '</section>',
              '<section class="panel"><h2>Alternative comparison</h2>']
    if valid:
        parts.append(_scatter(candidates))
    else:
        parts.append('<p class="warning">This table shows raw candidate claims. Trade-off graphics, Pareto conclusions and per-plan links are disabled until replay succeeds.</p>')
    parts.append('<div class="scroll comparison-scroll"><table id="candidate-comparison"><thead><tr><th>Candidate</th><th>Recorded status</th><th>Recorded Pareto flag</th><th>Energy (pJ/symbol)</th><th>Latency (ns)</th><th>Area (µm²)</th><th>Normalized noise RMS estimate</th></tr></thead><tbody>')
    for candidate in candidates:
        identifier = candidate.get('id')
        link = _local_plan_link(identifier) if valid and isinstance(identifier, str) and identifier in plan_ids else None
        label = f'<a href="{_escape(link)}">{_escape(identifier)}</a>' if link else _value(identifier)
        values = _mapping(candidate.get('metrics'))
        parts.append('<tr><td>' + label + '</td><td>' + _value(candidate.get('status')) + '</td><td>'
                     + _value(candidate.get('pareto')) + '</td>'
                     + ''.join('<td>' + _value(values.get(key)) + '</td>' for key in _AXES) + '</tr>')
    parts += ['</tbody></table></div><p class="muted">Linked retained candidates open their complete circuit report in this same output directory. Infeasible rows can lack aggregate metrics; missing values are not zero costs.</p></section>',
              '<section class="panel"><h2>Baseline and design changes</h2>']
    baseline = _mapping(inputs.get('baseline'))
    if not baseline:
        parts.append('<p>No baseline realization was supplied. Changes and metric deltas relative to a baseline are unavailable.</p>')
    else:
        parts += ['<h3>Baseline recipe and operating settings</h3>', _details(baseline.get('spec')),
                  '<h3>Baseline model metrics</h3>', _details(_mapping(baseline.get('evaluation')).get('metrics')),
                  '<p class="muted">Candidate comparisons explicitly report changes in operating conditions and technology assumptions. Deltas are interpretable only when the recorded comparison marks the metrics comparable.</p>']
    parts.append('</section>')
    rejections = [{**reason, 'candidate': candidate.get('id')} for candidate in candidates
                  for reason in _records(candidate.get('reasons'))]
    parts += ['<section class="panel"><h2>Rejection and exclusion overview</h2>',
              _table(rejections, table_id='rejection-overview'), '</section>']
    for index, candidate in enumerate(candidates):
        parts += [f'<details class="panel" id="candidate-detail-{index}"><summary>{_escape(candidate.get("id", "Unnamed candidate"))} · {_escape(candidate.get("status", "unrecorded"))}</summary>',
                  '<p>Recorded status: <strong>' + _value(candidate.get('status')) + '</strong>.</p>',
                  '<h3>Chosen recipe and settings</h3>', _details(candidate.get('spec')),
                  '<h3>Actual rejection guards and constraint diagnostics</h3>',
                  _table(_records(candidate.get('reasons')), table_id=f'candidate-reasons-{index}'),
                  '<h3>Recorded cost drivers</h3><p class="muted">Up to five evaluated instances, ordered by recorded energy and area. For an infeasible candidate this can be a partial ledger; these entries are not whole-circuit totals.</p>', _table(_records(candidate.get('cost_drivers')), table_id=f'cost-drivers-{index}'),
                  '<h3>Lowest margin among evaluated terminal receivers</h3>', _details(candidate.get('worst_receiver'))]
        comparison = _mapping(candidate.get('baseline_comparison'))
        if comparison:
            parts += ['<h3>Changes relative to baseline</h3>',
                      _details({key: value for key, value in comparison.items() if key != 'changed_instances'}),
                      _table(_records(comparison.get('changed_instances')), table_id=f'changed-instances-{index}',
                             columns=('id', 'change', 'before', 'after'))]
        parts.append('</details>')
    parts += ['<section class="panel"><h2>Single-constraint relaxation candidates</h2><p class="muted">These stored alternatives identify evaluated candidates that fail exactly one requirement. They do not relax that requirement automatically or establish feasibility outside the recorded model.</p>',
              _table(_records(result.get('single_constraint_relaxations')), table_id='single-constraint-relaxations'), '</section>',
              '<section class="panel"><h2>Technology evidence and model scope</h2>',
              _details({key: technology.get(key) for key in ('id', 'name', 'version', 'calibration_status', 'review', 'encoding')}),
              _details(result.get('scope')),
              '<h3>Assumptions and excluded effects</h3>',
              _details({key: technology.get(key) for key in ('assumptions', 'excluded_effects')}),
              '<h3>Parameter evidence</h3>', _table(_records(technology.get('sources')), table_id='technology-evidence'), '</section>',
              '<section class="panel"><h2>Reproducibility</h2>', _details(result.get('hashes')),
              '<details><summary>Complete stored design result</summary>', _value(original), '</details></section>',
              '<p class="muted">Standalone offline report with no scripts or remote assets. Relative links target retained circuit reports produced alongside this file. Nominal replay is a content and computation check, not an authorship signature or a measurement of hardware.</p></main></body></html>']
    return ''.join(parts)
