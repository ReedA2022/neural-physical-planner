"""Opt-in multiphysics command line; preserves the legacy command contracts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path

from npp import __version__
from npp.models import InputValidationError
from .contracts import canonical_hash
from .io import _read, init_project, keys, load_implementation, load_project, normalize_project, write_json
from .semantic import NumericalFailure


def _emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def output_directory(path, overwrite=False):
    directory = Path(path).resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir()) and not overwrite):
        raise ValueError(f"Output directory must be new or empty: {directory}; use --overwrite explicitly")
    return directory


def render_report(record, title='Multiphysics evaluation'):
    text = html.escape(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False))
    def escape(value):
        return html.escape(str(value))
    def amount(value):
        if not isinstance(value, dict):
            return escape(value)
        state = value.get('state')
        unit = escape(value.get('unit', ''))
        if state == 'known':
            return f'{value["value"]:.8g} {unit}'
        if state == 'bounded':
            return f'[{escape(value.get("lower", "unknown"))}, {escape(value.get("upper", "unknown"))}] {unit}'
        return 'Unavailable: ' + escape(value.get('reason', 'not declared'))
    evaluation = record.get('evaluation', record)
    ledger = evaluation.get('ledger', {})
    summary = '<p>Status: <strong>' + escape(evaluation.get('status', 'not evaluated')) + '</strong></p>'
    metrics = [(name, ledger[name]) for name in ('energy_pj', 'area_um2', 'duration_ns', 'memory_bytes') if name in ledger]
    schedule = evaluation.get('schedule', {})
    if 'makespan_ns' in schedule:
        metrics.append(('scheduled_duration_ns', {'state': 'known', 'value': schedule['makespan_ns'], 'unit': 'ns'}))
    if metrics:
        summary += '<table><caption>Modeled costs</caption><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
        summary += ''.join('<tr><td>' + escape(name) + '</td><td>' + amount(value) + '</td></tr>' for name, value in metrics)
        summary += '</tbody></table>'
    if evaluation.get('quality'):
        summary += '<h2>Quality on supplied inputs</h2><pre>' + escape(json.dumps(evaluation['quality'], indent=2)) + '</pre>'
    if evaluation.get('diagnostics'):
        summary += '<h2>Diagnostics</h2><pre>' + escape(json.dumps(evaluation['diagnostics'], indent=2)) + '</pre>'
    if schedule.get('tasks'):
        summary += '<h2>Schedule</h2><table><thead><tr><th>Task</th><th>Start (ns)</th><th>End (ns)</th></tr></thead><tbody>'
        summary += ''.join('<tr><td>' + escape(task['id']) + '</td><td>' + escape(task.get('start_ns')) + '</td><td>' + escape(task.get('end_ns')) + '</td></tr>' for task in schedule['tasks'])
        summary += '</tbody></table>'
    if evaluation.get('assumptions'):
        summary += '<h2>Assumptions</h2><ul>' + ''.join('<li>' + escape(a) + '</li>' for a in evaluation['assumptions']) + '</ul>'
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>' + html.escape(title) +
            '</title><style>body{font:16px system-ui;margin:3rem auto;max-width:1100px;padding:0 1rem;color:#182535}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:1.5rem;font-size:13px}'
            'table{width:100%;border-collapse:collapse;margin:1rem 0}th,td{text-align:left;padding:.5rem;border-bottom:1px solid #cbd5e1}'
            'caption{text-align:left;font-weight:700}summary{cursor:pointer;font-weight:700}'
            '.scope{border-left:4px solid #c98821;padding:1rem;background:#fff4df}</style><h1>' + html.escape(title) +
            '</h1><p class="scope">Declared-model research output. Hypothetical coefficients are not measured hardware.'
            ' Missing costs remain incomplete. Replay checks reproducibility, not independent physics.</p>' + summary +
            '<details><summary>Complete pinned record</summary><pre>' + text + '</pre></details></html>')


def evaluate_project(project):
    from .catalog import evaluate_model
    result = evaluate_model(project['model'], project['workload'], project['technology'],
                            family='digital', context=project['environment'])
    constraints = project['constraints']
    checks = []
    for name, quantity, limit in (
        ('energy_pj', result['ledger']['energy_pj'], constraints['max_energy_pj']),
        ('local_duration_ns', result['ledger']['duration_ns'], constraints['max_latency_ns']),
        ('sram_bytes', {'state': 'known', 'value': result['resources']['sram_bytes']}, constraints['max_memory_bytes']),
    ):
        lower = quantity.get('value') if quantity['state'] == 'known' else quantity.get('lower')
        upper = quantity.get('value') if quantity['state'] == 'known' else quantity.get('upper')
        status = 'failed' if lower is not None and lower > limit else 'passed' if upper is not None and upper <= limit else 'unresolved'
        checks.append({'metric': name, 'limit': limit, 'status': status})
    missing = sorted({'digital', 'memory', 'link', 'control'} - set(constraints['resources']))
    checks.append({'metric': 'declared_local_resources', 'missing': missing, 'status': 'failed' if missing else 'passed'})
    result['constraint_checks'] = checks
    if any(c['status'] == 'failed' for c in checks):
        result['status'] = 'resource_infeasible'
    elif any(c['status'] == 'unresolved' for c in checks) and result['status'] == 'model_feasible':
        result['status'] = 'conditional'
    record = {'schema_version': 'mp-evaluation/1', 'software_version': __version__,
              'inputs': deepcopy(project), 'family': 'digital', 'evaluation': result,
              'validation_kind': 'production evaluation; not independent validation'}
    record['record_hash'] = canonical_hash(record)
    return record


def check_evaluation(record):
    try:
        keys(record, {'schema_version', 'software_version', 'inputs', 'family', 'evaluation',
                      'validation_kind', 'record_hash'},
             {'schema_version', 'software_version', 'inputs', 'family', 'evaluation',
              'validation_kind', 'record_hash'}, 'evaluation record')
        if record['schema_version'] != 'mp-evaluation/1' or record['family'] != 'digital':
            raise ValueError('unsupported evaluation schema/family')
        inputs = record['inputs']
        if not isinstance(inputs, dict):
            raise ValueError('record inputs must be a portable project mapping')
        raw = {k: v for k, v in inputs.items() if k != 'input_hash'}
        if inputs.get('input_hash') != canonical_hash(raw):
            raise ValueError('normalized input hash mismatch')
        if any(not isinstance(raw.get(k), dict) for k in ('model', 'workload', 'technology', 'constraints', 'environment')):
            raise ValueError('replay requires embedded portable input snapshots, not file references')
        if 'weights' in raw['model']:
            raise ValueError('replay cannot load external weights')
        if canonical_hash(normalize_project(raw)) != canonical_hash(inputs):
            raise ValueError('recorded inputs are not the canonical normalized project')
        expected = evaluate_project(inputs)
        if canonical_hash(expected) != canonical_hash(record):
            raise ValueError('record differs from deterministic recomputation or pinned evaluator version')
        return {'valid': True, 'kind': 'replay', 'independent_validation': False, 'diagnostics': []}
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return {'valid': False, 'kind': 'replay', 'independent_validation': False,
                'diagnostics': [{'code': 'replay_mismatch', 'message': str(exc)}]}


def parser():
    cli = argparse.ArgumentParser(prog='npp mp', description='Explicit multiphysics research models and reproducible studies.')
    commands = cli.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create a commented small FFN project with saved weights.')
    init.add_argument('directory')
    init.add_argument('--overwrite', action='store_true')
    for name in ('validate', 'normalize', 'evaluate'):
        child = commands.add_parser(name)
        child.add_argument('--project', required=True)
        if name != 'validate':
            child.add_argument('--out-dir', required=True)
            child.add_argument('--overwrite', action='store_true')
    technology = commands.add_parser('technology', help='Export the explicitly hypothetical multiphysics pack.')
    technology.add_argument('--out', required=True)
    technology.add_argument('--overwrite', action='store_true')
    interfaces = commands.add_parser('interfaces', help='Export versioned, directional interface contracts and coefficients.')
    interfaces.add_argument('--out', required=True)
    interfaces.add_argument('--overwrite', action='store_true')
    compile_cmd = commands.add_parser('compile', help='Compile and execute a typed, scheduled FFN implementation.')
    compile_cmd.add_argument('--project', required=True)
    choice = compile_cmd.add_mutually_exclusive_group(required=True)
    choice.add_argument('--implementation', help='YAML or JSON implementation choices.')
    choice.add_argument('--kind', choices=('digital', 'analog', 'photonic', 'hybrid', 'fused'), help='Explicit reference implementation preset.')
    compile_cmd.add_argument('--out-dir', required=True)
    compile_cmd.add_argument('--overwrite', action='store_true')
    check_impl = commands.add_parser('check-implementation', help='Replay a compiled implementation and its full context.')
    check_impl.add_argument('--record', required=True)
    check = commands.add_parser('check', help='Replay a saved evaluation; this is not independent validation.')
    check.add_argument('--record', required=True)
    tile = commands.add_parser('tile', help='Evaluate a bounded signed analog matrix tile and its converter chain.')
    tile.add_argument('--spec', required=True)
    tile.add_argument('--out-dir', required=True)
    tile.add_argument('--overwrite', action='store_true')
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'init':
            _emit(init_project(args.directory, args.overwrite))
            return 0
        if args.command in ('technology', 'interfaces'):
            from .catalog import reference_pack
            from .interfaces import interface_catalog
            path = Path(args.out)
            if path.exists() and (not args.overwrite or not path.is_file()):
                raise ValueError(f'Output exists: {path}')
            write_json(path, reference_pack() if args.command == 'technology' else interface_catalog())
            _emit({'status': 'exported', args.command: str(path.resolve())})
            return 0
        if args.command == 'check':
            result = check_evaluation(_read(args.record))
            _emit(result)
            return 0 if result['valid'] else 2
        if args.command == 'check-implementation':
            from .graph import check_implementation
            result = check_implementation(_read(args.record))
            _emit(result)
            return 0 if result['valid'] else 2
        directory = output_directory(args.out_dir, args.overwrite) if args.command != 'validate' else None
        if args.command == 'tile':
            from .backends.analog import evaluate_linear
            spec = _read(args.spec)
            keys(spec, {'weights', 'inputs', 'bias', 'parameters', 'context'}, {'weights', 'inputs', 'parameters'}, 'tile')
            result = evaluate_linear(spec['weights'], spec['inputs'], spec.get('bias'),
                                     parameters=spec.get('parameters'), context=spec.get('context'))
            write_json(directory / 'tile.json', result)
            (directory / 'report.html').write_text(render_report(result, 'Analog matrix tile'), encoding='utf-8')
            _emit({'status': result.get('status'), 'result': str(directory / 'tile.json')})
            return 0 if result.get('status') in ('model_feasible', 'conditional') else 3 if result.get('status') == 'resource_infeasible' else 2
        project = load_project(args.project)
        if args.command == 'validate':
            _emit({'valid': True, 'input_hash': project['input_hash'], 'scope': 'Normalized FFN project; implementation compatibility is checked by compile.'})
            return 0
        if args.command == 'compile':
            from .graph import default_implementation, evaluate_implementation, check_implementation
            spec = load_implementation(args.implementation, project['model']) if args.implementation else default_implementation(project['model'], args.kind)
            record = evaluate_implementation(project, spec)
            check = check_implementation(record)
            if not check['valid']:
                raise ValueError(f'Internal implementation replay failure: {check}')
            write_json(directory / 'implementation.json', record)
            write_json(directory / 'check.json', check)
            files = ['implementation.json', 'check.json', 'report.html']
            for key in ('normalized_inputs', 'semantic_plan', 'physical_graph', 'interfaces', 'geometry', 'schedule', 'resources', 'ledger', 'quality', 'assumptions'):
                write_json(directory / (key + '.json'), record[key])
                files.append(key + '.json')
            (directory / 'report.html').write_text(render_report(record, 'Compiled multiphysics implementation'), encoding='utf-8')
            write_json(directory / 'manifest.json', {'software_version': __version__, 'input_hash': project['input_hash'],
                                                   'record_hash': record['record_hash'], 'files': files})
            _emit({'status': record['status'], 'record': str(directory / 'implementation.json'), 'replay_valid': True})
            return 0 if record['status'] in ('model_feasible', 'conditional') else 3 if record['status'] == 'resource_infeasible' else 2
        if args.command == 'normalize':
            write_json(directory / 'project.json', {k: v for k, v in project.items() if k != 'input_hash'})
            write_json(directory / 'manifest.json', {'software_version': __version__, 'input_hash': project['input_hash'], 'files': ['project.json']})
            _emit({'status': 'normalized', 'project': str(directory / 'project.json'), 'input_hash': project['input_hash']})
            return 0
        record = evaluate_project(project)
        check = check_evaluation(record)
        if not check['valid']:
            raise ValueError(f'Internal replay failure: {check["diagnostics"]}')
        write_json(directory / 'evaluation.json', record)
        write_json(directory / 'check.json', check)
        (directory / 'report.html').write_text(render_report(record), encoding='utf-8')
        write_json(directory / 'manifest.json', {'software_version': __version__, 'input_hash': project['input_hash'],
                                               'files': ['evaluation.json', 'check.json', 'report.html']})
        _emit({'status': record['evaluation'].get('status'), 'record': str(directory / 'evaluation.json'), 'replay_valid': True})
        return 0 if record['evaluation'].get('status') in ('model_feasible', 'conditional') else 3 if record['evaluation'].get('status') == 'resource_infeasible' else 2
    except NumericalFailure as exc:
        _emit({'status': 'numerical_failure', 'diagnostics': [{'code': 'numerical_failure', 'message': str(exc)}]})
        return 2
    except InputValidationError as exc:
        codes = {d.get('code') for d in exc.diagnostics}
        status = ('numerical_failure' if 'numerical_failure' in codes else 'unsupported' if 'unavailable_model' in codes
                  else 'resource_infeasible' if 'resource_infeasibility' in codes else 'invalid')
        _emit({'status': status, 'diagnostics': exc.diagnostics})
        return 3 if status == 'resource_infeasible' else 2
    except ArithmeticError as exc:
        _emit({'status': 'numerical_failure', 'diagnostics': [{'code': type(exc).__name__, 'message': str(exc)}]})
        return 2
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        _emit({'status': 'invalid', 'diagnostics': [{'code': type(exc).__name__, 'message': str(exc)}]})
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
