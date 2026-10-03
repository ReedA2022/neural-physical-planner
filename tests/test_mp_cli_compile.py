"""Public M2 workflow: portable compilation, inspectable failures and replay."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from npp.cli import main
from npp.multiphysics.cli import render_report
from npp.multiphysics.contracts import canonical_hash
from npp.multiphysics.io import init_project, load_implementation, load_project, write_json


class CompileWorkflowTests(unittest.TestCase):
    def invoke(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(['mp', *map(str, args)])
        return code, json.loads(out.getvalue())

    def test_starter_compilation_is_portable_and_writes_complete_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root / 'model with spaces été'
            init_project(project)
            command = ('compile', '--project', project / 'project.yaml', '--implementation', project / 'implementation.yaml', '--out-dir', root / 'run')
            code, result = self.invoke(*command)
            self.assertEqual(code, 0, result)
            path = Path(result['record'])
            record = json.loads(path.read_text())
            manifest = json.loads((path.parent / 'manifest.json').read_text())
            self.assertEqual(manifest['record_hash'], record['record_hash'])
            for filename in manifest['files']:
                self.assertTrue((path.parent / filename).is_file(), filename)
            for field in ('normalized_inputs', 'semantic_plan', 'physical_graph', 'interfaces', 'geometry', 'schedule', 'resources', 'ledger', 'quality', 'assumptions'):
                self.assertEqual(json.loads((path.parent / (field + '.json')).read_text()), record[field])
            (project / 'weights.npz').unlink()
            self.assertEqual(self.invoke('check-implementation', '--record', path)[0], 0)
            before = path.read_bytes()
            self.assertEqual(self.invoke(*command)[0], 2)
            self.assertEqual(path.read_bytes(), before)
            report = (path.parent / 'report.html').read_text()
            self.assertIn('Schedule</h2>', report)
            self.assertIn('Modeled costs', report)

    def test_preset_and_budget_violation_have_replayable_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            init_project(root / 'project')
            project = load_project(root / 'project/project.yaml')
            project.pop('input_hash')
            project['constraints']['max_energy_pj'] = 0
            write_json(root / 'tight.json', project)
            code, result = self.invoke('compile', '--project', root / 'tight.json', '--kind', 'digital', '--out-dir', root / 'run')
            self.assertEqual(code, 3, result)
            self.assertEqual(result['status'], 'resource_infeasible')
            self.assertEqual(self.invoke('check-implementation', '--record', result['record'])[0], 0)

    def test_rehashed_tampered_record_and_external_reference_are_rejected(self):
        from npp.multiphysics.graph import evaluate_implementation, check_implementation
        with tempfile.TemporaryDirectory() as folder:
            init_project(folder)
            record = evaluate_implementation(load_project(Path(folder) / 'project.yaml'))
            for mutation in ('cost', 'schedule', 'project_reference', 'weights_reference', 'nonmapping'):
                bad = deepcopy(record)
                if mutation == 'cost':
                    bad['ledger']['energy_pj']['value'] = 0
                elif mutation == 'schedule':
                    bad['schedule']['makespan_ns'] = 0
                elif mutation == 'project_reference':
                    bad['normalized_inputs']['project'] = str(Path(folder) / 'project.yaml')
                elif mutation == 'weights_reference':
                    bad['normalized_inputs']['project']['model']['weights'] = str(Path(folder) / 'weights.npz')
                else:
                    bad['normalized_inputs'] = []
                bad['record_hash'] = canonical_hash({k: v for k, v in bad.items() if k != 'record_hash'})
                with self.subTest(mutation=mutation):
                    self.assertFalse(check_implementation(bad)['valid'])

    def test_invalid_spec_does_not_create_output_and_interfaces_are_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            init_project(root / 'project')
            write_json(root / 'bad.json', {'schema_version': 'npp-implementation-1', 'operators': {}, 'secret': 2})
            code, result = self.invoke('compile', '--project', root / 'project/project.yaml', '--implementation', root / 'bad.json', '--out-dir', root / 'run')
            self.assertEqual(code, 2, result)
            self.assertFalse((root / 'run').exists())
            code, result = self.invoke('interfaces', '--out', root / 'interfaces.json')
            self.assertEqual(code, 0, result)
            exported = json.loads((root / 'interfaces.json').read_text())
            self.assertIn('signed_electro_optic_gate', [v['id'] for v in exported['interfaces']])
            self.assertEqual(self.invoke('interfaces', '--out', root / 'interfaces.json')[0], 2)

    def test_report_escapes_user_content(self):
        report = render_report({'status': '<script>x</script>', 'assumptions': ['<img src=x onerror=alert(1)>']}, '<svg onload=x>')
        self.assertNotIn('<script>', report)
        self.assertNotIn('<img', report)
        self.assertNotIn('<svg', report)
        self.assertIn('&lt;script&gt;', report)

    def test_separated_implementation_paths_resolve_from_declaring_file(self):
        from npp.multiphysics.graph import default_implementation
        from npp.multiphysics.interfaces import reference_parameters
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            init_project(root / 'project')
            project = load_project(root / 'project/project.yaml')
            spec = default_implementation(project['model'])
            write_json(root / 'choices/system.json', spec['system'])
            write_json(root / 'choices/interfaces.json', reference_parameters())
            spec['system'] = 'system.json'
            spec['interfaces'] = 'interfaces.json'
            write_json(root / 'choices/implementation.json', spec)
            loaded = load_implementation(root / 'choices/implementation.json', project['model'])
            self.assertEqual(loaded, default_implementation(project['model']))


if __name__ == '__main__':
    unittest.main()
