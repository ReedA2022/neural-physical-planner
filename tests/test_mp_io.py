"""M1 user-facing workflow and portable, checked evaluation boundaries."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from npp.cli import main
from npp.multiphysics.cli import check_evaluation, evaluate_project
from npp.multiphysics.contracts import canonical_hash
from npp.multiphysics.io import init_project, load_project, quantity, write_json


class MultiphysicsIOTests(unittest.TestCase):
    def invoke(self, *args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(['mp', *map(str, args)])
        return code, json.loads(stream.getvalue())

    def test_starter_portable_normalization_without_source_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'project with spaces été'
            init_project(root)
            original = load_project(root / 'project.yaml')
            code, normalized = self.invoke('normalize', '--project', root / 'project.yaml', '--out-dir', Path(folder) / 'portable')
            self.assertEqual(code, 0)
            (root / 'weights.npz').unlink()
            portable = load_project(normalized['project'])
            self.assertEqual(original, portable)
            record = evaluate_project(portable)
            self.assertTrue(check_evaluation(record)['valid'])
            self.assertFalse(check_evaluation(record)['independent_validation'])

    def test_actual_cli_evaluate_replay_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(self.invoke('init', root / 'project')[0], 0)
            args = ('evaluate', '--project', root / 'project/project.yaml', '--out-dir', root / 'run')
            code, result = self.invoke(*args)
            self.assertEqual(code, 0)
            self.assertTrue(self.invoke('check', '--record', result['record'])[1]['valid'])
            saved = Path(result['record']).read_bytes()
            self.assertEqual(self.invoke(*args)[0], 2)
            self.assertEqual(Path(result['record']).read_bytes(), saved)

    def test_recomputed_hash_does_not_bypass_input_or_evaluation_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            init_project(folder)
            record = evaluate_project(load_project(Path(folder) / 'project.yaml'))
            for kind in ('cost', 'constraint', 'file_reference'):
                bad = deepcopy(record)
                if kind == 'cost':
                    bad['evaluation']['ledger']['energy_pj']['value'] = 0
                elif kind == 'constraint':
                    bad['inputs']['constraints']['max_memory_bytes'] = -1
                else:
                    bad['inputs']['model'] = '/does/not/exist.json'
                bad['inputs']['input_hash'] = canonical_hash({k: v for k, v in bad['inputs'].items() if k != 'input_hash'})
                bad['record_hash'] = canonical_hash({k: v for k, v in bad.items() if k != 'record_hash'})
                self.assertFalse(check_evaluation(bad)['valid'])

    def test_budget_violation_is_preserved_as_inspectable_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            init_project(root / 'project')
            project = load_project(root / 'project/project.yaml')
            project.pop('input_hash')
            project['constraints']['max_energy_pj'] = 0
            write_json(root / 'tight.json', project)
            code, result = self.invoke('evaluate', '--project', root / 'tight.json', '--out-dir', root / 'run')
            self.assertEqual(code, 3)
            record = json.loads(Path(result['record']).read_text())
            self.assertEqual(record['evaluation']['status'], 'resource_infeasible')
            self.assertTrue(check_evaluation(record)['valid'])

    def test_equivalent_quantities_and_extreme_invalid_units(self):
        self.assertEqual(quantity('9 fJ', 'pJ', 'test'), quantity('0.009 pJ', 'pJ', 'test'))
        for value in ('1e-999 ns', '1e999999999 ns', '-1e-999 ns', '1 V', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantity(value, 'ns', 'test')

    def test_no_implicit_missing_resources_or_evidence_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            init_project(folder)
            project = load_project(Path(folder) / 'project.yaml')
            project['constraints']['resources'] = {'analog': 1}
            self.assertEqual(evaluate_project(project)['evaluation']['status'], 'resource_infeasible')
            project['workload']['evidence_policy'] = ['measured_instance']
            from npp.models import InputValidationError
            with self.assertRaises(InputValidationError):
                evaluate_project(project)


if __name__ == '__main__':
    unittest.main()
