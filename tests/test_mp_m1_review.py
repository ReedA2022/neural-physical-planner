"""Independent M1 gate regressions, authored separately from implementation."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from npp.models import InputValidationError
from npp.multiphysics.catalog import (
    RegisteredBackend, evaluate_model, normalize_pack, reference_pack,
    component_for, parameters_from_component,
)
from npp.multiphysics.backends.analog import evaluate_linear
from npp.multiphysics.backends import error_record
from npp.multiphysics.semantic import NumericalFailure, evaluate_ffn, NumericPolicy
from npp.multiphysics.cli import check_evaluation, evaluate_project
from npp.multiphysics.io import load_project


ROOT = Path(__file__).resolve().parents[1]
MODEL = {"W_up": [[1., 0.], [0., 1.]], "W_gate": [[1., 0.], [0., 1.]],
         "W_down": [[1., -1.]]}


class IndependentM1Review(unittest.TestCase):
    def test_scalar_analog_oracle_includes_quantizers_loading_and_joule_energy(self):
        """No production numerical helper is reused in this scalar reference."""
        parameters = parameters_from_component(component_for(reference_pack(), 'analog'))
        parameters.update(drift_per_ns=0., read_disturbance_relative=0.,
                          programming_noise_relative=0., line_resistance_ohm=37.)
        weights = [[.73, -.61, .02], [-.13, .88, -1.4]]
        inputs = [[.25, -.33, .58], [-.91, 1.03, -.82]]
        bias = [.21, -.12]
        p = parameters
        span = p['g_max_s'] - p['g_min_s']
        steps = 2**p['program_bits'] - 1
        rails = []
        for sign in (1, -1):
            rails.append([[p['g_min_s'] + round(max(sign*v, 0.)/p['weight_scale']*steps)/steps*span
                           for v in row] for row in weights])
        gp, gn = rails
        loading = [sum(gp[j][k] + gn[j][k] for j in range(2)) for k in range(3)]
        expected, read_energy, line_energy = [], 0., 0.
        dac_levels = 2**(p['dac_bits']-1)-1
        adc_levels = 2**(p['adc_bits']-1)-1
        voltage_limit = p['input_limit']*p['input_scale_v']
        for x in inputs:
            voltage = [round(v*p['input_scale_v']/voltage_limit*dac_levels)/dac_levels*voltage_limit for v in x]
            loaded = [v/(1+p['line_resistance_ohm']*g) for v, g in zip(voltage, loading)]
            row = []
            for j in range(2):
                positive = sum(v*g for v, g in zip(loaded, gp[j]))
                negative = sum(v*g for v, g in zip(loaded, gn[j]))
                current = positive-negative
                adc = round(current/p['output_current_limit_a']*adc_levels)/adc_levels*p['output_current_limit_a']
                row.append(adc*p['weight_scale']/(span*p['input_scale_v'])+bias[j])
            expected.append(row)
            read_energy += sum(v*v*g for v, g in zip(loaded, loading))*p['integration_ns']*1e3
            line_energy += sum((v-l)*l*g for v, l, g in zip(voltage, loaded, loading))*p['integration_ns']*1e3
        result = evaluate_linear(weights, inputs, bias, parameters)
        for actual_row, expected_row in zip(result['actual_output'], expected):
            for actual, reference in zip(actual_row, expected_row):
                self.assertTrue(math.isclose(actual, reference, rel_tol=1e-13, abs_tol=1e-13))
        entries = {entry['effect_id']: entry['energy_pj']['value'] for entry in result['ledger']['entries']}
        self.assertTrue(math.isclose(entries['conductance_dissipation'], read_energy, rel_tol=1e-13))
        self.assertTrue(math.isclose(entries['series_line_dissipation'], line_energy, rel_tol=1e-13))
        self.assertGreater(line_energy, 0.)

    def test_enforced_input_and_output_port_ranges(self):
        for port_index in (0, 1):
            with self.subTest(port=port_index):
                pack = reference_pack()
                pack['components'][0]['ports'][port_index]['signal']['value_range'] = {
                    'minimum': 0., 'maximum': .01}
                with self.assertRaises((InputValidationError, ValueError)):
                    evaluate_model(MODEL, {'inputs': [.2, -.4]}, pack)

    def test_unsupported_mechanism_metadata_cannot_silently_change(self):
        changes = {
            'digital_current_port': lambda c: c['ports'][0]['signal'].update(physical_variable='current'),
            'dimension_unit': lambda c: c['operating_envelope'][0].update(unit='GHz'),
            'digital_integration_time': lambda c: c['dynamics'].update(
                integration_ns={'state': 'known', 'unit': 'ns', 'value': 1e9}),
        }
        for name, change in changes.items():
            with self.subTest(mutation=name):
                pack = reference_pack()
                change(pack['components'][0])
                with self.assertRaises((InputValidationError, ValueError)):
                    normalize_pack(pack)

    def test_lifecycle_enforces_component_shape_envelope(self):
        component = reference_pack()['components'][0]
        component['operating_envelope'][0]['maximum'] = 1.
        backend = RegisteredBackend('digital.ffn')
        instance = backend.instantiate(component, '{}')
        self.assertEqual(instance.status, 'model_feasible')
        outcome = backend.evaluate(instance.payload_json, json.dumps({
            'request': {'model': MODEL, 'inputs': [.2, -.4]}}))
        self.assertNotIn(outcome.status, ('model_feasible', 'conditional'))
        self.assertTrue(outcome.diagnostics)

    def test_returned_digital_state_is_reusable_and_rejects_weight_change(self):
        backend = RegisteredBackend('digital.ffn')
        instance = backend.instantiate(reference_pack()['components'][0], '{}')
        request = {'request': {'model': MODEL, 'inputs': [.2, -.4]}}
        first = backend.evaluate(instance.payload_json, json.dumps(request))
        self.assertEqual(first.status, 'model_feasible')
        bound_state = json.dumps(json.loads(first.payload_json)['next_instance'])
        second = backend.evaluate(bound_state, json.dumps(request))
        self.assertEqual(second.status, 'model_feasible', second.diagnostics)
        self.assertEqual(json.loads(first.payload_json)['actual_output'], json.loads(second.payload_json)['actual_output'])
        request = copy.deepcopy(request)
        request['request']['model']['W_up'][0][0] = .5
        rejected = backend.evaluate(bound_state, json.dumps(request))
        self.assertEqual(rejected.status, 'invalid')
        self.assertEqual(rejected.diagnostics[0].code, 'domain_mismatch')

    def test_lifecycle_enforces_integration_and_supply(self):
        for family, component_index, request in (
                ('digital.ffn', 0, {'model': MODEL, 'inputs': [.2, -.4]}),
                ('analog.differential_conductance', 1, {'weights': [[1.]], 'inputs': [.2]})):
            backend = RegisteredBackend(family)
            instance = backend.instantiate(reference_pack()['components'][component_index], '{}')
            for field, value in (('temperature_c', 9999.), ('services', []), ('substrate', 'unmodeled')):
                with self.subTest(family=family, field=field):
                    environment = {'temperature_c': 25., 'region': 'room',
                                   'substrate': 'hypothetical_chiplet', 'services': ['power_supply']}
                    environment[field] = value
                    outcome = backend.evaluate(instance.payload_json, json.dumps({
                        'request': request, 'evaluation_context': {'environment': environment}}))
                    self.assertNotIn(outcome.status, ('model_feasible', 'conditional'))
                    self.assertTrue(outcome.diagnostics)

    def test_declared_absolute_track_a_budget_controls_status(self):
        model = copy.deepcopy(MODEL)
        model.update(semantic_track='A', quality_budget=0.)
        result = evaluate_model(model, {'inputs': [.2, -.4],
                                      'numeric_policy': {'format': 'float32', 'accumulation': 'float32'}})
        self.assertGreater(result['error']['max_abs'], 0.)
        self.assertFalse(result['semantic']['quality_passed'])
        self.assertEqual(result['status'], 'resource_infeasible')

    def test_nonzero_output_error_never_underflows_into_zero_rms(self):
        model = {'W_up': [[1.]], 'W_gate': [[1.]], 'W_down': [[5e-324]]}
        inputs = [[1.], [0.], [0.], [0.]]
        with self.assertRaises(NumericalFailure):
            evaluate_ffn(model, inputs, NumericPolicy(format='float32', accumulation='float32'))
        with self.assertRaises(InputValidationError) as caught:
            error_record([[5e-324], [0.], [0.], [0.]], [[0.], [0.], [0.], [0.]])
        self.assertEqual(caught.exception.diagnostics[0]['code'], 'numerical_failure')

    def test_unimplemented_track_r_execution_is_explicitly_unsupported(self):
        model = copy.deepcopy(MODEL)
        model.update(semantic_track='R', parent_model_identity='parent',
                     adaptation_identity='changed', adaptation_cost='unknown')
        with self.assertRaises(InputValidationError) as caught:
            evaluate_model(model, {'inputs': [.2, -.4]})
        self.assertEqual(caught.exception.diagnostics[0]['code'], 'unavailable_model')
        backend = RegisteredBackend('digital.ffn')
        instance = backend.instantiate(reference_pack()['components'][0], '{}')
        outcome = backend.evaluate(instance.payload_json, json.dumps({
            'request': {'model': model, 'inputs': [.2, -.4]}}))
        self.assertEqual(outcome.status, 'unsupported')

    def test_nonobject_sample_context_is_structured_invalid(self):
        backend = RegisteredBackend('analog.differential_conductance')
        instance = backend.instantiate(reference_pack()['components'][1], '{}')
        for context in ('null', '[]', '1', '"text"'):
            with self.subTest(context=context):
                outcome = backend.sample(instance.payload_json, context)
                self.assertEqual(outcome.status, 'invalid')
                self.assertTrue(outcome.diagnostics)

    def test_malformed_embedded_replay_inputs_are_structured(self):
        record = evaluate_project(load_project(ROOT / 'examples/multiphysics/ffn/project.yaml'))
        for malformed in ([], None, 'path.json', 1):
            with self.subTest(value=malformed):
                tampered = copy.deepcopy(record)
                tampered['inputs'] = malformed
                self.assertFalse(check_evaluation(tampered)['valid'])
        record['inputs'] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            path.write_text(json.dumps(record), encoding='utf8')
            command = subprocess.run([sys.executable, '-m', 'npp', 'mp', 'check', '--record', str(path)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(command.returncode, 2, command.stderr)
            self.assertFalse(json.loads(command.stdout)['valid'])
            self.assertNotIn('Traceback', command.stderr)


if __name__ == '__main__':
    unittest.main()
