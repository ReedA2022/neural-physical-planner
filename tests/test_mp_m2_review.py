"""Independent M2 gate regressions, authored after the implementation freeze."""
import copy
import math
import tempfile
import unittest

from npp.multiphysics.io import init_project, load_project
from npp.multiphysics.graph import default_implementation, evaluate_implementation, validate_graph
from npp.multiphysics.backends import photonic
from npp.multiphysics import interfaces
from npp.multiphysics.catalog import parameters_from_component


def project():
    with tempfile.TemporaryDirectory() as directory:
        init_project(directory)
        result = load_project(directory + '/project.yaml')
    result['constraints']['resources']['photonic'] = 2
    return result


class IndependentM2ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = project()
        cls.digital = evaluate_implementation(cls.project)
        cls.fused = evaluate_implementation(cls.project, default_implementation(cls.project['model'], 'fused'))

    def test_explicit_malformed_implementation_is_not_defaulted(self):
        for invalid in ([], {}, '', False, 0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluate_implementation(self.project, invalid)

    def test_registered_digital_operations_reject_uniform_wrong_representation(self):
        for key, value in [('unit', 'V'), ('encoding', 'unregistered_code'), ('scale', 2.)]:
            graph = copy.deepcopy(self.digital['physical_graph'])
            for node in graph['nodes']:
                for signal in node['ports'].values():
                    signal[key] = value
            # Both wire endpoints agree. The operation's native contract must
            # still reject an undeclared reinterpretation of numeric words.
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_graph(graph)

    def test_field_amplitude_operation_rejects_power_unit(self):
        graph = copy.deepcopy(self.fused['physical_graph'])
        for node in graph['nodes']:
            for signal in node['ports'].values():
                if signal['carrier'] == 'optical':
                    signal['unit'] = 'mW'
        with self.assertRaises(ValueError):
            validate_graph(graph)

    def test_nested_disclosure_owner_is_the_enclosing_composite(self):
        graph = copy.deepcopy(self.fused['physical_graph'])
        parent = next(node for node in graph['nodes'] if 'physical_subgraph' in node)
        parent['physical_subgraph']['nodes'][0]['accounting_parent'] = 'unrelated_owner'
        with self.assertRaises(ValueError):
            validate_graph(graph)

    def test_fused_field_input_honors_selected_component_envelope(self):
        p = copy.deepcopy(self.project)
        component = next(c for c in p['technology']['components'] if c['family'] == 'photonic')
        restricted = copy.deepcopy(component)
        restricted['id'] = 'restricted_down'
        for port in restricted['ports']:
            if port['name'] == 'input':
                port['signal']['value_range'] = {'minimum': 0., 'maximum': 0.}
        p['technology']['components'].append(restricted)
        spec = default_implementation(p['model'], 'fused')
        spec['operators']['up']['tiles'][0]['component'] = component['id']
        spec['operators']['down']['tiles'][0]['component'] = restricted['id']
        with self.assertRaises(ValueError):
            evaluate_implementation(p, spec)

    def test_fixed_analog_padding_includes_inactive_row_loading(self):
        p = copy.deepcopy(self.project)
        p['model'].update(W_up=[[1.], [1.2], [-1.], [1.5]], W_gate=[[1.]] * 4,
                          W_down=[[1., 1., -1., 1.]], b_up=[0.] * 4,
                          b_gate=[0.] * 4, b_down=[0.])
        p['workload']['inputs'] = [[1.2], [-1.3]]
        component = next(c for c in p['technology']['components'] if c['family'] == 'analog_electrical')
        overrides = dict(g_min_s=.001, g_max_s=.002, line_resistance_ohm=1000.,
                         program_bits=24, dac_bits=24, adc_bits=24, output_current_limit_a=.0001,
                         drift_per_ns=0., read_disturbance_relative=0.)
        for parameter in component['parameters']:
            if parameter['name'] in overrides:
                parameter['quantity']['value'] = overrides[parameter['name']]
        coefficients = parameters_from_component(component)
        m = p['model']
        def tile(w, x, b, installed_rows=4):
            # Independent scalar finite-conductance calculation. The active
            # one-row down tile retains all other installed g_min rail loads.
            rows, cols = len(w), len(w[0])
            span = coefficients['g_max_s'] - coefficients['g_min_s']
            steps = 2**coefficients['program_bits'] - 1
            def rail(weight):
                return coefficients['g_min_s'] + round(max(weight, 0.) / coefficients['weight_scale'] * steps) / steps * span
            gp = [[rail(w[r][c] if r < rows and c < cols else 0.) for c in range(4)] for r in range(installed_rows)]
            gn = [[rail(-w[r][c] if r < rows and c < cols else 0.) for c in range(4)] for r in range(installed_rows)]
            def quantize(value, limit, bits):
                levels = 2**(bits-1) - 1
                return round(value / limit * levels) / levels * limit
            voltage = [quantize((x[c] if c < len(x) else 0.) * coefficients['input_scale_v'],
                                coefficients['input_limit'] * coefficients['input_scale_v'], coefficients['dac_bits'])
                       for c in range(4)]
            loaded = [v / (1. + coefficients['line_resistance_ohm'] * math.fsum(gp[r][c] + gn[r][c] for r in range(installed_rows)))
                      for c,v in enumerate(voltage)]
            return [quantize(math.fsum(loaded[c] * (gp[r][c] - gn[r][c]) for c in range(4)),
                             coefficients['output_current_limit_a'], coefficients['adc_bits'])
                    * coefficients['weight_scale'] / (span * coefficients['input_scale_v']) + b[r] for r in range(rows)]
        expected, free_isolation = [], []
        for x in p['workload']['inputs']:
            up = tile(m['W_up'], x, m['b_up'])
            gate = tile(m['W_gate'], x, m['b_gate'])
            hidden = [u*g/(1.+math.exp(-g)) for u,g in zip(up,gate)]
            expected.append(tile(m['W_down'], hidden, m['b_down']))
            free_isolation.append(tile(m['W_down'], hidden, m['b_down'], installed_rows=1))
        self.assertGreater(max(abs(a[0]-b[0]) for a,b in zip(expected,free_isolation)), 1e-5)
        result = evaluate_implementation(p, default_implementation(p['model'], 'analog'))
        for actual, wanted in zip(result['output'], expected):
            self.assertAlmostEqual(actual[0], wanted[0], places=12)
        down = result['backend_evaluations']['run0.down.0']
        self.assertEqual(len(down['physical']['conductance_positive_s']), 4)
        self.assertEqual(len(down['physical']['conductance_positive_s'][0]), 4)

    def test_photonic_phase_quadrature_is_not_abs_value_readout(self):
        parameters = photonic.reference_parameters()
        parameters.update(weight_scale=1., insertion_loss_db=0., phase_bias_rad=math.pi/2)
        result = photonic.evaluate_linear([[.5]], [[.5]], None, parameters)
        # A 90-degree phase shift moves the signal to the orthogonal quadrature;
        # fixed zero-phase homodyne cannot output its absolute field amplitude.
        self.assertLess(abs(result['actual_output'][0][0]), 1e-5)
        self.assertGreater(abs(result['physical']['output_field']['imag'][0][0]), .1)
        self.assertLessEqual(result['physical']['matrix_output_power_mw'], result['physical']['input_power_mw'])

    def test_signed_gate_preserves_passivity_and_pays_control(self):
        parameters = interfaces.reference_parameters()
        encoded = interfaces.encode([.7, -.4], parameters)
        result = interfaces.signed_gate(encoded['field'], [-2., .5], parameters)
        before = sum(x*x+y*y for x,y in zip(encoded['field']['real'], encoded['field']['imag']))
        after = sum(x*x+y*y for x,y in zip(result['field']['real'], result['field']['imag']))
        self.assertLessEqual(after, before)
        self.assertLess(result['field']['real'][0], 0.)
        self.assertFalse(result['physical']['hidden_optical_readout'])
        self.assertTrue(any(e['category'] == 'conversion' and e['energy_pj']['value'] > 0 for e in result['ledger']['entries']))
        self.assertTrue(any(e['category'] == 'control' and e['energy_pj']['value'] > 0 for e in result['ledger']['entries']))


if __name__ == '__main__':
    unittest.main()
