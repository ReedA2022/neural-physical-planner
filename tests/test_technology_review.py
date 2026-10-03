"""Independent gate-1 review: physical invariants and malformed public inputs."""
import copy
import math
import unittest

from npp.models import InputValidationError
from npp.technology import (
    check_operating_point, db_transmission, evaluate_component, get_component,
    load_technology, photon_count, validate_technology,
)


class TechnologyReviewTests(unittest.TestCase):
    def setUp(self):
        self.pack = load_technology()

    def component(self, kind):
        return get_component(self.pack, kind)

    def test_passive_division_conserves_power_for_asymmetric_devices(self):
        for ratio in (0.01, 0.23, 0.5, 0.97):
            for loss in (0.0, 0.2, 13.0):
                component = copy.deepcopy(self.component('splitter'))
                component['parameters'].update(power_ratio=ratio, insertion_loss_db=loss)
                for power in (0.0, 0.001, 1.0, 10.0):
                    output = evaluate_component(component, input_power_mw=power)['output_powers_mw']
                    self.assertAlmostEqual(sum(output.values()), power * math.exp(-loss * math.log(10) / 10))
                    self.assertLessEqual(sum(output.values()), power * (1 + 1e-15))
                    self.assertAlmostEqual(output['out1'], ratio * sum(output.values()))

    def test_source_energy_duration_scaling_and_fixed_cost(self):
        component = copy.deepcopy(self.component('source'))
        component['costs']['fixed_energy_pj'] = 7.0
        outputs = [evaluate_component(component, source_power_mw=0.7, symbol_duration_ns=t)
                   for t in (0.5, 1.0, 2.0)]
        self.assertAlmostEqual(outputs[1]['energy_pj'] - 7, 2 * (outputs[0]['energy_pj'] - 7))
        self.assertAlmostEqual(outputs[2]['energy_pj'] - 7, 2 * (outputs[1]['energy_pj'] - 7))
        self.assertTrue(all(output['output_powers_mw']['out'] == 0.7 for output in outputs))

    def test_receiver_noise_separates_shot_and_electronic_scaling(self):
        detector = copy.deepcopy(self.component('detector'))
        detector['parameters']['input_referred_noise_mw_rms'] = 0.0
        short = evaluate_component(detector, input_power_mw=0.1, symbol_duration_ns=0.5)
        long = evaluate_component(detector, input_power_mw=0.1, symbol_duration_ns=2.0)
        stronger = evaluate_component(detector, input_power_mw=0.2, symbol_duration_ns=0.5)
        self.assertAlmostEqual(short['added_noise_variance'], 4 * long['added_noise_variance'])
        self.assertAlmostEqual(short['added_noise_variance'], 2 * stronger['added_noise_variance'])
        detector['parameters']['input_referred_noise_mw_rms'] = 0.0003
        noisy = evaluate_component(detector, input_power_mw=0.1, symbol_duration_ns=0.5)
        self.assertAlmostEqual(noisy['added_noise_variance'] - short['added_noise_variance'], 9e-6)

    def test_waveguide_cascade_composition_and_zero_length(self):
        component = self.component('waveguide')
        first = evaluate_component(component, input_power_mw=0.5, length_um=123.0)
        second = evaluate_component(component, input_power_mw=first['output_powers_mw']['out'], length_um=654.0)
        combined = evaluate_component(component, input_power_mw=0.5, length_um=777.0)
        self.assertAlmostEqual(second['output_powers_mw']['out'], combined['output_powers_mw']['out'])
        self.assertAlmostEqual(first['latency_ns'] + second['latency_ns'], combined['latency_ns'] + component['costs']['latency_ns'])
        zero = evaluate_component(component, input_power_mw=0.5, length_um=0.0)
        self.assertEqual(zero['output_powers_mw']['out'], 0.5)
        self.assertEqual(zero['area_um2'], component['costs']['area_um2'])

    def test_envelope_edges_are_inclusive_but_next_float_outside_is_rejected(self):
        for component in self.pack['components']:
            for parameter in ('wavelength_nm', 'temperature_c', 'symbol_duration_ns'):
                bounds = component['operating_envelope'][parameter]
                values = dict(wavelength_nm=1550.0, temperature_c=25.0, symbol_duration_ns=1.0)
                for edge, direction in (('minimum', -math.inf), ('maximum', math.inf)):
                    values[parameter] = bounds[edge]
                    self.assertEqual(check_operating_point(component, **values), [])
                    values[parameter] = math.nextafter(bounds[edge], direction)
                    self.assertTrue(check_operating_point(component, **values))

    def test_permuting_component_and_evidence_order_preserves_evaluation(self):
        permuted = copy.deepcopy(self.pack)
        permuted['components'].reverse()
        permuted['sources'].reverse()
        validated = validate_technology(permuted)
        for kind, arguments in [('source', {'source_power_mw': 0.7}),
                                ('splitter', {'input_power_mw': 0.7}),
                                ('waveguide', {'input_power_mw': 0.7, 'length_um': 51.0}),
                                ('detector', {'input_power_mw': 0.7}),
                                ('modulator', {'input_power_mw': 0.7})]:
            self.assertEqual(evaluate_component(get_component(validated, kind), **arguments),
                             evaluate_component(self.component(kind), **arguments))

    def test_valid_named_review_is_representable_without_changing_reference(self):
        pack = copy.deepcopy(self.pack)
        pack['sources'].append(dict(id='review_record', kind='review_report', citation='Named external report, supplied by user',
                                    version='1', coverage='Model consistency review', limitations='Not measured calibration'))
        pack['review'].update(status='reviewed', reviewer='Example reviewer', report_source_id='review_record',
                              notes='Test record; not the bundled reference review.')
        self.assertEqual(validate_technology(pack)['review']['status'], 'reviewed')
        self.assertEqual(self.pack['review']['status'], 'unreviewed')

    def test_blank_reviewer_cannot_satisfy_named_review_requirement(self):
        pack = copy.deepcopy(self.pack)
        pack['sources'].append(dict(id='review_record', kind='review_report', citation='Review report',
                                    version='1', coverage='Model review', limitations='Not calibration'))
        pack['review'].update(status='reviewed', reviewer=' \t\n ', report_source_id='review_record')
        with self.assertRaises(InputValidationError):
            validate_technology(pack)

    def test_blank_evidence_is_not_a_traceable_source(self):
        for field in ('citation', 'version', 'coverage', 'limitations'):
            with self.subTest(field=field):
                pack = copy.deepcopy(self.pack)
                pack['sources'][0][field] = ' \t\n '
                with self.assertRaises(InputValidationError):
                    validate_technology(pack)

    def test_large_integer_public_inputs_fail_as_validation_errors(self):
        # A finite Python integer need not fit in the backend's binary64 representation.
        huge = 10 ** 400
        for function, args in ((db_transmission, (huge,)), (photon_count, (huge, 1.0, 1550.0))):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(*args)
        for field in ('wavelength_nm', 'temperature_c', 'symbol_duration_ns', 'input_power_mw'):
            values = dict(wavelength_nm=1550.0, temperature_c=25.0, symbol_duration_ns=1.0, input_power_mw=0.1)
            values[field] = huge
            with self.subTest(field=field):
                self.assertTrue(check_operating_point(self.component('detector'), **values))


if __name__ == '__main__':
    unittest.main()
