import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from npp.models import InputValidationError
from npp.technology import (check_operating_point, db_transmission, evaluate_component,
                            get_component, load_technology, photon_count,
                            technology_schema, validate_technology)


class TechnologyTests(unittest.TestCase):
    def setUp(self):
        self.pack = load_technology()

    def component(self, kind):
        return get_component(self.pack, kind)

    def test_reference_is_explicitly_illustrative_and_unreviewed(self):
        self.assertEqual(self.pack['calibration_status'], 'illustrative')
        self.assertEqual(self.pack['review']['status'], 'unreviewed')
        for component in self.pack['components']:
            for group in ('parameters', 'operating_envelope', 'costs'):
                self.assertIn('illustrative_v1', component['evidence'][group])
            for port in component['ports']:
                self.assertEqual(port['max_connections'], 1)

    def test_validation_preserves_input_and_json_yaml_roundtrip(self):
        original = copy.deepcopy(self.pack)
        validate_technology(self.pack)
        self.assertEqual(original, self.pack)
        with tempfile.TemporaryDirectory() as tmp:
            for suffix, content in (('.json', json.dumps(original)), ('.yaml', yaml.safe_dump(original))):
                path = Path(tmp) / ('technology' + suffix)
                path.write_text(content)
                self.assertEqual(load_technology(path), original)
        self.assertEqual(technology_schema()['properties']['schema_version']['const'], '0.3')

    def test_bad_parameters_types_ranges_and_units_rejected(self):
        for key, bad in [('wall_plug_efficiency', 0.0), ('wall_plug_efficiency', 1.01),
                         ('wall_plug_efficiency', True), ('wall_plug_efficiency', '0.2'),
                         ('maximum_output_power_mw', '10 mW'), ('maximum_output_power_mw', math.inf),
                         ('relative_intensity_noise_rms', -1.0)]:
            with self.subTest(key=key, value=bad):
                pack = copy.deepcopy(self.pack)
                pack['components'][0]['parameters'][key] = bad
                with self.assertRaises(InputValidationError):
                    validate_technology(pack)

    def test_inconsistent_ranges_rejected(self):
        mutations = [
            lambda p: p['components'][0]['parameters'].update(minimum_output_power_mw=11.0),
            lambda p: p['components'][0]['operating_envelope']['temperature_c'].update(minimum=40.0),
            lambda p: p['components'][0]['operating_envelope']['temperature_c'].update(minimum=-274.0),
            lambda p: p['components'][1]['operating_envelope'].update(max_input_power_mw=None),
            lambda p: p['components'][3]['parameters'].update(minimum_full_scale_power_mw=2.0),
            lambda p: p['components'][1]['parameters'].update(power_ratio=1.0),
            lambda p: p['components'][1]['parameters'].update(ratio_tunability='free'),
        ]
        for mutate in mutations:
            pack = copy.deepcopy(self.pack)
            mutate(pack)
            with self.assertRaises(InputValidationError):
                validate_technology(pack)

    def test_evidence_reference_and_review_contracts(self):
        mutations = [
            lambda p: p['components'][0]['evidence'].update(parameters=['missing']),
            lambda p: p['components'][0]['evidence'].update(parameters=['nist_si']),
            lambda p: p['components'][0]['evidence'].update(costs=[]),
            lambda p: p['components'][0]['evidence'].update(model=['nist_si', 'nist_si']),
            lambda p: p['sources'].append(copy.deepcopy(p['sources'][0])),
            lambda p: p['review'].update(status='reviewed'),
            lambda p: p['review'].update(status='reviewed', reviewer='Researcher', report_source_id='nist_si'),
        ]
        for mutate in mutations:
            pack = copy.deepcopy(self.pack)
            mutate(pack)
            with self.assertRaises(InputValidationError) as caught:
                validate_technology(pack)
            self.assertTrue(caught.exception.diagnostics[0]['path'])

    def test_named_port_encoding_and_cardinality_contracts(self):
        mutations = [
            lambda p: p['components'][1]['ports'].pop(),
            lambda p: p['components'][1]['ports'][0].update(quantity='normalized_sample'),
            lambda p: p['components'][1]['ports'][0].update(name='wrong'),
            lambda p: p['components'][3]['ports'][1].update(max_connections=2),
            lambda p: p['components'][3]['ports'][1].update(max_connections=True),
            lambda p: p['encoding'].update(logical_minimum=-1.0),
            lambda p: p['encoding'].update(logical_minimum=False),
            lambda p: p['components'].append(copy.deepcopy(p['components'][0])),
            lambda p: p.update(free_copy=True),
        ]
        for mutate in mutations:
            pack = copy.deepcopy(self.pack)
            mutate(pack)
            with self.assertRaises(InputValidationError):
                validate_technology(pack)

    def test_loader_rejects_duplicate_keys_and_unsafe_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.yaml'
            for text in ('id: first\nid: second\n', '!!python/object/apply:os.system [echo bad]'):
                path.write_text(text)
                with self.assertRaises(InputValidationError):
                    load_technology(path)

    def test_loss_has_correct_power_not_amplitude_convention(self):
        self.assertAlmostEqual(db_transmission(10.0), 0.1)
        self.assertAlmostEqual(db_transmission(20.0), 0.01)
        splitter = copy.deepcopy(self.component('splitter'))
        splitter['parameters'].update(power_ratio=0.25, insertion_loss_db=10.0)
        result = evaluate_component(splitter, input_power_mw=2.0)
        self.assertAlmostEqual(result['output_powers_mw']['out1'], 0.05)
        self.assertAlmostEqual(result['output_powers_mw']['out2'], 0.15)
        self.assertLess(sum(result['output_powers_mw'].values()), 2.0)

    def test_source_energy_and_waveguide_delay_unit_oracles(self):
        source = evaluate_component(self.component('source'), source_power_mw=2.0, symbol_duration_ns=3.0)
        self.assertAlmostEqual(source['energy_pj'], 30.0)  # 2mW * 3ns / 20% = 30pJ
        waveguide = evaluate_component(self.component('waveguide'), input_power_mw=1.0, length_um=1000.0)
        self.assertAlmostEqual(waveguide['output_powers_mw']['out'], 10 ** (-0.2 / 10))
        self.assertAlmostEqual(waveguide['latency_ns'], 0.013342563807926082)
        self.assertAlmostEqual(waveguide['area_um2'], 500.0)

    def test_photon_receiver_noise_oracle(self):
        # Independent SI calculation: 1mW * 1ns = 1e-12J; 1550nm photon is ~1.28e-19J.
        expected_photons = 1e-12 / (6.62607015e-34 * 299792458 / 1.55e-6)
        self.assertAlmostEqual(photon_count(1.0, 1.0, 1550.0), expected_photons)
        result = evaluate_component(self.component('detector'), input_power_mw=0.1)
        expected_electrons = 0.8 * expected_photons * 0.1
        self.assertAlmostEqual(result['mean_photoelectrons'], expected_electrons)
        self.assertAlmostEqual(result['added_noise_variance'], 1 / expected_electrons + 1e-6)
        self.assertEqual(result['electrical_output_full_scale'], 1.0)

    def test_modulator_requires_carrier_and_has_real_cost(self):
        with self.assertRaises(ValueError):
            evaluate_component(self.component('modulator'))
        result = evaluate_component(self.component('modulator'), input_power_mw=1.0)
        self.assertAlmostEqual(result['output_powers_mw']['out'], 10 ** -0.1)
        self.assertEqual(result['energy_pj'], 0.1)
        self.assertEqual(result['added_noise_variance'], 0.002 ** 2)
        self.assertEqual({p['name'] for p in self.component('modulator')['ports']}, {'carrier', 'sample', 'out'})

    def test_unsupported_operating_points_and_context_rejected(self):
        component = self.component('detector')
        bad = check_operating_point(component, wavelength_nm=1600.0, temperature_c=10.0, symbol_duration_ns=0.1, input_power_mw=2.0)
        self.assertEqual(len(bad), 4)
        self.assertEqual(check_operating_point(component, wavelength_nm=1540.0, temperature_c=20.0, symbol_duration_ns=0.5, input_power_mw=1.0), [])
        for kwargs in ({'input_power_mw': 0.001}, {'input_power_mw': 2.0}, {'input_power_mw': True},
                       {'input_power_mw': 0.1, 'source_power_mw': 1.0}, {'input_power_mw': 0.1, 'length_um': 1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_component(component, **kwargs)
        with self.assertRaises(ValueError):
            evaluate_component(self.component('waveguide'), input_power_mw=1.0, length_um=10001.0)
        with self.assertRaises(ValueError):
            evaluate_component(self.component('source'), source_power_mw=10.01)
        with self.assertRaises(InputValidationError):
            get_component(self.pack, 'missing')

    def test_helpers_reject_nonfinite_and_wrong_type_numbers(self):
        for bad in (-1.0, math.nan, math.inf, True, '1'):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                db_transmission(bad)
        for kwargs in ({'power_mw': -1.0}, {'symbol_duration_ns': 0.0}, {'wavelength_nm': 0.0}):
            args = dict(power_mw=1.0, symbol_duration_ns=1.0, wavelength_nm=1550.0)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                photon_count(**args)

    def test_extreme_derived_numbers_raise_user_readable_errors(self):
        detector = copy.deepcopy(self.component('detector'))
        detector['parameters']['quantum_efficiency'] = 5e-324
        detector['parameters']['minimum_full_scale_power_mw'] = 5e-324
        with self.assertRaises(ValueError):
            evaluate_component(detector, input_power_mw=5e-324)
        detector['parameters']['quantum_efficiency'] = 0.8
        detector['parameters']['input_referred_noise_mw_rms'] = 1e308
        with self.assertRaises(ValueError):
            evaluate_component(detector, input_power_mw=0.1)
        with self.assertRaises(ValueError):
            photon_count(1e308, 1e308, 1550.0)


if __name__ == '__main__':
    unittest.main()
