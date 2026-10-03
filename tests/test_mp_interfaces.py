import copy
import json
import math
import unittest

import numpy as np

from npp.models import InputValidationError
from npp.multiphysics.contracts import CostLedger, Parameter, PortContract, known
from npp.multiphysics.interfaces import (
    reference_parameters, validate_parameters, validate_field, interface_catalog,
    registered_interfaces, evaluate_interface, encode, readout, signed_gate,
    attenuation_gate, dac, adc, optical_delay, digital_buffer, recompute_alignment,
)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.p = reference_parameters()
        self.ideal = {'mode': 'ideal_diagnostic'}

    def field(self, values):
        return encode(values, self.p, self.ideal)['field']

    def test_catalog_pins_direction_and_parameter_level_evidence(self):
        cat = interface_catalog()
        self.assertEqual(len(cat['interfaces']), 8)
        self.assertEqual(set(registered_interfaces()), {r['id'] for r in cat['interfaces']})
        for row in cat['interfaces']:
            self.assertEqual(row['evaluator_version'], '1.0')
            for port in row['ports']:
                PortContract.model_validate_json(json.dumps(port))
            for parameter in row['parameters']:
                typed = Parameter.model_validate_json(json.dumps(parameter))
                self.assertEqual(typed.evidence.kind, 'hypothetical')
                self.assertEqual(typed.uncertainty.kind, 'epistemic')

    def test_independent_encode_homodyne_signed_scalar_oracle(self):
        values = [[1.25, -.75], [-1.5, .2]]
        e = encode(values, self.p, self.ideal)
        r = readout(e['field'], self.p, self.ideal)
        np.testing.assert_allclose(e['field']['real'], [[v/2 for v in row] for row in values])
        np.testing.assert_allclose(r['values'], values, rtol=1e-14, atol=1e-14)
        source = next(v for v in e['ledger']['entries'] if v['effect_id'] == 'injected_carrier')
        expected = sum((v/2)**2 for row in values for v in row)/.2
        self.assertAlmostEqual(source['energy_pj']['value'], expected)
        lo = next(v for v in r['ledger']['entries'] if v['effect_id'] == 'local_oscillator')
        self.assertEqual(lo['energy_pj']['value'], 20.)

    def test_finite_roundtrip_quantizes_without_clipping(self):
        x = [.1234567, -.8765432]
        result = readout(encode(x, self.p)['field'], self.p)
        self.assertGreater(max(abs(a-b) for a, b in zip(result['values'], x)), 0)
        self.assertLess(max(abs(a-b) for a, b in zip(result['values'], x)), .001)
        with self.assertRaises(InputValidationError):
            encode([2.001], self.p)

    def test_signed_gate_exact_semantic_product_with_negative_phase(self):
        f = self.field([1.2, -.7, 0.])
        g = [-.5, 1.7, -1.]
        r = signed_gate(f, g, self.p, self.ideal)
        output = readout(r['field'], self.p, self.ideal)
        np.testing.assert_allclose(output['values'], [-.6, -1.19, 0.], atol=1e-14)
        self.assertEqual(r['physical']['sign_phase_rad'], [math.pi, 0., math.pi])
        self.assertFalse(r['physical']['hidden_optical_readout'])
        effects = {e['effect_id'] for e in r['ledger']['entries']}
        self.assertFalse(effects & {'balanced_detector', 'output_adc', 'local_oscillator'})
        self.assertEqual(r['field']['decode_scale'], 4.)

    def test_signed_gate_carries_both_quadratures_and_never_gains_power(self):
        f = self.field([1., -.5]); f['imag'] = [.3, -.1]
        r = signed_gate(f, [-2., .4], self.p, self.ideal)['field']
        np.testing.assert_allclose(r['imag'], [-.3, -.02])
        old = np.hypot(f['real'], f['imag'])**2
        new = np.hypot(r['real'], r['imag'])**2
        self.assertTrue(np.all(new <= old))
        for g in ([3., 0.], [.1]):
            with self.assertRaises(InputValidationError):
                signed_gate(f, g, self.p)

    def test_intensity_is_not_signed_field_or_signed_gate(self):
        payload = dict(carrier='optical', regime='classical_analog', encoding='intensity', power_mw=[1., 2.])
        self.assertEqual(attenuation_gate(payload, [.5, 0.])['intensity']['power_mw'], [.5, 0.])
        for function in (readout, validate_field):
            with self.assertRaises(InputValidationError):
                function(payload)
        with self.assertRaises(InputValidationError):
            attenuation_gate(payload, [-.1, .5])
        with self.assertRaises(InputValidationError):
            attenuation_gate(self.field([.2]), [.2])

    def test_local_oscillator_phase_and_reference_are_explicit(self):
        f = self.field([1.]); f['imag'] = [.25]
        p = copy.deepcopy(self.p); p['lo_phase_rad'] = math.pi/2
        r = readout(f, p, self.ideal)
        self.assertAlmostEqual(r['values'][0], .5)
        with self.assertRaises(InputValidationError):
            readout(f, self.p, {'reference_id': 'unrelated.laser'})

    def test_electrical_dac_adc_direction_and_scale(self):
        converted = dac([1.1, -.8], self.p, self.ideal)
        np.testing.assert_allclose(converted['analog']['voltage_v'], [.11, -.08])
        np.testing.assert_allclose(adc(converted['analog'], self.p, self.ideal)['values'], [1.1, -.8])
        with self.assertRaises(InputValidationError):
            adc(self.field([.2]))
        with self.assertRaises(InputValidationError):
            readout(converted['analog'])

    def test_delay_loss_uses_field_not_power_exponent(self):
        f = self.field([1., -.5])
        r = optical_delay(f, 20., self.p)
        transmission = 10**(-1/20)
        np.testing.assert_allclose(r['field']['real'], np.array(f['real'])*transmission)
        self.assertEqual(r['field']['decode_scale'], f['decode_scale'])
        self.assertEqual(r['duration_ns'], 20.)
        self.assertGreater(r['ledger']['energy_pj']['value'], 0.)
        self.assertGreater(r['ledger']['area_um2']['value'], 0.)

    def test_buffer_has_capacity_retention_access_and_hold_cost(self):
        r = digital_buffer([1., 2.], 10., self.p)
        self.assertEqual(r['values'], [1., 2.])
        self.assertEqual(r['alignment']['stored_bits'], 128)
        self.assertEqual(r['duration_ns'], 12.1)
        self.assertTrue({'buffer_write', 'buffer_read', 'buffer_hold'} <= {e['effect_id'] for e in r['ledger']['entries']})
        p = copy.deepcopy(self.p); p['buffer_capacity_bits'] = 32
        with self.assertRaises(InputValidationError):
            digital_buffer([1., 2.], 10., p)
        with self.assertRaises(InputValidationError):
            digital_buffer([1.], 1e7, self.p)

    def test_recompute_charges_every_upstream_repeat_and_elapsed_time(self):
        r = recompute_alignment([1.], 10., known(3., 'pJ').model_dump(), 4.)
        self.assertEqual(r['alignment']['repetitions'], 3)
        self.assertEqual(r['duration_ns'], 12.)
        charge = next(e for e in r['ledger']['entries'] if e['effect_id'] == 'recomputed_upstream')
        self.assertEqual(charge['energy_pj']['value'], 9.)
        with self.assertRaises(InputValidationError):
            recompute_alignment([1.], 10., known(3., 'pJ').model_dump(), 0.)

    def test_buffer_word_precision_is_not_free_binary64_in_binary32_storage(self):
        p = copy.deepcopy(self.p); p['buffer_word_bits'] = 32
        r = digital_buffer([1.123456789], 1., p)
        self.assertEqual(r['values'], [float(np.float32(1.123456789))])
        self.assertEqual(r['alignment']['stored_bits'], 32)
        with self.assertRaises(InputValidationError):
            digital_buffer([1e100], 1., p)

    def test_no_negative_wait_or_free_zero_delay_conversion(self):
        for fn, value in ((optical_delay, self.field([1.])), (digital_buffer, [1.])):
            with self.assertRaises(InputValidationError):
                fn(value, -1.)
            self.assertGreater(fn(value, 0.)['ledger']['energy_pj']['value'], 0.)

    def test_unknown_and_bounded_interface_costs_never_zero(self):
        for quantity in ({'state': 'unavailable', 'unit': 'pJ/sample', 'reason': 'not characterized'},
                         {'state': 'bounded', 'unit': 'pJ/sample', 'lower': .1, 'upper': .3}):
            p = copy.deepcopy(self.p); p['dac_energy_pj'] = quantity
            r = encode([.2], p)
            self.assertEqual(r['status'], 'conditional')
            self.assertFalse(r['ledger']['complete'])
            self.assertNotEqual(r['ledger']['energy_pj']['state'], 'known')

    def test_ledgers_unique_complete_and_all_category_dispositions(self):
        f = self.field([.2, -.5])
        outcomes = [encode([.2]), readout(f), signed_gate(f, [-.2, .3]),
                    optical_delay(f, 2.), digital_buffer([.2], 1.), dac([.2]),
                    adc(dac([.2])['analog'])]
        for r in outcomes:
            json.dumps(r, allow_nan=False)
            ledger = CostLedger.model_validate_json(json.dumps(r['ledger']['contract']))
            self.assertEqual(len(ledger.entries), len({e.contribution_id for e in ledger.entries}))
            self.assertEqual(ledger.aggregate('energy_pj', 'pJ').quantity.value, r['ledger']['energy_pj']['value'])
            self.assertEqual(len(r['ledger']['category_coverage']), 10)

    def test_inputs_parameters_and_context_snapshots_are_owned(self):
        x = [.2, -.4]; c = {'owner_id': 'my.encoder'}; p = copy.deepcopy(self.p)
        r = encode(x, p, c); frozen = copy.deepcopy(r)
        x[0] = 99; p['input_limit'] = 1; c['owner_id'] = 'changed'
        self.assertEqual(r, frozen)
        f = self.field([.5]); normalized = validate_field(f); f['real'][0] = 7.
        self.assertNotEqual(f, normalized)

    def test_strict_fields_shapes_modes_cost_units_and_boolean_guards(self):
        for x in ([True, .2], [], [[.1], [.2, .3]], [float('nan')], [1.]*65, [[[.1]]]):
            with self.assertRaises(InputValidationError):
                encode(x)
        for mutate in (lambda p: p.update(fake=1.), lambda p: p.update(dac_bits=True),
                       lambda p: p.update(wallplug_efficiency=1.1), lambda p: p.update(source_power_mw=0.),
                       lambda p: p.update(static_power_mw={'state': 'known', 'value': 1., 'unit': 'W'})):
            p = copy.deepcopy(self.p); mutate(p)
            with self.assertRaises(InputValidationError):
                validate_parameters(p)
        for context in ([], {'unknown': 1}, {'mode': 'physical_magic'}, {'owner_id': 'invalid owner'}):
            with self.assertRaises(InputValidationError):
                encode([.1], context=context)

    def test_numerical_underflow_is_not_zero_optical_energy(self):
        with self.assertRaises(InputValidationError) as caught:
            encode([1e-250], self.p, self.ideal)
        self.assertEqual(caught.exception.diagnostics[0]['code'], 'numerical_failure')

    def test_evaluator_registry_rejects_unknown_or_wrong_arguments(self):
        self.assertIn('field', evaluate_interface('digital_to_field', [.1]))
        self.assertIn('values', evaluate_interface('field_to_digital', self.field([.2])))
        self.assertIn('field', evaluate_interface('signed_electro_optic_gate', self.field([.2]), gate=[-.5]))
        for args, kw in ((('eval', [.1]), {}), (('digital_to_field', [.1]), {'gate': [.2]}),
                         (('signed_electro_optic_gate', self.field([.2])), {}),
                         (('digital_buffer', [.1]), {})):
            with self.assertRaises(InputValidationError):
                evaluate_interface(*args, **kw)


if __name__ == '__main__':
    unittest.main()
