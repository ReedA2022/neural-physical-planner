from copy import deepcopy
import unittest

from npp.multiphysics.infrastructure import default_system, normalize_system, system_parameters


class InfrastructureTests(unittest.TestCase):
    def test_explicit_units_equivalence_and_unknown_cost(self):
        a = normalize_system({'preset': 'hypothetical_system', 'overrides': {'clock_startup_energy_pj': '9 fJ'}})
        b = normalize_system({'preset': 'hypothetical_system', 'overrides': {'clock_startup_energy_pj': '.009 pJ'}})
        self.assertEqual(a, b)
        unknown = {'state': 'unavailable', 'unit': 'mW', 'reason': 'package background not characterized'}
        normalized = normalize_system({'preset': 'hypothetical_system', 'overrides': {'package_static_power_mw': unknown}})
        self.assertEqual(system_parameters(normalized)['package_static_power_mw']['state'], 'unavailable')

    def test_metadata_cannot_replace_mechanism_or_required_parameters(self):
        for mutation in ('unit', 'missing', 'duplicate', 'unknown', 'unresolved_time'):
            value = default_system()
            if mutation == 'unit':
                value['parameters'][0]['quantity']['unit'] = 'J'
            elif mutation == 'missing':
                value['parameters'].pop()
            elif mutation == 'duplicate':
                value['parameters'][0] = deepcopy(value['parameters'][1])
            elif mutation == 'unknown':
                value['equation'] = 'free energy'
            else:
                parameter = next(p for p in value['parameters'] if p['name'] == 'clock_startup_ns')
                parameter['quantity'] = {'state': 'unavailable', 'unit': 'ns', 'reason': 'unknown time'}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                normalize_system(value)

    def test_range_type_and_extreme_quantity_guards(self):
        for v in (True, -1, '1e-999 ns', '1e999999999 ns', '1 A'):
            with self.subTest(value=v), self.assertRaises(ValueError):
                normalize_system({'preset': 'hypothetical_system', 'overrides': {'clock_startup_ns': v}})
        with self.assertRaises(ValueError):
            normalize_system({'preset': 'hypothetical_system', 'overrides': {'electronic_buffer_capacity_bytes': 1.5}})


if __name__ == '__main__':
    unittest.main()
