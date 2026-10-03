"""Independent rewrite oracles and digital primitive accounting controls."""
import copy
import json
import math
import random
import unittest

import numpy as np

from npp.models import InputValidationError
from npp.multiphysics.catalog import component_for, parameters_from_component, reference_pack
from npp.multiphysics.backends.digital import (
    evaluate_elementwise_cost, evaluate_linear_cost, linear_weight_version,
)
from npp.multiphysics.rewrites import normalize_partition, transform_ffn
from npp.multiphysics.semantic import FFNWorkload, NumericPolicy, SemanticError, evaluate_ffn
from tests.test_mp_semantic import decimal_oracle, fixture


def tile(rows, columns, name="tile", family="digital"):
    return dict(id=name, rows=rows, columns=columns, family=family, component="unit", region="reference")


class RewriteTests(unittest.TestCase):
    def test_input_output_and_two_dimensional_partition_scalar_oracle(self):
        rng = random.Random(818)
        for n, m in ((2, 2), (3, 4), (5, 3), (7, 6)):
            w = [[rng.uniform(-2, 2) for _ in range(n)] for _ in range(m)]
            x, b = [rng.uniform(-1, 1) for _ in range(n)], [rng.uniform(2, 4) for _ in range(m)]
            row_groups = [list(range(0, m, 2)), list(range(1, m, 2))]
            col_groups = [list(range(0, n, 2)), list(range(1, n, 2))]
            for mode in ("input", "output", "two_dimensional"):
                rows = [list(range(m))] if mode == "input" else row_groups
                cols = [list(range(n))] if mode == "output" else col_groups
                tiles = [tile(list(reversed(r)), list(reversed(c)), f"t{i}_{j}", ["digital", "analog", "photonic"][(i+j)%3])
                         for i, r in enumerate(rows) for j, c in enumerate(cols)]
                normalized = normalize_partition((m, n), list(reversed(tiles)))
                partial = {(t["id"], r): sum(w[r][c] * x[c] for c in t["columns"])
                           for t in normalized["tiles"] for r in t["rows"]}
                actual = [math.fsum(partial[(name, r["row"])] for name in r["tiles"]) + b[r["row"]]
                          for r in normalized["output_reductions"]]
                expected = [math.fsum(a * value for a, value in zip(row, x)) + offset for row, offset in zip(w, b)]
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
                self.assertEqual(normalized["merge"]["row_order"], list(range(m)))
                self.assertEqual(normalized["merge"]["bias_application"], "once_after_reduction")
                self.assertTrue(set(t["family"] for t in normalized["tiles"]) <= {"digital", "analog_electrical", "photonic"})
                self.assertEqual(normalize_partition(normalized["shape"], normalized["tiles"]), normalized)

    def test_illegal_partition_cannot_hide_overlap_residual_or_bias(self):
        candidates = [
            [tile([0], [0])],
            [tile([0, 1], [0, 1], "a"), tile([0], [0], "b")],
            [tile([0, 0, 1], [0, 1])],
            [tile([0, 1], [0, 2])],
            [tile([False, 1], [0, 1])],
            [{**tile([0, 1], [0, 1]), "bias": [1, 2]}],
            [tile([0, 1], [0], "same"), tile([0, 1], [1], "same")],
            [tile([0, 1], [0, 1], family="quantum")],
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(SemanticError):
                normalize_partition((2, 2), candidate)
        # Labeling an omitted residual approximate does not make this exact
        # partition rule a reviewed low-rank/truncation transformation.
        with self.assertRaises(SemanticError):
            normalize_partition((2, 2), [tile([0], [0])], semantic_track="A", quality_budget=1)
        for kwargs in ({"semantic_track": "R"}, {"semantic_track": "A"}, {"quality_budget": True}):
            with self.assertRaises(SemanticError):
                normalize_partition((2, 2), [tile([0, 1], [0, 1])], **kwargs)

    def test_signed_permutations_with_bias_match_independent_decimal_oracle(self):
        model = fixture()
        rng = random.Random(829)
        for _ in range(20):
            permutation = rng.sample(range(3), 3)
            signs = [rng.choice([-1, 1]) for _ in range(3)]
            transformed = transform_ffn(model, permutation, signs)
            self.assertEqual(transformed["model"]["b_down"], model["b_down"])
            for x in ([0, 0], [1, -2], [-5, 3], [0.003, 0.125]):
                np.testing.assert_allclose(decimal_oracle(model, x), decimal_oracle(transformed["model"], x), rtol=1e-14, atol=1e-14)
            self.assertEqual(transformed["model"]["W_gate"], [model["W_gate"][i] for i in permutation])
            self.assertEqual(transformed["proof"]["source_model_identity"], FFNWorkload.from_dict(model).model_identity)
            json.dumps(transformed, allow_nan=False)

    def test_transform_guards_and_sign_through_silu_counterexample(self):
        model = fixture()
        for permutation, signs in (([0, 0, 2], [1, 1, 1]), ([0, 1], [1, 1]), ([True, 0, 2], [1, 1, 1]),
                                   ([0, 1, 2], [True, 1, 1]), ([0, 1, 2], [0, 1, 1]), ([0, 1, 2], [-2, 1, 1])):
            with self.assertRaises(SemanticError):
                transform_ffn(model, permutation, signs)
        valid = transform_ffn(model, [2, 0, 1], [-1, 1, -1])["model"]
        invalid = copy.deepcopy(valid)
        invalid["W_gate"][0] = [-v for v in invalid["W_gate"][0]]
        invalid["b_gate"][0] *= -1
        self.assertGreater(max(abs(a-b) for a, b in zip(decimal_oracle(model, [1, -2]), decimal_oracle(invalid, [1, -2]))), 0.01)


class DigitalPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.parameters = parameters_from_component(component_for(reference_pack(), "digital"))

    def test_linear_has_its_own_work_and_closed_form_cost(self):
        p = self.parameters
        output = evaluate_linear_cost([[1, 2], [-3, 4]], [[1, 2], [3, 4]], [5, -6], p)
        self.assertEqual(output["actual_output"], [[10, -1], [16, 1]])
        self.assertEqual(output["operations"]["matrix_mac"], 8)
        self.assertEqual(output["operations"]["bias_add"], 4)
        self.assertEqual(output["resources"]["sram_bytes"], 112)
        self.assertEqual(output["resources"]["sram_read_bytes"], 80)
        self.assertEqual(output["resources"]["sram_write_bytes"], 32)
        self.assertEqual(output["resources"]["dram_bytes"], 48)
        self.assertEqual(output["resources"]["link_bytes"], 160)
        duration = 8/p['macs_per_ns'] + 4/p['scalar_ops_per_ns'] + 112/p['sram_bytes_per_ns'] + 48/p['dram_bytes_per_ns'] + 160/p['link_bytes_per_ns'] + 48/p['program_bytes_per_ns'] + p['launch_ns']
        self.assertAlmostEqual(output["local_duration_ns"], duration)
        energy = (8*p['mac_energy_pj']['value'] + 4*p['add_energy_pj']['value'] + 80*p['sram_read_energy_pj_per_byte']['value'] + 32*p['sram_write_energy_pj_per_byte']['value'] + 48*p['dram_read_energy_pj_per_byte']['value'] + 160*p['link_energy_pj_per_byte']['value'] + 48*p['program_energy_pj_per_byte']['value'] + p['launch_energy_pj']['value'] + duration*p['static_power_mw']['value'])
        self.assertAlmostEqual(output["ledger"]["energy_pj"]["value"], energy)
        self.assertEqual(sum(phase['duration_ns'] for phase in output['phases']), output['local_duration_ns'])

    def test_programmed_weights_require_matching_version_and_still_pay_reads(self):
        w, x, b = [[1, 2]], [3, 4], [1]
        cold = evaluate_linear_cost(w, x, b, self.parameters)
        version = cold['state']['weight_version']
        self.assertEqual(version, linear_weight_version(w, b))
        context = {'weight_state': 'programmed', 'weight_version': version}
        warm = evaluate_linear_cost(w, x, b, self.parameters, context)
        self.assertEqual(cold['actual_output'], warm['actual_output'])
        self.assertEqual(warm['resources']['program_bytes'], 0)
        self.assertEqual(warm['resources']['dram_bytes'], 0)
        self.assertEqual(warm['resources']['sram_read_bytes'], cold['resources']['sram_read_bytes'])
        self.assertGreater(warm['resources']['link_bytes'], 0)
        self.assertLess(warm['ledger']['energy_pj']['value'], cold['ledger']['energy_pj']['value'])
        for changed in ({'weight_state': 'programmed'}, {**context, 'weight_version': 'bad'}, {**context, 'numeric_policy': NumericPolicy(format='float32').to_dict()}):
            with self.assertRaises(InputValidationError):
                evaluate_linear_cost(w, x, b, self.parameters, changed)
        with self.assertRaises(InputValidationError):
            evaluate_linear_cost([[1, 3]], x, b, self.parameters, context)
        repeated = evaluate_linear_cost(w, x, b, self.parameters, {**context, 'reuse_count': 3, 'weight_residency': 'dram'})
        self.assertEqual(repeated['resources']['dram_bytes'], 3 * cold['resources']['weight_bytes'])
        self.assertEqual(repeated['ledger']['area_um2'], cold['ledger']['area_um2'])

    def test_elementwise_gather_concat_and_bias_have_explicit_materialization(self):
        p = self.parameters
        cases = [('add', [[1, 2], [3, 4]], None, [4, 6]),
                 ('multiply', [[1, 2], [3, 4]], None, [3, 8]),
                 ('concat', [[2, 3], [1]], None, [2, 3, 1]),
                 ('gather', [[1, 2, 3]], {'indices': [2, 0, 0]}, [3, 1, 1]),
                 ('add_bias', [[[1, 2], [3, 4]], [5, 6]], None, [[6, 8], [8, 10]])]
        for kind, inputs, context, expected in cases:
            result = evaluate_elementwise_cost(kind, inputs, p, context)
            self.assertEqual(result['actual_output'], expected)
            self.assertGreater(result['resources']['sram_read_bytes'], 0)
            self.assertGreater(result['resources']['sram_write_bytes'], 0)
            self.assertGreater(result['resources']['link_bytes'], 0)
            self.assertGreater(result['local_duration_ns'], 0)
            self.assertEqual(result['resources']['program_bytes'], 0)
            self.assertEqual(result['state']['final_weight_state'], 'not_applicable')
        output = evaluate_elementwise_cost('silu', [[-1000, 0, 1000]], p)
        self.assertEqual(output['actual_output'], [-0.0, 0.0, 1000.0])

    def test_complete_ffn_composition_uses_primitives_and_matches_reference(self):
        m, p, x = fixture(), self.parameters, [0.31415926, -1.23456789]
        context = {'numeric_policy': NumericPolicy(format='float32', accumulation='float32').to_dict()}
        up = evaluate_linear_cost(m['W_up'], x, m['b_up'], p, context)
        gate = evaluate_linear_cost(m['W_gate'], x, m['b_gate'], p, context)
        activation = evaluate_elementwise_cost('silu', [gate['actual_output']], p, context)
        hidden = evaluate_elementwise_cost('multiply', [activation['actual_output'], up['actual_output']], p, context)
        down = evaluate_linear_cost(m['W_down'], hidden['actual_output'], m['b_down'], p, context)
        direct = evaluate_ffn(m, x, NumericPolicy(format='float32', accumulation='float32'))
        self.assertEqual(down['actual_output'], direct['output'])
        self.assertEqual(sum(r['operations'].get('matrix_mac', 0) for r in [up, gate, activation, hidden, down]), direct['operations']['matrix_mac'])

    def test_primitive_resource_unknown_and_malformed_controls(self):
        p = copy.deepcopy(self.parameters)
        p['sram_capacity_bytes'] = 1
        with self.assertRaises(InputValidationError):
            evaluate_linear_cost([[1]], [1], [0], p)
        p = copy.deepcopy(self.parameters)
        p['mac_energy_pj'] = {'state': 'unavailable', 'unit': 'pJ/MAC', 'reason': 'unknown'}
        self.assertEqual(evaluate_linear_cost([[1]], [1], [0], p)['status'], 'conditional')
        for kind, inputs, context in [('add', [[1], [1, 2]], None), ('silu', [[True]], None),
                                     ('gather', [[1, 2]], {'indices': [False]}), ('gather', [[1]], None),
                                     ('concat', [[[1]], [2]], None), ('add_bias', [[1, 2], [1]], None),
                                     ('silu', [[1]], {'weight_state': 'programmed', 'weight_version': 'fake'})]:
            with self.subTest(kind=kind), self.assertRaises((InputValidationError, SemanticError)):
                evaluate_elementwise_cost(kind, inputs, self.parameters, context)


if __name__ == '__main__':
    unittest.main()
