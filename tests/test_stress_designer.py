"""Regression and independent-order oracles for the bounded designer stress suite."""
from copy import deepcopy
import itertools
import json
import math
import unittest
from unittest.mock import patch

from npp.designer import plan_physical
from npp.implementation import realize
from npp.technology import load_technology
from scripts.stress_designer import AXES, FAMILIES, _audit_grid, _design, _network, run_campaign


class DesignerStressTests(unittest.TestCase):
    def test_seeded_campaign_covers_every_family_and_supported_limit(self):
        result = run_campaign(seed=314159, cases=48)
        self.assertTrue(result["passed"], json.dumps(result["failures"], indent=2))
        self.assertEqual(result["scenario_count"], 48)
        self.assertEqual(result["family_counts"], dict.fromkeys(FAMILIES, 8))
        self.assertTrue({"feasible", "infeasible", "excluded", "not_evaluated", "constraint_failed"}.issubset(result["candidate_status_counts"]))
        json.dumps(result, allow_nan=False)

    def test_fixed_seed_results_are_reproducible_except_elapsed_time(self):
        first, second = run_campaign(seed=2999, cases=12), run_campaign(seed=2999, cases=12)
        first.pop("elapsed_seconds")
        second.pop("elapsed_seconds")
        self.assertEqual(first, second)

    def test_campaign_records_and_continues_after_an_actual_failure(self):
        with patch("scripts.stress_designer._analytic_case", side_effect=AssertionError("injected incorrect cost")):
            result = run_campaign(seed=11, cases=7)
        self.assertFalse(result["passed"])
        self.assertEqual(result["scenario_count"], 7)
        self.assertEqual([f["scenario_index"] for f in result["failures"]], [0, 6])
        self.assertEqual(result["failures"][0]["reproduction"]["scenario_seed"], 11)
        self.assertEqual(result["failures"][0]["error_type"], "AssertionError")

    def test_independent_oracle_detects_coherent_looking_metric_corruption(self):
        network, technology, design = _network(), load_technology(), _design()
        result = plan_physical(network, technology, design)
        good = next(r for r in result["candidates"] if r["status"] == "feasible")
        good["metrics"]["energy_pj"] *= 1.01
        with self.assertRaisesRegex(AssertionError, "energy_pj"):
            _audit_grid(result, technology, network)

    def test_exact_pareto_order_is_permutation_invariant_at_ties_and_tiny_scales(self):
        network, technology = _network(1, 2), load_technology()
        base = realize(network, technology, {"source_node": "x", "source_power_mw": 0.2, "lengths_um": 0.0})
        tiny = math.nextafter(0.0, 1.0)
        sets = [
            [(1.0, 1.0, 1.0), (math.nextafter(1.0, 2.0), 1.0, 1.0), (1.0, 1.0, 1.0)],
            [(0.0, tiny, tiny), (tiny, 0.0, tiny), (tiny, tiny, 0.0)],
            [(1.0, 1.0 + 0.75e-12, 1.0 + 1.5e-12),
             (1.0 + 1.5e-12, 1.0, 1.0 + 0.75e-12),
             (1.0 + 0.75e-12, 1.0 + 1.5e-12, 1.0)],
        ]
        for vectors in sets:
            expected = {i for i, vector in enumerate(vectors) if not any(
                all(a <= b for a, b in zip(other, vector)) and any(a < b for a, b in zip(other, vector))
                for other in vectors)}
            self.assertTrue(expected)
            for order in itertools.permutations(range(3)):
                design = {"source_node": "x", "grid": {"recipes": ["passive"],
                          "source_powers_mw": [float(i + 1) for i in order], "route_lengths_um": [0.0]},
                          "objectives": list(AXES[:3])}

                def evaluation(_network, _technology, spec):
                    record = deepcopy(base)
                    record["spec"] = spec
                    record["evaluation"]["metrics"].update(dict(zip(AXES[:3], vectors[int(spec["source_power_mw"]) - 1])))
                    return record

                # Deliberately synthetic metrics isolate the search algorithm;
                # independent physical formulas are exercised by the campaign.
                with patch("npp.designer.realize", side_effect=evaluation):
                    result = plan_physical(network, technology, design)
                retained = {int(p["realization"]["spec"]["source_power_mw"]) - 1 for p in result["plans"]}
                self.assertEqual(retained, expected, (vectors, order))
                self.assertEqual(result["search"]["feasible_count"], 3)
                self.assertGreater(result["search"]["pareto_count"], 0)

    def test_stress_budget_itself_is_bounded(self):
        for cases in (0, -1, 2001, True, 1.5):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                run_campaign(cases=cases)


if __name__ == "__main__":
    unittest.main()
