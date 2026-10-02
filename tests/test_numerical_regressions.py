from copy import deepcopy
import unittest

from npp.checker import check_record
from npp.models import validate_inputs
from npp.physics import evaluate, simulate
from npp.planner import plan
from tests.fixtures import component, library, network, node, request, search_fixture


def scalar_source(value=0.0, noise_std=1e-13, max_abs_value=1.0, budgets=None):
    net = network([node("x", "input", input_bounds={"lower": [value], "upper": [value]})], ["x"])
    lib = library([component("encoder", ["input"], noise_std=noise_std, max_abs_value=max_abs_value)])
    net, lib, req = validate_inputs(net, lib, request(budgets=budgets or {}))
    return net, lib, req, {"components": {"x": "encoder"}, "fanout_rules": {}}


class NumericalRegressionTests(unittest.TestCase):
    def test_tiny_positive_error_violates_exact_zero_budget(self):
        net, lib, req, decision = scalar_source(budgets={"error_rms_bound": 0.0})
        result = evaluate(net, lib, req, decision)
        self.assertFalse(result["feasible"])
        self.assertGreater(result["metrics"]["error_rms_bound"], 0.0)
        self.assertIn("budget_exceeded", {d["code"] for d in result["diagnostics"]})

    def test_tiny_nominal_signal_exceeds_still_smaller_component_range(self):
        net, lib, req, decision = scalar_source(value=1e-13, max_abs_value=1e-20)
        result = evaluate(net, lib, req, decision)
        self.assertFalse(result["feasible"])
        self.assertIn("nominal_range_exceeded", {d["code"] for d in result["diagnostics"]})

    def test_checker_does_not_round_tiny_nonzero_error_to_zero(self):
        net, lib, req, _ = scalar_source()
        record = deepcopy(plan(net, lib, req)["plans"][0])
        self.assertGreater(record["evaluation"]["metrics"]["error_rms_bound"], 0.0)
        record["evaluation"]["metrics"]["error_rms_bound"] = 0.0
        result = check_record(net, lib, req, record)
        self.assertFalse(result["valid"])
        self.assertTrue(any("metrics.error_rms_bound" in d["path"] for d in result["diagnostics"]))

    def test_nondictionary_decision_returns_diagnostic(self):
        net, lib, req, _ = scalar_source()
        for decision in (None, [], "not a decision", 42):
            with self.subTest(decision=decision):
                result = evaluate(net, lib, req, decision)
                self.assertFalse(result["feasible"])
                self.assertTrue(result["diagnostics"])

    def test_checker_huge_integer_tamper_fails_without_crashing(self):
        net, lib, req = validate_inputs(*search_fixture())
        record = deepcopy(plan(net, lib, req)["plans"][0])
        record["evaluation"]["metrics"]["energy_pj"] = 10**1000
        result = check_record(net, lib, req, record)
        self.assertFalse(result["valid"])
        self.assertTrue(result["diagnostics"])

    def test_mc_cancellation_is_a_mismatch_not_zero_z_score(self):
        net, lib, req, decision = scalar_source(value=1e20, noise_std=1.0, max_abs_value=1e21)
        result = simulate(net, lib, req, decision, samples=100, seed=13)
        # IEEE float64 cannot retain order-one perturbations at 1e20.
        # Forward sampling therefore loses the source noise; the independent
        # analytical model still expects unit variance and must flag mismatch.
        self.assertEqual(result["mse_empirical"], 0.0)
        self.assertEqual(result["affine_expected_mse"], 1.0)
        self.assertFalse(result["consistent_with_bound"])
        self.assertIsNone(result["mse_difference_standard_errors"])


if __name__ == "__main__":
    unittest.main()
