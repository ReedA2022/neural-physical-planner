"""Adversarial regression gates discovered by the legacy stress campaign."""
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import permutations
import unittest
from unittest.mock import patch

import numpy as np

from npp.models import validate_inputs
from npp.physics import evaluate, simulate
from npp.planner import pareto_filter, plan
from scripts.stress_legacy import (
    _component, _library, _network, _precision_payload, _nonlinear_precision_payload, _request,
    _search_payload, run_campaign,
)


def records(points):
    return [dict(id=name, evaluation=dict(feasible=True, metrics=dict(zip(
                ["energy_pj", "latency_ns", "area_um2"], values))))
            for name, values in points]


class LegacyStressTests(unittest.TestCase):
    def test_seeded_campaign_covers_every_family(self):
        result = run_campaign(seed=6721, cases=84)
        self.assertEqual(result["actual_scenario_count"], 84)
        self.assertEqual(sum(result["family_counts"].values()), 84)
        self.assertEqual(len(result["family_counts"]), 8)
        self.assertEqual(result["failed"], 0, result["failures"])

    def test_positive_source_noise_never_passes_zero_budget(self):
        for index in range(7):
            with self.subTest(index=index):
                data = _precision_payload(index)
                result = evaluate(*(data[k] for k in ("network", "library", "request", "decision")))
                self.assertFalse(result["feasible"])
                if result["metrics"]:
                    self.assertGreater(result["metrics"]["error_rms_bound"], 0)
                else:
                    self.assertIn("numerical_range_exceeded", {d["code"] for d in result["diagnostics"]})

    def test_downstream_noise_underflow_fails_closed(self):
        for weight in [1e-160, 1e-200, 1e-300]:
            with self.subTest(weight=weight):
                net, lib, req = validate_inputs(_network([
                    dict(id="x", op="input", size=1, input_bounds=dict(lower=[0.], upper=[0.])),
                    dict(id="y", op="linear", size=1, inputs=["x"], weights=[[weight]])], ["y"]),
                    _library([_component("source", ["input"], noise_std=1e-100),
                              _component("op", ["linear"])]), _request(budgets={"error_rms_bound": 0.}))
                result = evaluate(net, lib, req, dict(components={"x": "source", "y": "op"}, fanout_rules={}))
                self.assertFalse(result["feasible"])
                self.assertIn("numerical_range_exceeded", {d["code"] for d in result["diagnostics"]})

    def test_zero_noise_remains_exactly_feasible(self):
        data = _precision_payload(0)
        data["library"]["components"][0]["noise_std"] = 0.
        result = evaluate(*(data[k] for k in ("network", "library", "request", "decision")))
        self.assertTrue(result["feasible"])
        self.assertEqual(result["metrics"]["error_rms_bound"], 0.)

    def test_nonlinear_underflow_cannot_certify_zero_error(self):
        for variant in range(3):
            with self.subTest(variant=variant):
                data = _nonlinear_precision_payload(variant)
                result = evaluate(*(data[k] for k in ("network", "library", "request", "decision")))
                self.assertFalse(result["feasible"])
                if variant < 2:
                    self.assertGreater(result["metrics"]["error_rms_bound"], 0.)
                    self.assertEqual(result["metrics"]["error_rms_bound"],
                                     result["analysis"]["per_output_rms_bounds"][0])
                else:
                    self.assertIn("numerical_range_exceeded", {d["code"] for d in result["diagnostics"]})

    def test_simulation_unrepresentable_mse_fails_before_sampling(self):
        for weight in (1e-60, 1e-160, 1e300):
            data = _nonlinear_precision_payload(0)
            data["network"]["nodes"][1]["weights"] = [[weight]]
            with self.subTest(weight=weight), patch("npp.physics.np.random.default_rng") as rng:
                with self.assertRaisesRegex(ValueError, "Modeled MSE"):
                    simulate(*(data[k] for k in ("network", "library", "request", "decision")), samples=2)
                rng.assert_not_called()

    def test_positive_optical_energy_cannot_silently_underflow(self):
        data = _precision_payload(0)
        c = data["library"]["components"][0]
        c.update(noise_std=0., domain="optical", photons=1e-100, wavelength_nm=1e308)
        data["request"]["budgets"] = {"energy_pj": 0.}
        result = evaluate(*(data[k] for k in ("network", "library", "request", "decision")))
        self.assertFalse(result["feasible"])
        self.assertIn("numerical_range_exceeded", {d["code"] for d in result["diagnostics"]})

    def test_nonzero_subnormal_optical_intermediate_cannot_pass_budget(self):
        data = _precision_payload(0)
        data["library"]["components"][0].update(noise_std=0., domain="optical", photons=3.7e-299)
        data["request"]["budgets"] = {"energy_pj": 4e-306}
        result = evaluate(*(data[k] for k in ("network", "library", "request", "decision")))
        self.assertFalse(result["feasible"])
        with localcontext() as context:
            context.prec = 80
            expected = float(Decimal("3.7e-299") * Decimal("6.62607015e-34") *
                             Decimal("299792458") / Decimal("1550e-9") * Decimal("1e12"))
        self.assertAlmostEqual(result["metrics"]["energy_pj"] / expected, 1.0, places=14)

    def test_near_tradeoff_chain_preserved_for_all_orders(self):
        data = records([("a", (1., 3.)), ("b", (1. + 1.5e-9, 1.)), ("c", (1. + .75e-9, 2.))])
        for ordering in permutations(data):
            self.assertEqual({r["id"] for r in pareto_filter(ordering, ["energy_pj", "latency_ns"])},
                             {"a", "b", "c"})

    def test_old_epsilon_cycle_retains_all_genuine_tradeoffs(self):
        e = .75e-9
        data = records([("a", (1., 1. + e, 1. + 2*e)),
                        ("b", (1. + 2*e, 1., 1. + e)),
                        ("c", (1. + e, 1. + 2*e, 1.))])
        for ordering in permutations(data):
            result = pareto_filter(ordering, ["energy_pj", "latency_ns", "area_um2"])
            self.assertEqual({r["id"] for r in result}, {"a", "b", "c"})

    def test_approximate_tie_does_not_discard_dominance_witness(self):
        data = records([("b", (1., 1.)), ("a", (1. + .75e-9, 1. + .75e-9)),
                        ("c", (1. + .5e-9, 2.))])
        result = pareto_filter(data, ["energy_pj", "latency_ns"])
        self.assertEqual({r["id"] for r in result}, {"a", "b"})

    def test_exact_objective_ties_still_have_canonical_representative(self):
        result = pareto_filter(records([("z", (1., 2.)), ("a", (1., 2.))]), ["energy_pj", "latency_ns"])
        self.assertEqual([r["id"] for r in result], ["a"])

    def test_streaming_plan_uses_same_transitive_significance_rule(self):
        net, lib, req = validate_inputs(_network([
            dict(id="x", op="input", size=1, input_bounds=dict(lower=[0.], upper=[0.])),
            dict(id="y", op="identity", size=1, inputs=["x"])], ["y"]),
            _library([_component("source", ["input"]),
                      _component("a", ["identity"], energy_pj=1., latency_ns=3.),
                      _component("b", ["identity"], energy_pj=1.+1.5e-9, latency_ns=1.),
                      _component("c", ["identity"], energy_pj=1.+.75e-9, latency_ns=2.)]),
            _request(objectives=["energy_pj", "latency_ns"]))
        result = plan(net, lib, req)
        self.assertEqual({r["decision"]["components"]["y"] for r in result["plans"]}, {"a", "b", "c"})

    def test_simulation_rejects_bad_seeds_before_random_allocation(self):
        payload = _search_payload(np.random.default_rng(5))
        args = [payload[k] for k in ("network", "library", "request")]
        for seed in (True, np.bool_(True), -1, 1.5, "0"):
            with self.subTest(seed=seed), patch("npp.physics.np.random.default_rng") as rng:
                with self.assertRaisesRegex(ValueError, "seed"):
                    simulate(*args, {}, samples=10, seed=seed)
                rng.assert_not_called()

    def test_simulation_rejects_work_before_evaluation_or_allocation(self):
        payload = _search_payload(np.random.default_rng(5))
        args = [payload[k] for k in ("network", "library", "request")]
        with patch("npp.physics.evaluate") as evaluate_mock, patch("npp.physics.np.random.default_rng") as rng:
            for samples in (1_000_001, 1_000_000):
                with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, "simulation limit|array work"):
                    simulate(*args, {}, samples=samples, seed=0)
            evaluate_mock.assert_not_called()
            rng.assert_not_called()

    def test_simulation_arithmetic_guard_precedes_execution(self):
        # A long skinny batch fits the array budget but exceeds MAC work.
        n = 128
        net, lib, req = validate_inputs(_network([
            dict(id="x", op="input", size=n, input_bounds=dict(lower=[0.]*n, upper=[0.]*n)),
            dict(id="y", op="linear", size=n, inputs=["x"], weights=np.eye(n).tolist())], ["y"]),
            _library([_component("source", ["input"], capacity=n), _component("op", ["linear"], capacity=n)]), _request())
        with patch("npp.physics.evaluate") as evaluator, patch("npp.physics.np.random.default_rng") as rng:
            with self.assertRaisesRegex(ValueError, "arithmetic"):
                simulate(net, lib, req, {}, samples=4000)
            evaluator.assert_not_called()
            rng.assert_not_called()


if __name__ == "__main__":
    unittest.main()
