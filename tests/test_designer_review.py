"""Independent Gate 4 checks, written after the implementation was submitted."""
from copy import deepcopy
import itertools
import json
import unittest
from unittest.mock import patch

from npp.designer import check_physical_design, plan_physical, validate_design_spec
from npp.implementation import realize
from npp.models import InputValidationError
from npp.technology import load_technology


def network(uses=4):
    return {"schema_version": "0.1", "name": "Gate four review", "nodes": [
        {"id": "x", "op": "input", "size": 1,
         "input_bounds": {"lower": [0.0], "upper": [1.0]}}],
        "outputs": ["x"] * uses}


def design():
    return {"source_node": "x", "grid": {
        "recipes": ["passive", "regenerate"],
        "source_powers_mw": [0.05, 0.1, 0.2],
        "regeneration_powers_mw": [0.1, 0.2],
        "route_lengths_um": [0.0, 10000.0]}}


class DesignerIndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.technology = load_technology()

    def test_every_objective_subset_matches_batch_pairwise_frontier(self):
        # Independently recompute each circuit, then use an all-pairs batch
        # definition rather than the implementation's incremental frontier.
        axes = ("energy_pj", "latency_ns", "area_um2", "noise_rms_estimate")
        initial = plan_physical(network(), self.technology, design())
        metrics = {}
        for row in initial["candidates"]:
            record = realize(network(), self.technology, row["spec"])
            if record["evaluation"]["feasible"]:
                metrics[row["id"]] = record["evaluation"]["metrics"]
        for length in range(1, len(axes) + 1):
            for selected in itertools.combinations(axes, length):
                with self.subTest(objectives=selected):
                    requested = design()
                    requested["objectives"] = list(selected)
                    result = plan_physical(network(), self.technology, requested)
                    expected = set()
                    for cid, candidate in metrics.items():
                        dominated = any(
                            all(other[axis] <= candidate[axis] for axis in selected)
                            and any(other[axis] < candidate[axis] for axis in selected)
                            for other_id, other in metrics.items() if other_id != cid)
                        if not dominated:
                            expected.add(cid)
                    self.assertEqual({p["id"] for p in result["plans"]}, expected)
                    self.assertEqual({r["id"] for r in result["candidates"] if r["pareto"]}, expected)

    def test_equal_selected_metrics_retain_distinct_route_implementations(self):
        requested = design()
        requested["grid"].update(recipes=["passive"], source_powers_mw=[0.1])
        requested["objectives"] = ["energy_pj"]
        result = plan_physical(network(), self.technology, requested)
        self.assertEqual(result["search"]["pareto_count"], 2)
        self.assertNotEqual(result["plans"][0]["realization"]["graph"], result["plans"][1]["realization"]["graph"])

    def test_conflicting_global_family_locks_explain_complete_empty_grid(self):
        alternative = deepcopy(next(c for c in self.technology["components"] if c["id"] == "detector"))
        alternative["id"] = "other_detector"
        self.technology["components"].append(alternative)
        requested = design()
        requested["grid"]["components"] = {"detector": ["detector", "other_detector"]}
        requested["locks"] = {"instances": {"lane0.use0.receiver": "detector", "lane0.use1.receiver": "other_detector"}}
        requested["max_evaluations"] = 1
        with patch("npp.designer.realize") as evaluate:
            result = plan_physical(network(), self.technology, requested)
        evaluate.assert_not_called()
        self.assertTrue(result["search"]["complete"])
        self.assertEqual(result["search"]["evaluated_count"], 0)
        self.assertEqual(result["search"]["excluded_count"], result["search"]["submitted_grid_size"])
        self.assertEqual(result["single_constraint_relaxations"], [])
        for row in result["candidates"]:
            self.assertEqual(row["status"], "excluded")
            self.assertEqual(len(row["reasons"]), 1)
            self.assertEqual(row["reasons"][0]["code"], "component_lock_mismatch")

    def test_operating_change_disables_like_for_like_comparison(self):
        baseline = realize(network(), self.technology,
                           {"source_node": "x", "source_power_mw": 0.1, "lengths_um": 0.0})
        requested = design()
        requested["grid"].update(recipes=["passive"], source_powers_mw=[0.1], route_lengths_um=[0.0])
        requested["operating"] = {"symbol_duration_ns": "2 ns"}
        requested["locks"] = {"recipe": "baseline", "source_power_mw": "baseline"}
        result = plan_physical(network(), self.technology, requested, baseline=baseline)
        comparison = result["candidates"][0]["baseline_comparison"]
        self.assertTrue(comparison["operating_point_changed"])
        self.assertFalse(comparison["technology_assumptions_changed"])
        self.assertFalse(comparison["comparable_metrics"])
        self.assertAlmostEqual(comparison["metric_deltas"]["energy_pj"], 0.5)
        self.assertAlmostEqual(comparison["metric_deltas"]["latency_ns"], 1.0)
        self.assertEqual(comparison["changed_instances"], [])

    def test_exact_budget_boundaries_pass_and_two_failures_offer_no_single_fix(self):
        requested = design()
        requested["grid"].update(recipes=["passive"], source_powers_mw=[0.1], route_lengths_um=[0.0])
        before = plan_physical(network(), self.technology, requested)
        row = before["candidates"][0]
        requested["constraints"] = {"max_" + axis: value for axis, value in row["metrics"].items()
                                     if axis in ("energy_pj", "latency_ns", "area_um2", "noise_rms_estimate")}
        requested["constraints"]["min_receiver_margin_mw"] = row["worst_receiver"]["sensitivity_margin_mw"]
        on_limits = plan_physical(network(), self.technology, requested)
        self.assertEqual(on_limits["candidates"][0]["status"], "feasible")
        requested["constraints"]["max_energy_pj"] /= 2
        requested["constraints"]["max_area_um2"] /= 2
        failed = plan_physical(network(), self.technology, requested)
        self.assertEqual(failed["candidates"][0]["status"], "constraint_failed")
        self.assertEqual(len(failed["candidates"][0]["reasons"]), 2)
        self.assertEqual(failed["single_constraint_relaxations"], [])

    def test_exclusions_after_evaluation_cap_are_still_classified(self):
        requested = design()
        requested["max_evaluations"] = 1
        requested["forbidden_recipes"] = ["regenerate"]
        result = plan_physical(network(), self.technology, requested)
        self.assertEqual(result["search"]["evaluated_count"], 1)
        self.assertEqual(result["search"]["not_evaluated_count"], 5)
        self.assertEqual(result["search"]["excluded_count"], 12)
        for row in result["candidates"]:
            if row["spec"]["recipe"] == "regenerate":
                self.assertEqual(row["status"], "excluded")
                self.assertEqual(row["reasons"][0]["code"], "forbidden_recipe")

    def test_equivalent_decimal_unit_locks_and_deduplication_are_exact(self):
        requested = design()
        requested["grid"].update(recipes=["passive"], source_powers_mw=["0.0001 W"], route_lengths_um=["0 m"])
        requested["locks"] = {"source_power_mw": 0.1}
        normalized = validate_design_spec(requested)
        self.assertEqual(normalized["grid"]["source_powers_mw"], [0.1])
        result = plan_physical(network(), self.technology, requested)
        self.assertEqual(result["search"]["feasible_count"], 1)
        requested["grid"]["source_powers_mw"].append("100 uW")
        with self.assertRaisesRegex(InputValidationError, "unique after unit normalization"):
            validate_design_spec(requested)

    def test_generated_record_limit_is_checked_beyond_input_snapshot_budget(self):
        requested = {"source_node": "x", "grid": {"recipes": ["passive"], "source_powers_mw": [2.0], "route_lengths_um": [0.0]}, "max_evaluations": 1}
        large = plan_physical(network(32), self.technology, requested)
        serialized_size = len(json.dumps(large, allow_nan=False))
        source_size = len(json.dumps({"network": large["inputs"]["network"], "technology": large["inputs"]["technology"]}, allow_nan=False))
        limit = source_size * 6 + 2
        self.assertLess(limit, serialized_size)
        with patch("npp.designer.MAX_RECORD_BYTES", limit):
            with self.assertRaisesRegex(InputValidationError, "generated portable design"):
                plan_physical(network(32), self.technology, requested)
            replay = check_physical_design(large)
            self.assertFalse(replay["valid"])
            self.assertEqual(replay["diagnostics"][0]["code"], "design_record_size_limit")

    def test_nested_explanations_locks_and_retained_plan_claims_are_replayed(self):
        requested = design()
        requested["constraints"] = {"max_energy_pj": 1.0}
        original = plan_physical(network(), self.technology, requested)
        mutations = [
            lambda r: r["resolved_locks"].update(source_power_mw=0.2),
            lambda r: r["candidates"][0]["cost_drivers"][0].update(energy_pj=0.0),
            lambda r: r["candidates"][0]["worst_receiver"].update(use_index=99),
            lambda r: r["single_constraint_relaxations"][0]["candidate_ids"].append("candidate-9999"),
            lambda r: r["plans"][0]["realization"]["evaluation"]["metrics"].update(energy_pj=0.0),
        ]
        self.assertTrue(check_physical_design(original)["valid"])
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                forged = deepcopy(original)
                mutate(forged)
                self.assertFalse(check_physical_design(forged)["valid"])


if __name__ == "__main__":
    unittest.main()
