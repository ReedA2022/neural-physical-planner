"""Independent finite-grid and hand-calculated physical-design scenarios.

These tests intentionally calculate reference-device powers, noise, cost and time
without calling the technology evaluator.  The numbers remain illustrative model
fixtures; passing the tests does not establish device calibration.
"""

from copy import deepcopy
import json
import math
import unittest

from npp.designer import check_physical_design, plan_physical, validate_design_spec
from npp.implementation import check_realization, realize
from npp.models import InputValidationError
from npp.technology import load_technology
from tests.fixtures import network, node


def four_destinations():
    return network([
        node("x", "input", input_bounds={"lower": [0.0], "upper": [1.0]})
    ], ["x", "x", "x", "x"])


def search_spec():
    return {
        "schema_version": "0.3", "source_node": "x",
        "grid": {
            "recipes": ["passive", "regenerate"],
            "source_powers_mw": [0.05, 0.1, 0.2],
            "regeneration_powers_mw": [0.1, 0.2],
            "route_lengths_um": [10000.0],
        },
        "objectives": ["energy_pj", "latency_ns", "area_um2", "noise_rms_estimate"],
    }


def choice(row):
    spec = row["spec"]
    return spec["recipe"], spec["source_power_mw"], spec["regeneration_power_mw"]


def detector_variance(power_mw):
    # One 1 ns full-scale symbol at 1550 nm, QE=.8, electronic RMS=.0001 mW.
    energy_j = power_mw * 1e-3 * 1e-9
    photon_energy_j = 6.62607015e-34 * 299792458.0 / (1550e-9)
    return photon_energy_j / (0.8 * energy_j) + (0.0001 / power_mw) ** 2


def oracle_metrics(recipe, source_power_mw, regeneration_power_mw):
    early_power = source_power_mw / 4 * 10 ** (-0.4 / 10)
    guide_transmission = 10 ** (-2 / 10)
    guide_delay_ns = 4 * 10000e-6 / 299792458.0 * 1e9
    if recipe == "passive":
        final_power = early_power * guide_transmission
        return {
            "energy_pj": source_power_mw / 0.2 + 4 * 0.1,
            "area_um2": 1000 + 3 * 20 + 4 * 100 + 4 * 10000 * 0.5,
            "latency_ns": 1 + 0.1 + 0.2 + guide_delay_ns,
            "noise_rms_estimate": math.sqrt(0.001 ** 2 + detector_variance(final_power)),
        }
    final_power = regeneration_power_mw * 10 ** (-1 / 10) * guide_transmission
    return {
        "energy_pj": (source_power_mw + 4 * regeneration_power_mw) / 0.2 + 8 * 0.1 + 4 * 0.1,
        "area_um2": 5 * 1000 + 3 * 20 + 8 * 100 + 4 * 100 + 4 * 10000 * 0.5,
        "latency_ns": 2 + 0.1 + 0.2 + 0.2 + 0.2 + guide_delay_ns,
        "noise_rms_estimate": math.sqrt(2 * 0.001 ** 2 + 0.002 ** 2
                                       + detector_variance(early_power)
                                       + detector_variance(final_power)),
    }


class PhysicalDesignOracleTests(unittest.TestCase):
    def setUp(self):
        self.network = four_destinations()
        self.technology = load_technology()

    def plan(self, design=None, **kwargs):
        return plan_physical(self.network, self.technology,
                             search_spec() if design is None else design, **kwargs)

    def test_full_grid_has_analytic_costs_and_three_specific_pareto_designs(self):
        result = self.plan()
        search = result["search"]
        # Passive branches do not repeat for the two carrier-power options.
        self.assertEqual(search["submitted_grid_size"], 3 + 3 * 2)
        self.assertEqual(search["evaluated_count"], 9)
        self.assertEqual(search["excluded_count"], 0)
        self.assertEqual(search["not_evaluated_count"], 0)
        self.assertEqual(search["feasible_count"], 8)
        self.assertTrue(search["complete"])
        expected_frontier = {
            ("passive", 0.1, None), ("passive", 0.2, None),
            ("regenerate", 0.2, 0.2),
        }
        self.assertEqual({choice(row) for row in result["candidates"] if row["pareto"]}, expected_frontier)
        self.assertEqual(search["pareto_count"], 3)
        self.assertEqual({row["id"] for row in result["candidates"] if row["pareto"]},
                         {plan["id"] for plan in result["plans"]})
        for row in result["candidates"]:
            with self.subTest(choice=choice(row)):
                if choice(row) == ("passive", 0.05, None):
                    self.assertEqual(row["status"], "infeasible")
                    self.assertIsNone(row["metrics"])
                    continue
                self.assertEqual(row["status"], "feasible", row["reasons"])
                for axis, expected in oracle_metrics(*choice(row)).items():
                    self.assertTrue(math.isclose(row["metrics"][axis], expected, rel_tol=1e-12, abs_tol=1e-14),
                                    (axis, row["metrics"][axis], expected))
        for plan in result["plans"]:
            self.assertTrue(check_realization(plan["realization"])["valid"])
        self.assertTrue(check_physical_design(json.loads(json.dumps(result, allow_nan=False)))["valid"])

    def test_low_launch_lock_keeps_regeneration_while_forbidding_it_proves_no_plan(self):
        design = search_spec()
        design["locks"] = {"source_power_mw": 0.05}
        result = self.plan(design)
        self.assertEqual(result["search"]["excluded_count"], 6)
        self.assertEqual(result["search"]["evaluated_count"], 3)
        self.assertEqual(result["search"]["feasible_count"], 2)
        self.assertTrue(result["search"]["complete"])
        early_power = 0.05 / 4 * 10 ** (-0.4 / 10)
        self.assertGreater(early_power, 0.01)
        self.assertLess(early_power * 10 ** (-2 / 10), 0.01)
        for plan in result["plans"]:
            record = plan["realization"]
            self.assertEqual(record["spec"]["source_power_mw"], 0.05)
            self.assertEqual(record["spec"]["recipe"], "regenerate")
            self.assertGreaterEqual(record["evaluation"]["metrics"]["energy_pj"], 3.45)
        design["forbidden_recipes"] = ["regenerate"]
        blocked = self.plan(design)
        self.assertEqual(blocked["search"]["evaluated_count"], 1)
        self.assertEqual(blocked["search"]["excluded_count"], 8)
        self.assertEqual(blocked["search"]["feasible_count"], 0)
        self.assertEqual(blocked["plans"], [])
        self.assertTrue(blocked["search"]["complete"])
        for row in blocked["candidates"]:
            if row["spec"]["recipe"] == "regenerate":
                self.assertIn("forbidden_recipe", {reason["code"] for reason in row["reasons"]})

    def test_larger_fresh_carrier_cannot_repair_failed_early_detection(self):
        design = search_spec()
        design["grid"].update(recipes=["regenerate"], source_powers_mw=[0.02],
                              regeneration_powers_mw=[0.1, 2.0])
        result = self.plan(design)
        self.assertEqual(result["search"]["evaluated_count"], 2)
        self.assertEqual(result["plans"], [])
        self.assertLess(0.02 / 4 * 10 ** (-0.4 / 10), 0.01)
        for row in result["candidates"]:
            self.assertEqual(row["status"], "infeasible")
            self.assertIsNone(row["metrics"])
            failures = [reason for reason in row["reasons"]
                        if "early_detector" in reason["path"] and "sensitivity" in reason["message"]]
            self.assertEqual(len(failures), 4, row["reasons"])

    def test_receiver_margin_constraint_identifies_weakest_actual_route(self):
        design = search_spec()
        design["grid"].update(recipes=["passive"], source_powers_mw=[0.1, 0.2],
                              route_lengths_um=[[0.0, 10000.0, 2000.0, 1000.0]])
        design["constraints"] = {"min_receiver_margin_mw": 0.01}
        result = self.plan(design)
        low, high = result["candidates"]
        self.assertEqual(low["status"], "constraint_failed")
        self.assertEqual(high["status"], "feasible")
        for row in result["candidates"]:
            worst = row["worst_receiver"]
            self.assertEqual(worst["port"]["instance"], "lane0.use1.receiver")
            expected_margin = row["spec"]["source_power_mw"] / 4 * 10 ** (-2.4 / 10) - 0.01
            self.assertAlmostEqual(worst["sensitivity_margin_mw"], expected_margin, places=13)
            self.assertAlmostEqual(worst["margin_slack_mw"], expected_margin - 0.01, places=13)
        reason = low["reasons"][0]
        self.assertEqual(reason["code"], "receiver_margin_failed")
        self.assertEqual(reason["instance"], "lane0.use1.receiver")
        self.assertAlmostEqual(reason["required_relaxation"], -low["worst_receiver"]["margin_slack_mw"], places=13)

    def test_single_constraint_relaxations_have_real_feasible_witnesses(self):
        design = search_spec()
        design["grid"].update(recipes=["passive"], source_powers_mw=[0.1, 0.2])
        design["constraints"] = {"max_energy_pj": 0.8}
        result = self.plan(design)
        self.assertEqual(result["plans"], [])
        self.assertEqual(result["single_constraint_relaxations"][0]["constraint"], "constraints.max_energy_pj")
        witnesses = result["single_constraint_relaxations"][0]["candidate_ids"]
        self.assertEqual(set(witnesses), {row["id"] for row in result["candidates"]})
        for row, expected_energy in zip(result["candidates"], (0.9, 1.4)):
            self.assertEqual(row["status"], "constraint_failed")
            self.assertEqual(len(row["reasons"]), 1)
            reason = row["reasons"][0]
            self.assertAlmostEqual(reason["actual"], expected_energy, places=13)
            self.assertAlmostEqual(reason["required_relaxation"], expected_energy - 0.8, places=13)
            witness = realize(self.network, self.technology, row["spec"])
            self.assertTrue(witness["evaluation"]["feasible"])

    def test_baseline_locks_preserve_values_and_comparison_is_zero(self):
        baseline = realize(self.network, self.technology,
                           {"source_node": "x", "source_power_mw": 0.1, "lengths_um": 10000.0})
        design = search_spec()
        design["locks"] = {"recipe": "baseline", "source_power_mw": "baseline",
                           "instances": {"lane0.use0.receiver": "baseline"}}
        result = self.plan(design, baseline=baseline)
        self.assertEqual(result["search"]["evaluated_count"], 1)
        self.assertEqual(result["search"]["excluded_count"], 8)
        self.assertEqual(result["resolved_locks"], {"recipe": "passive", "source_power_mw": 0.1,
                                                   "instances": {"lane0.use0.receiver": "detector"}})
        kept = next(row for row in result["candidates"] if row["pareto"])
        comparison = kept["baseline_comparison"]
        self.assertTrue(comparison["comparable_metrics"])
        self.assertFalse(comparison["technology_assumptions_changed"])
        self.assertFalse(comparison["operating_point_changed"])
        self.assertEqual(comparison["changed_instances"], [])
        self.assertTrue(all(delta == 0.0 for delta in comparison["metric_deltas"].values()))
        self.assertEqual(result["plans"][0]["realization"], baseline)
        corrupted = deepcopy(baseline)
        corrupted["evaluation"]["metrics"]["energy_pj"] = 0.0
        with self.assertRaises(InputValidationError):
            self.plan(design, baseline=corrupted)

    def test_changed_technology_is_explicitly_a_different_comparison_scenario(self):
        baseline = realize(self.network, self.technology,
                           {"source_node": "x", "source_power_mw": 0.1, "lengths_um": 10000.0})
        next(component for component in self.technology["components"]
             if component["id"] == "source")["parameters"]["wall_plug_efficiency"] = 0.25
        design = search_spec()
        design["locks"] = {"recipe": "baseline", "source_power_mw": "baseline"}
        result = self.plan(design, baseline=baseline)
        kept = next(row for row in result["candidates"] if row["pareto"])
        comparison = kept["baseline_comparison"]
        self.assertTrue(comparison["technology_assumptions_changed"])
        self.assertFalse(comparison["comparable_metrics"])
        self.assertAlmostEqual(comparison["metric_deltas"]["energy_pj"], -0.1, places=13)
        self.assertEqual(comparison["changed_instances"], [])

    def test_instance_lock_selects_documented_global_detector_family(self):
        sensitive = deepcopy(next(component for component in self.technology["components"]
                                  if component["id"] == "detector"))
        sensitive["id"] = "sensitive_detector"
        sensitive["parameters"]["minimum_full_scale_power_mw"] = 0.005
        sensitive["costs"]["fixed_energy_pj"] = 0.2
        sensitive["notes"] = "Illustrative test family: lower full-scale threshold at twice the fixed energy."
        self.technology["components"].append(sensitive)
        design = search_spec()
        design["grid"].update(recipes=["passive"], source_powers_mw=[0.05],
                              route_lengths_um=[[10000.0, 0.0, 0.0, 0.0]],
                              components={"detector": ["detector", "sensitive_detector"]})
        design["locks"] = {"instances": {"lane0.use0.receiver": "sensitive_detector"}}
        result = self.plan(design)
        self.assertEqual(result["search"]["submitted_grid_size"], 2)
        self.assertEqual(result["search"]["excluded_count"], 1)
        self.assertEqual(result["search"]["feasible_count"], 1)
        self.assertIn("every instance", result["scope"]["component_families"])
        record = result["plans"][0]["realization"]
        receivers = [instance for instance in record["graph"]["instances"]
                     if instance["id"].endswith(".receiver")]
        self.assertEqual(len(receivers), 4)
        self.assertEqual({instance["component"] for instance in receivers}, {"sensitive_detector"})
        self.assertAlmostEqual(record["evaluation"]["metrics"]["energy_pj"], 0.05 / 0.2 + 4 * 0.2, places=13)
        self.assertTrue(check_realization(record)["valid"])

    def test_branch_specific_modulator_family_does_not_duplicate_passive_choices(self):
        modulator = deepcopy(next(component for component in self.technology["components"]
                                  if component["id"] == "modulator"))
        modulator["id"] = "expensive_modulator"
        modulator["costs"]["fixed_energy_pj"] = 0.2
        self.technology["components"].append(modulator)
        design = search_spec()
        design["grid"].update(source_powers_mw=[0.1, 0.2],
                              components={"modulator": ["modulator", "expensive_modulator"]})
        result = self.plan(design)
        self.assertEqual(result["search"]["submitted_grid_size"], 2 + 2 * 2 * 2)
        passive = [row for row in result["candidates"] if row["spec"]["recipe"] == "passive"]
        regenerated = [row for row in result["candidates"] if row["spec"]["recipe"] == "regenerate"]
        self.assertEqual(len(passive), 2)
        self.assertEqual(len(regenerated), 8)
        self.assertTrue(all(row["spec"]["components"]["modulator"] == "modulator" for row in passive))
        self.assertTrue(all(not row["pareto"] for row in regenerated
                            if row["spec"]["components"]["modulator"] == "expensive_modulator"))

    def test_budget_truncation_remains_visible_and_cannot_claim_full_frontier(self):
        design = search_spec()
        design["max_evaluations"] = 2
        result = self.plan(design)
        self.assertEqual(result["search"]["submitted_grid_size"], 9)
        self.assertEqual(result["search"]["evaluated_count"], 2)
        self.assertEqual(result["search"]["not_evaluated_count"], 7)
        self.assertFalse(result["search"]["complete"])
        self.assertIn("unevaluated candidates may dominate", result["search"]["optimality_scope"])
        self.assertEqual({choice(row) for row in result["candidates"] if row["pareto"]}, {("passive", 0.1, None)})
        self.assertTrue(check_physical_design(result)["valid"])
        forged = deepcopy(result)
        forged["search"]["complete"] = True
        self.assertFalse(check_physical_design(forged)["valid"])

    def test_readable_units_preserve_numeric_scenario_and_reject_duplicate_choices(self):
        numeric = search_spec()
        readable = deepcopy(numeric)
        readable["grid"].update(source_powers_mw=["50 µW", "100000 nW", "0.2 mW"],
                               regeneration_powers_mw=["100000 nW", "200 μW"],
                               route_lengths_um=["10 mm"])
        self.assertEqual(validate_design_spec(readable), validate_design_spec(numeric))
        readable["locks"] = {"source_power_mw": 0.1, "regeneration_power_mw": 0.1}
        locked = self.plan(readable)
        self.assertEqual(locked["search"]["evaluated_count"], 1)
        self.assertEqual(locked["search"]["feasible_count"], 1)
        readable["grid"]["source_powers_mw"] = ["100000 nW", "0.1 mW"]
        with self.assertRaises(InputValidationError) as caught:
            validate_design_spec(readable)
        self.assertIn("duplicate_design_choice", {diagnostic["code"] for diagnostic in caught.exception.diagnostics})


if __name__ == "__main__":
    unittest.main()
