"""Strict contracts, replay and bounded-search regression tests for Part 4."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from npp.designer import (check_physical_design, design_schema, plan_physical,
                          validate_design_spec)
from npp.implementation import realize
from npp.models import InputValidationError
from npp.technology import load_technology


def network():
    return {"schema_version": "0.1", "name": "Designer test four-way fanout", "nodes": [
        {"id": "x", "op": "input", "size": 1, "input_bounds": {"lower": [0.0], "upper": [1.0]}}], "outputs": ["x"] * 4}


def design():
    return {"source_node": "x", "grid": {"recipes": ["passive", "regenerate"], "source_powers_mw": [0.05, 0.1, 0.2],
                                           "regeneration_powers_mw": [0.1, 0.2], "route_lengths_um": [10000.0]}}


class DesignerTests(unittest.TestCase):
    def setUp(self):
        self.network, self.technology, self.design = network(), load_technology(), design()

    def plan(self, spec=None, **kwargs):
        return plan_physical(self.network, self.technology, self.design if spec is None else spec, **kwargs)

    def baseline(self, recipe="passive", power=0.05):
        return realize(self.network, self.technology, {"source_node": "x", "recipe": recipe, "source_power_mw": power,
            "regeneration_power_mw": None if recipe == "passive" else 0.1, "lengths_um": 10000.0})

    def test_readable_units_normalize_to_canonical_spec(self):
        spec = design()
        spec["grid"].update(source_powers_mw=["50 uW", "0.1 mW", "0.0002 W"], regeneration_powers_mw=["100 µW", "200 μW"], route_lengths_um=["10 mm"])
        spec["operating"] = {"wavelength_nm": "1.55 um", "temperature_c": "25 °C", "symbol_duration_ns": "1000 ps"}
        spec["constraints"] = {"max_energy_pj": "1 nJ", "max_area_um2": "1 mm²", "max_latency_ns": "1 us", "max_noise_rms_estimate": ".1", "min_receiver_margin_mw": "1 uW"}
        normalized = validate_design_spec(spec)
        self.assertEqual(normalized["grid"], validate_design_spec(design())["grid"])
        self.assertEqual(normalized["constraints"]["max_area_um2"], 1e6)
        self.assertEqual(normalized["operating"]["symbol_duration_ns"], 1.0)
        self.assertEqual(normalized, validate_design_spec(normalized))

    def test_invalid_quantities_and_unknown_fields_are_actionable(self):
        for bad in [True, float("nan"), float("inf"), 10 ** 400, "1 dBm", "1 mm", "none", [], {}]:
            with self.subTest(bad=type(bad).__name__):
                spec = design()
                spec["grid"]["source_powers_mw"] = [bad]
                with self.assertRaises(InputValidationError):
                    self.plan(spec)
        spec = design()
        spec["grid"]["source_powers_mw"] = ["1 mm"]
        with self.assertRaisesRegex(InputValidationError, "unsupported power unit"):
            self.plan(spec)
        spec = design()
        spec["typo"] = 1
        with self.assertRaisesRegex(InputValidationError, "typo"):
            self.plan(spec)

    def test_duplicates_bool_budget_and_oversized_grid_rejected(self):
        for updates in [{"max_evaluations": True}, {"max_evaluations": 129}, {"objectives": ["energy_pj", "energy_pj"]},
                        {"forbidden_recipes": ["passive", "passive"]}]:
            with self.subTest(updates=updates), self.assertRaises(InputValidationError):
                self.plan({**design(), **updates})
        spec = design()
        spec["grid"]["source_powers_mw"] = [0.1, "100 uW"]
        with self.assertRaisesRegex(InputValidationError, "unique after unit"):
            self.plan(spec)
        spec = design()
        spec["grid"].update(source_powers_mw=[float(i) for i in range(1, 33)], regeneration_powers_mw=[float(i) for i in range(1, 33)], route_lengths_um=[float(i) for i in range(32)])
        with self.assertRaisesRegex(InputValidationError, "exceeds 4096"):
            self.plan(spec)

    def test_schema_exposes_strict_bounded_contract(self):
        schema = design_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["max_evaluations"]["maximum"], 128)

    def test_empty_solution_reasons_for_locks_and_forbidden_choices(self):
        spec = design()
        spec["locks"] = {"source_power_mw": 0.3}
        result = self.plan(spec)
        self.assertEqual(result["search"]["excluded_count"], 9)
        self.assertEqual(result["search"]["evaluated_count"], 0)
        self.assertTrue(result["search"]["complete"])
        self.assertEqual(result["plans"], [])
        self.assertTrue(all(row["reasons"][0]["path"] == "locks.source_power_mw" for row in result["candidates"]))
        spec = design()
        spec["forbidden_recipes"] = ["passive", "regenerate"]
        result = self.plan(spec)
        self.assertEqual(result["search"]["excluded_count"], 9)
        self.assertTrue(all(r["reasons"][0]["code"] == "forbidden_recipe" for r in result["candidates"]))

    def test_truncation_does_not_spend_budget_on_excluded_candidates(self):
        spec = design()
        spec.update(max_evaluations=1, locks={"recipe": "regenerate"})
        result = self.plan(spec)
        self.assertEqual(result["search"]["excluded_count"], 3)
        self.assertEqual(result["search"]["evaluated_count"], 1)
        self.assertEqual(result["search"]["not_evaluated_count"], 5)
        self.assertFalse(result["search"]["complete"])
        self.assertIn("may dominate", result["search"]["optimality_scope"])
        self.assertEqual(result["plans"][0]["id"], "candidate-0004")
        self.assertTrue(check_physical_design(result)["valid"])

    def test_baseline_locks_preserve_values_and_instance_presence(self):
        baseline = self.baseline("regenerate")
        spec = design()
        spec["locks"] = {"source_power_mw": "baseline", "instances": {"lane0.use0.modulator": "baseline"}}
        result = self.plan(spec, baseline=baseline)
        self.assertEqual(result["resolved_locks"]["source_power_mw"], 0.05)
        self.assertEqual(result["resolved_locks"]["instances"]["lane0.use0.modulator"], "modulator")
        self.assertTrue(all(p["realization"]["spec"]["recipe"] == "regenerate" for p in result["plans"]))
        self.assertTrue(all(p["realization"]["spec"]["source_power_mw"] == 0.05 for p in result["plans"]))
        self.assertTrue(any(r["code"] == "locked_instance_absent" for c in result["candidates"] for r in c["reasons"]))

    def test_passive_baseline_null_carrier_lock_excludes_regeneration(self):
        spec = design()
        spec["locks"] = {"regeneration_power_mw": "baseline"}
        result = self.plan(spec, baseline=self.baseline(power=0.1))
        self.assertIsNone(result["resolved_locks"]["regeneration_power_mw"])
        self.assertEqual(result["search"]["excluded_count"], 6)
        self.assertTrue(all(p["realization"]["spec"]["recipe"] == "passive" for p in result["plans"]))

    def test_unknown_locks_and_baseline_requests_rejected(self):
        for locks in [{"recipe": "baseline"}, {"instances": {"unknown.instance": "source"}},
                      {"instances": {"lane0.launch": "detector"}}, {"instances": {"lane0.use0.modulator": "baseline"}}]:
            with self.subTest(locks=locks), self.assertRaises(InputValidationError):
                self.plan({**design(), "locks": locks})
        spec = design()
        spec["grid"]["components"] = {"source": ["absent"]}
        with self.assertRaisesRegex(InputValidationError, "not a declared source"):
            self.plan(spec)

    def test_baseline_context_and_tampering_rejected(self):
        baseline = self.baseline()
        baseline["evaluation"]["feasible"] = True
        with self.assertRaisesRegex(InputValidationError, "baseline failed"):
            self.plan(baseline=baseline)
        baseline = self.baseline()
        other = network()
        other["name"] = "Different canonical context"
        with self.assertRaisesRegex(InputValidationError, "exact same canonical NN"):
            plan_physical(other, self.technology, design(), baseline=baseline)

    def test_baseline_changes_and_technology_scenario_flag(self):
        baseline = self.baseline(power=0.1)
        result = self.plan(baseline=baseline)
        same = next(c for c in result["candidates"] if c["spec"]["recipe"] == "passive" and c["spec"]["source_power_mw"] == 0.1)
        self.assertEqual(same["baseline_comparison"]["changed_instances"], [])
        self.assertTrue(same["baseline_comparison"]["comparable_metrics"])
        self.assertTrue(all(v == 0 for v in same["baseline_comparison"]["metric_deltas"].values()))
        technology = deepcopy(self.technology)
        technology["version"] = "changed illustrative scenario"
        result = plan_physical(self.network, technology, design(), baseline=baseline)
        comparison = result["plans"][0]["id"]
        row = next(c for c in result["candidates"] if c["id"] == comparison)
        self.assertTrue(row["baseline_comparison"]["technology_assumptions_changed"])
        self.assertFalse(row["baseline_comparison"]["comparable_metrics"])
        self.assertIsNotNone(row["baseline_comparison"]["metric_deltas"])

    def test_constraints_explain_numeric_witnesses_without_false_relaxations(self):
        spec = design()
        spec["constraints"] = {"max_energy_pj": 0.0}
        result = self.plan(spec)
        self.assertEqual(result["plans"], [])
        self.assertEqual(len(result["single_constraint_relaxations"]), 1)
        group = result["single_constraint_relaxations"][0]
        self.assertEqual(group["constraint"], "constraints.max_energy_pj")
        self.assertEqual(len(group["candidate_ids"]), 8)
        for row in result["candidates"]:
            if row["status"] == "constraint_failed":
                reason = row["reasons"][0]
                self.assertEqual(reason["actual"], row["metrics"]["energy_pj"])
                self.assertEqual(reason["required_relaxation"], reason["actual"])
        spec["constraints"]["max_area_um2"] = 0.0
        self.assertEqual(self.plan(spec)["single_constraint_relaxations"], [])

    def test_final_receiver_margin_and_cost_driver_explanation(self):
        spec = design()
        spec["grid"]["route_lengths_um"] = [[0.0, 1000.0, 2000.0, 10000.0]]
        spec["constraints"] = {"min_receiver_margin_mw": 0.1}
        result = self.plan(spec)
        for row in result["candidates"]:
            if row["metrics"] is not None:
                self.assertEqual(row["worst_receiver"]["port"]["instance"], "lane0.use3.receiver")
                self.assertEqual(row["worst_receiver"]["required_margin_mw"], 0.1)
                self.assertEqual(len(row["cost_drivers"]), 5)
                energies = [c["energy_pj"] for c in row["cost_drivers"]]
                self.assertEqual(energies, sorted(energies, reverse=True))
        self.assertTrue(any(r["code"] == "receiver_margin_failed" and r["instance"] == "lane0.use3.receiver" for c in result["candidates"] for r in c["reasons"]))

    def test_route_count_and_work_limits_precede_realizations(self):
        spec = design()
        spec["grid"]["route_lengths_um"] = [[1000.0, 2000.0]]
        with self.assertRaisesRegex(InputValidationError, "exactly 4"):
            self.plan(spec)
        wide = network()
        wide["nodes"][0].update(size=64, input_bounds={"lower": [0.0] * 64, "upper": [1.0] * 64})
        spec = design()
        spec["grid"]["source_powers_mw"] = [float(i) / 100 for i in range(1, 33)]
        with patch("npp.designer.realize") as mock, self.assertRaisesRegex(InputValidationError, "bounded instance/covariance"):
            plan_physical(wide, self.technology, spec)
        mock.assert_not_called()
        with patch("npp.designer.MAX_RECORD_BYTES", 100), patch("npp.designer.realize") as mock, self.assertRaisesRegex(InputValidationError, "snapshot"):
            self.plan()
        mock.assert_not_called()

    def test_replay_binds_all_claims_and_rejects_malformed_records(self):
        result = self.plan(baseline=self.baseline())
        self.assertTrue(check_physical_design(result)["valid"])
        for field in result:
            with self.subTest(field=field):
                bad = deepcopy(result)
                bad[field] = None
                self.assertFalse(check_physical_design(bad)["valid"])
        for bad in [None, [], {}, {**result, "extra": True}]:
            self.assertFalse(check_physical_design(bad)["valid"])
        for replacement in [True, float("nan"), float("inf"), "9"]:
            bad = deepcopy(result)
            bad["search"]["submitted_grid_size"] = replacement
            self.assertFalse(check_physical_design(bad)["valid"])
        bad = deepcopy(result)
        bad["candidates"][0]["reasons"] = []
        self.assertFalse(check_physical_design(bad)["valid"])


if __name__ == "__main__":
    unittest.main()
