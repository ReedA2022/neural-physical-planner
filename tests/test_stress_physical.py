"""Ensure the shipping stress campaign executes and its independent checks detect faults."""
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from scripts import stress_physical


class PhysicalStressCampaignTests(unittest.TestCase):
    def test_seeded_smoke_exercises_each_scenario_family(self):
        result = stress_physical.run_campaign(seed=9017, cases=12)
        self.assertEqual(result["scenario_count"], 12)
        self.assertEqual(result["passed_count"], 12, result["failures"])
        self.assertEqual(sum(result["family_counts"].values()), 12)
        self.assertEqual(len(result["family_counts"]), 7)
        self.assertFalse(result["external_validation"]["executed"])

    def test_scalar_oracle_detects_cost_timing_power_and_correlation_errors(self):
        # Bypass replay only inside this test: the scalar oracle must independently
        # reject even a self-consistent/replay-approved but numerically wrong result.
        original = stress_physical.realize
        for corruption in ("energy", "timing", "power", "correlation"):
            def corrupted(*args, **kwargs):
                record = deepcopy(original(*args, **kwargs))
                result = record["evaluation"]
                if corruption == "energy":
                    result["metrics"]["energy_pj"] *= 1.01
                elif corruption == "timing":
                    result["receivers"][0]["complete_symbol_ready_ns"] += .25
                elif corruption == "power":
                    result["receivers"][0]["full_scale_power_mw"] *= 1.01
                else:
                    result["receiver_covariance_estimate"][0][1] = 0.0
                return record
            with self.subTest(corruption=corruption):
                with patch.object(stress_physical, "realize", side_effect=corrupted), \
                     patch.object(stress_physical, "check_realization", return_value={"valid": True}):
                    with self.assertRaises(AssertionError):
                        stress_physical._oracle_case(random.Random(239), 1)

    def test_failure_report_keeps_reproducible_seed_and_case(self):
        with patch.object(stress_physical, "_component_case", side_effect=AssertionError("injected fault")):
            result = stress_physical.run_campaign(seed=41, cases=12)
        self.assertEqual(result["passed_count"], 11)
        self.assertEqual(result["failures"][0]["index"], 10)
        self.assertEqual(result["failures"][0]["family"], "component_conservation")
        self.assertIn("--seed 41 --cases 11", result["failures"][0]["reproduction"])

    def test_resource_budgets_and_seed_types_reject(self):
        for cases in (0, -1, 2001, True, 1.5):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                stress_physical.run_campaign(cases=cases)
        for seed in (True, 2.5, "two"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                stress_physical.run_campaign(seed=seed, cases=1)

    def test_positive_underflow_fails_closed_in_each_physical_contribution(self):
        for index in range(9):
            with self.subTest(case=index):
                stress_physical._underflow_case(random.Random(907), index)

    def test_true_zero_added_noise_remains_supported(self):
        pack = stress_physical.load_technology()
        fields = {"source": "relative_intensity_noise_rms",
                  "modulator": "added_sample_noise_rms",
                  "detector": "input_referred_noise_mw_rms"}
        for component in pack["components"]:
            if component["kind"] in fields:
                component["parameters"][fields[component["kind"]]] = 0.0
        result = stress_physical.realize(stress_physical.network(), pack,
                    stress_physical.spec(recipe="regenerate", regeneration_power_mw=.5))["evaluation"]
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertGreater(result["metrics"]["noise_rms_estimate"], 0)  # detector shot noise remains


if __name__ == "__main__":
    unittest.main()
