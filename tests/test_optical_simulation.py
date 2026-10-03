"""Export invariants and real SAX checks, with independent scalar power oracles."""
from copy import deepcopy
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

from npp.implementation import realize
from npp.models import InputValidationError
from npp.optical_simulation import export_optical_netlist, simulate_realization
from npp.technology import load_technology
from tests.test_implementation import scalar_fanout, vector_repeated_operand


SAX_INSTALLED = importlib.util.find_spec("sax") is not None


def fixture(count=3, recipe="passive", **changes):
    spec = {"source_node": "x", "source_power_mw": 1.0,
            "lengths_um": [100.0 * (i + 1) for i in range(count)], "recipe": recipe}
    if recipe == "regenerate":
        spec["regeneration_power_mw"] = 0.5
    spec.update(changes)
    return realize(scalar_fanout(count), load_technology(), spec)


class OpticalExportTests(unittest.TestCase):
    def test_export_needs_no_simulator_and_is_json_portable(self):
        record = fixture()
        with patch("npp.optical_simulation._load_sax", side_effect=AssertionError("export imported SAX")):
            exported = export_optical_netlist(record)
        self.assertEqual(json.loads(json.dumps(exported, allow_nan=False)), exported)
        self.assertEqual(exported["realization_hashes"], record["hashes"])
        self.assertEqual(exported["technology_status"]["calibration_status"], "illustrative")
        self.assertEqual(exported["scope"]["coverage"], "partial_optical_only")
        self.assertEqual(len(exported["segments"]), 1)
        segment = exported["segments"][0]
        self.assertEqual(len(segment["netlist"]["instances"]), 6)
        self.assertEqual(len(segment["outputs"]), 4)
        self.assertEqual(sum(o["kind"] == "matched_termination" for o in segment["outputs"]), 1)
        for name in segment["netlist"]["instances"]:
            self.assertTrue(name.isidentifier())
        # Returned snapshots are independent of caller-owned records.
        exported["realization_hashes"]["graph"] = "changed"
        self.assertNotEqual(record["hashes"]["graph"], "changed")

    def test_every_passive_port_appears_exactly_once(self):
        exported = export_optical_netlist(fixture(count=8))
        for segment in exported["segments"]:
            netlist = segment["netlist"]
            observed = list(netlist["ports"].values())
            for left, right in netlist["connections"].items():
                observed.extend([left, right])
            expected = [f"{iid},{port}" for iid, inst in netlist["instances"].items()
                        for port in exported["model_definitions"][inst["component"]]["ports"]]
            self.assertCountEqual(observed, expected)
            self.assertEqual(len(observed), len(set(observed)))

    def test_regeneration_cuts_every_electrical_boundary(self):
        exported = export_optical_netlist(fixture(recipe="regenerate"))
        self.assertEqual(len(exported["segments"]), 4)
        self.assertEqual(len(exported["excluded_electrical_connections"]), 3)
        launch = exported["segments"][0]
        self.assertEqual(launch["source"]["role"], "encoded_launch")
        self.assertEqual(sum(o["kind"] == "regeneration_detector_input" for o in launch["outputs"]), 3)
        for segment in exported["segments"][1:]:
            self.assertEqual(segment["source"]["role"], "constant_carrier")
            self.assertIsNone(segment["source"]["source_node"])
            self.assertEqual(segment["source"]["power_mw"], 0.5)
            self.assertEqual(len(segment["outputs"]), 1)
            self.assertEqual(segment["outputs"][0]["kind"], "receiver_input")
            self.assertCountEqual([i["component"] for i in segment["netlist"]["instances"].values()],
                                  ["npp_modulator_full_scale_v1", "npp_waveguide_v1"])
        self.assertIn("sample fixed to 1", exported["scope"]["boundary_conditions"])

    def test_vector_and_repeated_nn_uses_retain_distinct_bindings(self):
        record = realize(vector_repeated_operand(), load_technology(),
                         {"source_node": "x", "source_power_mw": 1.0})
        exported = export_optical_netlist(record)
        self.assertEqual([s["source"]["lane"] for s in exported["segments"]], [0, 1])
        bindings = [o["binding"] for s in exported["segments"] for o in s["outputs"] if o["kind"] == "receiver_input"]
        self.assertCountEqual(bindings, record["graph"]["receivers"])

    def test_tampering_rejected_before_engine_or_export(self):
        record = fixture()
        for field in ("metrics", "hash", "instance"):
            altered = deepcopy(record)
            if field == "metrics":
                altered["evaluation"]["metrics"]["energy_pj"] += 0.1
            elif field == "hash":
                altered["hashes"]["technology"] = "fake"
            else:
                altered["graph"]["instances"][0]["parameters"]["power_mw"] *= 2
            for function in (export_optical_netlist, simulate_realization):
                with self.subTest(field=field, function=function.__name__), self.assertRaises(InputValidationError):
                    function(altered)

    def test_infeasible_can_export_but_cannot_simulate(self):
        record = fixture(source_power_mw=0.01)
        self.assertFalse(export_optical_netlist(record)["baseline_feasible"])
        with patch("npp.optical_simulation._load_sax", side_effect=AssertionError("engine must not start")):
            with self.assertRaises(InputValidationError) as caught:
                simulate_realization(record)
        self.assertEqual(caught.exception.diagnostics[0]["code"], "infeasible_optical_realization")

    def test_strict_wavelengths_tolerances_and_limits(self):
        record = fixture()
        invalid_wavelengths = [[], [1550.0] * 2, [True], ["1550"], [float("nan")],
                               [float("inf")], [0.0], [-1.0], [10**400], 1550.0,
                               [1540.0 + i / 100 for i in range(34)]]
        for values in invalid_wavelengths:
            with self.subTest(values=repr(values)[:100]), self.assertRaises(InputValidationError):
                simulate_realization(record, values)
        for key in ("rtol", "atol_mw"):
            for value in (True, "0.1", float("inf"), float("nan"), -1.0, 10**400, 1.0):
                with self.subTest(key=key, value=str(value)[:50]), self.assertRaises(InputValidationError):
                    simulate_realization(record, **{key: value})
        with patch("npp.optical_simulation.MAX_SEGMENT_INSTANCES", 1), self.assertRaises(InputValidationError) as caught:
            export_optical_netlist(record)
        self.assertEqual(caught.exception.diagnostics[0]["code"], "optical_simulation_size_limit")

    def test_sweep_checks_envelope_before_engine(self):
        with patch("npp.optical_simulation._load_sax", side_effect=AssertionError("engine must not start")):
            with self.assertRaises(InputValidationError) as caught:
                simulate_realization(fixture(), [1550.0, 1570.0])
        self.assertEqual(caught.exception.diagnostics[0]["code"], "infeasible_optical_sweep")
        self.assertEqual(caught.exception.diagnostics[0]["path"], "wavelengths_nm.1")

    def test_missing_dependency_has_no_silent_fallback(self):
        import builtins
        original_import = builtins.__import__
        def without_sax(name, *args, **kwargs):
            if name == "sax":
                raise ImportError("test missing SAX")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_sax), self.assertRaises(InputValidationError) as caught:
            simulate_realization(fixture())
        self.assertEqual(caught.exception.diagnostics[0]["code"], "optional_dependency_missing")


@unittest.skipUnless(SAX_INSTALLED, "SAX optional photonics extra is not installed")
class RealSaxComparisonTests(unittest.TestCase):
    def test_real_solver_and_independent_two_four_eight_way_oracle(self):
        import sax
        for count in (2, 4, 8):
            with self.subTest(count=count), patch("sax.circuit", wraps=sax.circuit) as called:
                result = simulate_realization(fixture(count), [1540.0, 1550.0, 1560.0])
            self.assertEqual(called.call_count, 1)
            self.assertEqual(called.call_args.kwargs["backend"], "klu")
            self.assertTrue(result["passed"], result["diagnostics"])
            self.assertEqual(result["engine"]["name"], "SAX")
            depth = int(math.log2(count))
            for sample in result["samples"]:
                for row in sample["comparisons"]:
                    use = row["binding"]["use_index"]
                    expected = (0.5 * 10 ** (-0.2 / 10)) ** depth * 10 ** (-0.0002 * (use + 1) * 100 / 10)
                    self.assertAlmostEqual(row["sax_power_mw"], expected, places=12)
                    self.assertAlmostEqual(row["expected_power_mw"], expected, places=12)
            json.dumps(result, allow_nan=False)

    def test_non_power_of_two_accounts_for_terminated_power(self):
        result = simulate_realization(fixture(3))
        rows = result["samples"][0]["comparisons"]
        terminal = next(r for r in rows if r["kind"] == "matched_termination")
        self.assertAlmostEqual(terminal["sax_power_mw"], 0.25 * 10 ** (-0.4 / 10), places=13)
        self.assertEqual(len(rows), 4)
        self.assertTrue(result["passed"])

    def test_regeneration_validates_separate_launch_and_carrier_segments(self):
        result = simulate_realization(fixture(recipe="regenerate"), [1540., 1560.])
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["segment_count"], 4)
        self.assertEqual(result["summary"]["comparison_count"], 14)
        for sample in result["samples"]:
            for row in sample["comparisons"]:
                if row["kind"] == "receiver_input":
                    use = row["binding"]["use_index"]
                    expected = 0.5 * 10 ** (-1.0 / 10) * 10 ** (-0.0002 * (use + 1) * 100 / 10)
                else:
                    expected = (0.5 * 10 ** (-0.2 / 10)) ** 2
                self.assertAlmostEqual(row["sax_power_mw"], expected, places=12)
        self.assertIn("optical phase and group delay", result["scope"]["unchecked"])

    def test_wrong_independent_field_model_reports_failed_comparison(self):
        import npp.optical_simulation as adapter
        original = adapter._sax_models
        def wrong_models(sax, jnp):
            models = original(sax, jnp)
            splitter = models["npp_splitter_v1"]
            def wrong_splitter(*, wl=1.55, ratio=.5, loss_db=0.0):
                return {key: value * 1.1 for key, value in splitter(wl=wl, ratio=ratio, loss_db=loss_db).items()}
            models["npp_splitter_v1"] = wrong_splitter
            return models
        with patch.object(adapter, "_sax_models", side_effect=wrong_models):
            result = simulate_realization(fixture())
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["summary"]["max_relative_error"], 0.1)
        self.assertTrue(all(not row["passed"] for row in result["samples"][0]["comparisons"]))
        self.assertTrue(all(d["code"] == "optical_power_mismatch" for d in result["diagnostics"]))

    def test_solver_exception_is_structured_and_never_a_pass(self):
        with patch("sax.circuit", side_effect=RuntimeError("engine test failure")):
            with self.assertRaises(InputValidationError) as caught:
                simulate_realization(fixture())
        self.assertEqual(caught.exception.diagnostics[0]["code"], "optical_engine_error")
        self.assertIn("engine test failure", caught.exception.diagnostics[0]["message"])


if __name__ == "__main__":
    unittest.main()
