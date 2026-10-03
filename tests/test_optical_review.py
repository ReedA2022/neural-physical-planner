"""Independent Gate 3 regressions: export coverage and external optical power.

These tests use hand calculations and actual SAX composition. They do not
certify the illustrative reference devices or any excluded electrical model.
"""
from copy import deepcopy
import importlib.util
import json
import unittest
from unittest.mock import patch

from npp.implementation import realize
from npp.models import InputValidationError
from npp.optical_report import render_optical_simulation
from npp.optical_simulation import export_optical_netlist, simulate_realization
from npp.technology import load_technology
from tests.fixtures import network, node
from tests.test_optical_report import presentation_fixture


HAS_SAX = importlib.util.find_spec("sax") is not None


def make_record(count=3, lanes=1, recipe="passive", technology=None, **changes):
    name = "encoded.value:0-a"
    net = network([node(name, "input", size=lanes,
                        input_bounds={"lower": [0.0] * lanes, "upper": [1.0] * lanes})],
                  [name] * count)
    spec = {"source_node": name, "source_power_mw": 1.0, "recipe": recipe,
            "lengths_um": [float(100 * i) for i in range(count)]}
    if recipe == "regenerate":
        spec["regeneration_power_mw"] = 0.4
    spec.update(changes)
    return realize(net, technology or load_technology(), spec)


class OpticalReviewContractTests(unittest.TestCase):
    def test_multilane_regeneration_has_exact_component_and_boundary_coverage(self):
        record = make_record(count=5, lanes=2, recipe="regenerate")
        exported = json.loads(json.dumps(export_optical_netlist(record), allow_nan=False))
        graph = record["graph"]
        kinds = {c["id"]: c["kind"] for c in record["technology"]["components"]}
        expected_components = [i["id"] for i in graph["instances"]
                               if kinds[i["component"]] in {"splitter", "waveguide", "modulator"}]
        observed_components = [m["instance"] for s in exported["segments"]
                               for m in s["instance_mapping"].values()]
        self.assertCountEqual(observed_components, expected_components)
        self.assertEqual(len(observed_components), len(set(observed_components)))
        expected_sources = [i["id"] for i in graph["instances"] if kinds[i["component"]] == "source"]
        self.assertCountEqual([s["source"]["endpoint"]["instance"] for s in exported["segments"]], expected_sources)
        self.assertEqual(len(exported["segments"]), 12)
        outputs = [o for s in exported["segments"] for o in s["outputs"]]
        self.assertCountEqual([o["endpoint"]["instance"] for o in outputs if o["kind"] != "matched_termination"],
                              [i["id"] for i in graph["instances"] if kinds[i["component"]] == "detector"])
        self.assertCountEqual([o["binding"] for o in outputs if o["kind"] == "matched_termination"], graph["terminations"])
        self.assertCountEqual([o["binding"] for o in outputs if o["kind"] == "receiver_input"], graph["receivers"])
        self.assertEqual(len(exported["excluded_electrical_connections"]), 10)
        for segment in exported["segments"]:
            netlist = segment["netlist"]
            ports = list(netlist["ports"].values())
            ports.extend(p for pair in netlist["connections"].items() for p in pair)
            declared = [f"{identifier},{port}" for identifier, instance in netlist["instances"].items()
                        for port in exported["model_definitions"][instance["component"]]["ports"]]
            self.assertCountEqual(ports, declared)
            self.assertEqual(len(ports), len(set(ports)))

    def test_custom_component_names_map_to_models_by_kind(self):
        technology = load_technology()
        choices = {}
        for component in technology["components"]:
            component["id"] = f"custom.{component['kind']}:revision-7"
            choices[component["kind"]] = component["id"]
        record = make_record(technology=technology, components=choices)
        exported = export_optical_netlist(record)
        mappings = [m for s in exported["segments"] for m in s["instance_mapping"].values()]
        self.assertTrue(all(m["technology_component"].startswith("custom.") for m in mappings))
        self.assertTrue(all(name.isidentifier() for s in exported["segments"] for name in s["netlist"]["instances"]))
        self.assertIn("encoded.value:0-a", exported["segments"][0]["source"]["source_node"])

    def test_detector_only_wavelength_guard_prevents_external_engine_start(self):
        technology = load_technology()
        detector = next(c for c in technology["components"] if c["kind"] == "detector")
        detector["operating_envelope"]["wavelength_nm"] = {"minimum": 1549.0, "maximum": 1551.0}
        record = make_record(technology=technology)
        with patch("npp.optical_simulation._load_sax", side_effect=AssertionError("engine started")):
            with self.assertRaises(InputValidationError) as caught:
                simulate_realization(record, [1550.0, 1540.0])
        self.assertTrue(all(d["code"] == "infeasible_optical_sweep" for d in caught.exception.diagnostics))
        self.assertTrue(all(d["path"] == "wavelengths_nm.1" for d in caught.exception.diagnostics))
        self.assertTrue(any("receiver" in json.dumps(d["guard"]) for d in caught.exception.diagnostics))

    def test_malformed_portable_records_fail_before_optional_import(self):
        baseline = make_record()
        cases = [None, [], {}, {**baseline, "extra": "unrecorded"}]
        for field in ("graph", "technology", "evaluation", "hashes", "spec", "network"):
            altered = deepcopy(baseline)
            altered[field] = ["malformed"]
            cases.append(altered)
        with patch("npp.optical_simulation._load_sax", side_effect=AssertionError("engine started")):
            for index, record in enumerate(cases):
                for function in (export_optical_netlist, simulate_realization):
                    with self.subTest(index=index, function=function.__name__), self.assertRaises(InputValidationError):
                        function(record)


@unittest.skipUnless(HAS_SAX, "real SAX optional dependency is not installed")
class OpticalReviewExternalTests(unittest.TestCase):
    def test_custom_losses_three_way_split_unequal_routes_and_unused_leaf_oracle(self):
        technology = load_technology()
        for component in technology["components"]:
            if component["kind"] == "splitter":
                component["parameters"].update(insertion_loss_db=0.7)
            elif component["kind"] == "waveguide":
                component["parameters"]["attenuation_db_per_um"] = 0.0009
        lengths = [0.0, 321.0, 700.0]
        result = simulate_realization(make_record(technology=technology, lengths_um=lengths), [1560.0, 1540.0, 1550.0])
        self.assertTrue(result["passed"], result["diagnostics"])
        self.assertEqual([s["wavelength_nm"] for s in result["samples"]], [1560.0, 1540.0, 1550.0])
        for sample in result["samples"]:
            for row in sample["comparisons"]:
                if row["kind"] == "matched_termination":
                    expected = 0.25 * 10**(-1.4 / 10)
                else:
                    index = row["binding"]["use_index"]
                    expected = 0.25 * 10**(-(1.4 + lengths[index] * 0.0009) / 10)
                self.assertAlmostEqual(row["sax_power_mw"], expected, places=12)
                self.assertAlmostEqual(row["expected_power_mw"], expected, places=12)

    def test_scalar_oracle_perturbation_does_not_change_external_power(self):
        import npp.optical_simulation as adapter
        original = adapter.evaluate_implementation
        def wrong_scalar(*args, **kwargs):
            evaluation = original(*args, **kwargs)
            for row in evaluation["ledger"]:
                if row["kind"] == "detector":
                    row["input_power_mw"] *= 1.2
            return evaluation
        record = make_record(count=2, lengths_um=0.0)
        with patch.object(adapter, "evaluate_implementation", side_effect=wrong_scalar):
            comparison = simulate_realization(record)
        self.assertFalse(comparison["passed"])
        for row in comparison["samples"][0]["comparisons"]:
            expected = 0.5 * 10**(-0.2 / 10)
            self.assertAlmostEqual(row["sax_power_mw"], expected, places=13)
            self.assertAlmostEqual(row["expected_power_mw"], 1.2 * expected, places=13)
            self.assertFalse(row["passed"])

    def test_external_nonfinite_and_missing_transfer_never_produce_pass(self):
        class FakeCircuit:
            def __init__(self, coefficients):
                self.coefficients = coefficients
            def __call__(self, **kwargs):
                return self.coefficients
        record = make_record(count=2)
        for value in (complex(float("nan"), 0.0), complex(float("inf"), 0.0)):
            with self.subTest(value=value), patch("sax.circuit", return_value=(FakeCircuit({("output0", "input"): value}), {})):
                with self.assertRaises(InputValidationError) as caught:
                    simulate_realization(record)
                self.assertEqual(caught.exception.diagnostics[0]["code"], "optical_engine_nonfinite")
        with patch("sax.circuit", return_value=(FakeCircuit({}), {})):
            with self.assertRaises(InputValidationError) as caught:
                simulate_realization(record)
            self.assertEqual(caught.exception.diagnostics[0]["code"], "optical_engine_error")


class OpticalReviewReportTests(unittest.TestCase):
    def assertIncomplete(self, record):
        report = render_optical_simulation(record)
        self.assertIn("Incomplete or inconsistent comparison record", report)
        self.assertNotIn("Recorded optical-power agreement within tolerance", report)

    def test_outside_adapter_tolerance_contract_cannot_claim_agreement(self):
        result = presentation_fixture()
        result["tolerances"]["rtol"] = 1.0
        result["tolerances"]["atol_mw"] = 0.0
        for row in result["samples"][0]["comparisons"]:
            row["sax_power_mw"] = 2.0 * row["expected_power_mw"]
            row["absolute_error_mw"] = row["expected_power_mw"]
            row["relative_error"] = 1.0
            row["allowed_error_mw"] = row["expected_power_mw"]
        result["summary"]["max_absolute_error_mw"] = max(r["absolute_error_mw"] for r in result["samples"][0]["comparisons"])
        result["summary"]["max_relative_error"] = 1.0
        self.assertIncomplete(result)
        result = presentation_fixture()
        result["tolerances"]["atol_mw"] = 1.0
        for row in result["samples"][0]["comparisons"]:
            row["allowed_error_mw"] = 1.0 + result["tolerances"]["rtol"] * row["expected_power_mw"]
        self.assertIncomplete(result)

    def test_required_export_structures_and_versions_cannot_be_omitted(self):
        for mutation in ("segments", "provenance", "schema", "netlist_port", "mapping"):
            result = presentation_fixture()
            if mutation == "segments":
                for segment in result["export"]["segments"]:
                    segment.pop("netlist")
                    segment.pop("instance_mapping")
            elif mutation == "provenance":
                result["export"].pop("technology")
                result["export"].pop("model_definitions")
                result["export"]["realization_hashes"] = {"fake": 0}
                result["export"]["technology_status"] = {"fake": 0}
            elif mutation == "schema":
                result["schema_version"] = "999"
                result["export"]["schema_version"] = "999"
            elif mutation == "netlist_port":
                result["export"]["segments"][0]["netlist"]["ports"]["output0"] = "nonexistent,p_out"
            else:
                result["export"]["segments"][0]["instance_mapping"].clear()
            with self.subTest(mutation=mutation):
                self.assertIncomplete(result)

    def test_embedded_technology_must_match_its_recorded_content_hash(self):
        for mutation in ("snapshot", "hash"):
            result = presentation_fixture()
            if mutation == "snapshot":
                result["export"]["technology"]["name"] += " changed after export"
            else:
                result["export"]["realization_hashes"]["technology"] = "0" * 64
            with self.subTest(mutation=mutation):
                self.assertIncomplete(result)


if __name__ == "__main__":
    unittest.main()
