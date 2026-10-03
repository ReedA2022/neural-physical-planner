"""Physical distribution tests with scalar hand-calculation oracles.

Expected powers, costs and photoelectron noise are calculated here without calling
the technology evaluator or using the implementation graph's reported metrics.
The numeric devices are deliberately illustrative, not experimental evidence.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import unittest

import numpy as np

from npp.models import InputValidationError
from npp.implementation import (build_implementation, check_realization, evaluate_implementation,
                                realize, validate_implementation,
                                validate_realization_spec)
from npp.technology import load_technology
from tests.fixtures import network, node


def scalar_fanout(count):
    """Output repetitions are distinct physical destinations in the NN contract."""
    return network([
        node("x", "input", input_bounds={"lower": [0.0], "upper": [1.0]})
    ], ["x"] * count)


def vector_repeated_operand():
    return network([
        node("x", "input", size=2,
             input_bounds={"lower": [0.0, 0.0], "upper": [0.5, 0.5]}),
        node("sum", "add", ["x", "x"], size=2),
    ], ["sum", "x"])


def receiver_variance(power_mw, duration_ns=1.0):
    # E_photon = h*c/lambda; detector QE=.8 and electronics RMS=.0001 mW.
    incident_energy_j = power_mw * 1e-3 * duration_ns * 1e-9
    photon_energy_j = 6.62607015e-34 * 299_792_458.0 / (1550.0 * 1e-9)
    electrons = 0.8 * incident_energy_j / photon_energy_j
    return 1.0 / electrons + (0.0001 / power_mw) ** 2


def passive_power(launch_mw, depth, length_um):
    # Each balanced 50:50 splitter has .2 dB *excess* loss; guide .0002 dB/um.
    return launch_mw * (0.5 * 10.0 ** (-0.2 / 10.0)) ** depth * 10.0 ** (-0.0002 * length_um / 10.0)


def propagation_delay(length_um):
    return 4.0 * length_um * 1e-6 / 299_792_458.0 * 1e9


class IndependentDistributionOracleTests(unittest.TestCase):
    def setUp(self):
        self.tech = load_technology()

    def graph(self, count=2, **changes):
        spec = {"source_node": "x", "source_power_mw": 1.0,
                "lengths_um": 0.0}
        spec.update(changes)
        return build_implementation(scalar_fanout(count), self.tech, spec)

    def test_two_way_power_cost_time_and_correlated_noise(self):
        lengths = [100.0, 200.0]
        result = evaluate_implementation(self.graph(lengths_um=lengths), self.tech)
        self.assertTrue(result["feasible"], result["diagnostics"])
        powers = [passive_power(1.0, 1, length) for length in lengths]
        np.testing.assert_allclose([r["full_scale_power_mw"] for r in result["receivers"]], powers, rtol=1e-13)
        self.assertEqual(result["instance_counts"], {"source": 1, "splitter": 1, "waveguide": 2, "detector": 2})
        self.assertAlmostEqual(result["metrics"]["energy_pj"], 1.0 / 0.2 + 2 * 0.1, places=12)
        self.assertAlmostEqual(result["metrics"]["area_um2"], 1000.0 + 20.0 + 200.0 + 0.5 * sum(lengths), places=12)
        self.assertAlmostEqual(result["metrics"]["latency_ns"], 1.0 + 0.1 + 0.2 + propagation_delay(200.0), places=12)
        expected = np.full((2, 2), 0.001 ** 2)
        for index, power in enumerate(powers):
            expected[index, index] += receiver_variance(power)
        np.testing.assert_allclose(result["receiver_covariance_estimate"], expected, rtol=1e-12, atol=1e-18)
        self.assertAlmostEqual(result["metrics"]["noise_rms_estimate"], math.sqrt(max(np.diag(expected))), places=14)
        self.assertAlmostEqual(result["metrics"]["stacked_noise_rms_estimate"], math.sqrt(np.trace(expected)), places=14)

    def test_four_way_tree_pays_three_splitters_and_two_levels_of_loss(self):
        result = evaluate_implementation(self.graph(count=4), self.tech)
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertEqual(result["instance_counts"]["splitter"], 3)
        np.testing.assert_allclose([r["full_scale_power_mw"] for r in result["receivers"]], [passive_power(1.0, 2, 0.0)] * 4, rtol=1e-13)
        self.assertAlmostEqual(result["metrics"]["energy_pj"], 5.4, places=12)
        self.assertAlmostEqual(result["metrics"]["area_um2"], 1460.0, places=12)
        self.assertEqual(result["terminations"], [])

    def test_three_way_tree_terminates_fourth_leaf_without_redistribution(self):
        result = evaluate_implementation(self.graph(count=3), self.tech)
        self.assertTrue(result["feasible"], result["diagnostics"])
        leaf_power = passive_power(1.0, 2, 0.0)
        self.assertEqual(result["instance_counts"]["splitter"], 3)
        self.assertEqual(len(result["terminations"]), 1)
        self.assertAlmostEqual(result["metrics"]["terminated_power_mw"], leaf_power, places=13)
        self.assertAlmostEqual(sum(r["full_scale_power_mw"] for r in result["receivers"]) + result["metrics"]["terminated_power_mw"], 10.0 ** (-0.4 / 10.0), places=13)
        self.assertAlmostEqual(result["metrics"]["energy_pj"], 5.3, places=12)

    def test_regeneration_accounts_for_every_carrier_receiver_and_acquisition(self):
        lengths = [100.0, 200.0]
        result = evaluate_implementation(self.graph(recipe="regenerate", regeneration_power_mw=0.6, lengths_um=lengths), self.tech)
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertEqual(result["instance_counts"], {"source": 3, "splitter": 1, "waveguide": 2, "detector": 4, "modulator": 2})
        # Launch source, two fresh carriers, four receivers, two modulators.
        self.assertAlmostEqual(result["metrics"]["energy_pj"], (1.0 + 2 * 0.6) / 0.2 + 4 * 0.1 + 2 * 0.1, places=12)
        self.assertAlmostEqual(result["metrics"]["area_um2"], 3 * 1000.0 + 20.0 + 4 * 100.0 + 2 * 100.0 + sum(lengths) * 0.5, places=12)
        self.assertAlmostEqual(result["metrics"]["latency_ns"], 2.0 + 0.1 + 0.2 + 0.2 + 0.2 + propagation_delay(200.0), places=12)
        final_powers = [0.6 * 10.0 ** (-1.0 / 10.0) * 10.0 ** (-0.0002 * length / 10.0) for length in lengths]
        np.testing.assert_allclose([r["full_scale_power_mw"] for r in result["receivers"]], final_powers, rtol=1e-13)
        expected = np.full((2, 2), 0.001 ** 2)
        early_variance = receiver_variance(passive_power(1.0, 1, 0.0))
        for index, power in enumerate(final_powers):
            expected[index, index] += early_variance + 0.001 ** 2 + 0.002 ** 2 + receiver_variance(power)
        np.testing.assert_allclose(result["receiver_covariance_estimate"], expected, rtol=1e-12, atol=1e-18)

    def test_two_lanes_repeated_operands_and_output_sink_are_all_bound(self):
        graph = build_implementation(vector_repeated_operand(), self.tech,
                                     {"source_node": "x", "source_power_mw": 1.0, "lengths_um": [100.0, 200.0, 300.0]})
        result = evaluate_implementation(graph, self.tech)
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertEqual(result["instance_counts"], {"source": 2, "splitter": 6, "waveguide": 6, "detector": 6})
        self.assertEqual(len(graph["terminations"]), 2)
        observed = [(r["lane"], r["use"]["target"], r["use"]["index"], r["use"]["output"]) for r in result["receivers"]]
        self.assertEqual(observed, [(lane, target, index, output)
                                    for lane in (0, 1)
                                    for target, index, output in (("sum", 0, False), ("sum", 1, False), ("@output:1", 1, True))])
        self.assertAlmostEqual(result["metrics"]["energy_pj"], 10.6, places=12)
        covariance = np.asarray(result["receiver_covariance_estimate"])
        np.testing.assert_array_equal(covariance[:3, 3:], np.zeros((3, 3)))
        self.assertAlmostEqual(covariance[0, 1], 0.001 ** 2, places=16)
        self.assertAlmostEqual(covariance[3, 4], 0.001 ** 2, places=16)

    def test_more_source_power_pays_energy_reduces_receiver_noise_but_not_rin(self):
        low = evaluate_implementation(self.graph(source_power_mw=0.5), self.tech)
        high = evaluate_implementation(self.graph(source_power_mw=1.0), self.tech)
        self.assertTrue(low["feasible"] and high["feasible"])
        self.assertAlmostEqual(high["metrics"]["energy_pj"] - low["metrics"]["energy_pj"], 2.5, places=12)
        self.assertLess(high["metrics"]["noise_rms_estimate"], low["metrics"]["noise_rms_estimate"])
        self.assertAlmostEqual(high["receiver_covariance_estimate"][0][1], low["receiver_covariance_estimate"][0][1], places=16)

    def test_low_receiver_power_is_infeasible_with_no_partial_aggregate_metrics(self):
        graph = self.graph(count=4, source_power_mw=0.01)
        self.assertEqual(validate_implementation(graph, self.tech), graph)
        result = evaluate_implementation(graph, self.tech)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["metrics"])
        self.assertIsNone(result["receiver_covariance_estimate"])
        self.assertTrue(any("sensitivity" in d["message"] for d in result["diagnostics"]))

    def test_operating_power_path_and_temperature_limits_are_enforced(self):
        for changes in ({"source_power_mw": 11.0}, {"source_power_mw": 3.0},
                        {"lengths_um": 10001.0}, {"operating": {"temperature_c": 35.0}}):
            with self.subTest(changes=changes):
                result = evaluate_implementation(self.graph(**changes), self.tech)
                self.assertFalse(result["feasible"])
                self.assertIsNone(result["metrics"])
                self.assertTrue(result["diagnostics"])

    def test_portable_record_copies_inputs_and_hashes_parameter_changes(self):
        net = scalar_fanout(2)
        spec = {"source_node": "x", "source_power_mw": 1.0, "lengths_um": 100.0}
        record = realize(net, self.tech, spec)
        same = realize(deepcopy(net), deepcopy(self.tech), deepcopy(spec))
        self.assertEqual(record, same)
        changed = realize(net, self.tech, {**spec, "source_power_mw": 0.5})
        self.assertEqual(record["hashes"]["network"], changed["hashes"]["network"])
        self.assertNotEqual(record["hashes"]["spec"], changed["hashes"]["spec"])
        self.assertNotEqual(record["hashes"]["graph"], changed["hashes"]["graph"])
        net["outputs"].clear()
        spec["source_power_mw"] = 9.0
        self.tech["components"][0]["costs"]["area_um2"] = 0.0
        self.assertEqual(record["network"]["outputs"], ["x", "x"])
        self.assertEqual(record["spec"]["source_power_mw"], 1.0)
        self.assertEqual(record["technology"]["components"][0]["costs"]["area_um2"], 1000.0)

    def test_portable_json_roundtrip_replays_and_metric_tampering_fails(self):
        record = realize(scalar_fanout(2), self.tech,
                         {"source_node": "x", "source_power_mw": 1.0, "lengths_um": 100.0})
        record = json.loads(json.dumps(record, allow_nan=False))
        self.assertTrue(check_realization(record)["valid"])
        record["evaluation"]["metrics"]["energy_pj"] = 0.0
        checked = check_realization(record)
        self.assertFalse(checked["valid"])
        self.assertTrue(any(d["path"] == "evaluation" for d in checked["diagnostics"]))

    def test_consistently_rehashed_graph_still_must_match_declared_spec(self):
        record = realize(scalar_fanout(2), self.tech,
                         {"source_node": "x", "source_power_mw": 1.0, "lengths_um": 100.0})
        launch = next(i for i in record["graph"]["instances"] if i["role"] == "encoded_launch")
        launch["parameters"]["power_mw"] = 0.5
        record["hashes"]["graph"] = hashlib.sha256(json.dumps(record["graph"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        record["evaluation"] = evaluate_implementation(record["graph"], record["technology"])
        self.assertTrue(record["evaluation"]["feasible"])
        self.assertFalse(check_realization(record)["valid"])

    def test_accumulated_finite_noise_overflow_is_structured_infeasibility(self):
        # Two individually finite variance contributions overflow only after
        # graph composition. No numerical infinity may become a valid result.
        for component in self.tech["components"]:
            if component["kind"] == "source":
                component["parameters"]["relative_intensity_noise_rms"] = 1e154
            elif component["kind"] == "detector":
                component["parameters"]["input_referred_noise_mw_rms"] = 5e153
        result = evaluate_implementation(self.graph(), self.tech)
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["metrics"])
        self.assertIn("numeric_range_exceeded", {d["code"] for d in result["diagnostics"]})
        json.dumps(result, allow_nan=False)


class DistributionValidationTests(unittest.TestCase):
    def setUp(self):
        self.tech = load_technology()
        self.spec = {"source_node": "x", "source_power_mw": 1.0, "lengths_um": 100.0}
        self.graph = build_implementation(vector_repeated_operand(), self.tech, self.spec)

    def assert_invalid_graph(self, graph, code=None):
        with self.assertRaises(InputValidationError) as caught:
            validate_implementation(graph, self.tech)
        if code is not None:
            self.assertIn(code, {d["code"] for d in caught.exception.diagnostics})

    def test_dangling_input_and_unterminated_output_reject(self):
        graph = deepcopy(self.graph)
        graph["connections"].pop()
        self.assert_invalid_graph(graph)
        graph = deepcopy(self.graph)
        graph["terminations"].pop()
        self.assert_invalid_graph(graph, "unconnected_output")

    def test_implicit_output_copy_and_input_multidrive_reject(self):
        for field in ("source", "target"):
            graph = deepcopy(self.graph)
            graph["connections"][1][field] = deepcopy(graph["connections"][0][field])
            with self.subTest(field=field):
                self.assert_invalid_graph(graph, "port_connection_count")

    def test_unknown_port_and_electrical_to_optical_connection_reject(self):
        graph = deepcopy(self.graph)
        graph["connections"][0]["source"]["port"] = "missing"
        self.assert_invalid_graph(graph, "unknown_port")
        graph = deepcopy(self.graph)
        graph["connections"][0]["source"] = deepcopy(graph["receivers"][0]["port"])
        self.assert_invalid_graph(graph, "port_type_mismatch")

    def test_receiver_lane_swapping_cannot_relabel_physical_signals(self):
        graph = deepcopy(self.graph)
        graph["receivers"][0]["lane"], graph["receivers"][3]["lane"] = 1, 0
        self.assert_invalid_graph(graph, "receiver_lineage_mismatch")

    def test_missing_duplicate_or_wrong_nn_use_reject(self):
        for edit in ("missing", "duplicate", "wrong_source", "wrong_use"):
            graph = deepcopy(self.graph)
            if edit == "missing":
                graph["receivers"].pop()
            elif edit == "duplicate":
                graph["receivers"][1]["use_index"] = 0
            elif edit == "wrong_source":
                graph["receivers"][0]["source_node"] = "sum"
            else:
                graph["receivers"][0]["use_index"] = 10
            with self.subTest(edit=edit):
                self.assert_invalid_graph(graph)

    def test_complete_but_cyclic_subcircuit_rejects(self):
        graph = build_implementation(scalar_fanout(2), self.tech, self.spec)
        # Rewire a guide as a self-loop and bypass it to its receiver. All ports
        # remain occupied exactly once; only cycle validation can reject this.
        route_in = next(e for e in graph["connections"] if e["target"]["instance"].endswith("use0.route"))
        route_out = next(e for e in graph["connections"] if e["source"]["instance"].endswith("use0.route"))
        route_in["source"], route_out["source"] = deepcopy(route_out["source"]), deepcopy(route_in["source"])
        self.assert_invalid_graph(graph, "physical_cycle")

    def test_regeneration_requires_both_independent_carrier_and_sample(self):
        graph = build_implementation(scalar_fanout(2), self.tech, {**self.spec, "recipe": "regenerate", "regeneration_power_mw": 0.6})
        missing = deepcopy(graph)
        missing["connections"] = [e for e in missing["connections"] if not (e["target"]["instance"].endswith("use0.modulator") and e["target"]["port"] == "carrier")]
        self.assert_invalid_graph(missing)
        # Swap the encoded splitter branch with the constant carrier. This has
        # valid optical ports, but a constant is not an NN sample and x is not a carrier.
        early = next(e for e in graph["connections"] if e["target"]["instance"].endswith("use0.early_detector"))
        carrier = next(e for e in graph["connections"] if e["target"]["instance"].endswith("use0.modulator") and e["target"]["port"] == "carrier")
        early["source"], carrier["source"] = deepcopy(carrier["source"]), deepcopy(early["source"])
        self.assert_invalid_graph(graph)

    def test_recipe_label_cannot_hide_regeneration(self):
        graph = build_implementation(scalar_fanout(2), self.tech, {**self.spec, "recipe": "regenerate", "regeneration_power_mw": 0.6})
        graph["recipe"] = "passive"
        self.assert_invalid_graph(graph)

    def test_signed_or_overrange_network_values_are_not_implicitly_rescaled(self):
        for lower, upper in ((-0.01, 1.0), (0.0, 1.01)):
            net = scalar_fanout(2)
            net["nodes"][0]["input_bounds"] = {"lower": [lower], "upper": [upper]}
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(InputValidationError) as caught:
                    build_implementation(net, self.tech, self.spec)
                self.assertEqual(caught.exception.diagnostics[0]["code"], "unsupported_encoding_range")

    def test_strict_recipe_options_and_route_count(self):
        for changes in ({"recipe": "amplify"}, {"recipe": "regenerate"},
                        {"regeneration_power_mw": 0.5}, {"source_power_mw": True},
                        {"source_power_mw": float("nan")}, {"lengths_um": -1.0},
                        {"unknown_setting": 3}):
            with self.subTest(changes=changes):
                with self.assertRaises(InputValidationError):
                    validate_realization_spec({**self.spec, **changes})
        with self.assertRaises(InputValidationError) as caught:
            build_implementation(scalar_fanout(3), self.tech, {**self.spec, "lengths_um": [0.0, 1.0]})
        self.assertEqual(caught.exception.diagnostics[0]["code"], "route_count_mismatch")


if __name__ == "__main__":
    unittest.main()
