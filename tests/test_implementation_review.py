"""Independent gate-2 adversarial and conservation checks.

These tests exercise declared physical contracts through different graph shapes
and mutations, rather than recalculate the implementation's own ledger.
"""
from copy import deepcopy
import json
import math
import unittest

import numpy as np

from npp.implementation import (build_implementation, check_realization,
                                evaluate_implementation, realize,
                                validate_implementation)
from npp.models import InputValidationError
from npp.technology import load_technology


def scalar_network(count=2):
    return {"name": "Independent review network", "nodes": [
        {"id": "x", "op": "input", "size": 1,
         "input_bounds": {"lower": [0.0], "upper": [1.0]}}
    ], "outputs": ["x"] * count}


def spec(**changes):
    return {"source_node": "x", "source_power_mw": 1.0,
            "lengths_um": 0.0, **changes}


class ImplementationReviewTests(unittest.TestCase):
    def setUp(self):
        self.technology = load_technology()

    def test_eight_way_unequal_paths_conserve_power_with_explicit_loss(self):
        lengths = [0., 100., 250., 500., 1000., 2500., 5000., 10000.]
        record = realize(scalar_network(8), self.technology, spec(lengths_um=lengths))
        self.assertTrue(record["evaluation"]["feasible"])
        rows = record["evaluation"]["receivers"]
        splitter_loss = 10 ** (-3 * 0.2 / 10)
        expected = [splitter_loss / 8 * 10 ** (-length * .0002 / 10) for length in lengths]
        np.testing.assert_allclose([r["full_scale_power_mw"] for r in rows], expected, rtol=1e-13)
        self.assertEqual(record["evaluation"]["instance_counts"]["splitter"], 7)
        # Recover power at the branch inputs from independently stated losses.
        recovered = sum(r["full_scale_power_mw"] * 10 ** (length * .0002 / 10)
                        for r, length in zip(rows, lengths))
        self.assertAlmostEqual(recovered, splitter_loss, places=13)
        self.assertLess(sum(r["full_scale_power_mw"] for r in rows), 1.)
        np.testing.assert_allclose(np.array(record["evaluation"]["receiver_covariance_estimate"])[0, 1:],
                                   np.full(7, 1e-6), rtol=1e-13)

    def test_non_power_two_fanout_dumps_all_extra_leaves(self):
        record = realize(scalar_network(5), self.technology, spec())
        result = record["evaluation"]
        self.assertTrue(result["feasible"])
        self.assertEqual(len(record["graph"]["terminations"]), 3)
        power_after_tree = 10 ** (-.6 / 10)
        self.assertAlmostEqual(result["metrics"]["terminated_power_mw"], 3/8 * power_after_tree)
        self.assertAlmostEqual(sum(r["full_scale_power_mw"] for r in result["receivers"])
                               + result["metrics"]["terminated_power_mw"], power_after_tree)

    def test_selected_internal_node_can_encode_bounded_affine_of_signed_input(self):
        net = {"name": "Interior-node binding", "nodes": [
            {"id": "z", "op": "input", "size": 1,
             "input_bounds": {"lower": [-1.], "upper": [1.]}},
            {"id": "x", "op": "linear", "size": 1, "inputs": ["z"],
             "weights": [[.5]], "bias": [.5]},
            {"id": "sum", "op": "add", "size": 1, "inputs": ["x", "x"]}
        ], "outputs": ["x", "sum", "x"]}
        record = realize(net, self.technology, spec(lengths_um=[0., 1., 2., 3.]))
        self.assertTrue(record["evaluation"]["feasible"])
        self.assertEqual([row["use"] for row in record["evaluation"]["receivers"]], [
            {"target": "sum", "index": 0, "output": False},
            {"target": "sum", "index": 1, "output": False},
            {"target": "@output:0", "index": 0, "output": True},
            {"target": "@output:2", "index": 2, "output": True}])
        self.assertEqual(record["graph"]["instances"][0]["source_node"], "x")

    def test_regeneration_retains_symbol_duration_and_every_source_cost(self):
        # Nonzero fixed source costs must be charged to launch and each carrier.
        for component in self.technology["components"]:
            if component["kind"] == "source":
                component["costs"]["fixed_energy_pj"] = 1.25
        energies, latencies = [], []
        for duration in (1., 2.):
            result = realize(scalar_network(4), self.technology,
                             spec(recipe="regenerate", regeneration_power_mw=.5,
                                  operating={"symbol_duration_ns": duration}))["evaluation"]
            self.assertTrue(result["feasible"])
            energies.append(result["metrics"]["energy_pj"])
            latencies.append(result["metrics"]["latency_ns"])
        # Source optical energy: (1 + 4*.5) mW, efficiency .2.
        # Fixed costs: five sources, eight receivers, four modulators.
        self.assertAlmostEqual(energies[0], 3/.2 + 5*1.25 + 8*.1 + 4*.1)
        self.assertAlmostEqual(energies[1] - energies[0], 3/.2)
        self.assertAlmostEqual(latencies[1] - latencies[0], 2.)

    def test_shared_carrier_requires_costed_splitter_and_propagates_shared_noise(self):
        graph = build_implementation(scalar_network(), self.technology,
                                     spec(recipe="regenerate", regeneration_power_mw=1.))
        # Use one constant-carrier source through a physical splitter for two
        # modulators. Both samples still require their own early detector.
        graph["instances"] = [i for i in graph["instances"] if i["id"] != "lane0.use1.carrier"]
        graph["instances"].append({"id": "shared_carrier_split", "component": "splitter"})
        for edge in graph["connections"]:
            if edge["target"]["port"] == "carrier":
                index = 0 if edge["target"]["instance"] == "lane0.use0.modulator" else 1
                edge["source"] = {"instance": "shared_carrier_split", "port": f"out{index+1}"}
        graph["connections"].append({"source": {"instance": "lane0.use0.carrier", "port": "out"},
                                     "target": {"instance": "shared_carrier_split", "port": "in"}})
        result = evaluate_implementation(graph, self.technology)
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertEqual(result["instance_counts"]["source"], 2)
        self.assertEqual(result["instance_counts"]["splitter"], 2)
        self.assertAlmostEqual(result["metrics"]["energy_pj"], 2/.2 + 4*.1 + 2*.1)
        self.assertAlmostEqual(result["receiver_covariance_estimate"][0][1], 2e-6)
        expected_power = .5 * 10 ** (-.2/10) * 10 ** (-1./10)
        np.testing.assert_allclose([r["full_scale_power_mw"] for r in result["receivers"]], [expected_power]*2)

    def test_early_receiver_failure_cannot_be_repaired_by_later_regeneration(self):
        result = realize(scalar_network(8), self.technology,
                         spec(source_power_mw=.01, recipe="regenerate",
                              regeneration_power_mw=1.))["evaluation"]
        self.assertFalse(result["feasible"])
        self.assertIsNone(result["metrics"])
        self.assertEqual([r["status"] for r in result["receivers"]], ["blocked"]*8)
        self.assertTrue(any("early_detector" in d["path"] and "sensitivity" in d["message"]
                            for d in result["diagnostics"]))

    def test_all_top_level_record_fields_are_bound_by_reconstruction(self):
        original = realize(scalar_network(), self.technology, spec())
        for field, value in [("schema_version", "99"), ("kind", "different"),
                             ("network", None), ("technology", {}), ("spec", []),
                             ("graph", {}), ("hashes", {}), ("evaluation", {})]:
            with self.subTest(field=field):
                record = deepcopy(original)
                record[field] = value
                self.assertFalse(check_realization(record)["valid"])
        self.assertFalse(check_realization({**original, "extra": 1})["valid"])

    def test_replay_rejects_python_numeric_equality_and_nonfinite_json(self):
        original = realize(scalar_network(), self.technology, spec())
        paths = [("evaluation", "feasible"), ("spec", "source_power_mw"),
                 ("graph", "operating", "symbol_duration_ns")]
        for path in paths:
            record = deepcopy(original)
            target = record
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = 1 if path[-1] == "feasible" else True
            with self.subTest(path=path):
                self.assertFalse(check_realization(record)["valid"])
        for value in (float("nan"), float("inf"), -float("inf"), 10**10000):
            record = deepcopy(original)
            record["evaluation"]["metrics"]["energy_pj"] = value
            self.assertFalse(check_realization(record)["valid"])

    def test_graph_numeric_indices_are_not_booleans_or_strings(self):
        original = build_implementation(scalar_network(), self.technology, spec())
        for field in ("lane", "use_index"):
            for value in (False, "0", -1, 2**200):
                graph = deepcopy(original)
                graph["receivers"][0][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(InputValidationError):
                        validate_implementation(graph, self.technology)

    def test_no_electrical_sample_broadcast_even_with_fresh_optical_carriers(self):
        graph = build_implementation(scalar_network(), self.technology,
                                     spec(recipe="regenerate", regeneration_power_mw=.5))
        edges = [e for e in graph["connections"] if e["target"]["port"] == "sample"]
        edges[1]["source"] = deepcopy(edges[0]["source"])
        with self.assertRaises(InputValidationError) as caught:
            validate_implementation(graph, self.technology)
        self.assertIn("port_connection_count", {d["code"] for d in caught.exception.diagnostics})


if __name__ == "__main__":
    unittest.main()
