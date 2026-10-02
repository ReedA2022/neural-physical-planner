from copy import deepcopy
import unittest

from npp.checker import check_record
from npp.models import validate_inputs
from npp.physics import evaluate
from npp.planner import input_hash, plan
from tests.fixtures import search_fixture


class ReplayCheckerTests(unittest.TestCase):
    def setUp(self):
        self.net, self.lib, self.req = validate_inputs(*search_fixture())
        self.record = deepcopy(plan(self.net, self.lib, self.req)["plans"][0])

    def check(self):
        return check_record(self.net, self.lib, self.req, self.record)

    def test_original_generated_plan_replays(self):
        result = self.check()
        self.assertTrue(result["valid"], result["diagnostics"])
        self.assertTrue(result["recomputed_evaluation"]["feasible"])

    def test_modified_cost_is_detected(self):
        self.record["evaluation"]["metrics"]["energy_pj"] += 1
        result = self.check()
        self.assertFalse(result["valid"])
        self.assertTrue(any("metrics.energy_pj" in d["path"] for d in result["diagnostics"]))

    def test_modified_circuit_is_detected_even_with_unchanged_metrics(self):
        self.record["evaluation"]["hardware"]["connections"][0]["target"] = "made_up_component"
        result = self.check()
        self.assertFalse(result["valid"])
        self.assertTrue(any("hardware.connections" in d["path"] for d in result["diagnostics"]))

    def test_modified_input_weights_are_detected_by_snapshot_hash(self):
        self.net["nodes"][1]["weights"] = [[3.0]]
        result = self.check()
        self.assertFalse(result["valid"])
        self.assertIn("CHECK_INPUT_HASH", {d["code"] for d in result["diagnostics"]})

    def test_invalid_component_is_rejected_even_with_updated_decision_id(self):
        self.record["decision"]["components"]["y"] = "nonexistent_macro"
        self.record["id"] = "plan-" + input_hash(self.record["decision"])
        result = self.check()
        self.assertFalse(result["valid"])
        self.assertFalse(result["recomputed_evaluation"]["feasible"])

    def test_correctly_cached_infeasible_decision_is_not_a_valid_plan(self):
        self.req["budgets"] = {"energy_pj": 0.0}
        self.net, self.lib, self.req = validate_inputs(self.net, self.lib, self.req)
        self.record["input_hashes"] = {"network": input_hash(self.net),
                                        "library": input_hash(self.lib), "request": input_hash(self.req)}
        self.record["evaluation"] = evaluate(self.net, self.lib, self.req, self.record["decision"])
        result = self.check()
        self.assertFalse(result["valid"])
        self.assertIn("CHECK_INFEASIBLE", {d["code"] for d in result["diagnostics"]})

    def test_missing_analysis_and_nonfinite_cache_are_rejected(self):
        del self.record["evaluation"]["analysis"]
        self.record["evaluation"]["metrics"]["energy_pj"] = float("nan")
        result = self.check()
        self.assertFalse(result["valid"])
        self.assertTrue(result["diagnostics"])


if __name__ == "__main__":
    unittest.main()
