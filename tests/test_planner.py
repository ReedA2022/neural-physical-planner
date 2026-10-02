import unittest

from npp.models import validate_inputs
from npp.planner import pareto_filter, plan
from tests.fixtures import search_fixture


class PlannerTests(unittest.TestCase):
    def test_exhaustive_front_matches_independent_tiny_enumeration(self):
        net, lib, req = validate_inputs(*search_fixture())
        report = plan(net, lib, req)
        self.assertEqual(report["status"], "ok")
        actual = {p["decision"]["components"]["y"] for p in report["plans"]}
        # This oracle enumerates the explicit resource numbers directly,
        # without invoking evaluate or the planner's dominance helper.
        candidates = {"fast": (9, 1, 4), "small": (1, 9, 1), "dominated": (10, 10, 5)}
        expected = {a for a in candidates if not any(
            all(x <= y for x, y in zip(candidates[b], candidates[a]))
            and any(x < y for x, y in zip(candidates[b], candidates[a]))
            for b in candidates if a != b)}
        self.assertEqual(actual, expected)

    def test_truncated_no_feasible_search_does_not_prove_infeasibility(self):
        net, lib, req = search_fixture()
        # Every evaluated candidate will fail this budget, but truncation
        # still must not be reported as a proof of infeasibility.
        req["budgets"] = {"energy_pj": 0.0}
        req["search"]["max_evaluations"] = 1
        net, lib, req = validate_inputs(net, lib, req)
        report = plan(net, lib, req)
        self.assertEqual(report["status"], "search_exhausted")
        self.assertEqual(report["plans"], [])

    def test_complete_infeasible_search_can_report_infeasible(self):
        net, lib, req = search_fixture()
        req["budgets"] = {"energy_pj": 0.0}
        report = plan(*validate_inputs(net, lib, req))
        self.assertEqual(report["status"], "infeasible")
        self.assertEqual(report["plans"], [])

    def test_beam_search_is_not_called_globally_infeasible(self):
        net, lib, req = search_fixture()
        req["search"] = {"mode": "beam", "max_evaluations": 1, "beam_width": 1}
        req["budgets"] = {"energy_pj": 0.0}
        report = plan(*validate_inputs(net, lib, req))
        self.assertEqual(report["status"], "search_exhausted")

    def test_frontier_filter_retains_nonconvex_tradeoff(self):
        records = [
            {"id": str(i), "evaluation": {"feasible": True,
                "metrics": {"energy_pj": e, "latency_ns": t}}}
            for i, (e, t) in enumerate([(0, 10), (6, 6), (10, 0), (8, 8)])
        ]
        frontier = pareto_filter(records, ["energy_pj", "latency_ns"])
        # Middle point is nondominated but is unsupported by positive
        # weighted sums. Pareto filtering must preserve it.
        self.assertEqual({p["id"] for p in frontier}, {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
