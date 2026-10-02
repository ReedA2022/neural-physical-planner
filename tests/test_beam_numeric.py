"""Regression for overflow in beam ranking of otherwise valid finite inputs."""
import math
import unittest

from npp.models import validate_inputs
from npp.planner import plan, _operator_gain, _sensitivities
from tests.fixtures import component, library, network, node, request


class BeamNumericTests(unittest.TestCase):
    def test_large_finite_weights_do_not_overflow_beam_surrogates(self):
        net = network([
            node("x", "input", input_bounds={"lower": [0.0], "upper": [0.0]}),
            node("y", "linear", ["x"], weights=[[1e200]]),
            node("z", "linear", ["y"], weights=[[1e200]]),
        ], ["z"])
        lib = library([
            component("encoder", ["input"]),
            component("linear_a", ["linear"]),
            component("linear_b", ["linear"], energy_pj=1.0),
        ])
        req = request(search={"mode": "beam", "max_evaluations": 10, "beam_width": 1})
        net, lib, req = validate_inputs(net, lib, req)
        self.assertEqual(_operator_gain(net["nodes"][1]), 1e200)
        self.assertTrue(all(math.isfinite(value) for value in _sensitivities(net).values()))
        beam = plan(net, lib, req)
        self.assertEqual(beam["status"], "ok")
        self.assertFalse(beam["search"]["complete"])
        self.assertEqual(beam["search"]["evaluated"], 1)
        self.assertTrue(all(math.isfinite(value) for value in beam["plans"][0]["evaluation"]["metrics"].values()))
        req["search"]["mode"] = "exhaustive"
        exhaustive = plan(net, lib, req)
        self.assertEqual(beam["plans"][0]["decision"], exhaustive["plans"][0]["decision"])


if __name__ == "__main__":
    unittest.main()
