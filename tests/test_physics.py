import unittest

import numpy as np

from npp.models import validate_inputs
from npp.physics import evaluate, infer_bounds, outgoing_uses, simulate
from tests.fixtures import affine_fixture, component, library, network, node, request, rule


class AnalyticalPhysicsTests(unittest.TestCase):
    def test_shared_and_independent_noise_covariance(self):
        net, lib, req, decision = affine_fixture()
        net, lib, req = validate_inputs(net, lib, req)
        result = evaluate(net, lib, req, decision)
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertEqual(result["analysis"]["kind"], "exact_affine_gaussian")
        # e_left = z + a, e_right = 2*z + b, independently distributed
        # z ~ N(0,.2²), a ~ N(0,.3²), b ~ N(0,.4²).
        expected = np.array([[0.2**2 + 0.3**2, 2*0.2**2],
                             [2*0.2**2, 4*0.2**2 + 0.4**2]])
        np.testing.assert_allclose(result["analysis"]["output_covariance"], expected,
                                   rtol=1e-12, atol=1e-14)

    def test_forward_sampling_matches_analytic_mse_with_reported_uncertainty(self):
        net, lib, req, decision = affine_fixture()
        net, lib, req = validate_inputs(net, lib, req)
        result = simulate(net, lib, req, decision, samples=30000, seed=218)
        # From independent scalar propagation, total expected squared L2
        # error is .13+.32=.45, including the correlated component noise.
        expected_mse = 0.45
        self.assertAlmostEqual(result["affine_expected_mse"], expected_mse, places=12)
        self.assertGreater(result["mse_standard_error"], 0.0)
        self.assertLess(abs(result["mse_empirical"] - expected_mse),
                        6 * result["mse_standard_error"])

    def test_reconvergence_cancels_shared_noise(self):
        net = network([node("x", "input"), node("p", "linear", ["x"]),
                       node("m", "linear", ["x"], weights=[[-1.0]]),
                       node("sum", "add", ["p", "m"])], ["sum"])
        lib = library([component("encoder", ["input"], noise_std=0.7),
                       component("arithmetic", ["linear", "add"])])
        net, lib, req = validate_inputs(net, lib, request())
        decision = {"components": {"x": "encoder", "p": "arithmetic",
                                    "m": "arithmetic", "sum": "arithmetic"},
                    "fanout_rules": {"x": "copy"}}
        result = evaluate(net, lib, req, decision)
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertAlmostEqual(result["metrics"]["error_rms_bound"], 0.0, places=12)
        np.testing.assert_allclose(result["analysis"]["output_covariance"], [[0.0]], atol=1e-14)

    def test_repeated_operand_and_output_sink_count_as_three_uses(self):
        net = network([node("x", "input"), node("sum", "add", ["x", "x"])], ["sum", "x"])
        uses = outgoing_uses(net)["x"]
        self.assertEqual(len(uses), 3)
        self.assertEqual([(u["target"], u["index"]) for u in uses[:2]], [("sum", 0), ("sum", 1)])
        self.assertEqual(uses[2]["target"], "@output:1")

    def test_signed_affine_interval_propagation(self):
        net = network([
            node("x", "input", size=2, input_bounds={"lower": [-2., 1.], "upper": [3., 4.]}),
            node("y", "linear", ["x"], weights=[[2., -3.]], bias=[1.]),
        ], ["y"])
        bounds = infer_bounds(net)["y"]
        self.assertEqual(bounds["lower"], [-15.0])
        self.assertEqual(bounds["upper"], [4.0])

    def test_nominal_range_and_budget_reject_candidate(self):
        net, lib, req, decision = affine_fixture()
        lib["components"][2]["max_abs_value"] = 1.5
        net, lib, req = validate_inputs(net, lib, req)
        result = evaluate(net, lib, req, decision)
        self.assertFalse(result["feasible"])
        self.assertTrue(result["diagnostics"])
        net, lib, req, decision = affine_fixture()
        req["budgets"] = {"error_rms_bound": 0.01}
        net, lib, req = validate_inputs(net, lib, req)
        result = evaluate(net, lib, req, decision)
        self.assertFalse(result["feasible"])
        self.assertTrue(result["diagnostics"])

    def test_relu_error_bound_remains_conservative(self):
        net = network([node("x", "input", input_bounds={"lower": [0.], "upper": [0.]}),
                       node("y", "relu", ["x"])], ["y"])
        lib = library([component("encoder", ["input"], noise_std=0.3),
                       component("activation", ["relu"])])
        net, lib, req = validate_inputs(net, lib, request())
        result = evaluate(net, lib, req, {"components": {"x": "encoder", "y": "activation"},
                                         "fanout_rules": {}})
        self.assertTrue(result["feasible"], result["diagnostics"])
        self.assertEqual(result["analysis"]["kind"], "conservative_lipschitz_rms")
        # At x=0, max(0,Z) has RMS sigma/sqrt(2). A valid bound must
        # cover this despite the zero nominal activation.
        self.assertGreaterEqual(result["metrics"]["error_rms_bound"], 0.3 / np.sqrt(2))
        self.assertLessEqual(result["metrics"]["error_rms_bound"], 0.3 + 1e-12)


class OpticalPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.net = network([node("x", "input"), node("left", "identity", ["x"]),
                            node("right", "identity", ["x"])], ["left", "right"])
        self.lib = library([
            component("source", ["input"], domain="optical", source_scalable=True,
                      noise_std=0.1, photons=100., optical_efficiency=0.5, latency_ns=2.),
            component("digital", ["identity", "linear"]),
        ], [rule("passive", "passive_split"),
            rule("boost", "source_boost", gain=2., energy_pj=0.25),
            rule("amplify", "amplify", gain=2., added_noise_std=0.2, energy_pj=0.25),
            rule("regenerate", "regenerate", gain=2., added_noise_std=0.2, energy_pj=0.25),
            rule("serial", "serial_reencode", latency_ns=0.5, energy_pj=0.25),
            rule("sensitivity", "passive_split", split_policy="sensitivity")])
        self.req = request()
        self.decision = {"components": {"x": "source", "left": "digital", "right": "digital"},
                         "fanout_rules": {"x": "passive"}}
        self.photon_energy = 100 * 6.62607015e-34 * 299792458.0 / (1550e-9) * 1e12

    def evaluate_rule(self, rule_id):
        self.decision["fanout_rules"]["x"] = rule_id
        result = evaluate(*validate_inputs(self.net, self.lib, self.req), self.decision)
        self.assertTrue(result["feasible"], result["diagnostics"])
        return result

    def test_source_boost_pays_energy_and_reduces_only_branch_shot_noise(self):
        passive = self.evaluate_rule("passive")
        boost = self.evaluate_rule("boost")
        # Each branch receives P/2 photons, with eta=.5: shot variance=.04.
        # Gain 2 halves that term; inherited source variance .01 remains.
        np.testing.assert_allclose(passive["analysis"]["output_covariance"],
                                   [[0.05, 0.01], [0.01, 0.05]], atol=1e-14)
        np.testing.assert_allclose(boost["analysis"]["output_covariance"],
                                   [[0.03, 0.01], [0.01, 0.03]], atol=1e-14)
        self.assertAlmostEqual(passive["metrics"]["energy_pj"], self.photon_energy, places=12)
        self.assertAlmostEqual(boost["metrics"]["energy_pj"],
                               2*self.photon_energy + 0.25, places=12)

    def test_amplifier_noise_is_shared_and_can_outweigh_shot_improvement(self):
        result = self.evaluate_rule("amplify")
        # Inherited .01 + amplifier .04 is common; delivered shot .02 is independent.
        np.testing.assert_allclose(result["analysis"]["output_covariance"],
                                   [[0.07, 0.05], [0.05, 0.07]], atol=1e-14)
        passive = self.evaluate_rule("passive")
        self.assertGreater(result["metrics"]["error_rms_bound"],
                           passive["metrics"]["error_rms_bound"])

    def test_regeneration_keeps_inherited_and_shared_measurement_error(self):
        result = self.evaluate_rule("regenerate")
        # Common .01 + .04 + measurement shot .02 = .07.
        # Independent re-encode .04 + full-power gained shot .01 = .05.
        np.testing.assert_allclose(result["analysis"]["output_covariance"],
                                   [[0.12, 0.07], [0.07, 0.12]], atol=1e-14)
        self.assertAlmostEqual(result["metrics"]["energy_pj"],
                               5*self.photon_energy + 0.25, places=12)

    def test_intermediate_source_boost_is_rejected(self):
        net = network([node("x", "input"), node("middle", "identity", ["x"]),
                       node("left", "identity", ["middle"]),
                       node("right", "identity", ["middle"])], ["left", "right"])
        lib = library([component("encoder", ["input"]),
                       component("optical", ["identity"], domain="optical"),
                       component("digital", ["identity"])],
                      [rule("boost", "source_boost", gain=2.)])
        decision = {"components": {"x": "encoder", "middle": "optical", "left": "digital", "right": "digital"},
                    "fanout_rules": {"middle": "boost"}}
        result = evaluate(*validate_inputs(net, lib, self.req), decision)
        self.assertFalse(result["feasible"])
        self.assertIn("source_scaling_forbidden", {d["code"] for d in result["diagnostics"]})

    def test_boost_respects_physical_headroom(self):
        self.lib["components"][0]["max_abs_value"] = 1.0
        self.evaluate_rule("passive")
        self.decision["fanout_rules"]["x"] = "boost"
        result = evaluate(*validate_inputs(self.net, self.lib, self.req), self.decision)
        self.assertFalse(result["feasible"])
        self.assertIn("optical_headroom_exceeded", {d["code"] for d in result["diagnostics"]})

    def test_serial_encoding_requires_permission_and_pays_two_operations(self):
        self.decision["fanout_rules"]["x"] = "serial"
        result = evaluate(*validate_inputs(self.net, self.lib, self.req), self.decision)
        self.assertFalse(result["feasible"])
        self.assertIn("serialization_forbidden", {d["code"] for d in result["diagnostics"]})
        self.req["allow_serialization"] = True
        result = self.evaluate_rule("serial")
        np.testing.assert_allclose(result["analysis"]["output_covariance"],
                                   [[0.03, 0.0], [0.0, 0.03]], atol=1e-14)
        self.assertAlmostEqual(result["metrics"]["latency_ns"], 5.0, places=12)
        self.assertAlmostEqual(result["metrics"]["energy_pj"],
                               2*self.photon_energy + 0.5, places=12)

    def test_sensitivity_split_allocates_power_to_high_gain_branch(self):
        self.net["nodes"][1] = node("left", "linear", ["x"])
        self.net["nodes"][2] = node("right", "linear", ["x"], weights=[[4.]])
        self.lib["components"][0]["noise_std"] = 0.0
        passive = self.evaluate_rule("passive")
        sensitivity = self.evaluate_rule("sensitivity")
        # Optimal scalar allocation is f=(1/5,4/5), so variances after
        # weights (1,4) are (.1,.4), versus (.04,.64) for equal splitting.
        np.testing.assert_allclose(sensitivity["analysis"]["output_covariance"],
                                   [[0.1, 0.0], [0.0, 0.4]], atol=1e-14)
        self.assertLess(sensitivity["metrics"]["error_rms_bound"],
                        passive["metrics"]["error_rms_bound"])


if __name__ == "__main__":
    unittest.main()
