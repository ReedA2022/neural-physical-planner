"""Independent shipping counterexamples, separate from campaign generators.

These regressions test public entry points and preserve the independently found
false-zero and nonfinite-pass failures. They do not validate real devices.
"""
import json
import math
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from npp.config import load_inputs
from npp.designer import validate_design_spec
from npp.models import InputValidationError, validate_inputs
from npp.physics import evaluate, simulate
from tests.fixtures import component, library, network, node, request


class ReleaseIndependentReviewTests(unittest.TestCase):
    def nonlinear_fixture(self, weights):
        nodes = [node("x", "input", input_bounds={"lower": [0.], "upper": [0.]})]
        previous = "x"
        for index, weight in enumerate(weights):
            name = f"linear{index}"
            nodes.append(node(name, "linear", [previous], weights=[[weight]]))
            previous = name
        nodes.append(node("z", "relu", [previous]))
        net, lib, req = validate_inputs(network(nodes, ["z"]), library([
            component("source", ["input"], noise_std=1e-100),
            component("compute", ["linear", "relu"]),
        ]), request(budgets={"error_rms_bound": 0.}))
        decision = {"components": {n["id"]: "source" if n["id"] == "x" else "compute" for n in nodes},
                    "fanout_rules": {}}
        return net, lib, req, decision

    def test_nonlinear_positive_rms_survives_aggregation_or_fails_closed(self):
        # The true Lipschitz RMS is 1e-260: its square underflows binary64,
        # although the RMS itself is representable. A zero budget cannot pass.
        result = evaluate(*self.nonlinear_fixture([1e-160]))
        self.assertFalse(result["feasible"])
        if result["metrics"]:
            self.assertGreater(result["metrics"]["error_rms_bound"], 0)
            self.assertTrue(math.isclose(result["metrics"]["error_rms_bound"], 1e-260,
                                         rel_tol=1e-12, abs_tol=0))
        else:
            self.assertTrue(result["diagnostics"])

    def test_nonlinear_chained_attenuation_cannot_erase_positive_noise(self):
        # Python scalar multiplication can underflow even when NumPy's
        # floating-point exception context is active.
        result = evaluate(*self.nonlinear_fixture([1e-160, 1e-100]))
        self.assertFalse(result["feasible"])
        self.assertTrue(result["diagnostics"])

    def test_optical_positive_energy_cannot_pass_zero_budget(self):
        net, lib, req = validate_inputs(network([
            node("x", "input", input_bounds={"lower": [0.], "upper": [0.]})], ["x"]),
            library([component("source", ["input"], domain="optical", photons=1e-300)]),
            request(budgets={"energy_pj": 0.}))
        result = evaluate(net, lib, req, {"components": {"x": "source"}, "fanout_rules": {}})
        self.assertFalse(result["feasible"])
        if result["metrics"]:
            self.assertGreater(result["metrics"]["energy_pj"], 0)
        else:
            self.assertTrue(result["diagnostics"])

    def test_nonzero_subnormal_intermediate_does_not_underestimate_optical_cost(self):
        # The intermediate SI energy rounds to the smallest subnormal value;
        # converting that rounded value to pJ can understate the cost by 33%.
        photons = 3.7e-299
        expected = float(Decimal(str(photons)) * Decimal("6.62607015e-34") *
                         Decimal("299792458") / Decimal("1550e-9") * Decimal("1e12"))
        net, lib, req = validate_inputs(network([
            node("x", "input", input_bounds={"lower": [0.], "upper": [0.]})], ["x"]),
            library([component("source", ["input"], domain="optical", photons=photons)]),
            request(budgets={"energy_pj": 4e-306}))
        self.assertGreater(expected, 4e-306)
        result = evaluate(net, lib, req, {"components": {"x": "source"}, "fanout_rules": {}})
        self.assertFalse(result["feasible"])
        if result["metrics"]:
            self.assertTrue(math.isclose(result["metrics"]["energy_pj"], expected,
                                         rel_tol=1e-12, abs_tol=0))
        else:
            self.assertTrue(result["diagnostics"])

    def test_simulation_cannot_issue_nonfinite_uncertainty_pass(self):
        net, lib, req = validate_inputs(network([
            node("x", "input", input_bounds={"lower": [0.], "upper": [0.]})], ["x"]),
            library([component("source", ["input"], noise_std=1e150)]), request())
        decision = {"components": {"x": "source"}, "fanout_rules": {}}
        self.assertTrue(evaluate(net, lib, req, decision)["feasible"])
        try:
            result = simulate(net, lib, req, decision, samples=20, seed=0)
        except (ValueError, FloatingPointError) as error:
            self.assertTrue(str(error))  # Explicit numeric-range failure is valid.
        else:
            json.dumps(result, allow_nan=False)
            self.assertTrue(math.isfinite(result["mse_standard_error"]))
            self.assertGreater(result["mse_standard_error"], 0)

    def test_nonzero_underflowing_design_quantities_are_rejected(self):
        for value in ("1e-999 um", "-1e-999 um", "1e-999999999 um", "-1e-999999999 um"):
            with self.subTest(value=value), self.assertRaises(InputValidationError):
                validate_design_spec({"source_node": "x", "grid": {
                    "source_powers_mw": [.1], "route_lengths_um": [value]}})

    def test_nonzero_underflowing_hardware_quantities_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "project.json"
            for value in ("1e-999 pJ", "-1e-999 pJ", "1e-999999999 pJ"):
                project = {"name": "Independent review", "model": {
                    "input": {"size": 1, "bounds": [0, 1]}, "layers": [{"type": "identity"}]},
                    "hardware": {"extends": "example", "overrides": {
                        "components": {"optical_fixed": {"energy": value}}}}}
                path.write_text(json.dumps(project))
                with self.subTest(value=value), self.assertRaises(InputValidationError):
                    load_inputs(project_path=path)

    def test_representable_subnormal_and_explicit_zero_design_values_remain_valid(self):
        for value, expected in (("0 um", 0.), ("-0 um", -0.), ("5e-324 um", 5e-324)):
            result = validate_design_spec({"source_node": "x", "grid": {
                "source_powers_mw": [.1], "route_lengths_um": [value]}})
            self.assertEqual(result["grid"]["route_lengths_um"], [expected])


if __name__ == "__main__":
    unittest.main()
