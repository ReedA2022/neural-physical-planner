import copy
import unittest

from npp.models import InputValidationError, validate_inputs
from tests.fixtures import affine_fixture, component, library, network, node, request


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.net, self.lib, self.req, _ = affine_fixture()

    def assert_invalid(self):
        with self.assertRaises(InputValidationError) as caught:
            validate_inputs(self.net, self.lib, self.req)
        self.assertTrue(caught.exception.diagnostics)
        for diagnostic in caught.exception.diagnostics:
            self.assertIn("code", diagnostic)
            self.assertIn("path", diagnostic)
            self.assertIn("message", diagnostic)

    def test_duplicate_node_ids_rejected(self):
        self.net["nodes"][2]["id"] = "left"
        self.assert_invalid()

    def test_unknown_reference_rejected(self):
        self.net["nodes"][1]["inputs"] = ["missing"]
        self.assert_invalid()

    def test_wrong_bias_shape_rejected(self):
        self.net["nodes"][1]["bias"] = [0.0, 0.0]
        self.assert_invalid()

    def test_nonfinite_weights_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=bad):
                self.net["nodes"][1]["weights"] = [[bad]]
                self.assert_invalid()

    def test_negative_physical_cost_rejected(self):
        self.lib["components"][0]["energy_pj"] = -0.1
        self.assert_invalid()

    def test_invalid_input_bounds_rejected(self):
        self.net["nodes"][0]["input_bounds"] = {"lower": [2.0], "upper": [1.0]}
        self.assert_invalid()

    def test_unknown_fields_and_unsupported_random_access_are_not_ignored(self):
        for key in ("random_access", "require_random_access", "typo_latency_budget"):
            with self.subTest(key=key):
                self.req[key] = True
                self.assert_invalid()
                del self.req[key]

    def test_repeated_inputs_are_valid_separate_uses(self):
        net = network([node("x", "input"), node("sum", "add", ["x", "x"])], ["sum"])
        lib = library([component("encoder", ["input"]), component("adder", ["add"])])
        normalized, _, _ = validate_inputs(net, lib, request())
        self.assertEqual(normalized["nodes"][1]["inputs"], ["x", "x"])

    def test_validation_does_not_mutate_caller_inputs(self):
        before = copy.deepcopy((self.net, self.lib, self.req))
        validate_inputs(self.net, self.lib, self.req)
        self.assertEqual((self.net, self.lib, self.req), before)


if __name__ == "__main__":
    unittest.main()
