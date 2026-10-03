"""Regression contracts discovered by the input stress campaign."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from npp.config import _quantity, load_inputs, read_spec
from npp.models import InputValidationError
from npp.planner import input_hash
from scripts.stress_inputs import run_campaign


class InputStressRegressionTests(unittest.TestCase):
    def test_decimal_underflow_cannot_turn_nonzero_quantities_into_free_costs(self):
        for value in ("1e-999 pJ", "-1e-999 pJ", "1e-999999999 pJ", "-1e-999999999 pJ"):
            with self.subTest(value=value), self.assertRaises(InputValidationError):
                _quantity(value, "energy_pj", "energy")
        for value in ("0 pJ", "-0 pJ", "0e-999999999 pJ"):
            self.assertEqual(_quantity(value, "energy_pj", "energy"), 0)
        self.assertEqual(_quantity("5e-324 pJ", "energy_pj", "energy"), 5e-324)
        self.assertEqual(_quantity("5e-321 fJ", "energy_pj", "energy"), 5e-324)

    def test_inline_mixed_boolean_weights_and_bias_are_rejected_before_coercion(self):
        project = {"name": "strict numeric values", "hardware": "example", "model": {
            "input": {"size": 2, "bounds": [-1, 1]},
            "layers": [{"type": "linear", "weights": [[True, 2], [1, 2]], "bias": [0, 0]}]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "project.json"
            for field in ("weights", "bias"):
                if field == "bias":
                    project["model"]["layers"][0].update(weights=[[1, 2], [1, 2]], bias=[0, False])
                path.write_text(json.dumps(project))
                with self.subTest(field=field), self.assertRaises(InputValidationError) as error:
                    load_inputs(project_path=path)
                self.assertEqual(error.exception.diagnostics[0]["code"], "invalid_tensor")

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "optional onnx dependency unavailable")
    def test_onnx_mixed_boolean_input_bounds_are_rejected(self):
        import onnx
        from npp.onnx_import import import_onnx
        h = onnx.helper
        graph = h.make_graph([h.make_node("Identity", ["x"], ["y"])], "bounds", [
            h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])], [
            h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.onnx"
            onnx.save(h.make_model(graph), path)
            with self.assertRaises(InputValidationError):
                import_onnx(path, input_bounds=([False, -1], [1, 1]))

    def test_decimal_unit_equivalents_have_identical_normalization_and_hash(self):
        project = {"name": "unit equivalence", "hardware": "example", "model": {
            "input": {"size": 1, "bounds": [-1, 1]},
            "layers": [{"type": "identity"}]}, "design": {"limits": {"energy": "9 fJ"}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "project.json"
            path.write_text(json.dumps(project))
            left = load_inputs(project_path=path)
            project["design"]["limits"]["energy"] = "0.009 pJ"
            path.write_text(json.dumps(project))
            right = load_inputs(project_path=path)
        self.assertEqual(left, right)
        self.assertEqual(input_hash(left), input_hash(right))

    def test_alias_expansion_budget_is_structured_and_preserves_small_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aliases.yaml"
            path.write_text("x: &x [1, 2]\ny: *x\n")
            self.assertEqual(read_spec(path), {"x": [1, 2], "y": [1, 2]})
            # The 480-byte source represents millions of expanded values.
            # The guard must count them without actually traversing them all.
            path.write_text("a0: &a0 [0]\n" + "".join(
                f"a{i}: &a{i} [*a{i-1}, *a{i-1}]\n" for i in range(1, 23)))
            with self.assertRaises(InputValidationError) as error:
                read_spec(path)
            self.assertEqual(error.exception.diagnostics[0]["code"], "spec_resource_limit")

    def test_alias_budget_preserves_plain_tree_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp, patch("npp.config.MAX_SPEC_ALIAS_VALUES", 2):
            path = Path(tmp) / "tree.json"
            path.write_text('{"x":[0,0],"y":[0,0]}')
            self.assertEqual(read_spec(path), {"x": [0, 0], "y": [0, 0]})
            path = Path(tmp) / "tree.yaml"
            path.write_text('x: &x [0,0]\ny: *x\n')
            with self.assertRaises(InputValidationError):
                read_spec(path)

    def test_seeded_campaign_is_reproducible_and_counts_missing_dependencies(self):
        first = run_campaign(seed=321, cases=32)
        second = run_campaign(seed=321, cases=32)
        self.assertEqual(first["failures"], [])
        for report in (first, second):
            report.pop("elapsed_seconds")
            self.assertEqual(report["scenario_count"], report["passed_count"] + report["skipped_count"])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
