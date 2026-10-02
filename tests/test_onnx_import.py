"""Numerical import checks against ONNX's independent reference evaluator."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from npp.models import InputValidationError
from npp.onnx_import import import_onnx


HAS_ONNX = importlib.util.find_spec("onnx") is not None


def evaluate_vector_graph(network, value):
    results = {}
    for node in network["nodes"]:
        operands = [results[item] for item in node["inputs"]]
        if node["op"] == "input":
            result = np.asarray(value, dtype=float).reshape(-1)
        elif node["op"] == "linear":
            result = np.asarray(node["weights"]) @ operands[0] + node["bias"]
        elif node["op"] == "relu":
            result = np.maximum(operands[0], 0)
        elif node["op"] == "identity":
            result = operands[0]
        else:
            result = sum(operands)
        results[node["id"]] = result
    return [results[item] for item in network["outputs"]]


@unittest.skipUnless(HAS_ONNX, "optional onnx dependency is not installed")
class OnnxImportTests(unittest.TestCase):
    def setUp(self):
        import onnx
        self.onnx = onnx
        self.helper = onnx.helper
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "network.onnx"

    def tearDown(self):
        self.tmp.cleanup()

    def value(self, name, shape):
        return self.helper.make_tensor_value_info(name, self.onnx.TensorProto.FLOAT, shape)

    def tensor(self, name, value):
        return self.onnx.numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)

    def model(self, nodes, initializers=(), input_shape=(1, 2), outputs=None, inputs=None, value_info=()):
        graph = self.helper.make_graph(nodes, "Dense test", inputs or [self.value("x", input_shape)], outputs or [self.value("y", (1, 2))], initializer=initializers, value_info=value_info)
        model = self.helper.make_model(graph, opset_imports=[self.helper.make_opsetid("", 13)])
        self.onnx.save_model(model, self.path)
        return model

    def assert_reference(self, model, network, sample):
        from onnx.reference import ReferenceEvaluator
        reference = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: np.asarray(sample, dtype=np.float32)})
        actual = evaluate_vector_graph(network, sample)
        self.assertEqual(len(reference), len(actual))
        for expected, result in zip(reference, actual):
            np.testing.assert_allclose(result, np.asarray(expected).reshape(-1), rtol=2e-6, atol=2e-6)

    def assert_code(self, code):
        with self.assertRaises(InputValidationError) as caught:
            import_onnx(self.path)
        self.assertEqual(caught.exception.diagnostics[0]["code"], code)

    def test_gemm_transposed_scaling_bias_and_relu_match_reference(self):
        model = self.model([
            self.helper.make_node("Gemm", ["x", "W", "B"], ["/dense:0"], transB=1, alpha=0.5, beta=2.0),
            self.helper.make_node("Relu", ["/dense:0"], ["y"]),
        ], [self.tensor("W", [[1, -2], [-3, 4], [2, 0.5]]), self.tensor("B", [0.1, 0.2, -0.3])], outputs=[self.value("y", [1, 3])])
        network = import_onnx(self.path, input_bounds=([-2, -3], [2, 3]))
        self.assertEqual(network["nodes"][-1]["op"], "relu")
        self.assertEqual(network["nodes"][1]["id"], "node__dense:0")
        self.assertEqual(network["nodes"][0]["input_bounds"]["lower"], [-2, -3])
        self.assert_reference(model, network, [[0.25, -0.5]])

    def test_matmul_add_residual_multiple_outputs_keep_order(self):
        model = self.model([
            self.helper.make_node("MatMul", ["x", "W"], ["dense"]),
            self.helper.make_node("Add", ["B", "dense"], ["biased"]),
            self.helper.make_node("Relu", ["biased"], ["positive"]),
            self.helper.make_node("Add", ["positive", "x"], ["sum"]),
            self.helper.make_node("Identity", ["sum"], ["y"]),
        ], [self.tensor("W", [[1, -2], [3, 4]]), self.tensor("B", [0.1, -0.5])], input_shape=(2,), outputs=[self.value("y", [2]), self.value("dense", [2])])
        network = import_onnx(self.path)
        self.assertEqual(network["outputs"], ["y", "dense"])
        self.assert_reference(model, network, [0.3, -0.2])

    def test_symbolic_batch_means_batch_one_and_constant_matrix(self):
        weights = self.tensor("", [[2, 0], [0, -3]])
        model = self.model([
            self.helper.make_node("Constant", [], ["W"], value=weights),
            self.helper.make_node("MatMul", ["x", "W"], ["y"]),
        ], input_shape=("batch", 2), outputs=[self.value("y", ["batch", 2])])
        network = import_onnx(self.path)
        self.assertEqual(network["nodes"][0]["size"], 2)
        self.assert_reference(model, network, [[0.5, -0.25]])

    def test_gemm_nontransposed_matrix_and_scalar_bias(self):
        model = self.model([self.helper.make_node("Gemm", ["x", "W", "B"], ["y"])], [self.tensor("W", [[2, 3], [4, 5]]), self.tensor("B", 0.25)])
        self.assert_reference(model, import_onnx(self.path), [[0.1, 0.2]])

    def test_scalar_constant_bias_with_batch_axis(self):
        model = self.model([
            self.helper.make_node("Constant", [], ["B"], value_float=0.25),
            self.helper.make_node("Add", ["x", "B"], ["y"]),
        ])
        self.assert_reference(model, import_onnx(self.path), [[0.4, -0.2]])

    def test_rejects_unsupported_activation_instead_of_dropping_it(self):
        self.model([self.helper.make_node("Sigmoid", ["x"], ["y"])])
        self.assert_code("onnx_operator")

    def test_rejects_custom_domain(self):
        self.model([self.helper.make_node("Relu", ["x"], ["y"], domain="custom.hardware")])
        self.assert_code("onnx_domain")

    def test_rejects_fixed_batch_larger_than_one(self):
        self.model([self.helper.make_node("Identity", ["x"], ["y"])], input_shape=(4, 2), outputs=[self.value("y", [4, 2])])
        self.assert_code("onnx_batch")

    def test_rejects_dynamic_features(self):
        self.model([self.helper.make_node("Identity", ["x"], ["y"])], input_shape=(1, "features"), outputs=[self.value("y", [1, "features"])])
        self.assert_code("onnx_shape")

    def test_rejects_transposed_data_input(self):
        self.model([self.helper.make_node("Gemm", ["x", "W"], ["y"], transA=1)], [self.tensor("W", [[1, 2]])], outputs=[self.value("y", [2, 2])])
        self.assert_code("onnx_transpose")

    def test_rejects_overridable_parameters(self):
        self.model([self.helper.make_node("MatMul", ["x", "W"], ["y"])], [self.tensor("W", np.eye(2))], inputs=[self.value("x", [1, 2]), self.value("W", [2, 2])])
        self.assert_code("onnx_parameter_input")

    def test_rejects_dynamic_broadcast_add(self):
        self.model([
            self.helper.make_node("MatMul", ["x", "W"], ["scalar"]),
            self.helper.make_node("Add", ["x", "scalar"], ["y"]),
        ], [self.tensor("W", [[1], [1]])])
        self.assert_code("onnx_broadcast")

    def test_rejects_shape_annotation_disagreement(self):
        self.model([self.helper.make_node("Identity", ["x"], ["y"])], outputs=[self.value("y", [1, 3])])
        self.assert_code("onnx_shape")

    def test_rejects_external_weights_before_loading_payload(self):
        model = self.model([self.helper.make_node("MatMul", ["x", "W"], ["y"])], [self.tensor("W", np.eye(2))])
        self.onnx.external_data_helper.set_external_data(model.graph.initializer[0], location="missing-weights.bin")
        model.graph.initializer[0].ClearField("raw_data")
        self.path.write_bytes(model.SerializeToString())
        self.assert_code("onnx_external_data")

    def test_rejects_nonfinite_weights(self):
        self.model([self.helper.make_node("MatMul", ["x", "W"], ["y"])], [self.tensor("W", [[float("nan"), 0], [0, 1]])])
        self.assert_code("onnx_nonfinite")

    def test_rejects_mixed_integer_weights_and_float_input(self):
        weights = self.onnx.numpy_helper.from_array(np.eye(2, dtype=np.int64), name="W")
        model = self.model([self.helper.make_node("MatMul", ["x", "W"], ["y"])], [weights])
        # The default structural checker does not reject this invalid graph.
        self.onnx.checker.check_model(model)
        self.assert_code("onnx_invalid")

    def test_rejects_input_bounds_wrong_length(self):
        self.model([self.helper.make_node("Identity", ["x"], ["y"])])
        with self.assertRaises(InputValidationError) as caught:
            import_onnx(self.path, input_bounds=([-1, -1, -1], 1))
        self.assertEqual(caught.exception.diagnostics[0]["code"], "onnx_input_bounds")

    def test_duplicate_tensor_use_is_preserved(self):
        model = self.model([self.helper.make_node("Add", ["x", "x"], ["y"])])
        network = import_onnx(self.path)
        self.assertEqual(network["nodes"][1]["inputs"], ["x", "x"])
        self.assert_reference(model, network, [[0.3, -0.2]])


if __name__ == "__main__":
    unittest.main()
