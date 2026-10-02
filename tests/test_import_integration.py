"""End-to-end evidence that imported parameters retain computation and plans.

These fixtures use small, exactly representable numbers and deliberately
nonsymmetric weights.  The expected predictions are computed independently of
the input loader and of the planner's numerical error evaluator.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from npp.checker import check_record
from npp.config import load_inputs
from npp.models import validate_inputs
from npp.onnx_import import import_onnx
from npp.planner import plan
from tests.fixtures import component, library, network, node, request


ROOT = Path(__file__).resolve().parents[1]
W1 = np.array([[0.5, -0.75], [0.25, 0.125]], dtype=np.float64)
B1 = np.array([-0.125, 0.25], dtype=np.float64)
W2 = np.array([[0.625, -0.5]], dtype=np.float64)
B2 = np.array([0.125], dtype=np.float64)
SAMPLES = np.array([[-1.0, 0.75], [0.25, -0.75], [0.0, 0.0], [1.0, -1.0]])


def available(module):
    return importlib.util.find_spec(module) is not None


def expected_predictions():
    return np.maximum(SAMPLES @ W1.T + B1, 0) @ W2.T + B2


def graph_predictions(net):
    values = {}
    for operation in net["nodes"]:
        inputs = [values[name] for name in operation["inputs"]]
        if operation["op"] == "input":
            value = SAMPLES
        elif operation["op"] == "linear":
            value = inputs[0] @ np.asarray(operation["weights"]).T + operation["bias"]
        elif operation["op"] == "relu":
            value = np.maximum(inputs[0], 0)
        elif operation["op"] == "identity":
            value = inputs[0]
        elif operation["op"] == "add":
            value = sum(inputs)
        else:
            raise AssertionError(f"Unexpected operation: {operation['op']}")
        values[operation["id"]] = value
    return np.concatenate([values[name] for name in net["outputs"]], axis=1)


class ImportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.parameters = {"hidden.weight": W1, "hidden.bias": B1,
                           "readout.weight": W2, "readout.bias": B2}
        self.lib = library([
            component("encoder", ["input"], energy_pj=0.25, noise_std=0.01),
            component("activation", ["relu"], energy_pj=0.5, latency_ns=0.5, noise_std=0.02),
            component("fast", ["linear"], energy_pj=4, latency_ns=1, area_um2=2, noise_std=0.04),
            component("efficient", ["linear"], energy_pj=1, latency_ns=4, area_um2=1, noise_std=0.02),
        ])
        self.req = request(allowed_domains=["digital"])
        self.reference = network([
            node("x", "input", size=2, input_bounds={"lower": [-1, -1], "upper": [1, 1]}),
            node("hidden", "linear", ["x"], size=2, weights=W1.tolist(), bias=B1.tolist()),
            node("activation", "relu", ["hidden"], size=2),
            node("readout", "linear", ["activation"], weights=W2.tolist(), bias=B2.tolist()),
        ], ["readout"])
        self.reference["name"] = "Import integration"
        self.reference, self.lib, self.req = validate_inputs(self.reference, self.lib, self.req)
        self.write_json("hardware.json", self.lib)
        self.write_json("design.json", self.req)

    def write_json(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_yaml(self, name, payload):
        import yaml
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def model_spec(self, source, layout="out_in"):
        return {
            "name": "Import integration", "weights": source, "layout": layout,
            "input": {"name": "x", "size": 2, "bounds": [-1, 1]},
            "layers": [
                {"name": "hidden", "type": "linear", "weights": "hidden.weight", "bias": "hidden.bias"},
                {"name": "activation", "type": "relu"},
                {"name": "readout", "type": "linear", "weights": "readout.weight", "bias": "readout.bias"},
            ],
        }

    def project(self, model):
        self.write_yaml("nested/model.yaml", model)
        return self.write_yaml("project.yaml", {"name": "Integration project", "model": "nested/model.yaml",
                                                "hardware": "hardware.json", "design": "design.json"})

    def assert_equivalent(self, actual):
        self.assertEqual(actual["nodes"], self.reference["nodes"])
        self.assertEqual(actual["outputs"], self.reference["outputs"])
        np.testing.assert_array_equal(graph_predictions(actual), expected_predictions())
        # ReLU is semantically necessary on these samples. An importer that
        # merely preserves affine layers cannot satisfy the prediction check.
        without_relu = (SAMPLES @ W1.T + B1) @ W2.T + B2
        self.assertGreater(float(np.max(np.abs(without_relu - expected_predictions()))), 0.1)

    def assert_plans_equal(self, actual):
        reference_report = plan(self.reference, self.lib, self.req)
        actual_report = plan(actual, self.lib, self.req)
        self.assertEqual(actual_report["status"], "ok")
        self.assertTrue(actual_report["search"]["complete"])
        self.assertEqual(actual_report["search"]["evaluated"], 4)
        self.assertGreater(len(actual_report["plans"]), 1)
        expected = sorted((json.dumps(record["decision"], sort_keys=True), record["evaluation"]["metrics"])
                          for record in reference_report["plans"])
        result = sorted((json.dumps(record["decision"], sort_keys=True), record["evaluation"]["metrics"])
                        for record in actual_report["plans"])
        self.assertEqual(result, expected)
        for record in actual_report["plans"]:
            self.assertTrue(check_record(actual, self.lib, self.req, record)["valid"])

    def assert_import(self, source, layout="out_in"):
        project = self.project(self.model_spec(source, layout))
        actual, lib, req = load_inputs(project_path=project)
        self.assertEqual(lib, self.lib)
        self.assertEqual(req, self.req)
        self.assert_equivalent(actual)
        self.assert_plans_equal(actual)
        return actual

    def test_npz_relative_paths_recover_network_and_pareto_front(self):
        np.savez(self.root / "weights.npz", **self.parameters)
        self.assert_import("../weights.npz")

    @unittest.skipUnless(available("safetensors"), "optional safetensors dependency is not installed")
    def test_safetensors_recover_network_and_pareto_front(self):
        from safetensors.numpy import save_file
        save_file(self.parameters, str(self.root / "weights.safetensors"))
        self.assert_import("../weights.safetensors")

    @unittest.skipUnless(available("h5py"), "optional h5py dependency is not installed")
    def test_hdf5_and_keras_archives_recover_network_and_pareto_front(self):
        import h5py
        path = self.root / "model.weights.h5"
        with h5py.File(path, "w") as archive:
            for name, value in self.parameters.items():
                archive.create_dataset(name, data=value.T if name.endswith("weight") else value)
        with zipfile.ZipFile(self.root / "model.keras", "w") as archive:
            archive.write(path, "model.weights.h5")
            archive.writestr("config.json", '{"class_name":"IgnoredCustomCode"}')
        for source in ("../model.weights.h5", "../model.keras"):
            with self.subTest(source=source):
                self.assert_import(source, layout="in_out")

    @unittest.skipUnless(available("torch"), "optional torch dependency is not installed")
    def test_pytorch_state_dict_wrappers_and_framework_prediction(self):
        import torch
        tensors = {name: torch.from_numpy(value.copy()) for name, value in self.parameters.items()}
        for filename, payload, selector in (
                ("plain.pt", tensors, None),
                ("wrapped.pth", {"state_dict": tensors, "epoch": 7}, None),
                ("model.bin", {"model_state_dict": tensors, "epoch": 7}, None),
                ("custom.pt", {"trained_model": tensors, "epoch": 7}, "trained_model")):
            with self.subTest(filename=filename):
                torch.save(payload, self.root / filename)
                source = "../" + filename
                if selector:
                    source = {"file": source, "state_dict_key": selector}
                self.assert_import(source)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)).double()
        model.load_state_dict({"0.weight": tensors["hidden.weight"], "0.bias": tensors["hidden.bias"],
                               "2.weight": tensors["readout.weight"], "2.bias": tensors["readout.bias"]})
        with torch.no_grad():
            result = model(torch.from_numpy(SAMPLES)).numpy()
        np.testing.assert_array_equal(result, expected_predictions())

    def test_per_tensor_npy_files_and_explicit_square_transpose(self):
        spec = self.model_spec(None)
        del spec["weights"]
        for index, prefix in ((0, "hidden"), (2, "readout")):
            for kind in ("weight", "bias"):
                value = self.parameters[f"{prefix}.{kind}"]
                filename = f"{prefix}.{kind}.npy"
                np.save(self.root / filename, value.T if kind == "weight" else value)
                reference = {"file": "../" + filename, "tensor": "array"}
                if kind == "weight":
                    reference["layout"] = "in_out"
                spec["layers"][index]["weights" if kind == "weight" else "bias"] = reference
        actual, _, _ = load_inputs(project_path=self.project(spec))
        self.assert_equivalent(actual)
        self.assert_plans_equal(actual)

    def make_onnx(self):
        import onnx
        helper = onnx.helper
        model = helper.make_model(helper.make_graph([
            helper.make_node("Gemm", ["x", "hidden.weight", "hidden.bias"], ["hidden"], transB=1),
            helper.make_node("Relu", ["hidden"], ["activation"]),
            helper.make_node("Gemm", ["activation", "readout.weight", "readout.bias"], ["readout"], transB=1),
        ], "Import integration", [helper.make_tensor_value_info("x", onnx.TensorProto.DOUBLE, [1, 2])],
            [helper.make_tensor_value_info("readout", onnx.TensorProto.DOUBLE, [1, 1])],
            initializer=[onnx.numpy_helper.from_array(value, name=name) for name, value in self.parameters.items()]),
            opset_imports=[helper.make_opsetid("", 13)])
        path = self.root / "model.onnx"
        onnx.save_model(model, path)
        return path, model

    @unittest.skipUnless(available("onnx"), "optional onnx dependency is not installed")
    def test_onnx_graph_and_initializers_recover_same_predictions_and_plans(self):
        from onnx.reference import ReferenceEvaluator
        path, model = self.make_onnx()
        actual = import_onnx(path, input_bounds=(-1, 1), name="Import integration")
        self.assert_equivalent(actual)
        self.assert_plans_equal(actual)
        self.assert_import("../model.onnx")
        reference = ReferenceEvaluator(model)
        for index, sample in enumerate(SAMPLES):
            result = reference.run(None, {"x": sample.reshape(1, 2)})[0]
            np.testing.assert_array_equal(result, expected_predictions()[index:index + 1])

    def run_cli(self, *arguments, expect=0):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        elsewhere = self.root / "different_working_directory"
        elsewhere.mkdir(exist_ok=True)
        result = subprocess.run([sys.executable, "-m", "npp", *map(str, arguments)], cwd=elsewhere,
                                env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout)

    def test_cli_normalization_is_portable_and_saved_plans_replay(self):
        np.savez(self.root / "weights.npz", **self.parameters)
        project = self.project(self.model_spec("../weights.npz"))
        metadata = self.run_cli("inspect-weights", self.root / "weights.npz", "--json")
        self.assertEqual({item["name"] for item in metadata["tensors"]}, set(self.parameters))
        normalized = self.root / "normalized"
        self.run_cli("normalize", "--project", project, "--out-dir", normalized)
        first_run = self.root / "first_run"
        result = self.run_cli("plan", "--project", project, "--out-dir", first_run, "--samples", "128")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["all_saved_plans_checked"])
        # Normalized JSON embeds parameters and must survive source removal.
        (self.root / "weights.npz").unlink()
        second_run = self.root / "portable_run"
        inputs = ["--network", normalized / "network.json", "--library", normalized / "library.json",
                  "--request", normalized / "request.json"]
        self.run_cli("plan", *inputs, "--out-dir", second_run)
        first = json.loads((first_run / "report.json").read_text())
        second = json.loads((second_run / "report.json").read_text())
        self.assertEqual(first["input_hashes"], second["input_hashes"])
        self.assertEqual(first["plans"], second["plans"])
        saved = next((second_run / "plans").glob("*.json"))
        self.assertTrue(self.run_cli("check", *inputs, "--plan", saved)["valid"])

    @unittest.skipUnless(available("onnx"), "optional onnx dependency is not installed")
    def test_cli_onnx_import_and_generated_template_are_runnable(self):
        path, _ = self.make_onnx()
        output = self.root / "imported.json"
        self.run_cli("import-model", path, "--out", output, "--input-min", "-1", "--input-max", "1")
        self.assert_equivalent(json.loads(output.read_text()))
        directory = self.root / "new_project"
        self.run_cli("init", directory)
        self.run_cli("validate", "--project", directory / "project.yaml")
        run = self.run_cli("plan", "--project", directory / "project.yaml", "--out-dir", directory / "run")
        self.assertEqual(run["status"], "ok")
        self.assertTrue(run["all_saved_plans_checked"])

    def test_missing_tensor_and_bad_shape_fail_with_actionable_diagnostics(self):
        np.savez(self.root / "weights.npz", **self.parameters)
        spec = self.model_spec("../weights.npz")
        spec["layers"][0]["weights"] = "hidden.typo"
        project = self.project(spec)
        result = self.run_cli("validate", "--project", project, expect=2)
        self.assertEqual(result["status"], "invalid_input")
        self.assertIn("hidden.typo", json.dumps(result["diagnostics"]))
        bad = deepcopy(self.parameters)
        bad["hidden.weight"] = np.ones((2, 3))
        np.savez(self.root / "weights.npz", **bad)
        project = self.project(self.model_spec("../weights.npz"))
        result = self.run_cli("validate", "--project", project, expect=2)
        self.assertEqual(result["status"], "invalid_input")
        text = json.dumps(result["diagnostics"]).lower()
        self.assertTrue("shape" in text or "dimension" in text, text)

    def test_corrupt_weights_surface_structured_error_without_traceback(self):
        (self.root / "weights.npz").write_bytes(b"not a numpy archive")
        project = self.project(self.model_spec("../weights.npz"))
        result = self.run_cli("validate", "--project", project, expect=2)
        self.assertEqual(result["status"], "invalid_input")
        self.assertIn("weights.npz", json.dumps(result["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
