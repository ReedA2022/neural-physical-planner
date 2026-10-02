"""External tensor imports, format boundaries, and restricted checkpoint loading."""
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from npp.models import InputValidationError
from npp.weights import inspect_weights, load_weights


def available(name):
    return importlib.util.find_spec(name) is not None


class WeightsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.matrix = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)

    def test_npz_preserves_names_values_shapes_and_dtype(self):
        path = self.root / "weights.npz"
        np.savez(path, **{"layers.0.weight": self.matrix, "layers.0.bias": np.array([0., 1.])})
        result = load_weights(path)
        np.testing.assert_array_equal(result["layers.0.weight"], self.matrix)
        self.assertEqual(result["layers.0.weight"].dtype, np.float32)
        metadata = inspect_weights(path)
        self.assertEqual(metadata["format"], "numpy_npz")
        self.assertEqual(metadata["tensors"][1], {"name": "layers.0.weight", "shape": [2, 2], "dtype": "float32"})

    def test_npy_uses_documented_array_name(self):
        path = self.root / "weights.npy"
        np.save(path, self.matrix)
        np.testing.assert_array_equal(load_weights(path)["array"], self.matrix)
        self.assertIn("'array'", inspect_weights(path)["note"])

    def test_npz_keys_ending_in_npy_are_not_silently_aliased(self):
        path = self.root / "names.npz"
        np.savez(path, **{"weight": np.array([1.0]), "weight.npy": np.array([2.0])})
        tensors = load_weights(path)
        np.testing.assert_array_equal(tensors["weight"], [1.0])
        np.testing.assert_array_equal(tensors["weight.npy"], [2.0])

    def test_reject_nonreal_nonfinite_and_objects(self):
        for array in [np.array([np.nan]), np.array([np.inf]), np.array([1j]),
                      np.array([True]), np.array(["text"]), np.array([{}], dtype=object)]:
            with self.subTest(dtype=array.dtype, value=repr(array)):
                path = self.root / "invalid.npy"
                np.save(path, array)
                with self.assertRaises(InputValidationError):
                    load_weights(path)

    def test_malformed_missing_unsupported_and_empty(self):
        missing = self.root / "missing.npz"
        with self.assertRaisesRegex(InputValidationError, "does not exist"):
            load_weights(missing)
        with self.assertRaisesRegex(InputValidationError, "Unsupported weight extension"):
            load_weights(self.root / "unsupported.pkl")
        path = self.root / "invalid.npz"
        path.write_text("not weights")
        with self.assertRaises(InputValidationError):
            load_weights(path)
        np.savez(path)
        with self.assertRaisesRegex(InputValidationError, "No tensors"):
            load_weights(path)

    def test_npz_duplicate_names_rejected(self):
        path = self.root / "duplicate.npz"
        stream = io.BytesIO()
        np.save(stream, self.matrix)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("matrix.npy", stream.getvalue())
                archive.writestr("matrix.npy", stream.getvalue())
        with self.assertRaisesRegex(InputValidationError, "duplicate"):
            load_weights(path)

    def test_compressed_npz_expanded_budget_checked_before_loading(self):
        path = self.root / "compressed.npz"
        np.savez_compressed(path, weights=np.zeros(10_000))
        with patch("npp.weights.MAX_WEIGHT_BYTES", 10_000):
            with self.assertRaises(InputValidationError) as error:
                load_weights(path)
        self.assertEqual(error.exception.diagnostics[0]["code"], "weights_resource_limit")

    def test_declared_shape_budget_checked_before_loading(self):
        path = self.root / "huge.npy"
        with path.open("wb") as stream:
            np.lib.format.write_array_header_1_0(stream, {"shape": (2**30,), "fortran_order": False, "descr": "<f8"})
        with self.assertRaises(InputValidationError) as error:
            load_weights(path)
        self.assertEqual(error.exception.diagnostics[0]["code"], "weights_resource_limit")

    def test_state_dict_selector_restricted_to_pytorch(self):
        path = self.root / "weights.npy"
        np.save(path, self.matrix)
        with self.assertRaisesRegex(InputValidationError, "only supported for PyTorch"):
            load_weights(path, state_dict_key="weights")

    def test_missing_optional_dependency_explains_extra(self):
        path = self.root / "weights.safetensors"
        path.write_bytes(b"placeholder")
        with patch("npp.weights.importlib.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(InputValidationError, r"source directory.*pip install '\.\[safetensors\]'"):
                load_weights(path)

    def test_pytorch_restricted_arguments_and_no_unsafe_retry(self):
        from unittest.mock import Mock
        path = self.root / "weights.pt"
        path.write_bytes(b"not a zip")
        loader = Mock(side_effect=RuntimeError("restricted rejection"))
        fake_torch = types.SimpleNamespace(load=loader)
        with patch("npp.weights._dependency", return_value=fake_torch):
            with self.assertRaisesRegex(InputValidationError, "unsafe loading is never retried"):
                load_weights(path)
        loader.assert_called_once_with(path, weights_only=True, map_location="cpu")

    def test_torchscript_rejected_before_torch_load(self):
        from unittest.mock import Mock
        path = self.root / "model.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("model/constants.pkl", b"not loaded")
        fake_torch = types.SimpleNamespace(load=Mock())
        with patch("npp.weights._dependency", return_value=fake_torch):
            with self.assertRaisesRegex(InputValidationError, "TorchScript"):
                load_weights(path)
        fake_torch.load.assert_not_called()

    @unittest.skipUnless(available("safetensors"), "safetensors optional dependency unavailable")
    def test_safetensors_roundtrip(self):
        from safetensors.numpy import save_file
        path = self.root / "weights.safetensors"
        save_file({"weight": self.matrix}, str(path))
        np.testing.assert_array_equal(load_weights(path)["weight"], self.matrix)

    @unittest.skipUnless(available("torch"), "PyTorch optional dependency unavailable")
    def test_pytorch_plain_wrapped_and_explicit_state_dict(self):
        import torch
        state = {"weight": torch.from_numpy(self.matrix)}
        path = self.root / "weights.pt"
        for content, key in [(state, None), ({"state_dict": state, "epoch": 4}, None),
                             ({"model_state_dict": state}, None), ({"network": state}, "network")]:
            with self.subTest(key=key, wrapper=list(content)):
                torch.save(content, path)
                np.testing.assert_array_equal(load_weights(path, state_dict_key=key)["weight"], self.matrix)

    @unittest.skipUnless(available("torch"), "PyTorch optional dependency unavailable")
    def test_pytorch_bfloat_and_full_module(self):
        import torch
        path = self.root / "weights.pt"
        torch.save({"weight": torch.tensor([[1., -2.]], dtype=torch.bfloat16)}, path)
        np.testing.assert_array_equal(load_weights(path)["weight"], [[1., -2.]])
        torch.save(torch.nn.Linear(2, 2), path)
        with self.assertRaisesRegex(InputValidationError, "Restricted PyTorch loading failed"):
            load_weights(path)

    @unittest.skipUnless(available("h5py"), "h5py optional dependency unavailable")
    def test_hdf5_nested_paths_and_keras_archive(self):
        import h5py
        path = self.root / "model.weights.h5"
        with h5py.File(path, "w") as archive:
            archive.create_dataset("layers/dense/vars/0", data=self.matrix)
        key = "layers/dense/vars/0"
        np.testing.assert_array_equal(load_weights(path)[key], self.matrix)
        keras = self.root / "model.keras"
        with zipfile.ZipFile(keras, "w") as archive:
            archive.write(path, "model.weights.h5")
            archive.writestr("config.json", "This configuration must never be deserialized.")
        np.testing.assert_array_equal(load_weights(keras)[key], self.matrix)

    @unittest.skipUnless(available("h5py"), "h5py optional dependency unavailable")
    def test_hdf5_external_link_rejected(self):
        import h5py
        path = self.root / "linked.h5"
        with h5py.File(path, "w") as archive:
            archive["outside"] = h5py.ExternalLink("other.h5", "/weight")
        with self.assertRaisesRegex(InputValidationError, "external and soft links"):
            load_weights(path)

    @unittest.skipUnless(available("onnx"), "ONNX optional dependency unavailable")
    def test_onnx_initializer_weights(self):
        import onnx
        path = self.root / "model.onnx"
        tensor = onnx.numpy_helper.from_array(self.matrix, name="linear.weight")
        graph = onnx.helper.make_graph([], "empty_graph", [], [], initializer=[tensor])
        onnx.save_model(onnx.helper.make_model(graph), str(path))
        np.testing.assert_array_equal(load_weights(path)["linear.weight"], self.matrix)

    @unittest.skipUnless(available("onnx"), "ONNX optional dependency unavailable")
    def test_onnx_external_initializer_rejected(self):
        import onnx
        path = self.root / "external.onnx"
        tensor = onnx.numpy_helper.from_array(self.matrix, name="linear.weight")
        onnx.external_data_helper.set_external_data(tensor, location="external.bin")
        tensor.ClearField("raw_data")
        graph = onnx.helper.make_graph([], "empty_graph", [], [], initializer=[tensor])
        path.write_bytes(onnx.helper.make_model(graph).SerializeToString())
        with self.assertRaisesRegex(InputValidationError, "External ONNX tensor"):
            load_weights(path)


if __name__ == "__main__":
    unittest.main()
