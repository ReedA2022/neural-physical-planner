"""Import weights saved by real Keras without making Keras a runtime requirement."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from npp.config import load_inputs
from npp.weights import inspect_weights
from tests.fixtures import component, library, request


EXPORT_SCRIPT = r'''
from pathlib import Path
import sys
import keras
import numpy as np

root = Path(sys.argv[1])
keras.config.set_floatx("float64")
model = keras.Sequential([
    keras.Input(shape=(2,), dtype="float64"),
    keras.layers.Dense(2, name="hidden", dtype="float64"),
    keras.layers.ReLU(name="activation", dtype="float64"),
    keras.layers.Dense(1, name="readout", dtype="float64"),
])
w1 = np.array([[0.5, -0.75], [0.25, 0.125]], dtype=np.float64)
b1 = np.array([-0.125, 0.25], dtype=np.float64)
w2 = np.array([[0.625, -0.5]], dtype=np.float64)
b2 = np.array([0.125], dtype=np.float64)
samples = np.array([[-1.0, 0.75], [0.25, -0.75], [0.0, 0.0], [1.0, -1.0]])
model.set_weights([w1.T, b1, w2.T, b2])
model.save(root / "network.keras")
model.save_weights(root / "network.weights.h5")
np.savez(root / "reference.npz", samples=samples, predictions=model.predict(samples, verbose=0),
         w1=w1, b1=b1, w2=w2, b2=b2)
'''


HAS_KERAS_EXPORT = all(importlib.util.find_spec(module) is not None for module in ("keras", "torch", "h5py"))


@unittest.skipUnless(HAS_KERAS_EXPORT, "genuine Keras export test needs optional keras, torch and h5py")
class KerasExportTests(unittest.TestCase):
    def test_real_keras_and_hdf5_exports_preserve_weights_and_predictions(self):
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = os.environ.copy()
            environment["KERAS_BACKEND"] = "torch"
            exported = subprocess.run([sys.executable, "-c", EXPORT_SCRIPT, str(root)],
                                      env=environment, capture_output=True, text=True, timeout=90)
            self.assertEqual(exported.returncode, 0, exported.stdout + exported.stderr)
            with np.load(root / "reference.npz", allow_pickle=False) as archive:
                reference = {name: archive[name] for name in archive.files}

            # Discover names independently from the genuine HDF5 archive,
            # including Keras's generated layer/variable paths.
            datasets = {}
            with h5py.File(root / "network.weights.h5", "r") as archive:
                def collect(name, item):
                    if isinstance(item, h5py.Dataset):
                        datasets[name] = item[()]
                archive.visititems(collect)
            bindings = {}
            for name, expected in (("w1", reference["w1"].T), ("b1", reference["b1"]),
                                   ("w2", reference["w2"].T), ("b2", reference["b2"])):
                matches = [key for key, value in datasets.items() if np.array_equal(value, expected)]
                self.assertEqual(len(matches), 1, f"Ambiguous or missing saved Keras tensor: {name}")
                bindings[name] = matches[0]

            for extension in ("keras", "weights.h5"):
                with self.subTest(extension=extension):
                    weights = root / f"network.{extension}"
                    spec = {
                        "name": "Genuine Keras export", "weights": weights.name, "layout": "in_out",
                        "input": {"name": "x", "size": 2, "bounds": [-1, 1]},
                        "layers": [
                            {"name": "hidden", "type": "linear", "weights": bindings["w1"], "bias": bindings["b1"]},
                            {"name": "activation", "type": "relu"},
                            {"name": "readout", "type": "linear", "weights": bindings["w2"], "bias": bindings["b2"]},
                        ],
                    }
                    project = {
                        "model": spec,
                        "hardware": library([component("encoder", ["input"]),
                                             component("unit", ["linear", "relu"])]),
                        "design": request(allowed_domains=["digital"]),
                    }
                    path = root / "project.json"
                    path.write_text(json.dumps(project), encoding="utf-8")
                    # Importing the saved tensors requires neither Keras nor
                    # TensorFlow; block both imports during the actual check.
                    with patch.dict(sys.modules, {"keras": None, "tensorflow": None}):
                        metadata = inspect_weights(weights)
                        self.assertEqual({item["name"] for item in metadata["tensors"]}, set(datasets))
                        network, _, _ = load_inputs(project_path=path)
                    hidden, activation, readout = network["nodes"][1:]
                    self.assertEqual(activation["op"], "relu")
                    np.testing.assert_array_equal(hidden["weights"], reference["w1"])
                    np.testing.assert_array_equal(hidden["bias"], reference["b1"])
                    np.testing.assert_array_equal(readout["weights"], reference["w2"])
                    np.testing.assert_array_equal(readout["bias"], reference["b2"])
                    predictions = (np.maximum(reference["samples"] @ np.asarray(hidden["weights"]).T
                                               + hidden["bias"], 0)
                                   @ np.asarray(readout["weights"]).T + readout["bias"])
                    np.testing.assert_array_equal(predictions, reference["predictions"])


if __name__ == "__main__":
    unittest.main()
