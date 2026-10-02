"""Friendly syntax must normalize reproducibly and reject ambiguous mistakes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from npp.config import load_inputs, read_spec
from npp.models import InputValidationError, load_json, validate_inputs


EXAMPLES = Path(__file__).resolve().parents[1] / "npp" / "data"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps(value), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        return path

    def project(self):
        return {
            "name": "Test model",
            "model": {
                "input": {"size": 2, "bounds": [-1, 1]},
                "layers": [
                    {"name": "hidden", "type": "linear", "weights": [[1, 2], [3, 4]], "bias": [0.1, 0.2]},
                    {"name": "activation", "type": "relu"},
                    {"name": "out", "type": "linear", "weights": [[1, -1]]},
                ],
            },
            "hardware": "example",
            "design": {"minimize": ["energy", "latency", "error"], "search": "exhaustive"},
        }

    def load(self, value):
        return load_inputs(project_path=self.write("project.yaml", value))

    def test_legacy_inputs_are_identical(self):
        paths = [EXAMPLES / name for name in ("small_relu.json", "components.json", "request_default.json")]
        expected = validate_inputs(*(load_json(path) for path in paths))
        self.assertEqual(expected, load_inputs(network_path=paths[0], library_path=paths[1], request_path=paths[2]))

    def test_yaml_json_equivalence_and_sequential_expansion(self):
        project = self.project()
        from_yaml = self.load(project)
        from_json = load_inputs(project_path=self.write("project.json", project))
        self.assertEqual(from_yaml, from_json)
        network, library, request = from_yaml
        self.assertEqual(network["nodes"][0]["input_bounds"], {"lower": [-1.0, -1.0], "upper": [1.0, 1.0]})
        self.assertEqual(network["nodes"][-1]["bias"], [0.0])
        self.assertEqual(network["nodes"][2]["inputs"], ["hidden"])
        self.assertEqual(network["outputs"], ["out"])
        self.assertEqual(request["objectives"], ["energy_pj", "latency_ns", "error_rms_bound"])
        self.assertEqual(request["search"]["mode"], "exhaustive")
        self.assertIn("SYNTHETIC", library["provenance"])

    def test_each_declaring_spec_controls_relative_paths(self):
        project = self.project()
        model = project.pop("model")
        model["weights"] = "weights.npz"
        model["layers"][0] = {"name": "hidden", "type": "linear"}
        self.write("models/model.yml", model)
        np.savez(self.root / "models" / "weights.npz", **{"hidden.weight": [[1., 2.], [3., 4.]], "hidden.bias": [0.1, 0.2]})
        project["model"] = "models/model.yml"
        project["hardware"] = "hardware/hardware.yml"
        project["design"] = "design/design.yaml"
        self.write("hardware/hardware.yml", {"extends": "example", "overrides": {"components": {"optical_fixed": {"latency": "500 ps"}}}})
        self.write("design/design.yaml", {"minimize": ["latency"], "limits": {"latency": "2 us"}})
        network, library, request = self.load(project)
        self.assertEqual(network["nodes"][1]["bias"], [0.1, 0.2])
        self.assertEqual(request["budgets"]["latency_ns"], 2000.0)
        self.assertEqual(next(c for c in library["components"] if c["id"] == "optical_fixed")["latency_ns"], 0.5)

    def test_per_layer_reference_infers_bias_from_same_source(self):
        np.savez(self.root / "weights.npz", **{"fc.weight": [[1., 2.], [3., 4.]], "fc.bias": [5., 6.]})
        project = self.project()
        project["model"]["layers"] = [{"name": "different_name", "type": "linear", "weights": {"file": "weights.npz", "tensor": "fc.weight"}}]
        network, _, _ = self.load(project)
        self.assertEqual(network["nodes"][1]["bias"], [5.0, 6.0])
        project["model"]["layers"][0]["bias"] = None
        self.assertEqual(self.load(project)[0]["nodes"][1]["bias"], [0.0, 0.0])

    def test_global_and_per_layer_layout_override(self):
        project = self.project()
        project["model"]["layout"] = "in_out"
        project["model"]["layers"][2]["layout"] = "out_in"
        network, _, _ = self.load(project)
        self.assertEqual(network["nodes"][1]["weights"], [[1.0, 3.0], [2.0, 4.0]])
        self.assertEqual(network["nodes"][3]["weights"], [[1.0, -1.0]])

    def test_canonical_graph_external_tensors(self):
        np.savez(self.root / "weights.npz", **{"w": [[1., 2.]], "b": [0.5]})
        project = self.project()
        project["model"] = {
            "weights": "weights.npz", "nodes": [
                {"id": "x", "op": "input", "size": 2, "input_bounds": {"lower": [-1., -1.], "upper": [1., 1.]}},
                {"id": "y", "op": "linear", "size": 1, "inputs": ["x"], "weights": "w", "bias": "b"},
            ], "outputs": ["y"],
        }
        network, _, _ = self.load(project)
        self.assertEqual(network["nodes"][1]["weights"], [[1.0, 2.0]])
        self.assertEqual(network["nodes"][1]["bias"], [0.5])

    def test_units_editability_and_per_coordinate_bounds(self):
        project = self.project()
        project["model"]["input"]["bounds"] = {"lower": [-2, -1], "upper": [1, 3]}
        project["design"].update({"editable": True, "limits": {"latency": "2 µs", "area": "1 mm^2", "energy": "10 nJ", "error": 0.1}})
        project["hardware"] = {"extends": "example", "overrides": {"components": {"optical_fixed": {"area": "20 µm²", "energy": "1 nJ", "wavelength_nm": "1.55 um"}}}}
        network, library, request = self.load(project)
        self.assertEqual(network["nodes"][0]["input_bounds"]["upper"], [1., 3.])
        self.assertEqual(request["required_editable"], ["hidden", "out"])
        self.assertEqual(request["budgets"], {"latency_ns": 2000., "area_um2": 1e6, "energy_pj": 1e4, "error_rms_bound": 0.1})
        component = next(c for c in library["components"] if c["id"] == "optical_fixed")
        self.assertEqual(component["area_um2"], 20.)
        self.assertEqual(component["energy_pj"], 1000.)
        self.assertEqual(component["wavelength_nm"], 1550.)

    def test_custom_library_id_maps_and_defaults(self):
        library = load_json(EXAMPLES / "components.json")
        library["components"] = {item.pop("id"): item for item in library["components"]}
        for component in library["components"].values():
            component.pop("capacity")
            component.pop("max_abs_value")
        library["defaults"] = {"components": {"capacity": 64, "max_abs_value": 8}}
        project = self.project()
        project["hardware"] = library
        loaded = self.load(project)[1]
        self.assertEqual(loaded, validate_inputs(self.load(self.project())[0], load_json(EXAMPLES / "components.json"), {"name": "test"})[1])

    def test_default_names_and_default_design(self):
        project = self.project()
        del project["design"]
        for layer in project["model"]["layers"]:
            del layer["name"]
        network, _, request = self.load(project)
        self.assertEqual([node["id"] for node in network["nodes"]], ["x", "layer_1", "layer_2", "layer_3"])
        self.assertEqual(request["search"]["mode"], "beam")

    def test_rejects_unknown_fields_conflicts_and_wrong_units(self):
        mutations = [
            lambda p: p.update(desgin={}),
            lambda p: p["model"]["layers"][0].update(weigths=[[1, 2]]),
            lambda p: p["design"].update(objectives=["energy_pj"]),
            lambda p: p["design"].update(limits={"latency": "1 nJ"}),
            lambda p: p["design"].update(limits={"energy": "1e1000 pJ"}),
            lambda p: p["design"].update(limits={"area": True}),
            lambda p: p.update(hardware={"extends": "example", "overrides": {"components": {"missing_id": {"latency": "2 ns"}}}}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                project = self.project()
                mutate(project)
                with self.assertRaises(InputValidationError):
                    self.load(project)

    def test_rejects_shape_or_missing_tensor_with_context(self):
        project = self.project()
        project["model"]["layers"][0]["weights"] = [[1, 2, 3]]
        with self.assertRaises(InputValidationError) as captured:
            self.load(project)
        self.assertIn("layers[0].weights", str(captured.exception))
        self.assertIn("previous layer has size 2", str(captured.exception))
        np.savez(self.root / "weights.npz", known=np.ones((2, 2)))
        project["model"]["weights"] = "weights.npz"
        project["model"]["layers"][0]["weights"] = "typo"
        with self.assertRaises(InputValidationError) as captured:
            self.load(project)
        self.assertIn("known", str(captured.exception))
        self.assertEqual(captured.exception.diagnostics[0]["code"], "unknown_tensor")

    def test_duplicate_keys_unsafe_tags_nonfinite_and_parse_locations(self):
        samples = {
            "duplicate.yaml": "name: a\nname: b\n",
            "duplicate.json": '{"name":"a","name":"b"}',
            "unsafe.yaml": "name: !!python/object/apply:os.system ['echo unsafe']",
            "nonfinite.yaml": "name: .nan",
            "nonfinite.json": '{"value": 1e999}',
            "recursive.yaml": "name: &r [*r]",
            "broken.yaml": "name: [a,\n",
            "numeric_key.yaml": "1: name",
        }
        for name, content in samples.items():
            with self.subTest(name=name):
                path = self.root / name
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(InputValidationError) as captured:
                    read_spec(path)
                self.assertIn(name, str(captured.exception))
                if name in ("duplicate.yaml", "broken.yaml"):
                    self.assertIn("line", str(captured.exception))

    def test_missing_bounds_and_ambiguous_input_modes_rejected(self):
        project = self.project()
        del project["model"]["input"]["bounds"]
        with self.assertRaises(InputValidationError):
            self.load(project)
        path = self.write("project.yaml", self.project())
        with self.assertRaises(InputValidationError):
            load_inputs(project_path=path, network_path="ignored.json")
        with self.assertRaises(InputValidationError):
            load_inputs(network_path="only-one.json")


if __name__ == "__main__":
    unittest.main()
