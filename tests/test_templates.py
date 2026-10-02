"""Runnable templates and non-destructive project initialization."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import yaml

from npp.templates import create_project


class TemplateTests(unittest.TestCase):
    def test_initializer_writes_complete_numeric_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project with spaces"
            result = create_project(root)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["written_files"]), 5)
            self.assertTrue(all(Path(path).is_file() for path in result["written_files"]))
            manifest = yaml.safe_load((root / "project.yaml").read_text())
            self.assertEqual(set(manifest), {"name", "model", "hardware", "design"})
            for field in ("model", "hardware", "design"):
                self.assertIsInstance(yaml.safe_load((root / manifest[field]).read_text()), dict)
            with np.load(root / "weights.npz", allow_pickle=False) as tensors:
                self.assertEqual(set(tensors.files), {"hidden.weight", "hidden.bias", "readout.weight", "readout.bias"})
                self.assertEqual(tensors["hidden.weight"].shape, (3, 2))
                self.assertEqual(tensors["readout.weight"].shape, (1, 3))
                self.assertTrue(np.isfinite(tensors["hidden.weight"]).all())
            self.assertIn("--project", result["commands"]["plan"])
            self.assertIn("SYNTHETIC", (root / "hardware.yaml").read_text())

    def test_collision_preflight_does_not_write_partial_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # This is the last target: a loop that checks while writing would
            # already have overwritten or created the four specifications.
            (root / "weights.npz").write_bytes(b"keep my learned parameters")
            with self.assertRaisesRegex(ValueError, "already exist"):
                create_project(root)
            self.assertEqual([p.name for p in root.iterdir()], ["weights.npz"])
            self.assertEqual((root / "weights.npz").read_bytes(), b"keep my learned parameters")

    def test_explicit_overwrite_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_project(root)
            (root / "model.yaml").write_text("user model")
            (root / "notes.txt").write_text("keep these notes")
            create_project(root, overwrite=True)
            self.assertIn("Small dense network", (root / "model.yaml").read_text())
            self.assertEqual((root / "notes.txt").read_text(), "keep these notes")

    def test_directory_target_rejected_before_writes_even_with_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "design.yaml").mkdir()
            with self.assertRaisesRegex(ValueError, "regular files"):
                create_project(root, overwrite=True)
            self.assertEqual([p.name for p in root.iterdir()], ["design.yaml"])

    def test_generated_project_normalizes_and_finds_checked_plans(self):
        from npp.config import load_inputs
        from npp.planner import plan
        from npp.checker import check_record

        with tempfile.TemporaryDirectory() as tmp:
            create_project(tmp)
            network, library, request = load_inputs(project_path=Path(tmp) / "project.yaml")
            self.assertEqual([node["size"] for node in network["nodes"]], [2, 3, 3, 1])
            self.assertEqual(request["budgets"]["latency_ns"], 20.0)
            report = plan(network, library, request)
            self.assertEqual(report["status"], "ok")
            self.assertTrue(report["search"]["complete"])
            self.assertGreater(len(report["plans"]), 0)
            for record in report["plans"]:
                self.assertTrue(check_record(network, library, request, record)["valid"])


if __name__ == "__main__":
    unittest.main()
