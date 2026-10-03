"""User-facing physical workflow checks, including portable saved inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PhysicalCliTests(unittest.TestCase):
    def invoke(self, *arguments, expected=0):
        process = subprocess.run([sys.executable, "-m", "npp", *map(str, arguments)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return json.loads(process.stdout)

    def test_technology_export_can_be_reloaded_and_does_not_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "technology.json"
            original = self.invoke("technology")
            exported = self.invoke("technology", "--out", target)
            restored = self.invoke("technology", "--file", target)
            self.assertEqual(original["technology"], restored["technology"])
            self.assertEqual(original["technology_hash"], exported["technology_hash"])
            before = target.read_bytes()
            self.invoke("technology", "--out", target, expected=2)
            self.assertEqual(before, target.read_bytes())

    def test_realization_is_portable_and_records_infeasible_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            network = ROOT / "examples/physical/network.json"
            spec = root / "settings.json"
            spec.write_text(json.dumps({"source_node": "x", "recipe": "passive",
                                        "source_power_mw": 1.0, "lengths_um": 1000.0}))
            result = self.invoke("realize", "--network", network, "--spec", spec,
                                 "--out-dir", root / "first")
            first = json.loads(Path(result["realization"]).read_text())
            self.assertTrue(first["evaluation"]["feasible"])
            self.assertTrue(Path(result["report"]).is_file())
            spec.unlink()
            inputs = root / "first/inputs"
            result = self.invoke("realize", "--network", inputs / "network.json",
                                 "--technology", inputs / "technology.json",
                                 "--spec", inputs / "realization.json", "--out-dir", root / "second")
            second = json.loads(Path(result["realization"]).read_text())
            self.assertEqual(first, second)
            saved = Path(result["realization"])
            self.assertTrue(self.invoke("check-realization", "--realization", saved)["valid"])
            second["evaluation"]["metrics"]["energy_pj"] = 0.0
            saved.write_text(json.dumps(second))
            self.assertFalse(self.invoke("check-realization", "--realization", saved, expected=2)["valid"])
            impossible = dict(first["spec"], source_power_mw=0.01)
            spec.write_text(json.dumps(impossible))
            result = self.invoke("realize", "--network", network, "--spec", spec,
                                 "--out-dir", root / "infeasible", expected=3)
            self.assertEqual(result["status"], "infeasible")
            self.assertIsNone(result["metrics"])
            self.assertTrue(result["diagnostics"])
            bad = dict(impossible, invented_field=True)
            spec.write_text(json.dumps(bad))
            self.invoke("realize", "--network", network, "--spec", spec,
                        "--out-dir", root / "invalid", expected=2)
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
