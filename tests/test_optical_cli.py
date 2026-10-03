"""External-solver workflow and optional-dependency CLI boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from npp.implementation import realize
from npp.technology import load_technology


ROOT = Path(__file__).resolve().parents[1]
HAS_SAX = importlib.util.find_spec("sax") is not None


class OpticalCliTests(unittest.TestCase):
    def invoke(self, *arguments, expected=0):
        process = subprocess.run([sys.executable, "-m", "npp", *map(str, arguments)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return json.loads(process.stdout)

    def save_record(self, root):
        network = json.loads((ROOT / "examples/physical/network.json").read_text())
        record = realize(network, load_technology(), {"source_node": "x", "source_power_mw": 1.0})
        target = root / "realization.json"
        target.write_text(json.dumps(record))
        return target

    def test_portable_export_and_tampering_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.save_record(root)
            target = root / "export.json"
            result = self.invoke("export-optical", "--realization", source, "--out", target)
            self.assertEqual(result["status"], "exported")
            saved = target.read_bytes()
            netlist = json.loads(saved)
            self.assertEqual(netlist["kind"], "sax_optical_export")
            self.assertTrue(netlist["segments"])
            self.invoke("export-optical", "--realization", source, "--out", target, expected=2)
            self.assertEqual(saved, target.read_bytes())
            record = json.loads(source.read_text())
            record["evaluation"]["metrics"]["energy_pj"] = 0.0
            source.write_text(json.dumps(record))
            self.invoke("export-optical", "--realization", source, "--out", root / "bad.json", expected=2)
            self.assertFalse((root / "bad.json").exists())

    @unittest.skipUnless(HAS_SAX, "SAX optional dependency is not installed")
    def test_actual_solver_sweep_and_invalid_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.save_record(root)
            output = root / "simulation"
            result = self.invoke("simulate-realization", "--realization", source,
                                 "--wavelengths-nm", 1540, 1550, 1560, "--out-dir", output)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["passed"])
            simulation = json.loads(Path(result["simulation"]).read_text())
            self.assertEqual(len(simulation["samples"]), 3)
            self.assertTrue((output / "optical-netlist.json").is_file())
            self.assertTrue(Path(result["report"]).is_file())
            self.assertEqual(json.loads((output / "realization.json").read_text()), json.loads(source.read_text()))
            self.invoke("simulate-realization", "--realization", source,
                        "--wavelengths-nm", 1600, "--out-dir", root / "outside", expected=2)
            self.assertFalse((root / "outside").exists())

    @unittest.skipIf(HAS_SAX, "This check exercises an installation without SAX")
    def test_missing_solver_returns_actionable_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.save_record(root)
            result = self.invoke("simulate-realization", "--realization", source,
                                 "--out-dir", root / "simulation", expected=2)
            self.assertEqual(result["status"], "invalid_input")
            self.assertTrue(any(d["code"] == "optional_dependency_missing" for d in result["diagnostics"]))
            self.assertIn("photonics", json.dumps(result))
            self.assertFalse((root / "simulation").exists())


if __name__ == "__main__":
    unittest.main()
