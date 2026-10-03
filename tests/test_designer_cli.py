"""Designer-facing workflows using portable physical records and readable units."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from npp.implementation import realize
from npp.technology import load_technology
from tests.fixtures import network, node


ROOT = Path(__file__).resolve().parents[1]


class DesignerCliTests(unittest.TestCase):
    def invoke(self, *arguments, expected=0):
        process = subprocess.run([sys.executable, "-m", "npp", *map(str, arguments)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return json.loads(process.stdout)

    def fixture(self, directory):
        root = Path(directory)
        net = network([node("x", "input", input_bounds={"lower": [0.0], "upper": [1.0]})], ["x"] * 4)
        net_path = root / "network.json"
        net_path.write_text(json.dumps(net))
        baseline = realize(net, load_technology(), {"source_node": "x", "source_power_mw": 0.05,
                                                     "lengths_um": 10000.0})
        baseline_path = root / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))
        design = {"source_node": "x", "grid": {
            "recipes": ["passive", "regenerate"],
            "source_powers_mw": ["50 uW", "100 uW"],
            "regeneration_powers_mw": ["100 uW", "200 uW"],
            "route_lengths_um": ["10 mm"]},
            "objectives": ["energy_pj", "noise_rms_estimate"],
            "locks": {"source_power_mw": "baseline"}}
        design_path = root / "design.json"
        design_path.write_text(json.dumps(design))
        return root, net_path, baseline_path, design_path, design

    def test_locked_replan_is_portable_and_saves_inspectable_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, baseline, design_path, _ = self.fixture(directory)
            result = self.invoke("replan-physical", "--baseline", baseline, "--design", design_path,
                                 "--out-dir", root / "replan")
            self.assertTrue(result["search"]["complete"])
            self.assertTrue(result["plans"])
            saved = Path(result["result"])
            raw = json.loads(saved.read_text())
            for item in result["plans"]:
                realization = json.loads(Path(item["realization"]).read_text())
                self.assertEqual(realization["spec"]["source_power_mw"], 0.05)
                self.assertEqual(realization["spec"]["recipe"], "regenerate")
                self.assertTrue(realization["evaluation"]["feasible"])
                self.assertTrue(Path(item["realization"]).with_name("report.html").is_file())
            self.assertTrue(Path(result["report"]).is_file())
            before = saved.read_bytes()
            self.invoke("replan-physical", "--baseline", baseline, "--design", design_path,
                        "--out-dir", root / "replan", expected=2)
            self.assertEqual(saved.read_bytes(), before)
            baseline.unlink()
            design_path.unlink()
            self.assertTrue(self.invoke("check-physical-design", "--result", saved)["valid"])
            raw["search"]["feasible_count"] = 999
            saved.write_text(json.dumps(raw))
            self.assertFalse(self.invoke("check-physical-design", "--result", saved, expected=2)["valid"])

    def test_unlocked_plan_and_infeasible_or_invalid_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root, net, baseline, design_path, design = self.fixture(directory)
            design.pop("locks")
            design_path.write_text(json.dumps(design))
            result = self.invoke("plan-physical", "--network", net, "--design", design_path,
                                 "--out-dir", root / "unlocked")
            records = [json.loads(Path(item["realization"]).read_text()) for item in result["plans"]]
            self.assertTrue(any(r["spec"]["recipe"] == "passive" for r in records))
            impossible = deepcopy(design)
            impossible["constraints"] = {"max_energy_pj": "0 pJ"}
            design_path.write_text(json.dumps(impossible))
            result = self.invoke("plan-physical", "--network", net, "--design", design_path,
                                 "--out-dir", root / "infeasible", expected=3)
            self.assertEqual(result["status"], "infeasible")
            self.assertTrue(result["search"]["complete"])
            self.assertFalse(result["plans"])
            self.assertTrue(self.invoke("check-physical-design", "--result", result["result"])["valid"])
            impossible["unknown_option"] = True
            design_path.write_text(json.dumps(impossible))
            self.invoke("plan-physical", "--network", net, "--design", design_path,
                        "--out-dir", root / "invalid", expected=2)
            self.assertFalse((root / "invalid").exists())
            altered = json.loads(baseline.read_text())
            altered["spec"]["source_power_mw"] = 1.0
            baseline.write_text(json.dumps(altered))
            self.invoke("replan-physical", "--baseline", baseline, "--design", design_path,
                        "--out-dir", root / "bad-baseline", expected=2)
            self.assertFalse((root / "bad-baseline").exists())


if __name__ == "__main__":
    unittest.main()
