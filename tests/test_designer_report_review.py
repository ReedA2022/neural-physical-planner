"""Independent Gate 4 checks of report trust boundaries and CLI search status."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from npp.config import read_spec
from npp.designer import plan_physical
from npp.designer_report import render_physical_design
from npp.technology import load_technology

ROOT = Path(__file__).resolve().parents[1]


class DesignerReportReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = plan_physical(read_spec(ROOT / "examples/physical/fanout4.json"),
                                  load_technology(), read_spec(ROOT / "examples/physical/design.yaml"))

    def test_stale_plans_and_rejection_explanations_cannot_keep_verified_badge(self):
        for field in ("plans", "rejections", "search"):
            record = deepcopy(self.result)
            if field == "plans":
                record["plans"] = record["plans"] + record["plans"][:1]
            elif field == "rejections":
                rejected = next(row for row in record["candidates"] if row["reasons"])
                rejected["reasons"][0]["message"] = "An invented causal explanation"
            else:
                record["search"]["complete"] = False
            with self.subTest(field=field):
                report = render_physical_design(record)
                self.assertIn("Incomplete, inconsistent or unsupported design record", report)
                self.assertNotIn("Nominal design search replay passed", report)
                self.assertNotIn('<svg ', report)
                self.assertNotIn('<a href=', report)

    def test_recorded_nonfinite_values_stay_unverified_and_inspectable(self):
        for value in (True, float("nan"), float("inf"), 10 ** 400):
            record = deepcopy(self.result)
            candidate = next(row for row in record["candidates"] if row["metrics"] is not None)
            candidate["metrics"]["energy_pj"] = value
            with self.subTest(value_type=type(value).__name__):
                report = render_physical_design(record)
                self.assertIn("Incomplete, inconsistent or unsupported design record", report)
                self.assertNotIn("Nominal design search replay passed", report)
                self.assertNotIn('<svg ', report)

    def test_truncated_example_with_no_plan_is_search_exhausted_and_portable(self):
        with tempfile.TemporaryDirectory(prefix="npp gate4 review ") as temporary:
            root = Path(temporary)
            design = read_spec(ROOT / "examples/physical/design.yaml")
            design["max_evaluations"] = 1
            spec = root / "design with spaces.json"
            spec.write_text(json.dumps(design), encoding="utf-8")
            process = subprocess.run([sys.executable, "-m", "npp", "plan-physical", "--network",
                                      str(ROOT / "examples/physical/fanout4.json"), "--design", str(spec),
                                      "--out-dir", str(root / "result with spaces")], cwd=ROOT,
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 4, process.stdout + process.stderr)
            response = json.loads(process.stdout)
            self.assertEqual(response["status"], "search_exhausted")
            self.assertFalse(response["search"]["complete"])
            self.assertFalse(response["plans"])
            spec.unlink()
            check = subprocess.run([sys.executable, "-m", "npp", "check-physical-design", "--result",
                                    response["result"]], cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            self.assertTrue(json.loads(check.stdout)["valid"])
            report = Path(response["report"]).read_text(encoding="utf-8")
            self.assertIn("Incomplete search of the submitted finite grid.", report)
            self.assertIn("No feasible candidate in the evaluated search", report)
            self.assertNotIn("Complete for the submitted finite grid.", report)


if __name__ == "__main__":
    unittest.main()
