import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.fixtures import search_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliIntegrationTests(unittest.TestCase):
    def invoke(self, *arguments, expected_exit=0):
        process = subprocess.run([sys.executable, "-m", "npp", *map(str, arguments)],
                                 cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=60)
        self.assertEqual(process.returncode, expected_exit, process.stdout + process.stderr)
        return json.loads(process.stdout)

    def test_demo_exports_replayable_plan_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="npp-cli-test-") as temporary:
            output = Path(temporary) / "demo"
            result = self.invoke("demo", "--out-dir", output, "--samples", 100)
            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["pareto_plans"], 0)
            self.assertTrue(result["all_saved_plans_checked"])
            self.assertEqual(result["simulated_plans"], result["pareto_plans"])
            manifest = json.loads((output / "manifest.json").read_text())
            for relative in manifest["files"] + manifest["plans"]:
                self.assertTrue((output / relative).is_file(), relative)
            saved_plan = output / manifest["plans"][0]
            args = ["--network", output / "inputs/network.json",
                    "--library", output / "inputs/library.json",
                    "--request", output / "inputs/request.json", "--plan", saved_plan]
            checked = self.invoke("check", *args)
            self.assertTrue(checked["valid"], checked["diagnostics"])
            record = json.loads(saved_plan.read_text())
            record["evaluation"]["metrics"]["energy_pj"] += 1.0
            saved_plan.write_text(json.dumps(record), encoding="utf-8")
            rejected = self.invoke("check", *args, expected_exit=2)
            self.assertFalse(rejected["valid"])
            self.assertTrue(rejected["diagnostics"])

    def test_complete_impossible_request_exports_report_and_exit_three(self):
        with tempfile.TemporaryDirectory(prefix="npp-cli-test-") as temporary:
            directory = Path(temporary)
            net, lib, req = search_fixture()
            req["budgets"] = {"energy_pj": 0.0}
            args = []
            for key, value in (("network", net), ("library", lib), ("request", req)):
                destination = directory / f"{key}.json"
                destination.write_text(json.dumps(value), encoding="utf-8")
                args.extend(["--" + key, destination])
            output = directory / "impossible"
            result = self.invoke("plan", *args, "--out-dir", output, expected_exit=3)
            self.assertEqual(result["status"], "infeasible")
            self.assertTrue(result["search_complete"])
            self.assertEqual(result["pareto_plans"], 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "infeasible")
            self.assertEqual(report["plans"], [])


if __name__ == "__main__":
    unittest.main()
