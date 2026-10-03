"""A release harness must not turn skips, exceptions or a failed solver into PASS."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import stress_release as runner


class ReleaseRunnerTests(unittest.TestCase):
    def _run(self, result, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "evidence"
            module = SimpleNamespace(run_campaign=lambda **_: result,
                                     run_sax_checks=lambda: {"executed": True, "passed": False,
                                                            "comparison_count": 1})
            with patch.object(runner.importlib, "import_module", return_value=module):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = runner.run_release(out, seeds=[1], cases=1, areas=["inputs"], **kwargs)
            self.assertEqual(json.loads((out / "summary.json").read_text()), summary)
            return summary

    def test_required_optional_skips_are_incomplete(self):
        report = self._run({"scenario_count": 2, "failures": [], "skipped_count": 1}, require_optional=True)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual((report["passed_count"], report["skipped_count"]), (1, 1))

    def test_failed_external_comparison_preserves_executed_evidence(self):
        report = self._run({"scenario_count": 1, "failures": []}, sax=True)
        self.assertEqual(report["status"], "failed")
        self.assertIs(report["external_validation"]["executed"], True)
        self.assertEqual(report["external_validation"]["comparison_count"], 1)

    def test_malformed_or_contradictory_campaign_cannot_pass(self):
        for malformed in ({}, {"scenario_count": 1, "failures": [], "status": "failed"},
                          {"scenario_count": 1, "failures": [], "skipped_count": 2}):
            with self.subTest(malformed=malformed):
                report = self._run(malformed)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["failed_count"], 1)

    def test_changed_source_prevents_success_and_existing_directory_is_preserved(self):
        with patch.object(runner, "_source_snapshot", side_effect=[{"a": "old"}, {"a": "new"}]):
            report = self._run({"scenario_count": 1, "failures": []})
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["source_unchanged_during_run"])
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "keep"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                runner.run_release(folder, seeds=[1], cases=1, areas=["inputs"])
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_optimized_python_cannot_disable_release_assertions(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-O", str(runner.ROOT / "scripts/stress_release.py"),
                                     "--out-dir", str(Path(folder) / "evidence"),
                                     "--seeds", "1", "--cases", "1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("without -O", result.stderr)
            self.assertFalse((Path(folder) / "evidence").exists())

    def test_direct_campaigns_reject_optimized_python(self):
        for area in runner.AREAS:
            with self.subTest(area=area):
                code = f"from scripts.stress_{area} import run_campaign; run_campaign(seed=1, cases=1)"
                result = subprocess.run([sys.executable, "-O", "-c", code], cwd=runner.ROOT,
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("without -O", result.stderr)

    def test_real_campaigns_share_the_aggregate_result_contract(self):
        # Exercises the actual adapters; mocked reports cannot catch a result
        # key mismatch between a working campaign and the aggregate runner.
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                report = runner.run_release(Path(folder) / "evidence", seeds=[19], cases=1)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["scenario_count"], 30)
            self.assertEqual(report["passed_count"], 30)


if __name__ == "__main__":
    unittest.main()
