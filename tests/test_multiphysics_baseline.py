"""The migration gate must reject drift, damaged goldens and weakened evidence."""
from copy import deepcopy
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline_checker", ROOT / "scripts/check_multiphysics_baseline.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class MultiphysicsBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good = gate.check_baseline()

    @contextlib.contextmanager
    def frozen_copy(self):
        """Copy only the manifest's bounded data set, never production source."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads((ROOT / gate.BASELINE).read_text())
            for name in [gate.BASELINE, *manifest["files"]]:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            yield root

    def test_complete_current_software_passes_without_new_external_claim(self):
        self.assertEqual(self.good["status"], "passed", self.good)
        self.assertEqual(len(self.good["checks"]), 30)
        self.assertFalse(self.good["external_validation"]["executed_now"])
        self.assertEqual(self.good["external_validation"]["archived_evidence_integrity"], "passed")
        for case in ("designer-unlocked", "designer-locked", "designer-forbidden",
                     "designer-truncated", "designer-energy-limit", "physical-example-input-meanings"):
            self.assertTrue(any(c["case"] == case and c["status"] == "passed" for c in self.good["checks"]))

    def test_output_is_deterministic_and_never_imports_optional_solver(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"sax", "jax", "klujax"}:
                raise AssertionError(f"Core baseline gate attempted optional solver import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            self.assertEqual(gate.check_baseline(), self.good)

    def test_missing_and_changed_goldens_fail_before_runtime(self):
        mutations = (
            ("runs/demo/report.json", lambda p: p.unlink()),
            ("runs/validation-v0.3/fanout-4-passive/realization.json",
             lambda p: self.mutate_json(p, lambda d: d["evaluation"]["metrics"].update(energy_pj=0.0))),
            ("runs/validation-v0.3/baseline/realization.json",
             lambda p: self.mutate_json(p, lambda d: d["evaluation"].update(feasible=True))),
            ("runs/validation-v0.3/truncated/design-result.json",
             lambda p: self.mutate_json(p, lambda d: d["search"].update(complete=True))),
        )
        for name, mutate in mutations:
            with self.subTest(path=name), self.frozen_copy() as root:
                mutate(root / name)
                with patch.object(gate, "plan") as planner:
                    report = gate.check_baseline(root)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["checks"][0]["differences"][0]["path"], name)
                planner.assert_not_called()

    @staticmethod
    def mutate_json(path, mutate):
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))

    def test_inventory_cannot_be_weakened_by_deleting_a_fixture(self):
        with self.frozen_copy() as root:
            self.mutate_json(root / gate.BASELINE, lambda d: d["files"].pop("runs/demo/report.json"))
            report = gate.check_baseline(root)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["checks"][0]["case"], "baseline-inventory")

    def test_historical_solver_evidence_tamper_fails_without_reexecution(self):
        with self.frozen_copy() as root:
            name = "runs/validation-v0.3/fanout-2-passive/optical/simulation.json"
            self.mutate_json(root / name, lambda d: d.update(passed=False))
            report = gate.check_baseline(root)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["external_validation"]["executed_now"])

    def test_changed_runtime_metric_is_detected_against_frozen_outputs(self):
        original = gate.realize

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            if result["evaluation"]["feasible"]:
                result["evaluation"]["metrics"]["energy_pj"] *= 0.9
            return result

        with patch.object(gate, "realize", side_effect=changed):
            report = gate.check_baseline()
        self.assertEqual(report["status"], "failed")
        failed = next(c for c in report["checks"] if c["case"] == "realization-fanout-2-passive")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["differences"][0]["path"], "$.evaluation.metrics.energy_pj")

    def test_runtime_search_completeness_and_status_are_not_projected_away(self):
        original = gate.plan_physical

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            result["search"]["complete"] = not result["search"]["complete"]
            result["candidates"][0]["status"] = "feasible"
            return result

        with patch.object(gate, "plan_physical", side_effect=changed):
            report = gate.check_baseline()
        self.assertEqual(report["status"], "failed")
        for check in report["checks"]:
            if check["case"].startswith("designer-"):
                self.assertEqual(check["status"], "failed")
                self.assertTrue(any(d["path"] == "$.search.complete" for d in check["differences"]))

    def test_comparison_has_explicit_float_tolerance_but_exact_structure_and_types(self):
        self.assertFalse(gate.differences({"metric": 1.0}, {"metric": 1.0 + 1e-13}))
        for before, after in ((1.0, 1), (1, True), (True, 1), (1.0, float("nan")),
                              (1.0, float("inf")), (1.0, 1.01), ({"a": 1}, {}),
                              (["id1", "id2"], ["id2", "id1"]), ([1], [1, 2]),
                              ({"hash": "a"}, {"hash": "b"}), ("infeasible", "feasible")):
            with self.subTest(before=before, after=after):
                self.assertTrue(gate.differences(before, after))
        self.assertTrue(gate.differences(1.0, 1.0 + 1e-13, rtol=0.0, atol=0.0))

    def test_only_elapsed_value_is_excluded_and_its_contract_remains_checked(self):
        a = {"search": {"elapsed_seconds": 0.1, "evaluated": 3}}
        b = {"search": {"elapsed_seconds": 100.0, "evaluated": 3}}
        self.assertEqual(gate._without_elapsed(a), gate._without_elapsed(b))
        self.assertIn("elapsed_seconds", a["search"])
        for bad in (True, -1.0, float("nan"), None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gate._without_elapsed({"search": {"elapsed_seconds": bad}})
        with self.assertRaises(KeyError):
            gate._without_elapsed({"search": {}})

    def test_runtime_exception_returns_a_failed_case(self):
        with patch.object(gate, "plan", side_effect=RuntimeError("injected planner failure")):
            report = gate.check_baseline()
        self.assertEqual(report["status"], "failed")
        row = next(c for c in report["checks"] if c["case"] == "demo-macro-output")
        self.assertIn("injected planner failure", row["error"])

    def test_cli_saves_exact_report_and_refuses_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with patch.object(gate, "check_baseline", return_value=self.good), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gate.main(["--report", str(path)]), 0)
            self.assertEqual(json.loads(path.read_text()), self.good)
            before = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                gate.main(["--report", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_bytes(), before)
        failed = deepcopy(self.good)
        failed["status"] = "failed"
        with patch.object(gate, "check_baseline", return_value=failed), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate.main([]), 1)


if __name__ == "__main__":
    unittest.main()
