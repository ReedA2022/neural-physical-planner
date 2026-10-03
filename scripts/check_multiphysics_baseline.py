"""Check the frozen v0.3 software baseline before multiphysics migration.

This is a software compatibility gate, not independent physics validation.
It uses core dependencies only. Archived SAX results are integrity-checked,
never regenerated or presented as a newly executed external comparison.
There is deliberately no refresh/update mode.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npp.config import load_inputs, read_spec
from npp.designer import design_schema, plan_physical
from npp.implementation import implementation_schema, realization_schema, realize
from npp.models import schema_bundle
from npp.output_contracts import output_schema_bundle
from npp.planner import plan
from npp.technology import load_technology, technology_schema

BASELINE = "docs/multiphysics/baseline_v03.json"
# This digest pins the inventory itself; changing it requires an explicit review.
BASELINE_SHA256 = "d4338a72d5fd31adb57f651b336ec24cc23d3ba774d52bef6ff972464c6ad650"
RTOL = 1e-12
ATOL = 1e-15
MAX_DIFFERENCES = 20
CIRCUITS = tuple(f"fanout-{fanout}-{recipe}" for fanout in (2, 4, 8)
                 for recipe in ("passive", "regenerate")) + ("baseline",)
DESIGNS = ("unlocked", "locked", "forbidden", "truncated", "energy-limit")


def differences(expected: Any, actual: Any, *, rtol: float = RTOL,
                atol: float = ATOL) -> list[dict]:
    """Compare complete trees: exact types, keys, order and discrete values.

    Only finite float leaves receive the stated tolerance. In particular,
    bool/int/float substitutions, missing fields, IDs, hashes, statuses and
    candidate order can never be hidden by numeric coercion or projection.
    """
    result: list[dict] = []

    def add(path: str, reason: str, left: Any, right: Any) -> None:
        if len(result) < MAX_DIFFERENCES:
            result.append({"path": path, "reason": reason,
                           "expected": repr(left), "actual": repr(right)})

    def visit(left: Any, right: Any, path: str) -> None:
        if len(result) >= MAX_DIFFERENCES:
            return
        if type(left) is not type(right):
            add(path, "type_changed", type(left).__name__, type(right).__name__)
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                add(path, "keys_changed", sorted(left), sorted(right))
            for key in sorted(left.keys() & right.keys()):
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                add(path, "length_changed", len(left), len(right))
            for index, (before, after) in enumerate(zip(left, right)):
                visit(before, after, f"{path}[{index}]")
        elif isinstance(left, float):
            if not (math.isfinite(left) and math.isfinite(right)
                    and math.isclose(left, right, rel_tol=rtol, abs_tol=atol)):
                add(path, "numeric_changed", left, right)
        elif left != right:
            add(path, "value_changed", left, right)

    visit(expected, actual, "$")
    return result


def _read_json(path: Path) -> Any:
    return read_spec(path)


def _without_elapsed(record: dict) -> dict:
    """Ignore just the declared nondeterministic legacy wall-clock leaf."""
    result = deepcopy(record)
    elapsed = result["search"].pop("elapsed_seconds")
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("search.elapsed_seconds must remain a finite nonnegative float")
    return result


def check_baseline(root: Path = ROOT) -> dict:
    root = Path(root)
    report = {
        "schema_version": "npp-multiphysics-baseline-check/1",
        "status": "failed",
        "baseline_commit": "3ac1f30337753b1930fc538dc9ac58cbc1ac4c70",
        "scope": "software compatibility only; not independent semantic or physics validation",
        "comparison": {"float_relative_tolerance": RTOL, "float_absolute_tolerance": ATOL,
                       "types_and_discrete_fields": "exact", "schema_values": "exact",
                       "ignored_paths": ["legacy_macro.search.elapsed_seconds"],
                       "maximum_differences_per_check": MAX_DIFFERENCES},
        "checks": [],
        "external_validation": {"executed_now": False,
                                "archived_evidence_integrity": "not_checked",
                                "description": "Recorded v0.3 SAX comparisons only; no solver imported or executed."},
    }
    checks = report["checks"]
    try:
        raw = (root / BASELINE).read_bytes()
        if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
            raise ValueError("baseline inventory digest changed; an explicit reviewed migration is required")
        baseline = _read_json(root / BASELINE)
    except (OSError, ValueError) as exc:
        checks.append({"case": "baseline-inventory", "status": "failed", "error": str(exc)})
        return report

    bad_files = []
    for name, expected in sorted(baseline["files"].items()):
        try:
            digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if digest != expected["sha256"]:
                bad_files.append({"path": name, "reason": "digest_changed"})
        except OSError:
            bad_files.append({"path": name, "reason": "missing_or_unreadable"})
    checks.append({"case": "frozen-artifact-integrity",
                   "status": "failed" if bad_files else "passed",
                   "files_checked": len(baseline["files"]), "differences": bad_files})
    if bad_files:
        # Never use a changed or missing golden file as a new expected result.
        return report
    report["external_validation"]["archived_evidence_integrity"] = "passed"

    def compare(name: str, expected: Any, execute, *, exact: bool = False) -> None:
        try:
            actual = execute()
            changed = differences(expected, actual, rtol=0.0 if exact else RTOL,
                                  atol=0.0 if exact else ATOL)
            checks.append({"case": name, "status": "failed" if changed else "passed",
                           "differences": changed})
        except Exception as exc:
            checks.append({"case": name, "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}"})

    # Readable inputs, saved weights and canonical snapshots all keep their v0.3 meanings.
    for name, paths in (("demo", {"network_path": root / "examples/split_recombine.json",
                                 "library_path": root / "examples/components.json",
                                 "request_path": root / "examples/request_default.json"}),
                        ("friendly-v0.2", {"project_path": root / "examples/friendly/project.yaml"})):
        saved_inputs = tuple(_read_json(root / f"runs/{name}/inputs/{field}.json")
                             for field in ("network", "library", "request"))
        compare(f"{name}-input-meanings", list(saved_inputs),
                lambda paths=paths: list(load_inputs(**paths)), exact=True)
        expected = _without_elapsed(_read_json(root / f"runs/{name}/report.json"))
        compare(f"{name}-macro-output", expected,
                lambda saved_inputs=saved_inputs: _without_elapsed(plan(*saved_inputs)))

    for name in CIRCUITS:
        expected = _read_json(root / f"runs/validation-v0.3/{name}/realization.json")
        compare(f"realization-{name}", expected,
                lambda expected=expected: realize(expected["network"], expected["technology"], expected["spec"]))

    # Replay the original readable design files as well as the exact canonical snapshots.
    network = read_spec(root / "examples/physical/fanout4.json")
    technology = load_technology()
    baseline_record = _read_json(root / "runs/validation-v0.3/baseline/realization.json")
    compare("reference-technology", baseline_record["technology"], lambda: technology, exact=True)
    compare("baseline-input-meanings", baseline_record,
            lambda: realize(network, technology, read_spec(root / "examples/physical/baseline.yaml")))
    compare("physical-example-input-meanings",
            _read_json(root / "docs/multiphysics/baseline/physical_example.json"),
            lambda: realize(read_spec(root / "examples/physical/network.json"), technology,
                            read_spec(root / "examples/physical/realization.yaml")))
    design = read_spec(root / "examples/physical/design.yaml")
    locked = read_spec(root / "examples/physical/replan.yaml")
    forbidden, truncated, energy = deepcopy(locked), deepcopy(design), deepcopy(design)
    forbidden["forbidden_recipes"] = ["regenerate"]
    truncated["max_evaluations"] = 1
    energy["constraints"]["max_energy_pj"] = "0 pJ"
    requests = (design, locked, forbidden, truncated, energy)
    for name, request in zip(DESIGNS, requests):
        expected = _read_json(root / f"runs/validation-v0.3/{name}/design-result.json")
        previous = baseline_record if name in ("locked", "forbidden") else None
        compare(f"designer-{name}", expected,
                lambda request=request, previous=previous: plan_physical(network, technology, request, baseline=previous))

    schemas = {**schema_bundle(), **output_schema_bundle(), "technology": technology_schema(),
               "implementation": implementation_schema(), "realization": realization_schema(),
               "physical-design": design_schema()}
    compare("schema-inventory", baseline["schemas"], lambda: sorted(schemas), exact=True)
    for name in baseline["schemas"]:
        compare(f"schema-{name}", _read_json(root / f"schemas/{name}.schema.json"),
                lambda name=name: schemas[name], exact=True)
    report["status"] = "passed" if all(check["status"] == "passed" for check in checks) else "failed"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        help="Save a deterministic JSON report to a NEW file; never overwrite existing files.")
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        parser.error("--report must name a new file; existing files and baselines are never overwritten")
    try:
        report = check_baseline()
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.report is not None:
            with args.report.open("x", encoding="utf-8") as output:
                output.write(encoded)
        print(encoded, end="")
        return 0 if report["status"] == "passed" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
