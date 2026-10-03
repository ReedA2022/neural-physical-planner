"""Run bounded, reproducible release stress campaigns and preserve every outcome.

This is a source-checkout tool. Optional framework cases report explicit skips;
--require-optional makes any such skip fail the release gate. --sax requests an
actual external optical-power comparison, never a fallback simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from npp import __version__

AREAS = ("inputs", "legacy", "physical", "designer", "reports")


def _source_snapshot():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in ("npp/**/*.py", "npp/data/*.json", "scripts/stress_*.py")
            for path in sorted(ROOT.glob(pattern))}


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_release(directory, *, seeds, cases=400, areas=AREAS,
                require_optional=False, sax=False):
    """Create a new evidence directory; return the aggregate recorded outcome."""
    if not __debug__:
        raise ValueError("Stress campaigns require assertions; run Python without -O or -OO.")
    if type(cases) is not int or not 1 <= cases <= 2000:
        raise ValueError("cases must be an integer between 1 and 2000")
    if (not isinstance(seeds, (list, tuple)) or not 1 <= len(seeds) <= 5
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("choose 1–5 unique nonnegative 32-bit seeds")
    if not areas or any(area not in AREAS for area in areas) or len(set(areas)) != len(areas):
        raise ValueError("choose unique supported campaign areas")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source_hashes = _source_snapshot()
    source_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    started = time.monotonic()
    summary = {
        "schema_version": "npp-release-stress/1", "software_version": __version__,
        "python_version": platform.python_version(), "checkout_commit": commit,
        "evaluated_source_sha256": source_digest, "source_hashes": source_hashes,
        "seeds": list(seeds), "requested_cases_per_area_seed": cases,
        "areas": list(areas), "require_optional": require_optional,
        "runs": [], "scenario_count": 0, "passed_count": 0,
        "failed_count": 0, "skipped_count": 0, "status": "running",
        "external_validation": {"executed": False},
        "scope": "Software and declared-model research-prototype release checks; no hardware calibration, fabrication signoff or general neural physical synthesis.",
    }
    for seed in seeds:
        for area in areas:
            filename = f"{area}-{seed}.json"
            returned_result = None
            try:
                result = importlib.import_module(f"stress_{area}").run_campaign(seed=seed, cases=cases)
                returned_result = result
                count = result["scenario_count"]
                failures, skipped = result["failures"], result.get("skipped_count", 0)
                if (type(count) is not int or count <= 0 or not isinstance(failures, list)
                        or type(skipped) is not int or skipped < 0
                        or len(failures) + skipped > count):
                    raise ValueError("campaign returned inconsistent outcome counts")
                # A script may expose its status as a Boolean or a word. Neither
                # is allowed to disagree with the recorded failures.
                if result.get("passed") is False or result.get("status") == "failed":
                    if not failures:
                        raise ValueError("campaign declared failure without recording a failing scenario")
            except Exception as exc:
                result = {"campaign": area, "seed": seed, "scenario_count": 1,
                          "failures": [{"kind": "campaign_execution_error", "exception": type(exc).__name__,
                                        "message": str(exc)}], "skipped_count": 0}
                if returned_result is not None:
                    result["returned_result"] = returned_result
                count, failures, skipped = 1, result["failures"], 0
            _save(directory / filename, result)
            row = {"area": area, "seed": seed, "scenario_count": count,
                   "passed_count": count - len(failures) - skipped,
                   "failed_count": len(failures), "skipped_count": skipped, "report": filename}
            summary["runs"].append(row)
            for key in ("scenario_count", "passed_count", "failed_count", "skipped_count"):
                summary[key] += row[key]
            print(json.dumps(row), flush=True)
    if sax:
        try:
            result = importlib.import_module("stress_physical").run_sax_checks()
            if result.get("executed") is not True or type(result.get("passed")) is not bool:
                raise ValueError("requested SAX comparison returned no completed comparison outcome")
            summary["external_validation"] = result
        except Exception as exc:
            summary["external_validation"] = {"executed": False, "attempted": True, "passed": False,
                                               "exception": type(exc).__name__, "message": str(exc)}
        _save(directory / "external-sax.json", summary["external_validation"])
    summary["source_unchanged_during_run"] = _source_snapshot() == source_hashes
    failed = (summary["failed_count"] > 0 or summary["external_validation"].get("passed") is False
              or not summary["source_unchanged_during_run"])
    summary["status"] = ("failed" if failed else "incomplete" if require_optional and summary["skipped_count"] else "passed")
    summary["elapsed_seconds"] = time.monotonic() - started
    _save(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="New evidence directory; existing paths are never overwritten.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261003, 731, 987654321])
    parser.add_argument("--cases", type=int, default=400)
    parser.add_argument("--areas", choices=AREAS, nargs="+", default=list(AREAS))
    parser.add_argument("--require-optional", action="store_true")
    parser.add_argument("--sax", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_release(args.out_dir, seeds=args.seeds, cases=args.cases,
                             areas=args.areas, require_optional=args.require_optional, sax=args.sax)
        print(json.dumps({key: result[key] for key in ("status", "scenario_count", "passed_count", "failed_count", "skipped_count")}, indent=2))
        return {"passed": 0, "failed": 1, "incomplete": 2}[result["status"]]
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "invalid_request", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
