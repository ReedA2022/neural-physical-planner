"""Bounded report-corruption and real CLI workflow stress (no solver required).

Stored optical fixtures are rendered, not re-simulated. The CLI cases run actual
normalization, search, replay and small Monte Carlo jobs in temporary directories.
"""
from __future__ import annotations

from collections import Counter
import contextlib
from copy import deepcopy
import html
import io
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npp.cli import main as cli_main
from npp.designer_report import render_physical_design
from npp.implementation_report import render_implementation
from npp.optical_report import render_optical_simulation


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _paths(value, path=()):
    yield path
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _paths(item, path + (key,))
    elif isinstance(value, list):
        # Bound the mutation inventory independently of saved result size.
        for index, item in enumerate(value[:3]):
            yield from _paths(item, path + (index,))


def _replace(value, path, replacement):
    if not path:
        return replacement
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return value


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def _cli_campaign(root):
    results, failures = [], []

    def invoke(label, arguments, expected_code=0, expected_status=None, verify=None):
        output, errors = io.StringIO(), io.StringIO()
        argv = [str(a) for a in arguments]
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                code = cli_main(argv)
            _require(code == expected_code, f"exit {code} != {expected_code}: {output.getvalue()[:500]}")
            result = json.loads(output.getvalue())
            if expected_status is not None:
                _require(result.get("status") == expected_status, f"wrong CLI status: {result}")
            if code == 2:
                _require(result.get("valid") is False or result.get("status") == "invalid_input", "invalid command was not labeled invalid")
                _require(bool(result.get("diagnostics")), "invalid command omitted diagnostics")
            if verify:
                verify(result)
            results.append({"scenario": label, "passed": True, "exit_code": code})
            return result
        except Exception as error:
            failure = {"family": "cli", "scenario": label, "error_type": type(error).__name__, "message": str(error),
                       "reproduction": {"argv": argv, "stdout": output.getvalue()[:1000], "stderr": errors.getvalue()[:1000]}}
            failures.append(failure)
            results.append({"scenario": label, "passed": False})
            return None

    project = root / "Projet espace été"
    invoke("init_unicode_spaces", ["init", project], expected_status="ok")
    original_files = {path.name: path.read_bytes() for path in project.iterdir() if path.is_file()}
    invoke("init_refuses_overwrite", ["init", project], 2, "invalid_input",
           lambda _: _require(all((project / name).read_bytes() == data for name, data in original_files.items()), "init changed existing files"))
    invoke("inspect_npz", ["inspect-weights", project / "weights.npz", "--json"],
           verify=lambda r: _require(len(r["tensors"]) == 4, "starter tensors missing"))
    invoke("validate_friendly", ["validate", "--project", project / "project.yaml"], verify=lambda r: _require(r["valid"], "starter validation"))
    normalized = root / "portable inputs"
    invoke("normalize_external_weights", ["normalize", "--project", project / "project.yaml", "--out-dir", normalized], expected_status="ok")
    saved_network = (normalized / "network.json").read_bytes()
    invoke("normalize_refuses_overwrite", ["normalize", "--project", project / "project.yaml", "--out-dir", normalized], 2, "invalid_input",
           lambda _: _require((normalized / "network.json").read_bytes() == saved_network, "normalize changed prior network"))
    (project / "weights.npz").unlink()
    inputs = ["--network", normalized / "network.json", "--library", normalized / "library.json", "--request", normalized / "request.json"]
    invoke("portable_after_weights_removed", ["validate", *inputs], verify=lambda r: _require(r["valid"], "normalized inputs retained external dependency"))
    # Keep the real macro search tiny; this tests the workflow, not its scale.
    request = json.loads((normalized / "request.json").read_text())
    request.update(allowed_domains=["digital"], allow_serialization=False)
    request["search"] = {"mode": "exhaustive", "max_evaluations": 32, "beam_width": 8}
    _write(normalized / "request.json", request)
    macro_run = root / "macro run"
    invoke("macro_plan_with_samples", ["plan", *inputs, "--out-dir", macro_run, "--samples", 8], expected_status="ok",
           verify=lambda r: _require(r["all_saved_plans_checked"] and r["simulated_plans"] > 0, "saved plans not checked/simulated"))
    macro_plan = next((macro_run / "plans").glob("*.json"), macro_run / "missing.json")
    invoke("macro_replay", ["check", *inputs, "--plan", macro_plan], verify=lambda r: _require(r["valid"], "macro replay failed"))
    invoke("macro_simulate_small", ["simulate", *inputs, "--plan", macro_plan, "--samples", 8, "--seed", 17])
    invoke("simulate_rejects_one_sample", ["simulate", *inputs, "--plan", macro_plan, "--samples", 1], 2, "invalid_input")
    invoke("plan_rejects_negative_samples", ["plan", *inputs, "--out-dir", root / "invalid samples", "--samples", -1], 2, "invalid_input")
    bad = root / "bad saved.json"
    bad.write_text('{"broken": [}', encoding="utf-8")
    invoke("malformed_saved_json", ["check-realization", "--realization", bad], 2, "invalid_input")

    network = ROOT / "examples/physical/fanout4.json"
    specification = _write(root / "realization spec.json", {"source_node": "x", "source_power_mw": 0.2, "lengths_um": 0.0})
    physical = root / "physical feasible"
    invoke("physical_feasible", ["realize", "--network", network, "--spec", specification, "--out-dir", physical], expected_status="ok")
    record = physical / "realization.json"
    invoke("physical_replay", ["check-realization", "--realization", record], verify=lambda r: _require(r["valid"], "physical replay failed"))
    _write(specification, {"source_node": "x", "source_power_mw": 0.01, "lengths_um": 10000.0})
    infeasible = root / "physical infeasible"
    invoke("physical_infeasible", ["realize", "--network", network, "--spec", specification, "--out-dir", infeasible], 3, "infeasible",
           lambda r: _require(r["metrics"] is None and bool(r["diagnostics"]), "infeasible circuit invented totals"))
    invoke("infeasible_record_replays_honestly", ["check-realization", "--realization", infeasible / "realization.json"], verify=lambda r: _require(r["valid"], "legitimate infeasible record should replay"))
    corrupted = json.loads(record.read_text())
    corrupted["evaluation"]["metrics"]["energy_pj"] = 0.0
    _write(root / "tampered.json", corrupted)
    invoke("tampered_physical_rejected", ["check-realization", "--realization", root / "tampered.json"], 2)
    optical = root / "optical netlist.json"
    invoke("optical_export_no_solver", ["export-optical", "--realization", record, "--out", optical], expected_status="exported")
    previous_export = optical.read_bytes()
    invoke("optical_export_refuses_overwrite", ["export-optical", "--realization", record, "--out", optical], 2, "invalid_input",
           lambda _: _require(optical.read_bytes() == previous_export, "netlist overwrite was not prevented"))

    design = {"source_node": "x", "grid": {"recipes": ["passive"], "source_powers_mw": [0.2], "route_lengths_um": [0.0]}}
    design_file = _write(root / "design.json", design)
    designer_run = root / "designer feasible"
    invoke("designer_feasible", ["plan-physical", "--network", network, "--design", design_file, "--out-dir", designer_run], expected_status="ok")
    invoke("designer_replay", ["check-physical-design", "--result", designer_run / "design-result.json"], verify=lambda r: _require(r["valid"], "designer replay failed"))
    previous_design = (designer_run / "design-result.json").read_bytes()
    invoke("designer_refuses_overwrite", ["plan-physical", "--network", network, "--design", design_file, "--out-dir", designer_run], 2, "invalid_input",
           lambda _: _require((designer_run / "design-result.json").read_bytes() == previous_design, "designer overwrite changed result"))
    design["grid"]["source_powers_mw"] = [0.01]
    _write(design_file, design)
    invoke("designer_complete_infeasible", ["plan-physical", "--network", network, "--design", design_file, "--out-dir", root / "designer infeasible"], 3, "infeasible",
           lambda r: _require(r["search"]["complete"] and not r["plans"], "complete empty grid not identified"))
    design["grid"]["source_powers_mw"] = [0.01, 0.2]
    design["max_evaluations"] = 1
    _write(design_file, design)
    invoke("designer_truncated_empty", ["plan-physical", "--network", network, "--design", design_file, "--out-dir", root / "designer truncated"], 4, "search_exhausted",
           lambda r: _require(not r["search"]["complete"] and r["search"]["not_evaluated_count"] == 1, "truncation incorrectly claimed complete"))
    return results, failures


def run_campaign(seed=9237, cases=300):
    if not __debug__:
        raise ValueError("Stress campaigns require assertions; run Python without -O or -OO.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(cases, bool) or not isinstance(cases, int) or not 1 <= cases <= 2000:
        raise ValueError("cases must be an integer from 1 to 2000")
    started = time.monotonic()
    fixtures = [
        ("implementation", render_implementation, "runs/validation-v0.3/fanout-2-passive/realization.json",
         ("evaluation", "metrics"), ("network", "name"), "Recorded feasible within the declared model"),
        ("optical", render_optical_simulation, "runs/validation-v0.3/fanout-2-passive/optical/simulation.json",
         ("summary",), ("engine", "name"), "Recorded optical-power agreement within tolerance"),
        ("designer", render_physical_design, "runs/validation-v0.3/unlocked/design-result.json",
         ("search",), ("inputs", "network", "name"), "Nominal design search replay passed"),
    ]
    corpus = []
    for name, renderer, file, critical_path, injection_path, success in fixtures:
        value = json.loads((ROOT / file).read_text(encoding="utf-8"))
        corpus.append((name, renderer, file, value, list(_paths(value)), critical_path, injection_path, success))
    counts, modes, failures = Counter(), Counter(), []
    for index in range(cases):
        name, renderer, file, good, paths, critical, injection, success = corpus[index % 3]
        ordinal, rng = index // 3, random.Random(seed + 104729 * index)
        mode = "missing_required_fields" if ordinal % 5 == 0 else "html_injection" if ordinal % 5 == 1 else "nested_type_corruption"
        if mode == "missing_required_fields":
            path, replacement = critical, None
        elif mode == "html_injection":
            path, replacement = injection, '<img src=x onerror="alert(1)"><script>stress_probe</script>&\'"'
        else:
            path, replacement = rng.choice(paths), rng.choice([None, True, False, [], {}, "invalid", -1, 10 ** 400, 1e308])
        mutated = _replace(deepcopy(good), path, replacement)
        # All designer alterations must fail replay; avoid accidentally reusing a value.
        if mutated == good:
            replacement = "deliberately-different-stress-value"
            mutated = _replace(deepcopy(good), path, replacement)
        counts[name] += 1
        modes[mode] += 1
        try:
            output = renderer(mutated)
            _require(isinstance(output, str) and output.startswith("<!doctype html>"), "renderer omitted complete HTML document")
            if mode == "missing_required_fields" or name == "designer":
                _require(success not in output, "malformed report was promoted to success")
            if mode == "html_injection":
                _require(replacement not in output and html.escape(replacement, quote=True) in output, "HTML payload was not safely escaped")
            _require(not re.search(r'(?:cx|cy|x|y)="(?:nan|inf|-inf)"', output, re.I), "SVG contains nonfinite coordinates")
        except Exception as error:
            failures.append({"family": name, "scenario_index": index, "error_type": type(error).__name__, "message": str(error),
                             "reproduction": {"seed": seed, "fixture": file, "mode": mode, "path": list(path), "replacement": replacement}})
    with tempfile.TemporaryDirectory(prefix="npp-stress-reports-") as folder:
        cli_results, cli_failures = _cli_campaign(Path(folder))
    failures.extend(cli_failures)
    return {"campaign": "reports_and_cli", "seed": seed, "scenario_count": cases + len(cli_results),
            "report_scenario_count": cases, "cli_scenario_count": len(cli_results),
            "family_counts": {**dict(counts), "cli": len(cli_results)}, "report_mutation_counts": dict(modes),
            "cli_results": cli_results, "failures": failures, "passed": not failures,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "scope": "Stored report robustness and actual local CLI workflows. Optical fixture rendering is not a new independent solver comparison; no external solver was run."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9237)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run_campaign(args.seed, args.cases)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
