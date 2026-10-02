"""JSON-first command-line interface and reproducible run bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .checker import check_record
from .models import InputValidationError, load_json, schema_bundle, validate_inputs
from .output_contracts import output_schema_bundle
from .physics import simulate
from .planner import input_hash, plan
from .reports import save_reports


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _inputs(args) -> tuple[dict, dict, dict]:
    return validate_inputs(load_json(args.network), load_json(args.library), load_json(args.request))


def _output_directory(path: str, overwrite: bool) -> Path:
    directory = Path(path).resolve()
    if directory.exists() and (not directory.is_dir() or (any(directory.iterdir()) and not overwrite)):
        raise ValueError(f"Output directory is not empty: {directory}. Choose a new directory or use --overwrite.")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compile_run(network: dict, library: dict, request: dict, directory: Path,
                samples: int = 0) -> dict:
    """Compile, replay-check and export a run; inputs must be normalized."""
    report = plan(network, library, request)
    checks, simulations = {}, {}
    for record in report["plans"]:
        result = check_record(network, library, request, record)
        # The exported checker result stays compact; the full recomputation lives
        # in the plan and can be regenerated using the check command.
        checks[record["id"]] = {k: v for k, v in result.items() if k != "recomputed_evaluation"}
        if not result["valid"]:
            raise RuntimeError(f"Internal replay check failed: {result['diagnostics']}")
        if samples:
            simulations[record["id"]] = simulate(network, library, request, record["decision"],
                                                 samples=samples, seed=request["seed"])
    for name, value in (("network", network), ("library", library), ("request", request)):
        _write(directory / "inputs" / f"{name}.json", value)
    _write(directory / "report.json", report)
    _write(directory / "checks.json", checks)
    _write(directory / "simulations.json", simulations)
    for record in report["plans"]:
        _write(directory / "plans" / f"{record['id']}.json", record)
    save_reports(directory, report, network, checks, simulations)
    _write(directory / "manifest.json", {
        "schema_version": "0.1", "software_version": __version__,
        "input_hashes": report["input_hashes"],
        "plans": [f"plans/{record['id']}.json" for record in report["plans"]],
        "files": ["report.json", "checks.json", "simulations.json", "pareto.csv", "report.html",
                  "inputs/network.json", "inputs/library.json", "inputs/request.json"],
        "note": "This manifest identifies the current run. --overwrite may leave older, unreferenced files in the directory.",
    })
    return {"status": report["status"], "output_directory": str(directory),
            "evaluated": report["search"]["evaluated"], "feasible": report["search"]["feasible"],
            "pareto_plans": len(report["plans"]), "search_complete": report["search"]["complete"],
            "all_saved_plans_checked": all(c["valid"] for c in checks.values()),
            "simulated_plans": len(simulations),
            "simulation_consistency": all(s["consistent_with_bound"] for s in simulations.values()) if simulations else None,
            "report": str(directory / "report.html")}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="npp", description="Physics-aware neural hardware macro-plan exploration.")
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="command", required=True)
    for command, help_text in (("validate", "Validate three inputs and print their normalized hashes."),
                               ("plan", "Search, check and export a complete run bundle."),
                               ("check", "Recompute a saved plan and reject altered claims."),
                               ("simulate", "Sample primitive errors for a checked saved plan.")):
        child = commands.add_parser(command, help=help_text)
        for name in ("network", "library", "request"):
            child.add_argument(f"--{name}", required=True, metavar="JSON")
        if command == "plan":
            child.add_argument("--out-dir", required=True)
            child.add_argument("--samples", type=int, default=0, help="Monte Carlo samples per retained plan; 0 disables.")
            child.add_argument("--overwrite", action="store_true")
        if command in ("check", "simulate"):
            child.add_argument("--plan", required=True, metavar="JSON")
        if command == "simulate":
            child.add_argument("--samples", type=int, default=10000)
            child.add_argument("--seed", type=int)
    demo = commands.add_parser("demo", help="Compile the bundled affine reconvergence example.")
    demo.add_argument("--out-dir", default="runs/demo")
    demo.add_argument("--samples", type=int, default=10000)
    demo.add_argument("--overwrite", action="store_true")
    schemas = commands.add_parser("schemas", help="Export versioned JSON input and output schemas.")
    schemas.add_argument("--out-dir", required=True)
    schemas.add_argument("--overwrite", action="store_true")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "samples") and (args.samples < 0 or args.samples == 1 or
                                        (args.command == "simulate" and args.samples == 0)):
            raise ValueError("samples must be >= 2, or 0 to disable simulation in plan/demo")
        if args.command == "schemas":
            directory = _output_directory(args.out_dir, args.overwrite)
            for name, schema in {**schema_bundle(), **output_schema_bundle()}.items():
                _write(directory / f"{name}.schema.json", schema)
            _emit({"status": "ok", "output_directory": str(directory)})
            return 0
        if args.command == "demo":
            data = Path(__file__).parent / "data"
            network, library, request = validate_inputs(
                load_json(data / "split_recombine.json"), load_json(data / "components.json"),
                load_json(data / "request_default.json"))
        else:
            network, library, request = _inputs(args)
        if args.command == "validate":
            _emit({"valid": True, "schema_version": "0.1", "input_hashes": {
                "network": input_hash(network), "library": input_hash(library), "request": input_hash(request)}})
            return 0
        if args.command in ("plan", "demo"):
            directory = _output_directory(args.out_dir, args.overwrite)
            result = compile_run(network, library, request, directory, args.samples)
            _emit(result)
            return {"ok": 0, "infeasible": 3, "search_exhausted": 4}[result["status"]]
        record = load_json(args.plan)
        check = check_record(network, library, request, record)
        if args.command == "check" or not check["valid"]:
            _emit(check)
            return 0 if check["valid"] else 2
        seed = args.seed if args.seed is not None else request["seed"]
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        _emit(simulate(network, library, request, record["decision"], samples=args.samples, seed=seed))
        return 0
    except InputValidationError as exc:
        _emit({"status": "invalid_input", "diagnostics": exc.diagnostics})
        return 2
    except (OSError, ValueError, TypeError) as exc:
        _emit({"status": "invalid_input", "diagnostics": [{"code": "CLI_INPUT_ERROR", "message": str(exc)}]})
        return 2


if __name__ == "__main__":
    sys.exit(main())
