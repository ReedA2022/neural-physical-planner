"""JSON-first command-line interface and reproducible run bundles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .checker import check_record
from .config import load_inputs
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
    return load_inputs(network_path=args.network, library_path=args.library,
                       request_path=args.request, project_path=args.project)


def _available_output_directory(path: str, overwrite: bool) -> Path:
    directory = Path(path).resolve()
    if directory.exists() and (not directory.is_dir() or (any(directory.iterdir()) and not overwrite)):
        raise ValueError(f"Output directory is not empty: {directory}. Choose a new directory or use --overwrite.")
    return directory


def _output_directory(path: str, overwrite: bool) -> Path:
    directory = _available_output_directory(path, overwrite)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_physical_design(result: dict, output: str, overwrite: bool) -> dict:
    from .designer_report import render_physical_design
    from .implementation_report import render_implementation
    # The designer report replays the bounded nominal search before displaying
    # a verified comparison. The individual reports inspect the saved circuits.
    report = render_physical_design(result)
    directory = _output_directory(output, overwrite)
    _write(directory / "design-result.json", result)
    for name, value in result["inputs"].items():
        if value is not None:
            _write(directory / "inputs" / f"{name}.json", value)
    for plan_record in result["plans"]:
        target = directory / "plans" / plan_record["id"]
        _write(target / "realization.json", plan_record["realization"])
        (target / "report.html").write_text(render_implementation(plan_record["realization"]), encoding="utf-8")
    (directory / "report.html").write_text(report, encoding="utf-8")
    status = "ok" if result["plans"] else ("infeasible" if result["search"]["complete"] else "search_exhausted")
    return {"status": status, "result": str(directory / "design-result.json"),
            "report": str(directory / "report.html"), "search": result["search"],
            "plans": [{"id": plan_record["id"], "realization": str(directory / "plans" / plan_record["id"] / "realization.json")}
                      for plan_record in result["plans"]]}


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
    cli = argparse.ArgumentParser(prog="npp", description="Neural hardware planning and inspectable optical fanout subcircuits.")
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("mp", help="Opt-in multiphysics projects, models and research studies.",
                        add_help=False).add_argument("args", nargs=argparse.REMAINDER)
    for command, help_text in (("validate", "Validate a project or three input specs and print normalized hashes."),
                               ("plan", "Search, check and export a complete run bundle."),
                               ("normalize", "Resolve YAML, units and external weights to portable canonical JSON."),
                               ("check", "Recompute a saved plan and reject altered claims."),
                               ("simulate", "Sample primitive errors for a checked saved plan.")):
        child = commands.add_parser(command, help=help_text)
        child.add_argument("--project", "-p", metavar="YAML_OR_JSON",
                           help="Project file combining model, hardware and design settings.")
        for name in ("network", "library", "request"):
            child.add_argument(f"--{name}", metavar="YAML_OR_JSON",
                               help="Use all three input options together, instead of --project.")
        if command in ("plan", "normalize"):
            child.add_argument("--out-dir", required=True)
            child.add_argument("--overwrite", action="store_true")
        if command == "plan":
            child.add_argument("--samples", type=int, default=0, help="Monte Carlo samples per retained plan; 0 disables.")
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
    starter = commands.add_parser("init", help="Create a commented YAML starter project and example weight file.")
    starter.add_argument("directory")
    starter.add_argument("--overwrite", action="store_true")
    inspect = commands.add_parser("inspect-weights", help="List tensor names, dimensions and dtypes in a saved weight file.")
    inspect.add_argument("file")
    inspect.add_argument("--state-dict-key", help="Exact top-level PyTorch checkpoint key containing the state dictionary.")
    inspect.add_argument("--json", action="store_true", help="Return machine-readable JSON instead of a table.")
    importer = commands.add_parser("import-model", help="Import a supported ONNX graph into canonical network JSON.")
    importer.add_argument("file", help="Self-contained ONNX file with supported vector/dense operations.")
    importer.add_argument("--out", required=True)
    importer.add_argument("--input-min", required=True, type=float, help="Declared minimum for every input feature.")
    importer.add_argument("--input-max", required=True, type=float, help="Declared maximum for every input feature.")
    importer.add_argument("--name")
    importer.add_argument("--overwrite", action="store_true")
    technology = commands.add_parser("technology", help="Validate or export the physical technology pack and its evidence.")
    technology.add_argument("--file", help="Technology JSON/YAML; defaults to the illustrative reference pack.")
    technology.add_argument("--out", help="Write the fully normalized technology pack as JSON.")
    technology.add_argument("--overwrite", action="store_true")
    realization = commands.add_parser("realize", help="Generate and evaluate an explicit optical fanout subcircuit for a neural node.")
    model_input = realization.add_mutually_exclusive_group(required=True)
    model_input.add_argument("--project", help="Existing project JSON/YAML, including saved weight bindings.")
    model_input.add_argument("--network", help="Canonical network JSON/YAML, with concrete weights.")
    realization.add_argument("--technology", help="Physical technology pack; defaults to the illustrative reference.")
    realization.add_argument("--spec", required=True, help="Optical realization settings in JSON/YAML.")
    realization.add_argument("--out-dir", required=True)
    realization.add_argument("--overwrite", action="store_true")
    realization_check = commands.add_parser("check-realization", help="Rebuild and check every claim in a portable physical realization.")
    realization_check.add_argument("--realization", required=True, help="Saved realization.json file.")
    optical_export = commands.add_parser("export-optical", help="Export checked optical segments as a SAX netlist without running the solver.")
    optical_export.add_argument("--realization", required=True)
    optical_export.add_argument("--out", required=True)
    optical_export.add_argument("--overwrite", action="store_true")
    optical_sim = commands.add_parser("simulate-realization", help="Compare full-scale optical power with the external SAX circuit solver.")
    optical_sim.add_argument("--realization", required=True)
    optical_sim.add_argument("--wavelengths-nm", nargs="+", type=float, help="Wavelength sweep in nm; defaults to the recorded operating point.")
    optical_sim.add_argument("--rtol", type=float, default=1e-5)
    optical_sim.add_argument("--atol-mw", type=float, default=1e-9)
    optical_sim.add_argument("--out-dir", required=True)
    optical_sim.add_argument("--overwrite", action="store_true")
    design = commands.add_parser("plan-physical", help="Explore physical alternatives under designer locks and requirements.")
    design_model = design.add_mutually_exclusive_group(required=True)
    design_model.add_argument("--project", help="Project with a friendly model or external weight bindings.")
    design_model.add_argument("--network", help="Canonical network JSON/YAML.")
    design.add_argument("--technology", help="Physical technology pack; defaults to the illustrative reference.")
    design.add_argument("--design", required=True, help="Physical design grid, objectives and constraints in JSON/YAML.")
    design.add_argument("--out-dir", required=True)
    design.add_argument("--overwrite", action="store_true")
    replan = commands.add_parser("replan-physical", help="Regenerate alternatives relative to a checked physical realization.")
    replan.add_argument("--baseline", required=True, help="Baseline realization.json, including its network and technology.")
    replan.add_argument("--technology", help="Optional changed technology assumptions; otherwise use the baseline pack.")
    replan.add_argument("--design", required=True)
    replan.add_argument("--out-dir", required=True)
    replan.add_argument("--overwrite", action="store_true")
    design_check = commands.add_parser("check-physical-design", help="Recompute a saved physical search, constraints, alternatives and explanations.")
    design_check.add_argument("--result", required=True, help="Saved physical design result JSON.")
    return cli


def main(argv: list[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    if effective and effective[0] == "mp":
        from .multiphysics.cli import main as multiphysics_main
        return multiphysics_main(effective[1:])
    args = parser().parse_args(argv)
    try:
        if args.command == "check-physical-design":
            from .designer import check_physical_design
            result = check_physical_design(load_json(args.result))
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command in ("plan-physical", "replan-physical"):
            from .config import read_spec
            from .designer import plan_physical
            from .implementation import check_realization
            from .technology import load_technology
            _available_output_directory(args.out_dir, args.overwrite)
            baseline = None
            if args.command == "replan-physical":
                baseline = load_json(args.baseline)
                baseline_check = check_realization(baseline)
                if not baseline_check["valid"]:
                    raise InputValidationError(baseline_check["diagnostics"])
                network = baseline["network"]
                technology = load_technology(args.technology) if args.technology else baseline["technology"]
            else:
                network = load_inputs(project_path=args.project)[0] if args.project else read_spec(args.network)
                technology = load_technology(args.technology)
            result = plan_physical(network, technology, read_spec(args.design), baseline=baseline)
            saved = _save_physical_design(result, args.out_dir, args.overwrite)
            _emit(saved)
            return {"ok": 0, "infeasible": 3, "search_exhausted": 4}[saved["status"]]
        if args.command == "export-optical":
            from .optical_simulation import export_optical_netlist
            result = export_optical_netlist(load_json(args.realization))
            destination = Path(args.out).resolve()
            if destination.exists() and not args.overwrite:
                raise ValueError(f"Output exists: {destination}. Choose another file or use --overwrite.")
            _write(destination, result)
            _emit({"status": "exported", "netlist": str(destination),
                   "note": "Export only; no external simulation has run."})
            return 0
        if args.command == "simulate-realization":
            from .optical_simulation import simulate_realization
            from .optical_report import render_optical_simulation
            record = load_json(args.realization)
            result = simulate_realization(record, args.wavelengths_nm, rtol=args.rtol, atol_mw=args.atol_mw)
            directory = _output_directory(args.out_dir, args.overwrite)
            _write(directory / "realization.json", record)
            _write(directory / "optical-netlist.json", result["export"])
            _write(directory / "simulation.json", result)
            (directory / "report.html").write_text(render_optical_simulation(result), encoding="utf-8")
            _emit({"status": result["status"], "passed": result["passed"],
                   "simulation": str(directory / "simulation.json"),
                   "report": str(directory / "report.html"),
                   "netlist": str(directory / "optical-netlist.json"),
                   "diagnostics": result["diagnostics"]})
            return 0 if result["passed"] else 3
        if args.command == "check-realization":
            from .implementation import check_realization
            result = check_realization(load_json(args.realization))
            _emit(result)
            return 0 if result["valid"] else 2
        if args.command == "realize":
            from .config import read_spec
            from .implementation import realize
            from .implementation_report import render_implementation
            from .technology import load_technology
            network = load_inputs(project_path=args.project)[0] if args.project else read_spec(args.network)
            technology = load_technology(args.technology)
            specification = read_spec(args.spec)
            record = realize(network, technology, specification)
            directory = _output_directory(args.out_dir, args.overwrite)
            _write(directory / "realization.json", record)
            _write(directory / "inputs" / "network.json", record["network"])
            _write(directory / "inputs" / "technology.json", record["technology"])
            _write(directory / "inputs" / "realization.json", record["spec"])
            (directory / "report.html").write_text(render_implementation(record), encoding="utf-8")
            _emit({"status": "ok" if record["evaluation"]["feasible"] else "infeasible",
                   "realization": str(directory / "realization.json"),
                   "report": str(directory / "report.html"),
                   "metrics": record["evaluation"]["metrics"],
                   "diagnostics": record["evaluation"]["diagnostics"]})
            return 0 if record["evaluation"]["feasible"] else 3
        if args.command == "technology":
            from .technology import load_technology
            pack = load_technology(args.file)
            if args.out:
                destination = Path(args.out).resolve()
                if destination.exists() and not args.overwrite:
                    raise ValueError(f"Output exists: {destination}. Choose another file or use --overwrite.")
                _write(destination, pack)
                _emit({"status": "ok", "technology": str(destination), "technology_hash": input_hash(pack)})
            else:
                _emit({"status": "ok", "technology_hash": input_hash(pack), "technology": pack})
            return 0
        if args.command == "init":
            from .templates import create_project
            _emit(create_project(args.directory, overwrite=args.overwrite))
            return 0
        if args.command == "inspect-weights":
            from .weights import inspect_weights
            result = inspect_weights(args.file, state_dict_key=args.state_dict_key)
            if args.json:
                _emit(result)
            else:
                print(f"Format: {result['format']}")
                print(f"{'Tensor name':<48} {'Shape':<20} Dtype")
                for tensor in result["tensors"]:
                    shape = " × ".join(str(d) for d in tensor["shape"]) or "scalar"
                    print(f"{tensor['name']:<48} {shape:<20} {tensor['dtype']}")
                print("\nUse these exact tensor names in model.yaml. See docs/INPUT_GUIDE.md for weight layouts.")
            return 0
        if args.command == "import-model":
            from .onnx_import import import_onnx
            if Path(args.file).suffix.lower() != ".onnx":
                raise ValueError("import-model accepts ONNX graphs. For weights-only files, use inspect-weights and a model YAML description.")
            output = Path(args.out).resolve()
            if output.exists() and not args.overwrite:
                raise ValueError(f"Output exists: {output}. Choose another file or use --overwrite.")
            network = import_onnx(args.file, input_bounds=(args.input_min, args.input_max), name=args.name)
            _write(output, network)
            _emit({"status": "ok", "network": str(output), "nodes": len(network["nodes"]),
                   "outputs": network["outputs"], "input_bounds": [args.input_min, args.input_max]})
            return 0
        if hasattr(args, "samples") and (args.samples < 0 or args.samples == 1 or
                                        (args.command == "simulate" and args.samples == 0)):
            raise ValueError("samples must be >= 2, or 0 to disable simulation in plan/demo")
        if args.command == "schemas":
            from .technology import technology_schema
            from .implementation import implementation_schema, realization_schema
            from .designer import design_schema
            directory = _output_directory(args.out_dir, args.overwrite)
            for name, schema in {**schema_bundle(), **output_schema_bundle(), "technology": technology_schema(),
                                 "implementation": implementation_schema(), "realization": realization_schema(),
                                 "physical-design": design_schema()}.items():
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
        if args.command == "normalize":
            directory = _output_directory(args.out_dir, args.overwrite)
            for name, value in (("network", network), ("library", library), ("request", request)):
                _write(directory / f"{name}.json", value)
            _emit({"status": "ok", "output_directory": str(directory),
                   "files": ["network.json", "library.json", "request.json"],
                   "note": "Resolved tensors are embedded; these files no longer depend on the source weight files."})
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
