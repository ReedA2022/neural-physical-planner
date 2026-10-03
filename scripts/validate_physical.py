"""Reproduce bounded v0.3 integration cases with actual external SAX checks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npp import __version__
from npp.cli import _output_directory, _save_physical_design, _write
from npp.config import read_spec
from npp.designer import check_physical_design, plan_physical
from npp.implementation import check_realization, realize
from npp.implementation_report import render_implementation
from npp.optical_report import render_optical_simulation
from npp.optical_simulation import simulate_realization
from npp.technology import load_technology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    directory = _output_directory(args.out_dir, args.overwrite)
    technology = load_technology()
    wavelengths = [1540.0, 1545.0, 1550.0, 1555.0, 1560.0]
    summary = {"software_version": __version__, "scope": "illustrative full-scale optical subcircuits; no physical calibration or NN accuracy claim",
               "wavelengths_nm": wavelengths, "circuits": [], "designs": []}

    def save_circuit(record, target, label):
        assert check_realization(record)["valid"], label
        assert record["evaluation"]["feasible"], label
        comparison = simulate_realization(record, wavelengths)
        assert comparison["passed"], (label, comparison["diagnostics"])
        _write(target / "realization.json", record)
        (target / "report.html").write_text(render_implementation(record), encoding="utf-8")
        _write(target / "optical" / "simulation.json", comparison)
        _write(target / "optical" / "netlist.json", comparison["export"])
        (target / "optical" / "report.html").write_text(render_optical_simulation(comparison), encoding="utf-8")
        summary["circuits"].append({"case": label, "metrics": record["evaluation"]["metrics"],
                                    "engine": comparison["engine"], **comparison["summary"]})

    for fanout in (2, 4, 8):
        network = {"schema_version": "0.1", "name": f"validation_fanout_{fanout}",
                   "nodes": [{"id": "x", "op": "input", "inputs": [], "size": 1,
                              "input_bounds": {"lower": [0.0], "upper": [1.0]}}],
                   "outputs": ["x"] * fanout}
        for recipe in ("passive", "regenerate"):
            spec = {"source_node": "x", "recipe": recipe, "source_power_mw": 1.0,
                    "lengths_um": [250.0 + 750.0 * i for i in range(fanout)]}
            if recipe == "regenerate":
                spec["regeneration_power_mw"] = 0.2
            label = f"fanout-{fanout}-{recipe}"
            save_circuit(realize(network, technology, spec), directory / label, label)

    network = read_spec(ROOT / "examples/physical/fanout4.json")
    design = read_spec(ROOT / "examples/physical/design.yaml")
    baseline = realize(network, technology, read_spec(ROOT / "examples/physical/baseline.yaml"))
    assert check_realization(baseline)["valid"] and not baseline["evaluation"]["feasible"]
    _write(directory / "baseline" / "realization.json", baseline)
    (directory / "baseline" / "report.html").write_text(render_implementation(baseline), encoding="utf-8")
    locked = read_spec(ROOT / "examples/physical/replan.yaml")
    forbidden = deepcopy(locked)
    forbidden["forbidden_recipes"] = ["regenerate"]
    limited = deepcopy(design)
    limited["max_evaluations"] = 1
    energy_limited = deepcopy(design)
    energy_limited["constraints"]["max_energy_pj"] = "0 pJ"
    for name, request, previous in (("unlocked", design, None), ("locked", locked, baseline),
                                    ("forbidden", forbidden, baseline), ("truncated", limited, None),
                                    ("energy-limit", energy_limited, None)):
        result = plan_physical(network, technology, request, baseline=previous)
        assert check_physical_design(result)["valid"], name
        if name in ("unlocked", "locked"):
            assert result["plans"] and result["search"]["complete"], name
        if name in ("forbidden", "energy-limit"):
            assert not result["plans"] and result["search"]["complete"], name
        if name == "truncated":
            assert not result["plans"] and not result["search"]["complete"]
        if name == "energy-limit":
            assert result["single_constraint_relaxations"]
        if name == "locked":
            assert all(p["realization"]["spec"]["source_power_mw"] == 0.05 and
                       p["realization"]["spec"]["recipe"] == "regenerate" for p in result["plans"])
        _save_physical_design(result, str(directory / name), args.overwrite)
        summary["designs"].append({"case": name, **result["search"]})
        for retained in result["plans"]:
            save_circuit(retained["realization"], directory / name / "plans" / retained["id"], f"{name}/{retained['id']}")

    summary["all_checks_passed"] = True
    summary["optical_comparisons"] = sum(c["comparison_count"] for c in summary["circuits"])
    summary["maximum_absolute_power_error_mw"] = max(c["max_absolute_error_mw"] for c in summary["circuits"])
    _write(directory / "summary.json", summary)
    print(json.dumps({"status": "passed", "circuits": len(summary["circuits"]),
                      "design_cases": len(summary["designs"]), "optical_comparisons": summary["optical_comparisons"],
                      "max_absolute_power_error_mw": summary["maximum_absolute_power_error_mw"],
                      "summary": str(directory / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
