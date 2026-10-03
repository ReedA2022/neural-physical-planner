"""Seeded, bounded stress scenarios for the released optical designer.

The algebraic oracle below never calls the production device evaluator or
frontier builder. It checks the declared illustrative model, not real hardware.
Failures retain the seed, scenario number and inputs for deterministic replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npp.designer import check_physical_design, plan_physical, validate_design_spec
from npp.implementation import realize
from npp.models import InputValidationError
from npp.technology import load_technology

AXES = ("energy_pj", "latency_ns", "area_um2", "noise_rms_estimate")
KINDS = ("source", "splitter", "waveguide", "detector", "modulator")
FAMILIES = ("analytic_grid", "baseline_lock", "equivalent_units", "malformed_input", "replay_tamper", "supported_limits")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _close(actual, expected, label):
    _require(math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-14),
             f"{label}: got {actual!r}, expected {expected!r}")


def _network(lanes=1, uses=3):
    return {"schema_version": "0.1", "name": "Seeded designer stress",
            "nodes": [{"id": "x", "op": "input", "size": lanes,
                       "input_bounds": {"lower": [0.0] * lanes, "upper": [1.0] * lanes}}],
            "outputs": ["x"] * uses}


def _design():
    return {"source_node": "x", "grid": {"recipes": ["passive", "regenerate"],
            "source_powers_mw": [0.05, 0.2], "regeneration_powers_mw": [0.1],
            "route_lengths_um": [0.0, 4000.0]}}


def _oracle(spec, technology, lanes, uses):
    """Closed-form complete-tree calculation, independently of graph evaluation."""
    selected = {k: next(c for c in technology["components"] if c["id"] == spec["components"][k]) for k in KINDS}
    s, b, w, d, m = (selected[k] for k in KINDS)
    ps, pb, pw, pd, pm = (selected[k]["parameters"] for k in KINDS)
    depth = (uses - 1).bit_length()
    leaves = 2 ** depth
    reg = spec["recipe"] == "regenerate"
    launch, carrier = spec["source_power_mw"], spec["regeneration_power_mw"]
    lengths = spec["lengths_um"] if isinstance(spec["lengths_um"], list) else [spec["lengths_um"]] * uses
    op = spec["operating"]
    duration, wavelength = op["symbol_duration_ns"], op["wavelength_nm"]
    early = launch / leaves * 10 ** (-pb["insertion_loss_db"] * depth / 10)
    route_input = carrier * 10 ** (-pm["insertion_loss_db"] / 10) if reg else early
    powers = [route_input * 10 ** (-pw["attenuation_db_per_um"] * length / 10) for length in lengths]
    relevant = [s, b, w, d] + ([m] if reg else [])
    if any(not c["operating_envelope"][key]["minimum"] <= value <= c["operating_envelope"][key]["maximum"]
           for c in relevant for key, value in op.items()):
        return None
    if pb["power_ratio"] != 0.5 or any(length > pw["maximum_length_um"] for length in lengths):
        return None
    if any(not ps["minimum_output_power_mw"] <= value <= ps["maximum_output_power_mw"]
           for value in ([launch, carrier] if reg else [launch])):
        return None
    if launch > b["operating_envelope"]["max_input_power_mw"] or route_input > w["operating_envelope"]["max_input_power_mw"]:
        return None
    if reg and carrier > m["operating_envelope"]["max_input_power_mw"]:
        return None
    if any(not pd["minimum_full_scale_power_mw"] <= p <= d["operating_envelope"]["max_input_power_mw"]
           for p in powers + ([early] if reg else [])):
        return None
    counts = {"source": lanes * (1 + uses * reg), "splitter": lanes * (leaves - 1),
              "waveguide": lanes * uses, "detector": lanes * uses * (1 + reg), "modulator": lanes * uses * reg}
    energy = sum(counts[k] * selected[k]["costs"]["fixed_energy_pj"] for k in KINDS)
    energy += lanes * duration * (launch + (uses * carrier if reg else 0)) / ps["wall_plug_efficiency"]
    area = sum(counts[k] * selected[k]["costs"]["area_um2"] for k in KINDS) + lanes * sum(lengths) * pw["width_um"]
    latency = duration + s["costs"]["latency_ns"] + depth * b["costs"]["latency_ns"]
    if reg:
        latency += d["costs"]["latency_ns"] + duration + m["costs"]["latency_ns"]
    latency += w["costs"]["latency_ns"] + max(lengths) * pw["group_index"] * 1e3 / 299792458.0 + d["costs"]["latency_ns"]

    def detector_variance(power):
        photon_energy = 6.62607015e-34 * 299792458.0 / (wavelength * 1e-9)
        return photon_energy / (pd["quantum_efficiency"] * power * duration * 1e-12) + (pd["input_referred_noise_mw_rms"] / power) ** 2

    common = ps["relative_intensity_noise_rms"] ** 2
    if reg:
        common += ps["relative_intensity_noise_rms"] ** 2 + pm["added_sample_noise_rms"] ** 2 + detector_variance(early)
    noise = [common + detector_variance(p) for p in powers]
    return {"energy_pj": energy, "area_um2": area, "latency_ns": latency,
            "noise_rms_estimate": math.sqrt(max(noise)),
            "stacked_noise_rms_estimate": math.sqrt(lanes * sum(noise)),
            "terminated_power_mw": lanes * (leaves - uses) * early,
            "instance_count": sum(counts.values()), "receiver_count": lanes * uses,
            "margin_mw": min(powers) - pd["minimum_full_scale_power_mw"]}


def _expected_specs(design):
    """Explicit Cartesian enumeration; does not use designer private helpers."""
    grid = design["grid"]
    result = []
    for recipe in grid["recipes"]:
        carriers = [None] if recipe == "passive" else grid["regeneration_powers_mw"]
        alternatives = [grid["components"][k] for k in KINDS]
        if recipe == "passive":
            alternatives[-1] = alternatives[-1][:1]
        for p, q, lengths, components in itertools.product(grid["source_powers_mw"], carriers, grid["route_lengths_um"], itertools.product(*alternatives)):
            result.append({"schema_version": "0.3", "source_node": design["source_node"], "recipe": recipe,
                           "source_power_mw": p, "regeneration_power_mw": q, "lengths_um": lengths,
                           "operating": design["operating"], "components": dict(zip(KINDS, components))})
    return result


def _audit_grid(result, technology, network):
    design = result["inputs"]["design"]
    constraints, locks = design["constraints"], result["resolved_locks"]
    uses, lanes = len(network["outputs"]), network["nodes"][0]["size"]
    expected_specs = _expected_specs(design)
    _require(len(result["candidates"]) == len(expected_specs), "Cartesian candidate count")
    expected_status = Counter()
    evaluated = 0
    for index, (row, spec) in enumerate(zip(result["candidates"], expected_specs), 1):
        _require(row["id"] == f"candidate-{index:04d}" and row["spec"] == spec, "stable enumeration/spec")
        excluded = spec["recipe"] in design["forbidden_recipes"]
        excluded |= any(spec[field] != value for field, value in locks.items() if field != "instances")
        for iid, cid in locks["instances"].items():
            suffix = iid.rsplit(".", 1)[-1]
            kind = {"launch": "source", "carrier": "source", "receiver": "detector", "early_detector": "detector",
                    "modulator": "modulator", "route": "waveguide"}.get(suffix, "splitter")
            excluded |= (suffix in ("carrier", "early_detector", "modulator") and spec["recipe"] != "regenerate") or spec["components"][kind] != cid
        expected = _oracle(spec, technology, lanes, uses)
        if excluded:
            status = "excluded"
        elif evaluated == design["max_evaluations"]:
            status = "not_evaluated"
        else:
            evaluated += 1
            if expected is None:
                status = "infeasible"
            else:
                violations = [axis for axis in AXES if constraints["max_" + axis] is not None and expected[axis] > constraints["max_" + axis]]
                status = "constraint_failed" if violations or expected["margin_mw"] < constraints["min_receiver_margin_mw"] else "feasible"
                for metric, value in expected.items():
                    actual = row["worst_receiver"]["sensitivity_margin_mw"] if metric == "margin_mw" else row["metrics"][metric]
                    _close(actual, value, f"{row['id']}.{metric}")
        _require(row["status"] == status, f"{row['id']} status {row['status']} != {status}")
        if status in ("excluded", "not_evaluated", "infeasible"):
            _require(row["metrics"] is None and bool(row["reasons"]), "unevaluated/invalid candidate has metrics or no reason")
        expected_status[status] += 1
    eligible = [row for row in result["candidates"] if row["status"] == "feasible"]
    axes = design["objectives"]
    frontier = {row["id"] for row in eligible if not any(
        all(other["metrics"][a] <= row["metrics"][a] for a in axes) and any(other["metrics"][a] < row["metrics"][a] for a in axes)
        for other in eligible)}
    _require(frontier == {p["id"] for p in result["plans"]} == {r["id"] for r in result["candidates"] if r["pareto"]}, "all-pairs Pareto mismatch")
    search = result["search"]
    expected_search = {"submitted_grid_size": len(expected_specs), "evaluated_count": evaluated,
                       "excluded_count": expected_status["excluded"], "not_evaluated_count": expected_status["not_evaluated"],
                       "feasible_count": expected_status["feasible"], "pareto_count": len(frontier),
                       "complete": expected_status["not_evaluated"] == 0}
    _require(all(search[k] == v for k, v in expected_search.items()), "search summary mismatch")
    expected_relaxations = {}
    for row in result["candidates"]:
        if row["status"] == "constraint_failed" and len(row["reasons"]) == 1:
            expected_relaxations.setdefault(row["reasons"][0]["path"], []).append(row["id"])
    _require({r["constraint"]: r["candidate_ids"] for r in result["single_constraint_relaxations"]} == expected_relaxations, "relaxation witness mismatch")
    return dict(expected_status)


def _analytic_case(rng, ordinal, repro):
    lanes, uses = rng.choice([1, 2, 3]), rng.choice([2, 3, 4, 5, 7, 8])
    network, technology, design = _network(lanes, uses), load_technology(), _design()
    kind = KINDS[ordinal % len(KINDS)]
    alt = deepcopy(next(c for c in technology["components"] if c["id"] == kind))
    alt["id"] = "alternative_" + kind
    alt["costs"]["fixed_energy_pj"] += rng.choice([0.05, 0.15])
    alt["costs"]["area_um2"] += 7.0
    if kind == "source":
        alt["parameters"]["wall_plug_efficiency"] = 0.4
    elif kind == "waveguide":
        alt["parameters"]["attenuation_db_per_um"] = 0.0001
    elif kind == "detector":
        alt["parameters"]["minimum_full_scale_power_mw"] = 0.005
    elif kind == "splitter":
        alt["parameters"]["insertion_loss_db"] = 0.1
    else:
        alt["parameters"]["insertion_loss_db"] = 0.5
    technology["components"].append(alt)
    design["grid"].update(source_powers_mw=rng.sample([0.01, 0.04, 0.08, 0.2, 0.5, 3.0], 2),
                          route_lengths_um=[[rng.choice([0.0, 350.0, 2000.0, 10000.0]) for _ in range(uses)]],
                          components={kind: [kind, alt["id"]]}, regeneration_powers_mw=[rng.choice([0.02, 0.1, 0.3])])
    design["operating"] = {"wavelength_nm": rng.choice([1540.0, 1550.0, 1560.0]), "temperature_c": rng.choice([20.0, 25.0, 30.0]), "symbol_duration_ns": rng.choice([0.5, 1.0, 10.0])}
    design["objectives"] = rng.sample(list(AXES), rng.randint(1, 4))
    design["max_evaluations"] = rng.choice([1, 3, 128])
    design["constraints"] = rng.choice([{}, {"max_energy_pj": 2.0}, {"max_latency_ns": 3.0}, {"max_area_um2": 20000.0, "max_noise_rms_estimate": 0.03}, {"min_receiver_margin_mw": 0.02}])
    if ordinal % 4 == 1:
        design["forbidden_recipes"] = [rng.choice(["passive", "regenerate"])]
    elif ordinal % 4 == 2:
        design["locks"] = {"source_power_mw": rng.choice(design["grid"]["source_powers_mw"] + [0.123])}
    elif ordinal % 4 == 3:
        design["locks"] = {"instances": {"lane0.use0.modulator": "modulator"}}
    repro.update(network=network, technology=technology, design=design)
    result = plan_physical(network, technology, design)
    return _audit_grid(result, technology, network)


def _baseline_case(rng, ordinal, repro):
    network, technology, design = _network(rng.choice([1, 2]), rng.choice([2, 3, 5])), load_technology(), _design()
    spec = {"source_node": "x", "recipe": "passive" if ordinal % 2 else "regenerate", "source_power_mw": 0.2, "lengths_um": 4000.0}
    if spec["recipe"] == "regenerate":
        spec["regeneration_power_mw"] = 0.1
    baseline = realize(network, technology, spec)
    design["locks"] = rng.choice([{"recipe": "baseline"}, {"source_power_mw": "baseline"}, {"regeneration_power_mw": "baseline"}, {"instances": {"lane0.use0.receiver": "baseline"}}])
    changed_technology = ordinal % 3 == 1
    changed_operating = ordinal % 3 == 2
    if changed_technology:
        technology["components"][0]["costs"]["area_um2"] += 20
    if changed_operating:
        design["operating"] = {"symbol_duration_ns": 2.0}
    repro.update(network=network, technology=technology, design=design, baseline_spec=spec)
    result = plan_physical(network, technology, design, baseline=baseline)
    counts = _audit_grid(result, technology, network)
    for row in result["candidates"]:
        comparison = row["baseline_comparison"]
        if comparison is None:
            continue
        _require(comparison["technology_assumptions_changed"] == changed_technology, "technology comparison flag")
        _require(comparison["operating_point_changed"] == changed_operating, "operating comparison flag")
        comparable = baseline["evaluation"]["metrics"] is not None and row["metrics"] is not None and not changed_operating and not changed_technology
        _require(comparison["comparable_metrics"] == comparable, "like-for-like flag")
        if comparison["metric_deltas"] is not None:
            for axis in AXES:
                _close(comparison["metric_deltas"][axis], row["metrics"][axis] - baseline["evaluation"]["metrics"][axis], "baseline delta " + axis)
    _require(check_physical_design(result)["valid"], "baseline replay failed")
    return counts


def _unit_case(rng, ordinal, repro):
    numeric = _design()
    numeric["grid"].update(source_powers_mw=rng.sample([0.01, 0.04, 0.08, 0.2, 0.5], 2), route_lengths_um=[[0.0, 1000.0, 4000.0]])
    numeric.update(operating={"wavelength_nm": 1550.0, "symbol_duration_ns": 0.5, "temperature_c": 25.0},
                   constraints={"max_energy_pj": 2.0, "max_area_um2": 20000.0, "max_latency_ns": 3.0},
                   locks={"source_power_mw": numeric["grid"]["source_powers_mw"][0]})
    readable = deepcopy(numeric)
    power_unit, multiplier = rng.choice([("W", Decimal("0.001")), ("µW", Decimal("1000")), ("nW", Decimal("1000000"))])
    for field in ("source_powers_mw", "regeneration_powers_mw"):
        readable["grid"][field] = [f"{Decimal(str(v)) * multiplier} {power_unit}" for v in numeric["grid"][field]]
    readable["grid"]["route_lengths_um"] = [["0 m", "1 mm", "4000000 nm"]]
    readable["operating"] = {"wavelength_nm": "0.00000155 m", "symbol_duration_ns": "500 ps", "temperature_c": "25 °C"}
    readable["constraints"] = {"max_energy_pj": "2000 fJ", "max_area_um2": "0.02 mm²", "max_latency_ns": "0.003 us"}
    readable["locks"] = {"source_power_mw": f"{Decimal(str(numeric['locks']['source_power_mw'])) * multiplier} {power_unit}"}
    repro.update(numeric=numeric, readable=readable)
    canonical = validate_design_spec(numeric)
    _require(validate_design_spec(readable) == canonical, "equivalent quantities changed normalized specification")
    _require(validate_design_spec(canonical) == canonical, "normalization not idempotent")
    network, technology = _network(), load_technology()
    _require(plan_physical(network, technology, numeric) == plan_physical(network, technology, readable), "equivalent-unit full records differ")
    return {}


def _set_path(value, path, replacement):
    if not path:
        return replacement
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement
    return value


def _malformed_case(rng, ordinal, repro):
    invalid_values = [None, True, [], {}, "wrong", float("nan"), float("inf"), -float("inf")]
    invalid_fields = [("grid", "source_powers_mw", 0), ("grid", "route_lengths_um", 0),
                      ("operating", "symbol_duration_ns"), ("constraints", "max_latency_ns"), ("max_evaluations",)]
    design = validate_design_spec(_design())
    path = invalid_fields[ordinal % len(invalid_fields)]
    # Optional constraint null is legal; do not falsely call it malformed.
    options = invalid_values[1:] if path[0] == "constraints" else invalid_values
    replacement = rng.choice(options)
    design = _set_path(design, path, replacement)
    repro.update(path=list(path), replacement=repr(replacement), design=repr(design))
    try:
        validate_design_spec(design)
    except InputValidationError as error:
        _require(bool(error.diagnostics) and all(d.get("code") and d.get("path") and d.get("message") for d in error.diagnostics), "missing structured validation diagnostics")
    else:
        raise AssertionError(f"malformed design accepted at {path}")
    return {}


def _tamper_case(rng, ordinal, repro):
    network, technology, design = _network(), load_technology(), _design()
    design["grid"].update(recipes=["passive"], source_powers_mw=[0.2], route_lengths_um=[0.0])
    result = plan_physical(network, technology, design)
    paths = [("search", "complete"), ("candidates", 0, "status"), ("candidates", 0, "metrics", "energy_pj"),
             ("plans", 0, "realization", "graph", "instances"), ("inputs", "technology", "components"),
             ("inputs", "network", "nodes"), ("resolved_locks", "instances"), ("scope",),
             ("candidates", 0, "worst_receiver", "sensitivity_margin_mw"), ("hashes", "design")]
    path = paths[ordinal % len(paths)]
    replacement = rng.choice([None, "corrupted", [True], {"tampered": 1}])
    repro.update(path=list(path), replacement=replacement)
    corrupted = _set_path(deepcopy(result), path, replacement)
    checked = check_physical_design(corrupted)
    _require(not checked["valid"] and bool(checked["diagnostics"]), f"tampered record accepted at {path}")
    return {}


def _limits_case(rng, ordinal, repro):
    design, technology = _design(), load_technology()
    mode = ordinal % 8
    network = _network()
    if mode == 0:
        # Exactly 4096 candidate summaries, with no physics allocated to excluded choices.
        design["grid"].update(recipes=["regenerate"], source_powers_mw=[0.1 + i / 100 for i in range(16)],
                              regeneration_powers_mw=[0.1 + i / 100 for i in range(16)], route_lengths_um=[float(i) for i in range(16)])
        design.update(forbidden_recipes=["regenerate"], max_evaluations=1)
        result = plan_physical(network, technology, design)
        _require(result["search"]["submitted_grid_size"] == 4096 and result["search"]["excluded_count"] == 4096 and result["search"]["complete"], "4096-grid boundary")
    elif mode == 1:
        design["grid"].update(recipes=["regenerate"], source_powers_mw=[0.1 + i / 100 for i in range(17)],
                              regeneration_powers_mw=[0.1 + i / 100 for i in range(16)], route_lengths_um=[float(i) for i in range(16)])
        try:
            validate_design_spec(design)
        except InputValidationError as error:
            _require(any(d["code"] == "design_grid_limit" for d in error.diagnostics), "wrong grid-limit diagnostic")
        else:
            raise AssertionError("oversized Cartesian grid accepted")
    elif mode in (2, 3):
        network = _network(1 if mode == 2 else 64, 64 if mode == 2 else 2)
        design["grid"].update(recipes=["passive"], source_powers_mw=[2.0 if mode == 2 else 0.2], route_lengths_um=[0.0])
        design["max_evaluations"] = 1
        result = plan_physical(network, technology, design)
        _audit_grid(result, technology, network)
        _require(result["search"]["feasible_count"] == 1, "supported fanout/lane boundary failed")
    elif mode == 4:
        network = _network(65, 2)
        try:
            plan_physical(network, technology, design)
        except InputValidationError:
            pass
        else:
            raise AssertionError("unsupported 65-lane input accepted")
    elif mode == 5:
        design["source_node"] = "x" * 129
        try:
            validate_design_spec(design)
        except InputValidationError as error:
            _require(any(d["code"] == "design_identifier_limit" for d in error.diagnostics), "wrong identifier-limit diagnostic")
        else:
            raise AssertionError("129-character source identifier accepted")
    elif mode == 6:
        # Structurally valid request outside a physical envelope must not invent totals.
        design["operating"] = {"temperature_c": 31.0}
        result = plan_physical(network, technology, design)
        _audit_grid(result, technology, network)
        _require(all(r["status"] == "infeasible" for r in result["candidates"]), "out-of-envelope candidate status")
    else:
        design["grid"]["route_lengths_um"] = [[100.0, 200.0]]
        try:
            plan_physical(network, technology, design)
        except InputValidationError as error:
            _require(any(d["code"] == "route_count_mismatch" for d in error.diagnostics), "wrong per-use route diagnostic")
        else:
            raise AssertionError("wrong number of per-use route lengths accepted")
    repro.update(mode=mode, network=network, design=design)
    return {}


def run_campaign(seed=271828, cases=300):
    """Return bounded JSON-compatible results; each scenario is independently replayable."""
    if not __debug__:
        raise ValueError("Stress campaigns require assertions; run Python without -O or -OO.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(cases, bool) or not isinstance(cases, int) or not 1 <= cases <= 2000:
        raise ValueError("cases must be an integer from 1 to 2000")
    started = time.monotonic()
    family_counts, statuses, failures = Counter(), Counter(), []
    functions = (_analytic_case, _baseline_case, _unit_case, _malformed_case, _tamper_case, _limits_case)
    for index in range(cases):
        family_index = index % len(FAMILIES)
        family, ordinal = FAMILIES[family_index], index // len(FAMILIES)
        scenario_seed = seed + 104729 * index
        repro = {"seed": seed, "scenario_index": index, "scenario_seed": scenario_seed, "family": family, "ordinal": ordinal}
        family_counts[family] += 1
        try:
            statuses.update(functions[family_index](random.Random(scenario_seed), ordinal, repro))
        except Exception as error:
            failures.append({"family": family, "scenario_index": index, "error_type": type(error).__name__,
                             "message": str(error), "reproduction": repro})
    return {"campaign": "designer", "seed": seed, "scenario_count": cases, "family_counts": dict(family_counts),
            "candidate_status_counts": dict(statuses), "failures": failures, "passed": not failures,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "scope": "Independent algebraic and search-oracle stress of illustrative optical subcircuits; no hardware calibration or whole-network correctness claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run_campaign(args.seed, args.cases)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
