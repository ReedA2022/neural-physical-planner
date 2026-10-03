"""Bounded, replayable design exploration for declared optical fanout subcircuits.

This searches a user-submitted finite grid. It neither optimizes arbitrary circuits
nor claims physical calibration, whole-network accuracy, or manufacturing closure.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, DecimalException, localcontext
import itertools
import json
import math
import re
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .implementation import (OperatingPoint, SCOPE as IMPLEMENTATION_SCOPE, _hash,
                             _network_context, check_realization, realize,
                             validate_realization_spec)
from .models import Identifier, InputValidationError, Nonnegative, Positive, StrictModel
from .technology import validate_technology

MAX_GRID = 4096
MAX_EVALUATIONS = 128
MAX_INSTANCE_WORK = 250000
MAX_COVARIANCE_WORK = 2000000
MAX_RECORD_BYTES = 64000000
KINDS = ("source", "splitter", "waveguide", "detector", "modulator")
AXES = ("energy_pj", "latency_ns", "area_um2", "noise_rms_estimate")
Recipe = Literal["passive", "regenerate"]
Axis = Literal["energy_pj", "latency_ns", "area_um2", "noise_rms_estimate"]
PowerLock = Positive | Literal["baseline"] | None

SCOPE = {
    "design": "finite grid of balanced optical fanout subcircuits for one declared NN node",
    "optimality": "Pareto dominance only among evaluated, constraint-satisfying candidates in the submitted grid; all listed axes are minimized",
    "component_families": "one selected component ID per kind applies to every instance of that kind; instance locks constrain that global family and require the instance to exist",
    "route_options": "each route option is one scalar length for every use, or an ordered list with one length per actual NN use; not a physical router",
    "locks": "locks filter the submitted grid; they do not insert choices or silently change a candidate",
    "relaxations": "single-constraint suggestions are witnessed by physically feasible evaluated candidates that fail exactly one declared numeric requirement; no global minimum relaxation is claimed",
    "evidence": "model-based estimates using embedded technology assumptions; planning does not run external simulation or confer calibration",
    "implementation": IMPLEMENTATION_SCOPE,
}


class ComponentGrid(StrictModel):
    source: list[Identifier] = Field(default_factory=lambda: ["source"], min_length=1, max_length=16)
    splitter: list[Identifier] = Field(default_factory=lambda: ["splitter"], min_length=1, max_length=16)
    waveguide: list[Identifier] = Field(default_factory=lambda: ["waveguide"], min_length=1, max_length=16)
    detector: list[Identifier] = Field(default_factory=lambda: ["detector"], min_length=1, max_length=16)
    modulator: list[Identifier] = Field(default_factory=lambda: ["modulator"], min_length=1, max_length=16)


class DesignGrid(StrictModel):
    recipes: list[Recipe] = Field(default_factory=lambda: ["passive", "regenerate"], min_length=1, max_length=2)
    source_powers_mw: list[Positive] = Field(min_length=1, max_length=32)
    regeneration_powers_mw: list[Positive] = Field(default_factory=lambda: [0.1], min_length=1, max_length=32)
    route_lengths_um: list[Nonnegative | list[Nonnegative]] = Field(default_factory=lambda: [1000.0], min_length=1, max_length=32)
    components: ComponentGrid = Field(default_factory=ComponentGrid)

    @model_validator(mode="after")
    def bounded_routes(self):
        if any(isinstance(option, list) and not 2 <= len(option) <= 64 for option in self.route_lengths_um):
            raise ValueError("each route option must be a scalar or a list of 2..64 ordered lengths")
        return self


class DesignConstraints(StrictModel):
    max_energy_pj: Nonnegative | None = None
    max_latency_ns: Nonnegative | None = None
    max_area_um2: Nonnegative | None = None
    max_noise_rms_estimate: Nonnegative | None = None
    min_receiver_margin_mw: Nonnegative = 0.0


class DesignLocks(StrictModel):
    recipe: Recipe | Literal["baseline"] | None = None
    source_power_mw: PowerLock = None
    regeneration_power_mw: PowerLock = None
    instances: dict[Identifier, Identifier] = Field(default_factory=dict, max_length=256)


class DesignSpec(StrictModel):
    schema_version: Literal["0.3"] = "0.3"
    source_node: Identifier
    grid: DesignGrid
    operating: OperatingPoint = Field(default_factory=OperatingPoint)
    objectives: list[Axis] = Field(default_factory=lambda: list(AXES), min_length=1, max_length=4)
    constraints: DesignConstraints = Field(default_factory=DesignConstraints)
    locks: DesignLocks = Field(default_factory=DesignLocks)
    forbidden_recipes: list[Recipe] = Field(default_factory=list, max_length=2)
    max_evaluations: Annotated[int, Field(strict=True, ge=1, le=MAX_EVALUATIONS)] = MAX_EVALUATIONS


def _fail(code, path, message):
    raise InputValidationError([{"code": code, "path": path, "message": message}])


_NUMBER = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]*)")
_UNITS = {
    "power": {"mW": 1.0, "W": 1000.0, "uW": 0.001, "nW": 1e-6},
    "length": {"um": 1.0, "nm": 0.001, "mm": 1000.0, "m": 1e6},
    "wavelength": {"nm": 1.0, "um": 1000.0, "m": 1e9},
    "time": {"ns": 1.0, "ps": 0.001, "us": 1000.0, "ms": 1e6, "s": 1e9},
    "energy": {"pJ": 1.0, "fJ": 0.001, "nJ": 1000.0, "uJ": 1e6, "mJ": 1e9, "J": 1e12},
    "area": {"um2": 1.0, "nm2": 1e-6, "mm2": 1e6, "m2": 1e12},
    "temperature": {"C": 1.0, "degC": 1.0, "°C": 1.0},
    "noise": {},
}


def _quantity(value, dimension, path):
    if isinstance(value, bool):
        _fail("invalid_quantity", path, "expected a finite number or a number with a supported unit")
    try:
        if isinstance(value, (float, int)):
            result = float(value)
        elif isinstance(value, str):
            match = _NUMBER.fullmatch(value.strip())
            if not match:
                raise ValueError()
            number, unit = match.groups()
            unit = unit.replace("µ", "u").replace("μ", "u").replace("²", "2").replace("^2", "2")
            if unit and unit not in _UNITS[dimension]:
                _fail("unit_mismatch", path, f"unsupported {dimension} unit {unit!r}; expected {', '.join(_UNITS[dimension]) or 'a unitless normalized value'}")
            # Convert the decimal quantity before rounding to a binary float.
            # Otherwise equivalent inputs such as 100 uW and 0.1 mW can differ
            # by one float step, breaking locks and duplicate-choice checks.
            with localcontext() as context:
                context.prec = max(28, len(number) + 16)
                result = float(Decimal(number) * Decimal(str(_UNITS[dimension].get(unit, 1.0))))
        else:
            raise ValueError()
    except InputValidationError:
        raise
    except (ValueError, OverflowError, DecimalException):
        _fail("invalid_quantity", path, "quantity must be a finite number with a supported unit")
    if not math.isfinite(result):
        _fail("invalid_quantity", path, "quantity is outside the supported finite numeric range")
    return result


def _normalize_units(value):
    # Keep unknown fields intact so strict schema validation reports typos.
    value = deepcopy(value)
    if not isinstance(value, dict):
        return value
    grid = value.get("grid")
    if isinstance(grid, dict):
        for key in ("source_powers_mw", "regeneration_powers_mw"):
            if isinstance(grid.get(key), list):
                grid[key] = [_quantity(x, "power", f"grid.{key}.{i}") for i, x in enumerate(grid[key])]
        if isinstance(grid.get("route_lengths_um"), list):
            grid["route_lengths_um"] = [([_quantity(x, "length", f"grid.route_lengths_um.{i}.{j}") for j, x in enumerate(option)]
                                          if isinstance(option, list) else _quantity(option, "length", f"grid.route_lengths_um.{i}"))
                                         for i, option in enumerate(grid["route_lengths_um"])]
    for group, fields in {
        "operating": {"wavelength_nm": "wavelength", "temperature_c": "temperature", "symbol_duration_ns": "time"},
        "constraints": {"max_energy_pj": "energy", "max_latency_ns": "time", "max_area_um2": "area", "max_noise_rms_estimate": "noise", "min_receiver_margin_mw": "power"},
        "locks": {"source_power_mw": "power", "regeneration_power_mw": "power"},
    }.items():
        if isinstance(value.get(group), dict):
            for key, dimension in fields.items():
                if key in value[group] and value[group][key] is not None and not (group == "locks" and value[group][key] == "baseline"):
                    value[group][key] = _quantity(value[group][key], dimension, f"{group}.{key}")
    return value


def validate_design_spec(value: dict) -> dict:
    """Normalize readable quantities, then validate a bounded strict design grid."""
    try:
        if len(json.dumps(value, allow_nan=False)) > 2000000:
            _fail("design_size_limit", "design", "design specification exceeds 2 MB")
        result = DesignSpec.model_validate(_normalize_units(value)).model_dump(mode="json")
    except InputValidationError:
        raise
    except ValidationError as exc:
        raise InputValidationError([{"code": "design_validation_error", "path": ".".join(map(str, x["loc"])) or "design", "message": x["msg"]}
                                    for x in exc.errors(include_url=False, include_context=False)]) from exc
    except (ValueError, TypeError, OverflowError, RecursionError):
        _fail("design_validation_error", "design", "expected finite, bounded, JSON-compatible design fields")
    unique_lists = [("objectives", result["objectives"]), ("forbidden_recipes", result["forbidden_recipes"])]
    unique_lists += [(f"grid.{key}", result["grid"][key]) for key in ("recipes", "source_powers_mw", "regeneration_powers_mw", "route_lengths_um")]
    unique_lists += [(f"grid.components.{kind}", result["grid"]["components"][kind]) for kind in KINDS]
    for path, values in unique_lists:
        if len({_hash(x) for x in values}) != len(values):
            _fail("duplicate_design_choice", path, "choices must be unique after unit normalization")
    if _grid_size(result) > MAX_GRID:
        _fail("design_grid_limit", "grid", f"submitted grid exceeds {MAX_GRID} candidates; reduce the grid before searching")
    identifiers = [result["source_node"], *result["locks"]["instances"].keys(), *result["locks"]["instances"].values()]
    identifiers.extend(cid for choices in result["grid"]["components"].values() for cid in choices)
    if any(len(identifier) > 128 for identifier in identifiers):
        _fail("design_identifier_limit", "design", "design node, instance and component identifiers are limited to 128 characters")
    return result


def design_schema() -> dict:
    """Canonical numeric-unit JSON schema; YAML additionally accepts unit strings."""
    return DesignSpec.model_json_schema()


def _grid_size(design):
    grid = design["grid"]
    common = len(grid["source_powers_mw"]) * len(grid["route_lengths_um"])
    common *= math.prod(len(grid["components"][kind]) for kind in KINDS[:-1])
    return common * sum(1 if recipe == "passive" else len(grid["regeneration_powers_mw"]) * len(grid["components"]["modulator"]) for recipe in grid["recipes"])


def _candidate_specs(design):
    grid = design["grid"]
    for recipe in grid["recipes"]:
        regeneration = [None] if recipe == "passive" else grid["regeneration_powers_mw"]
        component_options = [grid["components"][kind] if kind != "modulator" or recipe == "regenerate" else grid["components"][kind][:1] for kind in KINDS]
        for power, regen, lengths, components in itertools.product(grid["source_powers_mw"], regeneration, grid["route_lengths_um"], itertools.product(*component_options)):
            yield validate_realization_spec({"source_node": design["source_node"], "recipe": recipe,
                "source_power_mw": power, "regeneration_power_mw": regen, "lengths_um": lengths,
                "operating": design["operating"], "components": dict(zip(KINDS, components))})


def _instance_kinds(lanes, uses, recipe):
    result = {}
    depth = (uses - 1).bit_length()
    for lane in range(lanes):
        result[f"lane{lane}.launch"] = "source"
        for level in range(depth):
            for position in range(2 ** level):
                result[f"lane{lane}.split{level}.{position}"] = "splitter"
        for use in range(uses):
            prefix = f"lane{lane}.use{use}"
            result[prefix + ".route"] = "waveguide"
            result[prefix + ".receiver"] = "detector"
            if recipe == "regenerate":
                result[prefix + ".early_detector"] = "detector"
                result[prefix + ".carrier"] = "source"
                result[prefix + ".modulator"] = "modulator"
    return result


def _resolve_locks(design, baseline, kinds, components):
    locks = deepcopy(design["locks"])
    active = {}
    for field in ("recipe", "source_power_mw", "regeneration_power_mw"):
        value = locks[field]
        if value == "baseline":
            if baseline is None:
                _fail("baseline_required", f"locks.{field}", "this lock requests a baseline value; supply a portable realization")
            value = baseline["spec"][field]
        elif value is None:
            continue
        active[field] = value
    active["instances"] = {}
    baseline_instances = {} if baseline is None else {i["id"]: i for i in baseline["graph"]["instances"]}
    for iid, cid in locks["instances"].items():
        if iid not in kinds:
            _fail("unknown_instance_lock", f"locks.instances.{iid}", "instance does not exist in either supported recipe for this NN node")
        if cid == "baseline":
            if iid not in baseline_instances:
                _fail("baseline_instance_required", f"locks.instances.{iid}", "the requested instance must exist in the supplied baseline")
            cid = baseline_instances[iid]["component"]
        if cid not in components or components[cid]["kind"] != kinds[iid]:
            _fail("invalid_component_lock", f"locks.instances.{iid}", f"{cid!r} is not a declared {kinds[iid]} in the current technology")
        active["instances"][iid] = cid
    return active


def _exclusions(spec, design, locks, instance_kinds):
    reasons = []
    if spec["recipe"] in design["forbidden_recipes"]:
        reasons.append({"code": "forbidden_recipe", "path": "forbidden_recipes", "message": f"recipe {spec['recipe']!r} is forbidden"})
    for field in ("recipe", "source_power_mw", "regeneration_power_mw"):
        if field in locks and spec[field] != locks[field]:
            reasons.append({"code": "lock_mismatch", "path": f"locks.{field}", "message": f"candidate {field} differs from the locked value", "required": locks[field], "actual": spec[field]})
    for iid, cid in locks["instances"].items():
        if iid not in instance_kinds:
            reasons.append({"code": "locked_instance_absent", "path": f"locks.instances.{iid}", "message": "the locked instance is absent from this recipe"})
        elif spec["components"][instance_kinds[iid]] != cid:
            reasons.append({"code": "component_lock_mismatch", "path": f"locks.instances.{iid}", "message": "the global component family does not match this instance lock", "required": cid, "actual": spec["components"][instance_kinds[iid]]})
    return reasons


def _summarize_evaluation(record, constraints):
    evaluation = record["evaluation"]
    evaluated_receivers = [r for r in evaluation["receivers"] if r["status"] == "evaluated"]
    worst = None
    if evaluated_receivers:
        row = min(evaluated_receivers, key=lambda r: (r["sensitivity_margin_mw"], r["port"]["instance"], r["port"]["port"]))
        worst = {key: deepcopy(row[key]) for key in ("port", "source_node", "lane", "use_index", "full_scale_power_mw", "sensitivity_margin_mw")}
        worst.update(required_margin_mw=constraints["min_receiver_margin_mw"], margin_slack_mw=row["sensitivity_margin_mw"] - constraints["min_receiver_margin_mw"])
    drivers = [{key: row[key] for key in ("instance", "component", "kind", "energy_pj", "area_um2")}
               for row in evaluation["ledger"] if row["status"] == "evaluated"]
    drivers.sort(key=lambda r: (-r["energy_pj"], -r["area_um2"], r["instance"]))
    reasons = deepcopy(evaluation["diagnostics"])
    if evaluation["feasible"]:
        for axis in AXES:
            constraint = "max_" + axis
            limit = constraints[constraint]
            actual = evaluation["metrics"][axis]
            if limit is not None and actual > limit:
                reasons.append({"code": "constraint_exceeded", "path": f"constraints.{constraint}", "message": f"{axis} exceeds its maximum", "actual": actual, "limit": limit, "required_relaxation": actual - limit})
        if worst["margin_slack_mw"] < 0:
            reasons.append({"code": "receiver_margin_failed", "path": "constraints.min_receiver_margin_mw", "message": "the final receiver with the smallest sensitivity margin misses the required margin", "instance": worst["port"]["instance"], "actual": worst["sensitivity_margin_mw"], "limit": constraints["min_receiver_margin_mw"], "required_relaxation": -worst["margin_slack_mw"]})
    return worst, drivers[:5], reasons


def _comparison(record, baseline):
    if baseline is None:
        return None
    before = {i["id"]: i for i in baseline["graph"]["instances"]}
    after = {i["id"]: i for i in record["graph"]["instances"]}
    changes = [{"id": iid, "change": "added" if iid not in before else "removed" if iid not in after else "changed", "before": before.get(iid), "after": after.get(iid)}
               for iid in sorted(before.keys() | after.keys()) if before.get(iid) != after.get(iid)]
    technology_changed = baseline["hashes"]["technology"] != record["hashes"]["technology"]
    operating_changed = baseline["spec"]["operating"] != record["spec"]["operating"]
    a, b = baseline["evaluation"]["metrics"], record["evaluation"]["metrics"]
    return {"technology_assumptions_changed": technology_changed, "operating_point_changed": operating_changed,
            "comparable_metrics": a is not None and b is not None and not technology_changed and not operating_changed,
            "metric_deltas": None if a is None or b is None else {axis: b[axis] - a[axis] for axis in AXES},
            "changed_instances": changes,
            "interpretation": "deltas are model-scenario differences; changed technology or operating assumptions prevent a like-for-like design comparison"}


def _dominates(a, b, axes):
    return all(a[axis] <= b[axis] for axis in axes) and any(a[axis] < b[axis] for axis in axes)


def plan_physical(network: dict, technology: dict, design: dict, *, baseline: dict | None = None) -> dict:
    """Explore a bounded grid and retain every nondominated evaluated realization."""
    design = validate_design_spec(design)
    technology = validate_technology(technology)
    network, node, uses, _ = _network_context(network, design["source_node"])
    if baseline is not None:
        checked = check_realization(baseline)
        if not checked["valid"]:
            raise InputValidationError([{"code": "invalid_design_baseline", "path": "baseline", "message": "baseline failed portable realization replay"}] + checked["diagnostics"])
        if _hash(network) != baseline["hashes"]["network"] or design["source_node"] != baseline["spec"]["source_node"]:
            _fail("baseline_context_mismatch", "baseline", "baseline must bind the exact same canonical NN and selected source node")
    components = {c["id"]: c for c in technology["components"]}
    for kind in KINDS:
        for cid in design["grid"]["components"][kind]:
            if cid not in components or components[cid]["kind"] != kind:
                _fail("invalid_component_choice", f"grid.components.{kind}", f"{cid!r} is not a declared {kind}")
    for i, route in enumerate(design["grid"]["route_lengths_um"]):
        if isinstance(route, list) and len(route) != len(uses):
            _fail("route_count_mismatch", f"grid.route_lengths_um.{i}", f"expected exactly {len(uses)} ordered NN-use lengths")
    kinds = {recipe: _instance_kinds(node["size"], len(uses), recipe) for recipe in ("passive", "regenerate")}
    locks = _resolve_locks(design, baseline, kinds["regenerate"], components)
    # Conservative limits are computed before any candidate physics evaluations.
    budget = min(design["max_evaluations"], _grid_size(design))
    maximum_instances = max(len(kinds[recipe]) for recipe in design["grid"]["recipes"])
    if budget * maximum_instances > MAX_INSTANCE_WORK or budget * (node["size"] * len(uses)) ** 2 > MAX_COVARIANCE_WORK:
        _fail("design_work_limit", "max_evaluations", "requested evaluation budget exceeds bounded instance/covariance work; reduce max_evaluations or the NN fanout/lane count")
    input_bytes = len(json.dumps({"network": network, "technology": technology}, allow_nan=False))
    if input_bytes * (2 * budget + 1) > MAX_RECORD_BYTES // 2:
        _fail("design_snapshot_work_limit", "max_evaluations", "embedded input snapshots multiplied by the evaluation budget exceed the portable design size allowance; reduce max_evaluations or the network context")
    candidates, frontier = [], []
    evaluated = excluded = 0
    for index, spec in enumerate(_candidate_specs(design), 1):
        cid = f"candidate-{index:04d}"
        reasons = _exclusions(spec, design, locks, kinds[spec["recipe"]])
        row = {"id": cid, "spec": spec, "status": "excluded", "metrics": None, "reasons": reasons,
               "worst_receiver": None, "cost_drivers": [], "pareto": False, "baseline_comparison": None}
        candidates.append(row)
        if reasons:
            excluded += 1
            continue
        if evaluated >= design["max_evaluations"]:
            row.update(status="not_evaluated", reasons=[{"code": "evaluation_limit", "path": "max_evaluations", "message": "candidate was not evaluated because the finite evaluation budget was reached"}])
            continue
        evaluated += 1
        try:
            record = realize(network, technology, spec)
        except InputValidationError as exc:
            row.update(status="infeasible", reasons=deepcopy(exc.diagnostics))
            continue
        worst, drivers, reasons = _summarize_evaluation(record, design["constraints"])
        row.update(metrics=record["evaluation"]["metrics"], reasons=reasons, worst_receiver=worst, cost_drivers=drivers,
                   baseline_comparison=_comparison(record, baseline),
                   status="infeasible" if not record["evaluation"]["feasible"] else "constraint_failed" if reasons else "feasible")
        if row["status"] != "feasible":
            continue
        if any(_dominates(other["realization"]["evaluation"]["metrics"], row["metrics"], design["objectives"]) for other in frontier):
            continue
        frontier = [other for other in frontier if not _dominates(row["metrics"], other["realization"]["evaluation"]["metrics"], design["objectives"])]
        frontier.append({"id": cid, "realization": record})
    retained = {p["id"] for p in frontier}
    for row in candidates:
        row["pareto"] = row["id"] in retained
    not_evaluated = sum(row["status"] == "not_evaluated" for row in candidates)
    relaxations = {}
    for row in candidates:
        if row["status"] == "constraint_failed" and len(row["reasons"]) == 1:
            reason = row["reasons"][0]
            relaxations.setdefault(reason["path"], []).append(row["id"])
    inputs = {"network": network, "technology": technology, "design": design, "baseline": deepcopy(baseline)}
    result = {"schema_version": "0.3", "kind": "physical_design", "inputs": inputs,
            "hashes": {key: _hash(value) for key, value in inputs.items()}, "resolved_locks": locks,
            "search": {"submitted_grid_size": len(candidates), "evaluated_count": evaluated, "excluded_count": excluded,
                       "not_evaluated_count": not_evaluated, "feasible_count": sum(row["status"] == "feasible" for row in candidates),
                       "pareto_count": len(frontier), "complete": not_evaluated == 0, "max_evaluations": design["max_evaluations"],
                       "optimality_scope": "complete submitted grid after hard choice exclusions" if not_evaluated == 0 else "evaluated subset only; unevaluated candidates may dominate retained plans"},
            "candidates": candidates, "plans": frontier,
            "single_constraint_relaxations": [{"constraint": key, "candidate_ids": ids, "note": "each named, evaluated candidate is feasible under the component model and violates only this numeric requirement; see its actual/limit/required_relaxation values"} for key, ids in sorted(relaxations.items())],
            "scope": deepcopy(SCOPE)}
    if len(json.dumps(result, allow_nan=False)) > MAX_RECORD_BYTES:
        _fail("design_record_size_limit", "result", "generated portable design exceeds 64 MB; reduce the grid or evaluation budget")
    return result


def check_physical_design(result: dict) -> dict:
    """Reconstruct all claims from bounded embedded inputs, including the baseline."""
    required = {"schema_version", "kind", "inputs", "hashes", "resolved_locks", "search", "candidates", "plans", "single_constraint_relaxations", "scope"}
    if not isinstance(result, dict) or set(result) != required or not isinstance(result.get("inputs"), dict) or set(result["inputs"]) != {"network", "technology", "design", "baseline"}:
        return {"valid": False, "diagnostics": [{"code": "invalid_physical_design", "path": "result", "message": "expected exactly the portable physical design record fields"}]}
    diagnostics = []
    try:
        if len(json.dumps(result, allow_nan=False)) > MAX_RECORD_BYTES:
            _fail("design_record_size_limit", "result", "portable design exceeds 64 MB replay limit")
        inputs = result["inputs"]
        replay = plan_physical(inputs["network"], inputs["technology"], inputs["design"], baseline=inputs["baseline"])
        for field in sorted(required):
            if _hash(result[field]) != _hash(replay[field]):
                diagnostics.append({"code": "physical_design_replay_mismatch", "path": field, "message": "stored content differs from deterministic reconstruction from embedded inputs"})
    except InputValidationError as exc:
        diagnostics.extend(exc.diagnostics)
    except (ValueError, TypeError, OverflowError, RecursionError):
        diagnostics.append({"code": "invalid_physical_design", "path": "result", "message": "record cannot be replayed within supported finite JSON limits"})
    return {"valid": not diagnostics, "diagnostics": diagnostics}
