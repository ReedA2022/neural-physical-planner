"""Finite physical-plan synthesis with explicit limits on search claims.

Exhaustive search enumerates the Cartesian product of structurally compatible
node implementations lazily. The evaluator, rather than the search heuristic,
decides physical and budget feasibility.

Beam search extends assignments in network topological order. Its ranking uses
additive component/rule/known-conversion costs, an ASAP latency estimate and a
downstream-sensitivity-weighted local noise surrogate. At every depth it retains
a deterministic round-robin mixture of equal-weight and single-objective ranks.
It never prunes on an alleged admissible error/cost bound. In particular, a
discarded prefix could lead to a better or feasible plan: beam output is always
an approximate frontier. Only complete retained assignments are evaluated by
the physical model. Ranking surrogates saturate at 1e300 to avoid overflow;
this changes heuristic ordering only and never clips the actual evaluation.

Pareto comparisons use relative tolerance 1e-9 and *no absolute tolerance*.
Thus a strictly positive error/energy never becomes equal to zero merely
because it is small. Objective ties retain one deterministic canonical plan.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
import hashlib
import json
import math
import time
from typing import Any, Iterable, Sequence

from .physics import evaluate, outgoing_uses


METRICS = ("energy_pj", "latency_ns", "area_um2", "error_rms_bound")
PARETO_RTOL = 1e-9
_SURROGATE_CAP = 1e300
_RULE_DOMAINS = {
    "digital_copy": "digital",
    "passive_split": "optical",
    "source_boost": "optical",
    "amplify": "optical",
    "regenerate": "optical",
    "serial_reencode": "optical",
}


def input_hash(obj: Any) -> str:
    """SHA-256 of canonical finite JSON; this is not a signature or attestation."""
    encoded = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=PARETO_RTOL, abs_tol=0.0)


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    no_worse = all(x <= y or _close(x, y) for x, y in zip(a, b))
    better = any(x < y and not _close(x, y) for x, y in zip(a, b))
    return no_worse and better


def _canonical_record_key(record: dict[str, Any]) -> str:
    if "decision" in record:
        return input_hash(record["decision"])
    return str(record.get("id", input_hash(record)))


def _objective_values(record: dict[str, Any], objectives: Sequence[str]) -> tuple[float, ...]:
    return tuple(float(record["evaluation"]["metrics"][name]) for name in objectives)


def _insert_frontier(
    frontier: list[dict[str, Any]], record: dict[str, Any], objectives: Sequence[str]
) -> None:
    """Incremental bounded-by-frontier retention; ties resolve by decision hash."""
    values = _objective_values(record, objectives)
    key = _canonical_record_key(record)
    to_remove: list[int] = []
    for i, other in enumerate(frontier):
        other_values = _objective_values(other, objectives)
        if all(_close(a, b) for a, b in zip(values, other_values)):
            if _canonical_record_key(other) <= key:
                return
            to_remove.append(i)
        elif _dominates(other_values, values):
            return
        elif _dominates(values, other_values):
            to_remove.append(i)
    for i in reversed(to_remove):
        del frontier[i]
    frontier.append(record)


def pareto_filter(records: Iterable[dict[str, Any]], objectives: Sequence[str]) -> list[dict[str, Any]]:
    """Retain feasible nondominated records for the requested minimized metrics.

    Finite metrics are required. Ordering and tie handling are deterministic;
    each output record is the original object rather than a rewritten copy.
    """
    objectives = tuple(objectives)
    if not objectives or any(name not in METRICS for name in objectives):
        raise ValueError("objectives must be a nonempty subset of supported metrics")
    candidates = []
    for record in records:
        if not record.get("evaluation", {}).get("feasible", True):
            continue
        values = _objective_values(record, objectives)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Pareto metrics must be finite")
        candidates.append(record)
    frontier: list[dict[str, Any]] = []
    for record in sorted(candidates, key=_canonical_record_key):
        _insert_frontier(frontier, record, objectives)
    return sorted(frontier, key=lambda r: (_objective_values(r, objectives), _canonical_record_key(r)))


@dataclass(frozen=True)
class _Choice:
    component: dict[str, Any]
    rule: dict[str, Any] | None

    @property
    def key(self) -> tuple[str, str]:
        return self.component["id"], "" if self.rule is None else self.rule["id"]


def _choices(
    network: dict[str, Any], library: dict[str, Any], request: dict[str, Any]
) -> tuple[list[list[_Choice]], dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    uses = outgoing_uses(network)
    allowed = set(request["allowed_domains"])
    editable = set(request.get("required_editable", []))
    choices: list[list[_Choice]] = []
    diagnostics: list[dict[str, str]] = []
    components = sorted(library["components"], key=lambda c: c["id"])
    rules = sorted(library.get("rules", []), key=lambda r: r["id"])
    for node in network["nodes"]:
        local: list[_Choice] = []
        matching_components = []
        for component in components:
            if node["op"] not in component["ops"]:
                continue
            if component["domain"] not in allowed or component["capacity"] < node["size"]:
                continue
            if node["op"] == "linear" and node["id"] in editable and not component["editable"]:
                continue
            matching_components.append(component)
            if len(uses[node["id"]]) <= 1:
                local.append(_Choice(component, None))
                continue
            for rule in rules:
                kind = rule["kind"]
                if _RULE_DOMAINS.get(kind) != component["domain"]:
                    continue
                if component["domain"] not in rule["domains"]:
                    continue
                if kind in ("source_boost", "serial_reencode") and not component.get("source_scalable", False):
                    continue
                if kind == "serial_reencode" and not request.get("allow_serialization", False):
                    continue
                local.append(_Choice(component, rule))
        local.sort(key=lambda choice: choice.key)
        choices.append(local)
        if not local:
            code = "NO_COMPATIBLE_COMPONENT" if not matching_components else "NO_COMPATIBLE_FANOUT_RULE"
            diagnostics.append({
                "code": code,
                "path": f"nodes.{node['id']}",
                "message": "No structurally compatible implementation under the request and rule guards.",
            })
    return choices, uses, diagnostics


def _decision(nodes: Sequence[dict[str, Any]], choices: Sequence[_Choice]) -> dict[str, Any]:
    return {
        "components": {node["id"]: choice.component["id"] for node, choice in zip(nodes, choices)},
        "fanout_rules": {
            node["id"]: choice.rule["id"]
            for node, choice in zip(nodes, choices) if choice.rule is not None
        },
    }


def _cap_surrogate(value: float) -> float:
    """Keep heuristic arithmetic finite; this is never a feasibility bound."""
    return min(max(value, 0.0), _SURROGATE_CAP) if math.isfinite(value) else _SURROGATE_CAP


def _surrogate_ratio(numerator: float, denominator: float) -> float:
    # Positive validated inputs can still underflow when multiplied together.
    return _cap_surrogate(numerator / denominator) if denominator > 0 else _SURROGATE_CAP


def _operator_gain(node: dict[str, Any]) -> float:
    if node["op"] == "linear":
        # The uncapped Frobenius norm upper-bounds the operator norm. hypot is
        # stable for finite weights whose squares would overflow.
        return _cap_surrogate(math.hypot(*(float(w) for row in node["weights"] for w in row)))
    return 1.0


def _sensitivities(network: dict[str, Any]) -> dict[str, float]:
    sensitivity = {node["id"]: 0.0 for node in network["nodes"]}
    for output in network["outputs"]:
        sensitivity[output] += 1.0
    for node in reversed(network["nodes"]):
        multiplier = _operator_gain(node)
        for source in node.get("inputs", []):
            sensitivity[source] = _cap_surrogate(sensitivity[source] + multiplier * sensitivity[node["id"]])
    return sensitivity


@dataclass(frozen=True)
class _Partial:
    selected: tuple[_Choice, ...]
    finishes: tuple[float, ...]
    costs: tuple[float, float, float, float]

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        return tuple(choice.key for choice in self.selected)


def _extend(
    partial: _Partial,
    choice: _Choice,
    node: dict[str, Any],
    indices: dict[str, int],
    conversions: dict[tuple[str, str], dict[str, Any]],
    use_count: int,
    output_count: int,
    sensitivity: float,
) -> _Partial:
    component = choice.component
    rule = choice.rule
    energy, latency, area, error = partial.costs
    energy += float(component["energy_pj"])
    area += float(component["area_um2"])
    own_time = float(component["latency_ns"])
    local_noise = float(component["noise_std"]) * math.sqrt(node["size"])
    if node["op"] == "linear":
        macs = sum(len(row) for row in node["weights"])
        energy += macs * float(component.get("energy_per_mac_pj", 0.0))
    if component["domain"] == "optical":
        photons = float(component["photons"])
        launch_multiplier = 1.0
        if rule is not None and rule["kind"] in ("source_boost", "serial_reencode"):
            launch_multiplier = float(use_count)
        photon_energy = _surrogate_ratio(
            photons * node["size"] * 6.62607015e-34 * 299792458.0,
            float(component["wavelength_nm"]) * 1e-9,
        )
        energy += _surrogate_ratio(photon_energy, float(component["wall_plug_efficiency"])) * 1e12 * launch_multiplier
        # Ranking surrogate only; the evaluator implements actual fan-out noise.
        local_noise += math.sqrt(_surrogate_ratio(node["size"], photons * float(component["optical_efficiency"])))
    if rule is not None:
        energy += float(rule["energy_pj"])
        area += float(rule["area_um2"])
        own_time += float(rule["latency_ns"])
        local_noise += float(rule["added_noise_std"]) * math.sqrt(node["size"])
        if rule["kind"] == "serial_reencode":
            own_time *= max(1, use_count)
    arrivals = []
    for source in node.get("inputs", []):
        i = indices[source]
        source_domain = partial.selected[i].component["domain"]
        arrival = partial.finishes[i]
        if source_domain != component["domain"]:
            conversion = conversions.get((source_domain, component["domain"]))
            if conversion is not None:
                energy += float(conversion["energy_pj"])
                area += float(conversion["area_um2"])
                arrival += float(conversion["latency_ns"])
                local_noise += float(conversion["noise_std"]) * math.sqrt(node["size"])
        arrivals.append(arrival)
    finish = (max(arrivals) if arrivals else 0.0) + own_time
    latency = max(latency, finish)
    if output_count and component["domain"] != "digital":
        conversion = conversions.get((component["domain"], "digital"))
        if conversion is not None:
            energy += output_count * float(conversion["energy_pj"])
            area += output_count * float(conversion["area_um2"])
            latency = max(latency, finish + float(conversion["latency_ns"]))
            local_noise += output_count * float(conversion["noise_std"]) * math.sqrt(node["size"])
    error += sensitivity * local_noise if sensitivity > 0 else 0.0
    return _Partial(
        partial.selected + (choice,), partial.finishes + (_cap_surrogate(finish),),
        tuple(_cap_surrogate(value) for value in (energy, latency, area, error)),
    )


def _retain_diverse(states: list[_Partial], width: int, objectives: Sequence[str]) -> list[_Partial]:
    if len(states) <= width:
        return sorted(states, key=lambda state: state.key)
    positions = [METRICS.index(name) for name in objectives]
    minimum = [min(state.costs[i] for state in states) for i in positions]
    maximum = [max(state.costs[i] for state in states) for i in positions]

    def normalized(state: _Partial) -> tuple[float, ...]:
        return tuple(
            0.0 if upper == lower else (state.costs[i] - lower) / (upper - lower)
            for i, lower, upper in zip(positions, minimum, maximum)
        )

    scores = {state.key: normalized(state) for state in states}
    # Balanced rank plus every requested single-objective rank. Ties use the
    # balanced score then the component/rule IDs, making the search reproducible.
    rankings = [sorted(states, key=lambda state: (sum(scores[state.key]), state.key))]
    rankings.extend(
        sorted(states, key=lambda state: (scores[state.key][j], sum(scores[state.key]), state.key))
        for j in range(len(positions))
    )
    pointers = [0] * len(rankings)
    seen: set[tuple[tuple[str, str], ...]] = set()
    retained = []
    while len(retained) < width:
        for j, ranking in enumerate(rankings):
            while pointers[j] < len(ranking) and ranking[pointers[j]].key in seen:
                pointers[j] += 1
            if pointers[j] < len(ranking):
                state = ranking[pointers[j]]
                pointers[j] += 1
                seen.add(state.key)
                retained.append(state)
                if len(retained) == width:
                    break
    return retained


def _beam_candidates(
    network: dict[str, Any], library: dict[str, Any], choices: Sequence[Sequence[_Choice]],
    uses: dict[str, list[dict[str, Any]]], request: dict[str, Any], telemetry: dict[str, Any],
) -> list[tuple[_Choice, ...]]:
    indices = {node["id"]: i for i, node in enumerate(network["nodes"])}
    conversions = {(c["from_domain"], c["to_domain"]): c for c in library.get("conversions", [])}
    output_counts = Counter(network["outputs"])
    sensitivities = _sensitivities(network)
    beam = [_Partial((), (), (0.0, 0.0, 0.0, 0.0))]
    width = request["search"]["beam_width"]
    for node, local in zip(network["nodes"], choices):
        expanded = [
            _extend(
                state, choice, node, indices, conversions, len(uses[node["id"]]),
                output_counts[node["id"]], sensitivities[node["id"]],
            )
            for state in beam for choice in local
        ]
        telemetry["partial_states_expanded"] += len(expanded)
        telemetry["partial_states_pruned"] += max(0, len(expanded) - width)
        beam = _retain_diverse(expanded, width, request["objectives"])
    return [state.selected for state in beam]


def plan(network: dict[str, Any], library: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Compile normalized, validated inputs into checked candidate hardware plans.

    ``status=infeasible`` is returned only for exhausted exhaustive enumeration
    or an empty structurally compatible candidate space. Beam/truncated searches
    with no feasible candidate return ``search_exhausted``.
    """
    started = time.perf_counter()
    hashes = {"network": input_hash(network), "library": input_hash(library), "request": input_hash(request)}
    choices, uses, generation_diagnostics = _choices(network, library, request)
    total = math.prod(len(local) for local in choices)
    search = request["search"]
    mode = search["mode"]
    maximum = search["max_evaluations"]
    telemetry: dict[str, Any] = {
        "mode": mode,
        "total_candidates": total,
        "candidate_counts": {node["id"]: len(local) for node, local in zip(network["nodes"], choices)},
        "evaluated": 0, "feasible": 0, "rejected": 0, "complete": False,
        "max_evaluations": maximum, "beam_width": search["beam_width"],
        "partial_states_expanded": 0, "partial_states_pruned": 0,
        "seed": request.get("seed", 0),
        "deterministic": True,
        "algorithm": "lazy_cartesian_enumeration" if mode == "exhaustive" else "diverse_surrogate_beam",
        "space_definition": "Structurally compatible component/rule combinations before numeric physical and budget guards.",
        "pareto_relative_tolerance": PARETO_RTOL,
        "pareto_absolute_tolerance": 0.0,
        "beam_surrogate_cap": _SURROGATE_CAP,
    }
    caveats = [
        "Feasibility and costs hold only under the declared component library and normalized additive-noise model.",
        "Error is an RMS model bound, not a deterministic all-sample or fabricated-device guarantee.",
        "Schedules dedicate one unit per neural node with modeled fan-out serialization; arbitrary resource sharing is not synthesized.",
        "Reported trade-offs concern the supplied finite implementation choices, not all physically possible hardware.",
        "A retained feasible plan is not a fabrication-ready layout or a measured-hardware result.",
    ]
    if mode == "beam":
        caveats.append(
            "Beam ranking uses incomplete additive cost and noise surrogates; it may discard every feasible continuation or miss superior trade-offs."
        )
    report: dict[str, Any] = {
        "schema_version": "0.1", "status": "search_exhausted", "input_hashes": hashes,
        "search": telemetry, "plans": [], "rejection_summary": [],
        "diagnostics": generation_diagnostics, "caveats": caveats,
    }
    if total == 0:
        telemetry["complete"] = True
        telemetry["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "infeasible"
        report["rejection_summary"] = [
            {"code": code, "count": count, "scope": "candidate_generation"}
            for code, count in sorted(Counter(d["code"] for d in generation_diagnostics).items())
        ]
        return report

    if mode == "exhaustive":
        candidates: Iterable[Sequence[_Choice]] = product(*choices)
    elif mode == "beam":
        candidates = _beam_candidates(network, library, choices, uses, request, telemetry)
    else:
        raise ValueError("Unsupported search mode; validate inputs before planning")
    rejection_counts: Counter[str] = Counter()
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        if telemetry["evaluated"] >= maximum:
            break
        decision = _decision(network["nodes"], candidate)
        evaluation = evaluate(network, library, request, decision)
        telemetry["evaluated"] += 1
        if evaluation.get("feasible", False):
            metrics = evaluation.get("metrics", {})
            if not all(name in metrics and math.isfinite(float(metrics[name])) for name in METRICS):
                evaluation = dict(evaluation)
                evaluation["feasible"] = False
                evaluation["diagnostics"] = list(evaluation.get("diagnostics", [])) + [{
                    "code": "INVALID_EVALUATION_METRICS", "path": "metrics",
                    "message": "Evaluator returned missing or nonfinite metrics for a purportedly feasible plan.",
                }]
        if evaluation.get("feasible", False):
            telemetry["feasible"] += 1
            record = {
                "schema_version": "0.1", "id": "plan-" + input_hash(decision),
                "input_hashes": dict(hashes), "decision": decision, "evaluation": evaluation,
            }
            _insert_frontier(frontier, record, request["objectives"])
        else:
            telemetry["rejected"] += 1
            codes = {d.get("code", "UNSPECIFIED_REJECTION") for d in evaluation.get("diagnostics", [])}
            rejection_counts.update(codes or {"UNSPECIFIED_REJECTION"})
    telemetry["complete"] = mode == "exhaustive" and telemetry["evaluated"] == total
    telemetry["elapsed_seconds"] = time.perf_counter() - started
    telemetry["pareto_retained"] = len(frontier)
    report["plans"] = pareto_filter(frontier, request["objectives"])
    report["rejection_summary"] = [
        {"code": code, "count": count} for code, count in sorted(rejection_counts.items())
    ]
    report["status"] = "ok" if report["plans"] else ("infeasible" if telemetry["complete"] else "search_exhausted")
    return report
