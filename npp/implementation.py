"""Explicit, NN-linked intensity fanout subcircuits (not a whole-NN backend).

Graph validity is separate from physical feasibility. Numerical noise is a
first-order full-scale estimate retaining shared sources, never an NN certificate.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .models import (Finite, Identifier, InputValidationError, Network, Nonnegative,
                     Positive, StrictModel)
from .physics import infer_bounds, outgoing_uses
from .technology import check_operating_point, evaluate_component, validate_technology

MAX_FANOUT = 64
MAX_LANES = 64
MAX_RECEIVERS = 256
MAX_INSTANCES = 4096
Index = Annotated[int, Field(strict=True, ge=0)]

SCOPE = {
    "implementation": "one NN node's scalar-lane optical fanout distribution subcircuit",
    "encoding": "normalized intensity P=P_full_scale*x, nominal 0<=x<=1; no implicit rescaling",
    "input_boundary": "each encoded_launch source is externally driven by the selected NN value; NN computation and encoding/driver internals are outside this subcircuit",
    "output_boundary": "complete calibrated electrical samples at receiver ports; wiring and consumer NN operators are outside this subcircuit",
    "energy": "full-scale per-symbol electrical energy upper bound under the declared nominal model; includes every launch and carrier source and component overhead; not workload averaged",
    "latency": "complete-symbol readiness from launch start; one acquisition interval at launch and an additional interval for each regeneration; fixed component overheads are additional",
    "carrier_schedule": "fresh carrier for each generated regenerative branch is gated to one symbol at reencoding; startup/transient and standby costs are excluded",
    "noise": "first-order normalized full-scale covariance and RMS estimate with shared upstream contributions; not a guaranteed NN error bound, clipping proof, or task-accuracy prediction",
    "termination": "unused optical leaves are ideal matched external termination boundaries; their power is lost and not redistributed; termination device area/cost is outside scope",
    "aggregation": "subcircuit instance costs only; not added to legacy neural macro estimates",
    "excluded": ["complete neural hardware", "signed encoding", "fabrication and layout", "routing spacing/bends/coupling", "complete control/memory/IO", "throughput", "noise nonlinearities and clipping", "calibration and manufacturing guarantees"],
}


class OperatingPoint(StrictModel):
    wavelength_nm: Positive = 1550.0
    temperature_c: Finite = 25.0
    symbol_duration_ns: Positive = 1.0

    @model_validator(mode="after")
    def physical_temperature(self):
        if self.temperature_c < -273.15:
            raise ValueError("temperature cannot be below absolute zero")
        return self


class ComponentChoices(StrictModel):
    source: Identifier = "source"
    splitter: Identifier = "splitter"
    waveguide: Identifier = "waveguide"
    detector: Identifier = "detector"
    modulator: Identifier = "modulator"


class RealizationSpec(StrictModel):
    schema_version: Literal["0.3"] = "0.3"
    source_node: Identifier
    recipe: Literal["passive", "regenerate"] = "passive"
    source_power_mw: Positive
    regeneration_power_mw: Positive | None = None
    lengths_um: Nonnegative | list[Nonnegative] = Field(default=1000.0)
    operating: OperatingPoint = Field(default_factory=OperatingPoint)
    components: ComponentChoices = Field(default_factory=ComponentChoices)

    @model_validator(mode="after")
    def fields_for_recipe(self):
        if self.recipe == "regenerate" and self.regeneration_power_mw is None:
            raise ValueError("regenerate requires regeneration_power_mw")
        if self.recipe == "passive" and self.regeneration_power_mw is not None:
            raise ValueError("regeneration_power_mw is only valid for regenerate")
        if isinstance(self.lengths_um, list) and not 2 <= len(self.lengths_um) <= MAX_FANOUT:
            raise ValueError(f"lengths_um must be a scalar or a list of 2..{MAX_FANOUT} route lengths")
        return self


class Endpoint(StrictModel):
    instance: Identifier
    port: Identifier


class Connection(StrictModel):
    source: Endpoint
    target: Endpoint


class InstanceParameters(StrictModel):
    power_mw: Positive | None = None
    length_um: Nonnegative | None = None


class Instance(StrictModel):
    id: Identifier
    component: Identifier
    parameters: InstanceParameters = Field(default_factory=InstanceParameters)
    role: Literal["encoded_launch", "constant_carrier"] | None = None
    source_node: Identifier | None = None
    lane: Index | None = None


class Receiver(StrictModel):
    port: Endpoint
    source_node: Identifier
    lane: Index
    use_index: Index


class Termination(StrictModel):
    port: Endpoint
    model: Literal["ideal_matched_external_boundary"] = "ideal_matched_external_boundary"
    reason: Literal["unused_splitter_leaf"] = "unused_splitter_leaf"


class ImplementationGraph(StrictModel):
    schema_version: Literal["0.3"] = "0.3"
    kind: Literal["intensity_fanout_subcircuit"] = "intensity_fanout_subcircuit"
    network: Network
    source_node: Identifier
    recipe: Literal["passive", "regenerate"]
    operating: OperatingPoint
    instances: list[Instance] = Field(min_length=1, max_length=MAX_INSTANCES)
    connections: list[Connection] = Field(max_length=MAX_INSTANCES * 2)
    launch_boundaries: list[Endpoint] = Field(min_length=1, max_length=MAX_LANES)
    receivers: list[Receiver] = Field(min_length=2, max_length=MAX_RECEIVERS)
    terminations: list[Termination] = Field(max_length=MAX_RECEIVERS)


def _error(code, path, message):
    raise InputValidationError([{"code": code, "path": path, "message": message}])


def _validate(model, value, code):
    try:
        return model.model_validate(value).model_dump(mode="json")
    except ValidationError as exc:
        raise InputValidationError([{"code": code, "path": ".".join(map(str, item["loc"])) or "input", "message": item["msg"]}
                                    for item in exc.errors(include_url=False, include_context=False)]) from exc
    except (OverflowError, RecursionError) as exc:
        _error(code, "input", "input exceeds supported finite numeric or nesting limits")


def validate_realization_spec(value: dict) -> dict:
    return _validate(RealizationSpec, value, "realization_validation_error")


def realization_schema() -> dict:
    return RealizationSpec.model_json_schema()


def implementation_schema() -> dict:
    return ImplementationGraph.model_json_schema()


def _network_context(network: dict, source_node: str):
    normalized = _validate(Network, network, "network_validation_error")
    if len(normalized["nodes"]) > 1024 or sum(n["size"] for n in normalized["nodes"]) > 100000:
        _error("implementation_size_limit", "network", "physical subcircuit context is limited to 1024 nodes and 100000 total scalar node values")
    nodes = {n["id"]: n for n in normalized["nodes"]}
    if source_node not in nodes:
        _error("unknown_source_node", "source_node", f"{source_node!r} does not exist in the NN")
    node = nodes[source_node]
    uses = outgoing_uses(normalized)[source_node]
    if not 2 <= len(uses) <= MAX_FANOUT:
        _error("fanout_out_of_range", "source_node", f"requires 2..{MAX_FANOUT} ordered NN uses (including repeated operands and outputs); got {len(uses)}")
    if node["size"] > MAX_LANES or node["size"] * len(uses) > MAX_RECEIVERS:
        _error("implementation_size_limit", "source_node", f"at most {MAX_LANES} lanes and {MAX_RECEIVERS} scalar receivers are supported")
    try:
        bounds = infer_bounds(normalized)[source_node]
    except (ValueError, OverflowError) as exc:
        _error("invalid_nominal_bounds", "network", str(exc))
    if any(lo < 0 or hi > 1 for lo, hi in zip(bounds["lower"], bounds["upper"])):
        _error("unsupported_encoding_range", "source_node", "normalized intensity requires inferred nominal bounds entirely in [0,1]; specify a suitable NN node or an explicit encoder outside this backend, not an implicit rescaling")
    return normalized, node, uses, bounds


def _ep(instance, port):
    return {"instance": instance, "port": port}


def _key(endpoint):
    return endpoint["instance"], endpoint["port"]


def build_implementation(network: dict, technology: dict, spec: dict) -> dict:
    """Generate a deterministic balanced splitter tree, repeated for every lane."""
    technology = validate_technology(technology)
    spec = validate_realization_spec(spec)
    network, node, uses, _ = _network_context(network, spec["source_node"])
    cmap = {c["id"]: c for c in technology["components"]}
    needed = ["source", "splitter", "waveguide", "detector"] + (["modulator"] if spec["recipe"] == "regenerate" else [])
    for kind in needed:
        cid = spec["components"][kind]
        if cid not in cmap or cmap[cid]["kind"] != kind:
            _error("invalid_component_choice", f"components.{kind}", f"{cid!r} is not a declared {kind}")
    ratio = cmap[spec["components"]["splitter"]]["parameters"]["power_ratio"]
    if ratio != 0.5:
        _error("unsupported_splitter_ratio", "components.splitter", "this balanced-tree recipe requires a declared fixed 50:50 splitter")
    lengths = spec["lengths_um"]
    if not isinstance(lengths, list):
        lengths = [lengths] * len(uses)
    if len(lengths) != len(uses):
        _error("route_count_mismatch", "lengths_um", f"expected exactly {len(uses)} lengths, one per ordered NN use")
    depth = (len(uses) - 1).bit_length()
    instances, connections, boundaries, receivers, terminations = [], [], [], [], []

    def add(iid, kind, *, power=None, length=None, role=None, lane=None):
        instances.append({"id": iid, "component": spec["components"][kind],
                          "parameters": {"power_mw": power, "length_um": length}, "role": role,
                          "source_node": spec["source_node"] if role == "encoded_launch" else None,
                          "lane": lane})
        return iid

    def connect(source, iid, port="in"):
        connections.append({"source": source, "target": _ep(iid, port)})

    for lane in range(node["size"]):
        launch = add(f"lane{lane}.launch", "source", power=spec["source_power_mw"], role="encoded_launch", lane=lane)
        boundaries.append(_ep(launch, "out"))
        leaves = [_ep(launch, "out")]
        for level in range(depth):
            next_leaves = []
            for position, upstream in enumerate(leaves):
                splitter = add(f"lane{lane}.split{level}.{position}", "splitter")
                connect(upstream, splitter)
                next_leaves.extend([_ep(splitter, "out1"), _ep(splitter, "out2")])
            leaves = next_leaves
        for use_index, (upstream, length) in enumerate(zip(leaves, lengths)):
            prefix = f"lane{lane}.use{use_index}"
            if spec["recipe"] == "regenerate":
                early = add(prefix + ".early_detector", "detector")
                connect(upstream, early)
                carrier = add(prefix + ".carrier", "source", power=spec["regeneration_power_mw"], role="constant_carrier")
                mod = add(prefix + ".modulator", "modulator")
                connect(_ep(early, "out"), mod, "sample")
                connect(_ep(carrier, "out"), mod, "carrier")
                upstream = _ep(mod, "out")
            guide = add(prefix + ".route", "waveguide", length=length)
            connect(upstream, guide)
            detector = add(prefix + ".receiver", "detector")
            connect(_ep(guide, "out"), detector)
            receivers.append({"port": _ep(detector, "out"), "source_node": spec["source_node"], "lane": lane, "use_index": use_index})
        terminations.extend({"port": port, "model": "ideal_matched_external_boundary", "reason": "unused_splitter_leaf"} for port in leaves[len(uses):])
    return validate_implementation({"schema_version": "0.3", "kind": "intensity_fanout_subcircuit", "network": network,
                                    "source_node": spec["source_node"], "recipe": spec["recipe"], "operating": spec["operating"],
                                    "instances": instances, "connections": connections, "launch_boundaries": boundaries,
                                    "receivers": receivers, "terminations": terminations}, technology)


def _structure(graph, technology):
    """Validate ports and causality, return lookup maps and topological order."""
    cmap = {c["id"]: c for c in technology["components"]}
    instances, ports = {}, {}
    for instance in graph["instances"]:
        iid = instance["id"]
        if iid in instances:
            _error("duplicate_instance", iid, "instance ids must be unique")
        instances[iid] = instance
        if instance["component"] not in cmap:
            _error("unknown_component", iid, "instance component is absent from technology pack")
        component = cmap[instance["component"]]
        kind, pars = component["kind"], instance["parameters"]
        if kind == "source":
            if instance["role"] is None or pars["power_mw"] is None or pars["length_um"] is not None:
                _error("invalid_instance_parameters", iid, "source requires role and power_mw, without length_um")
            if instance["role"] == "encoded_launch":
                if instance["source_node"] is None or instance["lane"] is None:
                    _error("missing_launch_binding", iid, "encoded launch requires source_node and lane")
            elif instance["source_node"] is not None or instance["lane"] is not None:
                _error("invalid_carrier_binding", iid, "constant carrier carries no NN source_node or lane")
        else:
            if instance["role"] is not None or instance["source_node"] is not None or instance["lane"] is not None or pars["power_mw"] is not None:
                _error("invalid_instance_parameters", iid, "only sources may declare role, NN binding or power_mw")
            if (kind == "waveguide") != (pars["length_um"] is not None):
                _error("invalid_instance_parameters", iid, "length_um is required exactly for waveguides")
        ports.update({(iid, p["name"]): p for p in component["ports"]})

    incoming, outgoing, successors = {}, {}, {iid: [] for iid in instances}
    indegree = {iid: 0 for iid in instances}
    for edge in graph["connections"]:
        src, dst = _key(edge["source"]), _key(edge["target"])
        if src not in ports or dst not in ports:
            _error("unknown_port", "connections", f"unknown endpoint in {src!r} -> {dst!r}")
        a, b = ports[src], ports[dst]
        if a["direction"] != "out" or b["direction"] != "in":
            _error("port_direction_mismatch", "connections", "connections must run from an output to an input")
        if any(a[field] != b[field] for field in ("domain", "quantity", "encoding")):
            _error("port_type_mismatch", "connections", f"incompatible physical ports: {src!r} -> {dst!r}")
        if src in outgoing or dst in incoming:
            _error("port_connection_count", "connections", "every physical port permits exactly one connection; no implicit electrical or optical copying/multidrive")
        incoming[dst], outgoing[src] = src, dst
        successors[src[0]].append(dst[0])
        indegree[dst[0]] += 1

    receiver_ports, termination_ports, boundary_ports = set(), set(), set()
    for receiver in graph["receivers"]:
        port = _key(receiver["port"])
        if port not in ports or ports[port]["direction"] != "out" or ports[port]["domain"] != "electrical":
            _error("invalid_receiver_port", "receivers", "receiver boundary must be an existing electrical output")
        if port in receiver_ports or port in outgoing:
            _error("port_connection_count", "receivers", "receiver output must have exactly one external boundary use")
        receiver_ports.add(port)
    for termination in graph["terminations"]:
        port = _key(termination["port"])
        if port not in ports or ports[port]["direction"] != "out" or ports[port]["domain"] != "optical":
            _error("invalid_termination_port", "terminations", "matched termination must be an optical output")
        if cmap[instances[port[0]]["component"]]["kind"] != "splitter":
            _error("invalid_termination_port", "terminations", "unused_splitter_leaf termination requires a splitter output")
        if port in termination_ports or port in outgoing:
            _error("port_connection_count", "terminations", "terminated output must have exactly one external boundary use")
        termination_ports.add(port)
    for boundary in graph["launch_boundaries"]:
        port = _key(boundary)
        if port not in ports or instances[port[0]]["role"] != "encoded_launch" or port[1] != "out":
            _error("invalid_launch_boundary", "launch_boundaries", "launch boundary must name encoded_launch source.out")
        if port in boundary_ports:
            _error("duplicate_launch_boundary", "launch_boundaries", "launch boundaries must be unique")
        boundary_ports.add(port)
    expected_launch = {(i["id"], "out") for i in instances.values() if i["role"] == "encoded_launch"}
    if boundary_ports != expected_launch:
        _error("missing_launch_boundary", "launch_boundaries", "every encoded launch must have one explicit boundary declaration")
    for key, port in ports.items():
        if port["direction"] == "in" and key not in incoming:
            _error("unconnected_input", str(key), "all physical inputs require a connection")
        if port["direction"] == "out" and key not in outgoing and key not in receiver_ports and key not in termination_ports:
            _error("unconnected_output", str(key), "output requires a connection, receiver boundary or explicit matched termination")
    ready = deque(iid for iid in instances if indegree[iid] == 0)
    order = []
    while ready:
        iid = ready.popleft()
        order.append(iid)
        for nxt in successors[iid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(instances):
        _error("physical_cycle", "connections", "physical subcircuit must be acyclic")
    return cmap, instances, ports, incoming, order


def validate_implementation(graph: dict, technology: dict) -> dict:
    """Check complete structural connectivity and exact nominal NN sample lineage.

    Device operating feasibility is evaluated separately, and may legitimately fail
    for a structurally valid candidate. Neither operation establishes calibration.
    """
    technology = validate_technology(technology)
    graph = _validate(ImplementationGraph, graph, "implementation_validation_error")
    _, node, uses, _ = _network_context(graph["network"], graph["source_node"])
    cmap, instances, ports, incoming, order = _structure(graph, technology)
    observed_launches = [(i["source_node"], i["lane"]) for i in instances.values() if i["role"] == "encoded_launch"]
    if Counter(observed_launches) != Counter((graph["source_node"], lane) for lane in range(node["size"])):
        _error("launch_binding_mismatch", "instances", "exactly one encoded launch per selected NN scalar lane is required")
    signal = {}
    for iid in order:
        instance = instances[iid]
        component = cmap[instance["component"]]
        kind = component["kind"]
        ins = {p["name"]: signal[incoming[(iid, p["name"])]] for p in component["ports"] if p["direction"] == "in"}
        if kind == "source":
            value = (instance["source_node"], instance["lane"], 0) if instance["role"] == "encoded_launch" else None
        elif kind == "modulator":
            if ins["carrier"] is not None:
                _error("encoded_carrier", iid, "modulator carrier must be an unencoded constant carrier, not a copy of the NN sample")
            if ins["sample"] is None:
                _error("missing_sample_lineage", iid, "modulator sample must descend from an encoded NN launch")
            value = (ins["sample"][0], ins["sample"][1], ins["sample"][2] + 1)
        else:
            value = ins["in"]
            if kind == "detector" and value is None:
                _error("missing_sample_lineage", iid, "detecting a constant carrier does not reproduce the NN sample")
        for p in component["ports"]:
            if p["direction"] == "out":
                signal[(iid, p["name"])] = value
    expected = Counter((graph["source_node"], lane, use) for lane in range(node["size"]) for use in range(len(uses)))
    observed = Counter((r["source_node"], r["lane"], r["use_index"]) for r in graph["receivers"])
    if expected != observed:
        _error("receiver_binding_mismatch", "receivers", "receivers must cover every ordered NN use and lane exactly once")
    stages = 0 if graph["recipe"] == "passive" else 1
    for receiver in graph["receivers"]:
        lineage = signal[_key(receiver["port"])]
        if lineage != (receiver["source_node"], receiver["lane"], stages):
            _error("receiver_lineage_mismatch", "receivers", "receiver must preserve its declared source/lane and use the recipe's regeneration stage count")
    if graph["recipe"] == "passive" and any(cmap[i["component"]]["kind"] == "modulator" or i["role"] == "constant_carrier" for i in instances.values()):
        _error("recipe_topology_mismatch", "recipe", "passive recipe cannot contain modulators or carrier sources")
    return graph


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evaluate_implementation(graph: dict, technology: dict) -> dict:
    """Evaluate all instances; infeasible candidates keep explicit diagnostics.

    A failed component blocks its descendants. Aggregate metrics are null on any
    failure, avoiding apparently complete costs for a partially evaluated circuit.
    """
    technology = validate_technology(technology)
    graph = validate_implementation(graph, technology)
    cmap, instances, ports, incoming, order = _structure(graph, technology)
    op = graph["operating"]
    signal, ledger, diagnostics = {}, [], []
    for iid in order:
        instance = instances[iid]
        component = cmap[instance["component"]]
        kind, pars = component["kind"], instance["parameters"]
        input_ports = [p["name"] for p in component["ports"] if p["direction"] == "in"]
        missing = [p for p in input_ports if incoming[(iid, p)] not in signal]
        entry = {"instance": iid, "component": component["id"], "kind": kind, "role": instance["role"]}
        static_errors = check_operating_point(component, **op)
        if kind == "waveguide" and pars["length_um"] > component["parameters"]["maximum_length_um"]:
            static_errors.append({"code": "component_infeasible", "path": iid, "message": "waveguide length outside declared range"})
        if static_errors:
            for error in static_errors:
                diagnostics.append({**error, "path": iid, "component_path": error["path"]})
            ledger.append({**entry, "status": "infeasible", "diagnostics": diagnostics[-len(static_errors):]})
            continue
        if missing:
            ledger.append({**entry, "status": "blocked", "reason": "an upstream instance is infeasible", "missing_inputs": missing})
            continue
        ins = {p: signal[incoming[(iid, p)]] for p in input_ports}
        input_power = None if kind == "source" else ins["carrier" if kind == "modulator" else "in"]["power_mw"]
        try:
            result = evaluate_component(component, **op, input_power_mw=input_power,
                                        source_power_mw=pars["power_mw"], length_um=pars["length_um"] or 0.0)
        except (InputValidationError, ValueError, OverflowError) as exc:
            errors = exc.diagnostics if isinstance(exc, InputValidationError) else [{"code": "component_infeasible", "path": iid, "message": str(exc)}]
            for error in errors:
                diagnostics.append({**error, "path": iid, "component_path": error["path"]})
            ledger.append({**entry, "status": "infeasible", "diagnostics": diagnostics[-len(errors):]})
            continue
        added = math.sqrt(result["added_noise_variance"])
        if kind == "source":
            loadings = {iid + ":rin": added} if added else {}
            ready = result["latency_ns"] + (op["symbol_duration_ns"] if instance["role"] == "encoded_launch" else 0.0)
        elif kind == "modulator":
            loadings = dict(ins["sample"]["noise_loadings"])
            for key, value in ins["carrier"]["noise_loadings"].items():
                loadings[key] = loadings.get(key, 0.0) + value
            if added:
                loadings[iid + ":modulation"] = added
            ready = max(ins["sample"]["ready_ns"], ins["carrier"]["ready_ns"]) + op["symbol_duration_ns"] + result["latency_ns"]
        else:
            loadings = dict(ins["in"]["noise_loadings"])
            if added:
                loadings[iid + ":detection"] = added
            ready = ins["in"]["ready_ns"] + result["latency_ns"]
        if not math.isfinite(ready) or any(not math.isfinite(v) for v in loadings.values()):
            diagnostics.append({"code": "numeric_range_exceeded", "path": iid, "message": "cumulative timing or noise exceeds finite range"})
            ledger.append({**entry, "status": "infeasible"})
            continue
        for port in component["ports"]:
            if port["direction"] == "out":
                signal[(iid, port["name"])] = {"power_mw": result["output_powers_mw"].get(port["name"]),
                                               "ready_ns": ready, "noise_loadings": loadings}
        ledger.append({**entry, "status": "evaluated", **result, "input_power_mw": input_power,
                       "complete_symbol_ready_ns": ready if instance["role"] != "constant_carrier" else None,
                       "carrier_ready_ns": ready if instance["role"] == "constant_carrier" else None,
                       "noise_loadings": loadings})
    uses = outgoing_uses(graph["network"])[graph["source_node"]]
    receivers = []
    for receiver in graph["receivers"]:
        key = _key(receiver["port"])
        row = {**receiver, "use": uses[receiver["use_index"]]}
        if key in signal:
            incoming_power = signal[incoming[(key[0], "in")]]["power_mw"]
            values = signal[key]
            try:
                variance = math.fsum(v*v for v in values["noise_loadings"].values())
            except OverflowError:
                variance = float("inf")
            if not math.isfinite(variance):
                diagnostics.append({"code": "numeric_range_exceeded", "path": key[0], "message": "receiver noise exceeds finite range"})
                row.update({"status": "infeasible"})
            else:
                detector = cmap[instances[key[0]]["component"]]
                row.update({"status": "evaluated", "full_scale_power_mw": incoming_power,
                            "sensitivity_margin_mw": incoming_power - detector["parameters"]["minimum_full_scale_power_mw"],
                            "overload_margin_mw": detector["operating_envelope"]["max_input_power_mw"] - incoming_power,
                            "complete_symbol_ready_ns": values["ready_ns"], "noise_loadings": values["noise_loadings"],
                            "noise_variance_estimate": variance, "noise_rms_estimate": math.sqrt(variance)})
        else:
            row["status"] = "blocked"
        receivers.append(row)
    terminations = [{**t, "full_scale_dissipated_power_mw": signal.get(_key(t["port"]), {}).get("power_mw")}
                    for t in graph["terminations"]]
    metrics = None
    covariance = None
    if not diagnostics:
        try:
            covariance = [[math.fsum(v * b["noise_loadings"].get(k, 0.0) for k, v in a["noise_loadings"].items())
                           for b in receivers] for a in receivers]
            metrics = {"energy_pj": math.fsum(row["energy_pj"] for row in ledger),
                       "area_um2": math.fsum(row["area_um2"] for row in ledger),
                       "latency_ns": max(row["complete_symbol_ready_ns"] for row in receivers),
                       "noise_rms_estimate": max(row["noise_rms_estimate"] for row in receivers),
                       "stacked_noise_rms_estimate": math.sqrt(math.fsum(row["noise_variance_estimate"] for row in receivers)),
                       "terminated_power_mw": math.fsum(row["full_scale_dissipated_power_mw"] for row in terminations),
                       "instance_count": len(instances), "receiver_count": len(receivers)}
            if any(not math.isfinite(v) for v in metrics.values()) or any(not math.isfinite(v) for row in covariance for v in row):
                raise ValueError("aggregate exceeds finite numeric range")
        except (ValueError, OverflowError):
            diagnostics.append({"code": "numeric_range_exceeded", "path": "metrics", "message": "aggregate exceeds finite numeric range"})
            metrics, covariance = None, None
    return {"feasible": not diagnostics, "diagnostics": diagnostics, "metrics": metrics, "ledger": ledger,
            "receivers": receivers, "receiver_covariance_estimate": covariance, "terminations": terminations,
            "instance_counts": dict(sorted(Counter(cmap[i["component"]]["kind"] for i in instances.values()).items())),
            "scope": json.loads(json.dumps(SCOPE)), "technology_status": {"id": technology["id"], "version": technology["version"],
                "calibration_status": technology["calibration_status"], "review": technology["review"]}}


def realize(network: dict, technology: dict, spec: dict) -> dict:
    """Create a portable record with immutable-by-hash snapshots and evaluation."""
    technology = validate_technology(technology)
    spec = validate_realization_spec(spec)
    graph = build_implementation(network, technology, spec)
    network = graph["network"]
    return {"schema_version": "0.3", "kind": "physical_realization", "network": network, "technology": technology,
            "spec": spec, "graph": graph,
            "hashes": {"network": _hash(network), "technology": _hash(technology), "spec": _hash(spec), "graph": _hash(graph)},
            "evaluation": evaluate_implementation(graph, technology)}


def check_realization(record: dict) -> dict:
    """Replay a generated record from its embedded inputs and compare all outputs.

    SHA256 binds content for reproducibility, not provenance/authenticity. This
    checker also verifies the spec-to-graph relationship; recomputing a tampered
    graph's hash alone does not make it the requested realization.
    """
    diagnostics = []
    required = {"schema_version", "kind", "network", "technology", "spec", "graph", "hashes", "evaluation"}
    if not isinstance(record, dict) or set(record) != required:
        return {"valid": False, "diagnostics": [{"code": "invalid_realization_record", "path": "record", "message": "expected exactly the portable realization record fields"}]}
    try:
        replay = realize(record["network"], record["technology"], record["spec"])
        for field in sorted(required):
            if _hash(record[field]) != _hash(replay[field]):
                diagnostics.append({"code": "realization_replay_mismatch", "path": field, "message": "stored content differs from deterministic reconstruction from embedded inputs"})
    except InputValidationError as exc:
        diagnostics.extend(exc.diagnostics)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        diagnostics.append({"code": "invalid_realization_record", "path": "record", "message": "record cannot be replayed within supported finite JSON limits"})
    return {"valid": not diagnostics, "diagnostics": diagnostics}
