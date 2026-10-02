"""Physical-plan evaluation for a deliberately restricted, explicit surrogate model.

This is not a coherent optical simulator.  A delivered optical channel is modeled
as a normalized additive Gaussian measurement, including optical-to-optical
links.  Component parameters and returned bounds are conditional on that model.

Affine programs propagate independent noise-source *loadings*, not just marginal
variances.  This is essential: noise inserted before a split is shared, and may
cancel or reinforce when branches meet.  Nonlinear programs use an unconditional
L2 Lipschitz/Minkowski upper bound.  Monte Carlo executes the physical operations
independently; it never samples the analytically computed output covariance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


_HC = 6.62607015e-34 * 299792458.0
_KINDS = {"digital_copy", "passive_split", "source_boost", "amplify", "regenerate", "serial_reencode"}
_ASSUMPTIONS = [
    "All component and rule parameters are declared model inputs; no measured-device accuracy is implied.",
    "Neural operators act on signed real normalized values. A concrete optical encoding of signs, weights and nonlinearities, digital fixed-point rounding, and physical macro internals are outside this backend.",
    "Optical links are normalized additive Gaussian computational channels, including optical-to-optical links; coherent-wave propagation and a device-level circuit backend are not implemented.",
    "For a delivered optical branch, shot-surrogate variance is 1/(photons*split_fraction*optical_efficiency*effective_gain), independent of the numerical signal value.",
    "All primitive noise sources are independent zero-mean isotropic Gaussian vectors. A source reused by branches remains shared, including inherited component noise and pre-split amplifier noise.",
    "Noise_std values are in normalized numerical units. Optical gain changes photon budget, not the ideal numerical value. Existing numerical error is never erased by amplification or regeneration.",
    "Nominal amplitude guards use conservative interval enclosures; sqrt(gain) times the maximum absolute nominal value must fit the originating optical macro's declared headroom. This field-amplitude surrogate is not a detailed encoding model.",
    "Nominal range guards do not bound Gaussian tails or guarantee absence of saturation on every stochastic sample. No clipping is applied in the additive model.",
    "Error is sqrt(E[||stacked digital outputs - ideal outputs||_2^2]); it is an RMS quantity, not a deterministic error certificate.",
    "Scheduling uses dedicated neural macros, unlimited compatible interconnect, ASAP dependencies, no contention except declared serial re-encoding; each graph evaluation is one inference, with no batching or throughput guarantee.",
    "Declared rule costs describe one complete fanout recipe. This version has no arity-dependent splitter-tree area model or maximum-port guard; use costs characterized for the intended fanout size. Serial dynamic costs repeat per encoding.",
    "Optical energy uses photons*dimension*hc/wavelength/wall_plug_efficiency. Amplifier and regeneration optical replenishment use the originating macro's wall-plug efficiency as an explicit surrogate, plus the rule's declared electrical overhead.",
    "Sensitivity splitting is a deterministic downstream Lipschitz heuristic, not a proof of globally optimal continuous power allocation.",
]


def outgoing_uses(network: dict) -> dict[str, list[dict]]:
    """Canonical outgoing edges; repeated operands and output sinks count separately."""
    uses: dict[str, list[dict]] = {n["id"]: [] for n in network["nodes"]}
    for node in network["nodes"]:
        for index, source in enumerate(node.get("inputs", [])):
            uses[source].append({"target": node["id"], "index": index, "output": False})
    for index, source in enumerate(network["outputs"]):
        uses[source].append({"target": f"@output:{index}", "index": index, "output": True})
    return uses


def infer_bounds(network: dict) -> dict[str, dict[str, list[float]]]:
    """Sound noiseless interval enclosures (correlations can make them loose)."""
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for node in network["nodes"]:
        op, inputs = node["op"], node.get("inputs", [])
        if op == "input":
            lo = np.asarray(node["input_bounds"]["lower"], dtype=float)
            hi = np.asarray(node["input_bounds"]["upper"], dtype=float)
        elif op == "linear":
            weight = np.asarray(node["weights"], dtype=float)
            bias = np.asarray(node.get("bias", [0.0] * node["size"]), dtype=float)
            lower, upper = bounds[inputs[0]]
            positive, negative = np.maximum(weight, 0), np.minimum(weight, 0)
            lo = positive @ lower + negative @ upper + bias
            hi = positive @ upper + negative @ lower + bias
        elif op == "relu":
            lower, upper = bounds[inputs[0]]
            lo, hi = np.maximum(lower, 0), np.maximum(upper, 0)
        elif op == "identity":
            lo, hi = bounds[inputs[0]]
        elif op == "add":
            lo = np.sum([bounds[source][0] for source in inputs], axis=0)
            hi = np.sum([bounds[source][1] for source in inputs], axis=0)
        else:
            raise ValueError(f"Unsupported operation: {op}")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError(f"Nonfinite nominal bounds at node {node['id']}")
        bounds[node["id"]] = (lo, hi)
    return {key: {"lower": lower.tolist(), "upper": upper.tolist()} for key, (lower, upper) in bounds.items()}


def _diag(code: str, path: str, message: str) -> dict:
    return {"code": code, "path": path, "message": message}


def _float(record: dict, field: str, default: float = 0.0) -> float:
    return float(record.get(field, default))


def _exceeds(actual: float, maximum: float) -> bool:
    """Relative tolerance only: positive quantities cannot pass a zero limit."""
    return actual > maximum and not math.isclose(actual, maximum, rel_tol=1e-12, abs_tol=0.0)


def _node_gain(node: dict) -> float:
    return float(np.linalg.norm(np.asarray(node["weights"], dtype=float), 2)) if node["op"] == "linear" else 1.0


def _sensitivities(network: dict, uses: dict) -> dict[str, list[float]]:
    nodes = {node["id"]: node for node in network["nodes"]}
    downstream: dict[str, float] = {}
    result = {}
    for node in reversed(network["nodes"]):
        values = [1.0 if edge["output"] else downstream[edge["target"]] * _node_gain(nodes[edge["target"]]) for edge in uses[node["id"]]]
        result[node["id"]] = values
        downstream[node["id"]] = sum(values)
    return result


def _photon_energy(component: dict, dimension: int) -> float:
    if component["domain"] != "optical":
        return 0.0
    return (_float(component, "photons") * dimension * _HC / (_float(component, "wavelength_nm") * 1e-9) / _float(component, "wall_plug_efficiency") * 1e12)


def _prepare(network: dict, library: dict, request: dict, decision: dict) -> dict:
    nodes = {node["id"]: node for node in network["nodes"]}
    components = {component["id"]: component for component in library["components"]}
    rules = {rule["id"]: rule for rule in library["rules"]}
    conversions = {(item["from_domain"], item["to_domain"]): item for item in library.get("conversions", [])}
    uses = outgoing_uses(network)
    bounds = infer_bounds(network)
    diagnostics = []
    chosen_components, chosen_rules = {}, {}
    component_decisions = decision.get("components", {})
    rule_decisions = decision.get("fanout_rules", {})
    if not isinstance(component_decisions, dict) or not isinstance(rule_decisions, dict):
        raise ValueError("Decision components and fanout_rules must be mappings")
    for unknown in set(component_decisions) - set(nodes):
        diagnostics.append(_diag("unknown_node", f"decision.components.{unknown}", "Component assignment references an unknown node."))
    fanout_nodes = {node_id for node_id, edges in uses.items() if len(edges) > 1}
    for unknown in set(rule_decisions) - fanout_nodes:
        diagnostics.append(_diag("unexpected_fanout_rule", f"decision.fanout_rules.{unknown}", "A rule is only allowed for a node with more than one outgoing use."))
    for node_id, node in nodes.items():
        path = f"decision.components.{node_id}"
        component_id = component_decisions.get(node_id)
        if component_id not in components:
            diagnostics.append(_diag("unknown_component", path, f"Missing or unknown component {component_id!r}."))
            continue
        component = components[component_id]
        chosen_components[node_id] = component
        if node["op"] not in component["ops"]:
            diagnostics.append(_diag("unsupported_operation", path, f"{component_id} does not implement {node['op']}."))
        if component["domain"] not in request["allowed_domains"]:
            diagnostics.append(_diag("forbidden_domain", path, f"Domain {component['domain']} is not allowed."))
        if node_id in request.get("required_editable", []) and not component["editable"]:
            diagnostics.append(_diag("editability_required", path, f"Node {node_id} must remain editable."))
        if node["size"] > component["capacity"]:
            diagnostics.append(_diag("capacity_exceeded", path, f"Output width {node['size']} exceeds macro capacity {component['capacity']}."))
        maximum = max(abs(value) for value in bounds[node_id]["lower"] + bounds[node_id]["upper"])
        if _exceeds(maximum, _float(component, "max_abs_value")):
            diagnostics.append(_diag("nominal_range_exceeded", path, f"Nominal interval amplitude {maximum:.9g} exceeds macro range {_float(component, 'max_abs_value'):.9g}."))
        if node_id not in fanout_nodes:
            continue
        rule_id = rule_decisions.get(node_id)
        if rule_id not in rules:
            diagnostics.append(_diag("unknown_fanout_rule", f"decision.fanout_rules.{node_id}", f"Missing or unknown fanout rule {rule_id!r}."))
            continue
        rule = rules[rule_id]
        chosen_rules[node_id] = rule
        kind = rule["kind"]
        rule_path = f"decision.fanout_rules.{node_id}"
        if kind not in _KINDS:
            diagnostics.append(_diag("unsupported_rule_kind", rule_path, f"No executable semantics for {kind}."))
        if component["domain"] not in rule["domains"] or (kind == "digital_copy") != (component["domain"] == "digital"):
            diagnostics.append(_diag("rule_domain_mismatch", rule_path, f"Rule {rule_id} cannot operate in {component['domain']}."))
        if kind in {"source_boost", "serial_reencode"} and (not component.get("source_scalable", False) or node["op"] != "input"):
            diagnostics.append(_diag("source_scaling_forbidden", rule_path, f"{kind} requires a source-scalable optical input encoder, not an intermediate value."))
        if kind == "serial_reencode" and not request.get("allow_serialization", False):
            diagnostics.append(_diag("serialization_forbidden", rule_path, "Sequential branch encoding is forbidden by the request."))
        if component["domain"] == "optical" and kind in {"source_boost", "amplify", "regenerate"}:
            physical_maximum = math.sqrt(_float(rule, "gain", 1.0)) * maximum
            if _exceeds(physical_maximum, _float(component, "max_abs_value")):
                diagnostics.append(_diag("optical_headroom_exceeded", rule_path, f"sqrt(gain)*nominal amplitude={physical_maximum:.9g} exceeds declared optical headroom {_float(component, 'max_abs_value'):.9g}."))
    result = {"nodes": nodes, "components": chosen_components, "rules": chosen_rules, "uses": uses, "bounds": bounds, "diagnostics": diagnostics, "network": network}
    if diagnostics:
        return result

    sensitivities = _sensitivities(network, uses)
    links, incoming, output_links, common_sources, component_sources = {}, {}, {}, {}, {}
    noise_sources = []

    def source(source_id: str, dimension: int, std: float, role: str, node_id: str) -> dict:
        if not math.isfinite(std) or not math.isfinite(std * std):
            raise ValueError(f"Noise source {source_id} exceeds finite floating-point range")
        record = {"id": source_id, "dimension": dimension, "std": std, "variance": std * std, "role": role, "node_id": node_id}
        if std > 0:
            noise_sources.append(record)
        return record

    for node_id, node in nodes.items():
        component = chosen_components[node_id]
        rule = chosen_rules.get(node_id)
        kind = rule["kind"] if rule else None
        edges = uses[node_id]
        dimension = node["size"]
        optical = component["domain"] == "optical"
        gain = _float(rule, "gain", 1.0) if rule and kind in {"source_boost", "amplify", "regenerate"} else 1.0
        component_sources[node_id] = source(f"component:{node_id}", dimension, 0.0 if kind == "serial_reencode" else _float(component, "noise_std"), "component_output", node_id)
        common_sources[node_id] = []
        if rule and kind != "serial_reencode":
            common_sources[node_id].append(source(f"rule:{node_id}:shared", dimension, _float(rule, "added_noise_std"), "shared_measurement" if kind == "regenerate" else "shared_rule_noise", node_id))
        if kind == "regenerate":
            common_sources[node_id].append(source(f"rule:{node_id}:measurement_shot", dimension, math.sqrt(1.0 / (_float(component, "photons") * _float(component, "optical_efficiency"))), "shared_regenerator_detection", node_id))
        fractions = [1.0] * len(edges)
        if optical and kind not in {"regenerate", "serial_reencode"} and len(edges) > 1:
            if rule.get("split_policy", "equal") == "sensitivity":
                weights = np.asarray(sensitivities[node_id], dtype=float)
                # Zero influence still gets a physical branch, with explicit positive allocation.
                weights = np.maximum(weights, max(float(np.max(weights)), 1.0) * 1e-9)
                fractions = (weights / np.sum(weights)).tolist()
            else:
                fractions = [1.0 / len(edges)] * len(edges)
        for use_index, edge in enumerate(edges):
            target_domain = "digital" if edge["output"] else chosen_components[edge["target"]]["domain"]
            conversion = None
            if component["domain"] != target_domain:
                conversion = conversions.get((component["domain"], target_domain))
                if conversion is None:
                    diagnostics.append(_diag("missing_conversion", f"edge.{node_id}.{use_index}", f"No {component['domain']}->{target_domain} conversion is available."))
            edge_id = f"edge:{node_id}:{use_index}"
            edge_noise = []
            if optical:
                shot_variance = 1.0 / (_float(component, "photons") * fractions[use_index] * _float(component, "optical_efficiency") * gain)
                edge_noise.append(source(f"{edge_id}:shot", dimension, math.sqrt(shot_variance), "independent_branch_shot_surrogate", node_id))
            if kind in {"serial_reencode", "regenerate"}:
                edge_noise.append(source(f"{edge_id}:encoding", dimension, _float(rule, "added_noise_std"), "independent_reencoding", node_id))
            if kind == "serial_reencode":
                edge_noise.append(source(f"{edge_id}:source", dimension, _float(component, "noise_std"), "independent_source_encoding", node_id))
            if conversion:
                edge_noise.append(source(f"{edge_id}:conversion", dimension, _float(conversion, "noise_std"), "domain_conversion", node_id))
            link = {"id": edge_id, "source": node_id, "use_index": use_index, **edge, "source_domain": component["domain"], "target_domain": target_domain, "split_fraction": fractions[use_index], "effective_gain": gain, "noise": edge_noise, "conversion": conversion}
            links[edge_id] = link
            if edge["output"]:
                output_links[edge["index"]] = edge_id
            else:
                incoming[(edge["target"], edge["index"])] = edge_id
    result.update({"links": links, "incoming": incoming, "output_links": output_links, "common_sources": common_sources, "component_sources": component_sources, "noise_sources": noise_sources})
    return result


def _plus_source(loadings: dict[str, np.ndarray], record: dict) -> dict[str, np.ndarray]:
    if record["std"] == 0:
        return loadings
    result = dict(loadings)
    result[record["id"]] = np.eye(record["dimension"]) * record["std"]
    return result


def _add_loadings(operands: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for operand in operands:
        for source_id, matrix in operand.items():
            result[source_id] = result[source_id] + matrix if source_id in result else matrix.copy()
    return result


def _analytic(prepared: dict) -> dict:
    exact = all(node["op"] != "relu" for node in prepared["network"]["nodes"])
    delivered: dict[str, dict[str, np.ndarray]] = {}
    delivered_bounds: dict[str, float] = {}
    for node_id, node in prepared["nodes"].items():
        operands = [prepared["incoming"][(node_id, index)] for index, _ in enumerate(node.get("inputs", []))]
        incoming_bounds = [delivered_bounds[edge_id] for edge_id in operands]
        op = node["op"]
        if op == "input":
            loadings, rms_bound = {}, 0.0
        elif op == "linear":
            weight = np.asarray(node["weights"], dtype=float)
            loadings = {key: weight @ matrix for key, matrix in delivered[operands[0]].items()} if exact else {}
            rms_bound = _node_gain(node) * incoming_bounds[0]
        elif op == "add":
            loadings = _add_loadings([delivered[edge_id] for edge_id in operands]) if exact else {}
            rms_bound = sum(incoming_bounds)
        else:
            loadings = dict(delivered[operands[0]]) if exact else {}
            rms_bound = incoming_bounds[0]
        sources = [prepared["component_sources"][node_id], *prepared["common_sources"][node_id]]
        for record in sources:
            if exact:
                loadings = _plus_source(loadings, record)
            rms_bound = math.hypot(rms_bound, math.sqrt(record["dimension"]) * record["std"])
        for use_index in range(len(prepared["uses"][node_id])):
            edge_id = f"edge:{node_id}:{use_index}"
            link_loadings = dict(loadings)
            link_bound = rms_bound
            for record in prepared["links"][edge_id]["noise"]:
                if exact:
                    link_loadings = _plus_source(link_loadings, record)
                link_bound = math.hypot(link_bound, math.sqrt(record["dimension"]) * record["std"])
            delivered[edge_id] = link_loadings
            delivered_bounds[edge_id] = link_bound
    output_ids = [prepared["output_links"][index] for index in range(len(prepared["network"]["outputs"]))]
    result: dict[str, Any] = {"kind": "exact_affine_gaussian" if exact else "conservative_lipschitz_rms", "nominal_bounds": prepared["bounds"], "assumptions": list(_ASSUMPTIONS)}
    if exact:
        dimensions = [prepared["nodes"][node_id]["size"] for node_id in prepared["network"]["outputs"]]
        dimension = sum(dimensions)
        covariance = np.zeros((dimension, dimension), dtype=float)
        source_dimensions = {record["id"]: record["dimension"] for record in prepared["noise_sources"]}
        source_ids = set().union(*(set(delivered[edge_id]) for edge_id in output_ids))
        for source_id in sorted(source_ids):
            matrix = np.vstack([delivered[edge_id].get(source_id, np.zeros((width, source_dimensions[source_id]))) for edge_id, width in zip(output_ids, dimensions)])
            covariance += matrix @ matrix.T
        result["output_covariance"] = covariance.tolist()
        result["error_rms_bound"] = math.sqrt(max(0.0, float(np.trace(covariance))))
        result["error_bound_method"] = "Exact covariance from independent source loadings, preserving all shared-source paths and repeated operands."
    else:
        result["error_rms_bound"] = math.sqrt(sum(delivered_bounds[edge_id] ** 2 for edge_id in output_ids))
        result["error_bound_method"] = "Spectral-norm propagation for linear nodes; ReLU is 1-Lipschitz; Minkowski at add nodes; quadrature only for newly independent zero-mean sources; Euclidean aggregation across output blocks."
        result["per_output_rms_bounds"] = [delivered_bounds[edge_id] for edge_id in output_ids]
    return result


def _hardware(prepared: dict) -> tuple[dict, dict, list]:
    instances, connections, schedule, trace = [], [], [], []
    energy, area = 0.0, 0.0
    delivered_time = {}
    for node_id, node in prepared["nodes"].items():
        component = prepared["components"][node_id]
        rule = prepared["rules"].get(node_id)
        kind = rule["kind"] if rule else None
        branch_count = len(prepared["uses"][node_id])
        serial = kind == "serial_reencode"
        repetitions = branch_count if serial else 1
        gain = _float(rule, "gain", 1.0) if rule and kind in {"source_boost", "amplify", "regenerate"} else 1.0
        nominal_photon_energy = _photon_energy(component, node["size"])
        launch_multiplier = repetitions if serial else 1.0
        if kind in {"source_boost", "amplify"}:
            launch_multiplier = gain
        elif kind == "regenerate":
            launch_multiplier = 1.0 + branch_count * gain
        macs = node["size"] * len(node["weights"][0]) if node["op"] == "linear" else 0
        component_energy = repetitions * (_float(component, "energy_pj") + macs * _float(component, "energy_per_mac_pj")) + nominal_photon_energy * launch_multiplier
        component_area = _float(component, "area_um2")
        energy += component_energy
        area += component_area
        node_instance = f"node:{node_id}"
        start = max((delivered_time[prepared["incoming"][(node_id, index)]] for index, _ in enumerate(node.get("inputs", []))), default=0.0)
        finish = start + _float(component, "latency_ns")
        instances.append({"id": node_instance, "kind": "neural_macro", "component_id": component["id"], "node_id": node_id, "op": node["op"], "domain": component["domain"], "size": node["size"], "editable": component["editable"], "macs": macs, "area_um2": component_area, "energy_pj": component_energy, "photon_energy_pj": nominal_photon_energy * launch_multiplier, "encoding_count": repetitions, "launch_energy_multiplier": launch_multiplier})
        schedule.append({"instance_id": node_instance, "operation": "compute", "start_ns": start, "end_ns": finish})
        rule_instance = None
        if rule:
            rule_instance = f"rule:{node_id}"
            rule_energy = repetitions * _float(rule, "energy_pj")
            energy += rule_energy
            area += _float(rule, "area_um2")
            instances.append({"id": rule_instance, "kind": "fanout_rule", "rule_id": rule["id"], "rule_kind": kind, "node_id": node_id, "gain": gain, "split_policy": rule.get("split_policy", "equal"), "area_um2": _float(rule, "area_um2"), "energy_pj": rule_energy})
            connections.append({"source": node_instance, "target": rule_instance, "role": "fanout_input"})
            trace.append({"node_id": node_id, "component_id": component["id"], "rule_id": rule["id"], "kind": kind, "guards_passed": True, "gain": gain, "split_fractions": [prepared["links"][f"edge:{node_id}:{index}"]["split_fraction"] for index in range(branch_count)], "inherited_error_preserved": True, "semantics": "shared measurement followed by independent full-power re-encoding" if kind == "regenerate" else "independent sequential encoding of exact source input" if serial else "shared normalized signal with independent branch detection"})
            if not serial:
                schedule.append({"instance_id": rule_instance, "operation": kind, "start_ns": finish, "end_ns": finish + _float(rule, "latency_ns")})
        for use_index, edge in enumerate(prepared["uses"][node_id]):
            edge_id = f"edge:{node_id}:{use_index}"
            link = prepared["links"][edge_id]
            cursor = finish
            upstream = node_instance
            if rule:
                if serial:
                    extra = use_index * (_float(component, "latency_ns") + _float(rule, "latency_ns"))
                    if use_index > 0:
                        schedule.append({"instance_id": node_instance, "operation": "reencode", "branch_index": use_index, "start_ns": start + extra, "end_ns": finish + extra})
                    cursor = finish + extra
                    schedule.append({"instance_id": rule_instance, "operation": kind, "branch_index": use_index, "start_ns": cursor, "end_ns": cursor + _float(rule, "latency_ns")})
                cursor += _float(rule, "latency_ns")
                upstream = rule_instance
            channel_id = f"channel:{node_id}:{use_index}"
            instances.append({"id": channel_id, "kind": "normalized_channel", "domain": component["domain"], "split_fraction": link["split_fraction"], "effective_gain": link["effective_gain"], "effective_detected_photons": _float(component, "photons") * link["split_fraction"] * _float(component, "optical_efficiency") * link["effective_gain"] if component["domain"] == "optical" else None, "noise_source_ids": [record["id"] for record in link["noise"] if record["std"] > 0], "area_um2": 0.0, "energy_pj": 0.0})
            connections.append({"source": upstream, "target": channel_id, "role": "branch", "branch_index": use_index})
            schedule.append({"instance_id": channel_id, "operation": "deliver", "start_ns": cursor, "end_ns": cursor})
            upstream = channel_id
            conversion = link["conversion"]
            if conversion:
                conversion_id = f"conversion:{node_id}:{use_index}"
                conversion_energy = _float(conversion, "energy_pj")
                conversion_area = _float(conversion, "area_um2")
                energy += conversion_energy
                area += conversion_area
                instances.append({"id": conversion_id, "kind": "domain_conversion", "conversion_id": conversion["id"], "from_domain": conversion["from_domain"], "to_domain": conversion["to_domain"], "area_um2": conversion_area, "energy_pj": conversion_energy})
                connections.append({"source": upstream, "target": conversion_id, "role": "conversion_input"})
                schedule.append({"instance_id": conversion_id, "operation": "convert", "start_ns": cursor, "end_ns": cursor + _float(conversion, "latency_ns")})
                cursor += _float(conversion, "latency_ns")
                upstream = conversion_id
            if edge["output"]:
                target_id = f"output:{edge['index']}"
                instances.append({"id": target_id, "kind": "output_sink", "domain": "digital", "node_id": node_id, "size": node["size"], "area_um2": 0.0, "energy_pj": 0.0})
                schedule.append({"instance_id": target_id, "operation": "output", "start_ns": cursor, "end_ns": cursor})
            else:
                target_id = f"node:{edge['target']}"
            connections.append({"source": upstream, "target": target_id, "input_index": edge["index"], "role": "output" if edge["output"] else "operand"})
            delivered_time[edge_id] = cursor
    latency = max(delivered_time[edge_id] for edge_id in prepared["output_links"].values())
    hardware = {"representation": "abstract_physical_implementation_plan_v1", "device_netlist": False, "instances": instances, "connections": connections, "schedule": sorted(schedule, key=lambda item: (item["start_ns"], item["end_ns"], item["instance_id"]))}
    return hardware, {"energy_pj": energy, "latency_ns": latency, "area_um2": area}, trace


def _failure(diagnostics: list[dict], bounds: dict | None = None, decision: dict | None = None) -> dict:
    return {"feasible": False, "metrics": {}, "diagnostics": diagnostics, "analysis": {"kind": "invalid_plan", "nominal_bounds": bounds or {}, "assumptions": list(_ASSUMPTIONS)}, "trace": [{"guards_passed": False, "diagnostics": diagnostics, "decision": decision or {}}], "hardware": {"representation": "abstract_physical_implementation_plan_v1", "device_netlist": False, "instances": [], "connections": [], "schedule": []}, "noise_sources": []}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(value)
    return True


def evaluate(network: dict, library: dict, request: dict, decision: dict) -> dict:
    """Evaluate a decision; infeasible candidates produce diagnostics, not exceptions."""
    if not isinstance(decision, dict):
        return _failure([_diag("invalid_physical_decision", "decision", "Decision must be a JSON object with component and fanout-rule mappings.")])
    try:
        prepared = _prepare(network, library, request, decision)
        if prepared["diagnostics"]:
            return _failure(prepared["diagnostics"], prepared["bounds"], decision)
        analysis = _analytic(prepared)
        hardware, metrics, trace = _hardware(prepared)
        metrics["error_rms_bound"] = analysis.pop("error_rms_bound")
        if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
            return _failure([_diag("nonfinite_physics", "metrics", "Evaluation overflowed or produced invalid metrics; rescale inputs or component parameters.")], prepared["bounds"], decision)
        diagnostics = []
        for name, maximum in request.get("budgets", {}).items():
            if maximum is not None and _exceeds(metrics[name], float(maximum)):
                diagnostics.append(_diag("budget_exceeded", f"request.budgets.{name}", f"Predicted {name}={metrics[name]:.9g} exceeds budget {maximum:.9g}."))
        result = {"feasible": not diagnostics, "metrics": metrics, "diagnostics": diagnostics, "analysis": analysis, "trace": trace, "hardware": hardware, "noise_sources": prepared["noise_sources"]}
        if not _finite_tree(result):
            return _failure([_diag("nonfinite_physics", "evaluation", "An intermediate physical quantity exceeded finite floating-point range; rescale model inputs.")], prepared["bounds"], decision)
        return result
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, OverflowError, np.linalg.LinAlgError) as exc:
        return _failure([_diag("invalid_physical_decision", "decision", str(exc))], decision=decision if isinstance(decision, dict) else None)


def _apply(node: dict, operands: list[np.ndarray], inputs: np.ndarray) -> np.ndarray:
    op = node["op"]
    if op == "input":
        return inputs.copy()
    if op == "linear":
        return operands[0] @ np.asarray(node["weights"], dtype=float).T + np.asarray(node.get("bias", [0.0] * node["size"]), dtype=float)
    if op == "add":
        return np.sum(operands, axis=0)
    if op == "relu":
        return np.maximum(operands[0], 0)
    return operands[0].copy()


def simulate(network: dict, library: dict, request: dict, decision: dict, samples: int = 10000, seed: int = 0) -> dict:
    """Independent physical forward sampling, with sampling uncertainty summaries.

    Inputs are uniform over their declared box.  Affine error is independent of
    this input distribution; the nonlinear analytic bound applies to every input.
    Samples demonstrate consistency with the model, not real-device validation.
    """
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer >= 2 to estimate sampling uncertainty")
    evaluation = evaluate(network, library, request, decision)
    if not evaluation["metrics"]:
        return {"feasible": False, "samples": 0, "seed": seed, "diagnostics": evaluation["diagnostics"]}
    prepared = _prepare(network, library, request, decision)
    rng = np.random.default_rng(seed)
    input_node = next(node for node in network["nodes"] if node["op"] == "input")
    inputs = rng.uniform(np.asarray(input_node["input_bounds"]["lower"], dtype=float), np.asarray(input_node["input_bounds"]["upper"], dtype=float), size=(samples, input_node["size"]))
    ideal, delivered = {}, {}

    def add_noise(values: np.ndarray, record: dict) -> np.ndarray:
        if record["std"] == 0:
            return values
        return values + rng.normal(0.0, record["std"], size=(samples, record["dimension"]))

    for node in network["nodes"]:
        node_id = node["id"]
        nominal_operands = [ideal[source] for source in node.get("inputs", [])]
        physical_operands = [delivered[prepared["incoming"][(node_id, index)]] for index, _ in enumerate(node.get("inputs", []))]
        ideal[node_id] = _apply(node, nominal_operands, inputs)
        physical = _apply(node, physical_operands, inputs)
        physical = add_noise(physical, prepared["component_sources"][node_id])
        for record in prepared["common_sources"][node_id]:
            physical = add_noise(physical, record)
        for use_index in range(len(prepared["uses"][node_id])):
            edge_id = f"edge:{node_id}:{use_index}"
            branch = physical.copy()
            for record in prepared["links"][edge_id]["noise"]:
                branch = add_noise(branch, record)
            delivered[edge_id] = branch
    ideal_output = np.concatenate([ideal[node_id] for node_id in network["outputs"]], axis=1)
    actual_output = np.concatenate([delivered[prepared["output_links"][index]] for index in range(len(network["outputs"]))], axis=1)
    error = actual_output - ideal_output
    squared_error = np.sum(error * error, axis=1)
    mse = float(np.mean(squared_error))
    mse_se = float(np.std(squared_error, ddof=1) / math.sqrt(samples))
    rms = math.sqrt(max(0.0, mse))
    rms_se = mse_se / (2 * rms) if rms > 0 else 0.0
    bound = evaluation["metrics"]["error_rms_bound"]
    exact = evaluation["analysis"]["kind"] == "exact_affine_gaussian"
    expected_mse = bound * bound
    difference = mse - expected_mse
    within_roundoff = math.isclose(mse, expected_mse, rel_tol=1e-12, abs_tol=0.0)
    consistent = within_roundoff or (abs(difference) <= 4 * mse_se if exact else difference <= 4 * mse_se)
    diagnostics = list(evaluation["diagnostics"])
    if not consistent:
        diagnostics.append(_diag(
            "monte_carlo_mismatch", "simulation.mse_empirical",
            "Empirical MSE does not agree with the exact affine prediction within four estimated standard errors. Possible causes include a modeling or implementation discrepancy, a sampling fluctuation, or floating-point cancellation."
            if exact else "Empirical MSE exceeds the modeled RMS bound plus four estimated standard errors.",
        ))
    result = {"feasible": evaluation["feasible"], "samples": int(samples), "seed": int(seed), "error_rms_empirical": rms, "error_rms_bound": bound, "mse_empirical": mse, "mse_standard_error": mse_se, "rms_standard_error": rms_se, "affine_expected_mse": expected_mse if exact else None, "consistent_with_bound": consistent, "sampling_diagnostic": "Affine predictions use a two-sided MSE comparison; nonlinear bounds use an upper-bound comparison. Four estimated standard errors and relative-only roundoff tolerance are allowed. This is a heuristic consistency check, not a formal confidence guarantee.", "numerical_limitations": "Finite-precision forward sampling can erase small noise added to very large signals; subtraction of nearly equal outputs can lose error information. An affine empirical MSE below its prediction is checked as a discrepancy, including zero estimated standard error.", "input_sampling": "independent uniform samples from the declared input box", "execution": "independent forward sampling of primitive noise sources; no output-covariance sampler", "outputs": {"dimension": int(error.shape[1]), "ideal_mean": np.mean(ideal_output, axis=0).tolist(), "physical_mean": np.mean(actual_output, axis=0).tolist(), "error_mean": np.mean(error, axis=0).tolist(), "error_std": np.std(error, axis=0, ddof=1).tolist()}, "diagnostics": diagnostics}
    if exact:
        result["mse_difference_standard_errors"] = difference / mse_se if mse_se > 0 else (0.0 if within_roundoff else None)
    return result
