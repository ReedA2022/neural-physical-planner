"""Offline inspection of a physical implementation record.

This module is a presentation layer: it neither evaluates a circuit nor upgrades
its validation, calibration, or review status. The machine-readable record remains
the authoritative artifact.
"""
from __future__ import annotations

from collections import defaultdict, deque
import html
import json
import math
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _value(value: Any) -> str:
    if value is None:
        return '<span class="muted">Not recorded</span>'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _escape(f"{value:.8g}")
    if isinstance(value, (dict, list)):
        return '<pre>' + _escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'
    return _escape(value)


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _detail_table(value: Any, *, table_id: str = "") -> str:
    values = _mapping(value)
    if not values:
        return '<p class="muted">No values recorded.</p>'
    rows = ''.join(f'<tr><th scope="row">{_escape(key)}</th><td>{_value(item)}</td></tr>'
                   for key, item in values.items())
    return f'<div class="scroll"><table id="{_escape(table_id)}"><tbody>{rows}</tbody></table></div>'


def _record_table(records: list[dict], *, table_id: str,
                  preferred: tuple[str, ...] = ()) -> str:
    if not records:
        return '<p class="muted">No entries recorded.</p>'
    keys = list(dict.fromkeys([key for key in preferred if any(key in r for r in records)]
                             + [key for row in records for key in row]))
    head = ''.join(f'<th scope="col">{_escape(key)}</th>' for key in keys)
    rows = ''.join('<tr>' + ''.join(f'<td>{_value(row.get(key))}</td>' for key in keys) + '</tr>'
                   for row in records)
    return (f'<div class="scroll"><table id="{_escape(table_id)}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def _endpoint(value: Any) -> tuple[str, str]:
    """Endpoints in the portable graph are explicit instance/port pairs."""
    if isinstance(value, dict):
        return str(value.get("instance", "")), str(value.get("port", ""))
    return "", ""


def _svg(instances: list[dict], connections: list[dict]) -> str:
    """Render a readable DAG or explain why the complete tables are preferable."""
    if not instances:
        return '<p class="muted">No circuit instances were recorded.</p>'
    if len(instances) > 96:
        return '<p class="muted">Diagram omitted for this large circuit; the complete instance and connection tables follow.</p>'
    identifiers = [str(instance.get("id", "")) for instance in instances]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        return '<p class="muted">Diagram unavailable: instance identifiers are missing or duplicated. Inspect the tables and validation diagnostics.</p>'
    nodes = dict(zip(identifiers, instances))
    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    edges = []
    for connection in connections:
        source, source_port = _endpoint(connection.get("source"))
        target, target_port = _endpoint(connection.get("target"))
        if source not in nodes or target not in nodes:
            return '<p class="muted">Diagram unavailable: a connection references an unknown instance. Inspect the complete connection table.</p>'
        outgoing[source].append(target)
        incoming[target].append(source)
        edges.append((source, source_port, target, target_port, connection.get("domain", "unspecified")))
    degree = {identifier: len(incoming[identifier]) for identifier in identifiers}
    queue = deque(identifier for identifier in identifiers if degree[identifier] == 0)
    ranks = {identifier: 0 for identifier in queue}
    ordered = []
    while queue:
        identifier = queue.popleft()
        ordered.append(identifier)
        for target in outgoing[identifier]:
            ranks[target] = max(ranks.get(target, 0), ranks[identifier] + 1)
            degree[target] -= 1
            if degree[target] == 0:
                queue.append(target)
    if len(ordered) != len(instances):
        return '<p class="muted">Diagram unavailable: the recorded graph contains a cycle. Inspect validation diagnostics and the tables.</p>'
    # Carrier readiness is not a display rank: place fresh optical carriers
    # alongside early detectors, directly above their modulator, so their edges
    # never appear to feed or pass through intermediate detector rectangles.
    for identifier in ordered:
        if (nodes[identifier].get("role") == "constant_carrier"
                and not incoming[identifier] and outgoing[identifier]):
            ranks[identifier] = min(ranks[child] for child in outgoing[identifier]) - 1
    groups: dict[int, list[str]] = defaultdict(list)
    for identifier in ordered:
        groups[ranks[identifier]].append(identifier)
    if len(groups) > 20 or max(map(len, groups.values())) > 20:
        return '<p class="muted">Diagram omitted to preserve legibility; the complete instance and connection tables follow.</p>'
    # Give each leaf a distinct lane and center ancestors over their descendants.
    # This keeps splitter trees ordered without crossing adjacent fanout branches.
    leaves = []
    seen = set()
    def visit(identifier: str):
        if identifier in seen:
            return
        seen.add(identifier)
        if not outgoing[identifier]:
            leaves.append(identifier)
        for child in outgoing[identifier]:
            visit(child)
    for identifier in identifiers:
        if not incoming[identifier]:
            visit(identifier)
    lane = {identifier: float(i) for i, identifier in enumerate(leaves)}
    for identifier in reversed(ordered):
        if outgoing[identifier]:
            lane[identifier] = sum(lane[child] for child in outgoing[identifier]) / len(outgoing[identifier])
    for group in groups.values():
        group.sort(key=lambda identifier: (lane[identifier], identifiers.index(identifier)))
    widest = max(len(group) for group in groups.values())
    width = max(690, widest * 208 + 32)
    height = len(groups) * 126 + 44
    position = {}
    for rank, group in groups.items():
        for index, identifier in enumerate(group):
            position[identifier] = ((width - 208 * len(group)) / 2 + 104 + index * 208, 46 + rank * 126)
    pieces = [f'<div class="scroll circuit"><svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Explicit physical circuit; full port labels appear in the connection table">',
              '<defs><marker id="circuit-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7Z" fill="#7890a4"/></marker></defs>']
    for source, source_port, target, target_port, domain in edges:
        x1, y1 = position[source]
        x2, y2 = position[target]
        midpoint = (y1 + 31 + y2 - 31) / 2
        colour = "#287ca7" if domain == "optical" else "#996514" if domain == "electrical" else "#7890a4"
        dash = ' stroke-dasharray="6 4"' if domain == "electrical" else ""
        label = f"{source}.{source_port} → {target}.{target_port} ({domain})"
        pieces.append(f'<path d="M{x1:g} {y1+31:g} C{x1:g} {midpoint:g} {x2:g} {midpoint:g} {x2:g} {y2-31:g}" fill="none" stroke="{colour}" stroke-width="1.8"{dash} marker-end="url(#circuit-arrow)"><title>{_escape(label)}</title></path>')
    for identifier in ordered:
        instance = nodes[identifier]
        x, y = position[identifier]
        kind = str(instance.get("kind", "unspecified"))
        colour = {"source": "#e4f0ff", "detector": "#fff2dc", "modulator": "#f1eaff",
                  "termination": "#eef0f3", "splitter": "#e4f5f0"}.get(kind, "#f4f7fa")
        # Long identifiers stay available in the title and the authoritative table.
        short_id = identifier if len(identifier) <= 23 else identifier[:20] + "…"
        short_kind = kind if len(kind) <= 23 else kind[:20] + "…"
        role = instance.get("role")
        identifier_y, kind_y = (y - 11, y + 5) if role else (y - 3, y + 17)
        pieces.append(f'<g class="instance-node" data-instance="{_escape(identifier)}"><title>{_escape(identifier)} — {_escape(kind)} — {_escape(role or "")}</title><rect x="{x-92:g}" y="{y-31:g}" width="184" height="62" rx="9" fill="{colour}" stroke="#b9c9d5"/><text x="{x:g}" y="{identifier_y:g}" text-anchor="middle" class="node-id">{_escape(short_id)}</text><text x="{x:g}" y="{kind_y:g}" text-anchor="middle">{_escape(short_kind)}</text>')
        if role:
            pieces.append(f'<text x="{x:g}" y="{y+21:g}" text-anchor="middle">{_escape(str(role).replace("_", " "))}</text>')
        pieces.append('</g>')
    pieces.append('</svg></div><p class="muted">Blue solid: optical. Brown dashed: electrical. Arrows follow signal flow; placement is schematic, not a fabrication layout or a timing scale. Exact ports and signal quantities are listed below.</p>')
    return ''.join(pieces)


_STYLE = """
body{margin:0;background:#f5f7fa;color:#183044;font:15px/1.55 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:32px 24px 64px}h1{font-size:30px;line-height:1.2}h2{font-size:21px;margin-top:0}h3{font-size:17px}p{max-width:100ch}.eyebrow{color:#386e8b;font-weight:700;letter-spacing:.07em;text-transform:uppercase;font-size:12px}.muted{color:#5e7384;font-size:13px}.banner{padding:16px 20px;background:#fff2d8;border-left:4px solid #c88e23}.panel{background:white;border:1px solid #dce4eb;border-radius:10px;padding:21px;margin-top:22px}.facts{display:flex;flex-wrap:wrap;gap:14px}.fact{min-width:170px;border:1px solid #dce4eb;border-radius:8px;padding:15px;background:white}.fact strong{display:block;font-size:21px}.scroll{overflow:auto}.circuit{padding-bottom:10px}svg{display:block;margin:auto}svg text{font:12px system-ui,sans-serif;fill:#263f52}svg .node-id{font-weight:650}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce4eb;padding:10px 9px;overflow-wrap:anywhere}th{font-weight:650;color:#436073}pre{font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:340px;overflow:auto;margin:0}details{margin-top:16px}summary{cursor:pointer;font-weight:650}.status{display:inline-block;font-weight:650;padding:5px 10px;border-radius:6px;background:#e9eef4}.warning{background:#fff0db;color:#704e15}.bad{background:#fde9e7;color:#892b25}@media(max-width:650px){main{padding:20px 12px}.panel{padding:14px}h1{font-size:25px}}@media print{body{background:white}.panel{break-inside:avoid}.scroll{overflow:visible}svg{max-width:100%;height:auto}pre{max-height:none}}
"""


def _display_graph(record: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Resolve declared component port types and add visible boundary markers.

    Resolution here is only for labels. It does not validate connectivity or
    infer any signal powers, costs, or performance.
    """
    graph = _mapping(record.get("graph"))
    technology = _mapping(record.get("technology"))
    components = {c["id"]: c for c in _records(technology.get("components"))
                  if isinstance(c.get("id"), str)}
    instances = []
    portmap = {}
    for instance in _records(graph.get("instances")):
        component_id = instance.get("component")
        component = components.get(component_id, {}) if isinstance(component_id, str) else {}
        ports = _records(component.get("ports"))
        instances.append({**instance, "kind": component.get("kind", "unresolved component"), "ports": ports})
        for port in ports:
            portmap[(str(instance.get("id", "")), str(port.get("name", "")))] = port
    connections = []
    for connection in _records(graph.get("connections")):
        source_type = portmap.get(_endpoint(connection.get("source")), {})
        target_type = portmap.get(_endpoint(connection.get("target")), {})
        connections.append({**connection, "domain": source_type.get("domain", "unresolved"),
                            "quantity": source_type.get("quantity", "unresolved"),
                            "encoding": source_type.get("encoding", "unresolved"),
                            "target_domain": target_type.get("domain", "unresolved"),
                            "target_quantity": target_type.get("quantity", "unresolved"),
                            "target_encoding": target_type.get("encoding", "unresolved")})
    drawing_nodes = list(instances)
    drawing_edges = list(connections)
    existing = {str(instance.get("id", "")) for instance in drawing_nodes}
    for index, termination in enumerate(_records(graph.get("terminations"))):
        identifier = f"termination_{index+1}"
        while identifier in existing:
            identifier = "_" + identifier
        existing.add(identifier)
        drawing_nodes.append({"id": identifier, "kind": "termination", "_boundary": True})
        drawing_edges.append({"source": termination.get("port"), "target": {"instance": identifier, "port": "boundary"}, "domain": "optical"})
    return drawing_nodes, drawing_edges, connections



def _presentation_issues(record: dict) -> list[str]:
    """Check recorded field coverage and bindings, without evaluating physics.

    These conservative checks only govern how stored claims are displayed; they
    do not authenticate hashes, establish numerical truth, or replace replay.
    """
    graph, evaluation = _mapping(record.get("graph")), _mapping(record.get("evaluation"))
    technology = _mapping(record.get("technology"))
    problems = []
    def rows(parent, key):
        value = parent.get(key)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            problems.append(f"{key}: expected a complete list of objects")
            return []
        return value
    def indexed(values, key, label):
        result = {}
        for value in values:
            identifier = value.get(key)
            if not isinstance(identifier, str) or not identifier or identifier in result:
                problems.append(f"{label}: identifiers must be unique nonempty strings")
            else:
                result[identifier] = value
        return result
    def number(value, *, nonnegative=True):
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value) and (not nonnegative or value >= 0)
        except (OverflowError, ValueError):
            return False
    def endpoint(value):
        if (not isinstance(value, dict) or set(value) != {"instance", "port"}
                or any(not isinstance(value.get(k), str) or not value[k] for k in ("instance", "port"))):
            return None
        return value["instance"], value["port"]

    instances = rows(graph, "instances")
    components = rows(technology, "components")
    imap = indexed(instances, "id", "instances")
    cmap = indexed(components, "id", "components")
    if not instances:
        problems.append("instances: no component instances recorded")
    ports = {}
    for instance_id, instance in imap.items():
        component_id = instance.get("component")
        component = cmap.get(component_id) if isinstance(component_id, str) else None
        if component is None:
            problems.append(f"{instance_id}: component reference is unresolved")
            continue
        for port in rows(component, "ports"):
            name = port.get("name")
            if not isinstance(name, str) or not name or (instance_id, name) in ports:
                problems.append(f"{instance_id}: malformed or duplicate port")
                continue
            ports[instance_id, name] = port
    for connection in rows(graph, "connections"):
        source, target = endpoint(connection.get("source")), endpoint(connection.get("target"))
        if source not in ports or target not in ports:
            problems.append("connections: unresolved instance or port reference")
            continue
        a, b = ports[source], ports[target]
        if (a.get("direction") != "out" or b.get("direction") != "in"
                or any(a.get(k) != b.get(k) for k in ("domain", "quantity", "encoding"))):
            problems.append("connections: declared endpoint types or directions disagree")

    ledger = rows(evaluation, "ledger")
    lmap = indexed(ledger, "instance", "ledger")
    if set(lmap) != set(imap) or len(ledger) != len(instances):
        problems.append("ledger: instance coverage differs from the graph")
    for instance_id, row in lmap.items():
        instance = imap.get(instance_id, {})
        component_id = instance.get("component")
        component = cmap.get(component_id, {}) if isinstance(component_id, str) else {}
        if (row.get("status") != "evaluated" or row.get("component") != component_id
                or row.get("kind") != component.get("kind") or row.get("role") != instance.get("role")):
            problems.append(f"{instance_id}: ledger status or component binding is inconsistent")
        for key in ("energy_pj", "area_um2", "latency_ns", "added_noise_variance"):
            if not number(row.get(key)):
                problems.append(f"{instance_id}: ledger {key} is missing or not a finite nonnegative number")

    declared_receivers = rows(graph, "receivers")
    receivers = rows(evaluation, "receivers")
    rows(evaluation, "diagnostics")
    network = _mapping(graph.get("network"))
    selected_source = graph.get("source_node")
    neural_nodes = rows(network, "nodes")
    source_nodes = [node for node in neural_nodes if node.get("id") == selected_source]
    source_size = source_nodes[0].get("size") if len(source_nodes) == 1 else None
    uses = []
    for node in neural_nodes:
        inputs = node.get("inputs", [])
        if not isinstance(inputs, list):
            problems.append("network: malformed input bindings")
            continue
        uses.extend({"target": node.get("id"), "index": index, "output": False}
                    for index, source in enumerate(inputs) if source == selected_source)
    outputs = network.get("outputs")
    if not isinstance(outputs, list):
        problems.append("network: malformed output bindings")
    else:
        uses.extend({"target": f"@output:{index}", "index": index, "output": True}
                    for index, source in enumerate(outputs) if source == selected_source)
    if not isinstance(selected_source, str) or type(source_size) is not int or source_size <= 0 or not uses:
        problems.append("receivers: source node and outgoing NN uses are unresolved")
    declared_bindings = []
    expected = {}
    for receiver in declared_receivers:
        lane, use_index = receiver.get("lane"), receiver.get("use_index")
        if (receiver.get("source_node") != selected_source or type(lane) is not int or type(use_index) is not int
                or type(source_size) is not int or not 0 <= lane < source_size or not 0 <= use_index < len(uses)):
            problems.append("receivers: declared NN lane or outgoing-use binding is invalid")
        else:
            declared_bindings.append((lane, use_index))
        ep = endpoint(receiver.get("port"))
        if ep is None or ep in expected or ep not in ports or ports[ep].get("domain") != "electrical" or ports[ep].get("direction") != "out":
            problems.append("receivers: malformed, duplicate or unresolved declared receiver port")
        else:
            expected[ep] = receiver
    if (type(source_size) is int and source_size > 0
            and (len(declared_bindings) != source_size * len(uses)
                 or len(set(declared_bindings)) != len(declared_bindings))):
        problems.append("receivers: declared bindings do not cover every source lane and ordered NN use exactly once")
    actual = set()
    for receiver in receivers:
        ep = endpoint(receiver.get("port"))
        if ep is None or ep not in expected or ep in actual:
            problems.append("receivers: evaluation has a duplicate or undeclared receiver port")
            continue
        actual.add(ep)
        if (receiver.get("status") != "evaluated"
                or any(receiver.get(k) != expected[ep].get(k) or type(receiver.get(k)) is not type(expected[ep].get(k))
                       for k in ("source_node", "lane", "use_index"))):
            problems.append("receivers: evaluation status or NN binding disagrees with the graph")
        use_index = receiver.get("use_index")
        if type(use_index) is not int or not 0 <= use_index < len(uses) or receiver.get("use") != uses[use_index]:
            problems.append("receivers: displayed NN use disagrees with the source network")
        for key in ("full_scale_power_mw", "sensitivity_margin_mw", "overload_margin_mw",
                    "complete_symbol_ready_ns", "noise_variance_estimate", "noise_rms_estimate"):
            if not number(receiver.get(key)):
                problems.append(f"receivers: {key} is missing or not a finite nonnegative number")
    if not expected or actual != set(expected) or len(receivers) != len(declared_receivers):
        problems.append("receivers: evaluation coverage differs from declared graph boundaries")

    metrics = evaluation.get("metrics")
    required = {"energy_pj", "area_um2", "latency_ns", "noise_rms_estimate", "stacked_noise_rms_estimate",
                "terminated_power_mw", "instance_count", "receiver_count"}
    if not isinstance(metrics, dict) or not required <= set(metrics):
        problems.append("metrics: required aggregate fields are missing")
    else:
        for key in required:
            if not number(metrics[key]):
                problems.append(f"metrics: {key} is not a finite nonnegative number")
        for key, count in (("instance_count", len(instances)), ("receiver_count", len(declared_receivers))):
            if type(metrics[key]) is not int or metrics[key] != count:
                problems.append(f"metrics: {key} disagrees with graph cardinality")
    return list(dict.fromkeys(problems))


def render_implementation(record: dict) -> str:
    """Render a portable ``realize`` result as self-contained, inert HTML.

    A malformed or partial record is inspectable but is never promoted to a
    complete success. This function runs no checker or physical simulation.
    """
    record = _mapping(record)
    graph = _mapping(record.get("graph"))
    evaluation = _mapping(record.get("evaluation"))
    technology = _mapping(record.get("technology"))
    network = _mapping(record.get("network"))
    spec = _mapping(record.get("spec"))
    instances = _records(graph.get("instances"))
    connections = _records(graph.get("connections"))
    boundaries = _records(graph.get("terminations"))
    receivers = _records(evaluation.get("receivers")) or _records(graph.get("receivers"))
    ledger = _records(evaluation.get("ledger"))
    diagnostics = _records(evaluation.get("diagnostics"))
    metrics = evaluation.get("metrics")
    consistency_issues = _presentation_issues(record)
    complete = not consistency_issues
    if evaluation.get("feasible") is False or diagnostics:
        status, status_class = "Recorded infeasible", "bad"
    elif evaluation.get("feasible") is True and complete:
        status, status_class = "Recorded feasible within the declared model", "status"
    else:
        status, status_class = "Incomplete or unrecorded evaluation", "warning"
    title = f"Physical implementation — {network.get('name', 'Unnamed circuit')}"
    pieces = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{_escape(title)}</title><style>{_STYLE}</style></head><body><main>',
              '<p class="eyebrow">Neural Physical Planner · physical subcircuit inspection</p>',
              f'<h1>{_escape(network.get("name", "Unnamed circuit"))}</h1>',
              f'<p>Source NN node: <strong>{_escape(graph.get("source_node", "Not recorded"))}</strong> · Recipe: <strong>{_escape(graph.get("recipe", "Not recorded"))}</strong></p>',
              f'<p><span class="status {status_class}">{status}</span></p>',
              '<p class="muted">Stored feasibility claim: ' + _value(evaluation.get("feasible"))
              + '. Presentation consistency: ' + ("recorded coverage and field types agree" if complete else "incomplete or inconsistent")
              + '. Authenticity and physical correctness remain unverified.</p>',
              '<div class="banner"><strong>Conditional subcircuit model; not hardware signoff.</strong> This report displays the stored record. Rendering does not check its hashes, replay its evaluation, or run an independent simulation. Calibration and expert review status are shown below. Full-scale energy and first-order noise estimates do not establish whole-network cost or task accuracy.</div>',
              '<section class="panel"><h2>Technology and evidence status</h2>',
              _detail_table({"technology_id": technology.get("id"), "version": technology.get("version"),
                             "calibration_status": technology.get("calibration_status"), "review": technology.get("review"),
                             "encoding": technology.get("encoding")}, table_id="technology-status"), '</section>',
              '<section class="panel"><h2>Recorded scope and assumptions</h2>',
              _detail_table(evaluation.get("scope"), table_id="scope"),
              '<h3>Technology assumptions and excluded effects</h3>',
              _detail_table({"assumptions": technology.get("assumptions"), "excluded_effects": technology.get("excluded_effects")}), '</section>',
              '<section class="panel"><h2>Circuit overview</h2><div class="facts">']
    for number, label in [(len(instances), "Component instances"), (len(connections), "Internal connections"),
                          (len(_records(graph.get("receivers"))), "Receiver boundaries"),
                          (len(boundaries), "Matched termination boundaries")]:
        pieces.append(f'<div class="fact"><strong>{number}</strong>{label}</div>')
    drawing_nodes, drawing_edges, typed_connections = _display_graph(record)
    pieces += ['</div>', _svg(drawing_nodes, drawing_edges),
               '<p class="muted">Termination markers represent declared external boundaries, not additional costed component instances. Receiver outputs end at calibrated electrical boundary ports.</p>', '</section>',
               '<section class="panel"><h2>Operating point and recipe settings</h2>', _detail_table(spec, table_id="recipe-settings"), '</section>',
               '<section class="panel"><h2>Aggregate metrics</h2>']
    if evaluation.get("feasible") is True and complete and not diagnostics:
        pieces += ['<p class="muted">Values copied from the stored evaluation. Keys include their units; noise values are normalized first-order estimates.</p>',
                   _detail_table(metrics, table_id="metrics")]
    else:
        pieces += ['<p class="warning">No complete feasible aggregate is available. Partial instance values below must not be interpreted as whole-circuit totals.</p>']
        if metrics is not None:
            pieces += ['<details><summary>Raw aggregate fields from the incomplete record</summary>', _value(metrics), '</details>']
    pieces += ['</section>', '<section class="panel"><h2>Presentation consistency</h2>']
    if consistency_issues:
        pieces.append('<ul>' + ''.join('<li>' + _escape(issue) + '</li>' for issue in consistency_issues) + '</ul>')
    else:
        pieces.append('<p>Stored field coverage and bindings are internally consistent. This is not hash authentication, numerical replay, or physical validation.</p>')
    pieces += ['</section>', '<section class="panel"><h2>Failed guards and diagnostics</h2>']
    if diagnostics:
        pieces.append(_record_table(diagnostics, table_id="diagnostics", preferred=("code", "path", "message")))
    elif evaluation.get("feasible") is True and complete:
        pieces.append('<p>No failed guards were recorded for this nominal evaluation. Independent verification is not established by this report.</p>')
    else:
        pieces.append('<p class="muted">No diagnostics recorded. This does not establish feasibility or evaluation completeness.</p>')
    resolved_instances = [instance for instance in drawing_nodes if not instance.get("_boundary")]
    pieces += ['</section>', '<section class="panel"><h2>Component instances and typed ports</h2>',
               _record_table(resolved_instances, table_id="instances", preferred=("id", "kind", "component", "role", "source_node", "lane", "parameters", "ports")), '</section>',
               '<section class="panel"><h2>Typed connections</h2><p class="muted">Source and target types are resolved from the recorded technology snapshot. The table displays declarations; it does not certify compatibility.</p>',
               _record_table(typed_connections, table_id="connections", preferred=("source", "target", "domain", "quantity", "encoding", "target_domain", "target_quantity", "target_encoding")), '</section>',
               '<section class="panel"><h2>External boundaries and unused optical leaves</h2><h3>Encoded launch ports</h3>',
               _record_table(_records(graph.get("launch_boundaries")), table_id="launch-boundaries", preferred=("instance", "port")),
               '<h3>Matched terminations</h3>',
               _record_table(_records(evaluation.get("terminations")) or boundaries, table_id="terminations", preferred=("port", "model", "reason", "full_scale_dissipated_power_mw")), '</section>',
               '<section class="panel"><h2>Receiver ledger</h2><p class="muted">Each receiver is linked to one scalar lane and one ordered neural use. Repeated operands retain separate receiver entries.</p>',
               _record_table(receivers, table_id="receivers", preferred=("port", "source_node", "lane", "use_index", "use", "status", "full_scale_power_mw", "sensitivity_margin_mw", "overload_margin_mw", "complete_symbol_ready_ns", "noise_rms_estimate")),
               '<details><summary>Recorded receiver covariance estimate</summary>', _value(evaluation.get("receiver_covariance_estimate")), '</details></section>',
               '<section class="panel"><h2>Instance resource and timing ledger</h2><p class="muted">Blocked or infeasible rows remain visible. Source roles distinguish an externally encoded launch from a fresh optical carrier.</p>',
               _record_table(ledger, table_id="resource-ledger", preferred=("instance", "kind", "component", "role", "status", "energy_pj", "area_um2", "latency_ns", "complete_symbol_ready_ns", "carrier_ready_ns", "input_power_mw")), '</section>',
               '<section class="panel"><h2>Evidence provenance</h2>',
               _record_table(_records(technology.get("sources")), table_id="evidence", preferred=("id", "kind", "citation", "version", "coverage", "limitations", "url")),
               '<h3>Component model bindings</h3>',
               _record_table([{key: component.get(key) for key in ("id", "kind", "model_version", "evidence", "notes")}
                              for component in _records(technology.get("components"))], table_id="component-evidence"), '</section>',
               '<section class="panel"><h2>Reproducibility</h2>', _detail_table(record.get("hashes"), table_id="hashes"),
               '<details><summary>Complete stored record</summary>', _value(record), '</details></section>',
               '<p class="muted">Generated locally without scripts, remote assets, or external network dependencies. Machine-readable JSON and the documented checker are the reproducible artifacts.</p></main></body></html>']
    return ''.join(pieces)
