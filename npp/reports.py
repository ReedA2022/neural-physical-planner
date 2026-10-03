"""Portable, self-contained reports and CSV exports (no external web assets)."""

from __future__ import annotations

import csv
from decimal import Decimal
import html
import io
import json
from pathlib import Path


METRICS = ("energy_pj", "latency_ns", "area_um2", "error_rms_bound")
LABELS = {"energy_pj": "Energy / inference (pJ)", "latency_ns": "Latency (ns)",
          "area_um2": "Macro area (µm²)", "error_rms_bound": "Output RMS error bound"}


def _escape(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return _escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _number(value):
    return f"{float(value):.6g}" if isinstance(value, (int, float)) else "—"


def csv_text(report: dict) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["plan_id", *METRICS,
                                               "analysis_kind", "components", "fanout_rules"])
    writer.writeheader()
    for plan in report.get("plans", []):
        evaluation = plan["evaluation"]
        writer.writerow({"plan_id": plan["id"], **evaluation["metrics"],
                         "analysis_kind": evaluation["analysis"].get("kind", "unknown"),
                         "components": json.dumps(plan["decision"]["components"], sort_keys=True),
                         "fanout_rules": json.dumps(plan["decision"]["fanout_rules"], sort_keys=True)})
    return stream.getvalue()


def _scatter(plans: list) -> str:
    if not plans:
        return '<p class="empty">No feasible designs were found. Inspect search completeness and rejection reasons.</p>'
    points = [(p["evaluation"]["metrics"]["energy_pj"],
               p["evaluation"]["metrics"]["latency_ns"]) for p in plans]
    # Plot in normalized coordinates before multiplying by pixel dimensions.
    # Valid finite costs can otherwise overflow during padding or `522 * x`,
    # and small costs should not disappear behind an absolute axis floor.
    scale_x = max(x for x, _ in points) or 1.0
    scale_y = max(y for _, y in points) or 1.0
    normalized = [(x / scale_x, y / scale_y) for x, y in points]
    xs, ys = zip(*normalized)
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
    dx, dy = max(xhi - xlo, .05), max(yhi - ylo, .05)
    xlo -= dx * .08
    xhi += dx * .08
    ylo -= dy * .08
    yhi += dy * .08
    items = ['<svg viewBox="0 0 620 300" role="img" aria-label="Energy versus latency for returned designs">',
             '<path d="M72 20 V246 H594" fill="none" stroke="#9eabc0"/>']
    for i in range(5):
        x = xlo + (xhi - xlo) * i / 4
        y = ylo + (yhi - ylo) * i / 4
        sx = 72 + 522 * i / 4
        sy = 246 - 226 * i / 4
        # Decimal labels also cover padded ticks just beyond the float range;
        # the actual recorded metrics remain untouched.
        label_x = Decimal(str(x)) * Decimal(str(scale_x))
        label_y = Decimal(str(y)) * Decimal(str(scale_y))
        items += [f'<text x="{sx}" y="266" text-anchor="middle">{label_x:.3g}</text>',
                  f'<text x="64" y="{sy+4}" text-anchor="end">{label_y:.3g}</text>',
                  f'<path d="M72 {sy} H594" stroke="#e6ebf3"/>']
    for i, ((x, y), (nx, ny)) in enumerate(zip(points, normalized)):
        sx = 72 + 522 * ((nx - xlo) / (xhi - xlo))
        sy = 246 - 226 * ((ny - ylo) / (yhi - ylo))
        items += [f'<a href="#p{i}"><circle cx="{sx}" cy="{sy}" r="6" fill="#176c93" stroke="white">',
                  f'<title>Design {i+1}: {x:.6g} pJ; {y:.6g} ns</title></circle></a>']
    items += ['<text x="335" y="293" text-anchor="middle">Energy per inference (pJ) →</text>',
              '<text x="19" y="140" transform="rotate(-90 19 140)" text-anchor="middle">Latency (ns) →</text></svg>',
              '<p class="muted">Two-dimensional projection. Designs may also trade area or numerical error; points need not form a monotone curve.</p>']
    return "".join(items)


def _network_svg(network: dict) -> str:
    nodes = network["nodes"]
    depths = {}
    for n in nodes:
        depths[n["id"]] = 0 if not n["inputs"] else 1 + max(depths[x] for x in n["inputs"])
    groups = {}
    for n in nodes:
        groups.setdefault(depths[n["id"]], []).append(n)
    positions = {}
    width = max(620, 180 * max(len(g) for g in groups.values()))
    height = 108 * (max(depths.values()) + 1)
    for depth, group in groups.items():
        for i, n in enumerate(group):
            positions[n["id"]] = (width * (i + 1) / (len(group) + 1), 46 + depth * 108)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Input neural computation graph">',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#a7b5c9"/></marker></defs>']
    for n in nodes:
        x, y = positions[n["id"]]
        for src in n["inputs"]:
            sx, sy = positions[src]
            parts.append(f'<path d="M{sx} {sy+26} L{x} {y-26}" stroke="#a7b5c9" marker-end="url(#arrow)"/>')
    for n in nodes:
        x, y = positions[n["id"]]
        color = "#e6f4ee" if n["id"] in network["outputs"] else "#eef3fa"
        parts += [f'<rect x="{x-78}" y="{y-26}" width="156" height="52" rx="9" fill="{color}" stroke="#c4d2e4"/>',
                  f'<text x="{x}" y="{y-3}" text-anchor="middle" font-weight="650">{_escape(n["id"])}</text>',
                  f'<text x="{x}" y="{y+15}" text-anchor="middle">{_escape(n["op"])} · {n["size"]} values</text>']
    return "".join(parts) + "</svg>"


def html_text(report: dict, network: dict, checks: dict | None = None,
              simulations: dict | None = None) -> str:
    checks, simulations = checks or {}, simulations or {}
    search = report.get("search", {})
    plans = report.get("plans", [])
    complete = search.get("complete", False)
    qualifier = "Complete within the declared candidate family" if complete else "Approximate set from an incomplete search"
    title = f"Neural Physical Planner — {network['name']}"
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{_escape(title)}</title>',
             '''<style>
body{margin:0;background:#f5f7fb;color:#182438;font:16px/1.5 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:34px 22px 70px}h1{font-size:32px;letter-spacing:-1px;margin:8px 0}h2{font-size:21px;margin:0 0 16px}h3{font-size:18px}p{max-width:90ch}.eyebrow{color:#176c93;text-transform:uppercase;font-size:12px;font-weight:750;letter-spacing:1.5px}.muted{color:#5f6c80;font-size:13px}.banner{background:#fff4dc;border-left:4px solid #dfa94e;padding:16px 20px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.stat,.panel{background:white;border:1px solid #e1e7ef;border-radius:12px;padding:20px}.stat strong{display:block;font-size:27px}.panel{margin-top:22px}.badge{display:inline-block;border-radius:20px;padding:4px 11px;background:#e6f4ee;color:#156747;font-size:12px;font-weight:700}.bad{background:#fce8e8;color:#922c2c}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid #e6ebf2}th{color:#617088;font-weight:600}a{color:#126d99}pre{overflow:auto;max-height:480px;background:#f4f7fb;padding:16px;border-radius:8px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}details{border-top:1px solid #e2e8f0;padding:10px 0}summary{cursor:pointer;font-weight:600}svg{width:100%;height:auto;max-height:530px}svg text{font-family:system-ui,sans-serif;font-size:12px;fill:#47576c}.scroll{overflow-x:auto}.empty{padding:30px;background:#f8f9fc}@media(max-width:650px){main{padding:20px 12px}h1{font-size:26px}.panel{padding:14px}}
</style></head><body><main>''',
             '<div class="eyebrow">NPP · reproducible planning report · v0.1</div>',
             f'<h1>{_escape(network["name"])}</h1><p>{_escape(qualifier)}.</p>',
             '<div class="banner"><strong>Research model, not fabrication signoff.</strong> The bundled hardware library contains illustrative synthetic parameters. Numerical bounds are conditional on the normalized additive Gaussian model; they are not measurements or guaranteed physical performance.</div>',
             '<div class="grid">']
    for value, label in [(report.get("status", "unknown"), "Search status"),
                         (len(plans), "Nondominated designs retained"),
                         (search.get("evaluated", 0), "Candidates evaluated"),
                         (search.get("total_candidates", "unknown"), "Structural candidate space")]:
        parts.append(f'<div class="stat"><strong>{_escape(value)}</strong>{_escape(label)}</div>')
    parts += ['</div><section class="panel"><h2>Design trade-offs</h2>', _scatter(plans),
              '<div class="scroll"><table><thead><tr><th>Design</th>']
    parts += [f'<th>{_escape(LABELS[m])}</th>' for m in METRICS]
    parts += ['<th>Replay check</th></tr></thead><tbody>']
    for i, p in enumerate(plans):
        check = checks.get(p["id"], {})
        parts.append(f'<tr><td><a href="#p{i}">{i+1}</a></td>')
        parts += [f'<td>{_number(p["evaluation"]["metrics"].get(m))}</td>' for m in METRICS]
        parts.append(f'<td>{"Passed" if check.get("valid") else "Failed" if check else "Not run"}</td></tr>')
    parts += ['</tbody></table></div></section><section class="panel"><h2>Neural computation</h2>', _network_svg(network), '</section>']
    for i, p in enumerate(plans):
        ev = p["evaluation"]
        parts += [f'<section class="panel" id="p{i}"><h2>Design {i+1}</h2>',
                  f'<p class="muted">{_escape(p["id"])}</p>',
                  f'<span class="badge">{_escape(ev["analysis"].get("kind", "unknown"))}</span>']
        for heading, value in [("Selected implementations", p["decision"]),
                               ("Applied rules and trace", ev.get("trace", [])),
                               ("Hardware macro graph and schedule", ev.get("hardware", {})),
                               ("Noise analysis and assumptions", ev.get("analysis", {})),
                               ("Replay checker", checks.get(p["id"], {"status": "not run"})),
                               ("Monte Carlo validation", simulations.get(p["id"], {"status": "not run"}))]:
            if heading == "Replay checker" and isinstance(value, dict):
                value = {k: v for k, v in value.items() if k != "recomputed_evaluation"}
            parts.append(f'<details><summary>{heading}</summary><pre>{_json(value)}</pre></details>')
        parts.append('</section>')
    parts += ['<section class="panel"><h2>Search and rejection diagnostics</h2>',
              f'<pre>{_json({"search":search,"rejection_summary":report.get("rejection_summary",[]),"input_hashes":report.get("input_hashes",{})})}</pre></section>',
              '<p class="muted">Generated locally. This report has no network dependencies or embedded tracking. The JSON files and source code are the authoritative reproducible artifacts.</p></main></body></html>']
    return "".join(parts)


def save_reports(directory: Path, report: dict, network: dict,
                 checks: dict | None = None, simulations: dict | None = None):
    (directory / "pareto.csv").write_text(csv_text(report), encoding="utf-8")
    (directory / "report.html").write_text(html_text(report, network, checks, simulations), encoding="utf-8")
