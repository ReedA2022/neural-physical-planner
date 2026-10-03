#!/usr/bin/env python3
"""Seeded adversarial checks for the v0.1 macro model (not device validation).

The affine oracle injects each independent disturbance into a separate direct
neural interpreter. Resource/schedule and tiny-space Pareto oracles never call
the production evaluator. Failed cases retain full normalized inputs and seed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npp.checker import check_record
from npp.models import validate_inputs
from npp.physics import evaluate, infer_bounds, simulate
from npp.planner import input_hash, pareto_filter, plan


METRICS = ("energy_pj", "latency_ns", "area_um2", "error_rms_bound")


def _component(identifier, ops, **overrides):
    result = dict(id=identifier, ops=ops, domain="digital", editable=True,
                  area_um2=0.0, energy_pj=0.0, latency_ns=0.0, noise_std=0.0,
                  max_abs_value=1e10, capacity=16)
    result.update(overrides)
    return result


def _rule(identifier="copy", kind="digital_copy", **overrides):
    result = dict(id=identifier, kind=kind,
                  domains=["digital" if kind == "digital_copy" else "optical"])
    result.update(overrides)
    return result


def _library(components, rules=None):
    return dict(schema_version="0.1", name="Seeded synthetic stress library",
                model="normalized_additive_gaussian_v1",
                provenance="Synthetic software oracle fixtures; no device calibration.",
                components=components, rules=rules or [_rule()], conversions=[
                    dict(id="d2o", from_domain="digital", to_domain="optical",
                         area_um2=0.0, energy_pj=0.0, latency_ns=0.0, noise_std=0.0),
                    dict(id="o2d", from_domain="optical", to_domain="digital",
                         area_um2=0.0, energy_pj=0.0, latency_ns=0.0, noise_std=0.0)])


def _request(**overrides):
    result = dict(name="Seeded stress request", objectives=list(METRICS),
                  search=dict(mode="exhaustive", max_evaluations=1000, beam_width=8))
    result.update(overrides)
    return result


def _network(nodes, outputs):
    return dict(name="Seeded stress graph", nodes=nodes, outputs=outputs)


def _uses(net):
    counts = Counter(net["outputs"])
    for node in net["nodes"]:
        counts.update(node["inputs"])
    return counts


def _forward(net, x, disturbances=None, ignore_bias=False):
    """Direct neural interpretation, independent of physics._apply/_prepare."""
    disturbances = disturbances or {}
    values = {}
    for node in net["nodes"]:
        name, op = node["id"], node["op"]
        if op == "input":
            value = np.array(x, dtype=float, copy=True)
        elif op == "linear":
            value = np.asarray(node["weights"]) @ values[node["inputs"][0]]
            if not ignore_bias:
                value = value + node["bias"]
        elif op == "add":
            value = sum((values[source] for source in node["inputs"]),
                        np.zeros(node["size"]))
        elif op == "relu":
            value = np.clip(values[node["inputs"][0]], 0, None)
        else:
            value = values[node["inputs"][0]].copy()
        values[name] = value + disturbances.get(name, 0.0)
    return values


def _covariance_oracle(net, lib, decision):
    components = {c["id"]: c for c in lib["components"]}
    rules = {r["id"]: r for r in lib["rules"]}
    dim = sum(next(n["size"] for n in net["nodes"] if n["id"] == name)
              for name in net["outputs"])
    expected = np.zeros((dim, dim))
    for node in net["nodes"]:
        name = node["id"]
        noises = [components[decision["components"][name]]["noise_std"]]
        if name in decision["fanout_rules"]:
            noises.append(rules[decision["fanout_rules"][name]]["added_noise_std"])
        for std in noises:
            for coordinate in range(node["size"]):
                impulse = np.zeros(node["size"])
                impulse[coordinate] = std
                out = _forward(net, np.zeros(net["nodes"][0]["size"]),
                               {name: impulse}, ignore_bias=True)
                column = np.concatenate([out[name] for name in net["outputs"]])
                expected += np.outer(column, column)
    return expected


def _digital_cost_oracle(net, lib, decision):
    components = {c["id"]: c for c in lib["components"]}
    rules = {r["id"]: r for r in lib["rules"]}
    energy = area = 0.0
    ready = {}
    for node in net["nodes"]:
        name = node["id"]
        c = components[decision["components"][name]]
        energy += c["energy_pj"]
        area += c["area_um2"]
        if node["op"] == "linear":
            energy += c["energy_per_mac_pj"] * sum(map(len, node["weights"]))
        ready[name] = max((ready[p] for p in node["inputs"]), default=0.0) + c["latency_ns"]
        if name in decision["fanout_rules"]:
            r = rules[decision["fanout_rules"][name]]
            ready[name] += r["latency_ns"]
            energy += r["energy_pj"]
            area += r["area_um2"]
    return dict(energy_pj=energy, area_um2=area,
                latency_ns=max(ready[name] for name in net["outputs"]))


def _dag_payload(rng, nonlinear=False):
    width = int(rng.integers(1, 5))
    nodes = [dict(id="x", op="input", size=width,
                  input_bounds=dict(lower=[-1.0] * width, upper=[1.0] * width))]
    for index in range(int(rng.integers(2, 8))):
        source = nodes[int(rng.integers(len(nodes)))]
        op = str(rng.choice(["linear", "add", "identity"]))
        if nonlinear and index == 0:
            op = "relu"
        node = dict(id=f"n{index}", op=op, inputs=[source["id"]], size=source["size"])
        if op == "linear":
            node["size"] = int(rng.integers(1, 5))
            node["weights"] = (rng.integers(-4, 5, (node["size"], source["size"])) / 4).tolist()
            node["bias"] = (rng.integers(-4, 5, node["size"]) / 4).tolist()
        elif op == "add":
            options = [n["id"] for n in nodes if n["size"] == source["size"]]
            node["inputs"] += [str(rng.choice(options)) for _ in range(int(rng.integers(1, 4)))]
        nodes.append(node)
    referenced = {source for n in nodes for source in n.get("inputs", [])}
    outputs = [n["id"] for n in nodes if n["id"] not in referenced]
    if rng.random() < 0.7:
        outputs += [str(rng.choice([n["id"] for n in nodes]))]
    if rng.random() < 0.5:
        outputs += [outputs[-1]]
    components = [_component(f"c{i}", [n["op"]],
                    noise_std=float(rng.choice([0.0, .01, .02, .04])),
                    energy_pj=float(rng.integers(1, 5)),
                    area_um2=float(rng.integers(1, 5)),
                    latency_ns=float(rng.integers(1, 5)), energy_per_mac_pj=.125)
                  for i, n in enumerate(nodes)]
    net, lib, req = validate_inputs(_network(nodes, outputs),
        _library(components, [_rule(added_noise_std=.003, energy_pj=.25,
                                   area_um2=.5, latency_ns=.125)]), _request())
    uses = _uses(net)
    decision = dict(components={n["id"]: f"c{i}" for i, n in enumerate(nodes)},
                    fanout_rules={name: "copy" for name, count in uses.items() if count > 1})
    return dict(network=net, library=lib, request=req, decision=decision)


def _check_dag(payload, nonlinear=False, sample_seed=0):
    net, lib, req, decision = (payload[k] for k in ("network", "library", "request", "decision"))
    result = evaluate(net, lib, req, decision)
    assert result["feasible"], result["diagnostics"]
    for key, value in _digital_cost_oracle(net, lib, decision).items():
        np.testing.assert_allclose(result["metrics"][key], value, rtol=1e-12, atol=0)
    bounds = infer_bounds(net)
    for corner in product([-1.0, 1.0], repeat=net["nodes"][0]["size"]):
        values = _forward(net, corner)
        for name, value in values.items():
            assert np.all(value >= np.array(bounds[name]["lower"]) - 1e-12)
            assert np.all(value <= np.array(bounds[name]["upper"]) + 1e-12)
    if not nonlinear:
        expected = _covariance_oracle(net, lib, decision)
        actual = np.array(result["analysis"]["output_covariance"])
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-17)
        assert np.linalg.eigvalsh(actual).min() >= -1e-12
        np.testing.assert_allclose(result["metrics"]["error_rms_bound"],
                                   math.sqrt(np.trace(expected)), rtol=3e-12, atol=1e-17)
    if nonlinear or sample_seed % 8 == 0:
        sampled = simulate(net, lib, req, decision, samples=4000, seed=sample_seed)
        error = sampled["mse_empirical"] - result["metrics"]["error_rms_bound"] ** 2
        # Seven SE is a deliberately conservative stochastic campaign gate;
        # the production simulator separately reports its own four-SE check.
        limit = 7 * sampled["mse_standard_error"] + 1e-14
        assert error <= limit if nonlinear else abs(error) <= limit, sampled


def _search_payload(rng):
    count = int(rng.integers(1, 4))
    nodes = [dict(id="x", op="input", size=1,
                  input_bounds=dict(lower=[-1.], upper=[1.]))]
    nodes += [dict(id=f"n{i}", op="linear", size=1,
                   inputs=["x" if i == 0 else f"n{i-1}"],
                   weights=[[float(rng.choice([-2, -1, 0, 1, 2]))]], bias=[.5])
              for i in range(count)]
    components = [_component("source", ["input"], energy_pj=1., latency_ns=1., area_um2=1.)]
    components += [_component(f"option{i}", ["linear"],
                    energy_pj=float(rng.integers(1, 10)),
                    area_um2=float(rng.integers(1, 10)),
                    latency_ns=float(rng.integers(1, 10))) for i in range(3)]
    objectives = [m for m in METRICS if rng.random() < .6] or ["energy_pj"]
    budgets = {"energy_pj": float(rng.integers(0, count * 10 + 2))} if rng.random() < .6 else {}
    net, lib, req = validate_inputs(_network(nodes, [nodes[-1]["id"]]),
                                   _library(components), _request(objectives=objectives, budgets=budgets))
    return dict(network=net, library=lib, request=req)


def _allpairs(records, objectives):
    def values(r):
        return [r["evaluation"]["metrics"][k] for k in objectives]
    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0)
    def dominates(a, b):
        return all(x <= y for x, y in zip(values(a), values(b))) and any(
            x < y and not close(x, y) for x, y in zip(values(a), values(b)))
    # Exact synthetic values avoid nontransitive approximate-equality chains.
    survivors = [r for r in records if not any(dominates(s, r) for s in records)]
    unique = []
    for r in sorted(survivors, key=lambda r: input_hash(r["decision"])):
        if not any(values(r) == values(s) for s in unique):
            unique.append(r)
    return {input_hash(r["decision"]) for r in unique}


def _check_search(payload):
    net, lib, req = (payload[k] for k in ("network", "library", "request"))
    records = []
    for options in product(["option0", "option1", "option2"], repeat=len(net["nodes"]) - 1):
        decision = dict(components=dict(zip([n["id"] for n in net["nodes"]], ["source", *options])),
                        fanout_rules={})
        metrics = _digital_cost_oracle(net, lib, decision)
        metrics["error_rms_bound"] = 0.0
        feasible = all(value is None or metrics[key] <= value for key, value in req["budgets"].items())
        if feasible:
            records.append(dict(decision=decision, evaluation=dict(metrics=metrics, feasible=True)))
    report = plan(net, lib, req)
    assert report["search"]["complete"]
    assert report["search"]["evaluated"] == 3 ** (len(net["nodes"]) - 1)
    assert report["search"]["feasible"] == len(records)
    assert {input_hash(p["decision"]) for p in report["plans"]} == _allpairs(records, req["objectives"])
    for record in report["plans"]:
        assert check_record(net, lib, req, record)["valid"]
    for mode in ("exhaustive", "beam"):
        bounded = deepcopy(req)
        bounded["budgets"]["energy_pj"] = 0.0
        bounded["search"].update(mode=mode, max_evaluations=1, beam_width=1)
        empty = plan(net, lib, bounded)
        assert empty["status"] == "search_exhausted", empty
        assert not empty["search"]["complete"]
        assert empty["search"]["evaluated"] == 1


def _optical_payload(rng):
    branches = int(rng.integers(2, 7))
    kind = str(rng.choice(["passive_split", "source_boost", "amplify", "regenerate", "serial_reencode"]))
    gain = float(rng.choice([1, 2, 4])) if kind in ("source_boost", "amplify", "regenerate") else 1.0
    photons, efficiency = float(rng.integers(50, 500)), float(rng.choice([.25, .5, 1.]))
    net, lib, req = validate_inputs(
        _network([dict(id="x", op="input", size=1, input_bounds=dict(lower=[-.5], upper=[.5]))],
                 ["x"] * branches),
        _library([_component("source", ["input"], domain="optical", source_scalable=True,
                             photons=photons, optical_efficiency=efficiency,
                             noise_std=.03, energy_pj=.5, latency_ns=.25)],
                 [_rule("r", kind, gain=gain, added_noise_std=.02, energy_pj=.1, latency_ns=.125)]),
        _request(allow_serialization=True))
    return dict(network=net, library=lib, request=req,
                decision=dict(components={"x": "source"}, fanout_rules={"x": "r"}))


def _check_optical(payload):
    net, lib, req, decision = (payload[k] for k in ("network", "library", "request", "decision"))
    c, r = lib["components"][0], lib["rules"][0]
    n, kind, gain = len(net["outputs"]), r["kind"], r["gain"]
    shot = 1 / (c["photons"] * c["optical_efficiency"])
    common = c["noise_std"] ** 2 + r["added_noise_std"] ** 2
    if kind == "serial_reencode":
        expected = np.eye(n) * (shot + c["noise_std"] ** 2 + r["added_noise_std"] ** 2)
        launch, reps = n, n
    elif kind == "regenerate":
        expected = np.full((n, n), common + shot) + np.eye(n) * (shot / gain + r["added_noise_std"] ** 2)
        launch, reps = 1 + n * gain, 1
    else:
        expected = np.full((n, n), common) + np.eye(n) * (n * shot / gain)
        launch, reps = gain, 1
    result = evaluate(net, lib, req, decision)
    assert result["feasible"], result["diagnostics"]
    np.testing.assert_allclose(result["analysis"]["output_covariance"], expected, rtol=1e-12, atol=1e-16)
    photon_energy = c["photons"] * 6.62607015e-34 * 299792458 / (c["wavelength_nm"] * 1e-9) / c["wall_plug_efficiency"] * 1e12
    np.testing.assert_allclose(result["metrics"]["energy_pj"],
                              reps * (c["energy_pj"] + r["energy_pj"]) + launch * photon_energy, rtol=1e-12)
    np.testing.assert_allclose(result["metrics"]["latency_ns"], reps * (c["latency_ns"] + r["latency_ns"]), rtol=1e-12)


def _check_tamper(payload, variant):
    net, lib, req, decision = (payload[k] for k in ("network", "library", "request", "decision"))
    record = dict(schema_version="0.1", id="plan-" + input_hash(decision), decision=decision,
                  input_hashes={"network": input_hash(net), "library": input_hash(lib), "request": input_hash(req)},
                  evaluation=evaluate(net, lib, req, decision))
    assert check_record(net, lib, req, record)["valid"]
    if variant % 5 == 0:
        record["evaluation"]["metrics"]["energy_pj"] += 1
    elif variant % 5 == 1:
        record["evaluation"]["feasible"] = 1
    elif variant % 5 == 2:
        record["evaluation"]["hardware"]["schedule"][0]["end_ns"] += 1
    elif variant % 5 == 3:
        record["evaluation"]["analysis"]["output_covariance"][0][0] += .5
    else:
        record["input_hashes"]["network"] = "0" * 64
    assert not check_record(net, lib, req, record)["valid"]


def _precision_payload(variant):
    exponent = [-13, -100, -150, -161, -170, -200, -300][variant % 7]
    net, lib, req = validate_inputs(
        _network([dict(id="x", op="input", size=1, input_bounds=dict(lower=[0.], upper=[0.]))], ["x"]),
        _library([_component("source", ["input"], noise_std=10.0 ** exponent)]),
        _request(budgets={"error_rms_bound": 0.0}))
    return dict(network=net, library=lib, request=req,
                decision=dict(components={"x": "source"}, fanout_rules={}))


def _check_precision(payload):
    result = evaluate(*(payload[k] for k in ("network", "library", "request", "decision")))
    assert not result["feasible"], "Positive source noise passed an exact-zero error budget"
    if result["metrics"]:
        assert result["metrics"]["error_rms_bound"] > 0


def _nonlinear_precision_payload(variant):
    # Covers both representable RMS with unrepresentable squared RMS, and
    # genuinely unrepresentable downstream scalar RMS multiplication.
    weight = [1e-100, 1e-160, 1e-300][variant % 3]
    net, lib, req = validate_inputs(_network([
        dict(id="x", op="input", size=1, input_bounds=dict(lower=[0.], upper=[0.])),
        dict(id="y", op="linear", size=1, inputs=["x"], weights=[[weight]]),
        dict(id="z", op="relu", size=1, inputs=["y"])], ["z"]),
        _library([_component("source", ["input"], noise_std=1e-100),
                  _component("op", ["linear", "relu"])]), _request(budgets={"error_rms_bound": 0.}))
    return dict(network=net, library=lib, request=req,
                decision=dict(components={"x": "source", "y": "op", "z": "op"}, fanout_rules={}))


def _check_near_tie(_):
    records = [dict(id=name, evaluation=dict(feasible=True, metrics=dict(energy_pj=e, latency_ns=t)))
               for name, e, t in [("a", 1., 3.), ("b", 1. + 1.5e-9, 1.), ("c", 1. + .75e-9, 2.)]]
    result = pareto_filter(records, ["energy_pj", "latency_ns"])
    assert {r["id"] for r in result} == {"a", "b", "c"}, "A tiny worsening was silently forgiven"


def run_campaign(seed=20261003, cases=320):
    """Run exactly ``cases`` recorded scenarios; no external calibration claims."""
    if not __debug__:
        raise ValueError("Stress campaigns require assertions; run Python without -O or -OO.")
    if isinstance(cases, bool) or not isinstance(cases, int) or cases < 1:
        raise ValueError("cases must be a positive integer")
    rng = np.random.default_rng(seed)
    started, counts, failures = time.perf_counter(), Counter(), []
    families = ["affine_dag"] * 6 + ["nonlinear_dag", "tiny_search", "tiny_search",
                "optical_rule", "optical_rule", "replay_tamper", "precision", "near_tie_frontier",
                "nonlinear_precision"]
    for index in range(cases):
        family = families[index % len(families)]
        variant = index // len(families)
        counts[family] += 1
        payload = {}
        try:
            if family in ("affine_dag", "nonlinear_dag", "replay_tamper"):
                payload = _dag_payload(rng, nonlinear=family == "nonlinear_dag")
                if family == "replay_tamper":
                    _check_tamper(payload, variant)
                else:
                    _check_dag(payload, nonlinear=family == "nonlinear_dag", sample_seed=seed + index)
            elif family == "tiny_search":
                payload = _search_payload(rng)
                _check_search(payload)
            elif family == "optical_rule":
                payload = _optical_payload(rng)
                _check_optical(payload)
            elif family in ("precision", "nonlinear_precision"):
                payload = (_precision_payload if family == "precision" else _nonlinear_precision_payload)(variant)
                _check_precision(payload)
            else:
                payload = dict(records=[["a", 1., 3.], ["b", 1. + 1.5e-9, 1.], ["c", 1. + .75e-9, 2.]])
                _check_near_tie(payload)
        except Exception as exc:
            failures.append(dict(case=index, family=family, seed=seed, variant=variant,
                                 exception=type(exc).__name__, message=str(exc), inputs=payload))
    return dict(schema_version="1", campaign="legacy_macro_model", seed=seed,
                requested_cases=cases, scenario_count=sum(counts.values()),
                actual_scenario_count=sum(counts.values()),
                family_counts=dict(counts), passed=sum(counts.values()) - len(failures),
                failed=len(failures), failures=failures, elapsed_seconds=time.perf_counter() - started,
                scope="Seeded synthetic software/model checks; no device calibration or arbitrary-space optimality claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--cases", type=int, default=320)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run_campaign(args.seed, args.cases)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}, indent=2))
    sys.exit(bool(result["failed"]))
