"""Seeded physical-contract stress checks with independent scalar oracles.

This checks software/model consistency for illustrative optical fanout, not device
calibration, fabrication, or NN accuracy. Optional SAX checks call the real external
engine and are reported separately. One scenario is one generated graph, malformed
input, or component bundle; individual assertions are not counted as scenarios.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npp.implementation import (build_implementation, check_realization,
                                evaluate_implementation, realize,
                                validate_implementation)
from npp.models import InputValidationError
from npp.technology import evaluate_component, load_technology, photon_count, validate_technology


def network(fanout=2, lanes=1):
    return {"name": "Seeded physical stress", "nodes": [
        {"id": "x", "op": "input", "size": lanes,
         "input_bounds": {"lower": [0.0] * lanes, "upper": [1.0] * lanes}}
    ], "outputs": ["x"] * fanout}


def spec(**changes):
    return {"source_node": "x", "source_power_mw": 1.0,
            "lengths_um": 0.0, **changes}


def _close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-17):
        raise AssertionError(f"{label}: actual={actual!r}, independent={expected!r}")


def _reject(call):
    try:
        call()
    except InputValidationError as exc:
        assert exc.diagnostics and all(d.get("code") and d.get("message") for d in exc.diagnostics)
        json.dumps(exc.diagnostics, allow_nan=False)
        return
    raise AssertionError("malformed input unexpectedly accepted")


def _oracle_case(rng, index, *, maximum=False):
    fanout = rng.choice([2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64])
    lanes = rng.choice([1, 2, 3])
    if maximum:
        fanout, lanes = [(64, 4), (4, 64), (63, 4), (3, 64)][index % 4]
    recipe = "regenerate" if index % 2 else "passive"
    pack = load_technology()
    c = {item["kind"]: item for item in pack["components"]}
    for component in c.values():
        component["costs"].update(area_um2=rng.uniform(0, 100),
                                  fixed_energy_pj=rng.uniform(0, 1),
                                  latency_ns=rng.uniform(0, 0.2))
        if component["kind"] != "source":
            component["operating_envelope"]["max_input_power_mw"] = 10.0
    p = {kind: component["parameters"] for kind, component in c.items()}
    p["source"]["wall_plug_efficiency"] = rng.uniform(0.01, 1)
    p["source"]["relative_intensity_noise_rms"] = rng.uniform(0, 0.02)
    p["splitter"]["insertion_loss_db"] = rng.uniform(0, 2)
    p["waveguide"].update(attenuation_db_per_um=rng.uniform(0, .001),
                            group_index=rng.uniform(1, 12), width_um=rng.uniform(.01, 2))
    p["detector"].update(minimum_full_scale_power_mw=1e-8,
                          quantum_efficiency=rng.uniform(.1, 1),
                          input_referred_noise_mw_rms=rng.uniform(0, 1e-5))
    p["modulator"].update(insertion_loss_db=rng.uniform(0, 4),
                           added_sample_noise_rms=rng.uniform(0, .01))
    duration = rng.uniform(.5, 10)
    wavelength = rng.uniform(1540, 1560)
    lengths = [rng.uniform(0, 10000) for _ in range(fanout)]
    power = rng.uniform(.1, 2)
    carrier = rng.uniform(.1, 1)
    request = spec(recipe=recipe, source_power_mw=power, lengths_um=lengths,
                   operating={"wavelength_nm": wavelength, "symbol_duration_ns": duration,
                              "temperature_c": rng.uniform(20, 30)})
    if recipe == "regenerate":
        request["regeneration_power_mw"] = carrier
    record = realize(network(fanout, lanes), pack, request)
    result = record["evaluation"]
    assert result["feasible"], result["diagnostics"]
    assert check_realization(json.loads(json.dumps(record, allow_nan=False)))["valid"]
    assert record["technology"]["calibration_status"] == "illustrative"
    depth = (fanout - 1).bit_length()
    leaves = 2 ** depth
    tree_power = power / leaves * math.exp(-math.log(10) * depth * p["splitter"]["insertion_loss_db"] / 10)
    costs = {kind: component["costs"] for kind, component in c.items()}
    counts = {"source": lanes, "splitter": lanes * (leaves - 1),
              "waveguide": lanes * fanout, "detector": lanes * fanout}
    if recipe == "regenerate":
        counts.update(source=lanes * (fanout + 1), detector=2 * lanes * fanout,
                      modulator=lanes * fanout)
    assert result["instance_counts"] == counts
    energy = sum(costs[k]["fixed_energy_pj"] * n for k, n in counts.items())
    energy += lanes * (power + (fanout * carrier if recipe == "regenerate" else 0)) * duration / p["source"]["wall_plug_efficiency"]
    area = sum(costs[k]["area_um2"] * n for k, n in counts.items()) + lanes * sum(lengths) * p["waveguide"]["width_um"]
    base_latency = duration + costs["source"]["latency_ns"] + depth * costs["splitter"]["latency_ns"]
    if recipe == "regenerate":
        base_latency = max(base_latency + costs["detector"]["latency_ns"], costs["source"]["latency_ns"]) + duration + costs["modulator"]["latency_ns"]
    base_latency += costs["waveguide"]["latency_ns"] + costs["detector"]["latency_ns"]
    expected_power = [(carrier * math.exp(-math.log(10) * p["modulator"]["insertion_loss_db"] / 10)
                       if recipe == "regenerate" else tree_power)
                      * math.exp(-math.log(10) * p["waveguide"]["attenuation_db_per_um"] * length / 10)
                      for length in lengths]
    def detection_variance(power_mw):
        photons = (power_mw * 1e-3) * (duration * 1e-9) / (6.62607015e-34 * 299792458 / (wavelength * 1e-9))
        return 1 / (p["detector"]["quantum_efficiency"] * photons) + (p["detector"]["input_referred_noise_mw_rms"] / power_mw) ** 2
    rin = p["source"]["relative_intensity_noise_rms"] ** 2
    independent = [detection_variance(value) for value in expected_power]
    if recipe == "regenerate":
        independent = [v + detection_variance(tree_power) + rin + p["modulator"]["added_sample_noise_rms"] ** 2 for v in independent]
    variance = [rin + v for v in independent]
    covariance = np.zeros((fanout * lanes, fanout * lanes))
    for lane in range(lanes):
        start = lane * fanout
        covariance[start:start + fanout, start:start + fanout] = rin
        covariance[range(start, start + fanout), range(start, start + fanout)] += independent
    np.testing.assert_allclose(result["receiver_covariance_estimate"], covariance, rtol=3e-12, atol=1e-18)
    if fanout * lanes <= 64:
        assert np.linalg.eigvalsh(result["receiver_covariance_estimate"])[0] >= -1e-14
    for lane in range(lanes):
        for use, length in enumerate(lengths):
            row = result["receivers"][lane * fanout + use]
            assert row["lane"] == lane and row["use_index"] == use
            _close(row["full_scale_power_mw"], expected_power[use], "receiver power")
            _close(row["complete_symbol_ready_ns"], base_latency + p["waveguide"]["group_index"] * length * 1e3 / 299792458, "receiver timing")
    metrics = result["metrics"]
    for name, expected in {"energy_pj": energy, "area_um2": area,
                           "latency_ns": base_latency + p["waveguide"]["group_index"] * max(lengths) * 1e3 / 299792458,
                           "noise_rms_estimate": math.sqrt(max(variance)),
                           "stacked_noise_rms_estimate": math.sqrt(lanes * sum(variance)),
                           "terminated_power_mw": lanes * (leaves - fanout) * tree_power,
                           "receiver_count": lanes * fanout, "instance_count": sum(counts.values())}.items():
        _close(metrics[name], expected, name)
    # Independent lane sources cannot correlate distinct lanes.
    if lanes > 1:
        assert np.count_nonzero(np.asarray(result["receiver_covariance_estimate"])[:fanout, fanout:]) == 0


def _guard_case(rng, index):
    pack = load_technology()
    requests = [spec(source_power_mw=11.0), spec(source_power_mw=.001),
                spec(lengths_um=10001), spec(operating={"wavelength_nm": 1539.9}),
                spec(operating={"temperature_c": 30.0001}), spec(operating={"symbol_duration_ns": .4999}),
                spec(source_power_mw=.01), spec(source_power_mw=10),
                spec(recipe="regenerate", regeneration_power_mw=.01),
                spec(recipe="regenerate", regeneration_power_mw=10),
                spec(recipe="regenerate", source_power_mw=.01, regeneration_power_mw=1)]
    result = realize(network(rng.choice([2, 3, 4, 8])), pack, requests[index % len(requests)])["evaluation"]
    assert not result["feasible"] and result["diagnostics"]
    assert result["metrics"] is None and result["receiver_covariance_estimate"] is None
    json.dumps(result, allow_nan=False)


def _malformed_case(rng, index):
    pack = load_technology()
    original = realize(network(3, 2), pack, spec(recipe="regenerate", regeneration_power_mw=.5))
    choice = index % 16
    graph = deepcopy(original["graph"])
    if choice == 0:
        graph["connections"].pop()
    elif choice == 1:
        graph["connections"][1]["source"] = deepcopy(graph["connections"][0]["source"])
    elif choice == 2:
        graph["connections"][1]["target"] = deepcopy(graph["connections"][0]["target"])
    elif choice == 3:
        graph["instances"].append(deepcopy(graph["instances"][0]))
    elif choice == 4:
        graph["receivers"][0]["lane"] = True
    elif choice == 5:
        graph["receivers"][0]["use_index"] = 3
    elif choice == 6:
        graph["receivers"][0]["port"] = deepcopy(graph["receivers"][1]["port"])
    elif choice == 7:
        graph["launch_boundaries"].pop()
    elif choice == 8:
        graph["terminations"].pop()
    elif choice == 9:
        graph["instances"][0]["component"] = "missing"
    elif choice == 10:
        graph["recipe"] = "passive"
    elif choice == 11:
        graph["operating"]["temperature_c"] = -273.16
    elif choice == 12:
        graph["connections"][0]["source"]["port"] = "unknown"
    elif choice == 13:
        graph["instances"][0]["role"] = "constant_carrier"
    elif choice == 14:
        graph["instances"][0]["parameters"]["power_mw"] = float("inf")
    else:
        graph["instances"][0]["parameters"]["power_mw"] = False
    _reject(lambda: validate_implementation(graph, pack))
    record = deepcopy(original)
    mutations = [("metrics", "energy_pj"), ("metrics", "area_um2"), ("metrics", "latency_ns")]
    first, second = rng.choice(mutations)
    record["evaluation"][first][second] *= 1.0001
    assert not check_realization(record)["valid"]
    record = deepcopy(original)
    record["hashes"]["graph"] = "0" * 64
    assert not check_realization(record)["valid"]


def _extreme_case(rng, index):
    pack = load_technology()
    choice = index % 8
    if choice < 3:
        field = ["area_um2", "fixed_energy_pj", "latency_ns"][choice]
        for component in pack["components"]:
            component["costs"][field] = 1e308
    elif choice == 3:
        pack["components"][0]["parameters"]["relative_intensity_noise_rms"] = 1e308
    elif choice == 4:
        next(c for c in pack["components"] if c["kind"] == "waveguide")["parameters"]["attenuation_db_per_um"] = 1e308
    elif choice == 5:
        next(c for c in pack["components"] if c["kind"] == "waveguide")["parameters"]["group_index"] = 1e308
    elif choice == 6:
        next(c for c in pack["components"] if c["kind"] == "detector")["parameters"]["quantum_efficiency"] = 5e-324
    else:
        pack["components"][0]["parameters"]["wall_plug_efficiency"] = 5e-324
    validate_technology(pack)
    result = realize(network(3), pack, spec(lengths_um=1000))["evaluation"]
    assert not result["feasible"] and result["metrics"] is None
    assert result["diagnostics"]
    json.dumps(result, allow_nan=False)


def _component_case(rng, index):
    pack = load_technology()
    component = next(c for c in pack["components"] if c["kind"] == "splitter")
    ratio = rng.uniform(.0001, .9999)
    loss = rng.uniform(0, 100)
    power = rng.uniform(0, 10)
    component["parameters"].update(power_ratio=ratio, insertion_loss_db=loss)
    outputs = evaluate_component(component, input_power_mw=power)["output_powers_mw"]
    total = power * math.exp(-loss * math.log(10) / 10)
    _close(outputs["out1"], total * ratio, "asymmetric splitter branch one")
    _close(outputs["out2"], total * (1-ratio), "asymmetric splitter branch two")
    assert math.fsum(outputs.values()) <= power * (1+1e-14)
    # Component model supports asymmetric splitting; the balanced graph recipe does not.
    if ratio != .5:
        _reject(lambda: build_implementation(network(), pack, spec()))
    guide = next(c for c in pack["components"] if c["kind"] == "waveguide")
    a, b = rng.uniform(0, 4000), rng.uniform(0, 4000)
    first = evaluate_component(guide, input_power_mw=power, length_um=a)
    second = evaluate_component(guide, input_power_mw=first["output_powers_mw"]["out"], length_um=b)
    together = evaluate_component(guide, input_power_mw=power, length_um=a+b)
    _close(second["output_powers_mw"]["out"], together["output_powers_mw"]["out"], "guide cascade")


def _underflow_case(rng, index):
    pack = load_technology()
    components = {c["kind"]: c for c in pack["components"]}
    choice = index % 9
    if choice < 3:
        kind, field = [("source", "relative_intensity_noise_rms"),
                       ("modulator", "added_sample_noise_rms"),
                       ("detector", "input_referred_noise_mw_rms")][choice]
        components[kind]["parameters"][field] = 10 ** rng.uniform(-290, -200)
        request = spec(recipe="regenerate", regeneration_power_mw=.5)
        result = realize(network(), pack, request)["evaluation"]
        assert not result["feasible"] and result["metrics"] is None
        assert any("underflow" in d["message"] for d in result["diagnostics"])
        json.dumps(result, allow_nan=False)
        return
    if choice == 3:
        component = components["source"]
        component["parameters"].update(minimum_output_power_mw=1e-250,
                                        maximum_output_power_mw=1e-150,
                                        wall_plug_efficiency=1e-200)
        component["operating_envelope"]["symbol_duration_ns"] = {"minimum": 1e-250, "maximum": 1e-150}
        call = lambda: evaluate_component(component, source_power_mw=1e-200, symbol_duration_ns=1e-200)
    elif choice < 7:
        component = components["waveguide"]
        component["parameters"][["width_um", "group_index", "attenuation_db_per_um"][choice-4]] = 1e-200
        call = lambda: evaluate_component(component, input_power_mw=1, length_um=1e-200)
    elif choice == 7:
        component = components["splitter"]
        component["parameters"]["insertion_loss_db"] = 4000
        call = lambda: evaluate_component(component, input_power_mw=1)
    else:
        call = lambda: photon_count(1e-200, 1e-200, 1550)
    try:
        call()
    except ValueError as exc:
        assert "underflow" in str(exc)
    else:
        raise AssertionError("positive physical contribution silently underflowed to zero")


def _boundary_case(rng, index):
    pack = load_technology()
    choice = index % 10
    if choice < 4:
        fanout, lanes = [(1, 1), (65, 1), (2, 65), (64, 5)][choice]
        _reject(lambda: realize(network(fanout, lanes), pack, spec()))
    elif choice == 4:
        _reject(lambda: realize(network(3), pack, spec(lengths_um=[0, 0])))
    else:
        names = ["wavelength_nm", "temperature_c", "symbol_duration_ns"]
        name = names[(choice-5) % 3]
        edge = "minimum" if index % 2 else "maximum"
        boundary = pack["components"][0]["operating_envelope"][name][edge]
        good = realize(network(), pack, spec(operating={name: boundary}))["evaluation"]
        assert good["feasible"], good["diagnostics"]
        outside = math.nextafter(boundary, -math.inf if edge == "minimum" else math.inf)
        bad = realize(network(), pack, spec(operating={name: outside}))["evaluation"]
        assert not bad["feasible"] and bad["metrics"] is None


def run_campaign(seed=730021, cases=384):
    """Return a JSON-safe report; failures retain exact deterministic replay indices."""
    if not __debug__:
        raise ValueError("Stress campaigns require assertions; run Python without -O or -OO.")
    if isinstance(cases, bool) or not isinstance(cases, int) or not 1 <= cases <= 2000:
        raise ValueError("cases must be an integer from 1 through 2000")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    started = time.monotonic()
    families = Counter()
    failures = []
    generators = [("scalar_oracle", _oracle_case)] * 6 + [("positive_underflow", _underflow_case),
        ("infeasible_envelope", _guard_case), ("malformed_graph_and_replay", _malformed_case),
        ("extreme_finite", _extreme_case), ("component_conservation", _component_case),
        ("size_and_envelope_boundaries", _boundary_case)]
    for index in range(cases):
        family, generate = generators[index % len(generators)]
        rng = random.Random(f"{seed}:{index}")
        families[family] += 1
        try:
            # Explicit four accepted high-size shapes are spread through the first
            # forty scenarios; most random checks use at most three lanes.
            if family == "scalar_oracle":
                generate(rng, index, maximum=index in (0, 13, 26, 39))
            else:
                generate(rng, index // len(generators))
        except Exception as exc:
            failures.append({"index": index, "family": family,
                             "exception": type(exc).__name__, "message": str(exc)[:2000],
                             "reproduction": f"python scripts/stress_physical.py --seed {seed} --cases {index + 1}"})
    return {"campaign": "physical_contracts", "seed": seed,
            "scenario_count": cases, "family_counts": dict(families),
            "passed_count": cases - len(failures), "failures": failures,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "scope": "software consistency against independent scalar/covariance oracles; illustrative device parameters, no calibration or whole-NN inference validation",
            "external_validation": {"executed": False, "reason": "separate optional run_sax_checks required"}}


def run_sax_checks():
    """Small actual external-engine sample, excluded from software scenario counts."""
    from npp.optical_simulation import simulate_realization
    started = time.monotonic()
    checks = []
    for fanout, lanes, recipe in [(3, 2, "passive"), (5, 1, "regenerate"),
                                   (17, 1, "passive"), (2, 2, "regenerate")]:
        request = spec(recipe=recipe, source_power_mw=1.0,
                       lengths_um=[i * 321.25 for i in range(fanout)])
        if recipe == "regenerate":
            request["regeneration_power_mw"] = .2
        result = simulate_realization(realize(network(fanout, lanes), load_technology(), request),
                                      [1540., 1550., 1560.])
        checks.append({"fanout": fanout, "lanes": lanes, "recipe": recipe,
                       "passed": result["passed"], "engine": result["engine"],
                       **result["summary"]})
    return {"executed": True, "circuits": checks,
            "comparison_count": sum(c["comparison_count"] for c in checks),
            "passed": all(c["passed"] for c in checks),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "scope": "actual SAX full-scale optical power composition only; no independent noise, timing, or device calibration"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=730021)
    parser.add_argument("--cases", type=int, default=384)
    parser.add_argument("--sax", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run_campaign(args.seed, args.cases)
    if args.sax:
        result["external_validation"] = run_sax_checks()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 1 if result["failures"] or result["external_validation"].get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
