"""Small explicit examples with analytically calculable expectations.

These are synthetic test parameters, not a characterized component library.
"""
from copy import deepcopy


def component(identifier, ops, **changes):
    result = {
        "id": identifier, "ops": ops, "domain": "digital", "editable": True,
        "source_scalable": False, "area_um2": 0.0, "latency_ns": 0.0,
        "energy_pj": 0.0, "energy_per_mac_pj": 0.0, "noise_std": 0.0,
        "max_abs_value": 100.0, "photons": 10000.0,
        "optical_efficiency": 1.0, "wall_plug_efficiency": 1.0,
        "wavelength_nm": 1550.0, "capacity": 16, "notes": "Synthetic test macro",
    }
    result.update(changes)
    return result


def rule(identifier="copy", kind="digital_copy", **changes):
    result = {
        "id": identifier, "kind": kind,
        "domains": ["digital" if kind == "digital_copy" else "optical"],
        "split_policy": "equal", "gain": 1.0, "added_noise_std": 0.0,
        "area_um2": 0.0, "latency_ns": 0.0, "energy_pj": 0.0,
        "notes": "Synthetic test rule",
    }
    result.update(changes)
    return result


def library(components, rules=None):
    return {
        "schema_version": "0.1", "name": "analytical_test_library",
        "model": "normalized_additive_gaussian_v1",
        "provenance": "Explicit synthetic parameters used only for regression tests.",
        "components": deepcopy(components), "rules": deepcopy(rules or [rule()]),
        "conversions": [
            {"id": "d2o", "from_domain": "digital", "to_domain": "optical",
             "area_um2": 0.0, "latency_ns": 0.0, "energy_pj": 0.0,
             "noise_std": 0.0, "notes": "Synthetic ideal test conversion"},
            {"id": "o2d", "from_domain": "optical", "to_domain": "digital",
             "area_um2": 0.0, "latency_ns": 0.0, "energy_pj": 0.0,
             "noise_std": 0.0, "notes": "Synthetic ideal test conversion"},
        ],
    }


def request(**changes):
    result = {
        "schema_version": "0.1", "name": "analytical_test_request",
        "objectives": ["energy_pj", "latency_ns", "area_um2", "error_rms_bound"],
        "budgets": {}, "required_editable": [],
        "allowed_domains": ["digital", "optical"], "allow_serialization": False,
        "search": {"mode": "exhaustive", "max_evaluations": 1000, "beam_width": 8},
        "seed": 17,
    }
    result.update(changes)
    return result


def node(identifier, op, inputs=None, **changes):
    result = {"id": identifier, "op": op, "inputs": inputs or [], "size": 1}
    if op == "input":
        result["input_bounds"] = {"lower": [-1.0], "upper": [1.0]}
    if op == "linear":
        result.update(weights=[[1.0]], bias=[0.0])
    result.update(changes)
    return result


def network(nodes, outputs):
    return {"schema_version": "0.1", "name": "analytical_test_network",
            "nodes": deepcopy(nodes), "outputs": outputs}


def affine_fixture():
    net = network([node("x", "input"), node("left", "linear", ["x"]),
                   node("right", "linear", ["x"], weights=[[2.0]])],
                  ["left", "right"])
    lib = library([
        component("encoder", ["input"], noise_std=0.2),
        component("left_unit", ["linear"], noise_std=0.3),
        component("right_unit", ["linear"], noise_std=0.4),
    ])
    decision = {"components": {"x": "encoder", "left": "left_unit", "right": "right_unit"},
                "fanout_rules": {"x": "copy"}}
    return net, lib, request(), decision


def search_fixture():
    net = network([node("x", "input"), node("y", "linear", ["x"], weights=[[2.0]])], ["y"])
    lib = library([
        component("encoder", ["input"]),
        component("fast", ["linear"], energy_pj=9, latency_ns=1, area_um2=4, noise_std=0.05),
        component("small", ["linear"], energy_pj=1, latency_ns=9, area_um2=1, noise_std=0.05),
        component("dominated", ["linear"], energy_pj=10, latency_ns=10, area_um2=5, noise_std=0.05),
    ])
    return net, lib, request()
