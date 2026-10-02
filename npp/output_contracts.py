"""Structural output schemas; replay checking establishes semantic consistency."""
from __future__ import annotations

from copy import deepcopy


def _object(properties: dict, required: list | None = None, extra: bool = False) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": extra}


def output_schema_bundle() -> dict[str, dict]:
    nonnegative = {"type": "number", "minimum": 0}
    string = {"type": "string"}
    hashes = _object({k: {"type": "string", "pattern": "^[0-9a-f]{64}$"}
                      for k in ("network", "library", "request")})
    diagnostic = _object({"code": string, "path": string, "message": string}, ["code", "message"], True)
    array_of_objects = {"type": "array", "items": {"type": "object"}}
    decision = _object({key: {"type": "object", "additionalProperties": string}
                        for key in ("components", "fanout_rules")})
    metrics = _object({key: nonnegative for key in (
        "energy_pj", "latency_ns", "area_um2", "error_rms_bound")})
    analysis = _object({
        "kind": {"enum": ["exact_affine_gaussian", "conservative_lipschitz_rms"]},
        "nominal_bounds": {"type": "object"},
        "assumptions": {"type": "array", "items": string},
        "output_covariance": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "error_bound_method": string,
    }, ["kind", "nominal_bounds", "assumptions", "error_bound_method"])
    hardware = _object({
        "representation": {"const": "abstract_physical_implementation_plan_v1"},
        "device_netlist": {"const": False},
        "instances": array_of_objects,
        "connections": {"type": "array", "items": _object({
            "source": string, "target": string, "role": string}, extra=True)},
        "schedule": {"type": "array", "items": _object({
            "instance_id": string, "operation": string, "start_ns": nonnegative,
            "end_ns": nonnegative}, extra=True)},
    })
    evaluation = _object({
        "feasible": {"const": True}, "metrics": metrics,
        "diagnostics": {"type": "array", "items": diagnostic, "maxItems": 0},
        "analysis": analysis, "hardware": hardware,
        "trace": array_of_objects, "noise_sources": array_of_objects,
    })
    record = _object({
        "schema_version": {"const": "0.1"},
        "id": {"type": "string", "pattern": "^plan-[0-9a-f]{64}$"},
        "input_hashes": hashes, "decision": decision, "evaluation": evaluation,
    })
    report = _object({
        "schema_version": {"const": "0.1"},
        "status": {"enum": ["ok", "infeasible", "search_exhausted"]},
        "input_hashes": hashes,
        "search": _object({
            "mode": {"enum": ["exhaustive", "beam"]},
            "total_candidates": {"type": "integer", "minimum": 0},
            "evaluated": {"type": "integer", "minimum": 0},
            "feasible": {"type": "integer", "minimum": 0},
            "rejected": {"type": "integer", "minimum": 0},
            "complete": {"type": "boolean"}, "algorithm": string,
        }, extra=True),
        "plans": {"type": "array", "items": deepcopy(record)},
        "rejection_summary": array_of_objects,
        "diagnostics": {"type": "array", "items": diagnostic},
        "caveats": {"type": "array", "items": string},
    })
    result = {"plan": record, "report": report}
    for name, schema in result.items():
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = f"Neural Physical Planner v0.1 {name}"
        schema["description"] = (
            "Structural contract for exported feasible plans and reports. Detailed instance/trace records "
            "are model-specific; npp check additionally recomputes their full semantic content. "
            "Hashes identify normalized inputs and decisions, not authorship or fabrication validity.")
    return result
