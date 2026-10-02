"""Validate an exported plan without trusting its cached evaluation.

This is a replay checker, not an independent theorem prover or a silicon
verification tool. It shares the physical model with the planner but never
reruns the search and recomputes all evaluation fields from the input snapshots.
"""

from __future__ import annotations

import math
from typing import Any

from .models import InputValidationError, validate_inputs
from .physics import evaluate
from .planner import input_hash


def _differences(expected: Any, actual: Any, path: str = "evaluation") -> list[dict]:
    result: list[dict] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            at = f"{path}.{key}"
            if key not in expected or key not in actual:
                result.append({"code": "CHECK_FIELD_MISMATCH", "path": at,
                               "message": "Missing or unexpected evaluation field."})
            else:
                result.extend(_differences(expected[key], actual[key], at))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            result.append({"code": "CHECK_LENGTH_MISMATCH", "path": path,
                           "message": "Array length differs from recomputation."})
        else:
            for i, (a, b) in enumerate(zip(expected, actual)):
                result.extend(_differences(a, b, f"{path}[{i}]"))
    elif (isinstance(expected, (int, float)) and not isinstance(expected, bool)
          and isinstance(actual, (int, float)) and not isinstance(actual, bool)):
        try:
            matches = math.isfinite(float(actual)) and math.isclose(
                float(expected), float(actual), rel_tol=1e-9, abs_tol=0.0)
        except (OverflowError, ValueError):
            matches = False
        if not matches:
            result.append({"code": "CHECK_VALUE_MISMATCH", "path": path,
                           "message": f"Stored {actual!r}; recomputed {expected!r}."})
    elif type(expected) is not type(actual) or expected != actual:
        result.append({"code": "CHECK_VALUE_MISMATCH", "path": path,
                       "message": "Value differs from recomputation."})
    return result


def check_record(network: dict, library: dict, request: dict, record: dict) -> dict:
    """Recheck a standalone plan against normalized, explicitly supplied inputs."""
    diagnostics: list[dict] = []
    try:
        network, library, request = validate_inputs(network, library, request)
    except InputValidationError as exc:
        return {"valid": False, "diagnostics": exc.diagnostics,
                "recomputed_evaluation": None}
    if not isinstance(record, dict):
        return {"valid": False, "diagnostics": [{"code": "CHECK_RECORD_TYPE",
                "path": "record", "message": "Plan must be a JSON object."}],
                "recomputed_evaluation": None}
    expected_hashes = {"network": input_hash(network), "library": input_hash(library),
                       "request": input_hash(request)}
    if record.get("schema_version") != "0.1":
        diagnostics.append({"code": "CHECK_VERSION", "path": "schema_version",
                            "message": "Expected plan schema version 0.1."})
    if record.get("input_hashes") != expected_hashes:
        diagnostics.append({"code": "CHECK_INPUT_HASH", "path": "input_hashes",
                            "message": "Plan inputs differ from the supplied normalized snapshots."})
    decision = record.get("decision")
    if not isinstance(decision, dict):
        diagnostics.append({"code": "CHECK_DECISION", "path": "decision",
                            "message": "Missing decision object."})
        return {"valid": False, "diagnostics": diagnostics,
                "recomputed_evaluation": None}
    try:
        decision_digest = input_hash(decision)
        expected_id = "plan-" + decision_digest
        if record.get("id") != expected_id:
            diagnostics.append({"code": "CHECK_PLAN_ID", "path": "id",
                                "message": "Plan ID does not match its decision."})
        recomputed = evaluate(network, library, request, decision)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        diagnostics.append({"code": "CHECK_EVALUATION_FAILED", "path": "decision",
                            "message": str(exc)})
        return {"valid": False, "diagnostics": diagnostics,
                "recomputed_evaluation": None}
    if not recomputed.get("feasible", False):
        diagnostics.append({"code": "CHECK_INFEASIBLE", "path": "evaluation.feasible",
                            "message": "This decision is not feasible under the supplied request."})
    diagnostics.extend(_differences(recomputed, record.get("evaluation")))
    return {"valid": not diagnostics, "diagnostics": diagnostics[:100],
            "recomputed_evaluation": recomputed,
            "scope": "Replay under the declared model; neither model calibration nor fabrication signoff is certified."}
