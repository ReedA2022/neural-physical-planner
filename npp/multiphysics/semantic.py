"""Bounded, carrier-independent SwiGLU semantics and digital arithmetic.

Matrices use output-by-input orientation. A rank-two input is a batch of row
vectors; it is never guessed from weight names or checkpoint architecture.
The ideal function is real arithmetic, while ``ideal_output`` is an explicitly
labeled binary64 *numerical reference* to that function, not exact real algebra.
Hardware timing, physical resources and evidence belong to other modules.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np


MAX_DIMENSION = 256
MAX_BATCH = 64
MAX_WEIGHTS = 262_144
MAX_MACS = 10_000_000


class SemanticError(ValueError):
    """Invalid workload or unsupported semantic contract."""


class NumericalFailure(SemanticError):
    """Numerical range failure, not a proof of physical infeasibility."""


def _array(value: Any, name: str, ranks: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise SemanticError(f"{name}: expected a rectangular numeric array") from exc
    if raw.ndim not in ranks or any(d < 1 or d > MAX_DIMENSION for d in raw.shape):
        raise SemanticError(f"{name}: rank must be {ranks} and axes must be 1..{MAX_DIMENSION}")
    for v in raw.flat:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
            raise SemanticError(f"{name}: entries must be real numbers, not booleans or strings")
    try:
        result = raw.astype(np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise NumericalFailure(f"{name}: entries exceed binary64 range") from exc
    if not np.isfinite(result).all():
        raise NumericalFailure(f"{name}: entries must be finite binary64 numbers")
    result[result == 0] = 0.0
    return result


def _tuple_matrix(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in _array(value, name, (2,)))


def _tuple_vector(value: Any, name: str, width: int) -> tuple[float, ...]:
    if value is None:
        return (0.0,) * width
    array = _array(value, name, (1,))
    if array.shape != (width,):
        raise SemanticError(f"{name}: expected exactly {width} values")
    return tuple(float(x) for x in array)


@dataclass(frozen=True)
class NumericPolicy:
    """IEEE round-to-nearest arithmetic with explicit non-fused accumulation.

    Inputs/weights and each product round to ``format``; every accumulator add
    rounds to ``accumulation``; bias addition uses that accumulator; the final
    result rounds to ``format``. SiLU uses stable binary64 exp and rounds once
    to ``format``. Subnormals and rounding to zero are permitted and reported as
    numerical approximation; overflow/nonfinite results raise NumericalFailure.
    """

    format: str = "float64"
    accumulation: str = "float64"
    rounding: str = "nearest_even"
    overflow: str = "reject"
    underflow: str = "gradual"

    def __post_init__(self) -> None:
        if self.format not in ("float32", "float64"):
            raise SemanticError("numeric format must be float32 or float64")
        if self.accumulation not in ("float32", "float64"):
            raise SemanticError("accumulation must be float32 or float64")
        if self.format == "float64" and self.accumulation != "float64":
            raise SemanticError("float64 operands require float64 accumulation")
        if (self.rounding, self.overflow, self.underflow) != ("nearest_even", "reject", "gradual"):
            raise SemanticError("supported numeric policy is nearest_even/reject/gradual")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NumericPolicy":
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise SemanticError("numeric policy contains unknown fields")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise SemanticError(str(exc)) from exc

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class FFNWorkload:
    """Immutable explicit model; missing biases normalize to zero vectors.

    Track A requires a nonnegative absolute output-error budget. Track R requires
    a distinct adaptation identity, its parent identity, and an explicit cost
    record (unknown costs must be represented as such by the evidence layer).
    This reference evaluates supplied weights; it never performs adaptation.
    """

    W_up: tuple[tuple[float, ...], ...]
    W_gate: tuple[tuple[float, ...], ...]
    W_down: tuple[tuple[float, ...], ...]
    b_up: tuple[float, ...] | None = None
    b_gate: tuple[float, ...] | None = None
    b_down: tuple[float, ...] | None = None
    name: str = "explicit_swiglu"
    semantic_track: str = "E"
    quality_budget: float | None = None
    parent_model_identity: str | None = None
    adaptation_identity: str | None = None
    adaptation_cost: str | None = None

    def __post_init__(self) -> None:
        for name in ("W_up", "W_gate", "W_down"):
            object.__setattr__(self, name, _tuple_matrix(getattr(self, name), name))
        h, n, m = len(self.W_up), len(self.W_up[0]), len(self.W_down)
        if len(self.W_gate) != h or len(self.W_gate[0]) != n or len(self.W_down[0]) != h:
            raise SemanticError("W_up/W_gate must be hidden x input; W_down must be output x hidden")
        if 2 * h * n + m * h > MAX_WEIGHTS:
            raise SemanticError(f"workload exceeds {MAX_WEIGHTS} matrix parameters")
        for name, width in (("b_up", h), ("b_gate", h), ("b_down", m)):
            object.__setattr__(self, name, _tuple_vector(getattr(self, name), name, width))
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 256:
            raise SemanticError("name must be a nonempty string of at most 256 characters")
        if not isinstance(self.semantic_track, str) or self.semantic_track not in ("E", "A", "R"):
            raise SemanticError("semantic_track must be E, A, or R")
        if self.quality_budget is not None:
            q = self.quality_budget
            if isinstance(q, (bool, np.bool_)) or not isinstance(q, Real):
                raise SemanticError("quality_budget must be a finite nonnegative absolute error")
            try:
                q = float(q)
            except (ValueError, OverflowError) as exc:
                raise SemanticError("quality_budget exceeds binary64 range") from exc
            if not math.isfinite(q) or q < 0:
                raise SemanticError("quality_budget must be a finite nonnegative absolute error")
            object.__setattr__(self, "quality_budget", q)
        if self.semantic_track == "A" and self.quality_budget is None:
            raise SemanticError("Track A requires an explicit quality_budget")
        adaptation = (self.parent_model_identity, self.adaptation_identity, self.adaptation_cost)
        if self.semantic_track == "R":
            if any(not isinstance(v, str) or not v.strip() for v in adaptation):
                raise SemanticError("Track R requires parent_model_identity, adaptation_identity, adaptation_cost")
            if self.parent_model_identity == self.adaptation_identity:
                raise SemanticError("Track R must identify a different adapted model")
        elif any(v is not None for v in adaptation):
            raise SemanticError("adaptation fields require Track R")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FFNWorkload":
        if not isinstance(value, dict):
            raise SemanticError("FFN workload must be an object")
        extra = set(value) - set(cls.__dataclass_fields__)
        missing = {"W_up", "W_gate", "W_down"} - set(value)
        if extra or missing:
            raise SemanticError(f"FFN fields: unknown={sorted(map(str, extra))}, missing={sorted(missing)}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise SemanticError(str(exc)) from exc

    @property
    def input_size(self) -> int:
        return len(self.W_up[0])

    @property
    def hidden_size(self) -> int:
        return len(self.W_up)

    @property
    def output_size(self) -> int:
        return len(self.W_down)

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        for name in ("W_up", "W_gate", "W_down"):
            result[name] = [list(row) for row in result[name]]
        for name in ("b_up", "b_gate", "b_down"):
            result[name] = list(result[name])
        return result

    @property
    def model_identity(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()

    def semantic_ir(self) -> dict[str, Any]:
        """Carrier-independent graph; bias is owned by its linear node once."""
        return {
            "schema": "mp.semantic.ffn.v1", "model_identity": self.model_identity,
            "semantic_track": self.semantic_track,
            "input_shape": [self.input_size], "output_shape": [self.output_size],
            "nodes": [
                {"id": "up", "op": "linear", "inputs": ["x"], "weight": "W_up", "bias": "b_up"},
                {"id": "gate_linear", "op": "linear", "inputs": ["x"], "weight": "W_gate", "bias": "b_gate"},
                {"id": "gate", "op": "silu", "inputs": ["gate_linear"]},
                {"id": "hidden", "op": "multiply", "inputs": ["gate", "up"]},
                {"id": "output", "op": "linear", "inputs": ["hidden"], "weight": "W_down", "bias": "b_down"},
            ],
            "ideal_function": "W_down @ (silu(W_gate @ x + b_gate) * (W_up @ x + b_up)) + b_down",
        }


def _policy(policy: NumericPolicy | None) -> NumericPolicy:
    if policy is None:
        return NumericPolicy()
    if not isinstance(policy, NumericPolicy):
        raise SemanticError("policy must be a NumericPolicy")
    return policy


def _cast(array: np.ndarray, dtype: Any, name: str) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            result = array.astype(dtype)
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalFailure(f"{name}: numeric overflow during format conversion") from exc
    if not np.isfinite(result).all():
        raise NumericalFailure(f"{name}: nonfinite result")
    return result


def linear(matrix: Any, inputs: Any, bias: Any = None, policy: NumericPolicy | None = None) -> np.ndarray:
    """Compute W @ x + b, or batched x @ W.T + b, without fused MACs."""
    p = _policy(policy)
    dtype, accumulator = getattr(np, p.format), getattr(np, p.accumulation)
    w = _array(matrix, "matrix", (2,))
    x = _array(inputs, "inputs", (1, 2))
    b = np.asarray(_tuple_vector(bias, "bias", len(w)))
    if x.shape[-1] != w.shape[1]:
        raise SemanticError("linear: input width differs from matrix width")
    batch = 1 if x.ndim == 1 else x.shape[0]
    if batch > MAX_BATCH or batch * w.size > MAX_MACS:
        raise SemanticError("linear: batch or work limit exceeded")
    w, x, b = (_cast(a, dtype, label) for a, label in ((w, "weights"), (x, "inputs"), (b, "bias")))
    rows = x.reshape(batch, -1)
    result = np.empty((batch, len(w)), dtype=dtype)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for i, row in enumerate(rows):
                for j, weights in enumerate(w):
                    total = accumulator(0)
                    for a, value in zip(weights, row):
                        product = dtype(a * value)
                        total = accumulator(total + accumulator(product))
                    result[i, j] = dtype(accumulator(total + accumulator(b[j])))
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalFailure("linear: non-fused arithmetic exceeded numeric range") from exc
    if not np.isfinite(result).all():
        raise NumericalFailure("linear: nonfinite output")
    return result[0] if x.ndim == 1 else result


def _stable_silu(value: float) -> float:
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


def silu(inputs: Any, policy: NumericPolicy | None = None) -> np.ndarray:
    p = _policy(policy)
    dtype = getattr(np, p.format)
    x = _cast(_array(inputs, "silu inputs", (1, 2)), dtype, "silu inputs")
    result = np.asarray([_stable_silu(float(v)) for v in x.flat]).reshape(x.shape)
    return _cast(result, dtype, "silu")


def _elementwise(left: Any, right: Any, policy: NumericPolicy | None, op: str) -> np.ndarray:
    p = _policy(policy)
    dtype = getattr(np, p.format)
    a, b = _array(left, "left", (1, 2)), _array(right, "right", (1, 2))
    if a.shape != b.shape:
        raise SemanticError(f"{op}: shapes must match exactly; implicit broadcasting is unsupported")
    a, b = _cast(a, dtype, "left"), _cast(b, dtype, "right")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            output = a * b if op == "multiply" else a + b
    except FloatingPointError as exc:
        raise NumericalFailure(f"{op}: arithmetic exceeded numeric range") from exc
    return _cast(output, dtype, op)


def multiply(left: Any, right: Any, policy: NumericPolicy | None = None) -> np.ndarray:
    return _elementwise(left, right, policy, "multiply")


def add(left: Any, right: Any, policy: NumericPolicy | None = None) -> np.ndarray:
    return _elementwise(left, right, policy, "add")


def _reference_linear(matrix: Any, inputs: np.ndarray, bias: Any) -> np.ndarray:
    output = []
    try:
        for row in inputs:
            values = []
            for weights, offset in zip(matrix, bias):
                products = [float(a) * float(x) for a, x in zip(weights, row)]
                value = math.fsum([*products, float(offset)])
                if not math.isfinite(value):
                    raise NumericalFailure("binary64 reference: nonfinite linear result")
                values.append(value)
            output.append(values)
    except (ValueError, OverflowError) as exc:
        raise NumericalFailure("binary64 reference: product or sum outside numeric range") from exc
    return np.asarray(output)


def operation_counts(workload: FFNWorkload, batch: int = 1) -> dict[str, int | str]:
    if not isinstance(workload, FFNWorkload):
        raise SemanticError("workload must be an FFNWorkload")
    if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= MAX_BATCH:
        raise SemanticError(f"batch must be an integer in 1..{MAX_BATCH}")
    n, h, m = workload.input_size, workload.hidden_size, workload.output_size
    macs = batch * (2 * n * h + h * m)
    if macs > MAX_MACS:
        raise SemanticError(f"FFN execution exceeds {MAX_MACS} matrix MACs")
    return {
        "batch": batch, "matrix_mac": macs, "bias_add": batch * (2 * h + m),
        "silu": batch * h, "elementwise_multiply": batch * h,
        "weight_elements": 2 * n * h + h * m, "bias_elements": 2 * h + m,
        "input_elements": batch * n, "output_elements": batch * m,
        "intermediate_elements": batch * 4 * h,
        "execution": "matrix_vector" if batch == 1 else "batched_matrix_vector",
        "count_boundary": "logical arithmetic and tensor sizes; no assumed hardware reuse or traffic",
    }


def evaluate_ffn(workload: FFNWorkload | dict[str, Any], inputs: Any,
                 policy: NumericPolicy | None = None) -> dict[str, Any]:
    """Execute both references, retaining E/A/R and finite precision boundaries."""
    w = FFNWorkload.from_dict(workload) if isinstance(workload, dict) else workload
    if not isinstance(w, FFNWorkload):
        raise SemanticError("workload must be an FFNWorkload or workload object")
    p = _policy(policy)
    x = _array(inputs, "inputs", (1, 2))
    if x.shape[-1] != w.input_size:
        raise SemanticError("FFN input width differs from explicit workload")
    single = x.ndim == 1
    rows = x.reshape(1, -1) if single else x
    counts = operation_counts(w, len(rows))
    ideal_up = _reference_linear(w.W_up, rows, w.b_up)
    ideal_gate_linear = _reference_linear(w.W_gate, rows, w.b_gate)
    ideal_gate = np.asarray([_stable_silu(float(v)) for v in ideal_gate_linear.flat]).reshape(ideal_gate_linear.shape)
    ideal_hidden = multiply(ideal_gate, ideal_up)
    ideal = _reference_linear(w.W_down, ideal_hidden, w.b_down)
    up = linear(w.W_up, rows, w.b_up, p)
    gate_linear = linear(w.W_gate, rows, w.b_gate, p)
    gate = silu(gate_linear, p)
    hidden = multiply(gate, up, p)
    output = linear(w.W_down, hidden, w.b_down, p)
    differences = [abs(float(a) - float(b)) for a, b in zip(output.flat, ideal.flat)]
    if not all(math.isfinite(d) for d in differences):
        raise NumericalFailure("output-reference difference exceeds binary64 range")
    max_error = max(differences)
    rms_error = 0.0 if max_error == 0 else max_error * math.sqrt(math.fsum((d / max_error) ** 2 for d in differences) / len(differences))
    if max_error > 0 and rms_error == 0:
        raise NumericalFailure("output-reference RMS error underflows binary64 reporting range")
    def serial(array: np.ndarray) -> Any:
        return (array[0] if single else array).tolist()
    return {
        "schema": "mp.digital.ffn.v1", "status": "evaluated", "model_identity": w.model_identity,
        "semantic_track": w.semantic_track, "numeric_policy": p.to_dict(),
        "ideal_reference": "binary64 products, compensated math.fsum, stable binary64 SiLU; numerical approximation to real ideal",
        "ideal_output": serial(ideal), "output": serial(output),
        "intermediates": {"up": serial(up), "gate_linear": serial(gate_linear), "gate": serial(gate), "hidden": serial(hidden)},
        "absolute_error": {"max": max_error, "rms": rms_error},
        "quality_budget": w.quality_budget,
        "quality_passed": None if w.quality_budget is None else max_error <= w.quality_budget,
        "quality_scope": "numeric execution versus this supplied model only; no task-quality or parent-model equivalence claim",
        "operations": counts,
    }
