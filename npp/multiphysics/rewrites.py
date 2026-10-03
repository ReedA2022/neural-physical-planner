"""Reviewed exact algebraic rewrites, separate from physical realizations.

These records justify the ideal function only. The graph builder must still pay
for gathers, distribution, partial sums, ordered merges and any re-encoding.
No generic basis change, truncated residual, or sign-through-SiLU rule exists.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .semantic import FFNWorkload, MAX_DIMENSION, SemanticError


def _indices(values, size, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise SemanticError(f"{name}: expected a nonempty index list")
    if any(type(v) is not int or not 0 <= v < size for v in values):
        raise SemanticError(f"{name}: integer indices must be in [0,{size})")
    if len(set(values)) != len(values):
        raise SemanticError(f"{name}: duplicate indices")
    return sorted(values)


def _track(track, quality_budget):
    if track not in ("E", "A"):
        raise SemanticError("reviewed rewrites support E or budgeted A; adaptation requires a new model workflow")
    if quality_budget is not None:
        if isinstance(quality_budget, bool) or not isinstance(quality_budget, (int, float)):
            raise SemanticError("quality_budget must be a finite nonnegative absolute error")
        try:
            valid = math.isfinite(quality_budget) and quality_budget >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise SemanticError("quality_budget must be a finite nonnegative absolute error")
    if track == "A" and quality_budget is None:
        raise SemanticError("Track A requires an explicit quality_budget")


def normalize_partition(shape, tiles, *, semantic_track="E", quality_budget=None):
    """Validate a complete disjoint rectangular tiling of one matrix.

    Each tile may select arbitrary ordered row/column subsets; canonicalization
    sorts those indices. Coverage is checked per matrix coefficient, so input,
    output and two-dimensional partitions share one correctness rule. Even Track
    A cannot silently omit coefficients: approximate operators need another
    explicit reviewed rule and its quality validation.
    """
    _track(semantic_track, quality_budget)
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(d) is not int or not 1 <= d <= MAX_DIMENSION for d in shape):
        raise SemanticError(f"shape must contain output/input dimensions in [1,{MAX_DIMENSION}]")
    rows, columns = shape
    if not isinstance(tiles, list) or not 1 <= len(tiles) <= 256:
        raise SemanticError("tiles must be a list of 1..256 explicit rectangles")
    covered, ids, normalized = set(), set(), []
    for position, item in enumerate(tiles):
        required = {"rows", "columns", "family", "component", "region"}
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required - {"id"}:
            raise SemanticError("each tile requires rows, columns, family, component, region and optional id; no tile owns bias")
        tile = dict(item)
        tile.setdefault("id", f"tile_{position}")
        for name in ("id", "component", "region"):
            value = tile[name]
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,159}", value):
                raise SemanticError(f"tile {name}: expected a bounded identifier")
        if tile["id"] in ids:
            raise SemanticError("duplicate tile identifier")
        ids.add(tile["id"])
        families = {"digital": "digital", "analog": "analog_electrical",
                    "analog_electrical": "analog_electrical", "photonic": "photonic",
                    "classical_photonic": "photonic"}
        if not isinstance(tile["family"], str) or tile["family"] not in families:
            raise SemanticError("tile family must be digital, analog_electrical, or photonic")
        tile["family"] = families[tile["family"]]
        tile["rows"] = _indices(tile["rows"], rows, "tile.rows")
        tile["columns"] = _indices(tile["columns"], columns, "tile.columns")
        cells = {(r, c) for r in tile["rows"] for c in tile["columns"]}
        if covered & cells:
            raise SemanticError("partition overlaps matrix coefficients; each coefficient must appear exactly once")
        covered.update(cells)
        normalized.append(tile)
    if len(covered) != rows * columns:
        raise SemanticError("partition omits coefficients/residual; incomplete coverage is not an exact rewrite")
    normalized.sort(key=lambda item: item["id"])
    reductions = [{"row": r, "tiles": [item["id"] for item in normalized if r in item["rows"]]} for r in range(rows)]
    return {
        "schema": "mp.rewrite.partition.v1", "semantic_track": semantic_track,
        "quality_budget": quality_budget, "shape": [rows, columns], "tiles": normalized,
        "output_reductions": reductions,
        "merge": {"row_order": list(range(rows)), "bias_owner": "merge", "bias_application": "once_after_reduction"},
        "proof": {
            "rule": "complete_disjoint_matrix_partition.v1",
            "source": "y = W @ x + b",
            "destination": "y[r] = sum(tile_W[r, columns] @ x[columns] for contributing tiles) + b[r]",
            "preconditions": ["every original matrix coefficient occurs exactly once", "input gather preserves selected index order", "partial sums are reduced for their original output row", "bias is added exactly once after reduction"],
            "rationale": "distributivity of finite real sums over a disjoint complete partition",
            "numerical_oracle": "independent scalar sums in tests/test_mp_rewrites.py",
            "runtime_obligations": ["pay each input gather/distribution", "pay partial-sum reads and reductions", "pay ordered output materialization and bias addition", "evaluate changed finite-precision and physical error"],
            "claim_boundary": "real-arithmetic equivalence; neither bitwise equality nor free movement is implied",
        },
    }


def transform_ffn(model: FFNWorkload | dict[str, Any], permutation, signs):
    """Absorb a hidden permutation and diagonal ±1 scaling into model weights.

    Signs multiply the up branch and down-projection columns, never SiLU's gate.
    Biases transform with their own branch; the output bias is unchanged.
    """
    original = FFNWorkload.from_dict(model) if isinstance(model, dict) else model
    if not isinstance(original, FFNWorkload):
        raise SemanticError("model must be an explicit FFN workload")
    if original.semantic_track == "R":
        raise SemanticError("Track R hardware/adaptation workflow is not an exact representation rewrite")
    h = original.hidden_size
    if not isinstance(permutation, (list, tuple)) or len(permutation) != h or any(type(v) is not int for v in permutation) or sorted(permutation) != list(range(h)):
        raise SemanticError("permutation must name each hidden channel exactly once")
    if not isinstance(signs, (list, tuple)) or len(signs) != h or any(type(v) not in (int, float) or v not in (-1, 1) for v in signs):
        raise SemanticError("signs must contain one explicit -1 or +1 per hidden channel")
    result = original.to_dict()
    result["W_gate"] = [list(original.W_gate[k]) for k in permutation]
    result["b_gate"] = [original.b_gate[k] for k in permutation]
    result["W_up"] = [[float(s) * v for v in original.W_up[k]] for k, s in zip(permutation, signs)]
    result["b_up"] = [float(s) * original.b_up[k] for k, s in zip(permutation, signs)]
    result["W_down"] = [[float(s) * row[k] for k, s in zip(permutation, signs)] for row in original.W_down]
    rewritten = FFNWorkload.from_dict(result)
    return {
        "model": rewritten.to_dict(),
        "proof": {
            "rule": "swiglu_hidden_signed_permutation.v1", "semantic_track": original.semantic_track,
            "source_model_identity": original.model_identity, "destination_model_identity": rewritten.model_identity,
            "permutation": list(permutation), "signs": [int(s) for s in signs],
            "source": "W_d @ (SiLU(W_g @ x + b_g) * (W_u @ x + b_u)) + b_d",
            "destination": "W_g'=P W_g; b_g'=P b_g; W_u'=D P W_u; b_u'=D P b_u; W_d'=W_d P.T D^-1; b_d'=b_d",
            "preconditions": ["P is a complete permutation", "D is diagonal with entries +/-1", "signs are applied only to the up branch and inverse down columns", "gate/output biases follow their declared transformations"],
            "rationale": "SiLU commutes with permutation; pointwise multiplication transports the up-branch diagonal sign, canceled by D^-1 in down projection",
            "numerical_oracle": "independent Decimal SwiGLU oracle in tests/test_mp_rewrites.py",
            "runtime_transformation": "none when absorbed into offline weights; original and transformed weights must each pay their own programming/configuration cost",
            "runtime_obligations": ["physical encoding must support every transformed sign/range", "placement and lane routing must follow the permutation", "finite precision and physical error must be reevaluated"],
            "claim_boundary": "ideal equivalent function, not task adaptation or unchanged physical error",
        },
    }
