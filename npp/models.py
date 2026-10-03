"""Strict, versioned input contracts for the neural physical planner.

Validation establishes structural/model validity, not hardware feasibility.
An otherwise valid request with impossible budgets is handled by the planner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


Version = Literal["0.1"]
Domain = Literal["digital", "optical"]
Operation = Literal["input", "linear", "relu", "add", "identity"]
Objective = Literal["energy_pj", "latency_ns", "area_um2", "error_rms_bound"]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")]
Text = Annotated[str, Field(min_length=1)]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Efficiency = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class InputValidationError(ValueError):
    """A user-readable collection of input diagnostics, safe to serialize."""

    def __init__(self, diagnostics: list[dict[str, str]]):
        self.diagnostics = diagnostics
        super().__init__("; ".join(f"{d['path']}: {d['message']}" for d in diagnostics))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)


class Bounds(StrictModel):
    lower: list[Finite] = Field(min_length=1)
    upper: list[Finite] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper must have equal lengths")
        if any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("each lower bound must be <= its upper bound")
        return self


class Node(StrictModel):
    id: Identifier
    op: Operation
    inputs: list[Identifier] = Field(default_factory=list)
    size: PositiveInt
    weights: list[list[Finite]] | None = None
    bias: list[Finite] | None = None
    input_bounds: Bounds | None = None

    @model_validator(mode="after")
    def operation_contract(self):
        if self.op == "input":
            if self.inputs:
                raise ValueError("input nodes cannot have predecessors")
            if self.input_bounds is None:
                raise ValueError("input nodes require finite input_bounds")
            if len(self.input_bounds.lower) != self.size:
                raise ValueError("input_bounds dimension must equal node size")
        else:
            if self.input_bounds is not None:
                raise ValueError("input_bounds is only valid on input nodes")
            required = 2 if self.op == "add" else 1
            if (self.op == "add" and len(self.inputs) < required) or (
                self.op != "add" and len(self.inputs) != required
            ):
                raise ValueError("add needs at least two inputs; other non-input ops need one")
        if self.op == "linear":
            if self.weights is None or len(self.weights) != self.size:
                raise ValueError("linear weights need one row per output element")
            if not self.weights[0] or any(len(row) != len(self.weights[0]) for row in self.weights):
                raise ValueError("linear weights must be a nonempty rectangular matrix")
            if self.bias is None:
                self.bias = [0.0] * self.size
            elif len(self.bias) != self.size:
                raise ValueError("linear bias dimension must equal output size")
        elif self.weights is not None or self.bias is not None:
            raise ValueError("weights and bias are only valid for linear nodes")
        return self


class Network(StrictModel):
    schema_version: Version = "0.1"
    name: Text
    nodes: list[Node] = Field(min_length=1)
    outputs: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def graph_contract(self):
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be unique")
        if sum(node.op == "input" for node in self.nodes) != 1:
            raise ValueError("v0.1 requires exactly one bounded input node")
        seen: dict[str, Node] = {}
        for node in self.nodes:
            for predecessor in node.inputs:
                if predecessor not in seen:
                    raise ValueError(f"{node.id}: input {predecessor!r} is missing or not earlier in topological order")
            if node.op == "linear":
                if len(node.weights[0]) != seen[node.inputs[0]].size:
                    raise ValueError(f"{node.id}: weight columns must match its input dimension")
            elif node.op in ("identity", "relu", "add"):
                if any(seen[predecessor].size != node.size for predecessor in node.inputs):
                    raise ValueError(f"{node.id}: all input dimensions must equal output size")
            seen[node.id] = node
        for output in self.outputs:
            if output not in seen:
                raise ValueError(f"unknown output node {output!r}")
        reachable: set[str] = set()
        pending = list(self.outputs)
        while pending:
            node_id = pending.pop()
            if node_id not in reachable:
                reachable.add(node_id)
                pending.extend(seen[node_id].inputs)
        unused = sorted(set(seen) - reachable)
        if unused:
            raise ValueError(f"all nodes must reach an output; unused nodes: {unused}")
        return self


class Component(StrictModel):
    id: Identifier
    ops: list[Operation] = Field(min_length=1)
    domain: Domain
    editable: bool = False
    source_scalable: bool = False
    area_um2: Nonnegative
    latency_ns: Nonnegative
    energy_pj: Nonnegative
    energy_per_mac_pj: Nonnegative = 0.0
    noise_std: Nonnegative
    max_abs_value: Positive
    photons: Positive = 1.0
    optical_efficiency: Efficiency = 1.0
    wall_plug_efficiency: Efficiency = 1.0
    wavelength_nm: Positive = 1550.0
    capacity: PositiveInt
    notes: str = ""

    @model_validator(mode="after")
    def capabilities(self):
        if len(set(self.ops)) != len(self.ops):
            raise ValueError("component ops must be unique")
        if "input" in self.ops and self.ops != ["input"]:
            raise ValueError("input encoders must be separate components with ops=['input']")
        if self.source_scalable and not (self.domain == "optical" and self.ops == ["input"]):
            raise ValueError("source_scalable is only supported for optical input encoders")
        if self.domain == "optical" and "relu" in self.ops:
            raise ValueError("v0.1 only implements digital ReLU; optical activation physics is unsupported")
        return self


RuleKind = Literal["digital_copy", "passive_split", "source_boost", "amplify", "regenerate", "serial_reencode"]


class Rule(StrictModel):
    id: Identifier
    kind: RuleKind
    domains: list[Domain] = Field(min_length=1)
    split_policy: Literal["equal", "sensitivity"] = "equal"
    gain: Annotated[float, Field(strict=True, ge=1, allow_inf_nan=False)] = 1.0
    added_noise_std: Nonnegative = 0.0
    area_um2: Nonnegative = 0.0
    latency_ns: Nonnegative = 0.0
    energy_pj: Nonnegative = 0.0
    notes: str = ""

    @model_validator(mode="after")
    def domain_contract(self):
        expected = ["digital"] if self.kind == "digital_copy" else ["optical"]
        if self.domains != expected:
            raise ValueError(f"{self.kind} requires domains={expected!r} in v0.1")
        if self.kind in ("digital_copy", "passive_split", "serial_reencode") and self.gain != 1.0:
            raise ValueError(f"{self.kind} requires gain=1; this rule does not amplify signals")
        return self


class Conversion(StrictModel):
    id: Identifier
    from_domain: Domain
    to_domain: Domain
    area_um2: Nonnegative
    latency_ns: Nonnegative
    energy_pj: Nonnegative
    noise_std: Nonnegative
    notes: str = ""

    @model_validator(mode="after")
    def changes_domain(self):
        if self.from_domain == self.to_domain:
            raise ValueError("conversion must change domain")
        return self


class ComponentLibrary(StrictModel):
    schema_version: Version = "0.1"
    name: Text
    model: Literal["normalized_additive_gaussian_v1"]
    provenance: Text
    components: list[Component] = Field(min_length=1)
    rules: list[Rule] = Field(min_length=1)
    conversions: list[Conversion] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def identifiers_and_conversions(self):
        for field in ("components", "rules", "conversions"):
            ids = [entry.id for entry in getattr(self, field)]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{field}: ids must be unique")
        pairs = {(entry.from_domain, entry.to_domain) for entry in self.conversions}
        if pairs != {("digital", "optical"), ("optical", "digital")}:
            raise ValueError("exactly one conversion in each cross-domain direction is required")
        return self


class Budgets(StrictModel):
    energy_pj: Nonnegative | None = None
    latency_ns: Nonnegative | None = None
    area_um2: Nonnegative | None = None
    error_rms_bound: Nonnegative | None = None


class Search(StrictModel):
    mode: Literal["exhaustive", "beam"] = "beam"
    max_evaluations: PositiveInt = 10000
    beam_width: PositiveInt = 128


class CompileRequest(StrictModel):
    schema_version: Version = "0.1"
    name: Text
    objectives: list[Objective] = Field(default_factory=lambda: ["energy_pj", "latency_ns", "error_rms_bound"], min_length=1)
    budgets: Budgets = Field(default_factory=Budgets)
    required_editable: list[Identifier] = Field(default_factory=list, description="Linear node IDs whose fixed-shape weights and bias must remain editable; other operations are unsupported here.")
    allowed_domains: list[Domain] = Field(default_factory=lambda: ["digital", "optical"], min_length=1)
    allow_serialization: bool = False
    search: Search = Field(default_factory=Search)
    seed: Annotated[int, Field(strict=True, ge=0)] = 0

    @model_validator(mode="after")
    def unique_lists(self):
        for field in ("objectives", "required_editable", "allowed_domains"):
            values = getattr(self, field)
            if len(values) != len(set(values)):
                raise ValueError(f"{field} must not contain duplicates")
        return self


def _location(root: str, loc: tuple[Any, ...]) -> str:
    return root + "".join(f"[{item}]" if isinstance(item, int) else f".{item}" for item in loc)


def validate_inputs(network: dict, library: dict, request: dict) -> tuple[dict, dict, dict]:
    """Validate and normalize inputs; fail with serializable diagnostics.

    The component library is allowed to offer no feasible implementation of a
    particular request: this is a search outcome, not malformed user input.
    """
    validated = []
    diagnostics = []
    for root, model, value in (("network", Network, network), ("library", ComponentLibrary, library), ("request", CompileRequest, request)):
        try:
            validated.append(model.model_validate(value))
        except ValidationError as exc:
            validated.append(None)
            diagnostics.extend({"code": error["type"], "path": _location(root, error["loc"]), "message": error["msg"]}
                               for error in exc.errors(include_url=False, include_input=False))
    if diagnostics:
        raise InputValidationError(diagnostics)
    net, lib, req = validated
    nodes_by_id = {node.id: node for node in net.nodes}
    for index, node_id in enumerate(req.required_editable):
        if node_id not in nodes_by_id:
            diagnostics.append({"code": "unknown_node", "path": f"request.required_editable[{index}]", "message": f"unknown network node {node_id!r}"})
        elif nodes_by_id[node_id].op != "linear":
            diagnostics.append({"code": "unsupported_editability", "path": f"request.required_editable[{index}]", "message": "v0.1 editability applies only to weights and bias of fixed-shape linear nodes"})
    if diagnostics:
        raise InputValidationError(diagnostics)
    return tuple(obj.model_dump(mode="json", exclude_none=True) for obj in (net, lib, req))


def load_json(path: str | Path) -> dict:
    """Read a JSON object, rejecting duplicate keys and nonfinite constants."""
    def unique_object(pairs):
        value = {}
        for key, entry in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = entry
        return value

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON numeric constant {value!r}")

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=unique_object, parse_constant=invalid_constant)
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value must be an object")
        return value
    except (OSError, ValueError, RecursionError) as exc:
        raise InputValidationError([{"code": "json_load_error", "path": str(path), "message": str(exc)}]) from exc


def schema_bundle() -> dict[str, dict]:
    """JSON Schema supplements; cross-object graph checks run in Python."""
    return {name: model.model_json_schema() for name, model in
            (("network", Network), ("library", ComponentLibrary), ("request", CompileRequest))}
