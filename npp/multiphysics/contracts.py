"""Versioned, immutable multiphysics contracts; deliberately no device equations.

These records describe assumptions and model validity. They do not certify a
fabricated component. Unknown quantities are explicit and never default to zero.
Only Python evaluators deliberately registered by the application can run.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(strict=True, min_length=1, max_length=160,
                                   pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")]
Text = Annotated[str, Field(strict=True, min_length=1, max_length=4096, pattern=r"\S")]
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=1, le=1_000_000_000)]
Digest = Annotated[str, Field(strict=True, pattern=r"^[a-f0-9]{64}$")]
Fidelity = Literal["algebraic_ideal", "analytical_cost", "reduced_order_dynamics",
                   "circuit_device", "measured_surrogate"]
Carrier = Literal["electrical", "optical", "acoustic"]
Regime = Literal["classical_digital", "classical_analog", "quantum"]
Status = Literal["invalid", "unsupported", "conditional", "model_feasible",
                 "numerical_failure", "resource_infeasible"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True,
                              validate_default=True, revalidate_instances="always")


class Interval(Contract):
    minimum: Finite
    maximum: Finite

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class Quantity(Contract):
    state: Literal["known", "bounded", "unavailable"]
    unit: Text
    value: Finite | None = None
    lower: Finite | None = None
    upper: Finite | None = None
    symbol: Identifier | None = None
    reason: Text | None = None

    @model_validator(mode="after")
    def meaningful(self):
        if self.state == "known":
            if self.value is None or any(x is not None for x in
                                         (self.lower, self.upper, self.symbol, self.reason)):
                raise ValueError("known quantity requires only a finite value and unit")
        elif self.state == "bounded":
            if self.value is not None or self.reason is not None:
                raise ValueError("bounded quantity cannot contain a value or unavailable reason")
            if self.lower is None or self.upper is None:
                if self.symbol is None:
                    raise ValueError("a partly bounded quantity requires an explicit symbol")
            if self.lower is not None and self.upper is not None and self.lower > self.upper:
                raise ValueError("quantity lower bound exceeds upper bound")
        elif self.reason is None or any(x is not None for x in
                                       (self.value, self.lower, self.upper, self.symbol)):
            raise ValueError("unavailable quantity requires a reason and cannot contain a value")
        return self


def known(value: float, unit: str) -> Quantity:
    return Quantity(state="known", value=value, unit=unit)


def unavailable(unit: str, reason: str) -> Quantity:
    return Quantity(state="unavailable", unit=unit, reason=reason)


class Evidence(Contract):
    kind: Literal["hypothetical", "source_parameterized", "independently_simulated",
                  "characterized", "measured_instance"]
    source: Text
    location: Text
    version: Text
    scope: Text


class Uncertainty(Contract):
    kind: Literal["none", "aleatoric", "epistemic", "mixed", "unknown"]
    description: Text
    distribution: Text | None = None
    correlation_group: Identifier | None = None
    bounds: Interval | None = None

    @model_validator(mode="after")
    def none_is_not_random(self):
        if self.kind == "none" and any(v is not None for v in
                                        (self.distribution, self.correlation_group, self.bounds)):
            raise ValueError("no uncertainty cannot also declare a distribution, correlation or bounds")
        return self


class Parameter(Contract):
    name: Identifier
    quantity: Quantity
    evidence: Evidence
    uncertainty: Uncertainty
    envelope: Interval | None = None
    dependencies: tuple[Identifier, ...] = ()
    instance_group: Identifier

    @model_validator(mode="after")
    def valid_parameter(self):
        if len(set(self.dependencies)) != len(self.dependencies) or self.name in self.dependencies:
            raise ValueError("parameter dependencies must be unique and not self-referential")
        if self.envelope is not None:
            values = (self.quantity.value, self.quantity.lower, self.quantity.upper)
            if any(v is not None and not self.envelope.minimum <= v <= self.envelope.maximum for v in values):
                raise ValueError("parameter value/domain is outside its declared valid envelope")
        return self


class ClockContract(Contract):
    domain: Identifier
    mode: Literal["synchronous", "asynchronous"]
    period_ns: Positive | None = None
    phase_ns: Nonnegative = 0.0

    @model_validator(mode="after")
    def valid_clock(self):
        if self.mode == "synchronous":
            if self.period_ns is None or self.phase_ns >= self.period_ns:
                raise ValueError("synchronous clock requires period and phase smaller than period")
        elif self.period_ns is not None or self.phase_ns != 0:
            raise ValueError("asynchronous clock cannot declare a period or phase")
        return self


class LifetimeContract(Contract):
    arrival_ns: Interval
    integration_ns: Nonnegative
    retention_ns: Quantity
    reset_required: bool

    @model_validator(mode="after")
    def temporal_ranges(self):
        if self.arrival_ns.minimum < 0 or self.retention_ns.unit != "ns":
            raise ValueError("arrival must be nonnegative and retention must use ns")
        _nonnegative_quantity(self.retention_ns)
        return self


class SignalContract(Contract):
    payload: Literal["tensor", "scalar", "control", "probability_sample", "quantum_register", "state_reference"]
    carrier: Carrier
    regime: Regime
    shape: tuple[Count, ...] = Field(min_length=1, max_length=8)
    axis_order: tuple[Identifier, ...] = Field(min_length=1, max_length=8)
    encoding: Identifier
    physical_variable: Text
    signed: bool
    scale: Positive
    unit: Text
    value_range: Interval
    bandwidth_hz: Quantity
    clock: ClockContract
    lifetime: LifetimeContract
    multiplexing: tuple[Identifier, ...] = ()
    disturbance_ids: tuple[Identifier, ...] = ()
    required_services: tuple[Identifier, ...] = ()
    phase_reference: Identifier | None = None
    quantum_platform: Text | None = None

    @model_validator(mode="after")
    def complete_signal(self):
        if len(self.shape) != len(self.axis_order) or len(set(self.axis_order)) != len(self.axis_order):
            raise ValueError("axis_order must uniquely name every shape axis")
        if not self.signed and self.value_range.minimum < 0:
            raise ValueError("unsigned signal range cannot contain negative values")
        if self.bandwidth_hz.unit != "Hz":
            raise ValueError("bandwidth must use Hz")
        _nonnegative_quantity(self.bandwidth_hz)
        for values in (self.multiplexing, self.disturbance_ids, self.required_services):
            if len(set(values)) != len(values):
                raise ValueError("multiplex, disturbance and service identifiers must be unique")
        if self.regime == "classical_digital" and self.carrier != "electrical":
            raise ValueError("this schema's digital reference carrier is electrical; other encodings need a new explicit contract")
        if self.regime == "quantum":
            if self.payload not in ("quantum_register", "state_reference") or self.quantum_platform is None:
                raise ValueError("quantum signal needs register/state-reference payload and physical platform")
        elif self.payload == "quantum_register" or self.quantum_platform is not None:
            raise ValueError("classical signal cannot expose a quantum register or quantum platform")
        if self.encoding in ("intensity", "normalized_intensity"):
            if self.carrier != "optical" or self.regime != "classical_analog" or self.signed or self.value_range.minimum < 0:
                raise ValueError("classical intensity encoding requires unsigned nonnegative optical signal")
            if self.physical_variable not in ("power", "intensity"):
                raise ValueError("intensity encoding must identify power or intensity, not field amplitude")
        if self.encoding in ("coherent_field", "signed_coherent_field"):
            if self.carrier != "optical" or self.regime != "classical_analog" or self.phase_reference is None:
                raise ValueError("coherent field encoding needs classical optical carrier and explicit phase reference")
            if self.physical_variable != "field_amplitude":
                raise ValueError("coherent encoding requires field_amplitude physical variable")
        return self


def signal_mismatches(source: SignalContract, target: SignalContract) -> tuple[str, ...]:
    """Exact wire compatibility. Conversions/resampling require an explicit block.

    Arrival, lifetime and inherited disturbances are validated by the scheduler;
    all representational and service fields must agree at a plain wire.
    """
    fields = ("payload", "carrier", "regime", "shape", "axis_order", "encoding",
              "physical_variable", "signed", "scale", "unit", "value_range",
              "bandwidth_hz", "clock", "multiplexing", "required_services",
              "phase_reference", "quantum_platform")
    return tuple(name for name in fields if getattr(source, name) != getattr(target, name))


class PortContract(Contract):
    name: Identifier
    direction: Literal["in", "out"]
    signal: SignalContract


class EnvelopeAxis(Contract):
    name: Identifier
    unit: Text
    minimum: Finite
    maximum: Finite

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("envelope bounds are reversed")
        if self.name == "temperature_c" and self.minimum < -273.15:
            raise ValueError("temperature below absolute zero")
        return self


class IntegrationContract(Contract):
    substrate: Text
    package: Text
    temperature_c: Interval
    required_services: tuple[Identifier, ...]
    interface_families: tuple[Identifier, ...]
    geometry_assumptions: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def physical_temperature(self):
        if self.temperature_c.minimum < -273.15:
            raise ValueError("temperature below absolute zero")
        return self


class DynamicsContract(Contract):
    startup_ns: Quantity
    programming_ns: Quantity
    settling_ns: Quantity
    integration_ns: Quantity
    reset_ns: Quantity
    retention_ns: Quantity
    calibration_ns: Quantity

    @model_validator(mode="after")
    def times(self):
        for name in type(self).model_fields:
            q = getattr(self, name)
            if q.unit != "ns":
                raise ValueError(f"{name} must use ns")
            _nonnegative_quantity(q)
        return self


CostCategory = Literal["compute", "movement", "conversion", "storage", "programming",
                       "calibration", "control", "support", "source", "reset"]


class AccountingPolicy(Contract):
    mode: Literal["leaf", "expanded", "inclusive"]
    boundary: Identifier
    cost_categories: tuple[CostCategory, ...] = Field(min_length=1)
    exclusions: tuple[Text, ...]
    amortization: Text

    @model_validator(mode="after")
    def unique_categories(self):
        if len(set(self.cost_categories)) != len(self.cost_categories):
            raise ValueError("duplicate cost categories")
        return self


class ValidationContract(Contract):
    reference_implementations: tuple[Text, ...]
    tests: tuple[Text, ...]
    known_failures: tuple[Text, ...]
    limitations: tuple[Text, ...] = Field(min_length=1)


class ComponentContract(Contract):
    id: Identifier
    version: Text
    family: Literal["digital", "analog_electrical", "photonic", "acoustic", "quantum", "interface", "coupling"]
    mechanism: Text
    evaluator_id: Identifier
    evaluator_version: Text
    ideal_relation: Text
    fidelity: Fidelity
    parameters: tuple[Parameter, ...] = Field(max_length=256)
    ports: tuple[PortContract, ...] = Field(min_length=1, max_length=256)
    operating_envelope: tuple[EnvelopeAxis, ...] = Field(min_length=1, max_length=64)
    integration: IntegrationContract
    dynamics: DynamicsContract
    imperfections: tuple[Text, ...]
    accounting: AccountingPolicy
    validation: ValidationContract

    @model_validator(mode="after")
    def unique_and_related(self):
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate parameter name")
        if len({p.name for p in self.ports}) != len(self.ports):
            raise ValueError("duplicate port name")
        if len({a.name for a in self.operating_envelope}) != len(self.operating_envelope):
            raise ValueError("duplicate envelope axis")
        parameters = {p.name: p for p in self.parameters}
        for p in self.parameters:
            if not set(p.dependencies) <= set(names):
                raise ValueError("parameter dependency names an absent parameter")
            for dependency in p.dependencies:
                if parameters[dependency].instance_group != p.instance_group:
                    raise ValueError("dependent parameters from incompatible instance groups cannot be combined")
        visiting: set[str] = set()
        visited: set[str] = set()
        def visit(name):
            if name in visiting:
                raise ValueError("cyclic parameter dependency")
            if name in visited:
                return
            visiting.add(name)
            for dependency in parameters[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
        for name in names:
            visit(name)
        return self


class TechnologyPack(Contract):
    schema_version: Literal["npp-multiphysics-1"] = "npp-multiphysics-1"
    id: Identifier
    version: Text
    components: tuple[ComponentContract, ...] = Field(min_length=1, max_length=256)
    assumptions: tuple[Text, ...]

    @model_validator(mode="after")
    def unique_components(self):
        if len({c.id for c in self.components}) != len(self.components):
            raise ValueError("duplicate component id")
        return self


class ResourceContract(Contract):
    id: Identifier
    owner: Identifier
    kind: Literal["compute", "converter", "port", "link", "buffer", "control", "memory", "source"]
    capacity: Count
    capacity_unit: Text
    region: Identifier


class Reservation(Contract):
    resource_id: Identifier
    owner: Identifier
    start_ns: Nonnegative
    end_ns: Nonnegative
    amount: Count

    @model_validator(mode="after")
    def ordered(self):
        if self.end_ns < self.start_ns:
            raise ValueError("reservation ends before it starts")
        return self


class StateContract(Contract):
    id: Identifier
    owner: Identifier
    version: Annotated[int, Field(strict=True, ge=0)]
    kind: Literal["weights", "cache", "charge", "excitation", "quantum_register", "calibration"]
    created_ns: Nonnegative
    retention_ns: Quantity
    read_mode: Literal["repeatable", "destructive", "bounded", "unsupported"]
    max_reads: Count | None = None
    reset_required: bool
    capacity: Count
    capacity_unit: Text

    @model_validator(mode="after")
    def read_contract(self):
        if self.retention_ns.unit != "ns":
            raise ValueError("state retention must use ns")
        _nonnegative_quantity(self.retention_ns)
        if self.read_mode == "bounded" and self.max_reads is None:
            raise ValueError("bounded read mode requires max_reads")
        if self.read_mode != "bounded" and self.max_reads is not None:
            raise ValueError("max_reads is only legal for bounded read mode")
        if self.kind == "quantum_register" and self.read_mode == "repeatable":
            raise ValueError("arbitrary quantum states do not have repeatable classical read semantics")
        return self


class StateAccess(Contract):
    state_id: Identifier
    version: Annotated[int, Field(strict=True, ge=0)]
    actor: Identifier
    mode: Literal["read", "write", "consume", "reset"]


def _nonnegative_quantity(quantity: Quantity):
    if any(v is not None and v < 0 for v in (quantity.value, quantity.lower, quantity.upper)):
        raise ValueError("cost/time/bandwidth quantities cannot be negative")


class AccountingOwner(Contract):
    id: Identifier
    mode: Literal["leaf", "expanded", "inclusive"]
    children: tuple[Identifier, ...] = ()
    resource_summary_complete: bool = False
    reservations: tuple[Reservation, ...] = ()

    @model_validator(mode="after")
    def composite(self):
        if len(set(self.children)) != len(self.children) or self.id in self.children:
            raise ValueError("owner children must be unique and cannot include self")
        if self.mode == "leaf" and self.children:
            raise ValueError("leaf owner cannot have children")
        if self.mode != "leaf" and not self.children:
            raise ValueError("composite owner requires children")
        if self.mode == "inclusive" and not self.resource_summary_complete:
            raise ValueError("inclusive composite must expose a complete resource reservation summary")
        if any(r.owner != self.id for r in self.reservations):
            raise ValueError("exported reservations must name their accounting owner")
        return self


class CostEntry(Contract):
    contribution_id: Identifier
    owner: Identifier
    boundary: Identifier
    category: CostCategory
    metric: Identifier
    quantity: Quantity
    basis: Text

    @model_validator(mode="after")
    def nonnegative(self):
        _nonnegative_quantity(self.quantity)
        return self


class Aggregate(Contract):
    metric: Identifier
    quantity: Quantity
    known_subtotal: Nonnegative
    complete: bool
    contributions: tuple[Identifier, ...]
    unresolved: tuple[Identifier, ...]


class CostLedger(Contract):
    boundary: Identifier
    owners: tuple[AccountingOwner, ...] = Field(min_length=1, max_length=1024)
    entries: tuple[CostEntry, ...] = Field(max_length=16384)
    expected_contributions: tuple[Identifier, ...] = Field(max_length=16384)

    @model_validator(mode="after")
    def ownership(self):
        owners = {o.id: o for o in self.owners}
        if len(owners) != len(self.owners):
            raise ValueError("duplicate owner id")
        ids = [e.contribution_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("every contribution must have exactly one owner; duplicate contribution id")
        if len(set(self.expected_contributions)) != len(self.expected_contributions) or set(ids) != set(self.expected_contributions):
            raise ValueError("ledger entries must cover every expected contribution exactly once, using unavailable entries for unknown costs")
        parents = {}
        for o in self.owners:
            for child in o.children:
                if child not in owners or child in parents:
                    raise ValueError("child owner must exist and have only one parent")
                parents[child] = o.id
        for name in owners:
            seen = set()
            node = name
            while node in parents:
                if node in seen:
                    raise ValueError("accounting ownership cycle")
                seen.add(node)
                node = parents[node]
        def reservation_key(reservation):
            return (reservation.resource_id, reservation.start_ns, reservation.end_ns, reservation.amount)
        # Inclusive owners may hide descendants, but not their declared capacity
        # reservations. Walk iteratively so deeply nested submitted records do
        # not depend on Python's recursion limit.
        for owner in self.owners:
            if owner.mode != "inclusive":
                continue
            required = Counter()
            pending = list(owner.children)
            while pending:
                child = owners[pending.pop()]
                required.update(reservation_key(r) for r in child.reservations)
                if child.mode == "expanded":
                    pending.extend(child.children)
            exported = Counter(reservation_key(r) for r in owner.reservations)
            if required - exported:
                raise ValueError("inclusive reservation summary omits a declared child reservation")
        for entry in self.entries:
            if entry.owner not in owners or entry.boundary != self.boundary:
                raise ValueError("entry must have a declared owner within the ledger system boundary")
            node = entry.owner
            while node in parents:
                node = parents[node]
                if owners[node].mode == "inclusive":
                    raise ValueError("inclusive composite already owns its descendants' costs; descendant costs cannot also be counted")
        return self

    def aggregate(self, metric: str, unit: str) -> Aggregate:
        entries = tuple(e for e in self.entries if e.metric == metric)
        if any(e.quantity.unit != unit for e in entries):
            raise ValueError("aggregate units differ; an explicit conversion is required")
        if not entries:
            return Aggregate(metric=metric, quantity=unavailable(unit, "No contribution declared for this metric"),
                             known_subtotal=0.0, complete=False, contributions=(), unresolved=())
        known_values = [e.quantity.value for e in entries if e.quantity.state == "known"]
        try:
            subtotal = math.fsum(known_values)
        except OverflowError as exc:
            raise ValueError("cost aggregate exceeds supported finite numeric range") from exc
        if not math.isfinite(subtotal):
            raise ValueError("cost aggregate exceeds supported finite numeric range")
        unresolved = tuple(e.contribution_id for e in entries if e.quantity.state != "known")
        ids = tuple(e.contribution_id for e in entries)
        if not unresolved:
            q = known(subtotal, unit)
        elif any(e.quantity.state == "unavailable" for e in entries):
            q = unavailable(unit, "Incomplete total: " + ", ".join(unresolved))
        else:
            def bound(which):
                values = []
                for e in entries:
                    value = e.quantity.value if e.quantity.state == "known" else getattr(e.quantity, which)
                    if value is None:
                        return None
                    values.append(value)
                try:
                    result = math.fsum(values)
                except OverflowError as exc:
                    raise ValueError("cost bound exceeds supported finite numeric range") from exc
                if not math.isfinite(result):
                    raise ValueError("cost bound exceeds supported finite numeric range")
                return result
            q = Quantity(state="bounded", unit=unit, lower=bound("lower"), upper=bound("upper"), symbol="aggregate." + metric)
        return Aggregate(metric=metric, quantity=q, known_subtotal=subtotal, complete=not unresolved,
                         contributions=ids, unresolved=unresolved)


class Diagnostic(Contract):
    code: Identifier
    message: Text
    path: Text


class EvaluationOutcome(Contract):
    status: Status
    diagnostics: tuple[Diagnostic, ...]
    obligations: tuple[Text, ...] = ()
    ledger: CostLedger | None = None
    result_digest: Digest | None = None
    payload_json: str | None = None

    @field_validator("payload_json")
    @classmethod
    def immutable_payload(cls, value):
        if value is None:
            return None
        try:
            return canonical_json(json.loads(value))
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError("payload_json must contain finite canonicalizable JSON") from exc

    @model_validator(mode="after")
    def successful_meaning(self):
        if self.payload_json is not None:
            digest = hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest()
            if self.result_digest is not None and self.result_digest != digest:
                raise ValueError("result_digest does not bind the outcome payload")
            object.__setattr__(self, "result_digest", digest)
        if self.status == "model_feasible" and self.obligations:
            raise ValueError("an outcome with unresolved obligations must be conditional")
        if self.status == "conditional" and not self.obligations:
            raise ValueError("conditional outcome must state unresolved obligations")
        if self.status not in ("conditional", "model_feasible") and not self.diagnostics:
            raise ValueError("unsuccessful outcome requires an actionable diagnostic")
        if self.status == "model_feasible" and self.ledger is not None:
            if any(e.quantity.state != "known" for e in self.ledger.entries):
                raise ValueError("unknown or bounded costs require a conditional outcome")
        return self


class RandomStream(Contract):
    id: Identifier
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**64)]
    role: Literal["search", "physical", "shared_disturbance"]
    source_id: Identifier | None = None

    @model_validator(mode="after")
    def shared_identity(self):
        if (self.role == "shared_disturbance") != (self.source_id is not None):
            raise ValueError("only shared disturbances require a source identity")
        return self


class EvaluationIdentity(Contract):
    schema_version: Literal["npp-multiphysics-1"] = "npp-multiphysics-1"
    model_version: Text
    pack_versions: tuple[Text, ...] = Field(min_length=1)
    component_versions: tuple[Text, ...] = Field(min_length=1)
    interface_versions: tuple[Text, ...]
    evaluator_versions: tuple[Text, ...] = Field(min_length=1)
    normalized_inputs_hash: Digest
    workload_hash: Digest
    environment_hash: Digest
    geometry_hash: Digest
    schedule_hash: Digest
    fidelity: Fidelity
    random_streams: tuple[RandomStream, ...]
    seed_policy: Literal["deterministic", "explicit_streams"]

    @model_validator(mode="after")
    def complete_identity(self):
        if len({s.id for s in self.random_streams}) != len(self.random_streams):
            raise ValueError("random streams need unique identifiers")
        shared = [s.source_id for s in self.random_streams if s.role == "shared_disturbance"]
        if len(shared) != len(set(shared)):
            raise ValueError("one shared disturbance source must have one stream, then be distributed to its consumers")
        if self.seed_policy == "deterministic" and self.random_streams:
            raise ValueError("deterministic seed policy cannot declare random streams")
        if self.seed_policy == "explicit_streams" and not self.random_streams:
            raise ValueError("explicit stream policy requires streams")
        roles = {s.role for s in self.random_streams}
        if "search" in roles and roles & {"physical", "shared_disturbance"}:
            search = {s.seed for s in self.random_streams if s.role == "search"}
            physical = {s.seed for s in self.random_streams if s.role in ("physical", "shared_disturbance")}
            if search & physical:
                raise ValueError("search and physical random streams must use separated seeds")
        return self

    def cache_key(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


def canonical_json(value: Any) -> str:
    """Finite JSON only, deterministic key order, no mutable reference retained."""
    seen: set[int] = set()
    nodes = 0
    def validate(obj, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 128 or nodes > 1_000_000:
            raise ValueError("canonical JSON exceeds bounded depth or value count")
        if obj is None or type(obj) in (str, bool, int):
            return
        if type(obj) is float:
            if not math.isfinite(obj):
                raise ValueError("canonical JSON rejects nonfinite floats")
            return
        if type(obj) not in (dict, list, tuple):
            raise ValueError("canonical JSON supports only JSON primitives, lists, tuples and string-key dictionaries")
        if id(obj) in seen:
            raise ValueError("canonical JSON rejects cyclic containers")
        seen.add(id(obj))
        if type(obj) is dict:
            if any(type(key) is not str for key in obj):
                raise ValueError("canonical JSON object keys must be strings")
            values = obj.values()
        else:
            values = obj
        for child in values:
            validate(child, depth + 1)
        seen.remove(id(obj))
    validate(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("value is not finite canonical JSON") from exc


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class BackendDescriptor(Contract):
    id: Identifier
    version: Text
    protocol_version: Literal["npp-backend-1"] = "npp-backend-1"
    families: tuple[Text, ...] = Field(min_length=1)
    capabilities: tuple[Identifier, ...]
    limitations: tuple[Text, ...] = Field(min_length=1)


@runtime_checkable
class BackendProtocol(Protocol):
    """Optional methods must return unsupported, never fake successful physics.

    Context and state cross the boundary as canonical JSON snapshots. This keeps
    one backend from mutating another backend's private Python objects.
    """
    def describe(self) -> BackendDescriptor: ...
    def validate(self, component: ComponentContract, context_json: str) -> EvaluationOutcome: ...
    def instantiate(self, component: ComponentContract, context_json: str) -> EvaluationOutcome: ...
    def evaluate(self, instance_json: str, context_json: str) -> EvaluationOutcome: ...
    def advance(self, state_json: str, context_json: str) -> EvaluationOutcome: ...
    def sample(self, instance_json: str, context_json: str) -> EvaluationOutcome: ...
    def export(self, instance_json: str, context_json: str) -> EvaluationOutcome: ...
    def compare(self, instance_json: str, reference_json: str, context_json: str) -> EvaluationOutcome: ...


class EvaluatorRegistry:
    """Application-owned dispatch; packs can select IDs, not execute formulas."""
    def __init__(self):
        self._entries: dict[tuple[str, str], tuple[BackendDescriptor, Callable]] = {}

    def register(self, descriptor: BackendDescriptor, evaluator: Callable[[str, str], EvaluationOutcome]):
        key = (descriptor.id, descriptor.version)
        if key in self._entries:
            raise ValueError("evaluator id/version already registered")
        if not callable(evaluator):
            raise ValueError("only application-provided callables may be registered")
        self._entries[key] = (descriptor, evaluator)

    def describe(self, evaluator_id: str, version: str) -> BackendDescriptor:
        if (evaluator_id, version) not in self._entries:
            raise ValueError("unregistered evaluator id/version")
        return self._entries[(evaluator_id, version)][0]

    def evaluate(self, evaluator_id: str, version: str, inputs: Any, context: Any) -> EvaluationOutcome:
        entry = self._entries.get((evaluator_id, version))
        if entry is None:
            return EvaluationOutcome(status="unsupported", diagnostics=(Diagnostic(
                code="unavailable_model", path="evaluator", message=f"No registered evaluator {evaluator_id}@{version}"),))
        try:
            inputs_json, context_json = canonical_json(inputs), canonical_json(context)
        except (ValueError, TypeError) as exc:
            return EvaluationOutcome(status="invalid", diagnostics=(Diagnostic(
                code="invalid_evaluation_input", path="inputs/context", message=str(exc)[:4000] or "Invalid evaluator inputs"),))
        try:
            result = entry[1](inputs_json, context_json)
            return EvaluationOutcome.model_validate(result)
        except ArithmeticError as exc:
            return EvaluationOutcome(status="numerical_failure", diagnostics=(Diagnostic(
                code="evaluation_failure", path="evaluator", message=str(exc)[:4000] or "Numerical evaluation failed"),))
        except (ValueError, TypeError) as exc:
            return EvaluationOutcome(status="invalid", diagnostics=(Diagnostic(
                code="invalid_evaluation_contract", path="evaluator", message=str(exc)[:4000] or "Invalid evaluator contract"),))
