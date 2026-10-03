"""Bounded deterministic scheduling and explicit owned, versioned state.

This is a non-preemptive list scheduler, not an optimal scheduler. Capacity is
reserved over half-open intervals. State operations occur at declared start/end
instants; conflicting accesses conservatively hold a lock for the full task.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contracts import (Contract, Identifier, Nonnegative, Count, ResourceContract,
                        StateContract, canonical_json)

MAX_TASKS = 2048


class StateError(ValueError):
    def __init__(self, code, message, status="invalid"):
        self.code, self.status = code, status
        super().__init__(message)


class StateStore:
    """Owns canonical snapshots. Writers/resets require the state owner.

    Explicit reader grants permit classical read/consume operations. Quantum
    payloads remain private: arbitrary classical reads are unsupported; consuming
    a register returns only its versioned handle, never simulator amplitudes.
    """
    def __init__(self):
        self._records = {}
        self.events = []

    @staticmethod
    def _occupancy(contract, payload, occupancy):
        def tensor_count(value):
            if type(value) in (int, float):
                return 1
            if type(value) is list:
                return sum(tensor_count(item) for item in value)
            raise StateError("invalid_state_payload", "word64/complex_lane state requires numeric tensor data")
        if contract.capacity_unit == "word64":
            amount, basis = tensor_count(payload), "derived_numeric_word_count"
        elif contract.capacity_unit == "complex_lane":
            if type(payload) is not dict or "real" not in payload or "imag" not in payload:
                raise StateError("invalid_state_payload", "complex_lane state requires real/imag tensor payloads")
            real, imag = tensor_count(payload["real"]), tensor_count(payload["imag"])
            if real != imag:
                raise StateError("invalid_state_payload", "complex lane real/imag payload sizes differ")
            amount, basis = real, "derived_complex_lane_count"
        else:
            if occupancy is None:
                raise StateError("unknown_state_occupancy", "Opaque state units require explicit occupancy; byte/qubit storage cannot be inferred from arbitrary JSON", "conditional")
            amount, basis = occupancy, "declared_in_capacity_units"
        if type(amount) is not int or amount < 0 or occupancy is not None and (type(occupancy) is not int or occupancy != amount):
            raise StateError("invalid_state_occupancy", "Occupancy must be a nonnegative integer matching the payload when derivable")
        if amount > contract.capacity:
            raise StateError("state_capacity", f"State {contract.id} needs {amount} {contract.capacity_unit}, capacity is {contract.capacity}", "resource_infeasible")
        return amount, basis

    def prepare(self, contract: StateContract | dict, payload, readers=(), occupancy=None):
        contract = _model(StateContract, contract)
        if contract.id in self._records:
            raise StateError("state_exists", f"State {contract.id} already exists; reset/write explicitly")
        if type(readers) not in (tuple, list) or any(type(a) is not str or not a for a in readers) or len(readers) != len(set(readers)):
            raise StateError("invalid_readers", "Reader grants must be distinct actor identifiers")
        encoded = canonical_json(payload)
        amount, basis = self._occupancy(contract, json.loads(encoded), occupancy)
        self._records[contract.id] = dict(contract=contract, payload_json=encoded, occupancy=amount, occupancy_basis=basis,
                                         readers=tuple(readers), reads=0, consumed=False, reset_pending=False,
                                         last_event_ns=contract.created_ns)
        self.events.append(dict(mode="create", state_id=contract.id, version=contract.version,
                                actor=contract.owner, time_ns=contract.created_ns))
        return self._handle(contract)

    @staticmethod
    def _handle(contract):
        return dict(state_id=contract.id, version=contract.version, owner=contract.owner)

    def _record(self, state_id, version, actor, time_ns, *, owner_only=False, require_live=True):
        if type(version) is not int or version < 0:
            raise StateError("invalid_version", "State version must be a nonnegative integer")
        if type(time_ns) not in (int, float) or not math.isfinite(time_ns) or time_ns < 0:
            raise StateError("invalid_time", "State operation needs a nonnegative finite time")
        if state_id not in self._records:
            raise StateError("missing_state", f"State {state_id} does not exist")
        record = self._records[state_id]
        contract = record["contract"]
        if version != contract.version:
            raise StateError("stale_state_version", f"State {state_id} is version {contract.version}, requested {version}")
        if actor != contract.owner and (owner_only or actor not in record["readers"]):
            raise StateError("state_owner_mismatch", f"Actor {actor} has no permitted access to state {state_id}")
        if time_ns < contract.created_ns or time_ns < record["last_event_ns"]:
            raise StateError("state_time_reversal", f"State {state_id} cannot be accessed before its creation/latest event")
        if require_live:
            if record["consumed"] or record["reset_pending"]:
                raise StateError("state_consumed", f"State {state_id} has been consumed or reset")
            age = time_ns - contract.created_ns
            retention = contract.retention_ns
            if retention.state == "unavailable":
                raise StateError("unknown_retention", f"Retention for {state_id} is unavailable: {retention.reason}", "conditional")
            if retention.state == "bounded":
                if retention.upper is not None and age > retention.upper:
                    raise StateError("retention_expired", f"State {state_id} exceeds its maximum retention", "resource_infeasible")
                if retention.lower is None or age > retention.lower:
                    raise StateError("uncertain_retention", f"Retention bounds do not guarantee state {state_id} at age {age} ns", "conditional")
            elif age > retention.value:
                raise StateError("retention_expired", f"State {state_id} expired after {retention.value} ns; actual age {age} ns", "resource_infeasible")
        return record

    def read(self, state_id, version, actor, time_ns):
        record = self._record(state_id, version, actor, time_ns)
        contract = record["contract"]
        if contract.kind == "quantum_register":
            raise StateError("quantum_readout_required", "Quantum state cannot be copied/read as a classical payload; use a measurement backend", "unsupported")
        if contract.read_mode == "unsupported":
            raise StateError("unsupported_state_read", f"State {state_id} does not support reads", "unsupported")
        if contract.read_mode == "bounded" and record["reads"] >= contract.max_reads:
            raise StateError("state_reads_exhausted", f"State {state_id} reached its read limit")
        result = json.loads(record["payload_json"])
        record["reads"] += 1
        if contract.read_mode == "destructive":
            record["consumed"] = True
        record["last_event_ns"] = time_ns
        self.events.append(dict(mode="read", state_id=state_id, version=version, actor=actor, time_ns=time_ns,
                                destructive=contract.read_mode == "destructive"))
        return result

    def consume(self, state_id, version, actor, time_ns):
        record = self._record(state_id, version, actor, time_ns)
        contract = record["contract"]
        if contract.read_mode == "unsupported":
            raise StateError("unsupported_state_read", f"State {state_id} does not support consumption", "unsupported")
        if contract.read_mode == "bounded" and record["reads"] >= contract.max_reads:
            raise StateError("state_reads_exhausted", f"State {state_id} reached its read limit")
        record["consumed"] = True
        record["reads"] += 1
        record["last_event_ns"] = time_ns
        self.events.append(dict(mode="consume", state_id=state_id, version=version, actor=actor, time_ns=time_ns))
        return self._handle(contract) if contract.kind == "quantum_register" else json.loads(record["payload_json"])

    def reset(self, state_id, version, actor, time_ns):
        record = self._record(state_id, version, actor, time_ns, owner_only=True, require_live=False)
        record["consumed"] = True
        record["reset_pending"] = True
        record["last_event_ns"] = time_ns
        record["payload_json"] = "null"
        self.events.append(dict(mode="reset", state_id=state_id, version=version, actor=actor, time_ns=time_ns))

    def write(self, state_id, version, actor, time_ns, payload, new_version, occupancy=None):
        record = self._record(state_id, version, actor, time_ns, owner_only=True, require_live=False)
        contract = record["contract"]
        if type(new_version) is not int or new_version != version + 1:
            raise StateError("invalid_version_transition", "A write must increment the state version by exactly one")
        if contract.reset_required and not record["reset_pending"]:
            raise StateError("reset_required", f"State {state_id} requires reset before writing")
        encoded = canonical_json(payload)
        amount, basis = self._occupancy(contract, json.loads(encoded), occupancy)
        updated = contract.model_dump(mode="json")
        updated.update(version=new_version, created_ns=float(time_ns))
        record.update(contract=StateContract.model_validate_json(canonical_json(updated)), payload_json=encoded,
                      occupancy=amount, occupancy_basis=basis, reads=0, consumed=False, reset_pending=False, last_event_ns=time_ns)
        self.events.append(dict(mode="write", state_id=state_id, version=new_version, previous_version=version,
                                actor=actor, time_ns=time_ns))
        return self._handle(record["contract"])

    def snapshot(self):
        result = []
        for state_id in sorted(self._records):
            record = self._records[state_id]
            contract = record["contract"]
            result.append(dict(contract=contract.model_dump(mode="json"), readers=list(record["readers"]),
                               reads=record["reads"], consumed=record["consumed"], reset_pending=record["reset_pending"],
                               occupancy=record["occupancy"], occupancy_basis=record["occupancy_basis"],
                               payload=None if contract.kind == "quantum_register" else json.loads(record["payload_json"]),
                               quantum_payload_private=contract.kind == "quantum_register"))
        return result


class StateEvent(Contract):
    mode: Literal["create", "read", "write", "consume", "reset"]
    at: Literal["start", "end"]
    state_id: Identifier
    version: Annotated[int, Field(strict=True, ge=0)]
    actor: Identifier
    contract: StateContract | None = None
    payload_json: str | None = None
    new_version: Annotated[int, Field(strict=True, ge=0)] | None = None
    readers: tuple[Identifier, ...] = ()
    bind_creation_time: bool = False
    occupancy: Annotated[int, Field(strict=True, ge=0)] | None = None

    @model_validator(mode="after")
    def event_shape(self):
        if self.mode == "create":
            if self.contract is None or self.payload_json is None:
                raise ValueError("create event requires contract and payload_json")
            if (self.contract.id, self.contract.version, self.contract.owner) != (self.state_id, self.version, self.actor):
                raise ValueError("create event must agree with state id, version and owner")
        elif self.contract is not None or self.readers or self.bind_creation_time:
            raise ValueError("only create events supply contracts/reader grants or bind creation time")
        if self.mode == "write":
            if self.payload_json is None or self.new_version != self.version + 1:
                raise ValueError("write requires payload and exactly the next version")
        elif self.new_version is not None:
            raise ValueError("new_version is only valid for write")
        if self.mode not in ("create", "write") and (self.payload_json is not None or self.occupancy is not None):
            raise ValueError("read/consume/reset cannot inject payloads or occupancy")
        if self.payload_json is not None:
            canonical_json(json.loads(self.payload_json))
        return self


class ResourceDemand(Contract):
    resource_id: Identifier
    amount: Count


class Task(Contract):
    id: Identifier
    owner: Identifier
    deps: tuple[Identifier, ...] = ()
    duration_ns: Nonnegative
    resources: tuple[ResourceDemand, ...] = ()
    release_ns: Nonnegative = 0.0
    events: tuple[StateEvent, ...] = ()

    @model_validator(mode="after")
    def consistent(self):
        if self.id in self.deps or len(set(self.deps)) != len(self.deps):
            raise ValueError("task dependencies must be distinct and cannot include itself")
        if len({d.resource_id for d in self.resources}) != len(self.resources):
            raise ValueError("task resource demands must be unique")
        if self.duration_ns == 0 and self.resources:
            raise ValueError("zero-duration tasks may carry state events, but cannot bypass positive resource reservations")
        if any(e.actor != self.owner for e in self.events):
            raise ValueError("task owner must be the actor of every state event")
        return self


def _model(cls, value):
    if isinstance(value, cls):
        return cls.model_validate(value)
    return cls.model_validate_json(canonical_json(value))


def _task(value):
    if isinstance(value, Task):
        return Task.model_validate(value)
    data = deepcopy(value)
    if isinstance(data, dict) and isinstance(data.get("resources"), dict):
        data["resources"] = [dict(resource_id=k, amount=v) for k, v in sorted(data["resources"].items())]
    return _model(Task, data)


def _overlap(a, b, c, d):
    return a < d and c < b


def _safe_add(left, right):
    result = left + right
    if not math.isfinite(result) or right > 0 and result <= left:
        raise ArithmeticError("schedule time exceeds supported finite precision")
    return result


def schedule(tasks, resources, initial_states=()):
    """Schedule <=2048 tasks, validate state events, return serializable evidence.

    Invalid/conditional states are retained as explicit diagnostics. No failed
    schedule is labeled model-feasible, and input objects are never mutated.
    """
    empty = dict(tasks=[], reservations=[], state_locks=[], makespan_ns=None,
                 critical_path=[], events=[], final_states=[], heuristic="deterministic_topological_earliest_fit",
                 scheduling_complete=False, state_execution_complete=False)
    store = None
    try:
        if type(tasks) not in (list, tuple) or not 1 <= len(tasks) <= MAX_TASKS:
            raise ValueError(f"schedule requires 1..{MAX_TASKS} tasks")
        if type(resources) not in (list, tuple) or len(resources) > 4096:
            raise ValueError("resources must be a list/tuple of at most 4096 contracts")
        task_list = [_task(t) for t in tasks]
        resource_list = [_model(ResourceContract, r) for r in resources]
        by_id = {t.id: t for t in task_list}
        resource_map = {r.id: r for r in resource_list}
        if len(by_id) != len(task_list) or len(resource_map) != len(resource_list):
            raise ValueError("duplicate task or resource id")
        for task in task_list:
            if not set(task.deps) <= set(by_id):
                raise ValueError(f"Task {task.id} has missing dependencies")
            for demand in task.resources:
                if demand.resource_id not in resource_map:
                    raise ValueError(f"Task {task.id} requests an undeclared resource")
                if demand.amount > resource_map[demand.resource_id].capacity:
                    raise StateError("resource_capacity", f"Task {task.id} requires more {demand.resource_id} capacity than supplied", "resource_infeasible")
        initial = []
        if type(initial_states) not in (list, tuple) or len(initial_states) > 4096:
            raise ValueError("initial_states must contain at most 4096 state snapshots")
        for state in initial_states:
            if type(state) is not dict or set(state) - {"contract", "payload", "readers", "occupancy"} or not {"contract", "payload"} <= set(state):
                raise ValueError("initial state needs contract/payload and optional explicit readers")
            initial.append(dict(contract=_model(StateContract, state["contract"]), payload=state["payload"], readers=state.get("readers", ()), occupancy=state.get("occupancy")))
        state_contracts = {s["contract"].id: s["contract"] for s in initial}
        for task in task_list:
            for event in task.events:
                if event.mode == "create":
                    if event.state_id in state_contracts:
                        raise ValueError("a state has more than one initial/create declaration")
                    state_contracts[event.state_id] = event.contract
        pending = set(by_id)
        order = []
        ordered = set()
        while pending:
            ready = sorted(t for t in pending if set(by_id[t].deps) <= ordered)
            if not ready:
                raise ValueError("dependency cycle")
            order.extend(ready)
            ordered.update(ready)
            pending.difference_update(ready)
        ancestors = {}
        for task_id in order:
            ancestors[task_id] = set(by_id[task_id].deps)
            for dependency in by_id[task_id].deps:
                ancestors[task_id].update(ancestors[dependency])
        placements, reservations, locks, predecessors = {}, [], [], {}
        resource_bookings = {name: [] for name in resource_map}
        state_bookings = {}
        for task_id in order:
            task = by_id[task_id]
            ready = max([task.release_ns] + [placements[d][1] for d in task.deps])
            task_states = {}
            for event in task.events:
                contract = state_contracts.get(event.state_id)
                exclusive = event.mode != "read" or contract is None or contract.read_mode in ("destructive", "bounded")
                task_states[event.state_id] = task_states.get(event.state_id, False) or exclusive
            forbidden = []
            for demand in task.resources:
                changes = {}
                for reservation in resource_bookings[demand.resource_id]:
                    if reservation["end_ns"] <= ready:
                        continue
                    changes.setdefault(reservation["start_ns"], []).append((reservation["task_id"], reservation["amount"]))
                    changes.setdefault(reservation["end_ns"], []).append((reservation["task_id"], -reservation["amount"]))
                active, usage, previous = {}, 0, None
                threshold = resource_map[demand.resource_id].capacity - demand.amount
                for when in sorted(changes):
                    if previous is not None and previous < when and usage > threshold:
                        forbidden.append((previous, when, tuple(active), "resource"))
                    for prior_id, amount in changes[when]:
                        usage += amount
                        if amount > 0:
                            active[prior_id] = amount
                        else:
                            active.pop(prior_id, None)
                    previous = when
            for state_id, exclusive in task_states.items():
                for lock in state_bookings.get(state_id, ()):
                    if exclusive or lock["exclusive"]:
                        forbidden.append((lock["start_ns"], lock["end_ns"], (lock["task_id"],), "state"))
            start, blockers = ready, set()
            for left, right, prior_ids, kind in sorted(forbidden):
                end = _safe_add(start, task.duration_ns)
                if start == end == left == right and kind == "state":
                    if any(prior not in ancestors[task_id] for prior in prior_ids):
                        raise StateError("simultaneous_state_hazard", "Conflicting zero-duration state events need an explicit dependency")
                    continue
                if start == end:
                    overlap = kind == "state" and left <= start < right
                elif left == right:
                    overlap = start < left < end
                else:
                    overlap = _overlap(start, end, left, right)
                if overlap:
                    start = right
                    blockers.update(prior_ids)
            end = _safe_add(start, task.duration_ns)
            chosen = (start, end)
            placements[task_id] = chosen
            candidates_predecessors = [d for d in (*task.deps, *sorted(blockers)) if placements[d][1] <= start]
            predecessors[task_id] = max(candidates_predecessors, key=lambda d: (placements[d][1], d)) if candidates_predecessors else None
            for demand in task.resources:
                reservation = dict(resource_id=demand.resource_id, owner=task.owner, task_id=task.id,
                                   start_ns=start, end_ns=end, amount=demand.amount)
                reservations.append(reservation)
                resource_bookings[demand.resource_id].append(reservation)
            for state_id, exclusive in task_states.items():
                lock = dict(state_id=state_id, task_id=task.id, start_ns=start, end_ns=end, exclusive=exclusive)
                locks.append(lock)
                state_bookings.setdefault(state_id, []).append(lock)
        last = max(order, key=lambda name: (placements[name][1], name))
        critical = []
        current = last
        while current is not None:
            critical.append(current)
            current = predecessors[current]
        empty.update(tasks=[dict(id=t, owner=by_id[t].owner, start_ns=placements[t][0], end_ns=placements[t][1]) for t in order],
                     reservations=reservations, state_locks=locks, makespan_ns=placements[last][1],
                     critical_path=list(reversed(critical)), scheduling_complete=True)
        store = StateStore()
        empty["events"] = store.events
        for state in initial:
            store.prepare(**state)
        event_queue = []
        for index, task_id in enumerate(order):
            task = by_id[task_id]
            start, end = placements[task_id]
            for event_index, event in enumerate(task.events):
                when = start if event.at == "start" else end
                # At the same timestamp, complete the producing predecessor
                # before beginning its consumer. Topological index handles zero
                # duration state chains without reversing dependency order.
                event_queue.append((when, index, event_index, event, task_id))
        event_queue.sort(key=lambda item: item[:3])
        for when, _, _, event, task_id in event_queue:
            if event.mode == "create":
                contract = event.contract
                if event.bind_creation_time:
                    data = contract.model_dump(mode="json")
                    data["created_ns"] = when
                    contract = _model(StateContract, data)
                if contract.created_ns != when:
                    raise StateError("state_creation_time", f"State {event.state_id} declares creation {contract.created_ns}, scheduled event is {when}")
                store.prepare(contract, json.loads(event.payload_json), event.readers, event.occupancy)
            elif event.mode == "write":
                store.write(event.state_id, event.version, event.actor, when, json.loads(event.payload_json), event.new_version, event.occupancy)
            else:
                getattr(store, event.mode)(event.state_id, event.version, event.actor, when)
            store.events[-1]["task_id"] = task_id
        empty.update(final_states=store.snapshot(), state_execution_complete=True)
        return dict(empty, status="model_feasible", diagnostics=[], obligations=[])
    except StateError as exc:
        if store is not None:
            empty["final_states"] = store.snapshot()
        return dict(empty, status=exc.status, diagnostics=[dict(code=exc.code, message=str(exc))],
                    obligations=[str(exc)] if exc.status == "conditional" else [])
    except ArithmeticError as exc:
        return dict(empty, status="numerical_failure", diagnostics=[dict(code="schedule_numeric_range", message=str(exc))], obligations=[])
    except (ValueError, TypeError, KeyError) as exc:
        return dict(empty, status="invalid", diagnostics=[dict(code="invalid_schedule", message=str(exc))], obligations=[])
