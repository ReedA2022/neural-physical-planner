import copy
import random
import unittest

from npp.multiphysics.contracts import StateContract, known, unavailable
from npp.multiphysics.scheduling import StateError, StateStore, schedule


def resource(name="alu", capacity=1, kind="compute"):
    return dict(id=name, owner="chip", kind=kind, capacity=capacity, capacity_unit="slot", region="chip")


def task(name, duration=2.0, deps=(), resources=None, **kwargs):
    return dict(id=name, owner="unit", duration_ns=duration, deps=list(deps),
                resources={"alu": 1} if resources is None else resources, **kwargs)


def state(**changes):
    values = dict(id="cache", owner="unit", version=0, kind="cache", created_ns=0.0,
                  retention_ns=known(100.0, "ns"), read_mode="repeatable", reset_required=False,
                  capacity=1, capacity_unit="word64")
    values.update(changes)
    return StateContract(**values)


def event(mode="read", version=0, at="start", **changes):
    data = dict(mode=mode, at=at, state_id="cache", version=version, actor="unit")
    data.update(changes)
    return data


class SchedulingTests(unittest.TestCase):
    def test_capacity_precedence_and_critical_path_actual_not_throughput(self):
        tasks = [task("a"), task("b", duration=3.0), task("c", duration=1.0, deps=("a", "b"))]
        result = schedule(tasks, [resource()])
        self.assertEqual(result["status"], "model_feasible", result)
        self.assertEqual([(x["start_ns"], x["end_ns"]) for x in result["tasks"]], [(0.0, 2.0), (2.0, 5.0), (5.0, 6.0)])
        self.assertEqual(result["makespan_ns"], 6.0)
        self.assertEqual(result["critical_path"], ["a", "b", "c"])
        parallel = schedule(tasks, [resource(capacity=2)])
        self.assertEqual(parallel["makespan_ns"], 4.0)
        self.assertEqual(tasks[0]["deps"], [])

    def test_every_resource_kind_is_reserved_and_no_partial_interval_overbooking(self):
        kinds = ["compute", "converter", "port", "link", "buffer", "control", "memory", "source"]
        for kind in kinds:
            result = schedule([task("a", resources={kind: 1}), task("b", resources={kind: 1})], [resource(kind, kind=kind)])
            self.assertEqual(result["makespan_ns"], 4.0, (kind, result))
        result = schedule([task("a", duration=4.0, release_ns=2.0), task("b", duration=3.0)], [resource()])
        self.assertEqual(result["tasks"][1]["start_ns"], 6.0)
        impossible = schedule([task("a", resources={"alu": 2})], [resource()])
        self.assertEqual(impossible["status"], "resource_infeasible")

    def test_cycles_missing_dependencies_strict_numbers_and_zero_reservation_guard(self):
        for tasks in [[task("a", deps=("b",)), task("b", deps=("a",))],
                      [task("a", deps=("missing",))], [task("a"), task("a")],
                      [task("a", duration=True)], [task("a", resources={"alu": True})],
                      [task("a", duration=0.0)], [task("a", resources={"undeclared": 1})]]:
            self.assertEqual(schedule(tasks, [resource()])["status"], "invalid")
        self.assertEqual(schedule([task("a", duration=0.0, resources={})], [])["status"], "model_feasible")
        result = schedule([task("a", duration=1.0, release_ns=1e300)], [resource()])
        self.assertEqual(result["status"], "numerical_failure")

    def test_store_snapshots_ownership_versions_and_expiration(self):
        store = StateStore()
        payload = [1.0]
        store.prepare(state(), payload, readers=("consumer",))
        payload[0] = 100
        result = store.read("cache", 0, "consumer", 0.0)
        result[0] = 500
        self.assertEqual(store.read("cache", 0, "unit", 1.0), [1.0])
        with self.assertRaisesRegex(StateError, "permitted access"):
            store.write("cache", 0, "consumer", 2.0, [3], 1)
        store.write("cache", 0, "unit", 2.0, [3], 1)
        with self.assertRaisesRegex(StateError, "version 1"):
            store.read("cache", 0, "unit", 3.0)
        self.assertEqual(store.read("cache", 1, "unit", 102.0), [3])
        with self.assertRaises(StateError) as error:
            store.read("cache", 1, "unit", 103.0)
        self.assertEqual(error.exception.code, "retention_expired")

    def test_destructive_and_bounded_read_reset_rewrite(self):
        store = StateStore()
        store.prepare(state(read_mode="destructive", reset_required=True), [1])
        self.assertEqual(store.read("cache", 0, "unit", 0.0), [1])
        with self.assertRaisesRegex(StateError, "consumed"):
            store.read("cache", 0, "unit", 1.0)
        with self.assertRaisesRegex(StateError, "reset"):
            store.write("cache", 0, "unit", 1.0, [2], 1)
        store.reset("cache", 0, "unit", 1.0)
        store.write("cache", 0, "unit", 2.0, [2], 1)
        self.assertEqual(store.read("cache", 1, "unit", 2.0), [2])
        bounded = StateStore()
        bounded.prepare(state(read_mode="bounded", max_reads=1), [0])
        bounded.read("cache", 0, "unit", 0.0)
        with self.assertRaisesRegex(StateError, "read limit"):
            bounded.consume("cache", 0, "unit", 0.0)

    def test_unknown_retention_conditional_and_quantum_no_cloning(self):
        unknown = StateStore()
        unknown.prepare(state(retention_ns=unavailable("ns", "not measured")), [1])
        with self.assertRaises(StateError) as error:
            unknown.read("cache", 0, "unit", 0.0)
        self.assertEqual(error.exception.status, "conditional")
        quantum = StateStore()
        quantum.prepare(state(kind="quantum_register", read_mode="destructive", capacity_unit="qubit"), {"amplitudes": [1, 0]}, occupancy=1)
        with self.assertRaises(StateError) as error:
            quantum.read("cache", 0, "unit", 0.0)
        self.assertEqual(error.exception.status, "unsupported")
        self.assertNotIn("amplitudes", quantum.consume("cache", 0, "unit", 0.0))
        self.assertIsNone(quantum.snapshot()[0]["payload"])
        with self.assertRaises(StateError):
            quantum.consume("cache", 0, "unit", 0.0)

    def test_waiting_expiration_actual_schedule_and_unknown_retention(self):
        initial = [dict(contract=state(retention_ns=known(1.0, "ns")), payload=[1])]
        result = schedule([task("busy", duration=5.0), task("read", events=[event()])], [resource()], initial)
        self.assertEqual(result["status"], "resource_infeasible", result)
        self.assertEqual(result["diagnostics"][0]["code"], "retention_expired")
        initial = [dict(contract=state(retention_ns=unavailable("ns", "unknown")), payload=[1])]
        result = schedule([task("read", events=[event()])], [resource()], initial)
        self.assertEqual(result["status"], "conditional")
        self.assertTrue(result["obligations"])
        self.assertTrue(result["scheduling_complete"])
        self.assertFalse(result["state_execution_complete"])
        self.assertEqual(result["makespan_ns"], 2.0)
        self.assertEqual(len(result["reservations"]), 1)
        self.assertEqual(len(result["final_states"]), 1)

    def test_read_write_hazards_serialize_and_stale_version_fails(self):
        initial = [dict(contract=state(), payload=[1])]
        write = task("a_write", duration=3.0, resources={}, events=[event("write", at="end", payload_json="[2]", new_version=1)])
        read = task("b_read", duration=2.0, resources={}, events=[event(version=1)])
        result = schedule([write, read], [], initial)
        self.assertEqual(result["status"], "model_feasible", result)
        self.assertEqual(result["tasks"][1]["start_ns"], 3.0)
        read["events"][0]["version"] = 0
        self.assertEqual(schedule([write, read], [], initial)["status"], "invalid")
        read["events"][0]["version"] = 0
        zero_write = task("a_write", duration=0.0, resources={}, events=[event("write", payload_json="[2]", new_version=1)])
        zero_read = task("b_read", duration=0.0, resources={}, events=[event(version=1)])
        self.assertEqual(schedule([zero_write, zero_read], [], initial)["diagnostics"][0]["code"], "simultaneous_state_hazard")
        zero_read["deps"] = ["a_write"]
        self.assertEqual(schedule([zero_write, zero_read], [], initial)["status"], "model_feasible")

    def test_state_creation_time_binding_and_read_at_exact_zero_retention(self):
        created = state(retention_ns=known(0.0, "ns"))
        create = event("create", at="end", contract=created.model_dump(mode="json"), payload_json="[1]",
                       readers=["unit"], bind_creation_time=True)
        result = schedule([task("a", events=[create]), task("b", deps=("a",), events=[event()])], [resource()])
        self.assertEqual(result["status"], "model_feasible", result)
        self.assertEqual(result["final_states"][0]["contract"]["created_ns"], 2.0)
        delayed = task("b", deps=("a",), release_ns=3.0, events=[event()])
        self.assertEqual(schedule([task("a", events=[create]), delayed], [resource()])["status"], "resource_infeasible")

    def test_payload_occupancy_and_capacity_checked_on_prepare_and_write(self):
        store = StateStore()
        with self.assertRaises(StateError) as error:
            store.prepare(state(), [1.0, 2.0])
        self.assertEqual(error.exception.code, "state_capacity")
        store.prepare(state(), [1.0])
        with self.assertRaises(StateError):
            store.write("cache", 0, "unit", 0.0, [1.0, 2.0], 1)
        self.assertEqual(store.read("cache", 0, "unit", 0.0), [1.0])
        opaque = StateStore()
        with self.assertRaises(StateError) as error:
            opaque.prepare(state(capacity_unit="custom_mode"), {"opaque": 1})
        self.assertEqual(error.exception.status, "conditional")
        opaque.prepare(state(capacity_unit="custom_mode"), {"opaque": 1}, occupancy=1)
        self.assertEqual(opaque.snapshot()[0]["occupancy_basis"], "declared_in_capacity_units")
        field = StateStore()
        field.prepare(state(capacity_unit="complex_lane"), {"real": [1.0], "imag": [0.0]})
        self.assertEqual(field.snapshot()[0]["occupancy"], 1)
        with self.assertRaises(StateError):
            field.write("cache", 0, "unit", 0.0, {"real": [1.0], "imag": [0.0, 1.0]}, 1)

    def test_seeded_capacity_and_precedence_against_discrete_time_oracle(self):
        rng = random.Random(2103)
        for case in range(120):
            capacities = {"alu": rng.randint(1, 3), "link": rng.randint(1, 3)}
            tasks = []
            for index in range(rng.randint(2, 16)):
                deps = rng.sample([t["id"] for t in tasks], min(len(tasks), rng.randint(0, 2)))
                demands = {name: rng.randint(1, capacity) for name, capacity in capacities.items() if rng.random() < 0.75}
                tasks.append(task(f"task{index:03d}", duration=float(rng.randint(1, 6)), deps=deps,
                                  resources=demands, release_ns=float(rng.randint(0, 5))))
            result = schedule(tasks, [resource(name, capacity=capacity, kind="link" if name == "link" else "compute")
                                      for name, capacity in capacities.items()])
            self.assertEqual(result["status"], "model_feasible", (case, result))
            planned = {t["id"]: t for t in result["tasks"]}
            original = {t["id"]: t for t in tasks}
            occupied = {name: {} for name in capacities}
            for actual in result["tasks"]:
                declared = original[actual["id"]]
                earliest = max([declared["release_ns"]] + [planned[d]["end_ns"] for d in declared["deps"]])
                self.assertGreaterEqual(actual["start_ns"], earliest)
                duration = int(declared["duration_ns"])
                # Independent discrete-time exhaustive start oracle for these
                # integer scenarios; production uses continuous event boundaries.
                possible = []
                for start in range(int(earliest), int(actual["start_ns"]) + 1):
                    if all(all(occupied[name].get(time, 0) + amount <= capacities[name]
                               for time in range(start, start + duration))
                           for name, amount in declared["resources"].items()):
                        possible.append(start)
                self.assertEqual(actual["start_ns"], min(possible))
                for name, amount in declared["resources"].items():
                    for time in range(int(actual["start_ns"]), int(actual["end_ns"])):
                        occupied[name][time] = occupied[name].get(time, 0) + amount
                        self.assertLessEqual(occupied[name][time], capacities[name])

    def test_schedule_is_deterministic_and_preserves_inputs(self):
        tasks = [task("c", deps=("a", "b")), task("b"), task("a")]
        snapshot = copy.deepcopy(tasks)
        first = schedule(tasks, [resource(capacity=2)])
        self.assertEqual(first, schedule(list(reversed(tasks)), [resource(capacity=2)]))
        self.assertEqual(tasks, snapshot)


if __name__ == "__main__":
    unittest.main()
