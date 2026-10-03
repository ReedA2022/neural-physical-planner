"""Independent M2 review probes for schedule/state and physical route ownership."""
import copy
import random
import tempfile
import unittest

from npp.multiphysics.contracts import known, unavailable
from npp.multiphysics.geometry import bus_distances, bus_geometry, grid_geometry, validate_geometry
from npp.multiphysics.graph import evaluate_implementation
from npp.multiphysics.io import init_project, load_project
from npp.multiphysics.scheduling import schedule


def state(retention=None, **changes):
    result = dict(id="store", owner="owner", version=0, kind="cache", created_ns=0.0,
                  retention_ns=retention or known(100.0, "ns").model_dump(mode="json"),
                  read_mode="repeatable", reset_required=False, capacity=1, capacity_unit="word64")
    result.update(changes)
    return result


def event(mode="read", version=0, at="start", **changes):
    result = dict(mode=mode, state_id="store", version=version, actor="owner", at=at)
    result.update(changes)
    return result


def task(name, duration=1.0, **changes):
    result = dict(id=name, owner="owner", duration_ns=duration, resources={}, deps=[])
    result.update(changes)
    return result


class IndependentM2ScheduleReview(unittest.TestCase):
    def test_state_lock_waiting_counts_toward_lifetime(self):
        tasks = [task("a_read", 4.0, events=[event()]),
                 task("b_consume", 1.0, events=[event("consume")])]
        result = schedule(tasks, [], [{"contract": state(known(2.0, "ns").model_dump(mode="json")), "payload": [7]}])
        self.assertEqual(result["status"], "resource_infeasible", result)
        self.assertEqual(result["tasks"][1]["start_ns"], 4.0)
        self.assertEqual(result["diagnostics"][0]["code"], "retention_expired")

    def test_bounded_and_unknown_retention_never_become_infinite(self):
        retention = dict(state="bounded", unit="ns", lower=2.0, upper=4.0, symbol="lifetime")
        for release, status in [(2.0, "model_feasible"), (3.0, "conditional"), (5.0, "resource_infeasible")]:
            result = schedule([task("read", release_ns=release, events=[event()])], [],
                              [{"contract": state(retention), "payload": [0]}])
            self.assertEqual(result["status"], status, result)
        result = schedule([task("read", events=[event()])], [],
                          [{"contract": state(unavailable("ns", "not characterized").model_dump(mode="json")), "payload": [0]}])
        self.assertEqual(result["status"], "conditional")
        self.assertFalse(result["state_execution_complete"])

    def test_two_granted_consumers_cannot_clone_destructive_state(self):
        tasks = [task(name, owner=name, events=[event("consume", actor=name)]) for name in ("alice", "bob")]
        result = schedule(tasks, [], [{"contract": state(read_mode="destructive"), "payload": [3], "readers": ["alice", "bob"]}])
        self.assertEqual(result["status"], "invalid", result)
        self.assertEqual(result["diagnostics"][0]["code"], "state_consumed")
        self.assertEqual(len([e for e in result["events"] if e["mode"] == "consume"]), 1)

    def test_reset_write_version_chain_and_reader_cannot_mutate(self):
        initial = [{"contract": state(reset_required=True), "payload": [2], "readers": ["guest"]}]
        chain = [task("reset", 0.0, events=[event("reset")]),
                 task("write", 0.0, deps=["reset"], events=[event("write", payload_json="[9]", new_version=1)]),
                 task("read", 1.0, deps=["write"], events=[event(version=1)])]
        result = schedule(chain, [], initial)
        self.assertEqual(result["status"], "model_feasible", result)
        self.assertEqual(result["final_states"][0]["payload"], [9])
        self.assertEqual(result["final_states"][0]["contract"]["version"], 1)
        attempted = copy.deepcopy(chain)
        attempted[0]["owner"] = attempted[0]["events"][0]["actor"] = "guest"
        result = schedule(attempted, [], initial)
        self.assertEqual(result["diagnostics"][0]["code"], "state_owner_mismatch")

    def test_fractional_capacity_reservations_against_independent_sweep(self):
        rng = random.Random(50621)
        for case in range(32):
            capacities = {"compute": rng.randint(1, 3), "link": rng.randint(1, 3)}
            tasks = []
            for index in range(12):
                deps = [t["id"] for t in tasks if rng.random() < .12]
                demands = {name: rng.randint(1, cap) for name, cap in capacities.items() if rng.random() < .8}
                tasks.append(task(f"t{index:02d}", rng.randint(1, 9) / 8,
                                  release_ns=rng.randint(0, 16) / 8, deps=deps, resources=demands))
            resources = [dict(id=name, owner="owner", kind=name, capacity=cap, capacity_unit="slot", region="chip")
                         for name, cap in capacities.items()]
            result = schedule(tasks, resources)
            self.assertEqual(result["status"], "model_feasible", (case, result))
            planned = {t["id"]: t for t in result["tasks"]}
            for original in tasks:
                actual = planned[original["id"]]
                self.assertGreaterEqual(actual["start_ns"], original["release_ns"])
                self.assertEqual(actual["end_ns"] - actual["start_ns"], original["duration_ns"])
                for dependency in original["deps"]:
                    self.assertLessEqual(planned[dependency]["end_ns"], actual["start_ns"])
            for name, cap in capacities.items():
                reservations = [r for r in result["reservations"] if r["resource_id"] == name]
                # Independent event sweep; end events release before equal-time starts.
                changes = sorted([(r["start_ns"], 1, r["amount"]) for r in reservations] +
                                 [(r["end_ns"], 0, -r["amount"]) for r in reservations])
                used = 0
                for _, _, delta in changes:
                    used += delta
                    self.assertGreaterEqual(used, 0)
                    self.assertLessEqual(used, cap, (case, name, result))
                self.assertEqual(used, 0)

    def test_bus_cannot_forward_through_unregistered_compute_interior(self):
        geometry = grid_geometry(["a", "b", "c"], [("a", "b"), ("b", "c")])
        for route in geometry["routes"]:
            route["direction"] = "bidirectional"
        self.assertEqual(validate_geometry(geometry)["status"], "model_feasible")
        # b has two distinct ports. Mere geometric connectivity cannot provide
        # free pass-through wiring or a port-to-port switch inside that compute block.
        with self.assertRaises(ValueError):
            bus_distances(geometry)

    def test_bus_rejects_unsupported_offcenter_junction_interior(self):
        geometry = bus_geometry(["a", "b"])
        junction = next(p for p in geometry["placements"] if p["id"] == "bus_junction_0")
        junction["ports"][0]["offset_um"] = .25
        route = next(r for r in geometry["routes"] if r["id"] == "bus_tap_0")
        start, end = route["points"]
        route["points"] = [start, dict(x_um=start["x_um"], y_um=end["y_um"] - 2),
                           dict(x_um=end["x_um"] - .25, y_um=end["y_um"] - 2),
                           dict(x_um=end["x_um"] - .25, y_um=end["y_um"])]
        # The current unit junction model owns center-to-edge 0.5um segments;
        # offcenter ports require another explicit interior route model.
        with self.assertRaises(ValueError):
            bus_distances(geometry)

    def test_shared_physical_bus_capacity_limits_parallel_compilation(self):
        with tempfile.TemporaryDirectory() as directory:
            init_project(directory)
            project = load_project(directory + "/project.yaml")
        for resource in project["constraints"]["resources"]:
            project["constraints"]["resources"][resource] = 4
        result = evaluate_implementation(project)
        self.assertEqual(result["status"], "model_feasible", result)
        # Both projections read the same input from the same memory placement
        # through one physical bus. Four abstract link instances cannot invent
        # four channels on a one-channel route/port.
        minimum_route_capacity = min(r["capacity"] for r in result["geometry"]["routes"])
        minimum_port_capacity = min(p["capacity"] for item in result["geometry"]["geometry"]["placements"] for p in item["ports"])
        tasks = {t["id"]: t for t in result["schedule"]["tasks"]}
        projections = [tasks["run0.gate.0"], tasks["run0.up.0"]]
        concurrent = projections[0]["start_ns"] < projections[1]["end_ns"] and projections[1]["start_ns"] < projections[0]["end_ns"]
        self.assertTrue(not concurrent or min(minimum_route_capacity, minimum_port_capacity) >= 2)


if __name__ == "__main__":
    unittest.main()
