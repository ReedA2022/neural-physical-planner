import copy
import unittest

from npp.multiphysics.geometry import bus_distances, bus_geometry, grid_geometry, validate_geometry


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = grid_geometry(["a", "b", "c"], [("a", "b"), ("b", "c")])

    def test_geometry_derives_lengths_bends_and_preserves_input(self):
        original = copy.deepcopy(self.geometry)
        result = validate_geometry(self.geometry)
        self.assertEqual(result["status"], "model_feasible", result)
        self.assertEqual([r["bends"] for r in result["routes"]], [2, 2])
        for route, facts in zip(self.geometry["routes"], result["routes"]):
            expected = sum(abs(a["x_um"] - b["x_um"]) + abs(a["y_um"] - b["y_um"]) for a, b in zip(route["points"], route["points"][1:]))
            self.assertEqual(facts["length_um"], expected)
        self.assertEqual(original, self.geometry)
        self.assertEqual(result, validate_geometry(self.geometry))

    def test_footprints_regions_services_and_thermal_compatibility(self):
        mutations = [("overlap", lambda g: g["placements"][1].update(x_um=11.0)),
                     ("outside", lambda g: g["placements"][0].update(x_um=1000.0)),
                     ("substrate", lambda g: g["placements"][0].update(substrate="incompatible")),
                     ("package", lambda g: g["placements"][0].update(package="incompatible")),
                     ("cold", lambda g: g["regions"][0].update(temperature_c=-200.0)),
                     ("services", lambda g: g["placements"][0].update(required_services=["cryogenic_support"])),
                     ("orientation_boolean", lambda g: g["placements"][0].update(orientation=False)),
                     ("negative_width", lambda g: g["placements"][0].update(width_um=-1.0))]
        for label, change in mutations:
            data = copy.deepcopy(self.geometry)
            change(data)
            self.assertEqual(validate_geometry(data)["status"], "invalid", label)

    def test_ports_orientation_units_and_route_capacities(self):
        for change in [lambda g: g["routes"][0]["points"][0].update(x_um=0.0),
                       lambda g: g["placements"][0].update(orientation=90),
                       lambda g: g["routes"][0].update(capacity=1, usage=2),
                       lambda g: g["routes"][0].update(capacity=True),
                       lambda g: g["routes"][0].update(carrier="quantum"),
                       lambda g: g["routes"][0]["points"][1].update(x_um=1.0),
                       lambda g: g["placements"][0]["ports"][0].update(offset_um=999.0)]:
            data = copy.deepcopy(self.geometry)
            change(data)
            self.assertEqual(validate_geometry(data)["status"], "invalid")
        rotated = grid_geometry(["a"])
        rotated["placements"][0]["orientation"] = 90
        self.assertEqual(validate_geometry(rotated)["status"], "model_feasible")

    def test_routes_do_not_tunnel_through_footprints_or_undeclared_crossings(self):
        data = grid_geometry(["a", "b"], [("a", "b")])
        # Vertical source route points downward through the source footprint.
        data["routes"][0]["points"] = [data["routes"][0]["points"][0],
                                      {"x_um": 15.0, "y_um": 5.0},
                                      {"x_um": 35.0, "y_um": 5.0},
                                      data["routes"][0]["points"][-1]]
        self.assertEqual(validate_geometry(data)["status"], "invalid")
        data = copy.deepcopy(self.geometry)
        data["routes"][0].update(crossing_policy="declared_component", crossing_component_ids=["a"])
        self.assertEqual(validate_geometry(data)["status"], "invalid")
        # A crossing graph cannot silently inherit free planar wire crossings.
        with self.assertRaisesRegex(ValueError, "crossing"):
            grid_geometry(["a", "b", "c", "d"], [("a", "c"), ("b", "d")])

    def test_bus_explicit_bidirectionality_and_internal_junction_lengths(self):
        geometry = bus_geometry(["digital", "memory", "source"],
                                kinds={"digital": "compute", "memory": "storage", "source": "source"})
        result = validate_geometry(geometry)
        self.assertEqual(result["status"], "model_feasible", result)
        self.assertTrue(all(r["direction"] == "bidirectional" for r in result["routes"]))
        distances = bus_distances(geometry)
        # Two 9.5um taps +19um external spine +2um internal junction path.
        self.assertEqual(distances["digital"]["memory"], 40.0)
        self.assertEqual(distances["digital"]["source"], 60.0)
        self.assertEqual(distances["memory"]["digital"], 40.0)
        self.assertEqual(distances["digital"]["digital"], 0.0)
        changed = copy.deepcopy(geometry)
        changed["routes"][0]["direction"] = "directional"
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            bus_distances(changed)

    def test_bus_rejects_free_transit_through_unmodeled_compute_ports(self):
        data = grid_geometry(["a", "b", "c"], [("a", "b"), ("b", "c")])
        for route in data["routes"]:
            route["direction"] = "bidirectional"
        self.assertEqual(validate_geometry(data)["status"], "model_feasible")
        with self.assertRaisesRegex(ValueError, "declared shared_electrical_bus_v1"):
            bus_distances(data)
        # A model name alone cannot authorize arbitrary internal forwarding.
        data["routing_model"] = "shared_electrical_bus_v1"
        self.assertEqual(validate_geometry(data)["status"], "invalid")
        with self.assertRaises(ValueError):
            bus_distances(data)

    def test_bus_junction_layout_and_complete_topology_are_explicit(self):
        data = bus_geometry(["a", "b"])
        junction = next(p for p in data["placements"] if p["id"] == "bus_junction_0")
        south = next(p for p in junction["ports"] if p["id"] == "south")
        south["offset_um"] = 0.25
        tap = next(r for r in data["routes"] if r["id"] == "bus_tap_0")
        x = tap["points"][-1]["x_um"]
        y = tap["points"][-1]["y_um"]
        tap["points"] = [tap["points"][0], {"x_um": x, "y_um": y - 0.5},
                         {"x_um": x - 0.25, "y_um": y - 0.5}, {"x_um": x - 0.25, "y_um": y}]
        result = validate_geometry(data)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("centered", result["diagnostics"][0]["message"])
        with self.assertRaises(ValueError):
            bus_distances(data)
        for mutation in (lambda g: g["routes"].pop(),
                         lambda g: g["placements"][0]["ports"].append({"id": "hidden_transit", "side": "east", "offset_um": 5.0, "capacity": 1}),
                         lambda g: g["placements"][1].update(orientation=180)):
            changed = bus_geometry(["a", "b"])
            mutation(changed)
            self.assertEqual(validate_geometry(changed)["status"], "invalid")
            with self.assertRaises(ValueError):
                bus_distances(changed)

    def test_bus_footprints_and_all_required_support_placements(self):
        names = ["digital", "analog", "photonic", "converter", "memory", "link", "control", "source", "buffer"]
        geometry = bus_geometry(names, footprints={"digital": (20.0, 30.0)},
                                kinds={"converter": "interface", "memory": "storage", "control": "control", "source": "source", "buffer": "buffer"})
        self.assertTrue(set(names) <= {p["id"] for p in geometry["placements"]})
        self.assertEqual(validate_geometry(geometry)["status"], "model_feasible")
        self.assertGreater(bus_distances(geometry)["digital"]["analog"], 40.0)
        with self.assertRaises(ValueError):
            bus_geometry(["a", "a"])
        with self.assertRaises(ValueError):
            bus_geometry(["a"], footprints={"a": (0.0, 1.0)})

    def test_interregion_links_need_explicit_interface(self):
        geometry = grid_geometry(["a", "b"], [("a", "b")])
        # Split the default chip into touching, non-overlapping declared regions.
        geometry["regions"][0]["width_um"] = 25.0
        second = copy.deepcopy(geometry["regions"][0])
        second.update(id="other", x_um=25.0, width_um=35.0)
        geometry["regions"].append(second)
        geometry["placements"][1]["region"] = "other"
        self.assertEqual(validate_geometry(geometry)["status"], "invalid")
        geometry["routes"][0]["interface_id"] = "explicit_package_link"
        self.assertEqual(validate_geometry(geometry)["status"], "model_feasible")

    def test_geometry_work_bound_precedes_quadratic_route_comparisons(self):
        data = copy.deepcopy(self.geometry)
        data["routes"] = [dict(data["routes"][0], id=f"route{i}") for i in range(400)]
        result = validate_geometry(data)
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["diagnostics"][0]["code"], "geometry_work_limit")

    def test_numerical_extreme_geometry_does_not_report_infinite_length(self):
        data = grid_geometry(["a"])
        data["regions"][0].update(x_um=1.7e308, width_um=1.7e308)
        self.assertEqual(validate_geometry(data)["status"], "numerical_failure")
        data = grid_geometry(["a"])
        data["placements"][0].update(x_um=1e300, width_um=1.0)
        data["placements"][0]["ports"][0]["offset_um"] = 0.5
        self.assertEqual(validate_geometry(data)["status"], "numerical_failure")


if __name__ == "__main__":
    unittest.main()
