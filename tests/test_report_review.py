"""Independent review regressions for conservative report completeness labels.

These mutations test the presentation boundary rather than asking the renderer
to independently reproduce physics or certify a stored realization.
"""
from copy import deepcopy
import re
import unittest
import xml.etree.ElementTree as ET

from npp.implementation import realize
from npp.implementation_report import render_implementation
from npp.technology import load_technology


class ReportReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        network = {
            "schema_version": "0.1", "name": "Report review",
            "nodes": [{"id": "x", "op": "input", "size": 1,
                       "input_bounds": {"lower": [0.0], "upper": [1.0]}}],
            "outputs": ["x", "x"],
        }
        cls.record = realize(network, load_technology(), {
            "source_node": "x", "source_power_mw": 1.0, "lengths_um": 100.0})

    def assertIncomplete(self, record):
        report = render_implementation(record)
        self.assertFalse("Recorded feasible within the declared model" in report,
                         "Malformed record was labeled complete and feasible")
        self.assertTrue("Incomplete or unrecorded evaluation" in report)
        self.assertFalse("No failed guards were recorded" in report)
        self.assertTrue("Partial instance values" in report)

    def test_duplicate_ledger_rows_do_not_establish_instance_coverage(self):
        record = deepcopy(self.record)
        record["evaluation"]["ledger"] = [record["evaluation"]["ledger"][0]] * len(record["graph"]["instances"])
        self.assertIncomplete(record)

    def test_unknown_ledger_instance_is_incomplete(self):
        record = deepcopy(self.record)
        record["evaluation"]["ledger"][0]["instance"] = "nonexistent"
        self.assertIncomplete(record)

    def test_receiver_coverage_requires_exact_declared_ports(self):
        for mutation in ("unknown", "missing", "duplicate"):
            with self.subTest(mutation=mutation):
                record = deepcopy(self.record)
                receivers = record["evaluation"]["receivers"]
                if mutation == "unknown":
                    receivers[0]["port"] = {"instance": "nonexistent", "port": "out"}
                elif mutation == "missing":
                    receivers.pop()
                else:
                    receivers[1] = deepcopy(receivers[0])
                self.assertIncomplete(record)

    def test_metrics_require_real_finite_numbers_and_known_fields(self):
        for value in (True, float("nan"), float("inf"), "1.0", None):
            with self.subTest(value=value):
                record = deepcopy(self.record)
                record["evaluation"]["metrics"]["energy_pj"] = value
                self.assertIncomplete(record)
        record = deepcopy(self.record)
        record["evaluation"]["metrics"] = {"arbitrary": 0}
        self.assertIncomplete(record)

    def test_malformed_component_identifiers_remain_inspectable(self):
        record = deepcopy(self.record)
        record["technology"]["components"][0]["id"] = ["invalid identifier"]
        self.assertIncomplete(record)

    def test_regeneration_connections_do_not_pass_through_other_components(self):
        """A carrier path must not appear to connect through an early detector."""
        for fanout in (2, 4, 8):
            with self.subTest(fanout=fanout):
                network = deepcopy(self.record["network"])
                network["outputs"] = ["x"] * fanout
                record = realize(network, load_technology(), {
                    "source_node": "x", "source_power_mw": 1.0,
                    "recipe": "regenerate", "regeneration_power_mw": 0.5,
                    "lengths_um": 100.0})
                report = render_implementation(record)
                svg_text = re.search(r"<svg\b.*?</svg>", report, re.S)
                self.assertIsNotNone(svg_text, "Supported fanout must retain its circuit diagram")
                svg = ET.fromstring(svg_text.group())
                ns = {"s": "http://www.w3.org/2000/svg"}
                rectangles = []
                for group in svg.findall("s:g", ns):
                    rect = group.find("s:rect", ns)
                    if rect is not None:
                        rectangles.append((group.get("data-instance"),
                                           *(float(rect.get(k)) for k in ("x", "y", "width", "height"))))
                for path in svg.findall("s:path", ns):
                    # Current renderer uses one cubic Bezier per connection.
                    coords = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path.get("d", ""))]
                    if len(coords) != 8 or "C" not in path.get("d", ""):
                        continue
                    points = list(zip(coords[::2], coords[1::2]))
                    for identifier, x, y, width, height in rectangles:
                        for step in range(1, 100):
                            t = step / 100
                            bx, by = [sum(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3)[i]*points[i][axis]
                                          for i in range(4)) for axis in (0, 1)]
                            self.assertFalse(x+2 < bx < x+width-2 and y+2 < by < y+height-2,
                                             f"Connection {path.findtext('s:title', namespaces=ns)} crosses component {identifier}")


if __name__ == "__main__":
    unittest.main()
