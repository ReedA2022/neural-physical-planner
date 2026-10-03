"""Inspection checks for the offline physical-circuit report."""
from copy import deepcopy
from html.parser import HTMLParser
import unittest

from npp.implementation_report import _svg, render_implementation
from npp.implementation import realize
from npp.technology import load_technology


class _Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tags = []
        self.table_rows = {}
        self.table_id = None
        self.in_body = False
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "table":
            self.table_id = attrs.get("id")
        elif tag == "tbody":
            self.in_body = True
        elif tag == "tr" and self.in_body and self.table_id:
            self.table_rows[self.table_id] = self.table_rows.get(self.table_id, 0) + 1

    def handle_endtag(self, tag):
        if tag == "table":
            self.table_id = None
        elif tag == "tbody":
            self.in_body = False


class CircuitDrawingTests(unittest.TestCase):
    def test_includes_sources_terminations_and_exact_ports(self):
        nodes = [{"id": "source", "kind": "source"}, {"id": "sink", "kind": "termination"}]
        edges = [{"source": {"instance": "source", "port": "out"},
                  "target": {"instance": "sink", "port": "in"}, "domain": "optical"}]
        drawing = _svg(nodes, edges)
        parsed = _Elements(drawing)
        self.assertEqual({attrs["data-instance"] for tag, attrs in parsed.tags if "data-instance" in attrs}, {"source", "sink"})
        self.assertIn("source.out → sink.in (optical)", drawing)

    def test_large_or_broken_graph_has_explicit_table_fallback(self):
        cases = [([{ "id": str(i)} for i in range(97)], [], "large circuit"),
                 ([{"id": "same"}, {"id": "same"}], [], "duplicated"),
                 ([{"id": "a"}], [{"source": {"instance": "a"}, "target": {"instance": "missing"}}], "unknown instance"),
                 ([{"id": "a"}], [{"source": {"instance": "a"}, "target": {"instance": "a"}}], "cycle")]
        for nodes, edges, message in cases:
            with self.subTest(message=message):
                drawing = _svg(nodes, edges)
                self.assertNotIn("<svg", drawing)
                self.assertIn(message, drawing)

    def test_diagram_escapes_identifiers_and_kinds(self):
        nodes = [{"id": '<script>alert("x")</script>', "kind": '<img src=x onerror="x">'}]
        snapshot = deepcopy(nodes)
        drawing = _svg(nodes, [])
        self.assertNotIn("<script", drawing)
        self.assertNotIn("<img", drawing)
        self.assertIn("&lt;script&gt;", drawing)
        self.assertEqual(nodes, snapshot)


def _record(fanout=3, *, recipe="passive", power=1.0):
    network = {"schema_version": "0.1", "name": "Scalar fanout fixture", "nodes": [
        {"id": "x", "op": "input", "size": 1,
         "input_bounds": {"lower": [0.0], "upper": [1.0]}}
    ] + [{"id": f"use_{index}", "op": "identity", "size": 1, "inputs": ["x"]}
         for index in range(fanout)], "outputs": [f"use_{index}" for index in range(fanout)]}
    spec = {"source_node": "x", "recipe": recipe, "source_power_mw": power,
            "lengths_um": 1000.0}
    if recipe == "regenerate":
        spec["regeneration_power_mw"] = 0.5
    return realize(network, load_technology(), spec)


class ImplementationReportTests(unittest.TestCase):
    def test_realization_counts_ports_terminations_and_ledger(self):
        record = _record(3)
        before = deepcopy(record)
        report = render_implementation(record)
        parsed = _Elements(report)
        for table, values in [("instances", record["graph"]["instances"]),
                              ("connections", record["graph"]["connections"]),
                              ("receivers", record["graph"]["receivers"]),
                              ("resource-ledger", record["evaluation"]["ledger"]),
                              ("terminations", record["graph"]["terminations"])]:
            self.assertEqual(parsed.table_rows[table], len(values), table)
        self.assertEqual(sum("data-instance" in attrs for tag, attrs in parsed.tags),
                         len(record["graph"]["instances"]) + len(record["graph"]["terminations"]))
        for text in ("Recorded feasible within the declared model", "power_mw", "normalized_sample",
                     "encoded_launch", "ideal_matched_external_boundary", "unreviewed", "illustrative",
                     "Rendering does not check its hashes", "unused_splitter_leaf"):
            self.assertIn(text, report)
        self.assertEqual(record, before)

    def test_passive_and_regenerated_fanout_2_4_8_have_readable_svg(self):
        for fanout in (2, 4, 8):
            for recipe in ("passive", "regenerate"):
                with self.subTest(fanout=fanout, recipe=recipe):
                    record = _record(fanout, recipe=recipe)
                    report = render_implementation(record)
                    parsed = _Elements(report)
                    self.assertTrue(any(tag == "svg" for tag, attrs in parsed.tags))
                    self.assertEqual(sum("data-instance" in attrs for tag, attrs in parsed.tags),
                                     len(record["graph"]["instances"]))
                    if recipe == "regenerate":
                        self.assertIn("constant_carrier", report)
                        self.assertIn('stroke-dasharray="6 4"', report)

    def test_infeasible_record_retains_failures_without_feasible_totals(self):
        record = _record(3, power=0.0001)
        self.assertFalse(record["evaluation"]["feasible"])
        report = render_implementation(record)
        parsed = _Elements(report)
        self.assertIn("Recorded infeasible", report)
        self.assertNotIn("Recorded feasible within the declared model", report)
        self.assertNotIn("metrics", parsed.table_rows)
        self.assertIn("Partial instance values", report)
        self.assertGreater(parsed.table_rows["diagnostics"], 0)
        self.assertEqual(parsed.table_rows["resource-ledger"], len(record["graph"]["instances"]))

    def test_partial_record_is_not_promoted_to_success(self):
        record = _record()
        record["evaluation"]["ledger"] = record["evaluation"]["ledger"][:1]
        report = render_implementation(record)
        self.assertIn("Incomplete or unrecorded evaluation", report)
        self.assertNotIn("Recorded feasible within the declared model", report)
        self.assertNotIn("No failed guards were recorded", report)
        self.assertIn("Incomplete or unrecorded evaluation", render_implementation({}))

    def test_all_user_content_is_inert_offline_text(self):
        record = _record()
        payload = '<script>alert("unsafe")</script><img src="https://invalid.test/tracker" onerror="alert(1)">'
        record["network"]["name"] = payload
        record["technology"]["review"]["notes"] = payload
        record["technology"]["sources"][0]["citation"] = payload
        record["technology"]["sources"][0]["url"] = 'javascript:alert(1)'
        report = render_implementation(record)
        parsed = _Elements(report)
        self.assertFalse(any(tag in ("script", "img", "iframe", "link") for tag, attrs in parsed.tags))
        self.assertFalse(any(key.lower().startswith("on") for tag, attrs in parsed.tags for key in attrs))
        self.assertFalse(any("src" in attrs or "href" in attrs for tag, attrs in parsed.tags))
        self.assertIn("&lt;script&gt;", report)
        self.assertIn("javascript:alert(1)", report)  # plain evidence text, never a link


if __name__ == "__main__":
    unittest.main()
