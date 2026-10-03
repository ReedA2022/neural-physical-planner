"""Release regressions for supported extreme costs and malformed saved inputs."""
import contextlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from npp.cli import main
from npp.models import validate_inputs
from npp.planner import plan
from npp.reports import _scatter
from tests.fixtures import component, library, network, node, request


class ReportStressTests(unittest.TestCase):
    def test_extreme_finite_costs_have_finite_visible_coordinates(self):
        for value in (0.0, 5e-324, 1e-200, 1.0, 1e306, sys.float_info.max):
            with self.subTest(value=value):
                n, l, r = validate_inputs(
                    network([node("x", "input")], ["x"]),
                    library([component("unit", ["input"], energy_pj=value, latency_ns=value)]),
                    request())
                report = plan(n, l, r)
                self.assertEqual(report["status"], "ok")
                html = _scatter(report["plans"])
                svg = ET.fromstring(re.search(r"<svg.*?</svg>", html, re.S)[0])
                circles = svg.findall(".//circle")
                self.assertEqual(len(circles), 1)
                for circle in circles:
                    x, y = float(circle.attrib["cx"]), float(circle.attrib["cy"])
                    self.assertTrue(math.isfinite(x) and 72 <= x <= 594)
                    self.assertTrue(math.isfinite(y) and 20 <= y <= 246)
                for text in svg.findall(".//text"):
                    self.assertNotIn(text.text.strip().lower(), ("inf", "-inf", "nan", "infinity"))

    def test_deep_saved_json_returns_a_structured_cli_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deep.json"
            path.write_text('{"x":' + '[' * 1500 + '0' + ']' * 1500 + '}')
            output = io.StringIO()
            original_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(500)
                with contextlib.redirect_stdout(output):
                    code = main(["check-realization", "--realization", str(path)])
            finally:
                sys.setrecursionlimit(original_limit)
            self.assertEqual(code, 2)
            result = json.loads(output.getvalue())
            self.assertTrue(result.get("status") == "invalid_input" or result.get("valid") is False)
            self.assertTrue(result["diagnostics"])

    def test_decoder_recursion_failure_is_structured_at_cli_boundary(self):
        # Python JSON decoder implementations differ in when they reject depth.
        # Exercise the error contract independently of the installed decoder.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "saved.json"
            path.write_text('{}')
            output = io.StringIO()
            with patch("npp.models.json.load", side_effect=RecursionError("input nesting limit")):
                with contextlib.redirect_stdout(output):
                    code = main(["check-realization", "--realization", str(path)])
            self.assertEqual(code, 2)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "invalid_input")
            self.assertEqual(result["diagnostics"][0]["code"], "json_load_error")


if __name__ == "__main__":
    unittest.main()
