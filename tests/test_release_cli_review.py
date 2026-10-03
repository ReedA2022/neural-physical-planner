"""Independent release checks for exported extremes and malformed saved records."""
import contextlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from npp.cli import main
from tests.fixtures import component, library, network, node, request


class ReleaseCLIReviewTests(unittest.TestCase):
    def invoke(self, arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(arguments)
        return code, json.loads(output.getvalue())

    def test_supported_extreme_costs_export_and_replay_without_overflow(self):
        for value in (0.0, 5e-324, 1e-200, 1.0, 1e306, sys.float_info.max):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = []
                payloads = {
                    'network': network([node('x', 'input')], ['x']),
                    'library': library([component('source', ['input'], energy_pj=value,
                                                   latency_ns=value, area_um2=value)]),
                    'request': request(),
                }
                for name, payload in payloads.items():
                    path = root / f'{name}.json'
                    path.write_text(json.dumps(payload))
                    args.extend([f'--{name}', str(path)])
                directory = root / 'output'
                code, result = self.invoke(['plan', *args, '--out-dir', str(directory)])
                self.assertEqual(code, 0)
                self.assertTrue(result['all_saved_plans_checked'])
                manifest = json.loads((directory / 'manifest.json').read_text())
                for relative in manifest['files'] + manifest['plans']:
                    self.assertTrue((directory / relative).is_file(), relative)
                record = json.loads((directory / manifest['plans'][0]).read_text())
                for metric in ('energy_pj', 'latency_ns', 'area_um2'):
                    self.assertEqual(record['evaluation']['metrics'][metric], value)
                code, checked = self.invoke(['check', *args, '--plan', str(directory / manifest['plans'][0])])
                self.assertEqual(code, 0)
                self.assertTrue(checked['valid'])
                document = (directory / 'report.html').read_text()
                projection = ET.fromstring(re.search(r'<svg.*?</svg>', document, re.S)[0])
                self.assertEqual(len(projection.findall('.//circle')), 1)
                for circle in projection.findall('.//circle'):
                    self.assertTrue(math.isfinite(float(circle.attrib['cx'])))
                    self.assertTrue(math.isfinite(float(circle.attrib['cy'])))

    def test_malformed_saved_json_has_structured_errors_for_every_record_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'invalid.json'
            for source in ('[]', '{', '{"v":NaN}', '{"v":1,"v":2}', '{"v":Infinity}'):
                path.write_text(source)
                for args in (
                    ['check-realization', '--realization', str(path)],
                    ['check-physical-design', '--result', str(path)],
                    ['export-optical', '--realization', str(path), '--out', str(root / 'netlist.json')],
                    ['simulate-realization', '--realization', str(path), '--out-dir', str(root / 'simulation')],
                    ['replan-physical', '--baseline', str(path), '--design', 'unused.json', '--out-dir', str(root / 'replan')],
                ):
                    with self.subTest(source=source, command=args[0]):
                        code, result = self.invoke(args)
                        self.assertEqual(code, 2)
                        self.assertEqual(result['status'], 'invalid_input')
                        self.assertTrue(result['diagnostics'])
                self.assertFalse((root / 'netlist.json').exists())
                self.assertFalse((root / 'simulation').exists())
                self.assertFalse((root / 'replan').exists())


if __name__ == '__main__':
    unittest.main()
