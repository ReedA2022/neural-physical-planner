"""Presentation coverage and conservative claims for optical comparison reports."""
from copy import deepcopy
from html.parser import HTMLParser
import importlib.util
import unittest

from npp.implementation import realize
from npp.optical_report import render_optical_simulation
from npp.optical_simulation import export_optical_netlist, simulate_realization
from npp.technology import load_technology


HAS_SAX = importlib.util.find_spec('sax') is not None


def realization(recipe='passive'):
    network = {'schema_version': '0.1', 'name': 'Optical report test',
               'nodes': [{'id': 'x', 'op': 'input', 'size': 1,
                          'input_bounds': {'lower': [0.0], 'upper': [1.0]}}],
               'outputs': ['x', 'x', 'x']}
    spec = {'source_node': 'x', 'source_power_mw': 1.0, 'lengths_um': [0.0, 1000.0, 2000.0],
            'recipe': recipe}
    if recipe == 'regenerate':
        spec['regeneration_power_mw'] = 0.5
    return realize(network, load_technology(), spec)


def presentation_fixture():
    """Synthetic stored-result fixture; this function does not claim a SAX run."""
    record = realization()
    exported = export_optical_netlist(record)
    ledger = {row['instance']: row for row in record['evaluation']['ledger']}
    comparisons = []
    for segment in exported['segments']:
        for output in segment['outputs']:
            endpoint = output['endpoint']
            row = ledger[endpoint['instance']]
            power = (row['output_powers_mw'][endpoint['port']] if output['kind'] == 'matched_termination'
                     else row['input_power_mw'])
            comparisons.append({'segment': segment['id'], 'source': segment['source']['endpoint'],
                                'sax_port': output['sax_port'], 'endpoint': endpoint, 'kind': output['kind'],
                                'binding': output['binding'], 'expected_power_mw': power, 'sax_power_mw': power,
                                'absolute_error_mw': 0.0, 'relative_error': 0.0,
                                'allowed_error_mw': 1e-9 + 1e-5 * power, 'passed': True})
    return {'schema_version': '0.3', 'kind': 'sax_optical_comparison', 'status': 'passed', 'passed': True,
            'engine': {'name': 'SAX', 'version': 'synthetic-fixture-no-engine-run', 'backend': 'klu',
                       'jax_version': 'synthetic-fixture', 'adapter_version': exported['adapter_version']},
            'tolerances': {'rtol': 1e-5, 'atol_mw': 1e-9,
                           'rule': 'absolute_error_mw <= atol_mw + rtol*abs(expected_power_mw)'},
            'scope': deepcopy(exported['scope']), 'export': exported,
            'samples': [{'wavelength_nm': 1550.0, 'comparisons': comparisons, 'passed': True}],
            'summary': {'segment_count': len(exported['segments']), 'wavelength_count': 1,
                        'comparison_count': len(comparisons), 'max_absolute_error_mw': 0.0, 'max_relative_error': 0.0},
            'diagnostics': []}


class Elements(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.tags = []
        self.table = None
        self.body = False
        self.rows = {}
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == 'table':
            self.table = attrs.get('id')
        elif tag == 'tbody':
            self.body = True
        elif tag == 'tr' and self.table and self.body:
            self.rows[self.table] = self.rows.get(self.table, 0) + 1

    def handle_endtag(self, tag):
        if tag == 'table':
            self.table = None
        elif tag == 'tbody':
            self.body = False


class OpticalReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = presentation_fixture()

    def assertIncomplete(self, result):
        report = render_optical_simulation(result)
        self.assertIn('Incomplete or inconsistent comparison record', report)
        self.assertNotIn('Recorded optical-power agreement within tolerance', report)
        return report

    def test_complete_boundary_table_scope_versions_and_no_mutation(self):
        result = deepcopy(self.fixture)
        snapshot = deepcopy(result)
        report = render_optical_simulation(result)
        parsed = Elements(report)
        self.assertIn('Recorded optical-power agreement within tolerance', report)
        self.assertEqual(parsed.rows['comparisons-0'], len(result['samples'][0]['comparisons']))
        self.assertEqual(parsed.rows['segments'], len(result['export']['segments']))
        for text in ('synthetic-fixture-no-engine-run', 'expected_power_mw', 'sax_power_mw',
                     'matched_termination', 'receiver_input', 'relative_error', 'partial_optical_only',
                     'unreviewed', 'illustrative', 'not experimental ground truth'):
            self.assertIn(text, report)
        self.assertEqual(result, snapshot)

    @unittest.skipUnless(HAS_SAX, 'SAX optional dependency is not installed')
    def test_actual_external_engine_sweep_is_rendered_consistently(self):
        result = simulate_realization(realization('regenerate'), [1540.0, 1550.0, 1560.0])
        report = render_optical_simulation(result)
        self.assertTrue(result['passed'])
        self.assertIn('Recorded optical-power agreement within tolerance', report)
        parsed = Elements(report)
        self.assertEqual(sum(value for key, value in parsed.rows.items() if key.startswith('comparisons-')),
                         result['summary']['comparison_count'])
        self.assertIn('regeneration_detector_input', report)
        self.assertIn('Excluded', report.title())

    def test_unicode_technology_uses_the_physical_realization_hash_contract(self):
        original = realization()
        technology = deepcopy(original['technology'])
        technology['name'] = 'Technology µ — 測定'
        physical = realize(original['network'], technology, original['spec'])
        result = presentation_fixture()
        result['export'] = export_optical_netlist(physical)
        report = render_optical_simulation(result)
        self.assertIn('Recorded optical-power agreement within tolerance', report)
        self.assertIn('Technology µ — 測定', report)

    def test_boundary_coverage_requires_unique_declared_outputs(self):
        for mutation in ('missing', 'duplicate', 'unknown', 'misbound'):
            with self.subTest(mutation=mutation):
                result = deepcopy(self.fixture)
                rows = result['samples'][0]['comparisons']
                if mutation == 'missing':
                    rows.pop()
                elif mutation == 'duplicate':
                    rows[1] = deepcopy(rows[0])
                elif mutation == 'unknown':
                    rows[0]['sax_port'] = 'not_exported'
                else:
                    rows[0]['endpoint'] = {'instance': 'wrong_detector', 'port': 'in'}
                self.assertIncomplete(result)

    def test_inconsistent_comparison_arithmetic_cannot_claim_agreement(self):
        for mutation in ('power', 'allowed', 'flag', 'summary', 'sample'):
            with self.subTest(mutation=mutation):
                result = deepcopy(self.fixture)
                row = result['samples'][0]['comparisons'][0]
                if mutation == 'power':
                    row['sax_power_mw'] *= 2.0
                elif mutation == 'allowed':
                    row['allowed_error_mw'] = 999.0
                elif mutation == 'flag':
                    row['passed'] = False
                elif mutation == 'summary':
                    result['summary']['comparison_count'] += 1
                else:
                    result['samples'][0]['passed'] = False
                self.assertIncomplete(result)

    def test_nonfinite_and_boolean_numbers_missing_tolerances_or_versions_are_incomplete(self):
        for value in (True, float('nan'), float('inf'), '0.1', None):
            with self.subTest(value=value):
                result = deepcopy(self.fixture)
                result['samples'][0]['comparisons'][0]['expected_power_mw'] = value
                self.assertIncomplete(result)
                result = deepcopy(self.fixture)
                result['tolerances']['atol_mw'] = value
                self.assertIncomplete(result)
        result = deepcopy(self.fixture)
        result['engine']['version'] = None
        self.assertIncomplete(result)
        result = deepcopy(self.fixture)
        result['summary']['wavelength_count'] = True
        self.assertIncomplete(result)

    def test_malformed_identifiers_empty_records_and_duplicate_wavelengths_do_not_crash(self):
        result = deepcopy(self.fixture)
        result['export']['segments'][0]['id'] = ['unhashable']
        self.assertIncomplete(result)
        result = deepcopy(self.fixture)
        result['samples'][0]['comparisons'][0]['sax_port'] = ['unhashable']
        self.assertIncomplete(result)
        result = deepcopy(self.fixture)
        result['samples'].append(deepcopy(result['samples'][0]))
        self.assertIncomplete(result)
        for result in ({}, {'passed': True}, {'samples': [None], 'export': {'segments': [None]}}):
            self.assertIncomplete(result)

    def test_honest_disagreement_is_visible(self):
        result = deepcopy(self.fixture)
        row = result['samples'][0]['comparisons'][0]
        row['sax_power_mw'] *= 2.0
        row['absolute_error_mw'] = abs(row['sax_power_mw'] - row['expected_power_mw'])
        row['relative_error'] = row['absolute_error_mw'] / row['expected_power_mw']
        row['passed'] = False
        result['samples'][0]['passed'] = False
        result['passed'], result['status'] = False, 'failed'
        result['summary']['max_absolute_error_mw'] = row['absolute_error_mw']
        result['summary']['max_relative_error'] = row['relative_error']
        result['diagnostics'] = [{'code': 'optical_power_mismatch', 'path': 'fixture', 'message': 'Deliberate test mismatch'}]
        report = render_optical_simulation(result)
        self.assertIn('Recorded optical-power disagreement', report)
        self.assertNotIn('Recorded optical-power agreement within tolerance', report)
        self.assertIn('Deliberate test mismatch', report)

    def test_hostile_content_remains_inert_without_network_assets(self):
        result = deepcopy(self.fixture)
        payload = '<script>alert("x")</script><img src="https://invalid.test/tracker" onerror="bad()">'
        result['engine']['version'] = payload
        result['scope']['unchecked'].append(payload)
        result['diagnostics'].append({'code': payload, 'path': payload, 'message': payload})
        report = render_optical_simulation(result)
        parsed = Elements(report)
        self.assertIn('&lt;script&gt;', report)
        self.assertFalse(any(tag in ('script', 'img', 'link', 'iframe') for tag, attrs in parsed.tags))
        self.assertFalse(any(key.startswith('on') or key in ('src', 'href') for tag, attrs in parsed.tags for key in attrs))


if __name__ == '__main__':
    unittest.main()
