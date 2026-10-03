"""Usable comparison reports gated by actual bounded design-search replay."""
from copy import deepcopy
from html.parser import HTMLParser
import unittest
from unittest.mock import patch

from npp.designer import plan_physical
from npp.designer_report import render_physical_design
from npp.implementation import realize
from npp.technology import load_technology


def inputs():
    network = {'schema_version': '0.1', 'name': 'Designer report fixture',
               'nodes': [{'id': 'x', 'op': 'input', 'size': 1,
                          'input_bounds': {'lower': [0.0], 'upper': [1.0]}}], 'outputs': ['x'] * 4}
    design = {'source_node': 'x', 'grid': {'recipes': ['passive', 'regenerate'],
              'source_powers_mw': [0.05, 0.1], 'regeneration_powers_mw': [0.1],
              'route_lengths_um': [10000.0]}}
    return network, load_technology(), design


def result_with(**updates):
    network, technology, design = inputs()
    design.update(updates)
    return plan_physical(network, technology, design)


class Elements(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.tags, self.rows = [], {}
        self.table, self.in_body = None, False
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == 'table':
            self.table = attrs.get('id')
        elif tag == 'tbody':
            self.in_body = True
        elif tag == 'tr' and self.table and self.in_body:
            self.rows[self.table] = self.rows.get(self.table, 0) + 1

    def handle_endtag(self, tag):
        if tag == 'table':
            self.table = None
        elif tag == 'tbody':
            self.in_body = False

    @property
    def links(self):
        return [attrs['href'] for tag, attrs in self.tags if 'href' in attrs]


class DesignerReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = result_with()

    def assertUnverified(self, result):
        report = render_physical_design(result)
        self.assertIn('Incomplete, inconsistent or unsupported design record', report)
        self.assertNotIn('Nominal design search replay passed', report)
        self.assertFalse(Elements(report).links)
        self.assertFalse(any(tag == 'svg' for tag, attrs in Elements(report).tags))
        return report

    def test_real_grid_counts_all_axes_rejections_and_retained_links(self):
        result = deepcopy(self.result)
        before = deepcopy(result)
        report = render_physical_design(result)
        parsed = Elements(report)
        self.assertIn('Nominal design search replay passed', report)
        self.assertIn('Complete for the submitted finite grid.', report)
        self.assertEqual(parsed.rows['candidate-comparison'], len(result['candidates']))
        self.assertEqual(parsed.rows['rejection-overview'], sum(len(row['reasons']) for row in result['candidates']))
        self.assertCountEqual(parsed.links, [f"plans/{plan['id']}/report.html" for plan in result['plans']])
        for text in ('Energy (pJ/symbol)', 'Latency (ns)', 'Area (µm²)', 'Normalized noise RMS estimate',
                     'unreviewed', 'illustrative', 'partial ledger', 'not whole-circuit totals'):
            self.assertIn(text, report)
        self.assertEqual(result, before)

    def test_constraint_rejections_show_actual_paths_and_witnesses(self):
        result = result_with(constraints={'max_energy_pj': 1.0})
        self.assertTrue(result['single_constraint_relaxations'])
        report = render_physical_design(result)
        parsed = Elements(report)
        self.assertIn('constraints.max_energy_pj', report)
        self.assertIn('constraint_exceeded', report)
        self.assertIn('required_relaxation', report)
        self.assertEqual(parsed.rows['single-constraint-relaxations'], len(result['single_constraint_relaxations']))
        self.assertEqual(parsed.rows['rejection-overview'], sum(len(row['reasons']) for row in result['candidates']))

    def test_baseline_locks_forbidden_choices_and_graph_changes(self):
        network, technology, design = inputs()
        baseline = realize(network, technology, {'source_node': 'x', 'source_power_mw': 0.1,
                                                'lengths_um': 10000.0})
        design['locks'] = {'source_power_mw': 'baseline'}
        result = plan_physical(network, technology, design, baseline=baseline)
        report = render_physical_design(result)
        self.assertIn('Nominal design search replay passed', report)
        self.assertIn('locks.source_power_mw', report)
        self.assertIn('lock_mismatch', report)
        parsed = Elements(report)
        for index, candidate in enumerate(result['candidates']):
            comparison = candidate['baseline_comparison']
            if comparison and comparison['changed_instances']:
                self.assertEqual(parsed.rows[f'changed-instances-{index}'], len(comparison['changed_instances']))
        self.assertIn('metric_deltas', report)
        design['forbidden_recipes'] = ['regenerate']
        forbidden = plan_physical(network, technology, design, baseline=baseline)
        report = render_physical_design(forbidden)
        self.assertIn('forbidden_recipes', report)
        self.assertIn('forbidden_recipe', report)

    def test_incomplete_search_does_not_claim_global_optimality(self):
        result = result_with(max_evaluations=2)
        self.assertFalse(result['search']['complete'])
        self.assertGreater(result['search']['not_evaluated_count'], 0)
        report = render_physical_design(result)
        self.assertIn('Incomplete search of the submitted finite grid.', report)
        self.assertNotIn('Complete for the submitted finite grid.', report)
        self.assertIn('evaluation_limit', report)
        self.assertIn('evaluated feasible candidates', report)
        self.assertIn('not a global optimum', report)

    def test_no_feasible_candidate_does_not_claim_hardware_impossibility(self):
        result = result_with(constraints={'max_energy_pj': 0.0})
        report = render_physical_design(result)
        self.assertIn('No feasible candidate in the evaluated search', report)
        self.assertNotIn('Nominal design search replay passed', report)
        self.assertFalse(Elements(report).links)
        self.assertIn('does not establish that every possible hardware implementation is infeasible', report)

    def test_coordinated_candidate_omissions_and_claim_tampering_fail_replay(self):
        for mutation in ('omitted', 'metrics', 'pareto', 'reasons', 'schema', 'identifier'):
            result = deepcopy(self.result)
            if mutation == 'omitted':
                result['candidates'].pop()
                result['search']['submitted_grid_size'] -= 1
                result['search']['evaluated_count'] -= 1
            elif mutation == 'metrics':
                result['candidates'][1]['metrics']['energy_pj'] = 0.0
                if result['plans']:
                    result['plans'][0]['realization']['evaluation']['metrics']['energy_pj'] = 0.0
            elif mutation == 'pareto':
                result['candidates'][1]['pareto'] = not result['candidates'][1]['pareto']
                result['plans'] = []
                result['search']['pareto_count'] = 0
            elif mutation == 'reasons':
                result['candidates'][0]['reasons'] = []
            elif mutation == 'schema':
                result['schema_version'] = '999'
            else:
                result['plans'][0]['id'] = '../outside'
            with self.subTest(mutation=mutation):
                self.assertUnverified(result)

    def test_malformed_records_remain_inspectable(self):
        for value in (None, [], {}, {'candidates': [None, {'id': ['bad'], 'metrics': [False]}]}):
            with self.subTest(value=value):
                self.assertUnverified(value)

    def test_out_of_bounds_replay_is_rejected_before_candidate_physics(self):
        result = deepcopy(self.result)
        result['inputs']['design']['max_evaluations'] = 10**6
        with patch('npp.designer.realize', side_effect=AssertionError('physics must not start')) as evaluate:
            self.assertUnverified(result)
            evaluate.assert_not_called()

    def test_report_replay_does_not_invoke_external_optical_engine(self):
        with patch('npp.optical_simulation._load_sax', side_effect=AssertionError('external solver must not start')) as external:
            report = render_physical_design(self.result)
            self.assertIn('Nominal design search replay passed', report)
            external.assert_not_called()
        self.assertIn('does not run an external SAX simulation', report)

    def test_valid_hostile_labels_are_escaped_and_links_stay_local(self):
        network, technology, design = inputs()
        payload = '<script>alert("x")</script><img src="https://invalid.test/a" onerror="bad()">'
        network['name'] = payload
        technology['review']['notes'] += payload
        result = plan_physical(network, technology, design)
        report = render_physical_design(result)
        parsed = Elements(report)
        self.assertIn('Nominal design search replay passed', report)
        self.assertIn('&lt;script&gt;', report)
        self.assertFalse(any(tag in ('script', 'img', 'iframe', 'link') for tag, attrs in parsed.tags))
        self.assertFalse(any(key.startswith('on') or key == 'src' for tag, attrs in parsed.tags for key in attrs))
        self.assertTrue(all(link.startswith('plans/candidate-') and link.endswith('/report.html') for link in parsed.links))


if __name__ == '__main__':
    unittest.main()
