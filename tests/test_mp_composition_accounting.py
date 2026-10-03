"""Integration regressions from the rewrite author's accounting audit.

This is integration support, not the independent milestone gate review.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from npp.multiphysics.graph import default_implementation, evaluate_implementation
from npp.multiphysics.composition_validation import (
    validate_graph_structure, validate_implementation_replay_inputs,
)
from npp.multiphysics.io import init_project, load_project
from npp.multiphysics.semantic import NumericPolicy, evaluate_ffn
from tests.test_mp_semantic import decimal_oracle


class CompositionAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as folder:
            init_project(folder)
            cls.project = load_project(Path(folder) / 'project.yaml')
            cls.project.pop('input_hash', None)

    def test_complete_digital_baseline_is_five_real_operators(self):
        for fmt in ('float64', 'float32'):
            p = deepcopy(self.project)
            policy = NumericPolicy(format=fmt)
            p['workload']['numeric_policy'] = policy.to_dict()
            result = evaluate_implementation(p)
            actual = evaluate_ffn(p['model'], p['workload']['inputs'], policy)
            self.assertEqual(result['output'], actual['output'])
            self.assertEqual(len(result['backend_evaluations']), 5)
            kinds = [r['backend_id'] for r in result['backend_evaluations'].values()]
            self.assertEqual(kinds.count('digital.linear'), 3)
            self.assertEqual(kinds.count('digital.silu'), 1)
            self.assertEqual(kinds.count('digital.multiply'), 1)

    def test_full_two_dimensional_partition_preserves_bias_exactly_once(self):
        p = deepcopy(self.project)
        p['model'].update(b_up=[.4, -.3], b_gate=[-.2, .1], b_down=[.7, -.6])
        spec = {'schema_version': 'npp-implementation-1', 'operators': {
            name: {'family': 'digital', 'input_cuts': [0, 1, 2], 'output_cuts': [0, 1, 2]}
            for name in ('up', 'gate', 'down')}}
        result = evaluate_implementation(p, spec)
        oracle = [decimal_oracle(p['model'], x) for x in p['workload']['inputs']]
        np.testing.assert_allclose(result['output'], oracle, rtol=1e-14, atol=1e-14)
        merges = [node for node in result['physical_graph']['nodes'] if node['id'].endswith('.bias')]
        self.assertEqual(len(merges), 3)
        self.assertGreater(len(result['backend_evaluations']), 5)

    def test_explicit_fifo_bandwidth_changes_scheduled_latency(self):
        p = deepcopy(self.project)
        ordinary = evaluate_implementation(p)
        spec = default_implementation(p['model'])
        spec['system'] = {'preset': 'hypothetical_system', 'overrides': {
            'electronic_buffer_read_bytes_per_ns': .001,
            'electronic_buffer_write_bytes_per_ns': .001}}
        slower = evaluate_implementation(p, spec)
        self.assertEqual(ordinary['output'], slower['output'])
        self.assertGreater(slower['schedule']['makespan_ns'], ordinary['schedule']['makespan_ns'] + 1000)
        self.assertGreater(slower['ledger']['energy_pj']['value'], ordinary['ledger']['energy_pj']['value'])

    def test_declared_pack_matrix_and_signal_envelopes_are_enforced(self):
        for kind in ('rows', 'input_range', 'output_range'):
            p = deepcopy(self.project)
            component = next(c for c in p['technology']['components'] if c['family'] == 'digital')
            if kind == 'rows':
                for axis in component['operating_envelope']:
                    if axis['name'] == 'rows':
                        axis['maximum'] = 1.
            else:
                port_name = 'input' if kind == 'input_range' else 'output'
                port = next(port for port in component['ports'] if port['name'] == port_name)
                port['signal']['value_range'] = {'minimum': -.001, 'maximum': .001}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evaluate_implementation(p)

    def test_unsupported_tile_region_cannot_be_ignored(self):
        p = deepcopy(self.project)
        spec = default_implementation(p['model'])
        spec['operators']['up']['tiles'][0]['region'] = 'absent_cryogenic_region'
        with self.assertRaises(ValueError):
            evaluate_implementation(p, spec)

    def test_evidence_policy_covers_system_support(self):
        p = deepcopy(self.project)
        p['workload']['evidence_policy'] = ['source_parameterized']
        # User-supplied evidence labels for local components cannot whitelist the
        # separate still-hypothetical clock/package/FIFO scenario coefficients.
        for component in p['technology']['components']:
            for parameter in component['parameters']:
                parameter['evidence']['kind'] = 'source_parameterized'
        with self.assertRaises(ValueError):
            evaluate_implementation(p)

    def test_unknown_system_support_keeps_totals_conditional(self):
        p = deepcopy(self.project)
        spec = default_implementation(p['model'])
        spec['system'] = {'preset': 'hypothetical_system', 'overrides': {
            'package_static_power_mw': {'state': 'unavailable', 'unit': 'mW', 'reason': 'uncharacterized package'}}}
        result = evaluate_implementation(p, spec)
        self.assertEqual(result['status'], 'conditional')
        self.assertFalse(result['ledger']['complete'])
        self.assertEqual(result['ledger']['energy_pj']['state'], 'unavailable')

    def test_repeated_invocations_pay_declared_cold_programming_each_time(self):
        p = deepcopy(self.project)
        once = evaluate_implementation(p)
        p['workload']['reuse_count'] = 3
        repeated = evaluate_implementation(p)
        def programming_bytes(record):
            return sum(e['multiplier'] for e in record['ledger']['entries'] if e['effect_id'] == 'weight_programming')
        self.assertGreater(programming_bytes(once), 0)
        self.assertEqual(programming_bytes(repeated), 3 * programming_bytes(once))
        self.assertEqual(repeated['output'], once['output'])
        self.assertGreater(repeated['schedule']['makespan_ns'], once['schedule']['makespan_ns'])
        self.assertEqual(repeated['workload_accounting']['reuse_count'], 3)

    def test_graph_structure_rejects_missing_drivers_direction_and_cycles(self):
        graph = evaluate_implementation(deepcopy(self.project))['physical_graph']
        self.assertTrue(validate_graph_structure(graph)['valid'])
        mutations = []
        missing = deepcopy(graph)
        missing['edges'].pop()
        mutations.append(missing)
        doubled = deepcopy(graph)
        doubled['edges'].append(deepcopy(doubled['edges'][0]))
        mutations.append(doubled)
        reverse = deepcopy(graph)
        reverse['edges'][0]['target']['port'] = 'out'
        mutations.append(reverse)
        cycle = deepcopy(graph)
        edge = cycle['edges'][0]
        target = next(n for n in cycle['nodes'] if n['id'] == edge['target']['node'])
        target['ports'][edge['target']['port']] = deepcopy(target['ports']['out'])
        edge['source'] = {'node': target['id'], 'port': 'out'}
        mutations.append(cycle)
        changed = deepcopy(graph)
        next(n for n in changed['nodes'] if n['operation'] != 'input')['operation'] = 'magic.convert'
        mutations.append(changed)
        for mutation in mutations:
            with self.assertRaises(ValueError):
                validate_graph_structure(mutation)

    def test_plain_optical_wire_cannot_gain_a_second_consumer(self):
        p = deepcopy(self.project)
        p['constraints']['resources']['photonic'] = 2
        result = evaluate_implementation(p, default_implementation(p['model'], 'fused'))
        graph = deepcopy(result['physical_graph'])
        source = next(node for node in graph['nodes'] if node['ports']['out']['carrier'] == 'optical')
        digital = next(node['ports']['out'] for node in graph['nodes'] if node['operation'] == 'input')
        graph['nodes'].append({'id': 'extra_readout', 'operation': 'interface.readout',
                              'ports': {'in0': deepcopy(source['ports']['out']), 'out': deepcopy(digital)},
                              'resource': 'converter', 'schedule_id': 'extra_readout'})
        graph['edges'].append({'source': {'node': source['id'], 'port': 'out'},
                              'target': {'node': 'extra_readout', 'port': 'in0'}, 'mechanism': 'plain_wire'})
        with self.assertRaisesRegex(ValueError, 'optical fanout'):
            validate_graph_structure(graph)

    def test_replay_validates_embedded_snapshots_before_any_file_access(self):
        record = evaluate_implementation(deepcopy(self.project))
        self.assertIn('project', validate_implementation_replay_inputs(record))
        mutations = []
        for name in ('model', 'workload', 'technology', 'constraints', 'environment'):
            bad = deepcopy(record)
            bad['normalized_inputs']['project'][name] = '/must/not/be/read.json'
            mutations.append(bad)
        weights = deepcopy(record)
        weights['normalized_inputs']['project']['model']['weights'] = '/must/not/be/read.npz'
        mutations.append(weights)
        geometry = deepcopy(record)
        geometry['normalized_inputs']['implementation']['geometry'] = '/must/not/be/read.json'
        mutations.append(geometry)
        proof = deepcopy(record)
        proof['normalized_inputs']['implementation']['operators']['up']['proof']['rationale'] = 'trust me'
        mutations.append(proof)
        nonfinite = deepcopy(record)
        nonfinite['normalized_inputs']['project']['workload']['inputs'][0][0] = float('nan')
        mutations.append(nonfinite)
        with patch('npp.multiphysics.io._read', side_effect=AssertionError('replay attempted file IO')) as read:
            for bad in mutations:
                with self.assertRaises(ValueError):
                    validate_implementation_replay_inputs(bad)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
