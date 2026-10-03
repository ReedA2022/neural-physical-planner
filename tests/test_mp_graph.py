"""Complete mixed FFN compiler tests with an independent scalar semantic oracle."""
import copy
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from npp.multiphysics.io import init_project, load_project
from npp.multiphysics.graph import (default_implementation, evaluate_implementation,
                                    check_implementation, normalize_implementation, validate_graph)
from npp.multiphysics.contracts import canonical_hash
from tests.test_mp_semantic import decimal_oracle


def project():
    with tempfile.TemporaryDirectory() as d:
        init_project(d)
        result=load_project(d+'/project.yaml')
    result['constraints']['resources']['photonic']=2
    return result


def scalar_oracle(model,inputs):
    def mv(w,x,b): return [math.fsum(a*v for a,v in zip(row,x))+bias for row,bias in zip(w,b)]
    out=[]
    for x in inputs:
        u=mv(model['W_up'],x,model['b_up']);g=mv(model['W_gate'],x,model['b_gate'])
        h=[a*b/(1+math.exp(-b)) for a,b in zip(u,g)]
        out.append(mv(model['W_down'],h,model['b_down']))
    return out


class GraphTests(unittest.TestCase):
    def test_complete_families_and_fused(self):
        p=project(); expected=scalar_oracle(p['model'],p['workload']['inputs'])
        for kind in ('digital','analog','photonic','hybrid','fused'):
            with self.subTest(kind=kind):
                r=evaluate_implementation(p,default_implementation(p['model'],kind))
                self.assertEqual(r['status'],'model_feasible')
                self.assertTrue(r['schedule']['state_execution_complete'])
                self.assertTrue(r['quality']['passed'])
                self.assertLess(max(abs(a-b) for row,x in zip(r['output'],expected) for a,b in zip(row,x)),.02)
                self.assertTrue(check_implementation(r)['valid'])
                self.assertGreater(r['ledger']['energy_pj']['value'],0)

    def test_two_dimensional_mixed_partition_and_signed_permutation(self):
        p=project();spec=default_implementation(p['model'])
        spec['operators']['up']={'tiles':[
            {'rows':[r],'columns':[c],'family':('digital','analog','photonic','digital')[2*r+c]}
            for r in range(2) for c in range(2)]}
        spec['transform']={'permutation':[1,0],'signs':[-1,1]}
        r=evaluate_implementation(p,spec)
        self.assertTrue(r['quality']['passed'])
        self.assertEqual(r['semantic_plan']['partitions']['up']['merge']['bias_application'],'once_after_reduction')
        self.assertTrue(any('sum' in n['id'] for n in r['physical_graph']['nodes']))
        self.assertTrue(check_implementation(r)['valid'])

    def test_invalid_partition_bounds_preflight(self):
        p=project();spec=default_implementation(p['model'])
        spec['operators']['up']={'tiles':[{'input':[0,10**12],'output':[0,2],'family':'digital'}]}
        with self.assertRaises(ValueError): normalize_implementation(spec,p['model'])
        spec['operators']['up']={'input_cuts':[0,1,1,2],'family':'digital'}
        with self.assertRaises(ValueError): normalize_implementation(spec,p['model'])

    def test_optical_alignment_is_paid_and_missing_capacity_rejected(self):
        p=project();spec=default_implementation(p['model'],'fused')
        records=[]
        for kind in ('delayed_launch','optical_delay','electronic_buffer','recompute'):
            s=copy.deepcopy(spec);s['alignment'].update(method=kind,delay_ns=100.)
            r=evaluate_implementation(p,s);records.append(r)
            self.assertEqual(r['status'],'model_feasible')
            self.assertTrue(r['schedule']['state_execution_complete'])
        self.assertGreater(records[1]['schedule']['makespan_ns'],records[0]['schedule']['makespan_ns'])
        self.assertGreater(records[3]['ledger']['energy_pj']['value'],records[0]['ledger']['energy_pj']['value'])
        p['constraints']['resources']['photonic']=1
        with self.assertRaises(ValueError): evaluate_implementation(p,spec)

    def test_replay_rejects_resigned_tamper(self):
        p=project();r=evaluate_implementation(p)
        r['output'][0][0]+=1
        r['record_hash']=canonical_hash({k:v for k,v in r.items() if k!='record_hash'})
        self.assertFalse(check_implementation(r)['valid'])
        for bad in ({'schema_version':'npp-implementation-result-1','normalized_inputs':[]},None,[],{}):
            self.assertFalse(check_implementation(bad)['valid'])

    def test_graph_type_mismatch(self):
        r=evaluate_implementation(project());g=copy.deepcopy(r['physical_graph'])
        edge=g['edges'][0];target=next(n for n in g['nodes'] if n['id']==edge['target']['node'])
        target['ports'][edge['target']['port']]['unit']='V'
        with self.assertRaises(ValueError): validate_graph(g)

    def test_unknown_system_cost_not_zero_and_bound_violation(self):
        p=project();s=default_implementation(p['model'])
        s['system']={'preset':'hypothetical_system','overrides':{'package_static_power_mw':{'state':'unavailable','unit':'mW','reason':'not characterized'}}}
        r=evaluate_implementation(p,s)
        self.assertEqual(r['status'],'conditional');self.assertEqual(r['ledger']['energy_pj']['state'],'unavailable')
        s['system']={'preset':'hypothetical_system','overrides':{'package_startup_energy_pj':{'state':'bounded','unit':'pJ','lower':1e9,'upper':2e9,'symbol':'package'}}}
        r=evaluate_implementation(p,s)
        self.assertEqual(r['status'],'resource_infeasible')

    def test_lifetime_and_services_are_not_ignored(self):
        p=project();s=default_implementation(p['model'])
        s['system']={'preset':'hypothetical_system','overrides':{'electronic_buffer_retention_ns':.0001}}
        with self.assertRaises(ValueError): evaluate_implementation(p,s)
        p['environment']['services'].remove('phase_reference')
        with self.assertRaises(ValueError): evaluate_implementation(p,default_implementation(p['model'],'photonic'))

    def test_signed_fusion_with_nonzero_bias_and_negative_gate(self):
        p=project();p['model']['b_down']=[.01,-.02]
        r=evaluate_implementation(p,default_implementation(p['model'],'fused'))
        gate=r['backend_evaluations']['run0.silu']['actual_output']
        self.assertTrue(any(v<0 for row in gate for v in row))
        expected=[decimal_oracle(p['model'],x) for x in p['workload']['inputs']]
        np.testing.assert_allclose(r['output'],expected,rtol=0,atol=.004)
        gate_ledger=r['backend_evaluations']['run0.signed_gate']['ledger']['entries']
        effects={e['effect_id'] for e in gate_ledger}
        self.assertIn('gate_dac',effects)
        self.assertIn('signed_electro_optic_gate',effects)
        self.assertTrue(any(e['category']=='source' for e in r['ledger']['entries']))
        self.assertTrue(r['schedule']['state_execution_complete'])

    def test_every_repeated_invocation_must_meet_quality_budget(self):
        p=project();p['workload'].update(reuse_count=3,seed=3)
        for component in p['technology']['components']:
            if component['family']=='analog_electrical':
                for parameter in component['parameters']:
                    if parameter['name']=='programming_noise_relative': parameter['quantity']['value']=.2
                    if parameter['name'] in ('adc_bits','dac_bits'): parameter['quantity']['value']=20.
        spec=default_implementation(p['model'],'analog')
        diagnostic=evaluate_implementation(p,spec)
        errors=[invocation['error']['rms'] for invocation in diagnostic['invocations']]
        self.assertEqual(len(errors),3)
        self.assertGreater(max(errors[:-1]),errors[-1])
        budget=(max(errors[:-1])+errors[-1])/2
        p['workload']['quality']['max_rms_error']=budget
        guarded=evaluate_implementation(p,spec)
        self.assertLess(guarded['invocations'][-1]['error']['rms'],budget)
        self.assertEqual(guarded['status'],'resource_infeasible')
        self.assertFalse(guarded['quality']['passed'])
        self.assertEqual(guarded['quality']['rms'],max(errors))
        self.assertTrue(check_implementation(guarded)['valid'])

    def test_fused_weights_have_owned_program_read_consume_lifecycle(self):
        p=project();p['workload']['reuse_count']=2
        spec=default_implementation(p['model'],'fused')
        result=evaluate_implementation(p,spec)
        states=result['owned_weight_states']
        self.assertEqual(len(states),2)
        self.assertEqual([state['epoch'] for state in states],[0,1])
        self.assertEqual(len({state['state_id'] for state in states}),2)
        self.assertTrue(all(state['physical_resource']=='photonic_down' for state in states))
        self.assertTrue(result['schedule']['state_execution_complete'])
        # A valid but too-short physical retention parameter must fail the
        # scheduled program-to-read interval, not merely update an annotation.
        p['workload']['reuse_count']=1
        for component in p['technology']['components']:
            if component['family']=='photonic':
                for parameter in component['parameters']:
                    if parameter['name']=='weight_retention_ns': parameter['quantity']['value']=.0001
                component['dynamics']['retention_ns']['value']=.0001
        with self.assertRaisesRegex(ValueError,'retention_expired'):
            evaluate_implementation(p,spec)

    def test_resource_and_service_omissions_fail_only_when_required(self):
        original=project()
        for resource in ('source','converter','control'):
            p=copy.deepcopy(original);p['constraints']['resources'].pop(resource)
            with self.subTest(resource=resource),self.assertRaises(ValueError):
                evaluate_implementation(p,default_implementation(p['model'],'fused'))
        p=copy.deepcopy(original);p['environment']['services'].remove('clock')
        with self.assertRaises(ValueError): evaluate_implementation(p)
        # Optional device families and their services do not alter the digital
        # arithmetic or require a source/converter reservation when unused.
        baseline=evaluate_implementation(original)
        p=copy.deepcopy(original)
        p['technology']['components']=[c for c in p['technology']['components'] if c['family']=='digital']
        p['environment']['services']=[s for s in p['environment']['services'] if s not in ('laser','phase_reference')]
        p['constraints']['resources'].pop('source');p['constraints']['resources'].pop('converter')
        isolated=evaluate_implementation(p)
        self.assertEqual(isolated['output'],baseline['output'])
        self.assertEqual(isolated['ledger']['energy_pj'],baseline['ledger']['energy_pj'])

    def test_explicit_physical_subgraphs_disclose_conversion_without_double_counting(self):
        p=project()
        for kind in ('analog','photonic','hybrid','fused'):
            result=evaluate_implementation(p,default_implementation(p['model'],kind))
            children=[node for parent in result['physical_graph']['nodes']
                      for node in parent.get('physical_subgraph',{}).get('nodes',[])]
            self.assertGreater(len(children),0)
            self.assertTrue(all(child['accounting']=='included_in_parent' for child in children))
            parent_ids={node['id'] for node in result['physical_graph']['nodes']}
            self.assertTrue(all(child['accounting_parent'] in parent_ids for child in children))
            ledger_owners={entry['owner_id'] for entry in result['ledger']['entries']}
            self.assertFalse(ledger_owners & {child['id'] for child in children})
            operations={child['operation'] for child in children}
            if kind=='analog':
                self.assertTrue({'interface.dac','analog.tile','interface.adc'}<=operations)
            if kind=='photonic':
                self.assertTrue({'interface.encode','photonic.mesh','interface.readout'}<=operations)

    def test_replay_never_loads_resigned_external_reference(self):
        record=evaluate_implementation(project())
        record['normalized_inputs']['project']['model']='/must/not/be/read.json'
        inputs=record['normalized_inputs']['project']
        inputs['input_hash']=canonical_hash({k:v for k,v in inputs.items() if k!='input_hash'})
        record['record_hash']=canonical_hash({k:v for k,v in record.items() if k!='record_hash'})
        with patch('npp.multiphysics.io._read',side_effect=AssertionError('replay attempted IO')) as read:
            self.assertFalse(check_implementation(record)['valid'])
            read.assert_not_called()


if __name__=='__main__': unittest.main()
