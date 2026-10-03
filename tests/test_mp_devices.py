import copy
import json
import math
import unittest

import numpy as np

from npp.models import InputValidationError
from npp.multiphysics.catalog import (reference_pack, normalize_pack, component_for,
                                     parameters_from_component, evaluate_model,
                                     RegisteredBackend)
from npp.multiphysics.backends.analog import evaluate_linear
from npp.multiphysics.backends.digital import evaluate_ffn_cost
from npp.multiphysics.contracts import TechnologyPack, CostLedger, BackendProtocol

MODEL={'W_up':[[1.,0.],[0.,1.]],'W_gate':[[1.,0.],[0.,1.]],'W_down':[[1.,-1.]],
       'b_up':[.1,0.],'b_gate':[0.,.2],'b_down':[.3]}


class MultiphysicsDeviceTests(unittest.TestCase):
    def setUp(self):
        self.pack=reference_pack()
        self.digital=parameters_from_component(component_for(self.pack,'digital'))
        self.analog=parameters_from_component(component_for(self.pack,'analog'))

    def ideal(self):
        p=copy.deepcopy(self.analog)
        p.update(mode='ideal_diagnostic',drift_per_ns=0.,read_disturbance_relative=0.,line_resistance_ohm=0.,programming_noise_relative=0.)
        return p

    def test_pack_uses_contracts_and_parameter_level_hypothetical_evidence(self):
        pack=TechnologyPack.model_validate_json(json.dumps(self.pack))
        self.assertEqual(normalize_pack(self.pack),self.pack)
        self.assertGreaterEqual(len(pack.components),2)
        self.assertTrue({'digital','analog_electrical'} <= {c.family for c in pack.components})
        for component in pack.components:
            self.assertTrue(component.ports and component.operating_envelope)
            for parameter in component.parameters:
                self.assertEqual(parameter.evidence.kind,'hypothetical')
                self.assertEqual(parameter.uncertainty.kind,'epistemic')

    def test_registered_only_no_formula_evaluation(self):
        for mutate in (lambda c:c.update(evaluator_id='evil.eval'), lambda c:c.update(evaluator_version='future'),
                       lambda c:c.update(family='quantum'),lambda c:c.update(fidelity='measured_surrogate')):
            p=copy.deepcopy(self.pack); mutate(p['components'][0])
            with self.assertRaises(InputValidationError): normalize_pack(p)

    def test_scalar_independent_analog_oracle_signed_bias_once(self):
        w=[[.7,-1.2],[-.2,.9]];x=[[-.3,.6],[.8,-.1]];b=[.13,-.21]
        result=evaluate_linear(w,x,b,self.ideal())
        expected=[[sum(row[k]*v[k] for k in range(2))+bias for row,bias in zip(w,b)] for v in x]
        np.testing.assert_allclose(result['actual_output'],expected,rtol=2e-14,atol=2e-14)
        self.assertTrue(all(v>=self.analog['g_min_s'] for row in result['physical']['conductance_positive_s'] for v in row))
        self.assertTrue(all(v>=self.analog['g_min_s'] for row in result['physical']['conductance_negative_s'] for v in row))

    def test_finite_quantization_changes_output_and_ideal_not_finite_claim(self):
        finite=evaluate_linear([[.71,-1.19]],[.31,-.27],None,self.analog)
        ideal=evaluate_linear([[.71,-1.19]],[.31,-.27],None,self.ideal())
        self.assertGreater(finite['error']['rms'],ideal['error']['rms'])
        self.assertIn('mathematical diagnostic',ideal['assumptions'][-1])
        self.assertEqual(finite['resources']['programmed_cells'],4)

    def test_differential_energy_includes_both_rails_and_voltage_squared(self):
        p=self.ideal(); r=evaluate_linear([[0.,0.]],[.2,-.3],None,p)
        expected=(.02**2+.03**2)*(2*p['g_min_s'])*p['integration_ns']*1e3
        e=next(e for e in r['ledger']['entries'] if e['effect_id']=='conductance_dissipation')
        self.assertAlmostEqual(e['energy_pj']['value'],expected,places=15)
        self.assertGreater(e['energy_pj']['value'],0.)

    def test_explicit_series_loading_reduces_current(self):
        p=copy.deepcopy(self.analog);p.update(programming_noise_relative=0.,line_resistance_ohm=0.)
        clean=evaluate_linear([[1.,1.]],[.4,.3],None,p)
        p['line_resistance_ohm']=1000.
        loaded=evaluate_linear([[1.,1.]],[.4,.3],None,p)
        self.assertLess(loaded['physical']['differential_current_a'][0][0],clean['physical']['differential_current_a'][0][0])

    def test_drift_and_read_disturbance_have_explicit_history(self):
        p=copy.deepcopy(self.analog);p.update(drift_per_ns=.001,read_disturbance_relative=.01)
        fresh=evaluate_linear([[1.]],[1.],None,p)
        aged=evaluate_linear([[1.]],[1.],None,p,{'age_ns':100.,'read_count':10})
        self.assertLess(aged['physical']['differential_current_a'][0][0],fresh['physical']['differential_current_a'][0][0])
        self.assertEqual(aged['state']['read_count'],10)

    def test_programming_samples_explicit_seed_and_stream(self):
        p=copy.deepcopy(self.analog);p['programming_noise_relative']=.03
        a=evaluate_linear([[.4,-.8]],[.3,.7],None,p,{'seed':9})
        b=evaluate_linear([[.4,-.8]],[.3,.7],None,p,{'seed':9})
        c=evaluate_linear([[.4,-.8]],[.3,.7],None,p,{'seed':10})
        self.assertEqual(a,b);self.assertNotEqual(a['physical'],c['physical'])

    def test_analog_rejects_range_dimension_current_temperature_and_boolean(self):
        for w,x,b,p,c in [([[3.]],[1.],None,self.analog,{}),([[1.]],[3.],None,self.analog,{}),
                           ([[1.]],[1.],None,self.analog,{'temperature_c':100.}),
                           ([[1.]],[True],None,self.analog,{}),([[1.,2.]],[1.],None,self.analog,{}),
                           ([[1.]],[1.],[1.,2.],self.analog,{})]:
            with self.assertRaises(InputValidationError):evaluate_linear(w,x,b,p,c)
        p=copy.deepcopy(self.analog);p['output_current_limit_a']=1e-8
        with self.assertRaises(InputValidationError):evaluate_linear([[1.]],[1.],None,p)

    def test_analog_ideal_mode_cannot_hide_nonideal_parameters(self):
        p=copy.deepcopy(self.analog);p['mode']='ideal_diagnostic'
        with self.assertRaises(InputValidationError):evaluate_linear([[1.]],[1.],None,p)

    def test_unknown_or_symbolic_cost_not_zero_or_fake_feasible(self):
        for q in ({'state':'unavailable','unit':'pJ/sample','reason':'not characterized'},
                  {'state':'bounded','unit':'pJ/sample','lower':.1,'upper':.5},
                  {'state':'bounded','unit':'pJ/sample','symbol':'read_cost'}):
            p=copy.deepcopy(self.analog);p['adc_energy_pj']=q
            result=evaluate_linear([[1.]],[1.],None,p)
            self.assertEqual(result['status'],'conditional')
            self.assertFalse(result['ledger']['complete'])
            self.assertNotEqual(result['ledger']['energy_pj']['state'],'known')
            CostLedger.model_validate_json(json.dumps(result['ledger']['contract']))

    def test_direct_cost_unit_and_negative_guard(self):
        for q in ({'state':'known','value':1.,'unit':'W'}, {'state':'known','value':-1.,'unit':'mW'}):
            p=copy.deepcopy(self.analog);p['static_power_mw']=q
            with self.assertRaises(InputValidationError):evaluate_linear([[1.]],[1.],None,p)

    def test_digital_complete_semantics_counts_and_movement(self):
        r=evaluate_ffn_cost(MODEL,[.2,-.4],self.digital)
        u=[.3,-.4]; g=[.2,-.2]
        expected=u[0]*g[0]/(1+math.exp(-g[0]))-u[1]*g[1]/(1+math.exp(-g[1]))+.3
        self.assertAlmostEqual(r['actual_output'][0],expected,places=14)
        self.assertEqual(r['operations']['matrix_mac'],10)
        self.assertEqual(r['operations']['bias_add'],5)
        self.assertEqual(r['operations']['silu'],2)
        self.assertEqual(r['operations']['elementwise_multiply'],2)
        self.assertEqual(r['resources']['weight_bytes'],15*8)
        for key in ('sram_read_bytes','sram_write_bytes','dram_bytes','link_bytes'):
            self.assertGreater(r['resources'][key],0)
        self.assertGreater(r['duration_breakdown']['control_ns'],0)

    def test_digital_reuse_pays_programming_once_and_movement_each_time(self):
        one=evaluate_ffn_cost(MODEL,[.2,-.4],self.digital)
        five=evaluate_ffn_cost(MODEL,[.2,-.4],self.digital,{'reuse_count':5})
        entries=lambda r:{e['effect_id']:e['energy_pj']['value'] for e in r['ledger']['entries']}
        a,b=entries(one),entries(five)
        self.assertEqual(a['weight_programming'],b['weight_programming'])
        self.assertEqual(5*a['matrix_mac'],b['matrix_mac'])
        self.assertEqual(5*a['sram_reads'],b['sram_reads'])
        self.assertEqual(a['dram_weight_loads'],b['dram_weight_loads'])
        dram=evaluate_ffn_cost(MODEL,[.2,-.4],self.digital,{'reuse_count':5,'weight_residency':'dram'})
        self.assertEqual(5*a['dram_weight_loads'],entries(dram)['dram_weight_loads'])

    def test_digital_capacity_rejected_not_infinite_bandwidth(self):
        p=copy.deepcopy(self.digital);p['sram_capacity_bytes']=1
        with self.assertRaises(InputValidationError):evaluate_ffn_cost(MODEL,[.2,-.4],p)
        p=copy.deepcopy(self.digital);p['macs_per_ns']=0
        with self.assertRaises(InputValidationError):evaluate_ffn_cost(MODEL,[.2,-.4],p)

    def test_context_identity_is_dependency_sensitive_and_inputs_unmodified(self):
        model=copy.deepcopy(MODEL);p=copy.deepcopy(self.digital)
        a=evaluate_ffn_cost(model,[.2,-.4],p,{'geometry':{'length_um':2.}})
        b=evaluate_ffn_cost(model,[.2,-.4],p,{'geometry':{'length_um':3.}})
        self.assertNotEqual(a['dependencies']['cache_key'],b['dependencies']['cache_key'])
        self.assertEqual(model,MODEL);self.assertEqual(p,self.digital)

    def test_catalog_environment_evidence_and_declared_envelope_guards(self):
        env={'temperature_c':25.,'region':'room','substrate':'hypothetical_chiplet','services':['power_supply']}
        self.assertEqual(evaluate_model(MODEL,{'inputs':[.2,-.4]},self.pack,context=env)['status'],'model_feasible')
        for key,value in [('temperature_c',99.),('services',[]),('substrate','alien')]:
            e=copy.deepcopy(env);e[key]=value
            with self.assertRaises(InputValidationError):evaluate_model(MODEL,{'inputs':[.2,-.4]},self.pack,context=e)
        with self.assertRaises(InputValidationError):
            evaluate_model(MODEL,{'inputs':[.2,-.4],'evidence_policy':['measured_instance']},self.pack)
        p=copy.deepcopy(self.pack);p['components'][0]['operating_envelope'][0]['maximum']=1.
        with self.assertRaises(InputValidationError):evaluate_model(MODEL,{'inputs':[.2,-.4]},p)

    def test_quality_budget_returns_inspectable_failure(self):
        r=evaluate_model({'weights':[[.7,-1.2]]},{'inputs':[.3,-.2],'quality':{'max_rms_error':0.}},self.pack,family='analog')
        self.assertEqual(r['status'],'resource_infeasible');self.assertFalse(r['quality']['passed'])
        self.assertIn('actual_output',r)

    def test_lifecycle_snapshot_isolation_advance_sampling_and_unsupported_comparison(self):
        backend=RegisteredBackend('analog.differential_conductance')
        self.assertIsInstance(backend,BackendProtocol)
        first=backend.instantiate(self.pack['components'][1],'{}')
        second=backend.instantiate(self.pack['components'][1],'{}')
        later=backend.advance(first.payload_json,'{"delta_ns":100,"reads":3}')
        self.assertEqual(json.loads(first.payload_json)['state']['age_ns'],0)
        self.assertEqual(first.payload_json,second.payload_json)
        self.assertEqual(json.loads(later.payload_json)['state']['read_count'],3)
        request={'request':{'weights':[[1.]],'inputs':[.2]},'evaluation_context':{'seed':11}}
        result=backend.sample(first.payload_json,json.dumps(request))
        self.assertEqual(result.status,'model_feasible')
        self.assertEqual(backend.compare(first.payload_json,'{}','{}').status,'unsupported')
        del request['evaluation_context']['seed']
        self.assertEqual(backend.sample(first.payload_json,json.dumps(request)).status,'invalid')
        wrong=RegisteredBackend('digital.ffn').evaluate(first.payload_json,'{}')
        self.assertEqual(wrong.status,'invalid')
        self.assertEqual(wrong.diagnostics[0].code,'domain_mismatch')

    def test_instance_private_state_and_component_mutation_never_leaks(self):
        backend=RegisteredBackend('analog.differential_conductance')
        component=copy.deepcopy(self.pack['components'][1]); instance=backend.instantiate(component,'{}')
        component['parameters'][0]['quantity']['value']=0
        self.assertNotEqual(json.loads(instance.payload_json)['component'],component)
        self.assertEqual(backend.export(instance.payload_json,'{}').status,'model_feasible')


if __name__=='__main__':unittest.main()
