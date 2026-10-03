import copy
import json
import math
import unittest

import numpy as np

from npp.models import InputValidationError
from npp.multiphysics.backends.photonic import evaluate_linear,reference_parameters
from npp.multiphysics.catalog import reference_pack,component_for,parameters_from_component,normalize_pack,evaluate_model,RegisteredBackend
from npp.multiphysics.interfaces import encode,readout,signed_gate
from npp.multiphysics.contracts import CostLedger


class CoherentPhotonicTests(unittest.TestCase):
    def setUp(self):
        self.p=reference_parameters()
        self.io={key[3:]:value for key,value in self.p.items() if key.startswith('io.')}

    def ideal(self):
        p=copy.deepcopy(self.p);p.update(insertion_loss_db=0.,phase_bias_rad=0.,phase_noise_std_rad=0.)
        return p

    def test_reference_pack_registry_has_distinct_signed_matrix(self):
        pack=reference_pack();c=component_for(pack,'photonic')
        self.assertEqual(c['evaluator_id'],'photonic.coherent_matrix')
        p=parameters_from_component(c)
        self.assertEqual(p,self.p)
        self.assertEqual(normalize_pack(pack),pack)
        self.assertTrue(all(item['evidence']['kind']=='hypothetical' for item in c['parameters']))

    def test_old_m1_pack_remains_valid(self):
        pack=reference_pack();pack['components']=pack['components'][:2]
        self.assertEqual(normalize_pack(pack),pack)

    def test_signed_matrix_ideal_scalar_oracle_bias_exactly_once(self):
        w=[[.3,-.2],[.1,.4]];x=[[.25,-.5],[-.2,.4]];bias=[.02,-.01]
        r=evaluate_linear(w,x,bias,self.ideal(),{'mode':'ideal_diagnostic'})
        expected=[[sum(a*b for a,b in zip(row,input_row))+bv for row,bv in zip(w,bias)] for input_row in x]
        np.testing.assert_allclose(r['actual_output'],expected,atol=2e-15,rtol=2e-15)
        self.assertLessEqual(r['physical']['maximum_singular_value'],1.)
        self.assertIn('phase',r['physical']['signed_encoding'])

    def test_field_output_has_no_free_available_digital_readout(self):
        r=evaluate_linear([[1.,-.5]],[.2,-.3],None,self.ideal(),{'mode':'ideal_diagnostic','output_mode':'field'})
        self.assertIsNone(r['actual_output'])
        self.assertIsNotNone(r['field'])
        self.assertIn('not an available classical readout',r['diagnostic_scope'])
        self.assertNotIn('output',r['ledger']['category_coverage'])
        np.testing.assert_allclose(r['diagnostic_decoded_output'],[.35],atol=1e-15)

    def test_fused_field_gate_down_projection_matches_unfused_algebra(self):
        wup=[[.3,-.2],[.1,.4]];wdown=[[.5,-.4]];x=[.25,-.5];bup=[.02,-.01];gate=[.7,-.3]
        up=evaluate_linear(wup,x,bup,self.ideal(),{'mode':'ideal_diagnostic','output_mode':'field'})
        gated=signed_gate(up['field'],gate,self.io,{'mode':'ideal_diagnostic','reference_id':'coherent_reference'})
        down=evaluate_linear(wdown,gated['field'],[.13],self.ideal(),{'mode':'ideal_diagnostic','input_mode':'field'})
        expected=(np.asarray(wup)@x+bup)*gate
        np.testing.assert_allclose(down['actual_output'],np.asarray(wdown)@expected+[.13],atol=1e-14)
        self.assertNotIn('input',down['ledger']['category_coverage'])
        self.assertEqual(gated['operation'],'signed_gate')

    def test_bias_field_combiner_conserves_power_and_pays_source(self):
        r=evaluate_linear([[1.]],[.3],[.2],self.ideal(),{'mode':'ideal_diagnostic','output_mode':'field'})
        phy=r['physical']
        self.assertAlmostEqual(phy['final_field_power_mw']+phy['terminated_power_mw'],phy['matrix_output_power_mw']+phy['bias_injected_power_mw'],places=15)
        self.assertGreater(phy['bias_injection_energy_pj'],0)
        self.assertGreater(r['duration_breakdown']['bias_ns'],0)
        np.testing.assert_allclose(r['diagnostic_decoded_output'],[.5],atol=1e-15)

    def test_power_loss_is_square_of_field_attenuation(self):
        p=self.ideal();p['insertion_loss_db']=6.
        r=evaluate_linear([[1.]],[1.],None,p,{'output_mode':'field'})
        # Quantized programmed transfer is inspected independently; all power
        # is |amplitude| squared, never a second ad-hoc intensity multiplier.
        transfer=complex(r['physical']['transfer_real'][0][0],r['physical']['transfer_imag'][0][0])
        self.assertAlmostEqual(r['physical']['matrix_output_power_mw'],r['physical']['input_power_mw']*abs(transfer)**2,places=15)
        p['insertion_loss_db']=0.
        no_loss=evaluate_linear([[1.]],[1.],None,p,{'output_mode':'field'})
        self.assertAlmostEqual(r['physical']['matrix_output_power_mw']/no_loss['physical']['matrix_output_power_mw'],10**(-.6),places=14)

    def test_passivity_guard_and_finite_program_projection(self):
        p=copy.deepcopy(self.p);p['weight_scale']=1.;p['program_bits']=2
        with self.assertRaises(InputValidationError):evaluate_linear([[1.,1.]],[.1,.1],None,p)
        r=evaluate_linear([[.7,.7]],[.1,.1],None,p)
        self.assertLessEqual(r['physical']['maximum_singular_value'],1.+1e-14)
        self.assertLessEqual(r['physical']['matrix_output_power_mw'],r['physical']['input_power_mw']*(1+1e-14))

    def test_reference_phase_noise_reproducible_and_identity_sensitive(self):
        p=copy.deepcopy(self.p);p['phase_noise_std_rad']=.2
        a=evaluate_linear([[.8]],[.4],None,p,{'seed':5})
        b=evaluate_linear([[.8]],[.4],None,p,{'seed':5})
        c=evaluate_linear([[.8]],[.4],None,p,{'seed':6})
        self.assertEqual(a,b);self.assertNotEqual(a['physical']['transfer_imag'],c['physical']['transfer_imag'])

    def test_intensity_quantum_and_reference_mismatch_not_cast(self):
        field=encode([.2],self.io,{'reference_id':'coherent_reference'})['field']
        for key,value in [('encoding','normalized_intensity'),('regime','quantum'),('reference_id','unlocked_reference')]:
            f=copy.deepcopy(field);f[key]=value
            with self.assertRaises(InputValidationError):evaluate_linear([[1.]],f,None,self.p,{'input_mode':'field'})

    def test_partial_unknown_cost_is_conditional_in_combined_ledger(self):
        p=copy.deepcopy(self.p);p['io.adc_energy_pj']={'state':'unavailable','unit':self.p['io.adc_energy_pj']['unit'],'reason':'uncharacterized ADC'}
        r=evaluate_linear([[1.]],[.3],None,p)
        self.assertEqual(r['status'],'conditional');self.assertFalse(r['ledger']['complete'])
        self.assertEqual(r['ledger']['energy_pj']['state'],'unavailable')
        CostLedger.model_validate_json(json.dumps(r['ledger']['contract']))

    def test_field_path_omits_only_explicit_conversion_owners(self):
        digital=evaluate_linear([[1.]],[.3],None,self.p)
        field=encode([.3],self.io,{'reference_id':'coherent_reference'})['field']
        optical=evaluate_linear([[1.]],field,None,self.p,{'input_mode':'field','output_mode':'field'})
        self.assertEqual(set(optical['ledger']['category_coverage']),{'matrix'})
        self.assertEqual(set(digital['ledger']['category_coverage']),{'input','matrix','output'})
        self.assertLess(optical['ledger']['energy_pj']['value'],digital['ledger']['energy_pj']['value'])
        self.assertGreater(optical['duration_breakdown']['calibration_ns'],0)

    def test_catalog_and_lifecycle_photonic_execution(self):
        pack=reference_pack()
        r=evaluate_model({'weights':[[1.]]},{'inputs':[.3]},pack,family='photonic')
        self.assertEqual(r['status'],'model_feasible')
        backend=RegisteredBackend('photonic.coherent_matrix')
        instance=backend.instantiate(component_for(pack,'photonic'),'{}')
        outcome=backend.evaluate(instance.payload_json,json.dumps({'request':{'weights':[[1.]],'inputs':[.3]}}))
        self.assertEqual(outcome.status,'model_feasible',outcome.diagnostics)
        self.assertEqual(backend.compare(instance.payload_json,'{}','{}').status,'unsupported')

    def test_malformed_public_contexts_fail_structurally(self):
        for context in ([],False,'bad',{'environment':'bad'},{'environment':[]},
                        {'environment':{'temperature_c':float('nan')}},{'environment':{'services':[]}},
                        {'owner_id':None},{'reference_id':3},{'reference_id':'bad reference'}):
            with self.subTest(context=context),self.assertRaises(InputValidationError):
                evaluate_linear([[1.]],[.2],None,self.p,context)
        for context in ([],False,{'owner_id':None},{'reference_id':3}):
            with self.assertRaises(InputValidationError):encode([.2],self.io,context)

    def test_programmed_weight_retention_is_explicit_and_guarded(self):
        component=component_for(reference_pack(),'photonic')
        self.assertEqual(component['dynamics']['retention_ns']['value'],self.p['weight_retention_ns'])
        good=evaluate_linear([[1.]],[.2],None,self.p,{'weight_age_ns':self.p['weight_retention_ns']})
        self.assertEqual(good['state']['weight_age_ns'],self.p['weight_retention_ns'])
        with self.assertRaises(InputValidationError):
            evaluate_linear([[1.]],[.2],None,self.p,{'weight_age_ns':self.p['weight_retention_ns']+1})

    def test_nested_physical_disclosure_uses_real_field_operations(self):
        from npp.multiphysics.composition_graphs import physical_subgraph
        from npp.multiphysics.composition_validation import validate_graph_structure
        for mode in ('digital','field'):
            result=evaluate_linear([[.4,-.5]],[.2,-.4],[.1],self.p,{'output_mode':mode})
            graph=physical_subgraph(result,[.2,-.4],self.p,'tile',output_mode=mode)
            self.assertTrue(validate_graph_structure(graph,_depth=1,_parent={'id':'tile','schedule_id':'tile'})['valid'])
            self.assertTrue(all(n['accounting_parent']=='tile' for n in graph['nodes']))
            operations=[n['operation'] for n in graph['nodes']]
            self.assertIn('photonic.mesh',operations)
            if mode=='field':
                self.assertIn('photonic.coherent_bias',operations)
                self.assertNotIn('interface.readout',operations)
            else:
                self.assertIn('interface.readout',operations)
                self.assertIn('digital.add_bias',operations)
            self.assertTrue(all('ledger' not in n for n in graph['nodes']))

    def test_nested_analog_disclosure_keeps_voltage_and_current_distinct(self):
        from npp.multiphysics.composition_graphs import physical_subgraph
        from npp.multiphysics.composition_validation import validate_graph_structure
        from npp.multiphysics.backends.analog import evaluate_linear as analog_evaluate
        params=parameters_from_component(component_for(reference_pack(),'analog'))
        result=analog_evaluate([[.4,-.5]],[.2,-.4],[.1],params)
        graph=physical_subgraph(result,[.2,-.4],params,'analog_tile')
        self.assertTrue(validate_graph_structure(graph,_depth=1,_parent={'id':'analog_tile','schedule_id':'analog_tile'})['valid'])
        nodes={n['operation']:n for n in graph['nodes'] if n['operation']!='input'}
        self.assertEqual(nodes['interface.dac']['ports']['out']['unit'],'V')
        self.assertEqual(nodes['analog.tile']['ports']['out']['unit'],'A')
        self.assertEqual(nodes['interface.adc']['ports']['out']['physical_variable'],'digital_word')
        self.assertIn('digital.add_bias',nodes)


if __name__=='__main__':unittest.main()
