import copy
import math
import unittest

import numpy as np

from npp.models import InputValidationError
from npp.multiphysics.array_padding import expand_array
from npp.multiphysics.catalog import reference_pack,component_for,parameters_from_component
from npp.multiphysics.backends.analog import evaluate_linear


class PhysicalArrayPaddingTests(unittest.TestCase):
    def test_full_rectangle_preserves_ideal_embedding_and_output_order(self):
        w=[[.3,-.2],[.4,.5]];x=[[.1,.2],[-.3,.4]];bias=[.2,-.1]
        snapshot=copy.deepcopy((w,x,bias))
        p=expand_array(w,x,bias,3,4)
        expected=np.asarray(x)@np.asarray(w).T+bias
        padded=np.asarray(p['inputs'])@np.asarray(p['weights']).T+p['bias']
        np.testing.assert_allclose(padded[:,p['output_indices']],expected)
        self.assertEqual(p['output_indices'],[0,1]);self.assertEqual(p['utilization'],1/3)
        self.assertEqual((w,x,bias),snapshot)
        self.assertTrue(np.all(padded[:,2:]==0))

    def test_finite_gmin_inactive_rows_change_loading_scalar_oracle(self):
        params=parameters_from_component(component_for(reference_pack(),'analog'))
        params.update(line_resistance_ohm=10000.,drift_per_ns=0.,read_disturbance_relative=0.,programming_noise_relative=0.,output_current_limit_a=.0001)
        padding=expand_array([[2.]],[2.],None,3,2)
        full=evaluate_linear(padding['weights'],padding['inputs'],padding['bias'],params)
        narrow=evaluate_linear([[2.]],[2.],None,params)
        # Independent physical oracle: the active + rail saturates at g_max;
        # active - rail and BOTH rails of each inactive row remain at g_min.
        conductance_load=params['g_max_s']+params['g_min_s']+2*(3-1)*params['g_min_s']
        applied_voltage=2.*params['input_scale_v']
        loaded_voltage=applied_voltage/(1.+params['line_resistance_ohm']*conductance_load)
        differential_current=loaded_voltage*(params['g_max_s']-params['g_min_s'])
        levels=2**(params['adc_bits']-1)-1
        digitized=round(differential_current/params['output_current_limit_a']*levels)/levels*params['output_current_limit_a']
        expected=digitized*params['weight_scale']/((params['g_max_s']-params['g_min_s'])*params['input_scale_v'])
        self.assertAlmostEqual(full['physical']['loaded_voltage_v'][0][0],loaded_voltage,places=15)
        self.assertAlmostEqual(full['actual_output'][0],expected,places=13)
        self.assertLess(full['actual_output'][0],narrow['actual_output'][0])
        self.assertGreater(narrow['actual_output'][0]-full['actual_output'][0],.02)
        self.assertEqual(full['actual_output'][1:],[0.,0.])
        gp=full['physical']['conductance_positive_s'];gn=full['physical']['conductance_negative_s']
        self.assertEqual(np.asarray(gp).shape,(3,2))
        self.assertTrue(all(gp[i][0]==gn[i][0]==params['g_min_s'] for i in (1,2)))
        counts={entry['effect_id']:entry['multiplier'] for entry in full['ledger']['entries']}
        self.assertEqual(counts['input_dac'],2)
        self.assertEqual(counts['output_adc'],3)
        self.assertEqual(counts['weight_programming'],12)
        expected_dissipation=loaded_voltage**2*conductance_load*params['integration_ns']*1e3
        dissip=next(e['energy_pj']['value'] for e in full['ledger']['entries'] if e['effect_id']=='conductance_dissipation')
        self.assertAlmostEqual(dissip,expected_dissipation,places=13)

    def test_same_shape_preserves_payload_and_rejects_field_padding(self):
        p=expand_array([[1.,-1.]],[.2,.3],[.4],1,2)
        self.assertEqual(p['weights'],[[1.,-1.]])
        self.assertEqual(p['inputs'],[.2,.3]);self.assertEqual(p['bias'],[.4]);self.assertEqual(p['utilization'],1.)
        with self.assertRaises(InputValidationError) as caught:
            expand_array([[1.]],{'encoding':'signed_coherent_field'},None,2,2)
        self.assertEqual(caught.exception.diagnostics[0]['code'],'domain_mismatch')

    def test_invalid_dimensions_inputs_and_bias_are_rejected(self):
        cases=[([[1.]],[1.],None,True,2),([[1.]],[1.],None,65,2),([[1.,2.]],[1.,2.],None,1,1),
               ([[True]],[1.],None,1,1),([[1.]],[1.,2.],None,1,2),([[1.]],[1.],[1.,2.],2,2)]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(InputValidationError):expand_array(*args)


if __name__=='__main__':unittest.main()
