import unittest
from tests.test_mp_graph import project
from npp.multiphysics.graph import default_implementation,evaluate_implementation
from npp.multiphysics.catalog import parameters_from_component

class FixedPoolShapeTests(unittest.TestCase):
    def _record(self,family):
        p=project()
        p['model'].update(W_up=[[.1],[.2],[-.1],[.3]],W_gate=[[.1],[.1],[.1],[.1]],
                          W_down=[[.2,.1,-.1,.1]],b_up=[0.]*4,b_gate=[0.]*4,b_down=[0.])
        p['workload']['inputs']=[[.2],[-.3]]
        return p,evaluate_implementation(p,default_implementation(p['model'],family))

    def test_analog_cross_oriented_tiles_install_rectangular_cells_and_io(self):
        p,r=self._record('analog')
        c=next(c for c in p['technology']['components'] if c['family']=='analog_electrical')
        params=parameters_from_component(c);pool=r['pool_provisioning']['analog.'+c['id']]
        self.assertEqual((pool['matrix_shape']['rows'],pool['matrix_shape']['columns']),(4,4))
        self.assertAlmostEqual(pool['area_effects']['cells']['value'],32*params['cell_area_um2']['value'])
        self.assertAlmostEqual(pool['area_effects']['converters']['value'],8*params['converter_area_um2']['value'])
        self.assertEqual(r['status'],'model_feasible')

    def test_photonic_rectangular_cells_hold_power_and_separate_io_maxima(self):
        p,r=self._record('photonic')
        c=next(c for c in p['technology']['components'] if c['family']=='photonic')
        params=parameters_from_component(c);pool=r['pool_provisioning']['photonic.'+c['id']]
        self.assertEqual((pool['matrix_shape']['rows'],pool['matrix_shape']['columns']),(4,4))
        self.assertAlmostEqual(pool['area_effects']['matrix.matrix_elements']['value'],16*params['element_area_um2']['value'])
        self.assertAlmostEqual(pool['static_power_effects']['matrix.phase_holding']['value'],16*params['holding_power_mw_per_element']['value'])
        self.assertAlmostEqual(pool['area_effects']['input.converter']['value'],4*params['io.converter_area_um2']['value'])
        self.assertAlmostEqual(pool['area_effects']['output.converter']['value'],4*params['io.converter_area_um2']['value'])
        self.assertEqual(r['status'],'model_feasible')

if __name__=='__main__':unittest.main()
