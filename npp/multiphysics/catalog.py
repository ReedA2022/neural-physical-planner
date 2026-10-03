"""Application-owned multiphysics device registry and hypothetical reference pack.

Pack data cannot supply executable formulas. All evaluator IDs are versioned and
selected from reviewed Python implementations below. Reference numbers are not
measured technology claims.
"""
from __future__ import annotations
import copy
import json
import math

from .contracts import (TechnologyPack, ComponentContract, SignalContract, BackendDescriptor,
                        EvaluationOutcome, Diagnostic, CostLedger, canonical_hash,
                        canonical_json)
from .backends import fail
from .backends import digital, analog, photonic

_BACKENDS = {('digital.ffn','1.0'):digital,('analog.differential_conductance','1.0'):analog,('photonic.coherent_matrix','1.0'):photonic}


def _known(value,unit):
    return {'state':'known','value':value,'unit':unit}


def _unit(name):
    from .backends import coefficient_unit
    if name.startswith('io.'):
        return photonic.parameter_units()[name]
    if name in photonic.LOCAL_UNITS:
        return photonic.LOCAL_UNITS[name]
    if name in digital.COST_FIELDS|analog.COST_FIELDS:
        return coefficient_unit(name)
    units={
        'macs_per_ns':'MAC/ns','scalar_ops_per_ns':'op/ns','sram_bytes_per_ns':'byte/ns','dram_bytes_per_ns':'byte/ns',
        'link_bytes_per_ns':'byte/ns','program_bytes_per_ns':'byte/ns','launch_ns':'ns','sram_capacity_bytes':'byte',
        'g_min_s':'S','g_max_s':'S','input_scale_v':'V','weight_scale':'1','input_limit':'1','output_current_limit_a':'A',
        'program_bits':'bit','dac_bits':'bit','adc_bits':'bit','programming_noise_relative':'1','drift_per_ns':'1/ns',
        'read_disturbance_relative':'1/read','line_resistance_ohm':'ohm','integration_ns':'ns','programming_ns_per_cell':'ns/cell',
        'dac_ns':'ns','adc_ns':'ns','control_ns':'ns','temperature_min_c':'degC','temperature_max_c':'degC',
    }
    return units[name]


def _parameter(name,value,unit=None):
    return {'name':name,'quantity':copy.deepcopy(value) if isinstance(value,dict) else _known(value,_unit(name) if unit is None else unit),
            'evidence':{'kind':'hypothetical','source':'NPP reference scenario; invented illustrative coefficients',
                        'location':'npp.multiphysics.catalog.reference_pack','version':'1.0',
                        'scope':'theoretical exploration only; no measured calibration or device claim'},
            'uncertainty':{'kind':'epistemic','description':'Chosen reference point; real device uncertainty has not been characterized'},
            'dependencies':[],'instance_group':'reference.v1'}


def _port(name,direction):
    return {'name':name,'direction':direction,'signal':{
        'payload':'tensor','carrier':'electrical','regime':'classical_digital','shape':[1], 'axis_order':['lane'],
        'encoding':'signed_numeric','physical_variable':'digital_word','signed':True,'scale':1.0,'unit':'1',
        'value_range':{'minimum':-1.7976931348623157e308,'maximum':1.7976931348623157e308},'bandwidth_hz':{'state':'unavailable','unit':'Hz','reason':'shape-bound local evaluation; no system clock contract yet'},
        'clock':{'domain':'local','mode':'asynchronous'},
        'lifetime':{'arrival_ns':{'minimum':0.0,'maximum':0.0},'integration_ns':0.0,'retention_ns':_known(0.,'ns'),'reset_required':False},
        'required_services':['power_supply']}}


def _component(family,identifier,backend,values):
    analog_family=family=='analog_electrical'
    return {'id':identifier,'version':'1.0','family':family,
        'mechanism':'differential conductance tile with owned input DAC and differential output ADC' if analog_family else 'serial digital SwiGLU FFN with explicit SRAM, DRAM and interconnect',
        'evaluator_id':backend.BACKEND_ID,'evaluator_version':backend.BACKEND_VERSION,
        'ideal_relation':'y = W x + b; signed bipolar voltages; W proportional to G_positive-G_negative' if analog_family else 'y = W_down (SiLU(W_gate x + b_gate) * (W_up x + b_up)) + b_down',
        'fidelity':'reduced_order_dynamics' if analog_family else 'analytical_cost',
        'parameters':[_parameter(k,v) for k,v in sorted(values.items())],
        'ports':[_port('input','in'),_port('output','out')],
        'operating_envelope':[{'name':'rows','unit':'count','minimum':1.,'maximum':64. if analog_family else 256.},
                              {'name':'columns','unit':'count','minimum':1.,'maximum':64. if analog_family else 256.},
                              {'name':'batch','unit':'count','minimum':1.,'maximum':128. if analog_family else 64.},
                              {'name':'temperature_c','unit':'degC','minimum':0.,'maximum':85.}],
        'integration':{'substrate':'hypothetical_chiplet','package':'abstract single die; no fabrication compatibility claim',
                       'temperature_c':{'minimum':0.,'maximum':85.},'required_services':['power_supply'],
                       'interface_families':['digital_word'],'geometry_assumptions':['lumped local device; package routing is outside M1 local boundary',
                                                                                  'ports are scalar lane templates; concrete vector shapes bind at graph instantiation']},
        'dynamics':{'startup_ns':_known(0.,'ns'),'programming_ns':{'state':'unavailable','unit':'ns','reason':'depends on bound matrix shape; evaluator reports exact local modeled duration'},
                    'settling_ns':_known(values.get('dac_ns',0.),'ns'),'integration_ns':_known(values.get('integration_ns',0.),'ns'),
                    'reset_ns':_known(0.,'ns'),'retention_ns':{'state':'unavailable','unit':'ns','reason':'retention beyond the explicit evaluated age is not guaranteed'},
                    'calibration_ns':_known(0.,'ns')},
        'imperfections':['finite arithmetic/quantization and explicitly declared local analog imperfections' if analog_family else 'explicit IEEE arithmetic; hypothetical costs do not model digital bit faults'],
        'accounting':{'mode':'inclusive','boundary':'local_component','cost_categories':['compute','movement','conversion','storage','programming','calibration','control','support'],
                      'exclusions':['cold chip startup, global clock distribution and package infrastructure are outside the active-component workload boundary'],
                      'amortization':'weight programming once per evaluation; other costs per explicit invocation; no free reuse'},
        'validation':{'reference_implementations':['npp.multiphysics.semantic' if not analog_family else 'independent differential-conductance scalar oracle in tests'],
                      'tests':['tests/test_mp_devices.py'],'known_failures':['out-of-envelope input rejected; unknown cost prevents complete numeric total'],
                      'limitations':['software model consistency only; no external hardware calibration','local duration excludes system resource contention','zero startup/calibration/reset mean excluded operations in this active-component boundary, not claims of free physical processes']}}


def _photonic_component():
    values=photonic.reference_parameters()
    component=_component('analog_electrical','photonic_reference',photonic,values)
    component.update(family='photonic',mechanism='passive signed coherent matrix with selectable explicit encoding and balanced-homodyne readout',
                     ideal_relation='field_out = T field_in, T=W/weight_scale and maximum singular value <= 1; decode_scale absorbs normalization, not physical gain')
    component['operating_envelope'][2]['maximum']=64.
    component['integration']['required_services']=['power_supply','phase_reference','control']
    component['integration']['interface_families']=['digital_word','signed_coherent_field','balanced_homodyne']
    component['dynamics']['settling_ns']=_known(values['settling_ns'],'ns')
    component['dynamics']['integration_ns']=_known(values['integration_ns'],'ns')
    component['dynamics']['calibration_ns']=_known(values['calibration_ns'],'ns')
    component['dynamics']['retention_ns']=_known(values['weight_retention_ns'],'ns')
    component['accounting']['cost_categories']=['compute','movement','conversion','storage','programming','calibration','control','support','source']
    component['validation']['reference_implementations']=['npp.multiphysics.backends.photonic; independently compared at M3, not by declaration']
    component['validation']['tests']=['tests/test_mp_photonic.py','tests/test_mp_interfaces.py']
    component['validation']['limitations']=['hypothetical reduced-order coherent contraction; no physical mesh synthesis or measured calibration',
                                           'field-mode decoded numbers are simulator diagnostics, not a physically available digital readout']
    return component


def reference_pack():
    digital_values={
        'mac_energy_pj':.2,'add_energy_pj':.03,'silu_energy_pj':.5,'multiply_energy_pj':.08,
        'sram_read_energy_pj_per_byte':.04,'sram_write_energy_pj_per_byte':.05,'dram_read_energy_pj_per_byte':2.,
        'link_energy_pj_per_byte':.02,'program_energy_pj_per_byte':.06,'launch_energy_pj':3.,'static_power_mw':.1,
        'compute_area_um2':1000.,'sram_area_um2_per_byte':.1,
        'macs_per_ns':32.,'scalar_ops_per_ns':8.,'sram_bytes_per_ns':64.,'dram_bytes_per_ns':16.,
        'link_bytes_per_ns':64.,'program_bytes_per_ns':16.,'launch_ns':2.,'sram_capacity_bytes':1048576,
    }
    analog_values={
        'g_min_s':1e-6,'g_max_s':1e-4,'input_scale_v':.1,'weight_scale':2.,'input_limit':2.,
        'output_current_limit_a':.01,'program_bits':10,'dac_bits':12,'adc_bits':16,
        'programming_noise_relative':0.,'drift_per_ns':1e-12,'read_disturbance_relative':1e-8,'line_resistance_ohm':1.,
        'integration_ns':10.,'programming_ns_per_cell':1.,'dac_ns':2.,'adc_ns':4.,'control_ns':1.,
        'temperature_min_c':0.,'temperature_max_c':85.,
        'dac_energy_pj':.1,'adc_energy_pj':1.,'programming_energy_pj_per_cell':5.,
        'readout_energy_pj_per_output':.2,'control_energy_pj_per_read':.2,'reference_energy_pj_per_read':.5,
        'bias_add_energy_pj_per_output':.03,'static_power_mw':.01,'cell_area_um2':.1,'converter_area_um2':10.,
    }
    return normalize_pack({'schema_version':'npp-multiphysics-1','id':'hypothetical_reference','version':'1.0',
                          'components':[_component('digital','digital_reference',digital,digital_values),
                                        _component('analog_electrical','analog_reference',analog,analog_values),_photonic_component()],
                          'assumptions':['All numerical reference coefficients are hypothetical and uncalibrated.',
                                         'M1 evaluates local active components; its durations are not an end-to-end schedule.',
                                         'Physical finite quantizers remain enabled in the reference analog component.']})


def parameters_from_component(component):
    c=component if isinstance(component,ComponentContract) else ComponentContract.model_validate_json(canonical_json(component))
    backend=_BACKENDS.get((c.evaluator_id,c.evaluator_version))
    if backend is None:
        fail('technology.evaluator','unregistered evaluator id/version','unavailable_model')
    if backend is photonic:
        photonic.reference_parameters()
    costs=backend.COST_FIELDS
    expected=(backend.PHYSICAL_FIELDS-{'mode'})|costs
    params={p.name:p for p in c.parameters}
    if set(params)!=expected:
        fail('technology.parameters',f'missing {sorted(expected-set(params))}; unknown {sorted(set(params)-expected)}')
    result={}
    for name,p in params.items():
        q=p.quantity
        if q.unit!=_unit(name):
            fail('technology.parameters.'+name,f'expected explicit canonical unit {_unit(name)}')
        if any(v is not None and v<0 for v in (q.value,q.lower,q.upper)) and name not in ('temperature_min_c','phase_bias_rad','io.lo_phase_rad'):
            fail('technology.parameters.'+name,'negative coefficient is not permitted')
        if name in costs:
            result[name]=q.model_dump(mode='json')
        else:
            if q.state!='known':
                fail('technology.parameters.'+name,'this mechanism requires a known operating parameter; unresolved cost coefficients remain supported','unavailable_model')
            result[name]=q.value
            if name in ('program_bits','dac_bits','adc_bits','bias_bits','sram_capacity_bytes') or (name.startswith('io.') and (name.endswith('_bits') or name.endswith('_capacity_bits'))):
                if not q.value.is_integer():
                    fail('technology.parameters.'+name,'requires integer-valued parameter')
                result[name]=int(q.value)
    if backend is analog:
        result['mode']='finite'
    return backend.validate_parameters(result)


def normalize_pack(raw):
    try:
        pack=raw if isinstance(raw,TechnologyPack) else TechnologyPack.model_validate_json(canonical_json(raw))
    except ValueError as exc:
        fail('technology',str(exc))
    for c in pack.components:
        if (c.evaluator_id,c.evaluator_version) not in _BACKENDS:
            fail('technology.evaluator',f'unregistered evaluator {c.evaluator_id}@{c.evaluator_version}','unavailable_model')
        expected={digital.BACKEND_ID:'digital',analog.BACKEND_ID:'analog_electrical',photonic.BACKEND_ID:'photonic'}[c.evaluator_id]
        if c.family!=expected:
            fail('technology.family','component family mismatches registered evaluator')
        params=parameters_from_component(c)
        expected_fidelity='analytical_cost' if expected=='digital' else 'reduced_order_dynamics'
        if c.fidelity!=expected_fidelity:
            fail('technology.fidelity','registered evaluator supports only '+expected_fidelity,'unavailable_model')
        if {port.name for port in c.ports}!={'input','output'}:
            fail('technology.ports','local evaluator requires one input and one output port template')
        for port in c.ports:
            sig=port.signal
            expected_signal=SignalContract.model_validate_json(canonical_json(_port(port.name,port.direction)['signal']))
            if port.direction!=('in' if port.name=='input' else 'out'):
                fail('technology.ports','input and output directions are fixed by this mechanism','domain_mismatch')
            # These evaluators have one precise inclusive boundary. A different
            # clock, variable, encoding, phase/multiplex or disturbance contract
            # requires an adapter/model; it is never decorative metadata.
            actual_contract=sig.model_dump(mode='json',exclude={'value_range','required_services'})
            expected_contract=expected_signal.model_dump(mode='json',exclude={'value_range','required_services'})
            if actual_contract!=expected_contract:
                fail('technology.ports.'+port.name,'unsupported signal metadata: require signed digital_word, unit-scale asynchronous scalar lane template with the declared lifetime/bandwidth model','domain_mismatch')
            if not set(sig.required_services)<=set(c.integration.required_services):
                fail('technology.ports','all signal services must be declared in component integration')
        for field,expected_time in (('startup_ns',0.),('reset_ns',0.),('calibration_ns',params.get('calibration_ns',0.)),
                                    ('settling_ns',params.get('settling_ns',params.get('dac_ns',0.))),
                                    ('integration_ns',params.get('integration_ns',0.))):
            q=getattr(c.dynamics,field)
            if q.state!='known' or q.value!=expected_time:
                fail('technology.dynamics.'+field,'must match this evaluator parameter-derived time '+str(expected_time)+' ns; unsupported metadata cannot override behavior','unavailable_model')
        for field in ('programming_ns','retention_ns'):
            if field=='retention_ns' and expected=='photonic':
                if c.dynamics.retention_ns.state!='known' or c.dynamics.retention_ns.value!=params['weight_retention_ns']:
                    fail('technology.dynamics.retention_ns','must match weight_retention_ns parameter')
                continue
            if getattr(c.dynamics,field).state!='unavailable':
                fail('technology.dynamics.'+field,'programming is shape-derived and retention is not guaranteed by this local model; use the explicit unavailable contract','unavailable_model')
        required_categories={'compute','movement','conversion','storage','programming','calibration','control','support'}|({'source'} if expected=='photonic' else set())
        if set(c.accounting.cost_categories)!=required_categories:
            fail('technology.accounting.cost_categories','must declare the complete evaluator category coverage; absent operations are explicitly excluded/not applicable in the ledger')
        if c.accounting.mode!='inclusive' or c.accounting.boundary!='local_component':
            fail('technology.accounting','this evaluator has a fixed inclusive local-component boundary')
        if {a.name for a in c.operating_envelope}!={'rows','columns','batch','temperature_c'}:
            fail('technology.operating_envelope','this evaluator requires rows, columns, batch and temperature_c envelope axes')
        for axis in c.operating_envelope:
            unit='degC' if axis.name=='temperature_c' else 'count'
            if axis.unit!=unit:
                fail('technology.operating_envelope.'+axis.name,'expected canonical unit '+unit)
            if axis.name!='temperature_c':
                cap=(128 if expected=='analog_electrical' else 64) if axis.name=='batch' else (64 if expected in ('analog_electrical','photonic') else 256)
                if not axis.minimum.is_integer() or not axis.maximum.is_integer() or axis.minimum<1 or axis.maximum>cap:
                    fail('technology.operating_envelope.'+axis.name,'must be integral within the implemented shape envelope 1..'+str(cap))
    return pack.model_dump(mode='json')


def component_for(technology,family):
    pack=normalize_pack(technology)
    family='analog_electrical' if family=='analog' else family
    selected=[c for c in pack['components'] if c['family']==family]
    if len(selected)!=1:
        fail('technology.components',f'exactly one component for family {family!r} is required, found {len(selected)}')
    return copy.deepcopy(selected[0])


def evaluate_model(model,workload,technology=None,family='digital',context=None):
    track=model.get('semantic_track','E') if isinstance(model,dict) else getattr(model,'semantic_track','E')
    if track=='R' or (isinstance(workload,dict) and workload.get('semantic_track')=='R'):
        fail('model.semantic_track','Track R hardware execution requires adaptation accounting and a matched parent-model baseline; metadata alone is unsupported','unavailable_model')
    pack=reference_pack() if technology is None else normalize_pack(technology)
    component=component_for(pack,family)
    params=parameters_from_component(component)
    if not isinstance(workload,dict):
        fail('workload','expected an explicit workload object')
    canonical_json(workload)
    allowed_workload={'inputs','numeric_policy','reuse_count','seed','weight_residency','semantic_track','quality','evidence_policy'}
    if set(workload)-allowed_workload:
        fail('workload',f'unknown workload fields: {sorted(set(workload)-allowed_workload)}')
    if 'inputs' not in workload:
        fail('workload.inputs','inputs are required')
    c=copy.deepcopy(context or {})
    environment_keys={'temperature_c','region','substrate','services'}
    if set(c)&environment_keys:
        if not set(c)<=environment_keys:
            fail('context','pass environment fields together or use nested environment context')
        c={'environment':c}
    env=c.get('environment',{'temperature_c':25.,'region':'reference','substrate':component['integration']['substrate'],
                             'services':list(component['integration']['required_services'])})
    if not isinstance(env,dict) or set(env)!=environment_keys:
        fail('environment','requires temperature_c, region, substrate and services')
    temperature=env['temperature_c']
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature):
        fail('environment.temperature_c','expected finite temperature')
    bounds=component['integration']['temperature_c']
    if not bounds['minimum']<=temperature<=bounds['maximum']:
        fail('environment.temperature_c','outside component integration envelope','envelope_violation')
    if env['substrate']!=component['integration']['substrate']:
        fail('environment.substrate','no declared integration model for requested substrate','domain_mismatch')
    if not isinstance(env['services'],list) or any(not isinstance(v,str) for v in env['services']):
        fail('environment.services','expected explicit service names')
    missing=set(component['integration']['required_services'])-set(env['services'])
    if missing:
        fail('environment.services',f'missing required services: {sorted(missing)}','domain_mismatch')
    c['environment']=copy.deepcopy(env)
    if family in ('analog','analog_electrical'):
        c['temperature_c']=temperature
    policy=workload.get('evidence_policy',['hypothetical'])
    if not isinstance(policy,list) or not policy or any(not isinstance(k,str) for k in policy):
        fail('workload.evidence_policy','expected a nonempty list of evidence kinds')
    kinds={p['evidence']['kind'] for p in component['parameters']}
    if kinds-set(policy):
        fail('workload.evidence_policy',f'component evidence excluded by policy: {sorted(kinds-set(policy))}','unavailable_model')
    if isinstance(model,dict) and 'semantic_track' in workload and model.get('semantic_track','E')!=workload['semantic_track']:
        fail('workload.semantic_track','model and workload tracks disagree')
    for key in ('numeric_policy','reuse_count','seed','weight_residency'):
        if key in workload:
            if key in c and c[key]!=workload[key]:
                fail('context.'+key,'conflicts with workload')
            c[key]=copy.deepcopy(workload[key])
    if family=='digital':
        result=digital.evaluate_ffn_cost(model,workload['inputs'],params,c)
    elif family in ('analog','analog_electrical','photonic'):
        if not isinstance(model,dict) or not {'weights'}<=set(model) or set(model)-{'weights','bias'}:
            fail('model','M1 analog model requires only weights and optional bias; full analog FFN is an M2 composition','unavailable_model')
        if 'numeric_policy' in c or 'weight_residency' in c:
            fail('workload','analog tile precision/residency are set by its own physical contract')
        evaluator=photonic if family=='photonic' else analog
        result=evaluator.evaluate_linear(model['weights'],workload['inputs'],model.get('bias'),params,c)
    else:
        fail('family','unsupported component family','unavailable_model')
    if family=='digital':
        from .semantic import FFNWorkload
        mm=FFNWorkload.from_dict(model) if isinstance(model,dict) else model
        dimensions={'rows':max(mm.hidden_size,mm.output_size),'columns':max(mm.input_size,mm.hidden_size)}
    else:
        dimensions={'rows':len(model['weights']),'columns':len(model['weights'][0])}
    xx=workload['inputs']
    dimensions.update(batch=len(xx) if isinstance(xx[0],list) else 1,temperature_c=temperature)
    for axis in component['operating_envelope']:
        if not axis['minimum']<=dimensions[axis['name']]<=axis['maximum']:
            fail('technology.operating_envelope.'+axis['name'],'bound workload is outside the declared component envelope','envelope_violation')
    for port in component['ports']:
        values=workload['inputs'] if port['name']=='input' else result['actual_output']
        if family=='photonic' and isinstance(values,dict):
            fail('workload.inputs','catalog component wrapper accepts digital ports; use explicit photonic field API in the typed implementation graph','domain_mismatch')
        if values is None:
            fail('context.output_mode','catalog component wrapper requires a digital output; field ports require typed graph instantiation','domain_mismatch')
        def scalar_values(tree):
            if isinstance(tree,list):
                for child in tree:
                    yield from scalar_values(child)
            else:
                yield tree
        bounds=port['signal']['value_range']
        if any(not bounds['minimum']<=value<=bounds['maximum'] for value in scalar_values(values)):
            fail('technology.ports.'+port['name']+'.value_range','actual signal is outside the declared input/output envelope','envelope_violation')
    quality=workload.get('quality')
    if quality is not None:
        if not isinstance(quality,dict) or set(quality)!={'max_rms_error'}:
            fail('workload.quality','requires max_rms_error')
        limit=quality['max_rms_error']
        if isinstance(limit,bool) or not isinstance(limit,(float,int)) or not math.isfinite(limit) or limit<0:
            fail('workload.quality.max_rms_error','expected finite nonnegative budget')
        passed=result['error']['rms']<=limit
        result['quality']={'max_rms_error':limit,'observed_rms_error':result['error']['rms'],'passed':passed,
                           'scope':'provided inputs versus this supplied model; not held-out task accuracy'}
        if not passed:
            result['status']='resource_infeasible'
            result['diagnostics']=[{'code':'quality_budget_exceeded','path':'workload.quality','message':'observed modeled RMS error exceeds declared budget'}]
    result['technology']={'pack_id':pack['id'],'pack_version':pack['version'],'pack_hash':canonical_hash(pack),
                          'component_id':component['id'],'component_version':component['version'],
                          'evidence_kinds':sorted({p['evidence']['kind'] for p in component['parameters']})}
    result['dependencies']['pack_hash']=canonical_hash(pack)
    result['dependencies']['component_hash']=canonical_hash(component)
    result['dependencies']['cache_key']=canonical_hash(result['dependencies'])
    return result


class RegisteredBackend:
    """JSON snapshot lifecycle; no mutable global or other backend state access.

    instantiate returns a private snapshot. evaluate/sample take that snapshot
    and a context containing ``request`` and optional ``evaluation_context``.
    advance returns a new snapshot and never mutates its predecessor.
    """
    def __init__(self,evaluator_id,version='1.0'):
        if (evaluator_id,version) not in _BACKENDS:
            raise ValueError('unregistered evaluator id/version')
        self._backend=_BACKENDS[(evaluator_id,version)]

    def describe(self):
        analog_family=self._backend is analog
        return BackendDescriptor(id=self._backend.BACKEND_ID,version=self._backend.BACKEND_VERSION,
                                 families=({analog:'analog_electrical',digital:'digital',photonic:'photonic'}[self._backend],),
                                 capabilities=('linear','sample_programming_noise','advance_weight_age') if analog_family else (('coherent_linear','sample_reference_phase','field_passthrough') if self._backend is photonic else ('complete_ffn',)),
                                 limitations=('hypothetical local model; no independently calibrated physical costs',
                                              'independent device comparison not available at M1; unsupported is explicit'))

    def _failure(self,exc):
        code='invalid_input'
        if hasattr(exc,'diagnostics'):
            code=exc.diagnostics[0].get('code',code)
        status={'unavailable_model':'unsupported','numerical_failure':'numerical_failure','resource_infeasibility':'resource_infeasible'}.get(code,'invalid')
        return EvaluationOutcome(status=status,diagnostics=(Diagnostic(code=code,path='backend',message=str(exc)[:4000]),))

    def _payload(self,value,status='model_feasible',ledger=None,obligations=()):
        diagnostics=tuple(Diagnostic(**d) for d in value.get('diagnostics',[])) if isinstance(value,dict) else ()
        return EvaluationOutcome(status=status,diagnostics=diagnostics,obligations=tuple(obligations),ledger=ledger,payload_json=canonical_json(value))

    def validate(self,component,context_json):
        try:
            c=component if isinstance(component,ComponentContract) else ComponentContract.model_validate_json(canonical_json(component))
            context=json.loads(context_json)
            canonical_json(context)
            if (c.evaluator_id,c.evaluator_version)!=(self._backend.BACKEND_ID,self._backend.BACKEND_VERSION):
                fail('component','backend cannot validate another evaluator instance','domain_mismatch')
            normalize_pack({'schema_version':'npp-multiphysics-1','id':'lifecycle_validation','version':'1.0','components':[c.model_dump(mode='json')],'assumptions':[]})
            p=parameters_from_component(c)
            return self._payload({'component_id':c.id,'parameters':p,'validated_context':context})
        except (ValueError,TypeError,OverflowError) as exc:
            return self._failure(exc)

    def instantiate(self,component,context_json):
        outcome=self.validate(component,context_json)
        if outcome.status!='model_feasible':
            return outcome
        c=component if isinstance(component,ComponentContract) else ComponentContract.model_validate_json(canonical_json(component))
        try:
            context=json.loads(context_json)
            if not isinstance(context,dict) or set(context)-{'owner_id'}:
                fail('instantiate.context','only owner_id is accepted; inputs belong to evaluation context')
            owner=context.get('owner_id',c.id)
            if not isinstance(owner,str) or not owner:
                fail('instantiate.owner_id','expected nonempty owner identifier')
            return self._payload({'backend_id':self._backend.BACKEND_ID,'backend_version':self._backend.BACKEND_VERSION,
                                  'component':c.model_dump(mode='json'),'state':{'owner_id':owner,'version':0,'age_ns':0.,'read_count':0,'weight_identity':None}})
        except (ValueError,TypeError) as exc:
            return self._failure(exc)

    def _instance(self,instance_json):
        instance=json.loads(instance_json)
        canonical_json(instance)
        if not isinstance(instance,dict) or set(instance)!={'backend_id','backend_version','component','state'}:
            fail('instance','invalid instance snapshot')
        if (instance['backend_id'],instance['backend_version'])!=(self._backend.BACKEND_ID,self._backend.BACKEND_VERSION):
            fail('instance','backend cannot access another evaluator state','domain_mismatch')
        c=ComponentContract.model_validate_json(canonical_json(instance['component']))
        if (c.evaluator_id,c.evaluator_version)!=(self._backend.BACKEND_ID,self._backend.BACKEND_VERSION):
            fail('instance.component','component/evaluator identity mismatch','domain_mismatch')
        state=instance['state']
        if not isinstance(state,dict) or set(state)!={'owner_id','version','age_ns','read_count','weight_identity'}:
            fail('instance.state','invalid state contract')
        for field in ('version','read_count'):
            if type(state[field]) is not int or state[field]<0:
                fail('instance.state.'+field,'expected nonnegative integer')
        if isinstance(state['age_ns'],bool) or not isinstance(state['age_ns'],(float,int)) or not math.isfinite(state['age_ns']) or state['age_ns']<0:
            fail('instance.state.age_ns','expected nonnegative finite age')
        identity=state['weight_identity']
        if identity is not None and (not isinstance(identity,str) or len(identity)!=64 or any(ch not in '0123456789abcdef' for ch in identity)):
            fail('instance.state.weight_identity','expected null or canonical weight digest')
        normalize_pack({'schema_version':'npp-multiphysics-1','id':'lifecycle_validation','version':'1.0','components':[c.model_dump(mode='json')],'assumptions':[]})
        return instance,parameters_from_component(c)

    def evaluate(self,instance_json,context_json):
        try:
            instance,p=self._instance(instance_json)
            context=json.loads(context_json)
            if not isinstance(context,dict) or set(context)-{'request','evaluation_context'} or 'request' not in context:
                fail('context','requires request and optional evaluation_context')
            request=context['request']
            extra=copy.deepcopy(context.get('evaluation_context',{}))
            if not isinstance(extra,dict):
                fail('evaluation_context','expected object')
            extra['owner_id']=instance['state']['owner_id']
            if self._backend in (analog,photonic):
                if not isinstance(request,dict) or set(request)-{'weights','inputs','bias'} or not {'weights','inputs'}<=set(request):
                    fail('request','analog request requires weights, inputs and optional bias')
                if set(extra)&{'age_ns','read_count'}:
                    fail('evaluation_context','state age/read count are owned by instance; use advance')
                if self._backend is analog:
                    extra.update(age_ns=instance['state']['age_ns'],read_count=instance['state']['read_count'])
                model={'weights':request['weights'],'bias':request.get('bias')}
                family='analog' if self._backend is analog else 'photonic'
            else:
                if not isinstance(request,dict) or set(request)!={'model','inputs'}:
                    fail('request','digital request requires model and inputs')
                model=request['model']
                family='digital'
            workload={'inputs':request['inputs']}
            for field in ('numeric_policy','reuse_count','seed','weight_residency','semantic_track','quality','evidence_policy'):
                if field in extra:
                    workload[field]=extra.pop(field)
            pack={'schema_version':'npp-multiphysics-1','id':'lifecycle_evaluation','version':'1.0','components':[instance['component']],'assumptions':[]}
            result=evaluate_model(model,workload,pack,family=family,context=extra)
            result_identity=result['state'].get('weight_identity',result['state'].get('weight_version'))
            if isinstance(result_identity,str) and result_identity.startswith('sha256:'):
                result_identity=result_identity[len('sha256:'):]
            if not isinstance(result_identity,str) or len(result_identity)!=64 or any(ch not in '0123456789abcdef' for ch in result_identity):
                fail('result.state.weight_identity','backend produced an invalid canonical weight digest','numerical_failure')
            old_identity=instance['state']['weight_identity']
            if old_identity is not None and old_identity!=result_identity:
                fail('request.weights','weights or programming seed differ from owned state; instantiate a newly programmed state','domain_mismatch')
            next_instance=copy.deepcopy(instance)
            next_instance['state']['weight_identity']=result_identity
            if old_identity is None:
                next_instance['state']['version']+=1
            result['next_instance']=next_instance
            result['state_evolution']='evaluate is pure; bind returned next_instance then advance explicitly for reads and elapsed time'
            ledger=CostLedger.model_validate_json(canonical_json(result['ledger']['contract']))
            return self._payload(result,result['status'],ledger,result['ledger']['obligations'])
        except (ValueError,TypeError,OverflowError,KeyError) as exc:
            return self._failure(exc)

    def advance(self,state_json,context_json):
        if self._backend is not analog:
            return self._unsupported('digital dynamic evolution is not modeled; explicit workload invocation handles costs')
        try:
            instance,_=self._instance(state_json)
            context=json.loads(context_json)
            if not isinstance(context,dict) or set(context)-{'delta_ns','reads'} or 'delta_ns' not in context:
                fail('advance','requires delta_ns and optional reads')
            delta=context['delta_ns']; reads=context.get('reads',0)
            if isinstance(delta,bool) or not isinstance(delta,(int,float)) or not math.isfinite(delta) or delta<0:
                fail('advance.delta_ns','expected nonnegative finite time')
            if type(reads) is not int or reads<0:
                fail('advance.reads','expected nonnegative integer')
            instance['state']['age_ns']+=delta
            instance['state']['read_count']+=reads
            instance['state']['version']+=1
            if instance['state']['age_ns']>1e15 or instance['state']['read_count']>10**9:
                fail('advance','state exceeds supported age/read envelope','envelope_violation')
            return self._payload(instance)
        except (ValueError,TypeError,OverflowError) as exc:
            return self._failure(exc)

    def sample(self,instance_json,context_json):
        if self._backend not in (analog,photonic):
            return self._unsupported('digital reference has no declared stochastic device model')
        try:
            context=json.loads(context_json)
            if not isinstance(context,dict) or not isinstance(context.get('evaluation_context',{}),dict):
                fail('sample.context','context and evaluation_context must be objects')
            if 'seed' not in context.get('evaluation_context',{}):
                fail('sample','physical sampling requires an explicit seed')
            return self.evaluate(instance_json,context_json)
        except (ValueError,TypeError) as exc:
            return self._failure(exc)

    def export(self,instance_json,context_json):
        try:
            instance,p=self._instance(instance_json)
            context=json.loads(context_json)
            if not isinstance(context,dict):
                fail('export.context','context must be an object')
            return self._payload({'schema':'npp-device-reference-input-1','instance':instance,'parameters':p,'context':context,
                                  'scope':'independent adapter input; exporting is not a completed comparison'})
        except (ValueError,TypeError) as exc:
            return self._failure(exc)

    def _unsupported(self,message):
        return EvaluationOutcome(status='unsupported',diagnostics=(Diagnostic(code='unavailable_model',path='backend',message=message),))

    def compare(self,instance_json,reference_json,context_json):
        return self._unsupported('M1 has no independent device adapter comparison; replay/reference reuse must not be reported as independent validation')


def registered_backends():
    return {key:RegisteredBackend(*key) for key in _BACKENDS}
