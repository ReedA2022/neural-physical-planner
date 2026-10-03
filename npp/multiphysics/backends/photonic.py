"""Bounded signed coherent matrix contraction with explicit classical interfaces.

The matrix is a passive complex field transfer, not an intensity matrix. Source,
encoding and homodyne readout are paid only when selected at this boundary.
Finite coefficient programming is projected onto the passive contraction set;
this reduced-order model does not synthesize a fabricated interferometer mesh.
"""
from __future__ import annotations
import copy
import hashlib
import math
import re

import numpy as np

from . import build_ledger, coefficient_unit, cost_quantity, error_record, evaluation_identity, fail
from ..contracts import CostEntry, CostLedger, AccountingOwner, Quantity, canonical_json

BACKEND_ID='photonic.coherent_matrix'
BACKEND_VERSION='1.0'
LOCAL_PHYSICAL_FIELDS={'weight_scale','program_bits','insertion_loss_db','phase_bias_rad','phase_noise_std_rad',
                       'propagation_ns','integration_ns','programming_ns_per_element','calibration_ns','settling_ns','weight_retention_ns',
                       'bias_limit','bias_bits','bias_driver_ns','bias_combiner_ns','bias_add_ns','wallplug_efficiency','temperature_min_c','temperature_max_c'}
LOCAL_COST_FIELDS={'programming_energy_pj_per_element','calibration_energy_pj','holding_power_mw_per_element',
                   'control_energy_pj_per_read','bias_driver_energy_pj_per_output','bias_add_energy_pj_per_output',
                   'element_area_um2','bias_combiner_area_um2'}
LOCAL_UNITS={'weight_scale':'1','program_bits':'bit','insertion_loss_db':'dB','phase_bias_rad':'rad','phase_noise_std_rad':'rad',
             'propagation_ns':'ns','integration_ns':'ns','programming_ns_per_element':'ns/element','calibration_ns':'ns','settling_ns':'ns','weight_retention_ns':'ns',
             'bias_limit':'1','bias_bits':'bit','bias_driver_ns':'ns','bias_combiner_ns':'ns','bias_add_ns':'ns','wallplug_efficiency':'1','temperature_min_c':'degC','temperature_max_c':'degC',
             'programming_energy_pj_per_element':'pJ/element','calibration_energy_pj':'pJ/calibration','holding_power_mw_per_element':'mW/element',
             'control_energy_pj_per_read':'pJ/read','bias_driver_energy_pj_per_output':'pJ/output','bias_add_energy_pj_per_output':'pJ/output',
             'element_area_um2':'um^2/element','bias_combiner_area_um2':'um^2/output'}
# Populated once the public interface module is imported, never by a pack.
PHYSICAL_FIELDS=set(LOCAL_PHYSICAL_FIELDS)
COST_FIELDS=set(LOCAL_COST_FIELDS)


def parameter_units():
    from ..interfaces import PARAMETER_UNITS
    return {**LOCAL_UNITS,**{'io.'+key:unit for key,unit in PARAMETER_UNITS.items()}}


def reference_parameters():
    from ..interfaces import reference_parameters as interface_parameters
    local={'weight_scale':8.,'program_bits':16,'insertion_loss_db':.1,'phase_bias_rad':0.,'phase_noise_std_rad':0.,
           'propagation_ns':.1,'integration_ns':1.,'programming_ns_per_element':.2,'calibration_ns':2.,'settling_ns':.2,'weight_retention_ns':1e6,
           'bias_limit':2.,'bias_bits':16,'bias_driver_ns':.2,'bias_combiner_ns':.1,'bias_add_ns':.1,'wallplug_efficiency':.2,'temperature_min_c':0.,'temperature_max_c':85.,
           'programming_energy_pj_per_element':.2,'calibration_energy_pj':2.,'holding_power_mw_per_element':.001,
           'control_energy_pj_per_read':.1,'bias_driver_energy_pj_per_output':.05,'bias_add_energy_pj_per_output':.03,
           'element_area_um2':4.,'bias_combiner_area_um2':2.}
    local={key:({'state':'known','value':value,'unit':LOCAL_UNITS[key]} if key in LOCAL_COST_FIELDS else value) for key,value in local.items()}
    interface=interface_parameters()
    COST_FIELDS.update('io.'+k for k,v in interface.items() if isinstance(v,dict))
    PHYSICAL_FIELDS.update('io.'+k for k,v in interface.items() if not isinstance(v,dict))
    return {**local,**{'io.'+key:value for key,value in interface.items()}}


def validate_parameters(parameters):
    reference=reference_parameters()
    if not isinstance(parameters,dict) or set(parameters)!=set(reference):
        fail('parameters','photonic model requires every declared local and io coefficient, with no unknown fields')
    p=copy.deepcopy(parameters)
    for key in LOCAL_PHYSICAL_FIELDS:
        value=p[key]
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            fail('parameters.'+key,'expected a finite real parameter')
        if key in ('program_bits','bias_bits'):
            if type(value) is not int or not 2<=value<=24:
                fail('parameters.'+key,'precision must be an integer in [2,24]')
        elif key not in ('temperature_min_c','temperature_max_c','phase_bias_rad') and value<0:
            fail('parameters.'+key,'must be nonnegative')
        if abs(value)>1e12:
            fail('parameters.'+key,'outside supported numerical envelope','envelope_violation')
    if not 1e-12<=p['weight_scale']<=1e12 or not 1e-12<=p['bias_limit']<=1e12 or not 0<p['wallplug_efficiency']<=1:
        fail('parameters','positive weight/bias scales and efficiency in (0,1] are required')
    if p['weight_retention_ns']<=0 or p['integration_ns']<=0 or p['temperature_min_c'] < -273.15 or p['temperature_min_c']>p['temperature_max_c']:
        fail('parameters','invalid integration time or temperature envelope')
    for key in LOCAL_COST_FIELDS:
        p[key]={name:value for name,value in cost_quantity(p[key],LOCAL_UNITS[key],'parameters.'+key).items() if value is not None}
    from ..interfaces import validate_parameters as validate_interfaces
    normalized_io=validate_interfaces({key[3:]:value for key,value in p.items() if key.startswith('io.')})
    p.update({'io.'+key:value for key,value in normalized_io.items()})
    return p


def _matrix(value,name,ranks=(1,2)):
    try:
        raw=np.asarray(value,dtype=object)
        if raw.ndim not in ranks or not all(1<=axis<=64 for axis in raw.shape):
            fail(name,'supported tensor axes are 1..64 with expected rank '+str(ranks),'envelope_violation')
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or np.iscomplexobj(v) for v in raw.flat):
            fail(name,'requires finite real values without bool/string/complex coercion')
        result=raw.astype(float)
        if not np.isfinite(result).all():
            fail(name,'nonfinite values are unsupported','numerical_failure')
        return result
    except (TypeError,ValueError,OverflowError) as exc:
        if hasattr(exc,'diagnostics'): raise
        fail(name,'invalid real tensor: '+str(exc))


def _combined_ledger(parts,duration):
    owners=[];entries=[];details=[];areas=[];coverage={}
    for label,record in parts:
        source=CostLedger.model_validate_json(canonical_json(record['contract']))
        owners.extend(AccountingOwner(id=label+'.'+owner.id,mode='leaf') for owner in source.owners)
        for entry in source.entries:
            entries.append(CostEntry(contribution_id=label+'.'+entry.contribution_id,owner=label+'.'+entry.owner,
                                     boundary='photonic_local',category=entry.category,metric=entry.metric,quantity=entry.quantity,basis=entry.basis))
        for row in record.get('entries',[]):
            item=copy.deepcopy(row);item['owner_id']=label+'.'+item['owner_id'];item['effect_id']=label+'.'+item['effect_id'];details.append(item)
        for row in record.get('area_entries',[]):
            item=copy.deepcopy(row);item['owner_id']=label+'.'+item['owner_id'];item['effect_id']=label+'.'+item['effect_id'];areas.append(item)
        coverage[label]=record.get('category_coverage',{})
    ledger=CostLedger(boundary='photonic_local',owners=tuple(owners),entries=tuple(entries),expected_contributions=tuple(e.contribution_id for e in entries))
    energy=ledger.aggregate('energy_pj','pJ');area=ledger.aggregate('area_um2','um^2')
    complete=energy.complete and area.complete
    return {'contract':ledger.model_dump(mode='json'),'energy_pj':energy.quantity.model_dump(mode='json',exclude_none=True),
            'area_um2':area.quantity.model_dump(mode='json',exclude_none=True),'duration_ns':{'state':'known','value':duration,'unit':'ns'},
            'complete':complete,'entries':details,'area_entries':areas,'category_coverage':coverage,
            'obligations':[] if complete else ['bounded or unavailable local/interface coefficients prevent a point-valued complete total']}


def evaluate_linear(weights,inputs,bias,parameters,context=None):
    p=validate_parameters(parameters);c=copy.deepcopy({} if context is None else context)
    allowed={'input_mode','output_mode','seed','reference_id','owner_id','environment','geometry','schedule','reuse_count','mode','input_source_owner','weight_age_ns'}
    if not isinstance(c,dict) or set(c)-allowed:
        fail('context','unsupported photonic context fields')
    for name,default in (('input_mode','digital'),('output_mode','digital'),('seed',0),('owner_id','photonic_tile'),('reference_id','coherent_reference'),('reuse_count',1),('mode','finite'),('weight_age_ns',0.)):
        c.setdefault(name,default)
    if c['input_mode'] not in ('digital','field') or c['output_mode'] not in ('digital','field') or c['mode'] not in ('finite','ideal_diagnostic'):
        fail('context','input/output modes are digital or field; precision mode finite or ideal_diagnostic')
    if type(c['seed']) is not int or not 0<=c['seed']<2**32 or type(c['reuse_count']) is not int or c['reuse_count']!=1:
        fail('context','seed must be uint32; M2 field evaluation uses one explicit invocation (schedule repeated invocations explicitly)')
    if any(not isinstance(c[key],str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.:-]{0,159}',c[key]) for key in ('owner_id','reference_id')):
        fail('context','owner/reference identities must be valid explicit identifiers')
    environment=c.get('environment',{})
    if not isinstance(environment,dict) or set(environment)-{'temperature_c','region','substrate','services'}:
        fail('environment','expected an environment mapping with declared temperature, region, substrate and services')
    for name in ('region','substrate'):
        if name in environment and (not isinstance(environment[name],str) or not environment[name]):
            fail('environment.'+name,'expected nonempty name')
    if 'services' in environment:
        services=environment['services']
        if not isinstance(services,list) or any(not isinstance(v,str) or not v for v in services) or len(set(services))!=len(services):
            fail('environment.services','expected unique explicit service names')
        if not {'power_supply','phase_reference','control'}<=set(services):
            fail('environment.services','coherent matrix requires power_supply, phase_reference and control','domain_mismatch')
    age=c['weight_age_ns']
    if isinstance(age,bool) or not isinstance(age,(int,float)) or not math.isfinite(age) or age<0:
        fail('context.weight_age_ns','expected finite nonnegative age')
    if age>p['weight_retention_ns']:
        fail('context.weight_age_ns','programmed coherent weights exceed supported retention; recalibration/programming required','envelope_violation')
    temperature=environment.get('temperature_c',25.)
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or not p['temperature_min_c']<=temperature<=p['temperature_max_c']:
        fail('environment.temperature_c','outside photonic temperature envelope','envelope_violation')
    w=_matrix(weights,'weights',(2,));outdim,indim=w.shape
    b=np.zeros(outdim) if bias is None else _matrix(bias,'bias',(1,))
    if b.shape!=(outdim,): fail('bias','bias size must match matrix rows')
    if np.max(np.abs(b))>p['bias_limit']:fail('bias','bias exceeds declared encoder envelope','envelope_violation')
    singular=float(np.linalg.norm(w,2))
    if not math.isfinite(singular) or singular>p['weight_scale']*(1+1e-12):
        fail('weights','signed coherent matrix must be a passive contraction after explicit weight_scale; choose a sufficiently large scale','envelope_violation')
    io={key[3:]:value for key,value in p.items() if key.startswith('io.')}
    from ..interfaces import encode,readout,validate_field
    parts=[];input_duration=0.;output_duration=0.
    ictx={'owner_id':c['owner_id']+'.input','reference_id':c['reference_id'],'mode':c['mode']}
    if c['input_mode']=='digital':
        x=_matrix(inputs,'inputs')
        encoded=encode(x.tolist(),io,ictx)
        field=encoded['field'];parts.append(('input',encoded['ledger']));input_duration=encoded['duration_ns']
    else:
        field=validate_field(inputs,io)
        if field['reference_id']!=c['reference_id']:
            fail('inputs.reference_id','field and local phase reference differ; an explicit phase-lock/interface model is required','domain_mismatch')
        x=np.asarray(field['real'],dtype=float)*field['decode_scale']
    real=np.asarray(field['real'],dtype=float);imag=np.asarray(field['imag'],dtype=float)
    a=real+1j*imag;vector=a.ndim==1
    if vector:a=a[None,:];x=x[None,:]
    if a.ndim!=2 or a.shape[1]!=indim or a.shape[0]>64:
        fail('inputs','field width must match matrix columns and batch is bounded to 64')
    scale=float(field['decode_scale'])
    t=w/p['weight_scale']
    if c['mode']=='finite':
        levels=2**(p['program_bits']-1)-1
        t=np.rint(t*levels)/levels
        salt=int.from_bytes(hashlib.sha256(c['reference_id'].encode()).digest()[:4],'big')
        phase=p['phase_bias_rad']+np.random.default_rng(np.random.SeedSequence([c['seed'],salt])).normal(0,p['phase_noise_std_rad'])
        t=t.astype(complex)*np.exp(1j*phase)
        norm=float(np.linalg.norm(t,2))
        t=t/max(1.,norm)
        amplitude_loss=10**(-p['insertion_loss_db']/20.)
    else:
        if p['phase_bias_rad']!=0 or p['phase_noise_std_rad']!=0 or p['insertion_loss_db']!=0:
            fail('context.mode','ideal_diagnostic requires explicit zero phase imperfections and zero insertion loss')
        amplitude_loss=1.
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            transfer=t*amplitude_loss
            output=a@transfer.T
            decoded_scale=scale*p['weight_scale']
            matrix_output=output.copy();matrix_scale=decoded_scale
            injected=None;terminated=None
            ideal=x@w.T+b
            bias_energy=0.;bias_power=0.;terminated_power=0.;bias_active=bias is not None and bool(np.any(b))
            if c['output_mode']=='field' and bias_active:
                levels=2**(p['bias_bits']-1)-1
                physical_bias=b if c['mode']=='ideal_diagnostic' else np.rint(b/p['bias_limit']*levels)/levels*p['bias_limit']
                injected=physical_bias/decoded_scale
                bias_power=float(np.sum(injected**2))*a.shape[0]
                bias_energy=bias_power*p['integration_ns']/p['wallplug_efficiency']
                terminated=(output-injected)/math.sqrt(2.)
                terminated_power=float(np.sum(np.abs(terminated)**2))
                output=(output+injected)/math.sqrt(2.)
                decoded_scale*=math.sqrt(2.)
            if not np.isfinite(output).all() or not math.isfinite(decoded_scale) or decoded_scale<=0:
                fail('evaluation','nonfinite coherent field result','numerical_failure')
    except (FloatingPointError,OverflowError,ZeroDivisionError) as exc:
        fail('evaluation','coherent matrix numerical failure: '+str(exc),'numerical_failure')
    serial=lambda value:(value[0] if vector else value).tolist()
    outgoing={'carrier':'optical','regime':'classical_analog','encoding':'signed_coherent_field',
              'real':serial(output.real),'imag':serial(output.imag),'decode_scale':decoded_scale,'reference_id':c['reference_id']}
    outgoing=validate_field(outgoing,io)
    counts=a.shape[0];cells=w.size
    times={'programming_ns':p['programming_ns_per_element']*cells,'calibration_ns':p['calibration_ns'],
           'settling_ns':p['settling_ns'],'read_ns':(p['propagation_ns']+p['integration_ns'])*counts,
           'input_conversion_ns':input_duration,'output_conversion_ns':0.,
           'bias_ns':((p['bias_driver_ns']+p['bias_combiner_ns']) if c['output_mode']=='field' and bias_active else (p['bias_add_ns'] if c['output_mode']=='digital' and bias is not None else 0.))*counts}
    actual=None;readout_before_bias=None
    if c['output_mode']=='digital':
        decoded=readout(outgoing,io,{'owner_id':c['owner_id']+'.output','reference_id':c['reference_id'],'mode':c['mode']})
        actual=np.asarray(decoded['values'],dtype=float)
        if vector:actual=actual[None,:]
        readout_before_bias=actual.copy()
        actual=actual+b
        parts.append(('output',decoded['ledger']));output_duration=decoded['duration_ns'];times['output_conversion_ns']=output_duration
    diagnostic=output.real*decoded_scale if actual is None else actual
    matrix_duration=math.fsum(times[k] for k in ('programming_ns','calibration_ns','settling_ns','read_ns','bias_ns'))
    local_entries=[('compute','passive_field_transfer',{'state':'known','value':0.,'unit':'pJ'},counts),
                   ('programming','matrix_programming',p['programming_energy_pj_per_element'],cells),
                   ('calibration','phase_calibration',p['calibration_energy_pj'],1),
                   ('control','read_control',p['control_energy_pj_per_read'],counts),
                   ('support','phase_holding',p['holding_power_mw_per_element'],cells*matrix_duration),
                   ('source','coherent_bias_source',{'state':'known','value':bias_energy,'unit':'pJ'},1),
                   ('conversion','coherent_bias_driver',p['bias_driver_energy_pj_per_output'],outdim*counts if c['output_mode']=='field' and bias_active else 0),
                   ('compute','digital_bias_addition',p['bias_add_energy_pj_per_output'],outdim*counts if c['output_mode']=='digital' and bias is not None else 0),
                   ('movement','passive_optical_transport',{'state':'known','value':0.,'unit':'pJ'},counts)]
    local=build_ledger(c['owner_id'],local_entries,matrix_duration,[('matrix_elements',p['element_area_um2'],cells),('bias_combiners',p['bias_combiner_area_um2'],outdim if c['output_mode']=='field' and bias_active else 0)])
    local['category_coverage']['storage']={'status':'covered_by_other_contributions','contribution_ids':['matrix_programming','phase_holding','area.matrix_elements'],
                                          'reason':'configured matrix state is held by the programmed physical matrix elements and their holding power; no duplicate storage charge'}
    parts.append(('matrix',local));duration=math.fsum(times.values());ledger=_combined_ledger(parts,duration)
    return {'backend_id':BACKEND_ID,'backend_version':BACKEND_VERSION,'status':'model_feasible' if ledger['complete'] else 'conditional',
            'ideal_output':serial(ideal),'actual_output':None if actual is None else serial(actual),
            'field':outgoing if c['output_mode']=='field' else None,'diagnostic_decoded_output':serial(diagnostic),
            'diagnostic_scope':'decoded simulation values are not an available classical readout when output_mode=field',
            'error':error_record(ideal,diagnostic,mode=c['mode']),'ledger':ledger,'local_duration_ns':duration,'duration_ns':duration,
            'duration_breakdown':times,'schedule_status':'local_serial_model_not_system_schedule',
            'resources':{'matrix_elements':cells,'field_input_lanes':indim,'field_output_lanes':outdim,'phase_reference':c['reference_id']},
            'physical':{'input_field':copy.deepcopy(field),'matrix_field':{'carrier':'optical','regime':'classical_analog','encoding':'signed_coherent_field','real':serial(matrix_output.real),'imag':serial(matrix_output.imag),'decode_scale':matrix_scale,'reference_id':c['reference_id']},
                        'output_field':copy.deepcopy(outgoing),'bias':b.tolist(),'bias_present':bias is not None,
                        'bias_injected_real':None if injected is None else np.broadcast_to(injected,output.shape).tolist(),
                        'terminated_real':None if terminated is None else serial(terminated.real),'terminated_imag':None if terminated is None else serial(terminated.imag),
                        'readout_before_bias':None if readout_before_bias is None else serial(readout_before_bias),
                        'transfer_real':transfer.real.tolist(),'transfer_imag':transfer.imag.tolist(),
                        'maximum_singular_value':float(np.linalg.norm(transfer,2)),'input_power_mw':float(np.sum(np.abs(a)**2)),
                        'matrix_output_power_mw':float(np.sum(np.abs(a@transfer.T)**2)),
                        'bias_injection_energy_pj':bias_energy,'bias_injected_power_mw':bias_power,'terminated_power_mw':terminated_power,
                        'final_field_power_mw':float(np.sum(np.abs(output)**2)),'signed_encoding':'complex field phase, never negative intensity'},
            'state':{'owner_id':c['owner_id'],'weight_identity':evaluation_identity(BACKEND_ID,BACKEND_VERSION,{'weights':w.tolist()},p,{'seed':c['seed'],'reference_id':c['reference_id'],'mode':c['mode']})['cache_key'],'programmed_once':True,'weight_age_ns':age,'weight_retention_ns':p['weight_retention_ns']},
            'dependencies':evaluation_identity(BACKEND_ID,BACKEND_VERSION,{'weights':w.tolist(),'inputs':field if c['input_mode']=='field' else x.tolist(),'vector_input':vector,'bias':None if bias is None else b.tolist()},p,c),
            'assumptions':['hypothetical coherent contraction; no fabricated interferometer mesh or calibration claim',
                           'normalization scale changes decoding; it is not optical gain',
                           'matrix input photons are supplied and charged by the explicit encoder or upstream field owner',
                           'global reference phase error is sampled once per reference identity and seed, preserving correlation',
                           'finite programming projects quantized transfer onto the passive contraction set',
                           'programmed transfer is stable only within the explicit hypothetical weight_retention_ns envelope; age beyond it is rejected',
                           'field bias uses a coherent source and 50:50 combiner, with the other output terminated',
                           'passive matrix energy is owned by its source; absorption is not charged a second time as electrical energy']}
