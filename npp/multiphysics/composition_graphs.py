"""Inspectable typed physical expansion of inclusive classical device results.

Every child refers to the parent's accounting and scheduled reservation. These
are internal physical relations, not additional executable/costed copies.
"""
from __future__ import annotations
import copy
import math

import numpy as np

from .contracts import SignalContract, canonical_json
from .backends import fail


def _signal(values,*,carrier='electrical',regime='classical_digital',encoding='signed_numeric',
            variable='digital_word',unit='1',scale=1.,reference=None):
    arr=np.asarray(values,dtype=float)
    if arr.ndim not in (1,2) or not arr.size or not np.isfinite(arr).all():
        fail('physical_subgraph.signal','expected an owned finite vector or matrix')
    fields={'payload':'tensor','carrier':carrier,'regime':regime,'shape':list(arr.shape),
            'axis_order':['lane'] if arr.ndim==1 else ['batch','lane'],'encoding':encoding,
            'physical_variable':variable,'signed':True,'scale':float(scale),'unit':unit,
            'value_range':{'minimum':float(np.min(arr)),'maximum':float(np.max(arr))},
            'bandwidth_hz':{'state':'unavailable','unit':'Hz','reason':'local asynchronous interface; timing comes from parent scheduled reservations'},
            'clock':{'domain':'reference','mode':'asynchronous'},
            'lifetime':{'arrival_ns':{'minimum':0.,'maximum':0.},'integration_ns':0.,
                        'retention_ns':{'state':'known','value':0.,'unit':'ns'},'reset_required':False},
            'required_services':['power_supply'] if carrier=='electrical' else ['phase_reference','control'],
            'phase_reference':reference}
    return SignalContract.model_validate_json(canonical_json(fields)).model_dump(mode='json')


def _field_signal(field):
    # Field envelope covers both quadratures. Neither is labeled intensity.
    real=np.asarray(field['real']);imag=np.asarray(field['imag'])
    sig=_signal(real,carrier='optical',regime='classical_analog',encoding='signed_coherent_field',
                variable='field_amplitude',unit='sqrt(mW)',scale=field['decode_scale'],reference=field['reference_id'])
    sig['value_range']={'minimum':float(min(np.min(real),np.min(imag))), 'maximum':float(max(np.max(real),np.max(imag)))}
    return sig


def physical_subgraph(result,input_values,params,node_id,*,input_mode='digital',output_mode='digital'):
    """Build only from the evaluator's actual returned physical trace.

    ``input_mode`` and ``output_mode`` are explicit; no digital representation is
    inferred from a decoded field diagnostic. The returned graph has no ledger.
    """
    backend=result.get('backend_id');physical=result.get('physical',{})
    if backend not in ('analog.differential_conductance','photonic.coherent_matrix'):
        fail('physical_subgraph','unsupported physical backend','unavailable_model')
    nodes=[];edges=[]
    def add(suffix,operation,signal,sources=(),resource='control',payload=None):
        identifier=node_id+'.physical.'+suffix
        ports={'out':copy.deepcopy(signal)}
        for i,source in enumerate(sources):
            src=next(n for n in nodes if n['id']==source)
            ports['in'+str(i)]=copy.deepcopy(src['ports']['out'])
            edges.append({'source':{'node':source,'port':'out'},'target':{'node':identifier,'port':'in'+str(i)},'mechanism':'plain_wire'})
        nodes.append({'id':identifier,'operation':operation,'ports':ports,'resource':resource,
                      'schedule_id':None if operation=='input' else node_id,'accounting_parent':node_id,
                      'accounting':'included_in_parent','physical_payload':copy.deepcopy(payload or {})})
        return identifier
    def like_input(values):
        arr=np.asarray(values)
        return arr[0].tolist() if np.asarray(input_values).ndim==1 and arr.ndim==2 else arr.tolist()
    if backend=='analog.differential_conductance':
        if input_mode!='digital' or output_mode!='digital':
            fail('physical_subgraph','analog composite currently has digital boundaries','domain_mismatch')
        source=add('input','input',_signal(input_values),resource='memory',payload={'values':input_values})
        volts=like_input(physical['dac_voltage_v'])
        voltage_signal=_signal(volts,regime='classical_analog',encoding='bipolar_voltage',variable='voltage',unit='V',scale=1./params['input_scale_v'])
        dac=add('dac','interface.dac',voltage_signal,[source],resource='converter',payload={'voltage_v':volts,'bits':params['dac_bits']})
        currents=like_input(physical['differential_current_a'])
        decode=params['weight_scale']/((params['g_max_s']-params['g_min_s'])*params['input_scale_v'])
        current_signal=_signal(currents,regime='classical_analog',encoding='differential_current',variable='current',unit='A',scale=decode)
        tile=add('conductance','analog.tile',current_signal,[dac],resource='analog',payload={
            'conductance_positive_s':physical['conductance_positive_s'],'conductance_negative_s':physical['conductance_negative_s'],
            'loaded_voltage_v':physical['loaded_voltage_v'],'differential_current_a':currents,
            'relation':'differential current = loaded_voltage @ (G_positive-G_negative).T; loading and rail currents owned here'})
        before=physical['readout_before_bias']
        adc=add('adc','interface.adc',_signal(before),[tile],resource='converter',payload={'values':before,'bits':params['adc_bits'],'decode_scale':decode})
        if physical['bias_present']:
            b=np.asarray(physical['bias']);b=np.broadcast_to(b,np.asarray(before).shape).tolist()
            constant=add('bias_value','input',_signal(b),resource='memory',payload={'values':b,'origin':'programmed bias'})
            add('bias_add','digital.add_bias',_signal(result['actual_output']),[adc,constant],resource='control',payload={'values':result['actual_output'],'relation':'bias added once after ADC'})
        return {'schema_version':'npp-typed-graph-1','nodes':nodes,'edges':edges}
    if input_mode not in ('digital','field') or output_mode not in ('digital','field'):
        fail('physical_subgraph','photonic boundary modes must be explicit')
    in_field=physical['input_field'];matrix_field=physical['matrix_field'];out_field=physical['output_field']
    if input_mode=='digital':
        source=add('input','input',_signal(input_values),resource='memory',payload={'values':input_values})
        source=add('encoder','interface.encode',_field_signal(in_field),[source],resource='converter',payload={'field':in_field,'source_accounting':'source and driver included in parent input ledger'})
    else:
        if not isinstance(input_values,dict) or input_values.get('encoding')!='signed_coherent_field':
            fail('physical_subgraph.input','field boundary requires a physical field payload','domain_mismatch')
        source=add('input','input',_field_signal(input_values),resource='source',payload={'field':input_values,'source_accounting':'owned by upstream field producer'})
    mesh=add('mesh','photonic.mesh',_field_signal(matrix_field),[source],resource='photonic',payload={
        'input_field':in_field,'output_field':matrix_field,'transfer_real':physical['transfer_real'],'transfer_imag':physical['transfer_imag'],
        'maximum_singular_value':physical['maximum_singular_value'],'relation':'passive complex field transfer; decode normalization is not optical gain'})
    current=mesh
    if physical['bias_injected_real'] is not None:
        b=np.asarray(physical['bias']);b=np.broadcast_to(b,np.asarray(out_field['real']).shape).tolist()
        constant=add('bias_value','input',_signal(b),resource='memory',payload={'values':b,'origin':'programmed coherent bias'})
        current=add('coherent_bias','photonic.coherent_bias',_field_signal(out_field),[mesh,constant],resource='source',payload={
            'output_field':out_field,'injected_real':physical['bias_injected_real'],
            'terminated_real':physical['terminated_real'],'terminated_imag':physical['terminated_imag'],
            'electrical_source_energy_pj':physical['bias_injection_energy_pj'],
            'relation':'E_out=(E_matrix+E_bias)/sqrt(2); second 50:50 output terminated; decode_scale multiplied by sqrt(2)'})
    if output_mode=='digital':
        before=physical['readout_before_bias']
        if before is None:fail('physical_subgraph.output','no physical readout was evaluated','domain_mismatch')
        current=add('readout','interface.readout',_signal(before),[current],resource='converter',payload={'values':before,'field':out_field,'mechanism':'paid balanced homodyne with LO, then ADC'})
        if physical['bias_present']:
            b=np.asarray(physical['bias']);b=np.broadcast_to(b,np.asarray(before).shape).tolist()
            constant=add('bias_value','input',_signal(b),resource='memory',payload={'values':b,'origin':'programmed digital bias'})
            add('bias_add','digital.add_bias',_signal(result['actual_output']),[current,constant],resource='control',payload={'values':result['actual_output'],'relation':'bias added once after physical readout'})
    elif result.get('actual_output') is not None:
        fail('physical_subgraph.output','field mode cannot expose a free digital output','domain_mismatch')
    return {'schema_version':'npp-typed-graph-1','nodes':nodes,'edges':edges}
