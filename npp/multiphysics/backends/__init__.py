"""Shared backend accounting helpers; no technology equations are shared."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
from npp.models import InputValidationError


def fail(path, message, code='invalid_input'):
    raise InputValidationError([{'path':path,'message':message,'code':code}])


def finite(value, path):
    value = float(value)
    if not math.isfinite(value):
        fail(path, 'nonfinite numerical result', 'numerical_failure')
    return value


def coefficient_unit(name):
    units={
        'mac_energy_pj':'pJ/MAC','add_energy_pj':'pJ/op','silu_energy_pj':'pJ/op','multiply_energy_pj':'pJ/op',
        'sram_read_energy_pj_per_byte':'pJ/byte','sram_write_energy_pj_per_byte':'pJ/byte',
        'dram_read_energy_pj_per_byte':'pJ/byte','link_energy_pj_per_byte':'pJ/byte','program_energy_pj_per_byte':'pJ/byte',
        'launch_energy_pj':'pJ/invocation','static_power_mw':'mW','compute_area_um2':'um^2','sram_area_um2_per_byte':'um^2/byte',
        'dac_energy_pj':'pJ/sample','adc_energy_pj':'pJ/sample','programming_energy_pj_per_cell':'pJ/cell',
        'readout_energy_pj_per_output':'pJ/output','control_energy_pj_per_read':'pJ/read','reference_energy_pj_per_read':'pJ/read',
        'bias_add_energy_pj_per_output':'pJ/output','cell_area_um2':'um^2/cell','converter_area_um2':'um^2/lane',
    }
    return units[name]


def cost_quantity(value,unit,path):
    from ..contracts import Quantity
    try:
        q=Quantity.model_validate(value)
    except ValueError as exc:
        fail(path,str(exc))
    if q.unit!=unit:
        fail(path,'expected canonical unit '+unit)
    if any(v is not None and v<0 for v in (q.value,q.lower,q.upper)):
        fail(path,'cost coefficient must be nonnegative')
    return q.model_dump(mode='json')


def quantity_scale(value, factor, unit=None):
    from ..contracts import Quantity
    q = Quantity.model_validate(value).model_dump(mode='json', exclude_none=True)
    if isinstance(factor, bool) or not isinstance(factor, (int,float)) or not math.isfinite(factor) or factor < 0:
        fail('ledger.factor', 'invalid nonnegative cost multiplier')
    if unit is not None:
        q['unit'] = unit
    if factor == 0:
        return {'state':'known','value':0.0,'unit':q['unit']}
    if q['state'] == 'known':
        original = q['value']
        q['value'] = finite(original*factor, 'ledger')
        if original > 0 and q['value'] == 0:
            fail('ledger', 'positive cost underflowed', 'numerical_failure')
    elif q['state'] == 'bounded':
        for key in ('lower','upper'):
            original = q.get(key)
            if original is None:
                continue
            q[key] = finite(original*factor, 'ledger')
            if original > 0 and q[key] == 0:
                fail('ledger', 'positive cost bound underflowed', 'numerical_failure')
    return q


def quantity_sum(values, unit):
    from ..contracts import Quantity
    values = list(values)
    if any(q['state'] == 'unavailable' for q in values):
        return {'state':'unavailable','unit':unit,'reason':'one or more required ledger contributions are unavailable'}
    if all(q['state'] == 'known' for q in values):
        return {'state':'known','value':finite(math.fsum(q['value'] for q in values),'ledger.total'),'unit':unit}
    bounds = {}
    for key in ('lower','upper'):
        parts = [q['value'] if q['state']=='known' else q.get(key) for q in values]
        bounds[key] = None if any(v is None for v in parts) else finite(math.fsum(parts),'ledger.'+key)
    return Quantity(state='bounded',unit=unit,symbol='aggregate',**bounds).model_dump(mode='json',exclude_none=True)


def build_ledger(owner_id, entries, duration_ns, area_entries):
    """Each local component owns every intrinsic contribution exactly once."""
    from ..contracts import CostLedger, AccountingOwner, CostEntry, Quantity
    ledger, contract_entries = [], []
    seen = set()
    for category, effect, coefficient, multiplier in entries:
        if effect in seen:
            fail('ledger', 'duplicate effect ownership: ' + effect)
        seen.add(effect)
        value = quantity_scale(coefficient,multiplier,'pJ')
        ledger.append({'owner_id':owner_id,'effect_id':effect,'category':category,
                       'coefficient':copy.deepcopy(coefficient),'multiplier':multiplier,'energy_pj':value})
        contract_entries.append(CostEntry(contribution_id=effect,owner=owner_id,boundary='local_component',category=category,
                                         metric='energy_pj',quantity=Quantity.model_validate(value),basis='coefficient times explicit event count or elapsed ns'))
    areas=[]
    for name,value,n in area_entries:
        value=quantity_scale(value,n,'um^2')
        areas.append({'owner_id':owner_id,'effect_id':name,'area_um2':value})
        contract_entries.append(CostEntry(contribution_id='area.'+name,owner=owner_id,boundary='local_component',category='support',
                                         metric='area_um2',quantity=Quantity.model_validate(value),basis='declared physical capacity, not invocation count'))
    contract=CostLedger(boundary='local_component',owners=(AccountingOwner(id=owner_id,mode='leaf'),),
                        entries=tuple(contract_entries),expected_contributions=tuple(e.contribution_id for e in contract_entries))
    energy=contract.aggregate('energy_pj','pJ')
    area=contract.aggregate('area_um2','um^2')
    complete=energy.complete and area.complete
    coverage={}
    for category in ('compute','movement','conversion','storage','programming','calibration','control','support'):
        ids=[e.contribution_id for e in contract_entries if e.category==category]
        if ids:
            coverage[category]={'status':'charged','contribution_ids':ids,'reason':'enumerated exactly once in the typed ledger'}
        elif category=='storage':
            storage_ids=[e.contribution_id for e in contract_entries if e.contribution_id in ('sram_reads','sram_writes','weight_programming','area.sram','area.cells','static_energy')]
            coverage[category]={'status':'covered_by_other_contributions','contribution_ids':storage_ids,
                                'reason':'weight/intermediate storage is represented by the existing programming, memory traffic, installed capacity and active static-energy owners; no duplicate charge'}
        elif category=='calibration':
            coverage[category]={'status':'excluded','contribution_ids':[],
                                'reason':'active configured-component boundary excludes a calibration procedure; no end-to-end chip calibration cost or free physical calibration is claimed'}
        elif category=='conversion':
            coverage[category]={'status':'not_applicable','contribution_ids':[],
                                'reason':'digital-only electrical word boundary has no DAC, ADC or physical domain crossing'}
        else:
            fail('ledger.category_coverage','missing required accounting category '+category)
    return {'contract':contract.model_dump(mode='json'),'boundary':'inclusive_local_component','entries':ledger,'area_entries':areas,'category_coverage':coverage,
            'energy_pj':energy.quantity.model_dump(mode='json',exclude_none=True),
            'area_um2':area.quantity.model_dump(mode='json',exclude_none=True),'complete':complete,
            'duration_ns':{'state':'known','value':finite(duration_ns,'duration_ns'),'unit':'ns'},
            'obligations':[] if complete else ['resolve bounded or unavailable cost coefficients before a point-valued total or comparison']}


def error_record(ideal, actual, mode='finite'):
    delta = np.asarray(actual,dtype=float)-np.asarray(ideal,dtype=float)
    if not np.isfinite(delta).all():
        fail('error', 'nonfinite error arithmetic', 'numerical_failure')
    largest=float(np.max(np.abs(delta)))
    rms = 0.0 if largest==0 else largest*math.sqrt(math.fsum((float(v)/largest)**2 for v in delta.ravel())/delta.size)
    if largest>0 and rms==0:
        fail('error.rms','positive output RMS error underflowed; cannot certify a zero error budget','numerical_failure')
    return {'rms':finite(rms,'error.rms'),'max_abs':finite(float(np.max(np.abs(delta))),'error.max_abs'),
            'approximation':'none','finite_precision':'binary64 roundoff only' if mode=='ideal_diagnostic' else 'explicit arithmetic or converter/program quantization',
            'stochastic':'fixed declared physical seed; realized sample, not a confidence bound',
            'epistemic':'reference coefficients are hypothetical; modeled error is not task accuracy'}


def evaluation_identity(backend_id, backend_version, workload, parameters, context):
    record = {'backend_id':backend_id,'backend_version':backend_version,'workload':workload,
              'parameters':parameters,'context':context}
    try:
        encoded = json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    except (TypeError,ValueError) as exc:
        fail('dependencies', 'context must be finite JSON data: '+str(exc))
    return {'cache_key':hashlib.sha256(encoded).hexdigest(),'snapshot':record}
