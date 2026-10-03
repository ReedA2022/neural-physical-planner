"""Bounded typed FFN implementation compiler with owned physical accounting.

This is an executable, deliberately conservative system model. Device tasks
reserve their complete resource sets for their complete local duration. It is
not a transistor layout or a claim of cycle-optimal scheduling.
"""
from __future__ import annotations
import copy
import math
import numpy as np

from .contracts import canonical_hash, canonical_json, SignalContract, signal_mismatches
from .backends import fail, quantity_sum, quantity_scale, error_record
from .io import keys, integer, number, quantity
from .semantic import FFNWorkload, NumericPolicy, evaluate_ffn

SCHEMA = 'npp-implementation-1'
RESULT_SCHEMA = 'npp-implementation-result-1'
VERSION = '1.0'
FAMILIES = {'digital':'digital', 'analog':'analog_electrical', 'analog_electrical':'analog_electrical',
            'photonic':'photonic', 'classical_photonic':'photonic'}
MAX_TILES = 96
MAX_INVOCATIONS = 64
MAX_TASKS = 2048


def _family(value):
    if not isinstance(value, str) or value not in FAMILIES:
        fail('implementation.family', 'supported families are digital, analog and photonic', 'unavailable_model')
    return FAMILIES[value]


def _cuts(value, width, path):
    if not isinstance(value, list) or len(value)<2 or value[0]!=0 or value[-1]!=width:
        fail(path, 'cuts must start at zero and end at the complete axis length')
    for v in value:
        integer(v,path,0,width)
    if any(a>=b for a,b in zip(value,value[1:])):
        fail(path,'cuts must be strictly increasing')
    return value


def default_implementation(model, kind='digital'):
    model = model.to_dict() if isinstance(model, FFNWorkload) else model
    m = FFNWorkload.from_dict(model)
    if kind not in ('digital','analog','photonic','hybrid','fused'):
        fail('implementation.kind','unknown default implementation')
    families = {name:kind for name in ('up','gate','down')}
    if kind in ('hybrid','fused'):
        families = {'up':'photonic','gate':'analog' if kind=='hybrid' else 'digital','down':'digital' if kind=='hybrid' else 'photonic'}
    return normalize_implementation({'schema_version':SCHEMA,'name':kind,
            'operators':{name:{'family':family} for name,family in families.items()},
            'fusion':'signed_photonic' if kind=='fused' else 'unfused'},m.to_dict())


def normalize_implementation(raw, model):
    """Normalize friendly cut grids or explicit disjoint matrix-cell tiles."""
    from .rewrites import normalize_partition, transform_ffn
    from .infrastructure import default_system, normalize_system
    from .interfaces import reference_parameters, validate_parameters
    m = model if isinstance(model,FFNWorkload) else FFNWorkload.from_dict(model)
    keys(raw, {'schema_version','name','operators','fusion','alignment','transform','geometry','system','interfaces'},
         {'schema_version','operators'}, 'implementation')
    if raw['schema_version']!=SCHEMA:
        fail('implementation.schema_version','unsupported implementation schema')
    name=raw.get('name','explicit implementation')
    if not isinstance(name,str) or not name or len(name)>256:
        fail('implementation.name','expected a nonempty name up to 256 characters')
    keys(raw['operators'], {'up','gate','down'}, {'up','gate','down'}, 'implementation.operators')
    result={'schema_version':SCHEMA,'name':name,'operators':{}}
    total=0
    for op,shape in [('up',(m.hidden_size,m.input_size)),('gate',(m.hidden_size,m.input_size)),('down',(m.output_size,m.hidden_size))]:
        cfg=raw['operators'][op]
        keys(cfg,{'family','component','region','input_cuts','output_cuts','tiles','shape','output_reductions','merge','proof','schema','semantic_track','quality_budget'},label='operator.'+op)
        if 'tiles' in cfg:
            if any(k in cfg for k in ('family','component','input_cuts','output_cuts')):
                fail('operator.'+op,'explicit tiles cannot be mixed with grid/default family fields')
            tiles=copy.deepcopy(cfg['tiles'])
            if not isinstance(tiles,list) or not 1<=len(tiles)<=MAX_TILES: fail('operator.tiles',f'expected 1..{MAX_TILES} tiles')
            for i,t in enumerate(tiles):
                keys(t,{'id','rows','columns','input','output','family','component','region'}, {'family'}, 'tile')
                if 'input' in t or 'output' in t:
                    if set(t)&{'rows','columns'} or not {'input','output'}<=set(t):
                        fail('tile','use either rows/columns or input/output half-open ranges')
                    for axis,field in [('columns','input'),('rows','output')]:
                        span=t.pop(field)
                        if not isinstance(span,list) or len(span)!=2 or any(type(v)is not int for v in span) or not 0<=span[0]<span[1]<=(shape[1] if field=='input' else shape[0]):
                            fail('tile.'+field,'expected a nonempty half-open integer range')
                        t[axis]=list(range(*span))
                t['family']=_family(t['family'])
                t.setdefault('id',f'{op}.tile{i}')
                t.setdefault('component','auto')
                t.setdefault('region','auto')
        else:
            family=_family(cfg.get('family','digital'))
            rows=_cuts(cfg.get('output_cuts',[0,shape[0]]),shape[0],'output_cuts')
            cols=_cuts(cfg.get('input_cuts',[0,shape[1]]),shape[1],'input_cuts')
            if (len(rows)-1)*(len(cols)-1)>MAX_TILES: fail('operator.tiles','grid exceeds supported tile count','resource_infeasibility')
            tiles=[]
            for ri,(lo,hi) in enumerate(zip(rows,rows[1:])):
                for ci,(a,b) in enumerate(zip(cols,cols[1:])):
                    tiles.append({'id':f'{op}.r{ri}c{ci}','rows':list(range(lo,hi)),'columns':list(range(a,b)),
                                  'family':family,'component':cfg.get('component','auto'), 'region':cfg.get('region','auto')})
        # The rewrite validator owns coverage and canonical row/column ordering.
        normalized=normalize_partition(shape,tiles,semantic_track=m.semantic_track,quality_budget=m.quality_budget)
        result['operators'][op]=normalized
        total+=len(normalized['tiles'])
    if total>MAX_TILES: fail('implementation.operators',f'exceeds {MAX_TILES} total tiles','resource_infeasibility')
    fusion=raw.get('fusion','unfused')
    if fusion not in ('unfused','signed_photonic'): fail('implementation.fusion','unsupported fusion','unavailable_model')
    if fusion=='signed_photonic':
        for op in ('up','down'):
            ts=result['operators'][op]['tiles']
            if len(ts)!=1 or ts[0]['family']!='photonic':
                fail('implementation.fusion','field-preserving fusion currently requires one complete photonic up and down tile','unavailable_model')
        if any(t['family']=='photonic' for t in result['operators']['gate']['tiles']):
            fail('implementation.fusion','fused gate projection currently supports digital/analog tiles; a third coherent engine needs a distinct allocation','unavailable_model')
    result['fusion']=fusion
    alignment=copy.deepcopy(raw.get('alignment',{}))
    keys(alignment,{'method','delay_ns','gate_release_ns'},label='alignment')
    alignment.setdefault('method','delayed_launch'); alignment.setdefault('delay_ns',0.); alignment.setdefault('gate_release_ns',0.)
    if alignment['method'] not in ('delayed_launch','optical_delay','electronic_buffer','recompute'):
        fail('alignment.method','unsupported alignment mechanism','unavailable_model')
    alignment['delay_ns']=number(quantity(alignment['delay_ns'],'ns','alignment.delay_ns'),'alignment.delay_ns',0,1e9)
    alignment['gate_release_ns']=number(quantity(alignment['gate_release_ns'],'ns','alignment.gate_release_ns'),'alignment.gate_release_ns',0,1e9)
    result['alignment']=alignment
    transform=copy.deepcopy(raw.get('transform',{'permutation':list(range(m.hidden_size)),'signs':[1]*m.hidden_size}))
    keys(transform,{'permutation','signs'},{'permutation','signs'},'transform')
    transform_ffn(m.to_dict(),transform['permutation'],transform['signs'])
    result['transform']=transform
    result['geometry']=copy.deepcopy(raw.get('geometry'))
    if result['geometry'] is not None and not isinstance(result['geometry'],dict): fail('geometry','expected a geometry mapping')
    result['system']=normalize_system(raw.get('system',default_system()))
    result['interfaces']=validate_parameters(raw.get('interfaces',reference_parameters()))
    return result


def _signal(values, *, field=False, physical=None, scale=1., reference='reference'):
    array=np.asarray(values,dtype=float)
    lo=float(np.min(array)); hi=float(np.max(array))
    # Wires describe the exact finite payload range, not a claimed physical envelope.
    return SignalContract.model_validate_json(canonical_json({'payload':'tensor','carrier':'optical' if field else 'electrical',
        'regime':'classical_analog' if field or physical else 'classical_digital',
        'shape':list(array.shape),'axis_order':['batch','lane'] if array.ndim==2 else ['lane'],
        'encoding':'signed_coherent_field' if field else ('bipolar_voltage' if physical=='voltage' else 'differential_current' if physical=='current' else 'signed_numeric'),
        'physical_variable':'field_amplitude' if field else physical or 'digital_word','signed':True,'scale':scale,
        'unit':'sqrt(mW)' if field else 'V' if physical=='voltage' else 'A' if physical=='current' else '1',
        'value_range':{'minimum':lo,'maximum':hi},'bandwidth_hz':{'state':'known','value':0.,'unit':'Hz'},
        'clock':{'domain':'reference','mode':'asynchronous'},'lifetime':{'arrival_ns':{'minimum':0.,'maximum':0.},
        'integration_ns':0.,'retention_ns':{'state':'known','value':0.,'unit':'ns'},'reset_required':False},
        'required_services':['laser','phase_reference','control'] if field else ['power_supply','clock'],
        'phase_reference':reference if field else None})).model_dump(mode='json')


def validate_graph(graph):
    """Validate types, directed ports, coverage and acyclic physical composition."""
    from .composition_validation import validate_graph_structure
    return validate_graph_structure(graph)


def _known(value,unit): return {'state':'known','value':float(value),'unit':unit}


class _Compiler:
    def __init__(self,project,spec):
        from .infrastructure import system_parameters
        from .catalog import parameters_from_component
        self.project=project; self.spec=spec; self.system=system_parameters(spec['system'])
        self.model=FFNWorkload.from_dict(project['model']); self.w=project['workload']; self.env=project['environment']
        self.components={c['id']:c for c in project['technology']['components']}
        self.params={k:parameters_from_component(v) for k,v in self.components.items()}
        self.nodes=[]; self.edges=[]; self.tasks=[]; self.results={}; self.values={}; self.signals={}
        self.energy=[]; self.area_by_resource={}; self.static_by_resource={}; self.used_resources=set()
        self.operation_resource={}; self.programmed=set(); self.task_ids=set(); self.current_deps=[]
        self.digital_component=self.component('digital',None); self.digital_params=self.params[self.digital_component['id']]
        self.memory_bytes=0; self.cold_programming_pj=[]; self.group_resource={}; self.external_weights={};self.weight_epochs={};self.last_weight_read={};self.owned_states=[];self.invocation_outputs=[];self.pool_shapes={};self.area_effects={};self.static_effects={};self.provisioning={}
        if {p['evidence']['kind'] for p in spec['system']['parameters']}-set(self.w['evidence_policy']):
            fail('workload.evidence_policy','system parameter evidence excluded by policy','unavailable_model')
        if 'hypothetical' not in self.w['evidence_policy'] and any(t['family']!='digital' for op in spec['operators'].values() for t in op['tiles']):
            fail('workload.evidence_policy','registered interface coefficients are hypothetical and excluded by policy','unavailable_model')
        # Resolve installed physical array dimensions before evaluating any
        # tile, so all invocations use the same fixed rectangular hardware.
        for op,partition in spec['operators'].items():
            for tile in partition['tiles']:
                family=tile['family'];component=self.component(family,tile.get('component'))
                resource='digital' if family=='digital' else 'analog' if family=='analog_electrical' else 'photonic_'+op if spec['fusion']=='signed_photonic' else 'photonic'
                group=resource+'.'+component['id']
                shape=self.pool_shapes.setdefault(group,{'family':family,'component_id':component['id'],'rows':0,'columns':0})
                shape['rows']=max(shape['rows'],len(tile['rows']));shape['columns']=max(shape['columns'],len(tile['columns']))

    def component(self,family,identifier):
        selected=[c for c in self.components.values() if c['family']==family and (identifier in (None,'auto') or c['id']==identifier)]
        if len(selected)!=1: fail('implementation.component',f'expected exactly one {family} component matching {identifier!r}')
        c=selected[0]; env=self.env; integration=c['integration']
        if env['substrate']!=integration['substrate']: fail('environment.substrate','component substrate mismatch','domain_mismatch')
        if not integration['temperature_c']['minimum']<=env['temperature_c']<=integration['temperature_c']['maximum']:
            fail('environment.temperature_c','component temperature envelope violated','envelope_violation')
        missing=set(integration['required_services'])-set(env['services'])
        if missing: fail('environment.services','missing services: '+', '.join(sorted(missing)),'domain_mismatch')
        if {p['evidence']['kind'] for p in c['parameters']}-set(self.w['evidence_policy']):
            fail('workload.evidence_policy','selected component evidence excluded','unavailable_model')
        return c

    def charge(self,owner,effect,category,coefficient,factor):
        self.energy.append({'owner_id':owner,'effect_id':effect,'category':category,'energy_pj':quantity_scale(coefficient,factor,'pJ'),
                            'coefficient':copy.deepcopy(coefficient),'multiplier':factor})

    def task(self,name,duration,resources,deps=(),release=0.,events=None):
        if name in self.task_ids: fail('schedule','duplicate compiler task')
        duration=number(duration,'task.duration',0,1e15)
        self.task_ids.add(name); self.used_resources.update(resources)
        self.tasks.append({'id':name,'owner':name,'duration_ns':duration,'deps':list(dict.fromkeys(deps)),
                           'resources':{r:1 for r in resources} if duration>0 else {},'release_ns':release,'events':events or []})
        if len(self.tasks)>MAX_TASKS: fail('implementation','task expansion exceeds bounded scheduler capacity','resource_infeasibility')
        return name

    def input(self,values):
        return self.constant('input',values)

    def constant(self,name,values):
        signal=_signal(values)
        self.nodes.append({'id':name,'operation':'input','ports':{'out':signal},'resource':'memory','schedule_id':None})
        self.values[name]=copy.deepcopy(values);self.signals[name]=signal; self.operation_resource[name]='memory'
        return name

    def add(self,name,operation,result,inputs,resources,component=None,release=0.,extra_deps=()):
        value=result.get('field')
        if value is None: value=result.get('actual_output')
        if value is None: value=result.get('values')
        if value is None: fail('compiler.backend','backend did not expose declared physical output')
        is_field=isinstance(value,dict) and value.get('carrier')=='optical'
        signal=_signal(value['real'] if is_field else value,field=is_field,
                       scale=value['decode_scale'] if is_field else 1.,reference=value['reference_id'] if is_field else 'reference')
        ports={'out':signal}; deps=list(extra_deps)
        for i,source in enumerate(inputs):
            ports['in'+str(i)]=copy.deepcopy(self.signals[source])
            self.edges.append({'source':{'node':source,'port':'out'},'target':{'node':name,'port':'in'+str(i)},'mechanism':'plain_wire'})
            if source in self.task_ids: deps.append(source)
        self.nodes.append({'id':name,'operation':operation,'ports':ports,'resource':resources[0], 'schedule_id':name,
                           'component_id':None if component is None else component['id'],
                           'accounting':'inclusive composite with complete task reservations' if '.composite' in operation else 'leaf',
                           'weight_version':result.get('state',{}).get('weight_identity',result.get('state',{}).get('weight_version'))})
        self.values[name]=copy.deepcopy(value);self.signals[name]=signal;self.results[name]=result
        self.operation_resource[name]=resources[0]
        duration=result.get('local_duration_ns',result.get('duration_ns'))
        self.task(name,duration,resources,deps,release)
        ledger=result['ledger']
        group=resources[0]+'.'+(component['id'] if component else operation)
        self.group_resource[group]=resources[0]
        static=[]
        for entry in ledger['entries']:
            coefficient=entry['coefficient']; unit=coefficient['unit']
            if unit.startswith('mW'):
                factor=result.get('resources',{}).get('matrix_elements',1) if unit=='mW/element' else 1
                power=quantity_scale(coefficient,factor,'mW')
                static.append(power)
                self.static_effects.setdefault(group,{}).setdefault(entry['effect_id'],[]).append(power)
            else:
                e=copy.deepcopy(entry);e['owner_id']=name; self.energy.append(e)
        if static: self.static_by_resource.setdefault(group,[]).append(quantity_sum(static,'mW'))
        self.area_by_resource.setdefault(group,[]).append(ledger['area_um2'])
        for entry in ledger.get('area_entries',[]):
            self.area_effects.setdefault(group,{}).setdefault(entry['effect_id'],[]).append(entry['area_um2'])
        return name

    def digital(self,name,kind,sources,extra=None,release=0.,extra_deps=()):
        from .backends.digital import evaluate_elementwise_cost
        context={'owner_id':name,'numeric_policy':self.w['numeric_policy'],'reuse_count':1,'seed':self.w['seed'],
                 'weight_residency':self.w['weight_residency'],'environment':self.env}
        context.update(extra or {})
        result=evaluate_elementwise_cost(kind,[self.values[s] for s in sources],self.digital_params,context)
        return self.add(name,'digital.'+kind,result,sources,['digital','memory','link','control'],self.digital_component,release,extra_deps)

    def interface(self,name,kind,sources,*args,extra_deps=()):
        from . import interfaces
        fn=getattr(interfaces,kind)
        payloads=[self.values[s] for s in sources]
        result=fn(*payloads,*args,parameters=self.spec['interfaces'],context={'owner_id':name,'reference_id':'reference'})
        resources={'encode':['converter','source','control'], 'readout':['converter','source','control'],
                   'signed_gate':['converter','control','source'], 'optical_delay':['buffer'],
                   'digital_buffer':['buffer','memory','control']}[kind]
        return self.add(name,'interface.'+kind,result,sources,resources,extra_deps=extra_deps)

    def linear(self,op,source,model,*,suffix='',field_output=False,field_input=False,extra_deps=(),release=0.):
        from .backends import digital,analog
        from .backends import photonic
        matrix=np.asarray(model['W_'+op]);bias=np.asarray(model['b_'+op])
        partition=self.spec['operators'][op]; tiles=partition['tiles']; outputs=[]
        for i,t in enumerate(tiles):
            family=t['family']; c=self.component(family,t.get('component'));p=self.params[c['id']]
            rows=t['rows']; columns=t['columns']; name=suffix+op+'.'+str(i)
            tile_source=source
            if len(columns)!=matrix.shape[1] or columns!=list(range(matrix.shape[1])):
                if field_input: fail('fusion','field input partition needs an explicit optical split/merge mechanism','unavailable_model')
                tile_source=self.digital(name+'.gather','gather',[source],{'indices':columns},extra_deps=extra_deps)
            if t['region'] not in ('auto',self.env['region']):
                fail('implementation.tiles.region','this bounded compiler requires the declared common integration region','domain_mismatch')
            dims={'rows':len(rows),'columns':len(columns),'batch':len(self.w['inputs']),'temperature_c':self.env['temperature_c']}
            for axis in c['operating_envelope']:
                if axis['name'] in dims and not axis['minimum']<=dims[axis['name']]<=axis['maximum']:
                    fail('component.operating_envelope.'+axis['name'],'bound tile exceeds the component envelope','envelope_violation')
            tile_w=matrix[np.ix_(rows,columns)].tolist()
            tile_bias=bias.tolist() if field_output or len(tiles)==1 else None
            x=self.values[tile_source]
            padding=None
            if family!='digital':
                resource='analog' if family=='analog_electrical' else 'photonic_'+op if self.spec['fusion']=='signed_photonic' else 'photonic'
                installed=self.pool_shapes[resource+'.'+c['id']]
                if field_input or field_output:
                    if installed['rows']!=len(rows) or installed['columns']!=len(columns):
                        fail('fusion.padding','field-array padding needs explicit optical lane interfaces; unsupported','unavailable_model')
                else:
                    from .array_padding import expand_array
                    padding=expand_array(tile_w,x,tile_bias,installed['rows'],installed['columns'])
                    if installed['columns']>len(columns):
                        zeros=np.zeros((len(self.w['inputs']),installed['columns']-len(columns))).tolist()
                        zero_id=self.constant(name+'.padding.zeros',zeros)
                        tile_source=self.digital(name+'.padding.concat','concat',[tile_source,zero_id],extra_deps=extra_deps)
                        x=self.values[tile_source]
                    tile_w=padding['weights'];tile_bias=padding['bias']
            context={'owner_id':name,'reuse_count':1,'seed':self.w['seed'],'environment':self.env}
            if family=='digital':
                context.update(numeric_policy=self.w['numeric_policy'],weight_residency=self.w['weight_residency'])
                result=digital.evaluate_linear_cost(tile_w,x,tile_bias,p,context)
                resources=['digital','memory','link','control'];operation='digital.linear'
            elif family=='analog_electrical':
                context.update(temperature_c=self.env['temperature_c'],stream_id='physical.'+name)
                result=analog.evaluate_linear(tile_w,x,tile_bias,p,context)
                resources=['analog','converter','memory','link','control'];operation='analog.composite'
            else:
                context.update(input_mode='field' if field_input else 'digital',output_mode='field' if field_output else 'digital',
                               reference_id='reference')
                result=photonic.evaluate_linear(tile_w,x,tile_bias,p,context)
                resources=['photonic_'+op if self.spec['fusion']=='signed_photonic' else 'photonic','converter','source','memory','link','control'];operation='photonic.composite'
            port_checks=[]
            for port in c['ports']:
                values=x if port['name']=='input' else result.get('actual_output')
                if values is None: values=result.get('diagnostic_decoded_output')
                field_diagnostic=isinstance(values,dict)
                if field_diagnostic:
                    # The selected component declares a semantic input range.
                    # Checking the represented in-phase value does not emit a
                    # digital value or bypass a physical homodyne/ADC interface.
                    try:
                        with np.errstate(over='raise',invalid='raise'):
                            arr=np.asarray(values['real'],dtype=float)*values['decode_scale']
                    except (FloatingPointError,OverflowError) as exc:
                        fail('component.ports.'+port['name'],'nonfinite semantic field range diagnostic: '+str(exc),'numerical_failure')
                else:
                    arr=np.asarray(values,dtype=float)
                if not np.isfinite(arr).all():
                    fail('component.ports.'+port['name'],'nonfinite bound signal diagnostic','numerical_failure')
                bounds=port['signal']['value_range']
                if np.any(arr<bounds['minimum']) or np.any(arr>bounds['maximum']):
                    fail('component.ports.'+port['name'],'bound signal exceeds the declared port range','envelope_violation')
                port_checks.append({'port':port['name'],'declared_range':copy.deepcopy(bounds),
                    'observed_minimum':float(np.min(arr)),'observed_maximum':float(np.max(arr)),
                    'validation_only':field_diagnostic or port['name']=='output' and field_output,
                    'scope':'semantic in-phase quadrature diagnostic; no physical digital readout is created' if field_diagnostic else 'declared semantic port envelope'})
            result['component_port_validation']=port_checks
            weight_transfer_ns=0.
            if family!='digital':
                weight_key=canonical_hash({'matrix':tile_w,'bias':tile_bias})
                weight_bytes=(len(tile_w)*len(tile_w[0])+(len(tile_bias) if tile_bias is not None else 0))*8
                first_load=weight_key not in self.external_weights
                self.external_weights[weight_key]=weight_bytes
                dram_bytes=weight_bytes if first_load or self.w['weight_residency']=='dram' else 0
                reads=weight_bytes if self.w['weight_residency']=='sram' else 0
                self.charge(name,'external_weight_dram','movement',self.digital_params['dram_read_energy_pj_per_byte'],dram_bytes)
                self.charge(name,'external_weight_sram','movement',self.digital_params['sram_read_energy_pj_per_byte'],reads)
                self.charge(name,'external_weight_link','movement',self.digital_params['link_energy_pj_per_byte'],weight_bytes)
                weight_transfer_ns=dram_bytes/self.digital_params['dram_bytes_per_ns']+reads/self.digital_params['sram_bytes_per_ns']+weight_bytes/self.digital_params['link_bytes_per_ns']
                result=copy.deepcopy(result)
                result['local_duration_ns']+=weight_transfer_ns
                result['external_weight_transfer']={'bytes':weight_bytes,'duration_ns':weight_transfer_ns,'first_load':first_load,'weight_identity':weight_key}
            if padding is not None:
                result['array_realization']={k:copy.deepcopy(v) for k,v in padding.items() if k not in ('weights','inputs','bias')}
                result['array_realization']['inactive_policy']='explicit zero-programmed full array; nonzero common-mode conductance/loading, all converters/readout and programming remain active and paid'
            self.add(name,operation,result,[tile_source],resources,c,release,extra_deps)
            group=resources[0]+'.'+c['id']
            shape=self.pool_shapes.setdefault(group,{'family':family,'component_id':c['id'],'rows':0,'columns':0})
            shape['rows']=max(shape['rows'],len(rows));shape['columns']=max(shape['columns'],len(columns))
            if field_input:
                setup_ns=weight_transfer_ns+sum(result['duration_breakdown'][k] for k in ('programming_ns','calibration_ns','settling_ns'))
                setup_id=name+'.program'
                previous=self.last_weight_read.get(resources[0])
                self.task(setup_id,setup_ns,[resources[0],'control','memory','link'],[previous] if previous else [])
                epoch=self.weight_epochs.get(resources[0],0);self.weight_epochs[resources[0]]=epoch+1
                state_id='weights.'+setup_id
                contract={'id':state_id,'owner':setup_id,'version':epoch,'kind':'weights','created_ns':0.,
                          'retention_ns':_known(p['weight_retention_ns'],'ns'),'read_mode':'repeatable','reset_required':False,
                          'capacity':len(rows)*len(columns),'capacity_unit':'weight_element'}
                program_task=next(t for t in self.tasks if t['id']==setup_id)
                program_task['events'].append({'mode':'create','at':'end','state_id':state_id,'version':epoch,'actor':setup_id,
                    'contract':contract,'payload_json':canonical_json({'matrix':tile_w,'weight_identity':result['state']['weight_identity']}),
                    'readers':[name],'bind_creation_time':True,'occupancy':len(rows)*len(columns)})
                self.last_weight_read[resources[0]]=name
                self.owned_states.append({'state_id':state_id,'physical_resource':resources[0],'epoch':epoch,
                    'weight_identity':result['state']['weight_identity'],'program_task':setup_id,'read_task':name})
                task=next(t for t in self.tasks if t['id']==name)
                task['duration_ns']-=setup_ns
                task['deps'].append(setup_id)
                task['events'].extend([{'mode':'read','at':'start','state_id':state_id,'version':epoch,'actor':name},
                                       {'mode':'consume','at':'end','state_id':state_id,'version':epoch,'actor':name}])
            if family!='digital':
                from .composition_graphs import physical_subgraph
                node=next(n for n in self.nodes if n['id']==name)
                node['physical_subgraph']=physical_subgraph(result,x,p,name,input_mode='field' if field_input else 'digital',output_mode='field' if field_output else 'digital')
            tile_output=name
            if padding is not None and padding['installed_rows']>len(rows):
                tile_output=self.digital(name+'.padding.select','gather',[name],{'indices':list(range(len(rows)))})
            outputs.append((tile_output,rows))
        if field_output or len(tiles)==1: return outputs[0][0]
        # Reduce input partitions per output row, then concatenate in original row order.
        row_nodes=[]
        for row in range(matrix.shape[0]):
            terms=[]
            for node,rows in outputs:
                if row in rows:
                    terms.append(self.digital(suffix+op+f'.row{row}.g{len(terms)}','gather',[node],{'indices':[rows.index(row)]}))
            current=terms[0]
            for j,term in enumerate(terms[1:]): current=self.digital(suffix+op+f'.row{row}.sum{j}','add',[current,term])
            row_nodes.append(current)
        out=row_nodes[0] if len(row_nodes)==1 else self.digital(suffix+op+'.concat','concat',row_nodes)
        # Bias remains outside every partition and is applied exactly once.
        biasnode=suffix+op+'.bias_value'; b=np.broadcast_to(bias,(len(self.w['inputs']),len(bias))).tolist()
        self.nodes.append({'id':biasnode,'operation':'input','ports':{'out':_signal(b)},'resource':'memory','schedule_id':None})
        self.values[biasnode]=b;self.signals[biasnode]=_signal(b);self.operation_resource[biasnode]='memory'
        return self.digital(suffix+op+'.bias','add',[out,biasnode])

    def provision_pools(self):
        """Install each device mechanism once at its full rectangular envelope.

        Shape-changing electrical/optical matrices cannot reshape a narrow
        physical array for free. The maximum independent row/column dimensions
        determine installed cells, with separately provisioned input/output I/O.
        Reconfigurable digital arithmetic retains its declared fixed engine.
        """
        for group in self.area_by_resource:
            areas={effect:_quantity_max(values,'um^2') for effect,values in self.area_effects.get(group,{}).items()}
            powers={effect:_quantity_max(values,'mW') for effect,values in self.static_effects.get(group,{}).items()}
            shape=self.pool_shapes.get(group)
            if shape and shape['family']!='digital':
                p=self.params[shape['component_id']];rows=shape['rows'];columns=shape['columns']
                if shape['family']=='analog_electrical':
                    areas['cells']=quantity_scale(p['cell_area_um2'],2*rows*columns,'um^2')
                    areas['converters']=quantity_scale(p['converter_area_um2'],rows+columns,'um^2')
                elif shape['family']=='photonic':
                    areas['matrix.matrix_elements']=quantity_scale(p['element_area_um2'],rows*columns,'um^2')
                    powers['matrix.phase_holding']=quantity_scale(p['holding_power_mw_per_element'],rows*columns,'mW')
            if areas: self.area_by_resource[group]=[quantity_sum(areas.values(),'um^2')]
            if powers: self.static_by_resource[group]=[quantity_sum(powers.values(),'mW')]
            self.provisioning[group]={'resource':self.group_resource[group],
                'matrix_shape':copy.deepcopy(shape),'area_effects':areas,'static_power_effects':powers,
                'rule':'independent maximum rows and columns for a fixed rectangular matrix; sum independently provisioned I/O mechanisms; digital engine remains reconfigurable'}

    def bind_states(self):
        from .contracts import canonical_json
        tasks={t['id']:t for t in self.tasks}
        consumers={name:[] for name in self.values}
        for e in self.edges: consumers[e['source']['node']].append(e['target']['node'])
        initial=[]; total_bytes=0
        for name,value in self.values.items():
            optical=isinstance(value,dict)
            count=np.asarray(value['real'] if optical else value).size
            storage_bytes=count*(16 if optical else 8)
            if not optical: total_bytes+=storage_bytes
            readers=list(dict.fromkeys(consumers[name]))
            if optical and len(readers)>1: fail('state','coherent field fanout requires a paid splitter; implicit copy is unsupported','domain_mismatch')
            contract={'id':'state.'+name,'owner':name,'version':0,'kind':'excitation' if optical else 'cache',
                      'created_ns':0.,'retention_ns':_known(0. if optical else self.system['electronic_buffer_retention_ns'],'ns'),
                      'read_mode':'destructive' if optical else 'repeatable','reset_required':False,
                      'capacity':int(count),'capacity_unit':'complex_lane' if optical else 'word64'}
            event={'mode':'create','at':'end','state_id':contract['id'],'version':0,'actor':name,
                   'contract':contract,'payload_json':canonical_json(value),'readers':readers,'bind_creation_time':True}
            if name in tasks: tasks[name]['events'].append(event)
            else: initial.append({'contract':contract,'payload':value,'readers':readers})
            for reader in readers:
                if reader not in tasks: continue
                tasks[reader]['events'].append({'mode':'consume' if optical else 'read','at':'start',
                                              'state_id':contract['id'],'version':0,'actor':reader})
            if not optical:
                # These are inter-component FIFOs, distinct from the local device's
                # SRAM traffic. Their physical capacity is explicitly installed.
                if name in tasks:
                    tasks[name]['duration_ns']+=storage_bytes/self.system['electronic_buffer_write_bytes_per_ns']
                    tasks[name]['resources']['memory']=1
                for reader in readers:
                    if reader in tasks:
                        tasks[reader]['duration_ns']+=storage_bytes/self.system['electronic_buffer_read_bytes_per_ns']
                        tasks[reader]['resources']['memory']=1
                self.charge(name,'system_fifo_write','storage',self.system['electronic_buffer_write_energy_pj_per_byte'],storage_bytes)
                self.charge(name,'system_fifo_reads','storage',self.system['electronic_buffer_read_energy_pj_per_byte'],storage_bytes*len(readers))
        total_bytes+=sum(self.external_weights.values()) if self.w['weight_residency']=='sram' else 0
        self.memory_bytes=total_bytes
        if total_bytes>self.system['electronic_buffer_capacity_bytes'] or total_bytes>self.project['constraints']['max_memory_bytes']:
            fail('resources.buffer',f'graph materializations require {total_bytes} bytes beyond capacity','resource_infeasibility')
        return initial

    def finalize(self,output,semantic_plan):
        from .scheduling import schedule
        from .geometry import bus_geometry, bus_distances, validate_geometry
        self.provision_pools()
        graph={'schema_version':'npp-typed-graph-1','nodes':self.nodes,'edges':self.edges}
        graph_validation=validate_graph(graph)
        def required_services(graph_value):
            services=set()
            for node in graph_value['nodes']:
                for signal in node['ports'].values(): services.update(signal.get('required_services',[]))
                if node.get('physical_subgraph'): services.update(required_services(node['physical_subgraph']))
            return services
        missing_services=required_services(graph)-set(self.env['services'])
        if missing_services:
            fail('environment.services','graph ports require missing services: '+', '.join(sorted(missing_services)),'domain_mismatch')
        capacities=copy.deepcopy(self.project['constraints']['resources'])
        if self.spec['fusion']=='signed_photonic':
            if capacities.get('photonic',0)<2:
                fail('constraints.resources.photonic','field-preserving up/down need two separately programmed photonic instances','resource_infeasibility')
            capacities['photonic_up']=1;capacities['photonic_down']=1
        missing=self.used_resources-set(capacities)
        if missing: fail('constraints.resources','missing explicit resource capacities: '+', '.join(sorted(missing)),'resource_infeasibility')
        kinds={'digital':'compute','analog':'compute','photonic':'compute','converter':'converter','memory':'memory',
               'link':'link','control':'control','source':'source','buffer':'buffer','photonic_up':'compute','photonic_down':'compute'}
        resources=[{'id':r,'owner':'system','kind':kinds[r],'capacity':capacities[r], 'capacity_unit':'instance',
                    'region':self.env['region']} for r in sorted(self.used_resources)]
        # Independent configured resources have separate startup/calibration owners.
        setup_duration=self.system['package_startup_ns']+self.system['clock_startup_ns']+self.system['calibration_ns']*sum(r['capacity'] for r in resources)
        setup=self.task('system.setup',setup_duration,['control'],())
        for t in self.tasks:
            if t['id']!=setup and not t['deps']: t['deps'].append(setup)
        self.charge('system','package_startup','support',self.system['package_startup_energy_pj'],1)
        self.charge('system','clock_startup','support',self.system['clock_startup_energy_pj'],1)
        self.charge('system','calibration','calibration',self.system['calibration_energy_pj_per_component'],sum(r['capacity'] for r in resources))
        # Geometry is built around installed resource pools rather than repeated
        # invocations: a device's footprint is not charged once per operation.
        physical_pools=sorted(self.used_resources)
        resource_placement={r:r for r in physical_pools}
        if self.spec['fusion']=='signed_photonic':
            # Coherent up/gate/down are a local configured optical assembly.
            # Its internal propagation is owned by the registered matrix/gate
            # equations. Exterior geometry describes its electrical boundary.
            for r in ('photonic_up','photonic_down','converter','source','buffer'):
                if r in resource_placement: resource_placement[r]='coherent_module'
        placed=sorted(set(resource_placement.values())|{'system_support'})
        footprints={}
        for placement in placed:
            requirements=[]
            for r in physical_pools:
                if resource_placement[r]==placement:
                    for group,values in self.area_by_resource.items():
                        if self.group_resource[group]==r:
                            q=_quantity_max(values,'um^2')
                            requirements.append(capacities[r]*q.get('value',q.get('upper') or 100.))
            if placement=='memory':
                estimated_bytes=sum(np.asarray(value).size*8 for value in self.values.values() if not isinstance(value,dict))
                estimated_bytes+=sum(self.external_weights.values()) if self.w['weight_residency']=='sram' else 0
                q=quantity_scale(self.system['electronic_buffer_area_um2_per_byte'],estimated_bytes,'um^2')
                requirements.append(q.get('value',q.get('upper') or 100.))
            if placement=='system_support':
                q=self.system['support_area_um2'];requirements.append(q.get('value',q.get('upper') or 100.))
            area_requirement=math.fsum(requirements) or 4.
            side=max(2.,math.sqrt(area_requirement))
            footprints[placement]=(side,side)
        geo=self.spec['geometry'] or bus_geometry(placed,region_id=self.env['region'],
            substrate=self.env['substrate'],temperature_c=self.env['temperature_c'],services=tuple(self.env['services']),footprints=footprints,
            kinds={p:('storage' if p=='memory' else 'control' if p in ('control','system_support') else 'interface' if p in ('converter','link') else 'source' if p=='source' else 'buffer' if p=='buffer' else 'compute') for p in placed})
        geometry=validate_geometry(geo)
        if geometry['status']!='model_feasible': fail('geometry',str(geometry['diagnostics']),'domain_mismatch')
        placements={p['id']:p for p in geometry['geometry']['placements']}
        regions={r['id']:r for r in geometry['geometry']['regions']}
        for placement in placed:
            if placement not in placements: fail('geometry.placements','missing physical pool '+placement,'domain_mismatch')
            item=placements[placement];region=regions[item['region']]
            if item['region']!=self.env['region'] or region['substrate']!=self.env['substrate'] or region['temperature_c']!=self.env['temperature_c']:
                fail('geometry.environment','placement contradicts the bound common integration environment','domain_mismatch')
            if set(self.env['services'])-set(region['services']):
                fail('geometry.services','placed region lacks declared system services','domain_mismatch')
            if item['width_um']*item['height_um']+1e-9<footprints[placement][0]*footprints[placement][1]:
                fail('geometry.footprint','placement is smaller than the declared installed resource footprint','resource_infeasibility')
        distances=bus_distances(geometry['geometry'])
        bus_capacities=[route['capacity'] for route in geometry['geometry']['routes']]
        bus_capacities.extend(port['capacity'] for item in geometry['geometry']['placements'] for port in item['ports'])
        if not bus_capacities or any(type(cap)is not int or cap<1 for cap in bus_capacities):
            fail('geometry.bus','cannot establish a positive physical shared-bus capacity','unavailable_model')
        physical_bus_capacity=min(bus_capacities)
        available_link_capacity=capacities['link']
        capacities['link']=min(available_link_capacity,physical_bus_capacity)
        for resource in resources:
            if resource['id']=='link': resource['capacity']=capacities['link']
        resource_bindings={'link':{'available_capacity':available_link_capacity,'physical_capacity':physical_bus_capacity,
            'effective_capacity':capacities['link'],'rule':'minimum available bound and instantiated shared-bus route/port bottleneck'}}
        for p in placed:
            if p not in distances: fail('geometry.placements','missing installed physical pool '+p,'domain_mismatch')
        geometry['resource_placements']=resource_placement
        geometry['coherent_boundary']='internal coherent routes belong to local registered matrix/gate transfer model; unsupported external optical routing is rejected'
        route_map={}
        for a in physical_pools:
            for b in physical_pools:
                pa,pb=resource_placement[a],resource_placement[b]
                if pa==pb: route_map[a,b]=0.
                elif pb in distances[pa]: route_map[a,b]=distances[pa][pb]
        # Explicit electrical bus transfers reserve the common link during the
        # complete receiving task. No optical field travels on this bus.
        tasks={t['id']:t for t in self.tasks}
        for i,edge in enumerate(self.edges):
            source=edge['source']['node'];target=edge['target']['node']
            if isinstance(self.values[source],dict): continue
            a=self.operation_resource[source];b=self.operation_resource[target]
            if a==b: continue
            if (a,b) not in route_map: fail('geometry.routes',f'missing route {a} -> {b}','domain_mismatch')
            length=route_map[a,b]; bytes_=np.asarray(self.values[source]).size*8
            duration=length*self.system['electrical_route_ns_per_um']+bytes_/self.system['electrical_link_bytes_per_ns']
            tasks[target]['duration_ns']+=duration
            tasks[target]['resources']['link']=1
            edge['route']={'source':a,'target':b,'length_um':length,'duration_ns':duration,'bytes':bytes_,
                           'reservation':'inclusive in receiving task'}
            self.charge('route.'+str(i),'electrical_transport','movement',self.system['electrical_route_energy_pj_per_byte_um'],length*bytes_)
        initial=self.bind_states()
        scheduling=schedule(self.tasks,resources,initial_states=initial)
        elapsed=scheduling.get('makespan_ns') or 0.
        if scheduling['status'] not in ('model_feasible','conditional'):
            fail('schedule',str(scheduling.get('diagnostics')),'numerical_failure' if scheduling['status']=='numerical_failure' else 'resource_infeasibility')
        for group,coeffs in self.static_by_resource.items():
            resource=self.group_resource[group]
            self.charge('installed.'+group,'static_over_elapsed','support',_quantity_max(coeffs,'mW'),elapsed*capacities[resource])
        self.charge('system','clock_static','support',self.system['clock_static_power_mw'],elapsed)
        self.charge('system','package_static','support',self.system['package_static_power_mw'],elapsed)
        self.charge('system','fifo_static','storage',self.system['electronic_buffer_static_power_mw_per_byte'],elapsed*self.memory_bytes)
        area=[]
        for group,values in self.area_by_resource.items():
            resource=self.group_resource[group]
            area.append({'owner_id':'installed.'+group,'resource':resource,'capacity':capacities[resource],
                         'area_um2':quantity_scale(_quantity_max(values,'um^2'),capacities[resource],'um^2')})
        area.extend([{'owner_id':'system.fifos','area_um2':quantity_scale(self.system['electronic_buffer_area_um2_per_byte'],self.memory_bytes,'um^2')},
                     {'owner_id':'system.support','area_um2':quantity_scale(self.system['support_area_um2'],1,'um^2')}])
        for item in geometry['geometry']['placements']:
            if item['id'] not in placed:
                area.append({'owner_id':'junction.'+item['id'],'area_um2':_known(item['width_um']*item['height_um'],'um^2')})
        for route in geometry['routes']:
            area.append({'owner_id':'route.'+route['id'],'area_um2':quantity_scale(self.system['electrical_route_area_um2_per_um'],route['length_um'],'um^2')})
        energy=quantity_sum([e['energy_pj'] for e in self.energy],'pJ'); total_area=quantity_sum([a['area_um2'] for a in area],'um^2')
        complete=energy['state']=='known' and total_area['state']=='known'
        ideal=evaluate_ffn(self.project['model'],self.w['inputs'],NumericPolicy())['ideal_output']
        invocation_quality=[{'node':node,'output':copy.deepcopy(self.values[node]),'error':error_record(ideal,self.values[node])} for node in self.invocation_outputs]
        error={'rms':max(q['error']['rms'] for q in invocation_quality),'max_abs':max(q['error']['max_abs'] for q in invocation_quality)}
        quality={'rms':error['rms'],'max_abs':error['max_abs'],'max_rms_error':self.w['quality']['max_rms_error'],
                 'model_max_absolute_error':self.model.quality_budget,'passed':error['rms']<=self.w['quality']['max_rms_error'] and
                 (self.model.quality_budget is None or error['max_abs']<=self.model.quality_budget),
                 'scope':'provided workload inputs versus original FFN; modeled realization, not held-out task quality'}
        constraints=self.project['constraints']; violations=[]
        if not quality['passed']: violations.append('quality_budget_exceeded')
        energy_lower=energy.get('value') if energy['state']=='known' else energy.get('lower')
        if energy_lower is not None and energy_lower>constraints['max_energy_pj']: violations.append('energy_budget_exceeded')
        if elapsed>constraints['max_latency_ns']: violations.append('latency_budget_exceeded')
        if scheduling['status'] not in ('model_feasible','conditional'): violations.append('schedule_infeasible')
        status='resource_infeasible' if violations else 'model_feasible' if complete and scheduling['status']=='model_feasible' else 'conditional'
        task_times={t['id']:t for t in scheduling['tasks']}
        completion_times=[task_times[n]['end_ns'] for n in self.invocation_outputs]
        steady_intervals=[b-a for a,b in zip(completion_times,completion_times[1:])]
        coverage={cat:{'status':'charged','owners':sorted({e['owner_id'] for e in self.energy if e['category']==cat})}
                  for cat in ('compute','movement','conversion','storage','programming','calibration','control','support')}
        if not coverage['conversion']['owners']: coverage['conversion']={'status':'not_applicable','reason':'all boundaries use digital words'}
        result={'schema_version':RESULT_SCHEMA,'compiler_version':VERSION,'status':status,
            'normalized_inputs':{'project':copy.deepcopy(self.project),'implementation':copy.deepcopy(self.spec)},
            'semantic_plan':semantic_plan,'physical_graph':graph,'graph_validation':graph_validation,'geometry':geometry,
            'schedule':scheduling,'resources':resources,'resource_bindings':resource_bindings,'pool_provisioning':self.provisioning,'ledger':{'boundary':'complete lumped FFN implementation',
                'entries':self.energy,'area_entries':area,'energy_pj':energy,'area_um2':total_area,'complete':complete,
                'category_coverage':coverage,'installed_area_policy':'sum distinct component/interface mechanisms, fixed matrix max-row by max-column envelope and independent I/O maxima, times capacity; separate FIFOs, junctions and support',
                'memory_bytes':self.memory_bytes},
            'output':copy.deepcopy(self.values[output]),'invocations':invocation_quality,'owned_weight_states':self.owned_states,'quality':quality,'diagnostics':[{'code':v} for v in violations],
            'backend_evaluations':self.results,'assumptions':[
                'all reference coefficients are explicitly hypothetical; no fabricated-device calibration or speedup claim',
                'bounded feed-forward batch; complete intermediate materialization in separately charged inter-component FIFOs',
                'inclusive local tasks reserve their complete converter/memory/control/source resources until completion',
                'each matrix composite owns its private I/O converters and sources; converter/source pools are shared service-capacity constraints, not an extra hardware area charge',
                'coherent fused up/gate/down and selected alignment devices form one local assembly with paid intrinsic propagation/loss; external optical routing is unsupported',
                'separate weight versions are programmed on each matrix invocation; no unproved free resident-weight reuse',
                'shared analog/photonic matrix pools execute full fixed rectangular arrays with zero padding; unused analog rows retain their modeled common-mode loading, and all lanes/cells remain charged',
                'digital zero-input concatenation and output selection are explicit paid materializations; inactive physical isolation is never assumed',
                'installed-resource static energy accrues over the complete system makespan, including waiting and setup',
                'optical fields cannot wait or fan out implicitly; each wire consumes an owned zero-retention excitation',
                'quantity bounds and missing coefficients remain conditional; point totals never silently substitute zero'],
            'workload_accounting':{'reuse_count':self.w['reuse_count'],'invocation_policy':'serial complete FFN invocations; matrix weights reprogrammed per invocation',
                'cold_workload_latency_ns':elapsed,'cold_first_invocation_latency_ns':completion_times[0],
                'steady_invocation_latency_samples_ns':steady_intervals,
                'steady_mean_latency_ns':math.fsum(steady_intervals)/len(steady_intervals) if steady_intervals else None,
                'amortized_latency_ns':elapsed/self.w['reuse_count'],
                'cold_workload_energy_pj':energy,'amortized_energy_pj':quantity_scale(energy,1/self.w['reuse_count'],'pJ'),
                'energy_scope':'complete cold workload and its arithmetic amortization; no isolated steady energy measurement is inferred'}}
        return result


def _quantity_max(values,unit):
    values=list(values)
    if any(q['state']=='unavailable' for q in values): return {'state':'unavailable','unit':unit,'reason':'installed footprint/static envelope has unresolved contributions'}
    if all(q['state']=='known' for q in values): return _known(max((q['value'] for q in values),default=0.),unit)
    bounds={}
    for key in ('lower','upper'):
        parts=[q['value'] if q['state']=='known' else q.get(key) for q in values]
        bounds[key]=None if any(v is None for v in parts) else max(parts,default=0.)
    return {'state':'bounded','unit':unit,'symbol':'installed_max',**bounds}


def evaluate_implementation(project,spec=None):
    """Compile, execute and pin one complete bounded FFN hardware implementation."""
    from .io import normalize_project
    from .rewrites import transform_ffn
    normalized=normalize_project({k:copy.deepcopy(v) for k,v in project.items() if k!='input_hash'})
    if normalized['model']['semantic_track']=='R': fail('model.semantic_track','Track R adaptation is unsupported by this compiler','unavailable_model')
    integer(normalized['workload']['reuse_count'],'workload.reuse_count',1,MAX_INVOCATIONS)
    spec=normalize_implementation(default_implementation(normalized['model']) if spec is None else spec,normalized['model'])
    transformed=transform_ffn(normalized['model'],spec['transform']['permutation'],spec['transform']['signs'])
    compiler=_Compiler(normalized,spec); input_id=compiler.input(normalized['workload']['inputs'])
    previous=None; output=None
    for repeat in range(normalized['workload']['reuse_count']):
        prefix=f'run{repeat}.'; deps=[] if previous is None else [previous]
        gate=compiler.linear('gate',input_id,transformed['model'],suffix=prefix,extra_deps=deps,release=spec['alignment']['gate_release_ns'])
        gate=compiler.digital(prefix+'silu','silu',[gate])
        if spec['fusion']=='unfused':
            up=compiler.linear('up',input_id,transformed['model'],suffix=prefix,extra_deps=deps)
            hidden=compiler.digital(prefix+'multiply','multiply',[up,gate])
            output=compiler.linear('down',hidden,transformed['model'],suffix=prefix)
        else:
            # Delay the physical launch until the digital control exists. The
            # earlier control is held in charged electrical storage; no optical
            # wait is silently granted. Other mechanisms add real storage work.
            method=spec['alignment']['method']; delay=spec['alignment']['delay_ns']
            launch_deps=[prefix+'down.0.program']+([gate] if method=='delayed_launch' else deps)
            up=compiler.linear('up',input_id,transformed['model'],suffix=prefix,field_output=True,extra_deps=launch_deps)
            if method=='optical_delay': up=compiler.interface(prefix+'alignment','optical_delay',[up],delay)
            elif method=='electronic_buffer':
                digital_value=compiler.interface(prefix+'alignment.readout','readout',[up])
                buffered=compiler.interface(prefix+'alignment.buffer','digital_buffer',[digital_value],delay)
                up=compiler.interface(prefix+'alignment.encode','encode',[buffered],extra_deps=[gate])
            elif method=='recompute':
                # A discarded first realization plus a new matrix execution.
                compiler.interface(prefix+'discard.readout','readout',[up])
                up=compiler.linear('up',input_id,transformed['model'],suffix=prefix+'recompute.',field_output=True,
                                   extra_deps=[prefix+'discard.readout',gate],release=delay)
            elif delay:
                # Explicit launch timestamp, not retention of a produced field.
                next(t for t in compiler.tasks if t['id']==up)['release_ns']=delay
            hidden=compiler.interface(prefix+'signed_gate','signed_gate',[up,gate])
            field_out=compiler.linear('down',hidden,transformed['model'],suffix=prefix,field_output=True,field_input=True)
            output=compiler.interface(prefix+'output.readout','readout',[field_out])
        previous=output
        compiler.invocation_outputs.append(output)
    result=compiler.finalize(output,{'operator':'complete_swiglu','transform':transformed['proof'],
                                      'partitions':spec['operators'],'fusion':spec['fusion'],
                                      'bias_rule':'every matrix bias applied once after partition reduction or paid coherent field injection'})
    from .interfaces import interface_catalog
    result['interfaces']=interface_catalog(spec['interfaces'])
    result['dependencies']={'compiler_version':VERSION,'project_hash':canonical_hash(normalized),
                            'implementation_hash':canonical_hash(spec),'geometry_hash':canonical_hash(result['geometry']),
                            'schedule_hash':canonical_hash(result['schedule']),'technology_hash':canonical_hash(normalized['technology']),
                            'seed_policy':{'physical_seed':normalized['workload']['seed'],'search_seed':'not used for fixed compilation'}}
    result['record_hash']=canonical_hash(result)
    return result


def check_implementation(record):
    """Full deterministic recomputation; an updated outer digest cannot bless edits."""
    try:
        if not isinstance(record,dict) or record.get('schema_version')!=RESULT_SCHEMA: raise ValueError('unsupported implementation record')
        from .composition_validation import validate_implementation_replay_inputs
        validate_implementation_replay_inputs(record)
        supplied=record.get('record_hash')
        content={k:v for k,v in record.items() if k!='record_hash'}
        if supplied!=canonical_hash(content): return {'valid':False,'reason':'record hash mismatch'}
        replay=evaluate_implementation(record['normalized_inputs']['project'],record['normalized_inputs']['implementation'])
        valid=canonical_hash(replay)==canonical_hash(record)
        return {'valid':valid,'reason':'complete deterministic replay matches' if valid else 'recomputed implementation differs',
                'scope':'replay self-consistency, not independent physical validation'}
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return {'valid':False,'reason':str(exc)}
