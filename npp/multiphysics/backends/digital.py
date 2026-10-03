"""Complete hypothetical digital FFN accounting, separate from CPU timings."""
from __future__ import annotations
import copy
import math
import numpy as np

from ..semantic import FFNWorkload, NumericPolicy, evaluate_ffn, operation_counts
from . import build_ledger, error_record, evaluation_identity, fail

BACKEND_ID = 'digital.ffn'
BACKEND_VERSION = '1.0'
COST_FIELDS = {
    'mac_energy_pj', 'add_energy_pj', 'silu_energy_pj', 'multiply_energy_pj',
    'sram_read_energy_pj_per_byte','sram_write_energy_pj_per_byte',
    'dram_read_energy_pj_per_byte','link_energy_pj_per_byte',
    'program_energy_pj_per_byte','launch_energy_pj', 'static_power_mw',
    'compute_area_um2','sram_area_um2_per_byte',
}
PHYSICAL_FIELDS = {'macs_per_ns','scalar_ops_per_ns','sram_bytes_per_ns','dram_bytes_per_ns',
                   'link_bytes_per_ns','program_bytes_per_ns','launch_ns','sram_capacity_bytes'}


def validate_parameters(parameters):
    if not isinstance(parameters, dict):
        fail('parameters', 'expected an object')
    expected = COST_FIELDS | PHYSICAL_FIELDS
    if set(parameters) != expected:
        fail('parameters', f'missing fields {sorted(expected-set(parameters))}; unknown fields {sorted(set(parameters)-expected)}')
    p = copy.deepcopy(parameters)
    for name in PHYSICAL_FIELDS:
        value = p[name]
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0 or (name != 'launch_ns' and value == 0):
            fail('parameters.'+name, 'expected a finite nonnegative launch time or positive capacity/rate')
    for name in PHYSICAL_FIELDS - {'sram_capacity_bytes'}:
        if p[name]>1e15 or (name!='launch_ns' and p[name]<1e-15):
            fail('parameters.'+name,'outside supported local numeric envelope [1e-15,1e15]', 'envelope_violation')
    if type(p['sram_capacity_bytes']) is not int:
        fail('parameters.sram_capacity_bytes','expected an integer byte capacity')
    from . import cost_quantity, coefficient_unit
    for name in COST_FIELDS:
        unit=coefficient_unit(name)
        p[name]=cost_quantity(p[name],unit,'parameters.'+name)
    return p


def evaluate_ffn_cost(workload, inputs, parameters, context=None):
    """Account for a complete FFN and explicit repeated invocations.

    The local reference serializes compute, transfers and control. This is an
    intentionally conservative, reproducible resource model, not GPU timing or
    a contention-aware system schedule. Weight programming is paid once; SRAM
    reads and interconnect transfers are paid on every invocation.
    """
    model = workload if isinstance(workload, FFNWorkload) else FFNWorkload.from_dict(workload)
    if model.semantic_track=='R':
        fail('model.semantic_track','Track R hardware execution requires adaptation accounting and a matched parent-model baseline; metadata alone is unsupported','unavailable_model')
    p = validate_parameters(parameters)
    c = copy.deepcopy(context or {})
    allowed = {'numeric_policy','reuse_count','seed','weight_residency','owner_id','environment','geometry','schedule','fidelity','seed_policy'}
    if set(c)-allowed:
        fail('context', f'unknown context fields {sorted(set(c)-allowed)}')
    c.setdefault('numeric_policy', NumericPolicy().to_dict())
    c.setdefault('reuse_count',1)
    c.setdefault('seed',0)
    c.setdefault('weight_residency','sram')
    c.setdefault('owner_id','digital_ffn')
    if type(c['reuse_count']) is not int or not 1 <= c['reuse_count'] <= 10**6:
        fail('context.reuse_count','expected integer in [1,1000000]')
    if type(c['seed']) is not int or not 0 <= c['seed'] < 2**32:
        fail('context.seed','expected integer in [0,2**32)')
    if c['weight_residency'] not in ('sram','dram'):
        fail('context.weight_residency','supported weight residency is sram or dram')
    if not isinstance(c['owner_id'],str) or not c['owner_id']:
        fail('context.owner_id','expected nonempty owner identifier')
    policy = NumericPolicy.from_dict(c['numeric_policy'])
    semantic = evaluate_ffn(model, inputs, policy)
    x = np.asarray(inputs,dtype=float)
    batch = 1 if x.ndim==1 else x.shape[0]
    counts = operation_counts(model,batch)
    count = c['reuse_count']
    byte_width = 8 if policy.format=='float64' else 4
    # Materialized unfused FFN: weights read once per batch invocation and
    # each intermediate written once then read once. Matrix-matrix batch reuse
    # is explicit here, not an assumed per-MAC free-weight access.
    weight_bytes = (counts['weight_elements']+counts['bias_elements'])*byte_width
    input_bytes = counts['input_elements']*byte_width
    output_bytes = counts['output_elements']*byte_width
    intermediate_bytes = counts['intermediate_elements']*byte_width
    workspace_bytes = input_bytes+output_bytes+intermediate_bytes
    required_sram = workspace_bytes+(weight_bytes if c['weight_residency']=='sram' else 0)
    if required_sram > p['sram_capacity_bytes']:
        fail('resources.sram',f'requires {required_sram} bytes; capacity is {p["sram_capacity_bytes"]}', 'resource_infeasibility')
    reads = (2*input_bytes+intermediate_bytes+(weight_bytes if c['weight_residency']=='sram' else 0))*count
    writes = (intermediate_bytes+output_bytes)*count
    dram_bytes = weight_bytes if c['weight_residency']=='sram' else weight_bytes*count
    link_bytes = (2*input_bytes+output_bytes+2*intermediate_bytes+weight_bytes)*count + dram_bytes
    program_bytes = weight_bytes
    scalar_ops = counts['bias_add']+counts['silu']+counts['elementwise_multiply']
    times = {
        'compute_ns':(counts['matrix_mac']/p['macs_per_ns']+scalar_ops/p['scalar_ops_per_ns'])*count,
        'sram_ns':(reads+writes)/p['sram_bytes_per_ns'],
        'dram_ns':dram_bytes/p['dram_bytes_per_ns'],
        'link_ns':link_bytes/p['link_bytes_per_ns'],
        'programming_ns':program_bytes/p['program_bytes_per_ns'],
        'control_ns':p['launch_ns']*count,
    }
    duration = math.fsum(times.values())
    entries = [
        ('compute','matrix_mac',p['mac_energy_pj'],counts['matrix_mac']*count),
        ('compute','bias_add',p['add_energy_pj'],counts['bias_add']*count),
        ('compute','silu',p['silu_energy_pj'],counts['silu']*count),
        ('compute','elementwise_multiply',p['multiply_energy_pj'],counts['elementwise_multiply']*count),
        ('movement','sram_reads',p['sram_read_energy_pj_per_byte'],reads),
        ('movement','sram_writes',p['sram_write_energy_pj_per_byte'],writes),
        ('movement','dram_weight_loads',p['dram_read_energy_pj_per_byte'],dram_bytes),
        ('movement','local_interconnect',p['link_energy_pj_per_byte'],link_bytes),
        ('programming','weight_programming',p['program_energy_pj_per_byte'],program_bytes),
        ('control','launch',p['launch_energy_pj'],count),
        ('support','static_energy',p['static_power_mw'],duration),
    ]
    ledger=build_ledger(c['owner_id'],entries,duration,[('compute',p['compute_area_um2'],1),('sram',p['sram_area_um2_per_byte'],p['sram_capacity_bytes'])])
    result={'backend_id':BACKEND_ID,'backend_version':BACKEND_VERSION,
            'status':'model_feasible' if ledger['complete'] else 'conditional',
            'ideal_output':semantic['ideal_output'],'actual_output':semantic['output'],
            'error':error_record(semantic['ideal_output'],semantic['output'],mode='ideal_diagnostic' if policy.format=='float64' else 'finite'),
            'semantic':semantic,'ledger':ledger,'local_duration_ns':duration,
            'schedule_status':'local_serial_model_not_system_schedule', 'duration_breakdown':times,
            'operations':counts,'resources':{'compute_engines':1,'sram_bytes':required_sram,'weight_bytes':weight_bytes,
                                           'sram_read_bytes':reads,'sram_write_bytes':writes,'dram_bytes':dram_bytes,'link_bytes':link_bytes},
            'state':{'owner_id':c['owner_id'],'weight_version':model.model_identity,'programmed_once':True,'reuse_count':count},
            'dependencies':evaluation_identity(BACKEND_ID,BACKEND_VERSION,{'model':model.to_dict(),'inputs':x.tolist()},p,c),
            'assumptions':['hypothetical resource coefficients; not measured CPU, GPU or ASIC timing',
                           'matrix-matrix batch reuses one weight load per invocation; matrix-vector is batch one',
                           'both input branches read SRAM separately; all intermediates written and read via charged links; no free broadcast or fusion',
                           'weight programming means register/SRAM configuration after external load; separate energy owners',
                           'serial local compute, movement and control; no system overlap or contention assumed',
                           'input dataset acquisition and task-level cache workloads are outside this FFN boundary']}

    result['quality_checks']=[]
    if semantic['quality_passed'] is not None:
        result['quality_checks'].append({'name':'model.max_absolute_error','passed':semantic['quality_passed'],
                                         'limit':semantic['quality_budget'],'observed':semantic['absolute_error']['max']})
    if semantic['quality_passed'] is False:
        result['status']='resource_infeasible'
        result['diagnostics']=[{'code':'quality_budget_exceeded','path':'model.quality_budget',
                               'message':'numeric execution exceeds the model declared maximum absolute error budget'}]
    return result


def _primitive_context(context, owner):
    """Normalize one explicit primitive; graph scheduling owns state existence."""
    if context is not None and not isinstance(context, dict):
        fail('context', 'expected an object')
    c = copy.deepcopy(context or {})
    allowed = {'numeric_policy','reuse_count','seed','weight_residency','owner_id',
               'environment','geometry','schedule','fidelity','seed_policy',
               'weight_state','weight_version','indices'}
    if set(c) - allowed:
        fail('context', f'unknown fields {sorted(set(c)-allowed)}')
    c.setdefault('numeric_policy', NumericPolicy().to_dict())
    c.setdefault('reuse_count', 1)
    c.setdefault('seed', 0)
    c.setdefault('weight_residency', 'sram')
    c.setdefault('owner_id', owner)
    c.setdefault('weight_state', 'cold')
    for key, upper in (('reuse_count', 10**6), ('seed', 2**32-1)):
        if type(c[key]) is not int or not (1 if key == 'reuse_count' else 0) <= c[key] <= upper:
            fail('context.'+key, 'integer outside supported range')
    if c['weight_residency'] not in ('sram', 'dram'):
        fail('context.weight_residency', 'expected sram or dram')
    if c['weight_state'] not in ('cold', 'programmed'):
        fail('context.weight_state', 'expected cold or programmed')
    if not isinstance(c['owner_id'], str) or not c['owner_id']:
        fail('context.owner_id', 'expected a nonempty owner identifier')
    return c, NumericPolicy.from_dict(c['numeric_policy'])


def linear_weight_version(matrix, bias=None, policy=None):
    """Digest of actual configured operands and arithmetic contract.

    A matching digest alone does not establish stored state. An orchestrator
    must bind it to an owned live state record before selecting programmed mode.
    """
    import hashlib
    import json
    from ..semantic import _array, _tuple_vector
    p = NumericPolicy() if policy is None else policy
    if not isinstance(p, NumericPolicy):
        fail('numeric_policy', 'expected NumericPolicy')
    weights = _array(matrix, 'matrix', (2,))
    offsets = _tuple_vector(bias, 'bias', weights.shape[0])
    snapshot = {'weights': weights.tolist(), 'bias': list(offsets), 'numeric_policy': p.to_dict()}
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _primitive_result(kind, inputs, ideal, actual, parameters, context, counts,
                      weight_elements=0, bias_elements=0, weight_version=None, model=None):
    """Shared accounting only; primitive transfer laws stay in digital kernels."""
    from . import finite
    p = parameters
    c, policy = context
    count = c['reuse_count']
    byte_width = 8 if policy.format == 'float64' else 4
    input_bytes = sum(np.asarray(x).size for x in inputs) * byte_width
    output_bytes = np.asarray(actual).size * byte_width
    weight_bytes = (weight_elements + bias_elements) * byte_width
    warm = c['weight_state'] == 'programmed'
    if warm:
        if weight_version is None:
            fail('context.weight_state', 'a weightless operator has no programmed weight state')
        if c.get('weight_version') != weight_version:
            fail('context.weight_version', 'programmed state must match all weights, biases and numerical policy')
    elif 'weight_version' in c and c['weight_version'] != weight_version:
        fail('context.weight_version', 'cold requested weight version does not match explicit operands')
    required_sram = input_bytes + output_bytes + (weight_bytes if c['weight_residency'] == 'sram' else 0)
    if required_sram > p['sram_capacity_bytes']:
        fail('resources.sram', f'requires {required_sram} bytes; capacity is {p["sram_capacity_bytes"]}', 'resource_infeasibility')
    reads = (input_bytes + (weight_bytes if c['weight_residency'] == 'sram' else 0)) * count
    writes = output_bytes * count
    dram_bytes = (0 if warm else weight_bytes) if c['weight_residency'] == 'sram' else weight_bytes * count
    program_bytes = 0 if warm else weight_bytes
    link_bytes = (input_bytes + output_bytes + weight_bytes) * count + dram_bytes
    scalar = counts.get('bias_add', 0) + counts.get('add', 0) + counts.get('silu', 0) + counts.get('elementwise_multiply', 0)
    times = {
        'dram_ns': dram_bytes / p['dram_bytes_per_ns'],
        'programming_ns': program_bytes / p['program_bytes_per_ns'],
        'sram_ns': (reads + writes) / p['sram_bytes_per_ns'],
        'link_ns': link_bytes / p['link_bytes_per_ns'],
        'control_ns': p['launch_ns'] * count,
        'compute_ns': (counts.get('matrix_mac', 0) / p['macs_per_ns'] + scalar / p['scalar_ops_per_ns']) * count,
    }
    duration = finite(math.fsum(times.values()), 'local_duration_ns')
    entries = [
        ('compute', 'matrix_mac', p['mac_energy_pj'], counts.get('matrix_mac', 0) * count),
        ('compute', 'bias_add', p['add_energy_pj'], counts.get('bias_add', 0) * count),
        ('compute', 'scalar_add', p['add_energy_pj'], counts.get('add', 0) * count),
        ('compute', 'silu', p['silu_energy_pj'], counts.get('silu', 0) * count),
        ('compute', 'elementwise_multiply', p['multiply_energy_pj'], counts.get('elementwise_multiply', 0) * count),
        ('movement', 'sram_reads', p['sram_read_energy_pj_per_byte'], reads),
        ('movement', 'sram_writes', p['sram_write_energy_pj_per_byte'], writes),
        ('movement', 'dram_weight_loads', p['dram_read_energy_pj_per_byte'], dram_bytes),
        ('movement', 'local_interconnect', p['link_energy_pj_per_byte'], link_bytes),
        ('programming', 'weight_programming', p['program_energy_pj_per_byte'], program_bytes),
        ('control', 'launch', p['launch_energy_pj'], count),
        ('support', 'static_energy', p['static_power_mw'], duration),
    ]
    ledger = build_ledger(c['owner_id'], entries, duration, [('compute', p['compute_area_um2'], 1),
                         ('sram', p['sram_area_um2_per_byte'], p['sram_capacity_bytes'])])
    phase_resources = {'dram_ns': ['dram'], 'programming_ns': ['programmer', 'sram'],
                       'sram_ns': ['sram'], 'link_ns': ['link'], 'control_ns': ['control'], 'compute_ns': ['compute']}
    phases = [{'name': key.removesuffix('_ns'), 'duration_ns': value, 'resources': phase_resources[key]}
              for key, value in times.items()]
    output = np.asarray(actual).tolist()
    reference = np.asarray(ideal).tolist()
    backend = 'digital.' + kind
    snapshot = {'inputs': [np.asarray(x).tolist() for x in inputs], 'model': model}
    return {
        'backend_id': backend, 'backend_version': '1.0',
        'status': 'model_feasible' if ledger['complete'] else 'conditional',
        'ideal_output': reference, 'actual_output': output,
        'error': error_record(reference, output, mode='ideal_diagnostic' if policy.format == 'float64' else 'finite'),
        'ledger': ledger, 'local_duration_ns': duration, 'duration_breakdown': times,
        'schedule_status': 'local_serial_model_not_system_schedule',
        'phases': phases, 'inclusive_resources': ['compute', 'sram', 'dram', 'link', 'programmer', 'control'],
        'reservation_policy': 'inclusive_all_resources_for_local_duration',
        'operations': dict(counts, input_elements=sum(np.asarray(x).size for x in inputs),
                           output_elements=np.asarray(actual).size, weight_elements=weight_elements, bias_elements=bias_elements),
        'resources': {'compute_engines': 1, 'sram_bytes': required_sram,
                      'installed_sram_bytes': p['sram_capacity_bytes'], 'weight_bytes': weight_bytes,
                      'sram_read_bytes': reads, 'sram_write_bytes': writes, 'dram_bytes': dram_bytes,
                      'link_bytes': link_bytes, 'program_bytes': program_bytes},
        'state': {'owner_id': c['owner_id'], 'weight_version': weight_version,
                  'initial_weight_state': c['weight_state'], 'final_weight_state': 'programmed' if weight_bytes else 'not_applicable',
                  'programming_required': bool(program_bytes), 'reuse_count': count,
                  'precondition': 'programmed mode requires an owned live state verified by the graph; digest matching alone does not prove prior programming'},
        'dependencies': evaluation_identity(backend, '1.0', snapshot, p, c),
        'assumptions': [
            'hypothetical resource coefficients, not measured processor timing',
            'local serial compute, movement, programming and control; graph scheduler must reserve shared physical instances',
            'primitive boundary materializes every output and charges all input reads; no implicit fusion or free gather/concat',
            'weights reused within one explicit batch; repeated invocations pay SRAM/link traffic and DRAM traffic when not resident',
            'area describes installed engine capacity and must be deduplicated by physical instance in a graph',
        ],
    }


def evaluate_linear_cost(matrix, inputs, bias, parameters, context=None):
    """One matrix kernel, with separate bias work and explicit cold/warm state."""
    from ..semantic import _array, _tuple_vector, _reference_linear, linear
    p = validate_parameters(parameters)
    c, policy = _primitive_context(context, 'digital_linear')
    if 'indices' in c:
        fail('context.indices', 'indices are only meaningful for gather')
    w, x = _array(matrix, 'matrix', (2,)), _array(inputs, 'inputs', (1, 2))
    b = _tuple_vector(bias, 'bias', w.shape[0])
    actual = linear(w, x, b, policy)
    batch = 1 if x.ndim == 1 else x.shape[0]
    ideal = _reference_linear(w, x.reshape(batch, -1), b)
    if x.ndim == 1:
        ideal = ideal[0]
    counts = {'matrix_mac': batch * w.size, 'bias_add': np.asarray(actual).size,
              'execution': 'matrix_vector' if batch == 1 else 'batched_matrix_vector'}
    version = linear_weight_version(w, b, policy)
    return _primitive_result('linear', [x], ideal, actual, p, (c, policy), counts,
                             w.size, len(b), version, {'weights': w.tolist(), 'bias': list(b)})


def evaluate_elementwise_cost(kind, inputs, parameters, context=None):
    """Digital scalar operators and explicit gather/concatenation materialization.

    Inputs is a list of arrays: one for SiLU/gather, two for add/multiply/add_bias,
    and one or more for concat. Gather uses explicit context.indices on the last
    axis. Only add_bias broadcasts, and only its declared one-dimensional bias.
    """
    from ..semantic import _array, _cast, add, multiply, silu, MAX_BATCH, MAX_DIMENSION
    p = validate_parameters(parameters)
    c, policy = _primitive_context(context, 'digital_' + str(kind))
    if kind not in ('silu', 'add', 'multiply', 'add_bias', 'gather', 'concat'):
        fail('kind', 'supported operators are silu, add, multiply, add_bias, gather, concat')
    if not isinstance(inputs, (list, tuple)) or not 1 <= len(inputs) <= 256:
        fail('inputs', 'expected a bounded list of operand arrays')
    arity = 1 if kind in ('silu', 'gather') else 2
    if kind != 'concat' and len(inputs) != arity:
        fail('inputs', f'{kind} requires {arity} operands')
    arrays = [_array(x, f'inputs[{i}]', (1, 2)) for i, x in enumerate(inputs)]
    if any(x.ndim == 2 and len(x) > MAX_BATCH for x in arrays):
        fail('inputs', f'batch exceeds {MAX_BATCH}')
    if kind != 'gather' and 'indices' in c:
        fail('context.indices', 'indices are only meaningful for gather')
    counts = {'matrix_mac': 0, 'bias_add': 0, 'add': 0, 'silu': 0, 'elementwise_multiply': 0}
    if kind == 'silu':
        ideal, actual = silu(arrays[0]), silu(arrays[0], policy)
        counts['silu'] = arrays[0].size
    elif kind in ('add', 'multiply'):
        function = add if kind == 'add' else multiply
        ideal, actual = function(*arrays), function(*arrays, policy)
        counts['add' if kind == 'add' else 'elementwise_multiply'] = arrays[0].size
    elif kind == 'add_bias':
        x, bias = arrays
        if bias.ndim != 1 or bias.shape[0] != x.shape[-1]:
            fail('inputs[1]', 'add_bias requires one bias per last-axis output channel')
        repeated = np.broadcast_to(bias, x.shape).copy()
        ideal, actual = add(x, repeated), add(x, repeated, policy)
        counts['bias_add'] = x.size
    elif kind == 'gather':
        indices = c.get('indices')
        if not isinstance(indices, list) or not 1 <= len(indices) <= MAX_DIMENSION or any(type(i) is not int or not 0 <= i < arrays[0].shape[-1] for i in indices):
            fail('context.indices', 'gather requires a bounded explicit list of valid integer last-axis indices')
        ideal = arrays[0][..., indices].copy()
        actual = _cast(ideal, getattr(np, policy.format), 'gather')
    else:
        if any(x.ndim != arrays[0].ndim or x.shape[:-1] != arrays[0].shape[:-1] for x in arrays):
            fail('inputs', 'concat operands must agree on all axes except the last')
        if sum(x.shape[-1] for x in arrays) > MAX_DIMENSION:
            fail('inputs', 'concatenated width exceeds supported dimension')
        ideal = np.concatenate(arrays, axis=-1)
        actual = _cast(ideal, getattr(np, policy.format), 'concat')
    return _primitive_result(kind, arrays, ideal, actual, p, (c, policy), counts)
