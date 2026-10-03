"""Structural and portable-replay guards for the M2 implementation graph.

Structural validity is not an independent physical validation. Replay inputs
must be fully embedded canonical snapshots before any normalizer may resolve
friendly project files or checkpoint references.
"""
from __future__ import annotations

from collections import deque
import re

from .contracts import SignalContract, canonical_hash, canonical_json, signal_mismatches
from .io import keys


_ARITY = {
    'input': 0, 'digital.linear': 1, 'digital.silu': 1, 'digital.add': 2,
    'digital.add_bias': 2, 'digital.multiply': 2, 'digital.gather': 1,
    'digital.concat': None, 'analog.composite': 1, 'photonic.composite': 1,
    'interface.encode': 1, 'interface.readout': 1, 'interface.signed_gate': 2,
    'interface.optical_delay': 1, 'interface.digital_buffer': 1,
    'interface.recompute': 1, 'interface.dac': 1, 'interface.adc': 1,
    'analog.tile': 1, 'photonic.mesh': 1, 'photonic.coherent_bias': 2,
}
_NODE_FIELDS = {'id', 'operation', 'ports', 'resource', 'schedule_id', 'component_id',
                'accounting', 'weight_version', 'physical_subgraph', 'accounting_parent',
                'physical_payload', 'evaluator_id', 'evaluator_version', 'parameters'}


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.:-]{0,159}', value):
        raise ValueError(label + ': expected a bounded identifier')


def _is_digital(signal):
    return (signal.carrier == 'electrical' and signal.regime == 'classical_digital'
            and signal.encoding == 'signed_numeric' and signal.physical_variable == 'digital_word'
            and signal.unit == '1' and signal.scale == 1.0 and signal.signed)


def _is_field(signal):
    return (signal.carrier == 'optical' and signal.regime == 'classical_analog'
            and signal.encoding == 'signed_coherent_field' and signal.physical_variable == 'field_amplitude'
            and signal.unit == 'sqrt(mW)' and signal.signed and signal.phase_reference is not None)


def _is_voltage(signal):
    return (signal.carrier == 'electrical' and signal.regime == 'classical_analog'
            and signal.encoding == 'bipolar_voltage' and signal.physical_variable == 'voltage'
            and signal.unit == 'V' and signal.signed)


def _is_current(signal):
    return (signal.carrier == 'electrical' and signal.regime == 'classical_analog'
            and signal.encoding == 'differential_current' and signal.physical_variable == 'current'
            and signal.unit == 'A' and signal.signed)


def _native_signal(signal):
    """Wire agreement cannot register an arbitrary encoding as an operator ABI.

    Digital words encode dimensionless values at unit scale. Voltage, current
    and coherent field scales remain explicit positive decoding factors: those
    factors are selected by the registered DAC, tile, mesh and gating models.
    Their numerical transfer is checked by evaluator replay, not by relabeling
    an otherwise uniformly compatible wire.
    """
    if not any(check(signal) for check in (_is_digital, _is_field, _is_voltage, _is_current)):
        raise ValueError('unregistered native signal representation, units or digital scale')
    expected_axes = ('lane',) if len(signal.shape) == 1 else ('batch', 'lane')
    if signal.payload != 'tensor' or len(signal.shape) not in (1, 2) or signal.axis_order != expected_axes:
        raise ValueError('registered FFN operators require a lane tensor or batch/lane tensor')


def _operation_ports(operation, signals):
    inputs = [signals['in'+str(i)] for i in range(len(signals)-1)]
    output = signals['out']
    if operation == 'input':
        return
    if operation.startswith('digital.') or operation in ('interface.digital_buffer', 'interface.recompute'):
        if not all(_is_digital(s) for s in inputs + [output]):
            raise ValueError(operation + ': digital operators require digital word ports; conversion cannot be implicit')
    elif operation in ('interface.encode', 'interface.readout'):
        correct = (_is_digital(inputs[0]) and _is_field(output)) if operation.endswith('encode') else (_is_field(inputs[0]) and _is_digital(output))
        if not correct:
            raise ValueError(operation + ': illegal conversion direction or encoding')
    elif operation in ('photonic.mesh', 'interface.optical_delay'):
        if not all(_is_field(s) for s in inputs + [output]):
            raise ValueError(operation + ': requires signed coherent fields')
    elif operation in ('interface.signed_gate', 'photonic.coherent_bias'):
        if not (_is_field(inputs[0]) and _is_digital(inputs[1]) and _is_field(output)):
            raise ValueError(operation + ': requires coherent signal and explicit digital control/bias')
    elif operation == 'interface.dac':
        if not (_is_digital(inputs[0]) and _is_voltage(output)):
            raise ValueError('DAC requires digital words -> declared analog voltage')
    elif operation == 'interface.adc':
        if not ((_is_voltage(inputs[0]) or _is_current(inputs[0])) and _is_digital(output)):
            raise ValueError('ADC requires analog electrical readout -> digital words')
    elif operation == 'analog.tile':
        if not (_is_voltage(inputs[0]) and _is_current(output)):
            raise ValueError('conductance tile requires voltage input and current output')
    elif operation == 'analog.composite':
        if not all(_is_digital(s) for s in inputs + [output]):
            raise ValueError('analog composite boundary requires its included DAC/ADC digital ports')
    elif operation == 'photonic.composite':
        if not all(_is_digital(s) or _is_field(s) for s in inputs + [output]):
            raise ValueError('photonic composite requires explicitly encoded digital or coherent-field ports')
    if operation in ('digital.silu', 'interface.encode', 'interface.readout', 'interface.dac',
                     'interface.adc', 'interface.optical_delay', 'interface.digital_buffer', 'interface.recompute'):
        if inputs[0].shape != output.shape:
            raise ValueError(operation + ': unary conversion/activation must preserve shape')
    if operation in ('digital.add', 'digital.multiply', 'interface.signed_gate'):
        if not all(s.shape == output.shape for s in inputs):
            raise ValueError(operation + ': pointwise operands must match the output shape')
    if operation in ('digital.add_bias', 'photonic.coherent_bias'):
        if inputs[0].shape != output.shape or inputs[1].shape not in (output.shape, (output.shape[-1],)):
            raise ValueError(operation + ': bias must match output channels with declared batch broadcasting')
    if operation == 'digital.concat':
        if any(s.shape[:-1] != output.shape[:-1] for s in inputs) or sum(s.shape[-1] for s in inputs) != output.shape[-1]:
            raise ValueError('concat shape must equal ordered last-axis concatenation')


def validate_graph_structure(graph, *, _depth=0, _parent=None):
    """Check every declared port, wire, DAG dependency and optical consumption."""
    if _depth > 8:
        raise ValueError('physical subgraph nesting exceeds supported depth')
    if (_depth == 0) != (_parent is None):
        raise ValueError('nested validation requires its actual enclosing composite')
    canonical_json(graph)
    keys(graph, {'schema_version', 'nodes', 'edges'}, {'schema_version', 'nodes', 'edges'}, 'graph')
    if graph['schema_version'] != 'npp-typed-graph-1':
        raise ValueError('unsupported typed graph schema')
    if not isinstance(graph['nodes'], list) or not 1 <= len(graph['nodes']) <= 16384:
        raise ValueError('graph requires a bounded nonempty node list')
    if not isinstance(graph['edges'], list) or len(graph['edges']) > 65536:
        raise ValueError('graph requires a bounded edge list')
    nodes, ports = {}, {}
    nested = 0
    for node in graph['nodes']:
        keys(node, _NODE_FIELDS, {'id', 'operation', 'ports', 'resource', 'schedule_id'}, 'graph.node')
        name = node['id']
        _identifier(name, 'node.id')
        _identifier(node['resource'], 'node.resource')
        if name in nodes:
            raise ValueError('duplicate graph node identifier')
        operation = node['operation']
        if not isinstance(operation, str) or operation not in _ARITY:
            raise ValueError('unregistered graph operation')
        if not isinstance(node['ports'], dict) or 'out' not in node['ports']:
            raise ValueError('each node requires one declared out port')
        arity = _ARITY[operation]
        count = len(node['ports']) - 1
        if count < 0 or (arity is None and count < 1) or (arity is not None and count != arity):
            raise ValueError(operation + ': incorrect input arity')
        if set(node['ports']) != {'out', *('in'+str(i) for i in range(count))}:
            raise ValueError('input ports must be consecutive in0, in1, ...; outputs use out')
        if operation == 'input':
            if node['schedule_id'] is not None:
                raise ValueError('input cannot claim a separately scheduled operation')
        else:
            _identifier(node['schedule_id'], 'node.schedule_id')
        if _depth:
            if node.get('accounting_parent') != _parent['id']:
                raise ValueError('nested accounting_parent must identify the actual enclosing composite')
            if node.get('accounting') != 'included_in_parent':
                raise ValueError('nested disclosure cannot independently own or duplicate composite costs')
            if operation != 'input' and node['schedule_id'] != _parent['schedule_id']:
                raise ValueError('nested disclosure must use its enclosing composite schedule reservation')
            if not isinstance(node.get('physical_payload'), dict):
                raise ValueError('nested disclosure requires an owned physical payload object')
        elif 'accounting_parent' in node:
            raise ValueError('accounting_parent belongs only to a nested composite disclosure')
        signals = {key: SignalContract.model_validate_json(canonical_json(value)) for key, value in node['ports'].items()}
        for signal in signals.values():
            _native_signal(signal)
        _operation_ports(operation, signals)
        nodes[name], ports[name] = node, signals
        if 'physical_subgraph' in node:
            if operation not in ('analog.composite', 'photonic.composite'):
                raise ValueError('physical disclosure requires a registered inclusive composite operation')
            details = validate_graph_structure(node['physical_subgraph'], _depth=_depth+1, _parent=node)
            nested += details['nodes'] + details['nested_nodes']
    incoming, outgoing, adjacency = {}, {}, {name: [] for name in nodes}
    indegree = {name: 0 for name in nodes}
    for edge in graph['edges']:
        keys(edge, {'source', 'target', 'mechanism', 'route'}, {'source', 'target', 'mechanism'}, 'graph.edge')
        if edge['mechanism'] != 'plain_wire':
            raise ValueError('edges cannot perform implicit conversions; use a registered node')
        for key in ('source', 'target'):
            keys(edge[key], {'node', 'port'}, {'node', 'port'}, 'edge.'+key)
            _identifier(edge[key]['node'], 'edge.'+key+'.node')
        src, dst = edge['source'], edge['target']
        if src['node'] not in nodes or dst['node'] not in nodes:
            raise ValueError('edge references an absent node')
        if src['port'] != 'out' or dst['port'] == 'out' or dst['port'] not in ports[dst['node']]:
            raise ValueError('wire direction must be out -> declared input')
        target = (dst['node'], dst['port'])
        if target in incoming:
            raise ValueError('an input port cannot have multiple drivers')
        mismatch = signal_mismatches(ports[src['node']]['out'], ports[dst['node']][dst['port']])
        if mismatch:
            raise ValueError('incompatible plain wire: ' + ', '.join(mismatch))
        incoming[target] = src['node']
        outgoing[src['node']] = outgoing.get(src['node'], 0) + 1
        adjacency[src['node']].append(dst['node'])
        indegree[dst['node']] += 1
    for name, signals in ports.items():
        for port in signals:
            if port != 'out' and (name, port) not in incoming:
                raise ValueError('unconnected required input: ' + name + '.' + port)
        if signals['out'].carrier == 'optical' and outgoing.get(name, 0) > 1:
            raise ValueError('optical fanout requires an explicit paid splitter; plain wires cannot copy a field')
    queue = deque(name for name, count in indegree.items() if count == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise ValueError('implementation graph contains a directed cycle')
    return {'valid': True, 'nodes': len(nodes), 'edges': len(graph['edges']), 'nested_nodes': nested}


def validate_implementation_replay_inputs(record):
    """Validate embedded canonical snapshots before any replay can touch files."""
    canonical_json(record)
    if not isinstance(record, dict) or record.get('schema_version') != 'npp-implementation-result-1':
        raise ValueError('unsupported implementation record')
    snapshots = record.get('normalized_inputs')
    keys(snapshots, {'project', 'implementation'}, {'project', 'implementation'}, 'normalized_inputs')
    project, implementation = snapshots['project'], snapshots['implementation']
    if not isinstance(project, dict) or not isinstance(implementation, dict):
        raise ValueError('replay requires embedded portable project and implementation objects')
    for name in ('model', 'workload', 'technology', 'constraints', 'environment'):
        if not isinstance(project.get(name), dict):
            raise ValueError('replay cannot resolve external project file references')
    if 'exploration' in project and not isinstance(project['exploration'], dict):
        raise ValueError('replay cannot resolve external exploration references')
    if any(name in project['model'] for name in ('weights', 'state_dict_key')):
        raise ValueError('replay cannot load checkpoint files')
    for name in ('W_up', 'W_gate', 'W_down', 'b_up', 'b_gate', 'b_down'):
        if not isinstance(project['model'].get(name), list):
            raise ValueError('replay requires every matrix and explicit bias embedded as values')
    for name in ('geometry', 'system', 'interfaces', 'transform', 'operators'):
        if name not in implementation or (implementation[name] is not None and not isinstance(implementation[name], dict)):
            raise ValueError('replay requires canonical embedded implementation fields')
    raw = {key: value for key, value in project.items() if key != 'input_hash'}
    if project.get('input_hash') != canonical_hash(raw):
        raise ValueError('normalized project input hash mismatch')
    from .io import normalize_project
    normalized = normalize_project(raw)
    if canonical_hash(normalized) != canonical_hash(project):
        raise ValueError('record project is not a canonical normalized snapshot')
    from .graph import normalize_implementation
    if canonical_hash(normalize_implementation(implementation, project['model'])) != canonical_hash(implementation):
        raise ValueError('record implementation is not a canonical normalized snapshot')
    return {'project': normalized, 'implementation': implementation}
