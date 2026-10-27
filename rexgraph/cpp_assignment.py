"""Exact native assignment learning over complete measured fields and bindings."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

import numpy as np

from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_evidence_admission import _references
from rexgraph.cpp_relational_field_program import _admission
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_ADAPTER = 'cpp_assignment_evidence'
_PROVIDER = {'name': 'rexgraph.cpp.assignment_evidence', 'version': '1'}
_PROFILE = 'rexgraph.available-assignment-learning.v1'
_QUERY = 'rexgraph.available-assignment-query.v1'
_TARGETS = 'rexgraph.available-assignment-observations.v1'
_SCHEMA = 'rexgraph.conditioned-assignment-response.v1'
_METHOD = 'exact-conditioned-assignment-response-v1'
_LIMITS = (4096, 16000000, 64000000, 1 << 24)
_READOUT = CoordinateSpace('native-conditioned-assignment-readout', ())
_lock = RLock()
_cache = OrderedDict()
_checkpoints = OrderedDict()
_outputs = OrderedDict()
_provider = None


def _binding():
    from rexgraph.cpp_model import _binding as binding
    module = binding()
    if not hasattr(module, 'NativeAssignmentEvidence'):
        raise ImportError('assignment learning requires its owning native evidence provider')
    return module


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _space(reference, configuration):
    return CoordinateSpace(f"cpp:{reference.native.namespace}:assignment:{configuration['parameter_carrier']}",
        tuple(str(k) for k in configuration['parameters']))


def validate_assignment_state(state):
    c, p = state.configuration, state.payload
    if (state.adapter != _ADAPTER or state.adapter_version != '1' or state.arithmetic != 'rational'
            or not isinstance(state.source, CppSourceReference) or state.output_axes
            or set(c) != {'profile', 'execution_provider', 'engine_receipt', 'limits', 'max_bytes', 'key',
                          'root_revision', 'parameter_carrier', 'parameters', 'root_sha256', 'finite_sha256'}
            or c['profile'] != _PROFILE or dict(c['execution_provider']) != _PROVIDER
            or set(p) != {'archive', 'sha256', 'revision', 'observation_count', 'status'}
            or state.space != _space(state.source, c)):
        raise ValueError('immutable native assignment model with its complete common parameter carrier required')
    _admission(c['limits'], c['max_bytes'])
    if (not isinstance(c['engine_receipt'], Mapping) or not c['engine_receipt']
            or any(type(c[name]) is not int or not 0 < c[name] < 1 << 64 for name in ('key', 'root_revision', 'parameter_carrier'))
            or any(type(k) is not int or not 0 < k < 1 << 64 for k in c['parameters'])
            or len(set(c['parameters'])) != len(c['parameters']) or len(c['parameters']) > c['limits'][0]
            or not _sha(c['root_sha256']) or not _sha(c['finite_sha256'])
            or type(p['archive']) is not bytes or not p['archive'] or len(p['archive']) > c['max_bytes']
            or sha256(p['archive']).hexdigest() != p['sha256']
            or type(p['revision']) is not int or p['revision'] != c['root_revision'] + state.step
            or type(p['observation_count']) is not int or p['observation_count'] < 0
            or p['status'] not in {'inconsistent', 'unresolved', 'identified'}
            or (state.step == 0 and (state.parent is not None or p['sha256'] != c['root_sha256'] or p['observation_count']))
            or (state.step > 0 and not _sha(state.parent))):
        raise ValueError('assignment source, parameter identities, immutable revision or native archive differs')


def _dependencies(module, native, authority, limits):
    return _references(module, native.dependency_archives(authority, max_records=limits[1]), limits)


def _check_native(module, native, state, authority):
    native.admit(authority)
    c = state.configuration; info = native.inspect(authority, max_records=c['limits'][1])
    finite = native.finite_source(authority)
    original = module.NativeSource(finite.asset().source_archive(), max_dimension=max(c['limits'][:2]), max_records=c['limits'][1])
    state.source.bind(original)
    if (info['key'] != c['key'] or info['root_revision'] != c['root_revision']
            or info['parameter_carrier'] != c['parameter_carrier'] or tuple(info['parameters']) != tuple(c['parameters'])
            or info['revision'] != state.payload['revision'] or info['observation_count'] != state.payload['observation_count']
            or info['status'] != state.payload['status']
            or sha256(native.archive(authority)).hexdigest() != state.payload['sha256']
            or sha256(finite.archive()).hexdigest() != c['finite_sha256']):
        raise ValueError('assignment complete native state or original finite source differs')
    deps = _dependencies(module, native, authority, c['limits'])
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a, b in zip(deps, state.dependencies, strict=True)):
        raise ValueError('assignment original publisher and source histories differ')


def _state(native, reference, configuration, authority, *, parent=None, step=0):
    image = native.archive(authority); info = native.inspect(authority, max_records=configuration['limits'][1])
    payload = dict(archive=image, sha256=sha256(image).hexdigest(), revision=info['revision'],
        observation_count=info['observation_count'], status=info['status'])
    state = ModelState(_ADAPTER, '1', reference, _space(reference, configuration), (), configuration, payload,
        'rational', step=step, parent=parent,
        dependencies=_dependencies(_binding(), native, authority, configuration['limits']))
    with _lock:
        _cache[id(state)] = (state, native)
        while len(_cache) > 8: _cache.popitem(last=False)
    return state



def _verify_lineage(native, state, authority):
    c = state.configuration
    ancestors = []; current = native
    while current is not None:
        ancestors.append(current)
        if len(ancestors) > state.step + 1: raise ValueError('assignment archive has extra immutable ancestors')
        current = current.parent(authority)
    if len(ancestors) != state.step + 1 or sha256(ancestors[-1].archive(authority)).hexdigest() != c['root_sha256']:
        raise ValueError('assignment original root or parent lineage differs')
    reconstructed = None
    for step, value in enumerate(reversed(ancestors)):
        reconstructed = _state(value, state.source, c, authority,
            parent=None if reconstructed is None else reconstructed.coefficient_digest, step=step)
    if reconstructed.coefficient_digest != state.coefficient_digest:
        raise ValueError('assignment complete checkpoint lineage differs')


def cpp_assignment_model(source, *, evidence, execution_context, reference=None):
    module = _binding(); authority = _context(module, execution_context)
    if not isinstance(source, module.NativeSource) or not isinstance(evidence, module.NativeAssignmentEvidence):
        raise TypeError('actual native source and assignment evidence root required')
    evidence.admit(authority)
    if evidence.parent(authority) is not None: raise ValueError('capture the original root and retain each immutable child update')
    policy = evidence.admission(); info = evidence.inspect(authority)
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    configuration = dict(profile=_PROFILE, execution_provider=_PROVIDER, engine_receipt=module.assignment_build_info(),
        limits=policy['limits'], max_bytes=policy['max_bytes'], key=info['key'], root_revision=info['root_revision'],
        parameter_carrier=info['parameter_carrier'], parameters=info['parameters'],
        root_sha256=sha256(evidence.archive(authority)).hexdigest(), finite_sha256=sha256(evidence.finite_source(authority).archive()).hexdigest())
    state = _state(evidence, ref, configuration, authority)
    _check_native(module, evidence, state, authority); enable_assignment_provider()
    return state


def prepare_assignment_model(state, source, *, execution_context):
    validate_assignment_state(state); state.bind(source)
    module = _binding(); authority = _context(module, execution_context); c = state.configuration
    if dict(c['engine_receipt']) != module.assignment_build_info(): raise ValueError('assignment model requires its pinned native implementation')
    with _lock:
        found = _cache.get(id(state)); native = found[1] if found is not None and found[0] is state else None
    if native is None:
        native = module.NativeAssignmentEvidence.from_archive(state.payload['archive'], authority,
            limits=c['limits'], max_bytes=c['max_bytes'])
    if found is None or found[0] is not state:
        _check_native(module, native, state, authority)
        _verify_lineage(native, state, authority)
    else:
        native.admit(authority)
    with _lock:
        _cache[id(state)] = (state, native); _cache.move_to_end(id(state))
        while len(_cache) > 8: _cache.popitem(last=False)
    return native


def clear_assignment_cache():
    with _lock: _cache.clear(); _checkpoints.clear(); _outputs.clear()


def _rows(value, maximum):
    if not isinstance(value, (tuple, list)) or not value or len(value) > maximum:
        raise ValueError('nonempty bounded assignment observation declarations required')
    rows = []
    for row in value:
        if (not isinstance(row, Mapping) or set(row) != {'key', 'origin', 'binding', 'measured', 'inputs'}
                or any(type(row[k]) is not int or not 0 < row[k] < 1 << 64 for k in ('key', 'origin'))
                or type(row['measured']) is not bytes or not row['measured'] or not isinstance(row['binding'], Mapping)
                or not isinstance(row['inputs'], (tuple, list)) or not row['inputs'] or len(row['inputs']) > maximum):
            raise ValueError('whole returned measurement, binding and original input archives required')
        for item in row['inputs']:
            if (not isinstance(item, Mapping) or set(item) != {'candidate', 'archive'}
                    or type(item['candidate']) is not int or not 0 < item['candidate'] < 1 << 64
                    or type(item['archive']) is not bytes or not item['archive']):
                raise ValueError('complete candidate input archive required')
        rows.append(thaw_tree(row))
    return tuple(rows)


def validate_assignment_targets(value, source, space, mask):
    if (not isinstance(value, Mapping) or set(value) != {'profile', 'model_digest', 'observations', 'origin_count'}
            or value['profile'] != _TARGETS or not isinstance(source, CppSourceReference) or not _sha(value['model_digest'])
            or mask.any() or type(value['origin_count']) is not int or value['origin_count'] < 1):
        raise ValueError('whole assignment measurements use encoded observations, not direct scalar parameter labels')
    _rows(value['observations'], 4096)
    return value


def _restore_rows(native, observations, authority, limits, maximum):
    module = _binding(); info = native.inspect(authority, max_records=limits[1])
    indices = {native.candidate(i, authority, max_records=limits[1])['key']: i for i in range(info['candidate_count'])}
    out = []
    for row in _rows(observations, limits[0]):
        measured = module.NativeAvailableFieldResult.from_archive(row['measured'], authority, limits=limits, max_bytes=maximum)
        inputs = []
        for item in row['inputs']:
            if item['candidate'] not in indices: raise ValueError('observation names an undeclared candidate')
            actual = native.restore_input(indices[item['candidate']], item['archive'], authority)
            inputs.append(dict(candidate=item['candidate'], input=actual))
        out.append(dict(key=row['key'], origin=row['origin'], binding=row['binding'], measured=measured, inputs=tuple(inputs)))
    return tuple(out)


def assignment_batch(state, source, observations, *, execution_context):
    native = prepare_assignment_model(state, source, execution_context=execution_context); c = state.configuration
    rows = _rows(observations, c['limits'][0]); actual = _restore_rows(native, rows, execution_context, c['limits'], c['max_bytes'])
    info = native.observation_dependencies(actual, execution_context, max_records=c['limits'][1])
    targets = dict(profile=_TARGETS, model_digest=state.coefficient_digest, observations=rows, origin_count=info['origin_count'])
    return ModelBatch(state.source, state.space, targets, np.zeros(len(state.space.keys), dtype=bool),
        dependencies=_references(_binding(), info['dependencies'], c['limits']))


def _train(state, source, batch, *, steps=1, execution_context):
    if steps != 1: raise ValueError('assignment observations append in one immutable native update')
    native = prepare_assignment_model(state, source, execution_context=execution_context); c = state.configuration; target = batch.targets
    if (target['profile'] != _TARGETS or target['model_digest'] != state.coefficient_digest
            or batch.inputs is not None or batch.target_input is not None):
        raise ValueError('assignment observation batch belongs to another model')
    rows = _restore_rows(native, target['observations'], execution_context, c['limits'], c['max_bytes'])
    info = native.observation_dependencies(rows, execution_context, max_records=c['limits'][1])
    deps = _references(_binding(), info['dependencies'], c['limits'])
    if info['origin_count'] != target['origin_count'] or len(deps) != len(batch.dependencies) or any(
            not a.matches(b) for a, b in zip(deps, batch.dependencies, strict=True)):
        raise ValueError('assignment batch original publishers or input/source ordering differs')
    child = native.append(rows, execution_context)
    if not child.parent(execution_context).same_owner(native): raise ValueError('assignment append lost its actual immutable parent')
    return _state(child, state.source, c, execution_context, parent=state.coefficient_digest, step=state.step + 1)


def assignment_query(state, input_archive, *, candidate=None):
    validate_assignment_state(state)
    if type(input_archive) is not bytes or not input_archive or len(input_archive) > state.configuration['max_bytes']:
        raise ValueError('explicit complete admitted input archive required')
    if candidate is not None and (type(candidate) is not int or candidate < 0): raise ValueError('candidate index must be a nonnegative integer')
    return freeze_tree(dict(profile=_QUERY, model_digest=state.coefficient_digest, source=state.source.as_record(),
        candidate=candidate, input_archive=input_archive))


def _infer(state, source, inputs, *, execution_context):
    native = prepare_assignment_model(state, source, execution_context=execution_context); c = state.configuration
    if (not isinstance(inputs, Mapping) or set(inputs) != {'profile', 'model_digest', 'source', 'candidate', 'input_archive'}
            or inputs['profile'] != _QUERY or inputs['model_digest'] != state.coefficient_digest
            or not source_from_record(inputs['source'], allow_cpp=True).matches(state.source)):
        raise ValueError('closed assignment query with original model/source identity required')
    assignment_query(state, inputs['input_archive'], candidate=inputs['candidate'])
    info = native.inspect(execution_context, max_records=c['limits'][1]); selected = inputs['candidate']
    if selected is None:
        selected = info['identified_candidate']
        if selected is None: raise ValueError('unresolved assignments need an explicitly conditional candidate query')
    value = native.execute(selected, native.restore_input(selected, inputs['input_archive'], execution_context), execution_context)
    image = value.archive(execution_context); actual = native.candidate(selected, execution_context, max_records=c['limits'][1])
    receipt = dict(profile=_METHOD, limits=c['limits'], max_bytes=c['max_bytes'], revision=info['revision'],
        key=info['key'], candidate=selected, candidate_key=actual['key'], binding=actual['binding'],
        interpretation='conditioned-correlated-native-family', executed_external_worker=False)
    archive = dict(schema=_SCHEMA, data=image, sha256=sha256(image).hexdigest(),
        model_sha256=state.payload['sha256'], engine_receipt=c['engine_receipt'])
    output = ModelOutput([], _READOUT, (), state.source, state.coefficient_digest, 'rational', _METHOD,
        dependencies=_dependencies(_binding(), value, execution_context, c['limits']), execution_receipt=receipt, field_archive=archive)
    with _lock:
        _outputs[id(output)] = (output, value)
        while len(_outputs) > 8: _outputs.popitem(last=False)
    return output


def read_assignment_result(output, *, execution_context, state=None):
    if (not isinstance(output, ModelOutput) or output.method != _METHOD or output.space != _READOUT or output.axes
            or output.values.size or output.grade is not None or output.variance != 'coordinate' or output.field_archive is None):
        raise TypeError('whole conditioned assignment response required')
    output.check_state(); module = _binding(); authority = _context(module, execution_context); a = output.field_archive; r = output.execution_receipt
    required = {'profile', 'limits', 'max_bytes', 'revision', 'key', 'candidate', 'candidate_key', 'binding', 'interpretation', 'executed_external_worker'}
    if (set(r) != required or r['profile'] != _METHOD or a['schema'] != _SCHEMA
            or r['interpretation'] != 'conditioned-correlated-native-family' or r['executed_external_worker'] is not False
            or dict(a['engine_receipt']) != module.assignment_build_info()):
        raise ValueError('conditioned response requires its complete receipt and pinned native implementation')
    limits = _admission(r['limits'], r['max_bytes'])
    with _lock: cached = _outputs.get(id(output))
    if cached is not None and cached[0] is output:
        value = cached[1]; value.admit(authority)
        if state is not None:
            validate_assignment_state(state)
            if (state.coefficient_digest != output.model_digest or state.payload['sha256'] != a['model_sha256']
                    or not state.source.matches(output.source)):
                raise ValueError('conditioned response belongs to another model checkpoint')
        return value
    value = module.NativeAssignmentResponse.from_archive(a['data'], authority, limits=limits, max_bytes=r['max_bytes'])
    native = value.model(authority); info = native.inspect(authority, max_records=limits[1]); selected = value.inspect(authority)['candidate']
    actual = native.candidate(selected, authority, max_records=limits[1])
    original = module.NativeSource(native.finite_source(authority).asset().source_archive(), max_dimension=max(limits[:2]), max_records=limits[1])
    output.source.bind(original)
    if (sha256(native.archive(authority)).hexdigest() != a['model_sha256'] or info['key'] != r['key']
            or info['revision'] != r['revision'] or selected != r['candidate'] or actual['key'] != r['candidate_key']
            or freeze_tree(actual['binding']) != r['binding']):
        raise ValueError('conditioned response model, selected binding or complete native ownership differs')
    deps = _dependencies(module, value, authority, limits)
    if len(deps) != len(output.dependencies) or any(not a.matches(b) for a, b in zip(deps, output.dependencies, strict=True)):
        raise ValueError('conditioned response original source histories differ')
    ancestors = []; current = native
    count = info['revision'] - info['root_revision'] + 1
    while current is not None:
        ancestors.append(current)
        if len(ancestors) > count: raise ValueError('conditioned response has extra immutable ancestors')
        current = current.parent(authority)
    if len(ancestors) != count: raise ValueError('conditioned response has missing immutable ancestors')
    reconstructed = cpp_assignment_model(original, evidence=ancestors[-1], execution_context=authority, reference=output.source)
    for step, child in enumerate(reversed(ancestors[:-1]), 1):
        reconstructed = _state(child, output.source, reconstructed.configuration, authority,
            parent=reconstructed.coefficient_digest, step=step)
    if reconstructed.coefficient_digest != output.model_digest:
        raise ValueError('conditioned response Core model identity differs from its complete native lineage')
    if state is not None:
        validate_assignment_state(state)
        if (state.coefficient_digest != output.model_digest or state.payload['sha256'] != a['model_sha256']
                or not state.source.matches(output.source)):
            raise ValueError('conditioned response belongs to another model checkpoint')
    with _lock:
        _outputs[id(output)] = (output, value)
        while len(_outputs) > 8: _outputs.popitem(last=False)
    return value


def assignment_checkpoint(state, source, *, execution_context):
    """Capture an identified structural worker without selecting free parameters."""
    from rexgraph.model_codec import pack_model
    from rexgraph.value_codec import pack_value
    native = prepare_assignment_model(state, source, execution_context=execution_context)
    info = native.inspect(execution_context, max_records=state.configuration['limits'][1])
    if info['identified_candidate'] is None or info['remainder_count']:
        raise ValueError('a learned worker requires one identified native candidate and no uncovered alternative')
    return pack_value(pack_model(state, native=True))


def assignment_checkpoint_state(checkpoint):
    from rexgraph.model_codec import unpack_model
    from rexgraph.value_codec import unpack_value
    if type(checkpoint) is not bytes or not checkpoint or len(checkpoint) > 256*1024*1024:
        raise ValueError('immutable assignment checkpoint bytes required')
    with _lock:
        found = _checkpoints.get(id(checkpoint))
        if found is not None and found[0] is checkpoint: return found[1]
    state = unpack_model(unpack_value(checkpoint)); validate_assignment_state(state)
    with _lock:
        _checkpoints[id(checkpoint)] = (checkpoint, state)
        while len(_checkpoints) > 8: _checkpoints.popitem(last=False)
    return state


def bind_assignment_owner(state, source, owner, *, execution_context):
    """Bind a declared runtime owner after checking the full stored model and source."""
    validate_assignment_state(state); state.bind(source)
    module = _binding(); authority = _context(module, execution_context)
    if not isinstance(owner, module.NativeAssignmentEvidence): raise TypeError('actual native assignment owner required')
    if dict(state.configuration['engine_receipt']) != module.assignment_build_info():
        raise ValueError('assignment runtime requires its pinned native implementation')
    with _lock: cached = _cache.get(id(state))
    if cached is not None and cached[0] is state and cached[1].same_owner(owner):
        owner.admit(authority)
    else:
        _check_native(module, owner, state, authority)
        _verify_lineage(owner, state, authority)
    with _lock:
        _cache[id(state)] = (state, owner)
        while len(_cache) > 8: _cache.popitem(last=False)
    return owner


def infer_assignment_checkpoint(checkpoint, source, input_archive, *, execution_context, owner=None):
    from rexgraph.model_runtime import infer_model
    state = assignment_checkpoint_state(checkpoint); enable_assignment_provider()
    native = (prepare_assignment_model(state, source, execution_context=execution_context) if owner is None else
              bind_assignment_owner(state, source, owner, execution_context=execution_context))
    info = native.inspect(execution_context, max_records=state.configuration['limits'][1])
    if info['identified_candidate'] is None or info['remainder_count']:
        raise ValueError('unresolved alternatives do not define an automatic worker')
    return infer_model(state, source, assignment_query(state, input_archive, candidate=info['identified_candidate']),
        execution_context=execution_context)


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError('assignment evidence adapter required')
    return cpp_assignment_model(source, **configuration)


def enable_assignment_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER['name'], '1', ((_ADAPTER, '1'),), _infer, create=_create,
                prepare=prepare_assignment_model, train=_train, target_profiles=(_TARGETS,),
                context_operations=('create', 'prepare', 'infer', 'train'))
        register_model_provider(_provider)
    return _provider
