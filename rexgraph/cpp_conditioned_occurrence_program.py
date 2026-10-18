"""Source conditioned native occurrence observations with typed target ports."""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np

from rexgraph import cpp_relational_field_evidence_admission as base
from rexgraph.conditioned_field import (
    LIMITS, MAX_BYTES, POLICY, PATTERN_PACKET, PreparedConditionedInput, check_conditioned_evidence,
)
from rexgraph.cpp_conditioned_field_program import (
    _closed, _grammar, _proof_dependencies, _history_source_dependencies,
)
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_program import _space
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelState, ModelBatch, ModelOutput, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value

ADAPTER = 'cpp_conditioned_occurrence_program'
PROVIDER = {'name': 'rexgraph.cpp.conditioned_occurrence_program', 'version': '1'}
PROFILE = 'rexgraph.conditioned-occurrence-program.v1'
TARGETS = 'rexgraph.conditioned-occurrence-training.v1'
QUERY = 'rexgraph.conditioned-occurrence-query.v1'
SCHEMA = 'rexgraph.conditioned-occurrence-field-family.v1'
METHOD = 'exact-conditioned-occurrence-program-v1'
_lock = RLock()
_models = WeakKeyDictionary()
_batches = WeakKeyDictionary()
_outputs = WeakKeyDictionary()
_provider = None


def _binding():
    module = base._binding()
    if (not hasattr(module, 'NativeOccurrenceTargetPort')
            or not hasattr(module.PreparedAvailableFieldEvidence, 'observation_origins')):
        raise ImportError('conditioned occurrence models require the typed native target provider')
    return module


def _receipts():
    module = _binding()
    return {'literal': module.field_interface_build_info(), 'evidence': module.field_evidence_admission_build_info()}


def _bytes(value):
    if type(value) is not bytes or not 0 < len(value) <= MAX_BYTES:
        raise ValueError('bounded immutable native packet required')
    return value


def validate_conditioned_occurrence_state(state):
    c, p = state.configuration, state.payload
    _closed(c, ('profile', 'execution_provider', 'engine_receipt', 'grammar', 'pattern', 'program_archive',
                'program_sha256', 'root_archive', 'root_sha256', 'limits', 'max_bytes'))
    _closed(p, ('available_archive', 'history', 'observation_count', 'native_dependency_count'))
    if (state.adapter != ADAPTER or state.adapter_version != '1' or state.arithmetic != 'rational'
            or c['profile'] != PROFILE or dict(c['execution_provider']) != PROVIDER or c['pattern'] != PATTERN_PACKET
            or tuple(c['limits']) != LIMITS or c['max_bytes'] != MAX_BYTES or state.output_axes
            or not isinstance(state.source, CppSourceReference)
            or (state.step == 0) != (state.parent is None)):
        raise ValueError('versioned source-conditioned occurrence model required')
    _grammar(c['grammar']); _closed(c['engine_receipt'], ('literal', 'evidence'))
    for packet, digest in (('program_archive', 'program_sha256'), ('root_archive', 'root_sha256')):
        if sha256(_bytes(c[packet])).hexdigest() != c[digest]: raise ValueError('original packet seal differs')
    _bytes(p['available_archive'])
    if (type(p['observation_count']) is not int or p['observation_count'] != len(p['history'])
            or type(p['native_dependency_count']) is not int or p['native_dependency_count'] < 4
            or len(state.dependencies) != p['native_dependency_count'] + 9*len(p['history'])):
        raise ValueError('complete native and literal dependency closure required')
    stages = []
    for item in p['history']:
        _closed(item, ('step', 'input', 'proof_source'))
        _bytes(item['input']); source_from_record(item['proof_source'], allow_cpp=True)
        if type(item['step']) is not int or not 1 <= item['step'] <= state.step:
            raise ValueError('ordered immutable training history required')
        stages.append(item['step'])
    if stages != sorted(stages) or set(stages) != set(range(1, state.step+1)):
        raise ValueError('source-conditioned append stages differ')


def _state(available, reference, configuration, history=(), parent=None, step=0):
    native = available.evidence(); program = native.program()
    dependencies = base._dependencies(_binding(), available, LIMITS)
    return ModelState(ADAPTER, '1', reference, _space(reference, program.identity()['key']), (), configuration,
        {'available_archive': available.archive(), 'history': tuple(history),
         'observation_count': native.identity()['observation_count'], 'native_dependency_count': len(dependencies)},
        'rational', step=step, parent=parent, dependencies=dependencies + _history_source_dependencies(history))


def _role_port(port):
    info = port.inspect(max_records=4096)
    expected = tuple(((1, 6+2*j), 84000, 85000+r) for j in range(12) for r in range(2))
    if info['node'] != 2 or tuple((c['occurrence'], c['basis'], c['direction']) for c in info['coordinates']) != expected:
        raise ValueError('actual complete ordered role24 target coordinates required')
    return info


def _check_sample(item, observation, program_archive, context, grammar):
    row = item.admit(context, program_archive, grammar=grammar)
    if (thaw_tree(observation['inputs']) != (row,) or observation['target_asset'] != row['asset']
            or observation['target_use'] != row['use'] or observation['key'] != item.literal.declaration()['key']-91000):
        raise ValueError('observation differs from complete original literal source proof')
    if 'port' in observation:
        _role_port(observation['port'])
        keys = tuple(observation['observed_coordinates'])
        if not keys or len(keys) != len(set(keys)):
            raise ValueError('distinct nonempty original target coordinate mask required')
    elif tuple(observation['observed']) != tuple((0, i) for i in range(4)):
        raise ValueError('lower observation retains its fixed whole-stalk domain')


def _history(available, inputs, source, context, grammar, program_archive):
    native = available.evidence()
    if native.program().archive() != program_archive or native.identity()['observation_count'] != len(inputs):
        raise ValueError('native program or complete training history differs')
    for i, item in enumerate(inputs):
        row = item.admit(context, program_archive, grammar=grammar)
        observation = native.observation(i, max_records=4096)
        if (observation['key'] != item.literal.declaration()['key']-91000
                or observation['input'] != item.literal.program().execute((item.literal.input(),)).archive()
                or observation['target_asset'] != row['asset'] or observation['target_use'] != row['use']):
            raise ValueError('native observation changes actual source-conditioned execution')
        if 'port' in observation:
            _role_port(observation['port'])
        elif observation['observed'] != tuple((0, k) for k in range(4)):
            raise ValueError('native lower mask changed domain')


def prepare_conditioned_occurrence_model(state, source, *, execution_context):
    state.check_state(); state.bind(source); c = state.configuration
    context = _context(_binding(), execution_context)
    if thaw_tree(c['engine_receipt']) != _receipts(): raise ValueError('occurrence model requires its pinned native implementation')
    with _lock: known = _models.get(state)
    if known is not None:
        if known[0] is not source: raise ValueError('live occurrence model requires actual admitted source owner; restore a cold state explicitly')
        known[1].admit(context)
        for item in known[2]: item.admit(context, c['program_archive'], grammar=c['grammar'])
        return known
    available = _binding().PreparedAvailableFieldEvidence.from_archive(state.payload['available_archive'], context, **POLICY)
    inputs = tuple(PreparedConditionedInput.from_archive(row['input'], context, grammar=c['grammar']) for row in state.payload['history'])
    _history(available, inputs, source, context, c['grammar'], c['program_archive'])
    stages = []; current = available
    while current is not None:
        stages.append(current)
        if len(stages) > state.step+1: raise ValueError('native history contains extra append stages')
        current = current.parent()
    if len(stages) != state.step+1 or stages[-1].archive() != c['root_archive']:
        raise ValueError('native root or parent lineage differs')
    check_conditioned_evidence(stages[-1], source, (), context, grammar=c['grammar'], program_archive=c['program_archive'])
    parent = None
    for step, member in enumerate(reversed(stages)):
        history = tuple(row for row in state.payload['history'] if row['step'] <= step)
        rebuilt = _state(member, state.source, c, history, parent, step)
        parent = rebuilt.coefficient_digest
    if parent != state.coefficient_digest: raise ValueError('complete native/literal dependency or parent seal differs')
    known = (source, available, inputs)
    with _lock: _models[state] = known
    return known


def cpp_conditioned_occurrence_model(source, *, available_evidence, grammar, execution_context, reference=None):
    module = _binding(); context = _context(module, execution_context); grammar = _grammar(grammar)
    if not isinstance(source, module.NativeSource) or not isinstance(available_evidence, module.PreparedAvailableFieldEvidence):
        raise TypeError('actual source and available evidence owners required')
    if available_evidence.admission() != POLICY or available_evidence.evidence().parent() is not None:
        raise ValueError('original root with fixed admission required')
    check_conditioned_evidence(available_evidence, source, (), context, grammar=grammar)
    program = available_evidence.evidence().program().archive(); root = available_evidence.archive()
    c = {'profile': PROFILE, 'execution_provider': PROVIDER, 'engine_receipt': _receipts(), 'grammar': grammar,
         'pattern': PATTERN_PACKET, 'program_archive': program, 'program_sha256': sha256(program).hexdigest(),
         'root_archive': root, 'root_sha256': sha256(root).hexdigest(), 'limits': LIMITS, 'max_bytes': MAX_BYTES}
    reference = CppSourceReference(source) if reference is None else reference.bind(source)
    state = _state(available_evidence, reference, c)
    with _lock: _models[state] = (source, available_evidence, ())
    enable_conditioned_occurrence_provider()
    return state


def _pack_rows(rows):
    return tuple(dict(row, port=row['port'].archive()) if 'port' in row else dict(row) for row in rows)


def _restore_rows(rows, available):
    program = available.evidence().program()
    return tuple(dict(row, port=_binding().NativeOccurrenceTargetPort.from_archive(row['port'], **POLICY).rebind(program))
                 if 'port' in row else thaw_tree(row) for row in rows)


def validate_conditioned_occurrence_targets(value, source, space, mask):
    _closed(value, ('profile', 'model_digest', 'observations', 'inputs', 'origin_count'))
    if (value['profile'] != TARGETS or type(value['model_digest']) is not str or len(value['model_digest']) != 64
            or type(value['origin_count']) is not int or value['origin_count'] < 1 or np.any(mask)
            or not value['observations'] or len(value['observations']) != len(value['inputs'])):
        raise ValueError('typed target mask is separate from source-cell batch coordinates')
    keys = set()
    for row, item in zip(value['observations'], value['inputs'], strict=True):
        _bytes(item)
        common = ('key', 'inputs', 'target_asset', 'target_use', 'target', 'origins')
        _closed(row, common + (('port', 'observed_coordinates') if 'port' in row else ('observed',)))
        _bytes(row['target_asset']); _bytes(row['target'])
        if type(row['key']) is not int or row['key'] <= 0 or row['key'] in keys: raise ValueError('distinct original observation identities required')
        keys.add(row['key'])
        if 'port' in row:
            _bytes(row['port']); _closed(row['origins'], ('asset', 'value', 'port'))
            if not row['observed_coordinates'] or any(type(k) is not int or k <= 0 for k in row['observed_coordinates']):
                raise ValueError('nonempty original target coordinate mask required')
        else: _closed(row['origins'], ('asset', 'value'))
    return value


def conditioned_occurrence_batch(state, source, samples, *, execution_context):
    _, available, _ = prepare_conditioned_occurrence_model(state, source, execution_context=execution_context)
    samples = tuple(samples); inputs = tuple(p for p, _ in samples); rows = tuple(dict(r) for _, r in samples)
    for item, row in zip(inputs, rows, strict=True): _check_sample(item, row, state.configuration['program_archive'], execution_context, state.configuration['grammar'])
    info = available.typed_observation_dependencies(rows, execution_context, max_records=LIMITS[1])
    target = {'profile': TARGETS, 'model_digest': state.coefficient_digest, 'observations': _pack_rows(rows),
              'inputs': tuple(item.archive() for item in inputs), 'origin_count': info['origin_count']}
    dependencies = base._references(_binding(), info['dependencies'], LIMITS) + _proof_dependencies(inputs)
    batch = ModelBatch(state.source, state.space, target, np.zeros(len(state.space.keys), dtype=bool), dependencies=dependencies)
    with _lock: _batches[batch] = (source, inputs, rows)
    return batch


def _train(state, source, batch, *, steps=1, execution_context):
    _, available, old_inputs = prepare_conditioned_occurrence_model(state, source, execution_context=execution_context)
    if steps != 1 or batch.targets['model_digest'] != state.coefficient_digest or batch.inputs is not None:
        raise ValueError('one immutable append on original model source required')
    with _lock: known = _batches.get(batch)
    if known is None:
        inputs = tuple(PreparedConditionedInput.from_archive(p, execution_context, grammar=state.configuration['grammar']) for p in batch.targets['inputs'])
        rows = _restore_rows(batch.targets['observations'], available)
    else:
        if known[0] is not source: raise ValueError('live batch requires original actual source owner')
        _, inputs, rows = known
    if _pack_rows(rows) != thaw_tree(batch.targets['observations']): raise ValueError('live observation differs from its immutable target declaration')
    for item, row in zip(inputs, rows, strict=True): _check_sample(item, row, state.configuration['program_archive'], execution_context, state.configuration['grammar'])
    info = available.typed_observation_dependencies(rows, execution_context, max_records=LIMITS[1])
    deps = base._references(_binding(), info['dependencies'], LIMITS) + _proof_dependencies(inputs)
    if (info['origin_count'] != batch.targets['origin_count'] or len(deps) != len(batch.dependencies)
            or any(not a.matches(b) for a, b in zip(deps, batch.dependencies, strict=True))):
        raise ValueError('complete original target/input/port dependency declaration differs')
    child = available.append_inputs(rows, execution_context)
    if child.parent().archive() != available.archive(): raise ValueError('native append changed parent evidence')
    first = available.evidence().identity()['observation_count']
    native_origins = tuple(origin for i in range(first, first+len(rows)) for origin in child.observation_origins(i, execution_context, max_records=LIMITS[1]))
    if tuple(o.receipt() for o in native_origins) != tuple(o.receipt() for o in info['origins']):
        raise ValueError('actual native typed dependency order differs')
    history = state.payload['history'] + tuple({'step': state.step+1, 'input': p.archive(),
        'proof_source': CppSourceReference(p.source.parent.model_source).as_record()} for p in inputs)
    result = _state(child, state.source, state.configuration, history, state.coefficient_digest, state.step+1)
    with _lock: _models[result] = (source, child, old_inputs+inputs)
    return result


@dataclass(frozen=True)
class ConditionedOccurrenceQuery(Mapping):
    declaration: object
    prepared: object
    def __getitem__(self, key): return self.declaration[key]
    def __iter__(self): return iter(self.declaration)
    def __len__(self): return len(self.declaration)


def conditioned_occurrence_query(state, prepared):
    state.check_state()
    if not isinstance(prepared, PreparedConditionedInput): raise TypeError('actual prepared literal source required')
    return ConditionedOccurrenceQuery(freeze_tree({'profile': QUERY, 'model_digest': state.coefficient_digest,
        'source': state.source.as_record(), 'input': prepared.archive()}), prepared)


def _infer(state, source, inputs, *, execution_context):
    _, available, history = prepare_conditioned_occurrence_model(state, source, execution_context=execution_context)
    _closed(inputs, ('profile', 'model_digest', 'source', 'input'))
    if (inputs['profile'] != QUERY or inputs['model_digest'] != state.coefficient_digest
            or not source_from_record(inputs['source'], allow_cpp=True).matches(state.source)):
        raise ValueError('source-conditioned query differs from model')
    item = inputs.prepared if isinstance(inputs, ConditionedOccurrenceQuery) else PreparedConditionedInput.from_archive(inputs['input'], execution_context, grammar=state.configuration['grammar'])
    if item.archive() != inputs['input']: raise ValueError('live input changed its original proof packet')
    row = item.admit(execution_context, state.configuration['program_archive'], grammar=state.configuration['grammar'])
    family = available.execute_inputs((row,), execution_context)
    packet = pack_value({'profile': PROFILE, 'source': source.archive, 'model': pack_model(state),
                         'input': item.archive(), 'native': family.archive()})
    _bytes(packet)
    deps = base._dependencies(_binding(), family, LIMITS) + _proof_dependencies(history+(item,))
    archive = {'schema': SCHEMA, 'data': packet, 'sha256': sha256(packet).hexdigest(),
               'program_sha256': state.configuration['program_sha256'], 'engine_receipt': state.configuration['engine_receipt']}
    output = ModelOutput([], CoordinateSpace('native-conditioned-occurrence-readout', ()), (), state.source,
        state.coefficient_digest, 'rational', METHOD, dependencies=deps, field_archive=archive,
        execution_receipt={'profile': METHOD, 'limits': LIMITS, 'max_bytes': MAX_BYTES, 'dependency_count': len(deps)})
    with _lock: _outputs[output] = (state, source, item, family)
    return output


def read_conditioned_occurrence_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if not isinstance(output, ModelOutput) or output.method != METHOD: raise TypeError('native conditioned occurrence output required')
    output.check_state(); a = output.field_archive; r = output.execution_receipt
    _closed(r, ('profile', 'limits', 'max_bytes', 'dependency_count'))
    if (r['profile'] != METHOD or tuple(r['limits']) != LIMITS or r['max_bytes'] != MAX_BYTES
            or r['dependency_count'] != len(output.dependencies) or a['schema'] != SCHEMA
            or thaw_tree(a['engine_receipt']) != _receipts() or output.values.size or output.axes
            or output.space != CoordinateSpace('native-conditioned-occurrence-readout', ())):
        raise ValueError('output receipt, field domain or pinned implementation differs')
    if limits is not None and (len(limits) != 4 or any(type(x) is not int or x < y for x, y in zip(limits, LIMITS, strict=True))):
        raise ValueError('requested admission excludes fixed profile')
    if max_bytes is not None and (type(max_bytes) is not int or len(a['data']) > max_bytes): raise ValueError('output byte admission exceeded')
    with _lock: known = _outputs.get(output)
    if known is None:
        row = unpack_value(a['data']); _closed(row, ('profile', 'source', 'model', 'input', 'native'))
        if row['profile'] != PROFILE: raise ValueError('output profile differs')
        model = unpack_model(row['model']); source = _binding().NativeSource(row['source'], max_bytes=MAX_BYTES)
        prepare_conditioned_occurrence_model(model, source, execution_context=execution_context)
        item = PreparedConditionedInput.from_archive(row['input'], execution_context, grammar=model.configuration['grammar'])
        family = _binding().NativeAvailableAffineResult.from_archive(row['native'], execution_context, **POLICY)
        known = (model, source, item, family)
        with _lock: _outputs[output] = known
    model, source, item, family = known
    _, available, history = prepare_conditioned_occurrence_model(model, source, execution_context=execution_context)
    item.admit(execution_context, model.configuration['program_archive'], grammar=model.configuration['grammar']); family.admit(execution_context)
    if (output.model_digest != model.coefficient_digest or not output.source.matches(model.source)
            or (state is not None and state.coefficient_digest != model.coefficient_digest)
            or family.evidence().archive() != available.archive()
            or family.input().result().archive() != item.literal.program().execute((item.literal.input(),)).archive()
            or a['program_sha256'] != model.configuration['program_sha256']):
        raise ValueError('complete model/input/target family differs')
    deps = base._dependencies(_binding(), family, LIMITS) + _proof_dependencies(history+(item,))
    if len(deps) != len(output.dependencies) or any(not x.matches(y) for x, y in zip(deps, output.dependencies, strict=True)):
        raise ValueError('output full dependency closure differs')
    return family


def _create(source, *, adapter=ADAPTER, **configuration):
    if adapter != ADAPTER: raise ValueError('occurrence adapter required')
    return cpp_conditioned_occurrence_model(source, **configuration)


def enable_conditioned_occurrence_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(PROVIDER['name'], '1', ((ADAPTER, '1'),), _infer, create=_create,
                prepare=prepare_conditioned_occurrence_model, train=_train, target_profiles=(TARGETS,),
                context_operations=('create', 'prepare', 'infer', 'train'))
            register_model_provider(_provider)
    return _provider
