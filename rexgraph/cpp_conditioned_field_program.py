"""Exact source conditioned field models with mandatory literal proof lineage."""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np

from rexgraph import cpp_relational_field_evidence_admission as base
from rexgraph.conditioned_field import (
    LIMITS, MAX_BYTES, POLICY, PATTERN_PACKET, PreparedConditionedInput,
    check_conditioned_model,
)
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelState, ModelBatch, ModelOutput, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value

PROFILE = 'rexgraph.conditioned-field-program.v1'
QUERY = 'rexgraph.conditioned-field-query.v1'
TARGETS = 'rexgraph.conditioned-field-training.v1'
SCHEMA = 'rexgraph.conditioned-field-family.v1'
METHOD = 'exact-conditioned-field-program-v1'
ADAPTER = 'cpp_conditioned_field_program'
PROVIDER = {'name': 'rexgraph.cpp.conditioned_field_program', 'version': '1'}
_lock = RLock()
_models = WeakKeyDictionary()
_batches = WeakKeyDictionary()
_outputs = WeakKeyDictionary()
_provider = None


def _binding():
    module = base._binding()
    if not hasattr(module, 'NativeLiteralInput') or not hasattr(module.NativeAvailabilityContext, 'recheck'):
        raise ImportError('conditioned field models require the native literal and availability provider')
    return module


def _closed(value, keys):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError('closed conditioned field contract required')


def _grammar(value):
    rows = tuple(thaw_tree(value))
    if len(rows) != 2:
        raise ValueError('one declared prefix and suffix predicate required')
    for index, row in enumerate(rows):
        _closed(row, ('key', 'kind', 'argument'))
        if (type(row['key']) is not int or not 0 < row['key'] < 1 << 64
                or row['kind'] != ('byte-prefix', 'byte-suffix')[index]
                or type(row['argument']) is not bytes or len(row['argument']) > MAX_BYTES):
            raise ValueError('closed exact predicate grammar required')
    if rows[0]['key'] == rows[1]['key']:
        raise ValueError('distinct original predicate identities required')
    return rows


def _receipts():
    module = _binding()
    return {'literal': module.field_interface_build_info(), 'evidence': module.field_evidence_admission_build_info()}


def _proof_dependencies(values):
    return tuple(CppSourceReference(value.source.parent.model_source) for value in values for _ in value.proofs)


def _state(base_state, history, configuration, parent=None):
    return ModelState(ADAPTER, '1', base_state.source, base_state.space, (), configuration,
        {'base_model': pack_model(base_state), 'history': tuple(history)}, 'rational', step=base_state.step,
        parent=parent, dependencies=base_state.dependencies+_history_source_dependencies(history))


def _history_source_dependencies(history):
    result = []
    for item in history:
        ref = source_from_record(item['proof_source'], allow_cpp=True)
        result.extend(ref for _ in range(9))
    return tuple(result)


def validate_conditioned_field_state(state):
    c, p = state.configuration, state.payload
    _closed(c, ('profile', 'execution_provider', 'engine_receipt', 'grammar', 'pattern', 'base_configuration', 'limits', 'max_bytes'))
    _closed(p, ('base_model', 'history'))
    if (state.adapter != ADAPTER or state.adapter_version != '1' or state.arithmetic != 'rational'
            or c['profile'] != PROFILE or dict(c['execution_provider']) != PROVIDER
            or c['pattern'] != PATTERN_PACKET or tuple(c['limits']) != LIMITS or c['max_bytes'] != MAX_BYTES
            or not isinstance(state.source, CppSourceReference) or state.output_axes):
        raise ValueError('versioned conditioned field model required')
    _grammar(c['grammar']); _closed(c['engine_receipt'], ('literal', 'evidence'))
    original = unpack_model(thaw_tree(p['base_model']))
    base.validate_available_field_learning_state(original)
    if (not original.source.matches(state.source) or original.space != state.space or original.output_axes
            or original.step != state.step or original.configuration != c['base_configuration']
            or (state.parent is None) != (state.step == 0)
            or len(p['history']) != original.payload['observation_count']):
        raise ValueError('conditioned state changes original source, coordinates or evidence count')
    stages = []
    for item in p['history']:
        _closed(item, ('step', 'input', 'proof_source'))
        source_from_record(item['proof_source'], allow_cpp=True)
        if type(item['step']) is not int or not 1 <= item['step'] <= state.step or type(item['input']) is not bytes or not 0 < len(item['input']) <= MAX_BYTES:
            raise ValueError('complete ordered conditioned input history required')
        stages.append(item['step'])
    if stages != sorted(stages) or set(stages) != set(range(1, state.step+1)):
        raise ValueError('conditioned evidence append lineage differs')
    if len(state.dependencies) != len(original.dependencies)+9*len(stages):
        raise ValueError('all original literal proof dependencies required')
    return original


def _admit_cached(value, context, grammar):
    source, original, available, inputs = value
    available.admit(context)
    for item in inputs:
        item.admit(context, original.configuration['program_archive'], grammar=grammar)
    return value


def prepare_conditioned_field_model(state, source, *, execution_context):
    state.check_state(); state.bind(source)
    context = _context(_binding(), execution_context)
    if thaw_tree(state.configuration['engine_receipt']) != _receipts():
        raise ValueError('conditioned model requires its pinned native implementation')
    with _lock:
        known = _models.get(state)
    if known is not None:
        if known[0] is not source:
            raise ValueError('live conditioned state requires its actual admitted source owner; restore a cold state explicitly')
        return _admit_cached(known, context, state.configuration['grammar'])
    original = validate_conditioned_field_state(state)
    available = base.inspect_available_field_learning_model(original, source, execution_context=context)
    inputs = tuple(PreparedConditionedInput.from_archive(item['input'], context, grammar=state.configuration['grammar']) for item in state.payload['history'])
    check_conditioned_model(original, source, inputs, context, available=available, grammar=state.configuration['grammar'])
    deps = original.dependencies+_proof_dependencies(inputs)
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a,b in zip(deps,state.dependencies,strict=True)):
        raise ValueError('conditioned historical proof source publications differ')
    stages = []; current = available
    while current is not None:
        stages.append(current); current = current.parent()
    parent = None; base_parent = None
    for step, member in enumerate(reversed(stages)):
        ancestor = base._state(member, state.source, original.configuration, base_parent, step)
        history = tuple(row for row in state.payload['history'] if row['step'] <= step)
        reconstructed = _state(ancestor, history, state.configuration, parent)
        parent = reconstructed.coefficient_digest; base_parent = ancestor.coefficient_digest
    if parent != state.coefficient_digest:
        raise ValueError('conditioned historical proof or immutable parent seal differs')
    value = (source, original, available, inputs)
    with _lock: _models[state] = value
    return value


def cpp_conditioned_field_model(source, *, available_evidence, grammar, execution_context, reference=None):
    grammar = _grammar(grammar); module = _binding(); context = _context(module, execution_context)
    original = base.cpp_available_field_learning_model(source, available_evidence=available_evidence,
        execution_context=context, reference=reference, **POLICY)
    check_conditioned_model(original, source, (), context, available=available_evidence, grammar=grammar)
    c = {'profile': PROFILE, 'execution_provider': PROVIDER, 'engine_receipt': _receipts(), 'grammar': grammar,
         'pattern': PATTERN_PACKET, 'base_configuration': original.configuration, 'limits': LIMITS, 'max_bytes': MAX_BYTES}
    state = _state(original, (), c)
    with _lock: _models[state] = (source, original, available_evidence, ())
    enable_conditioned_field_provider()
    return state


@dataclass(frozen=True)
class ConditionedFieldQuery(Mapping):
    declaration: object
    prepared: object
    def __getitem__(self, name): return self.declaration[name]
    def __iter__(self): return iter(self.declaration)
    def __len__(self): return len(self.declaration)


def conditioned_field_query(state, prepared):
    state.check_state()
    if not isinstance(prepared, PreparedConditionedInput):
        raise TypeError('actual prepared conditioned source input required')
    return ConditionedFieldQuery(freeze_tree({'profile': QUERY, 'model_digest': state.coefficient_digest,
        'source': state.source.as_record(), 'input': prepared.archive()}), prepared)


def validate_conditioned_field_targets(value, source, space, mask):
    _closed(value, ('profile', 'model_digest', 'base_batch', 'inputs'))
    if value['profile'] != TARGETS:
        raise ValueError('conditioned whole-field targets required')
    batch = unpack_model(thaw_tree(value['base_batch']))
    if (not isinstance(batch, ModelBatch) or not batch.source.matches(source) or batch.space != space
            or not np.array_equal(mask, batch.observed) or batch.inputs is not None or batch.target_input is not None
            or len(value['inputs']) != len(batch.targets['observations'])
            or any(type(p) is not bytes or not 0 < len(p) <= MAX_BYTES for p in value['inputs'])):
        raise ValueError('conditioned targets change complete native observations or source coordinates')
    return value


def _samples(original, inputs, rows, context, grammar):
    if len(inputs) != len(rows): raise ValueError('one literal proof per original observation required')
    for item, observation in zip(inputs, rows, strict=True):
        row = item.admit(context, original.configuration['program_archive'], grammar=grammar)
        if (thaw_tree(observation['inputs']) != (row,) or observation['target_asset'] != row['asset']
                or observation['target_use'] != row['use'] or observation['key'] != item.literal.declaration()['key']-91000
                or tuple(observation['observed']) != tuple((0, i) for i in range(4))):
            raise ValueError('observation differs from complete original literal proof')


def conditioned_field_batch(state, source, samples, *, execution_context):
    _, original, _, _ = prepare_conditioned_field_model(state, source, execution_context=execution_context)
    samples = tuple(samples); inputs = tuple(p for p, _ in samples); rows = tuple(r for _, r in samples)
    _samples(original, inputs, rows, execution_context, state.configuration['grammar'])
    native = base.available_field_program_batch(original, source, rows, execution_context=execution_context)
    target = {'profile': TARGETS, 'model_digest': state.coefficient_digest, 'base_batch': pack_model(native),
              'inputs': tuple(item.archive() for item in inputs)}
    batch = ModelBatch(state.source, state.space, target, native.observed, dependencies=native.dependencies+_proof_dependencies(inputs))
    with _lock: _batches[batch] = (source, native, inputs)
    return batch


def _train(state, source, batch, *, steps=1, execution_context):
    _, original, _, _ = prepare_conditioned_field_model(state, source, execution_context=execution_context)
    if steps != 1 or batch.targets['model_digest'] != state.coefficient_digest or batch.inputs is not None:
        raise ValueError('one immutable conditioned evidence append required')
    with _lock: bound = _batches.get(batch)
    if bound is None:
        native = unpack_model(thaw_tree(batch.targets['base_batch']))
        inputs = tuple(PreparedConditionedInput.from_archive(p, execution_context, grammar=state.configuration['grammar']) for p in batch.targets['inputs'])
    else:
        _, native, inputs = bound
        native.source.bind(source)
    _samples(original, inputs, native.targets['observations'], execution_context, state.configuration['grammar'])
    deps = native.dependencies+_proof_dependencies(inputs)
    if len(deps) != len(batch.dependencies) or any(not a.matches(b) for a,b in zip(deps,batch.dependencies,strict=True)):
        raise ValueError('conditioned batch original dependency publications differ')
    from rexgraph.model_runtime import train_model
    child = train_model(original, source, native, execution_context=execution_context)
    history = state.payload['history']+tuple({'step': state.step+1, 'input': item.archive(),
        'proof_source': CppSourceReference(item.source.parent.model_source).as_record()} for item in inputs)
    result = _state(child, history, state.configuration, state.coefficient_digest)
    prepared = base.inspect_available_field_learning_model(child, source, execution_context=execution_context)
    old_inputs = prepare_conditioned_field_model(state, source, execution_context=execution_context)[3]
    with _lock: _models[result] = (source, child, prepared, old_inputs+inputs)
    return result


def _execute_input_family(available, rows, context):
    return available.execute_inputs(rows, context)


def _infer(state, source, inputs, *, execution_context):
    _, original, available, _ = prepare_conditioned_field_model(state, source, execution_context=execution_context)
    _closed(inputs, ('profile', 'model_digest', 'source', 'input'))
    if (inputs['profile'] != QUERY or inputs['model_digest'] != state.coefficient_digest
            or not source_from_record(inputs['source'], allow_cpp=True).matches(state.source)):
        raise ValueError('query differs from original conditioned model')
    item = inputs.prepared if isinstance(inputs, ConditionedFieldQuery) else PreparedConditionedInput.from_archive(inputs['input'], execution_context, grammar=state.configuration['grammar'])
    if item.archive() != inputs['input']: raise ValueError('live query source proof differs from its saved declaration')
    row = item.admit(execution_context, original.configuration['program_archive'], grammar=state.configuration['grammar'])
    family = _execute_input_family(available, (row,), execution_context)
    native = base._output(original, family)
    packet = pack_value({'profile': PROFILE, 'source': source.archive, 'model': pack_model(state), 'input': item.archive(), 'native': pack_model(native)})
    if len(packet) > MAX_BYTES: raise ValueError('complete conditioned result exceeds byte admission')
    deps = native.dependencies+_proof_dependencies(prepare_conditioned_field_model(state, source, execution_context=execution_context)[3]+(item,))
    archive = {'schema': SCHEMA, 'data': packet, 'sha256': sha256(packet).hexdigest(),
               'program_sha256': original.configuration['program_sha256'], 'engine_receipt': state.configuration['engine_receipt']}
    output = ModelOutput([], CoordinateSpace('native-conditioned-field-readout', ()), (), state.source, state.coefficient_digest,
        'rational', METHOD, dependencies=deps, field_archive=archive,
        execution_receipt={'profile': METHOD, 'limits': LIMITS, 'max_bytes': MAX_BYTES, 'interpretation': 'complete-source-conditioned-correlated-field-family',
                           'dependency_count': len(deps)})
    with _lock: _outputs[output] = (state, source, item, native, family)
    return output


def read_conditioned_field_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if limits is not None and (len(limits) != 4 or any(type(a) is not int or a < b for a,b in zip(limits, LIMITS, strict=True))):
        raise ValueError("requested admission does not cover the fixed conditioned profile")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0 or len(output.field_archive["data"]) > max_bytes):
        raise ValueError("conditioned output exceeds requested byte admission")
    if (not isinstance(output, ModelOutput) or output.method != METHOD or output.space != CoordinateSpace('native-conditioned-field-readout', ())
            or output.axes or output.values.size or output.grade is not None or output.variance != 'coordinate'):
        raise TypeError('complete conditioned native field output required')
    output.check_state(); a = output.field_archive; r = output.execution_receipt
    _closed(r, ('profile', 'limits', 'max_bytes', 'interpretation', 'dependency_count'))
    if (r['profile'] != METHOD or tuple(r['limits']) != LIMITS or r['max_bytes'] != MAX_BYTES
            or r['interpretation'] != 'complete-source-conditioned-correlated-field-family' or r['dependency_count'] != len(output.dependencies)
            or a['schema'] != SCHEMA or len(a['data']) > MAX_BYTES or thaw_tree(a['engine_receipt']) != _receipts()):
        raise ValueError('conditioned output receipt or pinned implementation differs')
    with _lock: known = _outputs.get(output)
    if known is None:
        row = unpack_value(a['data']); _closed(row, ('profile', 'source', 'model', 'input', 'native'))
        if row['profile'] != PROFILE: raise ValueError('conditioned output profile differs')
        model = unpack_model(row['model'])
        original = unpack_model(thaw_tree(model.payload['base_model']))
        source = _binding().NativeSource(row['source'], max_bytes=MAX_BYTES)
        prepare_conditioned_field_model(model, source, execution_context=execution_context)
        item = PreparedConditionedInput.from_archive(row['input'], execution_context, grammar=model.configuration['grammar'])
        native = unpack_model(row['native']); family = base.read_available_field_learning_result(native, execution_context=execution_context, state=original)
        known = (model, source, item, native, family)
        with _lock: _outputs[output] = known
    model, source, item, native, family = known
    if state is not None and state.coefficient_digest != model.coefficient_digest:
        raise ValueError('conditioned output belongs to another model')
    _, original, _, history = prepare_conditioned_field_model(model, source, execution_context=execution_context)
    item.admit(execution_context, original.configuration['program_archive'], grammar=model.configuration['grammar']); family.admit(execution_context)
    if (output.model_digest != model.coefficient_digest or not output.source.matches(model.source)
            or family.input().result().archive() != item.literal.program().execute((item.literal.input(),)).archive()
            or a['program_sha256'] != original.configuration['program_sha256']):
        raise ValueError('conditioned result differs from complete original model/input execution')
    deps = native.dependencies+_proof_dependencies(history+(item,))
    if len(deps) != len(output.dependencies) or any(not a.matches(b) for a,b in zip(deps,output.dependencies,strict=True)):
        raise ValueError('conditioned result dependency lineage differs')
    return family


def _create(source, *, adapter=ADAPTER, **configuration):
    if adapter != ADAPTER: raise ValueError('conditioned field adapter required')
    return cpp_conditioned_field_model(source, **configuration)


def enable_conditioned_field_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(PROVIDER['name'], '1', ((ADAPTER, '1'),), _infer, create=_create,
                prepare=prepare_conditioned_field_model, train=_train, target_profiles=(TARGETS,),
                context_operations=('create', 'prepare', 'infer', 'train'))
            register_model_provider(_provider)
    return _provider
