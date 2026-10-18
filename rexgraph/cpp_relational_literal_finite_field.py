"""Complete available native reader families and explicit prospective choices."""
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock
from weakref import WeakKeyDictionary

from rexgraph import cpp_relational_finite_field as base
from rexgraph.cpp_relational_field_program import _LIMITS, _admission
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import source_from_record

ADAPTER = 'cpp_relational_literal_finite_field'
PROVIDER = {'name': 'rexgraph.cpp.relational_literal_finite_field', 'version': '1'}
PROFILE = 'rexgraph.available-relational-literal-finite-model.v1'
QUERY = 'rexgraph.available-relational-literal-finite-query.v1'
SCHEMA = 'rexgraph.available-relational-literal-finite-field-family.v1'
METHOD = 'exact-available-relational-literal-finite-field-family-v1'
CHOICE_SCHEMA = 'rexgraph.available-relational-literal-finite-field-evaluation.v1'
CHOICE_METHOD = 'exact-available-relational-literal-finite-field-evaluation-v1'
_lock = RLock()
_models = WeakKeyDictionary()
_outputs = WeakKeyDictionary()
_provider = None


def _binding():
    module = base._binding()
    if not hasattr(module, 'PreparedAvailableLiteralFiniteFieldMap'):
        raise ImportError('literal finite models require their complete native reader availability provider')
    return module


def _base(state):
    value = unpack_model(state.payload['base_checkpoint'])
    if not isinstance(value, ModelState):
        raise ValueError('original immutable finite declaration checkpoint required')
    base.validate_finite_field_state(value)
    if (value.coefficient_digest != state.configuration['base_digest'] or not value.source.matches(state.source)
            or value.space != state.space or value.output_axes != state.output_axes
            or tuple(value.configuration['limits']) != tuple(state.configuration['limits'])
            or value.configuration['max_bytes'] != state.configuration['max_bytes']):
        raise ValueError('reader source, coordinates, target declaration or admission differs')
    return value


def validate_literal_finite_state(state):
    c, p = state.configuration, state.payload
    if (state.adapter != ADAPTER or state.adapter_version != '1' or state.arithmetic != 'rational'
            or set(c) != {'profile', 'execution_provider', 'engine_receipt', 'limits', 'max_bytes',
                          'base_digest', 'map_sha256', 'reader_sha256'}
            or c['profile'] != PROFILE or dict(c['execution_provider']) != PROVIDER
            or not isinstance(c['engine_receipt'], Mapping) or not c['engine_receipt']
            or set(p) != {'map_archive', 'base_checkpoint'} or state.output_axes
            or state.step != 0 or state.parent is not None):
        raise ValueError('closed immutable available reader model required')
    _admission(c['limits'], c['max_bytes'])
    if (type(p['map_archive']) is not bytes or not p['map_archive'] or len(p['map_archive']) > c['max_bytes']
            or sha256(p['map_archive']).hexdigest() != c['map_sha256']
            or any(not base._sha(c[key]) for key in ('base_digest', 'map_sha256', 'reader_sha256'))):
        raise ValueError('complete available reader bytes or declaration differs')
    _base(state)


def _targets(declaration):
    return base._targets(declaration.payload['target_checkpoints'], declaration.configuration['target_digests'])


def _dependencies(module, native, declaration, authority):
    return base._dependencies(module, native, _targets(declaration), authority, declaration.configuration['limits'])


def _check(module, native, state, declaration, authority):
    c = state.configuration
    native.admit(authority)
    if (native.admission() != {'limits': tuple(c['limits']), 'max_bytes': c['max_bytes']}
            or sha256(native.archive()).hexdigest() != c['map_sha256']
            or sha256(native.reader(authority).archive()).hexdigest() != c['reader_sha256']):
        raise ValueError('complete reader recipe or policy differs')
    base._check_pins(module, _targets(declaration))
    base._check_map(module, native.base(authority), _targets(declaration), state.source, declaration.configuration, authority)
    deps = _dependencies(module, native, declaration, authority)
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a, b in zip(deps, state.dependencies, strict=True)):
        raise ValueError('complete reader publisher dependencies or ordering differ')


def cpp_available_literal_finite_field_model(source, *, available_map, targets, execution_context,
                                            limits=_LIMITS, max_bytes=256*1024*1024, reference=None):
    module = _binding()
    authority = _context(module, execution_context)
    limits = _admission(limits, max_bytes)
    if not isinstance(available_map, module.PreparedAvailableLiteralFiniteFieldMap):
        raise TypeError('actual complete native available reader map required')
    available_map.admit(authority)
    declaration = base.cpp_available_finite_field_model(source, available_map=available_map.base(authority),
        targets=targets, execution_context=authority, limits=limits, max_bytes=max_bytes, reference=reference)
    image = available_map.archive()
    configuration = {'profile': PROFILE, 'execution_provider': PROVIDER,
        'engine_receipt': module.available_literal_finite_field_build_info(), 'limits': limits, 'max_bytes': max_bytes,
        'base_digest': declaration.coefficient_digest, 'map_sha256': sha256(image).hexdigest(),
        'reader_sha256': sha256(available_map.reader(authority).archive()).hexdigest()}
    state = ModelState(ADAPTER, '1', declaration.source, declaration.space, (), configuration,
        {'map_archive': image, 'base_checkpoint': pack_model(declaration, native=True)}, 'rational',
        dependencies=_dependencies(module, available_map, declaration, authority))
    _check(module, available_map, state, declaration, authority)
    with _lock:
        _models[state] = available_map
    enable_literal_finite_field_provider()
    return state


def prepare_literal_finite_field_model(state, source, *, execution_context):
    validate_literal_finite_state(state)
    state.bind(source)
    module = _binding()
    authority = _context(module, execution_context)
    c = state.configuration
    if dict(c['engine_receipt']) != module.available_literal_finite_field_build_info():
        raise ValueError('reader model requires its pinned native implementation')
    declaration = _base(state)
    base._check_pins(module, _targets(declaration))
    with _lock:
        native = _models.get(state)
    if native is None:
        native = module.PreparedAvailableLiteralFiniteFieldMap.from_archive(state.payload['map_archive'], authority,
                    limits=c['limits'], max_bytes=c['max_bytes'])
        _check(module, native, state, declaration, authority)
        with _lock:
            _models[state] = native
    native.admit(authority)
    return native


def literal_finite_field_query(state):
    validate_literal_finite_state(state)
    return freeze_tree({'profile': QUERY, 'model_digest': state.coefficient_digest,
                       'source': state.source.as_record(), 'map_sha256': state.configuration['map_sha256']})


def _output(state, native, authority, selected=False):
    module = _binding()
    declaration = _base(state)
    image = native.archive()
    method, schema = (CHOICE_METHOD, CHOICE_SCHEMA) if selected else (METHOD, SCHEMA)
    family = native.family(authority) if selected else native
    model = family.model(authority)
    deps = _dependencies(module, native if selected else model, declaration, authority)
    archive = {'schema': schema, 'data': image, 'sha256': sha256(image).hexdigest(),
               'map_sha256': state.configuration['map_sha256'], 'engine_receipt': state.configuration['engine_receipt']}
    receipt = {'profile': method, 'limits': state.configuration['limits'], 'max_bytes': state.configuration['max_bytes'],
               'model_checkpoint': pack_model(state, native=True), 'dependency_count': len(deps)}
    native.admit(authority)
    output = ModelOutput([], base._READOUT, (), state.source, state.coefficient_digest, 'rational', method,
                         dependencies=deps, execution_receipt=receipt, field_archive=archive)
    with _lock:
        _outputs[output] = native
    return output


def _infer(state, source, inputs, *, execution_context):
    native = prepare_literal_finite_field_model(state, source, execution_context=execution_context)
    if (not isinstance(inputs, Mapping) or set(inputs) != {'profile', 'model_digest', 'source', 'map_sha256'}
            or inputs['profile'] != QUERY or inputs['model_digest'] != state.coefficient_digest
            or inputs['map_sha256'] != state.configuration['map_sha256']
            or not source_from_record(inputs['source'], allow_cpp=True).matches(state.source)):
        raise ValueError('closed query for the original complete reader required')
    return _output(state, native.execute(execution_context), execution_context)


def read_literal_finite_field_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if (not isinstance(output, ModelOutput) or output.method not in {METHOD, CHOICE_METHOD}
            or output.space != base._READOUT or output.axes or output.values.size
            or output.grade is not None or output.variance != 'coordinate'):
        raise TypeError('complete available reader family or prospective choice required')
    output.check_state()
    r, a = output.execution_receipt, output.field_archive
    if (set(r) != {'profile', 'limits', 'max_bytes', 'model_checkpoint', 'dependency_count'}
            or r['profile'] != output.method or r['dependency_count'] != len(output.dependencies)):
        raise ValueError('closed complete reader output receipt required')
    original = unpack_model(r['model_checkpoint'])
    validate_literal_finite_state(original)
    if (original.coefficient_digest != output.model_digest or not original.source.matches(output.source)
            or a['map_sha256'] != original.configuration['map_sha256']
            or tuple(r['limits']) != tuple(original.configuration['limits'])
            or r['max_bytes'] != original.configuration['max_bytes']
            or dict(a['engine_receipt']) != dict(original.configuration['engine_receipt'])
            or a['schema'] != (SCHEMA if output.method == METHOD else CHOICE_SCHEMA)):
        raise ValueError('reader output belongs to another complete model or policy')
    if state is not None:
        validate_literal_finite_state(state)
        if state.coefficient_digest != original.coefficient_digest:
            raise ValueError('reader output belongs to another model')
    module = _binding()
    authority = _context(module, execution_context)
    if dict(a['engine_receipt']) != module.available_literal_finite_field_build_info():
        raise ValueError('reader output requires its pinned native implementation')
    admitted = _admission(r['limits'], r['max_bytes'])
    maximum = r['max_bytes']
    if limits is not None:
        admitted = tuple(min(x, y) for x, y in zip(admitted, _admission(limits, maximum), strict=True))
    if max_bytes is not None:
        _admission(admitted, max_bytes)
        maximum = min(maximum, max_bytes)
    with _lock:
        native = _outputs.get(output) if admitted == tuple(r['limits']) and maximum == r['max_bytes'] else None
    if native is None:
        kind = module.NativeAvailableLiteralFiniteFieldResult if output.method == METHOD else module.NativeAvailableLiteralFiniteFieldEvaluation
        native = kind.from_archive(a['data'], authority, limits=admitted, max_bytes=maximum)
        family = native if output.method == METHOD else native.family(authority)
        _check(module, family.model(authority), original, _base(original), authority)
        deps = _dependencies(module, native if output.method == CHOICE_METHOD else family.model(authority), _base(original), authority)
        if len(deps) != len(output.dependencies) or any(not x.matches(y) for x, y in zip(deps, output.dependencies, strict=True)):
            raise ValueError('reader output dependency closure differs')
        if admitted == tuple(r['limits']) and maximum == r['max_bytes']:
            with _lock:
                _outputs[output] = native
    native.admit(authority)
    return native


def evaluate_literal_finite_field(output, ordinal, values, parameter_setting, choice_origin, *, execution_context, state):
    if output.method != METHOD:
        raise TypeError('complete unresolved reader family required')
    family = read_literal_finite_field_result(output, execution_context=execution_context, state=state)
    return _output(state, family.evaluate(ordinal, values, parameter_setting, choice_origin, execution_context), execution_context, True)


def literal_finite_field_branches(output, begin, count, *, execution_context):
    native = read_literal_finite_field_result(output, execution_context=execution_context)
    family = native if output.method == METHOD else native.family(execution_context)
    declaration = _base(unpack_model(output.execution_receipt['model_checkpoint']))
    targets = _targets(declaration)
    records = []
    for branch in family.select(begin, count, execution_context):
        info = branch.inspect(execution_context)
        raw = branch.raw(execution_context).result()
        index = info['target']
        records.append(dict(info, target_source=None if index is None else targets[index].source.as_record(),
                            fields=() if raw is None else raw.fields(), origin_output=output.coefficient_digest,
                            family_sha256=sha256(family.archive()).hexdigest()))
    return tuple(records)


def literal_finite_field_info(output, *, execution_context):
    native = read_literal_finite_field_result(output, execution_context=execution_context)
    family = native if output.method == METHOD else native.family(execution_context)
    declaration = _base(unpack_model(output.execution_receipt['model_checkpoint']))
    return {'map_sha256': output.field_archive['map_sha256'], 'archive_sha256': output.field_archive['sha256'],
            'branch_count': family.inspect(execution_context)['branch_count'],
            'target_count': len(declaration.configuration['target_digests']),
            'selected_ordinal': None if output.method == METHOD else native.inspect(execution_context)['ordinal'],
            'availability': 'complete-reader-and-independent-native-context-admitted', 'retains_complete_alternatives': True}


def literal_finite_field_stalks(output, node, cells, *, execution_context, coordinate_limit=256):
    if output.method != CHOICE_METHOD:
        raise TypeError('reader stalk selection requires an explicit prospective choice')
    native = read_literal_finite_field_result(output, execution_context=execution_context)
    ordinal = native.inspect(execution_context)['ordinal']
    branch = native.family(execution_context).select(ordinal, 1, execution_context)[0].inspect(execution_context)
    declaration = _base(unpack_model(output.execution_receipt['model_checkpoint']))
    target = _targets(declaration)[branch['target']]
    return tuple(dict(row, origin_output=output.coefficient_digest, map_sha256=output.field_archive['map_sha256'],
                      finite_ordinal=ordinal, target_source=target.source.as_record())
                 for row in native.raw(execution_context).result().select(node, cells, coordinate_limit=coordinate_limit))


def clear_literal_finite_field_cache():
    with _lock:
        _models.clear()
        _outputs.clear()


def _create(source, *, adapter=ADAPTER, **configuration):
    if adapter != ADAPTER:
        raise ValueError('complete reader adapter required')
    return cpp_available_literal_finite_field_model(source, **configuration)


def enable_literal_finite_field_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(PROVIDER['name'], '1', ((ADAPTER, '1'),), _infer, create=_create,
                prepare=prepare_literal_finite_field_model, context_operations=('create', 'prepare', 'infer'))
        register_model_provider(_provider)
    return _provider
