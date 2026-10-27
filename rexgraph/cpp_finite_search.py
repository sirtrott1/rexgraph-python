"""Admitted native finite projections with explicit cold reconstruction recipes."""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_evidence_admission import _references
from rexgraph.cpp_relational_field_program import _LIMITS, _admission
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value

_ADAPTER = 'cpp_finite_search'
_PROVIDER = {'name': 'rexgraph.cpp.finite_search', 'version': '1'}
_PROFILE = 'rexgraph.available-finite-projection-model.v1'
_QUERY = 'rexgraph.available-finite-projection-query.v1'
_RESULT = 'rexgraph.available-finite-projection-replay.v1'
_METHOD = 'exact-available-finite-projection-replay-v1'
_SPACE = CoordinateSpace('native-finite-projection-readout', ())
_SEARCH = (65536, 4194304, 16777216)
_FAMILY = (65536, 64000000, 16777216, 65536, 65536)
_lock = RLock()
_cache = OrderedDict()
_results = OrderedDict()
_provider = None


def _binding():
    from rexgraph.cpp_model import _binding as binding
    module = binding()
    if not hasattr(module, 'PreparedFiniteSearch'):
        raise ImportError('finite projections require their owning native search provider')
    return module


def _naturals(values, count=None):
    values = tuple(values)
    if (count is not None and len(values) != count) or any(type(v) is not int or not 0 <= v < 1 << 64 for v in values):
        raise ValueError('finite recipe requires nonnegative integer coordinates and limits')
    return values


def _recipe(value):
    required = {'key', 'revision', 'cells', 'window', 'fixed', 'distinct_relations', 'reuse_root', 'search_limits', 'family_limits'}
    if not isinstance(value, Mapping) or set(value) != required or type(value['reuse_root']) is not bool:
        raise ValueError('closed finite projection recipe required')
    key, revision = _naturals((value['key'], value['revision']), 2)
    if not key or not revision: raise ValueError('finite projection identity and revision required')
    cells = tuple(_naturals(row, 2) for row in value['cells'])
    groups = tuple(_naturals(row, 2) for row in value['distinct_relations'])
    fixed = tuple(_naturals(row, 3) for row in value['fixed'])
    window = None if value['window'] is None else tuple(_naturals(row) for row in value['window'])
    if len(set(cells)) != len(cells) or len({v[:2] for v in fixed}) != len(fixed):
        raise ValueError('finite recipe repeats an interface or fixed cell')
    if window is not None and len(window) != len(cells): raise ValueError('finite window arity differs')
    return freeze_tree(dict(key=key, revision=revision, cells=cells, window=window, fixed=fixed,
        distinct_relations=groups, reuse_root=value['reuse_root'], search_limits=_naturals(value['search_limits'], 3),
        family_limits=_naturals(value['family_limits'], 5)))


def _declaration(recipe):
    return {key: recipe[key] for key in ('key', 'revision', 'cells', 'window', 'fixed', 'distinct_relations')}


def _space(reference):
    return CoordinateSpace(f'cpp:{reference.native.namespace}:finite-projection-source',
        tuple(f'{g}:{key}' for g, row in enumerate(reference.native.coordinates) for key in row))


def validate_finite_search_state(state):
    c = state.configuration
    required = {'profile', 'execution_provider', 'engine_receipt', 'admission_receipt', 'limits', 'max_bytes',
                'available_sha256', 'source_sha256', 'recipe'}
    if (state.adapter != _ADAPTER or state.adapter_version != '1' or state.arithmetic != 'rational'
            or not isinstance(state.source, CppSourceReference) or set(c) != required
            or c['profile'] != _PROFILE or dict(c['execution_provider']) != _PROVIDER
            or set(state.payload) != {'available_archive'} or state.output_axes or state.step or state.parent is not None
            or state.space != _space(state.source)):
        raise ValueError('immutable admitted finite projection model required')
    _admission(c['limits'], c['max_bytes'])
    if _recipe(c['recipe']) != c['recipe']: raise ValueError('finite projection recipe is not canonical')
    archive = state.payload['available_archive']
    if type(archive) is not bytes or not archive or len(archive) > c['max_bytes'] or sha256(archive).hexdigest() != c['available_sha256']:
        raise ValueError('finite projection admitted source archive differs')
    for name in ('available_sha256', 'source_sha256'):
        value = c[name]
        if type(value) is not str or len(value) != 64 or any(ch not in '0123456789abcdef' for ch in value):
            raise ValueError('canonical finite source digest required')
    for name in ('engine_receipt', 'admission_receipt'):
        if not isinstance(c[name], Mapping) or not c[name]: raise ValueError('finite native implementation receipt required')


def _dependencies(module, available, context, limits):
    return _references(module, available.dependency_archives(context, max_records=limits[1]), limits)


def _check(module, available, state, authority):
    available.admit(authority)
    c = state.configuration
    actual = module.NativeSource(available.raw().asset().source_archive(), max_dimension=max(c['limits'][:2]), max_records=c['limits'][1])
    state.source.bind(actual)
    if (sha256(available.archive()).hexdigest() != c['available_sha256']
            or sha256(available.raw().archive()).hexdigest() != c['source_sha256']):
        raise ValueError('finite projection source or publisher closure differs')
    deps = _dependencies(module, available, authority, c['limits'])
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a, b in zip(deps, state.dependencies, strict=True)):
        raise ValueError('finite projection publisher histories differ')


@dataclass(frozen=True)
class PreparedFiniteProjection:
    """One admitted source and its actual native compiled owners."""
    _available: object
    _prepared: object
    _root: object
    _family: object
    preparation: object
    propagation: object

    def raw_family(self, execution_context):
        """Admit current authority before returning the explicitly raw math owner."""
        self._available.admit(_context(_binding(), execution_context))
        return self._family

    def raw_prepared(self, execution_context):
        self._available.admit(_context(_binding(), execution_context))
        return self._prepared

    def raw_propagation(self, execution_context):
        self._available.admit(_context(_binding(), execution_context))
        return self._root

    def materialize(self, execution_context, *, family_limits=_FAMILY, cancellation=None):
        authority = _context(_binding(), execution_context)
        self._available.admit(authority)
        if self._family is None:
            stage = self.propagation if self.propagation is not None else self.preparation
            return {'status': stage['status'], 'image': None, 'queries': None, 'work': None}
        result = self._family.materialize(family_limits=family_limits, cancellation=cancellation)
        self._available.admit(authority)
        return result


def _compile(available, recipe, authority, limits, max_bytes):
    module = _binding()
    available.admit(authority)
    result = module.PreparedFiniteSearch.prepare(available.raw(), recipe['distinct_relations'], search_limits=recipe['search_limits'])
    prepared = result['prepared']; preparation = freeze_tree({'status': result['status'], 'work': result['work']})
    root = family = propagation = None
    if prepared is not None:
        if recipe['reuse_root']:
            result = prepared.propagate(recipe['fixed'], search_limits=recipe['search_limits'])
            root = result['snapshot']; propagation = freeze_tree({'status': result['status'], 'work': result['work']})
        if not recipe['reuse_root'] or root is not None:
            family = module.NativeFiniteFamily.projection(root if root is not None else prepared,
                _declaration(recipe), limits=limits, max_bytes=max_bytes)
    available.admit(authority)
    return PreparedFiniteProjection(available, prepared, root, family, preparation, propagation)


def cpp_finite_search_model(source, *, available_source, projection, execution_context, reuse_root=True,
                            search_limits=_SEARCH, family_limits=_FAMILY, limits=_LIMITS,
                            max_bytes=256*1024*1024, reference=None):
    module = _binding(); authority = _context(module, execution_context); limits = _admission(limits, max_bytes)
    if not isinstance(source, module.NativeSource) or not isinstance(available_source, module.PreparedAvailableFiniteSource):
        raise TypeError('actual native source and available finite owner required')
    if available_source.admission() != {'limits': limits, 'max_bytes': max_bytes}:
        raise ValueError('finite source admission differs from its model')
    if not isinstance(projection, Mapping) or set(projection) != {'key', 'revision', 'cells', 'window', 'fixed', 'distinct_relations'}:
        raise ValueError('complete finite projection declaration required')
    recipe = _recipe(dict(projection, reuse_root=reuse_root, search_limits=search_limits, family_limits=family_limits))
    available_source.admit(authority)
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    archive = available_source.archive()
    config = dict(profile=_PROFILE, execution_provider=_PROVIDER, engine_receipt=module.finite_search_build_info(),
        admission_receipt=module.finite_field_admission_build_info(), available_sha256=sha256(archive).hexdigest(),
        source_sha256=sha256(available_source.raw().archive()).hexdigest(), recipe=recipe, limits=limits, max_bytes=max_bytes)
    state = ModelState(_ADAPTER, '1', ref, _space(ref), (), config, {'available_archive': archive}, 'rational',
        dependencies=_dependencies(module, available_source, authority, limits))
    _check(module, available_source, state, authority)
    compiled = _compile(available_source, recipe, authority, limits, max_bytes)
    with _lock:
        _cache[(id(source), state.coefficient_digest)] = (source, compiled)
        while len(_cache) > 4: _cache.popitem(last=False)
    enable_finite_search_provider()
    return state


def prepare_finite_search_model(state, source, *, execution_context):
    validate_finite_search_state(state); state.bind(source)
    module = _binding(); authority = _context(module, execution_context); c = state.configuration
    if dict(c['engine_receipt']) != module.finite_search_build_info() or dict(c['admission_receipt']) != module.finite_field_admission_build_info():
        raise ValueError('finite projection requires its pinned native implementation')
    key = (id(source), state.coefficient_digest)
    with _lock:
        cached = _cache.get(key)
        compiled = cached[1] if cached is not None and cached[0] is source else None
    if compiled is None:
        available = module.PreparedAvailableFiniteSource.from_archive(state.payload['available_archive'], authority,
            limits=c['limits'], max_bytes=c['max_bytes'])
        _check(module, available, state, authority)
        compiled = _compile(available, c['recipe'], authority, c['limits'], c['max_bytes'])
    _check(module, compiled._available, state, authority)
    with _lock:
        _cache[key] = (source, compiled); _cache.move_to_end(key)
        while len(_cache) > 4: _cache.popitem(last=False)
    return compiled


def clear_finite_search_cache():
    with _lock:
        _cache.clear(); _results.clear()


def finite_search_query(state):
    validate_finite_search_state(state)
    return freeze_tree(dict(profile=_QUERY, model_digest=state.coefficient_digest, source=state.source.as_record()))


def _infer(state, source, inputs, *, execution_context):
    if (not isinstance(inputs, Mapping) or set(inputs) != {'profile', 'model_digest', 'source'} or inputs['profile'] != _QUERY
            or inputs['model_digest'] != state.coefficient_digest
            or not source_from_record(inputs['source'], allow_cpp=True).matches(state.source)):
        raise ValueError('query must select the original immutable finite projection model')
    compiled = prepare_finite_search_model(state, source, execution_context=execution_context)
    result = compiled.materialize(execution_context, family_limits=state.configuration['recipe']['family_limits'])
    receipt = dict(profile=_RESULT, status=result['status'], preparation=compiled.preparation,
        propagation=compiled.propagation, work=result['work'],
        row_count=None if result['image'] is None else result['image'].inspect()['row_count'],
        query_count=None if result['queries'] is None else result['queries'].inspect()['query_count'],
        reconstruction='admitted-source-and-complete-request-reexecution')
    packet = pack_value({'profile': _RESULT, 'state': pack_model(state), 'receipt': thaw_tree(receipt)})
    if len(packet) > state.configuration['max_bytes']: raise ValueError('finite replay packet exceeds byte admission')
    archive = dict(schema=_RESULT, data=packet, sha256=sha256(packet).hexdigest(),
        program_sha256=state.coefficient_digest, engine_receipt=state.configuration['engine_receipt'])
    output = ModelOutput([], _SPACE, (), state.source, state.coefficient_digest, 'rational', _METHOD,
        dependencies=state.dependencies, execution_receipt=receipt, field_archive=archive)
    with _lock:
        _results[output.coefficient_digest] = (compiled, result)
        while len(_results) > 4: _results.popitem(last=False)
    return output


def read_finite_search_result(output, *, execution_context, state=None):
    if (not isinstance(output, ModelOutput) or output.method != _METHOD or output.space != _SPACE
            or output.axes or output.values.size or output.grade is not None or output.variance != 'coordinate'
            or output.field_archive is None or output.field_archive['schema'] != _RESULT):
        raise TypeError('complete finite projection replay output required')
    output.check_state()
    module = _binding(); authority = _context(module, execution_context)
    if dict(output.field_archive['engine_receipt']) != module.finite_search_build_info():
        raise ValueError('finite result requires its pinned native implementation')
    packet = unpack_value(output.field_archive['data'])
    if not isinstance(packet, Mapping) or set(packet) != {'profile', 'state', 'receipt'} or packet['profile'] != _RESULT:
        raise ValueError('closed finite replay packet required')
    original = unpack_model(packet['state'])
    validate_finite_search_state(original)
    if (original.coefficient_digest != output.model_digest or output.field_archive['program_sha256'] != original.coefficient_digest
            or not original.source.matches(output.source) or packet['receipt'] != output.execution_receipt
            or len(original.dependencies) != len(output.dependencies)
            or any(not a.matches(b) for a, b in zip(original.dependencies, output.dependencies, strict=True))
            or state is not None and state.coefficient_digest != original.coefficient_digest):
        raise ValueError('finite replay model, receipt or source lineage differs')
    c = original.configuration
    with _lock: cached = _results.get(output.coefficient_digest)
    if cached is None:
        available = module.PreparedAvailableFiniteSource.from_archive(original.payload['available_archive'], authority,
            limits=c['limits'], max_bytes=c['max_bytes'])
        _check(module, available, original, authority)
        compiled = _compile(available, c['recipe'], authority, c['limits'], c['max_bytes'])
        result = compiled.materialize(authority, family_limits=c['recipe']['family_limits'])
    else:
        compiled, result = cached
    _check(module, compiled._available, original, authority)
    expected = dict(profile=_RESULT, status=result['status'], preparation=compiled.preparation, propagation=compiled.propagation,
        work=result['work'], row_count=None if result['image'] is None else result['image'].inspect()['row_count'],
        query_count=None if result['queries'] is None else result['queries'].inspect()['query_count'],
        reconstruction='admitted-source-and-complete-request-reexecution')
    if freeze_tree(expected) != output.execution_receipt: raise ValueError('finite native reexecution differs from its complete receipt')
    with _lock:
        _results[output.coefficient_digest] = (compiled, result); _results.move_to_end(output.coefficient_digest)
        while len(_results) > 4: _results.popitem(last=False)
    return {'status': result['status'], 'image': result['image'],
            'queries': result['queries'],
            'work': None if result['work'] is None else freeze_tree(result['work'])}


def finite_search_branches(output, begin=0, count=16, *, execution_context):
    begin, count = _naturals((begin, count), 2)
    result = read_finite_search_result(output, execution_context=execution_context)
    image = result['image']
    if image is None: raise ValueError('incomplete or refuted preparation has no materialized image')
    size = image.inspect()['row_count']
    if begin > size or count > size - begin: raise IndexError('finite branch slice outside complete image')
    return tuple(dict(image.row(i).inspect(), ordinal=i, source=output.source.as_record(),
        origin_output=output.coefficient_digest, interpretation='conditional-existential-whole-source-witness') for i in range(begin, begin + count))


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError('finite search adapter required')
    return cpp_finite_search_model(source, **configuration)


def enable_finite_search_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER['name'], '1', ((_ADAPTER, '1'),), _infer,
                create=_create, prepare=prepare_finite_search_model, context_operations=('create', 'prepare', 'infer'))
        register_model_provider(_provider)
    return _provider
