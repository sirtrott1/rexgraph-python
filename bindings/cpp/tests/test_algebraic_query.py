"""Owning exact snapshot queries, supported coefficient windows and common records."""
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeAlgebraicArchive,
    NativeAlgebraicQueryCache,
    NativeAlgebraicQueryResult,
    NativeAlgebraicRecord,
    NativeAlgebraicSnapshot,
    NativeAlgebraicTensorWindow,
    PreparedAlgebraicQuery,
)
from test_algebraic_archive import algebraic_fixtures as algebraic_fixtures
from test_algebraic_archive import archive_case as archive_case
from test_algebraic_archive import exact_rows
from test_algebraic_graph import graph as graph
from test_algebraic_graph import graph_data as graph_data

from rexgraph.cpp_algebraic_query import (
    algebraic_query_observation,
    algebraic_query_record,
    algebraic_snapshot_record,
    algebraic_window_observation,
    algebraic_window_record,
    prepare_algebraic_query,
    read_algebraic_observation_record,
    read_algebraic_snapshot_record,
    validate_algebraic_observation,
)
from rexgraph.state import from_state, to_state
from rexgraph.value_codec import pack_value, unpack_value

AVAILABILITY = (4001, 5)


def publication(archive):
    return NativeAlgebraicSnapshot(archive, namespace=AVAILABILITY[0], sequence=AVAILABILITY[1])


def select(snapshot, **options):
    return PreparedAlgebraicQuery(snapshot, availability=AVAILABILITY, **options)


def execute(query, **options):
    return query.execute(availability=AVAILABILITY, **options)


def box(snapshot, key=90, lower=(1, 0), upper=(3, 2), **options):
    return NativeAlgebraicTensorWindow(snapshot.record(key), lower, upper, availability=AVAILABILITY, **options)


def reseal(value):
    value['seal'] = sha256(pack_value({k: v for k, v in value.items() if k != 'seal'})).hexdigest()
    return value


@pytest.fixture(scope='session')
def query_oracle(algebraic_fixtures):
    folder, _ = algebraic_fixtures
    executable = os.environ.get('REXGRAPH_CPP_QUERY_FIXTURE')
    if executable is None:
        from pathlib import Path
        build = folder / 'query-build'
        subprocess.run(['cmake', '-S', str(Path(__file__).resolve().parents[1]), '-B', str(build),
                        '-DREXGRAPH_BINDING_TEST_FIXTURE=ON', f'-DPython3_EXECUTABLE={sys.executable}'],
                       check=True, capture_output=True, text=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'algebraic_query_fixture', '-j1'],
                       check=True, capture_output=True, text=True)
        executable = build / 'algebraic_query_fixture'

    result = subprocess.run([str(executable), str(folder)], check=True, text=True, capture_output=True)
    return {row['profile']: row for row in map(json.loads, result.stdout.splitlines())}


def test_owning_snapshot_pins_preserve_actual_publication_and_record_domains(archive_case):
    archive, _ = archive_case
    first, second = publication(archive), publication(archive)
    assert first.same_publication(first) and not first.same_publication(second)
    assert first.archive_owner is archive
    assert first.preview(0)['native_history'] is False
    assert archive.record_count == len(archive.manifest['records'])
    assert archive.field_degree == archive.manifest['field_degree']
    vector, mapping = first.record(10), first.record(90)
    assert vector.domain == (2,) and mapping.domain == (3, 2)
    assert vector.value.coefficients == archive.record(10).coefficients
    assert vector.coordinate((1,)).coefficients[0] == vector.coefficients[1]
    assert mapping.coordinate((2, 1)).coefficients[0] == mapping.coefficients[5]
    assert vector.snapshot_owner is first
    assert vector.value.archive_owner is archive
    assert archive.record(30).apply(vector.value).coefficients == archive.record(30).apply(archive.record(10)).coefficients
    del first, second, archive
    gc.collect()
    assert vector.snapshot_owner.namespace == 4001 and len(vector.coefficients) == 2
    with pytest.raises(ValueError): vector.coordinate((0, 0))
    with pytest.raises(IndexError): mapping.coordinate((3, 0))


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_prepared_queries_match_independent_canonical_results(archive_case, query_oracle, workers):
    archive, oracle = archive_case
    snapshot = publication(archive)
    zero = archive.scalar([Fraction(0)] * archive.field_degree)
    query = select(snapshot, kind='vector', rows_per_tile=1)
    result = execute(query, cpu_workers=workers)
    expected = query_oracle[oracle['profile']]
    assert result.keys == tuple(expected['vectors'])
    assert result.query_owner is query and query.snapshot_owner is snapshot
    assert result.cpu_execution['worker_slots'] == min(workers, 2)
    assert result.cpu_execution['launched_threads'] == min(workers, 2) - 1
    assert result.cpu_execution['tiles'] == result.cpu_execution['completed_tiles'] == 2
    all_of = ({'coordinate': (0,), 'operation': 'eq', 'literal': zero},)
    equal = execute(select(snapshot, kind='vector', all_of=all_of))
    assert equal.keys == tuple(expected['zero_vectors']) == (11,)
    other = execute(select(snapshot, kind='vector', all_of=({**all_of[0], 'operation': 'ne'},)))
    assert other.keys == (10,)
    assert equal.record(0).coefficients == archive.record(11).coefficients
    assert all(q == 0 for row in equal.record(0).coefficients for q in row)
    literal = archive.scalar(archive.record(10).coefficients[0])
    chained = select(snapshot, kind='vector', all_of=(all_of[0], {**all_of[0], 'literal': literal}))
    assert execute(chained).keys == ()
    result = execute(select(snapshot, kind='map', source_carrier=archive.record(90), target_carrier=archive.record(90)))
    assert result.keys == (90,)


def test_occurrence_views_retain_inactive_underlying_maps_and_exact_contribution(archive_case, query_oracle):
    archive, oracle = archive_case
    snapshot = publication(archive)
    occurrences = query_oracle[oracle['profile']]['occurrences']
    observed = [snapshot.occurrence(1, i) for i in range(len(occurrences))]
    assert len({v['key'] for v in observed}) == len(occurrences)
    assert any(not v['active'] for v in observed)
    for value, expected in zip(observed, occurrences, strict=True):
        assert value['key'] == expected['key'] and value['active'] == expected['active']
        assert value['coefficient'] == Fraction(expected['coefficient'])
        for kind in ('underlying', 'weighted', 'contribution'):
            assert value[kind].coefficients == exact_rows(expected[kind])
            assert value[kind].archive_owner is archive
    with pytest.raises(IndexError): snapshot.occurrence(3, 0)


def test_rectangular_windows_keep_supported_zero_absence_and_declared_indices(archive_case, query_oracle):
    archive, oracle = archive_case
    snapshot = publication(archive)
    window = box(snapshot)
    expected = query_oracle[oracle['profile']]
    assert window.domain == (2, 2) and window.supported_count == 4
    assert window.preview()['coefficients'] == exact_rows(expected['window'])
    assert window.preview()['source_indices'] == tuple(expected['window_indices'])
    assert window.sum(availability=AVAILABILITY).coefficients == exact_rows(expected['sum'])
    for i, source in enumerate(expected['window_indices']):
        assert window.source_index(i) == source and window.window_index(source) == i
        assert window.coordinate(i).coefficients[0] == window.record_owner.coefficients[source]
    assert window.window_index(0) is None
    assert window.preview(1)['truncated'] and len(window.preview(1)['coefficients']) == 1
    zero = box(snapshot, key=11, lower=(0,), upper=(2,))
    empty = box(snapshot, lower=(1, 0), upper=(1, 2))
    assert zero.supported_count == 2 and zero.sum(availability=AVAILABILITY).kind == 'scalar'
    assert all(q == 0 for q in zero.sum(availability=AVAILABILITY).coefficients[0])
    assert empty.supported_count == 0 and empty.domain == (0, 2)
    assert empty.sum(availability=AVAILABILITY) is None and empty.preview()['coefficients'] == ()
    with pytest.raises(IndexError): empty.coordinate(0)
    with pytest.raises(IndexError): window.window_index(6)
    del snapshot, archive
    gc.collect()
    assert len(window.coordinate(0).coefficients[0]) == window.record_owner.snapshot_owner.archive_owner.field_degree


@pytest.mark.parametrize('owner', ['source', 'target', 'literal', 'literal_kind'])
def test_preparation_rejects_foreign_equal_values_or_wrong_scalar_kind(archive_case, owner):
    archive, _ = archive_case
    snapshot = publication(archive)
    foreign = NativeAlgebraicArchive(archive.archive)
    options = {'kind': 'vector'}
    if owner in {'source', 'target'}: options[owner + '_carrier'] = foreign.record(10)
    else:
        literal = foreign.scalar([0] * foreign.field_degree) if owner == 'literal' else archive.record(10)
        options['all_of'] = ({'coordinate': (0,), 'operation': 'eq', 'literal': literal},)
    with pytest.raises(ValueError, match='owner|scalar'):
        select(snapshot, **options)


def test_graph_views_follow_shared_owners_and_reject_independent_presentations(graph):
    a, alias, foreign = graph.archive(1), graph.archive(4), graph.archive(5)
    snapshot = publication(a)
    scalar = alias.scalar(a.record(10).coefficients[0])
    query = select(snapshot, kind='vector', source_carrier=alias.record(10),
                   all_of=({'coordinate': (0,), 'operation': 'eq', 'literal': scalar},))
    assert execute(query).keys == (10,)
    with pytest.raises(ValueError, match='owner'):
        select(snapshot, kind='vector', source_carrier=foreign.record(10))
    with pytest.raises(ValueError, match='owner'):
        select(snapshot, kind='vector', all_of=({'coordinate': (0,), 'operation': 'eq',
                                               'literal': foreign.scalar(a.record(10).coefficients[0])},))
    record = algebraic_query_record(execute(query))
    restored = read_algebraic_snapshot_record(record)
    assert restored.archive_owner.graph_key == 1
    assert restored.archive_owner.graph_owner.image == graph.image
    assert not restored.same_publication(snapshot)
    observation = read_algebraic_observation_record(record)
    assert observation['publication']['source']['kind'] == 'cpp_algebraic_graph_source'


@pytest.mark.parametrize('operation', ['query_prepare', 'query_execute', 'window_prepare', 'window_sum', 'cache_hit'])
def test_cancellation_is_checked_at_each_native_entry(archive_case, operation):
    archive, _ = archive_case
    snapshot = publication(archive)
    token = CancellationToken(); token.cancel()
    query = select(snapshot, kind='vector')
    cache = NativeAlgebraicQueryCache(10)
    execute(query, cache=cache)
    window = box(snapshot)
    with pytest.raises(ExecutionCancelled):
        if operation == 'query_prepare': select(snapshot, cancellation=token)
        elif operation == 'query_execute': execute(query, cancellation=token)
        elif operation == 'cache_hit': execute(query, cancellation=token, cache=cache)
        elif operation == 'window_prepare': box(snapshot, cancellation=token)
        else: window.sum(availability=AVAILABILITY, cancellation=token)
    assert cache.statistics['applications'] == 1 and cache.statistics['hits'] == 0


@pytest.mark.parametrize('cutoff', [(4001, 4), (4002, 5), (0, 5), (True, 5), (4001, True)])
def test_cutoffs_are_rechecked_for_windows_and_cache_hits(archive_case, cutoff):
    archive, _ = archive_case
    snapshot = publication(archive)
    query = select(snapshot, kind='vector')
    cache = NativeAlgebraicQueryCache(10)
    execute(query, cache=cache)
    window = box(snapshot)
    for operation in (lambda: query.execute(availability=cutoff, cache=cache),
                      lambda: window.sum(availability=cutoff),
                      lambda: PreparedAlgebraicQuery(snapshot, availability=cutoff),
                      lambda: NativeAlgebraicTensorWindow(snapshot.record(90), (0, 0), (3, 2), availability=cutoff)):
        with pytest.raises((ValueError, TypeError)):
            operation()
    assert cache.statistics['hits'] == 0


def test_exact_execution_requirements_admit_at_the_boundary_and_reject_below(archive_case):
    archive, _ = archive_case
    snapshot = publication(archive)
    literal = archive.scalar(archive.record(10).coefficients[0])
    clauses = ({'coordinate': (0,), 'operation': 'eq', 'literal': literal},)
    query = select(snapshot, kind='vector', all_of=clauses)
    for name, required in query.requirements.items():
        execute(query, execution_limits={name: required})
        if required:
            with pytest.raises(ExecutionLimitError): execute(query, execution_limits={name: required - 1})
    window = box(snapshot)
    for name, required in window.requirements.items():
        window.sum(availability=AVAILABILITY, execution_limits={name: required})
        if required:
            with pytest.raises(ExecutionLimitError):
                window.sum(availability=AVAILABILITY, execution_limits={name: required - 1})
    with pytest.raises(ExecutionLimitError): select(snapshot, execution_limits={'records': 1})
    with pytest.raises(ExecutionLimitError): box(snapshot, execution_limits={'records': 1})
    with pytest.raises(ExecutionLimitError): execute(query, cpu_workers=2, worker_limit=1)
    with pytest.raises(ExecutionLimitError): execute(query, tile_limit=0)


def test_actual_prepared_owner_controls_cache_hits_limits_clear_and_escaped_pins(archive_case):
    archive, _ = archive_case
    snapshot = publication(archive)
    query = select(snapshot, kind='vector', rows_per_tile=1)
    cache = NativeAlgebraicQueryCache(2)
    first = execute(query, cache=cache)
    hit = query.execute(availability=(4001, 8), cache=cache)
    assert hit.keys == first.keys and hit.cache_hit and hit.availability == (4001, 8)
    assert set(hit.cpu_execution.values()) == {0}
    equal = select(snapshot, kind='vector', rows_per_tile=1)
    assert not execute(equal, cache=cache).cache_hit
    assert cache.statistics == {'applications': 2, 'hits': 1, 'max_records': 2}
    below = dict(query.requirements, records=query.requirements['records'] - 1)
    with pytest.raises(ExecutionLimitError): execute(equal, cache=cache, execution_limits=below)
    cache.clear()
    assert not execute(query, cache=cache).cache_hit
    limited = NativeAlgebraicQueryCache(1)
    execute(query, cache=limited); assert not execute(query, cache=limited).cache_hit
    assert limited.statistics['applications'] == 2 and limited.statistics['hits'] == 0
    pin = first.record(1)
    del query, equal, snapshot, archive, first, hit, cache
    gc.collect()
    assert pin.key == 11 and all(q == 0 for row in pin.coefficients for q in row)


def test_concurrent_query_execution_retains_ordered_results_and_one_cache_owner(archive_case):
    archive, _ = archive_case
    query = select(publication(archive), kind='vector', rows_per_tile=1)
    cache = NativeAlgebraicQueryCache(2)
    with ThreadPoolExecutor(max_workers=4) as workers:
        outputs = list(workers.map(lambda _: execute(query, cpu_workers=2, cache=cache), range(12)))
    assert all(o.keys == (10, 11) for o in outputs)
    assert sum(not o.cache_hit for o in outputs) == 1
    assert cache.statistics == {'applications': 1, 'hits': 11, 'max_records': 2}
    assert all(o.record(0).snapshot_owner is query.snapshot_owner for o in outputs)


@pytest.mark.parametrize('declaration', [
    {'kind': 'scalar'}, {'rows_per_tile': 0}, {'rows_per_tile': True}, {'execution_limits': {'records': True}},
    {'all_of': ({'coordinate': (0,), 'operation': 'lt', 'literal': None},)},
    {'all_of': ({'coordinate': (True,), 'operation': 'eq', 'literal': None},)},
    {'all_of': ({'coordinate': (2,), 'operation': 'eq', 'literal': None},)},
    {'all_of': ({'coordinate': (0, 0), 'operation': 'eq', 'literal': None},)},
    {'all_of': ({'coordinate': (0,), 'operation': 'eq', 'literal': (0,)},)},
])
def test_invalid_exact_query_declarations_reject(archive_case, declaration):
    archive, _ = archive_case
    options = deepcopy(declaration)
    if 'all_of' in options and options['all_of'][0]['literal'] is None:
        options['all_of'][0]['literal'] = archive.scalar([0] * archive.field_degree)
    with pytest.raises((ValueError, TypeError, IndexError)):
        select(publication(archive), **dict({'kind': 'vector'}, **options))


@pytest.mark.parametrize('lower,upper', [((0,), (2,)), ((0, 0), (4, 2)), ((2, 1), (1, 2)), ((False, 0), (3, 2))])
def test_window_bounds_and_rank_are_explicit(archive_case, lower, upper):
    with pytest.raises((ValueError, TypeError, IndexError)):
        box(publication(archive_case[0]), lower=lower, upper=upper)


def test_uninitialized_and_reinitialized_handles_cannot_alias_old_publications(archive_case):
    archive, _ = archive_case
    snapshot = publication(archive)
    query = select(snapshot)
    window = box(snapshot)
    cache = NativeAlgebraicQueryCache(2)
    for operation in (lambda: snapshot.__init__(archive, namespace=7, sequence=0),
                      lambda: query.__init__(snapshot, availability=AVAILABILITY),
                      lambda: window.__init__(snapshot.record(90), (0, 0), (3, 2), availability=AVAILABILITY),
                      lambda: cache.__init__(3)):
        with pytest.raises(RuntimeError, match='immutable'): operation()
    for cls, name in ((NativeAlgebraicSnapshot, 'namespace'), (NativeAlgebraicRecord, 'domain'),
                      (NativeAlgebraicTensorWindow, 'supported_count'), (PreparedAlgebraicQuery, 'candidate_count'),
                      (NativeAlgebraicQueryResult, 'keys'), (NativeAlgebraicQueryCache, 'statistics')):
        with pytest.raises(RuntimeError, match='initialized'): getattr(cls.__new__(cls), name)


@pytest.mark.parametrize('kind', ['query', 'window', 'zero', 'empty'])
def test_complete_exact_observations_roundtrip_through_core_records(archive_case, kind):
    archive, _ = archive_case
    snapshot = publication(archive)
    if kind == 'query': record = algebraic_query_record(execute(select(snapshot, kind='vector')))
    else:
        window = box(snapshot) if kind == 'window' else box(snapshot, key=11, lower=(0,), upper=(2,) if kind == 'zero' else (0,))
        record = algebraic_window_record(window, availability=AVAILABILITY)
    expected = read_algebraic_observation_record(record)
    for restored in (from_state(to_state(record)),):
        assert read_algebraic_observation_record(restored) == expected
        current = read_algebraic_snapshot_record(restored)
        assert current.archive_owner.manifest == archive.manifest
        assert current.namespace == 4001 and current.sequence == 5
        assert not current.same_publication(snapshot)
    assert unpack_value(pack_value(expected)) == expected
    with pytest.raises(TypeError, match="structured cell attribute"):
        to_state(record, _native=False)
    if kind == 'empty':
        assert read_algebraic_observation_record(record, max_coordinates=0) == expected
    else:
        with pytest.raises(ValueError): read_algebraic_observation_record(record, max_coordinates=0)
    with pytest.raises(ValueError): read_algebraic_observation_record(record, max_archive_bytes=1)
    with pytest.raises((ValueError, RuntimeError)):
        read_algebraic_snapshot_record(record, limits={'max_file_bytes': len(archive.archive) - 1})


@pytest.mark.parametrize('kind', ['query', 'window'])
@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_observations_use_existing_whole_record_user_formats(archive_case, tmp_path, kind, suffix):
    from rexgraph.io import load, save
    snapshot = publication(archive_case[0])
    record = algebraic_query_record(execute(select(snapshot, kind='vector'))) if kind == 'query' else algebraic_window_record(box(snapshot), availability=AVAILABILITY)
    expected = read_algebraic_observation_record(record)
    path = tmp_path / ('observation' + suffix)
    save(path, record)
    restored = load(path)
    assert read_algebraic_observation_record(restored) == expected
    source = read_algebraic_snapshot_record(restored)
    assert source.archive_owner.archive == snapshot.archive_owner.archive


@pytest.mark.parametrize('mutation', ['float', 'basis', 'cutoff', 'namespace', 'shape', 'policy', 'predicate', 'sum', 'indices', 'bool', 'seal'])
def test_observation_readers_reject_resealed_inconsistent_metadata(archive_case, mutation):
    snapshot = publication(archive_case[0])
    use_query = mutation in {'float', 'basis', 'cutoff', 'namespace', 'shape', 'policy', 'predicate', 'bool', 'seal'}
    value = algebraic_query_observation(execute(select(snapshot, kind='vector'))) if use_query else algebraic_window_observation(box(snapshot), availability=AVAILABILITY)
    value = deepcopy(value)
    if mutation == 'float': value['records'][0]['coefficients'] = ((0.0,) * snapshot.archive_owner.field_degree,) * 2
    elif mutation == 'basis': value['coefficient_basis'] = 'embedding'
    elif mutation == 'cutoff': value['availability'] = (4001, 4)
    elif mutation == 'namespace': value['query']['namespace'] = 4002
    elif mutation == 'shape': value['records'][0]['domain'] = (3,)
    elif mutation == 'policy': value['query']['execution_limits']['records'] = 0
    elif mutation == 'predicate':
        value['query']['all_of'] = ({'coordinate': (0,), 'operation': 'eq', 'literal': (Fraction(0),) * snapshot.archive_owner.field_degree},)
        value['query']['predicate_count'] = 1
    elif mutation == 'sum': value['sum'] = (Fraction(0),) * snapshot.archive_owner.field_degree
    elif mutation == 'indices': value['window']['source_indices'] = (0, 1, 2, 3)
    elif mutation == 'bool': value['records'][0]['shape'] = (2, True)
    else: value['seal'] = '0' * 64
    if mutation != 'seal': reseal(value)
    with pytest.raises((ValueError, TypeError)):
        validate_algebraic_observation(value)


def test_offline_record_reader_checks_current_attachment_bytes_and_returns_owned_metadata(archive_case):
    snapshot = publication(archive_case[0])
    record = algebraic_query_record(execute(select(snapshot, kind='vector')))
    current = read_algebraic_observation_record(record)
    current['records'][0]['key'] = 999
    assert read_algebraic_observation_record(record)['records'][0]['key'] == 10
    attachment = record.get_metadata(1, 0, 'algebraic_archive')
    attachment['archive'][0] ^= np.uint8(1)
    with pytest.raises(ValueError, match='bytes'): read_algebraic_observation_record(record)


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_rcdb_commit_courier_and_fresh_process_retain_exact_query_observation(archive_case, tmp_path, backend):
    from rcdb import MemoryStore, copy_record, open_store
    snapshot = publication(archive_case[0])
    record = algebraic_query_record(execute(select(snapshot, kind='vector')))
    expected = read_algebraic_observation_record(record)
    url = f'sqlite:///{tmp_path}/query.sqlite' if backend == 'sqlite' else f'{backend}://{tmp_path}/queries'
    store = open_store(url, **({} if backend == 'sqlite' else {'read_only': False})).configure_security(require_commits=True)
    try:
        store.commit_mutation('query', record, expected_version=0)
        assert store.verify_commits('query')
        target = MemoryStore()
        try:
            receipt = copy_record(store, target, store.get_record('query'), return_receipt=True)
            assert receipt.source_digest == receipt.destination_digest
            copied = target.get_version('query', 1)
            assert read_algebraic_observation_record(copied) == expected
        finally: target.close()
    finally: store.close()
    code = '''
import importlib.abc,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('exact query imported Torch')
sys.meta_path.insert(0,NoTorch())
from rcdb import open_store
from rexgraph.cpp_algebraic_query import read_algebraic_observation_record,read_algebraic_snapshot_record
from rcql import Executor,parse
store=open_store(sys.argv[1])
try:
 record=store.get_version('query',1)
 observed=read_algebraic_observation_record(record)
 snapshot=read_algebraic_snapshot_record(record)
 queried=Executor(sources={'record':record},params={'request':{'kind':'vector'}}).execute(parse('FROM $record RETURN ALGEBRAIC_QUERY($request,5)'))
 assert queried.values[0]['seal']==observed['seal']
 assert queried.exactness[0].value=='algebraic'
 assert snapshot.sequence==5 and snapshot.namespace==4001
 print(json.dumps({'seal':observed['seal']}))
finally:store.close()
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, url], check=True, text=True, capture_output=True, cwd=tmp_path)
    assert json.loads(result.stdout) == {'seal': expected['seal']}


def test_core_observations_decode_without_optional_native_extension_or_torch(archive_case, tmp_path):
    from rexgraph.io.bundle import save_rcbd
    record = algebraic_query_record(execute(select(publication(archive_case[0]), kind='vector')))
    path = tmp_path / 'query.rcbd'
    save_rcbd(path, record)
    code = '''
import importlib.abc,json,sys
class NoOptional(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','rexgraph_cpp'}:raise AssertionError('observation decoding loaded optional runtime')
sys.meta_path.insert(0,NoOptional())
from rexgraph.io.bundle import load_rcbd
from rexgraph.cpp_algebraic_query import read_algebraic_observation_record
value=read_algebraic_observation_record(load_rcbd(sys.argv[1]))
assert value['arithmetic']=='algebraic' and value['coefficient_basis']=='power'
print(json.dumps({'seal':value['seal']}))
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(path)], check=True, text=True, capture_output=True, cwd=tmp_path)
    assert json.loads(result.stdout)['seal'] == read_algebraic_observation_record(record)['seal']


def test_rcql_agent_and_system_share_the_same_snapshot_query_and_window_results(archive_case):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    snapshot = publication(archive_case[0])
    record = algebraic_snapshot_record(snapshot)
    request = {'kind': 'vector'}
    expected = algebraic_query_observation(execute(prepare_algebraic_query(snapshot, request, availability=AVAILABILITY)))
    text = 'FROM $record RETURN ALGEBRAIC_QUERY($request,5), ALGEBRAIC_WINDOW(90,[1,0],[3,2],5)'
    result = Executor(sources={'record': record}, params={'request': request}).execute(parse(text))
    assert result.values[0] == expected
    assert result.values[1] == algebraic_window_observation(box(snapshot), availability=AVAILABILITY)
    assert all(item.value == 'algebraic' for item in result.exactness)
    hive = Hive('algebraic-query')
    hive.add_worker('query', lambda declaration: algebraic_query_observation(execute(prepare_algebraic_query(snapshot, declaration, availability=AVAILABILITY))), capability='analyze')
    assert hive.invoke('query', request, record=False) == expected
    runtime = RCQLRuntime()
    runtime.register('record', record, policy=SourcePolicy.allow('read'))
    assert runtime.execute(parse('FROM $record RETURN ALGEBRAIC_QUERY($request,5)'), params={'request': request}).values[0] == expected
    from rcql.capabilities import BoundSource
    denied = BoundSource(record, SourcePolicy.allow())
    with pytest.raises(PermissionError):
        Executor(sources={'record': denied}, params={'request': request}).execute(parse('FROM $record RETURN ALGEBRAIC_QUERY($request,5)'))
    for value in (snapshot, snapshot.record(10), box(snapshot), select(snapshot), execute(select(snapshot))):
        shown = json_value(value, max_values=1)
        assert shown['kind'] == type(value).__name__
        json.dumps(shown, allow_nan=False)


def test_observations_retain_the_effective_execution_policy(archive_case):
    query = select(publication(archive_case[0]), kind='vector')
    result = execute(query, execution_limits=query.requirements, cpu_workers=2, worker_limit=3)
    assert result.execution_limits == query.requirements
    assert result.cpu_policy == {'workers': 2, 'worker_limit': 3, 'tile_limit': 1000000}
    observed = algebraic_query_observation(result)
    assert observed['query']['execution_limits'] == query.requirements
    validate_algebraic_observation(observed)
