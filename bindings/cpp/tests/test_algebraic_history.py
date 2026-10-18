"""Native channel cutoffs, complete prefix records and shared application routes."""
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from rexgraph_cpp import (
    NativeAlgebraicArchive,
    NativeAlgebraicSnapshot,
    NativeAlgebraicTensorWindow,
    NativeSource,
)
from test_algebraic_archive import algebraic_fixtures as algebraic_fixtures

from rexgraph.cpp_algebraic_history import (
    algebraic_history_at,
    algebraic_history_record,
    algebraic_history_reference,
)
from rexgraph.cpp_algebraic_query import (
    algebraic_query_record,
    algebraic_snapshot_record,
    algebraic_window_record,
    prepare_algebraic_query,
    read_algebraic_observation_record,
    read_algebraic_snapshot_record,
    validate_algebraic_observation,
)
from rexgraph.state import from_state, to_state
from rexgraph.value_codec import pack_value

CHANNEL = (701, 1, 100)


@pytest.fixture(scope='session')
def history_fixtures(algebraic_fixtures, tmp_path_factory):
    folder = tmp_path_factory.mktemp('algebraic-history')
    executable = os.environ.get('REXGRAPH_CPP_HISTORY_FIXTURE')
    if executable is None:
        build = folder/'build'
        subprocess.run(['cmake', '-S', str(Path(__file__).resolve().parents[1]), '-B', str(build),
                        '-DREXGRAPH_BINDING_TEST_FIXTURE=ON', f'-DPython3_EXECUTABLE={sys.executable}'],
                       check=True, capture_output=True, text=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'algebraic_history_fixture', '-j1'],
                       check=True, capture_output=True, text=True)
        executable = build/'algebraic_history_fixture'
    subprocess.run([str(executable), str(algebraic_fixtures[0]), str(folder)], check=True, capture_output=True, text=True)
    return folder


@pytest.fixture(params=['rational', 'real_cubic', 'complex_quadratic', 'complex_quartic'])
def history_case(request, history_fixtures, algebraic_fixtures):
    name = request.param
    raw = (history_fixtures/(name+'-42.rfcxs')).read_bytes()
    return NativeSource(raw, max_bytes=len(raw)), NativeAlgebraicArchive((algebraic_fixtures[0]/(name+'.rfcxa')).read_bytes()), name


def snapshot(source, sequence=41):
    return NativeAlgebraicSnapshot.from_history(source, CHANNEL, sequence=sequence)


def observed_record(source, sequence=41, kind='query'):
    selected = snapshot(source, sequence)
    availability = (9001, sequence)
    if kind == 'query':
        result = prepare_algebraic_query(selected, {'kind': 'vector'}, availability=availability).execute(availability=availability)
        return algebraic_query_record(result)
    window = NativeAlgebraicTensorWindow(selected.record(90), (1, 0), (3, 2), availability=availability)
    return algebraic_window_record(window, availability=availability)


@pytest.mark.parametrize('sequence,offset', [(40, 0), (41, 7), (42, 99)])
def test_selected_prefix_and_archive_values_match_retained_native_revision(history_case, history_fixtures, sequence, offset):
    source, base, name = history_case
    selected = snapshot(source, sequence)
    owner = selected.archive_owner
    assert selected.namespace == 9001 and selected.sequence == sequence
    assert owner.history_owner.identity['sequence'] == sequence and owner.history_channel == CHANNEL
    assert owner.history_owner.archive == (history_fixtures/f'{name}-{sequence}.rfcxs').read_bytes()
    assert owner.history_owner.max_bytes == source.max_bytes
    expected = [list(row) for row in base.record(10).coefficients]
    expected[0][0] += Fraction(offset)
    assert selected.record(10).coefficients == tuple(map(tuple, expected))
    assert selected.preview(0)['native_history'] is True
    assert selected.preview(0)['history']['sequence'] == sequence
    assert owner.source_rfcx == base.source_rfcx
    assert owner.manifest['source_sizes'] == (3, 3, 2, 0)
    if sequence < 42:
        with pytest.raises((IndexError, ValueError)): owner.history_owner.at(sequence+1)
    assert NativeSource(owner.history_owner.archive).identity == owner.history_owner.identity
    with pytest.raises(ValueError, match='owner'): base.record(30).apply(selected.record(10).value)
    owner.record(30).apply(selected.record(10).value)


def test_secondary_channel_and_independent_restores_do_not_alias(history_case):
    source, base, _ = history_case
    first, second = snapshot(source), snapshot(source)
    original = NativeAlgebraicArchive.from_history(source, (701, 1, 101), sequence=41)
    assert original.archive == base.archive
    assert first.archive_owner.archive == second.archive_owner.archive
    with pytest.raises(ValueError, match='owner'): first.archive_owner.record(30).apply(second.record(10).value)
    with pytest.raises(ValueError, match='endpoint'): NativeAlgebraicSnapshot(first.archive_owner, namespace=9002, sequence=41)
    with pytest.raises(ValueError, match='endpoint'): NativeAlgebraicSnapshot(first.archive_owner, namespace=9001, sequence=42)
    assert algebraic_history_reference(base) is None


@pytest.mark.parametrize('channel', [(0, 1, 100), (701, 0, 100), (701, 1, 0), (True, 1, 100),
                                     (701.0, 1, 100), (701, 1), (701, 1, 100, 2), '701', {},
                                     (702, 1, 100), (701, 2, 100), (701, 1, 999), (701, 1, 200), (701, 1, 201)])
def test_wrong_or_opaque_channels_reject_without_changing_source(history_case, channel):
    source = history_case[0]
    before = source.identity
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        NativeAlgebraicArchive.from_history(source, channel, sequence=41)
    assert source.identity == before


@pytest.mark.parametrize('sequence', [39, 43, True, 40.0, -1, 1 << 64])
def test_cutoff_is_an_exact_retained_native_sequence(history_case, sequence):
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        snapshot(history_case[0], sequence)


@pytest.mark.parametrize('suffix', ['legacy', 'foreign'])
def test_legacy_integer_channel_and_foreign_source_reject(history_case, history_fixtures, suffix):
    _, _, name = history_case
    source = NativeSource((history_fixtures/f'{name}-{suffix}.rfcxs').read_bytes())
    with pytest.raises((ValueError, IndexError)): snapshot(source, source.identity['sequence'])


@pytest.mark.parametrize('limits', [{'max_records': 0}, {'field.max_degree': 0}, {'max_integer_bytes': 1}])
def test_canonical_archive_admission_is_applied_to_history_payload(history_case, limits):
    with pytest.raises((ValueError, RuntimeError)):
        NativeAlgebraicArchive.from_history(history_case[0], CHANNEL, sequence=41, limits=limits)


def test_history_records_keep_prefix_and_effective_reader_limits(history_case):
    source = history_case[0]
    record = algebraic_history_record(source, CHANNEL, sequence=41)
    selected = read_algebraic_snapshot_record(record)
    raw = selected.archive_owner.history_owner.archive
    restored = read_algebraic_snapshot_record(record, history_max_bytes=len(raw))
    assert restored.archive_owner.history_owner.max_bytes == len(raw)
    saved = algebraic_snapshot_record(restored)
    assert saved.get_metadata(1, 0, 'algebraic_snapshot')['publication']['history']['max_bytes'] == len(raw)
    with pytest.raises(ValueError): read_algebraic_snapshot_record(record, history_max_bytes=len(raw)-1)
    with pytest.raises((ValueError, RuntimeError)):
        read_algebraic_snapshot_record(record, limits={'max_records': 0})
    old = read_algebraic_snapshot_record(algebraic_history_at(record, 40))
    assert old.sequence == 40 and old.archive_owner.history_owner.identity['sequence'] == 40
    assert len(old.archive_owner.history_owner.archive) < len(raw)
    with pytest.raises((ValueError, IndexError)): algebraic_history_at(record, 42)


@pytest.mark.parametrize('kind', ['query', 'window'])
@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_complete_history_observations_roundtrip_existing_user_formats(history_case, tmp_path, kind, suffix):
    from rexgraph.io import load, save
    record = observed_record(history_case[0], kind=kind)
    expected = read_algebraic_observation_record(record)
    assert expected['publication']['history']['source']['native']['sequence'] == 41
    path = tmp_path/('history'+suffix)
    save(path, record)
    for current in [from_state(to_state(record)), load(path)]:
        assert read_algebraic_observation_record(current) == expected
        restored = read_algebraic_snapshot_record(current)
        assert restored.archive_owner.history_owner.identity['sequence'] == 41
        if kind == 'query':
            assert read_algebraic_observation_record(observed_record(restored.archive_owner.history_owner)) == expected
    with pytest.raises(ValueError): read_algebraic_observation_record(record, max_history_bytes=1)


@pytest.mark.parametrize('mutation', ['version', 'profile', 'namespace', 'sequence', 'max_bytes', 'channel', 'prefix', 'extra'])
def test_history_observation_receipts_reject_inconsistent_declarations(history_case, mutation):
    record = observed_record(history_case[0])
    value = deepcopy(read_algebraic_observation_record(record))
    history = value['publication']['history']
    if mutation == 'version': history['version'] = True
    elif mutation == 'profile': history['profile'] = 'native-algebraic-channel-v1'
    elif mutation in {'namespace', 'sequence'}: history['source']['native'][mutation] += 1
    elif mutation == 'max_bytes': history['max_bytes'] = True
    elif mutation == 'channel': history['channel'] = (701, 1, False)
    elif mutation == 'prefix': history['source']['native']['first_sequence'] = 42
    else: history['extra'] = None
    value['seal'] = __import__('hashlib').sha256(pack_value({k: v for k, v in value.items() if k != 'seal'})).hexdigest()
    with pytest.raises((ValueError, TypeError)): validate_algebraic_observation(value)


def test_current_prefix_bytes_are_checked_even_with_an_owning_restore_cache(history_case):
    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
    record = observed_record(history_case[0])
    cache = _AlgebraicRestoreCache()
    first = read_algebraic_snapshot_record(record, _archive_cache=cache)
    second = read_algebraic_snapshot_record(record, _archive_cache=cache)
    assert first.archive_owner is second.archive_owner
    assert cache._bytes == len(first.archive_owner.archive)+len(first.archive_owner.history_owner.archive)
    other = read_algebraic_snapshot_record(from_state(to_state(record)), _archive_cache=cache)
    assert other.archive_owner is not first.archive_owner
    record.get_metadata(1, 0, 'algebraic_history')['archive'][0] ^= np.uint8(1)
    with pytest.raises(ValueError, match='bytes'): read_algebraic_snapshot_record(record, _archive_cache=cache)
    with pytest.raises(ValueError, match='bytes'): read_algebraic_observation_record(record)


def test_native_restore_rechecks_channel_against_retained_archive_image(history_case):
    record = algebraic_history_record(history_case[0], CHANNEL, sequence=41)
    publication = record.get_metadata(1, 0, 'algebraic_snapshot')['publication']
    changed = deepcopy(publication['history']);changed['channel'] = (701, 1, 101)
    publication['history'] = changed
    record.get_metadata(1, 0, 'algebraic_history')['reference'] = changed
    with pytest.raises(ValueError, match='image'): read_algebraic_snapshot_record(record)


def test_pins_survive_owner_collection_and_concurrent_record_restore(history_case):
    source = history_case[0]
    record = observed_record(source)
    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
    cache = _AlgebraicRestoreCache()
    with ThreadPoolExecutor(max_workers=4) as pool:
        snapshots = list(pool.map(lambda _: read_algebraic_snapshot_record(record, _archive_cache=cache), range(12)))
    pin = snapshots[0].record(10)
    assert all(value.archive_owner is pin.snapshot_owner.archive_owner for value in snapshots)
    del snapshots, source
    gc.collect()
    assert pin.snapshot_owner.archive_owner.history_owner.identity['sequence'] == 41
    assert pin.coordinate((0,)).coefficients[0] == pin.coefficients[0]


def test_multi_chunk_native_archive_retains_all_named_records(history_fixtures, algebraic_fixtures):
    source = NativeSource((history_fixtures/'large.rfcxs').read_bytes())
    selected = snapshot(source, 40)
    assert len(selected.archive_owner.archive) > 4096
    base = NativeAlgebraicArchive((algebraic_fixtures[0]/'complex_quartic.rfcxa').read_bytes())
    assert selected.archive_owner.record_count == base.record_count + 128
    record = observed_record(source, 40)
    value = read_algebraic_observation_record(record)
    assert len(value['records']) == 130
    restored = read_algebraic_snapshot_record(record)
    assert restored.archive_owner.archive == selected.archive_owner.archive
    assert [item['key'] for item in value['records']][-1] == 1127


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_committed_rcdb_copy_and_fresh_process_retain_selected_history(history_case, tmp_path, backend):
    from rcdb import MemoryStore, copy_record, open_store
    record = observed_record(history_case[0])
    expected = read_algebraic_observation_record(record)
    url = f'sqlite:///{tmp_path}/history.sqlite' if backend == 'sqlite' else f'{backend}://{tmp_path}/history'
    store = open_store(url, **({} if backend == 'sqlite' else {'read_only': False})).configure_security(require_commits=True)
    try:
        store.commit_mutation('history', record, expected_version=0)
        assert store.verify_commits('history')
        target = MemoryStore()
        try:
            receipt = copy_record(store, target, store.get_record('history'), return_receipt=True)
            assert receipt.source_digest == receipt.destination_digest
            assert read_algebraic_observation_record(target.get_version('history', 1)) == expected
        finally: target.close()
    finally: store.close()
    code = '''
import importlib.abc,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='torch':raise AssertionError('native history imported Torch')
sys.meta_path.insert(0,NoTorch())
from rcdb import open_store
from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record,read_algebraic_observation_record
from rexgraph.cpp_algebraic_history import algebraic_history_at
from rcql import Executor,parse
store=open_store(sys.argv[1])
try:
 record=store.get_version('history',1)
 value=read_algebraic_observation_record(record)
 source=read_algebraic_snapshot_record(record)
 assert source.archive_owner.history_owner.identity['sequence']==41
 assert read_algebraic_snapshot_record(algebraic_history_at(record,40)).sequence==40
 result=Executor(sources={'record':record},params={'request':{'kind':'vector'}}).execute(parse('FROM $record RETURN ALGEBRAIC_QUERY($request,41)'))
 assert result.values[0]==value
 print(json.dumps({'seal':value['seal']}))
finally:store.close()
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, url], check=True, capture_output=True, text=True, cwd=tmp_path)
    assert json.loads(result.stdout) == {'seal': expected['seal']}


def test_rcql_agent_and_system_preserve_actual_history_selection(history_case):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from rcql.capabilities import BoundSource
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    record = algebraic_history_record(history_case[0], CHANNEL, sequence=42)
    request = {'kind': 'vector'}
    selection = parse('FROM $record RETURN ALGEBRAIC_AT(40)')
    selected = Executor(sources={'record': record}).execute(selection).values[0]
    text = 'FROM $record RETURN ALGEBRAIC_QUERY($request,40), ALGEBRAIC_HISTORY_INFO()'
    result = Executor(sources={'record': selected}, params={'request': request}).execute(parse(text))
    assert result.values[0]['publication']['sequence'] == 40
    assert result.values[1]['source']['native']['sequence'] == 40
    runtime = RCQLRuntime();runtime.register('record', record, policy=SourcePolicy.allow('read'))
    selected = runtime.execute(selection).values[0]
    runtime.register('selected', selected, policy=SourcePolicy.allow('read'))
    assert runtime.execute(parse(text.replace('$record', '$selected')), params={'request': request}).values == result.values
    hive = Hive('native-history')
    hive.add_worker('select', lambda sequence: algebraic_history_at(record, sequence), capability='transform')
    shown = hive.invoke('select', 40, record=False)
    assert read_algebraic_snapshot_record(shown).sequence == 40
    with pytest.raises(PermissionError):
        Executor(sources={'record': BoundSource(record, SourcePolicy.allow())}).execute(parse('FROM $record RETURN ALGEBRAIC_HISTORY_INFO()'))
    preview = json_value(read_algebraic_snapshot_record(record), max_values=1)
    assert preview['native_history'] is True
    json.dumps(preview, allow_nan=False)


def test_observations_decode_without_native_extension_or_torch(history_case, tmp_path):
    from rexgraph.io import save
    path = tmp_path/'history.rcbd'
    record = observed_record(history_case[0]);save(path, record)
    code = '''
import importlib.abc,json,sys
class NoOptional(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','rexgraph_cpp'}:raise AssertionError('history observation imported an optional runtime')
sys.meta_path.insert(0,NoOptional())
from rexgraph.io import load
from rexgraph.cpp_algebraic_query import read_algebraic_observation_record
from rexgraph.cpp_algebraic_history import validate_algebraic_history_reference
value=read_algebraic_observation_record(load(sys.argv[1]))
validate_algebraic_history_reference(value['publication']['history'],publication=value['publication'])
print(json.dumps({'seal':value['seal']}))
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(path)], check=True, capture_output=True, text=True, cwd=tmp_path)
    assert json.loads(result.stdout)['seal'] == read_algebraic_observation_record(record)['seal']
