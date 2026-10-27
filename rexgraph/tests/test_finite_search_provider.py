"""Finite projection models retain native owners and recheck publisher cutoffs."""
from dataclasses import replace

import pytest


@pytest.fixture(autouse=True)
def installed_native_backend():
    pytest.importorskip('rexgraph_cpp._bridge')


def fixture(publisher=None):
    pytest.importorskip('rexgraph_cpp._bridge')
    cpp = pytest.importorskip('rexgraph_cpp')
    from rexgraph.cpp_finite_search import cpp_finite_search_model, finite_search_query
    source = cpp.NativeSource.from_declaration(dict(profile='declared-role-source-v1', namespace=778001,
        semantic_scale=2, max_grade=1, role_aspects=('boundary',),
        types=({'name': 'relation', 'min_arity': 0, 'max_arity': 8},), witnesses=(11, 42, 99),
        relations=(), fields=(), metrics=(), carriers=()))
    asset = cpp.NativeInputUseAsset(source.archive, dict(profile='rexgraph.relational-input-use-asset.v1',
        key=11, channel=12, encoding='UTF-8', bytes=b'a / a', uses=(
            dict(key=21, cells=((0, 11), (0, 42), (0, 99)), occurrences=(), spans=((0, 1), (0, 0))),
            dict(key=22, cells=((0, 11),), occurrences=(), spans=((4, 5),)))))
    declaration = dict(profile='rexgraph.relational-finite-source.v1', key=31, revision=1,
        symbols=({'key': 101, 'definition': b'alternative'}, {'key': 102, 'definition': b'alternative'}),
        variables=tuple(dict(cell=(0, i), domain=(101, 102)) for i in range(3)),
        charts=({'key': 32, 'name': 'correlated field', 'patch': ((0, 0), (0, 1), (0, 2)),
                 'support': ((0, 0), (0, 1), (0, 2)), 'allowed': ((101, 101, 101), (102, 102, 102))},),
        bindings=({'key': 33, 'use': 21, 'cell': (0, 0)}, {'key': 34, 'use': 22, 'cell': (0, 0)},
                  {'key': 35, 'use': 21, 'cell': (0, 1)}, {'key': 36, 'use': 21, 'cell': (0, 2)}))
    limits = (4096, 16000000, 64000000, 16384 if publisher is None else 1 << 24)
    publisher = source if publisher is None else publisher
    origins = {
        41: cpp.NativeValueOrigin(41, publisher.archive, publisher.identity['sequence'], 'input-asset', asset.archive(), limits=limits),
        42: cpp.NativeValueOrigin(42, publisher.archive, publisher.identity['sequence'], 'evidence',
            cpp.finite_constraint_origin_payload(asset, declaration, 'observed'), limits=limits)}
    def context(remove=(), cutoff=None):
        return cpp.NativeAvailabilityContext(tuple(dict(origin=v, namespace=v.inspect()['namespace'],
            sequence=v.inspect()['sequence'] if cutoff is None else cutoff) for k, v in origins.items() if k not in remove))
    authority = context()
    available = cpp.PreparedAvailableFiniteSource(asset, declaration, 'observed', {'asset': 41, 'constraints': 42}, authority, limits=limits)
    projection = dict(key=51, revision=1, cells=((0, 0), (0, 2)), window=None, fixed=(), distinct_relations=())
    state = cpp_finite_search_model(source, available_source=available, projection=projection, execution_context=authority, limits=limits)
    return cpp, source, available, state, finite_search_query(state), context


def test_native_projection_model_uses_actual_owners_and_cold_recipe():
    cpp, source, available, state, query, context = fixture()
    from rexgraph.cpp_finite_search import (
        clear_finite_search_cache,
        finite_search_query,
        prepare_finite_search_model,
        read_finite_search_result,
    )
    from rexgraph.model_codec import pack_model, unpack_model
    from rexgraph.model_runtime import infer_model
    from rexgraph.value_codec import pack_value, unpack_value
    prepared = prepare_finite_search_model(state, source, execution_context=context())
    assert prepared.raw_prepared(context()).source().same_owner(available.raw())
    assert prepare_finite_search_model(state, source, execution_context=context()) is prepared
    output = infer_model(state, source, query, execution_context=context())
    result = read_finite_search_result(output, execution_context=context(), state=state)
    assert output.values.size == 0 and result['status'] == 'complete'
    assert result['image'].family().propagation().same_owner(prepared.raw_propagation(context()))
    assert result['image'].row(1).witness() == ((0, 0, 102), (0, 1, 102), (0, 2, 102))
    cold = unpack_model(unpack_value(pack_value(pack_model(output))))
    clear_finite_search_cache()
    replay = read_finite_search_result(cold, execution_context=context(), state=state)
    assert replay['image'].row(1).witness() == result['image'].row(1).witness()
    assert not replay['image'].family().source().same_owner(available.raw())
    restored = unpack_model(pack_model(state)); original = cpp.NativeSource(source.archive)
    assert infer_model(restored, original, finite_search_query(restored), execution_context=context()).field_archive['data'] == output.field_archive['data']


@pytest.mark.parametrize('key', (41, 42))
@pytest.mark.parametrize('route', ('prepare', 'infer', 'read', 'cold', 'raw'))
def test_warm_and_cold_routes_recheck_current_availability(key, route):
    _, source, _, state, query, context = fixture()
    from rexgraph.cpp_finite_search import (
        clear_finite_search_cache,
        prepare_finite_search_model,
        read_finite_search_result,
    )
    from rexgraph.model_runtime import infer_model
    output = infer_model(state, source, query, execution_context=context())
    prepared = prepare_finite_search_model(state, source, execution_context=context())
    denied = context(remove=(key,))
    if route == 'cold': clear_finite_search_cache()
    with pytest.raises(ValueError):
        if route == 'prepare': prepare_finite_search_model(state, source, execution_context=denied)
        elif route == 'infer': infer_model(state, source, query, execution_context=denied)
        elif route == 'raw': prepared.raw_family(denied)
        else: read_finite_search_result(output, execution_context=denied)


def test_changed_receipts_queries_and_ordinary_training_are_rejected():
    _, source, _, state, query, context = fixture()
    from rexgraph.cpp_finite_search import prepare_finite_search_model
    from rexgraph.model_contract import model_provider_inventory, resolve_model_provider
    from rexgraph.model_runtime import infer_model
    assert any(p['name'] == 'rexgraph.cpp.finite_search' for p in model_provider_inventory())
    with pytest.raises(NotImplementedError): resolve_model_provider(state, 'train')
    bad = replace(state, configuration={**state.configuration, 'engine_receipt': {'changed': True}})
    with pytest.raises(ValueError): prepare_finite_search_model(bad, source, execution_context=context())
    with pytest.raises(ValueError): infer_model(state, source, {**query, 'model_digest': '0' * 64}, execution_context=context())
    with pytest.raises((ValueError, TypeError)): infer_model(state, source, query)


@pytest.fixture
def publisher_history(tmp_path):
    import os
    import subprocess

    cpp = pytest.importorskip('rexgraph_cpp')
    executable = os.environ.get('REXGRAPH_CPP_TEST_FIXTURE')
    if executable is None: pytest.skip('native publisher history fixture writer required')
    subprocess.run([executable, str(tmp_path)], check=True, capture_output=True)
    return cpp.NativeSource((tmp_path / 'diagonal-history.rfcxs').read_bytes())


def test_actual_publisher_history_cutoff_is_rechecked_after_warm_execution(publisher_history):
    from rexgraph.cpp_finite_search import (
        clear_finite_search_cache,
        prepare_finite_search_model,
        read_finite_search_result,
    )
    from rexgraph.model_runtime import infer_model
    publisher = publisher_history
    assert publisher.identity['sequence'] > publisher.identity['first_sequence']
    _, source, _, state, query, context = fixture(publisher)
    output = infer_model(state, source, query, execution_context=context())
    denied = context(cutoff=publisher.identity['first_sequence'])
    for call in (lambda: prepare_finite_search_model(state, source, execution_context=denied),
                 lambda: infer_model(state, source, query, execution_context=denied),
                 lambda: read_finite_search_result(output, execution_context=denied)):
        with pytest.raises(ValueError, match='namespace availability'): call()
    clear_finite_search_cache()
    with pytest.raises(ValueError, match='namespace availability'):
        read_finite_search_result(output, execution_context=denied)


@pytest.mark.parametrize('format', ('rcbd', 'json', 'safetensors', 'h5'))
def test_model_and_result_roundtrip_through_existing_formats(tmp_path, format):
    _, source, _, state, query, context = fixture()
    from rexgraph.cpp_finite_search import (
        clear_finite_search_cache,
        finite_search_query,
        read_finite_search_result,
    )
    from rexgraph.io import load, save
    from rexgraph.io.model_record import model_record, read_model_record
    from rexgraph.model_runtime import infer_model
    output = infer_model(state, source, query, execution_context=context())
    record = model_record(source, state)
    record.attach_metadata(1, 0, 'projection', output)
    path = tmp_path / ('projection.' + format)
    save(path, record); clear_finite_search_cache()
    restored_record = load(path)
    restored, original = read_model_record(restored_record, execution_context=context())
    assert restored.coefficient_digest == state.coefficient_digest
    replay = read_finite_search_result(restored_record.get_metadata(1, 0, 'projection'), execution_context=context())
    assert replay['image'].row(0).witness() == ((0, 0, 101), (0, 1, 101), (0, 2, 101))
    assert infer_model(restored, original, finite_search_query(restored), execution_context=context()).field_archive['data'] == output.field_archive['data']


@pytest.mark.parametrize('kind', ('local', 'sqlite', 'object'))
def test_checked_cross_store_copy_and_rcql_keep_the_admitted_recipe(tmp_path, kind):
    pytest.importorskip('rcdb'); pytest.importorskip('rcql')
    from contextlib import closing

    from rcdb import LocalStore, MemoryStore, NativeObjectStore, SQLStore, copy_record
    from rcql import Executor, parse
    from rexgraph.cpp_finite_search import read_finite_search_result
    from rexgraph.io.model_record import model_record
    _, source, _, state, query, context = fixture()
    target = {'local': lambda: LocalStore(tmp_path / 'records'),
        'sqlite': lambda: SQLStore('sqlite:///' + str(tmp_path / 'records.db')),
        'object': lambda: NativeObjectStore('file://' + str(tmp_path / 'objects'))}[kind]()
    with closing(MemoryStore()) as origin, closing(target):
        original = origin.put('projection', model_record(source, state), analytics=False)
        receipt = copy_record(origin, target, original, return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        result = Executor(sources={'r': target.get('projection')}, params={'q': query, 'authority': context()}).execute(
            parse('FROM $r RETURN MODEL_INFER(MODEL_STATE(execution_context=$authority),$q,execution_context=$authority)'))
        output = result.values[0]
        assert read_finite_search_result(output, execution_context=context())['image'].inspect()['row_count'] == 2


def test_rcql_program_worker_runs_through_hive_with_current_authority(tmp_path):
    pytest.importorskip('rexgraph_cpp._bridge')
    pytest.importorskip('agent.relational'); pytest.importorskip('rcdb.engine'); pytest.importorskip('rcql.executor')
    from contextlib import closing

    from agent.hive import Hive

    from agent import RelationalSession
    from rcdb import MemoryStore
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, parse
    from rexgraph.cpp_finite_search import read_finite_search_result
    from rexgraph.io.model_record import model_record
    _, source, _, state, query, context = fixture()
    program = Program('finite_projection', (ProgramStep('solve', parse(
        'FROM $m RETURN MODEL_INFER(MODEL_STATE(execution_context=$authority),$q,execution_context=$authority)')),),
        (ProgramInput('q'), ProgramInput('authority')), (('result', OutputRef('solve')),))
    with closing(MemoryStore()) as store:
        session = RelationalSession(store)
        session.publish('model', model_record(source, state), expected_version=0)
        session.publish_program('projection_program', program, expected_version=0)
        current = [context()]
        hive = Hive()
        hive.add_program('project', session, 'projection_program', version=1,
            sources={'m': {'record_id': 'model', 'version': 1}}, runtime_parameters={'authority': lambda: current[0]})
        execution = hive.dispatch_capability('analyze', {'q': query})['result']
        output = execution.named_values['result']
        assert read_finite_search_result(output, execution_context=current[0])['image'].row(1).witness()[0] == (0, 0, 102)
        current[0] = context(remove=(42,))
        with pytest.raises(ValueError): hive.invoke('project', {'q': query})
        with pytest.raises(ValueError): hive.invoke('project', {'q': query, 'authority': context()})


def test_system_query_and_views_describe_whole_projection_recipes():
    pytest.importorskip('system'); pytest.importorskip('rcql')
    import asyncio

    from system.serialize import json_value
    from system.server.app import QueryRequest
    from system.server.app import query as system_query
    from system.state import sources

    from rexgraph.io.model_record import model_record
    _, source, _, state, query, context = fixture()
    assert json_value(state)['retains_whole_fields']
    name = 'finite_projection_provider_test'
    sources.register(name, model_record(source, state))
    try:
        request = QueryRequest(query='FROM $' + name + ' RETURN MODEL_INFER(MODEL_STATE(execution_context=$authority),$q,execution_context=$authority)',
            params={'authority': context(), 'q': query})
        preview = asyncio.run(system_query(request))
        assert preview['values'][0]['retains_whole_fields']
        assert preview['values'][0]['value_storage'] == 'admitted-source-and-complete-request-reexecution'
    finally:
        sources.remove(name)


def test_rcql_typed_creation_metadata_and_complete_branch_inspection():
    pytest.importorskip('rcql')
    from rcql import Executor, parse
    from rexgraph.io.model_record import model_record
    from rexgraph.model_runtime import infer_model
    _, source, available, state, query, context = fixture()
    output = infer_model(state, source, query, execution_context=context())
    configuration = dict(available_source=available, projection={k: state.configuration['recipe'][k]
        for k in ('key', 'revision', 'cells', 'window', 'fixed', 'distinct_relations')}, execution_context=context())
    executor = Executor(sources={'r': model_record(source, state)}, params={
        'state': state, 'output': output, 'authority': context(), 'configuration': configuration})
    for expression in ('MODEL_INFO($state,execution_context=$authority)',
                       'MODEL_VALUES($output,execution_context=$authority)',
                       'MODEL_FINITE_BRANCHES($output,begin=0,count=2,execution_context=$authority)',
                       'MODEL_INIT(adapter="cpp_finite_search",configuration=$configuration)'):
        assert executor.execute(parse('FROM $r RETURN ' + expression)).values
