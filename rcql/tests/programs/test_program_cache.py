from fractions import Fraction as Q

import pytest
from rcdb import MemoryStore

from rcql import Executor, OutputRef, Program, ProgramInput, ProgramStep, parse
from rcql.query_cache import QueryCache

from .helpers import native


def program():
    return Program('incremental', (
        ProgramStep('first', parse('FROM $r RETURN NATIVE_RESPONSE($x)')),
        ProgramStep('second', parse('FROM $r RETURN NATIVE_RESPONSE($x)')),
        ProgramStep('amount', parse('FROM $r RETURN SUM([$n,1])')),
    ), (ProgramInput('x'), ProgramInput('n', 'ExactInteger')),
        (('first', OutputRef('first')), ('second', OutputRef('second')), ('amount', OutputRef('amount'))))


def test_exact_port_projection_shares_calculation_and_invalidates_only_changed_inputs():
    r, x, _ = native()
    p = program()
    cache = QueryCache(MemoryStore())
    def execute(n, field=x):
        return Executor(sources={'r': r}, params={'x': field, 'n': n}, exactness='exact')
    reference = execute(2).execute_program(p)
    first = execute(2).execute_program(p, cache=cache)
    assert [result.native_plan['cache']['hit'] for _, result in first.steps] == [False, True, False]
    assert first.values[0].coefficient_digest == reference.values[0].coefficient_digest
    assert first.values[1].coefficient_digest == reference.values[1].coefficient_digest
    assert first.values[2] == reference.values[2] == 3
    changed = execute(7).execute_program(p, cache=cache)
    assert [result.native_plan['cache']['hit'] for _, result in changed.steps] == [True, True, False]
    assert changed.values[2] == 8
    assert dict(changed.steps)['first'].native_plan['program_input_ports'] == ('x',)
    field_changed = execute(7, x.with_values([Q(2), Q(1, 3), Q(5, 7)])).execute_program(p, cache=cache)
    assert [result.native_plan['cache']['hit'] for _, result in field_changed.steps] == [False, True, True]
    assert field_changed.values[0].coefficient_digest != first.values[0].coefficient_digest
    # The two original occurrences remain separate retained steps.
    assert tuple(name for name, _ in field_changed.steps) == ('first', 'second', 'amount')


def test_program_preserves_exact_policy_in_child_steps():
    r = native()[0]
    p = Program('float_input', (ProgramStep('value', parse('FROM $r RETURN $x')),), (ProgramInput('x'),))
    assert Executor(sources={'r': r}, params={'x': 1.25}).execute_program(p).values == (1.25,)
    with pytest.raises((ValueError, TypeError)):
        Executor(sources={'r': r}, params={'x': 1.25}, exactness='exact').execute_program(p)


def test_program_cache_keeps_linked_whole_fields_and_refuses_match():
    r, x, _ = native()
    p = Program('linked', (
        ProgramStep('field', parse('FROM $r RETURN NATIVE_RESPONSE($x)')),
        ProgramStep('read', parse('FROM $r RETURN $g'), (('g', OutputRef('field')),)),
    ), (ProgramInput('x'),))
    executor = Executor(sources={'r': r}, params={'x': x})
    cache = QueryCache(MemoryStore())
    one = executor.execute_program(p, cache=cache)
    two = executor.execute_program(p, cache=cache)
    assert one.values[0] is dict(one.steps)['field'].values[0]
    assert two.values[0].coefficient_digest == one.values[0].coefficient_digest
    assert all(result.native_plan['cache']['hit'] for _, result in two.steps)
    p = Program('iteration', (ProgramStep('each', parse('FROM $r MATCH e IN CELLS(1) RETURN e')),))
    with pytest.raises(TypeError, match='MATCH'):
        Executor(sources={'r': r}).execute_program(p, cache=cache)


def test_program_cache_requires_explicit_qualified_cache():
    r, x, _ = native()
    executor = Executor(sources={'r': r}, params={'x': x, 'n': 1})
    with pytest.raises(TypeError):
        executor.execute_program(program(), cache={})
    with pytest.raises(TypeError):
        executor.execute_program(program(), explain=True, cache=QueryCache(MemoryStore()))


@pytest.mark.parametrize('prepared', [False, True])
def test_native_model_cache_retains_exact_publication_and_rejects_foreign_cutoff(tmp_path, prepared):
    from dataclasses import replace

    binding = pytest.importorskip('rexgraph_cpp')
    from rcdb import open_store
    from rexgraph.cpp_model import cpp_model, cpp_rational_batch_input
    from rexgraph.io.model_record import model_record
    from rexgraph.type_accession import CoordinateSpace

    from rcql import Result
    from rcql.prepared import PreparedQueryCache
    cache_type = PreparedQueryCache if prepared else QueryCache
    source = binding.NativeSource.from_declaration({
        'profile':'declared-exact-source-v1', 'namespace':82008, 'semantic_scale':0, 'max_grade':1,
        'types':({'name':'cache','min_arity':2,'max_arity':2},), 'witnesses':(),
        'relations':[{'grade':1,'key':100,'type':0,'occurrences':[
            {'key':0,'lower':10,'exists':True,'negative':False,'magnitude':Q(1)},
            {'key':1,'lower':11,'exists':True,'negative':True,'magnitude':Q(1)}]}],
        'carriers':[{'key':10,'kind':'source_grade','grade':0,'coordinates':0,'size':2},
                    {'key':11,'kind':'source_grade','grade':1,'coordinates':0,'size':1}],
        'fields':[{'key':1001,'carrier':11,'entries':((0,Q(2,7)),),'support':None}], 'metrics':[]})
    model = cpp_model(source, (10,11))
    inputs = cpp_rational_batch_input(model, source, (1001,), CoordinateSpace('samples',('one',)))
    record = model_record(source, model)
    query = parse('FROM $r RETURN MODEL_INFER(MODEL_STATE(),$inputs)')
    executor = Executor(sources={'r':record}, params={'inputs':inputs}, exactness='exact')
    uri = 'local://'+str(tmp_path/'cache')
    store = open_store(uri)
    first = executor.execute_cached(query, cache_type(store))
    assert not first.native_plan['cache']['hit']
    store.close()

    store = open_store(uri)
    second = executor.execute_cached(query, cache_type(store))
    assert second.native_plan['cache']['hit']
    assert second.values[0].coefficient_digest == first.values[0].coefficient_digest
    assert second.values[0].source.coefficient_digest == first.values[0].source.coefficient_digest
    detached = Result.from_bytes(second.to_bytes()).values[0]
    altered = replace(detached.source.native, sequence=detached.source.native.sequence+1)
    forged = replace(detached, source=replace(detached.source, native=altered))
    bad_result = replace(second, values=(forged,))
    key = 'rcql_result/'+second.native_plan['cache']['key']
    cached = store.read_record(key).value
    cached.attach_metadata(1,0,'rcql_cache_payload',bad_result.to_bytes())
    store.commit_mutation(key,cached,expected_version=1,analytics=False)
    with pytest.raises(ValueError, match='unbound source'):
        executor.execute_cached(query, cache_type(store))
    store.close()


def test_prepared_reuse_retains_actual_whole_owners_and_private_receipts(monkeypatch):
    from rcql.prepared import PreparedQueryCache
    r, x, _ = native()
    store = MemoryStore()
    cache = PreparedQueryCache(store, max_results=2)
    query = parse('FROM $r RETURN NATIVE_RESPONSE($x)')
    executor = Executor(sources={'r': r}, params={'x': x}, exactness='exact')
    first = executor.execute_cached(query, cache)
    first.native_plan['cache']['key'] = 'caller mutation'
    def no_read(*args, **kwargs):
        raise AssertionError('a retained whole result must not reload RCDB')
    monkeypatch.setattr(store, 'read_record', no_read)
    second = executor.execute_cached(query, cache)
    assert second.values[0] is first.values[0]
    assert second.native_plan['cache']['key'] != 'caller mutation'
    assert second.native_plan['cache']['hit']
    assert second.execution[0]['methods'][0]['method'] == 'validated-prepared-result-reuse'


def test_prepared_result_bound_evicts_to_checked_durable_recovery():
    from rcql.prepared import PreparedQueryCache
    r, x, _ = native()
    store = MemoryStore()
    cache = PreparedQueryCache(store, max_results=1)
    query = parse('FROM $r RETURN NATIVE_RESPONSE($x)')
    def run(field):
        return Executor(sources={'r': r}, params={'x': field}).execute_cached(query, cache)
    first = run(x)
    changed = run(x.with_values([7, 8, 9]))
    assert len(cache._results) == 1
    assert first.values[0].coefficient_digest != changed.values[0].coefficient_digest
    restored = run(x)
    assert restored.native_plan['cache']['hit']
    assert restored.values[0].coefficient_digest == first.values[0].coefficient_digest
    assert restored.values[0] is not first.values[0]
    cache.clear()
    assert not cache._results


def test_implementation_snapshot_detects_same_size_change_and_allows_same_bytes(tmp_path, monkeypatch):
    import os

    from rcql import program_codec as codec
    source = tmp_path / 'operator.py'
    source.write_bytes(b'first')
    monkeypatch.setattr(codec, '_implementation_inputs', lambda: (((b'operator.py', source),), (b'v=1',)))
    owner = codec.ImplementationSnapshot()
    assert owner.check() == codec.implementation_identity()
    os.utime(source, None)
    assert owner.check() == codec.implementation_identity()
    timestamp = source.stat().st_mtime_ns
    source.write_bytes(b'other')
    os.utime(source, ns=(timestamp, timestamp))
    with pytest.raises(ValueError, match='implementation changed'):
        owner.check()


def test_nested_program_steps_share_only_explicit_resource_lifetime(monkeypatch):
    from contextlib import contextmanager

    import rcql.executor as module
    from rcql.execution_trace import current_query_resources, query_resource_scope
    observed = []
    @contextmanager
    def observe():
        with query_resource_scope():
            observed.append(current_query_resources())
            yield
    # The adapter context is observed without replacing any field execution.
    monkeypatch.setattr('rcql.execution_trace.query_resource_scope', observe)
    r, x, _ = native()
    executor = module.Executor(sources={'r': r}, params={'x': x, 'n': 0})
    executor.execute_program(program())
    assert len(observed) == 4 and all(owner is observed[0] for owner in observed)
    assert current_query_resources() is None
    executor.execute_program(program())
    assert observed[4] is not observed[0]


def test_prepared_reuse_rechecks_current_policy_and_evidence():
    from rcql import BoundSource, PreparedQueryCache, SourcePolicy
    from rcql.source_context import SnapshotContext
    r, x, _ = native()
    cache = PreparedQueryCache(MemoryStore())
    query = parse('FROM $r RETURN NATIVE_RESPONSE($x)')
    Executor(sources={'r': r}, params={'x': x}).execute_cached(query, cache)
    denied = BoundSource(r, SourcePolicy.allow('identity'))
    with pytest.raises(PermissionError):
        Executor(sources={'r': denied}, params={'x': x}).execute_cached(query, cache)
    # A different evidence closure cannot inherit an earlier cache admission.
    evidence = SnapshotContext({}, {})
    with pytest.raises(ValueError):
        Executor(sources={'r': r}, params={'x': x}, evidence=evidence).execute_cached(query, cache)
