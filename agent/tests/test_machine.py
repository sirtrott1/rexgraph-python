from fractions import Fraction as Q

import pytest
from agent.hive import Hive
from agent.machine import RelationalMachine
from agent.mcp_tools import Context

from agent import RelationalSession
from rcdb import MemoryStore, VersionConflictError, open_store
from rcql import OutputRef, Program, ProgramInput, ProgramStep, parse


def seed(store, context=None):
    from rexgraph import RexGraph
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.section_calculus import SectionFamily
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.type_accession import CoordinateSpace
    session = RelationalSession(store, context=context)
    graph = RexGraph.from_cells([4, [[0, 1], [1, 2], [2, 3]]])
    session.publish('data', graph, expected_version=0)
    snapshot = session.read('data', 1)
    source = FieldSource(snapshot.value, 'data', 1, snapshot.state_digest)
    space = NativeFieldCalculus.from_rex(snapshot.value).complex.spaces[1]
    x = TensorField(space, [1, Q(2, 3), Q(-1, 5)], source=source, grade=1, variance='chain')
    local = CoordinateSpace('alternatives', ('left', 'right', 'unobserved'))
    equations = CoordinateSpace('agreement', ('overlap',))
    action = CoordinateMap(local, equations, ((0, 0, 1), (0, 1, -1)))
    family = SectionFamily.solve(action, TensorField(equations, [0], source=source), 'retained')
    p = Program('field_machine', (
        ProgramStep('first', parse('FROM $r RETURN NATIVE_RESPONSE($x)')),
        ProgramStep('second', parse('FROM $r RETURN NATIVE_RESPONSE($x)')),
        ProgramStep('alternatives', parse('FROM $r RETURN $family')),
        ProgramStep('advance', parse('FROM $r RETURN SUM([$tick,1])')),
    ), (ProgramInput('x', 'TensorField'), ProgramInput('family', 'SectionFamily'),
        ProgramInput('tick', 'ExactInteger')),
        (('first', OutputRef('first')), ('second', OutputRef('second')),
         ('alternatives', OutputRef('alternatives')), ('tick', OutputRef('advance'))))
    session.publish_program('program', p, expected_version=0)
    state = {'x': x, 'family': family, 'tick': 0}
    return session, state, p


def create(session, state, name='machine'):
    return RelationalMachine.create(session, name, program_id='program', program_version=1,
        sources={'r': {'record_id': 'data', 'version': 1}}, state=state,
        message_ports=('x',), feedback={'tick': 'tick'})


def hits(result):
    return [step['native_plan']['cache']['hit'] for step in result.native_plan['step_results']]


def test_hive_program_and_machine_retain_fields_alternatives_and_incremental_ports(tmp_path, monkeypatch):
    monkeypatch.setenv('REXGRAPH_CONFIG_DIR', str(tmp_path / 'config'))
    session, state, _ = seed(MemoryStore())
    machine = create(session, state)
    hive = Hive('machine-tests')
    try:
        hive.add_program('operator', session, 'program', version=1,
                         sources={'r': {'record_id': 'data', 'version': 1}})
        operator = hive.invoke('operator', state)
        hive.add_machine('machine', machine)
        first = hive.invoke('machine', {'job_id': 'one', 'expected_version': 1})
        assert first.record.version == 2 and not first.replayed
        assert hits(first.result) == [False, True, False, False]
        for i in (0, 1, 2):
            assert operator.values[i].coefficient_digest == first.result.values[i].coefficient_digest
        assert first.result.values[2].directions.entries == state['family'].directions.entries
        assert len(first.result.values[2].directions.domain.keys) == 2
        second = hive.invoke('machine', {'job_id': 'two', 'expected_version': 2})
        assert hits(second.result) == [True, True, True, False]
        assert machine.inspect().value.values[0]['state']['tick'] == 2
        assert first.result.values[0].source.matches(second.result.values[0].source)
        assert {'operator', 'machine'} <= set(hive._complex.agents())
    finally:
        hive.stop_all()


def test_machine_cold_reopen_old_job_replay_and_guarded_messages(tmp_path):
    uri = 'local://' + str(tmp_path / 'records')
    store = open_store(uri)
    session, state, _ = seed(store)
    machine = create(session, state)
    one = machine.advance(job_id='one', expected_version=1)
    two = machine.advance(job_id='two', expected_version=2)
    store.close()
    store = open_store(uri)
    session = RelationalSession(store)
    machine = RelationalMachine(session, 'machine')
    replay = machine.advance(job_id='one', expected_version=1, optimized=False)
    assert replay.replayed and replay.record.version == one.record.version
    assert replay.result.to_bytes() == one.result.to_bytes()
    assert machine.inspect().record.version == two.record.version
    assert machine.inspect(1).value.values[0]['state']['tick'] == 0
    assert machine.inspect().value.values[0]['state']['tick'] == 2
    with pytest.raises(ValueError, match='different request'):
        machine.advance(job_id='one', expected_version=1, parameters={'x': state['x']})
    with pytest.raises(VersionConflictError):
        machine.advance(job_id='stale', expected_version=1)
    with pytest.raises(ValueError, match='undeclared'):
        machine.advance(job_id='wrong-port', expected_version=3, parameters={'tick': 99})
    assert len(store.history('machine')) == 3
    assert store.verify_commits('machine')
    store.close()


def test_failed_feedback_or_input_never_publishes_machine_state():
    session, state, _ = seed(MemoryStore())
    machine = create(session, state)
    with pytest.raises((ValueError, TypeError)):
        machine.advance(job_id='bad', expected_version=1, parameters={'x': 'no field'})
    assert machine.inspect().record.version == 1
    with pytest.raises(ValueError):
        machine.advance(job_id='bool', expected_version=True)


def test_reference_and_optimized_machine_agree_after_independent_input_change():
    session, state, _ = seed(MemoryStore())
    fast = create(session, state, 'fast')
    slow = create(session, state, 'reference')
    for index, field in enumerate((state['x'], state['x'].with_values([Q(3, 7), Q(-2), Q(4, 9)]))):
        message = {'x': field}
        a = fast.advance(job_id=str(index), expected_version=index+1, parameters=message)
        b = slow.advance(job_id=str(index), expected_version=index+1, parameters=message, optimized=False)
        for i in (0, 1, 2):
            assert a.result.values[i].coefficient_digest == b.result.values[i].coefficient_digest
        assert a.result.values[3] == b.result.values[3]
        assert a.result.values[0].source.as_record() == b.result.values[0].source.as_record()
    assert hits(a.result) == [False, True, True, False]


def test_machine_pins_old_sources_and_program_after_new_revisions():
    from rexgraph import RexGraph
    session, state, program = seed(MemoryStore())
    machine = create(session, state)
    session.publish('data', RexGraph.from_cells([2, [[0, 1]]]), expected_version=1)
    session.publish_program('program', program.rename_step('first', 'renamed'), expected_version=1)
    output = machine.advance(job_id='old', expected_version=1)
    assert output.result.native_plan['version'] == 1
    assert output.result.values[0].space.keys == state['x'].space.keys
    assert output.result.values[0].source.matches(state['x'].source)


def test_machine_keeps_workspace_and_actor_boundaries():
    store = MemoryStore()
    context = Context(workspace='alpha', identity='alice', auth_enabled=True, is_admin=False)
    session, state, _ = seed(store, context)
    machine = create(session, state)
    machine.advance(job_id='mine', expected_version=1)
    assert store.get_record('machine').meta['stored_by'] == 'alice'
    foreign = RelationalSession(store, context=Context(workspace='beta', identity='bob', auth_enabled=True))
    with pytest.raises(KeyError):
        RelationalMachine(foreign, 'machine').inspect()
    assert all((row.meta or {}).get('workspace') == 'alpha' for row in store.list())


def test_machine_requires_explicit_pins_and_port_contracts():
    session, state, _ = seed(MemoryStore())
    with pytest.raises(ValueError, match='explicit version'):
        RelationalMachine.create(session, 'bad', program_id='program', program_version=1,
                                 sources={'r': {'record_id': 'data'}}, state=state)
    with pytest.raises(TypeError, match='type'):
        RelationalMachine.create(session, 'bad', program_id='program', program_version=1,
            sources={'r': {'record_id': 'data', 'version': 1}}, state=state, feedback={'x': 'tick'})
    assert session.store.get_record('bad') is None


def test_equal_values_from_another_record_do_not_bind_to_machine():
    from dataclasses import replace

    from rexgraph.tensor_field import FieldSource
    session, state, _ = seed(MemoryStore())
    machine = create(session, state)
    original = state['x'].source
    foreign = replace(state['x'], source=FieldSource(original.source, 'another-record', 1, original.state_digest))
    with pytest.raises(ValueError, match='unbound source'):
        machine.advance(job_id='foreign', expected_version=1, parameters={'x':foreign})
    assert machine.inspect().record.version == 1


def test_competing_transitions_commit_only_one_state_revision(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    session, state, _ = seed(MemoryStore())
    first = create(session, state)
    second = RelationalMachine(session, 'machine')
    gate = Barrier(2)
    original = RelationalMachine._publish
    def publish(self, checkpoint, **kwargs):
        gate.wait(timeout=10)
        return original(self, checkpoint, **kwargs)
    monkeypatch.setattr(RelationalMachine, '_publish', publish)
    def advance(machine, job):
        try:
            return machine.advance(job_id=job, expected_version=1, optimized=False)
        except VersionConflictError as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(advance, first, 'first')
        b = pool.submit(advance, second, 'second')
        results = [a.result(), b.result()]
    assert sum(isinstance(value, VersionConflictError) for value in results) == 1
    assert first.inspect().record.version == 2
    assert first.inspect().value.values[0]['state']['tick'] == 1
    assert len(session.store.history('machine')) == 2
    assert session.store.verify_commits('machine')


def test_prepared_machine_keeps_pinned_record_owners_without_reloading(monkeypatch):
    session, state, _ = seed(MemoryStore())
    machine = create(session, state)
    original = session.store.read_record
    reads = []
    def read(record_id, *args, **kwargs):
        reads.append(record_id)
        return original(record_id, *args, **kwargs)
    monkeypatch.setattr(session.store, 'read_record', read)
    one = machine.advance(job_id='one', expected_version=1)
    owner = machine._preparation
    two = machine.advance(job_id='two', expected_version=2)
    assert machine._preparation is owner
    assert 'program' not in reads and 'data' not in reads
    assert one.result.values[0] is two.result.values[0]


def test_prepared_program_rejects_mutation_of_actual_pinned_carrier():
    session, state, _ = seed(MemoryStore())
    owner = session.prepare_program('program', version=1,
        sources={'r': {'record_id': 'data', 'version': 1}})
    first = owner.execute(state, exactness='exact')
    assert first.values[0].source.matches(state['x'].source)
    owner._bound['r'].value.attach_metadata(1, 0, 'changed', True)
    with pytest.raises(ValueError):
        owner.execute(state, exactness='exact')


def test_prepared_program_retains_owners_after_caller_release_and_serializes_scratch():
    import gc
    from concurrent.futures import ThreadPoolExecutor
    session, state, _ = seed(MemoryStore())
    owner = session.prepare_program('program', version=1,
        sources={'r': {'record_id': 'data', 'version': 1}})
    expected = owner.execute(state).values[0].coefficient_digest
    del session
    gc.collect()
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda _: owner.execute(state, exactness='exact'), range(6)))
    assert all(result.values[0].coefficient_digest == expected for result in outputs)
    owner.clear()
    assert not owner._resources
    assert owner.execute(state).values[0].coefficient_digest == expected
