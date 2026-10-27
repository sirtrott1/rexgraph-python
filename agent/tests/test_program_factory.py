"""Only owning native sections authorize prospective Program interpretation."""
from contextlib import closing
from pathlib import Path

import pytest


def assemble(witness, source):
    from rcql import Program, ProgramStep, ProgramInput, OutputRef, parse
    symbols = tuple(row[2] for row in witness)
    if len(witness) != 3 or len(set(symbols)) != 1:
        raise ValueError('complete synthetic correlated field required')
    expressions = {101: 'APPLY(DIRAC(),$x0,true)', 102: 'APPLY(ANTI_DIRAC(),$x0,true)',
                   103: 'GRADE_COMPONENT($x0,0)'}
    return Program('finite', (ProgramStep('step', parse('FROM $r0 RETURN ' + expressions[symbols[0]])),),
        (ProgramInput('x0', 'GradedChain'),), (('answer', OutputRef('step')),))


def reverse(program, source):
    expression = program.steps[0].query.returns[0]
    symbol = 103 if expression.name == 'GRADE_COMPONENT' else {'DIRAC': 101, 'ANTI_DIRAC': 102}[expression.args[0].name]
    return tuple((0, i, symbol) for i in range(3))


def bad_reverse(program, source):
    return ((0, 0, 999),)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pytest.importorskip('rexgraph_cpp._finite_search')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'bindings/cpp/tests'))
    from agent import FamilyVerifier, FiniteProgramFactory, RelationalSession
    from agent.hive import Hive
    from rcdb import LocalStore
    from rcql import Result
    from rexgraph_cpp import NativeFiniteSource
    from test_relational_finite_field import asset, declaration
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    from test_program_bundles import bundle
    with closing(LocalStore(tmp_path / 'records')) as store:
        session = RelationalSession(store); verifier = FamilyVerifier(session)
        session.save_result('source', Result((NativeFiniteSource(asset(), declaration()).archive(),), aliases=('source',)))
        session.save_result('map', Result((dict(version=1),), aliases=('map',)))
        session.save_result('inputs', bundle())
        task = verifier.create_question('task', source=verifier.pin('source'), answer_cells=((0, 0), (0, 2)), key=701)
        proof = verifier.verify(task, proof_id='proof')
        factory = FiniteProgramFactory(verifier, interpretation='correlated-synthetic', version=1,
            assemble=assemble, reverse=reverse, dependencies=(verifier.pin('map'),))
        hive = Hive('finite-programs')
        try:
            yield session, verifier, factory, hive, proof
        finally:
            hive.stop_all()


def selected(setup):
    session, verifier, factory, hive, proof = setup
    pin = factory.select(proof, 0, program_id='program', selection_id='selection')
    factory.register(hive, 'worker', pin, bundle=verifier.pin('inputs'),
        source_outputs={'r0': 0, 'r1': 1}, parameter_outputs={'x0': 2})
    return pin


def test_unresolved_family_retained_and_actual_hive_program_publishes_whole_result(setup):
    session, verifier, factory, hive, proof = setup
    assert verifier.assess(proof).named_values['accepted'] is None
    assert verifier.proof_owner(proof)['image'].inspect()['row_count'] == 3
    pin = selected(setup)
    assert factory.assess(pin).named_values['accepted'] is True
    receipt = factory.execute('worker', result_id='whole', receipt_id='execution')
    assert factory.assess_execution(receipt).named_values['accepted'] is True
    factory.publish(receipt, 'published')
    result = session.read('whole').value
    assert len(result.native_plan['step_results']) == 1
    assert result.values[0].source is result.native_plan['bundle_values'][0]
    assert verifier.proof_owner(proof)['image'].inspect()['row_count'] == 3
    assert len(session.read('execution').value.to_bytes()) < 4000


@pytest.mark.parametrize('defect', ['map', 'source', 'task', 'proof', 'program', 'selection', 'input', 'worker', 'provider'])
def test_current_publication_rechecks_every_actual_dependency(setup, defect, monkeypatch):
    session, verifier, factory, hive, proof = setup
    selected(setup)
    receipt = factory.execute('worker', result_id='whole', receipt_id='execution')
    if defect == 'worker': hive._bees['worker']._handler = lambda data: None
    elif defect == 'provider': verifier.withdraw()
    else:
        name = {'input': 'inputs'}.get(defect, defect)
        snapshot = session.read(name)
        session.publish(name, snapshot.value, expected_version=snapshot.record.version)
    assert factory.assess_execution(receipt).named_values['accepted'] is None
    with pytest.raises(ValueError, match='current evidence'):
        factory.publish(receipt, 'forbidden')
    assert session.store.get_record('forbidden') is None


def test_packets_need_fresh_replay_and_live_interpretation(setup):
    from agent import FamilyVerifier, FiniteProgramFactory
    session, verifier, factory, hive, proof = setup
    pin = selected(setup)
    fresh_verifier = FamilyVerifier(session)
    fresh = FiniteProgramFactory(fresh_verifier, interpretation='correlated-synthetic', version=1,
        assemble=assemble, reverse=reverse, dependencies=(verifier.pin('map'),))
    assert fresh.assess(pin).named_values['accepted'] is None
    fresh.restore(pin, proof_id='cold-proof')
    assert fresh.assess(pin).named_values['accepted'] is True
    factory._reverse = bad_reverse
    assert factory.assess(pin).named_values['accepted'] is None


def test_bad_reverse_or_interrupted_family_cannot_publish_a_program(setup):
    from agent import FiniteProgramFactory
    session, verifier, factory, hive, proof = setup
    bad = FiniteProgramFactory(verifier, interpretation='bad', version=1, assemble=assemble, reverse=bad_reverse)
    with pytest.raises(ValueError, match='complete native section'):
        bad.select(proof, 0, program_id='bad', selection_id='bad-selection')
    assert session.store.get_record('bad') is None
    with pytest.raises((ValueError, IndexError)):
        factory.select(proof, 30, program_id='outside', selection_id='outside-selection')
    assert session.store.get_record('outside') is None
    task = verifier.create_question('halted', source=verifier.pin('source'), answer_cells=((0, 0),), key=702,
        family_limits=(65536, 0, 16777216, 65536, 65536))
    incomplete = verifier.verify(task, proof_id='halted-proof')
    with pytest.raises(ValueError, match='complete native image'):
        factory.select(incomplete, 0, program_id='incomplete', selection_id='incomplete-selection')
    assert session.store.get_record('incomplete') is None


def source_interface_factory(setup, groups):
    from agent import FiniteProgramFactory
    session, verifier, _, _, _ = setup
    return FiniteProgramFactory(verifier, interpretation='synthetic-source-interface', version=2,
        assemble=assemble, reverse=reverse, dependencies=(verifier.pin('map'),), distinct_sources=groups)


def test_distinct_source_contract_is_pinned_and_keeps_equal_value_owners_distinct(setup):
    session, verifier, _, hive, proof = setup
    declared = [['r0', 'r1']]
    factory = source_interface_factory(setup, declared)
    declared[0].append('late')
    pin = factory.select(proof, 0, program_id='strict-program', selection_id='strict-selection')
    record = verifier._value(pin, 'program_selection')
    assert record['interpretation']['distinct_sources'] == (('r0', 'r1'),)
    factory.register(hive, 'strict', pin, bundle=verifier.pin('inputs'),
        source_outputs={'r0': 0, 'r1': 1}, parameter_outputs={'x0': 2})
    receipt = factory.execute('strict', result_id='strict-whole', receipt_id='strict-receipt')
    assert factory.assess_execution(receipt).named_values['accepted'] is True
    value = session.read('strict-whole').value
    first, second = value.native_plan['bundle_values'][:2]
    assert first is not second and value.values[0].source is first


@pytest.mark.parametrize('defect', ['same-index', 'same-owner', 'missing', 'bool', 'outside'])
def test_declared_distinct_sources_reject_invalid_bindings_before_hive_registration(setup, defect):
    from rcql import Result
    session, verifier, _, hive, proof = setup
    factory = source_interface_factory(setup, (('r0', 'r1'),))
    pin = factory.select(proof, 0, program_id='strict-program', selection_id='strict-selection')
    outputs = {'r0': 0, 'r1': 1}
    if defect == 'same-index': outputs['r1'] = 0
    elif defect == 'same-owner':
        values = session.read('inputs').value.values
        session.save_result('aliased', Result((values[0], values[0], values[2], values[2])))
    elif defect == 'missing': outputs.pop('r1')
    elif defect == 'bool': outputs['r1'] = True
    elif defect == 'outside': outputs['r1'] = 99
    bundle = verifier.pin('aliased' if defect == 'same-owner' else 'inputs')
    with pytest.raises((ValueError, TypeError)):
        factory.register(hive, 'forbidden', pin, bundle=bundle,
            source_outputs=outputs, parameter_outputs={'x0': 2})
    assert 'forbidden' not in hive._bees


def test_generic_factory_aliases_remain_valid_without_a_distinct_source_declaration(setup):
    session, verifier, factory, hive, proof = setup
    pin = factory.select(proof, 0, program_id='program', selection_id='selection')
    factory.register(hive, 'alias', pin, bundle=verifier.pin('inputs'),
        source_outputs={'r0': 0, 'r1': 0}, parameter_outputs={'x0': 2})
    receipt = factory.execute('alias', result_id='whole', receipt_id='receipt')
    assert factory.assess_execution(receipt).named_values['accepted'] is True


@pytest.mark.parametrize('groups', ['r0', ('r0', 'r1'), (('r0',),), (('r0', 'r0'),), ((1, 'r1'),)])
def test_source_interface_groups_require_explicit_distinct_port_names(setup, groups):
    with pytest.raises((ValueError, TypeError)):
        source_interface_factory(setup, groups)
