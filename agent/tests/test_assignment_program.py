"""Learned native child programs execute through ordinary Hive workers and RCDB."""
import importlib.util
import os
from contextlib import closing
from pathlib import Path

import pytest


def fixture():
    pytest.importorskip('rexgraph_cpp._assignment')
    path = Path(os.environ.get('REXGRAPH_ASSIGNMENT_PROVIDER_HELPER',
        str(Path(__file__).resolve().parents[2] / 'rexgraph/tests/test_assignment_provider.py')))
    if not path.is_file(): pytest.skip('native assignment provider fixture required')
    spec = importlib.util.spec_from_file_location('assignment_provider_fixture', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper.learned()


def test_learned_specialization_pins_actual_child_and_runs_two_held_inputs(tmp_path, monkeypatch):
    from agent.hive import Hive
    from rcql.assignment import prepare_assignment_program, restore_assignment_program

    from agent import RelationalSession
    from rcdb import MemoryStore
    from rcql import Program, ProgramTransformation, Result
    from rexgraph.cpp_assignment import read_assignment_result
    from rexgraph.io.model_record import model_record
    _, _, case, source, states = fixture()
    monkeypatch.setenv('REXGRAPH_CONFIG_DIR', str(tmp_path / 'config'))
    monkeypatch.setenv('REXGRAPH_CACHE_DIR', str(tmp_path / 'cache'))
    prepared = prepare_assignment_program(states[-1], source, execution_context=case.context())
    assert prepared.candidate == 0 and prepared.binding['worker'] == 11
    assert prepared.owner.branch(0, case.context())['coordinates']['free_dimension'] == 1
    prepared.transformation.verify(prepared.parent_program)
    record = prepared.program.to_record()
    record.attach_metadata(1, 0, 'rcql_transformation', prepared.transformation.to_bytes().decode('utf8'))
    with closing(MemoryStore()) as store:
        session = RelationalSession(store)
        for i, state in enumerate(states): session.publish('learner', model_record(source, state), expected_version=i)
        session.publish('worker', record, expected_version=0)
        loaded = Program.from_record(session.read('worker', 1).value)
        change = ProgramTransformation.from_bytes(session.read('worker', 1).value.get_metadata(1, 0, 'rcql_transformation'))
        change.verify(prepared.parent_program)
        assert loaded.coefficient_digest == prepared.program.coefficient_digest
        authority = [case.context()]; hive = Hive('learned-assignment')
        hive.add_program('learned', session, 'worker', version=1,
            sources={'r': {'record_id': 'learner', 'version': 3}}, runtime_parameters=prepared.runtime_parameters(lambda: authority[0]))
        responses = []
        for i, row in enumerate(case.data['held']):
            result = hive.invoke('learned', {'input': case.read(row['input'])})
            output = result.named_values['response']; value = read_assignment_result(output, execution_context=authority[0])
            assert value.model(authority[0]).same_owner(prepared.owner)
            assert value.archive(authority[0]) == case.read(row['response'])
            assert output.execution_receipt['executed_external_worker'] is False
            session.save_result('held_' + str(i), result, expected_version=0)
            cold = Result.from_bytes(session.read('held_' + str(i), 1).value.to_bytes())
            assert cold.to_bytes() == result.to_bytes()
            responses.append(value.field_archive(3, authority[0]))
        assert responses[0] != responses[1]
        restored = restore_assignment_program(loaded, source, execution_context=case.context())
        assert restored.program.to_bytes() == prepared.program.to_bytes()
        with pytest.raises(ValueError): hive.invoke('learned', {'input': case.read(case.data['held'][0]['input']), 'owner': prepared.owner})
        authority[0] = case.context(remove=(case.data['declaration']['origin'],))
        with pytest.raises(ValueError): hive.invoke('learned', {'input': case.read(case.data['held'][0]['input'])})


def test_unresolved_model_and_unknown_catalog_cannot_compile_an_automatic_worker():
    from rcql.assignment import prepare_assignment_program

    from rexgraph.cpp_assignment import (
        assignment_batch,
        cpp_assignment_model,
        prepare_assignment_model,
    )
    from rexgraph.model_runtime import train_model
    cpp, _, case, source, states = fixture()
    with pytest.raises(ValueError): prepare_assignment_program(states[0], source, execution_context=case.context())
    unknown = cpp.NativeAssignmentEvidence.from_archive(case.read('incomplete-catalog.bin'), case.context(),
        limits=(4096, 16000000, 64000000, 1 << 24))
    root = unknown.parent(case.context()).parent(case.context())
    state = cpp_assignment_model(source, evidence=root, execution_context=case.context())
    for i in range(2):
        observed = unknown.observation(i, case.context())
        rows = (dict(key=observed['key'], origin=observed['origin'], binding=observed['binding'],
            measured=observed['measured'].archive(), inputs=tuple(dict(candidate=v['candidate'], archive=v['input'].archive()) for v in observed['inputs'])),)
        batch = assignment_batch(state, source, rows, execution_context=case.context())
        state = train_model(state, source, batch, execution_context=case.context())
    assert state.step == 2 and prepare_assignment_model(state, source, execution_context=case.context()).remainder(0, 1, case.context()) == ((999,),)
    with pytest.raises(ValueError): prepare_assignment_program(state, source, execution_context=case.context())
