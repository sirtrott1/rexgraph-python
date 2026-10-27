"""Assignment models retain complete evidence and immutable training lineage."""
import importlib.util
import os
from dataclasses import replace
from pathlib import Path

import pytest


def fixture():
    cpp = pytest.importorskip('rexgraph_cpp')
    if not hasattr(cpp, 'NativeAssignmentEvidence'): pytest.skip('native assignment provider required')
    path = Path(os.environ.get('REXGRAPH_ASSIGNMENT_TEST_HELPER',
        str(Path(__file__).resolve().parents[2] / 'bindings/cpp/tests/test_assignment_evidence.py')))
    if not path.is_file(): pytest.skip('native assignment fixture loader required')
    spec = importlib.util.spec_from_file_location('assignment_fixture_loader', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    case = helper.Fixture()
    root = case.model(); source = cpp.NativeSource(root.finite_source(case.context()).asset().source_archive())
    from rexgraph.cpp_assignment import cpp_assignment_model
    state = cpp_assignment_model(source, evidence=root, execution_context=case.context())
    return cpp, helper, case, source, state


def rows(case, native, ordinal):
    row = case.observation(native, ordinal)
    return (dict(key=row['key'], origin=row['origin'], binding=row['binding'], measured=row['measured'].archive(),
        inputs=tuple(dict(candidate=item['candidate'], archive=item['input'].archive()) for item in row['inputs'])),)


def learned():
    cpp, helper, case, source, state = fixture()
    from rexgraph.cpp_assignment import assignment_batch, prepare_assignment_model
    from rexgraph.model_runtime import train_model
    states = [state]
    for i in range(2):
        native = prepare_assignment_model(states[-1], source, execution_context=case.context())
        batch = assignment_batch(states[-1], source, rows(case, native, i), execution_context=case.context())
        assert not batch.observed.any()
        states.append(train_model(states[-1], source, batch, execution_context=case.context()))
    return cpp, helper, case, source, tuple(states)


def test_source_axes_complete_observations_and_actual_parent_survive_learning():
    _, _, case, source, states = learned()
    from rexgraph.cpp_assignment import clear_assignment_cache, prepare_assignment_model
    from rexgraph.model_codec import pack_model, unpack_model
    for i, state in enumerate(states):
        assert state.space == states[0].space and state.source.matches(states[0].source)
        assert state.output_axes == states[0].output_axes and state.step == i
        assert state.payload['archive'] == case.read(f'model-v{i + 1}.bin')
        if i: assert state.parent == states[i - 1].coefficient_digest
        native = prepare_assignment_model(state, source, execution_context=case.context())
        assert native.archive(case.context()) == state.payload['archive']
        clear_assignment_cache()
        detached = unpack_model(pack_model(state, native=True))
        assert prepare_assignment_model(detached, source, execution_context=case.context()).archive(case.context()) == state.payload['archive']


@pytest.mark.parametrize('ordinal', (0, 1))
def test_input_conditioned_outputs_are_complete_families_with_actual_owners(ordinal):
    _, _, case, source, states = learned(); state = states[-1]
    from rexgraph.cpp_assignment import (
        assignment_query,
        prepare_assignment_model,
        read_assignment_result,
    )
    from rexgraph.model_codec import pack_model, unpack_model
    from rexgraph.model_runtime import infer_model
    native = prepare_assignment_model(state, source, execution_context=case.context())
    output = infer_model(state, source, assignment_query(state, case.read(case.data['held'][ordinal]['input'])), execution_context=case.context())
    response = read_assignment_result(output, execution_context=case.context(), state=state)
    assert output.values.size == 0 and response.model(case.context()).same_owner(native)
    assert read_assignment_result(output, execution_context=case.context()).model(case.context()).same_owner(native)
    assert response.archive(case.context()) == case.read(case.data['held'][ordinal]['response'])
    assert response.field_archive(3, case.context())['free_parameter_count'] == 1
    assert output.execution_receipt['executed_external_worker'] is False
    cold = unpack_model(pack_model(output, native=True))
    assert read_assignment_result(cold, execution_context=case.context(), state=state).archive(case.context()) == response.archive(case.context())


def test_cached_models_outputs_batches_and_checkpoints_recheck_current_authority():
    _, _, case, source, states = learned()
    from rexgraph.cpp_assignment import (
        assignment_query,
        clear_assignment_cache,
        prepare_assignment_model,
        read_assignment_result,
    )
    from rexgraph.model_runtime import infer_model
    state = states[-1]; output = infer_model(state, source, assignment_query(state, case.read(case.data['held'][0]['input'])), execution_context=case.context())
    denied = case.context(remove=(case.data['declaration']['origin'],))
    for operation in (lambda: prepare_assignment_model(state, source, execution_context=denied),
                      lambda: infer_model(state, source, assignment_query(state, case.read(case.data['held'][0]['input'])), execution_context=denied),
                      lambda: read_assignment_result(output, execution_context=denied)):
        with pytest.raises(ValueError): operation()
    clear_assignment_cache()
    with pytest.raises(ValueError): read_assignment_result(output, execution_context=denied)


def test_namespace_cutoff_rejects_warm_and_cold_learning_and_held_execution():
    _, _, case, source, states = learned()
    from rexgraph.cpp_assignment import (
        assignment_query,
        clear_assignment_cache,
        prepare_assignment_model,
        read_assignment_result,
    )
    from rexgraph.model_runtime import infer_model
    state = states[-1]
    output = infer_model(state, source, assignment_query(state, case.read(case.data['held'][0]['input'])), execution_context=case.context())
    denied = case.context(limits={case.data['declaration']['origin']: 0})
    for call in (lambda: prepare_assignment_model(state, source, execution_context=denied),
                 lambda: read_assignment_result(output, execution_context=denied),
                 lambda: infer_model(state, source, assignment_query(state, case.read(case.data['held'][0]['input'])), execution_context=denied)):
        with pytest.raises(ValueError, match='namespace availability'): call()
    clear_assignment_cache()
    with pytest.raises(ValueError, match='namespace availability'): read_assignment_result(output, execution_context=denied)


def test_changed_implementation_or_parent_lineage_is_rejected():
    _, _, case, source, states = learned()
    from rexgraph.cpp_assignment import prepare_assignment_model
    for altered in (replace(states[-1], parent='0' * 64),
                    replace(states[-1], configuration={**states[-1].configuration, 'root_sha256': '0' * 64}),
                    replace(states[-1], configuration={**states[-1].configuration, 'engine_receipt': {'changed': True}})):
        with pytest.raises(ValueError): prepare_assignment_model(altered, source, execution_context=case.context())


def test_cold_response_rejects_forged_core_model_identity_without_optional_state():
    _, _, case, source, states = learned()
    from rexgraph.cpp_assignment import (
        assignment_query,
        clear_assignment_cache,
        read_assignment_result,
    )
    from rexgraph.model_runtime import infer_model
    output = infer_model(states[-1], source, assignment_query(states[-1], case.read(case.data['held'][0]['input'])),
        execution_context=case.context())
    forged = replace(output, model_digest='0' * 64)
    clear_assignment_cache()
    with pytest.raises(ValueError, match='model identity'):
        read_assignment_result(forged, execution_context=case.context())


@pytest.mark.parametrize('format', ('rcbd', 'json', 'safetensors', 'h5'))
def test_whole_learning_model_and_response_roundtrip(tmp_path, format):
    if format == 'safetensors': pytest.importorskip('safetensors')
    if format == 'h5': pytest.importorskip('h5py')
    _, _, case, source, states = learned(); state = states[-1]
    from rexgraph.cpp_assignment import (
        assignment_query,
        clear_assignment_cache,
        read_assignment_result,
    )
    from rexgraph.io import load, save
    from rexgraph.io.model_record import model_record, read_model_record
    from rexgraph.model_runtime import infer_model
    output = infer_model(state, source, assignment_query(state, case.read(case.data['held'][0]['input'])), execution_context=case.context())
    record = model_record(source, state); record.attach_metadata(1, 0, 'response', output)
    path = tmp_path / ('assignment.' + format); save(path, record); clear_assignment_cache()
    restored = load(path); model, original = read_model_record(restored, execution_context=case.context())
    assert model.coefficient_digest == state.coefficient_digest
    assert read_assignment_result(restored.get_metadata(1, 0, 'response'), execution_context=case.context(), state=model).archive(case.context()) == output.field_archive['data']
    assert infer_model(model, original, assignment_query(model, case.read(case.data['held'][0]['input'])), execution_context=case.context()).field_archive['data'] == output.field_archive['data']
