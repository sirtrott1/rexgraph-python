"""Connected original diagrams, shared identities and joint exact evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from rexgraph_cpp import NativeSource, PreparedStalkTransport
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, stalk_transport_observation_batch, _prepare
from rexgraph.cpp_stalk_parameter_relations import stalk_context_diagrams_query, stalk_context_hypotheses
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from test_stalk_context_views import injection


def fixture(*, repeated=False, inactive=False, rectangular=False):
    relations = []; key = 0
    for row, cells in enumerate(((0, 1), (1, 2), (3, 4), (4, 5))):
        flags = []
        for i, cell in enumerate((*cells, cells[1]) if repeated else cells):
            key += 1
            flags.append(dict(key=key, lower=cell, exists=i < 2 and not (inactive and i == 1),
                negative=i == 0, magnitude=1, aspects=(0 if i == 0 else 1,)))
        relations.append(dict(grade=1, key=row+1, type=0, occurrences=tuple(flags)))
    source = NativeSource.from_declaration(dict(profile='declared-role-source-v1', namespace=826173,
        semantic_scale=2, max_grade=1, role_aspects=('head', 'tail'),
        types=(dict(name='context', min_arity=0, max_arity=0),), witnesses=tuple(range(6)),
        relations=tuple(relations), fields=(), metrics=(), carriers=()))
    axes = ((('meaning',),)*6, tuple(('x', 'y') if rectangular and row >= 2 else ('x',) for row in range(4)))
    arity = 3 if repeated else 2
    slots = tuple((p, 0, j) for p in range(4*arity) for j in range(len(axes[1][p//arity])))
    state = cpp_stalk_transport_model(source, axes, ((), ((),)*(4*arity)), slots, tuple(range(6)))
    frames = ((injection('meaning', ('meaning',), (0,)),)*6,
        tuple(injection('input', ('x', 'y'), range(len(a))) for a in axes[1]))
    field = tuple((int(row < 2),)*(len(a)) if row < 2 else (0,)*len(a) for row, a in enumerate(axes[1]))
    targets = ((0, (-2,)), (1, (-3,))) if inactive else ((0, (-2,)), (2, (3,)))
    state = train_model(state, source, stalk_transport_observation_batch(state, field, targets))
    return source, state, frames


def result(*, selected=(), **kwargs):
    source, state, frames = fixture(**kwargs)
    query = stalk_context_diagrams_query(state, ((2, 3),), frames, selected_diagrams=selected)
    return source, state, query, infer_model(state, source, query)


def test_complete_group_preserves_shared_stalk_and_original_family():
    source, state, query, output = result(selected=(((0, 2), (1, 3)),))
    acquisition = stalk_context_hypotheses(output); diagram = acquisition['diagrams'][0]
    assert acquisition['profile'] == 'acquired-context-transport-diagrams-v1'
    assert len(acquisition['blocks']) == 4 and len(acquisition['diagrams']) == 1
    assert diagram['row_map'] == ((0, 2), (1, 3))
    assert diagram['cell_map'] == ((0, 3), (1, 4), (2, 5)) and diagram['shared_cell_map'] == ((1, 4),)
    assert diagram['occurrence_map'] == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert diagram['status'] == 'open' and len(diagram['equations']) == 4
    assert acquisition['diagram_rejections'] == {'shared-boundary-identity': 1, 'shared-upper-identity': 2}
    assert output.execution_receipt['certificate']['evidence_fit']['kernel_shape'] == (8, 6)
    assert output.execution_receipt['conditional_family']['kernel_shape'] == (8, 2)
    assert output.execution_receipt['source_archive'] == source.archive
    assert state.coefficient_digest == output.model_digest
    with pytest.raises(TypeError): output.tensor()
    assert infer_model(state, source, dict(query, selected_diagrams=())).execution_receipt['conditional_family'] is None


def test_individually_feasible_stars_can_be_jointly_excluded():
    source, state, frames = fixture()
    for field, targets in ((((1,), (0,), (0,), (0,)), ((1, (2,)),)),
            (((0,), (1,), (0,), (0,)), ((1, (-3,)),)),
            (((0,), (0,), (1,), (1,)), ((4, (5,)),))):
        state = train_model(state, source, stalk_transport_observation_batch(state, field, targets))
    query = stalk_context_diagrams_query(state, ((2, 3),), frames, selected_diagrams=(((0, 2), (1, 3)),))
    output = infer_model(state, source, query); acquisition = stalk_context_hypotheses(output)
    diagram = acquisition['diagrams'][0]
    assert diagram['status'] == 'excluded' and not diagram['joint_consistent']
    assert all(acquisition['blocks'][i]['joint_consistent'] for i in diagram['blocks'])
    assert not output.execution_receipt['conditional_family']['consistent']


@pytest.mark.parametrize('repeated', [False, True])
@pytest.mark.parametrize('inactive', [False, True])
def test_repeated_inactive_flags_and_opposite_shared_signs_are_retained(repeated, inactive):
    _, _, _, output = result(repeated=repeated, inactive=inactive)
    acquisition = stalk_context_hypotheses(output); diagram = acquisition['diagrams'][0]
    assert len(diagram['occurrence_map']) == (6 if repeated else 4)
    assert diagram['shared_cell_map'] == ((1, 4),) and diagram['status'] == 'open'
    if repeated or inactive:
        assert any(gain == 0 for _, terms, _ in acquisition['relations'] for _, gain in terms)


def test_rectangular_original_inputs_keep_unshared_directions():
    _, _, _, output = result(rectangular=True, selected=(((0, 2), (1, 3)),))
    acquisition = stalk_context_hypotheses(output)
    assert all(acquisition['blocks'][i]['common_input']['shape'] == (2, 1) for i in acquisition['diagrams'][0]['blocks'])
    assert output.execution_receipt['certificate']['evidence_fit']['kernel_shape'] == (12, 10)
    assert output.execution_receipt['conditional_family']['kernel_shape'] == (12, 6)


def test_native_diagrams_are_owned_detached_and_concurrent():
    source, state, frames = fixture(); owner = _prepare(state, source)
    expected = owner.contextual_diagrams(((2, 3),), frames)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(out == expected for out in pool.map(lambda _: owner.contextual_diagrams(((2, 3),), frames), range(8)))
    expected['diagrams'][0]['cell_map'] = ()
    assert owner.contextual_diagrams(((2, 3),), frames)['diagrams'][0]['cell_map']
    empty = PreparedStalkTransport.__new__(PreparedStalkTransport)
    with pytest.raises(RuntimeError): empty.contextual_diagrams(((2, 3),), frames)


def test_absent_observations_preserve_all_original_unknowns():
    source, learned, frames = fixture(); c = learned.configuration
    state = cpp_stalk_transport_model(source, c['stalk_axes'], c['fixed_maps'], c['parameter_slots'], tuple(range(6)))
    output = infer_model(state, source, stalk_context_diagrams_query(state, ((2, 3),), frames))
    acquisition = stalk_context_hypotheses(output)
    assert acquisition['diagrams'] == () and acquisition['blocks'] == ()
    assert acquisition['diagram_rejections'] == {'no-observed-star-cover': 1}
    assert output.execution_receipt['certificate'] is None and output.values.shape == (0, 9)


def test_empty_patch_scope_does_not_invent_relations():
    source, state, frames = fixture()
    output = infer_model(state, source, stalk_context_diagrams_query(state, (), frames))
    assert stalk_context_hypotheses(output)['diagrams'] == () and output.values.shape == (0, 7)


@pytest.mark.parametrize('change', ['extra', 'source', 'source-kind', 'partial'])
def test_query_remains_closed_and_bound_to_actual_original_source(change):
    source, state, frames = fixture(); query = stalk_context_diagrams_query(state, ((2, 3),), frames)
    if change == 'extra': query['selected_blocks'] = ()
    if change == 'source': query['source_reference']['native']['archive_digest'] = '0'*64
    if change == 'source-kind': query['source_reference']['schema_version'] = Q(1)
    if change == 'partial': del query['selected_diagrams']
    with pytest.raises(ValueError): infer_model(state, source, query)


@pytest.mark.parametrize('patches', [((2,),), ((2, 2),), ((2, 7),), ((0, 2),), ((2, 3), (3, 2)),
    ((True, 3),), ((Q(2), 3),), ((2.0, 3),), '23'])
def test_invalid_or_disconnected_original_patches_reject(patches):
    source, state, frames = fixture()
    with pytest.raises((ValueError, TypeError)):
        infer_model(state, source, stalk_context_diagrams_query(state, patches, frames))


@pytest.mark.parametrize('selected', [(((0, 2),),), (((0, 2), (0, 3)),), (((0, 2), (1, 2)),),
    (((True, 2), (1, 3)),), (((1, 2), (0, 3)),), (((0, 2), (1, 3)), ((0, 2), (1, 3)))])
def test_selected_diagrams_require_complete_acquired_injective_maps(selected):
    source, state, frames = fixture()
    with pytest.raises((ValueError, TypeError)):
        infer_model(state, source, stalk_context_diagrams_query(state, ((2, 3),), frames, selected_diagrams=selected))


@pytest.mark.parametrize('field', ['row_map', 'cell_map', 'shared_cell_map', 'occurrence_map', 'equations', 'blocks', 'status'])
def test_changed_group_witness_rejects_exact_replay(field):
    _, _, _, output = result(); receipt = thaw_tree(output.execution_receipt)
    receipt['acquisition']['diagrams'][0][field] = ()
    with pytest.raises(ValueError, match='replay'): replace(output, execution_receipt=receipt)


@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_complete_group_records_roundtrip(tmp_path, suffix):
    source, state, query, output = result(selected=(((0, 2), (1, 3)),))
    record = model_record(source, state); record.attach_metadata(1, 0, 'query', query); record.attach_metadata(1, 0, 'output', output)
    path = tmp_path/('diagrams'+suffix); save_model(path, record); loaded = load_model(path); model, native = read_model_record(loaded)
    assert native.archive == source.archive
    assert loaded.get_metadata(1, 0, 'output').coefficient_digest == output.coefficient_digest
    assert infer_model(model, native, loaded.get_metadata(1, 0, 'query')).coefficient_digest == output.coefficient_digest


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_cold_five_package_group_replay(tmp_path, backend):
    from rcdb import open_store
    source, state, query, output = result(selected=(((0, 2), (1, 3)),))
    record = model_record(source, state); record.attach_metadata(1, 0, 'query', query); record.attach_metadata(1, 0, 'output', output)
    url = {'file': 'file://'+str(tmp_path/'files'), 'rex': 'rex://'+str(tmp_path/'rex'), 'sqlite': 'sqlite:///'+str(tmp_path/'db')}[backend]
    store = open_store(url, **({} if backend == 'sqlite' else {'read_only': False})).configure_security(require_commits=True)
    store.commit_mutation('diagrams', record, expected_version=0); assert store.verify_commits('diagrams'); store.close()
    script = '''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]);record=store.read_record('diagrams',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
output=result.values[0]
assert output.coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert result.values[1]['context_diagrams'] and json_value(state)['context_diagrams']
assert output.execution_receipt['acquisition']['diagrams'][0]['shared_cell_map']==((1,4),)
assert output.execution_receipt['conditional_family']['kernel_shape']==(8,2)
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('diagrams');hive.add_worker('diagrams',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('diagrams',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    process = subprocess.run([sys.executable, '-c', script, url, output.coefficient_digest], cwd=tmp_path,
        capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
