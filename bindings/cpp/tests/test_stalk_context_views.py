"""Faithful stalk views, complete shared inputs and original evidence replay."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from test_stalk_context_hypotheses import fixture
from rexgraph_cpp import PreparedStalkParameterRelations
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, stalk_transport_observation_batch, _prepare
from rexgraph.cpp_stalk_parameter_relations import stalk_context_views_query, stalk_context_hypotheses
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record


def injection(name, keys, positions):
    return (name, tuple(keys), tuple((axis, i, Q(1)) for i, axis in enumerate(positions)),
        tuple((i, axis, Q(1)) for i, axis in enumerate(positions)))


def model(*, inactive=False):
    source, _, _ = fixture(inactive=inactive)
    axes = ((('reading',),)*3, (('x', 'y'), ('x', 'y', 'z'), ('x', 'y')))
    slots = tuple((p, 0, j) for p in range(6) for j in range(len(axes[1][p//2])))
    state = cpp_stalk_transport_model(source, axes, ((), ((),)*6), slots, (0, 1, 2))
    frames = ((injection('reading', ('reading',), (0,)),)*3,
        tuple(injection('input', ('x', 'y', 'z'), range(len(row))) for row in axes[1]))
    state = train_model(state, source, stalk_transport_observation_batch(state,
        ((1, 0), (0, 0, 0), (0, 0)), ((0, (4,)),)))
    return source, state, frames


def output(*, selected=(), inactive=False):
    source, state, frames = model(inactive=inactive)
    return source, state, frames, infer_model(state, source,
        stalk_context_views_query(state, (1,), frames, selected_blocks=selected))


def test_different_original_dimensions_compare_complete_shared_inputs():
    source, state, frames, result = output(selected=((0, 1),))
    acquisition = stalk_context_hypotheses(result); block = acquisition['blocks'][0]
    assert acquisition['profile'] == 'acquired-context-transport-views-v1'
    assert block['cell_map'] == ((0, 1),) and block['occurrence_map'] == ((0, 2), (1, 3))
    assert block['common_input']['shape'] == (3, 2)
    assert block['origin_input']['shape'] == (2, 2) and block['target_input']['shape'] == (3, 2)
    assert block['response_shapes'] == ((1, 2), (1, 2))
    assert block['equation_coordinates'] == ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1))
    assert len(acquisition['relations']) == 4 and block['status'] == 'open'
    assert result.values.shape == (4, 14)
    family = result.execution_receipt['conditional_family']
    assert family['consistent'] and family['kernel_shape'] == (14, 9)
    assert result.execution_receipt['certificate']['evidence_fit']['kernel_shape'] == (14, 13)
    assert infer_model(state, source, stalk_context_views_query(state, (1,), frames)).execution_receipt['conditional_family'] is None
    with pytest.raises(TypeError): result.tensor()


def test_partial_input_intersection_retains_unshared_parameter_directions():
    source, state, frames = model()
    keys = ('a', 'b', 'c', 'd')
    frames = (frames[0], (injection('shared', keys, (0, 1)), injection('shared', keys, (1, 2, 3)), None))
    result = infer_model(state, source, stalk_context_views_query(state, (1,), frames, selected_blocks=((0, 1),)))
    block = stalk_context_hypotheses(result)['blocks'][0]
    assert block['common_input']['shape'] == (4, 1) and len(block['equations']) == 2
    assert block['origin_input']['entries'] == ((1, 0, Q(1)),)
    assert block['target_input']['entries'] == ((0, 0, Q(1)),)
    assert result.execution_receipt['conditional_family']['kernel_shape'] == (14, 11)


def test_nontranspose_reconstruction_and_redundant_coordinates_are_exact():
    source, state, frames = model()
    name, keys, a, _ = frames[1][0]
    chart = (name, keys, (*a, (2, 0, Q(1))), ((0, 0, Q(1, 2)), (0, 2, Q(1, 2)), (1, 1, Q(1))))
    frames = (frames[0], (chart, *frames[1][1:]))
    result = infer_model(state, source, stalk_context_views_query(state, (1,), frames))
    assert stalk_context_hypotheses(result)['blocks'][0]['common_input']['shape'] == (3, 2)
    assert len(stalk_context_hypotheses(result)['relations']) == 4
    assert len(result.execution_receipt['certificate']['evidence_fit']['free_columns']) == 13


def test_disjoint_inputs_and_absent_lower_views_do_not_supply_maps():
    source, state, frames = model()
    keys = ('a', 'b', 'c', 'd', 'e')
    none = (frames[0], (injection('shared', keys, (0, 1)), injection('shared', keys, (2, 3, 4)), None))
    result = infer_model(state, source, stalk_context_views_query(state, (1,), none))
    assert stalk_context_hypotheses(result)['rejections'] == {'empty-input-intersection': 1}
    assert result.values.shape == (0, 14) and result.execution_receipt['certificate'] is None
    none = ((None,)*3, frames[1])
    result = infer_model(state, source, stalk_context_views_query(state, (1,), none))
    assert stalk_context_hypotheses(result)['rejections'] == {'undeclared-lower-frame': 1}


def test_joint_context_rejects_inconsistent_observed_boundary_fields():
    source, state, frames = model()
    first = infer_model(state, source, stalk_context_views_query(state, (1,), frames))
    state = train_model(state, source, stalk_transport_observation_batch(state,
        ((0, 0), (1, 0, 0), (0, 0)), ((1, (6,)),)))
    second = infer_model(state, source, stalk_context_views_query(state, (1,), frames))
    assert stalk_context_hypotheses(first)['relations'] == stalk_context_hypotheses(second)['relations']
    assert set(second.execution_receipt['certificate']['statuses']) == {'open'}
    assert stalk_context_hypotheses(second)['blocks'][0]['status'] == 'excluded'


def test_inactive_maps_keep_zero_gain_support_and_original_unknowns():
    _, _, _, result = output(inactive=True)
    acquisition = stalk_context_hypotheses(result)
    assert acquisition['blocks'][0]['occurrence_map'] == ((0, 2), (1, 3))
    assert all(gain == 0 for _, terms, _ in acquisition['relations'][2:] for _, gain in terms)
    assert result.execution_receipt['certificate']['statuses'][2:] == ('forced', 'forced')
    assert result.execution_receipt['certificate']['evidence_fit']['kernel_shape'] == (14, 13)


def test_owned_native_views_are_detached_and_thread_safe():
    source, state, frames = model(); owner = _prepare(state, source)
    expected = owner.contextual_views((1,), frames)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r == expected for r in pool.map(lambda _: owner.contextual_views((1,), frames), range(8)))
    expected['blocks'][0]['common_input']['entries'] = ()
    assert owner.contextual_views((1,), frames)['blocks'][0]['common_input']['entries']


def test_utf8_accession_remains_separate_from_rectangular_semantic_views(monkeypatch):
    from rexgraph.linguistic_source import structural_language_source
    from rexgraph_cpp import PreparedRelationGuards
    documents = []
    for name, text in (('en', 'cat'), ('zh', '猫'), ('el', 'γάτα')):
        raw = text.encode('utf8')
        documents.append(dict(id=name, raw=raw, spans=((0, len(raw)),), assertions=(dict(
            id='reading', type='observed-reading', participants=(dict(token=0, concept=None, role='subject',
                exists=True, negative=False, magnitude=1),)),)))
    corpus = structural_language_source(documents, types=('observed-reading',), roles=('subject',))
    source = corpus.source; guard = PreparedRelationGuards(source.archive)
    rows = tuple(corpus.relation(name, 'reading') for name in ('en', 'zh', 'el'))
    lower = tuple(corpus.lower.index(('use', name, 0)) for name in ('en', 'zh', 'el'))
    semantic = {rows[0]: ('x', 'y'), rows[1]: ('x', 'y', 'z'), rows[2]: ('x', 'y')}
    axes = (tuple(('meaning',) if i in lower else ('bytes',) for i in range(len(corpus.lower))),
        tuple(semantic.get(r, ('encoding',)) for r in range(len(source.identity['coordinates'][1]))))
    occurrences = sum(len(guard.boundary(r)['occurrences']) for r in range(len(axes[1])))
    slots = tuple((guard.boundary(r)['occurrences'][0]['slot'], 0, j) for r in rows for j in range(len(axes[1][r])))
    state = cpp_stalk_transport_model(source, axes, ((), ((),)*occurrences), slots, lower, source_accession=corpus.accession())
    state = train_model(state, source, stalk_transport_observation_batch(state,
        tuple((1, 0) if r == rows[0] else (0,)*len(a) for r, a in enumerate(axes[1])), ((lower[0], (2,)),)))
    frames = (tuple(injection('meaning', ('meaning',), (0,)) if i in lower else None for i in range(len(axes[0]))),
        tuple(injection('semantic-input', ('x', 'y', 'z'), range(len(a))) if r in rows else None for r, a in enumerate(axes[1])))
    result = infer_model(state, source, stalk_context_views_query(state, rows[1:], frames))
    assert len(stalk_context_hypotheses(result)['blocks']) == 2
    assert all(b['status'] == 'open' for b in stalk_context_hypotheses(result)['blocks'])
    assert result.execution_receipt['source_archive'] == source.archive
    assert tuple(bytes.fromhex(c['raw_utf8_hex']) for c in state.configuration['source_accession']['charts']) == (
        b'cat', '猫'.encode('utf8'), 'γάτα'.encode('utf8'))


@pytest.mark.parametrize('value', [True, 1.0, Q(1)])
def test_indices_and_source_reference_scalar_kinds_are_not_coerced(value):
    source, state, frames = model()
    with pytest.raises((ValueError, TypeError)): stalk_context_views_query(state, (value,), frames)
    query = stalk_context_views_query(state, (1,), frames)
    query['source_reference']['schema_version'] = value
    with pytest.raises(ValueError): infer_model(state, source, query)


@pytest.mark.parametrize('change', ['nonfaithful', 'shape', 'carrier-keys', 'duplicate', 'float', 'bool', 'missing'])
def test_invalid_accession_reconstruction_declarations_reject(change):
    source, state, frames = model(); name, keys, a, r = frames[1][0]
    if change == 'nonfaithful': r = ((0, 0, Q(2)), (1, 1, Q(1)))
    if change == 'shape': r = ((2, 0, Q(1)),)
    if change == 'carrier-keys': keys = ('z', 'y', 'x')
    if change == 'duplicate': a = (*a, a[0])
    if change == 'float': a = ((0, 0, 1.0), (1, 1, Q(1)))
    if change == 'bool': a = ((True, 0, Q(1)), (1, 1, Q(1)))
    if change == 'missing': frames = (frames[0], frames[1][1:])
    else: frames = (frames[0], ((name, keys, a, r), *frames[1][1:]))
    with pytest.raises((ValueError, TypeError)):
        infer_model(state, source, stalk_context_views_query(state, (1,), frames))


@pytest.mark.parametrize('field', ['common_input', 'origin_input', 'target_input', 'response_shapes', 'equation_coordinates'])
def test_changed_native_view_witnesses_fail_replay(field):
    _, _, _, result = output(); receipt = thaw_tree(result.execution_receipt)
    receipt['acquisition']['blocks'][0][field] = ()
    with pytest.raises(ValueError, match='replay'): replace(result, execution_receipt=receipt)


def test_constant_equations_require_explicit_admission():
    source, state, _ = model(); owner = _prepare(state, source)
    relations = (('known-zero', (), Q(0)), ('contradiction', (), Q(1)))
    with pytest.raises(ValueError): PreparedStalkParameterRelations(owner, relations)
    proof = PreparedStalkParameterRelations(owner, relations, allow_constants=True)
    assert proof.receipt['statuses'] == ('forced', 'excluded')
    assert not proof.condition(('known-zero', 'contradiction'))['consistent']
    with pytest.raises(TypeError): PreparedStalkParameterRelations(owner, relations, allow_constants=1)


@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_view_hypotheses_roundtrip_with_complete_original_evidence(tmp_path, suffix):
    source, state, frames, result = output(selected=((0, 1),))
    query = stalk_context_views_query(state, (1,), frames, selected_blocks=((0, 1),))
    record = model_record(source, state); record.attach_metadata(1, 0, 'query', query); record.attach_metadata(1, 0, 'output', result)
    path = tmp_path/('views'+suffix); save_model(path, record); loaded = load_model(path); model_state, native = read_model_record(loaded)
    assert loaded.get_metadata(1, 0, 'output').coefficient_digest == result.coefficient_digest
    assert infer_model(model_state, native, loaded.get_metadata(1, 0, 'query')).coefficient_digest == result.coefficient_digest
    assert native.archive == source.archive


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_cold_five_package_view_replay(tmp_path, backend):
    from rcdb import open_store
    source, state, frames, result = output(selected=((0, 1),))
    record = model_record(source, state)
    record.attach_metadata(1, 0, 'query', stalk_context_views_query(state, (1,), frames, selected_blocks=((0, 1),)))
    record.attach_metadata(1, 0, 'output', result)
    url = {'file': 'file://'+str(tmp_path/'files'), 'rex': 'rex://'+str(tmp_path/'rex'), 'sqlite': 'sqlite:///'+str(tmp_path/'db')}[backend]
    store = open_store(url, **({} if backend == 'sqlite' else {'read_only': False})).configure_security(require_commits=True)
    store.commit_mutation('views', record, expected_version=0); assert store.verify_commits('views'); store.close()
    script = '''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]);record=store.read_record('views',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
output=result.values[0]
assert output.coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert result.values[1]['context_views'] and json_value(state)['context_views']
assert output.execution_receipt['acquisition']['blocks'][0]['common_input']['shape']==(3,2)
assert output.execution_receipt['conditional_family']['kernel_shape']==(14,9)
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('views');hive.add_worker('views',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('views',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    process = subprocess.run([sys.executable, '-c', script, url, result.coefficient_digest], cwd=tmp_path,
        capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
