"""Stored star selection preserves exact fields, joint weights and cold serving."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import subprocess
import sys
from unittest.mock import patch

import pytest

from rexgraph.cpp_stalk_pattern_model import (
    cpp_stalk_pattern_bank_model, select_stalk_pattern_bank_model,
    stalk_pattern_bank_sparse_query, stalk_pattern_bank_result,
)
from rexgraph.io.model_record import model_record, save_model
from rexgraph.model_runtime import infer_model
from rexgraph_cpp import NativeSource, PreparedStalkTransportPatternBank
from test_stalk_pattern_bank import correlated_bank, comparison, compile_bank


COORDINATES = {'weights': 1001, 'parameters': 1002}


def field(value):
    return dict(profile='complete-sparse-stalk-field-v1', frame=2, size=1,
                entries=((0, value),) if value else ())


@pytest.fixture(scope='module')
def parent():
    _, _, bank = correlated_bank()
    source = NativeSource(bank.archive())
    return bank, source, cpp_stalk_pattern_bank_model(source)


@pytest.mark.parametrize('patterns', [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)])
def test_selection_preserves_original_keys_and_the_complete_joint_family(parent, patterns):
    bank, _, _ = parent
    before = bank.inspect()
    selected = bank.select(patterns, model_namespace=906, coordinate_ids=COORDINATES)
    after = selected.inspect()
    assert after['pattern_keys'] == tuple(before['pattern_keys'][i] for i in patterns)
    assert after['occurrence_keys'] == tuple(before['occurrence_keys'][i] for i in patterns)
    assert after['weight_origin'] == tuple(before['weight_origin'][i] for i in patterns)
    assert after['input_images'] == tuple(before['input_images'][i] for i in patterns)
    for key in ('source_sha256', 'evidence_sha256', 'observations', 'peers', 'common_frames'):
        assert after[key] == before[key]
    assert after['parameter_directions']['shape'][1] == int(any(i < 2 for i in patterns))
    for values in ((1, 1, 1), (Q(3, 7), Q(-3, 7), Q(2, 11)), (0, 0, 0)):
        old = bank.response_sparse({i: field(values[i]) for i in patterns})
        new = selected.response_sparse({j: field(values[i]) for j, i in enumerate(patterns)})
        assert new['origin'] == old['origin'] and new['directions'] == old['directions']
        assert new['identified'] == old['identified']
    if patterns == (0, 1):
        cancelled = selected.response_sparse({0: field(1), 1: field(-1)})
        assert cancelled['identified'] and cancelled['origin'] == (0,)
    image = selected.archive()
    loaded = PreparedStalkTransportPatternBank(image, model_namespace=906)
    assert loaded.archive() == image and loaded.inspect() == after
    assert bank.inspect() == before


@pytest.mark.parametrize('redundant', [False, True])
def test_selection_keeps_full_rectangular_input_carriers_and_local_outputs(redundant):
    _, _, bank = compile_bank(redundant)
    selected = bank.select((1,), model_namespace=906, coordinate_ids=COORDINATES)
    x = (Q(2, 3), Q(-3, 7), 0, Q(2, 3) if redundant else 0)
    old = bank.response({1: x})
    new = selected.response({0: x})
    assert len(old['origin']) == 8 and len(new['origin']) == 4
    assert new['origin'] == old['origin'][4:]
    source = NativeSource(selected.archive())
    assert len(source.identity['coordinates'][1]) == 1 and len(source.identity['coordinates'][0]) == 2
    assert selected.inspect()['input_images'][0]['shape'] == bank.inspect()['input_images'][1]['shape']
    with pytest.raises(ValueError, match='outside'):
        selected.response({0: (0, 0, 1, 0)})
    if redundant:
        with pytest.raises(ValueError, match='outside'):
            selected.response({0: (1, 0, 0, 0)})


def test_selection_rebases_interfaces_and_preserves_exact_conflicts(parent):
    bank, _, _ = parent
    selected = bank.select((0, 2), model_namespace=906, coordinate_ids=COORDINATES)
    a = selected.glue_sparse({0: field(1), 1: field(1)}, (comparison('anchor', 0, 1),))
    b = bank.glue_sparse({0: field(1), 2: field(1)}, (comparison('anchor', 0, 2),))
    for key in ('origin', 'directions', 'offsets', 'identified'):
        assert a['section'][key] == b['section'][key]
    assert a['section']['domain_coordinates'] == tuple(b['section']['domain_coordinates'][i] for i in (0, 2))
    assert a['boundary_response']['origin'] == b['boundary_response']['origin']
    conflict = selected.glue_sparse({0: field(1), 1: field(1)},
        (comparison('first', 0, 1), comparison('second', 0, 1, Q(1, 2))))
    assert not conflict['consistent'] and conflict['section'] is None and any(conflict['contradiction'])
    outside = selected.response_sparse_candidates({0: field(0), 1: field(1)})
    assert all(item['supported'] for item in outside)


@pytest.mark.parametrize('patterns', [(), (1, 0), (0, 0), (3,), (-1,), (True,), (1.0,), '0', b'0'])
def test_selection_rejects_invalid_pattern_declarations(parent, patterns):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        parent[0].select(patterns, model_namespace=906, coordinate_ids=COORDINATES)


@pytest.mark.parametrize('coordinates', [
    {}, {'weights': 1001}, {'weights': 1001, 'parameters': 1002, 'extra': 1},
    {'weights': 0, 'parameters': 1002}, {'weights': 1001, 'parameters': 0},
    {'weights': 1001, 'parameters': 1001}, {'weights': True, 'parameters': 1002},
    {'weights': 1001.0, 'parameters': 1002}, {'weights': 101, 'parameters': 1002},
    {'weights': 1001, 'parameters': 104}, {'weights': 2, 'parameters': 1002},
])
def test_selection_requires_new_exact_coordinate_identities(parent, coordinates):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        parent[0].select((0,), model_namespace=906, coordinate_ids=coordinates)


@pytest.mark.parametrize('namespace', [0, 905, True, 1.0, -1])
def test_selection_requires_a_distinct_nonzero_namespace(parent, namespace):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        parent[0].select((0,), model_namespace=namespace, coordinate_ids=COORDINATES)


def test_concurrent_and_nested_selections_own_independent_compact_sources(parent):
    bank, source, model = parent
    before = bank.archive()
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = tuple(pool.map(lambda _: bank.select((0, 1), model_namespace=906,
            coordinate_ids=COORDINATES).archive(), range(8)))
    assert all(image == images[0] for image in images)
    state, local = select_stalk_pattern_bank_model(model, source, (0, 1),
        model_namespace=906, coordinate_ids=COORDINATES)
    assert local.archive == images[0] and state.payload == {} and state.step == 0
    state, nested = select_stalk_pattern_bank_model(state, local, (1,),
        model_namespace=907, coordinate_ids={'weights': 1003, 'parameters': 1004})
    query = stalk_pattern_bank_sparse_query(state, {0: field(Q(3, 7))})
    assert not stalk_pattern_bank_result(infer_model(state, nested, query))['identified']
    assert bank.archive() == before and source.archive == before


def test_selection_uses_the_stored_bank_and_binds_queries_to_its_new_source(parent):
    _, source, model = parent
    with patch('rexgraph.cpp_stalk_transport._prepare', side_effect=AssertionError('training accessed')):
        state, local = select_stalk_pattern_bank_model(model, source, (2,),
            model_namespace=906, coordinate_ids=COORDINATES)
    query = stalk_pattern_bank_sparse_query(model, {0: field(1)})
    with pytest.raises(ValueError):
        infer_model(state, local, query)
    with pytest.raises(ValueError):
        select_stalk_pattern_bank_model(model, local, (0,), model_namespace=907,
                                        coordinate_ids={'weights': 1003, 'parameters': 1004})


@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_cold_selected_model_serves_without_parent_or_training_source(parent, tmp_path, suffix):
    _, source, model = parent
    state, local = select_stalk_pattern_bank_model(model, source, (2,),
        model_namespace=906, coordinate_ids=COORDINATES)
    query = stalk_pattern_bank_sparse_query(state, {0: field(Q(3, 7))})
    output = infer_model(state, local, query)
    assert stalk_pattern_bank_result(output)['origin'] == (Q(6, 7),)
    record = model_record(local, state)
    record.attach_metadata(1, 0, 'query', query)
    path = tmp_path / ('selected' + suffix)
    save_model(path, record)
    script = '''
import sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_stalk_pattern_model import stalk_pattern_bank_result
import rexgraph.cpp_stalk_transport as training
def forbidden(*a,**k):raise AssertionError('training source accessed')
training._prepare=forbidden
record=load_model(sys.argv[1]);state,source=read_model_record(record)
assert state.step==0 and state.payload=={} and source.identity['namespace']==906
assert len(source.identity['coordinates'][1])==1
output=infer_model(state,source,record.get_metadata(1,0,'query'))
assert stalk_pattern_bank_result(output)['identified']
print(output.coefficient_digest)
'''
    completed = subprocess.run([sys.executable, '-c', script, str(path)], cwd='/tmp',
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == output.coefficient_digest


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_selected_model_cold_five_package_routes(parent, tmp_path, backend):
    from rcdb import open_store
    _, source, model = parent
    state, local = select_stalk_pattern_bank_model(model, source, (0, 1),
        model_namespace=906, coordinate_ids=COORDINATES)
    query = stalk_pattern_bank_sparse_query(state, {0: field(1), 1: field(-1)})
    expected = infer_model(state, local, query)
    record = model_record(local, state)
    record.attach_metadata(1, 0, 'query', query)
    url = {'file': 'file://' + str(tmp_path / 'files'), 'rex': 'rex://' + str(tmp_path / 'rex'),
           'sqlite': 'sqlite:///' + str(tmp_path / 'db')}[backend]
    store = open_store(url, **({} if backend == 'sqlite' else {'read_only': False})).configure_security(require_commits=True)
    store.commit_mutation('patterns', record, expected_version=0)
    assert store.verify_commits('patterns')
    store.close()
    script = '''
import sys,json
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_stalk_pattern_model import stalk_pattern_bank_result
store=open_store(sys.argv[1]);record=store.read_record('patterns',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
output=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q)')).values[0]
assert output.coefficient_digest==sys.argv[2]
assert stalk_pattern_bank_result(output)['identified']
json.dumps(json_value(output))
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('selected');hive.add_worker('selected',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('selected',query,record=False).coefficient_digest==sys.argv[2]
store.close()
'''
    completed = subprocess.run([sys.executable, '-c', script, url, expected.coefficient_digest],
                               cwd='/tmp', text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
