"""Joint acquisition of ordered field observations with complete original evidence."""
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import numpy as np
import pytest

from test_stalk_transport import model, truth, declaration, AXES, SLOTS, TRUTH
from rexgraph_cpp import NativeSource
from rexgraph.cpp_stalk_transport import (
    cpp_stalk_transport_model, stalk_transport_observation_batch, stalk_transport_observation_group,
    stalk_transport_query, _prepare, _receipt_matches,
)
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record


def batches(state, fields=(((1, 0, 0),), ((0, 1, 0),), ((0, 0, 1),)), rows=(0, 1)):
    return tuple(stalk_transport_observation_batch(state, field, tuple((row, truth(field)[row]) for row in rows)) for field in fields)


def acquire(state, source, records):
    return train_model(state, source, stalk_transport_observation_group(state, records), steps=len(records))


def test_joint_fit_equals_every_original_equation_and_full_sequential_family():
    source, initial = model(); records = batches(initial)
    sequential = initial
    for batch in records: sequential = train_model(sequential, source, batch)
    grouped = acquire(initial, source, records)
    assert grouped.step == 3 and grouped.parent == initial.coefficient_digest
    assert grouped.coefficient_digest != sequential.coefficient_digest
    assert _receipt_matches(grouped.payload['fit'], sequential.payload['fit'])
    assert grouped.payload['fit']['particular'] == TRUTH and grouped.payload['fit']['rank'] == 9
    assert len(grouped.payload['observations']) == 3
    for retained, original in zip(grouped.payload['observations'], records, strict=True):
        assert retained['batch'] == original.coefficient_digest
        assert unpack_model(retained['batch_checkpoint']).coefficient_digest == original.coefficient_digest
    field = ((Q(2, 3), Q(-7, 5), Q(11, 13)),)
    a = infer_model(grouped, source, stalk_transport_query(grouped, field))
    b = infer_model(sequential, source, stalk_transport_query(sequential, field))
    assert np.array_equal(a.values, b.values)
    assert a.execution_receipt['response'] == b.execution_receipt['response']


def test_group_appends_every_original_observation_once(monkeypatch):
    import rexgraph.cpp_stalk_transport as native
    source, initial = model(); records = batches(initial); group = stalk_transport_observation_group(initial, records)
    calls = []
    original = native._native
    def fit(c, source, observations):
        calls.append(len(observations)); return original(c, source, observations)
    monkeypatch.setattr(native, '_native', fit)
    child = train_model(initial, source, group, steps=3)
    assert child.step == 3 and calls == []
    assert _prepare(child, source).execution_stats['assembled_observations'] == 3
    assert _prepare(child, source).execution_stats['source_reused']


def test_single_member_group_has_the_same_canonical_child_as_single_training():
    source, initial = model(); batch = batches(initial)[0]
    assert acquire(initial, source, (batch,)).coefficient_digest == train_model(initial, source, batch).coefficient_digest


def test_separate_masks_and_unresolved_original_directions_are_retained():
    source, initial = model(); records = batches(initial, fields=(((1, 0, 0),), ((0, 1, 0),)), rows=(0,))
    group = stalk_transport_observation_group(initial, records)
    assert group.targets.shape == (2, 2, 2) and group.observed.tolist() == [True, False]
    child = train_model(initial, source, group, steps=2)
    assert child.payload['fit']['rank'] == 4 and child.payload['fit']['kernel_shape'] == (9, 5)
    assert child.payload['fit']['free_columns'] == (2, 5, 6, 7, 8)
    assert all(o['observed'] == (True, False) for o in child.payload['observations'])
    response = infer_model(child, source, stalk_transport_query(child, ((0, 0, 1),)))
    assert response.execution_receipt['boundary_identified'] == ((False, False), (False,))


def test_atomic_group_conflict_preserves_the_parent_and_accepts_a_valid_retry():
    source, initial = model(); records = batches(initial)
    conflict = stalk_transport_observation_batch(initial, ((1, 0, 0),), ((0, (999, 999)),))
    digest = initial.coefficient_digest
    with pytest.raises(ValueError, match='inconsistent'): acquire(initial, source, (*records, conflict))
    assert initial.coefficient_digest == digest and initial.step == 0
    assert acquire(initial, source, records).payload['fit']['particular'] == TRUTH


def test_group_resume_and_single_updates_replay_complete_previous_evidence():
    source, initial = model(); records = batches(initial)
    first = acquire(initial, source, records[:2]); final = train_model(first, source, records[2])
    assert first.step == 2 and final.step == 3 and final.parent == first.coefficient_digest
    assert _receipt_matches(final.payload['fit'], acquire(initial, source, records).payload['fit'])
    reverse = acquire(initial, source, tuple(reversed(records)))
    assert reverse.payload['fit']['particular'] == TRUTH
    assert tuple(o['batch'] for o in reverse.payload['observations']) == tuple(b.coefficient_digest for b in reversed(records))


def test_redundant_group_observations_remain_distinct_without_claiming_new_rank():
    source, initial = model(); batch = batches(initial)[0]; child = acquire(initial, source, (batch, batch, batch))
    assert child.step == 3 and child.payload['fit']['rank'] == 3
    assert len(child.payload['fit']['constraints']) == 9 and len(child.payload['observations']) == 3


def test_repeated_inactive_and_zero_flags_keep_their_original_map_directions():
    source = NativeSource.from_declaration(declaration(((10, -1, True), (10, 1, True),
        (10, 7, False), (10, 0, True)), (10,)))
    axes = ((('meaning',),), (('sense',),)); slots = tuple((i, 0, 0) for i in range(4))
    state = cpp_stalk_transport_model(source, axes, ((), ((), (), (), ())), slots, (0,))
    records = tuple(stalk_transport_observation_batch(state, ((x,),), ((0, (2*x,)),)) for x in (1, 3))
    child = acquire(state, source, records); sequential = state
    for batch in records: sequential = train_model(sequential, source, batch)
    assert _receipt_matches(child.payload['fit'], sequential.payload['fit'])
    assert child.payload['fit']['rank'] == 1 and child.payload['fit']['kernel_shape'] == (4, 3)
    output = infer_model(child, source, stalk_transport_query(child, ((5,),)))
    assert output.execution_receipt['boundary_identified'] == ((True,),)
    assert output.execution_receipt['response'] == infer_model(sequential, source, stalk_transport_query(sequential, ((5,),))).execution_receipt['response']
    assert output.execution_receipt['source_archive'] == source.archive


def test_named_dependent_parameter_basis_retains_its_full_original_family():
    source, original = model(); c = original.configuration
    state = cpp_stalk_transport_model(source, AXES, c['fixed_maps'], SLOTS, (0, 1),
        parameter_space={'name': 'declared', 'keys': ('gain', 'same-gain', 'unseen')},
        parameter_basis=tuple((i, j, Q(1)) for i in range(9) for j in (0, 1)))
    records = tuple(stalk_transport_observation_batch(state, field, ((0, (-3, -3)), (1, (2,))))
        for field in (((1, 0, 0),), ((0, 1, 0),)))
    child = acquire(state, source, records)
    assert child.adapter_version == '2' and child.payload['fit']['basis_rank'] == 1
    assert child.payload['fit']['kernel_shape'] == (3, 2) and child.payload['fit']['free_columns'] == (1, 2)
    assert child.payload['fit']['particular'] == (Q(6), Q(0), Q(0))


@pytest.mark.parametrize('steps', [1, 2, 4, True, 3.0])
def test_group_step_count_requires_the_exact_original_observation_count(steps):
    source, initial = model(); group = stalk_transport_observation_group(initial, batches(initial))
    with pytest.raises(ValueError): train_model(initial, source, group, steps=steps)


@pytest.mark.parametrize('change', ['target', 'float', 'mask', 'shape', 'partition', 'extra', 'checkpoint-kind', 'checkpoint-bytes'])
def test_changed_group_declarations_reject_before_fitting(change):
    source, initial = model(); group = stalk_transport_observation_group(initial, batches(initial))
    if change == 'float':
        values = group.targets.copy(); values[0, 0, 0] = 1.0
        with pytest.raises(TypeError, match='integer or Fraction'):
            replace(group, targets=values)
        assert initial.step == 0
        return
    if change == 'target':
        values = group.targets.copy(); values[0, 0, 0] = 999; group = replace(group, targets=values)
    if change == 'mask': group = replace(group, observed=np.zeros(2, dtype=bool))
    if change == 'shape': group = replace(group, targets=group.targets[:, :, :1])
    if change in {'partition', 'extra', 'checkpoint-kind', 'checkpoint-bytes'}:
        inputs = thaw_tree(group.inputs)
        if change == 'partition': inputs['partition'] = 'test'
        if change == 'extra': inputs['input'] = ()
        if change == 'checkpoint-kind': inputs['batches'] = (pack_model(initial), *inputs['batches'][1:])
        if change == 'checkpoint-bytes': inputs['batches'][0]['model_digest'][0] ^= 1
        group = replace(group, inputs=inputs)
    with pytest.raises((ValueError, TypeError)): train_model(initial, source, group, steps=3)
    assert initial.step == 0


@pytest.mark.parametrize('kind', ['empty', 'nested', 'foreign', 'heldout', 'wrong-object'])
def test_groups_require_original_valid_batches(kind):
    source, state = model(); records = batches(state)
    if kind == 'empty': records = ()
    if kind == 'nested': records = (stalk_transport_observation_group(state, records),)
    if kind == 'foreign':
        foreign = replace(records[0].source, record_id='foreign'); records = (replace(records[0], source=foreign),)
    if kind == 'heldout': records = (replace(records[0], inputs=dict(records[0].inputs, partition='test')),)
    if kind == 'wrong-object': records = (state,)
    with pytest.raises((ValueError, TypeError)): stalk_transport_observation_group(state, records)


def test_group_dimension_and_encoded_checkpoint_admissions_reject_early():
    source, original = model(); c = original.configuration
    state = cpp_stalk_transport_model(source, AXES, c['fixed_maps'], SLOTS, (0, 1), limits=(10, 16000000, 64000000, 16384))
    batch = batches(state)[0]
    with pytest.raises(ValueError, match='count|admitted'): stalk_transport_observation_group(state, (batch,)*11)
    state = cpp_stalk_transport_model(source, AXES, c['fixed_maps'], SLOTS, (0, 1), limits=(4096, 1024, 64000000, 16384))
    with pytest.raises(ValueError, match='entry'): stalk_transport_observation_group(state, batches(state))


@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_saved_group_and_fitted_child_roundtrip_and_resume(tmp_path, suffix):
    source, initial = model(); records = batches(initial); group = stalk_transport_observation_group(initial, records[:2])
    child = train_model(initial, source, group, steps=2); record = model_record(source, child)
    record.attach_metadata(1, 0, 'group', group); path = tmp_path/('group'+suffix); save_model(path, record)
    loaded = load_model(path); state, native = read_model_record(loaded)
    retained = loaded.get_metadata(1, 0, 'group'); assert retained.coefficient_digest == group.coefficient_digest
    assert unpack_model(pack_model(group)).coefficient_digest == group.coefficient_digest
    assert state.coefficient_digest == child.coefficient_digest and native.archive == source.archive
    assert train_model(state, native, records[2]).payload['fit']['particular'] == TRUTH


@pytest.mark.parametrize('backend', ['file', 'rex', 'sqlite'])
def test_cold_five_package_group_training_and_resume(tmp_path, backend):
    from rcdb import open_store
    source, initial = model(); group = stalk_transport_observation_group(initial, batches(initial))
    expected = train_model(initial, source, group, steps=3); record = model_record(source, initial)
    record.attach_metadata(1, 0, 'group', group)
    url = {'file': 'file://'+str(tmp_path/'files'), 'rex': 'rex://'+str(tmp_path/'rex'), 'sqlite': 'sqlite:///'+str(tmp_path/'db')}[backend]
    store = open_store(url, **({} if backend == 'sqlite' else {'read_only': False})).configure_security(require_commits=True)
    store.commit_mutation('group', record, expected_version=0); assert store.verify_commits('group'); store.close()
    script = '''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import train_model,infer_model
from rexgraph.cpp_stalk_transport import stalk_transport_query
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]);record=store.read_record('group',version=1).value
state,source=read_model_record(record);batch=record.get_metadata(1,0,'group')
result=Executor(sources={'r':record},params={'b':batch}).execute(parse(
 'FROM $r RETURN MODEL_TRAIN(MODEL_STATE(),$b,3),MODEL_INFO(MODEL_STATE())'))
child=result.values[0]
assert child.coefficient_digest==sys.argv[2] and child.step==3 and child.payload['fit']['rank']==9
assert result.values[1]['observation_groups'] and json_value(child)['observation_groups']
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('group');hive.add_worker('group',lambda b,**kw:train_model(state,source,b,steps=3),capability='transform')
 assert hive.invoke('group',batch,record=False).coefficient_digest==sys.argv[2]
assert infer_model(child,source,stalk_transport_query(child,((1,0,0),))).execution_receipt['boundary_identified']==((True,True),(True,))
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    process = subprocess.run([sys.executable, '-c', script, url, expected.coefficient_digest], cwd=tmp_path,
        capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
