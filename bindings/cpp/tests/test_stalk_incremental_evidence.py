"""Retained native source and exact observation rows across immutable updates."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction as Q
import gc

import pytest

from test_stalk_transport import model, truth, AXES, SLOTS, TRUTH
from rexgraph_cpp import PreparedStalkTransport
from rexgraph.cpp_stalk_transport import (
    _prepare, _receipt_matches, cpp_stalk_transport_model,
    stalk_transport_observation_batch, stalk_transport_query,
)
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record


def observation(field):
    return dict(input=field,target=truth(field),observed=(True,True))


def cold(source, state, observations):
    c=state.configuration;extra={k:c[k] for k in ('parameter_space','parameter_basis')} if state.adapter_version=='2' else {}
    return PreparedStalkTransport(source.archive,c['stalk_axes'],c['fixed_maps'],c['parameter_slots'],observations,
        limits=c['limits'],max_bytes=c['max_bytes'],**extra)


def test_sparse_insertions_before_previous_pivots_match_complete_cold_receipts():
    source,state=model();initial=_prepare(state,source);owner=initial;records=[]
    for field in (((0,0,1),),((1,0,0),),((0,1,0),),((Q(2,3),Q(-7,5),Q(11,13)),)):
        records.append(observation(field));owner=owner.append((records[-1],))
        assert _receipt_matches(owner.receipt,cold(source,state,records).receipt)
        assert owner.execution_stats['source_reused']
        assert owner.execution_stats['reused_observations']==len(records)-1
        assert owner.execution_stats['assembled_observations']==1
    assert owner.receipt['particular']==TRUTH and owner.receipt['kernel_shape']==(9,0)
    assert initial.receipt['rank']==0 and not initial.execution_stats['source_reused']


def test_group_partition_does_not_change_canonical_replay_cost_or_family():
    source,state=model();owner=_prepare(state,source)
    records=tuple(observation((field,)) for field in ((1,0,0),(0,1,0),(0,0,1),(2,3,5)))
    all_at_once=owner.append(records);two=owner.append(records[:2]).append(records[2:])
    single=owner
    for record in records:single=single.append((record,))
    assert _receipt_matches(all_at_once.receipt,two.receipt) and _receipt_matches(two.receipt,single.receipt)
    assert two.execution_stats['reused_observations']==2 and single.execution_stats['reused_observations']==3


def test_named_dependent_and_unseen_parameters_match_cold_full_families():
    source,original=model();c=original.configuration
    state=cpp_stalk_transport_model(source,AXES,c['fixed_maps'],SLOTS,(0,1),
        parameter_space={'name':'declared','keys':('gain','same-gain','unseen')},
        parameter_basis=tuple((i,j,Q(1)) for i in range(9) for j in (0,1)))
    records=tuple(dict(input=(field,),target=((-3,-3),(2,)),observed=(True,True)) for field in ((1,0,0),(0,1,0)))
    first=_prepare(state,source).append(records[:1]);child=first.append(records[1:])
    assert _receipt_matches(child.receipt,cold(source,state,records).receipt)
    assert child.receipt['particular']==(Q(6),Q(0),Q(0)) and child.receipt['kernel_shape']==(3,2)
    assert child.receipt['free_columns']==(1,2) and first.receipt['rank']==1


def test_model_training_uses_only_new_native_equations_and_cold_validation_replays_them():
    import rexgraph.cpp_stalk_transport as native
    source,state=model();initial=state
    for index,field in enumerate((((1,0,0),),((0,1,0),),((0,0,1),))):
        state=train_model(state,source,stalk_transport_observation_batch(state,field,tuple(enumerate(truth(field)))))
        owner=_prepare(state,source)
        assert owner.execution_stats['reused_observations']==index and owner.execution_stats['assembled_observations']==1
    assert state.payload['fit']['particular']==TRUTH
    with native._lock:native._cache.clear()
    restored=_prepare(state,source)
    assert not restored.execution_stats['source_reused'] and restored.execution_stats['assembled_observations']==3
    assert _receipt_matches(restored.receipt,state.payload['fit'])
    assert initial.step==0 and state.step==3


@pytest.mark.parametrize('change',['empty','target','float','boolean','partial-input','mask-kind','mask-size','unobserved-target','extra'])
def test_invalid_suffix_rejects_atomically_and_valid_retry_preserves_the_parent(change):
    source,state=model();owner=_prepare(state,source);record=observation(((1,0,0),));parent=owner.append((record,))
    changed=deepcopy(record);suffix=(changed,)
    if change=='empty':suffix=()
    if change=='target':changed['target']=((999,999),changed['target'][1])
    if change=='float':changed['input']=((1.0,0,0),)
    if change=='boolean':changed['input']=((True,0,0),)
    if change=='partial-input':changed['input']=((1,0),)
    if change=='mask-kind':changed['observed']=(1,True)
    if change=='mask-size':changed['observed']=(True,)
    if change=='unobserved-target':changed['observed']=(True,False)
    if change=='extra':changed['extra']=()
    before=parent.receipt
    with pytest.raises((ValueError,TypeError,RuntimeError)):parent.append(suffix)
    assert _receipt_matches(before,parent.receipt)
    assert parent.append((record,)).receipt['rank']==3


def test_appended_owners_keep_the_source_after_parent_and_source_collection():
    source,state=model();owner=_prepare(state,source);child=owner.append((observation(((1,0,0),)),))
    del source,state,owner;gc.collect()
    final=child.append((observation(((0,1,0),)),observation(((0,0,1),))))
    assert final.receipt['particular']==TRUTH


def test_concurrent_native_children_keep_distinct_observation_histories():
    source,state=model();owner=_prepare(state,source)
    records=tuple(observation((field,)) for field in ((1,0,0),(0,1,0),(0,0,1),(2,3,5)))
    with ThreadPoolExecutor(max_workers=4) as executor:children=tuple(executor.map(lambda record:owner.append((record,)),records))
    for child,record in zip(children,records,strict=True):assert _receipt_matches(child.receipt,cold(source,state,(record,)).receipt)
    assert owner.receipt['rank']==0


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_saved_incremental_model_restores_and_appends_original_observations(tmp_path,suffix):
    import rexgraph.cpp_stalk_transport as native
    source,state=model();field=((1,0,0),)
    first=train_model(state,source,stalk_transport_observation_batch(state,field,tuple(enumerate(truth(field)))))
    path=tmp_path/('model'+suffix);save_model(path,model_record(source,first))
    with native._lock:native._cache.clear()
    restored,source=read_model_record(load_model(path));field=((0,1,0),)
    second=train_model(restored,source,stalk_transport_observation_batch(restored,field,tuple(enumerate(truth(field)))))
    assert _prepare(second,source).execution_stats['reused_observations']==1
    assert _prepare(second,source).execution_stats['assembled_observations']==1
    output=infer_model(second,source,stalk_transport_query(second,((1,1,0),)))
    assert output.execution_receipt['boundary_identified']==((True,True),(True,))
