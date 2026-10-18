"""Whole field transport learning preserves every original chain interface."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import numpy as np
import pytest

from rexgraph_cpp import PreparedStalkBoundary, PreparedGradedStalkTransport
from rexgraph.cpp_graded_stalk_transport import (
    cpp_graded_stalk_transport_model, graded_stalk_transport_observation_batch,
    graded_stalk_transport_query, graded_stalk_transport_maps, graded_stalk_transport_field,
)
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.type_accession import CoordinateSpace
from test_stalk_boundary import third_grade


def declaration():
    source, axes, maps = third_grade()
    baseline = (maps[0],maps[1],tuple(() for _ in maps[2]),maps[3])
    owner = PreparedStalkBoundary(source.archive,axes,baseline)
    slots = tuple((p,i,j) for p,(up,lo) in enumerate(owner.receipt['occurrence_cells'][2])
        for i in range(len(axes[1][lo])) for j in range(len(axes[2][up])))
    return source,axes,baseline,slots


def model():
    source, axes, baseline, slots = declaration()
    state = cpp_graded_stalk_transport_model(source,axes,baseline,slots,(0,1,2),grade=2)
    return source,state


def observation():
    return dict(input=((1,0),(0,0)),target=((Q(3,2),0),(0,), (0,0)),observed=(True,False,False))


def trained():
    source,state = model();batch=graded_stalk_transport_observation_batch(state,((1,0),(0,0)),((0,(Q(3,2),0)),))
    return source,train_model(state,source,batch),batch


def dense(shape, entries):
    out = [[Q(0) for _ in range(shape[1])] for _ in range(shape[0])]
    for i,j,q in entries:out[i][j]=q
    return out


def test_full_chain_constraints_transfer_an_observed_face_field_through_shared_boundaries():
    source,state,batch=trained();fit=state.payload['fit']['fit']
    assert {k for k,_,_ in fit['chain_coordinates']}=={2,3}
    assert fit['chain_rank']>0 and fit['observation_rank']==2 and fit['rank']>fit['chain_rank']
    output=infer_model(state,source,graded_stalk_transport_query(state,((0,0),(3,0))))
    assert output.grade==1 and output.execution_receipt['response']['boundary']==((Q(9,2),0),(Q(9,2),),(-Q(9,2),0))
    assert all(all(row) for row in output.execution_receipt['boundary_identified'])
    assert graded_stalk_transport_field(output,row=0).values.tolist()==[[Q(9,2),0]]
    assert state.step==1 and state.payload['observations'][0]['batch']==batch.coefficient_digest
    assert unpack_model(state.payload['observations'][0]['batch_checkpoint']).coefficient_digest==batch.coefficient_digest
    state.bind(source)


def test_every_origin_and_free_direction_satisfies_the_complete_original_sparse_equations():
    _,state,_=trained();fit=state.payload['fit']['fit'];a=dense(fit['design_shape'],fit['design']);k=dense(fit['kernel_shape'],fit['kernel'])
    origin=fit['particular'];target=fit['target']
    for i,row in enumerate(a):
        assert sum(q*p for q,p in zip(row,origin,strict=True))==target[i]
        for j in range(fit['kernel_shape'][1]):assert sum(q*k[r][j] for r,q in enumerate(row))==0
    assert fit['kernel_shape'][0]==len(state.configuration['parameter_slots'])


def test_unobserved_named_stalk_directions_retain_the_whole_affine_family():
    source,state,_=trained();output=infer_model(state,source,graded_stalk_transport_query(state,((0,1),(0,0))))
    assert any(not all(row) for row in output.execution_receipt['boundary_identified'])
    with pytest.raises(TypeError):output.tensor()
    with pytest.raises(ValueError,match='explicit'):graded_stalk_transport_maps(state,source)
    assert output.execution_receipt['response']['receipt']['fit']['kernel_shape']==state.payload['fit']['fit']['kernel_shape']


@pytest.mark.parametrize('value',[-3,Q(2,7),0])
def test_materialized_full_tower_keeps_nonzero_route_channels_and_zero_chain_products(value):
    source,state,_=trained();free=(value,)*len(state.payload['fit']['fit']['free_columns'])
    maps=graded_stalk_transport_maps(state,source,free_parameters=free)
    assert maps['flag_maps'][1]==state.configuration['fixed_maps'][1] and maps['flag_maps'][3]==state.configuration['fixed_maps'][3]
    trace=PreparedStalkBoundary(source.archive,state.configuration['stalk_axes'],maps['flag_maps']).apply(3,((2,0),),boundary_steps=3)
    assert any(any(any(row) for row in channel) for channel in trace['intermediate_channels'][0])
    assert all(not any(row) for row in trace['fields'][2]) and all(not any(row) for row in trace['fields'][3])
    assert trace['receipt']['occurrence_keys']==state.payload['fit']['source']['occurrence_keys']


def test_original_unseen_inactive_and_zero_occurrence_parameters_remain_in_the_family():
    source,axes,baseline,slots=declaration();owner=PreparedStalkBoundary(source.archive,axes,baseline)
    plan=owner.fit_transports(2,slots,(observation(),));receipt=plan.receipt
    zero={p for p,q in enumerate(receipt['source']['occurrence_coefficients'][2]) if not q}
    assert zero and all((p,i,j) in receipt['parameter_slots'] for p,i,j in slots if p in zero)
    free=receipt['fit']['free_columns']
    assert all(i in free for i,(p,_,_) in enumerate(slots) if p in zero)
    result=plan.materialize(free_parameters=(7,)*len(free))
    assert all(result['entry_values'][i]==7 for i,(p,_,_) in enumerate(slots) if p in zero)
    del source,owner
    assert plan.apply(((0,0),(2,0)))['boundary']==((3,0),(3,),(-3,0))


def test_supported_zero_incidence_keeps_its_declared_fixed_local_map_when_parameters_are_added():
    source,axes,baseline,slots=declaration();family=list(baseline[2]);family[3]=((0,0,7),)
    fixed=(*baseline[:2],tuple(family),baseline[3]);owner=PreparedStalkBoundary(source.archive,axes,fixed)
    plan=owner.fit_transports(2,slots,(observation(),));free=plan.receipt['fit']['free_columns']
    values=plan.materialize(free_parameters=(5,)*len(free))
    assert values['flag_maps'][2][3][0]==(0,0,12)
    assert values['receipt']['source']['flag_maps'][2][3]==((0,0,7),)
    assert PreparedStalkBoundary(source.archive,axes,values['flag_maps']).apply(2,((1,0),(0,0)))['fields'][-1]==((Q(3,2),0),(Q(3,2),),(-Q(3,2),0))


def test_batch_and_sequential_native_fit_retain_the_same_complete_chain_and_evidence_family():
    source,axes,baseline,slots=declaration();owner=PreparedStalkBoundary(source.archive,axes,baseline)
    samples=(observation(),dict(input=((0,1),(0,0)),target=((2,0),(0,), (0,0)),observed=(True,False,False)))
    parent=owner.fit_transports(2,slots);first=parent.append(samples[:1]);second=first.append(samples[1:])
    assert second.receipt==owner.fit_transports(2,slots,samples).receipt
    bad=dict(samples[1],target=((3,0),(0,),(0,0)))
    with pytest.raises(ValueError):second.append((bad,))
    assert parent.receipt['observations']==() and first.receipt['observations']==samples[:1]


def test_unrealized_local_map_addition_obeys_bit_admission_even_on_zero_incidence():
    source,axes,baseline,_=declaration();family=list(baseline[2]);family[3]=((0,0,7),)
    owner=PreparedStalkBoundary(source.archive,axes,(*baseline[:2],tuple(family),baseline[3]),limits=(4096,16000000,64000000,16))
    plan=owner.fit_transports(2,((3,0,0),))
    with pytest.raises((ValueError,RuntimeError)):plan.materialize(free_parameters=((1<<16)-1,))
    assert plan.materialize(free_parameters=(2,))['flag_maps'][2][3]==((0,0,9),)


@pytest.mark.parametrize('limits',[(4096,16000000,1,16384),(4096,1,64000000,16384),(3,16000000,64000000,16384)])
def test_admission_failure_does_not_publish_an_unsupported_graded_fit(limits):
    source,axes,baseline,slots=declaration()
    with pytest.raises((ValueError,RuntimeError)):
        cpp_graded_stalk_transport_model(source,axes,baseline,slots,(0,1,2),grade=2,limits=limits)


def test_named_dependent_parameter_templates_keep_aliases_and_unused_parameters():
    source,axes,baseline,slots=declaration();basis=tuple((i,j,q) for i,(_,r,c) in enumerate(slots) if r==c==0 for j,q in ((0,1),(1,2)))
    state=cpp_graded_stalk_transport_model(source,axes,baseline,slots,(0,1,2),grade=2,
        parameter_space=CoordinateSpace('original-transport-entries',('left','right','unseen')),parameter_basis=basis)
    batch=graded_stalk_transport_observation_batch(state,((1,0),(0,0)),((0,(Q(3,2),0)),))
    state=train_model(state,source,batch);fit=state.payload['fit']['fit']
    assert fit['rank']==1 and fit['free_columns']==(1,2) and fit['particular']==(Q(3,2),0,0)
    first=graded_stalk_transport_maps(state,source,free_parameters=(0,0));second=graded_stalk_transport_maps(state,source,free_parameters=(Q(2,7),11))
    assert first['parameter_values']!=second['parameter_values'] and first['flag_maps']==second['flag_maps']
    output=infer_model(state,source,graded_stalk_transport_query(state,((0,0),(2,0))))
    assert output.execution_receipt['response']['boundary']==((3,0),(3,),(-3,0))


def test_observed_fields_cannot_break_an_unqueried_chain_interface():
    source,state=model();batch=graded_stalk_transport_observation_batch(state,((1,0),(0,0)),((0,(1,0)),(1,(2,))))
    with pytest.raises(ValueError,match='inconsistent exact transport observations'):train_model(state,source,batch)
    assert state.step==0
    assert train_model(state,source,graded_stalk_transport_observation_batch(state,((1,0),(0,0)),((0,(1,0)),))).step==1


@pytest.mark.parametrize('change',['shape','mask','float','bool','extra','unobserved','empty','duplicate','fixed','grade','missing-grade'])
def test_malformed_original_transport_declarations_and_observations_reject_atomically(change):
    source,axes,baseline,slots=declaration();owner=PreparedStalkBoundary(source.archive,axes,baseline);sample=observation();grade=2
    if change=='shape':sample['input']=((1,), (0,0))
    elif change=='mask':sample['observed']=(1,0,0)
    elif change=='float':sample['input']=((0.5,0),(0,0))
    elif change=='bool':sample['input']=((True,0),(0,0))
    elif change=='extra':sample['extra']=()
    elif change=='unobserved':sample['target']=((Q(3,2),0),(1,),(0,0))
    elif change=='empty':sample['observed']=(False,False,False);sample['target']=((0,0),(0,),(0,0))
    elif change=='duplicate':slots=(*slots,slots[0])
    elif change=='fixed':grade=1;slots=((0,0,0),)
    elif change=='grade':grade=0
    elif change=='missing-grade':grade=4
    with pytest.raises((ValueError,TypeError)):owner.fit_transports(grade,slots,(sample,))
    assert owner.fit_transports(2,declaration()[3],(observation(),)).apply(((0,0),(2,0)))['boundary']==((3,0),(3,),(-3,0))


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_complete_fit_source_original_batches_query_and_output_roundtrip_and_resume(tmp_path,suffix):
    source,state,_=trained();query=graded_stalk_transport_query(state,((0,0),(3,0)));output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'query',query);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('graded'+suffix);save_model(path,record);loaded=load_model(path)
    from rexgraph.cpp_graded_stalk_transport import _cache
    _cache.clear();restored,native=read_model_record(loaded)
    assert restored.coefficient_digest==state.coefficient_digest and native.archive==source.archive
    assert infer_model(restored,native,loaded.get_metadata(1,0,'query')).coefficient_digest==output.coefficient_digest
    assert loaded.get_metadata(1,0,'output').coefficient_digest==output.coefficient_digest
    next_batch=graded_stalk_transport_observation_batch(restored,((0,1),(0,0)),((0,(2,0)),))
    assert train_model(restored,native,next_batch).step==2


@pytest.mark.parametrize('change',['particular','kernel','chain','observation','checkpoint','source','output'])
def test_cold_replay_rejects_changed_evidence_or_chain_family(change):
    source,state,_=trained();payload=thaw_tree(state.payload);fit=payload['fit']['fit']
    if change=='particular':fit['particular']=(fit['particular'][0]+1,*fit['particular'][1:])
    elif change=='kernel':fit['kernel']=fit['kernel'][1:]
    elif change=='chain':fit['chain']=fit['chain'][1:]
    elif change=='observation':payload['observations'][0]['target']=((5,0),(0,),(0,0))
    elif change=='checkpoint':payload['observations'][0]['batch_checkpoint']=pack_model(state)
    elif change=='source':payload['fit']['source']['occurrence_keys']=((),(),(),())
    else:
        output=infer_model(state,source,graded_stalk_transport_query(state,((0,0),(3,0))))
        values=np.array(output.values,copy=True);values[0,0,0]+=1
        with pytest.raises(ValueError):replace(output,values=values)
        return
    from rexgraph.cpp_graded_stalk_transport import _cache
    _cache.clear()
    with pytest.raises((ValueError,TypeError)):infer_model(replace(state,payload=payload),source)


def test_concurrent_immutable_children_and_uninitialized_handles():
    source,axes,baseline,slots=declaration();owner=PreparedStalkBoundary(source.archive,axes,baseline);parent=owner.fit_transports(2,slots)
    with ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(lambda _:parent.append((observation(),)).receipt,range(6)))
    assert all(v==values[0] for v in values) and parent.receipt['observations']==()
    with pytest.raises(RuntimeError,match='immutable'):parent.__init__(owner,2,slots)
    empty=PreparedGradedStalkTransport.__new__(PreparedGradedStalkTransport)
    with pytest.raises(RuntimeError,match='not initialized'):empty.apply(())


@pytest.mark.parametrize('grade',[1,3])
def test_other_original_transport_grades_use_the_same_full_field_learning_contract(grade):
    source,axes,maps=third_grade();baseline=tuple(tuple(() for _ in family) if k==grade else family for k,family in enumerate(maps))
    owner=PreparedStalkBoundary(source.archive,axes,baseline)
    slots=tuple((p,i,j) for p,(up,lo) in enumerate(owner.receipt['occurrence_cells'][grade])
        for i in range(len(axes[grade-1][lo])) for j in range(len(axes[grade][up])))
    plan=owner.fit_transports(grade,slots);true=plan.materialize(free_parameters=(Q(2,7),)*len(plan.receipt['fit']['free_columns']))
    control=PreparedStalkBoundary(source.archive,axes,true['flag_maps'])
    field=tuple(tuple(1 if i==j==0 else 0 for j in range(len(row))) for i,row in enumerate(axes[grade]))
    target=control.apply(grade,field)['fields'][-1]
    state=cpp_graded_stalk_transport_model(source,axes,baseline,slots,tuple(range(len(target))),grade=grade)
    batch=graded_stalk_transport_observation_batch(state,field,tuple(enumerate(target)))
    state=train_model(state,source,batch);query=tuple(tuple(3*v for v in row) for row in field)
    output=infer_model(state,source,graded_stalk_transport_query(state,query))
    assert output.grade==grade-1
    assert output.execution_receipt['response']['boundary']==tuple(tuple(3*v for v in row) for row in target)
    assert all(all(row) for row in output.execution_receipt['boundary_identified'])


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
@pytest.mark.parametrize('affine_completion',[False,True])
def test_cold_five_package_training_inference_and_query_creation(tmp_path,backend,affine_completion):
    from rcdb import open_store
    if affine_completion:
        from rexgraph.cpp_graded_stalk_transport import cpp_affine_stalk_transport_model
        source,axes,maps,slots,grade=affine_declaration()
        initial=cpp_affine_stalk_transport_model(source,axes,maps,slots,(0,1,2),grade=grade)
        chosen=graded_stalk_transport_maps(initial,source,free_parameters=(Q(0),)*4)
        target=PreparedStalkBoundary(source.archive,axes,chosen['flag_maps']).apply(2,((1,),(1,)))['fields'][-1]
        batch=graded_stalk_transport_observation_batch(initial,((1,),(1,)),tuple(enumerate(target)))
        query=graded_stalk_transport_query(initial,((3,),(3,)))
    else:
        source,initial=model();batch=graded_stalk_transport_observation_batch(initial,((1,0),(0,0)),((0,(Q(3,2),0)),))
        query=graded_stalk_transport_query(initial,((0,0),(3,0)))
    expected=train_model(initial,source,batch);record=model_record(source,initial);record.attach_metadata(1,0,'batch',batch)
    record.attach_metadata(1,0,'query',query)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('graded',record,expected_version=0);assert store.verify_commits('graded');store.close()
    script='''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import train_model,infer_model
from rexgraph.cpp_graded_stalk_transport import graded_stalk_transport_query
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]);record=store.read_record('graded',version=1).value
state,source=read_model_record(record);batch=record.get_metadata(1,0,'batch')
query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'b':batch,'q':query}).execute(parse(
 'FROM $r RETURN MODEL_TRAIN(MODEL_STATE(),$b),MODEL_INFO(MODEL_STATE()),MODEL_INFER(MODEL_TRAIN(MODEL_STATE(),$b),$q)'))
child=result.values[0]
assert child.coefficient_digest==sys.argv[2] and child.step==1
assert child.adapter_version==sys.argv[3]
assert result.values[1]['strict_chain_family'] and result.values[1]['input_grade']==2
assert json_value(child)['strict_chain_family'] and json_value(child)['chain_rank']>0
assert result.values[2].grade==1 and result.exactness[2].value=='rational'
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('graded');hive.add_worker('graded',lambda b,**kw:train_model(state,source,b),capability='transform')
 assert hive.invoke('graded',batch,record=False).coefficient_digest==sys.argv[2]
output=infer_model(child,source,query)
assert all(all(row) for row in output.execution_receipt['boundary_identified'])
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    process=subprocess.run([sys.executable,'-B','-I','-c',script,url,expected.coefficient_digest,initial.adapter_version],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr


@pytest.mark.parametrize('affine_completion',[False,True])
def test_rcql_can_create_the_versioned_graded_transport_model(affine_completion):
    from rcql import Executor,parse
    if affine_completion:
        from rexgraph.cpp_graded_stalk_transport import cpp_affine_stalk_transport_model
        source,axes,maps,slots,grade=affine_declaration()
        state=cpp_affine_stalk_transport_model(source,axes,maps,slots,(0,1,2),grade=grade)
    else:
        source,state=model()
    record=model_record(source,state);c=state.configuration
    config={k:thaw_tree(c[k]) for k in ('stalk_axes','fixed_maps','parameter_slots','training_rows','grade','limits','max_bytes')}
    config['affine_completion']=affine_completion
    result=Executor(sources={'r':record},params={'config':config}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_graded_stalk_transport",$config)'))
    created=result.values[0]
    assert created.adapter=='cpp_graded_stalk_transport' and created.adapter_version==('2' if affine_completion else '1')
    assert created.payload['fit']==state.payload['fit'] and created.step==0
    config['affine_completion']=1
    with pytest.raises(TypeError,match='explicit boolean'):
        Executor(sources={'r':record},params={'config':config}).execute(parse(
            'FROM $r RETURN MODEL_INIT("cpp_graded_stalk_transport",$config)'))


@pytest.mark.parametrize('named',[False,True])
def test_incremental_appends_reuse_chain_and_assemble_only_new_fields(named):
    source,axes,fixed,slots=declaration();owner=PreparedStalkBoundary(source.archive,axes,fixed)
    options=dict(parameter_space=dict(name='exact-routes',keys=('a','alias','unseen')),
        parameter_basis=tuple((i,j,Q(1,j+1)) for i,(_,r,c) in enumerate(slots) if r==c==0 for j in (0,1))) if named else {}
    parent=owner.fit_transports(2,slots,**options);samples=(observation(),)*3
    prefix=parent.append(samples[:2]);child=prefix.append(samples[2:]);replay=owner.fit_transports(2,slots,samples,**options)
    assert child.receipt==replay.receipt==parent.append(samples).receipt
    assert child.execution_stats['chain_reused'] and child.execution_stats['source_reused']
    assert child.execution_stats['reused_observations']==2 and child.execution_stats['assembled_observations']==1
    assert not replay.execution_stats['chain_reused'] and replay.execution_stats['assembled_observations']==3
    assert child.execution_stats['guard_steps']<replay.execution_stats['guard_steps']
    assert parent.receipt['observations']==() and len(prefix.receipt['observations'])==2
    for field in (((1,0),(0,0)),((0,1),(0,0)),((0,0),(Q(2,7),0))):assert child.apply(field)==replay.apply(field)


def test_incremental_replay_reuses_native_rows_without_calling_cold_initialization(monkeypatch):
    import rexgraph.cpp_graded_stalk_transport as implementation
    source,state=model()
    def forbidden(*args,**kwargs):raise AssertionError('old source and fields were rebuilt')
    monkeypatch.setattr(implementation,'_native',forbidden)
    child=train_model(state,source,graded_stalk_transport_observation_batch(state,((1,0),(0,0)),((0,(Q(3,2),0)),)))
    assert child.step==1 and state.step==0
    assert implementation._cache[child.coefficient_digest].execution_stats['chain_reused']


def test_incremental_conflict_and_malformed_suffix_leave_parent_reusable():
    source,axes,fixed,slots=declaration();owner=PreparedStalkBoundary(source.archive,axes,fixed)
    parent=owner.fit_transports(2,slots,(observation(),));before=parent.receipt
    bad=dict(observation(),target=((9,0),(0,),(0,0)))
    with pytest.raises(ValueError):parent.append((bad,))
    with pytest.raises(TypeError):parent.append((dict(observation(),input=((True,0),(0,0))),))
    with pytest.raises(ValueError):parent.append(())
    assert parent.receipt==before
    assert parent.append((observation(),)).receipt==owner.fit_transports(2,slots,(observation(),)*2).receipt


def test_incremental_observation_generator_stops_at_declared_admission():
    source,axes,fixed,slots=declaration()
    parent=PreparedStalkBoundary(source.archive,axes,fixed,limits=(32,16000000,64000000,16384)).fit_transports(2,slots)
    consumed=[]
    def stream():
        while True:
            consumed.append(None);yield observation()
    with pytest.raises(ValueError,match='observation count'):parent.append(stream())
    assert len(consumed)==33 and parent.receipt['observations']==()


def test_incremental_children_keep_owned_chain_after_original_handles_are_released():
    source,axes,fixed,slots=declaration();owner=PreparedStalkBoundary(source.archive,axes,fixed)
    parent=owner.fit_transports(2,slots);child=parent.append((observation(),));wanted=child.receipt
    del source,owner,parent
    with ThreadPoolExecutor(max_workers=4) as pool:
        descendants=list(pool.map(lambda _:child.append((observation(),)),range(4)))
    assert all(v.receipt==descendants[0].receipt and v.execution_stats['chain_reused'] for v in descendants)
    assert child.receipt==wanted and child.apply(((0,0),(3,0)))['boundary']==((Q(9,2),0),(Q(9,2),),(-Q(9,2),0))


def affine_declaration(paired=True, *, unrepairable=False):
    """The native affine completion fixture with a retained nonflat fixed leg."""
    from rexgraph_cpp import NativeSource
    columns = [(1, ((10, -1), (20, 1))), (1, ((20, -1), (30, 1))),
               (1, ((10, -1), (30, 1)))]
    face = ((1, 1), (2, 1), (3, -1), (2, 1), (2, -1))
    columns.append((2, face))
    if paired:
        columns.extend(((2, face), (3, ((4, 1), (5, -1)))))
    relations = []; key = 0
    for i, (grade, column) in enumerate(columns):
        occurrences = []
        for lower, coefficient in column:
            key += 1
            occurrences.append(dict(key=key, lower=lower, exists=True, negative=coefficient < 0,
                                    magnitude=abs(coefficient), aspects=(0,)))
        relations.append(dict(grade=grade, key=i+1, type=0, occurrences=tuple(occurrences)))
    sizes = (3, 3, 2, 1) if paired else (3, 3, 1)
    source = NativeSource.from_declaration(dict(profile='declared-role-source-v1', namespace=941191,
        semantic_scale=0, max_grade=len(sizes)-1, role_aspects=('retained-boundary',),
        types=(dict(name='declared-column', min_arity=0, max_arity=0),), witnesses=(10,20,30),
        relations=tuple(relations), fields=(), metrics=(),
        carriers=tuple(dict(key=10+k, kind='source_grade', grade=k, coordinates=0, size=n)
                       for k, n in enumerate(sizes))))
    axes = tuple(tuple(('field',) for _ in range(n)) for n in sizes)
    selected = 1 if unrepairable else 2
    maps = [()]; slots = []
    for grade in range(1, len(sizes)):
        blocks = []; occurrence = 0
        for relation in relations:
            if relation['grade'] != grade: continue
            for _ in relation['occurrences']:
                if grade == selected:
                    fixed = not unrepairable and occurrence == 0
                    blocks.append(((0,0,1),) if fixed else ())
                    if not fixed: slots.append((occurrence,0,0))
                else:
                    absent = unrepairable and grade == 2 and relation['key'] == 5
                    blocks.append(() if absent else ((0,0,1),))
                occurrence += 1
        maps.append(tuple(blocks))
    return source, axes, tuple(maps), tuple(slots), selected


@pytest.mark.parametrize('paired', [False, True])
def test_nonflat_declaration_retains_rhs_and_native_affine_correlations(paired):
    from rexgraph_cpp import PreparedStalkDeclaration
    source, axes, maps, slots, grade = affine_declaration(paired)
    with pytest.raises(ValueError, match='complete strict chain'):
        PreparedStalkBoundary(source.archive, axes, maps)
    declaration = PreparedStalkDeclaration(source.archive, axes, maps)
    assert not declaration.receipt['strict_chain_family']
    assert not hasattr(declaration, 'apply') and not hasattr(declaration, 'inject_acquisition')
    plan = declaration.fit_transports(grade, slots); fit = plan.receipt['fit']
    chain_rows = fit['chain_shape'][0]
    target = fit['target'][:chain_rows]
    assert any(target) and len(fit['free_columns']) == (4 if paired else 2)
    reconstructed = [Q(0)] * chain_rows
    for row, column, value in fit['chain']:
        reconstructed[row] += value * fit['particular'][column]
    assert tuple(reconstructed) == target
    del source, declaration
    for parameter in (Q(-2,3), Q(0), Q(2,3)):
        chosen = plan.materialize(free_parameters=(parameter,)*len(fit['free_columns']))
        # A strict executor independently certifies all adjacent products.
        original, _, _, _, _ = affine_declaration(paired)
        completed = PreparedStalkBoundary(original.archive, axes, chosen['flag_maps'])
        assert completed.apply(2, tuple((1,) for _ in axes[2]), boundary_steps=2)['fields'][-1] == ((0,),(0,),(0,))
        if paired:
            assert completed.apply(3, ((1,),), boundary_steps=2)['fields'][-1] == ((0,),(0,),(0,))
        assert chosen['flag_maps'][2][0] == ((0,0,1),)


def test_nonflat_affine_parent_append_and_replay_preserve_complete_target():
    from rexgraph_cpp import PreparedStalkDeclaration
    source, axes, maps, slots, grade = affine_declaration()
    declaration = PreparedStalkDeclaration(source.archive, axes, maps)
    parent = declaration.fit_transports(grade, slots); before = parent.receipt
    chosen = parent.materialize(free_parameters=(Q(0),)*len(before['fit']['free_columns']))
    executor = PreparedStalkBoundary(source.archive, axes, chosen['flag_maps'])
    field = ((1,), (1,)); target = executor.apply(2, field)['fields'][-1]
    observation = dict(input=field, target=target, observed=(True,True,True))
    child = parent.append((observation,)); replay = declaration.fit_transports(grade, slots, (observation,))
    assert parent.receipt == before and parent.receipt['observations'] == ()
    for name in ('chain','target','particular','kernel','free_columns','chain_coordinates'):
        assert child.receipt['fit'][name] == replay.receipt['fit'][name]
    rows = before['fit']['chain_shape'][0]
    assert child.receipt['fit']['target'][:rows] == before['fit']['target'][:rows]
    assert child.execution_stats['chain_reused'] and child.receipt['strict_chain_family']
    bad = dict(observation, target=((target[0][0]+1,),*target[1:]))
    with pytest.raises(ValueError): child.append((bad,))
    assert parent.receipt == before


@pytest.mark.parametrize('suffix', ['.rcbd', '.safetensors'])
def test_nonflat_model_version_cold_codec_replay_and_full_output(tmp_path, suffix):
    from rexgraph.cpp_graded_stalk_transport import (
        cpp_affine_stalk_transport_model, replay_graded_stalk_transport_model, _cache,
    )
    source, axes, maps, slots, grade = affine_declaration()
    with pytest.raises(ValueError, match='complete strict chain'):
        cpp_graded_stalk_transport_model(source, axes, maps, slots, (0,1,2), grade=grade)
    state = cpp_affine_stalk_transport_model(source, axes, maps, slots, (0,1,2), grade=grade)
    assert state.adapter_version == '2' and state.configuration['execution_provider']['version'] == '2'
    fit = state.payload['fit']['fit']; rhs = fit['target'][:fit['chain_shape'][0]]
    assert any(rhs) and len(fit['free_columns']) == 4
    chosen = graded_stalk_transport_maps(state, source, free_parameters=(Q(0),)*4)
    executor = PreparedStalkBoundary(source.archive, axes, chosen['flag_maps'])
    field = ((1,), (1,)); target = executor.apply(2, field)['fields'][-1]
    batch = graded_stalk_transport_observation_batch(state, field, tuple(enumerate(target)))
    child = train_model(state, source, batch)
    assert state.step == 0 and child.parent == state.coefficient_digest
    query = graded_stalk_transport_query(child, field); output = infer_model(child, source, query)
    record = model_record(source, child); record.attach_metadata(1,0,'output',output); record.attach_metadata(1,0,'query',query)
    path = tmp_path/('affine'+suffix); save_model(path, record); _cache.clear()
    loaded = load_model(path); restored, native = read_model_record(loaded)
    assert restored.payload['fit']['fit']['target'][:len(rhs)] == rhs
    assert infer_model(restored,native,loaded.get_metadata(1,0,'query')).coefficient_digest == output.coefficient_digest
    assert loaded.get_metadata(1,0,'output').coefficient_digest == output.coefficient_digest
    replay = replay_graded_stalk_transport_model(restored, native)
    assert replay.adapter_version == '2' and replay.payload['fit']['fit']['target'] == restored.payload['fit']['fit']['target']
    script = '''
import sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_graded_stalk_transport import replay_graded_stalk_transport_model
record=load_model(sys.argv[1]);state,source=read_model_record(record)
rhs=state.payload['fit']['fit']['target'][:state.payload['fit']['fit']['chain_shape'][0]]
assert any(rhs) and state.adapter_version=='2'
assert infer_model(state,source,record.get_metadata(1,0,'query')).coefficient_digest==sys.argv[2]
assert replay_graded_stalk_transport_model(state,source).adapter_version=='2'
'''
    child_process = subprocess.run([sys.executable,'-B','-I','-c',script,str(path),output.coefficient_digest],
                                   cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert child_process.returncode == 0, child_process.stderr
    changed = thaw_tree(restored.payload)
    changed['fit']['fit']['target'] = tuple(Q(0) for _ in changed['fit']['fit']['target'])
    _cache.clear()
    with pytest.raises(ValueError): infer_model(replace(restored,payload=changed),native,query)


def test_nonflat_unrepairable_fixed_defect_is_not_published_as_a_fit():
    from rexgraph_cpp import PreparedStalkDeclaration
    source, axes, maps, slots, grade = affine_declaration(unrepairable=True)
    declaration = PreparedStalkDeclaration(source.archive, axes, maps)
    with pytest.raises(ValueError): declaration.fit_transports(grade, slots)
    assert not declaration.receipt['strict_chain_family']
    with pytest.raises(RuntimeError,match='immutable'): declaration.__init__(source.archive,axes,maps)
    empty = PreparedStalkDeclaration.__new__(PreparedStalkDeclaration)
    with pytest.raises(RuntimeError,match='not initialized'): empty.fit_transports(grade,slots)
