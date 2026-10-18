"""Exact learned incidence maps, affine ambiguity and complete package replay."""
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import numpy as np
import pytest

from rexgraph_cpp import NativeSource, PreparedStalkTransport, PreparedStalkBoundary, PreparedStalkFields
from rexgraph.cpp_stalk_transport import (
    cpp_stalk_transport_model, stalk_transport_observation_batch, stalk_transport_query,
    stalk_transport_field, stalk_transport_maps, enable_stalk_transport_provider,
)
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import ModelBatch, thaw_tree

AXES = ((('meaning','register'),('grammar',)), (('sense','role','diction'),))
SLOTS = tuple((0,r,c) for r in range(2) for c in range(3))+tuple((1,0,c) for c in range(3))
TRUTH = (Q(2),Q(1,3),Q(-1),Q(-2,5),Q(0),Q(7),Q(1),Q(2),Q(-3,2))


def declaration(flags=((10,Q(-1,2),True),(20,Q(1,3),True)), witnesses=(10,20), namespace=994116):
    return dict(profile='declared-role-source-v1',namespace=namespace,semantic_scale=0,max_grade=1,
        role_aspects=('original-boundary',),types=(dict(name='field-observation',min_arity=0,max_arity=0),),
        witnesses=witnesses,relations=(dict(grade=1,key=1,type=0,occurrences=tuple(
            dict(key=i+1,lower=lo,negative=q<0,magnitude=abs(q),exists=seen,aspects=(0,))
            for i,(lo,q,seen) in enumerate(flags))),),fields=(),metrics=(),
        carriers=(dict(key=10,kind='source_grade',grade=0,coordinates=0,size=len(witnesses)),
                  dict(key=11,kind='source_grade',grade=1,coordinates=0,size=1)))


def model(training=(0,1)):
    source=NativeSource.from_declaration(declaration())
    return source,cpp_stalk_transport_model(source,AXES,((),((),())),SLOTS,training)


def truth(field):
    x=field[0]
    return (tuple(-Q(1,2)*sum(TRUTH[3*r+c]*x[c] for c in range(3)) for r in range(2)),
            (Q(1,3)*sum(TRUTH[6+c]*x[c] for c in range(3)),))


def acquire(state,source,field,rows=(0,1)):
    target=truth(field)
    return train_model(state,source,stalk_transport_observation_batch(state,field,tuple((r,target[r]) for r in rows)))


@pytest.fixture
def learned():
    source,state=model()
    for field in (((1,0,0),),((0,1,0),),((0,0,1),)):
        state=acquire(state,source,field)
    return source,state


def test_basis_observations_recover_complete_rectangular_rational_maps(learned):
    _,state=learned;fit=state.payload['fit']
    assert fit['particular']==TRUTH and fit['rank']==9 and fit['kernel']==()
    assert fit['parameter_slots']==SLOTS
    assert fit['occurrence_coefficients']==((),(Q(-1,2),Q(1,3)))
    assert fit['occurrence_keys']==((),(1,2))
    assert fit['constraints']==tuple((sample,cell,j) for sample in range(3) for cell,n in enumerate((2,1)) for j in range(n))


@pytest.mark.parametrize('field',[((Q(2,3),Q(-7,5),Q(11,13)),),((0,0,0),),((2,-3,5),)])
def test_unseen_mixed_fields_use_the_learned_native_maps(learned,field):
    source,state=learned;output=infer_model(state,source,stalk_transport_query(state,field))
    assert output.execution_receipt['response']['boundary_family']==(truth(field),)
    assert output.execution_receipt['response']['occurrence_family']==(truth(field),)
    assert output.execution_receipt['boundary_identified']==((True,True),(True,))
    assert output.axes[-1].keys==('origin',)
    assert stalk_transport_field(output,0).values.tolist()==[list(truth(field)[0])]
    with pytest.raises(TypeError,match='ragged'):output.tensor()


def test_unsupported_padding_is_separate_from_learned_zero(learned):
    source,state=learned;output=infer_model(state,source,stalk_transport_query(state,((0,1,0),)))
    assert state.payload['fit']['particular'][4]==0
    assert output.execution_receipt['stalk_support']==((True,True),(True,False))
    assert output.values[1,1,0]==0 and output.values[0,1,0]==0


def test_rank_deficiency_identifies_training_span_without_filling_unknown_directions():
    source,state=model();state=acquire(state,source,((1,0,0),))
    assert state.payload['fit']['rank']==3 and state.payload['fit']['kernel_shape']==(9,6)
    known=infer_model(state,source,stalk_transport_query(state,((Q(5,7),0,0),)))
    unknown=infer_model(state,source,stalk_transport_query(state,((0,1,0),)))
    assert known.execution_receipt['boundary_identified']==((True,True),(True,))
    assert unknown.execution_receipt['boundary_identified']==((False,False),(False,))
    assert len(unknown.execution_receipt['response']['boundary_family'])==4
    assert len(unknown.execution_receipt['response']['zero_components'])==3
    with pytest.raises(TypeError,match='unidentified'):stalk_transport_field(unknown,0)
    assert stalk_transport_field(known,0).values.tolist()==[list(truth(((Q(5,7),0,0),))[0])]


def test_unobserved_stalk_retains_its_entire_map_family():
    source,state=model(training=(0,))
    for field in (((1,0,0),),((0,1,0),),((0,0,1),)):
        state=acquire(state,source,field,rows=(0,))
    assert state.payload['fit']['rank']==6 and state.payload['fit']['free_columns']==(6,7,8)
    output=infer_model(state,source,stalk_transport_query(state,((1,2,3),)))
    assert output.execution_receipt['boundary_identified']==((True,True),(False,))
    assert stalk_transport_field(output,0).values.tolist()==[list(truth(((1,2,3),))[0])]
    with pytest.raises(TypeError,match='unidentified'):stalk_transport_field(output,1)


def test_repeated_occurrences_retain_ambiguity_even_when_the_boundary_is_identified():
    source=NativeSource.from_declaration(declaration(((10,-1,True),(10,1,True)),(10,)))
    axes=((('meaning',),),(('sense',),));slots=((0,0,0),(1,0,0))
    state=cpp_stalk_transport_model(source,axes,((),((),())),slots,(0,))
    state=train_model(state,source,stalk_transport_observation_batch(state,((1,),),((0,(2,)),)))
    assert state.payload['fit']['rank']==1 and state.payload['fit']['kernel']==((0,0,1),(1,0,1))
    output=infer_model(state,source,stalk_transport_query(state,((3,),)))
    assert output.execution_receipt['boundary_identified']==((True,),)
    assert output.execution_receipt['occurrence_identified']==((False,),(False,))
    assert output.execution_receipt['response']['occurrence_family'][1]==((-3,),(3,))
    assert output.tensor().values.tolist()==[[6]]


def test_inactive_and_supported_zero_flags_keep_underlying_parameters_unidentified():
    source=NativeSource.from_declaration(declaration(((10,1,True),(10,7,False),(10,0,True)),(10,)))
    axes=((('meaning',),),(('sense',),));slots=tuple((i,0,0) for i in range(3))
    state=cpp_stalk_transport_model(source,axes,((),((),(),())),slots,(0,))
    state=train_model(state,source,stalk_transport_observation_batch(state,((1,),),((0,(4,)),)))
    fit=state.payload['fit']
    assert fit['rank']==1 and fit['free_columns']==(1,2)
    assert fit['occurrence_activity']==((),(True,False,True))
    assert fit['occurrence_coefficients']==((),(1,0,0))
    assert fit['kernel']==((1,0,1),(2,1,1))
    output=infer_model(state,source,stalk_transport_query(state,((2,),)))
    assert output.execution_receipt['response']['occurrence_family'][0]==((8,),(0,),(0,))


def test_sparse_unidentified_kernel_and_zero_query_do_not_expand_square_coefficients():
    source=NativeSource.from_declaration(declaration(((10,1,True),),(10,)))
    axes=((('meaning',),),(tuple('axis:'+str(i) for i in range(512)),))
    slots=tuple((0,0,i) for i in range(512))
    plan=PreparedStalkTransport(source.archive,axes,((),((),)),slots)
    assert plan.receipt['kernel_shape']==(512,512) and len(plan.receipt['kernel'])==512
    response=plan.apply(((0,)*512,))
    assert response['components']==('origin',) and len(response['zero_components'])==512
    assert response['boundary_family']==(((0,),),) and response['occurrence_family']==(((0,),),)


def test_conflicting_observation_rejects_without_mutating_the_parent():
    source,state=model();state=acquire(state,source,((1,0,0),));digest=state.coefficient_digest
    batch=stalk_transport_observation_batch(state,((1,0,0),),((0,(999,999)),))
    with pytest.raises(ValueError,match='inconsistent'):train_model(state,source,batch)
    assert state.coefficient_digest==digest and state.step==1
    child=acquire(state,source,((0,1,0),));assert child.step==2 and child.parent==digest


def test_redundant_observations_are_retained_without_claiming_new_rank():
    source,state=model();state=acquire(state,source,((1,0,0),));state=acquire(state,source,((1,0,0),))
    assert state.step==2 and len(state.payload['observations'])==2 and state.payload['fit']['rank']==3
    assert len(state.payload['fit']['constraints'])==6


def test_fixed_zero_and_unknown_coordinate_cannot_share_a_slot():
    source,_=model()
    with pytest.raises(ValueError,match='disjoint'):
        cpp_stalk_transport_model(source,AXES,((),(((0,0,0),),())),SLOTS,(0,1))


@pytest.mark.parametrize('value',[True,0.5,'1/2'])
def test_exact_transport_observations_reject_implicit_conversions(value):
    source,state=model()
    with pytest.raises(TypeError):stalk_transport_observation_batch(state,((value,0,0),),((0,(0,0)),))


@pytest.mark.parametrize('change',['slots','input','target','mask','unknown_target','steps','bits','entries'])
def test_native_admissions_and_whole_source_support_are_enforced(change):
    source,_=model();slots=SLOTS
    o=dict(input=((1,0,0),),target=((1,2),(3,)),observed=(True,True));limits=(4096,16000000,64000000,16384)
    if change=='slots':slots=(*slots,(2,0,0))
    elif change=='input':o['input']=((1,0),)
    elif change=='target':o['target']=((1,), (3,))
    elif change=='mask':o['observed']=(1,True)
    elif change=='unknown_target':o['observed']=(True,False)
    elif change=='steps':limits=(4096,16000000,1,16384)
    elif change=='bits':o['input']=((1<<20,0,0),);limits=(4096,16000000,64000000,8)
    else:limits=(4096,4,64000000,16384)
    with pytest.raises((ValueError,TypeError,OverflowError,IndexError,RuntimeError)):
        PreparedStalkTransport(source.archive,AXES,((),((),())),slots,(o,),limits=limits)


def test_held_out_partition_and_unlisted_training_stalk_cannot_acquire():
    source,state=model(training=(0,));batch=stalk_transport_observation_batch(state,((1,0,0),),((0,(1,2)),))
    wrong=ModelBatch(batch.source,batch.space,batch.targets,batch.observed,dict(batch.inputs,partition='test'))
    with pytest.raises(ValueError,match='training'):train_model(state,source,wrong)
    with pytest.raises(ValueError,match='training'):stalk_transport_observation_batch(state,((1,0,0),),((1,(2,)),))


def test_standard_exact_model_batches_retain_original_encoding_and_replay():
    source,state=model(training=(0,))
    batch=ModelBatch(state.source,state.space,np.asarray([[2,3],[0,0]],dtype=np.int64),np.asarray([True,False]),
        dict(profile='whole-stalk-transport-observation-v1',partition='training',input=((1,0,0),)))
    state=train_model(state,source,batch)
    from rexgraph.cpp_stalk_transport import _cache
    restored=unpack_model(pack_model(state));_cache.clear();enable_stalk_transport_provider().prepare(restored,source)
    assert restored.payload['observations'][0]['batch']==batch.coefficient_digest
    assert restored.payload['fit']['rank']==2


def test_rcql_creates_trains_and_evaluates_the_native_transport_profile():
    from rcql import Executor,parse
    source,state=model();record=model_record(source,state)
    field=((1,0,0),);batch=stalk_transport_observation_batch(state,field,((0,truth(field)[0]),))
    config=dict(stalk_axes=AXES,fixed_maps=((),((),())),parameter_slots=SLOTS,training_rows=(0,1))
    result=Executor(sources={'r':record},params={'c':config,'b':batch,'q':stalk_transport_query(state,field)}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_stalk_transport",$c),MODEL_INFER(MODEL_TRAIN(MODEL_STATE(),$b),$q)'))
    assert result.values[0].adapter=='cpp_stalk_transport' and result.values[0].payload['fit']['rank']==0
    assert result.values[1].execution_receipt['boundary_identified']==((True,True),(False,))
    assert result.exactness[1].value=='rational'


def test_native_stalk_input_accepts_exact_numpy_integer_coefficients():
    source,_=model();o=dict(input=((np.int64(1),0,0),),target=((np.int64(2),3),(0,)),observed=(True,False))
    plan=PreparedStalkTransport(source.archive,AXES,((),((),())),SLOTS,(o,))
    assert plan.receipt['rank']==2
    assert plan.apply(((np.int64(2),0,0),))['boundary_family'][0]==((4,6),(0,))


def test_shared_scalar_validation_accepts_numpy_integers_on_existing_stalk_paths():
    from rexgraph.cpp_stalk_fields import _exact
    assert _exact(np.int64(-3),64)==-3
    assert type(_exact(np.int64(-3),64).numerator) is int
    source,_=model();maps=((),(((0,0,np.int64(2)),),()))
    plan=PreparedStalkBoundary(source.archive,AXES,maps)
    assert plan.apply(1,((np.int64(3),0,0),))['fields'][-1]==((-3,0),(0,))


@pytest.mark.parametrize('field',['rank','particular','batch','receipt'])
def test_retained_fit_and_batch_tampering_fail_native_replay(learned,field):
    source,state=learned;payload=thaw_tree(state.payload)
    if field=='rank':payload['fit']['rank']=8
    elif field=='particular':payload['fit']['particular']=(999,*payload['fit']['particular'][1:])
    elif field=='batch':payload['observations'][0]['batch']='0'*64
    else:payload['fit']['preparation_steps']+=1
    with pytest.raises(ValueError):
        bad=replace(state,payload=payload);enable_stalk_transport_provider().prepare(bad,source)


def test_corrupted_affine_response_is_rejected(learned):
    source,state=learned;output=infer_model(state,source,stalk_transport_query(state,((1,2,3),)))
    receipt=thaw_tree(output.execution_receipt);receipt['boundary_identified']=((False,True),(True,))
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)


def test_learned_maps_feed_existing_native_boundary_and_green_actions(learned):
    source,state=learned;maps=stalk_transport_maps(state,source)['flag_maps'];field=((2,-3,5),)
    trace=PreparedStalkBoundary(source.archive,AXES,maps).apply(1,field)
    assert trace['fields'][-1]==truth(field)
    green=PreparedStalkFields(source.archive,AXES[0],AXES[1],maps[1])
    response=green.apply((field,))
    # The Green response uses the full learned rectangular blocks.
    assert response['boundaries'][0]==PreparedStalkBoundary(source.archive,AXES,maps).apply(1,response['fields'][0])['fields'][-1]
    assert response['occurrence_fields'][0]==PreparedStalkBoundary(source.archive,AXES,maps).apply(1,response['fields'][0])['occurrence_fields'][0]


def test_materialization_requires_every_free_parameter_and_retains_supported_zero():
    source,state=model();state=acquire(state,source,((1,0,0),))
    with pytest.raises(ValueError,match='explicit'):stalk_transport_maps(state,source)
    with pytest.raises(ValueError,match='complete'):stalk_transport_maps(state,source,free_parameters=(0,))
    materialized=stalk_transport_maps(state,source,free_parameters=(0,)*6)
    assert len(materialized['flag_maps'][1][0])==6 and len(materialized['flag_maps'][1][1])==3
    trace=PreparedStalkBoundary(source.archive,AXES,materialized['flag_maps']).apply(1,((0,1,0),))
    assert trace['fields'][-1]==((0,0),(0,))


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_exact_fit_family_output_and_training_continue_after_format_roundtrip(tmp_path,suffix):
    source,state=model();state=acquire(state,source,((1,0,0),))
    query=stalk_transport_query(state,((0,1,0),));output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('map'+suffix);save_model(path,record)
    restored_record=load_model(path);restored,native=read_model_record(restored_record)
    assert restored.coefficient_digest==state.coefficient_digest and native.archive==source.archive
    assert restored_record.get_metadata(1,0,'output').coefficient_digest==output.coefficient_digest
    assert infer_model(restored,native,query).coefficient_digest==output.coefficient_digest
    child=acquire(restored,native,((0,1,0),));assert child.payload['fit']['rank']==6


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_store_query_agent_and_system_preserve_the_affine_family(tmp_path,backend):
    from rcdb import open_store
    source,state=model();state=acquire(state,source,((1,0,0),))
    query=stalk_transport_query(state,((0,1,0),));output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'query',query);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('maps',record,expected_version=0);assert store.verify_commits('maps');store.close()
    script='''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]); record=store.read_record('maps',version=1).value
state,source=read_model_record(record); query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
assert result.values[0].coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert result.values[1]['unidentified_directions']==6 and result.values[1]['learned_transports']
assert json_value(state)['unidentified_directions']==6
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('transport'); hive.add_worker('maps',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('maps',query,record=False).coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    result=subprocess.run([sys.executable,'-c',script,url,output.coefficient_digest],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
