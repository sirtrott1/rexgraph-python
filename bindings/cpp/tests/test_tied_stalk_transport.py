"""Named parameter fields preserve original flags, whole observations and exact replay."""
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import numpy as np
import pytest

from rexgraph_cpp import NativeSource, PreparedStalkTransport, PreparedStalkBoundary, PreparedStalkFields
from rexgraph.cpp_stalk_transport import (
    cpp_stalk_transport_model, stalk_transport_observation_batch, stalk_transport_query,
    stalk_transport_maps, stalk_transport_field, enable_stalk_transport_provider,
)
from rexgraph.io.model_record import model_record,save_model,load_model,read_model_record
from rexgraph.model_codec import pack_model,unpack_model
from rexgraph.model_runtime import train_model,infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.type_accession import CoordinateSpace
from test_stalk_transport import AXES,SLOTS,declaration
from test_stalk_transport import model as independent_model, acquire

SPACE=CoordinateSpace('semantic-map-gains',('meaning','grammar'))
BASIS=((0,0,1),(1,0,Q(1,3)),(2,0,-1),(3,0,Q(-2,5)),(5,0,7),
       (6,1,1),(7,1,2),(8,1,Q(-3,2)))


def model(basis=BASIS,space=SPACE,training=(0,1)):
    source=NativeSource.from_declaration(declaration())
    state=cpp_stalk_transport_model(source,AXES,((),((),())),SLOTS,training,
        parameter_space=space,parameter_basis=basis)
    return source,state


def train_first_field(state,source):
    return train_model(state,source,stalk_transport_observation_batch(state,((1,0,0),),((0,(-1,Q(2,5))),(1,(1,)))))


def test_one_observation_identifies_shared_maps_and_transfers_to_unobserved_axes():
    source,state=model();state=train_first_field(state,source);fit=state.payload['fit']
    assert state.adapter_version=='2' and fit['rank']==fit['basis_rank']==2
    assert fit['particular']==(2,3) and fit['kernel_shape']==(2,0)
    field=((0,5,7),);output=infer_model(state,source,stalk_transport_query(state,field))
    assert output.values[:,:,0].tolist()==[[Q(16,3),-49],[Q(-1,2),0]]
    assert stalk_transport_field(output,row=0).values.tolist()==[[Q(16,3),-49]]
    assert stalk_transport_field(output,row=1).values.tolist()==[[Q(-1,2)]]
    maps=stalk_transport_maps(state,source)
    assert maps['parameter_values']==(2,3) and maps['entry_values']==(2,Q(2,3),-2,Q(-4,5),0,14,3,6,Q(-9,2))
    assert maps['parameter_space']==(SPACE.name,SPACE.keys)
    assert len(maps['flag_maps'][1][0])==6
    assert PreparedStalkBoundary(source.archive,AXES,maps['flag_maps']).apply(1,field)['fields'][-1]==((Q(16,3),-49),(Q(-1,2),))
    independent=cpp_stalk_transport_model(source,AXES,((),((),())),SLOTS,(0,1))
    independent=train_first_field(independent,source)
    control=infer_model(independent,source,stalk_transport_query(independent,field))
    assert independent.payload['fit']['rank']==3 and len(independent.payload['fit']['free_columns'])==6
    assert control.execution_receipt['boundary_identified']==((False,False),(False,))
    green=PreparedStalkFields(source.archive,AXES[0],AXES[1],maps['flag_maps'][1]).apply((field,))
    assert green['boundaries'][0]==PreparedStalkBoundary(source.archive,AXES,maps['flag_maps']).apply(1,green['fields'][0])['fields'][-1]


def test_unobserved_named_parameter_directions_remain_explicit():
    source,state=model();batch=stalk_transport_observation_batch(state,((1,0,0),),((0,(-1,Q(2,5))),))
    state=train_model(state,source,batch);output=infer_model(state,source,stalk_transport_query(state,((0,1,0),)))
    assert state.payload['fit']['free_columns']==(1,)
    assert output.execution_receipt['response']['components']==('origin','free:grammar')
    assert output.execution_receipt['boundary_identified']==((True,True),(False,))
    with pytest.raises(TypeError):output.tensor()
    with pytest.raises(ValueError,match='explicit'):stalk_transport_maps(state,source)
    assert stalk_transport_maps(state,source,free_parameters=(3,))['parameter_values']==(2,3)


def test_overlapping_templates_sum_before_gluing_and_keep_parameter_aliases():
    space=CoordinateSpace('aliases',('left','right','unused'))
    source,state=model(((0,0,1),(0,1,2)),space,training=(0,))
    batch=stalk_transport_observation_batch(state,((1,0,0),),((0,(-3,0)),))
    state=train_model(state,source,batch);fit=state.payload['fit']
    assert fit['basis_rank']==fit['rank']==1 and fit['free_columns']==(1,2)
    output=infer_model(state,source,stalk_transport_query(state,((3,0,0),)))
    assert output.execution_receipt['response']['components']==('origin',)
    assert output.execution_receipt['response']['zero_components']==('free:right','free:unused')
    assert output.values[:,:,0].tolist()==[[-9,0],[0,0]]
    first=stalk_transport_maps(state,source,free_parameters=(0,0))
    second=stalk_transport_maps(state,source,free_parameters=(Q(7,3),19))
    assert first['parameter_values']!=second['parameter_values']
    assert first['flag_maps']==second['flag_maps'] and first['entry_values'][0]==6


def test_shared_parameter_cancellation_does_not_identify_original_occurrences():
    source=NativeSource.from_declaration(declaration(((10,-1,True),(10,1,True)),(10,)))
    axes=((('meaning',),),(('sense',),));slots=((0,0,0),(1,0,0))
    state=cpp_stalk_transport_model(source,axes,((),((),())),slots,(0,),
        parameter_space=CoordinateSpace('shared',('gain',)),parameter_basis=((0,0,1),(1,0,1)))
    state=train_model(state,source,stalk_transport_observation_batch(state,((1,),),((0,(0,)),)))
    output=infer_model(state,source,stalk_transport_query(state,((3,),)))
    assert state.payload['fit']['rank']==0 and state.payload['fit']['basis_rank']==1
    assert output.execution_receipt['boundary_identified']==((True,),)
    assert output.execution_receipt['occurrence_identified']==((False,),(False,))
    assert output.execution_receipt['response']['occurrence_family'][1]==((-3,),(3,))


def test_inactive_and_zero_flags_retain_shared_template_support():
    source=NativeSource.from_declaration(declaration(((10,1,True),(10,7,False),(10,0,True)),(10,)))
    axes=((('meaning',),),(('sense',),));slots=tuple((i,0,0) for i in range(3))
    plan=PreparedStalkTransport(source.archive,axes,((),((),(),())),slots,
        (dict(input=((1,),),target=((4,),),observed=(True,)),),
        parameter_space=dict(name='shared',keys=('gain',)),parameter_basis=tuple((i,0,1) for i in range(3)))
    assert plan.receipt['particular']==(4,) and plan.receipt['rank']==1
    assert plan.receipt['occurrence_activity']==((),(True,False,True))
    assert plan.apply(((2,),))['occurrence_family'][0]==((8,),(0,),(0,))
    assert plan.materialize()['entry_values']==(4,4,4)


@pytest.mark.parametrize('change',['duplicate','row','column','index_bool','zero','float','bool','text','missing_space','missing_basis','name','keys','empty','duplicate_keys','metadata','steps','bits','entries'])
def test_named_basis_declarations_and_admissions_reject_invalid_inputs(change):
    source=NativeSource.from_declaration(declaration());basis=BASIS;space=dict(name=SPACE.name,keys=SPACE.keys)
    limits=(4096,16000000,64000000,16384)
    if change=='duplicate':basis=(*basis,basis[0])
    elif change=='row':basis=((len(SLOTS),0,1),)
    elif change=='column':basis=((0,2,1),)
    elif change=='index_bool':basis=((True,0,1),)
    elif change=='zero':basis=((0,0,0),)
    elif change=='float':basis=((0,0,0.5),)
    elif change=='bool':basis=((0,0,True),)
    elif change=='text':basis=((0,0,'1/2'),)
    elif change=='missing_space':space=None
    elif change=='missing_basis':basis=None
    elif change=='name':space['name']=' '
    elif change=='keys':space['keys']='meaning'
    elif change=='empty':space['keys']=()
    elif change=='duplicate_keys':space['keys']=('meaning','meaning')
    elif change=='metadata':space['hidden']=1
    elif change=='steps':limits=(4096,16000000,1,16384)
    elif change=='bits':basis=((0,0,1<<20),);limits=(4096,16000000,64000000,8)
    else:limits=(4096,4,64000000,16384)
    with pytest.raises((TypeError,ValueError,RuntimeError,OverflowError,IndexError)):
        PreparedStalkTransport(source.archive,AXES,((),((),())),SLOTS,parameter_space=space,parameter_basis=basis,limits=limits)


def test_conflicting_shared_observations_reject_without_changing_the_parent():
    source,state=model();state=train_first_field(state,source);digest=state.coefficient_digest
    batch=stalk_transport_observation_batch(state,((0,1,0),),((0,(999,0)),))
    with pytest.raises(ValueError,match='inconsistent'):train_model(state,source,batch)
    assert state.coefficient_digest==digest and state.step==1


@pytest.mark.parametrize('change',['basis','space','rank','particular','batch','receipt'])
def test_complete_named_template_fit_and_observation_receipts_are_replayed(change):
    source,state=model();state=train_first_field(state,source)
    configuration=thaw_tree(state.configuration);payload=thaw_tree(state.payload)
    if change=='basis':configuration['parameter_basis']=((0,0,2),*BASIS[1:])
    elif change=='space':configuration['parameter_space']['keys']=('grammar','meaning')
    elif change=='rank':payload['fit']['basis_rank']=0
    elif change=='particular':payload['fit']['particular']=(99,3)
    elif change=='batch':payload['observations'][0]['batch']='0'*64
    else:payload['fit']['preparation_steps']+=1
    with pytest.raises(ValueError):
        bad=replace(state,configuration=configuration,payload=payload);enable_stalk_transport_provider('2').prepare(bad,source)


@pytest.mark.parametrize('tied',[False,True])
@pytest.mark.parametrize('change',['rank','particular','coordinate','activity'])
def test_numerically_equal_receipt_type_changes_fail_exact_fit_replay(tied,change):
    source,state=model() if tied else independent_model()
    state=train_first_field(state,source) if tied else acquire(state,source,((1,0,0),))
    payload=thaw_tree(state.payload);fit=payload['fit']
    if change=='rank':fit['rank']=float(fit['rank'])
    elif change=='particular':fit['particular']=(float(fit['particular'][0]),*fit['particular'][1:])
    elif change=='coordinate':fit['pivots']=(float(fit['pivots'][0]),*fit['pivots'][1:])
    else:fit['occurrence_activity']=((),(1,*fit['occurrence_activity'][1][1:]))
    with pytest.raises(ValueError,match='replay'):
        bad=replace(state,payload=payload)
        enable_stalk_transport_provider(state.adapter_version).prepare(bad,source)


@pytest.mark.parametrize('tied',[False,True])
@pytest.mark.parametrize('change',['coefficient','boundary','occurrence','support'])
def test_numerically_equal_response_type_changes_fail_whole_field_replay(tied,change):
    source,state=model() if tied else independent_model()
    state=train_first_field(state,source) if tied else acquire(state,source,((1,0,0),))
    output=infer_model(state,source,stalk_transport_query(state,((1,0,0),)))
    receipt=thaw_tree(output.execution_receipt)
    if change=='coefficient':
        families=list(receipt['response']['boundary_family']);rows=list(families[0]);vector=list(rows[0])
        vector[0]=float(vector[0]);rows[0]=tuple(vector);families[0]=tuple(rows)
        receipt['response']['boundary_family']=tuple(families)
    else:
        key={'boundary':'boundary_identified','occurrence':'occurrence_identified','support':'stalk_support'}[change]
        rows=list(receipt[key]);rows[0]=(int(rows[0][0]),*rows[0][1:]);receipt[key]=tuple(rows)
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)


def test_named_output_method_and_exact_parameter_receipt_cannot_be_downgraded():
    source,state=model();state=train_first_field(state,source)
    output=infer_model(state,source,stalk_transport_query(state,((1,2,3),)))
    with pytest.raises(ValueError):replace(output,method='exact-affine-stalk-transport-v1')
    receipt=thaw_tree(output.execution_receipt);receipt['response']['profile']='whole-stalk-affine-response-v1'
    with pytest.raises(ValueError):replace(output,execution_receipt=receipt)
    with pytest.raises(ValueError):replace(output,values=np.asarray(output.values,dtype=np.float64),arithmetic='approximate')
    with pytest.raises(ValueError):replace(output,dependencies=(state.source,))
    with pytest.raises(ValueError):replace(state,dependencies=(state.source,))


@pytest.mark.parametrize('native',[False,True])
def test_unidentified_named_family_keeps_its_complete_portable_checkpoint(native):
    source,state=model();restored=unpack_model(pack_model(state,native=native))
    assert restored.coefficient_digest==state.coefficient_digest
    assert restored.configuration['parameter_space']['keys']==SPACE.keys
    assert restored.payload['fit']['kernel_shape']==(2,2)
    assert restored.payload['fit']['kernel']==((0,0,1),(1,1,1))


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_format_roundtrip_preserves_named_family_and_continued_observations(tmp_path,suffix):
    source,state=model();state=train_first_field(state,source)
    query=stalk_transport_query(state,((0,5,7),));output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('tied'+suffix);save_model(path,record)
    record=load_model(path);restored,native=read_model_record(record)
    assert restored.coefficient_digest==state.coefficient_digest
    assert record.get_metadata(1,0,'output').coefficient_digest==output.coefficient_digest
    assert infer_model(restored,native,query).coefficient_digest==output.coefficient_digest
    child=train_model(restored,native,stalk_transport_observation_batch(restored,((0,5,7),),((0,(Q(16,3),-49)),)))
    assert child.step==2 and child.payload['fit']['rank']==2


def test_query_creation_information_and_system_preview_use_named_parameter_count():
    from rcql import Executor,parse
    from system.serialize import json_value
    source,state=model();record=model_record(source,state)
    config=dict(stalk_axes=AXES,fixed_maps=((),((),())),parameter_slots=SLOTS,training_rows=(0,1),
        parameter_space=dict(name=SPACE.name,keys=SPACE.keys),parameter_basis=BASIS)
    batch=stalk_transport_observation_batch(state,((1,0,0),),((0,(-1,Q(2,5))),(1,(1,))))
    result=Executor(sources={'r':record},params={'c':config,'b':batch,'q':stalk_transport_query(state,((0,5,7),))}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_stalk_transport",$c),MODEL_INFO(MODEL_STATE()),MODEL_INFER(MODEL_TRAIN(MODEL_STATE(),$b),$q)'))
    assert result.values[0].adapter_version=='2'
    assert result.values[1]['parameter_count']==2 and result.values[1]['supported_map_entries']==9
    assert result.values[1]['basis_rank']==2 and result.exactness[2].value=='rational'
    assert json_value(state)['parameter_count']==2 and json_value(state)['supported_map_entries']==9


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_isolated_store_query_agent_and_system_keep_the_named_family(tmp_path,backend):
    from rcdb import open_store
    source,state=model();state=train_first_field(state,source)
    query=stalk_transport_query(state,((0,5,7),));output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'query',query);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('tied',record,expected_version=0);assert store.verify_commits('tied');store.close()
    script='''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
store=open_store(sys.argv[1]);record=store.read_record('tied',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
assert result.values[0].coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert result.values[1]['parameter_count']==2 and result.values[1]['supported_map_entries']==9
assert json_value(state)['parameter_space']['keys']==['meaning','grammar']
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('tied');hive.add_worker('maps',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('maps',query,record=False).coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    result=subprocess.run([sys.executable,'-I','-c',script,url,output.coefficient_digest],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr


def test_provider_creation_requires_its_declared_parameterization_version():
    source,_=model();config=dict(stalk_axes=AXES,fixed_maps=((),((),())),parameter_slots=SLOTS,training_rows=(0,1))
    with pytest.raises(ValueError,match='version 2'):
        enable_stalk_transport_provider('1').create(source,**config,parameter_space=SPACE,parameter_basis=BASIS)
    with pytest.raises(ValueError,match='named templates'):
        enable_stalk_transport_provider('2').create(source,**config)


def test_original_supported_coordinates_can_be_zero_without_a_basis_term():
    source,state=model((),CoordinateSpace('unused',('gain',)))
    output=infer_model(state,source,stalk_transport_query(state,((1,2,3),)))
    assert state.payload['fit']['basis_rank']==state.payload['fit']['rank']==0
    assert output.execution_receipt['response']['zero_components']==('free:gain',)
    maps=stalk_transport_maps(state,source,free_parameters=(Q(2,3),))
    assert maps['entry_values']==(0,)*9 and len(maps['flag_maps'][1][0])==6


def test_named_basis_accepts_numpy_integer_values_without_floating_conversion():
    source,state=model(((0,0,np.int64(1)),),CoordinateSpace('exact',('gain',)),training=(0,))
    state=train_model(state,source,stalk_transport_observation_batch(state,((np.int64(1),0,0),),((0,(-1,0)),)))
    assert state.payload['fit']['particular']==(2,)
    assert type(state.configuration['parameter_basis'][0][2].numerator) is int
