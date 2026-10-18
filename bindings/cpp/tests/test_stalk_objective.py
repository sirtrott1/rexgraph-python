"""Observed and conditional complete fields remain distinct across exact replay."""
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from rexgraph_cpp import NativeSource, PreparedStalkTransport, PreparedStalkFieldObjective
from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, stalk_transport_observation_batch, stalk_transport_query
from rexgraph.cpp_stalk_objective import cpp_stalk_objective_model, stalk_objective_field
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree
from rexgraph.io.model_record import save_model, load_model, model_record, read_model_record
from test_stalk_transport import declaration

AXES=((('meaning',),),(('reading',),('reading',)))
VIEWS=(('meaning',((0,0,1),(1,0,1))),)


def model():
    data=declaration(((10,1,True),),(10,),namespace=917901)
    data['relations']=(data['relations'][0],dict(grade=1,key=2,type=0,
        occurrences=(dict(key=2,lower=10,negative=True,magnitude=1,exists=True,aspects=(0,)),)))
    data['carriers'][1]['size']=2
    source=NativeSource.from_declaration(data)
    state=cpp_stalk_transport_model(source,AXES,((),((),())),((0,0,0),(1,0,0)),(0,))
    state=train_model(state,source,stalk_transport_observation_batch(state,((1,),(1,)),((0,(1,)),)))
    return source,state


def selected(source,evidence,f=1,**extra):
    return cpp_stalk_objective_model(source,evidence,(((1,),(1,)),),VIEWS,channel_coefficients=(0,0,f,0),**extra)


def test_F_selection_retains_exact_evidence_and_complete_original_fields():
    source,evidence=model();off=selected(source,evidence,0);on=selected(source,evidence)
    assert on.payload['selection']['particular']==(Q(1,2),Q(-1,2))
    assert on.payload['selection']['initial_energy']==1 and on.payload['selection']['minimum_energy']==0
    assert on.payload['selection']['rank']==1 and off.payload['selection']['rank']==0
    assert unpack_model(on.payload['evidence_checkpoint']).coefficient_digest==evidence.coefficient_digest
    assert off.payload['selection']['kernel']==evidence.payload['fit']['kernel']
    query=stalk_transport_query(evidence,((2,),(1,)))
    output=infer_model(on,source,query);original=infer_model(evidence,source,query)
    assert output.values[:,:,0].tolist()==[[Q(3,2)]]
    assert output.execution_receipt['evidence_response']==original.execution_receipt['response']
    assert output.execution_receipt['evidence_response']['boundary_family']==(((Q(2),),),((Q(1),),))
    assert output.execution_receipt['conditional_response']['occurrence_family']==(((Q(1),),(Q(1,2),)),)
    assert output.axes[-1].keys==('conditional-origin',)
    masks=output.execution_receipt['field_masks']
    assert masks['evidence_boundary_fixed']==((False,),) and masks['conditional_boundary_fixed']==((True,),)
    assert masks['evidence_occurrence_fixed']==((False,),(False,))
    assert masks['conditional_occurrence_fixed']==((True,),(True,))
    with pytest.raises(TypeError,match='explicit interpretation'):output.tensor()
    off_output=infer_model(off,source,query)
    assert len(off_output.execution_receipt['conditional_response']['boundary_family'])==2


@pytest.mark.parametrize('coefficients',[(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(1,2,3,4)])
@pytest.mark.parametrize('normalized',[False,True])
def test_channel_pullbacks_match_independent_two_relation_quadratic(coefficients,normalized):
    source,evidence=model()
    state=cpp_stalk_objective_model(source,evidence,(((1,),(1,)),),VIEWS,
        channel_coefficients=coefficients,trace_normalized=normalized)
    t,g,f,c=map(Q,coefficients)
    if normalized:t/=2;g/=2;f/=4;c/=2
    # The observed shared boundary fixes r1+r2=1. T and C penalize
    # r1-r2; raw G is constant on this evidence family.
    gain=t+2*f+c
    r=state.payload['selection']
    assert r['rank']==bool(gain)
    assert r['particular']==((Q(1,2),Q(-1,2)) if gain else (Q(1),Q(0)))
    assert r['minimum_energy']==g/2
    assert r['initial_energy']==(t+g+2*f+c)/2


def test_zero_views_preserve_declared_support_and_unresolved_family():
    source,evidence=model()
    state=cpp_stalk_objective_model(source,evidence,(((1,),(1,)),),
        (('supported-zero',((0,0,0),(1,0,0))),),mass=1)
    assert len(state.configuration['views'][0][1])==2
    assert state.payload['selection']['rank']==0 and state.payload['selection']['kernel_shape']==(2,1)


def test_additional_evidence_overrides_the_previous_objective_choice():
    source,evidence=model();state=selected(source,evidence)
    batch=stalk_transport_observation_batch(evidence,((1,),(0,)),((0,(3,)),))
    child=train_model(state,source,batch)
    retained=unpack_model(child.payload['evidence_checkpoint'])
    assert child.step==2 and child.parent==state.coefficient_digest
    assert retained.payload['fit']['particular']==(3,2)
    assert child.payload['selection']['particular']==(3,2) and child.payload['selection']['kernel_shape']==(2,0)
    assert child.payload['selection']['minimum_energy']==25
    assert state.payload['selection']['particular']==(Q(1,2),Q(-1,2))


def test_training_retains_the_new_native_objective_owner(monkeypatch):
    from rexgraph import cpp_stalk_objective as module
    source,evidence=model();state=selected(source,evidence)
    child=train_model(state,source,stalk_transport_observation_batch(evidence,((1,),(0,)),((0,(3,)),)))
    def forbidden(*args,**kwargs):
        raise AssertionError('the fitted objective was prepared again')
    monkeypatch.setattr(module,'_native',forbidden)
    output=infer_model(child,source,stalk_transport_query(evidence,((1,),(0,))))
    assert output.values[:,:,0].tolist()==[[Q(3)]]
    assert output.execution_receipt['evidence_response']['boundary_family']==(((Q(3),),),)


def test_concurrent_inference_keeps_each_complete_query_family():
    from concurrent.futures import ThreadPoolExecutor
    source,evidence=model();state=selected(source,evidence)
    def run(i):
        field=((i+1,),(1,));query=stalk_transport_query(evidence,field)
        output=infer_model(state,source,query)
        assert output.values[:,:,0].tolist()==[[Q(i+2,2)]]
        assert output.execution_receipt['query_declaration']['field']==field
        assert output.execution_receipt['evidence_response']['boundary_family'][0]==((Q(i+1),),)
        output.check_state()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run,range(12)))


@pytest.mark.parametrize('value',[True,1.0,complex(1),Q(1,1<<17000)])
def test_exact_objective_admission(value):
    source,evidence=model()
    with pytest.raises((TypeError,ValueError)):selected(source,evidence,value)


@pytest.mark.parametrize('terms',[((2,0,1),),((0,1,1),),((0,0,1),(0,0,0))])
def test_actual_original_occurrence_axes_are_required(terms):
    source,evidence=model()
    with pytest.raises(ValueError):cpp_stalk_objective_model(source,evidence,(((1,),(1,)),),(('meaning',terms),),mass=1)


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_complete_model_and_output_roundtrip(tmp_path,suffix):
    source,evidence=model();state=selected(source,evidence)
    query=stalk_transport_query(evidence,((2,),(1,)));output=infer_model(state,source,query)
    assert unpack_model(pack_model(output)).coefficient_digest==output.coefficient_digest
    path=tmp_path/('objective'+suffix);save_model(path,model_record(source,state))
    recovered,owner=read_model_record(load_model(path))
    assert recovered.coefficient_digest==state.coefficient_digest
    assert infer_model(recovered,owner,query).coefficient_digest==output.coefficient_digest


def test_altered_conditional_and_evidence_outputs_are_rejected():
    source,evidence=model();state=selected(source,evidence);output=infer_model(state,source,stalk_transport_query(evidence,((2,),(1,))))
    for key in ('conditional_response','evidence_response'):
        receipt=thaw_tree(output.execution_receipt);receipt[key]['boundary_family']=(((Q(99),),),)
        with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)
    receipt=thaw_tree(output.execution_receipt);receipt['field_masks']['evidence_boundary_fixed']=((True,),)
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)


def test_ragged_padding_remains_separate_from_supported_zero_fields():
    from test_stalk_transport import model as ragged_model
    source,evidence=ragged_model()
    state=cpp_stalk_objective_model(source,evidence,(((1,0,0),),),
        (('meaning',((0,0,1),)),('register',((0,1,1),)),('grammar',((1,0,1),))),mass=1)
    query=stalk_transport_query(evidence,((0,0,0),));output=infer_model(state,source,query)
    masks=output.execution_receipt['field_masks']
    assert masks['stalk_support']==((True,True),(True,False))
    assert masks['evidence_boundary_fixed']==((True,True),(True,))
    assert output.values[1,1,0]==output.values[0,1,0]==0
    with pytest.raises(TypeError,match='ragged'):stalk_objective_field(output,interpretation='evidence')
    assert stalk_objective_field(output,1,interpretation='conditional').values.tolist()==[[Q(0)]]


@pytest.mark.parametrize('field,interpretation,expected',[
    (((1,),(1,)),'evidence',Q(1)),
    (((1,),(1,)),'conditional',Q(1)),
    (((2,),(1,)),'conditional',Q(3,2)),
])
def test_fixed_field_extraction_requires_an_explicit_interpretation(field,interpretation,expected):
    source,evidence=model();output=infer_model(selected(source,evidence),source,stalk_transport_query(evidence,field))
    tensor=stalk_objective_field(output,interpretation=interpretation)
    assert tensor.values.tolist()==[[expected]] and tensor.axes[0].keys==('meaning',)
    assert len(tensor.provenance)==3
    with pytest.raises(TypeError):stalk_objective_field(output)
    with pytest.raises(ValueError):stalk_objective_field(output,interpretation='identified')
    if field==((2,),(1,)):
        with pytest.raises(TypeError,match='unresolved'):stalk_objective_field(output,interpretation='evidence')


def test_changed_native_minimum_certificate_is_rejected_on_prepare():
    source,evidence=model();state=selected(source,evidence)
    payload=thaw_tree(state.payload);payload['selection']['particular']=(Q(7),Q(6))
    forged=replace(state,payload=payload)
    with pytest.raises(ValueError,match='native replay'):infer_model(forged,source)


def test_foreign_carrier_and_unknown_direction_components_are_distinct():
    source,evidence=model();state=selected(source,evidence)
    data=declaration(namespace=917902);foreign=NativeSource.from_declaration(data)
    with pytest.raises(ValueError):infer_model(state,foreign)
    off=selected(source,evidence,0);output=infer_model(off,source,stalk_transport_query(evidence,((0,),(0,))))
    assert output.execution_receipt['conditional_response']['zero_components']==('objective-free:1',)
    assert output.execution_receipt['evidence_response']['zero_components']==('free:1',)
    assert off.payload['selection']['free_columns']==(1,)


def test_rcql_creates_the_same_declared_field_action_model():
    from rcql import Executor,parse
    source,evidence=model()
    config=dict(evidence=evidence,probes=(((1,),(1,)),),views=VIEWS,channel_coefficients=(0,0,1,0))
    result=Executor(sources={'source':model_record(source,evidence)},params={'config':config}).execute(parse(
        'FROM $source RETURN MODEL_INIT("cpp_stalk_objective",$config)'))
    assert result.values[0].coefficient_digest==selected(source,evidence).coefficient_digest
    assert result.values[0].arithmetic=='rational' and result.exactness[0].value=='structural'


def test_selected_owner_survives_original_python_handle_deletion():
    source,evidence=model()
    plan=PreparedStalkTransport(source.archive,AXES,((),((),())),((0,0,0),(1,0,0)),
        (dict(input=((1,),(1,)),target=((1,),),observed=(True,)),))
    objective=PreparedStalkFieldObjective(plan,(((1,),(1,)),),VIEWS,channel_coefficients=(0,0,1,0))
    del plan,source,evidence
    assert objective.apply(((2,),(1,)))['boundary_family']==(((Q(3,2),),),)
    receipt=objective.receipt;receipt['evidence_fit']['particular']=(99,99)
    assert objective.receipt['evidence_fit']['particular']==(1,0)


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_five_package_roundtrip_retains_conditional_and_observed_families(tmp_path,backend):
    from rcdb import open_store
    source,evidence=model();state=selected(source,evidence);query=stalk_transport_query(evidence,((2,),(1,)))
    output=infer_model(state,source,query);record=model_record(source,state)
    record.attach_metadata(1,0,'query',query);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('objective',record,expected_version=0);assert store.verify_commits('objective');store.close()
    script='''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
store=open_store(sys.argv[1]);record=store.read_record('objective',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
assert result.values[0].coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert result.values[1]['retains_evidence_family'] and result.values[1]['objective_rank']==1
assert json_value(state)['retains_evidence_family']
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('objective');hive.add_worker('objective',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('objective',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    result=subprocess.run([sys.executable,'-c',script,url,output.coefficient_digest],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
