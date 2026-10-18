"""Learned complete context rules retain independent evidence and conditional fields."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest

from test_stalk_context_hypotheses import fixture
from rexgraph.cpp_stalk_transport import stalk_transport_observation_batch, _prepare, stalk_transport_query
from rexgraph.cpp_stalk_parameter_relations import (
    stalk_context_rules_query, stalk_context_hypotheses, stalk_context_rule_response,
)
from rexgraph.model_runtime import train_model, infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record


def learn(*,different=False,partial=False):
    source,state,frames=fixture(repeated=False,dimension=2)
    for row in (0,1):
        for j in range(2 if row==0 or not partial else 1):
            field=tuple(tuple(int(i==j and r==row) for i in range(2)) for r in range(3))
            target=(Q(2)+int(different and row==1),Q(3,7)) if j==0 else (Q(-1,5),Q(4))
            state=train_model(state,source,stalk_transport_observation_batch(state,field,
                ((2*row,target),(2*row+1,tuple(2*v for v in target)))))
    return source,state,frames


def query(state,frames):
    return stalk_context_rules_query(state,(0,1),(2,),frames,minimum_peers=1)


def test_rules_learn_sharing_without_a_supplied_parameter_basis():
    source,state,frames=learn();before=state.coefficient_digest
    original=infer_model(state,source,stalk_transport_query(state,((0,0),(0,0),(Q(2,3),Q(-3,7)))))
    assert not all(original.execution_receipt['boundary_identified'][4])
    output=infer_model(state,source,query(state,frames));a=stalk_context_hypotheses(output)
    assert output.values.ndim==1 and output.axes==()
    assert output.values.size<len(a['relations'])*(1+len(state.payload['fit']['free_columns']))
    assert all(r['admitted'] for r in a['rules']) and a['selected_blocks']==((0,2),)
    assert a['evidence_scope']=='independent-original-entries'
    assert all(a['blocks'][i]['status']=='open' for i in a['selected_block_indices'])
    family=output.execution_receipt['conditional_family']
    assert family['consistent'] and family['kernel_shape']==(24,0)
    response=stalk_context_rule_response(output,((0,0),(0,0),(Q(2,3),Q(-3,7))))
    expected=(Q(149,105),Q(-10,7))
    assert response['boundary_family'][0][4:]==(expected,tuple(2*v for v in expected))
    assert len(response['boundary_family'])==1
    assert response['transfer_scope']=='conditional-on-observed-context-invariance'
    assert state.coefficient_digest==before and 'parameter_space' not in state.configuration
    assert state.payload['fit']['kernel_shape']==(24,8)


@pytest.mark.parametrize('case',['counterexample','partial','minimum','missing-frame'])
def test_rule_admission_preserves_contradictions_and_unresolved_peers(case):
    source,state,frames=learn(different=case=='counterexample',partial=case=='partial')
    if case=='missing-frame':frames=(frames[0][:4]+(None,frames[0][5]),frames[1])
    q=stalk_context_rules_query(state,(0,1),(2,),frames,minimum_peers=2 if case=='minimum' else 1)
    output=infer_model(state,source,q);a=stalk_context_hypotheses(output)
    assert a['selected_blocks']==() and output.execution_receipt['conditional_family'] is None
    response=stalk_context_rule_response(output,((0,0),(0,0),(1,0)))
    assert len(response['boundary_family'])>1
    if case=='counterexample':assert all(r['excluded_blocks'] for r in a['rules'])
    if case=='partial':assert all(r['open_blocks'] for r in a['rules'])


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_learned_rule_queries_and_fields_roundtrip(suffix,tmp_path):
    source,state,frames=learn();q=query(state,frames);output=infer_model(state,source,q)
    record=model_record(source,state);record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    field=((0,0),(0,0),(Q(2,3),Q(-3,7)));response=stalk_context_rule_response(output,field)
    record.attach_metadata(1,0,'conditional-field',thaw_tree(response))
    path=tmp_path/('model'+suffix);save_model(path,record);loaded=load_model(path)
    replay,native=read_model_record(loaded)
    new=infer_model(replay,native,loaded.get_metadata(1,0,'query'))
    assert new.coefficient_digest==output.coefficient_digest
    assert native.archive==source.archive
    assert stalk_context_rule_response(new,field)==response
    assert loaded.get_metadata(1,0,'output').coefficient_digest==output.coefficient_digest
    assert loaded.get_metadata(1,0,'conditional-field')==response


def test_rule_receipts_reject_edits_to_admission_or_selected_maps():
    source,state,frames=learn();output=infer_model(state,source,query(state,frames))
    r=thaw_tree(output.execution_receipt);r['acquisition']['rules'][0]['admitted']=False
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=r)
    r=thaw_tree(output.execution_receipt);r['acquisition']['selected_blocks']=()
    with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=r)


def test_native_rule_owners_retain_parallel_and_detached_receipts():
    source,state,frames=learn();owner=_prepare(state,source)
    expected=owner.contextual_rules((0,1),(2,),frames,minimum_peers=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r==expected for r in pool.map(lambda _:owner.contextual_rules((0,1),(2,),frames,minimum_peers=1),range(8)))
    expected['rules'][0]['admitted']=False
    assert owner.contextual_rules((0,1),(2,),frames,minimum_peers=1)['rules'][0]['admitted']


@pytest.mark.parametrize('training',[(True,), (Q(0),), (0.0,), (0,0), (2,), (3,), '0'])
def test_invalid_training_rows_reject(training):
    source,state,frames=learn()
    with pytest.raises((ValueError,TypeError,OverflowError)):
        infer_model(state,source,stalk_context_rules_query(state,training,(2,),frames,minimum_peers=1))


@pytest.mark.parametrize('peers',[0,-1,True,Q(1),1.0])
def test_invalid_peer_admission_reject(peers):
    _,state,frames=learn()
    with pytest.raises((ValueError,TypeError,OverflowError)):
        stalk_context_rules_query(state,(0,1),(2,),frames,minimum_peers=peers)


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_five_package_rules_reproduce_complete_conditional_fields(tmp_path,backend):
    from rcdb import open_store
    source,state,frames=learn();q=query(state,frames);output=infer_model(state,source,q)
    record=model_record(source,state);record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),
         'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('rules',record,expected_version=0);assert store.verify_commits('rules');store.close()
    script='''
import sys
from unittest.mock import patch
from fractions import Fraction as Q
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
from rexgraph.cpp_stalk_parameter_relations import stalk_context_rule_response,stalk_parameter_relation
store=open_store(sys.argv[1]);record=store.read_record('rules',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
output=result.values[0]
assert output.coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert output.space.name=='sparse-transport-relation-component' and output.values.ndim==1
assert output.execution_receipt['acquisition']['selected_blocks']==((0,2),)
assert output.execution_receipt['conditional_family']['kernel_shape']==(24,0)
field=stalk_context_rule_response(output,((0,0),(0,0),(Q(2,3),Q(-3,7))))
assert field['boundary_family'][0][4:]==((Q(149,105),Q(-10,7)),(Q(298,105),Q(-20,7)))
assert len(field['boundary_family'])==1
name=output.execution_receipt['acquisition']['relations'][0][0]
assert len(stalk_parameter_relation(output,name)['residual_family'])==9
assert json_value(state)['parameter_relations'] and json_value(output)['method']=='exact-stalk-parameter-relations-v1'
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('rules');hive.add_worker('rules',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('rules',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    completed=subprocess.run([sys.executable,'-I','-c',script,url,output.coefficient_digest],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert completed.returncode==0,completed.stderr
