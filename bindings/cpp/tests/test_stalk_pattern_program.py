"""Exact stage composition preserves original carriers and joint uncertainty."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import json
import subprocess
import sys

import pytest

from rexgraph_cpp import NativeSource, PreparedStalkTransportPatternBank
from rexgraph.cpp_stalk_pattern_model import (
    cpp_stalk_pattern_bank_model, stalk_pattern_bank_program_query,
    stalk_pattern_bank_result, clear_stalk_pattern_bank_cache,
)
from rexgraph.io.model_record import model_record, load_model, save_model, read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from test_stalk_observed_patterns import compile_observed
from test_stalk_pattern_bank import correlated_bank


def root(name, pattern, value, *, size=1):
    return dict(name=name, pattern=pattern, input=dict(profile='complete-sparse-stalk-field-v1',
        frame=2, size=size, entries=tuple((i,Q(value)) for i in range(size) if value)), edges=())


def edge(node, occurrence, gain=1, *, size=1):
    return dict(node=node, occurrence=occurrence, source_frame=1, target_frame=2,
        map=dict(shape=(size,1), entries=tuple((i,0,Q(gain)) for i in range(size) if gain)))


def stage(name, pattern, *edges):
    return dict(name=name, pattern=pattern, input=None, edges=edges)


@pytest.fixture(scope='module')
def learned():
    _,_,bank=correlated_bank();source=NativeSource(bank.archive())
    return bank,source,cpp_stalk_pattern_bank_model(source)


def program(value=Q(3,7)):
    return (root('original',2,value),stage('transport',2,edge(0,2,Q(1,2))),stage('next',2,edge(1,2)))


def test_native_exact_composition_and_complete_stage_fields(learned):
    bank,source,state=learned;nodes=program()
    result=bank.response_sparse_program(nodes)
    assert tuple(r['status'] for r in result)==('supported',)*3
    assert result[0]['response']==bank.response_sparse({2:nodes[0]['input']},shared=False)
    assert result[1]['input']['entries']==((0,Q(3,7)),)
    assert result[2]['response']['origin']==(0,0,Q(12,7))
    assert all(r['parameter_directions']==dict(shape=(3,1),entries=()) for r in result)
    query=stalk_pattern_bank_program_query(state,nodes)
    output=infer_model(state,source,query)
    assert stalk_pattern_bank_result(output)==result and output.values.size==0
    with pytest.raises(TypeError):output.tensor()


def test_joint_directions_cancel_without_assigning_unknown_parameters(learned):
    bank,_,_=learned
    nodes=(root('first',0,1),root('second',1,-1),root('anchor',2,1),
        stage('cancelled',2,edge(0,0),edge(1,1),edge(2,2)),
        stage('open',2,edge(0,0)),stage('blocked',2,edge(4,2)))
    result=bank.response_sparse_program(nodes)
    assert result[0]['parameter_directions']['entries']==((0,0,Q(1)),)
    assert result[1]['parameter_directions']['entries']==((1,0,Q(-1)),)
    assert result[3]['input_directions']['entries']==() and result[3]['response']['origin']==(0,0,4)
    assert result[4]['status']=='open-input' and result[4]['response'] is None
    assert result[4]['input_directions']['entries']==((0,0,Q(1)),)
    assert result[5]['status']=='blocked' and result[5]['input'] is None
    assert result[4]['parameter_directions'] is None and result[5]['input_directions'] is None


def test_supported_zero_and_explicit_zero_map_keep_full_carriers(learned):
    bank,_,_=learned
    result=bank.response_sparse_program((root('unknown',0,1),stage('zero-map',2,edge(0,0,0))))
    assert not result[0]['response']['identified']
    assert result[1]['input']['entries']==() and result[1]['response']['identified']
    assert result[1]['response']['origin']==(0,0,0)
    assert bank.response_sparse_program((root('zero',0,0),))[0]['response']['identified']


def test_outside_domain_is_distinct_from_open_and_blocked():
    _,_,bank=compile_observed(True);info=bank.inspect()
    field=dict(profile='complete-sparse-stalk-field-v1',frame=info['carrier_frames'][4],size=4,entries=((0,Q(1)),))
    result=bank.response_sparse_program((dict(name='outside',pattern=0,input=field,edges=()),))
    assert result[0]['status']=='outside-domain' and result[0]['input']==field and result[0]['response'] is None


@pytest.mark.parametrize('change', ['duplicate_name','forward','foreign_occurrence','source_frame','target_frame',
    'shape','duplicate_edge','extra','float','bool','no_input','both_inputs','empty','malformed_after_open'])
def test_all_program_declarations_validate_before_execution(learned,change):
    bank,source,state=learned;nodes=thaw_tree(program())
    if change=='duplicate_name':nodes[1]['name']=nodes[0]['name']
    elif change=='forward':nodes[1]['edges'][0]['node']=1
    elif change=='foreign_occurrence':nodes[1]['edges'][0]['occurrence']=0
    elif change=='source_frame':nodes[1]['edges'][0]['source_frame']=2
    elif change=='target_frame':nodes[1]['edges'][0]['target_frame']=1
    elif change=='shape':nodes[1]['edges'][0]['map']['shape']=(2,1)
    elif change=='duplicate_edge':nodes[1]['edges']=(*nodes[1]['edges'],nodes[1]['edges'][0])
    elif change=='extra':nodes[0]['latent']='value'
    elif change=='float':nodes[1]['edges'][0]['map']['entries']=((0,0,0.5),)
    elif change=='bool':nodes[1]['edges'][0]['node']=False
    elif change=='no_input':nodes[0]['input']=None
    elif change=='both_inputs':nodes[1]['input']=nodes[0]['input']
    elif change=='empty':nodes=()
    elif change=='malformed_after_open':nodes[0]['pattern']=0;nodes[1]['edges'][0]['occurrence']=0;nodes[2]['edges'][0]['source_frame']=99
    for execute in (lambda:bank.response_sparse_program(nodes),
            lambda:infer_model(state,source,stalk_pattern_bank_program_query(state,nodes))):
        with pytest.raises((ValueError,TypeError,RuntimeError,OverflowError)):execute()


@pytest.mark.parametrize('changed', ['result','query','source','family'])
def test_model_receipts_replay_all_stage_outputs(learned,changed):
    _,source,state=learned;output=infer_model(state,source,stalk_pattern_bank_program_query(state,program()))
    receipt=thaw_tree(output.execution_receipt)
    if changed=='result':receipt['result'][1]['status']='outside-domain'
    elif changed=='query':receipt['query_declaration']['inputs'][1]['edges'][0]['map']['entries']=((0,0,Q(1)),)
    elif changed=='source':receipt['query_declaration']['source_reference']['native']['namespace']+=1
    else:receipt['result'][0]['parameter_directions']['shape']=(3,0)
    clear_stalk_pattern_bank_cache()
    with pytest.raises(ValueError):replace(output,execution_receipt=receipt)


def test_program_has_one_action_budget_and_owning_concurrent_reuse(learned):
    bank,_,_=learned
    tight=PreparedStalkTransportPatternBank(bank.archive(),model_namespace=905,limits=(4096,16000000,64000,16384))
    with pytest.raises(RuntimeError,match='step limit'):
        tight.response_sparse_program(tuple(root('root:'+str(i),2,1) for i in range(3000)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses=tuple(pool.map(lambda _:bank.response_sparse_program(program()),range(12)))
    assert all(r==responses[0] for r in responses)


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_model_program_cold_restore_keeps_native_composition(learned,tmp_path,suffix):
    bank,source,state=learned;query=stalk_pattern_bank_program_query(state,program());output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'query',query);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('program'+suffix);save_model(path,record)
    retained=load_model(path);restored,owner=read_model_record(retained)
    assert infer_model(restored,owner,query).coefficient_digest==output.coefficient_digest
    script='''
import sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_stalk_pattern_model import stalk_pattern_bank_result
r=load_model(sys.argv[1]);s,o=read_model_record(r)
out=infer_model(s,o,r.get_metadata(1,0,'query'))
assert out.coefficient_digest==sys.argv[2]
assert stalk_pattern_bank_result(out)[2]['response']['origin'][2].numerator==12
assert 'torch' not in sys.modules
'''
    child=subprocess.run([sys.executable,'-I','-c',script,str(path),output.coefficient_digest],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr
    raw=tmp_path/'bank.rfcxs';raw.write_bytes(bank.archive())
    cold='''
import sys
from pathlib import Path
from rexgraph_cpp import PreparedStalkTransportPatternBank
b=PreparedStalkTransportPatternBank(Path(sys.argv[1]).read_bytes(),model_namespace=905)
f=dict(profile='complete-sparse-stalk-field-v1',frame=2,size=1,entries=((0,3),))
e=dict(node=0,occurrence=2,source_frame=1,target_frame=2,map=dict(shape=(1,1),entries=((0,0,1),)))
r=b.response_sparse_program((dict(name='first',pattern=2,input=f,edges=()),dict(name='next',pattern=2,input=None,edges=(e,))))
assert r[1]['response']['origin']==(0,0,12)
assert 'rexgraph.model_runtime' not in sys.modules
'''
    child=subprocess.run([sys.executable,'-I','-c',cold,str(raw)],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr


def test_system_exact_json_and_rcql_keep_complete_program(learned):
    from system.serialize import json_value
    from rcql import Executor,parse
    from rcql.executor import Result
    _,source,state=learned;query=stalk_pattern_bank_program_query(state,program())
    wire=json.loads(json.dumps(json_value(query)));output=infer_model(state,source,query)
    assert infer_model(state,source,wire).coefficient_digest==output.coefficient_digest
    result=Executor(sources={'r':model_record(source,state)},params={'q':wire}).execute(parse(
        'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_VALUES(MODEL_INFER(MODEL_STATE(),$q))'))
    restored=Result.from_bytes(result.to_bytes())
    assert restored.values[0].coefficient_digest==output.coefficient_digest
    assert restored.values[1][2]['status']=='supported'


@pytest.mark.parametrize('changed',['float','boolean','denominator','comparisons','edges','profile'])
def test_program_wire_values_reject_changed_exact_and_closed_declarations(learned,changed):
    from system.serialize import json_value
    _,source,state=learned;wire=json.loads(json.dumps(json_value(stalk_pattern_bank_program_query(state,program()))))
    if changed=='float':wire['inputs'][0]['input']['entries'][0][1]=0.5
    elif changed=='boolean':wire['inputs'][1]['edges'][0]['node']=False
    elif changed=='denominator':wire['inputs'][1]['edges'][0]['map']['entries'][0][2]['denominator']=0
    elif changed=='comparisons':wire['comparisons']=''
    elif changed=='edges':wire['inputs'][1]['edges']={}
    else:wire['profile']='stalk-pattern-bank-sparse-query-v1'
    with pytest.raises((ValueError,TypeError,RuntimeError)):infer_model(state,source,wire)


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_five_package_program_execution(learned,tmp_path,backend):
    from rcdb import open_store
    _,source,state=learned;query=stalk_pattern_bank_program_query(state,program());output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'query',query)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('program',record,expected_version=0);store.close()
    script='''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
store=open_store(sys.argv[1]);r=store.read_record('program',version=1).value;s,o=read_model_record(r);q=r.get_metadata(1,0,'query')
out=Executor(sources={'r':r},params={'q':q}).execute(parse('FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q)')).values[0]
assert out.coefficient_digest==sys.argv[2]
assert json_value(out)['execution_receipt']['result'][2]['status']=='supported'
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 h=Hive('program');h.add_worker('program',lambda q,**kw:infer_model(s,o,q),capability='predict')
 assert h.invoke('program',q,record=False).coefficient_digest==sys.argv[2]
store.close()
'''
    child=subprocess.run([sys.executable,'-c',script,url,output.coefficient_digest],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr
