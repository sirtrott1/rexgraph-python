"""Independent native peer coverage preserves original heterogeneous stalk fields."""
from dataclasses import replace
from fractions import Fraction as Q
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import pytest

from rexgraph.cpp_stalk_transport import cpp_stalk_transport_model, stalk_transport_observation_batch, _prepare
from rexgraph.cpp_stalk_parameter_relations import (
    stalk_context_view_rules_query, stalk_context_hypotheses, stalk_context_rule_response,
)
from rexgraph.io.model_record import model_record, save_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree
from test_stalk_context_hypotheses import fixture as frame_fixture


def learn(*, partial=False, conflict=False, redundant=False):
    source, old, _ = frame_fixture(repeated=False, dimension=2)
    # Two independent observed stars cover e0,e1. The target also retains e2.
    lower = old.configuration['stalk_axes'][0]
    upper = (('a', 'b'), ('a', 'b'), ('a', 'b', 'context'))
    axes = (lower, upper)
    slots = tuple((p, i, j) for p in range(6) for i in range(2) for j in range(len(upper[p//2])))
    state = cpp_stalk_transport_model(source, axes, ((), ((),)*6), slots, tuple(range(6)))
    for row in (0, 1):
        for j in range(1 if partial and row == 1 else 2):
            field = tuple(tuple(int(i == j and r == row) for i in range(len(local))) for r, local in enumerate(upper))
            values = (Q(2)+int(conflict and row == 1), Q(3,7)) if j == 0 else (Q(-1,5), Q(4))
            state = train_model(state, source, stalk_transport_observation_batch(state, field,
                ((2*row, values), (2*row+1, tuple(2*v for v in values)))))
    maps = (((0,0,1), (1,1,1)),)*2 + (((0,0,1), (1,1,1), (2,2,1)),)
    if redundant:
        maps = (tuple((*m, (3,0,1))) for m in maps)
    upper_views = []
    for a in maps:
        r = tuple((j,i,Q(1,2) if redundant and j == 0 else Q(1)) for i,j,_ in a)
        upper_views.append(('clause', ('a','b','context','redundant'), a, r))
    lower_views = tuple(('participant', ('x','y'), ((0,0,1),(1,1,1)), ((0,0,1),(1,1,1))) for _ in lower)
    return source, state, (lower_views, tuple(upper_views))


def query(state, views):
    return stalk_context_view_rules_query(state, (0,1), (2,), views, minimum_peers=1)


@pytest.mark.parametrize('redundant', [False, True])
def test_rectangular_rules_reuse_native_evidence_and_keep_new_field_directions(redundant):
    source, state, views = learn(redundant=redundant)
    digest = state.coefficient_digest
    output = infer_model(state, source, query(state, views))
    a = stalk_context_hypotheses(output)
    assert a['selected_blocks'] == ((0,2),)
    assert a['interpretation'] == 'learned-covered-context-view-rules-v1'
    certificate = output.execution_receipt['certificate']
    assert certificate['native_evidence_reused'] and certificate['preparation_steps'] == 0
    assert state.payload['fit']['kernel_shape'] == (28,12)
    assert output.execution_receipt['conditional_family']['kernel_shape'] == (28,4)
    assert output.values.ndim == 1 and output.axes == ()
    known = stalk_context_rule_response(output, ((0,0),(0,0),(Q(2,3),Q(-3,7),0)))
    assert known['boundary_family'][0][4:] == ((Q(149,105),Q(-10,7)),(Q(298,105),Q(-20,7)))
    assert len(known['boundary_family']) == 1 and len(known['zero_components']) == 4
    novel = stalk_context_rule_response(output, ((0,0),(0,0),(0,0,1)))
    assert len(novel['boundary_family']) == 5 and not novel['zero_components']
    assert novel['transfer_scope'] == 'conditional-on-covered-context-invariance'
    assert state.coefficient_digest == digest and 'parameter_space' not in state.configuration


@pytest.mark.parametrize('case', ['partial', 'conflict', 'minimum', 'missing'])
def test_unsupported_view_rules_preserve_the_complete_original_family(case):
    source,state,views=learn(partial=case=='partial',conflict=case=='conflict')
    if case == 'missing':views=(views[0],(*views[1][:2],None))
    q=stalk_context_view_rules_query(state,(0,1),(2,),views,minimum_peers=2 if case=='minimum' else 1)
    output=infer_model(state,source,q);a=stalk_context_hypotheses(output)
    assert not a['selected_blocks'] and output.execution_receipt['conditional_family'] is None
    assert len(stalk_context_rule_response(output,((0,0),(0,0),(0,0,1)))['boundary_family'])>1


def test_owner_acquisition_does_not_reconstruct_the_native_proof_from_python_relations(monkeypatch):
    import rexgraph_cpp
    from rexgraph.cpp_stalk_parameter_relations import _cache, _lock
    source,state,views=learn()
    # The ordinary Python constructor would assemble the entire evidence again.
    def forbidden(*a,**kw):raise AssertionError('duplicate evidence assembly')
    monkeypatch.setattr(rexgraph_cpp,'PreparedStalkParameterRelations',forbidden)
    with _lock:_cache.clear()
    output=infer_model(state,source,query(state,views))
    assert output.execution_receipt['certificate']['native_evidence_reused']


def test_native_owners_are_detached_and_parallel_reusable():
    source,state,views=learn();owner=_prepare(state,source)
    acquired,proof=owner.contextual_view_rules((0,1),(2,),views,minimum_peers=1)
    assert proof.receipt['preparation_steps']==0
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=tuple(pool.map(lambda _:owner.contextual_view_rules((0,1),(2,),views,minimum_peers=1)[0],range(8)))
    assert all(r==acquired for r in results)
    acquired['rules'][0]['admitted']=False
    assert owner.contextual_view_rules((0,1),(2,),views,minimum_peers=1)[0]['rules'][0]['admitted']
    raw=proof.receipt
    expected=tuple(sorted((*((i,0,q) for i,q in enumerate(raw['residual_origin']) if q),
        *((i,j+1,q) for i,j,q in raw['residual_directions'])),key=lambda e:(e[0],e[1])))
    assert proof.sparse_residual()==expected


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_covered_conditional_fields_and_unknowns_roundtrip_in_a_cold_process(tmp_path,suffix):
    source,state,views=learn(redundant=True);q=query(state,views);output=infer_model(state,source,q)
    record=model_record(source,state);record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('rules'+suffix);save_model(path,record)
    script='''
import sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_stalk_parameter_relations import stalk_context_rule_response
record=load_model(sys.argv[1]);state,source=read_model_record(record)
output=infer_model(state,source,record.get_metadata(1,0,'query'))
assert output.coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert output.execution_receipt['certificate']['native_evidence_reused']
assert len(stalk_context_rule_response(output,((0,0),(0,0),(0,0,1)))['boundary_family'])==5
assert 'torch' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-I','-c',script,str(path),output.coefficient_digest],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr


def test_view_rule_receipts_reject_peer_coverage_and_map_edits():
    source,state,views=learn();output=infer_model(state,source,query(state,views))
    for key,value in (('covering_peer_blocks',()),('selected_blocks',()),('minimum_peers',5)):
        receipt=thaw_tree(output.execution_receipt);receipt['acquisition'][key]=value
        with pytest.raises(ValueError,match='replay'):replace(output,execution_receipt=receipt)


@pytest.mark.parametrize('training',[(True,),(Q(0),),(0.0,),(0,0),(2,),(3,),'0'])
def test_invalid_training_rows_reject(training):
    source,state,views=learn()
    with pytest.raises((ValueError,TypeError,OverflowError)):
        infer_model(state,source,stalk_context_view_rules_query(state,training,(2,),views,minimum_peers=1))


@pytest.mark.parametrize('peers',[0,-1,True,Q(1),1.0])
def test_invalid_peer_counts_reject(peers):
    _,state,views=learn()
    with pytest.raises((ValueError,TypeError,OverflowError)):
        stalk_context_view_rules_query(state,(0,1),(2,),views,minimum_peers=peers)


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_five_package_views_retain_known_and_unshared_fields(tmp_path,backend):
    from rcdb import open_store
    source,state,views=learn(redundant=True)
    q=query(state,views);output=infer_model(state,source,q)
    record=model_record(source,state)
    record.attach_metadata(1,0,'query',q);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),
         'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('views',record,expected_version=0)
    assert store.verify_commits('views');store.close()
    script='''
import sys
from unittest.mock import patch
from fractions import Fraction as Q
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_stalk_parameter_relations import stalk_context_rule_response
store=open_store(sys.argv[1]);record=store.read_record('views',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
output=result.values[0]
assert output.coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert output.execution_receipt['certificate']['native_evidence_reused']
assert output.execution_receipt['conditional_family']['kernel_shape']==(28,4)
assert output.execution_receipt['acquisition']['selected_blocks']==((0,2),)
known=stalk_context_rule_response(output,((0,0),(0,0),(Q(2,3),Q(-3,7),0)))
assert known['boundary_family'][0][4:]==((Q(149,105),Q(-10,7)),(Q(298,105),Q(-20,7)))
assert len(known['boundary_family'])==1 and len(known['zero_components'])==4
novel=stalk_context_rule_response(output,((0,0),(0,0),(0,0,1)))
assert len(novel['boundary_family'])==5 and not novel['zero_components']
assert novel['transfer_scope']=='conditional-on-covered-context-invariance'
assert json_value(output)['method']=='exact-stalk-parameter-relations-v1'
assert json_value(state)['parameter_relations']
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('views');hive.add_worker('views',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('views',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    completed=subprocess.run([sys.executable,'-I','-c',script,url,output.coefficient_digest],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert completed.returncode==0,completed.stderr
