"""Independent evidenced candidates retain complete native response families."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from rexgraph.cpp_stalk_pattern_model import cpp_stalk_pattern_bank_model,stalk_pattern_bank_sparse_query,stalk_pattern_bank_result,clear_stalk_pattern_bank_cache
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph_cpp import NativeSource,PreparedStalkTransportPatternBank
from test_stalk_sparse_pattern_bank import field
from test_stalk_observed_patterns import compile_observed


@pytest.mark.parametrize('shared',[False,True])
def test_native_candidates_keep_unknown_families_and_outside_domains_separate(shared):
    _,_,bank=compile_observed(True);info=bank.inspect();frame=info['carrier_frames'][4]
    good=field(((0,Q(2,3)),(1,Q(2,3)),(3,Q(2,3))),size=4,frame=frame)
    bad=field(((0,1),),size=4,frame=frame)
    results=bank.response_sparse_candidates({0:good,1:bad},shared=shared)
    assert results[0]==dict(pattern=0,supported=True,response=bank.response_sparse({0:good},shared=shared))
    assert results[1]==dict(pattern=1,supported=False,response=None)
    zero=bank.response_sparse_candidates({0:field((),size=4,frame=frame)},shared=shared)
    assert zero[0]['supported'] and zero[0]['response']['identified']


def test_model_output_keeps_all_candidate_fields_and_replays_tampering():
    _,_,bank=compile_observed(True);source=NativeSource(bank.archive());state=cpp_stalk_pattern_bank_model(source)
    frame=bank.inspect()['carrier_frames'][4]
    inputs={0:field(((0,1),),size=4,frame=frame),1:field(((0,1),(1,1),(3,1)),size=4,frame=frame)}
    query=stalk_pattern_bank_sparse_query(state,inputs,operation='candidates')
    assert query['profile']=='stalk-pattern-bank-sparse-candidates-query-v1'
    output=infer_model(state,source,query)
    assert stalk_pattern_bank_result(output)==bank.response_sparse_candidates(inputs)
    clear_stalk_pattern_bank_cache();replace(output)
    changed=thaw_tree(output.execution_receipt);changed['result'][0]['supported']=True
    with pytest.raises(ValueError):replace(output,execution_receipt=changed)
    from system.serialize import json_value
    serialized=json_value(query)
    assert stalk_pattern_bank_result(infer_model(state,source,serialized))==bank.response_sparse_candidates(inputs)
    serialized['profile']='stalk-pattern-bank-sparse-query-v1'
    with pytest.raises(ValueError):infer_model(state,source,serialized)


@pytest.mark.parametrize('change',['frame','size','indices','empty','zero-entry'])
def test_invalid_candidate_declarations_reject_instead_of_becoming_unsupported(change):
    _,_,bank=compile_observed(True);frame=bank.inspect()['carrier_frames'][4]
    inputs={0:field(((0,1),),size=4,frame=frame)}
    if change=='frame':inputs[0]['frame']=999
    elif change=='size':inputs[0]['size']=1
    elif change=='indices':inputs={999:inputs[0]}
    elif change=='empty':inputs={}
    else:inputs[0]['entries']=((0,0),)
    with pytest.raises((ValueError,OverflowError,IndexError)):bank.response_sparse_candidates(inputs)


def test_resource_limit_is_not_an_outside_domain_status():
    _,_,bank=compile_observed(True);frame=bank.inspect()['carrier_frames'][4]
    limited=PreparedStalkTransportPatternBank(bank.archive(),model_namespace=907,limits=(4096,16000000,64000000,16384))
    with pytest.raises((ValueError,OverflowError)):
        limited.response_sparse_candidates({0:field(((0,Q(1,2**20000)),),size=4,frame=frame)})
