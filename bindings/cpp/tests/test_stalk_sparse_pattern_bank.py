"""Complete sparse input fields retain the same native model and exact families."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import json
import subprocess
import sys

import pytest

from rexgraph.cpp_stalk_pattern_model import (
    cpp_stalk_pattern_bank_model, stalk_pattern_bank_sparse_query,
    stalk_pattern_bank_result, clear_stalk_pattern_bank_cache,
)
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph_cpp import NativeSource, PreparedStalkTransportPatternBank
from test_stalk_pattern_bank import correlated_bank, comparison
from test_stalk_observed_patterns import compile_observed


def field(entries, *, size=1, frame=2):
    return dict(profile='complete-sparse-stalk-field-v1',frame=frame,size=size,entries=tuple(entries))


@pytest.fixture(scope='module')
def learned():
    _,_,bank=correlated_bank();source=NativeSource(bank.archive())
    return bank,source,cpp_stalk_pattern_bank_model(source)


@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('values',[(1,1,1),(Q(3,7),Q(-3,7),0),(0,0,0)])
def test_sparse_response_and_batch_preserve_complete_correlated_families(learned,shared,values):
    bank,_,_=learned
    sparse={i:field(((0,v),) if v else ()) for i,v in enumerate(values)}
    expected=bank.response({i:(v,) for i,v in enumerate(values)},shared=shared)
    assert bank.response_sparse(sparse,shared=shared)==expected
    assert bank.response_sparse_batch((sparse,)*7,shared=shared)==(expected,)*7


@pytest.mark.parametrize('conflict',[False,True])
def test_sparse_gluing_and_model_receipts_preserve_complete_interfaces(learned,conflict):
    bank,source,state=learned
    inputs={0:field(((0,1),)),2:field(((0,1),))}
    comparisons=(comparison('first',0,2),comparison('second',0,2,Q(1,2))) if conflict else ()
    expected=bank.glue({0:(1,),2:(1,)},comparisons)
    assert bank.glue_sparse(inputs,comparisons)==expected
    query=stalk_pattern_bank_sparse_query(state,inputs,operation='glue',comparisons=comparisons)
    output=infer_model(state,source,query)
    assert stalk_pattern_bank_result(output)==expected
    clear_stalk_pattern_bank_cache()
    assert replace(output).coefficient_digest==output.coefficient_digest
    record=thaw_tree(output.execution_receipt)
    if conflict:record['result']['contradiction']=(Q(0),)*3
    else:record['result']['section']['origin']=(Q(999),)
    with pytest.raises(ValueError):replace(output,execution_receipt=record)


def test_sparse_zero_is_a_complete_field_and_outside_observed_span_rejects():
    _,_,bank=compile_observed(True)
    info=bank.inspect();frame=info['carrier_frames'][4]
    zero={0:field((),size=4,frame=frame)}
    assert bank.response_sparse(zero)==bank.response({0:(0,0,0,0)})
    good={0:field(((0,Q(2,3)),(1,Q(2,3)),(3,Q(2,3))),size=4,frame=frame)}
    assert bank.response_sparse(good)==bank.response({0:(Q(2,3),Q(2,3),0,Q(2,3))})
    with pytest.raises(ValueError,match='outside'):
        bank.response_sparse({0:field(((0,1),),size=4,frame=frame)})
    with pytest.raises(ValueError,match='outside'):
        bank.response_sparse_batch((good,{0:field(((2,1),),size=4,frame=frame)}))
    assert bank.response_sparse(good)['identified']


@pytest.mark.parametrize('change',['frame','size','zero','duplicate','order','range','negative','float','bool_index','bool_frame','extra','partial','shared','empty_batch'])
def test_sparse_input_admission_is_closed_and_exact(learned,change):
    bank,_,_=learned;value=field(((0,1),));shared=True
    if change=='frame':value['frame']=99
    elif change=='size':value['size']=2
    elif change=='zero':value['entries']=((0,0),)
    elif change=='duplicate':value['entries']=((0,1),(0,2))
    elif change=='order':value['size']=2;value['entries']=((1,1),(0,1))
    elif change=='range':value['entries']=((1,1),)
    elif change=='negative':value['entries']=((-1,1),)
    elif change=='float':value['entries']=((0,1.0),)
    elif change=='bool_index':value['entries']=((True,1),)
    elif change=='bool_frame':value['frame']=True
    elif change=='extra':value['support']='full'
    elif change=='partial':value['profile']='partial-sparse-stalk-field-v1'
    elif change=='shared':shared=1
    with pytest.raises((TypeError,ValueError,RuntimeError,OverflowError)):
        if change=='empty_batch':bank.response_sparse_batch(())
        else:bank.response_sparse({0:value},shared=shared)


def test_sparse_batch_has_one_total_work_budget_and_failure_keeps_owner(learned):
    bank,_,_=learned
    tight=PreparedStalkTransportPatternBank(bank.archive(),model_namespace=905,limits=(4096,16000000,64000,16384))
    inputs={2:field(((0,1),))}
    assert tight.response_sparse(inputs)['origin']==(2,)
    with pytest.raises(RuntimeError,match='step limit'):
        tight.response_sparse_batch((inputs,)*2000)
    assert tight.response_sparse(inputs)['origin']==(2,)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=tuple(pool.map(lambda _:bank.response_sparse_batch((inputs,)*4),range(8)))
    assert all(r==results[0] for r in results)


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_sparse_query_and_serving_model_roundtrip_through_existing_io(learned,tmp_path,suffix):
    bank,source,state=learned;query=stalk_pattern_bank_sparse_query(state,{2:field(((0,Q(3,7)),))})
    output=infer_model(state,source,query)
    path=tmp_path/('model'+suffix);save_model(path,model_record(source,state))
    restored,owned=read_model_record(load_model(path))
    assert infer_model(restored,owned,query).coefficient_digest==output.coefficient_digest
    script='''
from pathlib import Path
from fractions import Fraction as Q
from rexgraph_cpp import PreparedStalkTransportPatternBank
import sys
b=PreparedStalkTransportPatternBank(Path(sys.argv[1]).read_bytes(),model_namespace=905)
f=dict(profile='complete-sparse-stalk-field-v1',frame=2,size=1,entries=((0,Q(3,7)),))
assert b.response_sparse_batch(({2:f},)*5)[0]['origin']==(Q(6,7),)
assert b.archive()==Path(sys.argv[1]).read_bytes()
assert 'rexgraph.model_runtime' not in sys.modules
'''
    raw=tmp_path/'bank.rfcxs';raw.write_bytes(bank.archive())
    child=subprocess.run([sys.executable,'-I','-c',script,str(raw)],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr


def test_sparse_system_json_and_rcql_keep_exact_coefficients(learned):
    from system.serialize import json_value
    from rcql import Executor,parse
    from rcql.executor import Result
    _,source,state=learned
    query=stalk_pattern_bank_sparse_query(state,{2:field(((0,Q(3,7)),))})
    wire=json.loads(json.dumps(json_value(query)))
    output=infer_model(state,source,query)
    assert infer_model(state,source,wire).coefficient_digest==output.coefficient_digest
    result=Executor(sources={'r':model_record(source,state)},params={'q':wire}).execute(parse(
        'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_VALUES(MODEL_INFER(MODEL_STATE(),$q))'))
    restored=Result.from_bytes(result.to_bytes())
    assert restored.values[0].coefficient_digest==output.coefficient_digest
    assert restored.values[1]['origin']==(Q(6,7),)


@pytest.mark.parametrize('change',['float','boolean','denominator','foreign_frame','extra'])
def test_sparse_wire_queries_reject_changed_support_and_approximation(learned,change):
    from system.serialize import json_value
    _,source,state=learned
    query=stalk_pattern_bank_sparse_query(state,{2:field(((0,Q(3,7)),))})
    wire=json.loads(json.dumps(json_value(query)));f=wire['inputs'][0][1]
    if change=='float':f['entries'][0][1]=0.5
    elif change=='boolean':f['entries'][0][0]=True
    elif change=='denominator':f['entries'][0][1]['denominator']=0
    elif change=='foreign_frame':f['frame']=99
    elif change=='extra':f['support']='partial'
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        infer_model(state,source,wire)
