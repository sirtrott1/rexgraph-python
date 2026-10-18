"""Serving banks through model records, exact receipts and package interfaces."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from rexgraph.cpp_stalk_pattern_model import (
    cpp_stalk_pattern_bank_model, stalk_pattern_bank_query, stalk_pattern_bank_result,
    inspect_stalk_pattern_bank_model, clear_stalk_pattern_bank_cache,
    enable_stalk_pattern_bank_provider,
)
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph_cpp import NativeSource
from test_stalk_pattern_bank import correlated_bank, comparison, compile_bank


@pytest.fixture(scope="module")
def learned():
    _, _, bank = correlated_bank()
    source = NativeSource(bank.archive())
    return bank, source, cpp_stalk_pattern_bank_model(source)


def request(state, mode):
    if mode == "unknown":
        return stalk_pattern_bank_query(state, {0:(1,)})
    if mode == "cancel":
        return stalk_pattern_bank_query(state, {0:(1,),1:(-1,)})
    if mode == "occurrences":
        return stalk_pattern_bank_query(state, {0:(1,),1:(1,),2:(1,)}, shared=False)
    comparisons = (comparison("first",0,2), comparison("second",0,2,Q(1,2))) if mode == "conflict" else ()
    return stalk_pattern_bank_query(state, {0:(1,),2:(1,)}, operation="glue", comparisons=comparisons)


@pytest.mark.parametrize("mode",["unknown","cancel","occurrences","glue","conflict"])
def test_complete_families_match_native_bank_and_preserve_correlations(learned, mode):
    bank, source, state = learned
    q = request(state,mode); output = infer_model(state,source,q)
    expected = (bank.glue(dict(q["inputs"]),q["comparisons"],shared=q["shared"]) if q["operation"]=="glue"
        else bank.response(dict(q["inputs"]),shared=q["shared"]))
    assert stalk_pattern_bank_result(output)==expected
    assert unpack_model(pack_model(output)).coefficient_digest==output.coefficient_digest
    assert output.values.size==0
    with pytest.raises(TypeError,match="joint field families"):
        output.tensor()
    assert state.payload=={} and state.step==0
    assert inspect_stalk_pattern_bank_model(state,source)==bank.inspect()


@pytest.mark.parametrize("change",["version","arithmetic","step","parent","payload","profile","provider","engine","space","limits","boolean_limit","bytes"])
def test_state_declarations_are_closed_and_serving_only(learned,change):
    _,_,state=learned
    from rexgraph.type_accession import CoordinateSpace
    config=thaw_tree(state.configuration);kwargs={}
    if change=="version":kwargs["adapter_version"]="2"
    elif change=="arithmetic":kwargs["arithmetic"]="approximate"
    elif change=="step":kwargs["step"]=1
    elif change=="parent":kwargs["parent"]="0"*64
    elif change=="payload":kwargs["payload"]={"observations":()}
    elif change=="profile":config["profile"]="other"
    elif change=="provider":config["execution_provider"]["version"]="2"
    elif change=="engine":config["engine_receipt"]["binding"]="0"*64
    elif change=="space":kwargs["space"]=CoordinateSpace(state.space.name,tuple(reversed(state.space.keys)))
    elif change=="limits":config["limits"]=(1,0,1,1)
    elif change=="boolean_limit":config["limits"]=(True,1,1,1)
    elif change=="bytes":config["max_bytes"]=0
    with pytest.raises((ValueError,TypeError)):
        replace(state,configuration=config,**kwargs)


@pytest.mark.parametrize("change",["source","model","operation","shared","index","bool_index","float","shape","comparison","duplicate","extra"])
def test_queries_cannot_rebind_or_silently_change_whole_fields(learned,change):
    _,source,state=learned;q=thaw_tree(request(state,"glue"))
    if change=="source":q["source_reference"]["native"]["namespace"]+=1
    elif change=="model":q["model_digest"]="0"*64
    elif change=="operation":q["operation"]="average"
    elif change=="shared":q["shared"]=1
    elif change=="index":q["inputs"]=((3,(Q(1),)),)
    elif change=="bool_index":q["inputs"]=((True,(Q(1),)),)
    elif change=="float":q["inputs"]=((0,(1.0,)),)
    elif change=="shape":q["inputs"]=((0,(Q(1),Q(2))),)
    elif change=="comparison":q["operation"]="response";q["comparisons"]=(comparison("first",0,2),)
    elif change=="duplicate":q["inputs"]=(q["inputs"][0],q["inputs"][0])
    elif change=="extra":q["extra"]=0
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        infer_model(state,source,q)


@pytest.mark.parametrize("mode,change",[("unknown","origin"),("unknown","directions"),("unknown","identified"),
    ("unknown","origin_kind"),("unknown","direction_kind"),("unknown","shape_kind"),
    ("unknown","domains"),("unknown","offsets"),("unknown","source"),("unknown","query"),
    ("unknown","extra"),("glue","parameter_origin"),("glue","section"),("glue","forced"),
    ("glue","section_cells"),("conflict","contradiction"),("conflict","defect"),("conflict","consistent")])
def test_outputs_replay_every_field_and_conflict_witness(learned,mode,change):
    _,source,state=learned;output=infer_model(state,source,request(state,mode))
    r=thaw_tree(output.execution_receipt);result=r["result"]
    if change=="origin":result["origin"]=(Q(1),)
    elif change=="origin_kind":result["origin"]=(0,)
    elif change=="direction_kind":result["directions"]["entries"]=((0,0,1),)
    elif change=="shape_kind":result["directions"]["shape"]=(True,1)
    elif change=="directions":result["directions"]["entries"]=()
    elif change=="identified":result["identified"]=True
    elif change=="domains":result["domain_coordinates"]=()
    elif change=="offsets":result["offsets"]=(0,0)
    elif change=="source":r["source_archive"]=r["source_archive"][:-1]
    elif change=="query":r["query_declaration"]["inputs"]=((0,(Q(2),)),)
    elif change=="extra":result["extra"]=0
    elif change=="parameter_origin":result["parameter_origin"]=(Q(3),)
    elif change=="section":result["section"]["origin"]=(Q(3),)
    elif change=="forced":result["forced"]=True
    elif change=="section_cells":result["section_cells"]=()
    elif change=="contradiction":result["contradiction"]=(Q(0),)*3
    elif change=="defect":result["defect_directions"]["entries"]=()
    elif change=="consistent":result["consistent"]=True
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        replace(output,execution_receipt=r)


def test_tight_admission_and_actual_source_profile_apply_on_cold_prepare(learned):
    bank,source,state=learned
    assert cpp_stalk_pattern_bank_model(source,max_bytes=len(bank.archive())).coefficient_digest
    with pytest.raises(ValueError):cpp_stalk_pattern_bank_model(source,max_bytes=len(bank.archive())-1)
    with pytest.raises((ValueError,RuntimeError)):cpp_stalk_pattern_bank_model(source,limits=(1,1,1,1))
    original,_,_=correlated_bank()
    with pytest.raises((ValueError,RuntimeError)):cpp_stalk_pattern_bank_model(original)
    _,_,other=compile_bank()
    with pytest.raises(ValueError):infer_model(state,NativeSource(other.archive()),request(state,"glue"))
    assert set(enable_stalk_pattern_bank_provider().operations)=={"create","prepare","infer"}
    clear_stalk_pattern_bank_cache()
    assert stalk_pattern_bank_result(infer_model(state,source,request(state,"glue")))["section"]["origin"]==(2,)


def test_serving_bank_rejects_training_and_parameter_transport(learned):
    from rexgraph.model_runtime import train_model, transport_model
    from rexgraph.model_state import ModelBatch
    from rexgraph.source_binding import CppSourceReference
    _,source,state=learned
    batch=ModelBatch(state.source,state.space,(Q(0),)*3,(True,)*3)
    with pytest.raises(NotImplementedError,match="train"):
        train_model(state,source,batch)
    with pytest.raises(NotImplementedError,match="transport"):
        transport_model(state,source,CppSourceReference(source),{})


def test_restore_enforces_model_byte_limit_before_native_source_parse(learned):
    bank,source,state=learned
    config=thaw_tree(state.configuration);config["max_bytes"]=len(bank.archive())-1
    tight=replace(state,configuration=config)
    record=model_record(source,tight)
    with patch("rexgraph.cpp_model._binding",side_effect=AssertionError("source parsed before byte admission")):
        with pytest.raises(ValueError,match="byte limit"):
            read_model_record(record)


def test_query_order_and_input_mutation_do_not_change_sealed_output(learned):
    _,source,state=learned
    q=stalk_pattern_bank_query(state,{2:(1,),0:(1,)},operation="glue")
    assert q==request(state,"glue")
    output=infer_model(state,source,q);digest=output.coefficient_digest
    q["inputs"]=((0,(Q(9),)),)
    assert output.coefficient_digest==digest
    assert stalk_pattern_bank_result(output)["section"]["origin"]==(2,)


def test_native_content_witness_avoids_duplicate_execution_and_replays_after_eviction(learned):
    import rexgraph.cpp_stalk_pattern_model as provider
    _,source,state=learned;clear_stalk_pattern_bank_cache()
    with patch.object(provider,"_analysis",wraps=provider._analysis) as analyze:
        output=infer_model(state,source,request(state,"unknown"))
        assert analyze.call_count==1
        assert replace(output).coefficient_digest==output.coefficient_digest
        assert analyze.call_count==1
        clear_stalk_pattern_bank_cache()
        assert replace(output).coefficient_digest==output.coefficient_digest
        assert analyze.call_count==2
        r=thaw_tree(output.execution_receipt);r["result"]["identified"]=True
        with pytest.raises(ValueError):replace(output,execution_receipt=r)
        assert analyze.call_count==3


def test_concurrent_queries_keep_each_native_family_and_receipt_bound_to_its_input(learned):
    _,source,state=learned
    queries=tuple(request(state,mode) for mode in ("unknown","cancel","glue","conflict"))
    expected=tuple(infer_model(state,source,q).coefficient_digest for q in queries)
    clear_stalk_pattern_bank_cache()
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests=tuple(pool.map(lambda i:infer_model(state,source,queries[i%4]).coefficient_digest,range(32)))
    assert digests==expected*8


@pytest.mark.parametrize("mode",["unknown","glue","conflict"])
def test_system_endpoint_decodes_exact_query_records_and_native_result_envelopes(learned,mode):
    from fastapi.testclient import TestClient
    from system.server.app import app
    from system.state import sources
    from system.serialize import json_value
    from rcql import Executor,parse,SourcePolicy
    from rcql.executor import Result
    _,source,state=learned;record=model_record(source,state);q=request(state,mode)
    wire=json.loads(json.dumps(json_value(q)))
    native=infer_model(state,source,q)
    assert infer_model(state,source,wire).coefficient_digest==native.coefficient_digest
    direct=Executor(sources={"r":record},params={"q":wire}).execute(parse(
        'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_VALUES(MODEL_INFER(MODEL_STATE(),$q))'))
    assert Result.from_bytes(direct.to_bytes()).values[1]==stalk_pattern_bank_result(native)
    sources.register("bank_serving_test",record,policy=SourcePolicy.allow("read"))
    try:
        with TestClient(app) as client:
            text='FROM $bank_serving_test RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_VALUES(MODEL_INFER(MODEL_STATE(),$q))'
            response=client.post('/api/query',json={"query":text,"params":{"q":wire}})
            assert response.status_code==200,response.text
            assert response.json()["values"][0]["digest"]==native.coefficient_digest
            response=client.post('/api/query',json={"query":text,"params":{"q":wire},"result_format":"native"})
            assert response.status_code==200,response.text
            assert Result.from_bytes(response.content).values[0].coefficient_digest==native.coefficient_digest
    finally:sources.remove("bank_serving_test")


@pytest.mark.parametrize("change",["float","boolean","zero_denominator","negative_denominator","integer_extra","integer_leading_zero","foreign_source","extra_matrix"])
def test_json_queries_reject_approximation_and_foreign_declarations(learned,change):
    from system.serialize import json_value
    _,source,state=learned;q=json.loads(json.dumps(json_value(request(state,"conflict"))))
    value=q["inputs"][0][1][0]
    if change=="float":q["inputs"][0][1][0]=1.0
    elif change=="boolean":value["numerator"]=True
    elif change=="zero_denominator":value["denominator"]=0
    elif change=="negative_denominator":value["denominator"]=-1
    elif change=="integer_extra":value["numerator"]={"kind":"Integer","decimal":"1","extra":0}
    elif change=="integer_leading_zero":value["numerator"]={"kind":"Integer","decimal":"01"}
    elif change=="foreign_source":q["source_reference"]["native"]["namespace"]+=1
    elif change=="extra_matrix":q["comparisons"][0]["left_map"]["extra"]=0
    with pytest.raises((ValueError,TypeError,RuntimeError)):
        infer_model(state,source,q)


def test_json_source_identity_preserves_unsigned_publications_beyond_browser_integer_width():
    from test_stalk_context_view_rules import learn,query
    from rexgraph.cpp_stalk_pattern import compile_stalk_context_pattern_bank
    from system.serialize import json_value
    source,state,views=learn()
    bank=compile_stalk_context_pattern_bank(state,source,query(state,views),(0,1),
        model_namespace=(1<<63)+3,coordinate_ids={"weights":101,"parameters":102,"domains":(100,103)})
    owner=NativeSource(bank.archive());model=cpp_stalk_pattern_bank_model(owner)
    q=stalk_pattern_bank_query(model,{0:(Q(2,3),Q(-3,7),0,0)})
    wire=json.loads(json.dumps(json_value(q)))
    assert wire["source_reference"]["native"]["namespace"]=={"kind":"Integer","decimal":str((1<<63)+3)}
    assert infer_model(model,owner,wire).coefficient_digest==infer_model(model,owner,q).coefficient_digest
    assert json_value(model)["source"]["native"]["namespace"]==wire["source_reference"]["native"]["namespace"]
    assert json_value(infer_model(model,owner,q))["source"]["native"]["namespace"]==wire["source_reference"]["native"]["namespace"]


def test_interface_array_metadata_uses_action_admission_separately_from_archive_bytes(learned):
    import numpy as np
    bank,source,_=learned;state=cpp_stalk_pattern_bank_model(source,max_bytes=len(bank.archive()))
    comparisons=[]
    for i in range(200):
        item=comparison("reading:"+str(i),0,2)
        for side in ("left_map","right_map"):
            item[side]["entries"]=np.asarray(((0,0,1),),dtype=np.int64)
        comparisons.append(item)
    q=stalk_pattern_bank_query(state,{0:(1,),2:(1,)},operation="glue",comparisons=comparisons)
    result=stalk_pattern_bank_result(infer_model(state,source,q))
    assert result["consistent"] and result["section"]["origin"]==(2,)


@pytest.mark.parametrize("suffix",[".rcbd",".safetensors"])
def test_serving_model_record_stores_bank_once_without_training(learned,tmp_path,suffix):
    bank,source,state=learned;record=model_record(source,state)
    assert record.get_metadata(1,0,"model_source")["archive"].tobytes()==bank.archive()
    path=tmp_path/("bank"+suffix);save_model(path,record)
    loaded,data=read_model_record(load_model(path))
    assert loaded.coefficient_digest==state.coefficient_digest and data.archive==bank.archive()
    assert loaded.payload=={} and "observations" not in loaded.configuration
    assert infer_model(loaded,data,request(loaded,"unknown")).execution_receipt["result"]["identified"] is False


@pytest.mark.parametrize("backend",["file","rex","sqlite"])
@pytest.mark.parametrize("mode",["unknown","glue","conflict"])
def test_cold_five_package_serving_preserves_unknowns_and_conflicts(learned,tmp_path,backend,mode):
    from rcdb import open_store
    _,source,state=learned;query=request(state,mode);output=infer_model(state,source,query)
    record=model_record(source,state)
    record.attach_metadata(1,0,"query",query);record.attach_metadata(1,0,"output",output)
    url={"file":"file://"+str(tmp_path/"files"),"rex":"rex://"+str(tmp_path/"rex"),
        "sqlite":"sqlite:///"+str(tmp_path/"db")}[backend]
    store=open_store(url,**({} if backend=="sqlite" else {"read_only":False})).configure_security(require_commits=True)
    store.commit_mutation("patterns",record,expected_version=0)
    assert store.verify_commits("patterns");store.close()
    script='''
import sys,json
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
from rexgraph.cpp_stalk_pattern_model import stalk_pattern_bank_result
store=open_store(sys.argv[1]);record=store.read_record('patterns',version=1).value
state,source=read_model_record(record);q=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':q}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_VALUES(MODEL_INFER(MODEL_STATE(),$q)),MODEL_INFO(MODEL_STATE())'))
output=result.values[0];family=stalk_pattern_bank_result(output)
assert output.coefficient_digest==sys.argv[2]
assert result.values[1]==family and result.exactness[0].value=='rational'
assert result.exactness[1].value=='structural'
assert result.values[2]['parameter_count']==3 and result.values[2]['unidentified_directions']==1
assert result.values[2]['trainable'] is False
if sys.argv[3]=='unknown':assert not family['identified'] and family['directions']['shape']==(1,1)
elif sys.argv[3]=='glue':assert family['section']['origin']==(2,) and family['parameter_directions']['shape']==(1,0)
else:assert not family['consistent'] and family['section'] is None and any(family['contradiction'])
rendered=json_value(output);json.dumps(rendered)
assert rendered['value_storage']=='execution-receipt' and 'values' not in rendered
assert json_value(state)['parameter_count'] is None
assert json_value(state)['parameter_storage']=='native-source-archive'
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('patterns');hive.add_worker('patterns',lambda query,**kw:infer_model(state,source,query),capability='predict')
 assert hive.invoke('patterns',q,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    done=subprocess.run([sys.executable,"-I","-c",script,url,output.coefficient_digest,mode],cwd=tmp_path,
        capture_output=True,text=True,timeout=60)
    assert done.returncode==0,done.stderr


def test_rcql_creation_uses_native_bank_publication_and_existing_source_policy(learned):
    from rcql import Executor,parse,BoundSource
    from rcql.capabilities import SourcePolicy
    _,source,state=learned;record=model_record(source,state)
    created=Executor(sources={"r":record}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_stalk_pattern_bank")')).values[0]
    assert created.source.native==state.source.native
    assert created.space==state.space
    with pytest.raises(PermissionError):
        Executor(sources={"r":BoundSource(record,SourcePolicy.allow())},params={"q":request(state,"unknown")}).execute(parse(
            "FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q)"))
