"""Whole finite model checkpoints and common package execution contracts."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import numpy as np
from rexgraph_cpp import NativeSource
from rexgraph.cpp_relational_field_admission import cpp_available_field_program_model
from rexgraph.cpp_relational_field_evidence_admission import cpp_available_field_learning_model, _state
from rexgraph.cpp_relational_finite_field import (
    cpp_available_finite_field_model, finite_field_query, read_finite_field_result,
    evaluate_finite_field, finite_field_branches, finite_field_stalks, inspect_finite_field_model,
)
from rexgraph.field_archive import freeze_field_archive
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import resolve_model_provider
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.model_state import ModelBatch
from rexgraph.source_binding import CppSourceReference
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from test_relational_field_admission import context, publish
from test_relational_finite_field_admission import fixture


def model_fixture(**kwargs):
    f=fixture(**kwargs);owner,literal,finite,evidence,targets,spec,model,origins=f;authority=context(origins)
    fixed=cpp_available_field_program_model(owner,available_program=targets[0],execution_context=authority)
    parents=[];current=evidence
    while current is not None:parents.append(current);current=current.parent()
    learned=cpp_available_field_learning_model(owner,available_evidence=parents[-1],execution_context=authority)
    for step,value in enumerate(reversed(parents[:-1]),1):learned=_state(value,learned.source,learned.configuration,learned.coefficient_digest,step)
    unused=cpp_available_field_program_model(owner,available_program=targets[2],execution_context=authority)
    target_states=(fixed,learned,unused);source=NativeSource(literal.source_archive())
    state=cpp_available_finite_field_model(source,available_map=model,targets=target_states,execution_context=authority)
    return f,source,state,target_states


def complete(f,source,state,*,ordinal=1,values=(7,11,13),key=601):
    origins=f[-1];authority=context(origins)
    output=infer_model(state,source,finite_field_query(state),execution_context=authority)
    family=read_finite_field_result(output,execution_context=authority,state=state);setting=600 if ordinal==1 else 0
    if setting:origins[setting]=publish(setting,f[0],'prospective',f[3].parameter_setting_payload(values))
    origins[key]=publish(key,f[0],'prospective',family.completion_payload(ordinal,values,setting,context(origins)))
    selected=evaluate_finite_field(output,ordinal,values,setting,key,execution_context=context(origins),state=state)
    return output,selected,context(origins)


def test_finite_model_retains_target_checkpoints_original_sources_whole_families_and_stalks():
    f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state)
    assert state.adapter=='cpp_relational_finite_field' and state.adapter_version=='1' and not state.output_axes
    assert state.step==0 and state.parent is None and state.source.native.namespace!=targets[0].source.native.namespace
    assert tuple(unpack_model(t).coefficient_digest for t in state.payload['target_checkpoints'])==tuple(t.coefficient_digest for t in targets)
    assert targets[1].step==1 and targets[1].parent is not None
    assert 'map_sha256' in output.field_archive and 'program_sha256' not in output.field_archive
    family=read_finite_field_result(output,execution_context=authority,state=state)
    value=read_finite_field_result(selected,execution_context=authority,state=state)
    assert value.family(authority).archive()==family.archive() and value.inspect(authority)['ordinal']==1
    branches=finite_field_branches(output,0,4,execution_context=authority)
    assert tuple(b['status'] for b in branches)==('interpreted','interpreted','interpreted','unsupported')
    assert branches[1]['target_source']==targets[1].source.as_record() and branches[3]['target_source'] is None
    stalk=finite_field_stalks(selected,1,((1,0),),execution_context=authority)[0]
    assert stalk['values']==(2,3) and stalk['target_source']==targets[1].source.as_record() and stalk['finite_ordinal']==1
    assert stalk['target_source']!=selected.source.as_record()
    assert len(selected.dependencies)>len(output.dependencies)>len(state.source.native.coordinates)
    with pytest.raises(TypeError):output.tensor()
    with pytest.raises(TypeError):finite_field_stalks(output,1,((1,0),),execution_context=authority)


def test_immutable_instance_query_rejects_silent_source_input_and_model_rebinding():
    f,source,state,targets=model_fixture();authority=context(f[-1]);query=dict(finite_field_query(state))
    for bad in (None,{},dict(query,inputs=()),dict(query,map_sha256='0'*64),dict(query,model_digest='0'*64)):
        with pytest.raises(ValueError):infer_model(state,source,bad,execution_context=authority)
    with pytest.raises(ValueError):infer_model(state,f[0],query,execution_context=authority)
    assert state.coefficient_digest==unpack_model(pack_model(state,native=True)).coefficient_digest


@pytest.mark.parametrize('key',(4,115,400,401,443,80,500,501,600,601))
def test_common_result_and_model_readers_recheck_original_evidence_and_choice_roles(key):
    f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state)
    denied=context(f[-1],remove=(key,))
    with pytest.raises(ValueError):read_finite_field_result(selected,execution_context=denied,state=state)
    if key not in (600,601):
        with pytest.raises(ValueError):inspect_finite_field_model(state,source,execution_context=denied)
        with pytest.raises(ValueError):infer_model(state,source,finite_field_query(state),execution_context=denied)


def test_fixed_choice_and_identified_affine_choice_keep_separate_setting_rules():
    f,source,state,targets=model_fixture();output,fixed,authority=complete(f,source,state,ordinal=0,values=())
    native=read_finite_field_result(fixed,execution_context=authority,state=state)
    assert native.inspect(authority)['parameter_setting_receipt'] is None
    g,other,identified,target_states=model_fixture(identified=True)
    _,selected,available=complete(g,other,identified,values=())
    assert read_finite_field_result(selected,execution_context=available,state=identified).inspect(available)['parameter_setting_receipt'] is not None


def test_exact_value_packing_model_records_persistence_and_current_restore_context(tmp_path):
    f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state)
    for value in (state,output,selected):
        data=pack_value(pack_model(value,native=True));assert pack_value(pack_model(unpack_model(unpack_value(data)),native=True))==data
    record=model_record(source,state);restored,bound=read_model_record(record,execution_context=authority)
    assert restored.coefficient_digest==state.coefficient_digest and restored.source.matches(state.source)
    filename=tmp_path/'finite-model.rcbd';save_model(filename,record,execution_context=authority)
    restored,bound=read_model_record(load_model(filename,execution_context=authority),execution_context=authority)
    assert infer_model(restored,bound,finite_field_query(restored),execution_context=authority).field_archive['data']==output.field_archive['data']
    with pytest.raises(ValueError):read_model_record(record,execution_context=context(f[-1],remove=(443,)))
    with pytest.raises((TypeError,ValueError)):read_model_record(record)


def test_target_pin_and_complete_checkpoint_grafts_reject_before_native_map_acceptance():
    f,source,state,targets=model_fixture();authority=context(f[-1])
    wrong=replace(targets[1],configuration=dict(targets[1].configuration,engine_receipt=dict(targets[1].configuration['engine_receipt'],binding='0'*64)))
    c=dict(state.configuration,target_digests=(targets[0].coefficient_digest,wrong.coefficient_digest,targets[2].coefficient_digest))
    p=dict(state.payload,target_checkpoints=(state.payload['target_checkpoints'][0],pack_model(wrong,native=True),state.payload['target_checkpoints'][2]))
    bad=replace(state,configuration=c,payload=p)
    with pytest.raises(ValueError,match='pinned native implementation'):inspect_finite_field_model(bad,source,execution_context=authority)
    from rexgraph.cpp_relational_field_program import _space
    changed_key=targets[0].configuration['key']+1
    other=replace(targets[0],configuration=dict(targets[0].configuration,key=changed_key),space=_space(targets[0].source,changed_key))
    with pytest.raises(ValueError,match='program or source carrier'):
        cpp_available_finite_field_model(source,available_map=f[-2],targets=(other,*targets[1:]),execution_context=authority)
    with pytest.raises(ValueError,match='publisher closure'):
        cpp_available_finite_field_model(source,available_map=f[-2],targets=(targets[2],*targets[1:]),execution_context=authority)


def test_map_envelope_and_output_receipt_grafts_preserve_distinct_carriers():
    f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state)
    with pytest.raises(ValueError):freeze_field_archive(dict(output.field_archive,program_sha256=output.field_archive['map_sha256']))
    with pytest.raises(ValueError):freeze_field_archive(dict(output.field_archive,map_sha256='X'*64))
    wrong=replace(output,field_archive=dict(output.field_archive,map_sha256='0'*64))
    with pytest.raises(ValueError):read_finite_field_result(wrong,execution_context=authority)
    wrong=replace(output,execution_receipt=dict(output.execution_receipt,finite_key=999))
    with pytest.raises(ValueError):read_finite_field_result(wrong,execution_context=authority)


def test_ordinary_training_transport_and_old_local_promotion_reject():
    f,source,state,targets=model_fixture();authority=context(f[-1])
    assert resolve_model_provider(state,'infer').context_operations==('create','prepare','infer')
    with pytest.raises(NotImplementedError):resolve_model_provider(state,'train')
    with pytest.raises(NotImplementedError):resolve_model_provider(state,'transport')
    batch=ModelBatch(state.source,state.space,np.zeros(len(state.space.keys)),np.ones(len(state.space.keys),dtype=bool))
    with pytest.raises(NotImplementedError):train_model(state,source,batch,execution_context=authority)
    with pytest.raises(NotImplementedError):transport_model(state,source,CppSourceReference(source),{})
    from rexgraph.cpp_relational_field_local import solve_available_field_output
    output,selected,authority=complete(f,source,state)
    with pytest.raises(TypeError):solve_available_field_output(selected,1,{}, {}, {},execution_context=authority)


def test_concurrent_cold_checkpoint_provider_execution_retains_whole_alternatives():
    f,source,state,targets=model_fixture();authority=context(f[-1]);image=pack_value(pack_model(state,native=True))
    def run(_):
        child=unpack_model(unpack_value(image))
        return infer_model(child,source,finite_field_query(child),execution_context=authority).field_archive['data']
    with ThreadPoolExecutor(max_workers=6) as pool:outputs=tuple(pool.map(run,range(12)))
    assert outputs==(outputs[0],)*12


def test_rcql_system_and_agent_worker_use_common_finite_model_contracts():
    f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state);record=model_record(source,state)
    from rcql.model_operators import model_values,model_stalk,model_info,model_finite_branches,model_finite_complete
    assert model_values(record,output,authority)['branch_count']==4
    assert model_info(record,state,authority)['trainable'] is False
    assert model_finite_branches(record,output,0,4,authority)[1]['assignment']==(101,101,101)
    assert model_stalk(record,selected,(1,0),node=1,execution_context=authority)['target_source']==targets[1].source.as_record()
    assert model_finite_complete(record,output,1,(7,11,13),600,601,state,authority).field_archive['data']==selected.field_archive['data']
    from system.serialize import json_value
    summary=json_value(state);assert summary['target_count']==3 and summary['trainable'] is False
    view=json_value(selected);assert 'map_sha256' in view['field_archive'] and 'program_sha256' not in view['field_archive']
    from agent.hive import Hive
    hive=Hive();hive.add_worker('finite-fields',lambda _:infer_model(state,source,finite_field_query(state),execution_context=authority),capability='analyze')
    assert hive.dispatch_capability('analyze',None)['result'].field_archive['data']==output.field_archive['data']


def test_typed_rcql_finite_creation_completion_branch_and_stalk_routes():
    from rcql import Executor,parse
    f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state);record=model_record(source,state)
    config={'available_map':f[-2],'targets':targets,'execution_context':authority}
    request='FROM $r RETURN MODEL_INIT("cpp_relational_finite_field",$config),MODEL_INFER(MODEL_STATE(execution_context=$authority),$query,execution_context=$authority),MODEL_FINITE_BRANCHES($output,begin=1,count=2,execution_context=$authority),MODEL_FINITE_COMPLETE($output,ordinal=1,values=[7,11,13],parameter_setting=600,choice_origin=601,state=$state,execution_context=$authority),MODEL_STALK($selected,[1,0],node=1,execution_context=$authority)'
    results=Executor(sources={'r':record},params={'config':config,'query':finite_field_query(state),'output':output,
        'state':state,'selected':selected,'authority':authority}).execute(parse(request)).values
    assert results[0].coefficient_digest==state.coefficient_digest and results[1].field_archive['data']==output.field_archive['data']
    assert tuple(v['ordinal'] for v in results[2])==(1,2)
    assert results[3].field_archive['data']==selected.field_archive['data'] and results[4]['finite_ordinal']==1


@pytest.mark.parametrize('backend',('file','rex','sqlite'))
def test_checked_rcdb_copies_keep_complete_finite_model_outputs_and_current_context(tmp_path,backend):
    from rcdb import open_store,copy_record
    f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state);record=model_record(source,state)
    record.attach_metadata(1,0,'finite-response',selected)
    stores=[open_store('sqlite:///'+str(tmp_path/(name+'.sqlite'))) if backend=='sqlite' else
            open_store(backend+'://'+str(tmp_path/name),read_only=False) for name in ('first','second')]
    try:
        stored=stores[0].put('finite',record);copy_record(stores[0],stores[1],stored)
        copied=stores[1].get('finite');restored,bound=read_model_record(copied,execution_context=authority)
        response=copied.get_metadata(1,0,'finite-response')
        assert restored.coefficient_digest==state.coefficient_digest and response.model_digest==state.coefficient_digest
        assert read_finite_field_result(response,execution_context=authority,state=restored).archive()==selected.field_archive['data']
        with pytest.raises(ValueError):read_finite_field_result(response,execution_context=context(f[-1],remove=(601,)))
    finally:
        for store in stores:store.close()


def test_rex_client_finite_record_transfer_preserves_workspace_and_audit_boundaries(tmp_path,monkeypatch):
    from contextlib import closing
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from rcdb import MemoryStore,record_packet
    from agent.client import RexClient
    from agent.rcdb import reset_default_store
    from agent.server import audit,auth
    from agent.server.routes.rex import router
    monkeypatch.setenv('REXGRAPH_CONFIG_DIR',str(tmp_path));monkeypatch.setenv('REXGRAPH_RCDB_URI','local://'+str(tmp_path/'received'))
    monkeypatch.setenv('REXGRAPH_AUDIT_JOURNAL',str(tmp_path/'audit.jsonl'));monkeypatch.setenv('REXGRAPH_FRAME_KEY','finite-frame-key')
    monkeypatch.setattr(auth,'_CONFIG_DIR',tmp_path);reset_default_store();auth.reset_auth_manager();audit.reset_cache()
    app=FastAPI();app.include_router(router);f,source,state,targets=model_fixture();output,selected,authority=complete(f,source,state)
    record=model_record(source,state);record.attach_metadata(1,0,'response',selected)
    try:
        manager=auth.get_auth_manager();manager.enable_auth();token=manager.create_token('finite-reader',['alpha'],role='user')
        other_token=manager.create_token('other-reader',['beta'],role='user')
        with closing(MemoryStore()) as upstream:
            upstream.put('finite',record,meta={'workspace':'forged'});packet=record_packet(upstream,'finite')
        with TestClient(app) as transport:
            def strip(kw):return {key:value for key,value in kw.items() if key!='timeout'}
            monkeypatch.setattr(httpx,'stream',lambda method,url,**kw:transport.stream(method,url,**strip(kw)))
            monkeypatch.setattr(httpx,'get',lambda url,**kw:transport.get(url,**strip(kw)))
            client=RexClient('http://testserver',api_key=token,workspace='alpha',frame_key='finite-frame-key')
            receipt=client.rex_store_record(packet,source='finite-source',courier='finite-carrier')
            received=client.rex_fetch_record(receipt.destination_record_id,version=receipt.destination_version).snapshot()
            restored,bound=read_model_record(received.value,execution_context=authority)
            response=received.value.get_metadata(1,0,'response')
            assert restored.coefficient_digest==state.coefficient_digest and bound.archive==source.archive
            assert read_finite_field_result(response,state=restored,execution_context=authority).archive()==selected.field_archive['data']
            assert received.record.meta['workspace']=='alpha' and received.record.meta['stored_by']=='finite-reader'
            assert client.rex_audit()['integrity']['valid']
            other=RexClient('http://testserver',api_key=other_token,workspace='beta',frame_key='finite-frame-key')
            with pytest.raises(httpx.HTTPStatusError) as denied:other.rex_fetch_record(receipt.destination_record_id)
            assert denied.value.response.status_code==404
    finally:
        reset_default_store();auth.reset_auth_manager();audit.reset_cache()
