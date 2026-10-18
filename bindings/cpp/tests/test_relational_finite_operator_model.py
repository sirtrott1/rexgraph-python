"""Original checkpoints, independent instances and common whole field execution."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph_cpp import NativeSource, PreparedAvailableFiniteFieldMap, PreparedAvailableFiniteFieldOperator
from rexgraph.cpp_relational_finite_operator import (
    cpp_available_finite_operator_model, finite_operator_query, read_finite_operator_result,
    evaluate_finite_operator, finite_operator_branches, finite_operator_stalks,
    inspect_finite_operator_model,
)
from rexgraph.cpp_relational_finite_field import cpp_available_finite_field_model
from rexgraph.cpp_relational_field_admission import cpp_available_field_program_model
from rexgraph.cpp_relational_field_evidence_admission import (
    cpp_available_field_learning_model, available_field_program_batch, _prepare as prepare_evidence, _state,
)
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.model_contract import resolve_model_provider
from rexgraph.model_state import ModelBatch
from rexgraph.source_binding import CppSourceReference
from rexgraph.field_archive import freeze_field_archive
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from test_relational_field_admission import context, publish
from test_relational_field_evidence_admission import measured_observation
from test_relational_field_evidence import tensor, reference
from test_relational_finite_field import values
from test_relational_finite_operator_admission import independent, complete as native_complete
from rexgraph_cpp import finite_interpretation_origin_payload, finite_operator_origin_payload
from test_relational_finite_operator_admission import reference_fixture


def reference_model(f,origins,*,evidence=None,evidence_state=None,available_map=None):
    owner,literal,finite,old_evidence,targets,spec,old_map,_=f
    authority=context(origins);evidence=old_evidence if evidence is None else evidence
    fixed=cpp_available_field_program_model(owner,available_program=targets[0],execution_context=authority)
    if evidence_state is None:
        parents=[];current=evidence
        while current is not None:parents.append(current);current=current.parent()
        evidence_state=cpp_available_field_learning_model(owner,available_evidence=parents[-1],execution_context=authority)
        for step,value in enumerate(reversed(parents[:-1]),1):
            evidence_state=_state(value,evidence_state.source,evidence_state.configuration,evidence_state.coefficient_digest,step)
    unused=cpp_available_field_program_model(owner,available_program=targets[2],execution_context=authority)
    source=NativeSource(literal.source_archive())
    parent=cpp_available_finite_field_model(source,available_map=old_map if available_map is None else available_map,
        targets=(fixed,evidence_state,unused),execution_context=authority)
    return source,parent,(fixed,evidence_state,unused)


def fixture(**kwargs):
    data=independent(**kwargs);f,op,instance,origins,finite,rows,d,keys,held=data
    source,parent,targets=reference_model(f,origins)
    state=cpp_available_finite_operator_model(source,reference_state=parent,available_operator=op,execution_context=context(origins))
    return data,source,parent,state,targets


def complete_model(data,source,state,*,ordinal=1,xi=(7,11,13),key=1501):
    f,op,instance,origins,finite,rows,d,keys,held=data
    query=finite_operator_query(state,instance,execution_context=context(origins))
    output=infer_model(state,source,query,execution_context=context(origins))
    family=read_finite_operator_result(output,execution_context=context(origins),state=state)
    native_complete(family,f,origins,held,xi,ordinal,key)
    selected=evaluate_finite_operator(output,ordinal,xi,1500 if ordinal==1 else 0,key,
        execution_context=context(origins),state=state)
    return query,output,selected,context(origins)


def test_original_checkpoint_child_lineage_independent_sources_and_whole_fields():
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    assert state.adapter=='cpp_relational_finite_operator' and state.step==parent.step+1 and state.parent==parent.coefficient_digest
    assert state.source.matches(parent.source) and state.space==parent.space and state.output_axes==parent.output_axes
    assert unpack_model(state.payload['reference_checkpoint']).coefficient_digest==parent.coefficient_digest
    assert output.source.matches(state.source) and output.execution_receipt['actual_input']['source']!=state.source.as_record()
    assert output.values.size==0 and 'operator_sha256' in output.field_archive and 'map_sha256' not in output.field_archive
    family=read_finite_operator_result(output,execution_context=authority,state=state)
    value=read_finite_operator_result(selected,execution_context=authority,state=state)
    assert family.instance(authority).source(authority).raw().asset().recover(311)==(b'oak',b'',b'ak')
    assert family.instance(authority).use_correspondences(0,2,authority)==({'reference':111,'input':311},{'reference':112,'input':312})
    assert len(values(value.raw().raw().result().field_archive(2,occurrences=True)))==19
    branches=finite_operator_branches(output,0,4,execution_context=authority)
    assert tuple(r['status'] for r in branches)==('interpreted','interpreted','interpreted','unsupported')
    assert branches[1]['target_source']==targets[1].source.as_record() and branches[3]['fields']==()
    stalk=finite_operator_stalks(selected,1,((1,0),),execution_context=authority)[0]
    assert stalk['values']==(2,3) and stalk['actual_input_source']==output.execution_receipt['actual_input']['source']
    assert stalk['target_source']==targets[1].source.as_record() and stalk['target_source']!=selected.source.as_record()
    assert len(selected.dependencies)>len(output.dependencies)>len(state.dependencies)
    with pytest.raises(TypeError):output.tensor()
    with pytest.raises(TypeError):finite_operator_stalks(output,1,((1,0),),execution_context=authority)


@pytest.mark.parametrize('route', ('infer', 'family', 'evaluation'))
def test_declared_source_admission_survives_smaller_constructor_defaults(monkeypatch, route):
    import rexgraph.cpp_relational_finite_operator as provider
    data,source,parent,state,targets=fixture()
    query,output,selected,authority=complete_model(data,source,state)
    module=provider._binding()

    class SmallerDefaults:
        def __getattr__(self,name):
            return getattr(module,name)

        @staticmethod
        def NativeSource(image,*,max_dimension=1,max_records=1,**policy):
            return NativeSource(image,max_dimension=max_dimension,max_records=max_records,**policy)

    with pytest.raises(ValueError):
        SmallerDefaults.NativeSource(source.archive)
    monkeypatch.setattr(provider,'_binding',lambda:SmallerDefaults())
    if route=='infer':
        replay=infer_model(state,source,query,execution_context=authority)
        assert pack_value(pack_model(replay,native=True))==pack_value(pack_model(output,native=True))
    else:
        original=selected if route=='evaluation' else output
        replay=read_finite_operator_result(original,execution_context=authority,state=state)
        assert replay.archive()==original.field_archive['data']
        with pytest.raises(ValueError):
            read_finite_operator_result(original,execution_context=authority,state=state,
                                        limits=(4096,1,64000000,16384))


@pytest.mark.parametrize('key',(1,4,115,400,401,443,80,82,500,501,1000,1001,1100,1101,1113,1114,1115,1121,1140,1141,1142,1143,1144,1500,1501))
def test_common_checkpoint_output_and_standalone_readers_recheck_every_role_family(key):
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    denied=context(data[3],remove=(key,))
    with pytest.raises(ValueError):read_finite_operator_result(selected,execution_context=denied,state=state)
    if key not in (1500,1501):
        with pytest.raises(ValueError):infer_model(state,source,query,execution_context=denied)
    if key<1100:
        with pytest.raises(ValueError):inspect_finite_operator_model(state,source,execution_context=denied)


def test_query_source_operator_instance_and_unsupported_structural_updates_reject():
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    for wrong in (None,{},dict(query,inputs=()),dict(query,operator_sha256='0'*64),dict(query,model_digest='0'*64),
                  dict(query,instance_sha256='0'*64),dict(query,source=output.execution_receipt['actual_input']['source'])):
        with pytest.raises(ValueError):infer_model(state,source,wrong,execution_context=authority)
    with pytest.raises(ValueError):infer_model(state,data[-1],query,execution_context=authority)
    with pytest.raises(TypeError):finite_operator_query(state,data[1],execution_context=authority)
    batch=ModelBatch(state.source,state.space,np.zeros(len(state.space.keys)),np.ones(len(state.space.keys),dtype=bool))
    with pytest.raises(NotImplementedError):train_model(state,source,batch,execution_context=authority)
    with pytest.raises(NotImplementedError):transport_model(state,source,CppSourceReference(source),{})
    with pytest.raises(NotImplementedError):resolve_model_provider(state,'train')
    assert resolve_model_provider(state,'infer').context_operations==('create','prepare','infer')


def test_complete_original_checkpoint_pins_lineage_and_input_receipt_grafts_reject():
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    with pytest.raises(ValueError):replace(state,parent='0'*64)
    with pytest.raises(ValueError):replace(state,step=0)
    wrong=replace(state,configuration=dict(state.configuration,engine_receipt=dict(state.configuration['engine_receipt'],binding='0'*64)))
    with pytest.raises(ValueError,match='pinned native implementation'):inspect_finite_operator_model(wrong,source,execution_context=authority)
    bad_parent=replace(parent,configuration=dict(parent.configuration,engine_receipt=dict(parent.configuration['engine_receipt'],binding='0'*64)))
    wrong=replace(state,parent=bad_parent.coefficient_digest,configuration=dict(state.configuration,reference_digest=bad_parent.coefficient_digest),
        payload=dict(state.payload,reference_checkpoint=pack_model(bad_parent,native=True)))
    with pytest.raises(ValueError,match='pinned native implementation'):inspect_finite_operator_model(wrong,source,execution_context=authority)
    wrong=replace(output,execution_receipt=dict(output.execution_receipt,actual_input=dict(output.execution_receipt['actual_input'],key=999)))
    with pytest.raises(ValueError,match='actual input'):read_finite_operator_result(wrong,execution_context=authority)
    wrong=replace(output,dependencies=tuple(reversed(output.dependencies)))
    with pytest.raises(ValueError):read_finite_operator_result(wrong,execution_context=authority)
    with pytest.raises(ValueError):freeze_field_archive(dict(output.field_archive,map_sha256=state.configuration['operator_sha256']))
    wrong=replace(output,field_archive=dict(output.field_archive,operator_sha256='0'*64))
    with pytest.raises(ValueError):read_finite_operator_result(wrong,execution_context=authority)


def test_fixed_and_zero_axis_affine_choices_still_have_distinct_prospective_laws():
    data,source,parent,state,targets=fixture(identified=True)
    query,output,selected,authority=complete_model(data,source,state,xi=())
    value=read_finite_operator_result(selected,execution_context=authority,state=state)
    assert value.inspect(authority)['parameter_setting_receipt'] is not None
    _,_,fixed,authority=complete_model(data,source,state,ordinal=0,xi=(),key=1502)
    assert read_finite_operator_result(fixed,execution_context=authority).inspect(authority)['parameter_setting_receipt'] is None
    family=read_finite_operator_result(output,execution_context=authority)
    with pytest.raises(ValueError):family.completion_payload(1,(),0,authority)
    with pytest.raises(ValueError):evaluate_finite_operator(output,3,(),0,1502,execution_context=authority,state=state)


def test_canonical_values_model_persistence_and_separate_current_context(tmp_path):
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    for value in (state,output,selected):
        image=pack_value(pack_model(value,native=True));assert pack_value(pack_model(unpack_model(unpack_value(image)),native=True))==image
    record=model_record(source,state);record.attach_metadata(1,0,'operator-response',selected)
    restored,bound=read_model_record(record,execution_context=authority)
    assert infer_model(restored,bound,query,execution_context=authority).field_archive['data']==output.field_archive['data']
    path=tmp_path/'operator.rcbd';save_model(path,record,execution_context=authority)
    restored,bound=read_model_record(load_model(path,execution_context=authority),execution_context=authority)
    assert read_finite_operator_result(selected,state=restored,execution_context=authority).archive()==selected.field_archive['data']
    with pytest.raises((TypeError,ValueError)):read_model_record(record)
    with pytest.raises(ValueError):read_model_record(record,execution_context=context(data[3],remove=(1001,)))


def test_cold_concurrent_provider_and_output_readers_keep_actual_input_lineage():
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    image=pack_value(pack_model(state,native=True));response=pack_value(pack_model(selected,native=True))
    def run(_):
        current=unpack_model(unpack_value(image));value=unpack_model(unpack_value(response))
        assert read_finite_operator_result(value,execution_context=authority,state=current).archive()==selected.field_archive['data']
        return infer_model(current,source,query,execution_context=authority).field_archive['data']
    with ThreadPoolExecutor(max_workers=6) as pool:assert tuple(pool.map(run,range(6)))==(output.field_archive['data'],)*6


def test_typed_rcql_system_and_existing_agent_worker_routes():
    from rcql import Executor,parse
    from rcql.model_operators import model_values,model_stalk,model_info,model_finite_branches,model_finite_complete
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state);record=model_record(source,state)
    assert model_values(record,output,authority)['actual_input']==output.execution_receipt['actual_input']
    assert model_info(record,state,authority)['reference_digest']==parent.coefficient_digest
    assert model_finite_branches(record,output,0,4,authority)[3]['status']=='unsupported'
    assert model_stalk(record,selected,(1,0),node=1,execution_context=authority)['target_source']==targets[1].source.as_record()
    assert model_finite_complete(record,output,1,(7,11,13),1500,1501,state,authority).field_archive['data']==selected.field_archive['data']
    config={'reference_state':parent,'available_operator':data[1],'execution_context':authority}
    request='FROM $r RETURN MODEL_INIT("cpp_relational_finite_operator",$config),MODEL_INFER(MODEL_STATE(execution_context=$a),$q,execution_context=$a),MODEL_FINITE_BRANCHES($o,begin=1,count=2,execution_context=$a),MODEL_FINITE_COMPLETE($o,ordinal=1,values=[7,11,13],parameter_setting=1500,choice_origin=1501,state=$s,execution_context=$a),MODEL_STALK($v,[1,0],node=1,execution_context=$a)'
    result=Executor(sources={'r':record},params={'config':config,'a':authority,'q':query,'o':output,'s':state,'v':selected}).execute(parse(request)).values
    assert result[0].coefficient_digest==state.coefficient_digest and result[1].field_archive['data']==output.field_archive['data']
    assert tuple(r['ordinal'] for r in result[2])==(1,2) and result[4]['actual_input_source']==output.execution_receipt['actual_input']['source']
    from system.serialize import json_value
    assert json_value(state)['reference_digest']==parent.coefficient_digest and json_value(output)['retains_complete_alternatives']
    assert json_value(selected)['field_archive']['operator_sha256']==state.configuration['operator_sha256']
    from agent.hive import Hive
    hive=Hive();hive.add_worker('operator-fields',lambda _:infer_model(state,source,query,execution_context=authority),capability='analyze')
    assert hive.dispatch_capability('analyze',None)['result'].field_archive['data']==output.field_archive['data']


@pytest.mark.parametrize('backend',('file','rex','sqlite'))
def test_existing_checked_rcdb_copy_retains_operator_inputs_and_context(tmp_path,backend):
    from rcdb import open_store,copy_record
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    record=model_record(source,state);record.attach_metadata(1,0,'operator-response',selected)
    stores=[open_store('sqlite:///'+str(tmp_path/(name+'.sqlite'))) if backend=='sqlite' else
        open_store(backend+'://'+str(tmp_path/name),read_only=False) for name in ('first','second')]
    try:
        row=stores[0].put('operator',record);copy_record(stores[0],stores[1],row);copied=stores[1].get('operator')
        restored,bound=read_model_record(copied,execution_context=authority);response=copied.get_metadata(1,0,'operator-response')
        assert read_finite_operator_result(response,state=restored,execution_context=authority).archive()==selected.field_archive['data']
        with pytest.raises(ValueError):read_finite_operator_result(response,execution_context=context(data[3],remove=(1144,)))
    finally:
        for store in stores:store.close()


def learned_operator_pair():
    f,old_operator,base,operator_declaration=reference_fixture()
    owner=f[0];origins=dict(base);root=f[3].parent()
    initial=cpp_available_field_learning_model(owner,available_evidence=root,execution_context=context(origins))
    observed=measured_observation(owner,root.program().program(),origins,11,(1,2,3,4),(2,-1,3,0,0),mask=((0,0),))
    batch=available_field_program_batch(initial,owner,(observed,),execution_context=context(origins))
    learned=train_model(initial,owner,batch,execution_context=context(origins))
    assert learned.source.matches(initial.source) and learned.space==initial.space and learned.output_axes==initial.output_axes
    assert learned.parent==initial.coefficient_digest and learned.step==initial.step+1
    with pytest.raises(ValueError):train_model(initial,owner,batch,execution_context=context(origins,remove=(115,)))
    models=[]
    for target_state in (initial,learned):
        grants=dict(origins);native=prepare_evidence(target_state,owner,execution_context=context(grants))
        targets=(f[4][0],native,f[4][2])
        payload=finite_interpretation_origin_payload(f[2],targets,f[5],'observed',context(grants))
        grants[500]=publish(500,owner,'maps',payload);grants[501]=publish(501,owner,'evidence',payload)
        field_map=PreparedAvailableFiniteFieldMap(f[2],targets,f[5],'observed',{'maps':500,'relationship':501},context(grants))
        source,parent,_=reference_model(f,grants,evidence=native,evidence_state=target_state,available_map=field_map)
        payload=finite_operator_origin_payload(field_map,operator_declaration,'observed',context(grants))
        grants[1000]=publish(1000,owner,'maps',payload);grants[1001]=publish(1001,owner,'evidence',payload)
        op=PreparedAvailableFiniteFieldOperator(field_map,operator_declaration,'observed',{'maps':1000,'relationship':1001},context(grants))
        state=cpp_available_finite_operator_model(source,reference_state=parent,available_operator=op,execution_context=context(grants))
        models.append((source,state,op,grants,target_state))
    return f,initial,learned,tuple(models)


def held_instance(op,base,x,*,namespace=990010,key=8801):
    from rexgraph_cpp import PreparedAvailableFiniteSource,finite_constraint_origin_payload,finite_operator_input_origin_payloads
    template=independent(x,namespace=namespace);asset=template[4].raw().asset();held=template[-1]
    grants=dict(base);d=template[6];d=dict(d,key=key);transport=d['transport']
    schema=op.source_declaration(asset,transport)
    grants[1100]=publish(1100,held,'input-asset',asset.archive())
    grants[1101]=publish(1101,held,'evidence',finite_constraint_origin_payload(asset,schema,'observed'))
    source=PreparedAvailableFiniteSource(asset,schema,'observed',{'asset':1100,'constraints':1101},context(grants))
    rows=[]
    for i,p in enumerate(op.select_ports(0,4,context(grants))):
        use=312 if p['rule']==6403 else 311;actual=op.input_schema(asset,transport,p['key']);field=tensor(x)
        payloads=finite_operator_input_origin_payloads(actual,use,field)
        keys={'asset':1100,'value':1110+3*i,'accession':1111+3*i,'reconstruction':1112+3*i}
        for name,label in (('value','input-value'),('accession','input-accession'),('reconstruction','reconstruction')):
            grants[keys[name]]=publish(keys[name],held,label,payloads[name])
        rows.append({'port':p['key'],'schema':actual,'use':use,'field':field,'origins':keys})
    keys=dict(zip(('transport_maps','transport_relationship','maps','relationship','instance'),range(1140,1145),strict=True))
    payload=op.transport_payload(source,transport,'observed',context(grants))
    grants[1140]=publish(1140,held,'maps',payload);grants[1141]=publish(1141,held,'evidence',payload)
    payload=op.child_map_payload(source,d,tuple(rows),'observed',context(grants))
    grants[1142]=publish(1142,held,'maps',payload);grants[1143]=publish(1143,held,'evidence',payload)
    payload=op.instance_payload(source,d,tuple(rows),'observed',keys,context(grants))
    grants[1144]=publish(1144,held,'prospective',payload)
    return op.instantiate(source,d,tuple(rows),'observed',keys,context(grants)),grants


@pytest.mark.parametrize('trial',range(12))
def test_g3_common_training_changes_complete_independently_held_fields_and_retains_alternatives(trial):
    f,initial,learned,models=learned_operator_pair()
    x=(Q(trial+1,3),Q(trial-2,5),Q(2*trial+1,7),Q(3-trial,2));families=[];images=[]
    for source,state,op,grants,target in models:
        instance,available=held_instance(op,grants,x,namespace=990010+trial,key=8800+trial)
        authority=context(available);query=finite_operator_query(state,instance,execution_context=authority)
        output=infer_model(state,source,query,execution_context=authority)
        family=read_finite_operator_result(output,execution_context=authority,state=state)
        affine=family.select(1,1,authority)[0].raw().raw().result();families.append(affine)
        assert tuple(b.inspect(authority)['status'] for b in family.select(0,4,authority))==('interpreted','interpreted','interpreted','unsupported')
        assert family.instance(authority).use_correspondences(0,2,authority)==({'reference':111,'input':311},{'reference':112,'input':312})
        assert output.source.matches(state.source) and output.execution_receipt['actual_input']['source']!=state.source.as_record()
        assert values(affine.field_archive(1)['particular'])==x
        assert affine.field_archive(2)['direction_nonzeros']>0 and len(values(affine.field_archive(2,occurrences=True)['particular']))==19
        images.append(output.field_archive['sha256'])
    assert images[0]!=images[1] and families[0].field_archive(2)['particular']!=families[1].field_archive(2)['particular']
    assert families[0].field_archive(2)['free_parameter_count']==5 and families[1].field_archive(2)['free_parameter_count']==3
    expected,flags,later=reference(x,(-10,5,0,0,0))
    assert values(families[1].field_archive(2)['particular'])==expected
    assert values(families[1].field_archive(2,occurrences=True)['particular'])==flags
    assert values(families[1].field_archive(3)['particular'])==later
    assert learned.payload['observation_count']==initial.payload['observation_count']+1


def test_standalone_operator_output_replays_with_external_authority_and_no_model_file(tmp_path):
    import os,struct,subprocess,sys
    from pathlib import Path
    data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
    (tmp_path/'response.value').write_bytes(pack_value(pack_model(selected,native=True)))
    roles=('program','maps','metrics','input-asset','input-value','input-accession','reconstruction',
           'target-asset','target-value','interface','evidence','prospective')
    pieces=[struct.pack('<Q',len(data[3]))]
    for key,origin in data[3].items():
        info=origin.inspect();history=origin.source_archive();payload=origin.payload()
        pieces.extend((struct.pack('<QB',key,roles.index(info['role'])),struct.pack('<Q',len(history)),history,
                       struct.pack('<Q',len(payload)),payload,struct.pack('<QQ',info['namespace'],info['sequence'])))
    (tmp_path/'authority.bin').write_bytes(b''.join(pieces))
    code='from pathlib import Path;import sys;from test_relational_finite_field_admission import authority_from_fixture;from rexgraph.model_codec import unpack_model;from rexgraph.value_codec import unpack_value;from rexgraph.cpp_relational_finite_operator import read_finite_operator_result;p=Path(sys.argv[1]);assert not (p/"model.value").exists();o=unpack_model(unpack_value((p/"response.value").read_bytes()));a=authority_from_fixture(p/"authority.bin");r=read_finite_operator_result(o,execution_context=a);assert r.archive()==o.field_archive["data"] and r.family(a).instance(a).source(a).raw().asset().recover(311)==(b"oak",b"",b"ak")'
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run((sys.executable,'-c',code,str(tmp_path)),cwd=tmp_path,env=env,check=True)


def test_rex_client_operator_record_transfer_preserves_workspace_and_audit_boundaries(tmp_path,monkeypatch):
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
    app=FastAPI();app.include_router(router);data,source,parent,state,targets=fixture();query,output,selected,authority=complete_model(data,source,state)
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
            assert read_finite_operator_result(response,state=restored,execution_context=authority).archive()==selected.field_archive['data']
            assert received.record.meta['workspace']=='alpha' and received.record.meta['stored_by']=='finite-reader'
            assert client.rex_audit()['integrity']['valid']
            other=RexClient('http://testserver',api_key=other_token,workspace='beta',frame_key='finite-frame-key')
            with pytest.raises(httpx.HTTPStatusError) as denied:other.rex_fetch_record(receipt.destination_record_id)
            assert denied.value.response.status_code==404
    finally:
        reset_default_store();auth.reset_auth_manager();audit.reset_cache()
