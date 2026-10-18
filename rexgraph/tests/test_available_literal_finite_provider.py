"""Available reader lifecycle retains whole native recipes and external authority."""
from contextlib import closing
from dataclasses import replace
import importlib.util
import os
import pytest


def prepared(future_key=None):
    cpp=pytest.importorskip('rexgraph_cpp')
    if not hasattr(cpp,'PreparedAvailableLiteralFiniteFieldMap'):pytest.skip('native available reader SDK required')
    helper=os.environ.get('REXGRAPH_CPP_AVAILABLE_READER_TEST_HELPER')
    if not helper:pytest.skip('original complete reader fixtures required')
    spec=importlib.util.spec_from_file_location('native_available_reader_fixture',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    f=mod.fixture(future_key=future_key);ctx=mod.context(f['origins']);source=f['obj']['owner'];policy=f['policy']
    from rexgraph.cpp_relational_field_admission import cpp_available_field_program_model
    from rexgraph.cpp_relational_field_evidence_admission import cpp_available_field_learning_model
    from rexgraph.cpp_relational_literal_finite_field import cpp_available_literal_finite_field_model,literal_finite_field_query
    targets=(cpp_available_field_program_model(source,available_program=f['program'],execution_context=ctx,**policy),
             cpp_available_field_learning_model(source,available_evidence=f['evidence'],execution_context=ctx,**policy))
    state=cpp_available_literal_finite_field_model(source,available_map=f['model'],targets=targets,execution_context=ctx,**policy)
    return cpp,mod,f,source,targets,state,literal_finite_field_query(state)


@pytest.fixture(scope='module')
def case():return prepared()


def test_model_whole_family_canonical_choices_and_pin_cache(case):
    cpp,mod,f,source,targets,state,q=case;ctx=mod.context(f['origins'])
    from rexgraph.cpp_relational_literal_finite_field import (read_literal_finite_field_result,prepare_literal_finite_field_model,
        evaluate_literal_finite_field,literal_finite_field_stalks,clear_literal_finite_field_cache,literal_finite_field_query)
    from rexgraph.model_runtime import infer_model
    from rexgraph.model_codec import pack_model,unpack_model
    from rexgraph.value_codec import pack_value,unpack_value
    output=infer_model(state,source,q,execution_context=ctx);family=read_literal_finite_field_result(output,execution_context=ctx,state=state)
    assert family.model(ctx).same_owner(f['model'])
    assert prepare_literal_finite_field_model(state,source,execution_context=ctx).same_owner(f['model'])
    assert family.raw(ctx).archive()==f['reader'].execute().family().archive()
    assert output.values.size==0 and family.select(1,1,ctx)[0].raw(ctx).result().field_archive(2)['free_parameter_count']==1
    f['publish'](14,'prospective',f['evidence'].parameter_setting_payload((7,)))
    f['publish'](15,'prospective',family.completion_payload(1,(7,),14,mod.context(f['origins'])))
    choice=evaluate_literal_finite_field(output,1,(7,),14,15,execution_context=mod.context(f['origins']),state=state)
    assert literal_finite_field_stalks(choice,2,((0,0),),execution_context=mod.context(f['origins']))[0]['target_source']==targets[1].source.as_record()
    packet=pack_value(pack_model(choice));cold=unpack_model(unpack_value(packet));assert pack_value(pack_model(cold))==packet
    clear_literal_finite_field_cache()
    assert read_literal_finite_field_result(cold,execution_context=mod.context(f['origins']),state=state).archive()==choice.field_archive['data']
    restored=unpack_model(pack_model(state));original=cpp.NativeSource(source.archive)
    assert infer_model(restored,original,literal_finite_field_query(restored),execution_context=mod.context(f['origins'])).field_archive['data']==output.field_archive['data']
    from rexgraph.model_contract import model_provider_inventory,resolve_model_provider
    assert any(r['name']=='rexgraph.cpp.relational_literal_finite_field' for r in model_provider_inventory())
    with pytest.raises(NotImplementedError):resolve_model_provider(state,'train')
    bad=replace(state,configuration={**state.configuration,'engine_receipt':{'engine_headers':'changed'}})
    with pytest.raises(ValueError):prepare_literal_finite_field_model(bad,source,execution_context=ctx)


@pytest.mark.parametrize('key',range(1,14))
@pytest.mark.parametrize('route',('infer','prepare','output','cold-output','cold-model'))
def test_all_reader_and_base_dependencies_rechecked(case,key,route):
    cpp,mod,f,source,targets,state,q=case
    from rexgraph.cpp_relational_literal_finite_field import read_literal_finite_field_result,prepare_literal_finite_field_model,literal_finite_field_query
    from rexgraph.model_runtime import infer_model
    from rexgraph.model_codec import pack_model,unpack_model
    output=infer_model(state,source,q,execution_context=mod.context(f['origins']));denied=mod.context(f['origins'],(key,))
    with pytest.raises(ValueError):
        if route=='infer':infer_model(state,source,q,execution_context=denied)
        elif route=='prepare':prepare_literal_finite_field_model(state,source,execution_context=denied)
        elif route=='output':read_literal_finite_field_result(output,execution_context=denied)
        elif route=='cold-output':read_literal_finite_field_result(unpack_model(pack_model(output)),execution_context=denied)
        else:
            cold=unpack_model(pack_model(state));infer_model(cold,cpp.NativeSource(source.archive),literal_finite_field_query(cold),execution_context=denied)


@pytest.mark.parametrize('key',(3,6,7,10,13))
def test_actual_future_publication_cutoff(case,key):
    cpp,mod,f,source,targets,state,q=prepared(future_key=key)
    from rexgraph.cpp_relational_literal_finite_field import read_literal_finite_field_result,prepare_literal_finite_field_model
    from rexgraph.model_runtime import infer_model
    from rexgraph.model_codec import pack_model,unpack_model
    output=infer_model(state,source,q,execution_context=mod.context(f['origins']));denied=mod.context(f['origins'],cutoffs={key:0})
    assert f['origins'][key].inspect()['sequence']==1
    for call in (lambda:infer_model(state,source,q,execution_context=denied),
                 lambda:prepare_literal_finite_field_model(state,source,execution_context=denied),
                 lambda:read_literal_finite_field_result(output,execution_context=denied),
                 lambda:read_literal_finite_field_result(unpack_model(pack_model(output)),execution_context=denied)):
        with pytest.raises(ValueError,match='namespace availability'):call()


@pytest.mark.parametrize('fmt',('rcbd','safetensors','h5','zarr','json'))
def test_complete_model_and_output_five_container_roundtrip(case,tmp_path,fmt):
    cpp,mod,f,source,targets,state,q=case;ctx=mod.context(f['origins'])
    from rexgraph.model_runtime import infer_model
    from rexgraph.io.model_record import model_record,read_model_record
    from rexgraph.io import save,load
    from rexgraph.cpp_relational_literal_finite_field import read_literal_finite_field_result,literal_finite_field_query
    output=infer_model(state,source,q,execution_context=ctx);record=model_record(source,state);record.attach_metadata(1,0,'whole-reader-output',output)
    path=tmp_path/('whole-reader.'+fmt);save(path,record);loaded=load(path)
    cold,owner=read_model_record(loaded,execution_context=ctx);assert cold.coefficient_digest==state.coefficient_digest
    saved=loaded.get_metadata(1,0,'whole-reader-output');assert read_literal_finite_field_result(saved,execution_context=ctx,state=cold).archive()==output.field_archive['data']
    assert infer_model(cold,owner,literal_finite_field_query(cold),execution_context=ctx).field_archive['data']==output.field_archive['data']
    with pytest.raises(ValueError):read_literal_finite_field_result(saved,execution_context=mod.context(f['origins'],(13,)))


@pytest.mark.parametrize('kind',('local','sqlite','object'))
def test_checked_store_copies_and_rcql(case,tmp_path,kind):
    from rcdb import MemoryStore,LocalStore,SQLStore,NativeObjectStore,copy_record
    from rexgraph.io.model_record import model_record
    from rcql import Executor,parse
    from rexgraph.cpp_relational_literal_finite_field import read_literal_finite_field_result
    cpp,mod,f,source,targets,state,q=case;ctx=mod.context(f['origins'])
    store={'local':lambda:LocalStore(tmp_path/'records'),'sqlite':lambda:SQLStore('sqlite:///'+str(tmp_path/'reader.db')),
           'object':lambda:NativeObjectStore('file://'+str(tmp_path/'objects'))}[kind]()
    with closing(MemoryStore()) as origin,closing(store) as target:
        original=origin.put('reader',model_record(source,state),analytics=False)
        receipt=copy_record(origin,target,original,return_receipt=True);assert receipt.source_digest==receipt.destination_digest
        executor=Executor(sources={'r':target.get('reader')},params={'authority':ctx,'q':q})
        result=executor.execute(parse('FROM $r RETURN MODEL_INFER(MODEL_STATE(execution_context=$authority),$q,execution_context=$authority)'))
        output=result.values[0];assert read_literal_finite_field_result(output,execution_context=ctx).inspect(ctx)['branch_count']==2


def test_rcql_operations_System_preview_and_agent_recheck(case):
    from rcql import Executor,parse
    from rexgraph.io.model_record import model_record
    from rexgraph.cpp_relational_literal_finite_field import (read_literal_finite_field_result,evaluate_literal_finite_field)
    from rexgraph.model_runtime import infer_model
    from system.serialize import json_value
    from agent.hive import Hive
    import asyncio
    from system.server.app import QueryRequest,query as system_query
    from system.state import sources
    cpp,mod,f,source,targets,state,q=case;ctx=mod.context(f['origins']);record=model_record(source,state)
    assert json_value(state)['retains_whole_fields']
    output=infer_model(state,source,q,execution_context=ctx);family=read_literal_finite_field_result(output,execution_context=ctx)
    f['publish'](14,'prospective',f['evidence'].parameter_setting_payload((7,)));ctx=mod.context(f['origins'])
    f['publish'](15,'prospective',family.completion_payload(1,(7,),14,ctx));ctx=mod.context(f['origins'])
    params={'authority':ctx,'q':q,'state':state,'output':output,'configuration':{'available_map':f['model'],'targets':targets,**f['policy'],'execution_context':ctx}}
    executor=Executor(sources={'r':record},params=params)
    queries=('MODEL_STATE(execution_context=$authority)',
             'MODEL_INFO($state,execution_context=$authority)',
             'MODEL_VALUES($output,execution_context=$authority)',
             'MODEL_FINITE_BRANCHES($output,begin=0,count=2,execution_context=$authority)',
             'MODEL_FINITE_COMPLETE($output,ordinal=1,values=[7],parameter_setting=14,choice_origin=15,state=$state,execution_context=$authority)')
    for expression in queries:assert executor.execute(parse('FROM $r RETURN '+expression)).values
    assert executor.execute(parse('FROM $r RETURN MODEL_INIT(adapter="cpp_relational_literal_finite_field",configuration=$configuration)')).values
    selected=evaluate_literal_finite_field(output,1,(7,),14,15,execution_context=ctx,state=state)
    params['selected']=selected
    assert Executor(sources={'r':record},params=params).execute(parse('FROM $r RETURN MODEL_STALK($selected,cell=[0,0],node=2,execution_context=$authority)')).values
    name='available_literal_reader';sources.register(name,record)
    try:
        request=QueryRequest(query='FROM $'+name+' RETURN MODEL_INFER(MODEL_STATE(execution_context=$authority),$q,execution_context=$authority)',params={'authority':ctx,'q':q})
        preview=asyncio.run(system_query(request));assert preview['values'][0]['retains_whole_fields']
        authority=[ctx];hive=Hive();hive.add_worker(name,lambda query:infer_model(state,source,query,execution_context=authority[0]),capability='analyze')
        assert hive.dispatch_capability('analyze',q)['result'].model_digest==state.coefficient_digest
        authority[0]=mod.context(f['origins'],(13,))
        with pytest.raises(ValueError):hive.dispatch_capability('analyze',q)
        request.params['authority']=authority[0]
        with pytest.raises(Exception) as error:asyncio.run(system_query(request))
        assert getattr(error.value,'status_code',None)==400
    finally:sources.remove(name)


def test_source_backed_evidence_child_recomposition_preserves_parent_and_future_target():
    cpp,mod,f,source,targets,parent,q=prepared(future_key=21)
    from rexgraph.model_runtime import infer_model,train_model
    from rexgraph.cpp_relational_field_evidence_admission import available_field_program_batch
    from rexgraph.cpp_relational_literal_finite_field import (cpp_available_literal_finite_field_model,
        literal_finite_field_query,read_literal_finite_field_result)
    from rexgraph.model_codec import pack_model,unpack_model
    policy=f['policy'];ctx=mod.context(f['origins']);old=infer_model(parent,source,q,execution_context=ctx)
    # A constructed exact native measurement exercises the software lifecycle;
    # it supplies no linguistic annotation or language quality evidence.
    tensor=f['evidence'].execute_inputs((dict(f['submission'],asset=f['obj']['asset'].archive()),),ctx).result().evaluate((2,)).field_archive(2)
    publisher=cpp.NativeSource.from_declaration({'profile':'declared-role-source-v1','namespace':270001,'semantic_scale':3,
        'max_grade':1,'role_aspects':('control-measurement',),'types':({'name':'control','min_arity':1,'max_arity':1},),
        'witnesses':(1,),'relations':({'grade':1,'key':10001,'type':0,'occurrences':({'key':1,'lower':1,'aspects':(0,),
        'exists':True,'negative':False,'magnitude':1},)},),'fields':(),'metrics':(),'carriers':()})
    asset=cpp.NativeInputUseAsset(publisher.archive,{'profile':'rexgraph.relational-input-use-asset.v1','key':71001,'channel':71002,
        'encoding':'RFCXT-control','bytes':tensor,'uses':({'key':1,'cells':((0,1),(1,10001)),
        'occurrences':((1,1),),'spans':((0,len(tensor)),)},)},**policy)
    f['origins'][20]=cpp.NativeValueOrigin(20,publisher.archive,publisher.identity['sequence'],'target-asset',asset.archive(),**policy)
    f['publish'](21,'target-value',cpp.field_target_origin_payload(f['obj']['program'],2,asset.archive(),1,tensor,**policy))
    observation={'key':72001,'inputs':(dict(f['submission'],asset=f['obj']['asset'].archive()),),
        'target_asset':asset.archive(),'target_use':1,'target':tensor,'observed':((0,0),),'origins':{'asset':20,'value':21}}
    ctx=mod.context(f['origins']);native_child=f['evidence'].append_inputs((observation,),ctx)
    batch=available_field_program_batch(targets[1],source,(observation,),execution_context=ctx)
    learned=train_model(targets[1],source,batch,steps=1,execution_context=ctx)
    assert learned.parent==targets[1].coefficient_digest and learned.step==1
    assert native_child.parent().archive()==f['evidence'].archive()
    assert native_child.evidence().identity()['free_parameter_count']==0
    next_targets=(f['program'],native_child)
    payload=cpp.finite_interpretation_origin_payload(f['finite'],next_targets,f['spec'],'observed',ctx,**policy)
    f['publish'](31,'maps',payload);f['publish'](32,'evidence',payload);ctx=mod.context(f['origins'])
    base=cpp.PreparedAvailableFiniteFieldMap(f['finite'],next_targets,f['spec'],'observed',{'maps':31,'relationship':32},ctx,**policy)
    reader=cpp.PreparedLiteralFiniteFieldMap(f['finite'].raw(),(f['program'].program(),native_child.evidence()),f['request'],**policy)
    f['publish'](33,'interface',cpp.literal_finite_interface_origin_payload(base,reader,((10,),(10,)),ctx,**policy));ctx=mod.context(f['origins'])
    native=cpp.PreparedAvailableLiteralFiniteFieldMap(base,reader,{'inputs':((10,),(10,)),'interface':33},ctx,**policy)
    child=cpp_available_literal_finite_field_model(source,available_map=native,targets=(targets[0],learned),execution_context=ctx,**policy)
    after=infer_model(child,source,literal_finite_field_query(child),execution_context=ctx)
    fields=read_literal_finite_field_result(after,execution_context=ctx).select(1,1,ctx)[0].raw(ctx).result()
    assert fields.field_archive(2)['free_parameter_count']==0 and fields.field_archive(2)['particular']==tensor
    assert infer_model(parent,source,q,execution_context=ctx).field_archive['data']==old.field_archive['data']
    assert f['evidence'].evidence().identity()['free_parameter_count']==1
    complete=read_literal_finite_field_result(after,execution_context=ctx)
    with pytest.raises(ValueError):complete.completion_payload(1,(),0,ctx)
    f['publish'](34,'prospective',native_child.parameter_setting_payload(()));ctx=mod.context(f['origins'])
    f['publish'](35,'prospective',complete.completion_payload(1,(),34,ctx));ctx=mod.context(f['origins'])
    selected=complete.evaluate(1,(),34,35,ctx)
    assert selected.inspect(ctx)['parameter_setting_receipt']==f['origins'][34].receipt()
    denied=mod.context(f['origins'],cutoffs={21:0});assert f['origins'][21].inspect()['sequence']==1
    with pytest.raises(ValueError,match='namespace availability'):infer_model(child,source,literal_finite_field_query(child),execution_context=denied)
    with pytest.raises(ValueError,match='namespace availability'):read_literal_finite_field_result(unpack_model(pack_model(after)),execution_context=denied)
