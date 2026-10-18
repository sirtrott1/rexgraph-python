"""Installed product evidence ownership, native whole fields and application lifecycle."""
import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
from rexgraph_cpp import (
    NativeSource, NativeInputUseAsset, NativeValueOrigin, PreparedAvailableFieldProgram,
    PreparedAvailableKnownProductEvidence, NativeKnownProductFamily, known_product_root_payload,
    field_target_origin_payload,
)
from test_relational_known_tensor_square import fixture as square_fixture, context, tensor, identity


def fixture():
    producer,submission,square,input,origins,current=square_fixture()
    supplied=input.execute(current);program=supplied.result().program()
    parent_asset=NativeInputUseAsset.from_archive(input.parent()['asset'])
    source=NativeSource(parent_asset.source_archive())
    declaration={"profile":"rexgraph.relational-field-parameters.v1","key":720,"revision":1,"node":2,
        "variables":tuple((1,0,i) for i in range(6)),"basis":identity(6),"parameters":tuple(range(1000,1006))}
    def publish(key,role,payload):origins[key]=NativeValueOrigin(key,source.archive,source.identity['sequence'],role,payload)
    publish(17,'evidence',known_product_root_payload(program,declaration))
    current=context(origins)
    available=PreparedAvailableFieldProgram(program,{'program':9,'maps':10,'metrics':11},current)
    root=PreparedAvailableKnownProductEvidence(available,declaration,17,current)
    target=NativeInputUseAsset(source.archive,{"profile":"rexgraph.relational-input-use-asset.v1","key":900,"channel":700,
        "encoding":"UTF-8","bytes":b"caf\xc3\xa9 / caf\xc3\xa9","uses":({'key':111,'cells':((0,10),(0,20),(0,30),(1,100)),
        'occurrences':tuple((1,i+1) for i in range(5)),'spans':((0,13),)},)})
    expected=tensor((0,5,0))
    publish(31,'target-asset',target.archive())
    publish(32,'target-value',field_target_origin_payload(program,2,target.archive(),111,expected))
    current=context(origins)
    row={'key':1,'input':supplied,'lineage':None,'target_asset':target,'target_use':111,'target':expected,
         'observed':((0,0),(0,1),(0,2)),'origins':{'asset':31,'value':32},'interface_origin':33,'evidence_origin':34}
    descriptor=root.observation_payload(row,current)
    publish(33,'interface',descriptor);publish(34,'evidence',descriptor)
    current=context(origins)
    return source,root,row,current,origins


def test_complete_family_parent_update_and_shared_cold_bytes():
    source,root,row,current,origins=fixture()
    before=root.execute(row['input'],current).result()
    assert not before.identified(2,1)
    child=root.append((row,),current);family=child.execute(row['input'],current)
    assert family.result().field_archive(2)['particular']==row['target']
    assert family.result().field_archive(2)['directions']
    assert all(family.result().identified(2,i) for i in range(3))
    assert child.evidence().identity()['free_parameter_count']==5
    assert child.parent().archive()==root.archive()
    assert child.origin_count()==23
    cold=PreparedAvailableKnownProductEvidence.from_archive(child.archive(),current)
    restored=NativeKnownProductFamily.from_archive(family.archive(),current)
    assert cold.archive()==child.archive() and restored.archive()==family.archive()
    with pytest.raises(ValueError,match='actual.*owner|different actual'):cold.execute(row['input'],current)


@pytest.mark.parametrize('key',tuple(range(1,16))+(17,31,32,33,34))
def test_all_product_root_target_and_observer_dependencies_rechecked(key):
    source,root,row,current,origins=fixture();child=root.append((row,),current);family=child.execute(row['input'],current)
    hidden=context(origins,(key,))
    for operation in (lambda:child.admit(hidden),lambda:family.admit(hidden),
                      lambda:PreparedAvailableKnownProductEvidence.from_archive(child.archive(),hidden)):
        with pytest.raises(ValueError,match='origin|availability'):operation()


def test_owner_lifetime_mutation_and_concurrent_whole_fields():
    source,root,row,current,origins=fixture();family=root.append((row,),current).execute(row['input'],current)
    packet=family.archive();del source,root,row;gc.collect()
    assert family.result().field_archive(2)['particular']==tensor((0,5,0))
    with pytest.raises((AttributeError,TypeError)):family.authority=True
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert tuple(pool.map(lambda _:NativeKnownProductFamily.from_archive(packet,current).archive(),range(3)))==(packet,)*3


def test_core_train_infer_cold_and_changed_source_rejection():
    from rexgraph.cpp_relational_known_product import (cpp_known_product_model,known_product_batch,known_product_query,
        read_known_product_result,clear_known_product_cache,inspect_known_product_model)
    from rexgraph.model_runtime import train_model,infer_model
    from rexgraph.model_codec import pack_model,unpack_model
    source,root,row,current,origins=fixture()
    state=cpp_known_product_model(source,available_evidence=root,execution_context=current)
    batch=known_product_batch(state,source,(row,),execution_context=current)
    learned=train_model(state,source,batch,execution_context=current)
    query=known_product_query(learned,row['input'])
    output=infer_model(learned,source,query,execution_context=current)
    assert output.values.size==0 and len(batch.dependencies)>=19
    assert read_known_product_result(output,execution_context=current,state=learned).result().field_archive(2)['particular']==row['target']
    assert learned.parent==state.coefficient_digest and learned.source.matches(state.source)
    assert unpack_model(pack_model(learned,native=True)).coefficient_digest==learned.coefficient_digest
    clear_known_product_cache();cold=inspect_known_product_model(learned,source,execution_context=current)
    restored=infer_model(learned,source,known_product_query(learned,row['input'],restore=True),execution_context=current)
    assert restored.field_archive['data']==output.field_archive['data']
    assert cold.parameter_schema_archive()==root.parameter_schema_archive()
    with pytest.raises(ValueError,match='warm query'):infer_model(learned,source,query,execution_context=current)


def test_rcql_and_system_inspect_fields_without_terminal_readout():
    from rexgraph.cpp_relational_known_product import cpp_known_product_model,known_product_batch,known_product_query
    from rexgraph.model_runtime import train_model,infer_model
    from rcql.model_operators import model_values,model_stalk
    from system.serialize import json_value
    source,root,row,current,origins=fixture();state=cpp_known_product_model(source,available_evidence=root,execution_context=current)
    learned=train_model(state,source,known_product_batch(state,source,(row,),execution_context=current),execution_context=current)
    output=infer_model(learned,source,known_product_query(learned,row['input']),execution_context=current)
    from rexgraph.io.model_record import model_record
    record=model_record(source,learned)
    assert model_values(record,output,execution_context=current)['fields']
    assert model_stalk(record,output,(0,1),node=2,execution_context=current)['identified']==(True,)
    summary=json_value(output)
    assert summary['retains_whole_fields'] and summary['retains_complete_alternatives']


def test_equal_query_packets_retain_distinct_actual_input_owners():
    from rexgraph.cpp_relational_known_product import cpp_known_product_model,known_product_query
    from rexgraph.model_runtime import infer_model
    source,root,row,current,origins=fixture()
    state=cpp_known_product_model(source,available_evidence=root,execution_context=current)
    first=known_product_query(state,row['input'])
    expected=infer_model(state,source,first,execution_context=current)
    other_source,other_root,other_row,other_current,other_origins=fixture()
    second=known_product_query(state,other_row['input'])
    assert first['input']==second['input'] and first['lineage']==second['lineage']
    assert first['dispatch_token']!=second['dispatch_token']
    assert infer_model(state,source,first,execution_context=current).field_archive['data']==expected.field_archive['data']
    with pytest.raises(ValueError,match='actual.*owner|different actual'):
        infer_model(state,source,second,execution_context=other_current)
    with pytest.raises(ValueError,match='origin|availability'):
        infer_model(state,source,first,execution_context=context(origins,(12,)))
    assert known_product_query(state,row['input'],restore=True)['dispatch_token'] is None


def test_warm_model_retains_owner_under_fresh_independent_current_context():
    from rexgraph.cpp_relational_known_product import cpp_known_product_model,known_product_query
    from rexgraph.model_runtime import infer_model
    source,root,row,current,origins=fixture()
    state=cpp_known_product_model(source,available_evidence=root,execution_context=current)
    query=known_product_query(state,row['input'])
    expected=infer_model(state,source,query,execution_context=current)
    assert infer_model(state,source,query,execution_context=context(origins)).field_archive['data']==expected.field_archive['data']
    with pytest.raises(ValueError,match='origin|availability'):
        infer_model(state,source,query,execution_context=context(origins,(12,)))
    assert infer_model(state,source,query,execution_context=context(origins)).field_archive['data']==expected.field_archive['data']


def lineage_fixture():
    from rexgraph_cpp import NativeAvailableInitialInputLineage,initial_input_lineage_payload
    source,root,row,current,origins=fixture()
    flags=((10,True,False,1),(20,True,True,1),(10,True,True,1),(20,False,False,1),(30,True,False,0))
    original_source=NativeSource.from_declaration({'profile':'declared-role-source-v1','namespace':998200,
        'semantic_scale':2,'max_grade':1,'role_aspects':('observed-use',),
        'types':({'name':'context','min_arity':1,'max_arity':8},),'witnesses':(10,20,30),
        'relations':({'grade':1,'key':100,'type':0,'occurrences':tuple({'key':i+1,'lower':lower,
            'exists':active,'negative':negative,'magnitude':magnitude,'aspects':(0,)}
            for i,(lower,active,negative,magnitude) in enumerate(flags))},),'carriers':(),'fields':(),'metrics':()})
    original=NativeInputUseAsset(original_source.archive,{'profile':'rexgraph.relational-input-use-asset.v1',
        'key':599,'channel':700,'encoding':'UTF-8','bytes':b'caf\xc3\xa9 / caf\xc3\xa9',
        'uses':({'key':111,'cells':((1,100),),'occurrences':tuple((1,i+1) for i in range(5)),
            'spans':((0,5),(8,13),(0,0),(0,5))},)})
    declaration={'profile':'rexgraph.initial-input-lineage.v1','key':41111,'revision':1,'uses':((111,111),)}
    origins[35]=NativeValueOrigin(35,original_source.archive,original_source.identity['sequence'],'input-asset',original.archive())
    origins[36]=NativeValueOrigin(36,source.archive,source.identity['sequence'],'interface',
        initial_input_lineage_payload(original,row['input'],declaration))
    current=context(origins)
    lineage=NativeAvailableInitialInputLineage(original,row['input'],declaration,
        {'original':35,'initial':4,'interface':36},current)
    row={**row,'lineage':lineage}
    descriptor=root.observation_payload(row,current)
    for key,role in ((33,'interface'),(34,'evidence')):
        origins[key]=NativeValueOrigin(key,source.archive,source.identity['sequence'],role,descriptor)
    return source,root,row,context(origins),origins,original


def test_initial_lineage_retains_original_bytes_flags_and_complete_learning():
    from rexgraph_cpp import NativeAvailableInitialInputLineage
    source,root,row,current,origins,original=lineage_fixture()
    lineage=row['lineage']
    assert lineage.asset_archive()==original.archive()
    assert lineage.asset_archive('initial')==row['input'].input().parent()['asset']
    assert len(lineage.dependency_archives())==3
    cold=NativeAvailableInitialInputLineage.from_archive(lineage.archive(),row['input'],current)
    assert cold.archive()==lineage.archive()
    child=root.append((row,),current)
    family=child.execute(row['input'],current,lineage=lineage)
    restored=NativeKnownProductFamily.from_archive(family.archive(),current)
    assert restored.lineage().archive()==lineage.archive()
    assert restored.archive()==family.archive()
    assert restored.result().field_archive(2)['particular']==row['target']
    with pytest.raises((AttributeError,TypeError)):lineage.__init__(original,row['input'],{}, {},current)


@pytest.mark.parametrize('key',(35,4,36))
def test_initial_lineage_rechecks_all_three_origins(key):
    from rexgraph_cpp import NativeAvailableInitialInputLineage
    source,root,row,current,origins,original=lineage_fixture();hidden=context(origins,(key,))
    for operation in (lambda:row['lineage'].admit(hidden),
        lambda:NativeAvailableInitialInputLineage.from_archive(row['lineage'].archive(),row['input'],hidden)):
        with pytest.raises(ValueError,match='origin|availability'):operation()


def test_finite_teacher_guard_rejects_generic_rational_carrier():
    from rexgraph_cpp import require_four_use_grammar_image
    from test_relational_known_tensor_square import program,source
    producer,submission,square,input,origins,current=square_fixture()
    parent=PreparedAvailableFieldProgram(program(source()),{'program':1,'maps':2,'metrics':3},current)
    with pytest.raises(ValueError,match='four|grammar|teacher'):
        require_four_use_grammar_image(parent,submission,current)


def test_typed_rcql_product_lifecycle_retains_complete_fields():
    from rcql import Executor,parse
    from rexgraph.cpp_relational_known_product import cpp_known_product_model,known_product_batch,known_product_query,clear_known_product_cache
    from rexgraph.io.model_record import model_record
    source,root,row,current,origins=fixture()
    initial=cpp_known_product_model(source,available_evidence=root,execution_context=current)
    record=model_record(source,initial)
    executor=Executor(sources={'model':record},params={'config':{'available_evidence':root,'execution_context':current}})
    created=executor.execute(parse('FROM $model RETURN MODEL_INIT("cpp_relational_known_product", $config)')).values[0]
    assert created.coefficient_digest==initial.coefficient_digest
    batch=known_product_batch(created,source,(row,),execution_context=current)
    clear_known_product_cache()
    learned=Executor(sources={'model':record},params={'state':created,'batch':batch,'context':current}).execute(parse(
        'FROM $model RETURN MODEL_TRAIN($state,$batch,1,$context)')).values[0]
    query=known_product_query(learned,row['input'],restore=True)
    result=Executor(sources={'model':model_record(source,learned)},params={'state':learned,'query':query,'context':current,'cell':(0,1)}).execute(parse(
        'FROM $model LET o = MODEL_INFER($state,$query,$context) RETURN MODEL_VALUES(o,$context), MODEL_STALK(o,$cell,2,256,$context), MODEL_INFO($state,$context)'))
    assert result.values[0]['fields'] and result.values[1]['identified']==(True,)
    assert result.values[2]['retains_correlated_families'] and result.values[2]['trainable']


@pytest.mark.parametrize('backend',('sqlite','file','rex'))
def test_checked_rcdb_copy_and_current_restore(tmp_path,backend):
    from rcdb import open_store,copy_record
    from rexgraph.cpp_relational_known_product import cpp_known_product_model,known_product_batch,known_product_query,read_known_product_result
    from rexgraph.model_runtime import train_model,infer_model
    from rexgraph.io.model_record import model_record,read_model_record
    source,root,row,current,origins=fixture();initial=cpp_known_product_model(source,available_evidence=root,execution_context=current)
    state=train_model(initial,source,known_product_batch(initial,source,(row,),execution_context=current),execution_context=current)
    output=infer_model(state,source,known_product_query(state,row['input']),execution_context=current)
    record=model_record(source,state);record.attach_metadata(1,0,'complete-product-response',output)
    stores=[open_store('sqlite:///'+str(tmp_path/(name+'.sqlite'))) if backend=='sqlite' else
            open_store(backend+'://'+str(tmp_path/name),read_only=False) for name in ('one','two')]
    try:
        saved=stores[0].put('product',record);copy_record(stores[0],stores[1],saved);copied=stores[1].get('product')
        restored,bound=read_model_record(copied,execution_context=current)
        value=copied.get_metadata(1,0,'complete-product-response')
        assert read_known_product_result(value,execution_context=current,state=restored).archive()==output.field_archive['data']
        query=known_product_query(restored,row['input'],restore=True)
        assert infer_model(restored,bound,query,execution_context=current).field_archive['data']==output.field_archive['data']
        with pytest.raises(ValueError):read_model_record(copied,execution_context=context(origins,(4,)))
    finally:
        for store in stores:store.close()


def test_agent_capability_worker_calls_common_exact_runtime():
    from agent.hive import Hive
    from rexgraph.cpp_relational_known_product import cpp_known_product_model,known_product_batch,known_product_query
    from rexgraph.model_runtime import train_model,infer_model
    source,root,row,current,origins=fixture();state=cpp_known_product_model(source,available_evidence=root,execution_context=current)
    state=train_model(state,source,known_product_batch(state,source,(row,),execution_context=current),execution_context=current)
    query=known_product_query(state,row['input'])
    hive=Hive()
    hive.add_worker('exact-product',lambda request:infer_model(state,source,request,execution_context=current),capability='analyze')
    result=hive.dispatch_capability('analyze',query)['result']
    assert result.values.size==0 and result.field_archive['data']
