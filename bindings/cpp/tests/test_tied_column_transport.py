"""Named native incidence templates with complete encoded column evidence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import subprocess
import sys

import pytest
from rexgraph_cpp import PreparedStalkTransport, encode_stalk_column_tensor, stalk_transport_build_info
from rexgraph.cpp_column_transport import column_transport_observation, column_transport_observation_group
from rexgraph.cpp_stalk_transport import (
    cpp_stalk_transport_model, enable_stalk_transport_provider, replay_stalk_transport_model,
    stalk_transport_query, stalk_transport_field, _prepare, _cache,
)
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_state import thaw_tree
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from test_tied_stalk_transport import model, train_first_field, SPACE, BASIS


@pytest.fixture(scope='module')
def wide_sparse_bank():
    from rexgraph_cpp import NativeSource
    declaration=dict(profile='declared-role-source-v1',namespace=997081,semantic_scale=0,max_grade=1,
        role_aspects=('boundary',),types=(dict(name='context',min_arity=1,max_arity=1),),witnesses=(10,),
        relations=tuple(dict(grade=1,key=i+1,type=0,occurrences=(
            dict(key=i+1,lower=10,negative=False,magnitude=1,exists=True,aspects=(0,)),)) for i in range(3)),
        fields=(),metrics=(),carriers=(dict(key=10,kind='source_grade',grade=0,coordinates=0,size=1),
            dict(key=11,kind='source_grade',grade=1,coordinates=0,size=3)))
    source=NativeSource.from_declaration(declaration)
    lower=tuple('out:'+str(i) for i in range(40));upper=tuple('in:'+str(i) for i in range(30))
    slots=tuple((p,i,j) for p in range(3) for i in range(40) for j in range(30))
    native=PreparedStalkTransport(source.archive,((lower,),(upper,)*3),((),((),)*3),slots,
        parameter_space=dict(name='shared-map',keys=tuple('p:'+str(i) for i in range(1200))),
        parameter_basis=tuple((i,i%1200,1) for i in range(3600)),column_fields=True,
        limits=(16384,16000000,64000000,16384))
    native=native.append_columns(tuple(dict(input=((i,encode_stalk_column_tensor(30,((i,1),))),),
        target=((0,encode_stalk_column_tensor(40,())),)) for i in range(3)))
    identity=lambda n:tuple((i,i,1) for i in range(n))
    frames=((('outputs',lower,identity(40),identity(40)),),
        (('inputs',upper,identity(30),identity(30)),)*3)
    bank=native.learned_pattern_bank((0,1,2),(0,),frames,minimum_peers=2,declared_templates=True,
        model_namespace=997082,coordinate_ids=dict(weights=997083,parameters=997084,domains=(997085,)),
        evidence_digest='7'*64)
    assert bank.inspect()['parameter_directions']['shape']==(1200,1080)
    return bank


@pytest.mark.parametrize('suffix',['.safetensors','.rcbd'])
def test_large_sparse_bank_uses_declared_codec_limits_across_model_storage(wide_sparse_bank,tmp_path,suffix):
    from rexgraph_cpp import NativeSource,PreparedStalkTransportPatternBank,PreparedRelationGuards
    from rexgraph.cpp_stalk_pattern_model import (
        stalk_pattern_bank_source,cpp_stalk_pattern_bank_model,stalk_pattern_bank_query,select_stalk_pattern_bank_model,
    )
    image=wide_sparse_bank.archive()
    with pytest.raises(ValueError,match='logical size'):NativeSource(image)
    with pytest.raises(ValueError,match='logical size'):PreparedRelationGuards(image)
    guards=PreparedRelationGuards(image,limits=(wide_sparse_bank.limits[1],wide_sparse_bank.limits[2],wide_sparse_bank.limits[3]))
    assert guards.boundary(0)['type']=='context'
    source=stalk_pattern_bank_source(image,limits=wide_sparse_bank.limits)
    state=cpp_stalk_pattern_bank_model(source,limits=wide_sparse_bank.limits)
    path=tmp_path/('wide'+suffix);save_model(path,model_record(source,state))
    restored,owner=read_model_record(load_model(path))
    bank=PreparedStalkTransportPatternBank(owner.archive,model_namespace=997082,limits=wide_sparse_bank.limits)
    assert bank.archive()==image and bank.inspect()==wide_sparse_bank.inspect()
    known=(1,)+(0,)*29;unknown=(0,)*29+(1,)
    for value in (known,unknown):
        a=infer_model(state,source,stalk_pattern_bank_query(state,{0:value}))
        b=infer_model(restored,owner,stalk_pattern_bank_query(restored,{0:value}))
        assert a.execution_receipt['result']==b.execution_receipt['result'];b.check_state()
    selected,local=select_stalk_pattern_bank_model(restored,owner,(0,),model_namespace=997091,
        coordinate_ids=dict(weights=997092,parameters=997093))
    infer_model(selected,local,stalk_pattern_bank_query(selected,{0:unknown})).check_state()


def test_source_cache_cannot_bypass_narrower_tensor_reader_admission(wide_sparse_bank):
    from rexgraph.cpp_stalk_pattern_model import stalk_pattern_bank_source,cpp_stalk_pattern_bank_model
    from rexgraph.cpp_model import cpp_source_record,restore_cpp_source,_SourceRestoreCache
    source=stalk_pattern_bank_source(wide_sparse_bank.archive(),limits=wide_sparse_bank.limits)
    state=cpp_stalk_pattern_bank_model(source,limits=wide_sparse_bank.limits)
    record=cpp_source_record(source,state.source);cache=_SourceRestoreCache()
    options=dict(max_dimension=max(wide_sparse_bank.limits[:2]),max_records=wide_sparse_bank.limits[1])
    first=restore_cpp_source(record,state.source,_source_cache=cache,**options)
    assert restore_cpp_source(record,state.source,_source_cache=cache,**options) is first
    with pytest.raises(ValueError,match='logical size'):
        restore_cpp_source(record,state.source,_source_cache=cache)


def columns(basis=BASIS,space=SPACE):
    source,dense=model(basis,space)
    c=dense.configuration
    return source,dense,cpp_stalk_transport_model(source,c['stalk_axes'],c['fixed_maps'],c['parameter_slots'],
        c['training_rows'],parameter_space=space,parameter_basis=basis,column_fields=True)


def acquire(state,source,targets=None):
    packet=column_transport_observation(state,{0:((0,1),)},targets if targets is not None else {0:((0,-1),(1,Q(2,5))),1:((0,1),)})
    batch=column_transport_observation_group(state,(packet,))
    return train_model(state,source,batch),batch


def mathematical_fit(fit):
    fields=('particular','kernel','kernel_shape','rank','pivots','free_columns','constraints','design','target',
        'parameter_slots','parameter_space','parameter_basis','basis_shape','basis_rank','parameterization',
        'occurrence_keys','occurrence_coefficients','occurrence_activity')
    return {key:fit[key] for key in fields}


@pytest.mark.parametrize('basis,space',[(BASIS,SPACE),
    (tuple((i,j,Q(1)) for i in range(9) for j in (0,1)),dict(name='dependent',keys=('left','right')))])
def test_column_fit_equals_complete_named_whole_field_family(basis,space):
    source,dense,sparse=columns(basis,space)
    if basis==BASIS:
        dense=train_first_field(dense,source);sparse,batch=acquire(sparse,source)
    else:
        from rexgraph.cpp_stalk_transport import stalk_transport_observation_batch
        dense=train_model(dense,source,stalk_transport_observation_batch(dense,((1,0,0),),((0,(-1,-1)),(1,(Q(2,3),)))))
        sparse,batch=acquire(sparse,source,{0:((0,-1),(1,-1)),1:((0,Q(2,3)),)})
    assert sparse.adapter_version=='4' and batch.targets['profile']=='encoded-original-stalk-targets-v1'
    assert mathematical_fit(dense.payload['fit'])==mathematical_fit(sparse.payload['fit'])
    assert _prepare(sparse,source).execution_stats['materialized_field_entries']==0
    for field in (((0,5,7),),((Q(1,2),Q(-2,3),5),),((0,0,0),)):
        a=infer_model(dense,source,stalk_transport_query(dense,field))
        b=infer_model(sparse,source,stalk_transport_query(sparse,field))
        assert a.execution_receipt['response']==b.execution_receipt['response']
        assert a.execution_receipt['boundary_identified']==b.execution_receipt['boundary_identified']
        b.check_state()


def test_named_columns_retain_partial_identification_and_supported_zeros():
    source,_,state=columns();state,_=acquire(state,source,{0:((0,-1),(1,Q(2,5)))})
    assert state.payload['fit']['rank']==1 and state.payload['fit']['kernel_shape']==(2,1)
    output=infer_model(state,source,stalk_transport_query(state,((0,5,7),)))
    assert output.execution_receipt['boundary_identified']==((True,True),(False,))
    assert stalk_transport_field(output,0).values.tolist()==[[Q(16,3),-49]]
    with pytest.raises(TypeError):stalk_transport_field(output,1)


@pytest.mark.parametrize('suffix',['.safetensors','.rcbd'])
def test_cold_named_column_history_and_response_roundtrip(tmp_path,suffix):
    source,_,state=columns();state,batch=acquire(state,source)
    path=tmp_path/('named-columns'+suffix);save_model(path,model_record(source,state))
    _cache.clear();restored,owner=read_model_record(load_model(path))
    assert restored.coefficient_digest==state.coefficient_digest
    assert mathematical_fit(restored.payload['fit'])==mathematical_fit(state.payload['fit'])
    assert unpack_model(pack_model(batch)).coefficient_digest==batch.coefficient_digest
    query=stalk_transport_query(restored,((0,5,7),))
    assert stalk_transport_field(infer_model(restored,owner,query),0).values.tolist()==[[Q(16,3),-49]]
    code='''
import sys
from rexgraph.io.model_record import load_model,read_model_record
from rexgraph.cpp_stalk_transport import enable_stalk_transport_provider,stalk_transport_query,stalk_transport_field
from rexgraph.model_runtime import infer_model
enable_stalk_transport_provider('4')
state,source=read_model_record(load_model(sys.argv[1]))
assert state.adapter_version=='4'
from fractions import Fraction
assert stalk_transport_field(infer_model(state,source,stalk_transport_query(state,((0,5,7),))),1).values.tolist()==[[Fraction(-1,2)]]
'''
    subprocess.run([sys.executable,'-c',code,str(path)],check=True)


def test_replay_preserves_named_basis_and_exact_column_family():
    source,_,state=columns();state,_=acquire(state,source)
    replay=replay_stalk_transport_model(state,source)
    assert replay.adapter_version=='4' and mathematical_fit(replay.payload['fit'])==mathematical_fit(state.payload['fit'])
    assert replay.configuration['parameter_basis']==state.configuration['parameter_basis']


@pytest.mark.parametrize('change',['basis','history','fit','version'])
def test_changed_native_named_column_witnesses_reject(change):
    source,_,state=columns();state,_=acquire(state,source)
    configuration=thaw_tree(state.configuration);payload=thaw_tree(state.payload)
    version='4'
    if change=='basis':configuration['parameter_basis']=((0,0,Q(2)),*configuration['parameter_basis'][1:])
    if change=='history':payload['observations'][0]['input']=((0,encode_stalk_column_tensor(3,((1,1),))),)
    if change=='fit':payload['fit']['particular']=(Q(3),Q(3))
    if change=='version':version='3'
    _cache.clear()
    with pytest.raises(ValueError):
        broken=replace(state,configuration=configuration,payload=payload,adapter_version=version)
        infer_model(broken,source,stalk_transport_query(broken,((1,0,0),)))


def test_native_named_columns_do_not_accept_whole_field_packets():
    source,dense,state=columns();c=state.configuration
    native=PreparedStalkTransport(source.archive,c['stalk_axes'],c['fixed_maps'],c['parameter_slots'],
        parameter_space=dict(name=SPACE.name,keys=SPACE.keys),parameter_basis=BASIS,column_fields=True)
    assert native.receipt['profile']==stalk_transport_build_info(4)['profile']
    with pytest.raises(ValueError,match='compact'):native.append(())
    with pytest.raises(ValueError,match='compact'):
        PreparedStalkTransport(source.archive,c['stalk_axes'],c['fixed_maps'],c['parameter_slots'],
            (dict(input=((1,0,0),),target=((-1,Q(2,5)),(1,)),observed=(True,True)),),
            parameter_space=dict(name=SPACE.name,keys=SPACE.keys),parameter_basis=BASIS,column_fields=True)
    with pytest.raises(ValueError,match='column'): _prepare(dense,source).append_columns(())


def test_provider_creation_keeps_the_two_column_versions_distinct():
    source,_,state=columns();c=state.configuration
    config={key:c[key] for key in ('stalk_axes','fixed_maps','parameter_slots','training_rows')}
    with pytest.raises(ValueError,match='named'):enable_stalk_transport_provider('4').create(source,**config)
    with pytest.raises(ValueError,match='independent'):
        enable_stalk_transport_provider('3').create(source,**config,parameter_space=SPACE,parameter_basis=BASIS)
    created=enable_stalk_transport_provider('4').create(source,**config,parameter_space=SPACE,parameter_basis=BASIS)
    assert created.adapter_version=='4'
    with pytest.raises(ValueError,match='version 4'):
        enable_stalk_transport_provider('2').create(source,**config,parameter_space=SPACE,parameter_basis=BASIS,column_fields=True)


def test_native_reads_preserve_shared_owner_across_threads():
    source,_,state=columns();state,_=acquire(state,source);native=_prepare(state,source)
    digest=state.coefficient_digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=tuple(pool.map(native.apply,(((0,5,7),),)*16))
    assert all(result==results[0] for result in results)
    assert state.coefficient_digest==digest and native.execution_stats['materialized_field_entries']==0


def test_named_column_provider_runs_through_rcql_and_system_summary():
    from rcql import Executor,parse
    from system.serialize import json_value
    source,_,initial=columns();state,batch=acquire(initial,source)
    query=stalk_transport_query(state,((0,5,7),))
    result=Executor(sources={'r':model_record(source,initial)},params={'b':batch,'q':query}).execute(parse(
        'FROM $r RETURN MODEL_INFER(MODEL_TRAIN(MODEL_STATE(),$b),$q)'))
    assert result.values[0].execution_receipt['response']['boundary_family']==(((Q(16,3),-49),(Q(-1,2),)),)
    summary=json_value(state)
    assert summary['observation_group_profile']=='column-stalk-transport-observation-group-v1'
    assert summary['parameter_count']==2 and summary['supported_map_entries']==9
    assert summary['parameter_space']['keys']==list(SPACE.keys)


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_named_column_provider_can_be_retained_by_rcdb(tmp_path,backend):
    from rcdb import open_store
    source,_,state=columns();state,_=acquire(state,source)
    path={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(path,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('shared',model_record(source,state),expected_version=0)
    assert store.verify_commits('shared');store.close()
    store=open_store(path);record=store.read_record('shared',version=1).value
    restored,owner=read_model_record(record)
    query=stalk_transport_query(restored,((0,5,7),))
    assert stalk_transport_field(infer_model(restored,owner,query),1).values.tolist()==[[Q(-1,2)]]
    store.close()


def test_agent_worker_uses_the_native_named_column_provider():
    from unittest.mock import patch
    from agent.hive import Hive
    source,_,state=columns();state,_=acquire(state,source)
    query=stalk_transport_query(state,((0,5,7),))
    with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
        hive=Hive('shared');hive.add_worker('maps',lambda q,**kw:infer_model(state,source,q),capability='predict')
        response=hive.invoke('maps',query,record=False)
        assert response.model_digest==state.coefficient_digest
