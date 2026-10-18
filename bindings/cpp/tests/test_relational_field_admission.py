"""Installed native origin admission, complete field replay and package routes."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import struct

import pytest
from rexgraph_cpp import (
    NativeSource, NativeInputUseAsset, NativeInputAccession, NativeValueOrigin, NativeAvailabilityContext,
    PreparedFieldProgram, PreparedAccessionFieldProgram,
    PreparedAvailableFieldProgram, NativeAvailableFieldResult, field_input_origin_payloads,
)
from rexgraph.cpp_relational_field_admission import (
    cpp_available_field_program_model, available_field_query, read_available_field_result,
    clear_available_field_cache,
)
from rexgraph.cpp_relational_field_program import read_field_program_result, field_program_stalks
from rexgraph.model_runtime import infer_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from test_relational_field_program import source, tensor
from test_relational_field_evidence import learning_declaration, actual_asset, input_row, HELD_INPUT

PROGRAM_ROLES = ('program', 'maps', 'metrics')
INPUT_ROLES = ('input-asset', 'input-value', 'input-accession', 'reconstruction')


def context(origins, *, remove=(), limits=None, namespaces=None):
    limits = {} if limits is None else limits; namespaces = {} if namespaces is None else namespaces
    return NativeAvailabilityContext(tuple({'origin': value, 'namespace': namespaces.get(key,value.inspect()['namespace']),
        'sequence': limits.get(key,value.inspect()['sequence'])} for key,value in origins.items() if key not in remove))


def publish(key, history, role, payload, *, cutoff=None, limits=(4096,16000000,64000000,16384)):
    return NativeValueOrigin(key, history.archive, history.identity['sequence'] if cutoff is None else cutoff, role, payload,limits=limits)


def prepared(values=HELD_INPUT, *, key_offset=0):
    owner = source(); fixed = PreparedFieldProgram(owner.archive, learning_declaration())
    program = PreparedAccessionFieldProgram(fixed.archive())
    asset = actual_asset(999991,111,b'independently supplied held input')
    row = input_row(asset,111,values); payloads = field_input_origin_payloads(program,row)
    origins = {i+1:publish(i+1,owner,role,program.archive()) for i,role in enumerate(PROGRAM_ROLES)}
    for i,(name,role) in enumerate(zip(('asset','value','accession','reconstruction'),INPUT_ROLES,strict=True),4):
        key=i+key_offset;origins[key]=publish(key,owner,role,payloads[name])
    keys={'program':1,'maps':2,'metrics':3};row['origins']={name:i+key_offset for i,name in enumerate(('asset','value','accession','reconstruction'),4)}
    authority=context(origins); available=PreparedAvailableFieldProgram(program,keys,authority)
    return owner,program,available,row,origins,authority


def _reseal(image):
    image=bytearray(image);value=1469598103934665603
    for byte in image[32:]:value=((value^byte)*1099511628211)&((1<<64)-1)
    struct.pack_into('<Q',image,16,len(image));struct.pack_into('<Q',image,24,value)
    return bytes(image)


def graft_native(image, replacement):
    size=struct.unpack_from('<Q',image,32)[0]
    return _reseal(image[:32]+struct.pack('<Q',len(replacement))+replacement+image[40+size:])


def test_complete_fields_and_receipts_roundtrip_under_independent_authority():
    owner,bare,available,row,origins,authority=prepared()
    result=available.execute_inputs((row,),authority)
    asset=NativeInputUseAsset.from_archive(row['asset'])
    accession=NativeInputAccession(asset,bare,row['accession'])
    expected=bare.execute((accession.admit(row['use'],row['field']),))
    assert result.result().archive()==expected.archive()
    assert result.program_archive(authority)==available.archive()
    for node in (1,2,3):assert result.result().field_archive(node)==expected.field_archive(node)
    assert result.result().field_archive(2,occurrences=True)==expected.field_archive(2,occurrences=True)
    assert len(result.dependency_archives())==7
    assert NativeAvailableFieldResult.from_archive(result.archive(),authority).archive()==result.archive()
    assert PreparedAvailableFieldProgram.from_archive(available.archive(),authority).archive()==available.archive()
    assert origins[5].inspect()['role']=='input-value'
    assert NativeSource(origins[5].source_archive()).identity['first_sequence']==owner.identity['first_sequence']


@pytest.mark.parametrize('key',range(1,8))
def test_missing_or_foreign_origin_rejects_each_role(key):
    _,_,available,row,origins,authority=prepared(); result=available.execute_inputs((row,),authority)
    missing=context(origins,remove=(key,))
    with pytest.raises(ValueError):available.execute_inputs((row,),missing)
    with pytest.raises(ValueError):NativeAvailableFieldResult.from_archive(result.archive(),missing,limits=(1,16000000,64000000,16384))
    foreign=context(origins,namespaces={key:origins[key].inspect()['namespace']+1})
    with pytest.raises(ValueError):available.execute_inputs((row,),foreign)


@pytest.mark.parametrize('role',('target-value','prospective','metrics'))
def test_target_or_setting_never_becomes_input_by_equal_coefficients(role):
    owner,_,available,row,origins,_=prepared()
    origins[5]=publish(5,owner,role,origins[5].payload())
    with pytest.raises(ValueError,match='role or complete content'):available.execute_inputs((row,),context(origins))


def test_byte_asset_does_not_supply_missing_numeric_origin():
    _,_,available,row,origins,_=prepared()
    with pytest.raises(ValueError,match='missing actual value origin'):available.execute_inputs((row,),context(origins,remove=(5,)))
    changed=deepcopy(row);changed['field']=tensor((2,3,5,11))
    with pytest.raises(ValueError,match='role or complete content'):available.execute_inputs((changed,),context(origins))


def test_live_owner_and_declared_cold_adoption_stay_distinct():
    owner,bare,available,row,origins,authority=prepared();result=available.execute_inputs((row,),authority)
    cloned={key:publish(key,NativeSource(value.source_archive()),value.inspect()['role'],value.payload()) for key,value in origins.items()}
    other=context(cloned)
    with pytest.raises(ValueError,match='live value origin owner'):available.admit(other)
    with pytest.raises(ValueError,match='live value origin owner'):result.admit(other)
    assert PreparedAvailableFieldProgram.from_archive(available.archive(),other).archive()==available.archive()
    assert NativeAvailableFieldResult.from_archive(result.archive(),other).archive()==result.archive()
    assert not origins[1].same_owner(cloned[1]) and origins[1].receipt()==cloned[1].receipt()


def test_independent_equal_value_origins_retain_distinct_use_lineage():
    _,_,a,row,origins,authority=prepared();_,_,b,other,other_origins,other_authority=prepared(key_offset=100)
    first=a.execute_inputs((row,),authority);second=b.execute_inputs((other,),other_authority)
    assert first.result().archive()==second.result().archive()
    assert first.archive()!=second.archive() and origins[5].receipt()!=other_origins[105].receipt()
    assert NativeAvailableFieldResult.from_archive(second.archive(),other_authority).archive()==second.archive()


def test_cold_grafted_field_rejects_before_action_preparation():
    _,_,available,row,origins,authority=prepared();result=available.execute_inputs((row,),authority)
    _,_,other,changed,other_origins,other_authority=prepared((2,3,5,11));new=other.execute_inputs((changed,),other_authority)
    forged=graft_native(result.archive(),new.result().archive())
    with pytest.raises(ValueError,match='role or complete content'):
        NativeAvailableFieldResult.from_archive(forged,authority,limits=(1,16000000,64000000,16384))
    altered=bytearray(result.archive());altered[-1]^=1
    with pytest.raises(ValueError):NativeAvailableFieldResult.from_archive(bytes(altered),authority)


def test_late_origin_and_current_context_reject_warm_and_cold(native_fixtures):
    history=NativeSource((native_fixtures[0]/'diagonal-history.rfcxs').read_bytes())
    assert history.identity['sequence']>history.identity['first_sequence']
    owner,bare,_,row,origins,_=prepared()
    origins[5]=publish(5,history,'input-value',origins[5].payload(),limits=(4096,16000000,64000000,1<<24))
    authority=context(origins);available=PreparedAvailableFieldProgram(bare,{'program':1,'maps':2,'metrics':3},authority)
    result=available.execute_inputs((row,),authority)
    withheld=context(origins,limits={5:history.identity['first_sequence']})
    with pytest.raises(ValueError,match='namespace availability'):available.execute_inputs((row,),withheld)
    with pytest.raises(ValueError,match='namespace availability'):result.admit(withheld)
    with pytest.raises(ValueError,match='namespace availability'):NativeAvailableFieldResult.from_archive(result.archive(),withheld)
    assert result.archive()==NativeAvailableFieldResult.from_archive(result.archive(),authority).archive()
    origins[1]=publish(1,history,'program',origins[1].payload(),limits=(4096,16000000,64000000,1<<24))
    late_program=context(origins,limits={1:history.identity['first_sequence']})
    with pytest.raises(ValueError,match='namespace availability'):PreparedAvailableFieldProgram(bare,{'program':1,'maps':2,'metrics':3},late_program)


def test_native_origin_context_and_prepared_owners_are_immutable():
    _,bare,available,row,origins,authority=prepared()
    with pytest.raises(TypeError):origins[1].__init__(1,bare.archive(),0,'program',bare.archive())
    with pytest.raises(TypeError):authority.__init__(())
    with pytest.raises(TypeError):available.__init__(bare,{'program':1,'maps':2,'metrics':3},authority)
    with pytest.raises((AttributeError,TypeError)):authority.hidden_authority=True
    with pytest.raises(ValueError):NativeAvailabilityContext(({'origin':origins[1],'namespace':origins[1].inspect()['namespace'],'sequence':0},)*2)
    with pytest.raises(RuntimeError):NativeAvailabilityContext.__new__(NativeAvailabilityContext).__init__(({'origin':NativeValueOrigin.__new__(NativeValueOrigin),'namespace':1,'sequence':0},))


def test_concurrent_calls_retain_exact_owners_after_publishers_escape():
    _,_,available,row,origins,authority=prepared();expected=available.execute_inputs((row,),authority).archive()
    del origins
    with ThreadPoolExecutor(max_workers=6) as pool:
        outputs=tuple(pool.map(lambda _:available.execute_inputs((row,),authority).archive(),range(6)))
    assert outputs==(expected,)*6


def test_core_context_required_and_full_result_lineage_verified():
    owner,_,available,row,origins,authority=prepared()
    state=cpp_available_field_program_model(owner,available_program=available,execution_context=authority)
    query=available_field_query(state,(row,))
    with pytest.raises(ValueError,match='separate execution context'):infer_model(state,owner,query)
    output=infer_model(state,owner,query,execution_context=authority)
    result=read_available_field_result(output,state=state,execution_context=authority)
    assert result.archive()==read_field_program_result(output,state=state,execution_context=authority).archive()
    assert len(field_program_stalks(output,2,((0,0),),execution_context=authority)[0]['values'])==2
    with pytest.raises(TypeError):read_field_program_result(output)
    with pytest.raises(ValueError):read_available_field_result(output,execution_context=context(origins,remove=(5,)))
    restored=unpack_model(unpack_value(pack_value(pack_model(state,native=True))))
    assert infer_model(restored,owner,query,execution_context=authority).field_archive['sha256']==output.field_archive['sha256']
    with pytest.raises(ValueError):infer_model(restored,owner,query,execution_context=context(origins,remove=(2,)))
    clear_available_field_cache()
    assert infer_model(restored,owner,query,execution_context=authority).field_archive['sha256']==output.field_archive['sha256']
    receipt=dict(output.execution_receipt);receipt['available_program_sha256']='f'*64
    with pytest.raises(ValueError):read_available_field_result(replace(output,execution_receipt=receipt),execution_context=authority)


@pytest.mark.parametrize('extension',('.rcbd','.safetensors'))
def test_saved_record_requires_independent_restore_authority(tmp_path,extension):
    owner,_,available,row,origins,authority=prepared()
    state=cpp_available_field_program_model(owner,available_program=available,execution_context=authority)
    record=model_record(owner,state);path=tmp_path/('available'+extension);save_model(path,record,execution_context=authority)
    with pytest.raises(ValueError,match='separate execution context'):load_model(path)
    cold=load_model(path,execution_context=authority);restored,data=read_model_record(cold,execution_context=authority)
    query=available_field_query(restored,(row,));output=infer_model(restored,data,query,execution_context=authority)
    assert read_available_field_result(output,state=restored,execution_context=authority).result().field_archive(1)==row['field']
    with pytest.raises(ValueError):read_model_record(cold,execution_context=context(origins,remove=(3,)))


@pytest.mark.parametrize('backend',('file','rex','sqlite'))
def test_rcql_checked_copies_system_and_agent_keep_external_authority(tmp_path,backend):
    from rcql import Executor,parse
    from rcdb import open_store,copy_record
    from system.serialize import json_value
    from agent.hive import Hive
    owner,_,available,row,origins,authority=prepared()
    state=cpp_available_field_program_model(owner,available_program=available,execution_context=authority)
    record=model_record(owner,state);query=available_field_query(state,(row,))
    output=infer_model(state,owner,query,execution_context=authority)
    config={'source_backed':True,'available_fields':True,'available_program':available,'execution_context':authority}
    values=Executor(sources={'r':record},params={'c':config,'q':query,'o':output,'authority':authority}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_relational_field_program",$c),'
        'MODEL_INFER(MODEL_STATE(execution_context=$authority),$q,execution_context=$authority),'
        'MODEL_STALK($o,[0,0],node=2,max_coordinates=2,execution_context=$authority),'
        'MODEL_VALUES($o,execution_context=$authority),MODEL_INFO(MODEL_STATE(execution_context=$authority),execution_context=$authority)')).values
    assert values[0].coefficient_digest==state.coefficient_digest and values[1].field_archive['sha256']==output.field_archive['sha256']
    assert len(values[2]['values'])==2 and len(values[3]['fields'])==3 and values[4]['adapter_version']=='5'
    assert json_value(state)['availability']=='independent-native-context-required' and json_value(output)['retains_whole_fields']
    stores=[open_store('sqlite:///'+str(tmp_path/(n+'.sqlite'))) if backend=='sqlite' else open_store(backend+'://'+str(tmp_path/n),read_only=False) for n in ('first','second')]
    try:
        stored=stores[0].put('available',record);copy_record(stores[0],stores[1],stored)
        restored,data=read_model_record(stores[1].get('available'),execution_context=authority)
        hive=Hive();hive.add_worker('available',lambda value:infer_model(restored,data,value,execution_context=authority),capability='analyze')
        response=hive.dispatch_capability('analyze',query)['result']
        assert read_available_field_result(response,execution_context=authority).archive()==read_available_field_result(output,execution_context=authority).archive()
    finally:
        for store in stores:store.close()
