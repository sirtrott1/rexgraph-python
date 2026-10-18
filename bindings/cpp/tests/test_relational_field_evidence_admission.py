"""Actual whole field evidence authority, native bulk updates and global choices."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q

import pytest
from rexgraph_cpp import (
    PreparedFieldProgram, PreparedAccessionFieldProgram, PreparedAvailableFieldProgram,
    PreparedAvailableFieldEvidence, NativeAvailableAffineResult, NativeAvailableAffineEvaluation,
    field_input_origin_payloads, field_evidence_origin_payload, field_target_origin_payload,
)
from test_relational_field_admission import publish, context
from test_relational_field_evidence import (
    source, learning_declaration, parameters, actual_asset, input_row, reference, tensor, matrix,
    read_matrix, HELD_INPUT,
)
from rexgraph.cpp_relational_field_evidence_admission import (
    cpp_available_field_learning_model, available_field_program_batch, available_field_learning_query,
    read_available_field_learning_result, evaluate_available_field_program,
    inspect_available_field_learning_model, clear_available_field_learning_cache,
)
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model


def prepared():
    owner = source(); program = PreparedAccessionFieldProgram(PreparedFieldProgram(owner.archive, learning_declaration()).archive())
    origins = {i: publish(i, owner, role, program.archive()) for i, role in enumerate(('program','maps','metrics'), 1)}
    origins[4] = publish(4, owner, 'evidence', field_evidence_origin_payload(program, parameters()))
    authority = context(origins)
    available = PreparedAvailableFieldProgram(program, {'program':1,'maps':2,'metrics':3}, authority)
    evidence = PreparedAvailableFieldEvidence(available, parameters(), 4, authority)
    return owner, program, evidence, origins


def independent_input(owner, program, origins, key, values, *, accession=None):
    asset = actual_asset(990000+key, key, b'independently supplied original input')
    row = input_row(asset, key, values)
    if accession is not None: row['accession'] = accession
    payloads = field_input_origin_payloads(program, row)
    names = ('asset','value','accession','reconstruction')
    roles = ('input-asset','input-value','input-accession','reconstruction')
    keys = {name: key*10+i for i, name in enumerate(names)}
    for name, role in zip(names, roles, strict=True):
        origins[keys[name]] = publish(keys[name], owner, role, payloads[name])
    row['origins'] = keys
    return row


def measured_observation(owner, program, origins, key, values, theta, *, mask=((0,0),(0,1),(0,2)), accession=None):
    row = independent_input(owner, program, origins, key, values, accession=accession)
    measured = actual_asset(980000+key, key+1000, b'complete historical measurement')
    target = tensor(reference(values, theta)[0]); keys = {'asset':key*10+4, 'value':key*10+5}
    origins[keys['asset']] = publish(keys['asset'], owner, 'target-asset', measured.archive())
    origins[keys['value']] = publish(keys['value'], owner, 'target-value',
        field_target_origin_payload(program, 2, measured.archive(), key+1000, target))
    return {'key':key,'inputs':(row,),'target_asset':measured.archive(),'target_use':key+1000,
            'target':target,'observed':mask,'origins':keys}


def learned(*, mask=((0,0),(0,1),(0,2))):
    owner, program, root, origins = prepared(); theta = tuple(map(Q,(2,-1,3,0,0)))
    observation = measured_observation(owner, program, origins, 11, tuple(map(Q,(1,2,3,4))), theta, mask=mask)
    child = root.append_inputs((observation,), context(origins))
    return owner, program, root, child, origins, theta


@pytest.mark.parametrize('values', [HELD_INPUT, (11,-2,3,1), (-3,0,2,5), (0,4,-1,2), (1,0,0,1), (0,0,3,7), (5,7,0,0), (-2,-3,-5,-7)])
def test_native_admitted_multi_map_learning_changes_complete_independent_fields(values):
    owner, program, root, child, origins, theta = learned()
    held = independent_input(owner, program, origins, 81, values); authority = context(origins)
    before = root.execute_inputs((held,), authority).result()
    family = child.execute_inputs((held,), authority); complete = family.result()
    lower, flags, later = reference(values, theta)
    for node, expected in ((1,values),(2,lower),(3,later)):
        assert complete.field_archive(node)['particular'] == tensor(expected)
        assert complete.field_archive(node)['free_parameter_count'] == 2
    assert complete.field_archive(2, occurrences=True)['particular'] == tensor(flags)
    assert before.field_archive(2)['particular'] != complete.field_archive(2)['particular']
    assert child.parent().archive() == root.archive()
    assert child.evidence().identity()['free_parameter_count'] == 2
    assert child.origin_count() == 13 and len(child.dependency_archives()) == 15
    assert NativeAvailableAffineResult.from_archive(family.archive(), authority).archive() == family.archive()
    assert PreparedAvailableFieldEvidence.from_archive(child.archive(), authority).archive() == child.archive()


def test_partial_stalks_keep_correlated_unknowns_and_one_global_setting_reused():
    owner, program, root, child, origins, theta = learned(mask=((0,0),))
    assert child.evidence().identity()['free_parameter_count'] == 3
    free = tuple(map(Q,(7,11,13)))
    origins[900] = publish(900, owner, 'prospective', child.parameter_setting_payload(free))
    families = []
    for key, values in ((81,HELD_INPUT),(82,(11,-2,3,1))):
        row = independent_input(owner, program, origins, key, values); authority = context(origins)
        family = child.execute_inputs((row,), authority); families.append(family)
        assert family.result().field_archive(2)['direction_nonzeros'] > 0
        selected = family.evaluate(free, 900, authority)
        assert selected.family().archive() == family.archive()
        assert selected.result().free_values_archive()
        assert NativeAvailableAffineEvaluation.from_archive(selected.archive(), authority).archive() == selected.archive()
    assert families[0].archive() != families[1].archive()
    maps = child.evidence().occurrence_map(1,3).inspect()
    assert any(read_matrix(image)[0][0] for image in maps['directions'])


def test_full_raw_suffix_admission_precedes_first_invalid_accession_action():
    owner, program, root, origins = prepared(); theta = tuple(map(Q,(2,-1,3,0,0)))
    from test_relational_field_evidence import input_accession
    invalid = input_accession(); invalid['accession'] = matrix(((0,0,0,0),)*4)
    first = measured_observation(owner, program, origins, 11, (1,2,3,4), theta, accession=invalid)
    last = measured_observation(owner, program, origins, 12, (4,-1,2,3), theta)
    archive = root.archive()
    with pytest.raises(ValueError, match='missing actual value origin'):
        root.append_inputs((first,last), context(origins, remove=(last['origins']['value'],)))
    with pytest.raises(ValueError): root.append_inputs((first,last), context(origins))
    assert root.archive() == archive and root.parent() is None


def test_current_root_evidence_input_target_and_setting_authority_rechecked():
    owner, program, root, child, origins, theta = learned()
    held = independent_input(owner, program, origins, 81, HELD_INPUT); authority = context(origins)
    family = child.execute_inputs((held,), authority)
    for key in (4, 115, held['origins']['value']):
        denied = context(origins, remove=(key,))
        with pytest.raises(ValueError): family.admit(denied)
        with pytest.raises(ValueError): NativeAvailableAffineResult.from_archive(family.archive(), denied)
    origins[900] = publish(900, owner, 'prospective', child.parameter_setting_payload((7,11)))
    selected = family.evaluate((7,11), 900, context(origins))
    with pytest.raises(ValueError): selected.admit(authority)
    with pytest.raises(ValueError): NativeAvailableAffineEvaluation.from_archive(selected.archive(), authority)
    with pytest.raises(ValueError): family.evaluate((7,), 900, context(origins))


def test_role_copy_and_foreign_live_owners_reject_but_declared_cold_adoption_works():
    owner, program, root, child, origins, theta = learned()
    held = independent_input(owner, program, origins, 81, HELD_INPUT)
    key = held['origins']['value']; original = origins[key]
    origins[key] = publish(key, owner, 'target-value', original.payload())
    with pytest.raises(ValueError, match='role or complete content'): child.execute_inputs((held,), context(origins))
    origins[key] = original
    from rexgraph_cpp import NativeSource
    cloned = {key:publish(key, NativeSource(value.source_archive()), value.inspect()['role'], value.payload()) for key,value in origins.items()}
    other = context(cloned)
    with pytest.raises(ValueError, match='live value origin owner'): child.admit(other)
    assert PreparedAvailableFieldEvidence.from_archive(child.archive(), other).archive() == child.archive()


def test_owning_available_family_lifetime_concurrency_and_empty_handles():
    owner, program, root, child, origins, theta = learned()
    held = independent_input(owner, program, origins, 81, HELD_INPUT); authority = context(origins)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(child.execute_inputs,[(held,)]*6,[authority]*6))
    del root, child, held, program, origins, owner
    assert len({result.archive() for result in results}) == 1
    for cls in (PreparedAvailableFieldEvidence,NativeAvailableAffineResult,NativeAvailableAffineEvaluation):
        with pytest.raises(RuntimeError): cls.__new__(cls).archive()
    with pytest.raises(TypeError): results[0].evidence().__init__(None,{},1,authority)


@pytest.mark.parametrize('kind',['root','input','target','setting'])
def test_actual_value_availability_separate_from_early_asset_and_retrospective_target(native_fixtures,kind):
    from rexgraph_cpp import NativeSource
    history = NativeSource((native_fixtures[0]/'diagonal-history.rfcxs').read_bytes())
    owner,program,root,origins = prepared(); theta = tuple(map(Q,(2,-1,3,0,0)))
    if kind=='root':
        origins[4] = publish(4,history,'evidence',origins[4].payload(),limits=(4096,16000000,64000000,1<<24))
        with pytest.raises(ValueError,match='namespace availability'):
            PreparedAvailableFieldEvidence(root.program(),parameters(),4,context(origins,limits={4:history.identity['first_sequence']}))
        return
    observed = measured_observation(owner,program,origins,11,(1,2,3,4),theta)
    key = 111 if kind=='input' else 115
    if kind in {'input','target'}:
        origins[key] = publish(key,history,'input-value' if kind=='input' else 'target-value',origins[key].payload(),limits=(4096,16000000,64000000,1<<24))
        denied = context(origins,limits={key:history.identity['first_sequence']})
        with pytest.raises(ValueError,match='namespace availability'):root.append_inputs((observed,),denied)
        child = root.append_inputs((observed,),context(origins))
        with pytest.raises(ValueError,match='namespace availability'):PreparedAvailableFieldEvidence.from_archive(child.archive(),denied)
        assert child.evidence().identity()['observation_count'] == 1
        return
    child = root.append_inputs((observed,),context(origins));held = independent_input(owner,program,origins,81,HELD_INPUT)
    origins[900] = publish(900,history,'prospective',child.parameter_setting_payload((7,11)),limits=(4096,16000000,64000000,1<<24))
    family = child.execute_inputs((held,),context(origins));denied = context(origins,limits={900:history.identity['first_sequence']})
    with pytest.raises(ValueError,match='namespace availability'):family.evaluate((7,11),900,denied)
    selected = family.evaluate((7,11),900,context(origins))
    with pytest.raises(ValueError,match='namespace availability'):NativeAvailableAffineEvaluation.from_archive(selected.archive(),denied)


def core_learned(*, partial=False):
    owner, program, root, origins = prepared(); theta = tuple(map(Q,(2,-1,3,0,0)))
    state = cpp_available_field_learning_model(owner,available_evidence=root,execution_context=context(origins))
    observed = measured_observation(owner,program,origins,11,(1,2,3,4),theta,
        mask=((0,0),) if partial else ((0,0),(0,1),(0,2)))
    authority = context(origins)
    batch = available_field_program_batch(state,owner,(observed,),execution_context=authority)
    child = train_model(state,owner,batch,execution_context=authority)
    return owner,program,state,child,origins,theta,batch


def test_core_train_infer_cold_model_output_and_parent_lineage_require_current_authority():
    owner,program,parent,child,origins,theta,batch = core_learned()
    assert child.parent == parent.coefficient_digest and child.step == 1
    assert child.source.matches(parent.source) and child.space == parent.space and child.output_axes == parent.output_axes
    with pytest.raises(ValueError,match='separate execution context'): train_model(parent,owner,batch)
    with pytest.raises(ValueError): train_model(parent,owner,batch,execution_context=context(origins,remove=(115,)))
    held = independent_input(owner,program,origins,81,HELD_INPUT); authority = context(origins)
    query = available_field_learning_query(child,(held,))
    with pytest.raises(ValueError,match='separate execution context'):infer_model(child,owner,query)
    output = infer_model(child,owner,query,execution_context=authority)
    result = read_available_field_learning_result(output,state=child,execution_context=authority)
    assert result.result().field_archive(2)['particular'] == tensor(reference(HELD_INPUT,theta)[0])
    restored = unpack_model(unpack_value(pack_value(pack_model(child,native=True))))
    clear_available_field_learning_cache()
    assert infer_model(restored,owner,query,execution_context=authority).field_archive['data'] == output.field_archive['data']
    with pytest.raises(ValueError): read_available_field_learning_result(output,execution_context=context(origins,remove=(held['origins']['value'],)))


def test_core_admission_declaration_and_false_parent_dependency_rejection():
    from dataclasses import replace
    owner,program,root,origins = prepared()
    with pytest.raises(ValueError,match='admission differs'):
        cpp_available_field_learning_model(owner,available_evidence=root,execution_context=context(origins),limits=(1,16000000,64000000,16384))
    owner,program,parent,child,origins,theta,batch = core_learned()
    clear_available_field_learning_cache()
    with pytest.raises(ValueError):inspect_available_field_learning_model(replace(child,parent='0'*64),owner,execution_context=context(origins))
    with pytest.raises(ValueError):inspect_available_field_learning_model(replace(child,dependencies=child.dependencies[:-1]),owner,execution_context=context(origins))


@pytest.mark.parametrize('extension',['.rcbd','.safetensors'])
def test_core_record_cold_replay_and_prospective_family_are_separate(extension,tmp_path):
    owner,program,parent,child,origins,theta,batch = core_learned(partial=True)
    free = (7,11,13)
    native = inspect_available_field_learning_model(child,owner,execution_context=context(origins))
    origins[900] = publish(900,owner,'prospective',native.parameter_setting_payload(free))
    held = independent_input(owner,program,origins,81,HELD_INPUT); authority = context(origins)
    query = available_field_learning_query(child,(held,)); output = infer_model(child,owner,query,execution_context=authority)
    selected = evaluate_available_field_program(output,free,900,state=child,execution_context=authority)
    complete = read_available_field_learning_result(selected,state=child,execution_context=authority)
    assert complete.family().archive() == read_available_field_learning_result(output,execution_context=authority).archive()
    path = tmp_path/('available-evidence'+extension);save_model(path,model_record(owner,child),execution_context=authority)
    with pytest.raises(ValueError,match='separate execution context'):load_model(path)
    record = load_model(path,execution_context=authority); restored,data = read_model_record(record,execution_context=authority)
    assert infer_model(restored,data,query,execution_context=authority).field_archive['data'] == output.field_archive['data']


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_rcql_system_checked_store_copy_and_agent_worker_routes(backend,tmp_path):
    from rcdb import open_store, copy_record
    from rcql.executor import Executor
    from rcql.parser import parse
    from system.serialize import json_value
    from agent.hive import Hive
    owner,program,parent,child,origins,theta,batch = core_learned()
    held = independent_input(owner,program,origins,81,HELD_INPUT); authority = context(origins)
    request = available_field_learning_query(child,(held,)); output = infer_model(child,owner,request,execution_context=authority)
    native_parent = inspect_available_field_learning_model(parent,owner,execution_context=authority)
    configuration = {'source_backed':True,'available_fields':True,'learning':True,'available_evidence':native_parent,'execution_context':authority}
    record = model_record(owner,child)
    values = Executor(sources={'r':record},params={'cfg':configuration,'batch':batch,'q':request,'authority':authority,'o':output}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_relational_field_program",$cfg),'
        'MODEL_INFER(MODEL_STATE(execution_context=$authority),$q,execution_context=$authority),'
        'MODEL_VALUES($o,execution_context=$authority),'
        'MODEL_STALK($o,[0,0],node=2,max_coordinates=2,execution_context=$authority),'
        'MODEL_INFO(MODEL_STATE(execution_context=$authority),execution_context=$authority)')).values
    assert values[0].coefficient_digest == parent.coefficient_digest
    results = values[1:]
    trained = Executor(sources={'r':model_record(owner,parent)},params={'authority':authority,'batch':batch}).execute(parse(
        'FROM $r RETURN MODEL_TRAIN(MODEL_STATE(execution_context=$authority),$batch,execution_context=$authority)')).values[0]
    assert trained.coefficient_digest == child.coefficient_digest
    assert results[0].field_archive['data'] == output.field_archive['data']
    assert json_value(child)['trainable'] and json_value(child)['availability'] == 'independent-native-context-required'
    assert json_value(output)['retains_whole_fields']
    stores = [open_store('sqlite:///'+str(tmp_path/(name+'.sqlite'))) if backend=='sqlite' else
              open_store(backend+'://'+str(tmp_path/name),read_only=False) for name in ('first','second')]
    try:
        stored = stores[0].put('available',record);copy_record(stores[0],stores[1],stored)
        restored,data = read_model_record(stores[1].get('available'),execution_context=authority)
        hive = Hive();hive.add_worker('available-evidence',lambda row:infer_model(restored,data,row,execution_context=authority),capability='analyze')
        response = hive.dispatch_capability('analyze',request)['result']
        assert response.field_archive['data'] == output.field_archive['data']
    finally:
        for store in stores:store.close()
