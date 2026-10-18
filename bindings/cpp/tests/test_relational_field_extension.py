"""Faithful native extensions, whole completion families and application lineage."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph_cpp import (PreparedFieldProgram, PreparedAccessionFieldProgram, PreparedAvailableFieldProgram,
    PreparedAvailableFieldExtension, NativeAvailableExtensionFamily, NativeAvailableExtensionEvaluation,
    NativeInputUseAsset, NativeInputAccession, field_extension_origin_payload, field_input_origin_payloads)
from rexgraph.cpp_relational_field_admission import cpp_available_field_program_model
from rexgraph.cpp_relational_field_extension import (cpp_available_field_extension_model, field_extension_query,
    read_field_extension_result, evaluate_field_extension, inspect_field_extension_model)
from rexgraph.cpp_relational_field_program import read_field_program_result, field_program_stalks
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import ModelBatch
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.io.model_record import model_record, read_model_record, save_model, load_model
from test_relational_field_admission import prepared, context, publish, graft_native
from test_relational_field_evidence import learning_declaration, matrix, read_matrix, HELD_INPUT
from test_relational_field_program import LOWER, UPPER, COEFFICIENTS, action
from test_relational_field_local import values


def larger_declaration(old=None):
    old=learning_declaration() if old is None else deepcopy(old)
    target=deepcopy(old);target['revision']+=1;stalks=[];maps=[];offsets={0:[],1:[]};original={0:[],1:[]};sizes={}
    for index,s in enumerate(old['stalks']):
        n=len(s['directions']);grade=s['cell'][0];offsets[grade].append(sum(sizes[c]+1 for c in sizes if c[0]==grade))
        original[grade].append(sum(sizes[c] for c in sizes if c[0]==grade));sizes[s['cell']]=n
        metric=read_matrix_bytes_declaration(s['metric']);basis=12000+100*target['revision']+index
        stalks.append({'cell':s['cell'],'basis':basis,'directions':tuple(30000+100*index+i+10000*target['revision'] for i in range(n+1)),
                       'metric':matrix(tuple(tuple(metric[i][j] if i<n and j<n else Q(i==j) for j in range(n+1)) for i in range(n+1)))})
        maps.append({'cell':s['cell'],'source_basis':s['basis'],'target_basis':basis,
                     'inclusion':matrix(tuple(tuple(Q(i==j) for j in range(n)) for i in range(n+1))),
                     'reconstruction':matrix(tuple(tuple(Q(i==j) for j in range(n+1)) for i in range(n)))})
    target['stalks']=tuple(stalks);flags=[]
    for slot,f in enumerate(old['occurrences']):
        m=read_matrix_bytes_declaration(f['map']);nr,nc=len(m),len(m[0]);flag=dict(f)
        flag['map']=matrix(tuple(tuple(m[i][j] if i<nr and j<nc else Q(slot+1,3) if i==0 and j==nc else Q(0)
                                     for j in range(nc+1)) for i in range(nr+1)));flags.append(flag)
    target['occurrences']=tuple(flags)
    n0=sum(n+1 for c,n in sizes.items() if c[0]==0);n1=sum(n+1 for c,n in sizes.items() if c[0]==1)
    lifted=[[Q(0)]*n0 for _ in range(n1)];old_map=read_matrix_bytes_declaration(old['nodes'][2]['map'])
    for c,n in sizes.items():
        if c[0]!=1:continue
        for low,m in sizes.items():
            if low[0]!=0:continue
            for i in range(n):
                for j in range(m):lifted[offsets[1][c[1]]+i][offsets[0][low[1]]+j]=old_map[original[1][c[1]]+i][original[0][low[1]]+j]
    nodes=list(target['nodes']);nodes[2]=dict(nodes[2],map=matrix(lifted));target['nodes']=tuple(nodes)
    d={'key':9001,'revision':1,'free_carrier':9002,'free_basis':9003,'free_directions':(9004,9005),
       'source_map':{'cells':((0,1,2),(0,1)),'occurrences':((),tuple(range(8)))},'stalks':tuple(maps),'isometric':True}
    return target,d,offsets,original,sizes


def read_matrix_bytes_declaration(value):
    rows,cols=value['shape'];out=[[Q(0)]*cols for _ in range(rows)]
    for i,j,q in value['entries']:out[i][j]=Q(q)
    return tuple(tuple(row) for row in out)


def fixture(v=HELD_INPUT, *, status='prospective'):
    owner,program,parent,row,origins,authority=prepared(v)
    target,d,*_=larger_declaration();next_program=PreparedAccessionFieldProgram(PreparedFieldProgram(owner.archive,target).archive())
    for key,role in zip((901,902,903),('program','maps','metrics'),strict=True):origins[key]=publish(key,owner,role,next_program.archive())
    next_available=PreparedAvailableFieldProgram(next_program,{'program':901,'maps':902,'metrics':903},context(origins))
    origins[904]=publish(904,owner,'evidence' if status=='observed' else 'prospective',field_extension_origin_payload(parent,next_available,d,status))
    authority=context(origins);extension=PreparedAvailableFieldExtension(parent,next_available,d,status,904,authority)
    return owner,parent,extension,row,origins,authority


def complete(extension,row,owner,origins,free=(Q(2),Q(-3)),key=905):
    family=extension.execute_inputs((row,),context(origins));origins[key]=publish(key,owner,'prospective',family.completion_payload(free))
    return family,family.evaluate(free,key,context(origins)),context(origins)


@pytest.mark.parametrize('v',[HELD_INPUT,(11,-2,3,1),(-3,0,2,5),(0,0,0,0)])
def test_whole_correlated_family_and_every_flag_agree_with_independent_action(v):
    owner,parent,extension,row,origins,authority=fixture(v);family,evaluation,authority=complete(extension,row,owner,origins)
    assert family.input().field_archive(1)==row['field']
    x=(Q(v[0]),Q(v[1]),Q(2),Q(v[2]),Q(v[3]),Q(-3));assert values(evaluation.field_archive(1))==x
    target,_,offsets,_,sizes=larger_declaration();expected=[[Q(0)]*(sizes[(0,i)]+1) for i in range(3)];flags=[]
    for slot,flag in enumerate(target['occurrences']):
        upper=x[3*UPPER[slot]:3*UPPER[slot]+3];coefficient=Q(-1) if slot==7 else COEFFICIENTS[slot]
        q=tuple(coefficient*a for a in action(read_matrix_bytes_declaration(flag['map']),upper));flags.extend(q)
        for i,a in enumerate(q):expected[LOWER[slot]][i]+=a
    whole=tuple(q for stalk in expected for q in stalk)
    assert values(evaluation.field_archive(2))==whole and values(evaluation.field_archive(2,occurrences=True))==tuple(flags)
    assert values(evaluation.field_archive(3))==action(read_matrix_bytes_declaration(target['nodes'][2]['map']),whole)
    for node in (1,2,3):
        f=family.field_archive(node);assert f['free_parameter_count']==2
        assert tuple(a+b for a,b in zip(values(f['particular']),action(read_matrix(f['directions']),(Q(2),Q(-3))),strict=True))==values(evaluation.field_archive(node))
    maps=extension.info()['grade_maps'];assert len(read_matrix(maps[1]['kernel'])[0])==2
    assert NativeAvailableExtensionFamily.from_archive(family.archive(),authority).archive()==family.archive()
    assert NativeAvailableExtensionEvaluation.from_archive(evaluation.archive(),authority).archive()==evaluation.archive()
    selected=family.select(1,((1,0),));assert selected[0]['identified']==(True,True,False)
    assert selected[0]['particular']['directions']==target['stalks'][3]['directions']
    with pytest.raises((ValueError,RuntimeError)):family.evaluate((Q(2),),905,authority)
    with pytest.raises((ValueError,OverflowError,RuntimeError)):family.select(1,((1,0),),coordinate_limit=2)


@pytest.mark.parametrize('key',(1,2,3,4,5,6,7,901,902,903,904,905))
def test_current_complete_authority_required_for_live_and_cold_completion(key):
    owner,parent,extension,row,origins,authority=fixture();family,evaluation,authority=complete(extension,row,owner,origins)
    missing=context(origins,remove=(key,))
    with pytest.raises(ValueError):evaluation.admit(missing)
    with pytest.raises(ValueError):NativeAvailableExtensionEvaluation.from_archive(evaluation.archive(),missing)
    if key!=905:
        with pytest.raises(ValueError):extension.execute_inputs((row,),missing)


def test_equal_numeric_input_keeps_original_owner_and_completion_is_input_specific():
    owner,parent,extension,row,origins,authority=fixture();family,evaluation,authority=complete(extension,row,owner,origins)
    from test_relational_field_evidence import actual_asset,input_row
    other=input_row(actual_asset(999992,222,b'equal coefficients distinct original bytes'),222,HELD_INPUT)
    payloads=field_input_origin_payloads(extension.source(),other);other['origins']={}
    for i,(name,role) in enumerate(zip(('asset','value','accession','reconstruction'),('input-asset','input-value','input-accession','reconstruction'),strict=True),910):
        other['origins'][name]=i;origins[i]=publish(i,owner,role,payloads[name])
    second=extension.execute_inputs((other,),context(origins))
    assert second.field_archive(2)==family.field_archive(2) and second.archive()!=family.archive()
    with pytest.raises(ValueError):second.evaluate((Q(2),Q(-3)),905,context(origins))
    with pytest.raises(ValueError):NativeAvailableExtensionEvaluation.from_archive(graft_native(evaluation.archive(),b'foreign'),authority)
    asset=NativeInputUseAsset.from_archive(row['asset']);accession=NativeInputAccession(asset,parent.program(),row['accession'])
    transported=extension.transport_accession(accession,920,authority)
    assert transported.admit(row['use'],row['field']).inspect()['field']==row['field']
    assert read_matrix(transported.admit(row['use'],row['field']).inspect()['accession'])[0]==(Q(1),Q(0),Q(0),Q(0))
    foreign=NativeInputAccession(asset,PreparedAccessionFieldProgram(parent.program().archive()),row['accession'])
    with pytest.raises(ValueError):extension.transport_accession(foreign,921,authority)


@pytest.mark.parametrize('suffix',('.rcbd','.safetensors'))
def test_core_versioned_parent_source_record_and_explicit_completion(tmp_path,suffix):
    owner,parent,extension,row,origins,authority=fixture()
    before=cpp_available_field_program_model(owner,available_program=parent,execution_context=authority)
    state=cpp_available_field_extension_model(owner,parent=before,available_extension=extension,execution_context=authority)
    assert state.adapter_version=='7' and state.parent==before.coefficient_digest and state.step==before.step+1
    query=field_extension_query(state,(row,))
    with pytest.raises(ValueError,match='separate execution context'):infer_model(state,owner,query)
    output=infer_model(state,owner,query,execution_context=authority);family=read_field_program_result(output,state=state,execution_context=authority)
    origins[905]=publish(905,owner,'prospective',family.completion_payload((2,-3)));authority=context(origins)
    completed=evaluate_field_extension(output,(2,-3),905,execution_context=authority,state=state)
    assert field_program_stalks(completed,1,((1,0),),execution_context=authority)[0]['values']==(Q(2),Q(3),Q(2))
    assert field_program_stalks(output,1,((1,0),),execution_context=authority)[0]['identified']==(True,True,False)
    record=model_record(owner,state);path=tmp_path/('extension'+suffix);save_model(path,record,execution_context=authority)
    restored,data=read_model_record(load_model(path,execution_context=authority),execution_context=authority)
    assert restored.coefficient_digest==state.coefficient_digest
    assert infer_model(restored,data,query,execution_context=authority).field_archive['data']==output.field_archive['data']
    assert unpack_model(unpack_value(pack_value(pack_model(completed,native=True)))).field_archive['data']==completed.field_archive['data']
    with pytest.raises(NotImplementedError):train_model(state,owner,ModelBatch(state.source,state.space,np.zeros(len(state.space.keys)),np.ones(len(state.space.keys),dtype=bool)),execution_context=authority)
    altered=dict(completed.execution_receipt);altered['extension_sha256']='f'*64
    with pytest.raises(ValueError):read_field_extension_result(replace(completed,execution_receipt=altered),execution_context=authority)
    with pytest.raises(ValueError):read_field_extension_result(completed,execution_context=context(origins,remove=(905,)))
    from rcql.model_operators import model_values,model_stalk,model_info,model_complete
    assert model_values(record,output,authority)['relationship']['key']==9001
    assert model_stalk(record,completed,(1,0),node=1,execution_context=authority)['values']==(Q(2),Q(3),Q(2))
    assert model_info(record,state,authority)['trainable'] is False
    assert model_complete(record,output,(2,-3),905,state,authority).field_archive['data']==completed.field_archive['data']
    from system.serialize import json_value
    assert json_value(state)['relationship_status']=='prospective'
    from agent.hive import Hive
    hive=Hive();hive.add_worker('extension',lambda inputs:infer_model(state,owner,inputs,execution_context=authority),capability='analyze')
    assert hive.dispatch_capability('analyze',query)['result'].field_archive['data']==output.field_archive['data']


def test_selected_learned_parent_retains_evidence_choice_and_independent_child_grants():
    from test_relational_field_evidence_admission import core_learned,independent_input
    from rexgraph.cpp_relational_field_evidence_admission import inspect_available_field_learning_model
    owner,original,root,state,origins,theta,batch=core_learned()
    evidence=inspect_available_field_learning_model(state,owner,execution_context=context(origins))
    held=independent_input(owner,original,origins,81,HELD_INPUT);family=evidence.execute_inputs((held,),context(origins));free=(Q(3),Q(-2))
    origins[800]=publish(800,owner,'prospective',evidence.parameter_setting_payload(free));selected=family.evaluate(free,800,context(origins))
    old=learning_declaration();old['revision']=selected.result().program().identity()['revision']
    for slot,flag in enumerate(old['occurrences']):
        info=evidence.evidence().occurrence_map(1,slot).inspect();base=read_matrix(info['particular']);directions=tuple(read_matrix(x) for x in info['directions'])
        flag['map']=matrix(tuple(tuple(base[i][j]+sum((q*d[i][j] for q,d in zip(free,directions,strict=True)),Q(0)) for j in range(len(base[0]))) for i in range(len(base))))
    target,d,*_=larger_declaration(old);program=PreparedAccessionFieldProgram(PreparedFieldProgram(owner.archive,target).archive())
    for key,role in zip((901,902,903),('program','maps','metrics'),strict=True):origins[key]=publish(key,owner,role,program.archive())
    admitted=PreparedAvailableFieldProgram(program,{'program':901,'maps':902,'metrics':903},context(origins))
    origins[904]=publish(904,owner,'evidence',field_extension_origin_payload(selected,admitted,d,'observed'))
    extension=PreparedAvailableFieldExtension(selected,admitted,d,'observed',904,context(origins))
    child=cpp_available_field_extension_model(owner,parent=state,available_extension=extension,execution_context=context(origins))
    new_input=independent_input(owner,extension.source(),origins,82,(11,-2,3,1))
    response=infer_model(child,owner,field_extension_query(child,(new_input,)),execution_context=context(origins))
    native=read_field_extension_result(response,state=child,execution_context=context(origins))
    assert native.model().root().archive()==selected.archive()
    assert child.parent==state.coefficient_digest and child.step==state.step+1 and child.configuration['status']=='observed'
    assert native.field_archive(2)['particular']!=extension.execute_inputs((new_input,),context(origins)).input().field_archive(2)
    for key in (4,114,115,800,901,904):
        with pytest.raises(ValueError):inspect_field_extension_model(child,owner,execution_context=context(origins,remove=(key,)))


def test_completed_extension_is_actual_third_local_rhs_and_cancel_joins():
    from rexgraph_cpp import local_field_origin_payloads,submit_local_field_solve,NativeLocalFieldResult
    from rexgraph.cpp_relational_field_local import solve_available_field_output,read_local_field_result
    owner,parent,extension,row,origins,authority=fixture();family,evaluation,authority=complete(extension,row,owner,origins)
    d={'key':9501,'revision':1,'coordinate_carrier':9502,'coordinate_basis':9503,'grade':1,'kind':'green','lambda':Q(1,3),'harmonic_frame':None,'interior':(0,2,4)}
    identity=matrix(tuple(tuple(Q(i==j) for j in range(6)) for i in range(6)))
    r={'patch':((0,1,2),(0,1)),'view':{'key':9504,'basis':9505,'directions':tuple(range(9506,9512)),
       'accession':identity,'reconstruction':identity,'physical':identity}}
    payloads=local_field_origin_payloads(evaluation,1,d,r);keys={'problem':9601,'interface':9602,'reconstruction':9603,'gauge':0}
    for name,role in (('problem','maps'),('interface','interface'),('reconstruction','reconstruction')):origins[keys[name]]=publish(keys[name],owner,role,payloads[name])
    authority=context(origins);run=submit_local_field_solve(evaluation,1,d,r,keys,authority);result=run.wait(authority)
    assert isinstance(result.rhs(),NativeAvailableExtensionEvaluation) and result.rhs().archive()==evaluation.archive()
    assert NativeLocalFieldResult.from_archive(result.archive(),authority).archive()==result.archive() and run.receipt()['joined']
    cancelled=submit_local_field_solve(evaluation,1,d,r,keys,authority,pause_at='preparation')
    import time
    deadline=time.monotonic()+10
    while cancelled.phase()!='preparation':assert time.monotonic()<deadline;time.sleep(.001)
    cancelled.cancel();assert cancelled.receipt()['joined'] and cancelled.receipt()['status']=='cancelled'
    before=cpp_available_field_program_model(owner,available_program=parent,execution_context=authority)
    state=cpp_available_field_extension_model(owner,parent=before,available_extension=extension,execution_context=authority)
    output=infer_model(state,owner,field_extension_query(state,(row,)),execution_context=authority)
    completed=evaluate_field_extension(output,(2,-3),905,execution_context=authority,state=state)
    packet=solve_available_field_output(completed,1,d,r,keys,execution_context=authority,state=state)
    assert read_local_field_result(packet,state=state,execution_context=authority).archive()==result.archive()
    with pytest.raises(ValueError):read_local_field_result(packet,execution_context=context(origins,remove=(904,)))


@pytest.mark.parametrize('backend',('sqlite','file'))
def test_checked_cross_store_copy_retains_complete_structural_parent_and_outputs(tmp_path,backend):
    from rcdb import open_store,copy_record
    owner,parent,extension,row,origins,authority=fixture()
    before=cpp_available_field_program_model(owner,available_program=parent,execution_context=authority)
    state=cpp_available_field_extension_model(owner,parent=before,available_extension=extension,execution_context=authority)
    output=infer_model(state,owner,field_extension_query(state,(row,)),execution_context=authority)
    record=model_record(owner,state);record.attach_metadata(1,0,'extension-response',output)
    stores=[open_store('sqlite:///'+str(tmp_path/(name+'.sqlite'))) if backend=='sqlite' else
            open_store(backend+'://'+str(tmp_path/name),read_only=False) for name in ('first','second')]
    try:
        saved=stores[0].put('extension',record);copy_record(stores[0],stores[1],saved)
        copied=stores[1].get('extension');restored,data=read_model_record(copied,execution_context=authority)
        assert restored.parent==before.coefficient_digest
        response=copied.get_metadata(1,0,'extension-response')
        assert read_field_extension_result(response,state=restored,execution_context=authority).archive()==output.field_archive['data']
        with pytest.raises(ValueError):read_model_record(copied,execution_context=context(origins,remove=(904,)))
    finally:
        for store in stores:store.close()


def test_model_extension_rejects_foreign_parent_checkpoint_and_immutable_native_reinitialization():
    owner,parent,extension,row,origins,authority=fixture()
    before=cpp_available_field_program_model(owner,available_program=parent,execution_context=authority)
    with pytest.raises(TypeError):extension.__init__(parent,extension.target(),{},'prospective',904,authority)
    with pytest.raises((AttributeError,TypeError)):extension.hidden_parent=before
    for cls in (PreparedAvailableFieldExtension,NativeAvailableExtensionFamily,NativeAvailableExtensionEvaluation):
        with pytest.raises(RuntimeError):cls.__new__(cls).archive()
    different=dict(before.configuration);different['available_sha256']='f'*64
    with pytest.raises(ValueError):replace(before,configuration=different)


def test_typed_extension_and_completion_routes_preserve_explicit_destination_binding():
    from rcql.binding import bind
    from rcql.capabilities import SourcePolicy
    from rcql.executor import Executor
    from rcql.parser import parse
    owner,parent,extension,row,origins,authority=fixture()
    before=cpp_available_field_program_model(owner,available_program=parent,execution_context=authority)
    target=cpp_available_field_program_model(owner,available_program=extension.target(),execution_context=authority)
    destination=bind('target',model_record(owner,target),SourcePolicy.allow('*'))
    request='FROM $r RETURN MODEL_EXTEND($parent,$destination,$extension,execution_context=$authority)'
    state=Executor(sources={'r':model_record(owner,before)},params={'parent':before,'destination':destination,'extension':extension,'authority':authority}).execute(parse(request)).values[0]
    assert state.parent==before.coefficient_digest and state.adapter_version=='7'
    output=infer_model(state,owner,field_extension_query(state,(row,)),execution_context=authority)
    family=read_field_extension_result(output,execution_context=authority)
    origins[905]=publish(905,owner,'prospective',family.completion_payload((2,-3)));authority=context(origins)
    request='FROM $r RETURN MODEL_COMPLETE($output,[2,-3],905,$state,execution_context=$authority),MODEL_VALUES($output,execution_context=$authority)'
    result=Executor(sources={'r':model_record(owner,state)},params={'output':output,'state':state,'authority':authority}).execute(parse(request)).values
    assert read_field_extension_result(result[0],execution_context=authority).family().archive()==family.archive()
    assert result[1]['relationship']['free_directions']==(9004,9005)


def test_new_relations_and_higher_grade_require_explicit_target_source_child():
    from rexgraph_cpp import NativeSource
    owner,parent,_,row,origins,authority=fixture()
    # A different source retains every original occurrence, then adds a relation
    # and a cancelling higher attaching history without identifying grade/scale.
    original_flags=((10,1,True),(20,-1,True),(10,-1,True),(20,1,False),(30,0,True),
                    (10,1,True),(20,Q(1,2),True),(30,-1,True))
    relations=[]
    for key,start,end in ((100,0,5),(101,5,8)):
        relations.append({'grade':1,'key':key,'type':0,'occurrences':tuple({'key':i+1,'lower':low,'negative':q<0,'magnitude':abs(q),'exists':active,'aspects':(0,)} for i,(low,q,active) in enumerate(original_flags[start:end],start))})
    relations.append({'grade':1,'key':102,'type':0,'occurrences':tuple({'key':9+i,'lower':low,'negative':q<0,'magnitude':abs(q),'exists':active,'aspects':(0,)} for i,(low,q,active) in enumerate(((10,1,True),(20,-1,True),(40,Q(1,2),True),(40,-1,False))))})
    relations.append({'grade':2,'key':200,'type':0,'occurrences':tuple({'key':20+i,'lower':low,'negative':negative,'magnitude':q,'exists':active,'aspects':(0,)} for i,(low,negative,q,active) in enumerate(((100,False,1,True),(100,True,1,True),(101,False,1,False),(101,False,0,True))))})
    target_source=NativeSource.from_declaration({'profile':'declared-role-source-v1','namespace':991002,'semantic_scale':3,'max_grade':2,'role_aspects':('boundary',),
        'types':({'name':'relation','min_arity':0,'max_arity':8},),'witnesses':(10,20,30,40),'relations':tuple(relations),'fields':(),'metrics':(),
        'carriers':({'key':10,'kind':'source_grade','grade':0,'coordinates':0,'size':4},{'key':11,'kind':'source_grade','grade':1,'coordinates':0,'size':3},{'key':12,'kind':'source_grade','grade':2,'coordinates':0,'size':1})})
    target,d,*_=larger_declaration();stalks=list(target['stalks'])
    for cell,basis in (((0,3),9700),((1,2),9701),((2,0),9702)):stalks.append({'cell':cell,'basis':basis,'directions':(basis+10,basis+20),'metric':matrix(((1,0),(0,1)))})
    target['stalks']=tuple(sorted(stalks,key=lambda x:x['cell']));flags=list(target['occurrences'])
    for slot,n in ((8,3),(9,4),(10,2),(11,2)):flags.append({'grade':1,'slot':slot,'role':'coefficient_independent','map':matrix(tuple((Q(i+1),Q(i-1)) for i in range(n)))})
    for slot in range(4):flags.append({'grade':2,'slot':slot,'role':'coefficient_independent','map':matrix(((1,0),(0,1),(0,0)))})
    target['occurrences']=tuple(flags);nodes=list(target['nodes']);old_map=read_matrix_bytes_declaration(nodes[2]['map'])
    nodes[2]=dict(nodes[2],map=matrix(tuple(tuple(old_map[i][j] if i<6 and j<10 else Q(0) for j in range(12)) for i in range(8))));target['nodes']=tuple(nodes)
    d['free_directions']=(9004,9005,9006,9007)
    program=PreparedAccessionFieldProgram(PreparedFieldProgram(target_source.archive,target).archive())
    for key,role in zip((901,902,903),('program','maps','metrics'),strict=True):origins[key]=publish(key,target_source,role,program.archive())
    admitted=PreparedAvailableFieldProgram(program,{'program':901,'maps':902,'metrics':903},context(origins))
    origins[904]=publish(904,target_source,'prospective',field_extension_origin_payload(parent,admitted,d,'prospective'))
    authority=context(origins);extension=PreparedAvailableFieldExtension(parent,admitted,d,'prospective',904,authority)
    before=cpp_available_field_program_model(owner,available_program=parent,execution_context=authority)
    state=cpp_available_field_extension_model(target_source,parent=before,available_extension=extension,execution_context=authority)
    assert not state.source.matches(before.source) and len(state.space.keys)==8 and state.parent==before.coefficient_digest
    output=infer_model(state,target_source,field_extension_query(state,(row,)),execution_context=authority)
    family=read_field_extension_result(output,state=state,execution_context=authority)
    assert len(family.model().info()['grade_maps'])==3 and family.field_archive(1)['free_parameter_count']==4
    assert family.model().info()['source_map']['occurrences'][1]==tuple(range(8))
    with pytest.raises(ValueError):infer_model(state,owner,field_extension_query(state,(row,)),execution_context=authority)
    with pytest.raises(ValueError):train_model(before,target_source,ModelBatch(before.source,before.space,np.zeros(len(before.space.keys)),np.ones(len(before.space.keys),dtype=bool)),execution_context=authority)


@pytest.mark.parametrize('changed',('key','revision','program_sha256'))
def test_matching_parent_archive_cannot_hide_forged_program_metadata(changed):
    from rexgraph.cpp_relational_field_program import _space
    owner,parent,extension,row,origins,authority=fixture()
    before=cpp_available_field_program_model(owner,available_program=parent,execution_context=authority)
    c=dict(before.configuration);c[changed]='f'*64 if changed=='program_sha256' else c[changed]+1
    forged=replace(before,configuration=c,space=_space(before.source,c['key']))
    with pytest.raises(ValueError,match='parent program metadata'):
        cpp_available_field_extension_model(owner,parent=forged,available_extension=extension,execution_context=authority)
