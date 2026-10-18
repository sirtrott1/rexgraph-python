"""Complete exact local responses, original authority and joined native runs."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
import time

import pytest
from rexgraph_cpp import (
    NativeLocalFieldResult, NativeLocalFieldWork, submit_local_field_solve, local_field_origin_payloads,
    NativeAvailableAffineEvaluation,
)
from test_relational_field_admission import prepared, context, publish, graft_native
from test_relational_field_evidence_admission import learned, independent_input
from test_relational_field_evidence import matrix, read_matrix, reference, HELD_INPUT
from test_relational_field_program import action
from test_relational_field_evidence import Reader


def declaration(*, kind='green', frame=None, interior=(0,2,4)):
    return {'key':8501,'revision':1,'coordinate_carrier':8502,'coordinate_basis':8503,
            'grade':0,'kind':kind,'lambda':Q(1,3) if kind=='green' else Q(0),
            'harmonic_frame':frame,'interior':interior}


def readout():
    a=tuple(tuple(Q(i==j) if i<7 else Q(j%2) for j in range(7)) for i in range(8))
    r=tuple(tuple(Q(i==j) for j in range(8)) for i in range(7))
    return {'patch':((0,1,2),(0,1)), 'view':{'key':8504,'basis':8505,'directions':tuple(range(8600,8608)),
            'accession':matrix(a),'reconstruction':matrix(r),'physical':matrix(tuple(tuple(Q(i==j) for j in range(7)) for i in range(7)))}}


def grant(owner,rhs,origins,d,r):
    payloads=local_field_origin_payloads(rhs,2,d,r);keys={'problem':8801,'interface':8802,'reconstruction':8803,'gauge':0}
    for name,role in (('problem','maps'),('interface','interface'),('reconstruction','reconstruction')):
        origins[keys[name]]=publish(keys[name],owner,role,payloads[name])
    if d['harmonic_frame'] is not None:
        keys['gauge']=8804;origins[8804]=publish(8804,owner,'maps',payloads['gauge'])
    return keys,context(origins)


def fixture(values=HELD_INPUT, *, d=None, r=None):
    owner,program,available,row,origins,authority=prepared(values)
    rhs=available.execute_inputs((row,),authority);d=declaration() if d is None else d;r=readout() if r is None else r
    keys,authority=grant(owner,rhs,origins,d,r)
    return owner,program,rhs,origins,authority,d,r,keys


def solve(a,b):
    rows=[list(map(Q,row))+[Q(value)] for row,value in zip(a,b,strict=True)];n=len(rows)
    for k in range(n):
        pivot=next(i for i in range(k,n) if rows[i][k]);rows[k],rows[pivot]=rows[pivot],rows[k]
        q=rows[k][k];rows[k]=[v/q for v in rows[k]]
        for i in range(n):
            if i!=k:
                q=rows[i][k];rows[i]=[v-q*w for v,w in zip(rows[i],rows[k],strict=True)]
    return tuple(row[-1] for row in rows)


def values(image):
    r=Reader(image);shape=tuple(r.u64() for _ in range(r.u64()));assert len(shape)==1
    full=r.take(1)[0]
    if not full:
        for _ in range(r.u64()):r.u64()
    length,count=shape[0],r.u64();out=[Q(0)]*length
    for _ in range(count):
        index=r.u64();out[index]=r.rational()
    assert r.offset==len(image) and length==shape[0]
    return tuple(out)


def green_reference(v, theta=(0,0,0,0,0)):
    c=tuple(zip(*(reference(tuple(Q(i==j) for i in range(4)),tuple(map(Q,theta)))[0] for j in range(4)),strict=True))
    n=((2,1,0,0,0,0,0),(1,3,0,0,0,0,0),(0,0,1,0,0,0,0),(0,0,0,1,0,0,0),
       (0,0,0,0,1,0,0),(0,0,0,0,0,1,0),(0,0,0,0,0,0,1))
    m=((3,1,0,0),(1,2,0,0),(0,0,1,0),(0,0,0,1))
    columns=[]
    for j in range(7):
        e=tuple(Q(i==j) for i in range(7));dual=action(n,e)
        adjoint=action(tuple(zip(*c,strict=True)),dual);up=action(c,solve(m,adjoint))
        columns.append(tuple(e[i]+Q(1,3)*up[i] for i in range(7)))
    operator=tuple(zip(*columns,strict=True));return solve(operator,reference(v,tuple(map(Q,theta)))[0]),c,n,m


@pytest.mark.parametrize('v',[HELD_INPUT,(11,-2,3,1),(-3,0,2,5),(0,4,-1,2),(1,0,0,1),(0,0,3,7),(5,7,0,0),(-2,-3,-5,-7)])
def test_complete_green_fields_match_independent_fraction(v):
    owner,program,rhs,origins,authority,d,r,keys=fixture(v)
    run=submit_local_field_solve(rhs,2,d,r,keys,authority); result=run.wait(authority)
    expected,c,n,m=green_reference(tuple(map(Q,v)))
    assert values(result.field_archive())==expected
    assert values(result.field_archive('patch'))==expected
    assert values(result.field_archive('reconstructed'))==expected
    a=tuple(tuple(Q(i==j) if i<7 else Q(j%2) for j in range(7)) for i in range(8))
    assert values(result.field_archive('visible'))==action(a,expected)
    assert not any(values(result.field_archive('residual')))
    info=result.info();assert read_matrix(info['matrices']['upper_boundary'])==c
    assert read_matrix(info['matrices']['metric'])==tuple(tuple(map(Q,row)) for row in n)
    assert read_matrix(info['matrices']['above_metric'])==tuple(tuple(map(Q,row)) for row in m)
    full=values(result.field_archive('augmented_response'));b=values(result.field_archive('augmented_rhs'))
    assert action(read_matrix(info['matrices']['augmented_operator']),full)==b
    assert action(read_matrix(info['matrices']['schur']),values(result.field_archive('interface_response')))==values(result.field_archive('interface_rhs'))
    assert run.receipt()['joined'] and run.receipt()['status']=='succeeded'
    assert result.rhs().archive()==rhs.archive();assert info['patch']==r['patch']
    selected=result.select(2,((0,1),(0,0)),coordinate_limit=5)
    assert selected[0]['values']==expected[2:5] and selected[1]['values']==expected[:2]
    assert selected[0]['directions']==(201,202,203)
    with pytest.raises((ValueError,IndexError)):result.select(1,((0,1),))
    with pytest.raises((ValueError,OverflowError,RuntimeError)):result.select(2,((0,1),(0,0)),coordinate_limit=4)
    assert NativeLocalFieldResult.from_archive(result.archive(),authority).archive()==result.archive()
    assert NativeLocalFieldWork.replay(result.archive(),authority).wait(authority).archive()==result.archive()


@pytest.mark.parametrize('key',(1,2,3,4,5,6,7,8801,8802,8803))
def test_every_original_and_local_role_checked_before_action_and_cold_replay(key):
    owner,program,rhs,origins,authority,d,r,keys=fixture()
    result=submit_local_field_solve(rhs,2,d,r,keys,authority).wait(authority)
    missing=context(origins,remove=(key,))
    with pytest.raises(ValueError):result.admit(missing)
    with pytest.raises(ValueError):NativeLocalFieldResult.from_archive(result.archive(),missing)
    run=submit_local_field_solve(rhs,2,d,r,keys,missing)
    with pytest.raises(ValueError):run.wait(missing)
    assert run.receipt()['joined'] and run.receipt()['status']=='failed'


@pytest.mark.parametrize('phase',('preparation','action','replay'))
def test_actual_native_cancel_joins_and_keeps_old_immutable_owners(phase):
    owner,program,rhs,origins,authority,d,r,keys=fixture();original=rhs.archive()
    complete=submit_local_field_solve(rhs,2,d,r,keys,authority).wait(authority)
    if phase=='replay':run=NativeLocalFieldWork.replay(complete.archive(),authority,pause_at=phase)
    else:run=submit_local_field_solve(rhs,2,d,r,keys,authority,pause_at=phase)
    deadline=time.monotonic()+10
    while run.phase()!=phase:
        assert time.monotonic()<deadline;time.sleep(.001)
    run.cancel();receipt=run.receipt();assert receipt['joined'] and receipt['status']=='cancelled'
    with pytest.raises(RuntimeError):run.wait(authority)
    assert rhs.archive()==original and complete.rhs().archive()==original
    assert submit_local_field_solve(rhs,2,d,r,keys,authority).wait(authority).archive()==complete.archive()


def test_full_harmonic_gauge_and_bad_readout_reject_without_synthetic_zeros():
    d=declaration(kind='harmonic-inverse',interior=());owner,program,rhs,origins,authority,d,r,keys=fixture(d=d)
    result=submit_local_field_solve(rhs,2,d,r,keys,authority).wait(authority)
    frame=read_matrix(result.info()['matrices']['harmonic_frame']);assert len(frame)==7 and len(frame[0])==3
    n=read_matrix(result.info()['matrices']['metric']);response=values(result.field_archive())
    assert action(tuple(zip(*frame,strict=True)),action(n,response))==(Q(0),)*3
    supplied=deepcopy(d);supplied['harmonic_frame']=matrix(frame);keys,authority=grant(owner,rhs,origins,supplied,r)
    assert submit_local_field_solve(rhs,2,supplied,r,keys,authority).wait(authority).field_archive()==result.field_archive()
    wrong=deepcopy(supplied);wrong['harmonic_frame']=matrix(tuple(row[:1] for row in frame));keys,bad=grant(owner,rhs,origins,wrong,r)
    with pytest.raises(ValueError):submit_local_field_solve(rhs,2,wrong,r,keys,bad).wait(bad)
    unclosed=deepcopy(r);unclosed['patch']=((0,1),(0,1));keys,bad=grant(owner,rhs,origins,d,unclosed)
    with pytest.raises(ValueError):submit_local_field_solve(rhs,2,d,unclosed,keys,bad).wait(bad)


def selected_fixture():
    owner,program,root,child,origins,theta=learned();held=independent_input(owner,program,origins,91,HELD_INPUT)
    family=child.execute_inputs((held,),context(origins));free=(Q(3),Q(-2))
    origins[8899]=publish(8899,owner,'prospective',child.parameter_setting_payload(free));authority=context(origins)
    selected=family.evaluate(free,8899,authority);d,r=declaration(),readout();keys,authority=grant(owner,selected,origins,d,r)
    return owner,program,root,child,selected,origins,authority,d,r,keys,theta


def test_selected_child_full_evidence_setting_and_held_owner_survive_local_and_concurrent_cold_replay():
    owner,program,root,child,rhs,origins,authority,d,r,keys,theta=selected_fixture()
    result=submit_local_field_solve(rhs,2,d,r,keys,authority).wait(authority)
    assert values(result.field_archive())==green_reference(HELD_INPUT,theta)[0]
    restored=result.rhs();assert isinstance(restored,NativeAvailableAffineEvaluation)
    assert restored.archive()==rhs.archive() and restored.family().evidence().archive()==child.archive()
    assert len(result.dependency_archives())==len(rhs.dependency_archives())+3
    with ThreadPoolExecutor(max_workers=6) as pool:
        images=list(pool.map(lambda _:NativeLocalFieldResult.from_archive(result.archive(),authority).archive(),range(12)))
    assert all(image==result.archive() for image in images)
    for key in (4,114,115,8899,911,910):
        with pytest.raises(ValueError):NativeLocalFieldResult.from_archive(result.archive(),context(origins,remove=(key,)))


def test_queued_input_buffers_are_frozen_and_packet_graft_is_rejected():
    owner,program,rhs,origins,authority,d,r,keys=fixture()
    run=submit_local_field_solve(rhs,2,d,r,keys,authority,pause_at='preparation')
    d['lambda']=Q(5);r['view']['directions']=(1,);run.resume();result=run.wait(authority)
    assert result.info()['lambda']=='1/3' and result.info()['readout_directions']==tuple(range(8600,8608))
    with pytest.raises(ValueError):NativeLocalFieldResult.from_archive(graft_native(result.archive(),b'foreign'),authority)
    with pytest.raises((ValueError,OverflowError,RuntimeError)):NativeLocalFieldResult.from_archive(result.archive(),authority,limits=(2,100,100,32))


@pytest.mark.parametrize('role',('problem','interface','reconstruction','gauge'))
def test_local_numeric_publication_cutoffs_remain_independent_of_original_source(native_fixtures,role):
    from rexgraph_cpp import NativeSource
    history=NativeSource((native_fixtures[0]/'diagonal-history.rfcxs').read_bytes())
    d=declaration(kind='harmonic-inverse',interior=()) if role=='gauge' else declaration()
    owner,program,rhs,origins,authority,d,r,keys=fixture(d=d)
    if role=='gauge':
        solved=submit_local_field_solve(rhs,2,d,r,keys,authority).wait(authority)
        d['harmonic_frame']=matrix(read_matrix(solved.info()['matrices']['harmonic_frame']))
        keys,authority=grant(owner,rhs,origins,d,r)
    key=keys[role];prior=origins[key]
    origins[key]=publish(key,history,'interface' if role=='interface' else 'reconstruction' if role=='reconstruction' else 'maps',
                         prior.payload(),limits=(4096,16000000,64000000,1<<24))
    admitted=context(origins);result=submit_local_field_solve(rhs,2,d,r,keys,admitted).wait(admitted)
    denied=context(origins,limits={key:history.identity['first_sequence']})
    with pytest.raises(ValueError):result.admit(denied)
    with pytest.raises(ValueError):NativeLocalFieldResult.from_archive(result.archive(),denied)
    run=submit_local_field_solve(rhs,2,d,r,keys,denied)
    with pytest.raises(ValueError):run.wait(denied)
    assert run.receipt()['joined']


def test_core_portable_packet_routes_without_a_separate_model_file(tmp_path):
    from rexgraph.cpp_relational_field_admission import cpp_available_field_program_model,available_field_query
    from rexgraph.cpp_relational_field_local import solve_available_field_output,read_local_field_result
    from rexgraph.cpp_relational_field_program import field_program_stalks
    from rexgraph.model_runtime import infer_model
    from rexgraph.value_codec import pack_value,unpack_value
    from rexgraph.model_codec import pack_model,unpack_model
    from system.serialize import json_value
    from rcql.model_operators import model_values,model_stalk
    owner,program,available,row,origins,authority=prepared()
    state=cpp_available_field_program_model(owner,available_program=available,execution_context=authority)
    rhs=infer_model(state,owner,available_field_query(state,(row,)),execution_context=authority)
    native=available.execute_inputs((row,),authority);d,r=declaration(),readout();keys,authority=grant(owner,native,origins,d,r)
    output=solve_available_field_output(rhs,2,d,r,keys,execution_context=authority,state=state)
    portable=unpack_model(unpack_value(pack_value(pack_model(output,native=True))))
    restored=read_local_field_result(portable,execution_context=authority)
    assert restored.archive()==output.field_archive['data'] and not portable.values.size
    assert field_program_stalks(portable,2,((0,1),),execution_context=authority)[0]['values']==values(restored.field_archive())[2:5]
    assert json_value(portable)['packet_context']=='complete-global-context'
    from rexgraph.io.model_record import model_record
    record=model_record(owner,state)
    assert model_values(record,portable,execution_context=authority)['packet_context']=='complete-global-context'
    assert model_stalk(record,portable,(0,1),node=2,execution_context=authority)['values']==values(restored.field_archive())[2:5]
    missing=context(origins,remove=(8801,))
    with pytest.raises(ValueError):read_local_field_result(portable,execution_context=missing)
    with pytest.raises(TypeError):read_local_field_result(portable,execution_context=None)
    with pytest.raises(ValueError):read_local_field_result(replace(portable,dependencies=portable.dependencies[:-1]),execution_context=authority)


@pytest.mark.parametrize('backend',('file','rex','sqlite'))
def test_typed_rcql_checked_record_copy_and_agent_local_worker(backend,tmp_path):
    from rexgraph.cpp_relational_field_admission import cpp_available_field_program_model,available_field_query
    from rexgraph.cpp_relational_field_local import solve_available_field_output,read_local_field_result
    from rexgraph.model_runtime import infer_model
    from rexgraph.io.model_record import model_record,read_model_record
    from rcdb import open_store,copy_record
    from rcql.executor import Executor
    from rcql.parser import parse
    from agent.hive import Hive
    owner,program,available,row,origins,authority=prepared()
    state=cpp_available_field_program_model(owner,available_program=available,execution_context=authority)
    request=available_field_query(state,(row,));rhs=infer_model(state,owner,request,execution_context=authority)
    d,r=declaration(),readout();keys,authority=grant(owner,available.execute_inputs((row,),authority),origins,d,r)
    result=solve_available_field_output(rhs,2,d,r,keys,execution_context=authority)
    record=model_record(owner,state);record.attach_metadata(1,0,'local-response',result)
    query='FROM $r RETURN MODEL_LOCAL($rhs,2,$d,$view,$keys,execution_context=$authority),MODEL_VALUES($local,execution_context=$authority),MODEL_STALK($local,[0,1],node=2,execution_context=$authority)'
    response=Executor(sources={'r':record},params={'rhs':rhs,'d':d,'view':r,'keys':keys,'authority':authority,'local':result}).execute(parse(query)).values
    assert response[0].field_archive['data']==result.field_archive['data'] and response[1]['packet_context']=='complete-global-context'
    assert response[2]['values']==values(read_local_field_result(result,execution_context=authority).field_archive())[2:5]
    stores=[open_store('sqlite:///'+str(tmp_path/(name+'.sqlite'))) if backend=='sqlite' else
            open_store(backend+'://'+str(tmp_path/name),read_only=False) for name in ('first','second')]
    try:
        stored=stores[0].put('local-serving',record);copy_record(stores[0],stores[1],stored)
        copied=stores[1].get('local-serving');parent,data=read_model_record(copied,execution_context=authority)
        restored=copied.get_metadata(1,0,'local-response')
        assert read_local_field_result(restored,execution_context=authority).archive()==result.field_archive['data']
        hive=Hive();hive.add_worker('local-serving',lambda row:solve_available_field_output(
            infer_model(parent,data,row,execution_context=authority),2,d,r,keys,execution_context=authority),capability='analyze')
        assert hive.dispatch_capability('analyze',request)['result'].field_archive['data']==result.field_archive['data']
    finally:
        for store in stores:store.close()
