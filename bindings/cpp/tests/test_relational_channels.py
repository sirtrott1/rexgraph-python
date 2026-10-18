"""Exact native channel equations, original identities and five package replay."""
from fractions import Fraction as Q
import subprocess
import sys

import numpy as np
import pytest

from rexgraph_cpp import NativeSource, PreparedRelationalChannels
from rexgraph.cpp_channels import cpp_channel_model, relational_channel_query
from rexgraph.io.model_record import model_record, save_model, load_model, read_model_record
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import ModelState, ModelOutput, thaw_tree


def declaration(namespace=719231):
    columns = (((10,-1,True),(20,Q(1,2),True),(30,Q(1,2),True)),
               ((10,Q(1,4),True),(20,-1,True),(30,Q(3,4),True),(40,7,False)),
               ((10,0,True),(20,1,True),(40,-1,True)))
    offset = 0; relations = []
    for j, column in enumerate(columns):
        flags = tuple(dict(key=offset+i+1, lower=lo, negative=q<0, magnitude=abs(q), exists=active, aspects=(0,))
                      for i,(lo,q,active) in enumerate(column))
        relations.append(dict(grade=1,key=100+j,type=0,occurrences=flags)); offset += len(flags)
    return dict(profile='declared-role-source-v1', namespace=namespace, semantic_scale=0, max_grade=2,
                role_aspects=('original-participant',), types=(dict(name='branch',min_arity=0,max_arity=0),),
                witnesses=(10,20,30,40), relations=tuple(relations), fields=(), metrics=(),
                carriers=(dict(key=10,kind='source_grade',grade=0,coordinates=0,size=4),
                          dict(key=11,kind='source_grade',grade=1,coordinates=0,size=3),
                          dict(key=12,kind='source_grade',grade=2,coordinates=0,size=0)))


def oracle(weights, field, count):
    b = ((Q(-1),Q(1,4),Q(0)), (Q(1,2),Q(-1),Q(1)),
         (Q(1,2),Q(3,4),Q(0)), (Q(0),Q(0),Q(-1)))
    t = [[sum((row[i]*row[j]*weights[i]*weights[j] for row in b),Q(0)) for j in range(3)] for i in range(3)]
    g = [[sum((abs(row[i])*abs(row[j])*weights[i]*weights[j] for row in b),Q(0)) for j in range(3)] for i in range(3)]
    ac = [[Q(bool(q)) if count else abs(q) for q in row] for row in b]
    k = [[sum((row[i]*row[j] for row in ac),Q(0)) for j in range(3)] for i in range(3)]
    f = [[t[i][j]-g[i][j] for j in range(3)] for i in range(3)]
    c = [[-q for q in row] for row in k]
    for i in range(3):
        f[i][i] += sum(map(abs,f[i]),Q(0)); c[i][i] += sum(k[i],Q(0))
    matrices = (t,g,f,c)
    values = tuple(tuple(sum((q*x for q,x in zip(row,field,strict=True)),Q(0)) for row in matrix) for matrix in matrices)
    traces = tuple(sum((matrix[i][i] for i in range(3)),Q(0)) for matrix in matrices)
    return values,traces


@pytest.mark.parametrize('reading',['share','count'])
@pytest.mark.parametrize('normalized',[False,True])
@pytest.mark.parametrize('weights',[(1,1,1),(-2,Q(3,5),0),(0,0,0)])
def test_complete_native_fields_match_independent_fraction_oracle(reading,normalized,weights):
    source = NativeSource.from_declaration(declaration()); field = (Q(2,7),-3,Q(1,5))
    plan = PreparedRelationalChannels(source.archive,weights,coparticipation=reading)
    result = plan.apply(field,trace_normalized=normalized)
    expected,traces = oracle(weights,field,reading=='count')
    if normalized:
        expected = tuple(tuple(q/trace if trace else q for q in row) for row,trace in zip(expected,traces,strict=True))
    assert result['fields'] == expected and plan.receipt['traces'] == traces
    assert result['energies'] == tuple(sum((x*y for x,y in zip(field,row,strict=True)),Q(0)) for row in expected)
    assert all(q>=0 for q in result['energies'])
    assert plan.receipt['occurrences']==10 and plan.receipt['occurrence_keys']==tuple(range(1,11))
    assert plan.receipt['occurrence_activity']==(True,)*6+(False,)+(True,)*3
    assert plan.receipt['signed_incidence'][6:8]==(0,0)
    assert len(source.identity['coordinates'])==3
    state = cpp_channel_model(source,weights,coparticipation=reading,trace_normalized=normalized)
    output = infer_model(state,source,relational_channel_query(state,field))
    assert tuple(tuple(output.values[:,k]) for k in range(4))==expected
    assert output.tensor().axes[0].keys==('T','G','F','C')


@pytest.mark.parametrize('bad',[1.0,True,complex(1),Q(1,1<<130)])
def test_exact_admission_refuses_implicit_numeric_profiles(bad):
    source=NativeSource.from_declaration(declaration())
    with pytest.raises((TypeError,ValueError)):
        PreparedRelationalChannels(source.archive,(1,bad,1),limits=(32,1000,10000,128))


def test_repeated_participants_refuse_without_losing_source_identity():
    data=declaration();data['relations'][0]['occurrences'][1]['lower']=10
    source=NativeSource.from_declaration(data); archive=source.archive
    with pytest.raises(ValueError,match='distinct participants'):
        PreparedRelationalChannels(archive,(1,1,1))
    assert source.archive==archive


def test_numpy_integer_fields_and_weights_remain_exact():
    source=NativeSource.from_declaration(declaration());values=np.array([1,-2,3],dtype=np.int64)
    plan=PreparedRelationalChannels(source.archive,values)
    assert plan.apply(values)['fields']==oracle((1,-2,3),(1,-2,3),False)[0]
    state=cpp_channel_model(source,values);output=infer_model(state,source,relational_channel_query(state,values))
    assert all(isinstance(q,Q) for q in output.values.flat)


def test_source_binding_admission_and_owner_are_explicit():
    source=NativeSource.from_declaration(declaration()); plan=PreparedRelationalChannels(source.archive,(1,1,1))
    with pytest.raises(RuntimeError,match='immutable'):plan.__init__(source.archive,(1,1,1))
    with pytest.raises(TypeError):plan.apply((1,2,3),trace_normalized=1)
    with pytest.raises(ValueError):plan.apply((1,2))
    with pytest.raises(ValueError):plan.apply((1,2,3,4))
    with pytest.raises((ValueError,RuntimeError)):
        PreparedRelationalChannels(source.archive,(1,1,1),limits=(3,1000,10000,128))
    bounded=PreparedRelationalChannels(source.archive,(1,1,1),limits=(32,64,10000,128))
    with pytest.raises((ValueError,RuntimeError)):bounded.apply((1,2,3))
    state=cpp_channel_model(source); foreign=NativeSource.from_declaration(declaration(719232))
    query=relational_channel_query(state,(1,2,3)); query['source_reference']=cpp_channel_model(foreign).source.as_record()
    with pytest.raises(ValueError,match='actual original source'):infer_model(state,source,query)


def test_forged_diagonal_and_floating_receipt_are_rejected():
    source=NativeSource.from_declaration(declaration()); state=cpp_channel_model(source)
    c=thaw_tree(state.configuration);p=c['native_plan'];rows=p['diagonals']
    p['diagonals']=(*rows[:2],(rows[2][0]+1,*rows[2][1:]),rows[3])
    p['traces']=(*p['traces'][:2],p['traces'][2]+1,p['traces'][3])
    with pytest.raises(ValueError,match='diagonal differs'):
        ModelState(state.adapter,state.adapter_version,state.source,state.space,state.output_axes,c,{},'rational')
    c=thaw_tree(state.configuration);c['native_plan']['weights']=(1.0,*c['native_plan']['weights'][1:])
    with pytest.raises(TypeError):
        ModelState(state.adapter,state.adapter_version,state.source,state.space,state.output_axes,c,{},'rational')
    query=relational_channel_query(state,(1,2,3));output=infer_model(state,source,query)
    bad=output.values.copy();bad[0,2]+=1
    with pytest.raises(ValueError,match='field or energy'):
        ModelOutput(bad,output.space,output.axes,output.source,output.model_digest,'rational',output.method,
                    grade=1,variance='chain',execution_receipt=output.execution_receipt)


def test_rcql_creates_the_same_closed_native_channel_profile():
    from rcql import Executor,parse
    source=NativeSource.from_declaration(declaration());state=cpp_channel_model(source)
    record=model_record(source,state);configuration=dict(weights=(-2,Q(3,5),0),coparticipation='count',trace_normalized=True)
    created=Executor(sources={'r':record},params={'c':configuration}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_channels",$c)')).values[0]
    assert created.adapter=='cpp_channels' and created.arithmetic=='rational'
    query=relational_channel_query(created,(1,2,3))
    result=Executor(sources={'r':record},params={'m':created,'q':query}).execute(parse(
        'FROM $r RETURN MODEL_INFER($m,$q),MODEL_INFO($m)'))
    expected=infer_model(created,source,query)
    assert result.values[0].coefficient_digest==expected.coefficient_digest
    assert result.values[1]['coparticipation']=='count' and result.exactness[0].value=='rational'


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_exact_channel_fields_roundtrip_through_both_model_formats(tmp_path,suffix):
    source=NativeSource.from_declaration(declaration());state=cpp_channel_model(source,(-2,Q(3,5),0),trace_normalized=True)
    query=relational_channel_query(state,(1,Q(-3,4),2));output=infer_model(state,source,query)
    assert unpack_model(pack_model(state)).coefficient_digest==state.coefficient_digest
    record=model_record(source,state);record.attach_metadata(1,0,'output',output)
    path=tmp_path/('channels'+suffix);save_model(path,record);restored_record=load_model(path)
    restored,native=read_model_record(restored_record)
    assert native.archive==source.archive and restored_record.get_metadata(1,0,'output').coefficient_digest==output.coefficient_digest
    assert infer_model(restored,native,query).coefficient_digest==output.coefficient_digest


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_cold_store_rcql_agent_and_system_use_one_native_channel_contract(tmp_path,backend):
    from rcdb import open_store
    source=NativeSource.from_declaration(declaration());state=cpp_channel_model(source)
    query=relational_channel_query(state,(1,2,3));output=infer_model(state,source,query)
    record=model_record(source,state);record.attach_metadata(1,0,'query',query);record.attach_metadata(1,0,'output',output)
    url={'file':'file://'+str(tmp_path/'files'),'rex':'rex://'+str(tmp_path/'rex'),'sqlite':'sqlite:///'+str(tmp_path/'db')}[backend]
    store=open_store(url,**({} if backend=='sqlite' else {'read_only':False})).configure_security(require_commits=True)
    store.commit_mutation('channels',record,expected_version=0);assert store.verify_commits('channels');store.close()
    script='''
import sys
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.cpp_channels import enable_channel_provider
from rexgraph.model_runtime import infer_model
from rexgraph.io.model_record import read_model_record
enable_channel_provider()
store=open_store(sys.argv[1]);record=store.read_record('channels',version=1).value
state,source=read_model_record(record);query=record.get_metadata(1,0,'query')
result=Executor(sources={'r':record},params={'q':query}).execute(parse(
 'FROM $r RETURN MODEL_INFER(MODEL_STATE(),$q),MODEL_INFO(MODEL_STATE())'))
assert result.values[0].coefficient_digest==sys.argv[2] and result.exactness[0].value=='rational'
assert result.values[1]['channels']==('T','G','F','C') and not result.values[1]['trainable']
assert json_value(state)['channels']==['T','G','F','C']
with patch('agent.activity.record'),patch('agent.activity.open_use',return_value=0),patch('agent.activity.close_use'):
 hive=Hive('channels');hive.add_worker('channels',lambda q,**kw:infer_model(state,source,q),capability='predict')
 assert hive.invoke('channels',query,record=False).coefficient_digest==sys.argv[2]
assert record.get_metadata(1,0,'output').coefficient_digest==sys.argv[2]
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
store.close()
'''
    result=subprocess.run([sys.executable,'-c',script,url,output.coefficient_digest],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
