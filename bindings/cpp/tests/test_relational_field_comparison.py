"""Whole native comparison fields, actual source owners and independent oracles."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from dataclasses import replace
from copy import deepcopy
import pytest
from rexgraph_cpp import (PreparedFieldProgram,PreparedAccessionFieldProgram,PreparedFieldComparison,
    NativeInputAccession,NativeComparisonFieldResult,NativeComparisonPair)
from test_relational_field_program import source,tensor,MAPS,UPPER,LOWER,action
from test_relational_field_evidence import learning_declaration,actual_asset,input_accession,read_matrix
from test_relational_field_program import matrix,LATER

COEFFICIENTS=tuple(map(Q,(1,-1,-1,0,0,1,Q(1,2),-1)))
PHI=(*MAPS[:7],tuple(tuple(-q for q in row) for row in MAPS[7]))
METRICS=(((2,1),(1,3)),((1,0,0),(0,1,0),(0,0,1)),((1,0),(0,1)))


def comparison_declaration(policy='share'):
    return {'profile':'rexgraph.relational-comparison-forms.v1','key':4101,'revision':1,'grade':1,
        'participation':policy,'weights':({'cell':(1,0),'value':Q(-2)},{'cell':(1,1),'value':Q(3)})}


def oracle(x,channel,policy):
    operators=[]
    for p,m in enumerate(PHI):
        rows=[]
        for row in m:
            whole=[Q(0)]*4;whole[2*UPPER[p]:2*UPPER[p]+2]=map(Q,row);rows.append(tuple(whole))
        operators.append(tuple(rows))
    k=[[Q(0)]*4 for _ in range(4)];boundaries=[];unsigned=[]
    def add_form(op,metric,weight):
        for i in range(4):
            for j in range(4):k[i][j]+=weight*sum((op[a][i]*Q(metric[a][b])*op[b][j] for a in range(len(op)) for b in range(len(op))),Q(0))
    for v,metric in enumerate(METRICS):
        flags=[p for p in range(8) if LOWER[p]==v];signed=[[Q(0)]*4 for _ in metric];raw=[[Q(0)]*4 for _ in metric]
        for p in flags:
            w=Q((-2,3)[UPPER[p]])
            for i,row in enumerate(operators[p]):
                for j,q in enumerate(row):signed[i][j]+=w*COEFFICIENTS[p]*q;raw[i][j]+=w*abs(COEFFICIENTS[p])*q
        boundaries.extend(action(signed,x));unsigned.extend(action(raw,x))
        if channel in ('T','rawG'):add_form(signed if channel=='T' else raw,metric,Q(1))
        else:
            for i,p in enumerate(flags):
                for q in flags[i+1:]:
                    sign=Q(1)
                    if channel=='C':
                        beta=lambda f:abs(COEFFICIENTS[f]) if policy=='share' else Q(COEFFICIENTS[f]!=0)
                        weight=beta(p)*beta(q)
                    else:
                        product=Q((-2,3)[UPPER[p]])*Q((-2,3)[UPPER[q]])
                        weight=2*abs(product*COEFFICIENTS[p]*COEFFICIENTS[q]) if COEFFICIENTS[p]*COEFFICIENTS[q]<0 else Q(0)
                        sign=Q(1) if product>=0 else Q(-1)
                    op=tuple(tuple(a-sign*b for a,b in zip(ar,br,strict=True)) for ar,br in zip(operators[p],operators[q],strict=True))
                    add_form(op,metric,weight)
    covector=action(k,x);a,b=covector[:2]
    output=((2*a-b)/5,(-a+3*b)/5,*covector[2:])
    energy=sum((a*b for a,b in zip(x,covector,strict=True)),Q(0))
    flags=tuple(action(m,x[2*UPPER[p]:2*UPPER[p]+2]) for p,m in enumerate(PHI))
    return tuple(output),tuple(boundaries),tuple(unsigned),flags,energy


@pytest.fixture
def prepared():
    owner=source();program=PreparedAccessionFieldProgram(PreparedFieldProgram(owner.archive,learning_declaration()).archive())
    asset=actual_asset(999991,111,b'actual original comparison input')
    return program,asset


@pytest.mark.parametrize('trial',range(4))
@pytest.mark.parametrize('policy',['share','legacy_count'])
@pytest.mark.parametrize('channel',['T','rawG','C','F'])
def test_complete_factored_fields_occurrences_energy_and_native_replay_match_fraction(prepared,trial,policy,channel):
    program,asset=prepared;forms=PreparedFieldComparison(program,comparison_declaration(policy))
    x=(Q(trial-4,3),Q(trial%7-2,5),Q(trial%13-3,7),Q(trial%11+2,9))
    accession=NativeInputAccession(asset,forms.program(),input_accession());parent=forms.program().execute((accession.admit(111,tensor(x)),))
    result=forms.apply(parent,node=1,channel=channel);output,signed,unsigned,flags,energy=oracle(x,channel,policy)
    assert result.field_archive('input')==tensor(x) and result.field_archive()==tensor(output)
    assert result.field_archive('signed_boundary')==tensor(signed) and result.field_archive('unsigned_boundary')==tensor(unsigned)
    for p,values in enumerate(flags):assert result.occurrence_field_archive(p)==tensor(values)
    assert result.inspect()['quadrance']==energy and result.inspect()['occurrences']==8
    assert result.select(((1,0),(1,1)),coordinate_limit=4)[0]['values']==output[:2]
    assert NativeComparisonFieldResult.from_archive(result.archive()).archive()==result.archive()
    assert result.input().archive()==parent.archive()
    assert PreparedFieldComparison.from_archive(forms.archive()).archive()==forms.archive()
    operator=read_matrix(forms.operator_matrix(channel,max_entries=16));assert action(operator,x)==output


def test_actual_owners_wrong_node_grade_bounds_and_pair_parent_survive_producers(prepared):
    program,asset=prepared;forms=PreparedFieldComparison(program,comparison_declaration());x=tuple(map(Q,(2,3,5,7)))
    accession=NativeInputAccession(asset,forms.program(),input_accession());parent=program.execute((accession.admit(111,tensor(x)),))
    foreign=PreparedFieldComparison.from_archive(forms.archive())
    with pytest.raises(ValueError):foreign.apply(parent,node=1,channel='T')
    with pytest.raises(ValueError):forms.apply(parent,node=2,channel='T')
    result=forms.apply(parent,node=1,channel='F');pair=result.pair(1,6,channel='F');view=pair.inspect()
    values=tuple(a+b for a,b in zip(action(PHI[1],x[:2]),action(PHI[6],x[2:]),strict=True))
    assert view['admitted'] and view['weight']==6 and view['residual']==tensor(values)
    assert view['quadrance']==6*sum((q*q for q in values),Q(0))
    assert not result.pair(0,5,channel='F').inspect()['admitted']
    assert result.pair(1,3,channel='C').inspect()['weight']==0
    with pytest.raises(ValueError):result.pair(0,1,channel='C')
    with pytest.raises(RuntimeError):result.select(((1,0),),coordinate_limit=1)
    with pytest.raises(RuntimeError):forms.inspect(max_records=1)
    with pytest.raises(RuntimeError):forms.operator_matrix('T',max_entries=1)
    image=result.archive();del result,parent,forms,program,asset,accession
    assert pair.parent().archive()==image and pair.parent().input().input_uses()[0]['use']==111
    for cls,method in ((PreparedFieldComparison,'archive'),(NativeComparisonFieldResult,'archive'),(NativeComparisonPair,'inspect')):
        with pytest.raises(RuntimeError):getattr(cls.__new__(cls),method)()
    with ThreadPoolExecutor(4) as pool:images=list(pool.map(lambda _:pair.parent().archive(),range(8)))
    assert len(set(images))==1


@pytest.mark.parametrize('channel',['T','rawG','C','F'])
@pytest.mark.parametrize('policy',['share','legacy_count'])
def test_complete_comparison_fields_transform_with_declared_oblique_bases_and_metrics(prepared,channel,policy):
    def multiply(a,b):
        return tuple(tuple(sum((Q(a[i][k])*Q(b[k][j]) for k in range(len(b))),Q(0)) for j in range(len(b[0]))) for i in range(len(a)))
    def transpose(a):return tuple(zip(*a,strict=True))
    def blocks(*parts):
        size=sum(len(p) for p in parts);out=[[Q(0)]*size for _ in range(size)];offset=0
        for p in parts:
            for i,row in enumerate(p):out[offset+i][offset:offset+len(row)]=map(Q,row)
            offset+=len(p)
        return tuple(tuple(r) for r in out)
    identity=lambda n:tuple(tuple(Q(i==j) for j in range(n)) for i in range(n))
    s=((1,1),(0,1));si=((1,-1),(0,1));q=((1,0),(1,1));qi=((1,0),(-1,1))
    upper=blocks(s,identity(2));lower,lower_inverse=blocks(q,identity(3),identity(2)),blocks(qi,identity(3),identity(2))
    spec=deepcopy(learning_declaration())
    for i,stalk in enumerate(spec['stalks']):
        if i in (0,3):
            inverse=qi if i==0 else si;old=((2,1),(1,3)) if i==0 else ((3,1),(1,2))
            stalk['metric']=matrix(multiply(multiply(transpose(inverse),old),inverse));stalk['basis']=9000+i
            stalk['directions']=(9100+2*i,9101+2*i)
    for p,entry in enumerate(spec['occurrences']):
        entry['map']=matrix(multiply(multiply(q if LOWER[p]==0 else identity(len(PHI[p])),PHI[p]),si if UPPER[p]==0 else identity(2)))
    spec['nodes'][2]['map']=matrix(multiply(multiply(upper,LATER),lower_inverse))
    program,asset=prepared;x=tuple(map(Q,(2,3,5,7)));original=PreparedFieldComparison(program,comparison_declaration(policy))
    acc=NativeInputAccession(asset,original.program(),input_accession());parent=original.program().execute((acc.admit(111,tensor(x)),));before=original.apply(parent,node=1,channel=channel)
    transformed=PreparedAccessionFieldProgram(PreparedFieldProgram(source().archive,spec).archive());forms=PreparedFieldComparison(transformed,comparison_declaration(policy))
    declaration_=input_accession();stalks=list(declaration_['stalks']);stalks[0]['metric']=spec['stalks'][3]['metric'];declaration_['stalks']=tuple(stalks)
    accession=NativeInputAccession(asset,forms.program(),declaration_);xx=action(upper,x);input_=forms.program().execute((accession.admit(111,tensor(xx)),));after=forms.apply(input_,node=1,channel=channel)
    expected=oracle(x,channel,policy)
    assert after.field_archive()==tensor(action(upper,expected[0])) and after.inspect()['quadrance']==before.inspect()['quadrance']
    assert after.field_archive('signed_boundary')==tensor(action(lower,expected[1]))
    assert after.field_archive('unsigned_boundary')==tensor(action(lower,expected[2]))
    for p,field in enumerate(expected[3]):assert after.occurrence_field_archive(p)==tensor(action(q,field) if LOWER[p]==0 else field)
    assert NativeComparisonFieldResult.from_archive(after.archive()).archive()==after.archive()


@pytest.mark.parametrize('suffix',['.rcbd','.safetensors'])
def test_fixed_comparison_provider_complete_archives_dependencies_and_cold_records(prepared,tmp_path,suffix):
    from rexgraph.cpp_relational_field_comparison import cpp_field_comparison_model,field_comparison_query,read_field_comparison_result,clear_field_comparison_cache
    from rexgraph.model_runtime import infer_model
    from rexgraph.io.model_record import model_record,save_model,load_model,read_model_record
    from rexgraph.model_codec import pack_model,unpack_model
    _,asset=prepared;owner=source();state=cpp_field_comparison_model(owner,learning_declaration(),comparison=comparison_declaration(),node=1,channel='C')
    row={'asset':asset.archive(),'accession':input_accession(),'use':111,'field':tensor((2,3,5,7))}
    query=field_comparison_query(state,(row,));output=infer_model(state,owner,query);result=read_field_comparison_result(output,state=state)
    assert state.adapter_version=='4' and output.values.size==0 and output.dependencies[0].native.namespace==999991
    assert result.field_archive()==tensor(oracle(tuple(map(Q,(2,3,5,7))),'C','share')[0])
    record=model_record(owner,state);record.attach_metadata(1,0,'response',output);path=tmp_path/('comparison'+suffix);save_model(path,record)
    clear_field_comparison_cache();loaded=load_model(path);restored,actual=read_model_record(loaded)
    assert restored.coefficient_digest==state.coefficient_digest
    assert infer_model(restored,actual,query).field_archive['sha256']==output.field_archive['sha256']
    standalone=unpack_model(pack_model(output,native=True));clear_field_comparison_cache()
    assert read_field_comparison_result(standalone).archive()==result.archive()
    with pytest.raises(ValueError):read_field_comparison_result(replace(output,dependencies=()))
    other=cpp_field_comparison_model(owner,learning_declaration(),comparison=comparison_declaration(),node=1,channel='F')
    with pytest.raises(ValueError):read_field_comparison_result(replace(output,model_digest=other.coefficient_digest),state=other)


@pytest.mark.parametrize('backend',['file','rex','sqlite'])
def test_comparison_rcql_system_checked_copies_and_agent_routes(prepared,tmp_path,backend):
    from rexgraph.cpp_relational_field_comparison import cpp_field_comparison_model,field_comparison_query,read_field_comparison_result
    from rexgraph.model_runtime import infer_model,train_model
    from rexgraph.model_state import ModelBatch
    from rexgraph.io.model_record import model_record,read_model_record
    from rcql import Executor,parse
    from rcdb import open_store,copy_record
    from system.serialize import json_value
    from agent.hive import Hive
    _,asset=prepared;owner=source();spec=learning_declaration();weights=comparison_declaration()
    state=cpp_field_comparison_model(owner,spec,comparison=weights,node=1,channel='F')
    row={'asset':asset.archive(),'accession':input_accession(),'use':111,'field':tensor((2,3,5,7))}
    query=field_comparison_query(state,(row,));output=infer_model(state,owner,query);record=model_record(owner,state)
    config={'source_backed':True,'comparison_fields':True,'declaration':spec,'comparison':weights,'node':1,'channel':'F'}
    values=Executor(sources={'r':record},params={'c':config,'q':query,'o':output}).execute(parse(
        'FROM $r RETURN MODEL_INIT("cpp_relational_field_program",$c),MODEL_INFER(MODEL_STATE(),$q),'
        'MODEL_STALK($o,[1,0],max_coordinates=2),MODEL_VALUES($o),MODEL_INFO(MODEL_STATE())')).values
    assert values[0].coefficient_digest==state.coefficient_digest and values[1].field_archive['sha256']==output.field_archive['sha256']
    assert len(values[2]['values'])==2 and values[3]['comparison']['channel']=='F' and values[4]['comparison_grade']==1
    assert json_value(state)['comparison_channel']=='F' and json_value(output)['retains_whole_fields']
    with pytest.raises(NotImplementedError):train_model(state,owner,ModelBatch(state.source,state.space,[0]*len(state.space.keys),[True]*len(state.space.keys)))
    stores=[open_store('sqlite:///'+str(tmp_path/(n+'.sqlite'))) if backend=='sqlite' else open_store(backend+'://'+str(tmp_path/n),read_only=False) for n in ('first','second')]
    try:
        stored=stores[0].put('comparison',record);copy_record(stores[0],stores[1],stored);restored,actual=read_model_record(stores[1].get('comparison'))
        assert infer_model(restored,actual,query).field_archive['sha256']==output.field_archive['sha256']
        hive=Hive();hive.add_worker('comparisons',lambda value:infer_model(restored,actual,value),capability='analyze')
        response=hive.dispatch_capability('analyze',query)['result'];assert read_field_comparison_result(response).archive()==read_field_comparison_result(output).archive()
    finally:
        for store in stores:store.close()
