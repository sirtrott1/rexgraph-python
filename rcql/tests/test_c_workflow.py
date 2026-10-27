"""Closed C constructors, owner identity, prospective plans and cold RCQL replay."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import inspect
from dataclasses import replace
import os
from pathlib import Path
import pytest

from rexgraph import RexGraph
from rexgraph.tensor_field import FieldSource
from rcql import CWorkflow,CInterface,Executor,Result,SourcePolicy,SourceRef,bind,query,source,call
from rcql.c_workflow import _pack,KINDS,primitive_ledger
from rcql.c_workflow_contracts import ARGUMENTS
from rcql.execution_trace import capture_methods
from rcql.operators import get_operator
from rcql.signatures import lookup

@pytest.fixture
def backend():return pytest.importorskip('rexgraph_cpp._c_workflow')

def metadata():
    return dict(producer_namespace=77,producer=88,revision=3,evidence_namespace=99,evidence_cutoff=12,purpose='exact synthetic C test')

def numerical(backend,dependencies=()):
    p=backend.Payload
    def row(kind,key,payload,deps=()):return dict(kind=kind,key=key,revision=1,dependencies=deps,payload=payload.take())
    records=(row('Coordinate',1,p().string('s').u64(0)),
             row('Domain',2,p().rational(-2).rational(2).flag(True).flag(True).vectors(()),(0,)),
             row('Expression',3,p().vector((0,1)).vector((0,1)),(1,)),
             row('Carrier',4,p().string('rows').u64(1).u64(0)),
             row('Carrier',5,p().string('columns').u64(1).u64(0)),
             row('UnivariateFamily',6,p(),(1,3,4,2)))
    return CWorkflow.create(metadata(),records,(5,2,1,2),dependencies=dependencies)

def binding(policy=None,version=1):
    r=RexGraph.from_cells([2,[[0,1]]])
    return bind('r',r,policy or SourcePolicy.allow('*'),source_ref=SourceRef('r',record_id='source',record_version=version))

def test_original_denominator_hole_and_native_type_dispatch(backend):
    w=numerical(backend)
    assert w.exports[0].evaluate('rank',1)['fields']['rank']==(1,)
    assert w.exports[1].evaluate('expression',1)['fields']['value']==1
    # Cancellation of x/x never deletes the original x=0 exclusion.
    for operation,index in (('rank',5),('expression',2)):
        with pytest.raises((ValueError,RuntimeError)):w.value(index).evaluate(operation,0)
    with pytest.raises(ValueError):w.value(0).evaluate('rank',1)
    with pytest.raises(ValueError):w.value(5).evaluate('arbitrary_kernel',1)
    assert w.declaration['records'][2]['payload']
    assert w.native().metadata==metadata()

def test_exact_payloads_and_bounded_admission(backend):
    p=backend.Payload
    q=Q(2**2000+1,7)
    raw=p().rational(q).take()
    assert raw and b'0.142857' not in raw
    for value in (True,0.1):
        with pytest.raises(TypeError):p().rational(value)
    with pytest.raises(ValueError):p().matrix(((1,),),(2,1))
    with pytest.raises((ValueError,IndexError)):numerical(backend).value(-1)
    w=numerical(backend)
    for limits,exact in (((len(w.native().to_bytes())-1,128,32,2000000,256,2048),None),((32<<20,1,32,2000000,256,2048),None),((32<<20,128,1,2000000,256,2048),None)):
        with pytest.raises((ValueError,RuntimeError)):
            CWorkflow.decode(w.native().to_bytes(),limits=limits)
    cancelled=backend.Cancellation();cancelled.cancel()
    with pytest.raises(RuntimeError):CWorkflow.decode(w.native().to_bytes(),cancel=cancelled)
    with pytest.raises((ValueError,RuntimeError)):w.value(2).inspect(0)

@pytest.mark.parametrize('change',('implementation','records','exports','metadata','native','limits','parent'))
def test_recomputed_outer_hash_never_certifies_changed_descriptors(backend,change):
    w=numerical(backend);d=w.declaration
    if change=='implementation':d['implementation']={**d['implementation'],'binding_digest':'foreign'}
    elif change=='records':
        rows=list(d['records']);rows[2]={**rows[2],'payload':backend.Payload().vector((2,)).vector((1,)).take()};d['records']=tuple(rows)
    elif change=='exports':d['exports']=(0,)
    elif change=='metadata':d['metadata']={**d['metadata'],'purpose':'unrelated prediction'}
    elif change=='native':d['archive']=d['archive'][:-1]+bytes((d['archive'][-1]^1,))
    elif change=='limits':d['limits']=()
    else:d['parent']=b'foreign revision'
    with pytest.raises((ValueError,RuntimeError)):
        CWorkflow.from_bytes(_pack('c_workflow',d)).native()

def test_actual_owners_shared_aliases_and_concurrent_cold_initialization(backend):
    w=numerical(backend);other=CWorkflow.from_bytes(w.data)
    assert not w.value(2).same_owner(other.value(2))
    assert not w.native().same_owner(other.native())
    pending=CWorkflow.from_bytes(w.data)
    with ThreadPoolExecutor(8) as pool:owners=tuple(pool.map(lambda _:pending.native(),range(16)))
    assert all(v is owners[0] for v in owners)
    raw=Result((w,w.value(2),w.value(2),other,other.value(2))).to_bytes()
    restored=Result.from_bytes(raw)
    a,x,y,b,z=restored.values
    assert x.workflow is y.workflow is a and z.workflow is b and a is not b
    assert a.native().same_owner(x.workflow.native()) and not a.native().same_owner(b.native())
    assert x.evaluate('expression',1)['fields']['value']==1
    assert restored.to_bytes()==raw

def test_explicit_source_version_policy_and_purpose_gate(backend):
    b=binding();ref=FieldSource(b.value,b.ref.record_id,b.ref.record_version)
    w=numerical(backend,(ref,));v=w.value(2)
    with pytest.raises(ValueError):v.inspect()
    with pytest.raises(ValueError):v.inspect(sources={'r':binding(version=2)})
    denied=bind('r',b.value,SourcePolicy.allow('identity'),source_ref=b.ref)
    with pytest.raises(PermissionError):v.inspect(sources={'r':denied})
    assert v.inspect(sources={'r':b})
    w.require_scope(metadata())
    for name,value in (('evidence_cutoff',11),('evidence_namespace',100),('purpose','prediction')):
        with pytest.raises(ValueError):w.require_scope({**metadata(),name:value})
    revised=w.modify(())
    assert revised.declaration['parent']==w.data and revised.dependencies[0].matches(ref)
    assert revised.declaration['metadata']['revision']==w.declaration['metadata']['revision']+1

def test_rcql_native_queries_and_explain_are_separate(backend,monkeypatch):
    b=binding();w=numerical(backend)
    ex=Executor(sources={'r':b.source})
    expression=query(source('r'),call('C_EVALUATE',w.value(2),'expression',1))
    value=ex.execute(expression).values[0]
    assert value['fields']['value']==1
    def forbidden(*args,**kwargs):raise AssertionError('EXPLAIN executed a native kernel')
    monkeypatch.setattr(CWorkflow,'native',forbidden)
    plan=ex.execute(replace(expression,explain=True))
    assert plan

@pytest.mark.parametrize('name',tuple(ARGUMENTS))
def test_each_c_registration_has_an_executable_contract(backend,name):
    w=numerical(backend);b=binding();v=w.value(2)
    cases={
      'C_WORKFLOW_READ':(w.native().to_bytes(),),
      'C_WORKFLOW_BUILD':(metadata(),w.declaration['records'],w.declaration['exports']),
      'C_WORKFLOW_REPLAY':(w,w.declaration['implementation']),
      'C_WORKFLOW_EXPORT':(w,), 'C_WORKFLOW_MODIFY':(w,()), 'C_VALUE':(w,2),
      'C_INSPECT':(v,), 'C_EVALUATE':(v,'expression',1), 'C_PRIMITIVES':(),
      'LAYER_C_VALUE':('expression',v),
    }
    args=cases.get(name)
    params=list(inspect.signature(get_operator(name).fn).parameters.values())[1:]
    assert lookup(name).arity==(sum(p.default is p.empty for p in params),len(params))
    if args is None:
        with pytest.raises(TypeError):
            with capture_methods(binding=b):get_operator(name).fn(b.value,'bad',v)
    else:
        with capture_methods(binding=b):assert get_operator(name).fn(b.value,*args) is not None
        assert Executor(sources={'r':b.source}).execute(query(source('r'),call(name,*args))).values

def archive_root():
    path=os.environ.get('REXGRAPH_C_WORKFLOW_FIXTURES')
    if not path:pytest.skip('set REXGRAPH_C_WORKFLOW_FIXTURES to explicitly generated native C fixtures')
    return Path(path)

@pytest.mark.parametrize('name',('characters','branch','periods','surface-factors','radical-normalization','univariate','jets-and-words','rational-family','holonomy','rotating','refinement','orbit'))
def test_original_native_c_workflow_cold_replay(backend,name):
    raw=(archive_root()/(name+'.bin')).read_bytes()
    w=CWorkflow.decode(raw)
    assert w.native().to_bytes()==raw
    observations=tuple(v.inspect() for v in w.exports)
    cold=Result.from_bytes(Result((w,*w.exports)).to_bytes()).values
    assert cold[0] is not w and all(v.workflow is cold[0] for v in cold[1:])
    assert tuple(v.inspect() for v in cold[1:])==observations
    for i in range(len(w.declaration['records'])):assert w.value(i).inspect()

def test_complete_layer_interface_runtime_revisions_and_foreign_owners(backend):
    from rcql import Layer,LayerProduct,LayerOperator
    w=CWorkflow.decode((archive_root()/'refinement.bin').read_bytes())
    rows=w.declaration['records'];proof=next(i for i,r in enumerate(rows) if KINDS[r['kind']-1]=='RefinementDiagram')
    participants=rows[proof]['dependencies'][:2]
    layers=tuple(Layer.c_value(name,w.value(index)) for name,index in zip(('coarse','fine'),participants,strict=True))
    interface=CInterface.declare('refinement',w.value(proof))
    p=LayerProduct.create(layers,(interface,))
    b=binding();bindings={'r':b}
    op=LayerOperator.compile(p,bindings,{})
    before=op.apply(bindings,{})
    assert before.constructor=='RefinementDiagram'
    q=w.modify(())
    newlayers=tuple(Layer.c_value(name,q.value(index)) for name,index in zip(('coarse','fine'),participants,strict=True))
    with pytest.raises(ValueError):p.modify((newlayers[0],))
    p2=p.modify(newlayers,(CInterface.declare('refinement',q.value(proof)),))
    assert p2.declaration['parent']==p.data
    op2=LayerOperator.compile(p2,bindings,{})
    assert op2.product.data!=op.product.data and op.apply(bindings,{}).inspect()==before.inspect()
    other=CWorkflow.from_bytes(w.data)
    with pytest.raises(ValueError):LayerProduct.create((layers[0],Layer.c_value('fine',other.value(participants[1]))),(interface,))
    with pytest.raises(ValueError):LayerProduct.create(layers,())
    ex=Executor(sources={k:v.source for k,v in bindings.items()})
    assert ex.execute(query(source('r'),call('LAYER_APPLY',op))).values[0].constructor=='RefinementDiagram'
    assert p.solve(sources=bindings).verify(bindings)


def test_complete_rational_family_stays_a_family(backend):
    from rcql import Layer,LayerProduct,LayerOperator
    w=CWorkflow.decode((archive_root()/'rational-family.bin').read_bytes())
    gauge=w.exports[2].inspect()
    assert gauge['shapes']['gauge']==(0,1) and gauge['fields']['gauge']==()
    rows=w.declaration['records'];i=next(i for i,r in enumerate(rows) if KINDS[r['kind']-1]=='LayerFamily')
    from rcql.c_workflow import _proof_participants
    participants=_proof_participants(w,i)
    layers=tuple(Layer.c_value(name,w.value(j)) for name,j in zip(('left','right'),participants,strict=True))
    p=LayerProduct.create(layers,(CInterface.declare('joint',w.value(i)),))
    value=LayerOperator.compile(p,{'r':binding()},{}).apply({'r':binding()},{})
    assert value.constructor=='LayerFamily' and value.inspect()['fields']['kernel.shape']==(2,1)
    assert value.evaluate('layer_at',1)['fields']['kernel']==((Q(1),),(Q(1),))
    with pytest.raises((ValueError,RuntimeError)):value.evaluate('layer_at',0)
    with pytest.raises(ValueError):LayerOperator.compile(p,{'r':binding()},{'choose':0})
    with pytest.raises(ValueError):p.solve(pins=(('left','a',1),))


def test_named_primitive_ledger_keeps_profiles_distinct():
    rows=primitive_ledger()
    assert [r['name'] for r in rows if r['profile']=='closed-C-declarations-v1']==list(KINDS)
    assert any(r['name']=='PreparedNativeEnsemble' and r['status']=='native-unregistered' for r in rows)
    assert len(set(KINDS))==66


def test_uninitialized_native_owners_refuse(backend):
    raw=backend.NativeArchive.__new__(backend.NativeArchive)
    for fn in (raw.to_bytes,raw.records,lambda:raw.exports,lambda:raw.metadata,lambda:raw.inspect(0),lambda:raw.evaluate(0,'rank')):
        with pytest.raises(RuntimeError):fn()


def test_prospective_program_data_and_cache_distinguish_revisions(backend):
    from rcql.program_codec import dumps,loads
    from rcql.query_cache import QueryCache
    from rcdb import open_store
    from contextlib import closing
    from rcql import parse
    w=numerical(backend)
    prospective=loads(dumps(w.value(2)))
    assert prospective.workflow._owner is None
    assert prospective.coefficient_digest==w.value(2).coefficient_digest
    b=binding();ex=Executor(sources={'r':b.source},params={'v':w.value(2)})
    expression=parse('FROM $r RETURN C_EVALUATE($v, "expression", 1)')
    with closing(open_store('memory://')) as store:
        cache=QueryCache(store)
        assert ex.execute_cached(expression,cache).values[0]['fields']['value']==1
        assert ex.execute_cached(expression,cache).values[0]['fields']['value']==1
        from rcql.c_workflow import LIMITS
        tight=CWorkflow.from_bytes(w.data,limits=(LIMITS[0],1,*LIMITS[2:]))
        ex.params['v']=tight.value(2)
        with pytest.raises((ValueError,RuntimeError)):ex.execute_cached(expression,cache)
        p=backend.Payload
        d=w.declaration['records'][2]
        revised=w.modify(((2,{**d,'revision':2,'payload':p().vector((2,)).vector((1,)).take()}),))
        ex.params['v']=revised.value(2)
        assert ex.execute_cached(expression,cache).values[0]['fields']['value']==2
        assert w.value(2).evaluate('expression',1)['fields']['value']==1


def test_cold_reader_uses_current_receiving_limits(backend):
    w=numerical(backend)
    saved=w.declaration
    saved['limits']=(1<<50,)*6
    saved['exact']=(1<<50,)*4
    from rcql.c_workflow import LIMITS
    limited=CWorkflow.from_bytes(_pack('c_workflow',saved),limits=(LIMITS[0],1,*LIMITS[2:]))
    with pytest.raises((ValueError,RuntimeError)):limited.native()
    restored=CWorkflow.from_bytes(w.data)
    assert restored.native().to_bytes()==w.native().to_bytes()


def test_cache_keeps_views_on_their_actual_input_owner(backend):
    from contextlib import closing
    from rcdb import open_store
    from rcql.query_cache import QueryCache
    from rcql import Program,ProgramStep,ProgramInput,OutputRef,parse
    w=numerical(backend);other=CWorkflow.from_bytes(w.data);b=binding()
    select=parse('FROM $r RETURN C_VALUE($w,2)')
    observe=parse('FROM $r RETURN C_INSPECT(C_VALUE($w,2))')
    with closing(open_store('memory://')) as store:
        cache=QueryCache(store)
        for owner in (w,other,w):
            ex=Executor(sources={'r':b.source},params={'w':owner})
            value=ex.execute_cached(select,cache)
            assert value.values[0].workflow is owner
            assert value.native_plan['cache']['reason']=='retained-native-input-owner'
            ex.execute_cached(observe,cache)
            assert ex.execute_cached(observe,cache).native_plan['cache']['hit']
        p=Program('owning_views',(
            ProgramStep('read',parse('FROM $r RETURN C_WORKFLOW_READ($archive)')),
            ProgramStep('view',parse('FROM $r RETURN C_VALUE($w,2)'),(('w',OutputRef('read')),))),
            (ProgramInput('archive'),), (('workflow',OutputRef('read')),('view',OutputRef('view'))))
        ex=Executor(sources={'r':b.source},params={'archive':w.native().to_bytes()})
        for _ in range(2):
            result=ex.execute_program(p,cache=cache)
            assert result.values[1].workflow is result.values[0]
            assert result.step('view').values[0].workflow is result.step('read').values[0]


def test_explicit_replay_retains_original_data_and_refuses_receipt_tampering(backend):
    original=numerical(backend)
    d=original.declaration
    d['implementation']={**d['implementation'],'engine_digest':'a'*64}
    saved=CWorkflow.from_bytes(_pack('c_workflow',d))
    with pytest.raises(ValueError):saved.native()
    receipt=d['implementation']
    current=saved.replay(receipt)
    assert current.declaration['parent']==saved.data
    assert current.declaration['archive']==d['archive']
    assert current.declaration['metadata']==d['metadata']
    assert current.declaration['implementation']==backend.build_info()
    assert not current.native().same_owner(original.native())
    assert current.value(2).evaluate('expression',1)['fields']['value']==1
    assert saved.declaration['implementation']==receipt
    with pytest.raises(ValueError):saved.replay({**receipt,'binding_digest':'0'*64})
    limited=CWorkflow.from_bytes(saved.data,limits=(1,128,32,2000000,256,2048))
    with pytest.raises((ValueError,RuntimeError)):limited.replay(receipt)
    cancelled=backend.Cancellation();cancelled.cancel()
    with pytest.raises(RuntimeError):saved.replay(receipt,cancel=cancelled)
    forged={**d,'metadata':{**d['metadata'],'purpose':'foreign purpose'}}
    with pytest.raises(ValueError):CWorkflow.from_bytes(_pack('c_workflow',forged)).replay(receipt)


def test_replay_requires_current_sources_and_explain_is_prospective(backend,monkeypatch):
    from dataclasses import replace
    b=binding();w=numerical(backend,(FieldSource(b.value,b.ref.record_id,b.ref.record_version),))
    with pytest.raises(ValueError):w.replay(w.declaration['implementation'])
    with pytest.raises(ValueError):w.replay(w.declaration['implementation'],sources={'r':binding(version=2)})
    assert w.replay(w.declaration['implementation'],sources={'r':b}).dependencies[0].matches(w.dependencies[0])
    ex=Executor(sources={'r':b.source})
    expression=query(source('r'),call('C_WORKFLOW_REPLAY',w,w.declaration['implementation']))
    def forbidden(*args,**kwargs):raise AssertionError('EXPLAIN executed replay')
    monkeypatch.setattr(CWorkflow,'replay',forbidden)
    assert ex.execute(replace(expression,explain=True))
