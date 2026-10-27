"""Retained interfaces, exact families and first-class runtime revisions."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
from rexgraph import RexGraph
from rexgraph.type_accession import CoordinateSpace
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.section_calculus import UnderdeterminedSectionError
from rexgraph.tensor_field import FieldSource
from rcql import (Layer, LayerProduct, LayerInterface, LayerSolution, LayerOperator, LayerReadout,
                  Program, ProgramInput, ProgramStep, Executor, SourcePolicy, SourceRef, bind, parse)


def diagram(programs=False, dependencies=()):
    space=CoordinateSpace('local parameters',('a','latent'))
    common=CoordinateSpace('overlap',('shared',))
    action=CoordinateMap(space,common,((0,0,1),))
    first=Program('first',(ProgramStep('step',parse('FROM $r RETURN SUM([$x,$a])')),),
                  (ProgramInput('x','ExactRational'),ProgramInput('a','ExactRational')))
    second=replace(first,name='second')
    layers=tuple(Layer.affine(key,space,role=role,program=program if programs else None,
        coefficients=(('a','a'),) if programs else (),inputs=(('x','x'),) if programs and key=='left' else ())
        for key,role,program in (('left','convergent',first),('right','divergent',second)))
    interface=LayerInterface.equality('agreement',tuple((v.key,action) for v in layers))
    return LayerProduct.create(layers,(interface,),dependencies=dependencies,
        links=(('right','x','left','0'),) if programs else (),outputs=(('result','right','0'),) if programs else ())


def bindings(policy=None):
    r=RexGraph.from_cells([2,[[0,1]]])
    return {'r':bind('r',r,policy or SourcePolicy.allow('*'))}


def read_a(solution, layer='right'):
    system=solution.product.system()
    selection=system.selection(layer)
    target=CoordinateSpace('prediction',('a',))
    return LayerReadout.create(solution,CoordinateMap(system.space,target,
        tuple((0,j,v) for i,j,v in selection.entries if i==0)))


def test_full_affine_family_and_invariant_prediction():
    product=diagram();free=product.solve()
    assert free.family().dimension==3
    with pytest.raises(UnderdeterminedSectionError): read_a(free).value()
    solution=product.solve((('left','a',Q(2,7)),))
    assert solution.verify() and solution.family().dimension==2
    assert read_a(solution).value().values.tolist()==[Q(2,7)]
    assert tuple(v.declaration['role'] for v in product.layers)==('convergent','divergent')


def test_duplicate_observations_are_retained_and_contradiction_certified():
    product=diagram()
    good=product.solve((('left','a',2),('left','a',2)))
    assert good.family().dimension==2
    bad=product.solve((('left','a',2),('left','a',3)))
    assert bad.declaration['status']=='inconsistent' and bad.verify()
    assert bad.declaration['certificate']['witness']
    with pytest.raises(ValueError): bad.family()


def test_explicit_maps_and_original_carriers():
    product=diagram();d=product.layers[0].declaration
    foreign=Layer.affine('left',CoordinateSpace('different',('a','latent')))
    with pytest.raises(ValueError,match='domain'): product.modify((foreign,))
    with pytest.raises(ValueError): LayerProduct.create((product.layers[0],product.layers[0]))
    assert d['space']==('local parameters',('a','latent'))


def test_runtime_operator_revision_preserves_prior_version():
    sources=bindings();product=diagram(True)
    solution=product.solve((('left','a',Q(2,7)),))
    operation=solution.compile(sources,{'x':Q(3,7)})
    assert operation.apply(sources,{'x':Q(3,7)})==(Q(1),)
    first=product.layers[0];space=CoordinateSpace(*first.declaration['space'])
    multiply=Program('double',(ProgramStep('step',parse('FROM $r RETURN SUM([$x,$a,$a])')),),
        (ProgramInput('x','ExactRational'),ProgramInput('a','ExactRational')))
    revised=product.modify((Layer.affine('left',space,program=multiply,coefficients=(('a','a'),),inputs=(('x','x'),)),))
    assert revised.declaration['parent']==product.data
    new=revised.solve((('left','a',Q(2,7)),)).compile(sources,{'x':Q(3,7)})
    assert new.apply(sources,{'x':Q(3,7)})==(Q(9,7),)
    assert operation.apply(sources,{'x':Q(3,7)})==(Q(1),)
    with pytest.raises(UnderdeterminedSectionError): product.solve().compile(sources,{'x':Q(1)})


def test_affine_layer_execution_retains_portable_program_history():
    from rcql import Result, query, source, call
    sources=bindings()
    operation=diagram(True).solve((('left','a',Q(2,7)),)).compile(sources,{'x':Q(3,7)})
    result=Executor(sources={k:v.source for k,v in sources.items()}).execute(
        query(source('r'),call('LAYER_APPLY',operation,sources,{'x':Q(3,7)})))
    execution=result.values[0]
    assert execution.values==operation.apply(sources,{'x':Q(3,7)})==(Q(1),)
    assert len(execution.steps)==2 and execution.sources
    assert tuple(step.values for _,step in execution.steps)==((Q(5,7),),(Q(1),))
    raw=result.to_bytes()
    restored=Result.from_bytes(raw)
    assert restored.values[0].steps==execution.steps
    assert restored.to_bytes()==raw


def test_current_source_version_and_policy_are_rechecked():
    sources=bindings();ref=FieldSource(sources['r'].value,'evidence',1)
    product=diagram(True,(ref,))
    sources={'r':bind('r',sources['r'].value,SourcePolicy.allow('*'),source_ref=SourceRef('r',record_id='evidence',record_version=1))}
    solution=product.solve((('left','a',1),),sources=sources)
    operation=solution.compile(sources,{'x':Q(1)})
    assert operation.apply(sources,{'x':Q(1)})==(3,)
    with pytest.raises(ValueError,match='exact current source version'): operation.apply(bindings(),{'x':Q(1)})
    narrow={'r':bind('r',sources['r'].value,SourcePolicy.allow('read'),source_ref=sources['r'].ref)}
    # SUM is a read calculation; a narrower policy remains sufficient.
    assert operation.apply(narrow,{'x':Q(1)})==(3,)
    denied={'r':bind('r',sources['r'].value,SourcePolicy.allow(),source_ref=sources['r'].ref)}
    with pytest.raises(PermissionError): operation.apply(denied,{'x':Q(1)})


def test_rcql_constructs_solves_revises_and_persists():
    product=diagram();e=Executor(sources={'r':RexGraph.from_cells([1,[]])},params={'layers':product.layers,
        'interfaces':tuple(LayerInterface(v) for v in product.declaration['interfaces'])})
    result=e.execute(parse('FROM $r LET p=LAYER_PRODUCT($layers,$interfaces) LET s=LAYER_SOLVE(p,[["left","a",2]]) RETURN p,s,LAYER_INFO(s)'))
    assert result.values[2]['dimension']==2
    from rcql.result_codec import pack_result,unpack_result
    reopened=unpack_result(pack_result(result))
    assert reopened.values[0].data==result.values[0].data
    assert reopened.values[1].data==result.values[1].data
    assert LayerProduct.from_record(result.values[0].to_record()).data==result.values[0].data


def test_programmatic_declarations_can_be_captured_and_modified_by_programs():
    product=diagram()
    program=Program('revise',(ProgramStep('step',parse('FROM $r RETURN LAYER_MODIFY($product,[])')),),
        (ProgramInput('product','LayerProduct'),)).specialize({'product':product})
    restored=Program.from_bytes(program.to_bytes())
    output=restored.execute(Executor(sources={'r':RexGraph.from_cells([1,[]])})).values[0]
    assert output.declaration['parent']==product.data
    assert product.declaration['parent'] is None


def test_explain_does_not_solve(monkeypatch):
    product=diagram();e=Executor(sources={'r':RexGraph.from_cells([1,[]])},params={'p':product})
    monkeypatch.setattr(LayerSolution,'compute',lambda *a,**k:pytest.fail('solve during explain'))
    e.execute(parse('EXPLAIN FROM $r RETURN LAYER_SOLVE($p)'))


@pytest.mark.parametrize('kind',[LayerProduct,LayerSolution,LayerOperator,LayerReadout])
def test_declaration_tampering_is_rejected(kind):
    product=diagram(True);solution=product.solve((('left','a',1),))
    values={LayerProduct:product,LayerSolution:solution,LayerOperator:solution.compile(bindings(),{'x':Q(1)}),LayerReadout:read_a(solution)}
    raw=values[kind].data
    with pytest.raises(ValueError): kind(raw[:-1]+bytes([raw[-1]^1]))


@pytest.mark.parametrize('backend',['rex','local'])
def test_cold_storage_replay_preserves_evidence_versions(tmp_path,backend):
    from rcdb import open_store
    from rcql.source_context import SnapshotContext,SourceSelection
    uri=str(tmp_path/'layers.rexdb') if backend=='rex' else 'local://'+str(tmp_path/'layers')
    store=open_store(uri,**({"read_only":False} if backend=="rex" else {}))
    root=RexGraph.from_cells([2,[[0,1]]])
    store.commit_mutation('evidence',root,expected_version=0,tx_time=10,analytics=False)
    snapshot=store.read_record('evidence')
    product=diagram(dependencies=(FieldSource(root,'evidence',1,snapshot.state_digest),))
    store.commit_mutation('diagram',product.to_record(),expected_version=0,tx_time=20,analytics=False)
    store.close()
    store=open_store(uri,**({"read_only":False} if backend=="rex" else {}))
    try:
        ctx=SnapshotContext.select(store,(SourceSelection('r','evidence'),SourceSelection('d','diagram')),cutoff=25)
        saved=LayerProduct.from_record(ctx.sources['d'].value)
        result=Executor(sources={'r':ctx.sources['r']},evidence=ctx,params={'p':saved}).execute(parse('FROM $r RETURN LAYER_INFO(LAYER_SOLVE($p,[["left","a",7]]))'))
        assert result.values[0]['dimension']==2
        later=RexGraph.from_cells([2,[[0,1]]]);later.attach_metadata(0,0,'later','revision')
        store.commit_mutation('evidence',later,expected_version=1,tx_time=30,analytics=False)
        future=SnapshotContext.select(store,(SourceSelection('r','evidence'),),cutoff=35)
        with pytest.raises(ValueError):
            Executor(sources=future.sources,evidence=future,params={'p':saved}).execute(parse('FROM $r RETURN LAYER_SOLVE($p)'))
        # The old pinned source remains valid even after the new commit.
        assert ctx.sources['r'].ref.record_version==1
    finally:
        store.close()


def test_recomputed_frame_does_not_authorize_a_false_solution():
    from rcql.layer_product import _pack
    solution=diagram().solve((('left','a',1),))
    altered=LayerSolution(_pack('layer_solution',{**solution.declaration,'pins':(('left','a',2),)}))
    with pytest.raises(ValueError,match='replay'): altered.verify()


def test_layer_products_work_as_named_runtime_modifiers():
    from rcql import NameRelation
    product=diagram();r=RexGraph.from_cells([1,[]]);binding=bind('r',r,SourcePolicy.allow('*'))
    modifier=NameRelation.operator('LAYER_MODIFY').bind('replacements',()).bind('interfaces',None)
    restored=NameRelation.from_bytes(modifier.to_bytes())
    revised=restored.apply(binding,{'product':product})
    assert revised.declaration['parent']==product.data
    assert product.declaration['parent'] is None


def test_recursive_runtime_can_revise_layer_products_and_reopen_history():
    from rcql import RecursiveDefinition,RecursiveProgram,RecursionResult,recur
    from rcql.ast import Parameter,Literal,Comparison,Call,ListExpr
    n,p=Parameter('n'),Parameter('p')
    revised=Call('LAYER_MODIFY',(p,ListExpr(())))
    decrease=Call('SUM',(ListExpr((n,Literal(-1))),))
    definition=RecursiveDefinition('revise',(ProgramInput('n','ExactInteger'),ProgramInput('p','LayerProduct')),
        Comparison('==',n,Literal(0)),p,recur('revise',decrease,revised),ProgramInput('result','LayerProduct'),
        result_like='p',decreases='n')
    program=RecursiveProgram('layer_revisions',(definition,))
    r=RexGraph.from_cells([1,[]]);product=diagram()
    result=program.execute(bind('r',r,SourcePolicy.allow('*')),'revise',[3,product])
    assert result.calls==4
    current=result.value
    for _ in range(3): current=LayerProduct(current.declaration['parent'])
    assert current.data==product.data
    reopened=RecursionResult.from_record(result.to_record(),result.dependencies)
    assert reopened.coefficient_digest==result.coefficient_digest


def test_query_cache_retains_layer_declarations_and_checks_revisions():
    from rcql.query_cache import QueryCache
    from rcdb import open_store
    from contextlib import closing
    with closing(open_store('memory://')) as store:
        product=diagram();r=RexGraph.from_cells([1,[]]);cache=QueryCache(store)
        e=Executor(sources={'r':r},params={'p':product})
        query=parse('FROM $r RETURN LAYER_MODIFY($p,[])')
        first=e.execute_cached(query,cache);second=e.execute_cached(query,cache)
        assert first.values[0].data==second.values[0].data
        assert second.native_plan['cache']['hit']
        e.params['p']=product.modify()
        third=e.execute_cached(query,cache)
        assert third.values[0].data!=first.values[0].data
        assert not third.native_plan['cache']['hit']
