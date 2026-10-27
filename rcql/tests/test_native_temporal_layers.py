"""Independent exact examples through the compiled temporal executor."""
from fractions import Fraction as Q
import pytest
pytest.importorskip('rexgraph_temporal')
from rcql import TemporalModel,Layer,LayerProduct,LayerOperator,Executor,parse,SourcePolicy,bind
from rexgraph import RexGraph


def native_fixture():
    model=TemporalModel.create(0x5243514C).declare('coordinate',dict(name='t',clock='event',unit='seconds',origin='zero'))
    coordinate=model.selected
    model=model.declare('function',dict(coordinate=coordinate,pieces=({'numerator':(1,2,3)},),name='q'))
    function=model.selected
    model=model.declare('program',{'opcode':'input'});input_program=model.selected
    model=model.declare('program',{'opcode':'derivative','operands':(input_program,)})
    derivative=model.selected
    return model,coordinate,function,input_program,derivative


def test_whole_functions_exact_derivative_and_runtime_revision():
    model,c,f,x,dx=native_fixture()
    first=model.apply(dx,{1:f})
    assert first.value(2)==14 and first.info()['pieces'][0]['denominator']==(1,)
    revised=first.revise_program(dx,{'opcode':'multiply','operands':(x,x)})
    second=revised.apply(revised.selected,{1:f})
    assert second.value(2)==17**2
    assert first.value(2)==14 and model.select(f,'function').value(2)==17
    assert second.verify()
    assert TemporalModel.from_record(second.to_record()).value(Q(1,2))==Q(121,16)


def test_original_denominator_holes_are_not_filled():
    m,c,_,_,_=native_fixture()
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':(-1,1),'denominator':(-1,1)},)))
    assert m.value(1) is None and m.value(2)==1
    function=m.selected
    m=m.declare('program',{'opcode':'input'});x=m.selected
    m=m.declare('program',{'opcode':'derivative','operands':(x,)})
    result=m.apply(m.selected,{1:function})
    assert result.value(1) is None and result.value(2)==0


def test_function_layer_operator_changes_native_functions_at_runtime():
    m,c,f,x,dx=native_fixture()
    m=m.declare('support',{'coordinate':c,'components':((0,2,True,True),)});support=m.selected
    layers=(Layer.function('left',m,support,program=dx,role='convergent'),Layer.function('right',m,support,program=dx,role='divergent'))
    product=LayerProduct.create(layers)
    operation=LayerOperator.compile(product,{}, {})
    result=operation.apply({}, {1:f})
    assert result.value(2)==14
    revision=m.revise_program(dx,{'opcode':'multiply','operands':(x,x)});square=revision.selected
    replacement=tuple(Layer.function(v.key,revision,support,program=square,role=v.declaration['role']) for v in layers)
    revised=product.modify(replacement)
    second=LayerOperator.compile(revised,{},{}).apply({}, {1:f})
    assert second.value(2)==289 and result.value(2)==14
    assert revised.declaration['parent']==product.data


def test_incompatible_cover_and_exact_chart():
    m,c,f,x,dx=native_fixture()
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':(1,4,12)},)));other=m.selected
    m=m.declare('support',{'coordinate':c,'components':((0,1,True,True),)});support=m.selected
    product=LayerProduct.create((Layer.function('left',m,support,function=f),Layer.function('right',m,support,function=other,scale=Q(1,2))))
    # q_right(t/2) = q_left(t), using actual declared charts.
    assert product.solve().declaration['status']=='compatible'
    bad=product.modify((Layer.function('right',m,support,function=other),))
    assert bad.solve().declaration['status']=='inconsistent'


def test_rcql_native_execution_and_portable_model():
    m,c,f,x,dx=native_fixture()
    e=Executor(sources={'r':RexGraph.from_cells([1,[]])},params={'m':m,'f':f,'dx':dx})
    e.params['inputs']={1:f}
    result=e.execute(parse('FROM $r LET a=FUNCTION_APPLY($m,$dx,$inputs) RETURN a,FUNCTION_VALUE(a,2),FUNCTION_VERIFY(a)'))
    assert result.values[1:]==(14,True)
    from rcql.result_codec import pack_result,unpack_result
    reopened=unpack_result(pack_result(result))
    assert reopened.values[0].value(2)==14


def test_native_binding_rejects_inexact_input_and_missing_backend_during_decode(monkeypatch):
    m,c,f,x,dx=native_fixture()
    with pytest.raises(TypeError): m.value(0.5,f)
    from rcql.program_codec import dumps,loads
    raw=dumps(m)
    import rcql.native_temporal as module
    monkeypatch.setattr(module,'_backend',lambda:pytest.fail('kernel imported during decode'))
    assert loads(raw).data==m.data


def learnable(limits=()):
    m=TemporalModel.create(123,limits).declare('coordinate',dict(name='t',clock='event',unit='s',origin='0'));c=m.selected
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':(0,0,1)},)));f=m.selected
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':(0,2)},)));target=m.selected
    m=m.declare('program',{'opcode':'input'});x=m.selected
    m=m.declare('program',{'opcode':'derivative','operands':(x,)});dx=m.selected
    m=m.declare('support',{'coordinate':c,'components':((0,1,True,True),)});s=m.selected
    m=m.declare('context',{'support':s});context=m.selected
    m=m.declare('set',{'coordinate':c,'members':((101,1,f,s,True),)});members=m.selected
    return m,c,f,target,x,dx,s,context,members


def test_complete_native_learning_and_qualified_transfer():
    m,c,f,target,x,dx,s,context,members=learnable()
    fit=m.fit(members,context,(x,dx),target)
    info=fit.info()
    assert info['complete'] and info['forced'] and tuple(v['program'] for v in info['matches'])==(dx,)
    assert fit.apply(0,{1:f},fit.selected).value(Q(1,3))==Q(2,3)
    restored=TemporalModel.from_record(fit.to_record())
    assert restored.info()==info and restored.verify()


def test_incomplete_learning_is_not_a_unique_prediction():
    # One candidate application cannot qualify a two-candidate search.
    limits=(1,128,1024,100000,100000,4096,1000000,10000000,1000000)
    m,c,f,target,x,dx,s,context,members=learnable(limits)
    fit=m.fit(members,context,(x,dx),target)
    assert not fit.info()['complete'] and not fit.info()['forced']
    with pytest.raises(ValueError): fit.apply(0,{1:f},fit.selected)


def test_correlated_native_families_preserve_all_global_witnesses():
    m,c,f,target,x,dx,s,context,members=learnable()
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':()},)));zero=m.selected
    m=m.declare('set',{'coordinate':c,'members':((102,1,zero,s,True),)});members=m.selected
    m=m.fit(members,context,(x,dx),zero);a=m.selected
    m=m.declare('context',{'support':s});other=m.selected
    m=m.fit(members,other,(x,dx),zero);b=m.selected
    transition={'left':context,'right':other,'same_program':True}
    product=LayerProduct.create((Layer.family('past',m,a),Layer.family('future',m,b)),transitions=(transition,))
    solution=product.solve();native=solution.native();info=native.info()
    assert info['complete'] and info['compatible']
    joint=native.select(info['output'],'joint').info()
    assert not joint['forced'] and len(set(joint['witnesses']))==2
    # Explicitly empty relation is different from an omitted restriction.
    empty=LayerProduct.create(product.layers,transitions=({**transition,'allowed_pairs':()},))
    assert empty.solve().declaration['status']=='inconsistent'
    assert solution.verify()


def test_rcql_fit_requires_current_training_permission():
    m,c,f,target,x,dx,s,context,members=learnable()
    params=dict(m=m,members=members,context=context,candidates=(x,dx),target=target)
    r=RexGraph.from_cells([1,[]])
    text=parse('FROM $r RETURN FUNCTION_FIT($m,$members,$context,$candidates,$target)')
    with pytest.raises(PermissionError):
        Executor(sources={'r':bind('r',r,SourcePolicy.allow('read')).source},params=params).execute(text)
    result=Executor(sources={'r':bind('r',r,SourcePolicy.allow('read','train')).source},params=params).execute(text)
    assert result.values[0].info()['forced']


def test_native_source_creation_and_operator_composition_in_one_rcql_query():
    m,c,f,x,dx=native_fixture()
    m=m.declare('support',{'coordinate':c,'components':((0,2,True,True),)});s=m.selected
    r=RexGraph.from_cells([1,[]])
    params=dict(m=m,s=s,dx=dx,inputs={1:f})
    e=Executor(sources={'r':r},params=params)
    result=e.execute(parse('FROM $r LET a=LAYER_FUNCTION("a",$m,$s,program=$dx) LET b=LAYER_FUNCTION("b",$m,$s,program=$dx) LET p=LAYER_PRODUCT([a,b]) LET op=LAYER_COMPILE(p) LET f=LAYER_APPLY(op,parameters=$inputs) RETURN FUNCTION_VALUE(f,2),op'))
    assert result.values[0]==14
    assert result.values[1].product.profile=='functions'


def test_explicit_native_function_embedding_preserves_cross_moments():
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.coordinate_map import CoordinateMap
    from rexgraph.tensor_moment import CoordinatePairing
    m,c,f,x,dx=native_fixture()
    r=RexGraph.from_cells([2,[[0,1]]])
    space=NativeFieldCalculus.from_rex(r).complex.spaces[1]
    injection=CoordinateMap(CoordinateSpace('amplitude',('q',)),space,((0,0,1),))
    powers=CoordinateSpace('polynomial powers',('0','1','2'))
    pairing=CoordinatePairing(space,space,((0,0,1),))
    params=dict(m=m.select(f,'function'),injection=injection,powers=powers,pairing=pairing)
    e=Executor(sources={'r':r},params=params)
    result=e.execute(parse('FROM $r LET field=FUNCTION_FIELD($m,$injection,1,$powers) RETURN field,MOMENT_CONTRACT(FIELD_PAIR(field,field,$pairing)),NATIVE_RESPONSE(field),$m'))
    assert result.values[0].values.tolist()==[[1,2,3]]
    assert result.values[1].values.tolist()==[[[1,2,3],[2,4,6],[3,6,9]]]
    assert len(result.values[2].axes)==1
    assert m.select(f,'function').coefficient_digest in result.values[0].provenance
    from rcql.result_codec import pack_result,unpack_result
    assert unpack_result(pack_result(result)).values[1].values.tolist()==result.values[1].values.tolist()


def test_embedding_refuses_filling_original_rational_holes():
    from rexgraph.native_field import NativeFieldCalculus
    from rexgraph.type_accession import CoordinateSpace
    from rexgraph.coordinate_map import CoordinateMap
    m,c,f,x,dx=native_fixture()
    m=m.declare('function',dict(coordinate=c,pieces=({'numerator':(-1,1),'denominator':(-1,1)},)))
    r=RexGraph.from_cells([2,[[0,1]]]);space=NativeFieldCalculus.from_rex(r).complex.spaces[1]
    injection=CoordinateMap(CoordinateSpace('amplitude',('q',)),space,((0,0,1),))
    params=dict(m=m,injection=injection,powers=CoordinateSpace('powers',('0','1')))
    with pytest.raises(ValueError,match='denominator exclusions'):
        Executor(sources={'r':r},params=params).execute(parse('FROM $r RETURN FUNCTION_FIELD($m,$injection,1,$powers)'))


def test_uninitialized_native_source_is_refused():
    from rexgraph_temporal import NativeTemporal
    source=NativeTemporal.__new__(NativeTemporal)
    with pytest.raises(RuntimeError): source.objects()
    with pytest.raises(RuntimeError): source.verify()


def test_complete_archive_interoperability_with_canonical_relational_runtime():
    module=pytest.importorskip('rexgraph.relational_runtime')
    pytest.importorskip('rexgraph_cpp._temporal_program')
    Runtime=module.RelationalRuntime
    r=Runtime.create(0xB01).extend('coordinate',dict(name='t',clock='t-clock',unit='s',origin='0'))
    c=r.selected
    r=r.extend('function',dict(coordinate=c,name='q',provenance='declared',pieces=[dict(identity=1,
        domain=(None,None,False,False),numerator=(1,2,3),denominator=(1,),provenance='q')]))
    m=TemporalModel.from_relational(r,kind='function')
    assert m.value(2)==17
    original=r.to_bytes()
    m=m.declare('program',{'opcode':'input'});x=m.selected
    m=m.declare('program',{'opcode':'derivative','operands':(x,)})
    m=m.apply(m.selected,{1:r.selected})
    exported=m.to_relational()
    assert exported.evaluate(2)==14 and exported.function()['coordinate']==c
    assert r.to_bytes()==original and r.evaluate(2)==17
    assert TemporalModel.from_relational(exported,kind='function').value(2)==14
    from rexgraph.tensor_field import FieldSource
    bound=m.with_dependencies((FieldSource(RexGraph.from_cells([1,[]])),))
    with pytest.raises(ValueError,match='discard dependencies'): bound.to_relational()


def test_temporal_support_and_probability_replay():
    m,c,_,_,_=native_fixture()
    m=m.declare('support',dict(coordinate=c,components=((0,2,True,True),)))
    first=m.selected
    m=m.declare('support',dict(coordinate=c,components=((1,3,True,True),)))
    second=m.selected
    common=m.declare('support_intersection',dict(inputs=(first,second)))
    assert common.info()==((1,2,True,True),)
    law=common.declare('law',dict(inputs=(first,),rationals=(Q(1),)))
    assert law.kind=='law' and law.verify()
    restored=TemporalModel.from_record(law.to_record())
    assert restored.verify()


def test_public_native_archive_receiving_retains_complete_source_and_dependencies():
    from rexgraph.tensor_field import FieldSource
    from rcql.layer_product import _execution_sources
    model,c,function,x,dx=native_fixture()
    evidence=RexGraph.from_cells([1,[]])
    dependency=FieldSource(evidence,'evidence',1)
    declaration=model.declaration
    restored=TemporalModel.from_archive(declaration['source'],declaration['endpoint'],
        selected=function,kind='function',dependencies=(dependency,))
    assert restored.value(2)==17 and restored.verify()
    assert restored.declaration['source']==declaration['source']
    assert restored.declaration['endpoint']==declaration['endpoint']
    assert restored.select(dx,'program').info()==model.select(dx,'program').info()
    assert restored.dependencies[0].matches(dependency)
    assert TemporalModel.from_record(restored.to_record()).data==restored.data
    with pytest.raises(ValueError,match='exact current source version'):
        _execution_sources(restored,{})
    from rcql import SourceRef
    binding=bind('r',evidence,SourcePolicy.allow('read'),
        source_ref=SourceRef('r',record_id='evidence',record_version=1))
    assert _execution_sources(restored,{'r':binding})['r'].ref==binding.ref


def test_public_native_archive_receiving_rejects_corruption_and_false_selection():
    model,c,function,x,dx=native_fixture();d=model.declaration
    def receive(source=d['source'],endpoint=d['endpoint'],selected=function,kind='function'):
        return TemporalModel.from_archive(source,endpoint,selected=selected,kind=kind)
    for source,endpoint in ((d['source'][:-1],d['endpoint']),(d['source'],d['endpoint'][:-1])):
        with pytest.raises((ValueError,RuntimeError)):receive(source,endpoint)
    with pytest.raises(TypeError):receive(bytearray(d['source']))
    for selected,kind in ((function,'program'),(2**64-1,'function'),(True,'function'),
                          (0,'function'),(function,'source'),(function,'unknown')):
        with pytest.raises(ValueError):receive(selected=selected,kind=kind)
    source=receive(selected=0,kind='source')
    assert source.verify() and source.declaration['source']==d['source']


def test_explicit_function_replay_retains_original_producer_and_source():
    from rcql.native_temporal import _frame
    model,_,function,_,_=native_fixture()
    selected=model.select(function,'function')
    d=selected.declaration;d['build']={**d['build'],'temporal_digest':'a'*64}
    historical=TemporalModel(_frame(d));limits=(65536,128,256,1000000,1000000,4096,16000000,64000000,4096)
    with pytest.raises(ValueError):historical.verify()
    replay=historical.replay(d['build'],limits)
    assert replay.declaration['parent']==historical.coefficient_digest
    assert replay.declaration['source']==d['source'] and replay.declaration['endpoint']==d['endpoint']
    assert replay.value(2)==17 and replay.verify()
    assert historical.declaration==d
    with pytest.raises(ValueError):historical.replay({**d['build'],'rcql_wrapper':'0'*64},limits)
    with pytest.raises(ValueError):historical.replay(d['build'],())


def test_function_replay_requires_current_dependencies_and_explain_is_prospective(monkeypatch):
    from dataclasses import replace
    from rexgraph.tensor_field import FieldSource
    from rcql import query,source,call,SourceRef
    model,*_=native_fixture();g=RexGraph.from_cells([1,[]])
    b=bind('r',g,SourcePolicy.allow('*'),source_ref=SourceRef('r',record_id='r',record_version=1))
    model=model.with_dependencies((FieldSource(g,'r',1),));limits=(65536,128,256,1000000,1000000,4096,16000000,64000000,4096)
    with pytest.raises(ValueError):model.replay(model.declaration['build'],limits)
    replay=model.replay(model.declaration['build'],limits,{'r':b})
    assert replay.dependencies[0].matches(model.dependencies[0])
    expr=query(source('r'),call('FUNCTION_REPLAY',model,model.declaration['build'],limits))
    def forbidden(*args,**kwargs):raise AssertionError('EXPLAIN ran function replay')
    monkeypatch.setattr(TemporalModel,'replay',forbidden)
    assert Executor(sources={'r':b.source}).execute(replace(expr,explain=True))
