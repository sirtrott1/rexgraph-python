"""Original graded stalk fields, strict chain families and retained channels."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from rexgraph_cpp import NativeSource, PreparedStalkBoundary

AXES = ((('literal','register'),('literal','register','role'),('literal',)),
        (('literal','context'),('literal',),('literal','context')),
        (('literal','unused'),))
IDENTITY = ((0,0,1),(1,1,1))
MAPS = ((),(IDENTITY,IDENTITY,IDENTITY,IDENTITY,IDENTITY,((0,0,1),),
            ((0,0,1),),((0,0,1),),IDENTITY,((0,0,1),)),
        (((0,0,1),),((0,0,1),),((0,0,1),),((0,0,1),)))


def declaration(namespace=941115):
    columns = (((0,-1,True),(1,1,True),(0,Q(1,2),True),(0,Q(-1,2),True),(1,9,False),(2,0,True)),
               ((1,-1,True),(2,1,True)),((0,-1,True),(2,1,True)),
               ((0,1,True),(1,1,True),(2,-1,True),(1,0,True)))
    relations = []; key = 0
    for i,column in enumerate(columns):
        occurrences = []
        for lo,coefficient,exists in column:
            key += 1
            occurrences.append(dict(key=key,lower=10*(lo+1) if i<3 else lo+1,negative=coefficient<0,magnitude=abs(coefficient),
                                    exists=exists,aspects=(0,)))
        relations.append(dict(grade=1 if i<3 else 2,key=i+1,type=0,occurrences=tuple(occurrences)))
    return dict(profile='declared-role-source-v1',namespace=namespace,semantic_scale=2,max_grade=2,
        role_aspects=('retained-boundary',),types=(dict(name='declared-column',min_arity=0,max_arity=0),),
        witnesses=(10,20,30),relations=tuple(relations),fields=(),metrics=(),
        carriers=tuple(dict(key=10+k,kind='source_grade',grade=k,coordinates=0,size=n) for k,n in enumerate((3,3,1))))


def owner(**kwargs):
    source = NativeSource.from_declaration(declaration())
    return source,PreparedStalkBoundary(source.archive,AXES,MAPS,**kwargs)


def test_higher_boundary_retains_nonzero_channels_whose_exact_sum_is_zero():
    _,plan = owner(); trace = plan.apply(2,((1,7),),boundary_steps=2)
    assert trace['grades']==(2,1,0)
    assert trace['fields']==(((1,7),),((1,0),(1,),(-1,0)),((0,0),(0,0,0),(0,)))
    assert trace['intermediate_channels'][0]==(((-1,0),(1,0,0),(0,)),
        ((0,0),(-1,0,0),(1,)),((1,0),(0,0,0),(-1,)))
    assert trace['intermediate_channels'][1]==()
    assert len(trace['occurrence_fields'][0])==4 and len(trace['occurrence_fields'][1])==10
    assert trace['occurrence_fields'][1][2:6]==((Q(1,2),0),(Q(-1,2),0),(0,0,0),(0,))
    assert plan.receipt['strict_chain_family'] and plan.receipt['dimensions']==(6,5,2)
    assert plan.receipt['occurrence_activity'][1][4] is False
    assert plan.receipt['occurrence_coefficients'][1][4:6]==(0,0)
    assert plan.receipt['occurrence_keys']==((),tuple(range(1,11)),tuple(range(11,15)))


def test_equal_object_boundaries_do_not_identify_whole_relation_fields():
    _,plan = owner()
    a = plan.apply(1,((1,0),(1,),(0,0)))
    b = plan.apply(1,((0,0),(0,), (1,0)))
    assert a['fields'][-1]==b['fields'][-1]
    assert a['fields'][0]!=b['fields'][0] and a['occurrence_fields']!=b['occurrence_fields']


@pytest.mark.parametrize('changed', ['literal','unused'])
def test_full_chain_validation_rejects_bad_maps_even_on_unqueried_directions(changed):
    source,_ = owner(); maps = deepcopy(MAPS); family = list(maps[2])
    family[0] = ((0,0,2),) if changed=='literal' else ((0,0,1),(1,1,1))
    with pytest.raises(ValueError,match='complete strict chain'):
        PreparedStalkBoundary(source.archive,AXES,(maps[0],maps[1],tuple(family)))


def test_object_fields_and_identity_traces_retain_their_actual_grade():
    _,plan = owner(); field=((1,-1),(2,3,4),(5,))
    trace=plan.apply(0,field,boundary_steps=0)
    assert trace['grades']==(0,) and trace['fields']==(field,)
    assert trace['occurrence_fields']==trace['intermediate_channels']==()
    with pytest.raises(ValueError):plan.apply(0,field)


@pytest.mark.parametrize('value',[True,0.5,'1/2'])
def test_exact_original_stalk_coefficients_refuse_implicit_conversions(value):
    _,plan=owner()
    with pytest.raises(TypeError):plan.apply(2,((value,0),))
    source,_=owner(); family=list(MAPS[2]);family[0]=((0,0,value),)
    with pytest.raises(TypeError):PreparedStalkBoundary(source.archive,AXES,(*MAPS[:2],tuple(family)))


@pytest.mark.parametrize('change',['grades','flags','dimensions','duplicate','inactive'])
def test_complete_original_flags_and_ragged_local_directions_are_required(change):
    source,_=owner(); axes=AXES;maps=MAPS
    if change=='grades':axes=AXES[:-1];maps=MAPS[:-1]
    elif change=='flags':maps=(*MAPS[:2],MAPS[2][:-1])
    elif change=='dimensions':axes=(AXES[0],AXES[1],(('literal','literal'),))
    else:
        family=list(MAPS[1]);family[4]=((0,0,0),(0,0,0)) if change=='duplicate' else ((3,0,1),)
        maps=((),tuple(family),MAPS[2])
    with pytest.raises(ValueError):PreparedStalkBoundary(source.archive,axes,maps)


def test_native_acquisition_injects_into_the_declared_higher_stalk():
    from test_relation_response import declared
    _,plan=owner()
    acquired=NativeSource.from_declaration(declared((((0,0,2,True),(1,1,-1,True)),)))
    result=plan.inject_acquisition(2,acquired.archive,(0,1),(3,))
    assert result['field']==((6,-3),) and result['boundary']==(6,-3)
    assert result['grade']==2 and result['columns']==1 and result['retained_occurrences']==2
    assert plan.apply(2,result['field'],boundary_steps=2)['fields'][-1]==((0,0),(0,0,0),(0,))
    with pytest.raises(ValueError):plan.inject_acquisition(2,acquired.archive,(0,0),(3,))


def test_owner_refusal_and_concurrent_traces_preserve_all_original_values():
    source,plan=owner(); del source
    expected=plan.apply(2,((Q(2,3),9),),boundary_steps=2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(value==expected for value in pool.map(lambda _:plan.apply(2,((Q(2,3),9),),boundary_steps=2),range(9)))
    with pytest.raises(ValueError):plan.apply(2,((1<<16385,0),))
    assert plan.apply(2,((Q(2,3),9),),boundary_steps=2)==expected
    with pytest.raises(RuntimeError,match='immutable'):plan.__init__(b'',(),())
    empty=PreparedStalkBoundary.__new__(PreparedStalkBoundary)
    with pytest.raises(RuntimeError,match='not initialized'):empty.apply(0,())


@pytest.mark.parametrize('bounds',[(12,10000,100000,16384),(4096,12,100000,16384),(4096,10000,1,16384)])
def test_native_admissions_refuse_before_publishing_a_boundary_owner(bounds):
    with pytest.raises((ValueError,RuntimeError)):owner(limits=bounds)


def third_grade():
    data=declaration();data['max_grade']=3
    face=deepcopy(data['relations'][-1]);face['key']=5
    face['occurrences']=tuple(dict(o,key=o['key']+4) for o in face['occurrences'])
    higher=dict(grade=3,key=6,type=0,occurrences=(
        dict(key=19,lower=4,exists=True,negative=False,magnitude=1,aspects=(0,)),
        dict(key=20,lower=5,exists=True,negative=True,magnitude=1,aspects=(0,))))
    data['relations']+= (face,higher)
    data['carriers']=tuple(dict(c,size=2) if c['grade']==2 else c for c in data['carriers'])+(
        dict(key=13,kind='source_grade',grade=3,coordinates=0,size=1),)
    axes=(*AXES[:2],AXES[2]*2,(('assertion','unused'),))
    maps=(*MAPS[:2],MAPS[2]*2,(((0,0,1),),((0,0,1),)))
    return NativeSource.from_declaration(data),axes,maps


def test_third_grade_keeps_distinct_equal_boundary_faces_and_their_channels():
    source,axes,maps=third_grade();plan=PreparedStalkBoundary(source.archive,axes,maps)
    trace=plan.apply(3,((2,9),),boundary_steps=3)
    assert trace['grades']==(3,2,1,0)
    assert trace['fields'][1]==((2,0),(-2,0))
    assert trace['fields'][2]==((0,0),(0,),(0,0))
    assert trace['intermediate_channels'][0]==(((2,0),(2,),(-2,0)),((-2,0),(-2,),(2,0)))
    assert trace['receipt']['occurrence_keys'][3]==(19,20)


def test_complete_chain_check_includes_the_second_adjacent_interface():
    source,axes,maps=third_grade();bad=(*maps[:3],(((0,0,1),(0,1,1)),((0,0,1),)))
    with pytest.raises(ValueError,match='complete strict chain'):
        PreparedStalkBoundary(source.archive,axes,bad)


def test_rational_head_and_tail_shares_are_retained_as_original_occurrence_vectors():
    shares=(-Q(1),Q(1,2),Q(1,3),Q(1,6));data=declaration()
    data.update(max_grade=1,witnesses=(0,1,2,3),relations=(dict(grade=1,key=1,type=0,
        occurrences=tuple(dict(key=i+1,lower=i,exists=True,negative=q<0,magnitude=abs(q),aspects=(0,))
                          for i,q in enumerate(shares))),),
        carriers=(dict(key=10,kind='source_grade',grade=0,coordinates=0,size=4),
                  dict(key=11,kind='source_grade',grade=1,coordinates=0,size=1)))
    source=NativeSource.from_declaration(data)
    plan=PreparedStalkBoundary(source.archive,((('boundary',),)*4,(('relation',),)),
        ((),(((0,0,1),),)*4))
    trace=plan.apply(1,((6,),))
    assert trace['fields'][1]==((-6,),(3,),(2,),(1,))
    assert trace['occurrence_fields'][0]==((-6,),(3,),(2,),(1,))
    assert sum(shares[1:])==1 and sum(q*q for q in shares)==Q(25,18)
