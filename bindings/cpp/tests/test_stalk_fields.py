"""Whole stalk response oracles, occurrence support and declared acquisition maps."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q

import pytest

from rexgraph_cpp import NativeSource, PreparedStalkFields
from test_relation_response import declared, solve

LOWER = (("shared", "role"), ("semantic", "register", "context"))
UPPER = (("money", "river"), ("category",), ("formal", "casual"))
COLUMNS = (((0,0,1,True),(1,1,-2,True)),
           ((0,0,1,True),(0,0,-1,True)),
           ((1,1,Q(2,3),True),(0,0,9,False),(1,1,0,True)))
MAPS = (((0,0,1),(1,1,1)), ((0,0,Q(1,2)),(1,1,Q(1,3)),(2,0,1),(2,1,1)),
        ((0,0,1),), ((0,0,1),), ((0,0,1),(1,1,1),(2,0,-1)),
        ((0,0,1),(1,1,1)), ((2,0,1),(2,1,-1)))


def owner(**kwargs):
    source = NativeSource.from_declaration(declared(COLUMNS))
    return source, PreparedStalkFields(source.archive, LOWER, UPPER, MAPS, **kwargs)


def reference(fields, parameter=Q(1)):
    upper = [0,2,3,5]; lower = [0,2,5]
    boundary = [[Q(0)]*5 for _ in range(5)]; maps = []; slot = 0
    for i, column in enumerate(COLUMNS):
        for cell, _, coefficient, exists in column:
            entries = MAPS[slot]
            realized = tuple((r,c,Q(value)*coefficient if exists else Q(0)) for r,c,value in entries)
            maps.append((i,cell,realized)); slot += 1
            for row,col,value in realized:
                boundary[lower[cell]+row][upper[i]+col] += value
    matrix = [[Q(i==j)+parameter*sum(row[i]*row[j] for row in boundary)
               for j in range(5)] for i in range(5)]
    results = []
    for field in fields:
        response = solve(matrix, tuple(v for row in field for v in row))
        result = tuple(response[a:b] for a,b in zip(upper[:-1],upper[1:],strict=True))
        b = tuple(sum(value*x for value,x in zip(row,response,strict=True)) for row in boundary)
        b = tuple(b[a:z] for a,z in zip(lower[:-1],lower[1:],strict=True))
        occurrences = []
        for i,cell,entries in maps:
            vector = [Q(0)]*len(LOWER[cell])
            for row,col,value in entries:
                vector[row] += value*result[i][col]
            occurrences.append(tuple(vector))
        results.append((result,b,tuple(occurrences)))
    return tuple(results)


@pytest.mark.parametrize("parameter",[Q(1),Q(2,3),Q(17,20)])
def test_native_ragged_tensor_fields_match_independent_full_block_oracle(parameter):
    fields = (((Q(1,3),-2),(5,),(3,4)), ((1,-1),(0,),(0,0)), ((0,0),(1,),(0,0)))
    source,plan = owner(parameter=parameter); actual = plan.apply(fields)
    expected = reference(fields,parameter)
    assert tuple(zip(actual['fields'],actual['boundaries'],actual['occurrence_fields'],strict=True)) == expected
    assert actual['residual_exact_zero']
    assert plan.receipt['upper_axes']==UPPER and plan.receipt['lower_axes']==LOWER
    assert len(actual['fields'])==3 and len(actual['occurrence_fields'][0])==7
    assert actual['occurrence_fields'][0][-2:]==((0,0),(0,0,0))
    assert actual['fields'][2][1]==(1,) and all(v==0 for row in actual['boundaries'][2] for v in row)
    assert NativeSource(source.archive).archive==source.archive


def test_zero_scalar_total_does_not_erase_a_relational_stalk_reading():
    _,plan=owner()
    positive=((1,-1),(0,),(0,0)); zero=((0,0),(0,),(0,0))
    assert sum(v for row in positive for v in row)==sum(v for row in zero for v in row)
    result=plan.apply((positive,zero))
    assert result['fields'][0]!=result['fields'][1]
    assert result['occurrence_fields'][0]!=result['occurrence_fields'][1]


@pytest.mark.parametrize('value',[0.1,True,'1/2'])
def test_implicit_stalk_coefficients_reject_before_native_execution(value):
    with pytest.raises(TypeError):owner(parameter=value)
    _,plan=owner()
    with pytest.raises(TypeError):plan.apply((((value,0),(0,),(0,0)),))


def test_unused_inactive_flag_directions_and_duplicate_coordinates_are_checked():
    source,_=owner(); changed=list(MAPS)
    changed[5]=((2,0,1),)
    with pytest.raises(ValueError,match='coordinates'):
        PreparedStalkFields(source.archive,LOWER,UPPER,changed)
    changed[5]=((0,0,0),(0,0,0))
    with pytest.raises(ValueError,match='distinct flag'):
        PreparedStalkFields(source.archive,LOWER,UPPER,changed)


def test_complete_maps_and_named_axes_are_required_even_for_zero_fields():
    source,plan=owner()
    with pytest.raises(ValueError):PreparedStalkFields(source.archive,LOWER,UPPER,MAPS[:-1])
    with pytest.raises(ValueError):PreparedStalkFields(source.archive,LOWER,(('money','money'),*UPPER[1:]),MAPS)
    with pytest.raises(ValueError):plan.apply(((),))
    with pytest.raises(ValueError):plan.apply((((0,),(0,),(0,0)),))
    with pytest.raises(ValueError):plan.apply(())


def test_acquisition_is_a_native_boundary_followed_by_an_explicit_injection():
    _,plan=owner()
    columns=(((0,0,2,True),(1,1,-2,True)),((0,0,Q(1,3),True),(1,1,0,True)))
    acquisition=NativeSource.from_declaration(declared(columns))
    injected=plan.inject_acquisition(acquisition.archive,(1,0),(3,6))
    assert injected['boundary']==(8,-6)
    assert injected['field']==((-6,8),(0,),(0,0))
    assert injected['retained_occurrences']==4 and injected['columns']==2
    assert plan.apply((injected['field'],))['fields'][0]==reference((injected['field'],))[0][0]
    for mapping in ((0,0),(0,), (99,0)):
        with pytest.raises(ValueError):plan.inject_acquisition(acquisition.archive,mapping,(1,1))


def test_owner_lifetime_concurrency_and_refusal_preserve_later_actions():
    source,plan=owner(); del source
    fields=(((1,-1),(0,),(0,0)),)
    expected=plan.apply(fields)
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(out==expected for out in pool.map(lambda _:plan.apply(fields),range(9)))
    with pytest.raises(ValueError):plan.apply((((1<<16385,0),(0,),(0,0)),))
    assert plan.apply(fields)==expected
    with pytest.raises(RuntimeError,match='immutable'):plan.__init__(b'',(),(),())
    empty=PreparedStalkFields.__new__(PreparedStalkFields)
    with pytest.raises(RuntimeError,match='not initialized'):empty.apply(fields)


def test_higher_differentials_and_dimension_limits_refuse_explicitly():
    source,_=owner()
    with pytest.raises((ValueError,RuntimeError),match='dimension'):
        PreparedStalkFields(source.archive,LOWER,UPPER,MAPS,limits=(9,10000,100000,16384))
    data=declared(COLUMNS,max_grade=2)
    data['relations']+=(dict(grade=2,key=99,type=0,occurrences=()),)
    data['carriers']=tuple(dict(c,size=1) if c['grade']==2 else c for c in data['carriers'])
    source=NativeSource.from_declaration(data)
    with pytest.raises(ValueError,match='higher differential'):
        PreparedStalkFields(source.archive,LOWER,UPPER,MAPS)
