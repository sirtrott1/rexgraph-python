"""Exact response oracles and retained incidence roles on native sources."""
from copy import deepcopy
from fractions import Fraction as Q

import pytest

from rexgraph_cpp import NativeSource, PreparedRelationResponse


def declared(columns, *, max_grade=1):
    relations = []
    key = 1
    for i, members in enumerate(columns):
        occurrences = []
        for lower, role, value, exists in members:
            occurrences.append(dict(key=key, lower=lower, exists=exists,
                negative=value < 0, magnitude=abs(value), aspects=(role,)))
            key += 1
        relations.append(dict(grade=1, key=i+1, type=0, occurrences=tuple(occurrences)))
    return dict(profile="declared-role-source-v1", namespace=71007, semantic_scale=0,
        max_grade=max_grade, role_aspects=("left", "right"),
        types=(dict(name="retained-membership", min_arity=0, max_arity=0),),
        witnesses=(0, 1), relations=tuple(relations), fields=(), metrics=(),
        carriers=tuple(dict(key=10+k, kind="source_grade", grade=k, coordinates=0,
            size=2 if k==0 else len(columns) if k==1 else 0) for k in range(max_grade+1)))


def solve(matrix, rhs):
    rows = [[Q(v) for v in row]+[Q(b)] for row, b in zip(matrix, rhs, strict=True)]
    for i in range(len(rows)):
        pivot = next(j for j in range(i, len(rows)) if rows[j][i])
        rows[i], rows[pivot] = rows[pivot], rows[i]
        factor = rows[i][i]
        rows[i] = [v/factor for v in rows[i]]
        for j in range(len(rows)):
            if j != i:
                factor = rows[j][i]
                rows[j] = [a-factor*b for a, b in zip(rows[j], rows[i], strict=True)]
    return tuple(row[-1] for row in rows)


def oracle(columns, fields, parameter, gains=None):
    n = len(columns)
    boundary = [[Q(0)]*n for _ in range(6)]
    for i, members in enumerate(columns):
        for lower, role, value, exists in members:
            value = Q(value) if exists else Q(0)
            membership, retained_role = (gains or {}).get(("left","right")[role],(1,1))
            boundary[3*lower][i] += value*membership
            boundary[3*lower+role+1][i] += value*retained_role
    matrix = [[Q(i==j)+parameter*sum(row[i]*row[j] for row in boundary)
               for j in range(n)] for i in range(n)]
    responses = [solve(matrix, field) for field in fields]
    gram = tuple(tuple(sum(Q(a)*b for a,b in zip(field,response,strict=True))
                       for response in responses) for field in fields)
    return gram, responses[0], tuple(sum(a*b for a,b in zip(row,responses[0],strict=True)) for row in boundary)


def prepared(columns, **kwargs):
    source = NativeSource.from_declaration(declared(columns))
    axes = tuple(("membership", "left", "right") for _ in range(2))
    slots = tuple(role+1 for members in columns for _,role,_,_ in members)
    return source, PreparedRelationResponse(source.archive, axes, slots, **kwargs)


def test_native_original_map_green_matches_independent_fraction_oracle():
    columns = (((0,0,1,True),(1,1,2,True)), ((0,1,1,True),),
               ((0,0,1,True),(0,0,-1,True),(1,1,9,False),(1,0,0,True)))
    fields = ((Q(1,3),-2,5),(0,1,0),(1,1,0))
    source, plan = prepared(columns, parameter=Q(2,3))
    result = plan.moments(fields)
    gram, response, boundary = oracle(columns,fields,Q(2,3))
    assert (result["gram"],result["response"],result["boundary"]) == (gram,response,boundary)
    assert result["residual_exact_zero"] is True
    assert len(result["contributions"]) == 7
    assert result["contributions"][-2:] == (0,0)
    assert plan.receipt["retained_occurrences"] == 7
    assert NativeSource(source.archive).archive == source.archive


def test_role_axes_preserve_order_and_shared_membership_connects_distinct_roles():
    columns = (((0,0,1,True),),((0,1,1,True),),((0,0,1,True),))
    _, plan = prepared(columns)
    result = plan.moments(((1,0,0),(0,1,0),(0,0,1)))
    assert result["gram"][0][1] != 0
    assert result["gram"][0][1] != result["gram"][0][2]
    assert result["gram"][0][1] == result["gram"][1][0]


def test_simultaneous_column_and_source_reorientation_preserves_pairing():
    columns = (((0,0,1,True),(1,1,1,True)), ((0,0,1,True),))
    _, first = prepared(columns)
    changed = (tuple((a,b,-v,e) for a,b,v,e in columns[0]),columns[1])
    _, second = prepared(changed)
    assert first.moments(((2,1),(3,4)))["gram"] == second.moments(((-2,1),(-3,4)))["gram"]


@pytest.mark.parametrize("value", [0.5, True, "1/2"])
def test_floating_and_implicit_parameter_conversion_are_rejected(value):
    with pytest.raises(TypeError):
        prepared((((0,0,1,True),),),parameter=value)


def test_malformed_roles_limits_and_upper_differential_fail_explicitly():
    columns = (((0,0,1,True),),)
    source, plan = prepared(columns)
    axes = (("membership","left","right"),)*2
    with pytest.raises(ValueError,match="role aspect"):
        PreparedRelationResponse(source.archive,axes,(2,))
    with pytest.raises((ValueError,IndexError,RuntimeError),match="limit|dimension"):
        PreparedRelationResponse(source.archive,axes,(1,),limits=(1,100,100,100))
    with pytest.raises(ValueError):
        plan.moments(((1,2),))
    with pytest.raises(TypeError):
        plan.moments(((0.1,),))
    declaration = declared(columns,max_grade=2)
    empty = NativeSource.from_declaration(declaration)
    assert PreparedRelationResponse(empty.archive,axes,(1,)).moments(((1,),))["residual_exact_zero"]
    declaration["relations"] += (dict(grade=2,key=99,type=0,occurrences=()),)
    declaration["carriers"] = tuple(dict(c,size=1) if c["grade"]==2 else c for c in declaration["carriers"])
    higher = NativeSource.from_declaration(declaration)
    with pytest.raises(ValueError,match="higher stalk differential"):
        PreparedRelationResponse(higher.archive,axes,(1,))


def test_owner_cannot_be_reinitialized_or_used_before_initialization():
    columns = (((0,0,1,True),),)
    source, plan = prepared(columns)
    with pytest.raises(RuntimeError,match="immutable"):
        plan.__init__(source.archive,(("membership","left","right"),)*2,(1,))
    owner = PreparedRelationResponse.__new__(PreparedRelationResponse)
    with pytest.raises(RuntimeError,match="not initialized"):
        owner.moments(((1,),))


def test_declaration_rejects_foreign_or_duplicate_role_flags():
    declaration = declared((((0,0,1,True),),))
    for roles in ((2,),(0,0)):
        changed = deepcopy(declaration)
        changed["relations"][0]["occurrences"][0]["aspects"] = roles
        with pytest.raises(ValueError,match="occurrence role aspect"):
            NativeSource.from_declaration(changed)


def test_scalar_admission_rejects_growth_before_native_conversion():
    _,plan=prepared((((0,0,1,True),),),limits=(32,1024,100000,4096))
    with pytest.raises(ValueError,match="integer bit admission"):
        plan.moments(((1<<4097,),))


def test_repeated_field_actions_obey_the_declared_work_admission():
    _,plan=prepared((((0,0,1,True),),),limits=(128,16000,512,4096))
    assert plan.moments(((1,),))["residual_exact_zero"]
    with pytest.raises(RuntimeError,match="step limit"):
        plan.moments(((1,),)*32)
    assert plan.moments(((1,),))["residual_exact_zero"]


@pytest.mark.parametrize('membership',[Q(0),Q(1,2),Q(1),Q(2)])
@pytest.mark.parametrize('role',[Q(0),Q(1,3),Q(1),Q(2)])
def test_declared_role_gains_match_full_independent_green_and_flag_oracle(membership,role):
    columns=(((0,0,1,True),(1,1,2,True)),((0,1,1,True),),
        ((0,0,1,True),(0,0,-1,True),(1,1,9,False),(1,0,0,True)))
    table=(("left",membership,role),("right",Q(2,3),Q(3,2)))
    fields=((Q(1,3),-2,5),(0,1,0),(1,1,0))
    source,plan=prepared(columns,parameter=Q(2,3),role_transports=table)
    result=plan.moments(fields)
    expected=oracle(columns,fields,Q(2,3),dict((name,(a,b)) for name,a,b in table))
    assert (result['gram'],result['response'],result['boundary'])==expected
    offsets=(0,3);boundary=[Q(0)]*6;slot=0
    for column in columns:
        for lower,index,_,_ in column:
            boundary[offsets[lower]]+=result['membership_contributions'][slot]
            boundary[offsets[lower]+index+1]+=result['role_contributions'][slot];slot+=1
    assert tuple(boundary)==result['boundary']
    assert plan.receipt['retained_occurrences']==7 and source.identity['coordinates'][1]
    assert result['contributions'][-2:]==(0,0)
    assert plan.receipt['role_transports']==table


@pytest.mark.parametrize('table',[(('left',True,1),),(('left',0.5,1),),(('left',-1,1),),
    (('foreign',1,1),),(('left',1,1),('left',1,1)),(('membership',1,1),),(('left',1),)])
def test_invalid_native_role_transports_reject(table):
    with pytest.raises((TypeError,ValueError,IndexError)):
        prepared((((0,0,1,True),),),role_transports=table)
