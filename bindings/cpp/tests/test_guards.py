"""Whole native context maps, occurrence semantics and finite guard admissions."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import pytest

from rexgraph_cpp import NativeSource, PreparedRelationGuards


def source(rows):
    relations=[];key=0
    for row in rows:
        flags=[]
        for lower,role,exists,negative,magnitude in row:
            key+=1
            flags.append(dict(key=key,lower=lower,aspects=(role,),exists=exists,negative=negative,magnitude=magnitude))
        relations.append(dict(grade=1,key=len(relations)+1,type=0,occurrences=tuple(flags)))
    return NativeSource.from_declaration(dict(profile="declared-role-source-v1",namespace=951107,
        semantic_scale=2,max_grade=1,role_aspects=("left","target","right"),
        types=(dict(name="context",min_arity=0,max_arity=0),),witnesses=tuple(range(8)),
        relations=tuple(relations),carriers=(),fields=(),metrics=()))


def row(lowers):
    return tuple((lower,i,True,False,1) for i,lower in enumerate(lowers))


def oracle(rows,origins,targets,pins):
    forward={};reverse={}
    for a,b in zip(origins,targets,strict=True):
        if len(rows[a])!=len(rows[b]):return False
        for x,y in zip(rows[a],rows[b],strict=True):
            if x[1:]!=y[1:]:return False
            if x[0] in forward and forward[x[0]]!=y[0]:return False
            if y[0] in reverse and reverse[y[0]]!=x[0]:return False
            forward[x[0]]=y[0];reverse[y[0]]=x[0]
    return all(forward[i]==i for i in pins)


def test_native_guard_matches_independent_ordered_occurrence_oracle():
    rows=tuple(row(values) for values in product(range(3),repeat=3))
    plan=PreparedRelationGuards(source(rows).archive)
    for a in range(len(rows)):
        for b in range(len(rows)):
            pins=(rows[a][1][0],)
            result=plan.match((a,),(b,),fixed_lowers=pins)
            assert result["matched"]==oracle(rows,(a,),(b,),pins)
            assert result["strict_morphism"]==result["chain_map"]==result["matched"]
            if result["matched"]:
                assert len(result["occurrence_map"])==3
                assert all(q-p==3*(b-a) for p,q in result["occurrence_map"])


def test_union_guard_prevents_shared_identity_splitting_between_contexts():
    rows=(row((0,1,2)),row((0,3,4)),row((5,1,2)),row((6,3,4)))
    plan=PreparedRelationGuards(source(rows).archive)
    assert plan.match((0,),(2,))["matched"]
    assert plan.match((1,),(3,))["matched"]
    result=plan.match((0,1),(2,3))
    assert not result["matched"] and result["reason"]=="shared-identity-split"


@pytest.mark.parametrize("field,value",[(1,0),(2,False),(3,True),(4,0),(4,2)])
def test_guard_checks_zero_inactive_orientation_roles_and_exact_magnitude(field,value):
    first=row((0,1,2));changed=[list(flag) for flag in first];changed[2][field]=value
    plan=PreparedRelationGuards(source((first,tuple(tuple(flag) for flag in changed))).archive)
    assert not plan.match((0,),(1,))["matched"]
    assert plan.match((1,),(1,))["matched"]
    boundary=plan.boundary(1)
    assert len(boundary["occurrences"])==3


def test_equal_aggregated_boundaries_do_not_hide_changed_occurrence_order():
    a=row((0,1,2));b=(a[2],a[1],a[0])
    plan=PreparedRelationGuards(source((a,b)).archive)
    assert not plan.match((0,),(1,),fixed_lowers=(0,1,2))["matched"]


def test_guard_owner_is_immutable_and_concurrent_calls_have_private_scratch():
    plan=PreparedRelationGuards(source((row((0,1,2)),row((0,1,3)))).archive)
    expected=plan.match((0,),(1,),fixed_lowers=(0,1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result==expected for result in pool.map(lambda _:plan.match((0,),(1,),fixed_lowers=(0,1)),range(12)))
    with pytest.raises(RuntimeError,match="immutable"):
        plan.__init__(source((row((0,1,2)),)).archive)
    empty=PreparedRelationGuards.__new__(PreparedRelationGuards)
    with pytest.raises(RuntimeError,match="initialized"):empty.match((0,),(0,))
    with pytest.raises(RuntimeError,match="initialized"):empty.boundary(0)


@pytest.mark.parametrize("origins,targets,pins",[((),(),()),((0,),(5,),()),((0,0),(0,1),()),
    ((0,1),(0,0),()),((0,),(0,),(0,0)),((0,),(0,),(7,)),((True,),(0,),()),((0.5,),(0,),())])
def test_invalid_guard_declarations_are_rejected(origins,targets,pins):
    plan=PreparedRelationGuards(source((row((0,1,2)),row((0,1,3)))).archive)
    with pytest.raises((ValueError,IndexError,TypeError)):
        plan.match(origins,targets,fixed_lowers=pins)


def test_guard_admission_rejects_archive_entries_work_and_integer_size():
    archive=source((row((0,1,2)),)).archive
    with pytest.raises(ValueError,match="byte admission"):PreparedRelationGuards(archive,max_bytes=len(archive)-1)
    with pytest.raises(RuntimeError,match="entry limit"):PreparedRelationGuards(archive,limits=(2,64000000,16384))
    with pytest.raises(RuntimeError,match="step limit"):PreparedRelationGuards(archive,limits=(1000000,2,16384))
    huge=source((((0,0,True,False,1<<32),),)).archive
    with pytest.raises(RuntimeError,match="bit limit"):PreparedRelationGuards(huge,limits=(1000000,64000000,8))
    plan=PreparedRelationGuards(archive)
    with pytest.raises(RuntimeError,match="step limit"):plan.match((0,),(0,),work_limit=1)
