"""Finite native section agreement, conflict cores and resource admissions."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import pytest

from rexgraph_cpp import NativeSource, PreparedRelationalSections


def source():
    key = 0; relations = []
    for row in ((0,1), (1,2), (2,0), (0,0,1)):
        occurrences = []
        for lower in row:
            key += 1
            occurrences.append(dict(key=key,lower=lower,exists=True,negative=False,magnitude=1))
        relations.append(dict(grade=1,key=len(relations)+1,type=0,occurrences=tuple(occurrences)))
    return NativeSource.from_declaration(dict(profile="declared-exact-source-v1",namespace=947107,
        semantic_scale=0,max_grade=1,types=(dict(name="constraint",min_arity=0,max_arity=0),),
        witnesses=(0,1,2),relations=tuple(relations),fields=(),metrics=(),carriers=()))


def chart(name, relation, support, allowed):
    return dict(name=name,relation=relation,support=support,allowed=allowed)


EQ = ((0,0),(1,1))
NE = ((0,1),(1,0))


def test_all_native_sections_agree_with_independent_finite_predicates():
    s = source()
    charts = (chart("x=y",0,(0,1),EQ),chart("y=z",1,(1,2),EQ))
    plan = PreparedRelationalSections(s.archive,((0,1),)*3,charts)
    out = plan.sections()
    oracle = tuple(row for row in product((0,1),repeat=3) if row[0]==row[1] and row[1]==row[2])
    assert out["sections"] == oracle == ((0,0,0),(1,1,1))
    assert out["satisfiable"] and out["constraint_complete"] and out["candidate_count"]==8
    assert out["conflict_charts"]==()
    charts[0]["allowed"] = NE
    assert plan.sections()==out
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(result==out for result in pool.map(lambda _:plan.sections(),range(8)))


def test_exact_global_conflict_core_is_subset_minimal_and_omits_redundancy():
    s = source()
    charts = (chart("x=y",0,(0,1),EQ),chart("y=z",1,(1,2),EQ),chart("z!=x",2,(2,0),NE),
              chart("tautology",0,(0,),((0,),(1,))))
    out = PreparedRelationalSections(s.archive,((0,1),)*3,charts).sections()
    assert not out["satisfiable"] and out["sections"]==()
    assert out["subset_minimal"] and out["conflict_charts"]==(0,1,2) and out["conflict_cells"]==(0,1,2)
    for dropped in out["conflict_charts"]:
        assert PreparedRelationalSections(s.archive,((0,1),)*3,
            tuple(c for i,c in enumerate(charts[:3]) if i!=dropped)).sections()["satisfiable"]


def test_repeated_occurrences_remain_primary_without_duplicate_variables():
    s = source()
    out = PreparedRelationalSections(s.archive,((0,1),)*3,(chart("branch",3,(0,1),NE),)).sections()
    assert len(out["sections"])==4
    assert all(row[0]!=row[1] for row in out["sections"])


def test_declared_cover_glues_whole_local_tuples_before_native_validation():
    s=source()
    charts=(chart("x=y",0,(0,1),EQ),chart("y=z",1,(1,2),EQ))
    full=PreparedRelationalSections(s.archive,((0,1),)*3,charts).sections()
    joined=PreparedRelationalSections(s.archive,((0,1),)*3,charts,enumeration_charts=(0,1)).sections()
    assert joined["sections"]==full["sections"]
    assert joined["domain_count"]==8 and joined["candidate_count"]==4
    for selectors in ((0,), (0,0), (2,)):
        with pytest.raises(ValueError):
            PreparedRelationalSections(s.archive,((0,1),)*3,charts,enumeration_charts=selectors)


def test_reentrant_initialization_cannot_replace_the_owner():
    s=source();plan=PreparedRelationalSections.__new__(PreparedRelationalSections)
    def domains():
        with pytest.raises(RuntimeError,match="immutable"):
            plan.__init__(s.archive,((0,1),)*3,())
        yield from ((0,1),)*3
    plan.__init__(s.archive,domains(),())
    assert len(plan.sections()["sections"])==8


@pytest.mark.parametrize("failure",["duplicate","outside","empty","arity","domain","float","candidate","work","bytes"])
def test_invalid_declarations_and_unadmitted_searches_are_rejected(failure):
    s = source(); domains=((0,1),)*3; charts=(chart("x=y",0,(0,1),EQ),); kwargs={}
    if failure=="duplicate":domains=((0,0),(0,1),(0,1))
    elif failure=="outside":charts=(chart("wrong",0,(2,),((0,),)),)
    elif failure=="empty":charts=(chart("wrong",0,(0,),()),)
    elif failure=="arity":charts=(chart("wrong",0,(0,1),((0,),)),)
    elif failure=="domain":charts=(chart("wrong",0,(0,),((2,),)),)
    elif failure=="float":domains=((0.0,1),)*3
    elif failure=="candidate":kwargs["limits"]=(7,1000000,64000000)
    elif failure=="work":kwargs["limits"]=(8,1000000,1)
    elif failure=="bytes":kwargs["max_bytes"]=len(s.archive)-1
    with pytest.raises((ValueError,IndexError)):
        PreparedRelationalSections(s.archive,domains,charts,**kwargs)


def test_uninitialized_and_reinitialized_owners_reject():
    empty=PreparedRelationalSections.__new__(PreparedRelationalSections)
    with pytest.raises(RuntimeError):empty.sections()
    s=source();plan=PreparedRelationalSections(s.archive,((0,1),)*3,())
    with pytest.raises(RuntimeError):plan.__init__(s.archive,((0,1),)*3,())
