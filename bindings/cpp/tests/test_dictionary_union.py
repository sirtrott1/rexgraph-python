"""Complete dictionary union maps, retained alternatives and owning execution."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import pytest

from rexgraph_cpp import NativeSource,PreparedRelationGuards,PreparedDictionaryGuards


def alias_source(*,split=False,duplicates=False):
    rows=[(0,4,0),(1,4,0),(2,5,0),(3,6 if split else 5,0),
          (4,7,1),(5,8,1),(6,9,1),(7,10,2),(8,10,2),(9,10,2)]
    if duplicates:rows.extend(((4,7,1),(5,8,1),(7,10,2),(8,10,2)))
    relations=[];key=37
    for a,b,kind in rows:
        flags=[]
        for side,lower in enumerate((a,b)):
            key+=13
            flags.append(dict(key=key,lower=lower,exists=True,negative=False,magnitude=1,aspects=(kind+side,)))
        relations.append(dict(grade=1,key=len(relations)*17+5,type=kind,occurrences=tuple(flags)))
    return NativeSource.from_declaration(dict(profile="declared-role-source-v1",namespace=971007,
        semantic_scale=2,max_grade=1,witnesses=tuple(range(11)),
        role_aspects=("dictionary-record","dictionary-entry","dictionary-sense","dictionary-synset"),
        types=tuple(dict(name=name,min_arity=0,max_arity=0) for name in ("record-entry","entry-sense","sense-synset")),
        relations=tuple(relations),fields=(),metrics=(),carriers=())),relations


@pytest.mark.parametrize("pairs",[((0,2),(1,3)),((2,0),(3,1))])
def test_individual_licenses_do_not_allow_shared_entry_split_or_merge(pairs):
    native,_=alias_source(split=True);owner=PreparedRelationGuards(native.archive)
    catalog=owner.prepare_dictionary(pairs)
    assert all(p["licenses"] for p in catalog.comparison["pairs"])
    result=catalog.sections((0,1))
    assert result["sections"]==() and result["complete"]
    assert result["domain_count"]==result["candidate_count"]==result["overlap_rejections"]==1
    assert result["native_calls"]==0


def test_shared_original_dictionary_columns_glue_once_in_one_whole_map():
    native,_=alias_source();owner=PreparedRelationGuards(native.archive)
    catalog=owner.prepare_dictionary(((0,2),(1,3)))
    result=catalog.sections((0,1));assert len(result["sections"])==1
    section=result["sections"][0]
    assert section["source_rows"]==(0,1,4,7)
    assert len(section["relation_map"])==4 and len(section["occurrence_map"])==8
    replay=owner.match(section["source_rows"],section["target_rows"],fixed_lowers=section["fixed_lowers"])
    assert all(section[key]==replay[key] for key in ("cell_map","relation_map","occurrence_map","execution_steps","strict_morphism","chain_map","ordered"))


def union_oracle(relations,routes):
    row_map={};inverse_rows={};cell_map={};inverse_cells={}
    for license in routes:
        for a,b in zip(license["source_rows"],license["target_rows"],strict=True):
            if a in row_map and row_map[a]!=b or b in inverse_rows and inverse_rows[b]!=a:return False
            row_map[a]=b;inverse_rows[b]=a
            for x,y in zip(relations[a]["occurrences"],relations[b]["occurrences"],strict=True):
                if any(x[key]!=y[key] for key in ("aspects","exists","negative","magnitude")):return False
                old,new=x["lower"],y["lower"]
                if old in cell_map and cell_map[old]!=new or new in inverse_cells and inverse_cells[new]!=old:return False
                cell_map[old]=new;inverse_cells[new]=old
    return all(cell_map[l["synset"]]==l["synset"] for l in routes)


@pytest.mark.parametrize("split",[False,True])
def test_all_route_combinations_agree_with_independent_original_column_oracle(split):
    native,relations=alias_source(split=split,duplicates=True)
    catalog=PreparedRelationGuards(native.archive).prepare_dictionary(((0,2),(1,3)))
    alternatives=tuple(p["licenses"] for p in catalog.comparison["pairs"])
    expected=set()
    for choices in product(*(range(len(a)) for a in alternatives)):
        if union_oracle(relations,tuple(a[i] for a,i in zip(alternatives,choices,strict=True))):expected.add(choices)
    result=catalog.sections((0,1))
    assert result["domain_count"]==len(alternatives[0])*len(alternatives[1])
    assert set(s["choices"] for s in result["sections"])==expected
    assert result["candidate_count"]==result["domain_count"]
    assert result["native_calls"]+result["overlap_rejections"]==result["domain_count"]


def test_seed_context_and_dictionary_routes_share_the_same_map():
    native,_=alias_source();catalog=PreparedRelationGuards(native.archive).prepare_dictionary(((0,2),(1,3)))
    assert catalog.sections((0,1),source_rows=(0,),target_rows=(2,))["sections"]
    result=catalog.sections((0,1),source_rows=(0,),target_rows=(3,))
    assert not result["sections"] and result["native_calls"]==1
    assert result["overlap_rejections"]==1
    identity=catalog.sections((),source_rows=(0,),target_rows=(0,),fixed_lowers=(0,))
    assert len(identity["sections"])==1 and identity["domain_count"]==1


def test_prepared_catalog_retains_source_and_has_private_concurrent_scratch():
    native,_=alias_source();owner=PreparedRelationGuards(native.archive)
    catalog=owner.prepare_dictionary(((0,2),(1,3)));expected=catalog.sections((0,1));del owner,native
    returned=catalog.comparison;returned["pairs"][0]["licenses"][0]["source_rows"]=(999,)
    assert catalog.sections((0,1))==expected
    with pytest.raises(RuntimeError,match="prepare_dictionary"):catalog.__init__()
    empty=PreparedDictionaryGuards.__new__(PreparedDictionaryGuards)
    with pytest.raises(RuntimeError,match="initialized"):empty.sections((0,))
    with pytest.raises(RuntimeError,match="initialized"):_ = empty.comparison
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(r==expected for r in pool.map(lambda _:catalog.sections((0,1)),range(12)))


@pytest.mark.parametrize("indices,kwargs",[((0,0),{}),((2,),{}),((True,),{}),((0.5,),{}),
    ((),{}),((0,),dict(source_rows=(),target_rows=(0,))),((0,),dict(candidate_limit=0)),
    ((0,),dict(source_rows=(100,),target_rows=(0,))),((0,),dict(source_rows=(0,),target_rows=(0,),fixed_lowers=(10,)))])
def test_invalid_dictionary_union_requests_refuse_without_partial_results(indices,kwargs):
    native,_=alias_source();catalog=PreparedRelationGuards(native.archive).prepare_dictionary(((0,2),(1,3)))
    with pytest.raises((ValueError,IndexError,TypeError)):catalog.sections(indices,**kwargs)


def test_full_route_domain_is_admitted_before_pruning_and_work_includes_whole_map():
    native,_=alias_source(duplicates=True);catalog=PreparedRelationGuards(native.archive).prepare_dictionary(((0,2),(1,3)))
    result=catalog.sections((0,1))
    with pytest.raises(RuntimeError,match="candidate admission"):catalog.sections((0,1),candidate_limit=result["domain_count"]-1)
    with pytest.raises(RuntimeError,match="work|step"):catalog.sections((0,1),work_limit=result["execution_steps"]-1)
    exact=catalog.sections((0,1),work_limit=result["execution_steps"])
    assert exact==result and all(s["strict_morphism"] and s["chain_map"] for s in result["sections"])


def test_pair_without_any_dictionary_route_has_a_complete_empty_domain():
    native,_=alias_source();catalog=PreparedRelationGuards(native.archive).prepare_dictionary(((0,6),))
    result=catalog.sections((0,))
    assert result["domain_count"]==result["candidate_count"]==0
    assert result["complete"] and result["sections"]==()
