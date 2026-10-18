"""Retained dictionary column comparisons and complete alternative witnesses."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from rexgraph_cpp import NativeSource,PreparedRelationGuards


def dictionary_source(*,fault=None):
    rows=((0,2,0),(1,3,0),(2,4,1),(2,5,1),(3,6,1),(3,7,1),
          (4,8,2),(5,9,2),(6,8,2),(7,9,2),(6,8,2))
    relations=[];key=100
    for row,(a,b,kind) in enumerate(rows):
        flags=[]
        for side,lower in enumerate((a,b)):
            key+=7
            flag=dict(key=key,lower=lower,exists=True,negative=False,magnitude=1,aspects=(kind+side,))
            if row==8 and side==1:
                if fault=="inactive":flag["exists"]=False
                elif fault=="zero":flag["magnitude"]=0
                elif fault=="negative":flag["negative"]=True
                elif fault=="magnitude":flag["magnitude"]=2
                elif fault=="role":flag["aspects"]=(0,)
            flags.append(flag)
        relations.append(dict(grade=1,key=500+row*11,type=kind,occurrences=tuple(flags)))
    return NativeSource.from_declaration(dict(profile="declared-role-source-v1",namespace=961007,
        semantic_scale=2,max_grade=1,witnesses=tuple(range(12)),
        role_aspects=("dictionary-record","dictionary-entry","dictionary-sense","dictionary-synset"),
        types=tuple(dict(name=name,min_arity=0,max_arity=0) for name in ("record-entry","entry-sense","sense-synset")),
        relations=tuple(relations),fields=(),metrics=(),carriers=()))


def test_all_common_synsets_and_duplicate_columns_retain_native_maps():
    owner=PreparedRelationGuards(dictionary_source().archive)
    result=owner.shared_synsets(((0,1),(0,11)))
    assert result["pairs"][1]["licenses"]==()
    licenses=result["pairs"][0]["licenses"]
    assert tuple(l["synset"] for l in licenses)==(8,8,9)
    assert tuple(l["target_rows"][-1] for l in licenses)==(8,10,9)
    for license in licenses:
        native=owner.match(license["source_rows"],license["target_rows"],fixed_lowers=license["fixed_lowers"])
        assert all(license[key]==native[key] for key in ("cell_map","relation_map","occurrence_map","execution_steps","strict_morphism","chain_map","ordered"))
        assert len(license["occurrence_map"])==6
        assert (license["synset"],license["synset"]) in license["cell_map"]
    assert result["execution_steps"]>sum(l["execution_steps"] for l in licenses)


def test_dictionary_join_matches_an_independent_full_route_oracle_for_all_lower_pairs():
    native=dictionary_source();owner=PreparedRelationGuards(native.archive)
    columns=tuple(owner.boundary(row) for row in range(11))
    routes={i:[] for i in range(12)}
    for a in range(2):
        record,entry=(f["lower"] for f in columns[a]["occurrences"])
        for b in range(2,6):
            x,sense=(f["lower"] for f in columns[b]["occurrences"])
            if x!=entry:continue
            for c in range(6,11):
                y,synset=(f["lower"] for f in columns[c]["occurrences"])
                if y==sense:routes[record].append((synset,(a,b,c)))
    pairs=tuple((a,b) for a in range(12) for b in range(12))
    result=owner.shared_synsets(pairs)
    for pair,native_pair in zip(pairs,result["pairs"],strict=True):
        expected=tuple((s,left,right) for s,left in routes[pair[0]] for t,right in routes[pair[1]] if s==t)
        actual=tuple((l["synset"],l["source_rows"],l["target_rows"]) for l in native_pair["licenses"])
        assert actual==expected


@pytest.mark.parametrize("fault",["inactive","zero"])
def test_inactive_or_supported_zero_column_cannot_supply_a_synset_license(fault):
    owner=PreparedRelationGuards(dictionary_source(fault=fault).archive)
    result=owner.shared_synsets(((0,1),))
    assert tuple(l["target_rows"][-1] for l in result["pairs"][0]["licenses"])==(10,9)
    assert len(owner.boundary(8)["occurrences"])==2


@pytest.mark.parametrize("fault",["role","negative","magnitude"])
def test_dictionary_profile_refuses_unsupported_role_and_coefficient_declarations(fault):
    owner=PreparedRelationGuards(dictionary_source(fault=fault).archive)
    with pytest.raises(ValueError,match="dictionary"):
        owner.shared_synsets(((0,1),))


@pytest.mark.parametrize("pairs",[((0,1),(0,1)),((0,),),((0,1,2),),((0,12),),((True,1),),((0.0,1),)])
def test_dictionary_pair_declarations_are_exact_and_bounded(pairs):
    owner=PreparedRelationGuards(dictionary_source().archive)
    with pytest.raises((ValueError,TypeError)):
        owner.shared_synsets(pairs)


def test_dictionary_comparison_work_includes_all_nested_map_checks():
    owner=PreparedRelationGuards(dictionary_source().archive)
    result=owner.shared_synsets(((0,1),))
    with pytest.raises(RuntimeError,match="work|step"):
        owner.shared_synsets(((0,1),),work_limit=result["execution_steps"]-1)
    assert owner.shared_synsets(((0,1),),work_limit=result["execution_steps"])==result
    assert owner.shared_synsets(())["pairs"]==()
    empty=PreparedRelationGuards.__new__(PreparedRelationGuards)
    with pytest.raises(RuntimeError,match="initialized"):empty.shared_synsets(((0,1),))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(value==result for value in pool.map(lambda _:owner.shared_synsets(((0,1),)),range(8)))


def test_dictionary_results_share_the_native_entry_admission():
    archive=dictionary_source().archive
    owner=PreparedRelationGuards(archive,limits=(600,64000000,16384))
    with pytest.raises(RuntimeError,match="entry limit"):
        owner.shared_synsets(((0,1),))
