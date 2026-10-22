"""Owning finite search preserves symbols, sources and sound budget outcomes."""
import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_relational_finite_field import asset, declaration

from rexgraph_cpp import NativeFiniteSource


@pytest.mark.parametrize("method", ["bounded_global_section", "propagated_global_section"])
def test_witness_is_a_complete_original_section_and_archive_is_unchanged(method):
    owner = NativeFiniteSource(asset(), declaration())
    before = owner.archive()
    family = owner.sections().select(0, 3)
    solve = getattr(owner, method)
    result = solve(((0, 1, 102),))
    assert result["status"] == "satisfiable"
    assert result["witness"] == ((0, 0, 102), (0, 1, 102), (0, 2, 102))
    assert tuple(v[2] for v in result["witness"]) in family
    assert result["work"]["variables"] == 3
    assert result["work"]["steps"] > 0
    assert owner.archive() == before
    restored = NativeFiniteSource.from_archive(before)
    assert getattr(restored, method)(((0, 1, 102),)) == result
    del owner, restored, solve
    gc.collect()
    assert result["witness"][0] == (0, 0, 102)


@pytest.mark.parametrize("method", ["bounded_global_section", "propagated_global_section"])
def test_refutation_is_distinct_from_work_admission_and_unknown(method):
    owner = NativeFiniteSource(asset(), declaration())
    solve = getattr(owner, method)
    refuted = solve(((0, 0, 101), (0, 2, 102)))
    assert refuted["status"] == "unsatisfiable" and refuted["witness"] is None
    for limits in ({"max_steps": 0}, {"max_variables": 2}):
        unresolved = solve(**limits)
        assert unresolved["status"] == "budget_exceeded" and unresolved["witness"] is None
    for symbol in (101, 102, 103):
        assert solve(((0, 0, symbol),))["witness"] == tuple((0, i, symbol) for i in range(3))


def test_propagated_scratch_admission_and_receipts_are_per_invocation():
    owner = NativeFiniteSource(asset(), declaration())
    full = owner.propagated_global_section(((0, 1, 103),))
    assert full["work"]["domain_reductions"] > 0
    assert full["work"]["peak_entries"] > 0
    refused = owner.propagated_global_section(max_entries=0)
    assert refused["status"] == "budget_exceeded" and refused["witness"] is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: owner.propagated_global_section(((0, 1, 103),)), range(8)))
    assert results == [full] * 8


@pytest.mark.parametrize("fixed", [((0, 0, 101), (0, 0, 101)), ((0, 0, 999),), ((0, 99, 101),), ((0, 0),), ((False, 0, 101),)])
@pytest.mark.parametrize("method", ["bounded_global_section", "propagated_global_section"])
def test_invalid_fixed_assignments_are_rejected(method, fixed):
    owner = NativeFiniteSource(asset(), declaration())
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        getattr(owner, method)(fixed)


@pytest.mark.parametrize("limits", [{"max_steps": -1}, {"max_steps": True}, {"max_variables": 1.5}, {"max_entries": -1}])
def test_invalid_limits_are_rejected_without_mutating_source(limits):
    owner = NativeFiniteSource(asset(), declaration())
    before = owner.archive()
    with pytest.raises((ValueError, OverflowError)):
        owner.propagated_global_section(**limits)
    assert owner.archive() == before
