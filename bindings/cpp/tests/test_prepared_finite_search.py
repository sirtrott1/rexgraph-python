"""Compiled finite owners retain original predicates and whole conditional fields."""
import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_relational_finite_field import asset, declaration

from rexgraph_cpp import (
    FiniteSearchCancellation,
    NativeFiniteFamily,
    NativeFiniteSource,
    PreparedFiniteSearch,
)


def fixture(fixed=()):
    source = NativeFiniteSource(asset(), declaration())
    result = PreparedFiniteSearch.prepare(source)
    assert result['status'] == 'ready'
    prepared = result['prepared']
    propagated = prepared.propagate(fixed)
    assert propagated['status'] == 'fixed_point'
    return source, prepared, propagated['snapshot']


def projection(fixed=(), window=None):
    return dict(key=701, revision=1, cells=((0, 0), (0, 2)),
                window=window, fixed=fixed, distinct_relations=())


def test_actual_search_root_owner_and_complete_field_survive_parent_release():
    source, prepared, root = fixture(((0, 1, 102),))
    assert prepared.source().same_owner(source)
    assert root.prepared().same_owner(prepared)
    assert root.source().same_owner(source)
    assert root.same_owner(root)
    assert root.inspect()['fixed'] == ((0, 1, 102),)
    assert root.select_domains(0, 3) == tuple((0, i, (102,)) for i in range(3))
    archive = source.archive()
    cold_source = NativeFiniteSource.from_archive(archive)
    cold = PreparedFiniteSearch.prepare(cold_source)['prepared'].propagate(((0, 1, 102),))['snapshot']
    assert not cold.same_owner(root) and not cold.source().same_owner(source)
    assert cold.select_domains(0, 3) == root.select_domains(0, 3)
    del source, prepared, cold, cold_source
    gc.collect()
    assert root.continue_global_section()['witness'] == tuple((0, i, 102) for i in range(3))
    assert root.source().archive() == archive


@pytest.mark.parametrize('profile', ['source', 'prepared', 'root'])
def test_correlated_image_contains_whole_original_witnesses(profile):
    source, prepared, root = fixture()
    family = NativeFiniteFamily.projection(dict(source=source, prepared=prepared, root=root)[profile], projection())
    assert family.declaration() == projection() and family.comparison() is None
    assert family.source_mapping() == {'cells': ((0, 0), (0, 2)), 'symbols': ((101, 102, 103), (101, 102, 103))}
    result = family.materialize()
    assert result['status'] == 'complete'
    image = result['image']
    assert image.inspect() == {'row_count': 3, 'query_count': 9}
    assert image.family().same_owner(family)
    assert [image.row(i).inspect()['tuple'] for i in range(3)] == [(101, 101), (102, 102), (103, 103)]
    assert family.extend((101, 102))['status'] == 'unsatisfiable'
    for i in range(3):
        row = image.row(i)
        assert row.family().same_owner(family) and row.field_size() == 3
        assert row.witness() == tuple((0, c, 101 + i) for c in range(3))
        assert row.witness(begin=1, count=1) == ((0, 1, 101 + i),)
    queries = image.select_queries(0, 9)
    assert result['queries'].family().same_owner(family)
    assert result['queries'].select(0, 9) == queries
    assert sum(q['status'] == 'satisfiable' for q in queries) == 3
    if profile == 'root':
        assert family.propagation().same_owner(root)
        assert result['work']['reused_propagations'] == 1
        assert result['work']['continued_source_queries'] == 9
        assert result['work']['prior_propagation_steps'] == root.inspect()['propagation_work']['steps']
    else:
        assert family.propagation() is None
    archive = source.archive()
    del source, prepared, root, family
    gc.collect()
    assert image.row(1).witness() == ((0, 0, 102), (0, 1, 102), (0, 2, 102))
    assert image.family().source().archive() == archive


def test_root_request_must_match_original_fixed_and_ordered_group_request():
    fixed = ((0, 1, 102),)
    source, prepared, root = fixture(fixed)
    with pytest.raises(ValueError):
        NativeFiniteFamily.projection(root, projection())
    bad = projection(fixed)
    bad['distinct_relations'] = ((1, 0),)
    with pytest.raises(ValueError):
        NativeFiniteFamily.projection(root, bad)
    family = NativeFiniteFamily.projection(root, projection(fixed))
    assert family.materialize()['image'].inspect()['row_count'] == 1
    assert family.extend((101, 101))['status'] == 'unsatisfiable'
    assert root.continue_global_section(((0, 0, 101),))['status'] == 'unsatisfiable'
    assert root.continue_global_section()['status'] == 'satisfiable'


def test_intersection_retains_actual_parents_correlations_and_leaf_witnesses():
    source, prepared, root = fixture()
    a = NativeFiniteFamily.projection(root, projection())
    b = NativeFiniteFamily.projection(source, projection(window=((102, 103), (101, 102))))
    c = NativeFiniteFamily.intersect_images(a, b, dict(key=702, revision=1,
        right_positions=(0, 1), right_symbols=((101, 102, 103), (101, 102, 103))))
    assert c.parent().same_owner(a) and c.parent(right=True).same_owner(b)
    assert c.comparison() == dict(key=702, revision=1, right_positions=(0, 1), right_symbols=((101, 102, 103), (101, 102, 103)))
    assert c.propagation(0).same_owner(root) and c.propagation(1) is None
    image = c.materialize()['image']
    assert image.inspect()['row_count'] == 1
    row = image.row(0)
    assert row.inspect() == {'tuple': (102, 102), 'leaf_count': 2}
    assert row.witness(0) == row.witness(1) == ((0, 0, 102), (0, 1, 102), (0, 2, 102))
    assert c.extend((101, 102))['extension'] is None
    with pytest.raises(ValueError):
        NativeFiniteFamily.intersect_images(a, b, dict(key=703, revision=1,
            right_positions=(0, 0), right_symbols=((101, 102, 103), (101, 102, 103))))


def test_empty_window_is_complete_empty_but_interruption_never_is():
    _, prepared, root = fixture()
    empty = NativeFiniteFamily.projection(root, projection(window=((), (101, 102, 103))))
    complete = empty.materialize()
    assert complete['status'] == 'complete' and complete['image'].inspect()['row_count'] == 0
    family = NativeFiniteFamily.projection(root, projection())
    for limits in ((65536, 0, 16777216, 65536, 65536), (65536, 64000000, 0, 65536, 65536),
                   (65536, 64000000, 16777216, 1, 65536), (65536, 64000000, 16777216, 65536, 1)):
        refused = family.materialize(family_limits=limits)
        assert refused['status'] == 'budget_exceeded' and refused['image'] is None
        assert refused['queries'].family().same_owner(family)
    partial = family.materialize(family_limits=(65536, 64000000, 16777216, 65536, 1))
    assert partial['queries'].inspect()['query_count'] > 0
    assert partial['queries'].select(0, 1)[0]['status'] == 'satisfiable'
    stop = FiniteSearchCancellation(); stop.cancel()
    assert PreparedFiniteSearch.prepare(root.source(), cancellation=stop)['status'] == 'cancelled'
    assert prepared.propagate(cancellation=stop)['snapshot'] is None
    assert root.continue_global_section(cancellation=stop)['status'] == 'cancelled'
    assert family.materialize(cancellation=stop)['image'] is None
    assert family.extend((101, 101), cancellation=stop)['extension'] is None


def test_eight_concurrent_readers_share_root_but_keep_private_search_work():
    _, _, root = fixture()
    family = NativeFiniteFamily.projection(root, projection())
    def read(_):
        result = family.materialize()
        return result['work'], tuple(result['image'].row(i).witness() for i in range(3))
    expected = read(0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(read, range(8))) == [expected] * 8


@pytest.mark.parametrize('fixed', [((0, 0, 101), (0, 0, 101)), ((0, 9, 101),),
                                   ((False, 0, 101),), ((0, 0, 999),)])
def test_invalid_requests_reject_without_mutating_root(fixed):
    _, prepared, root = fixture()
    before = root.select_domains(0, 3)
    for solve in (prepared.global_section, prepared.propagate, root.continue_global_section):
        with pytest.raises((ValueError, IndexError, OverflowError)):
            solve(fixed)
    assert root.select_domains(0, 3) == before


def test_explicit_slice_and_immutability_admission():
    _, prepared, root = fixture()
    family = NativeFiniteFamily.projection(root, projection())
    image = family.materialize()['image']; row = image.row(0)
    for action in (lambda: root.select_domains(0, 3, max_values=1), lambda: row.witness(max_records=1),
                   lambda: image.select_queries(0, 9, max_values=1), lambda: family.inspect(max_records=1)):
        with pytest.raises((ValueError, RuntimeError)):
            action()
    for owner in (prepared, root, family, image, row):
        with pytest.raises(AttributeError):
            owner.replacement = 1
    with pytest.raises(IndexError): row.witness(begin=4)
    with pytest.raises(IndexError): root.select_domains(2, 2)
    with pytest.raises(IndexError): image.row(3)
