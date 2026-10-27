"""Complete joint verdicts and native whole-witness replay packets."""
import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_prepared_finite_search import projection
from test_relational_finite_field import asset, declaration

from rexgraph_cpp import NativeFiniteFamily, NativeFiniteSource


def family(**changes):
    source = NativeFiniteSource(asset(), declaration())
    view = projection()
    view.update(changes)
    return source, NativeFiniteFamily.projection(source, view)


def test_classification_preserves_joint_correlations_and_unsupported_carriers():
    source, view = family()
    image = view.materialize()["image"]
    claims = (((0, 0, 101),), ((0, 2, 102),), ((0, 0, 101), (0, 2, 102)),
              (), ((0, 1, 101),), ((0, 0, 999),))
    rows = image.classify_equalities(claims)
    assert tuple(r["status"] for r in rows) == (
        "alternatives", "alternatives", "refuted", "entailed", "unsupported", "unsupported")
    assert (rows[0]["satisfying"], rows[0]["refuting"]) == (1, 2)
    assert (rows[2]["satisfying"], rows[2]["refuting"]) == (0, 3)
    archive = source.archive()
    del source, view
    gc.collect()
    assert image.family().source().archive() == archive
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: image.classify_equalities(claims), range(8))) == [rows] * 8


def test_empty_joint_image_is_not_vacuous_entailment():
    _, view = family(window=((), (101, 102, 103)))
    result = view.materialize()
    assert result["status"] == "complete"
    assert result["image"].classify_equalities(((), ((0, 0, 101),), ((0, 0, 999),))) == (
        dict(status="inconsistent_family", satisfying=0, refuting=0),
        dict(status="inconsistent_family", satisfying=0, refuting=0),
        dict(status="unsupported", satisfying=0, refuting=0))


def test_proof_packet_retains_source_view_all_queries_and_whole_witnesses():
    from test_finite_pruning_proof import decode_proof
    source, view = family(fixed=((0, 1, 102),))
    result = view.materialize()
    image = result["image"]
    packet = image.proof_archive()
    decoded = decode_proof(packet)
    assert decoded['profile'] == b"rexgraph.finite-projection-proof.v2"
    assert decoded['source'] == source.archive()
    assert (decoded['key'], decoded['revision']) == (701, 1)
    assert decoded['cells'] == ((0, 0), (0, 2)) and decoded['window'] is None
    assert decoded['fixed'] == ((0, 1, 102),) and decoded['groups'] == ()
    assert decoded['pruning']['complete'] and decoded['pruning']['window'] == ((102,), (102,))
    count = len(decoded['queries'])
    assert count == image.inspect()["query_count"]
    for recorded, actual in zip(decoded['queries'], image.select_queries(0, count), strict=True):
        assert recorded == dict(actual, status=0 if actual['status'] == 'satisfiable' else 1)
    assert decoded['image'] == (((102, 102), (tuple((0, i, 102) for i in range(3)),)),)
    restored = NativeFiniteSource.from_archive(source.archive())
    replay = NativeFiniteFamily.projection(restored, view.declaration()).materialize()
    assert replay["image"].proof_archive() == packet
    assert image.classify_equalities((((0, 0, 102), (0, 2, 102)),))[0]["status"] == "entailed"


def test_interrupted_packet_preserves_receipts_without_an_image():
    from test_finite_pruning_proof import decode_proof
    _, view = family()
    result = view.materialize(family_limits=(65536, 64000000, 16777216, 65536, 1))
    assert result["status"] == "budget_exceeded" and result["image"] is None
    packet = result["queries"].proof_archive()
    assert decode_proof(packet)['image'] is None
    assert result["queries"].inspect()["query_count"] > 0


def test_propagated_owner_is_classified_but_cannot_be_exported_as_a_raw_recipe():
    from test_prepared_finite_search import fixture
    _, _, root = fixture(((0, 1, 102),))
    view = NativeFiniteFamily.projection(root, projection(((0, 1, 102),)))
    result = view.materialize()
    assert result["image"].classify_equalities((((0, 0, 102),),))[0]["status"] == "entailed"
    for owner in (result["image"], result["queries"]):
        with pytest.raises(ValueError, match="unprepared single-source projection"):
            owner.proof_archive()


@pytest.mark.parametrize("method,kwargs", [
    ("proof_archive", {"max_bytes": 0}),
    ("classify_equalities", {"claims": (((0, 0, 101),),), "max_checks": 0}),
    ("classify_equalities", {"claims": (((True, 0, 101),),)}),
    ("classify_equalities", {"claims": (((0, 0, 101, 4),),)}),
])
def test_native_admission_errors_do_not_change_owning_image(method, kwargs):
    _, view = family()
    image = view.materialize()["image"]
    before = image.proof_archive()
    with pytest.raises((ValueError, TypeError, OverflowError, RuntimeError)):
        getattr(image, method)(**kwargs)
    assert image.proof_archive() == before
