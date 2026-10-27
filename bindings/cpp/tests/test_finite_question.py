"""Answer projections and observable fibers over complete native joint images."""
import gc
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_finite_claim_verification import family
from test_relational_finite_field import asset, declaration

from rexgraph_cpp import FiniteSearchCancellation, NativeFiniteFamily, NativeFiniteSource


def test_question_keeps_joint_answers_and_certifies_a_sufficient_observation():
    source, view = family()
    image = view.materialize()["image"]
    result = image.analyze_question(((0, 0), (0, 2)), ((0, 2),))
    assert result["status"] == "alternatives" and result["answer"] is None
    assert result["answer_count"] == result["observation_count"] == 3
    assert result["max_remaining_answers"] == 1
    assert result["observation_determines_answer"] is True
    assert result["obstruction"] is None
    assert image.classify_equalities((((0, 0, 101), (0, 2, 102)),))[0]["status"] == "refuted"
    original = source.archive()
    del source, view
    gc.collect()
    assert image.family().source().archive() == original
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: image.analyze_question(((0, 0), (0, 2)), ((0, 2),)), range(8))) == [result] * 8


def test_obstruction_is_an_actual_pair_in_one_observation_fiber():
    _, view = family()
    image = view.materialize()["image"]
    result = image.analyze_question(((0, 2),))
    assert result["status"] == "alternatives"
    assert result["answer_count"] == result["max_remaining_answers"] == 3
    assert result["observation_count"] == 1
    assert result["observation_determines_answer"] is False
    left, right = result["obstruction"]
    assert image.row(left).inspect()["tuple"][1] != image.row(right).inspect()["tuple"][1]
    assert image.row(left).family().same_owner(image.family())


def test_unique_answer_does_not_require_a_unique_whole_section():
    d = declaration()
    d["charts"] = (dict(d["charts"][0], support=((0, 0),), allowed=((102,),)),)
    source = NativeFiniteSource(asset(), d)
    view = NativeFiniteFamily.projection(source, dict(key=701, revision=1, cells=((0, 2), (0, 0)),
        window=None, fixed=(), distinct_relations=()))
    image = view.materialize()["image"]
    result = image.analyze_question(((0, 0),), ((0, 2),))
    assert image.inspect()["row_count"] == 3
    assert result["status"] == "unique" and result["answer"] == (102,)
    assert result["answer_count"] == 1 and result["observation_count"] == 3
    assert result["observation_determines_answer"] is True


def test_question_order_is_the_declared_order():
    d = declaration()
    d["charts"] = (dict(d["charts"][0], support=((0, 0), (0, 2)), allowed=((101, 103),)),)
    view = NativeFiniteFamily.projection(NativeFiniteSource(asset(), d), dict(key=701, revision=1,
        cells=((0, 0), (0, 2)), window=None, fixed=(), distinct_relations=()))
    assert view.materialize()["image"].analyze_question(((0, 2), (0, 0)))["answer"] == (103, 101)


def test_empty_and_unprojected_questions_never_supply_an_answer():
    _, view = family(window=((), (101, 102, 103)))
    empty = view.materialize()["image"].analyze_question(((0, 0),))
    assert empty["status"] == "inconsistent_family" and empty["answer"] is None
    assert empty["answer_count"] == 0 and empty["observation_determines_answer"] is None
    _, view = family()
    unsupported = view.materialize()["image"].analyze_question(((0, 1),))
    assert unsupported["status"] == "unsupported" and unsupported["answer_count"] is None


@pytest.mark.parametrize("kwargs,status", [({"max_checks": 0}, "budget_exceeded"), ({"max_checks": 9}, "budget_exceeded"),
    ({"cancel": True}, "cancelled")])
def test_interrupted_analysis_never_returns_partial_counts_or_descent(kwargs, status):
    _, view = family()
    image = view.materialize()["image"]
    before = image.proof_archive()
    if kwargs.pop("cancel", False):
        token = FiniteSearchCancellation()
        token.cancel()
        kwargs["cancellation"] = token
    result = image.analyze_question(((0, 0),), ((0, 2),), **kwargs)
    assert result["status"] == status
    assert all(result[k] is None for k in ("answer", "answer_count", "observation_count", "max_remaining_answers",
        "observation_determines_answer", "obstruction"))
    assert image.proof_archive() == before


@pytest.mark.parametrize("question,observation", [((), ()), (((0, 0), (0, 0)), ()), (((True, 0),), ()),
    (((0, 0, 101),), ()), (((0, 0),), ((0, 2), (0, 2)))])
def test_malformed_maps_are_rejected(question, observation):
    _, view = family()
    image = view.materialize()["image"]
    with pytest.raises((ValueError, TypeError, OverflowError)):
        image.analyze_question(question, observation)
