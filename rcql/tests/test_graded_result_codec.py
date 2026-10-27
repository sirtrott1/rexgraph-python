"""Whole graded results retain complete bases, arithmetic and source owners."""
from fractions import Fraction as Q
import hashlib

import numpy as np
import pytest

from rexgraph.cochain import Chain
from rexgraph.graded_boundary import solid_octahedron_3rex
from rexgraph.graph import RexGraph
from rexgraph.value_codec import pack_value, unpack_value
from rexgraph.weighted_dirac import GradedChain, weighted_dirac
from rcql import Executor, Program, ProgramStep, Result, parse
from rcql.result_codec import pack_result, unpack_result


def make():
    return RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2])


@pytest.mark.parametrize("width", [None, 0, 3])
@pytest.mark.parametrize("domain", ["rational", "real", "complex"])
def test_complete_graded_roundtrip_and_continued_operator_execution(width, domain):
    rex = make()
    shape = (2,) if width is None else (2, width)
    value = Q(2, 7) if domain == "rational" else 0.25 if domain == "real" else 1+2j
    array = np.full(shape, value, dtype=object if domain == "rational" else float if domain == "real" else complex)
    original = GradedChain(rex, (Chain(1, array, source=rex),))
    raw = pack_result(Result((rex, original, original.component(1))))
    restored, whole, component = unpack_result(raw).values
    assert whole.source is restored and component.source is restored
    assert all(c.source is restored for c in whole.components)
    assert whole.sizes == original.sizes and whole.exact is original.exact
    for before, after in zip(original.components, whole.components, strict=True):
        assert after.values.dtype == before.values.dtype and after.values.shape == before.values.shape
        np.testing.assert_array_equal(after.values, before.values)
        with pytest.raises(ValueError):
            after.values[...] = 1
    left = weighted_dirac(rex).apply(original, exact=original.exact)
    right = weighted_dirac(restored).apply(whole, exact=whole.exact)
    for a, b in zip(left.components, right.components, strict=True):
        np.testing.assert_array_equal(a.values, b.values)
    assert unpack_result(raw).to_bytes() == raw


def test_shared_and_equal_valued_distinct_owners_are_retained():
    a, b = make(), make()
    x, y = GradedChain(a), GradedChain(b)
    raw = Result((a, x, {"repeat": x}, b, y)).to_bytes()
    left, first, nested, right, second = Result.from_bytes(raw).values
    assert left is not right and first.source is left and second.source is right
    assert nested["repeat"].source is left
    assert all(c.source is left for c in first.components)
    assert all(c.source is right for c in second.components)
    assert len(unpack_value(raw[37:])["graphs"]) == 2
    with pytest.raises(TypeError, match="bound to its source"):
        weighted_dirac(left).apply(second, exact=True)


def test_higher_grades_and_empty_towers_keep_all_explicit_components():
    for rex in (RexGraph.from_cells(solid_octahedron_3rex()), RexGraph.from_graph([], [])):
        state = GradedChain(rex)
        restored = Result.from_bytes(Result((state,)).to_bytes()).values[0]
        assert len(restored.components) == len(state.components)
        assert restored.sizes == state.sizes and restored.exact
        assert all(c.grade == grade for grade, c in enumerate(restored.components))
        assert all(c.values.shape == old.values.shape for c, old in zip(restored.components, state.components, strict=True))


def test_real_program_outputs_and_intermediates_roundtrip_without_execution(monkeypatch):
    from rcql import OutputRef, ProgramInput
    rex = make()
    program = Program("whole_fields", (
        ProgramStep("d", parse("FROM $r RETURN APPLY(DIRAC(),GRADED_CHAIN([SHARE(CELL(1,0))]),true)")),
        ProgramStep("a", parse("FROM $r RETURN APPLY(ANTI_DIRAC(),$x,true)"),
                    (("x", OutputRef("d")),), contracts=(ProgramInput("x", "GradedChain"),))),
        outputs=(("whole", OutputRef("a")),))
    run = Executor(sources={"r": rex}, exactness="exact").execute_program(program)
    original = Result((rex, run.values[0], tuple(result.values[0] for _, result in run.steps)),
                      aliases=("source", "whole", "steps"))
    raw = original.to_bytes()
    monkeypatch.setattr(Executor, "execute", lambda *a, **k: pytest.fail("decoder executed a query"))
    back = Result.from_bytes(raw)
    assert back.to_bytes() == raw and back.values[1].source is back.values[0]
    assert all(field.source is back.values[0] for field in back.values[2])
    assert back.values[1].component(0).values.tolist() == [Q(-1), Q(2), Q(-1)]


@pytest.mark.parametrize("mutation", ["version", "boolean-version", "missing", "reordered", "duplicate",
    "foreign-owner", "missing-source", "subset-basis", "wrong-shape", "block-axes", "cochain", "third-axis"])
def test_malformed_whole_graded_packets_are_refused(mutation):
    rex, foreign = make(), make()
    state = GradedChain(rex, (Chain(1, np.ones((2, 2), dtype=int), source=rex),))
    payload = Result((state, foreign)).to_bytes()
    record = unpack_value(payload[37:])
    nodes = list(record["values"][1]); node = list(nodes[0]); components = list(node[3][1])
    if mutation == "version": node[1] = 2
    elif mutation == "boolean-version": node[1] = True
    elif mutation == "missing-source": node[2] = ("value", pack_value(None))
    elif mutation == "missing": components.pop()
    elif mutation == "reordered": components.reverse()
    elif mutation == "duplicate": components[1] = components[0]
    else:
        component = list(components[1])
        if mutation == "foreign-owner": component[4] = ("graph", 1)
        elif mutation == "subset-basis": component[3] = ("value", pack_value((0, 1)))
        elif mutation == "wrong-shape": component[2] = ("value", pack_value(np.zeros((1, 2), dtype=int)))
        elif mutation == "block-axes": component[2] = ("value", pack_value(np.zeros((2, 3), dtype=int)))
        elif mutation == "cochain": component[0] = "Cochain"
        elif mutation == "third-axis": component[2] = ("value", pack_value(np.zeros((2, 2, 1), dtype=int)))
        components[1] = tuple(component)
    node[3] = ("tuple", tuple(components)); nodes[0] = tuple(node)
    record["values"] = ("tuple", tuple(nodes)); raw = pack_value(record)
    with pytest.raises((ValueError, TypeError)):
        Result.from_bytes(payload[:5] + hashlib.sha256(raw).digest() + raw)


def test_changed_source_population_is_refused_at_encoding():
    rex = make()
    state = GradedChain(rex)
    object.__setattr__(state, "source", RexGraph.from_graph([0], [1]))
    with pytest.raises(ValueError, match="source bases|source grade"):
        Result((state,)).to_bytes()


def test_encoding_does_not_construct_numerical_operator_views(monkeypatch):
    state = GradedChain(make())
    monkeypatch.setattr("rexgraph.linear_operator._boundaries",
                        lambda *a: pytest.fail("numerical boundary view constructed during encoding"))
    assert Result((state,)).to_bytes()
