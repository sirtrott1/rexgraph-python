"""Joint context selection retains full participant maps and other alternatives."""
from fractions import Fraction as Q
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("rexgraph_cpp._temporal_program")
_spec = importlib.util.spec_from_file_location("evolution_declarations", Path(__file__).with_name("test_temporal_program.py"))
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
coordinate, function, program = _helper.coordinate, _helper.function, _helper.program


def fixture():
    from rexgraph.relational_runtime import RelationalRuntime
    r, t = coordinate(RelationalRuntime.create(89011))
    r, arrangement = coordinate(r, "arrangement")
    r, identity = program(r, "input")
    r, derivative = program(r, "derivative", (identity,))
    r, zero = function(r, t, (0,))
    r, quadratic = function(r, t, (0, 0, 1))
    r, observed = function(r, t, (0, 2))
    r, held = function(r, t, (0, 2, 0, 1))
    r = r.extend("support", dict(coordinate=arrangement, domains=[(0, 1, True, True)], label="support")); support = r.selected
    contexts = []
    for label in ("a", "b", "c"):
        r = r.extend("context", dict(support=support, meaning=label)); contexts.append(r.selected)
    r = r.extend("function_set", dict(coordinate=arrangement, members=[(101, 1, zero, support, True)], label="original zero use")); a = r.selected
    r = r.extend("function_set", dict(coordinate=arrangement, members=[(201, 1, quadratic, support, True)], label="original quadratic use")); b = r.selected
    families = []
    for context, population, expected in zip(contexts, (a, b, a), (zero, observed, zero), strict=True):
        r = r.extend("fit", dict(function_set=population, context=context, candidates=[identity, derivative], expected=expected, observed_domain=None))
        families.append(r.selected)
    transition = dict(left=contexts[0], right=contexts[1], same_program=True, allowed_pairs=None, participation=[])
    r = r.extend("join", dict(families=families, transitions=[transition]))
    return r, contexts, held, derivative


def test_joint_selected_whole_response_and_cold_premise_lineage():
    from rexgraph.relational_runtime import RelationalRuntime
    parent, contexts, held, derivative = fixture()
    family, original = parent.selected, parent.to_bytes()
    assert not parent.evolution()["forced"]
    selected = parent.extend("select_evolution", dict(family=family, context=contexts[0]))
    choice = selected.selection()
    assert choice["match"]["program"] == derivative and choice["match"]["inputs"][0][2] == 101
    assert choice["evolution"] == family and choice["context"] == contexts[0]
    assert selected.extends(parent)
    child = selected.extend("apply_learned", dict(family=selected.selected, bindings=[(1, held)], label="held whole response"))
    assert child.function()["pieces"][0]["numerator"] == (Q(2), Q(0), Q(3))
    assert parent.to_bytes() == original
    reopened = RelationalRuntime.from_bytes(child.to_bytes())
    assert reopened.to_bytes() == child.to_bytes() and reopened.selection(selected.selected) == choice
    direct = parent.extend("apply_evolution", dict(family=family, context=contexts[0], bindings=[(1, held)], label="direct"))
    assert direct.function()["pieces"][0]["numerator"] == (Q(2), Q(0), Q(3))
    assert direct.extends(parent)


def test_unscoped_unresolved_foreign_and_failed_execution_publish_no_child():
    parent, contexts, held, _ = fixture()
    before, family = parent.to_bytes(), parent.selected
    for operation, spec in (
        ("apply_learned", dict(family=family, bindings=[(1, held)], label="unscoped")),
        ("select_evolution", dict(family=family, context=contexts[2])),
        ("select_evolution", dict(family=family, context=999999)),
        ("apply_evolution", dict(family=family, context=contexts[0], bindings=[], label="missing input")),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            parent.extend(operation, spec)
        assert parent.to_bytes() == before


def test_same_program_and_value_cannot_select_distinct_membership_uses():
    from rexgraph.relational_runtime import RelationalRuntime
    r, t = coordinate(RelationalRuntime.create(89012))
    r, a = coordinate(r, "arrangement")
    r, x = program(r, "input")
    r, d = program(r, "derivative", (x,))
    r, zero = function(r, t, (0,))
    r, other = function(r, t, (0,))
    r = r.extend("support", dict(coordinate=a, domains=[(0, 1, True, True)], label="validity")); support = r.selected
    r = r.extend("context", dict(support=support, meaning="two original uses")); context = r.selected
    r = r.extend("function_set", dict(coordinate=a, members=[(301, 1, zero, support, True),
        (302, 1, other, support, True)], label="same values and program, distinct participants")); population = r.selected
    r = r.extend("fit", dict(function_set=population, context=context, candidates=[d], expected=zero, observed_domain=None)); local = r.selected
    assert len(r.fit()["matches"]) == 2
    r = r.extend("join", dict(families=[local], transitions=[]))
    with pytest.raises(ValueError):
        r.extend("select_evolution", dict(family=r.selected, context=context))


def test_author_facade_cold_adoption_of_selection_and_verified_layer_diagram():
    pytest.importorskip("rexgraph_temporal")
    from rcql import TemporalModel
    r, contexts, held, _ = fixture()
    selected = r.extend("select_evolution", dict(family=r.selected, context=contexts[0]))
    child = selected.extend("apply_learned", dict(family=selected.selected, bindings=[(1, held)], label="held response"))
    facade = TemporalModel.from_relational(child, kind="function")
    assert facade.verify()
    assert facade.to_relational().selection(selected.selected) == selected.selection()
    # Retain an actual author function layer diagram, then verify it in Core.
    m = TemporalModel.create(89013).declare("coordinate", dict(name="t", clock="c", unit="s", origin="0")); t = m.selected
    m = m.declare("function", dict(coordinate=t, pieces=({"numerator": (1, 2, 3)},))); f = m.selected
    m = m.declare("support", dict(coordinate=t, components=((0, 1, True, True),))); support = m.selected
    m = m.product("functions", [(1, f, support, 1, 0, "left"), (2, f, support, 1, 0, "right")])
    retained = m.to_relational()
    info = retained.layer()
    assert info["verified"] and info["compatible"] and len(info["layers"]) == 2
    assert info["diagram"] in retained.inspect()["verified_layer_diagrams"]
    assert [layer["role"] for layer in info["layers"]] == ["left", "right"]
