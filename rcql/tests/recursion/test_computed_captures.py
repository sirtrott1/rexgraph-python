"""Computed function captures are values at runtime and descriptors during planning."""
from dataclasses import replace

import pytest

from rcql import (Executor, NameRelation, ProgramInput, RecursiveDefinition, RecursiveProgram,
                  parse, recur)
from rcql.ast import Call, Comparison, ListExpr, Literal, Parameter
from .helpers import source


@pytest.mark.parametrize("modifier", ["NAME_BIND", "NAME_REBIND"])
@pytest.mark.parametrize("nested", [False, True])
def test_computed_capture_plans_without_serializing_types(modifier, nested, monkeypatch):
    from .helpers import binding
    from rcql.operators import _REGISTRY
    op = "COUNT" if nested else "SUM"
    seeded = NameRelation.operator(op).bind("values", [10])
    target = seeded if modifier == "NAME_REBIND" else NameRelation.operator(op)
    values = "[[NAME_APPLY($seed,[]),2]]" if nested else "[NAME_APPLY($seed,[]),2]"
    text = f'FROM $r LET revised={modifier}($target,"values",{values}) RETURN NAME_APPLY(revised,[])'
    r = source()
    executor = Executor(sources={"r": r}, params={"seed": seeded, "target": target})
    entry = _REGISTRY[op]
    with monkeypatch.context() as scope:
        scope.setitem(_REGISTRY, op, replace(entry, fn=lambda *a: pytest.fail("EXPLAIN executed a capture")))
        explained = executor.execute(parse("EXPLAIN " + text))
        assert explained.execution == ()
    assert executor.execute(parse(text)).values == ((1 if nested else 12),)
    assert seeded.apply(binding(r), []) == (1 if nested else 10)


def test_recursive_accumulator_revises_function_with_computed_value():
    n, op = Parameter("n"), Parameter("operation")
    value = Call("NAME_APPLY", (op, ListExpr(())))
    revised = Call("NAME_REBIND", (op, Literal("values"), ListExpr((value, n))))
    previous = Call("SUM", (ListExpr((n, Literal(-1))),))
    program = RecursiveProgram("Accumulator", (RecursiveDefinition("accumulate",
        (ProgramInput("n", "ExactInteger"), ProgramInput("operation", "NameRelation")),
        Comparison("==", n, Literal(0)), value, recur("accumulate", previous, revised),
        ProgramInput("result", "ExactInteger"), decreases="n"),))
    seed = NameRelation.operator("SUM").bind("values", [0])
    result = Executor(sources={"r": source()}, params={"program": program, "seed": seed}).execute(parse(
        'FROM $r LET result=RECURSIVE_RUN($program,"accumulate",[10,$seed]) '
        'RETURN RECURSIVE_VALUE(result),NAME_APPLY($seed,[])'))
    assert result.values == (55, 0)
