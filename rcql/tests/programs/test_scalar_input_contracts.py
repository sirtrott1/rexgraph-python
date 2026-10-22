"""Primitive program ports retain declared types through links and specialization."""
from fractions import Fraction as Q

import pytest

from rcql import Executor, OutputRef, Program, ProgramInput, ProgramStep, parse
from rexgraph import RexGraph


def program(kind):
    return Program("TypedScalar", (
        ProgramStep("first", parse("FROM $r RETURN $value")),
        ProgramStep("linked", parse("FROM $r RETURN $item"), (("item", OutputRef("first")),),
                    (ProgramInput("item", kind),))), (ProgramInput("value", kind),))


@pytest.mark.parametrize("kind,value", [
    ("Text", "record/Ω"), ("Text", ""), ("Bytes", b"\x00\xffrecord"), ("Bytes", b""),
    ("Boolean", True), ("ExactInteger", 2**150), ("ExactRational", Q(2**150, 7)),
])
def test_typed_scalar_inputs_links_and_sealed_captures(kind, value):
    r = RexGraph.from_cells([2, [[0, 1]]])
    declaration = program(kind)
    executor = Executor(sources={"r": r}, params={"value": value})
    assert declaration.execute(executor).values == (value,)
    assert declaration.execute(executor, explain=True)["evaluation"].startswith("none")
    captured = Program.from_bytes(declaration.specialize({"value": value}).to_bytes())
    assert captured.execute(Executor(sources={"r": r})).values == (value,)


@pytest.mark.parametrize("kind,value", [("Text", b"text"), ("Bytes", "bytes"), ("Text", 7), ("Bytes", 7)])
def test_typed_text_and_bytes_refuse_other_scalar_types(kind, value):
    r = RexGraph.from_cells([2, [[0, 1]]])
    executor = Executor(sources={"r": r}, params={"value": value})
    with pytest.raises(TypeError, match="declared type"):
        program(kind).execute(executor)
