"""Program downloads retain complete steps and their shared source owners."""
from dataclasses import replace
from fractions import Fraction as Q
import hashlib

import pytest

from rcql import Executor, OutputRef, Program, ProgramStep, Result, parse
from rexgraph import RexGraph
from rexgraph.value_codec import pack_value, unpack_value


def executed():
    source = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    source.attach_metadata(1, 0, "reading", Q(1, 7))
    program = Program("whole_fields", (
        ProgramStep("first", parse("FROM $r RETURN APPLY(DIRAC(),GRADED_CHAIN([SHARE(CELL(1,0))]),true)")),
        ProgramStep("second", parse("FROM $r RETURN APPLY(ANTI_DIRAC(),$previous,true)"),
                    (("previous", OutputRef("first")),))),
        outputs=(("first", OutputRef("first")), ("second", OutputRef("second"))))
    return Executor(sources={"r": source}).execute_program(program)


def test_program_run_download_preserves_all_steps_and_shared_owners(monkeypatch):
    original = executed()
    raw = Result((original,), aliases=("program",)).to_bytes()
    monkeypatch.setattr(Executor, "execute", lambda *a, **k: pytest.fail("decode executed a query"))
    monkeypatch.setattr(Executor, "execute_program", lambda *a, **k: pytest.fail("decode executed a program"))
    restored = Result.from_bytes(raw).values[0]
    assert restored.program_digest == original.program_digest
    assert restored.aliases == original.aliases and restored.sources == original.sources
    assert tuple(name for name, _ in restored.steps) == ("first", "second")
    owner = restored.values[0].source
    assert restored.values[1].source is owner
    assert all(step.values[0].source is owner for _, step in restored.steps)
    assert restored.dependencies[0].source is owner
    assert owner.get_metadata(1, 0, "reading") == Q(1, 7)
    for old, new in zip(original.values, restored.values, strict=True):
        assert all(a.values.tolist() == b.values.tolist() for a, b in zip(old.components, new.components, strict=True))
    for (_, old), (_, new) in zip(original.steps, restored.steps, strict=True):
        assert old.execution == new.execution and old.provenance == new.provenance
        assert old.exactness == new.exactness and old.native_plan == new.native_plan
    assert Result((restored,), aliases=("program",)).to_bytes() == raw


def test_actual_program_run_query_result_is_portable():
    from rcql import SourcePolicy, bind
    source = RexGraph.from_graph([0], [1])
    program = Program("count", (ProgramStep("one", parse("FROM $r RETURN COUNT(CELLS(1))")),))
    result = Executor(sources={"r": source}, params={"p": program,
        "sources": {"r": bind("r", source, SourcePolicy.allow("*"))}, "params": {}}).execute(
            parse("FROM $r RETURN PROGRAM_RUN($p,$sources,$params)"))
    restored = Result.from_bytes(result.to_bytes())
    assert restored.values[0].values == (1,)
    assert restored.values[0].step("one").values == (1,)
    assert restored.to_bytes() == result.to_bytes()


def test_nested_results_share_the_outer_graph_table_and_metadata():
    source = RexGraph.from_graph([0], [1])
    inner = Result((source,), plan=("retained",), native_plan={"cells": 1},
                   provenance=({"use": "input"},), execution=({"operator": "identity"},), aliases=("source",))
    raw = Result((source, inner)).to_bytes()
    record = unpack_value(raw[37:])
    assert len(record["graphs"]) == 1
    restored = Result.from_bytes(raw)
    assert restored.values[0] is restored.values[1].values[0]
    assert restored.values[1].native_plan == inner.native_plan
    assert restored.to_bytes() == raw


@pytest.mark.parametrize("mutation", ("version", "digest", "aliases", "steps", "sources", "dependencies", "nested-fields"))
def test_program_result_receiving_rejects_malformed_records(mutation):
    raw = Result((executed(),)).to_bytes();record = unpack_value(raw[37:])
    program = list(record["values"][1][0])
    if mutation == "version":program[1] = True
    elif mutation == "digest":program[4] = "unknown"
    elif mutation == "aliases":program[6] = ("one",)
    elif mutation == "steps":
        first = program[3][1][0];program[3] = ("tuple", (first, first))
    elif mutation == "sources":program[5] = ("tuple", (("value", pack_value({"state_digest": "fake"})),))
    elif mutation == "dependencies":program[7] = ("tuple", (("value", pack_value(1)),))
    else:
        steps = list(program[3][1]);name, nested = steps[0][1]
        nested = list(nested);nested[2]["callback"] = "execute"
        steps[0] = ("tuple", (name, tuple(nested)));program[3] = ("tuple", tuple(steps))
    record["values"] = ("tuple", (tuple(program),))
    data = pack_value(record)
    with pytest.raises(ValueError):Result.from_bytes(raw[:5]+hashlib.sha256(data).digest()+data)


def test_program_result_storage_and_copy_preserve_complete_packet():
    from contextlib import closing
    rcdb = pytest.importorskip("rcdb")
    from rcql.result_codec import register_result_storage_codec
    register_result_storage_codec()
    result = Result((executed(),), aliases=("program",))
    with closing(rcdb.open_store("memory://")) as source, closing(rcdb.open_store("memory://")) as destination:
        source.put_record("run", result, codec=rcdb.QUERY_RESULT_CODEC, expected_version=0)
        rcdb.copy_record(source, destination, source.get_record("run"), destination_id="copied")
        assert destination.read_record("copied").value.to_bytes() == result.to_bytes()


def test_program_result_encoder_refuses_incomplete_aliases():
    with pytest.raises(ValueError, match="program result"):
        Result((replace(executed(), aliases=()),)).to_bytes()
