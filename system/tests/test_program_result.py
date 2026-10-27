"""Program previews are bounded; native downloads retain every step."""
import asyncio
import json

from rcql import Executor, OutputRef, Program, ProgramStep, Result, SourcePolicy, bind, parse
from rexgraph import RexGraph
from system.serialize import json_value


def program():
    return Program("counts", (
        ProgramStep("one", parse("FROM $r RETURN COUNT(CELLS(1))")),
        ProgramStep("two", parse("FROM $r RETURN $previous"), (("previous", OutputRef("one")),))),
        outputs=(("first", OutputRef("one")), ("second", OutputRef("two"))))


def test_program_preview_bounds_outputs_steps_and_source_dependencies():
    source = RexGraph.from_graph([0], [1])
    result = Executor(sources={"r": source}).execute_program(program())
    preview = json_value(result, max_values=1)
    assert preview["kind"] == "ProgramResult"
    assert preview["output_count"] == preview["step_count"] == 2
    assert preview["values"] == [1] and preview["aliases"] == ["first"]
    assert preview["outputs_truncated"] and preview["steps_truncated"]
    assert preview["steps"] == [{"name": "one", "output_count": 1, "aliases": [None], "execution_records": len(result.step("one").execution)}]
    assert "values" not in preview["steps"][0] and "native_plan" not in preview["steps"][0]
    empty = json_value(result, max_values=0)
    assert not empty["values"] and not empty["steps"] and not empty["dependencies"]
    assert empty["outputs_truncated"] and empty["steps_truncated"] and empty["dependencies_truncated"]
    json.dumps(preview, allow_nan=False)


def test_system_program_native_and_preview_share_the_same_program():
    from system.server.app import QueryRequest, query
    from system.state import sources
    source = RexGraph.from_graph([0], [1]);p = program()
    sources.register("program_test", source)
    parameters = dict(p=p, bindings={"r": bind("r", source, SourcePolicy.allow("*"))}, params={})
    text = 'FROM REX("program_test") RETURN PROGRAM_RUN($p,$bindings,$params)'
    try:
        native = asyncio.run(query(QueryRequest(query=text, params=parameters, result_format="native")))
        received = Result.from_bytes(native.body).values[0]
        preview = asyncio.run(query(QueryRequest(query=text, params=parameters)))
        assert received.values == (1, 1) and len(received.steps) == 2
        assert preview["values"][0]["program_digest"] == received.program_digest
        assert preview["values"][0]["step_count"] == 2
    finally:
        sources.remove("program_test")
