"""Live program resources belong to worker registration, outside stored inputs."""
from fractions import Fraction as Q

import pytest
from agent.hive import Hive
from agent.hive_workflow import WorkflowStep

from agent import RelationalSession
from rcdb import MemoryStore
from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
from rexgraph import RexGraph


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    session = RelationalSession(MemoryStore())
    session.publish("data", RexGraph.from_cells([2, [[0, 1]]]), expected_version=0)
    program = Program("current", (ProgramStep("sum", parse("FROM $r RETURN SUM([$x,$offset])")),),
        (ProgramInput("x", "ExactInteger"), ProgramInput("offset", "ExactInteger")),
        (("value", OutputRef("sum")),))
    session.publish_program("program", program, expected_version=0)
    hive = Hive("runtime-program-inputs")
    yield hive, session, program, {"r": {"record_id": "data", "version": 1}}
    hive.stop_all()
    session.store.close()


def test_current_resolvers_are_copied_called_per_invocation_and_never_captured(setup):
    hive, session, program, sources = setup
    current = {"offset": 2}
    calls = []
    opaque = object()

    def resolve():
        calls.append(opaque)
        return current["offset"]

    resolvers = {"offset": resolve}
    original = program.to_bytes()
    bee = hive.add_program("program", session, "program", version=1, sources=sources,
        runtime_parameters=resolvers)
    resolvers["offset"] = lambda: 999
    sources["r"]["version"] = 99
    assert hive.invoke("program", {"x": 3}).values == (5,)
    current["offset"] = 7
    workflow = hive.run_workflow([WorkflowStep("run", "program", {"x": 3})])
    assert workflow.outputs["run"].values == (10,)
    assert calls == [opaque, opaque]
    packet = workflow.to_result().to_bytes()
    assert Result.from_bytes(packet).to_bytes() == packet
    assert program.to_bytes() == original
    assert session.read("program", 1).value is not None
    assert bee._context is session.context
    assert {"offset", "x"} == {p.name for p in program.inputs}


@pytest.mark.parametrize("value", [None, 2, "replacement"])
def test_caller_collision_is_refused_before_any_resolver_or_program(setup, value):
    hive, session, _, sources = setup
    calls = []
    hive.add_program("program", session, "program", version=1, sources=sources,
        runtime_parameters={"offset": lambda: calls.append("called") or 2})
    with pytest.raises(ValueError, match="registration-owned"):
        hive.invoke("program", {"x": 3, "offset": value})
    assert calls == []


@pytest.mark.parametrize("bindings,error", [
    ([], TypeError), ("offset", TypeError),
    ({"undeclared": lambda: 1}, ValueError), ({"offset": 1}, ValueError),
])
def test_runtime_binding_names_and_resolvers_are_explicit(setup, bindings, error):
    hive, session, _, sources = setup
    with pytest.raises(error, match="runtime parameters"):
        hive.add_program("program", session, "program", version=1, sources=sources,
            runtime_parameters=bindings)
    assert hive.get("program") is None


def test_resolver_failure_and_input_contract_do_not_fall_back(setup):
    hive, session, _, sources = setup

    def unavailable():
        raise LookupError("current resource unavailable")

    hive.add_program("missing", session, "program", version=1, sources=sources,
        runtime_parameters={"offset": unavailable})
    with pytest.raises(LookupError, match="current resource"):
        hive.invoke("missing", {"x": 3})
    hive.add_program("wrong", session, "program", version=1, sources=sources,
        runtime_parameters={"offset": lambda: Q(1, 2)})
    with pytest.raises(TypeError, match="declared type"):
        hive.invoke("wrong", {"x": 3})
