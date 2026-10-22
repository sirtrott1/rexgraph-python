"""Hive controls capability execution, dependencies and guarded record updates."""
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
from agent.hive import Hive
from agent.hive_workflow import WorkflowCheck as Check
from agent.hive_workflow import WorkflowRef as Ref
from agent.hive_workflow import WorkflowStep as Step


@pytest.fixture
def hive(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    value = Hive("workflow-tests")
    yield value
    value.stop_all()


def test_whole_outputs_pass_by_declared_accession_in_process(hive):
    import threading

    from rexgraph import RexGraph
    original = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    parent_thread = threading.get_ident()
    hive.add_worker("source", lambda data: original, capability="analyze")

    def consume(data):
        assert data["field"] is original
        assert threading.get_ident() != parent_thread
        return {"count": data["field"].nE}
    hive.add_worker("consumer", consume, capability="analyze")
    # Declaration order does not imply execution order.
    result = hive.run_workflow([
        Step("consume", "consumer", {"field": Ref("produce")}),
        Step("produce", "source"),
    ])
    assert result.ok and result.outputs["consume"] == {"count": 2}
    assert result.outputs["produce"] is original
    assert result.steps[0]["dependencies"] == ("produce",)
    assert {"source", "consumer"} <= set(hive._complex.agents())


@pytest.mark.parametrize("verdict,state", [(True, "completed"), (False, "rejected"), (None, "unresolved")])
def test_declared_checks_retain_whole_evidence_and_block_unaccepted_dependents(hive, verdict, state):
    from rcql import Result
    from rexgraph import RexGraph

    whole = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    observed, writes = [], []
    evidence = Result((verdict, whole), aliases=("accepted", "whole"))
    hive.add_worker("produce", lambda data: whole)

    def observe(data):
        assert data["value"] is whole
        observed.append(data)
        return evidence
    hive.add_worker("observe", observe, capability="analyze")
    hive.add_worker("write", lambda data: writes.append(data), capability="transform")
    result = hive.run_workflow([
        Step("first", "produce", checks=(Check("observe", {"value": Ref("first")}),)),
        Step("write", "write", Ref("first")),
    ])
    assert result.steps[0]["state"] == state
    assert result.steps[0]["checks"][0]["result"] is evidence
    assert result.outputs["first"] is whole and len(observed) == 1
    assert result.ok is (verdict is True)
    assert writes == ([whole] if verdict is True else [])
    assert result.steps[1]["state"] == ("completed" if verdict is True else "skipped")
    # The unchecked write has no result obligation, even when all calls complete.
    assert not result.verified


def test_all_obligations_are_observed_and_a_rejection_is_not_hidden_by_an_open_one(hive):
    from rcql import Result
    calls = []
    hive.add_worker("produce", lambda data: "original")
    for name, verdict in [("reject", False), ("open", None), ("accept", True)]:
        def observe(data, name=name, verdict=verdict):
            calls.append(name)
            return Result((verdict,), aliases=("accepted",))
        hive.add_worker(name, observe)
    result = hive.run_workflow([Step("x", "produce", checks=tuple(Check(name) for name in
        ("reject", "open", "accept")))])
    assert calls == ["reject", "open", "accept"]
    assert result.steps[0]["state"] == "rejected" and result.outputs["x"] == "original"
    assert [o["state"] for o in result.steps[0]["checks"]] == ["rejected", "unresolved", "accepted"]


@pytest.mark.parametrize("bad", [
    True, {"accepted": True},
    ("scalar", 1), ("scalar", "true"),
    ("missing", True), ("duplicate", True), ("length", True),
])
def test_malformed_verdict_fails_closed_without_discarding_the_return(hive, bad):
    from rcql import Result
    if isinstance(bad, tuple):
        kind, value = bad
        aliases = {"scalar": ("accepted",), "missing": ("other",),
                   "duplicate": ("accepted", "accepted"), "length": ("accepted", "other")}[kind]
        values = (value, value) if kind == "duplicate" else (value,)
        bad = Result(values, aliases=aliases)
    writes = []
    hive.add_worker("task", lambda data: b"original\x00\xff")
    hive.add_worker("invalid", lambda data: bad)
    hive.add_worker("write", lambda data: writes.append(data))
    result = hive.run_workflow([Step("task", "task", checks=(Check("invalid"),)),
                                Step("write", "write", Ref("task"))])
    assert result.steps[0]["state"] == "failed" and not result.ok and writes == []
    assert result.outputs["task"] == b"original\x00\xff"
    assert result.steps[0]["checks"][0]["result"] is bad
    assert result.steps[0]["checks"][0]["error"]["type"] == "TypeError"


def test_guard_refusal_prevents_the_operation_and_retains_the_guard(hive):
    from rcql import Result
    calls = []
    hive.add_worker("guard", lambda data: Result((None, data), aliases=("admitted", "original")))
    hive.add_worker("write", lambda data: calls.append(data), capability="transform")
    result = hive.run_workflow([Step("write", "write", {"original": b"\x00\xff"},
        guards=(Check("guard", {"original": b"\x00\xff"}, "admitted"),))])
    assert not result.ok and not result.verified and calls == [] and not result.outputs
    assert result.steps[0]["state"] == "unresolved"
    assert result.steps[0]["guards"][0]["result"].values[1] == {"original": b"\x00\xff"}


@pytest.mark.parametrize("steps", [
    [Step("task", "worker", checks=(Check("missing"),))],
    [Step("task", "worker", guards=(Check("observer", Ref("task")),))],
    [Step("task", "worker", checks=(Check("observer", Ref("missing")),))],
    [Step("a", "worker", checks=(Check("observer", Ref("b")),)),
     Step("b", "worker", checks=(Check("observer", Ref("a")),))],
])
def test_check_workers_and_reference_cycles_are_validated_before_any_call(hive, steps):
    calls = []
    hive.add_worker("worker", lambda data: calls.append("worker"))
    hive.add_worker("observer", lambda data: calls.append("observer"))
    with pytest.raises(ValueError):
        hive.run_workflow(steps)
    assert calls == []


def test_changed_handler_on_the_same_worker_is_refused(hive):
    calls = []
    bee = hive.add_worker("consumer", lambda data: calls.append("original"))

    def rebind(data):
        bee._handler = lambda data: calls.append("replacement")
        return "input"
    hive.add_worker("rebind", rebind)
    result = hive.run_workflow([Step("rebind", "rebind"), Step("consumer", "consumer", Ref("rebind"))])
    assert not result.ok and not calls
    assert result.steps[1]["error"]["type"] == "RuntimeError"


def test_pinned_rcql_checks_named_outputs_and_cold_database_copy(hive, tmp_path):
    from agent import RelationalSession
    from rcdb import copy_record, open_store
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
    from rexgraph import RexGraph

    uri = "local://" + str(tmp_path / "records")
    store = open_store(uri)
    session = RelationalSession(store)
    session.publish("source", RexGraph.from_cells([3, [[0, 1], [1, 2]]]), expected_version=0)
    program = Program("observed_count", (ProgramStep("observe", parse(
        "FROM $r RETURN COUNT(CELLS(1)) = $expected, COUNT(CELLS(1))")),),
        (ProgramInput("expected", "ExactInteger"),),
        (("accepted", OutputRef("observe", 0)), ("actual", OutputRef("observe", 1))))
    session.publish_program("check", program, expected_version=0)
    hive.add_program("observer", session, "check", version=1,
                     sources={"r": {"record_id": "source", "version": 1}})
    hive.add_worker("produce", lambda data: Result((2,), aliases=("expected",)))
    # New source history must not silently retarget the pinned observer.
    session.publish("source", RexGraph.from_cells([2, [[0, 1]]]), expected_version=1)
    result = hive.run_workflow([Step("task", "produce", checks=(Check("observer",
        {"expected": Ref("task", ("expected",))}),))])
    assert result.ok and result.verified
    check = result.steps[0]["checks"][0]["result"]
    assert check.named_values == {"accepted": True, "actual": 2}
    assert check.native_plan["record_id"] == "check" and check.native_plan["version"] == 1
    packet = result.to_result()
    payload = packet.to_bytes()
    session.save_result("outcome", packet, expected_version=0)
    store.close()
    store = open_store(uri)
    recovered = RelationalSession(store).read("outcome").value
    assert recovered.to_bytes() == payload
    trace = recovered.values[0]["steps"][0]
    assert Result.from_bytes(trace["checks"][0]["result"]["payload"]).to_bytes() == check.to_bytes()
    target = open_store("local://" + str(tmp_path / "copy"))
    copy_record(store, target, store.get_record("outcome"))
    assert RelationalSession(target).read("outcome").value.to_bytes() == payload
    target.close()
    store.close()


def test_duplicate_result_alias_refuses_before_consumer_runs(hive):
    from rcql import Result
    calls = []
    hive.add_worker("producer", lambda data: Result((1, 2), aliases=("x", "x")))
    hive.add_worker("consumer", lambda data: calls.append(data))
    result = hive.run_workflow([Step("a", "producer"), Step("b", "consumer", Ref("a", ("x",)))])
    assert not result.ok and not calls
    assert result.steps[1]["error"]["type"] == "ValueError"


@pytest.mark.parametrize("block", [False, True])
def test_typed_program_guard_and_exact_whole_field_residual(hive, block):
    from dataclasses import replace

    from agent import RelationalSession
    from rcdb import MemoryStore
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, parse
    from rexgraph import RexGraph
    from rexgraph.model_runtime import native_model
    from rexgraph.tensor_field import FieldSource, TensorField
    from rexgraph.type_accession import CoordinateSpace

    session = RelationalSession(MemoryStore())
    session.publish("source", RexGraph.from_cells([4, [[0, 1], [1, 2], [2, 3]]]), expected_version=0)
    snapshot = session.read("source", 1)
    source = FieldSource(snapshot.value, "source", 1, snapshot.state_digest)
    model = native_model(snapshot.value, reference=source)
    axes = (CoordinateSpace("uses", ("first", "second")),) if block else ()
    values = [[1, 2], [0, 0], [Q(1, 3), Q(2, 3)]] if block else [1, 0, Q(1, 3)]
    x = TensorField(model.space, values, axes=axes, source=source, grade=1, variance="chain")
    guard = Program("admit_chain", (ProgramStep("ports", parse("FROM $r RETURN true,$x")),),
        (ProgramInput("x", "TensorField", 1, "chain"),),
        (("accepted", OutputRef("ports")), ("input", OutputRef("ports", 1))))
    transform = Program("response", (ProgramStep("response", parse("FROM $r RETURN NATIVE_RESPONSE($x)")),),
        (ProgramInput("x", "TensorField", 1, "chain"),), (("response", OutputRef("response")),))
    check = Program("verify_response", (ProgramStep("certificate", parse(
        "FROM $r LET cert=MODEL_CERTIFY($model,$x,$candidate) RETURN cert.exact_solution,cert")),),
        (ProgramInput("model", "ModelState"), ProgramInput("x", "TensorField", 1, "chain"),
         ProgramInput("candidate", "TensorField", 1, "chain")),
        (("accepted", OutputRef("certificate")), ("certificate", OutputRef("certificate", 1))))
    for name, program in (("guard", guard), ("transform", transform), ("check", check)):
        session.publish_program(name, program, expected_version=0)
        hive.add_program(name, session, name, version=1,
                         sources={"r": {"record_id": "source", "version": 1}})
    writes = []
    hive.add_worker("write", lambda data: writes.append(data), capability="transform")

    def run(worker, value):
        return hive.run_workflow([Step("solve", worker, {"x": value},
            guards=(Check("guard", {"x": value}),),
            checks=(Check("check", {"model": model, "x": value,
                                    "candidate": Ref("solve", ("response",))}),)),
            Step("write", "write", Ref("solve", ("response",)))])

    first = run("transform", x)
    assert first.ok and first.steps[0]["validation"] == "accepted" and len(writes) == 1
    certificate = first.steps[0]["checks"][0]["result"].named_values["certificate"]
    assert certificate["exact_solution"] and certificate["residual"].space == x.space
    assert certificate["residual"].axes == x.axes
    assert all(v == 0 for v in certificate["residual"].values.flat)
    if block:
        assert certificate["quadrance_bound"].contract_support().shape == (2, 2)
    # An equally shaped input with the wrong declared variance cannot reach the operator.
    refused = run("transform", replace(x, variance="cochain"))
    assert refused.steps[0]["state"] == "failed" and not refused.outputs and len(writes) == 1
    assert refused.steps[0]["guards"][0]["error"]["type"] == "TypeError"
    session.publish("other-source", snapshot.value, expected_version=0)
    foreign = session.read("other-source", 1)
    foreign_input = replace(x, source=FieldSource(foreign.value, "other-source", 1, foreign.state_digest))
    refused = run("transform", foreign_input)
    assert refused.steps[0]["state"] == "failed" and not refused.outputs and len(writes) == 1
    assert refused.steps[0]["guards"][0]["error"]["type"] == "ValueError"
    from rcql import Result
    hive.add_worker("incorrect", lambda data: Result((data["x"],), aliases=("response",)))
    wrong = run("incorrect", x)
    assert wrong.steps[0]["state"] == "rejected" and len(writes) == 1
    assert wrong.outputs["solve"].named_values["response"] is x
    assert not wrong.steps[0]["checks"][0]["result"].named_values["certificate"]["exact_solution"]
    next_values = [[0, 0], [Q(2, 5), Q(1, 5)], [1, 2]] if block else [0, Q(2, 5), 1]
    second = run("transform", replace(x, values=next_values))
    assert second.ok and len(writes) == 2
    assert first.outputs["solve"].values[0].coefficient_digest != second.outputs["solve"].values[0].coefficient_digest
    restored = Result.from_bytes(wrong.to_result().to_bytes()).values[0]
    observation = Result.from_bytes(restored["steps"][0]["checks"][0]["result"]["payload"])
    assert observation.named_values["certificate"]["residual"].coefficient_digest == (
        wrong.steps[0]["checks"][0]["result"].named_values["certificate"]["residual"].coefficient_digest)


def test_outcome_codec_refuses_unsupported_handles_and_cycles(hive):
    hive.add_worker("handle", lambda data: object())
    result = hive.run_workflow([Step("handle", "handle")])
    with pytest.raises(TypeError, match="portable codec"):
        result.to_result().to_bytes()
    loop = []
    loop.append(loop)
    hive.add_worker("cycle", lambda data: loop)
    result = hive.run_workflow([Step("cycle", "cycle")])
    with pytest.raises(ValueError, match="acyclic"):
        result.to_result()


def test_failed_model_decision_blocks_writes_and_preserves_independent_results(hive):
    calls = []

    def invalid(data):
        raise ValueError("model did not select an admitted action")
    hive.add_worker("decision", invalid, capability="predict")
    hive.add_worker("write", lambda data: calls.append(data), capability="transform")
    hive.add_worker("independent", lambda data: "retained", capability="analyze")
    result = hive.run_workflow([Step("decision", "decision"),
        Step("write", "write", Ref("decision")), Step("audit", "independent")])
    assert not result.ok and calls == []
    assert result.outputs == {"audit": "retained"}
    assert result.steps[0]["error"]["type"] == "ValueError"
    assert result.steps[1]["state"] == "skipped"
    assert result.steps[1]["blocked_by"] == ("decision",)


@pytest.mark.parametrize("bad", [
    [Step("write", "worker"), Step("later", "worker", Ref("missing"))],
    [Step("write", "worker"), Step("later", "worker", Ref("later"))],
    [Step("write", "worker"), Step("later", "missing")],
    [Step("write", "worker"), Step("write", "worker")],
])
def test_invalid_workflow_refuses_before_any_worker_runs(hive, bad):
    calls = []
    hive.add_worker("worker", lambda data: calls.append(data))
    with pytest.raises(ValueError):
        hive.run_workflow(bad)
    assert calls == []


def test_scheduler_failure_does_not_replay_an_already_completed_write(hive, monkeypatch):
    calls = []
    hive.add_worker("worker", lambda data: calls.append(data) or data, capability="transform")

    def partially_run(units, placement, **kwargs):
        units[0]["fn"]()
        raise OSError("scheduler failed after the first publication")
    coordinator = SimpleNamespace(cost=None, plan=lambda units: {u["id"]: "thread" for u in units},
        pools=SimpleNamespace(run=partially_run))
    monkeypatch.setattr(hive, "_coordinator_obj", lambda: coordinator)
    result = hive.run_workflow([Step("first", "worker", 1), Step("second", "worker", 2),
        Step("dependent", "worker", Ref("first"))])
    assert calls == [1] and result.outputs == {"first": 1}
    assert not result.ok
    assert result.steps[1]["scheduler_error"]["type"] == "OSError"
    assert result.steps[2]["state"] == "skipped"


def test_bad_output_path_fails_consumer_before_call(hive):
    calls = []
    hive.add_worker("producer", lambda data: {"actual": [3]})
    hive.add_worker("consumer", lambda data: calls.append(data))
    result = hive.run_workflow([Step("first", "producer"),
        Step("bad", "consumer", Ref("first", ("missing",)))])
    assert not result.ok and calls == []
    assert result.steps[1]["error"]["type"] == "KeyError"


def test_hive_orchestrates_real_program_revision_history_and_complete_results(hive, tmp_path, monkeypatch):
    import base64

    from agent.mcp_tools import Context

    from agent import rcdb
    from rcdb import open_store
    from rcql import Result
    store = open_store("local://" + str(tmp_path / "records"))
    monkeypatch.setattr(rcdb, "default_store", lambda: store)
    context = Context(workspace="alpha", identity="alice", is_admin=False, auth_enabled=True)
    names = ["rexgraph_rcdb_publish", "rexgraph_rcql_publish_program", "rexgraph_rcql_run_program",
             "rexgraph_rcql_transform_program"]
    hive.add_tools(context=context, names=names)
    declaration = {"name": "Quote", "inputs": [{"name": "x", "kind": "ExactRational"},
        {"name": "fee", "kind": "ExactRational"}], "steps": [
        {"name": "subtotal", "query": "FROM $data RETURN SUM([$x,$fee])"},
        {"name": "twice", "query": "FROM $data RETURN SUM([$subtotal,$subtotal])",
         "inputs": {"subtotal": {"step": "subtotal", "output": 0}}}]}
    sources = {"data": {"record_id": Ref("data", ("record", "id")),
                         "version": Ref("data", ("record", "version"))}}
    identity = {"record_id": Ref("program", ("record", "id")),
                "version": Ref("program", ("record", "version"))}
    parameters = {"x": {"numerator": 3, "denominator": 2}, "fee": {"numerator": 1, "denominator": 4}}
    result = hive.run_workflow([
        Step("data", "rexgraph_rcdb_publish", {"record_id": "data", "cells": [3, [[0, 1], [1, 2]]], "expected_version": 0}),
        Step("program", "rexgraph_rcql_publish_program", {"record_id": "quote", "declaration": declaration, "expected_version": 0}),
        Step("before", "rexgraph_rcql_run_program", {**identity, "parameters": parameters, "sources": sources, "save_as": "before"}),
        Step("changed", "rexgraph_rcql_transform_program", {**identity, "rule": "specialize",
            "arguments": {"fee": {"numerator": 3, "denominator": 4}},
            "parameters": {"x": parameters["x"]}, "sources": sources}, needs=("before",)),
        Step("after", "rexgraph_rcql_run_program", {"record_id": Ref("changed", ("record", "id")),
            "version": Ref("changed", ("record", "version")), "parameters": {"x": parameters["x"]},
            "sources": sources, "save_as": "after"}),
        Step("replay", "rexgraph_rcql_run_program", {**identity, "parameters": parameters, "sources": sources,
             "save_as": "replay"}, needs=("after",)),
    ])
    assert result.ok, result.steps
    for name, expected in [("before", Q(7, 2)), ("after", Q(9, 2)), ("replay", Q(7, 2))]:
        packet = base64.b64decode(result.outputs[name]["result_base64"])
        value = Result.from_bytes(packet)
        assert value.values == (expected,)
        assert len(value.native_plan["step_results"]) == 2
        assert store.read_record(name).value.to_bytes() == packet
        assert store.get_record(name).meta["stored_by"] == "alice"
    assert len(store.history("quote")) == 2
    assert store.verify_commits("quote")
    store.close()


def test_workflow_tools_keep_workspace_boundary_and_stale_write_guard(hive, monkeypatch):
    from agent.mcp_tools import Context

    from agent import RelationalSession, rcdb
    from rcdb import MemoryStore
    from rexgraph import RexGraph
    store = MemoryStore()
    monkeypatch.setattr(rcdb, "default_store", lambda: store)
    other = RelationalSession(store, context=Context(workspace="beta", identity="bob", auth_enabled=True))
    other.publish("secret", RexGraph.from_cells([2, [[0, 1]]]), expected_version=0)
    context = Context(workspace="alpha", identity="alice", auth_enabled=True, is_admin=False)
    hive.add_tools(context=context, names=["rexgraph_rcdb_inspect", "rexgraph_rcdb_publish"])
    result = hive.run_workflow([
        Step("foreign", "rexgraph_rcdb_inspect", {"record_id": "secret"}),
        Step("hidden-copy", "rexgraph_rcdb_publish", {"record_id": "copy", "value_record": {"record_id": "secret"},
            "expected_version": 0}, needs=("foreign",)),
        Step("create", "rexgraph_rcdb_publish", {"record_id": "ours", "cells": [2, [[0, 1]]], "expected_version": 0}),
        Step("stale", "rexgraph_rcdb_publish", {"record_id": "ours", "cells": [2, [[0, 1]]], "expected_version": 0}, needs=("create",)),
    ])
    assert not result.ok
    assert result.steps[0]["error"]["type"] == "KeyError"
    assert result.steps[1]["state"] == "skipped" and store.get_record("copy") is None
    assert result.steps[3]["error"]["type"] == "VersionConflictError"
    assert len(store.history("ours")) == len(store.history("secret")) == 1
