"""Stored RCQL workers run actual native solves with separately current authority."""
from pathlib import Path

import pytest

pytest.importorskip("rexgraph_cpp._task_outcome")


@pytest.mark.parametrize("mode", ["satisfied", "refuted", "open"])
def test_direct_program_workers_preserve_solve_families_and_current_availability(tmp_path, monkeypatch, mode):
    from agent.hive import Hive
    from agent.hive_workflow import WorkflowCheck as Check
    from agent.hive_workflow import WorkflowRef as Ref
    from agent.hive_workflow import WorkflowStep as Step

    from agent import RelationalSession
    from rcdb import open_store
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
    from rexgraph.cpp_relational_field_admission import (
        available_field_query,
        cpp_available_field_program_model,
    )
    from rexgraph.cpp_relational_field_evidence_admission import (
        available_field_learning_query,
        cpp_available_field_learning_model,
    )
    from rexgraph.io.model_record import model_record
    from rexgraph_cpp import NativeInputUseAsset

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "bindings/cpp/tests"))
    from test_native_task_outcome import contract
    from test_relational_field_admission import context, prepared
    from test_relational_field_evidence import HELD_INPUT, reference
    if mode == "open":
        from test_relational_field_evidence_admission import independent_input
        from test_relational_field_evidence_admission import prepared as root
        owner, program, evidence, origins = root()
        row = independent_input(owner, program, origins, 81, HELD_INPUT)
        authority = context(origins)
        state = cpp_available_field_learning_model(owner, available_evidence=evidence, execution_context=authority)
        inputs = available_field_learning_query(state, (row,))
        original_response = evidence.execute_inputs((row,), authority)
    else:
        owner, program, available, row, origins, authority = prepared()
        state = cpp_available_field_program_model(owner, available_program=available, execution_context=authority)
        inputs = available_field_query(state, (row,))
        original_response = available.execute_inputs((row,), authority)
    target = list(reference(HELD_INPUT, (0, 0, 0, 0, 0))[0])
    if mode == "refuted":
        target[0] += 1
    _, task, binding, authority = contract(original_response, NativeInputUseAsset.from_archive(row["asset"]),
        owner, origins, target, construct=False)
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    session = RelationalSession(open_store("local://" + str(tmp_path / "records")))
    session.publish("model", model_record(owner, state), expected_version=0)
    sources = {"r": {"record_id": "model", "version": 1}}
    producer = Program("native_solve", (
        ProgramStep("execute", parse("FROM $r RETURN MODEL_INFER(MODEL_STATE(0,$authority),$inputs,$authority)")),
        ProgramStep("assess", parse("FROM $r RETURN TASK_OUTCOME($response,$task,$asset,$binding,$authority)"),
            (("response", OutputRef("execute")),))),
        tuple(ProgramInput(n) for n in ("inputs", "task", "asset", "binding", "authority")),
        (("outcome", OutputRef("assess")),))
    observer = Program("native_check", (ProgramStep("check", parse(
        "FROM $r RETURN TASK_VERDICT($outcome,$authority),$outcome,TASK_STATUS($outcome,$authority),TASK_CERTIFICATE($outcome,$authority)")),),
        (ProgramInput("outcome", "TaskOutcome"), ProgramInput("authority")),
        tuple((n, OutputRef("check", i)) for i, n in enumerate(("accepted", "whole", "status", "certificate"))))
    for name, value in (("producer", producer), ("observer", observer)):
        session.publish_program(name, value, expected_version=0)
    hive = Hive("direct-native-programs")
    current = {"authority": authority}
    resolutions = []

    def resolve():
        resolutions.append(current["authority"])
        return current["authority"]

    for name, record in (("solve", "producer"), ("check", "observer")):
        hive.add_program(name, session, record, version=1, sources=sources,
            runtime_parameters={"authority": resolve})

    def publish(data):
        return session.save_result("accepted", Result((data,), aliases=("outcome",))).to_wire_dict()

    hive.add_worker("publish", publish, capability="transform")
    data = {"inputs": inputs, "task": task, "asset": row["asset"], "binding": binding}
    try:
        workflow = hive.run_workflow([
            Step("solve", "solve", data, checks=(Check("check", {"outcome": Ref("solve", ("outcome",))}),)),
            Step("publish", "publish", Ref("solve", ("outcome",)))])
        diagnostic = workflow.steps[0].get("error") or tuple(c.get("error", c.get("state")) for c in workflow.steps[0].get("checks", ()))
        assert workflow.steps[0]["state"] == {"satisfied": "completed", "refuted": "rejected", "open": "unresolved"}[mode], diagnostic
        observed = workflow.outputs["solve"].named_values["outcome"]
        assert observed.inspect(authority)["status"] == mode
        assert observed.native(authority).response(authority).archive() == original_response.archive()
        assert (session.store.get_record("accepted") is not None) is (mode == "satisfied")
        packet = workflow.to_result().to_bytes()
        assert Result.from_bytes(packet).to_bytes() == packet
        assert len(resolutions) == 2
        before = len(resolutions)
        with pytest.raises(ValueError, match="registration-owned"):
            hive.invoke("solve", {**data, "authority": authority})
        assert len(resolutions) == before
        current["authority"] = context(origins, remove=(task["origin"],))
        with pytest.raises(ValueError):
            hive.invoke("check", {"outcome": observed})
        current["authority"] = authority
        assert hive.invoke("check", {"outcome": observed}).named_values["status"] == mode
    finally:
        hive.stop_all()
        session.store.close()
