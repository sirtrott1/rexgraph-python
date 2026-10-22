"""Native task assessment through actual RCQL programs, Hive and record copies."""
from pathlib import Path

import pytest

pytest.importorskip("rexgraph_cpp._task_outcome")


@pytest.mark.parametrize("refuted", [False, True])
def test_native_owner_reaches_rcql_checks_hive_mutation_gate_and_cold_rcdb(tmp_path, monkeypatch, refuted):
    from agent.hive import Hive
    from agent.hive_workflow import WorkflowCheck as Check
    from agent.hive_workflow import WorkflowRef as Ref
    from agent.hive_workflow import WorkflowStep as Step

    from agent import RelationalSession
    from rcdb import copy_record, open_store
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
    from rexgraph.cpp_relational_field_admission import (
        available_field_query,
        cpp_available_field_program_model,
        read_available_field_result,
    )
    from rexgraph.io.model_record import model_record
    from rexgraph_cpp import NativeInputUseAsset

    # Only fixture modules enter sys.path; installed runtime packages stay installed.
    fixtures = str(Path(__file__).resolve().parents[2] / "bindings/cpp/tests")
    monkeypatch.syspath_prepend(fixtures)
    from test_native_task_outcome import contract
    from test_relational_field_admission import prepared
    from test_relational_field_evidence import HELD_INPUT, reference
    owner, program, available, row, origins, original_context = prepared()
    state = cpp_available_field_program_model(owner, available_program=available, execution_context=original_context)
    selected = available_field_query(state, (row,))
    # Goal fixed before any Hive task; it is never copied from a produced response.
    target = list(reference(HELD_INPUT, (0, 0, 0, 0, 0))[0])
    if refuted:
        target[0] += 1
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    uri = "local://" + str(tmp_path / "records")
    store = open_store(uri)
    session = RelationalSession(store)
    session.publish("model", model_record(owner, state), expected_version=0)
    sources = {"r": {"record_id": "model", "version": 1}}
    observer = Program("observe_task", (ProgramStep("observe", parse(
        "FROM $r RETURN TASK_VERDICT($outcome,$authority),$outcome,TASK_STATUS($outcome,$authority),"
        "TASK_CERTIFICATE($outcome,$authority),TASK_FIELD($outcome,3,$authority)")),),
        (ProgramInput("outcome", "TaskOutcome"), ProgramInput("authority")),
        tuple((name, OutputRef("observe", i)) for i, name in enumerate(("accepted", "outcome", "status", "certificate", "field"))))
    session.publish_program("observer", observer, expected_version=0)
    hive = Hive("native-task-workflow")
    current = {}

    def execute(data):
        inferred = session.query("FROM $r RETURN MODEL_INFER(MODEL_STATE(0,$authority),$inputs,$authority)",
            parameters={"inputs": selected, "authority": original_context}, sources=sources).values[0]
        actual = read_available_field_result(inferred, execution_context=original_context)
        asset = NativeInputUseAsset.from_archive(row["asset"])
        _, task, binding, authority = contract(actual, asset, owner, origins, target, construct=False)
        current["authority"] = authority
        result = session.query("FROM $r RETURN TASK_OUTCOME($output,$task,$asset,$binding,$authority)",
            parameters={"output": inferred, "task": task, "asset": row["asset"], "binding": binding,
                        "authority": authority}, sources=sources)
        current["inference"] = inferred
        return result.values[0]

    def observe(data):
        # Current native authority is executor-owned, not serialized as a program capture.
        return session.run_program("observer", version=1, sources=sources,
            parameters={"outcome": data, "authority": current["authority"]}, exactness="exact")

    def publish(data):
        return session.save_result("accepted", Result((data,), aliases=("outcome",))).to_wire_dict()

    hive.add_worker("execute", execute, capability="transform", worker_type="program:rcql")
    hive.add_worker("observe", observe, capability="analyze", worker_type="program:rcql")
    hive.add_worker("publish", publish, capability="transform", worker_type="record:rcql-result")
    try:
        workflow = hive.run_workflow([Step("execution", "execute", checks=(Check("observe", Ref("execution")),)),
            Step("publication", "publish", Ref("execution"))])
        assert workflow.steps[0]["state"] != "failed", workflow.steps[0].get("error", workflow.steps[0].get("checks"))
        assert workflow.ok is (not refuted)
        assert workflow.steps[0]["state"] == ("rejected" if refuted else "completed"), workflow.steps
        observed = workflow.steps[0]["checks"][0]["result"]
        assert observed.named_values["accepted"] is (not refuted)
        assert observed.named_values["status"] == ("refuted" if refuted else "satisfied")
        value = workflow.outputs["execution"]
        assert value.inspect(current["authority"])["executed_response"]
        assert (store.get_record("accepted") is not None) is (not refuted)
        assert value.native(current["authority"]).response(current["authority"]).result().field_archive(3) == (
            read_available_field_result(current["inference"], execution_context=original_context).result().field_archive(3))
        packet = workflow.to_result()
        original = packet.to_bytes()
        session.save_result("workflow", packet)
        store.close()
        store = open_store(uri)
        cold_session = RelationalSession(store)
        recovered = cold_session.read("workflow", 1).value
        assert recovered.to_bytes() == original
        restored = recovered.values[0]["outputs"]["execution"]
        assert restored._owner is None
        assert restored.inspect(current["authority"])["status"] == observed.named_values["status"]
        from dataclasses import replace

        from rexgraph.source_binding import CppSourceReference
        forged_identity = replace(restored.source.native, archive_digest="f" * 64)
        forged = replace(restored, source=CppSourceReference(None, native=forged_identity))
        with pytest.raises(ValueError, match="source differs"):
            forged.native(current["authority"])
        destination = open_store("local://" + str(tmp_path / "copy"))
        try:
            receipt = copy_record(store, destination, store.get_record("workflow"), return_receipt=True)
            assert receipt.source_digest == receipt.destination_digest
            assert RelationalSession(destination).read("workflow", 1).value.to_bytes() == original
        finally:
            destination.close()
    finally:
        hive.stop_all()
        store.close()
