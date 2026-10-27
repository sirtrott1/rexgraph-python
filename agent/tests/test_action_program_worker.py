"""Stored action programs execute with current authority through common services."""
from contextlib import closing
import importlib.util
import os
from pathlib import Path

import pytest


def declarations():
    path = Path(os.environ.get("REXGRAPH_ACTION_TEST_HELPER", str(
        Path(__file__).resolve().parents[2] / "bindings/cpp/tests/test_action_program.py")))
    spec = importlib.util.spec_from_file_location("action_declarations", path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper


def test_hive_current_action_authority_and_checked_copy(tmp_path, monkeypatch):
    pytest.importorskip("rexgraph_cpp._action_program")
    from agent import RelationalSession
    from agent.hive import Hive
    from rcdb import MemoryStore, LocalStore, copy_record
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
    from rexgraph.io.action_record import action_record, read_action_record
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    owner, evidence = declarations().fixture("signed", core=True)
    program = Program("whole_action_worker", (
        ProgramStep("packet", parse('FROM $r RETURN ACTION_PACKET()')),
        ProgramStep("execute", parse('FROM $r RETURN ACTION_EXECUTE($packet,$evidence,$availability,owner=$owner)'),
                    inputs=(("packet", OutputRef("packet")),)),
        ProgramStep("fields", parse('FROM $r RETURN ACTION_RESULT($output,$evidence,$availability)'),
                    inputs=(("output", OutputRef("execute")),)),
    ), (ProgramInput("evidence"), ProgramInput("availability"), ProgramInput("owner")),
       (("archive", OutputRef("execute")), ("fields", OutputRef("fields"))))
    with closing(MemoryStore()) as store, closing(LocalStore(str("local://" + str(tmp_path / "actions.sqlite")).removeprefix("local://"))) as destination:
        session = RelationalSession(store)
        session.publish("action", action_record(owner), expected_version=0)
        session.publish_program("worker", program, expected_version=0)
        cutoff = [((790, 7),)]
        hive = Hive("native-action-worker")
        try:
            hive.add_program("act", session, "worker", version=1,
                sources={"r": {"record_id": "action", "version": 1}},
                runtime_parameters={"owner": lambda: owner, "evidence": lambda: evidence, "availability": lambda: cutoff[0]})
            result = hive.invoke("act", {})
            assert result.named_values["fields"]["totals"] == (5, -2, 3, -2)
            assert len(result.named_values["fields"]["fields"]) == 6
            assert Result.from_bytes(result.to_bytes()).to_bytes() == result.to_bytes()
            actual = owner.execute(evidence, cutoff[0])
            session.publish("output", action_record(actual), expected_version=0)
            copied = copy_record(store, destination, store.get_record("output"))
            restored = read_action_record(destination.get(copied.id), evidence=evidence, availability=cutoff[0])
            assert restored.to_bytes() == actual.to_bytes()
            with pytest.raises(ValueError, match="registration-owned"):
                hive.invoke("act", {"availability": cutoff[0]})
            cutoff[0] = ((790, 6),)
            with pytest.raises((ValueError, RuntimeError)):
                hive.invoke("act", {})
            cutoff[0] = ((790, 7),)
            assert hive.invoke("act", {}).named_values["fields"]["totals"] == (5, -2, 3, -2)
        finally:
            hive.stop_all()


def test_system_native_action_results_use_common_query_path():
    pytest.importorskip("rexgraph_cpp._action_program")
    from fastapi.testclient import TestClient
    from rcql import Result
    from rcdb import MemoryStore
    from rexgraph.io.action_record import action_record
    from system.server.app import app
    from system.state import sources
    owner, evidence = declarations().fixture("signed", core=True)
    with closing(MemoryStore()) as store:
        store.put("program", action_record(owner))
        store.put("evidence", action_record(evidence))
        sources.register("native-actions", store)
        try:
            with TestClient(app) as client:
                query = ('FROM RCDB("native-actions") '
                    'LET p=ACTION_PACKET(RCDB_GET("program")) LET e=ACTION_PACKET(RCDB_GET("evidence")) '
                    'RETURN ACTION_RESULT(ACTION_EXECUTE(p,e,$availability),e,$availability)')
                response = client.post("/api/query", json={"query": query,
                    "params": {"availability": [[790, 7]]}, "result_format": "native"})
                assert response.status_code == 200, response.text
                result = Result.from_bytes(response.content)
                assert result.values[0]["totals"] == (5, -2, 3, -2)
                assert len(result.values[0]["fields"]) == 6
                refused = client.post("/api/query", json={"query": query,
                    "params": {"availability": [[790, 6]]}, "result_format": "native"})
                assert refused.status_code == 400
        finally:
            sources.remove("native-actions")
