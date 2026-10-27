"""Hive executes learned native function programs and persists complete children."""
from contextlib import closing
from fractions import Fraction as Q
import importlib.util
import os
from pathlib import Path

import pytest


def test_hive_learned_native_transfer_and_checked_store_copy(tmp_path, monkeypatch):
    pytest.importorskip("rexgraph_cpp._temporal_program")
    from agent import RelationalSession
    from agent.hive import Hive
    from rcdb import MemoryStore, LocalStore, copy_record
    from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
    from rexgraph.io.relational_record import read_relational_record, relational_record
    from rexgraph.relational_runtime import RelationalRuntime
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    path = Path(os.environ.get("REXGRAPH_RELATIONAL_TEST_HELPER", str(
        Path(__file__).resolve().parents[2] / "bindings/cpp/tests/test_temporal_program.py")))
    spec = importlib.util.spec_from_file_location("relational_worker_declarations", path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    _, state, t, _, _, _, _, _ = helper.fit_fixture(RelationalRuntime.create(89001))
    family = state.selected
    state, cubic = helper.function(state, t, (2, -1, 3, 1), label="fresh independent input")
    packet = state.to_bytes()
    template = Program("native_transfer", (ProgramStep("apply", parse(
        'FROM $r RETURN RELATIONAL_EXTEND(packet=$packet,operation="apply_learned",specification=$spec,owner=$owner)')),),
        (ProgramInput("packet", "Bytes"), ProgramInput("spec"), ProgramInput("owner")), (("child", OutputRef("apply")),))
    program = template.specialize({"packet": packet})
    with closing(MemoryStore()) as store, closing(LocalStore(str("local://" + str(tmp_path / "copy.db")).removeprefix("local://"))) as destination:
        session = RelationalSession(store)
        session.publish("source", relational_record(state), expected_version=0)
        session.publish_program("worker", program, expected_version=0)
        hive = Hive("relational-program")
        try:
            hive.add_program("learned", session, "worker", version=1,
                sources={"r": {"record_id": "source", "version": 1}}, runtime_parameters={"owner": lambda: state})
            specification = dict(family=family, bindings=[(1, cubic)], label="held whole response")
            result = hive.invoke("learned", {"spec": specification})
            child = RelationalRuntime.from_bytes(result.named_values["child"])
            assert child.extends(state)
            assert child.function()["pieces"][0]["numerator"] == (Q(16), Q(48), Q(24))
            assert child.evaluate(Q(1, 3)) == Q(104, 3)
            session.publish("child", relational_record(child), expected_version=0)
            session.save_result("result", result, expected_version=0)
            assert Result.from_bytes(result.to_bytes()).to_bytes() == result.to_bytes()
            copied = copy_record(store, destination, store.get_record("child"))
            assert read_relational_record(destination.get(copied.id)).to_bytes() == child.to_bytes()
            with pytest.raises(ValueError, match="registration-owned"):
                hive.invoke("learned", {"spec": specification, "owner": state})
        finally:
            hive.stop_all()
