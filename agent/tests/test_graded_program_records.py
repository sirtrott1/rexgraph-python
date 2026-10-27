"""Hive program records preserve whole graded intermediates and their owners."""
from contextlib import closing

import pytest

from agent import RelationalSession
from agent.hive import Hive
from rcdb import LocalStore, MemoryStore, RecordPacket, copy_record, record_packet
from rcql import OutputRef, Program, ProgramInput, ProgramStep, Result, parse
from rexgraph.graph import RexGraph
from rexgraph.weighted_dirac import GradedChain


@pytest.mark.parametrize("backend", ["memory", "local"])
def test_hive_program_and_checked_copy_retain_complete_graded_results(backend, tmp_path):
    program = Program("graded_pipeline", (
        ProgramStep("boundary", parse("FROM $r RETURN APPLY(DIRAC(),GRADED_CHAIN([SHARE(CELL(1,0))]),true)")),
        ProgramStep("response", parse("FROM $r RETURN APPLY(ANTI_DIRAC(),$field,true)"),
            (("field", OutputRef("boundary")),), contracts=(ProgramInput("field", "GradedChain"),))),
        outputs=(("whole", OutputRef("response")),))
    selected = MemoryStore() if backend == "memory" else LocalStore(tmp_path / "source")
    with closing(selected) as store, closing(LocalStore(tmp_path / "destination")) as destination:
        session = RelationalSession(store)
        session.publish("data", RexGraph.from_hypergraph([0, 2, 4], [0, 1, 1, 2]), expected_version=0)
        session.publish_program("pipeline", program, expected_version=0)
        hive = Hive("graded-program-records")
        try:
            hive.add_program("run", session, "pipeline", version=1,
                sources={"r": {"record_id": "data", "version": 1}})
            result = hive.invoke("run", {})
            assert isinstance(result.named_values["whole"], GradedChain)
            session.save_result("whole", result, expected_version=0)
            restored = session.read("whole").value
            assert restored.to_bytes() == result.to_bytes()
            owner = restored.named_values["whole"].source
            for step in restored.native_plan["step_results"]:
                state = step["values"][0]
                assert state.source is owner and all(c.source is owner for c in state.components)
                assert all(not c.values.flags.writeable for c in state.components)
            packet = RecordPacket.from_bytes(record_packet(store, "whole").to_bytes())
            assert packet.snapshot().value.to_bytes() == result.to_bytes()
            receipt = copy_record(packet.source(), destination, packet.record,
                destination_id="copied", expected_digest=packet.state_digest, return_receipt=True)
            assert receipt.source_digest == receipt.destination_digest
            assert destination.read_record("copied").value.to_bytes() == result.to_bytes()
            # An explicitly pinned worker continues to execute its historical source.
            session.publish("data", RexGraph.from_hypergraph([0, 3, 5], [0, 1, 2, 1, 2]), expected_version=1)
            historical = hive.invoke("run", {})
            assert historical.to_bytes() == result.to_bytes()
            hive.add_program("current", session, "pipeline", version=1,
                sources={"r": {"record_id": "data", "version": 2}})
            updated = hive.invoke("current", {})
            assert updated.to_bytes() != result.to_bytes()
            assert Result.from_bytes(updated.to_bytes()).to_bytes() == updated.to_bytes()
        finally:
            hive.stop_all()
