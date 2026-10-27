"""Actual native family workers, current-evidence guards and RCDB proof references."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pytest.importorskip("rexgraph_cpp._finite_search")
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_ACTIVITY_JOURNAL", "off")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "bindings/cpp/tests"))
    from test_relational_finite_field import asset, declaration
    from agent import FamilyVerifier, RelationalSession
    from rcdb import open_store
    from rcql import Result
    from rexgraph_cpp import NativeFiniteSource
    uri = "local://" + str(tmp_path / "records")
    store = open_store(uri)
    session = RelationalSession(store)
    verifier = FamilyVerifier(session)
    session.save_result("source", Result((NativeFiniteSource(asset(), declaration()).archive(),), aliases=("source",)))
    yield verifier, session, uri
    store.close()


def task(verifier, *, name="task", fixed=((0, 1, 102),), window=None,
         terms=((0, 0, 102), (0, 2, 102)), **options):
    return verifier.create_task(name, source=verifier.pin("source"),
        projection=dict(key=701, revision=1, cells=((0, 0), (0, 2)), window=window,
                        fixed=fixed, distinct_relations=()),
        claims=({"name": "joint-response", "terms": terms},), **options)


@pytest.mark.parametrize("options,status,accepted", [
    ({}, "entailed", True),
    ({"terms": ((0, 0, 101),)}, "refuted", False),
    ({"fixed": ()}, "alternatives", None),
    ({"terms": ((0, 0, 999),)}, "unsupported", None),
    ({"window": ((), (101, 102, 103))}, "inconsistent_family", False),
    ({"family_limits": (65536, 0, 16777216, 65536, 65536)}, "incomplete", None),
])
def test_native_outcomes_remain_distinct_and_publication_is_guarded(setup, options, status, accepted):
    verifier, session, _ = setup
    selected = task(verifier, **options)
    proof = verifier.verify(selected, proof_id="proof")
    observed = verifier.assess(proof).named_values
    assert observed["status"] == status
    assert observed["accepted"] is accepted
    packet = session.read("proof").value.named_values["proof"]
    assert isinstance(packet["native_proof"], bytes)
    assert verifier.proof_owner(proof)["status"] == packet["status"]
    if accepted:
        answer = verifier.publish(proof, "answer")
        assert session.read(answer["record_id"]).value.named_values["answer"]["proof"] == proof
        queried = session.query('FROM $db RETURN RCDB_GET("answer")')
        assert queried.values[0].to_bytes() == session.read("answer").value.to_bytes()
    else:
        with pytest.raises(ValueError, match="cannot publish"):
            verifier.publish(proof, "answer")
        assert session.store.get_record("answer") is None


def test_hive_uses_small_proof_refs_and_cold_replay_executes_again(setup):
    from agent import FamilyVerifier, RelationalSession
    from agent.hive import Hive
    from agent.hive_workflow import WorkflowCheck as Check, WorkflowRef as Ref, WorkflowStep as Step
    from rcdb import open_store
    verifier, session, uri = setup
    selected = task(verifier)
    hive = Hive("finite-family-verification")
    hive.add_worker("execute", lambda _: verifier.verify(selected, proof_id="proof"), capability="transform")
    hive.add_worker("guard", verifier.assess, capability="analyze")
    hive.add_worker("publish", lambda pin: verifier.publish(pin, "answer"), capability="transform")
    try:
        workflow = hive.run_workflow((Step("execution", "execute", checks=(Check("guard", Ref("execution")),)),
            Step("publication", "publish", Ref("execution"), guards=(Check("guard", Ref("execution")),))))
        assert workflow.ok, workflow.steps
        pin = workflow.outputs["execution"]
        packet = workflow.to_result().to_bytes()
        native = session.read("proof").value.named_values["proof"]["native_proof"]
        assert native not in packet
        assert len(packet) < 15000
        session.save_result("workflow", workflow.to_result())
        cold_store = open_store(uri)
        try:
            cold_session = RelationalSession(cold_store)
            cold = FamilyVerifier(cold_session)
            assert cold.assess(pin).named_values["status"] == "unissued"
            with pytest.raises(ValueError, match="unissued"):
                cold.publish(pin, "forged")
            fresh = cold.replay(pin, proof_id="cold-proof")
            assert cold.assess(fresh).named_values["accepted"] is True
            assert cold_session.read("cold-proof").value.named_values["proof"]["native_proof"] == native
            cold.publish(fresh, "cold-answer")
            assert cold_session.read("workflow").value.to_bytes() == packet
        finally:
            cold_store.close()
    finally:
        hive.stop_all()
    # A separate interpreter has neither the live owning image nor issued grants.
    script = """
import sys
from contextlib import closing
from agent import FamilyVerifier, RelationalSession
from rcdb import open_store
with closing(open_store(sys.argv[1])) as store:
    session = RelationalSession(store)
    verifier = FamilyVerifier(session)
    pin = session.read('answer').value.named_values['answer']['proof']
    assert verifier.assess(pin).named_values['status'] == 'unissued'
    fresh = verifier.replay(pin, proof_id='process-proof')
    assert verifier.assess(fresh).named_values['accepted'] is True
    verifier.publish(fresh, 'process-answer')
    print('fresh native replay and publication')
"""
    executed = subprocess.run([sys.executable, "-I", "-B", "-c", script, uri],
        cwd=Path(uri.removeprefix("local://")).parent, env=dict(os.environ),
        capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout.strip() == "fresh native replay and publication"


@pytest.mark.parametrize("changed", ["source", "dependency", "task", "proof", "provider"])
def test_stale_evidence_or_provider_blocks_publication(setup, changed):
    from rcql import Result
    verifier, session, _ = setup
    session.save_result("input", Result((b"original observed bytes",), aliases=("input",)))
    selected = task(verifier, dependencies=(verifier.pin("input"),))
    proof = verifier.verify(selected, proof_id="proof")
    assert verifier.assess(proof).named_values["accepted"] is True
    if changed == "provider":
        verifier.withdraw()
        verifier.resume()
    else:
        record_id = "input" if changed == "dependency" else changed
        old = session.read(record_id).value
        session.save_result(record_id, old, expected_version=1)
    assert verifier.assess(proof).named_values["accepted"] is None
    with pytest.raises(ValueError, match="cannot publish"):
        verifier.publish(proof, "answer")
    assert session.store.get_record("answer") is None


def test_dependency_change_during_native_execution_is_observed(setup, monkeypatch):
    from rcql import Result
    verifier, session, _ = setup
    session.save_result("input", Result((b"before",), aliases=("input",)))
    selected = task(verifier, dependencies=(verifier.pin("input"),))
    real = verifier._provider
    receipt, source_type, family_type = real()

    class UpdatingFamily:
        @staticmethod
        def projection(source, declaration):
            session.save_result("input", Result((b"after",), aliases=("input",)), expected_version=1)
            return family_type.projection(source, declaration)

    monkeypatch.setattr(verifier, "_provider", lambda: (receipt, source_type, UpdatingFamily))
    proof = verifier.verify(selected, proof_id="proof")
    assert verifier.assess(proof).named_values["accepted"] is None
    with pytest.raises(ValueError, match="cannot publish"):
        verifier.publish(proof, "answer")


def test_guard_checks_published_envelopes_without_reloading_whole_payloads(setup, monkeypatch):
    verifier, session, _ = setup
    proof = verifier.verify(task(verifier), proof_id="proof")
    def unexpected(*args, **kwargs):
        raise AssertionError("current proof guards must not decode native payloads")
    monkeypatch.setattr(session, "read", unexpected)
    assert verifier.assess(proof).named_values["accepted"] is True


def test_reference_forgery_and_changed_provider_are_refused(setup, monkeypatch):
    verifier, _, _ = setup
    proof = verifier.verify(task(verifier), proof_id="proof")
    for key, value in (("state_digest", "f" * 64), ("store_id", "f" * 64), ("version", 2)):
        forged = dict(proof, **{key: value})
        assert verifier.assess(forged).named_values["accepted"] is None
    provider = verifier._provider
    receipt, source, family = provider()
    monkeypatch.setattr(verifier, "_provider", lambda: (dict(receipt, binding_digest="changed"), source, family))
    assert verifier.assess(proof).named_values["status"] == "provider_changed"


def test_cancelled_native_execution_preserves_proof_but_never_publishes(setup):
    from rexgraph_cpp import FiniteSearchCancellation
    verifier, _, _ = setup
    cancellation = FiniteSearchCancellation()
    cancellation.cancel()
    proof = verifier.verify(task(verifier), proof_id="proof", cancellation=cancellation)
    assert verifier.proof_owner(proof)["status"] == "cancelled"
    assert verifier.assess(proof).named_values["accepted"] is None


def test_missing_native_provider_has_no_python_fallback(setup, monkeypatch):
    verifier, _, _ = setup
    selected = task(verifier)
    def missing():
        raise ImportError("native finite provider unavailable")
    monkeypatch.setattr(verifier, "_provider", missing)
    with pytest.raises(ImportError, match="native finite provider unavailable"):
        verifier.verify(selected, proof_id="proof")


def test_publication_rechecks_after_a_previously_accepted_guard(setup):
    from rcql import Result
    verifier, session, _ = setup
    session.save_result("input", Result((b"evidence",), aliases=("input",)))
    selected = task(verifier, dependencies=(verifier.pin("input"),))
    proof = verifier.verify(selected, proof_id="proof")
    assert verifier.assess(proof).named_values["accepted"] is True
    session.save_result("input", Result((b"new evidence",), aliases=("input",)), expected_version=1)
    with pytest.raises(ValueError, match="cannot publish"):
        verifier.publish(proof, "answer")


def test_publication_obeys_expected_destination_version(setup):
    verifier, session, _ = setup
    proof = verifier.verify(task(verifier), proof_id="proof")
    first = verifier.publish(proof, "answer")
    assert first["version"] == 1
    with pytest.raises((ValueError, RuntimeError)):
        verifier.publish(proof, "answer", expected_version=0)
    second = verifier.publish(proof, "answer", expected_version=1)
    assert second["version"] == 2


def test_releasing_live_owners_leaves_durable_proof_for_fresh_execution(setup):
    verifier, session, _ = setup
    proof = verifier.verify(task(verifier), proof_id="proof")
    verifier.release(proof)
    assert verifier.assess(proof).named_values["status"] == "unissued"
    assert session.read("proof").value.named_values["proof"]["native_proof"]
    fresh = verifier.replay(proof, proof_id="replayed")
    assert verifier.assess(fresh).named_values["accepted"] is True


def test_corrupt_source_packet_cannot_create_a_proof_or_publication(setup):
    from rcql import Result
    verifier, session, _ = setup
    raw = session.read("source").value.named_values["source"]
    session.save_result("source", Result((raw + b"undeclared suffix",), aliases=("source",)), expected_version=1)
    selected = task(verifier)
    with pytest.raises((ValueError, RuntimeError)):
        verifier.verify(selected, proof_id="proof")
    assert session.store.get_record("proof") is None
    assert session.store.get_record("answer") is None


def test_restored_task_receives_the_same_closed_schema_validation(setup):
    from rcql import Result
    verifier, session, _ = setup
    selected = task(verifier)
    spec = dict(session.read(selected["record_id"]).value.named_values["task"])
    spec["claims"] = ({"name": "claim", "terms": ((0, 0, 102),), "ignored": True},)
    session.save_result("bad-task", Result((spec,), aliases=("task",)))
    with pytest.raises(ValueError, match="name and terms"):
        verifier.verify(verifier.pin("bad-task"), proof_id="proof")
    assert session.store.get_record("proof") is None
