"""Source-relative answer publication and native clarification analysis."""
import pytest
from pathlib import Path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pytest.importorskip("rexgraph_cpp")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "bindings/cpp/tests"))
    from test_relational_finite_field import asset, declaration
    from agent import FamilyVerifier, RelationalSession
    from rcdb import open_store
    from rcql import Result
    from rexgraph_cpp import NativeFiniteSource
    store = open_store("local://" + str(tmp_path / "records"))
    session = RelationalSession(store)
    verifier = FamilyVerifier(session)

    def source(mode="weak", name="source", expected_version=0):
        d = declaration()
        if mode == "strong":
            d["charts"] = (dict(d["charts"][0], allowed=((102, 102),)), d["charts"][1])
        elif mode == "inconsistent":
            d["charts"] = (dict(d["charts"][0], allowed=((101, 101),)),
                           dict(d["charts"][1], allowed=((102, 102),)))
        session.save_result(name, Result((NativeFiniteSource(asset(), d).archive(),), aliases=("source",)),
                            expected_version=expected_version)
        return verifier.pin(name)

    yield verifier, session, source
    store.close()


def question(verifier, source, **options):
    return verifier.create_question("task", source=source, answer_cells=((0, 0),),
        observation_cells=((0, 2),), key=701, **options)


@pytest.mark.parametrize("mode,options,status,accepted,answer", [
    ("strong", {}, "unique", True, (102,)),
    ("weak", {}, "alternatives", None, None),
    ("inconsistent", {}, "inconsistent_family", False, None),
    ("strong", {"max_checks": 0}, "incomplete", None, None),
    ("strong", {"family_limits": (65536, 0, 16777216, 65536, 65536)}, "incomplete", None, None),
])
def test_only_complete_nonempty_unique_answers_publish(setup, mode, options, status, accepted, answer):
    verifier, session, source = setup
    task = question(verifier, source(mode), **options)
    proof = verifier.verify(task, proof_id="proof")
    result = verifier.assess(proof).named_values
    assert (result["status"], result["accepted"], result["answer"]) == (status, accepted, answer)
    if accepted:
        verifier.publish(proof, "answer")
        value = session.read("answer").value.named_values["answer"]
        assert value["answer"] == answer and value["scope"] == "declared-source"
    else:
        with pytest.raises(ValueError, match="cannot publish"):
            verifier.publish(proof, "answer")
        assert session.store.get_record("answer") is None


def test_sufficient_clarification_is_not_an_observed_answer(setup):
    verifier, _, source = setup
    proof = verifier.verify(question(verifier, source()), proof_id="proof")
    result = verifier.assess(proof).named_values
    assert result["question"]["observation_determines_answer"] is True
    assert result["question"]["answer_count"] == 3
    assert result["accepted"] is None and result["answer"] is None


@pytest.mark.parametrize("profile", ["question", "claim"])
def test_task_snapshots_source_reference_before_record_reads(setup, monkeypatch, profile):
    verifier, session, source = setup
    original = source("strong")
    read = verifier._read

    def mutate_caller(pin, **kwargs):
        selected = read(pin, **kwargs)
        original["version"] = 900
        return selected

    monkeypatch.setattr(verifier, "_read", mutate_caller)
    if profile == "question":
        task = question(verifier, original)
    else:
        task = verifier.create_task("task", source=original,
            projection=dict(key=701, revision=1, cells=((0, 0),), window=None, fixed=(), distinct_relations=()),
            claims=({"name": "value", "terms": ((0, 0, 102),)},))
    assert session.read("task").value.named_values["task"]["source"]["version"] == 1
    assert verifier.assess(verifier.verify(task, proof_id="proof")).named_values["accepted"] is True


def test_observer_results_do_not_alias_live_authority(setup):
    verifier, _, source = setup
    proof = verifier.verify(question(verifier, source()), proof_id="proof")
    report = verifier.assess(proof).named_values["question"]
    report.update(status="unique", answer=(999,), answer_count=1)
    actual = verifier.assess(proof).named_values
    assert actual["status"] == "alternatives" and actual["answer"] is None
    with pytest.raises(ValueError, match="cannot publish"):
        verifier.publish(proof, "answer")


def test_source_withdrawal_reopens_question_and_old_grant_is_stale(setup):
    verifier, _, source = setup
    first = source("strong")
    task = question(verifier, first)
    pin = verifier.verify(task, proof_id="proof")
    assert verifier.assess(pin).named_values["answer"] == (102,)
    weak = source("weak", expected_version=1)
    assert verifier.assess(pin).named_values["accepted"] is None
    next_task = question(verifier, weak, expected_version=1)
    next_pin = verifier.verify(next_task, proof_id="proof", expected_version=1)
    assert verifier.assess(next_pin).named_values["status"] == "alternatives"


def test_cold_replay_recomputes_question_and_whole_native_packet(setup):
    from agent import FamilyVerifier
    verifier, session, source = setup
    pin = verifier.verify(question(verifier, source("strong")), proof_id="proof")
    cold = FamilyVerifier(session)
    assert cold.assess(pin).named_values["status"] == "unissued"
    replayed = cold.replay(pin, proof_id="cold-proof")
    assert cold.assess(replayed).named_values["answer"] == (102,)
    assert session.read("cold-proof").value.named_values["proof"]["native_proof"] == session.read("proof").value.named_values["proof"]["native_proof"]


def test_cancelled_question_does_not_answer(setup):
    from rexgraph_cpp import FiniteSearchCancellation
    verifier, _, source = setup
    token = FiniteSearchCancellation()
    token.cancel()
    pin = verifier.verify(question(verifier, source("strong")), proof_id="proof", cancellation=token)
    assert verifier.assess(pin).named_values["status"] == "incomplete"


@pytest.mark.parametrize("mutation", ["window", "fixed", "distinct_relations", "cells"])
def test_saved_question_cannot_restrict_its_own_answer(setup, mutation):
    from rcql import Result
    verifier, session, source = setup
    question(verifier, source())
    spec = session.read("task").value.named_values["task"]
    changes = {"window": ((101,), (101,)), "fixed": ((0, 0, 101),),
        "distinct_relations": ((1, 0),), "cells": ((0, 0),)}
    spec["projection"][mutation] = changes[mutation]
    session.save_result("task", Result((spec,), aliases=("task",)), expected_version=1)
    with pytest.raises(ValueError):
        verifier.verify(verifier.pin("task"), proof_id="proof")
