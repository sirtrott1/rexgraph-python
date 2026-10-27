"""Evidence selection and real checking operations across Agent entry points."""
from types import SimpleNamespace as NS

import pytest
from agent.hallucination import detect_hallucinations_exchange
from agent.model_context import select_context
from agent.pipeline_runner import PipelineRunner
from agent.guard import GuardRule, OutputGuard
from agent.reactive_hive import ReactiveHive
from agent.hive import Hive
from agent import rcdb
from rexgraph import RexGraph


def source():
    return RexGraph.from_cells([3, [[0, 1], [1, 2]]])


def doc(name, text):
    return NS(doc_id=name, text=text, rex=source(), meta={"record_id": name, "record_version": 1})


def chunk(text):
    return NS(text=text, char_start=0, char_end=len(text), kappa=1.0)


def test_context_keeps_distinct_uses_and_byte_maps():
    text = "meaning αβ"
    first, second = doc("a", text), doc("b", text)
    context = select_context([(first, chunk(text)), (second, chunk(text))])
    record = context.as_record()
    assert context.text == text + "\n\n" + text
    assert [p["doc_id"] for p in record["parts"]] == ["a", "b"]
    assert all(p["source_byte_end"] == len(text.encode()) for p in record["parts"])
    assert record["parts"][0]["record_id"] != record["parts"][1]["record_id"]
    record["parts"][0]["doc_id"] = "changed"
    assert context.as_record()["parts"][0]["doc_id"] == "a"


def test_unmapped_chunk_positions_are_not_inferred_from_equal_text():
    context = select_context([(doc("a", "repeat repeat"), NS(text="repeat"))])
    part = context.as_record()["parts"][0]
    assert part["mapping_status"] == "unmapped" and part["source_char_start"] is None


def test_pipeline_checks_the_context_that_the_model_actually_receives(monkeypatch):
    first, second = doc("a", "alpha beta "*1500), doc("b", "zebra yak")
    corpus = NS(documents=[first, second], query=lambda *a, **k: NS(query_text="alpha",
        ranked_sections=[{"doc_id": "a"}, {"doc_id": "b"}]))
    observed = {}
    class Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, *, json):
            observed["wire"] = json["messages"][0]["content"].removeprefix("Document context:\n")
            return NS(raise_for_status=lambda: None, json=lambda: {"choices": [{"message": {"content": "alpha beta"}}]})
    from agent.hallucination import HallucinationReport
    def check(text, response):
        observed["checked"] = text
        return HallucinationReport()
    monkeypatch.setattr("httpx.Client", Client)
    monkeypatch.setattr("agent.hallucination.detect_hallucinations_exchange", check)
    _, response, report = PipelineRunner(model_url="http://unused.invalid")._query_and_check(
        corpus, [("a", [chunk(first.text)]), ("b", [chunk(second.text)])], {}, "alpha", 0)
    assert response == "alpha beta"
    assert observed["checked"] == observed["wire"] == first.text[:12000]
    assert report["context"]["text"] == observed["wire"]
    assert report["context"]["truncated"]
    assert "zebra" not in observed["checked"]


def test_no_exchange_and_no_shared_entities_do_not_assert_consistency():
    missing = detect_hallucinations_exchange("x", "y")
    different = detect_hallucinations_exchange("alpha beta gamma delta", "zebra yak quartz neon")
    assert missing.evaluation_status == "not_evaluated" and missing.overall_score is None
    assert different.evaluation_status == "unsupported"
    assert "structurally consistent" not in missing.summary() + different.summary()
    assert missing.as_record()["observation_kind"] == "exchange_structure"


def test_stream_match_completed_earlier_in_the_text_is_emitted():
    guard = OutputGuard([GuardRule("late", r"a.*z", "late"), GuardRule("early", r"bad", "early")])
    events = list(guard.scan_stream(["abad", "z"]))
    assert [hit["rule"] for hit in events[0][1]] == ["early"]
    assert [hit["rule"] for hit in events[1][1]] == ["late"]


def test_reactive_worker_needs_a_real_implementation():
    h = Hive("real-check")
    reactive = ReactiveHive(h, store=rcdb.MemoryStore())
    actions = reactive.require("verify")
    assert actions[0]["unmet"] == "verify" and h.get("verifier") is None
    h.add_worker("unreviewed", lambda value: value, capability="analyze", worker_type="unreviewed")
    assert reactive.require("review")[0]["unmet"] == "review"


def test_supplied_verifier_executes_and_retains_its_actual_result():
    from rcql import Result
    h = Hive("supplied-check")
    seen = []
    def verify(value, **kw):
        seen.append(value)
        return Result((value == "supported",), aliases=("accepted",))
    reactive = ReactiveHive(h, store=rcdb.MemoryStore(), implementations={"verify": verify})
    assert reactive.require("verify")[0]["deployed"] == "verifier"
    assert h.invoke("verifier", "wrong").values == (False,)
    assert seen == ["wrong"]
    assert reactive.require("verify") == []


def test_triple_view_explicitly_rejects_a_wider_source_without_losing_it():
    from agent.knowledge import join
    from rexgraph.object_identity import object_digest
    value = RexGraph.from_cells([3, [[0, 1, 2]]])
    before = object_digest(value)
    with pytest.raises(ValueError, match="triples"):
        join(value)
    assert object_digest(value) == before and len(value.boundary_idx) == 3


def test_execution_flow_requires_an_actual_path_and_task_condition():
    from agent.integrations.langgraph_rex import RexStateGraph
    graph = RexStateGraph()
    graph.add_transition("a", "b"); graph.add_transition("b", "c"); graph.add_transition("c", "a")
    graph.log_visit("a")
    assert graph.should_continue()["evaluation_status"] == "not_evaluated"
    graph.log_visit("b")
    report = graph.should_continue()
    assert report["evaluation_status"] == "evaluated" and report["path"] == ["a", "b"]
    assert report["recommendation"] is None
    assert "pct_gradient" in report
