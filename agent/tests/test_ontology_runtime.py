"""Actual ontology histories and whole-source RCQL workers run through Hive."""
from contextlib import closing
from fractions import Fraction

import pytest

from agent import RelationalSession
from agent.hive import Hive
from agent.ontology_runtime import read_ontology, publish_ontology
from rcdb import MemoryStore
from rcql import OutputRef, Program, ProgramInput, ProgramStep, parse
from rexgraph import RexGraph
from rexgraph.ontology import OntologyState
from rexgraph.value import ExactTime


def ontology():
    return OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:ontology",
        context="use-a", known_at=ExactTime(Fraction(1)), signature={"types": ("edge",)},
        universe={"scope": "original", "unknown": ("first", "second")})


def test_actual_publication_time_is_checked_separately_from_declared_horizon():
    value = ontology()
    with closing(MemoryStore()) as store:
        store.put("ontology", value.record(), analytics=False, _tx_time=20.0)
        session = RelationalSession(store)
        with pytest.raises(ValueError, match="publication"):
            read_ontology(session, "ontology", version=1, cutoff=ExactTime(Fraction(10)))
        assert read_ontology(session, "ontology", version=1, cutoff=ExactTime(Fraction(20))).digest == value.digest


def test_actual_evidence_closure_refuses_a_later_source_even_for_backdated_declarations():
    from rexgraph.tensor_field import FieldSource
    with closing(MemoryStore()) as store:
        evidence = RexGraph.from_graph([0], [1])
        store.put("evidence", evidence, analytics=False, _tx_time=30.0)
        value = OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:ontology",
            context="one", known_at=ExactTime(Fraction(1)), signature={}, universe={},
            dependencies=(FieldSource(evidence, "evidence", 1),))
        store.put("ontology", value.record(), analytics=False, _tx_time=10.0)
        session = RelationalSession(store)
        with pytest.raises(ValueError, match="cutoff"):
            read_ontology(session, "ontology", version=1, cutoff=ExactTime(Fraction(20)))
        assert read_ontology(session, "ontology", version=1, cutoff=ExactTime(Fraction(30))).digest == value.digest


def test_agent_file_reader_and_automatic_loader_preserve_the_complete_source(tmp_path):
    from agent.auto import auto_rex
    from agent.adapters.formats import read
    from rexgraph.io import save_ontology
    value = ontology()
    path = tmp_path / "ontology.rcbo"
    save_ontology(path, value)
    assert OntologyState.from_record(read(path)).digest == value.digest
    assert OntologyState.from_record(auto_rex(path)).digest == value.digest


def test_hive_program_reads_complete_ontology_and_revises_through_checked_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    value = ontology()
    program = Program("ontology_worker", (
        ProgramStep("whole", parse('FROM $r LET o = ONTOLOGY_AT(ONTOLOGY_READ(), $cutoff) '
                                  'RETURN o, ONTOLOGY_RECORD(o), ONTOLOGY_INFO(o)')),
    ), (ProgramInput("cutoff", "ExactInteger"),),
       (("ontology", OutputRef("whole", 0)), ("source", OutputRef("whole", 1)),
        ("inspection", OutputRef("whole", 2))))
    with closing(MemoryStore()) as store:
        session = RelationalSession(store)
        publish_ontology(session, "ontology", value, expected_version=0)
        session.publish_program("worker", program, expected_version=0)
        hive = Hive("ontology-programs")
        try:
            hive.add_program("ontology-reader", session, "worker", version=1,
                sources={"r": {"record_id": "ontology", "version": 1}})
            result = hive.invoke("ontology-reader", {"cutoff": 1})
            assert result.values[0].digest == value.digest
            assert OntologyState.from_record(result.values[1]).digest == value.digest
            assert result.values[2]["revision"] == 1
            child = value.revise(value.record(), known_at=ExactTime(Fraction(2)),
                kind="ontology-revision", maps=("declared-comparison",),
                universe={"scope": "expanded", "unknown": ("first", "second", "third")})
            assert publish_ontology(session, "ontology", child, expected_version=1).version == 2
            assert hive.invoke("ontology-reader", {"cutoff": 1}).values[0].digest == value.digest
            with pytest.raises(ValueError, match="parent"):
                publish_ontology(session, "ontology", value, expected_version=2)
        finally:
            hive.stop_all()


def test_hive_native_revision_worker_requires_registration_permission_and_actual_parent(tmp_path, monkeypatch):
    native = pytest.importorskip("rexgraph_cpp")
    pytest.importorskip("rexgraph_cpp._source_evolution")
    from agent.ontology_runtime import native_revision_program
    from rexgraph.native_ontology import PROVIDER, native_ontology_component, register_native_ontology_provider
    owner = native.NativeSource.from_declaration({"profile": "declared-exact-source-v1",
        "namespace": 9051, "semantic_scale": 1, "max_grade": 1,
        "types": (), "witnesses": (10,), "relations": (), "carriers": (), "fields": (), "metrics": ()})
    parent = OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:revision",
        context="one", known_at=ExactTime(Fraction(1)), universe={}, provider=dict(PROVIDER),
        signature={"native_sources": (native_ontology_component("part", owner),)})
    spec = {"profile": "source-signature-additions-v1", "aspects": (),
            "types": ({"name": "new exact relation", "min_arity": 0, "max_arity": 9},)}
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_CACHE_DIR", str(tmp_path / "cache"))
    register_native_ontology_provider()
    with closing(MemoryStore()) as store:
        session = RelationalSession(store)
        publish_ontology(session, "ontology", parent, expected_version=0)
        program = native_revision_program("native_revision", operation="add")
        session.publish_program("worker", program, expected_version=0)
        hive = Hive("native-ontology")
        args = {"component": "part", "specification": spec, "cutoff": 2}
        selectors = {"r": {"record_id": "ontology", "version": 1}}
        try:
            hive.add_program("reader", session, "worker", version=1, sources=selectors)
            with pytest.raises(PermissionError):
                hive.invoke("reader", args)
            with pytest.raises(TypeError, match="boolean"):
                hive.add_program("bad", session, "worker", version=1, sources=selectors, allow_write=1)
            hive.add_program("reviser", session, "worker", version=1, sources=selectors, allow_write=True)
            child = hive.invoke("reviser", args).named_values["ontology"]
            assert child.is_child_of(parent)
            assert child.dependencies[-1].record_id == "ontology"
            assert child.verify(sources={}, parent=parent)["native_source_integrity"] == "supported"
            assert publish_ontology(session, "ontology", child, expected_version=1).version == 2
            # The worker keeps the explicitly selected parent, even after publication.
            assert hive.invoke("reviser", args).named_values["ontology"].digest == child.digest
            with pytest.raises(ValueError):
                hive.invoke("reviser", dict(args, allow_write=True))
            with pytest.raises(ValueError, match="parent"):
                publish_ontology(session, "ontology", child, expected_version=2)
        finally:
            hive.stop_all()
