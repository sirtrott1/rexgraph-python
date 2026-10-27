"""Tests for the finalized Connector seam (agent.connectors + agent.interfaces).

Anchored to structural facts: a triangle has nV=3, nE=3, β₁=1, and the engine
returns a per edge character of shape (nE, 4). The contract's length invariants
must fail loudly on a malformed connector.
"""

import numpy as np
import pytest
from agent.connectors import BaseConnector, ConnectorError
from agent.interfaces import Capabilities, Connector, apply_label_privacy, configure, reset


class TriangleConnector(BaseConnector):
    """Trivial in memory connector: a 3 cycle A->B->C->A. Runs immediately, needs
    no live service: the smallest thing that exercises the whole contract."""

    CAPABILITIES = Capabilities(weights=True, schemes=("memory",))

    def read(self, source):
        labels = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C"), ("C", "A")]
        src = np.array([0, 1, 2])
        tgt = np.array([1, 2, 0])
        return self.result(
            (src, tgt),
            vertex_labels=labels,
            edges=edges,
            source="memory://triangle",
            weights=[1.0, 1.0, 1.0],
        )


def test_connector_is_recognized_by_the_protocol():
    conn = TriangleConnector()
    # runtime_checkable Protocol: read + capabilities present
    assert isinstance(conn, Connector)


def test_capabilities_descriptor():
    caps = TriangleConnector().capabilities()
    assert caps.topology is True          # always
    assert caps.weights is True
    assert caps.modality is False and caps.faces is False
    assert caps.schemes == ("memory",)
    assert caps.summary() == "topology+weights"


def test_meta_shape_matches_contract():
    rex, meta = TriangleConnector().read(None)
    assert set(["vertex_labels", "edges", "source", "nV", "nE"]) <= set(meta)
    assert meta["nV"] == 3 and meta["nE"] == 3
    assert len(meta["vertex_labels"]) == meta["nV"]
    assert len(meta["edges"]) == meta["nE"]
    assert len(meta["weights"]) == meta["nE"]
    assert meta["source"] == "memory://triangle"
    assert all(len(e) == 2 for e in meta["edges"])


def test_result_enforces_length_invariants():
    c = TriangleConnector()
    with pytest.raises(ConnectorError):        # too few labels
        c.result((np.array([0]), np.array([1])),
                 vertex_labels=[], edges=[("A", "B")], source="x")
    with pytest.raises(ConnectorError):        # weights length != nE
        c.result((np.array([0]), np.array([1])),
                 vertex_labels=["A", "B"], edges=[("A", "B")],
                 source="x", weights=[1.0, 2.0])


def test_output_builds_a_complex_in_the_engine():
    """The (rex, meta) a connector emits must actually construct in the engine."""
    from rexgraph.graph import RexGraph
    (src, tgt), meta = TriangleConnector().read(None)
    g = RexGraph(sources=src, targets=tgt)
    chi = np.asarray(g.structural_character, dtype=float)
    assert chi.shape == (meta["nE"], 4)        # per edge character over [T,G,F,C]


def test_labels_are_the_privacy_surface():
    """meta['vertex_labels'] is what label privacy tokenizes: same names must
    survive the round trip as stable tokens (structure preserving)."""
    reset()
    try:
        configure(label_privacy="hash", label_salt="s")
        _, meta = TriangleConnector().read(None)
        tok = apply_label_privacy(meta)
        assert tok["_label_privacy"] == "hash"
        assert len(tok["vertex_labels"]) == meta["nV"]
        assert tok["vertex_labels"] != ["A", "B", "C"]   # names hidden
        # deterministic: same name -> same token
        _, meta2 = TriangleConnector().read(None)
        assert apply_label_privacy(meta2)["vertex_labels"] == tok["vertex_labels"]
    finally:
        reset()


# template and validation harness

from agent.connectors import faces_to_csc, to_rexgraph
from agent.connectors.template import ExampleEdgesConnector
from agent.connectors.validate import validate_connector


def test_template_worked_example_passes_the_harness():
    rep = validate_connector(ExampleEdgesConnector(), None)
    assert rep.ok, str(rep)
    names = {c.name for c in rep.checks}
    assert {"contract shape", "builds in engine", "chain condition ∂²=0",
            "betti / signature", "RCDB round-trip", "read-only probe",
            "capability consistency"} <= names


def test_faces_to_csc_roundtrips_a_triangle_face():
    # one triangular face over 3 edges -> a single CSC column with 3 rows
    B2 = np.array([[1.0], [1.0], [1.0]])
    cp, rp, vp = faces_to_csc(B2)
    assert list(cp) == [0, 3]
    assert list(rp) == [0, 1, 2]
    assert list(vp) == [1.0, 1.0, 1.0]


def test_to_rexgraph_fills_a_cycle_with_a_valid_face():
    # triangle + its face -> cycle filled: b1 drops to 0, chain stays valid
    B2 = np.array([[1.0], [1.0], [1.0]])
    meta = {"vertex_labels": ["A", "B", "C"],
            "edges": [("A", "B"), ("B", "C"), ("C", "A")],
            "source": "mem://tri", "faces": B2}
    g = to_rexgraph((np.array([0, 1, 2]), np.array([1, 2, 0])), meta)
    assert g.chain_valid is True
    assert tuple(g.betti) == (1, 0, 0)


class _BadFaceConnector(BaseConnector):
    CAPABILITIES = Capabilities(faces=True)

    def read(self, source):
        B2 = np.zeros((3, 1)); B2[0, 0] = 1.0        # single edge is not a loop
        return self.result((np.array([0, 1, 2]), np.array([1, 2, 0])),
                           vertex_labels=["A", "B", "C"],
                           edges=[("A", "B"), ("B", "C"), ("C", "A")],
                           faces=B2, source="mem://bad")


class _OverclaimConnector(BaseConnector):
    CAPABILITIES = Capabilities(weights=True)        # claims weights, emits none

    def read(self, source):
        return self.result((np.array([0, 1]), np.array([1, 0])),
                           vertex_labels=["A", "B"],
                           edges=[("A", "B"), ("B", "A")], source="mem://oc")


def test_harness_fails_chain_condition_on_bad_faces():
    rep = validate_connector(_BadFaceConnector(), None)
    assert not rep.ok
    chk = next(c for c in rep.checks if c.name == "chain condition ∂²=0")
    assert not chk.passed


def test_harness_fails_capability_overclaim():
    rep = validate_connector(_OverclaimConnector(), None)
    assert not rep.ok
    chk = next(c for c in rep.checks if c.name == "capability consistency")
    assert not chk.passed


def test_sql_sources_remain_optional_and_do_not_open_on_discovery():
    from agent.connectors import open_connector
    from agent.connectors.sql import SQLConnector
    from agent.connectors.warehouse import WarehouseConnector
    assert isinstance(open_connector("postgresql://host/database"),SQLConnector)
    assert isinstance(open_connector("snowflake://host/database"),WarehouseConnector)
    assert isinstance(open_connector("sqlite"),SQLConnector)


# document / semantic / generic / skeletons / registry

from agent.connectors import open_connector

_SUB = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


def test_document_connector_infers_and_validates():
    from agent.connectors.document import DocumentConnector
    cols = {"users": [{"_id": 1}],
            "posts": [{"_id": 10, "user_id": 1}],
            "comments": [{"_id": 100, "post_id": 10, "user_id": 1}]}
    conn = DocumentConnector()
    rex, meta = conn.read(cols)
    assert "users" in meta["vertex_labels"]
    assert validate_connector(conn, cols).ok


def test_semantic_connector_edges_align_with_rex():
    from agent.connectors.semantic import SemanticConnector
    triples = [("Dog", _SUB, "Animal"), ("Cat", _SUB, "Animal"),
               ("Animal", _SUB, "Organism")]
    conn = SemanticConnector()
    rex, meta = conn.read(triples)
    assert len(meta["edges"]) == rex.nE          # contract aligns with engine
    assert ("Dog", "Animal") in meta["edges"]
    assert validate_connector(conn, triples).ok


def test_generic_connector_weights_and_cycle():
    conn = open_connector("edges")
    rows = [("a", "b", 3.0), ("b", "c", 1.0), ("c", "a", 2.0)]
    rex, meta = conn.read(rows)
    assert meta["weights"] == [3.0, 1.0, 2.0]
    assert tuple(to_rexgraph(rex, meta).betti) == (1, 1, 0)   # a 3 cycle
    assert validate_connector(conn, rows).ok


def test_graph_connector_in_memory_stand_in():
    conn = open_connector("neo4j")
    g = {"nodes": ["Person", "Company", "Address"],
         "relationships": [("Person", "Company"), ("Person", "Address"),
                           ("Company", "Address")]}
    assert validate_connector(conn, g).ok
    with pytest.raises(NotImplementedError):
        conn.read("neo4j://live-server")          # live path needs host driver


def test_stream_connector_in_memory_stand_in():
    conn = open_connector("kafka")
    s = {"topics": ["orders", "shipments", "invoices"],
         "references": [("shipments", "orders"), ("invoices", "orders")]}
    assert validate_connector(conn, s).ok
    with pytest.raises(NotImplementedError):
        conn.read("kafka://live-broker")


def test_registry_routes_schemes_and_rejects_unknown():
    from agent.connectors.document import DocumentConnector
    from agent.connectors.generic import GenericConnector
    from agent.connectors.graph import GraphConnector
    from agent.connectors.semantic import SemanticConnector
    from agent.connectors.stream import StreamConnector
    assert isinstance(open_connector("mongodb://h/db"), DocumentConnector)
    assert isinstance(open_connector("ontology"), SemanticConnector)
    assert isinstance(open_connector("neo4j://h"), GraphConnector)
    assert isinstance(open_connector("kafka://h"), StreamConnector)
    assert isinstance(open_connector("edges"), GenericConnector)
    from agent.connectors.records import RCDBConnector, NativeFileConnector
    assert isinstance(open_connector("rcdb"), RCDBConnector)
    assert isinstance(open_connector("rcbf"), NativeFileConnector)
    assert isinstance(open_connector("rcbd"), NativeFileConnector)
    with pytest.raises(ConnectorError):
        open_connector("ftp://nope")
