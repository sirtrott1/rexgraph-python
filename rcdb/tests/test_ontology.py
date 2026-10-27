"""Ontology snapshots share the existing typed codec and checked copy boundary."""
from contextlib import closing
from fractions import Fraction

import pytest

from rcdb import LocalStore, MemoryStore, ONTOLOGY_CODEC, copy_record
from rcdb.header import DEFAULT_RECORD_CODECS
from rexgraph import RexGraph
from rexgraph.ontology import OntologyState
from rexgraph.value import ExactTime


def ontology():
    return OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:types",
        context="context-a", known_at=ExactTime(Fraction(1)), signature={"types": ("relation",)},
        universe={"scope": "declared", "unknown": ("a", "b")})


def test_typed_versions_reopen_and_cross_store_copy_preserve_complete_objects(tmp_path):
    a = ontology()
    b = a.revise(a.record(), known_at=ExactTime(Fraction(2)), kind="ontology-revision",
                 maps=("declared-identity-comparison",), axioms=({"new": "constraint"},))
    with closing(LocalStore(tmp_path / "origin")) as store:
        store.put_record("ontology", a, codec=ONTOLOGY_CODEC, tx_time=10.0, expected_version=0)
        store.put_record("ontology", b, codec=ONTOLOGY_CODEC, tx_time=20.0, expected_version=1)
        assert store.get("ontology", as_of=15.0).digest == a.digest
        assert store.get_version("ontology", 2).is_child_of(a)
    with closing(LocalStore(tmp_path / "origin")) as store, closing(MemoryStore()) as destination:
        assert store.get_version("ontology", 1).to_bytes() == a.to_bytes()
        receipt = copy_record(store, destination, store.read_record("ontology", version=2).record,
                              return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        assert destination.get_version("ontology", receipt.destination_version).to_bytes() == b.to_bytes()


def test_old_store_codec_inventory_is_not_silently_extended(tmp_path):
    old = tuple(ref for ref in DEFAULT_RECORD_CODECS if ref != ONTOLOGY_CODEC)
    with closing(LocalStore(tmp_path / "old", record_codecs=old)) as store:
        with pytest.raises(ValueError):
            store.put_record("ontology", ontology(), codec=ONTOLOGY_CODEC)
    with closing(LocalStore(tmp_path / "old")) as store:
        assert ONTOLOGY_CODEC not in store.header.record_codecs
