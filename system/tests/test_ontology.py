"""Ontology previews keep source identity and omit complete archive payloads."""
from fractions import Fraction
import json

from rexgraph import RexGraph
from rexgraph.ontology import OntologyState
from rexgraph.value import ExactTime
from system.serialize import json_value


def test_preview_is_bounded_without_decoding_the_complete_ontology_again(monkeypatch):
    value = OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:ontology",
        context="one", known_at=ExactTime(Fraction(1)), signature={"native_archive": b"x" * 200_000},
        universe={"alternatives": ("a", "b")}, axioms=({"unknown": "retained"},))
    def forbidden(*args, **kwargs):
        raise AssertionError("inspection expanded a complete ontology archive")
    monkeypatch.setattr(OntologyState, "declaration", property(forbidden))
    preview = json_value(value, max_values=2)
    assert preview["digest"] == value.digest and preview["axiom_count"] == 1
    assert preview["mathematical_validation"] == "requires-explicit-provider"
    assert len(json.dumps(preview)) < 2000
