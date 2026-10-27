"""Native enclosure validation does not claim graph equivalence or axiom proofs."""
from fractions import Fraction

import pytest

from rexgraph import RexGraph
from rexgraph.ontology import OntologyState
from rexgraph.value import ExactTime


def native_source(namespace):
    backend = pytest.importorskip("rexgraph_cpp")
    return backend.NativeSource.from_declaration(dict(profile="declared-role-source-v1",
        namespace=namespace, semantic_scale=2, max_grade=1, role_aspects=("boundary",),
        types=({"name": "retained relation", "min_arity": 0, "max_arity": 4},),
        witnesses=(10, 11), relations=({"grade": 1, "key": 7, "type": 0,
            "occurrences": ({"key": 100, "lower": 10, "exists": True, "negative": True,
                "magnitude": 1, "aspects": (0,)}, {"key": 101, "lower": 11,
                "exists": True, "negative": False, "magnitude": 1, "aspects": (0,)} )},),
        fields=(), metrics=(), carriers=()))


def ontology(owner):
    from rexgraph.native_ontology import PROVIDER, native_ontology_component
    return OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:native",
        context="one", known_at=ExactTime(Fraction(1)),
        signature={"native_sources": (native_ontology_component("component", owner),)},
        universe={"scope": "independent-native-enclosure"}, provider=dict(PROVIDER),
        axioms=({"arbitrary_unimplemented_theory": "retained"},))


def test_actual_native_decode_parent_prefix_and_unproved_dimensions():
    from rexgraph.native_ontology import register_native_ontology_provider
    a = ontology(native_source(99))
    register_native_ontology_provider()
    report = OntologyState.from_bytes(a.to_bytes()).verify(sources={})
    assert report["native_source_integrity"] == "supported"
    assert report["axiom_profile_result"] == report["current_availability"] == "not_evaluated"
    assert report["source_accession"]["graph_to_native_morphism"] == "not_evaluated"
    child = a.revise(a.record(), known_at=ExactTime(Fraction(2)),
                     kind="ontology-revision", maps=("declared-comparison",))
    assert child.verify(sources={}, parent=a)["native_source_integrity"] == "supported"
    with pytest.raises((TypeError, ValueError)):
        child.verify(sources={})
    with pytest.raises(ValueError, match="root"):
        a.verify(sources={}, parent=a)
    different = ontology(native_source(100))
    wrong = a.revise(different.record(), known_at=ExactTime(Fraction(2)),
        kind="ontology-revision", maps=("asserted-comparison",), signature=different.declaration["signature"])
    with pytest.raises(ValueError, match="parent prefix"):
        wrong.verify(sources={}, parent=a)
    altered = a.declaration["signature"]
    altered["native_sources"][0]["identity"]["namespace"] += 1
    bad = a.revise(a.record(), known_at=ExactTime(Fraction(2)), kind="ontology-revision",
                   maps=("comparison",), signature=altered)
    with pytest.raises(ValueError, match="publication"):
        bad.verify(sources={}, parent=a)


def additions():
    return {"profile": "source-signature-additions-v1", "aspects": (
        {"name": "visible", "role": "visibility", "builder_default": True,
         "fill": False, "assignments": ()},), "types": ()}


def test_native_authoring_retains_actual_parent_and_evidence():
    pytest.importorskip("rexgraph_cpp._source_evolution")
    from rexgraph.native_ontology import add_native_ontology_declarations, register_native_ontology_provider
    from rexgraph.tensor_field import FieldSource
    register_native_ontology_provider()
    parent = ontology(native_source(9042))
    source = parent.record()
    reference = FieldSource(source, "ontology", 1)
    child = add_native_ontology_declarations(parent, "component", additions(),
        parent_source=reference, known_at=ExactTime(Fraction(2)))
    assert child.is_child_of(parent)
    assert child.dependencies[-1].matches(reference)
    assert child.declaration["change"]["kind"] == "declaration-recode"
    assert child.declaration["axioms"] == parent.declaration["axioms"]
    assert child.verify(sources={}, parent=parent)["native_source_integrity"] == "supported"
    assert OntologyState.from_bytes(child.to_bytes()).to_bytes() == child.to_bytes()
    with pytest.raises(ValueError, match="selected source"):
        add_native_ontology_declarations(parent, "component", additions(),
            parent_source=FieldSource(child.record(), "ontology", 2), known_at=ExactTime(Fraction(3)))
    with pytest.raises(ValueError, match="positive parent"):
        add_native_ontology_declarations(parent, "component", additions(),
            parent_source=FieldSource(source), known_at=ExactTime(Fraction(2)))
    source.attach_metadata(1, 0, "changed", True)
    with pytest.raises(ValueError, match="changed"):
        add_native_ontology_declarations(parent, "component", additions(),
            parent_source=reference, known_at=ExactTime(Fraction(2)))
