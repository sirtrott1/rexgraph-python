"""Ontology queries retain complete source state and explicit runtime validation."""
from fractions import Fraction

import pytest

from rexgraph import RexGraph
from rexgraph.ontology import OntologyState
from rexgraph.value import ExactTime
from rcql import Executor, Result, parse


def ontology():
    return OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:types",
        context="first", known_at=ExactTime(Fraction(10)), signature={"types": ("edge",)},
        universe={"scope": "original", "unknown": ("a", "b")})


def test_typed_read_export_import_and_full_result_roundtrip(tmp_path):
    value = ontology()
    runtime = Executor(sources={"r": value.record()})
    query = parse('FROM $r LET o = ONTOLOGY_AT(ONTOLOGY_READ(), 10) '
                  'RETURN ONTOLOGY_INFO(o), ONTOLOGY_EXPORT(o), ONTOLOGY_RECORD(o), o')
    result = runtime.execute(query)
    assert result.values[0]["digest"] == value.digest
    assert OntologyState.from_bytes(result.values[1]).digest == value.digest
    assert OntologyState.from_record(result.values[2]).digest == value.digest
    assert result.values[3].digest == value.digest
    restored = Result.from_bytes(result.to_bytes())
    assert restored.values[3].to_bytes() == value.to_bytes()
    received = Executor(sources={"r": value.record()}, params={"archive": value.to_bytes()})
    assert received.execute(parse('FROM $r RETURN ONTOLOGY_INFO(ONTOLOGY_IMPORT($archive))')).values[0]["digest"] == value.digest
    with pytest.raises(ValueError, match="horizon"):
        runtime.execute(parse('FROM $r RETURN ONTOLOGY_AT(ONTOLOGY_READ(), 9)'))
    with pytest.raises(LookupError, match="provider"):
        runtime.execute(parse('FROM $r RETURN ONTOLOGY_VERIFY(ONTOLOGY_READ(), 10)'))


def test_file_source_and_planning_do_not_invoke_a_provider(tmp_path, monkeypatch):
    from rexgraph.io import FileCatalog, save_ontology
    value = ontology()
    save_ontology(tmp_path / "types.rcbo", value)
    runtime = Executor(sources={"files": FileCatalog([tmp_path])})
    query = parse('FROM FILE("files", "root0/types.rcbo") RETURN ONTOLOGY_INFO(ONTOLOGY_READ())')
    assert runtime.execute(query).values[0]["digest"] == value.digest
    def forbidden(*args, **kwargs): pytest.fail("planning executed an ontology provider")
    monkeypatch.setattr(OntologyState, "verify", forbidden)
    explanation = Executor(sources={"r": value.record()}).execute(
        parse('EXPLAIN FROM $r RETURN ONTOLOGY_VERIFY(ONTOLOGY_READ(), 10)'))
    assert explanation is not None


def test_declared_revision_requires_training_policy_and_keeps_parent_source():
    from rcql import BoundSource, SourcePolicy
    value = ontology()
    parameters = {"changes": {"known_at": 20, "kind": "ontology-revision", "maps": ("declared-map",),
                              "universe": {"scope": "new", "unknown": ("a", "b", "c")}},
                  "graph": value.record()}
    query = parse('FROM $r RETURN ONTOLOGY_REVISE(ONTOLOGY_READ(), $graph, $changes)')
    denied = Executor(sources={"r": BoundSource(value.record(), SourcePolicy.allow("read"))}, params=parameters)
    with pytest.raises(PermissionError):
        denied.execute(query)
    runtime = Executor(sources={"r": BoundSource(value.record(), SourcePolicy.allow("read", "train"))}, params=parameters)
    child = runtime.execute(query).values[0]
    assert child.is_child_of(value)
    assert child.declaration["universe"]["unknown"] == ("a", "b", "c")
    assert value.declaration["universe"]["unknown"] == ("a", "b")


def test_native_revision_rechecks_selected_parent_and_source_permissions():
    native = pytest.importorskip("rexgraph_cpp")
    pytest.importorskip("rexgraph_cpp._source_evolution")
    from rcql import BoundSource, SourcePolicy
    from rcql.types import SourceRef
    from rexgraph.native_ontology import PROVIDER, native_ontology_component
    owner = native.NativeSource.from_declaration({"profile": "declared-exact-source-v1",
        "namespace": 9061, "semantic_scale": 1, "max_grade": 1,
        "types": (), "witnesses": (10,), "relations": (), "carriers": (), "fields": (), "metrics": ()})
    parent = OntologyState.create(RexGraph.from_graph([0], [1]), identity="example:typed-revision",
        context="one", known_at=ExactTime(Fraction(1)), universe={}, provider=dict(PROVIDER),
        signature={"native_sources": (native_ontology_component("part", owner),)})
    policy = SourcePolicy.allow("read", "train")
    ref = SourceRef("r", state_digest=parent.source_digest, policy_digest=policy.digest,
                    record_id="ontology", record_version=1)
    params = {"parent": parent, "spec": {"profile": "source-signature-additions-v1", "aspects": (),
        "types": ({"name": "extra", "min_arity": 0, "max_arity": 9},)}}
    query = parse('FROM $r RETURN ONTOLOGY_NATIVE_ADD($parent, "part", $spec, 2)')
    runtime = Executor(sources={"r": BoundSource(parent.record(), policy, ref=ref)}, params=params)
    child = runtime.execute(query).values[0]
    assert child.is_child_of(parent)
    assert Result.from_bytes(runtime.execute(query).to_bytes()).values[0].digest == child.digest
    denied = Executor(sources={"r": BoundSource(parent.record(), SourcePolicy.allow("read"), ref=ref)}, params=params)
    with pytest.raises(PermissionError):
        denied.execute(query)
    wrong = Executor(sources={"r": BoundSource(parent.record(), policy, ref=ref)},
                     params=dict(params, parent=child))
    with pytest.raises(ValueError, match="(dependency|selected query source)"):
        wrong.execute(query)


def test_ontology_signature_registration_keeps_metadata_planning_lightweight():
    import subprocess
    import sys
    program = (
        "import sys, rcql, rcql.planning; "
        "signature = rcql.lookup('ONTOLOGY_REVISE'); "
        "assert not signature.inputs[1].accepts(object()); "
        "assert 'numpy' not in sys.modules; "
        "assert 'rcql.operators' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", program], check=True)


@pytest.mark.parametrize("graph", [{}, "not a graph", (1, 2)])
def test_ontology_revision_rejects_non_graph_during_explain(graph):
    value = ontology()
    changes = {"known_at": 20, "kind": "ontology-revision", "maps": ("declared-map",)}
    runtime = Executor(sources={"r": value.record()}, params={"graph": graph, "changes": changes})
    with pytest.raises(TypeError, match="argument graph"):
        runtime.execute(parse('EXPLAIN FROM $r RETURN ONTOLOGY_REVISE(ONTOLOGY_READ(), $graph, $changes)'))
