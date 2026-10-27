"""Whole ontology source, selected identity, revision and stream admission."""
from fractions import Fraction
from io import BytesIO
import struct

import pytest

from rexgraph import Absent, Relations, RexGraph, VertexTable
from rexgraph.ontology import OntologyState
from rexgraph.io import FileCatalog, load_ontology, save_ontology
from rexgraph.io.rcbo import RCBOFormatError, read_rcbo, write_rcbo
from rexgraph.object_identity import object_digest
from rexgraph.value import ExactTime, TimeRange
from rexgraph.value_codec import pack_value


def source():
    return RexGraph.from_relations(Relations.from_supports(
        [[0, 1, 2], [2, 2]], vertices=VertexTable(("a", "b", "c", "unobserved")),
        relation_ids=["assertion", "supported-zero"], weights=[Fraction(2, 3), 0],
        attributes={1: {0: {"raw": (b"a\x00\xff", Fraction(1, 7), Absent)}}}))


def ontology(context="observation-a"):
    return OntologyState.create(source(), identity="example:ontology", context=context,
        known_at=ExactTime(Fraction(10)), validity=TimeRange(ExactTime(Fraction(1)), ExactTime(Fraction(3))),
        signature={"rules": ("typed-assertion",), "native_series": b"opaque source retained as data"},
        universe={"scope": "declared-source", "unresolved": ("extension-a", "extension-b")},
        axioms=({"nonempty": "a"},), interpretations=({"meanings": ("first", "second")},))


@pytest.mark.parametrize("extension", ["rcbo", "rcbf", "rcbd", "safetensors"])
def test_complete_roundtrip_and_interoperation_preserve_selected_ontology(tmp_path, extension):
    if extension in {"rcbd", "safetensors"}:
        pytest.importorskip("safetensors")
    original = ontology()
    target = tmp_path / ("ontology." + extension)
    save_ontology(target, original)
    restored = load_ontology(target)
    assert restored.digest == original.digest
    assert restored.to_bytes() == original.to_bytes()
    assert restored.source_digest == object_digest(original.record())
    assert restored.record().relation_supports() == original.record().relation_supports()
    assert restored.record().relation_keys == original.record().relation_keys
    assert pack_value(restored.declaration) == pack_value(original.declaration)
    if extension == "rcbo":
        catalog = FileCatalog([tmp_path])
        assert catalog.info("root0/ontology.rcbo").kind == "rcbo"
        assert object_digest(catalog.load("root0/ontology.rcbo")) == original.source_digest


def test_source_and_declarations_are_owned_and_parent_branching_is_explicit():
    parent = ontology()
    before = parent.to_bytes()
    copy = parent.record()
    copy.attach_metadata(1, 0, "external", {"new": True})
    declaration = parent.declaration
    declaration["universe"]["unresolved"] = ()
    assert parent.to_bytes() == before
    child = parent.revise(copy, known_at=ExactTime(Fraction(20)), kind="ontology-revision",
                          maps=({"profile": "declared-inclusion", "old_source": parent.source_digest},),
                          universe={"scope": "expanded", "new-symbol": "d"}, validity=None)
    assert child.is_child_of(parent) and child.digest != parent.digest
    assert child.declaration["validity"] is None
    assert "new-symbol" not in parent.declaration["universe"]
    assert not child.is_child_of(ontology("observation-b"))
    with pytest.raises((AttributeError, TypeError)):
        parent._anchor = (1, 1)
    with pytest.raises(ValueError, match="maps"):
        parent.revise(copy, known_at=ExactTime(Fraction(20)), kind="source-extension", maps=())
    with pytest.raises(ValueError, match="horizon"):
        parent.revise(copy, known_at=ExactTime(Fraction(9)), kind="restriction", maps=("map",))
    with pytest.raises(LookupError, match="provider"):
        parent.verify(sources={})


def test_two_anchors_in_the_identical_graph_do_not_establish_the_same_parent():
    graph = ontology().record()
    second = ontology().declaration
    second["axioms"] = ({"nonempty": "b"},)
    graph.attach_metadata(1, 1, "relational_ontology", second)
    a = OntologyState.from_record(graph, anchor=(1, 0))
    b = OntologyState.from_record(graph, anchor=(1, 1))
    assert a.source_digest == b.source_digest and a.digest != b.digest
    child = a.revise(a.record(), known_at=ExactTime(Fraction(20)), kind="ontology-revision", maps=("comparison",))
    assert child.is_child_of(a) and not child.is_child_of(b)
    assert OntologyState.from_bytes(b.to_bytes()).digest == b.digest


class ShortStream(BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 7))

    def write(self, data):
        return super().write(data[:11])

    def seek(self, *args):
        raise AssertionError("the format requires no stream seek")


def test_forward_only_short_reads_and_writes_are_complete():
    value = ontology()
    stream = ShortStream()
    write_rcbo(stream, value)
    raw = stream.getvalue()
    assert read_rcbo(ShortStream(raw)).digest == value.digest
    assert raw == value.to_bytes()


@pytest.mark.parametrize("damage", ["truncated", "trailing", "checksum", "version", "flags", "oversize"])
def test_damaged_or_unsupported_frames_never_decode_sources(damage, monkeypatch):
    raw = bytearray(ontology().to_bytes())
    if damage == "truncated": raw = raw[:-1]
    elif damage == "trailing": raw += b"x"
    elif damage == "checksum": raw[-1] ^= 1
    elif damage == "version": struct.pack_into("<I", raw, 8, 2)
    elif damage == "flags": struct.pack_into("<I", raw, 12, 1)
    else: struct.pack_into("<Q", raw, 24, 1 << 40)
    import rexgraph.io.rcbo as container
    def forbidden(*args, **kwargs): pytest.fail("damaged frame reached graph decoder")
    monkeypatch.setattr(container, "_source_record", forbidden)
    with pytest.raises(RCBOFormatError):
        read_rcbo(BytesIO(bytes(raw)))


def test_atomic_failed_write_preserves_existing_file_and_admission(tmp_path):
    path = tmp_path / "ontology.rcbo"
    save_ontology(path, ontology())
    old = path.read_bytes()
    with pytest.raises(ValueError, match="max_bytes"):
        save_ontology(path, ontology(), max_bytes=32)
    assert path.read_bytes() == old
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ontology.rcbo"]
    with pytest.raises(RCBOFormatError, match="max_bytes"):
        read_rcbo(BytesIO(old), max_bytes=len(old)-1)
    with pytest.raises(TypeError, match="ExactTime"):
        ontology().require_horizon(10.0)
    with pytest.raises(ValueError, match="horizon"):
        ontology().require_horizon(ExactTime(Fraction(9)))


def test_rcbo_creation_needs_no_optional_tensor_backend(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "safetensors" or name.startswith("safetensors."):
            pytest.fail("RCBO construction imported an optional tensor backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    value = ontology()
    assert OntologyState.from_bytes(value.to_bytes()).digest == value.digest


def test_declared_legacy_payload_roundtrips_without_rewriting_its_original_bytes():
    pytest.importorskip("safetensors")
    from hashlib import sha256
    from rexgraph.io.rcbo import MAGIC
    from rexgraph.io.safetensors_bridge import state_to_safetensors_bytes
    from rexgraph.state import to_state
    value = ontology()
    raw = state_to_safetensors_bytes(to_state(value.record()))
    header = pack_value({"anchor": (1, 0), "state_digest": value.source_digest,
        "ontology_digest": value.digest, "state_codec": "rexgraph.safetensors-v10"})
    frame = MAGIC + struct.pack("<IIQQ", 1, 0, len(header), len(raw)) + header + raw
    frame += sha256(frame).digest()
    restored = OntologyState.from_bytes(frame)
    assert restored.to_bytes() == frame
    assert restored.digest == value.digest
    assert object_digest(restored.record()) == value.source_digest
