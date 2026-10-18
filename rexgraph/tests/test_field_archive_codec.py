"""Portable field envelopes retain bytes; executable replay is provider owned."""
from dataclasses import replace
from hashlib import sha256

import pytest

from rexgraph.field_archive import FIELD_ARCHIVE_METHOD, FIELD_ARCHIVE_SCHEMA
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_state import ModelOutput
from rexgraph.source_binding import CppSourceIdentity, CppSourceReference
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_spec, unpack_spec


def archive():
    data = b"opaque native result envelope fixture"
    return {"schema": FIELD_ARCHIVE_SCHEMA, "data": data, "sha256": sha256(data).hexdigest(),
            "program_sha256": "c" * 64, "engine_receipt": {"engine_digest": "d" * 64}}


def output(record=None):
    source = CppSourceReference(None, native=CppSourceIdentity(
        88, 2, 0, "a" * 64, "b" * 64, (("10", "11"), ("100",), ())))
    return ModelOutput([], CoordinateSpace("program-readout", ()), (), source, "e" * 64,
                       "rational", FIELD_ARCHIVE_METHOD, field_archive=archive() if record is None else record)


@pytest.mark.parametrize("native", [False, True])
def test_archive_output_roundtrip_keeps_full_bytes_and_empty_readout(native):
    original = output()
    packet = pack_model(original, native=native)
    assert unpack_spec(packet)["version"] == 3
    restored = unpack_model(packet)
    assert restored.coefficient_digest == original.coefficient_digest
    assert restored.source.matches(original.source)
    assert restored.field_archive == original.field_archive
    assert restored.values.size == 0
    with pytest.raises(TypeError, match="explicit native node and stalk selection"):
        restored.tensor()


def test_archive_freezes_receipts_and_seals_native_bytes():
    declaration = archive()
    retained = output(declaration)
    declaration["engine_receipt"]["engine_digest"] = "f" * 64
    assert retained.field_archive["engine_receipt"]["engine_digest"] == "d" * 64
    with pytest.raises(TypeError):
        retained.field_archive["data"] = b"changed"
    changed = archive()
    changed["data"] += b" changed"
    changed["sha256"] = sha256(changed["data"]).hexdigest()
    object.__setattr__(retained, "field_archive", changed)
    with pytest.raises(ValueError, match="model output changed"):
        retained.check_state()


@pytest.mark.parametrize("kind", ["extra", "missing", "schema", "mutable", "empty", "digest",
                                 "program", "uppercase_program", "receipt", "handle"])
def test_archive_requires_closed_immutable_portable_declarations(kind):
    value = archive()
    if kind == "extra": value["owner"] = "hidden"
    elif kind == "missing": del value["program_sha256"]
    elif kind == "schema": value["schema"] = "unknown"
    elif kind == "mutable": value["data"] = bytearray(value["data"])
    elif kind == "empty": value.update(data=b"", sha256=sha256(b"").hexdigest())
    elif kind == "digest": value["sha256"] = "f" * 64
    elif kind == "program": value["program_sha256"] = "g" * 64
    elif kind == "uppercase_program": value["program_sha256"] = "C" * 64
    elif kind == "receipt": value["engine_receipt"] = {}
    else: value["engine_receipt"]["owner"] = object()
    with pytest.raises((TypeError, ValueError)):
        output(value)


@pytest.mark.parametrize("changes", [{"method": "other"}, {"arithmetic": "approximate"},
                                    {"field_archive": None}])
def test_archive_has_a_declared_exact_program_output_contract(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(output(), **changes)


@pytest.mark.parametrize("version", [1, 2])
def test_archive_cannot_be_hidden_in_legacy_model_schema(version):
    packet = pack_model(output(), native=True)
    declaration = unpack_spec(packet)
    declaration["version"] = version
    packet["spec"] = pack_spec(declaration, [name for name in packet if name != "spec"], native=True)
    with pytest.raises(ValueError):
        unpack_model(packet)


def test_schema_three_requires_a_nonempty_archive_value():
    original = output()
    legacy = ModelOutput([], original.space, (), original.source, original.model_digest,
                         "rational", "portable-readout")
    packet = pack_model(legacy, native=True)
    declaration = unpack_spec(packet)
    declaration.update(version=3, field_archive=["none"])
    packet["spec"] = pack_spec(declaration, [name for name in packet if name != "spec"], native=True)
    with pytest.raises(ValueError):
        unpack_model(packet)
