"""Bounded, checksummed RCBO streams containing complete ontology source state."""
from __future__ import annotations

from hashlib import sha256
from hmac import compare_digest
from io import BytesIO
from pathlib import Path
import struct

MAGIC = b"RCBO\0\0\0\0"
_HEADER = struct.Struct("<IIQQ")
_MAX_HEADER = 64 << 10
_DEFAULT_MAX_BYTES = 256 << 20


class RCBOFormatError(ValueError):
    """An ontology stream is damaged, unsupported or exceeds admission."""


def _budget(max_bytes):
    if type(max_bytes) is not int or not 0 < max_bytes <= _DEFAULT_MAX_BYTES:
        raise ValueError("RCBO max_bytes requires a positive integer up to 256 MiB")
    return max_bytes


def write_rcbo(stream, ontology, *, max_bytes=_DEFAULT_MAX_BYTES):
    """Write one frame to a binary stream without seeking or closing the stream."""
    from rexgraph.ontology import OntologyState
    from rexgraph.value_codec import pack_value
    if not isinstance(ontology, OntologyState):
        raise TypeError("RCBO requires an OntologyState")
    budget = _budget(max_bytes)
    header = pack_value({"anchor": ontology._anchor, "state_digest": ontology.source_digest,
                         "ontology_digest": ontology.digest,
                         "state_codec": ontology._codec})
    if len(header) > _MAX_HEADER:
        raise RCBOFormatError("RCBO declaration exceeds admission")
    chunks = (MAGIC, _HEADER.pack(1, 0, len(header), len(ontology._state)), header, ontology._state)
    if sum(map(len, chunks)) + 32 > budget:
        raise RCBOFormatError("RCBO exceeds max_bytes")
    digest = sha256()

    def put(data):
        view = memoryview(data)
        while view:
            count = stream.write(view)
            if type(count) is not int or not 0 < count <= len(view):
                raise OSError("RCBO stream did not accept the complete frame")
            view = view[count:]

    for chunk in chunks:
        put(chunk); digest.update(chunk)
    put(digest.digest())


def read_rcbo(stream, *, max_bytes=_DEFAULT_MAX_BYTES):
    """Read one complete binary stream with bounded framing and no provider loading."""
    from rexgraph.ontology import OntologyState
    from rexgraph.value_codec import unpack_value
    budget = _budget(max_bytes)
    consumed = 0
    digest = sha256()

    def take(size, *, checksum=True):
        nonlocal consumed
        if size < 0 or consumed + size > budget:
            raise RCBOFormatError("RCBO exceeds max_bytes")
        parts = bytearray()
        while len(parts) < size:
            part = stream.read(min(size - len(parts), 1 << 20))
            if type(part) is not bytes or not part or len(part) > size - len(parts):
                raise RCBOFormatError("truncated or nonbinary RCBO stream")
            parts.extend(part)
        consumed += size
        raw = bytes(parts)
        if checksum:
            digest.update(raw)
        return raw

    try:
        if take(len(MAGIC)) != MAGIC:
            raise RCBOFormatError("invalid RCBO magic")
        version, flags, header_size, state_size = _HEADER.unpack(take(_HEADER.size))
        if version != 1 or flags != 0:
            raise RCBOFormatError("unsupported RCBO version or flags")
        if not 0 < header_size <= _MAX_HEADER or state_size == 0:
            raise RCBOFormatError("invalid RCBO header or state size")
        if consumed + header_size + state_size + 32 > budget:
            raise RCBOFormatError("RCBO declared frame exceeds max_bytes")
        header = unpack_value(take(header_size))
        if (type(header) is not dict or set(header) != {"anchor", "state_digest", "ontology_digest", "state_codec"}
                or header["state_codec"] not in {"rexgraph.rcbf-v6", "rexgraph.safetensors-v10"}):
            raise RCBOFormatError("unsupported RCBO state declaration")
        raw = take(state_size)
        if not compare_digest(digest.digest(), take(32, checksum=False)):
            raise RCBOFormatError("RCBO stream checksum differs")
        if stream.read(1) != b"":
            raise RCBOFormatError("RCBO has trailing data")
        result = OntologyState.from_record(_source_record(raw, header["state_codec"]), anchor=header["anchor"])
        if result.source_digest != header["state_digest"] or result.digest != header["ontology_digest"]:
            raise RCBOFormatError("RCBO source differs from its declared state")
        if header["state_codec"] == "rexgraph.safetensors-v10":
            # Retain the declared legacy payload for owning record-codec replay.
            # A new graph accession uses the current complete RCBF encoding.
            object.__setattr__(result, "_state", raw)
            object.__setattr__(result, "_codec", header["state_codec"])
        return result
    except (TypeError, ValueError, OverflowError, KeyError, struct.error, RecursionError) as exc:
        if isinstance(exc, RCBOFormatError):
            raise
        raise RCBOFormatError("invalid RCBO source or ontology declaration") from exc


def _source_record(raw, codec):
    if codec == "rexgraph.rcbf-v6":
        from .rcbf import _load_state_stream
        return _load_state_stream(BytesIO(raw), max_bytes=_DEFAULT_MAX_BYTES)
    if codec == "rexgraph.safetensors-v10":
        from .safetensors_bridge import load_safetensors_bytes
        return load_safetensors_bytes(raw)
    raise RCBOFormatError("unsupported RCBO source codec")


def ontology_bytes(ontology, *, max_bytes=_DEFAULT_MAX_BYTES):
    stream = BytesIO()
    write_rcbo(stream, ontology, max_bytes=max_bytes)
    return stream.getvalue()


def save_rcbo(path, ontology, *, max_bytes=_DEFAULT_MAX_BYTES):
    """Atomically publish a complete ontology; failed writes preserve the old file."""
    from .publication import staged_publication
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(target) as staged, staged.open("wb") as stream:
        write_rcbo(stream, ontology, max_bytes=max_bytes)
    return target


def load_rcbo(path, *, max_bytes=_DEFAULT_MAX_BYTES):
    with Path(path).open("rb") as stream:
        return read_rcbo(stream, max_bytes=max_bytes)


def save_ontology(path, ontology, *, format=None, **options):
    """Save an ontology through RCBO or an ordinary complete graph container."""
    from rexgraph.ontology import OntologyState
    from . import _detect_format, save
    if not isinstance(ontology, OntologyState):
        raise TypeError("ontology save requires an OntologyState")
    kind = _detect_format(path, format)
    return save(path, ontology if kind == "rcbo" else ontology.record(), format=kind, **options)


def load_ontology(path, *, format=None, anchor=(1, 0), **options):
    """Read a complete ontology across RCBO and the common graph formats."""
    from rexgraph.ontology import OntologyState
    from . import load
    value = load(path, format=format, **options)
    return value if isinstance(value, OntologyState) else OntologyState.from_record(value, anchor=anchor)


__all__ = ["RCBOFormatError", "read_rcbo", "write_rcbo", "load_rcbo", "save_rcbo",
           "load_ontology", "save_ontology"]
