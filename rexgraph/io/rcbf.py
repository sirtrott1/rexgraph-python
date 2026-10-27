"""Sequential RCBF containers for complete canonical static and temporal state.

Version 6 stores the common state header and named tensors with bounded framing
and a complete stream digest. Versions 4 and 5 retain their checked C0--C2 import
contract, including explicit selection of a current snapshot from temporal tails.
"""
from __future__ import annotations

import hashlib
import hmac
import struct
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np

__all__ = ["RCBFFormatError", "is_rcbf_file", "load_rcbf", "save_rcbf"]


MAGIC = b"RCBF\0\0\0\0"
LEGACY_MAGIC = b"REXFILE\0"
_HEADER = struct.Struct("<7IQ")
_SUPPORTED_VERSIONS = frozenset({4, 5})
_STATE_VERSION = 6
_STATE_HEADER = struct.Struct("<IIQQ")
_STATE_ENTRY = struct.Struct("<QQ")
_MAX_STATE_HEADER = 16 * 1024 * 1024
_MAX_STATE_TENSORS = 100_000
_DEFAULT_MAX_BYTES = 1024 * 1024 * 1024
_MAX_NAME_BYTES = 1024 * 1024
_MAX_CELLS = 10_000_000
_MAX_SIGNALS = 100_000
_MAX_HARMONIC_DIM = 100_000
_MAX_EXACT_F64_INT = 2**53
_MAX_ATTRIBUTE_ROWS = 10_000_000
_FIXED_NAME_BYTES = 64

_FLAG_ATTRS = 0x1
_FLAG_STATE_FULL = 0x2
_FLAG_ATTR_STR = 0x4
_FLAG_SLOT_ATTR = 0x8
_FLAG_ATTR_KIND = 0x10


class RCBFFormatError(ValueError):
    """The stream is malformed or cannot be represented as a RexGraph exactly."""


def _byte_limit(max_bytes):
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("RCBF max_bytes must be a positive integer")
    return max_bytes


def save_rcbf(path, obj, *, max_bytes=_DEFAULT_MAX_BYTES):
    """Atomically write complete canonical state as an RCBF version 6 stream.

    The byte budget includes framing and the checksum. Tensors are encoded and
    written individually using the existing closed native value codec. Optional
    derived caches are not calculated or serialized by this container.
    """
    from rexgraph.graph import RexGraph, TemporalRex
    from .publication import staged_publication

    max_bytes = _byte_limit(max_bytes)
    if isinstance(obj, TemporalRex):
        from rexgraph.temporal_state import to_temporal_state
        state = to_temporal_state(obj)
    elif isinstance(obj, RexGraph):
        from rexgraph.state import to_state
        state = to_state(obj)
    else:
        raise TypeError("RCBF requires RexGraph or TemporalRex")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(target) as staged, staged.open("wb") as handle:
        _write_state_stream(handle, state, max_bytes=max_bytes)
    return target


def _write_state_stream(handle, state, *, max_bytes=_DEFAULT_MAX_BYTES):
    """Write canonical complete state; containers share this exact wire codec."""
    from rexgraph.value_codec import pack_value
    max_bytes = _byte_limit(max_bytes)
    header = pack_value(state.header)
    names = sorted(state.tensors)
    if len(header) > _MAX_STATE_HEADER or len(names) > _MAX_STATE_TENSORS:
        raise ValueError("RCBF state header or tensor count exceeds its limit")
    digest = hashlib.sha256()
    written = 0

    def put(data):
        nonlocal written
        if written + len(data) + digest.digest_size > max_bytes:
            raise ValueError("RCBF state exceeds max_bytes")
        handle.write(data)
        digest.update(data)
        written += len(data)

    put(MAGIC)
    put(_STATE_HEADER.pack(_STATE_VERSION, 0, len(header), len(names)))
    put(header)
    for name in names:
        encoded = name.encode("utf-8")
        if not encoded or len(encoded) > _MAX_NAME_BYTES:
            raise ValueError("RCBF tensor name exceeds its limit")
        payload = pack_value(state.tensors[name])
        put(_STATE_ENTRY.pack(len(encoded), len(payload)))
        put(encoded)
        put(payload)
    handle.write(digest.digest())


def _load_state_stream(handle, *, max_bytes):
    from rexgraph.value_codec import unpack_value

    handle.seek(0, 2)
    size = handle.tell()
    if size > _byte_limit(max_bytes):
        raise RCBFFormatError("RCBF state exceeds max_bytes")
    end = size - hashlib.sha256().digest_size
    handle.seek(0)
    digest = hashlib.sha256()

    def take(count):
        if count < 0 or handle.tell() + count > end:
            raise RCBFFormatError("truncated RCBF state frame")
        value = _read_exact(handle, count, section="state frame")
        digest.update(value)
        return value

    try:
        if take(8) != MAGIC:
            raise RCBFFormatError("RCBF version 6 requires the RCBF magic")
        version, flags, header_size, count = _STATE_HEADER.unpack(take(_STATE_HEADER.size))
        if version != _STATE_VERSION or flags != 0:
            raise RCBFFormatError("unsupported RCBF state version or flags")
        if header_size > _MAX_STATE_HEADER or count > _MAX_STATE_TENSORS:
            raise RCBFFormatError("RCBF state header or tensor count exceeds its limit")
        header = unpack_value(take(header_size))
        if type(header) is not dict:
            raise RCBFFormatError("RCBF state header must be a mapping")
        tensors = {}
        previous = None
        for _ in range(count):
            name_size, value_size = _STATE_ENTRY.unpack(take(_STATE_ENTRY.size))
            if not 0 < name_size <= _MAX_NAME_BYTES:
                raise RCBFFormatError("RCBF tensor name exceeds its limit")
            name = take(name_size).decode("utf-8")
            if previous is not None and name <= previous:
                raise RCBFFormatError("RCBF tensor names must be distinct and ordered")
            value = unpack_value(take(value_size))
            if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                raise RCBFFormatError("RCBF state tensors require closed numeric or string arrays")
            tensors[name] = value
            previous = name
        if handle.tell() != end:
            raise RCBFFormatError("RCBF state has unclaimed trailing data")
        if not hmac.compare_digest(digest.digest(), _read_exact(handle, 32, section="checksum")):
            raise RCBFFormatError("RCBF stream checksum differs")
        if header.get("object_type") == "TemporalRex":
            from rexgraph.temporal_state import TemporalState, from_temporal_state
            return from_temporal_state(TemporalState(tensors, header))
        if header.get("object_type") != "RexGraph":
            raise RCBFFormatError("unsupported RCBF state object type")
        from rexgraph.state import RexState, from_state
        return from_state(RexState(tensors, header))
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        if isinstance(exc, RCBFFormatError):
            raise
        raise RCBFFormatError(f"invalid RCBF canonical state: {exc}") from exc


def _read_exact(handle: BinaryIO, size: int, *, section: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise RCBFFormatError(f"truncated RCBF {section}")
    return data


def _require_remaining(handle: BinaryIO, size: int, *, section: str) -> None:
    position = handle.tell()
    handle.seek(0, 2)
    remaining = handle.tell() - position
    handle.seek(position)
    if size > remaining:
        raise RCBFFormatError(f"truncated RCBF {section}")


def _read_array(handle: BinaryIO, dtype: str, count: int, *, section: str) -> np.ndarray:
    if count < 0:
        raise RCBFFormatError(f"invalid negative RCBF {section} count")
    itemsize = np.dtype(dtype).itemsize
    _require_remaining(handle, count * itemsize, section=section)
    return np.frombuffer(
        _read_exact(handle, count * itemsize, section=section), dtype=dtype
    ).copy()


def _read_cstring(handle: BinaryIO, *, section: str) -> str:
    data = bytearray()
    while len(data) <= _MAX_NAME_BYTES:
        byte = _read_exact(handle, 1, section=section)
        if byte == b"\0":
            try:
                return bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RCBFFormatError(f"invalid UTF-8 in RCBF {section}") from exc
        data.extend(byte)
    raise RCBFFormatError(f"RCBF {section} exceeds {_MAX_NAME_BYTES} bytes")


def _read_fixed_string(handle: BinaryIO, size: int, *, section: str) -> str:
    raw = _read_exact(handle, size, section=section).split(b"\0", 1)[0]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RCBFFormatError(f"invalid UTF-8 in RCBF {section}") from exc


def _read_text(handle: BinaryIO, size: int, *, section: str) -> str:
    if size > _MAX_NAME_BYTES:
        raise RCBFFormatError(f"RCBF {section} exceeds {_MAX_NAME_BYTES} bytes")
    try:
        return _read_exact(handle, size, section=section).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RCBFFormatError(f"invalid UTF-8 in RCBF {section}") from exc


def _require_integral(values: np.ndarray, *, section: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise RCBFFormatError(
            f"RCBF {section} must use finite integral coefficients for an exact import"
        )
    if np.any(np.abs(values) > _MAX_EXACT_F64_INT):
        raise RCBFFormatError(
            f"RCBF {section} exceeds the exact integer range of its float64 carrier"
        )
    return np.ascontiguousarray(values.astype(np.int64))


def _skip(handle: BinaryIO, size: int, *, section: str) -> None:
    _require_remaining(handle, size, section=section)
    handle.seek(size, 1)


def _read_attribute_tail(
    handle: BinaryIO,
    *,
    flags: int,
    n_edges: int,
) -> tuple[list[list[dict[str, object]]], list[list[dict[str, object]]]]:
    """Read the current C1 and C1-slot metadata tail without retyping it."""
    attributes: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
    slots: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
    if not (flags & _FLAG_ATTRS):
        if flags & (_FLAG_ATTR_KIND | _FLAG_ATTR_STR):
            raise RCBFFormatError("RCBF attribute text/kind section lacks attribute counts")
    counts = (
        _read_array(handle, "<u2", n_edges, section="edge attribute counts")
        if flags & _FLAG_ATTRS else np.zeros(n_edges, dtype="<u2")
    )
    total = int(np.asarray(counts, dtype=np.uint64).sum())
    if total > _MAX_ATTRIBUTE_ROWS:
        raise RCBFFormatError("RCBF edge attribute count exceeds the import limit")
    for edge, count in enumerate(counts):
        for attribute in range(int(count)):
            key = _read_fixed_string(
                handle,
                _FIXED_NAME_BYTES,
                section=f"edge {edge} attribute {attribute} key",
            )
            numeric = struct.unpack(
                "<d",
                _read_exact(
                    handle,
                    8,
                    section=f"edge {edge} attribute {attribute} numeric value",
                ),
            )[0]
            if not np.isfinite(numeric):
                raise RCBFFormatError("RCBF edge attribute numeric values must be finite")
            attributes[edge].append({"key": key, "kind": "number", "value": numeric})

    if flags & _FLAG_ATTR_KIND:
        kind_names = {0: "number", 1: "text", 2: "reference"}
        for edge, rows in enumerate(attributes):
            for attribute, row in enumerate(rows):
                kind, grade, size = struct.unpack(
                    "<BBH",
                    _read_exact(
                        handle,
                        4,
                        section=f"edge {edge} attribute {attribute} kind",
                    ),
                )
                if kind not in kind_names:
                    raise RCBFFormatError("RCBF edge attribute has an unknown kind")
                row["kind"] = kind_names[kind]
                if kind == 0:
                    continue
                text = _read_text(
                    handle,
                    size,
                    section=f"edge {edge} attribute {attribute} text",
                )
                row["value"] = text
                if kind == 2:
                    row["reference_grade"] = int(grade)
    elif flags & _FLAG_ATTR_STR:
        for edge, rows in enumerate(attributes):
            for attribute, row in enumerate(rows):
                size = struct.unpack(
                    "<H",
                    _read_exact(
                        handle,
                        2,
                        section=f"edge {edge} attribute {attribute} text length",
                    ),
                )[0]
                if size:
                    row["kind"] = "text"
                    row["value"] = _read_text(
                        handle,
                        size,
                        section=f"edge {edge} attribute {attribute} text",
                    )

    if flags & _FLAG_SLOT_ATTR:
        for edge in range(n_edges):
            count = struct.unpack(
                "<H", _read_exact(handle, 2, section=f"edge {edge} slot attribute count")
            )[0]
            if count > _MAX_ATTRIBUTE_ROWS:
                raise RCBFFormatError("RCBF slot attribute count exceeds the import limit")
            for attribute in range(count):
                slot = struct.unpack(
                    "<i",
                    _read_exact(
                        handle,
                        4,
                        section=f"edge {edge} slot attribute {attribute} slot",
                    ),
                )[0]
                key = _read_fixed_string(
                    handle,
                    _FIXED_NAME_BYTES,
                    section=f"edge {edge} slot attribute {attribute} key",
                )
                numeric = struct.unpack(
                    "<d",
                    _read_exact(
                        handle,
                        8,
                        section=f"edge {edge} slot attribute {attribute} numeric value",
                    ),
                )[0]
                if not np.isfinite(numeric):
                    raise RCBFFormatError("RCBF slot attribute numeric values must be finite")
                size = struct.unpack(
                    "<H",
                    _read_exact(
                        handle,
                        2,
                        section=f"edge {edge} slot attribute {attribute} text length",
                    ),
                )[0]
                row: dict[str, object] = {"slot": int(slot), "key": key, "value": numeric}
                if size:
                    row["kind"] = "text"
                    row["value"] = _read_text(
                        handle,
                        size,
                        section=f"edge {edge} slot attribute {attribute} text",
                    )
                else:
                    row["kind"] = "number"
                slots[edge].append(row)
    return attributes, slots


def _chain_is_exact(
    boundary_ptr: np.ndarray,
    boundary_idx: np.ndarray,
    face_ptr: np.ndarray,
    face_rows: np.ndarray,
    face_values: np.ndarray,
) -> bool:
    """Check ``B1 B2 == 0`` over rational coefficients, without float tolerance."""
    for face in range(face_ptr.size - 1):
        accumulated: dict[int, Fraction] = {}
        for offset in range(int(face_ptr[face]), int(face_ptr[face + 1])):
            edge = int(face_rows[offset])
            coefficient = int(face_values[offset])
            lo, hi = int(boundary_ptr[edge]), int(boundary_ptr[edge + 1])
            support = boundary_idx[lo:hi]
            arity = int(support.size)
            if arity == 1:
                vertex = int(support[0])
                accumulated[vertex] = accumulated.get(vertex, Fraction()) + coefficient
                continue
            if arity == 2 and int(support[0]) == int(support[1]):
                continue
            head = int(support[0])
            accumulated[head] = accumulated.get(head, Fraction()) - coefficient
            share = Fraction(coefficient, arity - 1)
            for vertex in support[1:]:
                item = int(vertex)
                accumulated[item] = accumulated.get(item, Fraction()) + share
        if any(value for value in accumulated.values()):
            return False
    return True


def is_rcbf_file(path: str | Path) -> bool:
    """Return whether *path* has an RCBF or legacy REXFILE magic prefix."""
    try:
        with Path(path).open("rb") as handle:
            return _read_exact(handle, 8, section="magic") in {MAGIC, LEGACY_MAGIC}
    except (OSError, RCBFFormatError):
        return False


def load_rcbf(path: str | Path, *, allow_current_snapshot: bool = False,
              max_bytes: int = _DEFAULT_MAX_BYTES):
    """Read complete version 6 state or a supported legacy C0--C2 snapshot.

    Version 6 uses the common static/temporal state decoder and a total byte
    budget. Legacy versions 4 and 5 require homogeneous C1 directedness, finite C1
    weights, and integral C2 coefficients that satisfy the chain condition over
    the rationals. Signals are preserved as metadata and harmonic data is not
    treated as a shared canonical carrier. A stream with stored temporal states
    raises unless ``allow_current_snapshot=True`` expressly requests only its
    current C0--C2 state.
    """
    source = Path(path)
    with source.open("rb") as handle:
        magic = _read_exact(handle, 8, section="magic")
        if magic not in {MAGIC, LEGACY_MAGIC}:
            raise RCBFFormatError("not an RCBF stream (unknown magic)")
        version = struct.unpack("<I", _read_exact(handle, 4, section="version"))[0]
        if version == _STATE_VERSION:
            return _load_state_stream(handle, max_bytes=max_bytes)
        handle.seek(8)
        version, flags, n_vertices, n_edges, n_faces, n_signals, harmonic_dim, _reserved = (
            _HEADER.unpack(_read_exact(handle, _HEADER.size, section="header"))
        )
        if version not in _SUPPORTED_VERSIONS:
            raise RCBFFormatError(
                f"unsupported RCBF version {version}; supported versions are 4 and 5"
            )
        if max(n_vertices, n_edges, n_faces) > _MAX_CELLS:
            raise RCBFFormatError(
                f"RCBF C0--C2 cell count exceeds the {_MAX_CELLS:,} import limit"
            )
        if n_signals > _MAX_SIGNALS or harmonic_dim > _MAX_HARMONIC_DIM:
            raise RCBFFormatError("RCBF signal or harmonic dimension exceeds the import limit")

        labels = [
            _read_cstring(handle, section=f"vertex label {vertex}")
            for vertex in range(n_vertices)
        ]

        # ``src`` and ``tgt`` are legacy pairwise convenience fields.  The following
        # branching boundary section is the authoritative C1 carrier.
        _read_array(handle, "<i4", n_edges, section="edge sources")
        _read_array(handle, "<i4", n_edges, section="edge targets")
        weights = _read_array(handle, "<f8", n_edges, section="edge weights")
        signs = _require_integral(
            _read_array(handle, "<f8", n_edges, section="edge signs"),
            section="edge signs",
        )
        if not np.all(np.isin(signs, (-1, 0, 1))):
            raise RCBFFormatError("RCBF edge signs must be exact -1, 0, or 1 orientations")
        if not np.all(np.isfinite(weights)):
            raise RCBFFormatError("RCBF edge weights must be finite")
        directed_values = _read_array(handle, "u1", n_edges, section="edge directedness")
        if not np.all(np.isin(directed_values, (0, 1))):
            raise RCBFFormatError("RCBF edge directedness must be 0 or 1")
        if directed_values.size and not np.all(directed_values == directed_values[0]):
            raise RCBFFormatError(
                "mixed per-relation directedness has no exact RexGraph carrier"
            )
        edge_type_codes = _read_array(handle, "u1", n_edges, section="edge type codes")
        edge_names = [
            _read_cstring(handle, section=f"edge name {edge}") for edge in range(n_edges)
        ]
        edge_types = [
            _read_cstring(handle, section=f"edge type {edge}") for edge in range(n_edges)
        ]
        edge_chi = _read_array(handle, "<f8", n_edges * 4, section="edge chi").reshape(
            n_edges, 4
        )

        arities = _read_array(handle, "<u2", n_edges, section="edge boundary arities")
        if np.any(arities == 0):
            raise RCBFFormatError("RCBF C1 relations must have nonempty boundary support")
        boundary_count = int(np.asarray(arities, dtype=np.uint64).sum())
        if boundary_count > np.iinfo(np.int32).max:
            raise RCBFFormatError("RCBF C1 boundary is too large for the RexGraph carrier")
        boundary_idx = _read_array(
            handle, "<i4", boundary_count, section="edge boundary participants"
        )
        if np.any(boundary_idx < 0) or np.any(boundary_idx >= n_vertices):
            raise RCBFFormatError("RCBF C1 boundary refers to a vertex outside C0")
        boundary_ptr = np.empty(n_edges + 1, dtype=np.int32)
        boundary_ptr[0] = 0
        boundary_ptr[1:] = np.cumsum(arities, dtype=np.int64).astype(np.int32)

        face_ptr = np.zeros(n_faces + 1, dtype=np.int32)
        face_rows = np.zeros(0, dtype=np.int32)
        face_values = np.zeros(0, dtype=np.int64)
        if n_faces:
            face_arities = _read_array(handle, "<u2", n_faces, section="face arities")
            if np.any(face_arities == 0):
                raise RCBFFormatError("RCBF C2 cells must have nonempty relation support")
            face_count = int(np.asarray(face_arities, dtype=np.uint64).sum())
            if face_count > np.iinfo(np.int32).max:
                raise RCBFFormatError("RCBF C2 boundary is too large for the RexGraph carrier")
            face_rows = _read_array(handle, "<i4", face_count, section="face relations")
            if np.any(face_rows < 0) or np.any(face_rows >= n_edges):
                raise RCBFFormatError("RCBF C2 boundary refers to a relation outside C1")
            face_values = _require_integral(
                _read_array(handle, "<f8", face_count, section="face coefficients"),
                section="face coefficients",
            )
            face_ptr[1:] = np.cumsum(face_arities, dtype=np.int64).astype(np.int32)
            for face in range(n_faces):
                lo, hi = int(face_ptr[face]), int(face_ptr[face + 1])
                if np.unique(face_rows[lo:hi]).size != hi - lo:
                    raise RCBFFormatError("RCBF C2 boundary repeats a C1 relation")
        face_names = [
            _read_cstring(handle, section=f"face name {face}")
            for face in range(n_faces)
        ]

        signal_values: dict[str, np.ndarray] = {}
        for signal in range(n_signals):
            name = _read_cstring(handle, section=f"signal name {signal}")
            values = _read_array(
                handle, "<f8", n_edges, section=f"signal values {signal}"
            )
            if name in signal_values:
                raise RCBFFormatError("RCBF repeats a signal name; cannot preserve distinct signals")
            if not np.all(np.isfinite(values)):
                raise RCBFFormatError("RCBF signal values must be finite")
            signal_values[name] = values
        _skip(
            handle,
            int(harmonic_dim) * int(n_edges) * np.dtype("<f8").itemsize,
            section="harmonic basis",
        )
        temporal_states = struct.unpack("<I", _read_exact(handle, 4, section="state count"))[0]
        if temporal_states and not allow_current_snapshot:
            raise RCBFFormatError(
                "RCBF temporal states require a temporal converter; pass "
                "allow_current_snapshot=True to import only the current C0--C2 state"
            )
        # State full payloads only exist after temporal records. A current snapshot
        # import deliberately stops before those records, while a timeless RCBF can
        # carry its complete C1 and per boundary metadata into the RexGraph sidecar.
        edge_attributes: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
        slot_attributes: list[list[dict[str, object]]] = [[] for _ in range(n_edges)]
        if temporal_states == 0:
            edge_attributes, slot_attributes = _read_attribute_tail(
                handle,
                flags=int(flags),
                n_edges=int(n_edges),
            )
            if handle.read(1):
                raise RCBFFormatError("RCBF has unclaimed trailing data")

    if not _chain_is_exact(
        boundary_ptr, boundary_idx, face_ptr, face_rows, face_values
    ):
        raise RCBFFormatError("RCBF C2 boundary violates the exact chain condition")

    from rexgraph.graph import RexGraph

    kwargs = {
        "boundary_ptr": boundary_ptr,
        "boundary_idx": boundary_idx,
        "directed": bool(directed_values[0]) if directed_values.size else False,
        "w_E": weights,
        "signs": signs,
    }
    if n_faces:
        kwargs.update(
            B2_col_ptr=face_ptr,
            B2_row_idx=face_rows,
            B2_vals=np.asarray(face_values, dtype=np.float64),
        )
    rex = RexGraph(**kwargs)
    # An empty or isolated C0 basis is meaningful and cannot be recovered solely
    # from C1 supports, so retain the explicitly stored C0 cardinality.
    rex._nV = int(n_vertices)
    rex._agent_meta = {
        "vertex_labels": labels,
        "rcbf": {
            "version": int(version),
            "flags": int(flags),
            "signals": {name: values.tolist() for name, values in signal_values.items()},
            "temporal_states_unimported": int(temporal_states),
            "trailing_metadata_unimported": bool(
                temporal_states and flags & (_FLAG_ATTRS | _FLAG_SLOT_ATTR)
            ),
            "edge_attributes": edge_attributes,
            "slot_attributes": slot_attributes,
        },
    }
    for edge, (name, edge_type, code, chi) in enumerate(
        zip(edge_names, edge_types, edge_type_codes, edge_chi, strict=True)
    ):
        if name:
            rex.attach_metadata(1, edge, "rcbf_name", name)
        if edge_type:
            rex.attach_metadata(1, edge, "rcbf_type", edge_type)
        rex.attach_metadata(1, edge, "rcbf_type_code", int(code))
        if np.all(np.isfinite(chi)):
            rex.attach_metadata(1, edge, "rcbf_chi", chi.tolist())
    for face, name in enumerate(face_names):
        if name:
            rex.attach_metadata(2, face, "rcbf_name", name)
    return rex
