"""Named algebraic archive channels on complete native publication prefixes."""
from collections.abc import Mapping
from hashlib import sha256

import numpy as np

from rexgraph.cpp_algebraic import _key
from rexgraph.source_binding import (
    CppAlgebraicSourceReference,
    CppSourceReference,
    source_from_record,
)

__all__ = ["algebraic_history_reference", "validate_algebraic_history_reference",
           "algebraic_history_record", "algebraic_history_at"]


def _closed(value, names):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("incomplete algebraic history declaration")


def validate_algebraic_history_reference(value, *, publication=None):
    """Check the selected namespace, prefix identity, channel and byte admission."""
    _closed(value, ("kind", "version", "profile", "source", "channel", "max_bytes"))
    if (value["kind"] != "cpp_algebraic_history_reference" or type(value["version"]) is not int
            or value["version"] != 1 or value["profile"] != "native-algebraic-channel-v2"):
        raise ValueError("unsupported algebraic history reference")
    source = source_from_record(value["source"], allow_cpp=True)
    if not isinstance(source, CppSourceReference) or not source.native.namespace:
        raise ValueError("algebraic history requires a native publication prefix")
    names = value["channel"]
    if isinstance(names, (str, bytes, Mapping)) or len(names) != 3 or not all(_key(v) for v in names):
        raise ValueError("algebraic history requires three nonzero channel names")
    _key(value["max_bytes"])
    if publication is not None:
        archive = source_from_record(publication["source"], allow_cpp=True)
        if (not isinstance(archive, CppAlgebraicSourceReference)
                or _key(publication["namespace"]) != source.native.namespace or _key(publication["sequence"]) != source.native.sequence
                or archive.native.graph_digest is not None
                or tuple(map(len, source.native.coordinates)) != archive.native.source_sizes):
            raise ValueError("algebraic history differs from its archive publication")
    return source


def algebraic_history_reference(archive):
    """Describe the actual retained native prefix of a history backed archive."""
    owner = archive.history_owner
    if owner is None:
        return None
    value = {"kind": "cpp_algebraic_history_reference", "version": 1,
             "profile": "native-algebraic-channel-v2", "source": CppSourceReference(owner).as_record(),
             "channel": archive.history_channel, "max_bytes": owner.max_bytes}
    validate_algebraic_history_reference(value)
    return value


def _attach_history(record, archive):
    reference = algebraic_history_reference(archive)
    if reference is not None:
        record.attach_metadata(1, 0, "algebraic_history", {"kind": "cpp_algebraic_history", "version": 1,
            "reference": reference, "archive": np.frombuffer(archive.history_owner.archive, dtype=np.uint8).copy()})


def _check_history_attachment(record, index, publication, max_bytes):
    reference = publication["history"]
    source = validate_algebraic_history_reference(reference, publication=publication)
    attachment = record.get_metadata(1, index, "algebraic_history")
    _closed(attachment, ("kind", "version", "reference", "archive"))
    if (attachment["kind"] != "cpp_algebraic_history" or type(attachment["version"]) is not int
            or attachment["version"] != 1 or attachment["reference"] != reference):
        raise ValueError("algebraic history attachment differs from its publication")
    raw = attachment["archive"]
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("algebraic history requires a one dimensional uint8 archive")
    bound = min(_key(max_bytes), reference["max_bytes"])
    if raw.size < 32 or raw.size > bound:
        raise ValueError("algebraic history exceeds its independent byte admission")
    if sha256(raw.tobytes()).hexdigest() != source.native.archive_digest:
        raise ValueError("algebraic history prefix bytes differ")
    return raw, source, bound


def _restore_history_archive(record, index, publication, limits, checked, cache):
    from rexgraph_cpp import NativeAlgebraicArchive, NativeSource

    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
    raw, source, bound = checked
    reference = publication["history"]
    archive_tensor = record.get_metadata(1, index, "algebraic_archive")["archive"]
    key = ("native-algebraic-channel-v2", int(index), source.coefficient_digest, tuple(reference["channel"]),
           bound, tuple(sorted(limits.items())), id(archive_tensor), publication["source"]["state_digest"])
    if cache is not None and not isinstance(cache, _AlgebraicRestoreCache):
        raise TypeError("algebraic history cache requires its bounded restore implementation")
    owner = None if cache is None else cache.get(key, record, raw)
    if owner is None:
        native = NativeSource(raw.tobytes(), max_bytes=bound)
        source.bind(native)
        owner = NativeAlgebraicArchive.from_history(native, reference["channel"],
                                                   sequence=source.native.sequence, limits=limits)
        if sha256(owner.archive).hexdigest() != publication["source"]["state_digest"]:
            raise ValueError("selected history archive differs from its retained algebraic image")
        if cache is not None:
            owner = cache.put(key, record, raw, owner, int(raw.size + archive_tensor.size))
    return owner


def algebraic_history_record(source, channel, *, sequence, limits=None):
    """Save a selected channel with its complete native prefix and algebraic image."""
    from rexgraph_cpp import NativeAlgebraicSnapshot

    from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
    return algebraic_snapshot_record(NativeAlgebraicSnapshot.from_history(source, channel, sequence=sequence, limits=limits))


def algebraic_history_at(record, sequence, index=0, *, limits=None, history_max_bytes=256 << 20):
    """Select a retained endpoint and save a prefix ending at that publication."""
    from rexgraph_cpp import NativeAlgebraicSnapshot

    from rexgraph.cpp_algebraic_query import (
        algebraic_snapshot_record,
        read_algebraic_snapshot_record,
    )
    snapshot = read_algebraic_snapshot_record(record, index, limits=limits, history_max_bytes=history_max_bytes)
    if snapshot.graph_revision_owner is not None:
        from rexgraph.cpp_algebraic_graph_history import algebraic_graph_history_record
        revision = snapshot.graph_revision_owner
        return algebraic_graph_history_record(revision.history_owner, revision.history_channel, sequence=sequence,
            archive_key=snapshot.archive_owner.graph_key, limits=revision.limits)
    archive = snapshot.archive_owner
    if archive.history_owner is None:
        raise ValueError("selected record does not retain native algebraic history")
    selected = NativeAlgebraicSnapshot.from_history(archive.history_owner, archive.history_channel,
                                                   sequence=sequence, limits=archive.limits)
    return algebraic_snapshot_record(selected)
