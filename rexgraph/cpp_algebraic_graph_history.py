"""Anchored graph revisions retained with their complete native prefixes."""
from collections.abc import Mapping
from hashlib import sha256

import numpy as np

from rexgraph.cpp_algebraic import _key, _portable_limits
from rexgraph.source_binding import (
    CppAlgebraicSourceReference,
    CppSourceReference,
    _sha,
    source_from_record,
)
from rexgraph.value_codec import pack_value

__all__ = ["algebraic_graph_history_record", "algebraic_graph_history_at", "algebraic_graph_history_reference"]
_EXTRA_LIMITS = ("max_snapshot_bytes", "max_source_bindings", "max_archive_bindings",
                 "max_source_coordinates", "max_history_revisions")


def _closed(value, names):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("incomplete algebraic graph history declaration")


def _limits(value):
    if not isinstance(value, Mapping) or set(_EXTRA_LIMITS) - set(value):
        raise ValueError("incomplete graph revision limits")
    _portable_limits({k: v for k, v in value.items() if k not in _EXTRA_LIMITS}, graph=True)
    for name in _EXTRA_LIMITS:
        _key(value[name])
    return dict(value)


def _validate_reference(value, *, publication=None):
    _closed(value, ("kind", "version", "profile", "source", "channel", "max_bytes", "limits", "revision"))
    if (value["kind"] != "cpp_algebraic_graph_history_reference" or type(value["version"]) is not int
            or value["version"] != 1 or value["profile"] != "native-algebraic-graph-channel-v1"):
        raise ValueError("unsupported graph history reference")
    source = source_from_record(value["source"], allow_cpp=True)
    if not isinstance(source, CppSourceReference) or not source.native.namespace:
        raise ValueError("graph history requires a native publication prefix")
    channel = value["channel"]
    if isinstance(channel, (str, bytes, Mapping)) or len(channel) != 3 or not all(_key(v) for v in channel):
        raise ValueError("graph history requires three nonzero channel names")
    _key(value["max_bytes"])
    limits = _limits(value["limits"])
    r = value["revision"]
    _closed(r, ("codec", "namespace", "sequence", "anchor_bundle", "snapshot_digest", "graph_digest", "archives", "sources"))
    if (r["codec"] != "rfcxh-v1" or _key(r["namespace"]) != source.native.namespace
            or _key(r["sequence"]) != source.native.sequence or not _key(r["anchor_bundle"])):
        raise ValueError("graph revision differs from its native prefix")
    if source.native.sequence - source.native.first_sequence + 1 > limits["max_history_revisions"]:
        raise ValueError("graph prefix exceeds its revision admission")
    _sha(r["snapshot_digest"]); _sha(r["graph_digest"])
    sources, archives = r["sources"], r["archives"]
    if (isinstance(sources, (str, bytes, Mapping)) or isinstance(archives, (str, bytes, Mapping))
            or len(sources) > limits["max_source_bindings"] or len(archives) > limits["max_archive_bindings"]):
        raise ValueError("graph binding tables exceed their admission")
    bundle_keys, coordinates, anchor = [], 0, None
    for binding in sources:
        _closed(binding, ("bundle", "kind", "cells", "occurrences"))
        key = _key(binding["bundle"])
        if not key or binding["kind"] not in {"independent", "anchor_morphism"}:
            raise ValueError("invalid graph source binding")
        bundle_keys.append(key)
        cells, occurrences = binding["cells"], binding["occurrences"]
        for rows in (cells, occurrences):
            if isinstance(rows, (str, bytes, Mapping)):
                raise ValueError("graph source map requires original grade rows")
            for row in rows:
                if isinstance(row, (str, bytes, Mapping)):
                    raise ValueError("graph source map requires coordinate rows")
                coordinates += len(row)
                if coordinates > limits["max_source_coordinates"]:
                    raise ValueError("graph source maps exceed their coordinate admission")
                for v in row:
                    _key(v)
        if binding["kind"] == "independent":
            if cells or occurrences or key == r["anchor_bundle"]:
                raise ValueError("independent graph source cannot declare an anchor map")
        elif len(cells) != len(occurrences):
            raise ValueError("graph source map must retain every grade")
        if key == r["anchor_bundle"]:
            anchor = binding
    if bundle_keys != sorted(set(bundle_keys)) or anchor is None:
        raise ValueError("graph source bindings require distinct sorted keys and an anchor")
    if (tuple(map(len, anchor["cells"])) != tuple(map(len, source.native.coordinates))
            or any(tuple(row) != tuple(range(len(row))) for name in ("cells", "occurrences") for row in anchor[name])):
        raise ValueError("graph anchor requires its complete original identity map")
    archive_keys = []
    for row in archives:
        if isinstance(row, (str, bytes, Mapping)) or len(row) != 2 or not _key(row[0]) or _key(row[1]) not in bundle_keys:
            raise ValueError("invalid graph archive binding")
        archive_keys.append(row[0])
    if archive_keys != sorted(set(archive_keys)):
        raise ValueError("graph archive bindings require distinct sorted keys")
    if publication is not None:
        archive = source_from_record(publication["source"], allow_cpp=True)
        if (not isinstance(archive, CppAlgebraicSourceReference) or archive.native.graph_digest != r["graph_digest"]
                or archive.native.archive_key not in archive_keys
                or publication["namespace"] != r["namespace"] or publication["sequence"] != r["sequence"]):
            raise ValueError("graph history differs from its selected archive publication")
    return source


def algebraic_graph_history_reference(snapshot):
    """Describe the actual graph revision retained by a database archive pin."""
    owner = snapshot.graph_revision_owner
    if owner is None:
        return None
    value = {"kind": "cpp_algebraic_graph_history_reference", "version": 1,
             "profile": "native-algebraic-graph-channel-v1", "source": CppSourceReference(owner.history_owner).as_record(),
             "channel": owner.history_channel, "max_bytes": owner.history_owner.max_bytes,
             "limits": owner.limits, "revision": owner.manifest}
    _validate_reference(value)
    return value


def _attach_graph_history(record, snapshot):
    reference = algebraic_graph_history_reference(snapshot)
    if reference is not None:
        owner = snapshot.graph_revision_owner
        record.attach_metadata(1, 0, "algebraic_graph_history", {
            "kind": "cpp_algebraic_graph_history", "version": 1, "reference": reference,
            "archive": np.frombuffer(owner.history_owner.archive, np.uint8).copy(),
            "snapshot": np.frombuffer(owner.image, np.uint8).copy()})


def _check_graph_history_attachment(record, index, publication, max_bytes):
    reference = publication["graph_history"]
    source = _validate_reference(reference, publication=publication)
    value = record.get_metadata(1, index, "algebraic_graph_history")
    _closed(value, ("kind", "version", "reference", "archive", "snapshot"))
    if (value["kind"] != "cpp_algebraic_graph_history" or type(value["version"]) is not int
            or value["version"] != 1 or value["reference"] != reference):
        raise ValueError("graph history attachment differs from its publication")
    bound = min(_key(max_bytes), reference["max_bytes"])
    for name, limit, digest in (("archive", bound, source.native.archive_digest),
            ("snapshot", reference["limits"]["max_snapshot_bytes"], reference["revision"]["snapshot_digest"])):
        raw = value[name]
        if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("graph history requires one dimensional uint8 attachments")
        if raw.size < 32 or raw.size > limit or sha256(raw.tobytes()).hexdigest() != digest:
            raise ValueError("graph history bytes differ or exceed their independent admission")
    return value["archive"], value["snapshot"], source, bound


def _restore_graph_history_snapshot(record, index, publication, limits, checked, cache):
    from rexgraph_cpp import (
        NativeAlgebraicGraphDatabase,
        NativeAlgebraicGraphRevision,
        NativeSource,
    )

    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
    raw, image, source, bound = checked
    reference = publication["graph_history"]
    policy = NativeAlgebraicGraphRevision.default_limits()
    if limits is not None:
        if not isinstance(limits, Mapping) or set(limits) - set(policy):
            raise ValueError("unknown graph revision reader admission")
        policy.update({k: _key(v) for k, v in limits.items()})
    policy = {k: min(v, reference["limits"][k]) for k, v in policy.items()}
    if image.size > policy["max_snapshot_bytes"]:
        raise ValueError("graph snapshot exceeds its reader byte admission")
    tensor = record.get_metadata(1, index, "algebraic_archive")["archive"]
    key = ("native-algebraic-graph-channel-v1", index, sha256(pack_value(publication)).hexdigest(),
           bound, tuple(sorted(policy.items())), id(image), id(tensor))
    if cache is not None and not isinstance(cache, _AlgebraicRestoreCache):
        raise TypeError("graph history cache requires its bounded restore implementation")
    result = None if cache is None else cache.get(key, record, raw)
    if result is None:
        native = NativeSource(raw.tobytes(), max_bytes=bound)
        source.bind(native)
        revision = NativeAlgebraicGraphRevision(native, reference["channel"], sequence=source.native.sequence, limits=policy)
        if revision.manifest != reference["revision"] or revision.image != image.tobytes():
            raise ValueError("restored graph revision differs from its complete declaration")
        database = NativeAlgebraicGraphDatabase(revision, availability=(publication["namespace"], publication["sequence"]))
        archive_key = source_from_record(publication["source"], allow_cpp=True).native.archive_key
        result = database.archive(archive_key, availability=(publication["namespace"], publication["sequence"]))
        if not source_from_record(publication["source"], allow_cpp=True).matches(CppAlgebraicSourceReference(result.archive_owner)):
            raise ValueError("selected graph archive differs from its complete attachment")
        if cache is not None:
            result = cache.put(key, record, raw, result, int(raw.size + image.size + tensor.size))
    return result


def algebraic_graph_history_record(source, channel, *, sequence, archive_key, limits=None):
    """Save one native graph database archive with its anchored declaration and prefix."""
    from rexgraph_cpp import NativeAlgebraicGraphDatabase, NativeAlgebraicGraphRevision

    from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
    revision = NativeAlgebraicGraphRevision(source, channel, sequence=sequence, limits=limits)
    availability = (revision.manifest["namespace"], sequence)
    snapshot = NativeAlgebraicGraphDatabase(revision, availability=availability).archive(archive_key, availability=availability)
    return algebraic_snapshot_record(snapshot)


def algebraic_graph_history_at(record, sequence, index=0, *, archive_key=None, limits=None, history_max_bytes=256 << 20):
    """Select a prior retained endpoint with its declared archive key and source maps."""
    from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record
    snapshot = read_algebraic_snapshot_record(record, index, limits=limits, history_max_bytes=history_max_bytes)
    revision = snapshot.graph_revision_owner
    if revision is None:
        raise ValueError("selected record does not retain native graph history")
    return algebraic_graph_history_record(revision.history_owner, revision.history_channel, sequence=sequence,
        archive_key=snapshot.archive_owner.graph_key if archive_key is None else archive_key, limits=revision.limits)
