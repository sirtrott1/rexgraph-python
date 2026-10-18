"""Canonical C++ algebraic archives attached to ordinary RexGraph records."""
from collections import OrderedDict
from hashlib import sha256
from numbers import Integral
from threading import RLock

import numpy as np


def algebraic_record(archive):
    """Retain the complete native archive as a checked byte attachment."""
    from rexgraph_cpp import NativeAlgebraicArchive

    from rexgraph.graph import RexGraph
    if type(archive) is not NativeAlgebraicArchive:
        raise TypeError("algebraic records require an owning NativeAlgebraicArchive")
    manifest = archive.manifest
    attachment = {"kind": "cpp_algebraic_archive", "version": 1,
                  "archive_digest": manifest["archive_digest"], "source_digest": manifest["source_digest"],
                  "archive": np.frombuffer(archive.archive, np.uint8).copy()}
    if archive.graph_owner is not None:
        attachment.update(kind="cpp_algebraic_graph_archive", graph_digest=manifest["graph_digest"],
                          archive_key=archive.graph_key,
                          archive=np.frombuffer(archive.graph_owner.image, np.uint8).copy())
    record = RexGraph.from_hypergraph([0, 1], [0])
    record.attach_metadata(1, 0, "algebraic_archive", attachment)
    return record


def algebraic_graph_record(graph):
    """Retain the complete graph and its named owner and comparison tables."""
    from rexgraph_cpp import NativeAlgebraicArchiveGraph

    from rexgraph.graph import RexGraph
    if type(graph) is not NativeAlgebraicArchiveGraph:
        raise TypeError("algebraic graph records require an owning NativeAlgebraicArchiveGraph")
    record = RexGraph.from_hypergraph([0, 1], [0])
    record.attach_metadata(1, 0, "algebraic_graph", {
        "kind": "cpp_algebraic_graph", "version": 1, "graph_digest": graph.manifest["graph_digest"],
        "archive": np.frombuffer(graph.image, np.uint8).copy()})
    return record


class _AlgebraicRestoreCache:
    """Keep at most eight validated owners within one query's archive byte budget."""
    def __init__(self):
        self._entries = OrderedDict()
        self._lock = RLock()
        self._bytes = 0

    def get(self, key, record, tensor):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] is not record or entry[1] is not tensor:
                return None
            self._entries.move_to_end(key)
            return entry[2]

    def put(self, key, record, tensor, owner, size):
        if size > 64 << 20:
            return owner
        with self._lock:
            known = self._entries.get(key)
            if known is not None and known[0] is record and known[1] is tensor:
                self._entries.move_to_end(key)
                return known[2]
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[3]
            self._entries[key] = (record, tensor, owner, size)
            self._bytes += size
            while len(self._entries) > 8 or self._bytes > 64 << 20:
                self._bytes -= self._entries.popitem(last=False)[1][3]
        return owner


def _read_attachment(record, index, name, limits, cache):
    from collections.abc import Mapping

    from rexgraph_cpp import NativeAlgebraicArchive, NativeAlgebraicArchiveGraph
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("algebraic record cell index must be a nonnegative integer")
    attachment = record.get_metadata(1, int(index), name)
    if not isinstance(attachment, Mapping):
        raise ValueError("selected record has no complete algebraic archive attachment")
    graph = attachment.get("kind") in {"cpp_algebraic_graph_archive", "cpp_algebraic_graph"}
    standalone_graph = name == "algebraic_graph"
    names = {"kind", "version", "archive"}
    names |= {"graph_digest"} if standalone_graph else {"archive_digest", "source_digest"}
    if graph and not standalone_graph:
        names |= {"graph_digest", "archive_key"}
    if set(attachment) != names:
        raise ValueError("selected record has no complete algebraic archive attachment")
    expected = "cpp_algebraic_graph" if standalone_graph else "cpp_algebraic_graph_archive" if graph else "cpp_algebraic_archive"
    if (attachment["kind"] != expected or type(attachment["version"]) is not int
            or attachment["version"] != 1):
        raise ValueError("unsupported algebraic archive attachment schema")
    archive = attachment["archive"]
    if not isinstance(archive, np.ndarray) or archive.dtype != np.uint8 or archive.ndim != 1:
        raise ValueError("algebraic archive attachment requires a one dimensional uint8 tensor")
    from rexgraph.source_binding import _sha, _uint
    for key in ("graph_digest",) if standalone_graph else ("archive_digest", "source_digest") + (("graph_digest",) if graph else ()):
        _sha(attachment[key])
    if graph and not standalone_graph and _uint(attachment["archive_key"]) == 0:
        raise ValueError("algebraic graph archive key must be nonzero")
    owner_type = NativeAlgebraicArchiveGraph if graph else NativeAlgebraicArchive
    policy = owner_type.default_limits()
    if limits is not None:
        if not isinstance(limits, Mapping) or set(limits) - set(policy):
            raise ValueError("unknown algebraic archive admission fields")
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("algebraic archive admission requires integers")
            if not 0 <= value <= np.iinfo(np.uintp).max:
                raise OverflowError("algebraic archive admission is outside its declared width")
            policy[name] = int(value)
    if archive.size > policy["max_file_bytes"]:
        raise ValueError("algebraic archive attachment exceeds its byte admission")
    raw = archive.tobytes()
    digest = attachment["graph_digest" if graph else "archive_digest"]
    if sha256(raw).hexdigest() != digest:
        raise ValueError("algebraic archive attachment digest differs")
    if cache is not None and not isinstance(cache, _AlgebraicRestoreCache):
        raise TypeError("algebraic archive cache requires its bounded restore implementation")
    key = ("rfcxg" if graph else "rfcxa", id(record), id(archive), digest, tuple(policy.items()))
    if not graph:
        key += (int(index), attachment["source_digest"])
    result = None
    if cache is not None:
        known = cache.get(key, record, archive)
        if known is not None:
            result = known
    if result is None:
        result = owner_type(raw, limits=policy)
        if cache is not None:
            result = cache.put(key, record, archive, result, len(raw))
    if standalone_graph:
        return result
    if graph:
        result = result.archive(attachment["archive_key"])
    manifest = result.manifest
    if (manifest["source_digest"] != attachment["source_digest"]
            or manifest["archive_digest"] != attachment["archive_digest"]):
        raise ValueError("algebraic archive attachment source differs")
    return result


def read_algebraic_record(record, index=0, *, limits=None, _archive_cache=None):
    """Restore a standalone archive or a selected view with its complete graph owner."""
    return _read_attachment(record, index, "algebraic_archive", limits, _archive_cache)


def read_algebraic_graph_record(record, index=0, *, limits=None, _archive_cache=None):
    """Check the complete RFCXG attachment and restore a fresh shared owner graph."""
    return _read_attachment(record, index, "algebraic_graph", limits, _archive_cache)
