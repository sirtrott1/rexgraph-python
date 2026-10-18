"""Owning algebraic queries and complete polynomial coordinate observations."""
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from math import prod
from numbers import Integral

from rexgraph.cpp_algebraic import _carrier, _key, _portable_limits
from rexgraph.source_binding import CppAlgebraicSourceReference, _sha, source_from_record
from rexgraph.value_codec import pack_value

__all__ = ["prepare_algebraic_query", "algebraic_query_observation", "algebraic_window_observation",
           "validate_algebraic_observation", "algebraic_snapshot_record", "read_algebraic_snapshot_record",
           "algebraic_query_record", "algebraic_window_record", "read_algebraic_observation_record"]
_LIMITS = {"records": 4000000, "rational_slots": 16000000, "sparse_entries": 16000000,
           "scalar_work": 64000000, "integer_bits": 1 << 24}
_KINDS = ("vector", "covector", "map", "form", "metric")
_OBSERVATIONS = {"cpp_algebraic_query_observation", "cpp_algebraic_window_observation"}


def _closed(value, names):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("incomplete algebraic observation declaration")


def _limits(value, *, complete=False):
    result = dict(_LIMITS)
    if complete and value is None:
        raise ValueError("complete algebraic execution admission is required")
    if value is not None:
        if not isinstance(value, Mapping) or set(value) - set(result) or complete and set(value) != set(result):
            raise ValueError("unknown or incomplete algebraic execution admission")
        result.update({key: _key(v) for key, v in value.items()})
    return result


def _publication(snapshot):
    from rexgraph.cpp_algebraic_history import algebraic_history_reference
    value = {"namespace": snapshot.namespace, "sequence": snapshot.sequence,
             "source": CppAlgebraicSourceReference(snapshot.archive_owner).as_record()}
    history = algebraic_history_reference(snapshot.archive_owner)
    if history is not None:
        value["history"] = history
    from rexgraph.cpp_algebraic_graph_history import algebraic_graph_history_reference
    graph_history = algebraic_graph_history_reference(snapshot)
    if graph_history is not None:
        value["graph_history"] = graph_history
    return value


def _check_publication(value):
    names = {"namespace", "sequence", "source"}
    if isinstance(value, Mapping) and "history" in value:
        names.add("history")
    if isinstance(value, Mapping) and "graph_history" in value:
        if "history" in value:
            raise ValueError("algebraic publication requires one native history declaration")
        names.add("graph_history")
    _closed(value, names)
    if not _key(value["namespace"]):
        raise ValueError("algebraic publication namespace must be nonzero")
    _key(value["sequence"])
    source = source_from_record(value["source"], allow_cpp=True)
    if not isinstance(source, CppAlgebraicSourceReference):
        raise ValueError("algebraic publication requires its complete archive source")
    if "history" in value:
        from rexgraph.cpp_algebraic_history import validate_algebraic_history_reference
        validate_algebraic_history_reference(value["history"], publication=value)
    if "graph_history" in value:
        from rexgraph.cpp_algebraic_graph_history import _validate_reference
        _validate_reference(value["graph_history"], publication=value)
    return source


def _cutoff(publication, availability):
    if isinstance(availability, (str, bytes, Mapping)) or len(availability) != 2:
        raise ValueError("algebraic availability requires a namespace and cutoff")
    ns, cutoff = map(_key, availability)
    if ns != publication["namespace"] or cutoff < publication["sequence"]:
        raise ValueError("algebraic observation exceeds its declared availability")


def _blocks(rows, count, degree, bits):
    if isinstance(rows, (str, bytes, Mapping)) or len(rows) != count:
        raise ValueError("algebraic observation coefficient count differs")
    for row in rows:
        if isinstance(row, (str, bytes, Mapping)) or len(row) != degree:
            raise ValueError("algebraic observation requires complete polynomial coordinates")
        for q in row:
            if type(q) is not Fraction:
                raise TypeError("algebraic observations require exact Fraction coordinates")
            if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > bits:
                raise ValueError("algebraic observation exceeds its integer bit admission")


def _admit_coordinates(count, degree, maximum):
    maximum = _key(maximum)
    if count * degree > maximum:
        raise ValueError("complete algebraic observation exceeds its coordinate admission")


def prepare_algebraic_query(snapshot, request=None, *, availability, execution_limits=None, cancellation=None):
    """Bind named record carriers and complete scalar literals to one actual snapshot."""
    from rexgraph_cpp import CancellationToken, NativeAlgebraicSnapshot, PreparedAlgebraicQuery
    if type(snapshot) is not NativeAlgebraicSnapshot:
        raise TypeError("algebraic query requires an owning NativeAlgebraicSnapshot")
    request = {} if request is None else request
    names = {"kind", "source_key", "target_key", "all_of", "rows_per_tile"}
    if not isinstance(request, Mapping) or set(request) - names:
        raise ValueError("unknown algebraic query request fields")
    limits = _limits(execution_limits)
    _cutoff({"namespace": snapshot.namespace, "sequence": snapshot.sequence}, availability)
    if cancellation is not None and not isinstance(cancellation, CancellationToken):
        raise TypeError("algebraic execution requires a native CancellationToken")
    if cancellation is not None and cancellation.cancelled:
        from rexgraph_cpp import ExecutionCancelled
        raise ExecutionCancelled("exact execution cancelled at a phase boundary")
    clauses = request.get("all_of", ())
    if isinstance(clauses, (str, bytes, Mapping)):
        raise TypeError("algebraic query predicates require a sized sequence")
    archive = snapshot.archive_owner
    rows, degree = archive.record_count, archive.field_degree
    if (6 * rows + rows * len(clauses) + len(clauses) > limits["records"]
            or len(clauses) * degree > limits["rational_slots"]
            or rows * len(clauses) * (degree + 2) > limits["scalar_work"]):
        raise ValueError("algebraic query declaration exceeds execution admission")
    predicates = []
    records = 6 * rows + rows * len(clauses) + len(clauses)
    for item in clauses:
        if len(predicates) >= len(clauses):
            raise ValueError("algebraic predicate count changed")
        _closed(item, ("coordinate", "operation", "literal"))
        coordinate = item["coordinate"]
        if isinstance(coordinate, (str, bytes, Mapping)):
            raise TypeError("algebraic predicate coordinates require a sized sequence")
        records += len(coordinate)
        if records > limits["records"]:
            raise ValueError("algebraic query coordinates exceed execution admission")
        coordinate = tuple(_key(v) for v in coordinate)
        literal = item["literal"]
        if isinstance(literal, (str, bytes, Mapping)) or len(literal) != degree:
            raise ValueError("algebraic predicate requires complete polynomial coordinates")
        for q in literal:
            if isinstance(q, bool) or not isinstance(q, (Integral, Fraction)):
                raise TypeError("algebraic literals require integers or Fraction values")
            q = Fraction(int(q)) if isinstance(q, Integral) else q
            if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > limits["integer_bits"]:
                raise ValueError("algebraic predicate exceeds its execution integer bit admission")
        predicates.append({"coordinate": coordinate, "operation": item["operation"],
                           "literal": archive.scalar(literal)})
    if len(predicates) != len(clauses):
        raise ValueError("algebraic predicate count changed")
    return PreparedAlgebraicQuery(snapshot, availability=availability, kind=request.get("kind"),
        source_carrier=None if request.get("source_key") is None else archive.record(_key(request["source_key"])),
        target_carrier=None if request.get("target_key") is None else archive.record(_key(request["target_key"])),
        all_of=predicates, rows_per_tile=request.get("rows_per_tile", 256),
        execution_limits=limits, cancellation=cancellation)


def _observation(kind, snapshot, availability, details):
    from rexgraph_cpp import build_info
    value = {"kind": kind, "version": 1, "arithmetic": "algebraic", "coefficient_basis": "power",
             "publication": _publication(snapshot), "availability": tuple(availability),
             "engine": build_info(), **details}
    value["seal"] = sha256(pack_value(value)).hexdigest()
    return value


def algebraic_query_observation(result, *, max_coordinates=65536):
    """Capture complete ordered query records without identifying them as a model field."""
    from rexgraph_cpp import NativeAlgebraicQueryResult
    if type(result) is not NativeAlgebraicQueryResult:
        raise TypeError("query observation requires an owning native result")
    query = result.query_owner
    snapshot = query.snapshot_owner
    degree = snapshot.archive_owner.field_degree
    pins = tuple(result.record(i) for i in range(len(result.keys)))
    count = sum(pin.preview(0)["coefficient_count"] for pin in pins)
    _admit_coordinates(count, degree, max_coordinates)
    declaration = query.preview((1 << 64) - 1)
    declaration["execution_limits"] = result.execution_limits
    declaration["all_of"] = tuple({**p, "literal": p["literal"]["coefficients"][0]}
                                   for p in declaration["all_of"])
    value = _observation("cpp_algebraic_query_observation", snapshot, result.availability,
                         {"query": declaration, "records": tuple(pin.preview((1 << 64) - 1) for pin in pins)})
    validate_algebraic_observation(value, max_coordinates=max_coordinates)
    return value


def algebraic_window_observation(window, *, availability, execution_limits=None, cancellation=None,
                                 max_coordinates=65536):
    """Capture the complete coefficient box and its optional exact scalar sum."""
    from rexgraph_cpp import NativeAlgebraicTensorWindow
    if type(window) is not NativeAlgebraicTensorWindow:
        raise TypeError("window observation requires an owning native tensor window")
    snapshot = window.record_owner.snapshot_owner
    degree = snapshot.archive_owner.field_degree
    _admit_coordinates(window.supported_count, degree, max_coordinates)
    total = window.sum(availability=availability, execution_limits=execution_limits, cancellation=cancellation)
    details = window.preview((1 << 64) - 1)
    if execution_limits is not None:
        details["execution_limits"] = dict(window.execution_limits, **execution_limits)
    value = _observation("cpp_algebraic_window_observation", snapshot, availability,
        {"window": details, "record": window.record_owner.preview(0),
         "sum": None if total is None else total.coefficients[0]})
    validate_algebraic_observation(value, max_coordinates=max_coordinates)
    return value


def validate_algebraic_observation(value, *, max_coordinates=65536, max_records=1000000,
                                   max_integer_bits=1 << 24):
    """Check complete portable coordinates and receipts without loading the optional engine."""
    if not isinstance(value, Mapping) or value.get("kind") not in _OBSERVATIONS:
        raise ValueError("unsupported algebraic observation schema")
    query = value["kind"] == "cpp_algebraic_query_observation"
    _closed(value, {"kind", "version", "arithmetic", "coefficient_basis", "publication", "availability", "engine", "seal"}
                  | ({"query", "records"} if query else {"window", "record", "sum"}))
    if type(value["version"]) is not int or value["version"] != 1 or value["arithmetic"] != "algebraic" or value["coefficient_basis"] != "power":
        raise ValueError("unsupported algebraic observation profile")
    source = _check_publication(value["publication"])
    _cutoff(value["publication"], value["availability"])
    engine = value["engine"]
    _closed(engine, ("engine_version", "engine_digest", "binding_digest", "binding_version", "scalar_policy"))
    _sha(engine["engine_digest"]); _sha(engine["binding_digest"])
    if (not isinstance(engine["engine_version"], str) or not engine["engine_version"]
            or engine["binding_version"] != "1" or engine["scalar_policy"] != "gmp-rational-exact"):
        raise ValueError("algebraic observation requires an exact native implementation receipt")
    max_records, bits = _key(max_records), _key(max_integer_bits)
    records = value["records"] if query else (value["record"],)
    if isinstance(records, (str, bytes, Mapping)) or len(records) > max_records:
        raise ValueError("algebraic observation exceeds its record admission")
    keys, total = [], 0
    for record in records:
        _closed(record, ("kind", "shape", "source_carrier", "target_carrier", "coefficient_count", "coefficients",
                         "truncated", "key", "domain", "namespace", "sequence"))
        if record["kind"] not in _KINDS or type(record["truncated"]) is not bool:
            raise ValueError("invalid algebraic record observation kind")
        keys.append(_key(record["key"]))
        if record["namespace"] != value["publication"]["namespace"] or record["sequence"] != value["publication"]["sequence"]:
            raise ValueError("algebraic record publication differs")
        _key(record["namespace"]); _key(record["sequence"])
        s, t = _carrier(source, record["source_carrier"]), _carrier(source, record["target_carrier"])
        domain = (s["rank"],) if record["kind"] in {"vector", "covector"} else (t["rank"], s["rank"])
        if (tuple(record["domain"]) != domain or tuple(record["shape"]) != ((s["rank"], 1) if len(domain) == 1 else domain)
                or _key(record["coefficient_count"]) != prod(domain)):
            raise ValueError("algebraic record domain differs from its declared carriers")
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (*record["domain"], *record["shape"])):
            raise ValueError("algebraic record dimensions require integers")
        total += prod(domain) if query else 0
        _admit_coordinates(total, source.native.field_degree, max_coordinates)
        if query:
            if record["truncated"]:
                raise ValueError("complete query observation cannot contain truncated records")
            _blocks(record["coefficients"], prod(domain), source.native.field_degree, bits)
        elif record["coefficients"] or record["truncated"] != (prod(domain) != 0):
            raise ValueError("window receipt must retain its record declaration without full values")
    if len(set(keys)) != len(keys):
        raise ValueError("algebraic query records must retain distinct named keys")
    detail = value["query" if query else "window"]
    names = (("profile", "namespace", "sequence", "candidate_count", "kind", "source_carrier", "target_carrier",
              "all_of", "predicate_count", "truncated", "rows_per_tile", "requirements", "execution_limits") if query else
             ("profile", "domain", "source_domain", "lower", "upper", "supported_count", "record_key",
              "namespace", "sequence", "requirements", "execution_limits", "coefficients", "source_indices", "truncated"))
    _closed(detail, names)
    policy = _limits(detail["execution_limits"], complete=True)
    required = _limits(detail["requirements"], complete=True)
    if any(required[k] > policy[k] for k in policy):
        raise ValueError("algebraic observation exceeds its execution admission")
    if query:
        _closed(detail, ("profile", "namespace", "sequence", "candidate_count", "kind", "source_carrier", "target_carrier",
                         "all_of", "predicate_count", "truncated", "rows_per_tile", "requirements", "execution_limits"))
        if detail["profile"] != "prepared-algebraic-query-v1" or detail["kind"] is not None and detail["kind"] not in _KINDS:
            raise ValueError("unsupported prepared algebraic query observation")
        if (detail["truncated"] is not False or not _key(detail["rows_per_tile"])
                or len(records) > _key(detail["candidate_count"]) or detail["candidate_count"] > max_records
                or _key(detail["predicate_count"]) != len(detail["all_of"]) or len(detail["all_of"]) > max_records):
            raise ValueError("incomplete or oversized algebraic query observation")
        for name in ("source_carrier", "target_carrier"):
            if detail[name] is not None:
                _carrier(source, detail[name])
        for predicate in detail["all_of"]:
            _closed(predicate, ("coordinate", "operation", "literal"))
            if predicate["operation"] not in {"eq", "ne"}:
                raise ValueError("unsupported exact algebraic equality predicate")
            if len(predicate["coordinate"]) > max_records:
                raise ValueError("algebraic predicate coordinate count exceeds record admission")
            for index in predicate["coordinate"]:
                _key(index)
            _blocks((predicate["literal"],), 1, source.native.field_degree, bits)
        for record in records:
            if detail["kind"] is not None and record["kind"] != detail["kind"]:
                raise ValueError("query observation record kind differs from its predicate")
            for name in ("source_carrier", "target_carrier"):
                if detail[name] is not None and record[name] != detail[name]:
                    raise ValueError("query observation carrier differs from its predicate")
            for predicate in detail["all_of"]:
                coordinate = tuple(_key(i) for i in predicate["coordinate"])
                if len(coordinate) != len(record["domain"]) or any(i >= n for i, n in zip(coordinate, record["domain"], strict=True)):
                    raise ValueError("query observation predicate coordinate differs")
                flat = sum(i * prod(record["domain"][axis + 1:]) for axis, i in enumerate(coordinate))
                equal = tuple(record["coefficients"][flat]) == tuple(predicate["literal"])
                if equal != (predicate["operation"] == "eq"):
                    raise ValueError("query observation violates its complete equality predicate")
    else:
        _closed(detail, ("profile", "domain", "source_domain", "lower", "upper", "supported_count", "record_key",
                         "namespace", "sequence", "requirements", "execution_limits", "coefficients", "source_indices", "truncated"))
        record = records[0]
        if detail["profile"] != "algebraic-coefficient-window-v1" or detail["record_key"] != record["key"] or tuple(detail["source_domain"]) != tuple(record["domain"]):
            raise ValueError("algebraic window source record differs")
        rank = len(record["domain"])
        if any(len(detail[name]) != rank for name in ("domain", "lower", "upper")):
            raise ValueError("algebraic window rank differs")
        for lo, hi, size, extent in zip(detail["lower"], detail["upper"], record["domain"], detail["domain"], strict=True):
            if not 0 <= _key(lo) <= _key(hi) <= size or _key(extent) != hi - lo:
                raise ValueError("algebraic window bounds differ")
        count = prod(detail["domain"])
        if _key(detail["supported_count"]) != count or detail["truncated"] is not False:
            raise ValueError("complete algebraic window cannot be truncated")
        _admit_coordinates(count, source.native.field_degree, max_coordinates)
        _blocks(detail["coefficients"], count, source.native.field_degree, bits)
        expected = []
        for i in range(count):
            index, remaining = 0, i
            for axis in range(rank - 1, -1, -1):
                position = remaining % detail["domain"][axis] + detail["lower"][axis]
                remaining //= detail["domain"][axis]
                index += position * prod(record["domain"][axis + 1:])
            expected.append(index)
        if tuple(detail["source_indices"]) != tuple(expected) or any(type(v) is not int for v in detail["source_indices"]):
            raise ValueError("algebraic window source coordinates differ")
        if not count:
            if value["sum"] is not None:
                raise ValueError("empty algebraic window has no sum")
        else:
            _blocks((value["sum"],), 1, source.native.field_degree, bits)
            if tuple(value["sum"]) != tuple(sum((row[j] for row in detail["coefficients"]), Fraction(0)) for j in range(source.native.field_degree)):
                raise ValueError("algebraic window sum differs from its complete coordinates")
    if (detail["namespace"] != value["publication"]["namespace"] or detail["sequence"] != value["publication"]["sequence"]):
        raise ValueError("algebraic observation publication differs")
    _key(detail["namespace"]); _key(detail["sequence"])
    _sha(value["seal"])
    body = {key: v for key, v in value.items() if key != "seal"}
    if sha256(pack_value(body)).hexdigest() != value["seal"]:
        raise ValueError("algebraic observation seal differs")
    return value


def algebraic_snapshot_record(snapshot):
    """Retain complete RFCXA or RFCXG bytes and explicit publication labels."""
    from rexgraph_cpp import NativeAlgebraicSnapshot

    from rexgraph.io.algebraic_record import algebraic_record
    if type(snapshot) is not NativeAlgebraicSnapshot:
        raise TypeError("snapshot record requires an owning native publication")
    archive = snapshot.archive_owner
    record = algebraic_record(archive)
    record.attach_metadata(1, 0, "algebraic_snapshot", {"kind": "cpp_algebraic_snapshot", "version": 1,
        "profile": "algebraic-snapshot-publication-v1", "publication": _publication(snapshot),
        "archive_limits": archive.limits if archive.graph_owner is None else archive.graph_owner.limits})
    from rexgraph.cpp_algebraic_history import _attach_history
    _attach_history(record, archive)
    from rexgraph.cpp_algebraic_graph_history import _attach_graph_history
    _attach_graph_history(record, snapshot)
    return record


def read_algebraic_snapshot_record(record, index=0, *, limits=None, history_max_bytes=256 << 20, _archive_cache=None):
    """Restore independent owning publication pins under saved and reader archive limits."""
    from rexgraph_cpp import (
        NativeAlgebraicArchive,
        NativeAlgebraicArchiveGraph,
        NativeAlgebraicSnapshot,
    )

    from rexgraph.io.algebraic_record import read_algebraic_record
    index = _key(index)
    declared = record.get_metadata(1, index, "algebraic_snapshot")
    _closed(declared, ("kind", "version", "profile", "publication", "archive_limits"))
    if (declared["kind"] != "cpp_algebraic_snapshot" or type(declared["version"]) is not int or declared["version"] != 1
            or declared["profile"] != "algebraic-snapshot-publication-v1"):
        raise ValueError("unsupported algebraic snapshot record")
    reference = _check_publication(declared["publication"])
    graph = reference.native.graph_digest is not None
    graph_history = "graph_history" in declared["publication"]
    _portable_limits(declared["archive_limits"], graph=graph)
    reader = (NativeAlgebraicArchiveGraph if graph else NativeAlgebraicArchive).default_limits()
    allowed = set(reader)
    if graph_history:
        from rexgraph.cpp_algebraic_graph_history import _EXTRA_LIMITS
        allowed.update(_EXTRA_LIMITS)
    if limits is not None:
        if not isinstance(limits, Mapping) or set(limits) - allowed:
            raise ValueError("unknown algebraic snapshot reader admission")
        reader.update({key: _key(v) for key, v in limits.items() if key in reader})
    effective = {key: min(v, declared["archive_limits"][key]) for key, v in reader.items()}
    if graph_history:
        from rexgraph.cpp_algebraic_graph_history import _restore_graph_history_snapshot
        _, checked = _check_retained_snapshot(record, index, effective["max_file_bytes"], history_max_bytes, _with_history=True)
        policy = dict(limits or {})
        policy.update(effective)
        return _restore_graph_history_snapshot(record, index, declared["publication"], policy, checked, _archive_cache)
    if "history" in declared["publication"]:
        from rexgraph.cpp_algebraic_history import _restore_history_archive
        _, checked = _check_retained_snapshot(record, index, effective["max_file_bytes"], history_max_bytes,
                                               _with_history=True)
        archive = _restore_history_archive(record, index, declared["publication"], effective, checked, _archive_cache)
    else:
        archive = read_algebraic_record(record, index, limits=effective, _archive_cache=_archive_cache)
    if not reference.matches(CppAlgebraicSourceReference(archive)):
        raise ValueError("snapshot record source differs from its complete archive attachment")
    return NativeAlgebraicSnapshot(archive, namespace=declared["publication"]["namespace"],
                                  sequence=declared["publication"]["sequence"])


def algebraic_query_record(result, *, max_coordinates=65536):
    """Retain a query observation beside its complete owning source publication."""
    observation = algebraic_query_observation(result, max_coordinates=max_coordinates)
    record = algebraic_snapshot_record(result.query_owner.snapshot_owner)
    record.attach_metadata(1, 0, "algebraic_observation", observation)
    return record


def algebraic_window_record(window, *, availability, execution_limits=None, cancellation=None, max_coordinates=65536):
    """Retain a coefficient window observation beside its complete source publication."""
    observation = algebraic_window_observation(window, availability=availability, execution_limits=execution_limits,
        cancellation=cancellation, max_coordinates=max_coordinates)
    record = algebraic_snapshot_record(window.record_owner.snapshot_owner)
    record.attach_metadata(1, 0, "algebraic_observation", observation)
    return record


def _check_retained_snapshot(record, index, max_archive_bytes, max_history_bytes=256 << 20, *, _with_history=False):
    import numpy as np
    declared = record.get_metadata(1, index, "algebraic_snapshot")
    _closed(declared, ("kind", "version", "profile", "publication", "archive_limits"))
    if (declared["kind"] != "cpp_algebraic_snapshot" or type(declared["version"]) is not int or declared["version"] != 1
            or declared["profile"] != "algebraic-snapshot-publication-v1"):
        raise ValueError("unsupported algebraic snapshot record")
    source = _check_publication(declared["publication"])
    graph = source.native.graph_digest is not None
    _portable_limits(declared["archive_limits"], graph=graph)
    attachment = record.get_metadata(1, index, "algebraic_archive")
    _closed(attachment, {"kind", "version", "archive", "archive_digest", "source_digest"}
                       | ({"graph_digest", "archive_key"} if graph else set()))
    expected = "cpp_algebraic_graph_archive" if graph else "cpp_algebraic_archive"
    if attachment["kind"] != expected or type(attachment["version"]) is not int or attachment["version"] != 1:
        raise ValueError("unsupported retained algebraic archive attachment")
    raw = attachment["archive"]
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("retained algebraic archive requires a one dimensional uint8 tensor")
    if (raw.size > _key(max_archive_bytes) or raw.size > declared["archive_limits"]["max_file_bytes"]
            or attachment["archive_digest"] != source.native.archive_digest
            or attachment["source_digest"] != source.native.source_digest):
        raise ValueError("retained algebraic archive differs or exceeds reader admission")
    digest = source.native.graph_digest if graph else source.native.archive_digest
    if graph and (attachment["graph_digest"] != digest or _key(attachment["archive_key"]) != source.native.archive_key):
        raise ValueError("retained algebraic graph selection differs")
    if sha256(raw.tobytes()).hexdigest() != digest:
        raise ValueError("retained algebraic archive bytes differ")
    checked = None
    if "history" in declared["publication"]:
        from rexgraph.cpp_algebraic_history import _check_history_attachment
        checked = _check_history_attachment(record, index, declared["publication"], max_history_bytes)
    if "graph_history" in declared["publication"]:
        from rexgraph.cpp_algebraic_graph_history import _check_graph_history_attachment
        checked = _check_graph_history_attachment(record, index, declared["publication"], max_history_bytes)
    if _with_history:
        return declared, checked
    return declared


def read_algebraic_observation_record(record, index=0, *, max_archive_bytes=256 << 20, max_history_bytes=256 << 20, **admission):
    """Read checked exact observations without restoring or importing the native engine."""
    index = _key(index)
    observation = record.get_metadata(1, index, "algebraic_observation")
    validate_algebraic_observation(observation, **admission)
    declared = _check_retained_snapshot(record, index, max_archive_bytes, max_history_bytes)
    if declared["publication"] != observation["publication"]:
        raise ValueError("algebraic observation differs from its retained snapshot declaration")
    return deepcopy(observation)
