"""Whole grade algebraic fields, original stalk witnesses and exact shape windows."""
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from math import prod

from rexgraph.cpp_algebraic import _key
from rexgraph.cpp_algebraic_query import (
    _admit_coordinates,
    _blocks,
    _check_publication,
    _check_retained_snapshot,
    _closed,
    _cutoff,
    _limits,
    _publication,
    algebraic_snapshot_record,
    read_algebraic_snapshot_record,
)
from rexgraph.source_binding import _sha
from rexgraph.value_codec import pack_value

__all__ = ["prepare_algebraic_grade", "algebraic_grade_observation", "algebraic_grade_window_observation",
           "validate_algebraic_grade_observation", "algebraic_grade_record",
           "read_algebraic_grade_observation_record", "read_algebraic_grade_record",
           "restore_algebraic_grade_observation"]
_FIELD = "original-algebraic-grade-field-v1"
_WINDOW = "original-algebraic-grade-window-v1"
_META = {"profile", "namespace", "sequence", "grade", "derived", "cell_count", "domain",
         "record_keys", "stalk_offsets", "requirements", "execution_limits"}
_VALUES = {"coefficients", "source_coordinates", "truncated"}


def _matches_publication(saved, current):
    """Compare complete source witnesses while admitting narrower reader policies."""
    _check_publication(saved)
    _check_publication(current)
    declared, restored = dict(saved), dict(current)
    for key in ("history", "graph_history"):
        if key not in declared or key not in restored:
            continue
        old, new = dict(declared[key]), dict(restored[key])
        if new["max_bytes"] > old["max_bytes"]:
            return False
        old.pop("max_bytes"); new.pop("max_bytes")
        if key == "graph_history":
            allowed, actual = old.pop("limits"), new.pop("limits")
            if set(allowed) != set(actual) or any(actual[name] > allowed[name] for name in allowed):
                return False
        declared[key], restored[key] = old, new
    return declared == restored


def prepare_algebraic_grade(snapshot, request, *, availability, execution_limits=None, cancellation=None):
    """Select one named vector per original cell and apply declared native boundary steps."""
    from rexgraph_cpp import NativeAlgebraicGradeTensor
    _closed(request, ("grade", "record_keys", "boundary_steps"))
    grade, steps = _key(request["grade"]), _key(request["boundary_steps"])
    if grade > 65535 or steps > grade:
        raise ValueError("algebraic grade or boundary step count is outside its domain")
    tensor = NativeAlgebraicGradeTensor(snapshot, grade, request["record_keys"], availability=availability,
                                       execution_limits=execution_limits, cancellation=cancellation)
    for _ in range(steps):
        tensor = tensor.boundary(availability=availability, cancellation=cancellation)
    return tensor


def _metadata(tensor):
    return {key: value for key, value in tensor.preview(0).items() if key in _META}


def _lineage(tensor):
    owners = []
    while tensor is not None:
        owners.append(tensor)
        tensor = tensor.input_owner
    return tuple(_metadata(owner) for owner in reversed(owners))


def _observe(tensor, availability, detail, operation, contraction=None, *, max_coordinates):
    from rexgraph_cpp import build_info
    publication = _publication(tensor.snapshot_owner)
    _cutoff(publication, availability)
    value = {"kind": "cpp_algebraic_grade_observation", "version": 1,
             "arithmetic": "algebraic", "coefficient_basis": "power", "operation": operation,
             "publication": publication, "availability": tuple(availability), "engine": build_info(),
             "lineage": _lineage(tensor), "detail": detail, "sum": contraction}
    value["seal"] = sha256(pack_value(value)).hexdigest()
    validate_algebraic_grade_observation(value, max_coordinates=max_coordinates)
    return value


def algebraic_grade_observation(tensor, *, availability, max_coordinates=65536):
    """Capture complete exact grade coordinates and the selected boundary lineage."""
    from rexgraph_cpp import NativeAlgebraicGradeTensor
    if type(tensor) is not NativeAlgebraicGradeTensor:
        raise TypeError("grade observation requires an owning native grade tensor")
    _admit_coordinates(tensor.domain[0], tensor.snapshot_owner.archive_owner.field_degree, max_coordinates)
    return _observe(tensor, availability, tensor.preview(tensor.domain[0]), "field", max_coordinates=max_coordinates)


def algebraic_grade_window_observation(window, *, availability, execution_limits=None, cancellation=None,
                                      max_coordinates=65536):
    """Capture every supported window coordinate and its native coefficient sum."""
    from rexgraph_cpp import NativeAlgebraicGradeTensorWindow
    if type(window) is not NativeAlgebraicGradeTensorWindow:
        raise TypeError("grade window observation requires an owning native window")
    tensor = window.tensor_owner
    _admit_coordinates(window.supported_count, tensor.snapshot_owner.archive_owner.field_degree, max_coordinates)
    scalar = window.sum(availability=availability, execution_limits=execution_limits, cancellation=cancellation)
    contraction = None if scalar is None else scalar.coefficients[0]
    return _observe(tensor, availability, window.preview(window.supported_count), "window", contraction,
                    max_coordinates=max_coordinates)


def _check_admission(detail, reader):
    requirements = _limits(detail["requirements"], complete=True)
    limits = _limits(detail["execution_limits"], complete=True)
    for key, required in requirements.items():
        if required > min(limits[key], reader[key]):
            raise ValueError("algebraic grade declaration exceeds execution admission")
    return min(limits["integer_bits"], reader["integer_bits"])


def _check_metadata(value, publication, sizes, reader, max_records):
    _closed(value, _META)
    if value["profile"] != _FIELD or type(value["derived"]) is not bool:
        raise ValueError("unsupported algebraic grade declaration")
    grade, count = _key(value["grade"]), _key(value["cell_count"])
    if grade >= len(sizes) or count != sizes[grade]:
        raise ValueError("algebraic grade differs from its original source cells")
    if value["namespace"] != publication["namespace"] or value["sequence"] != publication["sequence"]:
        raise ValueError("algebraic grade publication differs")
    _key(value["namespace"]); _key(value["sequence"])
    domain, offsets, keys = value["domain"], value["stalk_offsets"], value["record_keys"]
    if any(isinstance(v, (str, bytes, Mapping)) for v in (domain, offsets, keys)):
        raise ValueError("algebraic grade coordinates require sized sequences")
    if len(domain) != 1 or len(offsets) != count + 1 or len(offsets) + len(keys) > max_records:
        raise ValueError("algebraic grade offsets or key count differs")
    n = _key(domain[0])
    offsets = tuple(_key(i) for i in offsets)
    if offsets[0] != 0 or offsets[-1] != n or any(a > b for a, b in zip(offsets, offsets[1:], strict=False)):
        raise ValueError("algebraic grade stalk offsets differ")
    keys = tuple(_key(key) for key in keys)
    if value["derived"]:
        if keys:
            raise ValueError("derived algebraic grade cannot declare selected record keys")
    elif len(keys) != count or any(key == 0 for key in keys):
        raise ValueError("selected algebraic grade requires a record for every original cell")
    return _check_admission(value, reader)


def _witnesses(detail, metadata, indices):
    witnesses = detail["source_coordinates"]
    if isinstance(witnesses, (str, bytes, Mapping)) or len(witnesses) != len(indices):
        raise ValueError("algebraic grade original coordinate count differs")
    offsets = metadata["stalk_offsets"]
    for coordinate, index in zip(witnesses, indices, strict=True):
        if isinstance(coordinate, (str, bytes, Mapping)) or len(coordinate) != 3:
            raise ValueError("algebraic grade requires original cell coordinate triples")
        grade, cell, local = map(_key, coordinate)
        if (grade != metadata["grade"] or cell >= metadata["cell_count"]
                or local >= offsets[cell + 1] - offsets[cell] or offsets[cell] + local != index):
            raise ValueError("algebraic grade original stalk coordinate differs")


def validate_algebraic_grade_observation(value, *, max_coordinates=65536, max_records=4000000,
                                        execution_limits=None):
    """Check closed exact observations and receipts without importing the native engine."""
    _closed(value, ("kind", "version", "arithmetic", "coefficient_basis", "operation", "publication",
                    "availability", "engine", "lineage", "detail", "sum", "seal"))
    if (value["kind"] != "cpp_algebraic_grade_observation" or type(value["version"]) is not int
            or value["version"] != 1 or value["arithmetic"] != "algebraic"
            or value["coefficient_basis"] != "power" or value["operation"] not in {"field", "window"}):
        raise ValueError("unsupported algebraic grade observation")
    source = _check_publication(value["publication"])
    _cutoff(value["publication"], value["availability"])
    engine = value["engine"]
    _closed(engine, ("engine_version", "engine_digest", "binding_digest", "binding_version", "scalar_policy"))
    _sha(engine["engine_digest"]); _sha(engine["binding_digest"])
    if (not isinstance(engine["engine_version"], str) or not engine["engine_version"]
            or engine["binding_version"] != "1" or engine["scalar_policy"] != "gmp-rational-exact"):
        raise ValueError("unsupported algebraic grade engine receipt")
    lineage, reader, max_records = value["lineage"], _limits(execution_limits), _key(max_records)
    if isinstance(lineage, (str, bytes, Mapping)) or not lineage or len(lineage) > len(source.native.source_sizes):
        raise ValueError("algebraic grade requires its selected boundary lineage")
    if len(lineage) > max_records:
        raise ValueError("algebraic grade lineage exceeds record admission")
    used = len(lineage)
    for i, declaration in enumerate(lineage):
        _check_metadata(declaration, value["publication"], source.native.source_sizes, reader, max_records)
        used += len(declaration["record_keys"]) + len(declaration["stalk_offsets"])
        if used > max_records:
            raise ValueError("algebraic grade lineage exceeds record admission")
        if declaration["derived"] != (i != 0) or i and declaration["grade"] != lineage[i - 1]["grade"] - 1:
            raise ValueError("algebraic grade boundary lineage differs")
        if i and any(declaration["execution_limits"][key] > lineage[i - 1]["execution_limits"][key] for key in reader):
            raise ValueError("algebraic grade boundary enlarged retained execution admission")
    metadata, detail = lineage[-1], value["detail"]
    degree = source.native.field_degree
    if value["operation"] == "field":
        _closed(detail, _META | _VALUES)
        if {key: detail[key] for key in _META} != metadata or value["sum"] is not None:
            raise ValueError("algebraic field observation differs from its lineage")
        count, bits = metadata["domain"][0], _check_admission(metadata, reader)
        indices = range(count)
    else:
        _closed(detail, ("profile", "namespace", "sequence", "grade", "domain", "source_domain", "lower", "upper",
                        "supported_count", "requirements", "execution_limits", "source_indices", *_VALUES))
        if (detail["profile"] != _WINDOW or detail["grade"] != metadata["grade"]
                or detail["namespace"] != metadata["namespace"] or detail["sequence"] != metadata["sequence"]):
            raise ValueError("algebraic grade window source differs")
        shape = detail["source_domain"]
        if any(isinstance(detail[key], (str, bytes, Mapping)) for key in ("source_domain", "domain", "lower", "upper", "source_indices")):
            raise ValueError("algebraic grade window requires sized coordinate sequences")
        if len(shape) > max_records or prod(tuple(_key(n) for n in shape)) != metadata["domain"][0]:
            raise ValueError("algebraic grade window shape differs from its complete grade")
        if any(len(detail[key]) != len(shape) for key in ("domain", "lower", "upper")):
            raise ValueError("algebraic grade window rank differs")
        for size, lo, hi, extent in zip(shape, detail["lower"], detail["upper"], detail["domain"], strict=True):
            if not 0 <= _key(lo) <= _key(hi) <= size or _key(extent) != hi - lo:
                raise ValueError("algebraic grade window bounds differ")
        count, bits = prod(detail["domain"]), _check_admission(detail, reader)
        if _key(detail["supported_count"]) != count:
            raise ValueError("algebraic grade window support count differs")
        if any(detail["execution_limits"][key] > metadata["execution_limits"][key] for key in reader):
            raise ValueError("algebraic grade window enlarged retained execution admission")
        _admit_coordinates(count, degree, max_coordinates)
        if len(detail["source_indices"]) != count:
            raise ValueError("algebraic grade window source index count differs")
        indices = []
        for i in range(count):
            index, remaining = 0, i
            for axis in range(len(shape) - 1, -1, -1):
                position = remaining % detail["domain"][axis] + detail["lower"][axis]
                remaining //= detail["domain"][axis]
                index += position * prod(shape[axis + 1:])
            indices.append(index)
        if tuple(_key(i) for i in detail["source_indices"]) != tuple(indices):
            raise ValueError("algebraic grade window source indices differ")
    _admit_coordinates(count, degree, max_coordinates)
    if detail["truncated"] is not False:
        raise ValueError("complete algebraic grade observation cannot be truncated")
    _blocks(detail["coefficients"], count, degree, bits)
    _witnesses(detail, metadata, indices)
    if value["operation"] == "window":
        if not count:
            if value["sum"] is not None:
                raise ValueError("empty algebraic grade window has no sum")
        else:
            _blocks((value["sum"],), 1, degree, bits)
            if tuple(value["sum"]) != tuple(sum((row[j] for row in detail["coefficients"]), Fraction(0)) for j in range(degree)):
                raise ValueError("algebraic grade window sum differs from its complete coordinates")
    _sha(value["seal"])
    if sha256(pack_value({key: v for key, v in value.items() if key != "seal"})).hexdigest() != value["seal"]:
        raise ValueError("algebraic grade observation seal differs")
    return value


def algebraic_grade_record(value, *, availability, execution_limits=None, cancellation=None, max_coordinates=65536):
    """Retain a complete grade or window observation beside its owning source publication."""
    from rexgraph_cpp import NativeAlgebraicGradeTensor, NativeAlgebraicGradeTensorWindow
    if type(value) is NativeAlgebraicGradeTensor:
        if execution_limits is not None or cancellation is not None:
            raise ValueError("field record observations use their retained execution declaration")
        observation = algebraic_grade_observation(value, availability=availability, max_coordinates=max_coordinates)
        tensor = value
    elif type(value) is NativeAlgebraicGradeTensorWindow:
        observation = algebraic_grade_window_observation(value, availability=availability,
            execution_limits=execution_limits, cancellation=cancellation, max_coordinates=max_coordinates)
        tensor = value.tensor_owner
    else:
        raise TypeError("grade record requires an owning native field or window")
    record = algebraic_snapshot_record(tensor.snapshot_owner)
    record.attach_metadata(1, 0, "algebraic_grade_observation", observation)
    return record


def read_algebraic_grade_observation_record(record, index=0, *, max_archive_bytes=256 << 20,
                                           max_history_bytes=256 << 20, **admission):
    """Read checked exact observations without restoring native carrier or proof owners."""
    index = _key(index)
    observation = record.get_metadata(1, index, "algebraic_grade_observation")
    validate_algebraic_grade_observation(observation, **admission)
    declared = _check_retained_snapshot(record, index, max_archive_bytes, max_history_bytes)
    if declared["publication"] != observation["publication"]:
        raise ValueError("algebraic grade observation differs from its retained publication")
    return deepcopy(observation)


def read_algebraic_grade_record(record, index=0, *, limits=None, history_max_bytes=256 << 20,
                               execution_limits=None, max_coordinates=65536, cancellation=None, _archive_cache=None):
    """Restore actual source owners and replay the retained native grade boundary lineage."""
    observation = read_algebraic_grade_observation_record(record, index, max_history_bytes=history_max_bytes,
        execution_limits=execution_limits, max_coordinates=max_coordinates)
    snapshot = read_algebraic_snapshot_record(record, index, limits=limits, history_max_bytes=history_max_bytes,
                                              _archive_cache=_archive_cache)
    return restore_algebraic_grade_observation(snapshot, observation, execution_limits=execution_limits,
        max_coordinates=max_coordinates, cancellation=cancellation)


def restore_algebraic_grade_observation(snapshot, observation, *, execution_limits=None,
                                      max_coordinates=65536, cancellation=None):
    """Replay a complete observation on an explicitly supplied owning source publication."""
    from rexgraph_cpp import (
        NativeAlgebraicGradeTensor,
        NativeAlgebraicGradeTensorWindow,
        NativeAlgebraicSnapshot,
    )
    if type(snapshot) is not NativeAlgebraicSnapshot:
        raise TypeError("grade restoration requires an owning native snapshot")
    validate_algebraic_grade_observation(observation, execution_limits=execution_limits,
                                        max_coordinates=max_coordinates)
    if not _matches_publication(observation["publication"], _publication(snapshot)):
        raise ValueError("algebraic grade observation belongs to another source publication")
    reader = _limits(execution_limits)
    availability = observation["availability"]
    tensor = None
    for declaration in observation["lineage"]:
        policy = {key: min(reader[key], saved) for key, saved in declaration["execution_limits"].items()}
        if tensor is None:
            tensor = NativeAlgebraicGradeTensor(snapshot, declaration["grade"], declaration["record_keys"],
                availability=availability, execution_limits=policy, cancellation=cancellation)
        else:
            tensor = tensor.boundary(availability=availability, execution_limits=policy, cancellation=cancellation)
        current = _metadata(tensor)
        if {key: v for key, v in current.items() if key != "execution_limits"} != {key: v for key, v in declaration.items() if key != "execution_limits"}:
            raise ValueError("restored algebraic grade lineage differs from its actual source owners")
    if observation["operation"] == "field":
        value = tensor
        actual = algebraic_grade_observation(value, availability=availability, max_coordinates=max_coordinates)
    else:
        detail = observation["detail"]
        policy = {key: min(reader[key], saved) for key, saved in detail["execution_limits"].items()}
        value = NativeAlgebraicGradeTensorWindow(tensor, detail["source_domain"], detail["lower"], detail["upper"],
            availability=availability, execution_limits=policy, cancellation=cancellation)
        actual = algebraic_grade_window_observation(value, availability=availability, cancellation=cancellation,
                                                    max_coordinates=max_coordinates)
    if ({key: v for key, v in actual["detail"].items() if key != "execution_limits"}
            != {key: v for key, v in observation["detail"].items() if key != "execution_limits"}
            or actual["sum"] != observation["sum"]):
        raise ValueError("restored algebraic grade coordinates differ from their native action")
    return value
