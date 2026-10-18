"""Exact whole grade metric observations on explicitly retained native sources."""
from copy import deepcopy
from hashlib import sha256

from rexgraph.cpp_algebraic import _key
from rexgraph.cpp_algebraic_grade import (
    _matches_publication,
    algebraic_grade_observation,
    restore_algebraic_grade_observation,
    validate_algebraic_grade_observation,
)
from rexgraph.cpp_algebraic_query import (
    _admit_coordinates,
    _blocks,
    _check_publication,
    _check_retained_snapshot,
    _closed,
    _cutoff,
    algebraic_snapshot_record,
    read_algebraic_snapshot_record,
)
from rexgraph.source_binding import _sha
from rexgraph.value_codec import pack_value

__all__ = ["algebraic_grade_metric_observation", "validate_algebraic_grade_metric_observation",
           "algebraic_grade_metric_record", "read_algebraic_grade_metric_observation_record",
           "read_algebraic_grade_metric_record", "restore_algebraic_grade_metric_observation"]
_PROFILE = "original-algebraic-grade-metric-v1"
_ACTIONS = {"hodge", "coboundary", "inner", "adjacent_quadrance"}


def _evaluate(field, tensor, operation, availability, other, cancellation):
    if operation in {"hodge", "coboundary"}:
        if other is not None:
            raise ValueError("metric vector actions accept one grade input")
        return field.apply(tensor, operation, availability=availability, cancellation=cancellation)
    if operation == "inner":
        if other is None:
            raise ValueError("metric inner product requires a second owning grade input")
        return field.inner(tensor, other, availability=availability, cancellation=cancellation)
    if operation == "adjacent_quadrance":
        if other is not None:
            raise ValueError("adjacent quadrance accepts one grade input")
        return field.adjacent_quadrance(tensor, availability=availability, cancellation=cancellation)
    raise ValueError("unsupported whole-grade metric operation")


def _detail(result, operation):
    if operation in {"hodge", "coboundary"}:
        return result.preview(result.domain[0])
    return {"profile": "algebraic-metric-scalar-v1", "coefficients": result.coefficients}


def algebraic_grade_metric_observation(field, tensor, operation, *, availability, other=None,
                                       cancellation=None, max_coordinates=65536):
    """Execute a canonical metric action and retain complete original grade inputs."""
    from rexgraph_cpp import NativeAlgebraicGradeTensor, NativeAlgebraicMetricField, build_info
    if type(field) is not NativeAlgebraicMetricField or type(tensor) is not NativeAlgebraicGradeTensor:
        raise TypeError("metric observations require an owning metric field and original grade tensor")
    if not tensor.matches_source(field.snapshot_owner):
        raise ValueError("metric input publication owner differs")
    if other is not None and (type(other) is not NativeAlgebraicGradeTensor or not other.matches_source(field.snapshot_owner)):
        raise ValueError("metric pairing input publication owner differs")
    if type(operation) is not str or operation not in _ACTIONS:
        raise ValueError("unsupported whole-grade metric operation")
    source = field.snapshot_owner
    publication = algebraic_grade_observation(tensor, availability=availability, max_coordinates=max_coordinates)
    second = None if other is None else algebraic_grade_observation(other, availability=availability, max_coordinates=max_coordinates)
    if operation == "coboundary":
        grade = tensor.grade + 1
        if grade >= len(source.archive_owner.manifest["source_sizes"]):
            raise ValueError("top grade has no coboundary target")
        carriers = (source.archive_owner.record(key).preview(0)["source_carrier"] for key in field.metric_keys)
        count = sum(item["rank"] for item in carriers if item["grade"] == grade)
        _admit_coordinates(count, source.archive_owner.field_degree, max_coordinates)
    elif operation == "hodge":
        _admit_coordinates(tensor.domain[0], source.archive_owner.field_degree, max_coordinates)
    else:
        _admit_coordinates(1, source.archive_owner.field_degree, max_coordinates)
    result = _evaluate(field, tensor, operation, availability, other, cancellation)
    value = {"kind": "cpp_algebraic_grade_metric_observation", "version": 1, "profile": _PROFILE,
             "arithmetic": "algebraic", "coefficient_basis": "power", "operation": operation,
             "publication": publication["publication"], "availability": tuple(availability), "engine": build_info(),
             "metric_keys": field.metric_keys, "max_retained_coordinates": field.max_retained_coordinates,
             "input": publication, "other": second, "detail": _detail(result, operation)}
    value["seal"] = sha256(pack_value(value)).hexdigest()
    validate_algebraic_grade_metric_observation(value, max_coordinates=max_coordinates)
    return value


def validate_algebraic_grade_metric_observation(value, *, max_coordinates=65536,
                                                max_records=4000000, execution_limits=None, max_integer_bits=1048576):
    """Check complete portable metric receipts without constructing native proof owners."""
    _closed(value, ("kind", "version", "profile", "arithmetic", "coefficient_basis", "operation", "publication",
                    "availability", "engine", "metric_keys", "max_retained_coordinates", "input", "other", "detail", "seal"))
    if (value["kind"] != "cpp_algebraic_grade_metric_observation" or type(value["version"]) is not int
            or value["version"] != 1 or value["profile"] != _PROFILE or value["arithmetic"] != "algebraic"
            or value["coefficient_basis"] != "power" or type(value["operation"]) is not str or value["operation"] not in _ACTIONS):
        raise ValueError("unsupported whole-grade metric observation")
    source = _check_publication(value["publication"])
    _cutoff(value["publication"], value["availability"])
    max_records = _key(max_records)
    keys = value["metric_keys"]
    if isinstance(keys, (str, bytes, dict)) or len(keys) != sum(source.native.source_sizes) or len(keys) > max_records:
        raise ValueError("metric field requires every original cell metric")
    if any(_key(key) == 0 for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("metric field requires distinct explicit metric records")
    _key(value["max_retained_coordinates"])
    first, second = value["input"], value["other"]
    operation = value["operation"]
    if (operation == "inner") != (second is not None):
        raise ValueError("metric operation input count differs")
    for item in (first,) if second is None else (first, second):
        validate_algebraic_grade_observation(item, max_coordinates=max_coordinates,
                                             max_records=max_records, execution_limits=execution_limits)
        if (item["operation"] != "field" or item["publication"] != value["publication"]
                or item["engine"] != value["engine"] or item["availability"] != value["availability"]):
            raise ValueError("metric action input source or engine differs")
    if second is not None and (first["detail"]["grade"] != second["detail"]["grade"]
                               or first["detail"]["stalk_offsets"] != second["detail"]["stalk_offsets"]):
        raise ValueError("metric pairing requires identical original grade carriers")
    detail, degree = value["detail"], source.native.field_degree
    bits = _key(max_integer_bits)
    if operation in {"inner", "adjacent_quadrance"}:
        _closed(detail, ("profile", "coefficients"))
        if detail["profile"] != "algebraic-metric-scalar-v1":
            raise ValueError("metric scalar profile differs")
        _admit_coordinates(1, degree, max_coordinates)
        _blocks(detail["coefficients"], 1, degree, bits)
    else:
        _closed(detail, ("profile", "grade", "cell_count", "domain", "stalk_offsets", "coefficients", "source_coordinates", "truncated"))
        grade = _key(detail["grade"])
        if (detail["profile"] != "original-algebraic-metric-values-v1" or detail["truncated"] is not False
                or grade != first["detail"]["grade"] + (operation == "coboundary")
                or grade >= len(source.native.source_sizes) or _key(detail["cell_count"]) != source.native.source_sizes[grade]):
            raise ValueError("metric result original grade differs")
        offsets, domain = detail["stalk_offsets"], detail["domain"]
        if isinstance(offsets, (str, bytes, dict)) or isinstance(domain, (str, bytes, dict)):
            raise ValueError("metric result requires original stalk offsets")
        if len(offsets) != detail["cell_count"] + 1 or len(domain) != 1 or len(offsets) > max_records:
            raise ValueError("metric result coordinate extent differs")
        offsets = tuple(_key(n) for n in offsets)
        count = _key(domain[0])
        if offsets[0] != 0 or offsets[-1] != count or any(a > b for a, b in zip(offsets, offsets[1:], strict=False)):
            raise ValueError("metric result stalk offsets differ")
        if operation == "hodge" and offsets != first["detail"]["stalk_offsets"]:
            raise ValueError("Hodge result must retain its original grade carriers")
        _admit_coordinates(count, degree, max_coordinates)
        _blocks(detail["coefficients"], count, degree, bits)
        expected = tuple((grade, cell, local) for cell, (a, b) in enumerate(zip(offsets, offsets[1:], strict=False)) for local in range(b - a))
        if detail["source_coordinates"] != expected:
            raise ValueError("metric result original cell coordinates differ")
    _sha(value["seal"])
    if sha256(pack_value({key: item for key, item in value.items() if key != "seal"})).hexdigest() != value["seal"]:
        raise ValueError("metric observation seal differs")
    return value


def algebraic_grade_metric_record(field, tensor, operation, *, availability, **options):
    """Save complete source publication, metrics and original input lineage with one exact result."""
    observation = algebraic_grade_metric_observation(field, tensor, operation, availability=availability, **options)
    record = algebraic_snapshot_record(field.snapshot_owner)
    record.attach_metadata(1, 0, "algebraic_grade_metric_observation", observation)
    return record


def read_algebraic_grade_metric_observation_record(record, index=0, *, max_archive_bytes=256 << 20,
                                                   max_history_bytes=256 << 20, **admission):
    """Read checked metric observations while leaving native source and metric proofs unmaterialized."""
    index = _key(index)
    value = record.get_metadata(1, index, "algebraic_grade_metric_observation")
    validate_algebraic_grade_metric_observation(value, **admission)
    retained = _check_retained_snapshot(record, index, max_archive_bytes, max_history_bytes)
    if retained["publication"] != value["publication"]:
        raise ValueError("metric observation differs from its retained source publication")
    return deepcopy(value)


def restore_algebraic_grade_metric_observation(snapshot, value, *, max_retained_coordinates=1000000,
                                                max_coordinates=65536, execution_limits=None, cancellation=None):
    """Replay complete grade inputs and canonical metric actions on their actual native owners."""
    from rexgraph_cpp import NativeAlgebraicMetricField, build_info

    from rexgraph.cpp_algebraic_query import _publication
    validate_algebraic_grade_metric_observation(value, max_coordinates=max_coordinates, execution_limits=execution_limits)
    if not _matches_publication(value["publication"], _publication(snapshot)):
        raise ValueError("metric observation source publication differs")
    current = build_info()
    if any(value["engine"][key] != current[key] for key in ("engine_version", "engine_digest", "scalar_policy")):
        raise ValueError("metric replay requires its retained canonical engine implementation")
    field = NativeAlgebraicMetricField(snapshot, value["metric_keys"], availability=value["availability"],
        max_retained_coordinates=min(_key(max_retained_coordinates), value["max_retained_coordinates"]), cancellation=cancellation)
    options = {"max_coordinates": max_coordinates, "execution_limits": execution_limits, "cancellation": cancellation}
    tensor = restore_algebraic_grade_observation(snapshot, value["input"], **options)
    other = None if value["other"] is None else restore_algebraic_grade_observation(snapshot, value["other"], **options)
    result = _evaluate(field, tensor, value["operation"], value["availability"], other, cancellation)
    if _detail(result, value["operation"]) != value["detail"]:
        raise ValueError("metric result differs from its canonical native action")
    return field, tensor, other, result


def read_algebraic_grade_metric_record(record, index=0, *, limits=None, history_max_bytes=256 << 20,
                                       **admission):
    """Restore complete native publications and verify original metric action outputs."""
    value = read_algebraic_grade_metric_observation_record(record, index, max_history_bytes=history_max_bytes,
        **{key: item for key, item in admission.items() if key in {"max_coordinates", "execution_limits"}})
    snapshot = read_algebraic_snapshot_record(record, index, limits=limits, history_max_bytes=history_max_bytes)
    restored = restore_algebraic_grade_metric_observation(snapshot, value, **admission)
    return value, snapshot, restored
