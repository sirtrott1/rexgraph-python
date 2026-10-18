"""Declared algebraic coefficient features for sparse integer classifiers."""
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock

import numpy as np

from rexgraph.exact_value import exact_fraction
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_state import ModelBatch, ModelOutput, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.value_codec import pack_value, unpack_value

_PROFILE = "algebraic-coefficient-features-v1"
_MAX_BYTES = 256 << 20
_NAMES = {"profile", "source", "primary_space", "feature_carrier", "feature_coordinates",
          "feature_count", "grade", "carrier", "max_origin_bytes", "max_entries", "origins", "bindings"}
_origin_cache = OrderedDict()
_origin_lock = RLock()
_origin_cache_bytes = 0


def _decoded_origin(raw):
    global _origin_cache_bytes
    with _origin_lock:
        if raw in _origin_cache:
            _origin_cache.move_to_end(raw)
            return _origin_cache[raw]
    output = _output(unpack_model(unpack_value(raw)))
    with _origin_lock:
        if raw in _origin_cache:
            _origin_cache.move_to_end(raw)
            return _origin_cache[raw]
        if len(raw) <= 32 << 20:
            _origin_cache[raw] = output
            _origin_cache_bytes += len(raw)
            while len(_origin_cache) > 8 or _origin_cache_bytes > 32 << 20:
                previous, _ = _origin_cache.popitem(last=False)
                _origin_cache_bytes -= len(previous)
    return output


def _uint(value, bits=64):
    from rexgraph.cpp_learning import _uint as checked
    return checked(value, bits)


def _maximum(value):
    value = _uint(value)
    if not 0 < value <= _MAX_BYTES:
        raise ValueError("coefficient feature origin admission requires 1 through 268435456 bytes")
    return value


def _output(output):
    if not isinstance(output, ModelOutput) or output.arithmetic != "algebraic":
        raise TypeError("coefficient features require complete exact algebraic model outputs")
    output.check_state()
    from rexgraph.cpp_algebraic import validate_algebraic_output
    validate_algebraic_output(output)
    if (output.execution_receipt.get("kind") not in {"vector", "covector", "scalar"}
            or output.values.ndim != 2 or len(output.axes) != 1):
        raise ValueError("coefficient features require a declared carrier vector, covector or scalar")
    return output


def validate_coefficient_features(record, *, state=None, max_origin_bytes=None):
    """Check retained producers and return the declared exact integer feature rows."""
    if not isinstance(record, Mapping) or set(record) != _NAMES or record["profile"] != _PROFILE:
        raise ValueError("unsupported algebraic coefficient feature declaration")
    maximum = _maximum(record["max_origin_bytes"])
    if max_origin_bytes is not None:
        maximum = min(maximum, _maximum(max_origin_bytes))
    if len(pack_value(thaw_tree(record))) > maximum:
        raise ValueError("coefficient feature origin byte admission exceeded")
    source = source_from_record(record["source"], allow_cpp=True)
    if not isinstance(source, CppSourceReference):
        raise TypeError("coefficient features require the learner's native source publication")
    space = record["primary_space"]
    if (not isinstance(space, tuple) or len(space) != 2 or not isinstance(space[0], str)
            or not isinstance(space[1], tuple) or not space[1] or len(set(space[1])) != len(space[1])):
        raise ValueError("coefficient features require distinct named learner rows")
    count = _uint(record["feature_count"])
    carrier, coordinates = _uint(record["feature_carrier"]), _uint(record["feature_coordinates"])
    grade, primary_carrier = _uint(record["grade"], 16), _uint(record["carrier"])
    from rexgraph.cpp_model import _space
    if grade >= len(source.native.coordinates) or space != (
            _space(source, grade, primary_carrier).name, source.native.coordinates[grade]):
        raise ValueError("coefficient features differ from the declared native learner grade")
    entries_limit = _uint(record["max_entries"])
    if not count or not coordinates:
        raise ValueError("coefficient features require a nonempty explicit parameter carrier")
    if state is not None:
        c = state.configuration
        if (state.adapter != "cpp_perceptron" or record["source"] != state.source.as_record()
                or space != (state.space.name, state.space.keys) or carrier != c["feature_carrier"]
                or coordinates != c["feature_coordinates"] or count != c["feature_count"]
                or grade != c["grade"] or primary_carrier != c["carrier"]
                or entries_limit > c["limits"][2]):
            raise ValueError("coefficient features differ from the learner's declared coordinates")
    origins, bindings = record["origins"], record["bindings"]
    if not isinstance(origins, tuple) or not isinstance(bindings, tuple) or len(bindings) != len(space[1]):
        raise ValueError("coefficient features must explicitly bind every ordered learner row")
    if not origins or len(origins) > len(bindings) or any(type(raw) is not bytes for raw in origins):
        raise ValueError("coefficient features require bounded complete producer outputs")
    outputs = tuple(_decoded_origin(raw) for raw in origins)
    if len(set(origins)) != len(origins):
        raise ValueError("coefficient feature producers must be stored once in first use order")
    from rexgraph.cpp_learning import _int64
    rows, used, entries = [], [], 0
    for key, binding in zip(space[1], bindings, strict=True):
        if not isinstance(binding, Mapping) or set(binding) != {"row", "origin", "producer", "features"} or binding["row"] != key:
            raise ValueError("coefficient feature binding changed its declared learner row")
        origin = _uint(binding["origin"])
        if origin >= len(outputs):
            raise ValueError("coefficient feature binding references an absent producer")
        if origin not in used:
            used.append(origin)
        output = outputs[origin]
        if binding["producer"] != output.coefficient_digest:
            raise ValueError("coefficient feature binding changed its declared producer output")
        if not isinstance(binding["features"], tuple):
            raise ValueError("coefficient feature coordinates require an ordered declaration")
        entries += len(binding["features"])
        if entries > entries_limit:
            raise ValueError("coefficient feature entry admission exceeded")
        row, seen = [], set()
        for feature in binding["features"]:
            if not isinstance(feature, tuple) or len(feature) != 4:
                raise ValueError("coefficient features require parameter, coordinate, power and scale")
            index, coordinate, power = (_uint(v) for v in feature[:3])
            scale = exact_fraction(feature[3])
            if (index >= count or index in seen or coordinate >= len(output.space.keys)
                    or power >= len(output.axes[0].keys) or not scale):
                raise ValueError("coefficient feature lies outside its declared coordinates or repeats a parameter")
            seen.add(index)
            value = exact_fraction(output.values[coordinate, power]) * scale
            if value.denominator != 1:
                raise ValueError("declared coefficient feature scale does not produce an integer")
            # Retain explicit supported zeros in their declared feature order.
            row.append((index, _int64(value.numerator)))
        rows.append(tuple(row))
    if used != list(range(len(origins))):
        raise ValueError("coefficient feature producer table differs from first use order")
    return tuple(rows)


def cpp_coefficient_perceptron_input(state, source, outputs, *, row_keys, features,
                                    max_origin_bytes=16 << 20, cpu_workers=None,
                                    rows_per_tile=256, prepare=False):
    """Bind each learner row to explicit polynomial coefficient features.

    Each feature is (parameter_index, coordinate, power, exact_scale). Scaling
    must yield a signed 64 bit integer. Producer outputs retain their original
    field and carrier declarations. These features are coefficient observations.
    """
    from rexgraph.cpp_perceptron import cpp_perceptron_input
    state.bind(source)
    if state.adapter != "cpp_perceptron":
        raise TypeError("coefficient feature input requires a cpp_perceptron model")
    if tuple(row_keys) != state.space.keys:
        raise ValueError("coefficient feature row links must match the declared learner order")
    outputs, features = tuple(outputs), tuple(features)
    if len(outputs) != len(state.space.keys) or len(features) != len(outputs):
        raise ValueError("coefficient feature declarations must match all learner rows")
    origins, bindings, known, retained = [], [], {}, 0
    for key, output, row_features in zip(state.space.keys, outputs, features, strict=True):
        _output(output)
        digest = output.coefficient_digest
        index = known.get(digest)
        if index is None:
            raw = pack_value(pack_model(output, native=True))
            retained += len(raw)
            if retained > _maximum(max_origin_bytes):
                raise ValueError("coefficient feature origin byte admission exceeded")
            index = len(origins)
            known[digest] = index
            origins.append(raw)
        bindings.append({"row": key, "origin": index, "producer": digest,
                         "features": tuple(tuple(feature) for feature in row_features)})
    c = state.configuration
    record = freeze_tree({"profile": _PROFILE, "source": state.source.as_record(),
        "primary_space": (state.space.name, state.space.keys), "feature_carrier": c["feature_carrier"],
        "feature_coordinates": c["feature_coordinates"], "feature_count": c["feature_count"],
        "grade": c["grade"], "carrier": c["carrier"],
        "max_origin_bytes": _maximum(max_origin_bytes), "max_entries": c["limits"][2],
        "origins": tuple(origins), "bindings": tuple(bindings)})
    rows = validate_coefficient_features(record, state=state)
    inputs = dict(cpp_perceptron_input(state, source, rows, cpu_workers=cpu_workers,
                                     rows_per_tile=rows_per_tile))
    inputs["coefficient_features"] = record
    inputs = freeze_tree(inputs)
    if type(prepare) is not bool:
        raise TypeError("coefficient feature input preparation must be boolean")
    if prepare:
        from rexgraph.cpp_perceptron import _prepare_input_rows, prepare_perceptron
        _prepare_input_rows(prepare_perceptron(state, source), inputs)
    return inputs


def cpp_coefficient_perceptron_batch(state, source, outputs, labels, *, row_keys, features,
                                    rate=1, event_cutoff, knowledge_cutoff, observed=None,
                                    max_origin_bytes=16 << 20):
    """Capture declared coefficient features, signed labels and learner causal cutoffs."""
    from rexgraph.cpp_learning import _int64
    inputs = dict(cpp_coefficient_perceptron_input(state, source, outputs, row_keys=row_keys,
        features=features, max_origin_bytes=max_origin_bytes))
    inputs.update(rate=_int64(rate), event_cutoff=_uint(event_cutoff), knowledge_cutoff=_uint(knowledge_cutoff))
    targets = np.asarray(tuple(_int64(v) for v in labels), dtype=object)
    observed = np.ones(len(state.space.keys), dtype=bool) if observed is None else observed
    return ModelBatch(state.bind(source), state.space, targets, observed, inputs)


def validate_coefficient_feature_history(state, history=None):
    """Check all retained coefficient observations with a total serialized byte bound."""
    total = 0
    for entry in state.payload.get("history", ()) if history is None else history:
        if not isinstance(entry, Mapping) or "coefficient_features" not in entry:
            continue
        if set(entry) != {"rows", "targets", "observed", "rate", "event_cutoff", "knowledge_cutoff", "coefficient_features"}:
            raise ValueError("unsupported coefficient feature epoch checkpoint")
        record = entry["coefficient_features"]
        total += len(pack_value(thaw_tree(record)))
        if total > _MAX_BYTES:
            raise ValueError("coefficient feature history byte admission exceeded")
        if validate_coefficient_features(record, state=state) != entry["rows"]:
            raise ValueError("integer checkpoint rows differ from their retained coefficient producers")


def validate_coefficient_feature_output(output):
    """Check the learner coordinates of a retained coefficient feature inference."""
    r = output.execution_receipt
    record = r["coefficient_features"]
    validate_coefficient_features(record)
    if (output.method != "checked-int64-perceptron-v1" or output.arithmetic != "rational"
            or output.axes or output.variance != "coordinate"
            or output.grade != record["grade"]
            or record["source"] != output.source.as_record()
            or record["primary_space"] != (output.space.name, output.space.keys)
            or record["feature_carrier"] != r["feature_carrier"]
            or record["feature_coordinates"] != r["feature_coordinates"]
            or record["max_entries"] > r["limits"][2]):
        raise ValueError("coefficient feature inference differs from its declared learner coordinates")
