"""Exact field model declarations over owning C++ sources."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from numbers import Integral
from threading import RLock

import numpy as np

from rexgraph.exact_value import exact_fraction
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value

_provider = None
_provider_lock = RLock()
_limits = (4000000, 16000000, 16000000, 64000000, 1 << 24)


def _binding():
    try:
        import rexgraph_cpp
    except ModuleNotFoundError as exc:
        if exc.name != "rexgraph_cpp": raise
        raise ImportError("this model requires the optional rexgraph-cpp Python bindings") from exc
    return rexgraph_cpp


def _space(reference, grade, carrier):
    return CoordinateSpace(f"cpp:{reference.native.namespace}:grade:{grade}:carrier:{carrier}",
                           reference.native.coordinates[grade])


def _prepare(state, source):
    if state.adapter == "cpp_perceptron":
        from rexgraph.cpp_perceptron import prepare_perceptron
        return prepare_perceptron(state, source)
    if state.adapter != "cpp_field":
        from rexgraph.cpp_learning import prepare_learning_model
        return prepare_learning_model(state, source)
    binding = _binding()
    if dict(state.configuration["engine_receipt"]) != binding.build_info():
        raise ValueError("model requires a different native semantic implementation")
    state.bind(source)
    c = state.configuration
    expected = cpp_model(source, c["grade_carriers"], grade=c["grade"],
                         operation=c["operation"], parameter=c["parameter"],
                         metric_keys=c["metric_keys"], metric_selection=c["metric_selection"],
                         profile=c["profile"], dimension_limit=c["dimension_limit"],
                         limits=c["limits"], reference=state.source)
    if "transport" in state.payload:
        from rexgraph.cpp_transport import restore_transport, retained_payload
        expected, offset = restore_transport(state, source)
        if state.step != offset or state.payload != retained_payload(state, dict(expected.payload)):
            raise ValueError("transported field model differs from its declared fixed profile")
    elif expected.coefficient_digest != state.coefficient_digest:
        raise ValueError("native model differs from its declared fixed field profile")
    return source.prepare(c["grade_carriers"], metric_keys=c["metric_keys"],
                          metric_selection=c["metric_selection"], profile=c["profile"],
                          dimension_limit=c["dimension_limit"], limits=c["limits"])


def _infer(state, source, inputs):
    if state.adapter == "cpp_perceptron":
        from rexgraph.cpp_perceptron import infer_perceptron
        return infer_perceptron(state, source, inputs)
    if state.adapter != "cpp_field":
        from rexgraph.cpp_learning import infer_learning_model
        return infer_learning_model(state, source, inputs)
    binding = _binding()
    c = state.configuration
    native = _prepare(state, source)
    if isinstance(inputs, Mapping):
        record = _rational_batch_record(state, source, inputs)
        fields = tuple(source.field(key) for key in record["field_keys"])
        result = native.execute_batch(fields, operation=c["operation"], grade=c["grade"], parameter=c["parameter"])
        return _rational_batch_output(state, source, record, result,
            {"metric_keys": c["metric_keys"], "metric_selection": c["metric_selection"],
             "operation": c["operation"], "operator_grade": c["grade"], "parameter": c["parameter"]})
    if not isinstance(inputs, binding.NativeField):
        raise TypeError("this exact native profile requires a stored NativeField input")
    result = native.execute(inputs, operation=c["operation"], grade=c["grade"], parameter=c["parameter"])
    reference = state.source.bind(source)
    output_space = _space(reference, result["grade"], result["output_carrier"])
    receipt = {key: value for key, value in result.items() if key != "values"}
    receipt["engine"] = binding.build_info()
    from rexgraph.cpp_transport import transport_receipt
    receipt.update(transport_receipt(state))
    receipt.update({"metric_keys": c["metric_keys"], "metric_selection": c["metric_selection"],
                    "operation": c["operation"], "operator_grade": c["grade"],
                    "parameter": c["parameter"], "limits": c["limits"]})
    return ModelOutput(result["values"], output_space, (), reference, state.coefficient_digest,
                       "rational", result["profile"], grade=result["grade"], variance="chain",
                       execution_receipt=receipt)


def _train(state, source, batch, *, steps):
    if state.adapter == "cpp_perceptron":
        from rexgraph.cpp_perceptron import train_perceptron
        return train_perceptron(state, source, batch, steps=steps)
    if state.adapter != "cpp_metric":
        raise NotImplementedError("this native model profile does not support training")
    from rexgraph.cpp_learning import train_learning_model
    return train_learning_model(state, source, batch, steps=steps)


def _create(source, *args, adapter="cpp_field", **kwargs):
    if adapter == "cpp_perceptron":
        from rexgraph.cpp_perceptron import cpp_perceptron_model
        return cpp_perceptron_model(source, *args, **kwargs)
    if adapter == "cpp_field":
        return cpp_model(source, *args, **kwargs)
    from rexgraph.cpp_learning import cpp_integer_model, cpp_trainable_model
    if adapter == "cpp_integer":
        return cpp_integer_model(source, *args, **kwargs)
    if adapter == "cpp_metric":
        return cpp_trainable_model(source, *args, **kwargs)
    raise ValueError("unsupported native model adapter")


def _transport(state, source, destination, mapping, *, optimizer):
    from rexgraph.cpp_transport import transport_cpp_model
    return transport_cpp_model(state, source, destination, mapping, optimizer=optimizer)


def enable_cpp_provider():
    """Register native field and learning profiles without importing Torch."""
    global _provider
    _binding()
    with _provider_lock:
        if _provider is None:
            _provider = ModelProvider("rexgraph.cpp", "1", (("cpp_field", "1"),
                ("cpp_integer", "1"), ("cpp_metric", "1"), ("cpp_perceptron", "1")), _infer,
                create=_create, prepare=_prepare, train=_train, transport=_transport)
        register_model_provider(_provider)
        return _provider


def cpp_model(source, grade_carriers, *, grade=1, operation="hodge", parameter=None,
              metric_keys=(), metric_selection="unit", profile="finite", dimension_limit=64,
              limits=_limits, reference=None):
    """Declare a prepared exact action on explicitly selected native grade carriers."""
    binding = _binding()
    if not isinstance(source, binding.NativeSource):
        raise TypeError("native model requires an owning NativeSource")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    if not isinstance(ref, CppSourceReference):
        raise TypeError("native model requires a C++ source reference")
    if (isinstance(grade, bool) or not isinstance(grade, Integral)
            or not 0 <= grade < len(ref.native.coordinates)):
        raise ValueError("model grade is outside the retained native source")
    grade = int(grade)
    if operation not in {"boundary", "adjoint", "hodge_down", "hodge_up", "hodge", "green", "harmonic"}:
        raise ValueError("unsupported native field model operation")
    if operation in {"boundary", "adjoint"} and not grade:
        raise ValueError("grade zero has no boundary or adjoint")
    parameter = exact_fraction(Fraction(1) if parameter is None and operation == "green"
                               else Fraction(0) if parameter is None else parameter)
    if operation != "green" and parameter:
        raise ValueError("only a Green action accepts a model parameter")
    if operation in {"green", "harmonic"} and profile != "global":
        raise ValueError("Green and harmonic actions require the explicit global profile")
    carriers, metrics, limits = tuple(grade_carriers), tuple(metric_keys), tuple(limits)
    source.prepare(carriers, metric_keys=metrics, metric_selection=metric_selection,
                   profile=profile, dimension_limit=dimension_limit, limits=limits)
    configuration = {"execution_provider": {"name": "rexgraph.cpp", "version": "1"},
                     "engine_receipt": binding.build_info(), "grade": grade,
                     "operation": operation, "parameter": parameter, "grade_carriers": carriers,
                     "metric_keys": metrics, "metric_selection": metric_selection,
                     "profile": profile, "dimension_limit": dimension_limit, "limits": limits,
                     "input_profile": "stored-native-field-v1"}
    if profile == "finite":
        configuration["batch_input_profile"] = "stored-native-field-batch-v1"
    input_grade = grade - 1 if operation == "adjoint" else grade
    state = ModelState("cpp_field", "1", ref, _space(ref, input_grade, carriers[input_grade]), (),
                       configuration, {}, "rational")
    enable_cpp_provider()
    return state


def cpp_rational_batch_input(state, source, field_keys, samples):
    """Capture ordered stored fields on an explicit sample coordinate space."""
    if not isinstance(samples, CoordinateSpace):
        raise TypeError("rational batch requires a named sample CoordinateSpace")
    _prepare(state, source)
    return _rational_batch_record(state, source, {
        "profile": "stored-native-field-batch-v1", "source": state.source.as_record(),
        "samples": {"name": samples.name, "keys": samples.keys}, "field_keys": tuple(field_keys)})


def _rational_batch_record(state, source, record):
    if (state.adapter not in {"cpp_field", "cpp_metric"}
            or state.configuration.get("batch_input_profile") != "stored-native-field-batch-v1"):
        raise ValueError("model does not declare finite stored field batch inference")
    if (not isinstance(record, Mapping) or set(record) != {"profile", "source", "samples", "field_keys"}
            or record["profile"] != "stored-native-field-batch-v1"):
        raise ValueError("unsupported stored rational batch declaration")
    reference = source_from_record(record["source"], allow_cpp=True)
    if not isinstance(reference, CppSourceReference) or not reference.matches(state.bind(source)):
        raise ValueError("rational batch uses another source publication")
    axis = record["samples"]
    if not isinstance(axis, Mapping) or set(axis) != {"name", "keys"}:
        raise ValueError("rational batch requires named ordered sample coordinates")
    samples = CoordinateSpace(axis["name"], axis["keys"])
    keys = tuple(record["field_keys"])
    c = state.configuration
    grade = c["grade"] if state.adapter == "cpp_field" else c["declaration"]["grade"]
    output_grade = grade-1 if state.adapter == "cpp_field" and c["operation"] == "boundary" else grade
    output_space = _space(reference, output_grade, c["grade_carriers"][output_grade])
    if not keys or len(keys) != len(samples.keys) or samples.name in {state.space.name, output_space.name}:
        raise ValueError("rational batch dimensions differ from the declared sample axis")
    if (len(keys) > state.configuration["limits"][0]
            or len(keys) * len(state.space.keys) > state.configuration["limits"][1]):
        raise ValueError("rational batch exceeds its record or coefficient bound")
    if any(isinstance(key, bool) or not isinstance(key, Integral) or not 0 < key < 1 << 64 for key in keys):
        raise ValueError("native batch field keys require positive unsigned 64 bit integers")
    keys = tuple(int(key) for key in keys)
    if state.adapter == "cpp_field" and c["operation"] == "adjoint": grade -= 1
    for key in keys:
        if source.field(key).carrier != c["grade_carriers"][grade]:
            raise ValueError("rational batch field differs from the declared input carrier")
    return freeze_tree({"profile": record["profile"], "source": reference.as_record(),
                       "samples": {"name": samples.name, "keys": samples.keys}, "field_keys": keys})


def _rational_batch_output(state, source, record, result, additions):
    samples = CoordinateSpace(record["samples"]["name"], record["samples"]["keys"])
    reference = state.source.bind(source)
    output_space = _space(reference, result["grade"], result["output_carrier"])
    receipt = {key: value for key, value in result.items()
               if key not in {"values", "workspace_capacity", "workspace_slots"}}
    receipt.update(additions, engine=_binding().build_info(), limits=state.configuration["limits"],
                   input_digest=sha256(pack_value(thaw_tree(record))).hexdigest())
    from rexgraph.cpp_transport import transport_receipt
    receipt.update(transport_receipt(state))
    return ModelOutput(result["values"], output_space, (samples,), reference, state.coefficient_digest,
                       "rational", result["profile"], grade=result["grade"], variance="chain", execution_receipt=receipt)


def cpp_source_record(source, reference):
    """Capture canonical native archive bytes beside their source declaration."""
    reference.bind(source)
    return {"kind": "cpp_source_archive", "version": 1, "reference": reference.as_record(),
            "archive": np.frombuffer(source.archive, dtype=np.uint8).copy()}


class _SourceRestoreCache:
    """Bounded owning publications for repeated reads of one archive attachment."""

    def __init__(self, *, max_sources=8, max_archive_bytes=64 * 1024 * 1024):
        if (type(max_sources) is not int or max_sources < 0 or
                type(max_archive_bytes) is not int or max_archive_bytes < 0):
            raise ValueError("native source cache bounds require nonnegative integers")
        self._max_sources = max_sources
        self._max_archive_bytes = max_archive_bytes
        self._entries = OrderedDict()
        self._archive_bytes = 0
        self._lock = RLock()

    def get(self, record, archive, raw, *, read_limits=(1000000, 1000000, 0)):
        key = (id(record), id(archive), read_limits)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            owner, buffer, source, size = entry
            if owner is record and buffer is archive and source.archive == raw:
                self._entries.move_to_end(key)
                return source
            self._entries.pop(key)
            self._archive_bytes -= size
            return None

    def put(self, record, archive, source, *, read_limits=(1000000, 1000000, 0)):
        size = len(source.archive)
        if not self._max_sources or size > self._max_archive_bytes:
            return
        key = (id(record), id(archive), read_limits)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._archive_bytes -= previous[3]
            self._entries[key] = (record, archive, source, size)
            self._archive_bytes += size
            while (len(self._entries) > self._max_sources or
                   self._archive_bytes > self._max_archive_bytes):
                _, previous = self._entries.popitem(last=False)
                self._archive_bytes -= previous[3]


def restore_cpp_source(record, reference, *, max_bytes=256 * 1024 * 1024, _source_cache=None,
                       max_dimension=1000000, max_records=1000000, max_logical_size=None):
    """Validate a bounded source attachment and restore its owning publication."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral):
        raise TypeError("native source byte limit requires an integer")
    max_bytes = int(max_bytes)
    if not 0 <= max_bytes <= np.iinfo(np.uintp).max:
        raise ValueError("native source byte limit is outside its declared width")
    read_limits = []
    for value in (max_dimension, max_records, 0 if max_logical_size is None else max_logical_size):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("native source read limits require integers")
        value = int(value)
        if not 0 <= value <= np.iinfo(np.uintp).max:
            raise ValueError("native source read limit is outside its declared width")
        read_limits.append(value)
    max_dimension, max_records, max_logical_size = read_limits = tuple(read_limits)
    if (not isinstance(record, Mapping) or set(record) != {"kind", "version", "reference", "archive"}
            or record["kind"] != "cpp_source_archive" or type(record["version"]) is not int
            or record["version"] != 1):
        raise ValueError("unsupported native source archive attachment")
    selected = source_from_record(record["reference"], allow_cpp=True)
    if not isinstance(selected, CppSourceReference) or not selected.matches(reference):
        raise ValueError("native source attachment reference differs from the model")
    archive = np.asarray(record["archive"])
    if archive.dtype != np.uint8 or archive.ndim != 1 or archive.size > max_bytes:
        raise ValueError("invalid native source archive tensor or byte limit")
    raw = archive.tobytes()
    if sha256(raw).hexdigest() != reference.state_digest:
        raise ValueError("native source attachment digest differs")
    if _source_cache is not None and type(_source_cache) is not _SourceRestoreCache:
        raise TypeError("native source reuse requires an owning source cache")
    source = None if _source_cache is None else _source_cache.get(record, archive, raw, read_limits=read_limits)
    if source is None:
        source = _binding().NativeSource(raw, max_bytes=max_bytes, max_dimension=max_dimension,
                                       max_records=max_records, max_logical_size=max_logical_size)
    reference.bind(source)
    if _source_cache is not None:
        _source_cache.put(record, archive, source, read_limits=read_limits)
    enable_cpp_provider()
    return source
