"""Exact packed Boolean fitting on declared source rows and parameter states."""
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock

import numpy as np

from rexgraph.cpp_learning import _uint
from rexgraph.cpp_model import _binding, _space
from rexgraph.exact_value import exact_fraction
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, _sha
from rexgraph.value_codec import pack_value

_PROFILE = "packed-boolean-input-v1"
_METHOD = "packed-boolean-gate-v1"
_INPUT_KEYS = {"profile", "source", "primary_space", "channels", "lanes", "first", "second"}
_CONFIG_KEYS = {"execution_provider", "engine_receipt", "grade", "carrier", "parameter_key",
    "parameter_carrier", "parameter_coordinates", "parameter_states", "input_channels", "availability",
    "time_coordinates", "initial_parameter_knowledge_time", "max_lanes", "max_history_bytes",
    "objective", "tie_policy", "input_profile", "training_profile"}
_STATES = ((0, 0), (0, 1), (1, 0), (1, 1))
_provider = None
_lock = RLock()
_cache = OrderedDict()
_input_cache = OrderedDict()


def _bit(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    q = exact_fraction(value)
    if q not in (0, 1):
        raise ValueError("Boolean coefficients must be exactly zero or one")
    return int(q)


def pack_binary_plane(values):
    """Pack ordered Boolean coefficients with the first lane in the least significant bit."""
    values = tuple(_bit(v) for v in values)
    return np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little").tobytes()


def _check_plane(raw, lanes):
    if type(raw) is not bytes or len(raw) != (lanes + 7) // 8:
        raise ValueError("packed Boolean plane differs from its declared lane count")
    if lanes % 8 and raw[-1] >> (lanes % 8):
        raise ValueError("packed Boolean plane has nonzero tail padding")


def _plane(raw, lanes):
    _check_plane(raw, lanes)
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little", count=lanes)


def _engine(record):
    names = {"engine", "version", "engine_digest", "binding_digest", "profile"}
    if not isinstance(record, Mapping) or set(record) != names or record["engine"] != "rexgraph-cpp" or record["profile"] != "binary-truth-gate-v1":
        raise ValueError("unsupported packed Boolean engine receipt")
    if not isinstance(record["version"], str) or not record["version"]:
        raise ValueError("packed Boolean engine version is absent")
    _sha(record["engine_digest"]); _sha(record["binding_digest"])


def _names(value):
    if isinstance(value, (str, bytes)):
        raise ValueError("Boolean input requires two declared channel names")
    names = tuple(value)
    if len(names) != 2 or any(not isinstance(v, str) or not v or len(v) > 256 for v in names) or names[0] == names[1]:
        raise ValueError("Boolean input requires two distinct declared channel names")
    return names


def _availability(values, count):
    result = []
    for item in values:
        if not isinstance(item, Mapping) or set(item) != {"event_time", "knowledge_time", "label_available_time", "partition"}:
            raise ValueError("unsupported Boolean label availability declaration")
        if item["partition"] not in {"training", "evaluation"}:
            raise ValueError("unknown Boolean sample partition")
        result.append({**{key: _uint(item[key]) for key in ("event_time", "knowledge_time", "label_available_time")},
                       "partition": item["partition"]})
    if len(result) != count:
        raise ValueError("Boolean availability differs from the declared source row order")
    return tuple(result)


def _input(state, record):
    if not isinstance(record, Mapping) or set(record) != _INPUT_KEYS or record["profile"] != _PROFILE:
        raise ValueError("unsupported packed Boolean input declaration")
    c = state.configuration; count = len(state.space.keys)
    if (record["source"] != state.source.as_record() or record["primary_space"] != (state.space.name, state.space.keys)
            or record["channels"] != c["input_channels"] or _uint(record["lanes"]) != count):
        raise ValueError("Boolean input differs from its source rows or declared channels")
    _check_plane(record["first"], count); _check_plane(record["second"], count)
    return record


def _epoch(state, entry, parameter_time):
    names = {"inputs", "targets", "observed", "event_cutoff", "knowledge_cutoff"}
    if not isinstance(entry, Mapping) or set(entry) != names:
        raise ValueError("unsupported packed Boolean fit checkpoint")
    inputs = _input(state, entry["inputs"]); count = len(state.space.keys)
    _check_plane(entry["targets"], count)
    observed = entry["observed"]
    event, knowledge = _uint(entry["event_cutoff"]), _uint(entry["knowledge_cutoff"])
    if (not isinstance(observed, tuple) or len(observed) != count or any(type(v) is not bool for v in observed)
            or not any(observed) or parameter_time > knowledge):
        raise ValueError("Boolean fit requires observed labels and available current parameters")
    for use, timing in zip(observed, state.configuration["availability"], strict=True):
        if use and (timing["partition"] != "training" or timing["event_time"] > event
                    or max(timing["knowledge_time"], timing["label_available_time"]) > knowledge):
            raise ValueError("evaluation or future Boolean labels cannot fit parameters")
    if all(observed):
        return inputs["first"], inputs["second"], entry["targets"], count, knowledge
    a, b, targets = _plane(inputs["first"], count), _plane(inputs["second"], count), _plane(entry["targets"], count)
    mask = np.asarray(observed, dtype=bool)
    return pack_binary_plane(a[mask]), pack_binary_plane(b[mask]), pack_binary_plane(targets[mask]), sum(observed), knowledge


def validate_binary_gate_state(state):
    """Check portable source, parameter, observation and checkpoint declarations."""
    c = state.configuration
    if (state.adapter != "cpp_binary_gate" or state.adapter_version != "1" or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or state.output_axes or state.dependencies
            or set(c) != _CONFIG_KEYS or dict(c["execution_provider"]) != {"name": "rexgraph.cpp.binary", "version": "1"}):
        raise ValueError("unsupported packed Boolean model declaration")
    _engine(c["engine_receipt"])
    grade, carrier = _uint(c["grade"], 16), _uint(c["carrier"])
    if grade >= len(state.source.native.coordinates) or state.space != _space(state.source, grade, carrier) or not state.space.keys:
        raise ValueError("Boolean model rows differ from its original native grade")
    for key in ("parameter_key", "parameter_carrier", "parameter_coordinates", "time_coordinates", "initial_parameter_knowledge_time"):
        _uint(c[key])
    if not c["parameter_coordinates"] or not c["time_coordinates"] or c["parameter_states"] != _STATES:
        raise ValueError("Boolean parameters require the four ordered input states and explicit coordinates")
    _names(c["input_channels"]); _availability(c["availability"], len(state.space.keys))
    if (not len(state.space.keys) <= _uint(c["max_lanes"]) <= 64000000
            or not 0 < _uint(c["max_history_bytes"]) <= 256 << 20
            or c["objective"] != "hamming-count" or c["tie_policy"] != "lowest-truth-table"
            or c["input_profile"] != _PROFILE or c["training_profile"] != "canonical-binary-gate-fit-v1"):
        raise ValueError("unsupported Boolean fitting policy or admission")
    p = state.payload
    if set(p) != {"truth_table", "last_hamming_error", "history"} or _uint(p["truth_table"]) > 15:
        raise ValueError("Boolean parameters must be one declared four-bit truth table")
    history = p["history"]
    if not isinstance(history, tuple) or len(history) != state.step or state.step > 1024 or (state.step == 0) != (state.parent is None):
        raise ValueError("Boolean checkpoint step or parent differs from its fit history")
    if len(pack_value(thaw_tree(history))) > c["max_history_bytes"]:
        raise ValueError("Boolean fit history byte admission exceeded")
    parameter_time = c["initial_parameter_knowledge_time"]
    for entry in history:
        *_, count, parameter_time = _epoch(state, entry, parameter_time)
    if (not history and p["last_hamming_error"] is not None
            or history and not 0 <= _uint(p["last_hamming_error"]) <= count):
        raise ValueError("Boolean checkpoint loss differs from its observed label count")


def cpp_binary_gate_model(source, carrier, parameter_key, availability, *, parameter_coordinates,
                          time_coordinates, input_channels=("a", "b"), grade=1,
                          initial_parameter_knowledge_time=0, max_lanes=1048576,
                          max_history_bytes=16 << 20, reference=None):
    """Declare original source rows and four Boolean parameter states for exact fitting."""
    binding = _binding()
    if not isinstance(source, binding.NativeSource):
        raise TypeError("Boolean fitting requires an owning NativeSource")
    reference = CppSourceReference(source) if reference is None else reference.bind(source)
    carrier, parameter_key, grade = _uint(carrier), _uint(parameter_key), _uint(grade, 16)
    primary = source.carrier(carrier); parameter = source.field(parameter_key); space = source.carrier(parameter.carrier)
    if primary["kind"] != "source_grade" or primary["grade"] != grade or not primary["size"]:
        raise ValueError("Boolean model requires nonempty original SourceGrade rows")
    if (space["kind"] != "explicit" or space["size"] != 4 or space["coordinates"] != _uint(parameter_coordinates)
            or parameter.support != (0, 1, 2, 3)):
        raise ValueError("Boolean parameters require full support on four explicit input states")
    table = sum(_bit(value) << index for index, value in enumerate(parameter.values))
    config = {"execution_provider": {"name": "rexgraph.cpp.binary", "version": "1"},
        "engine_receipt": binding.binary_build_info(), "grade": grade, "carrier": carrier,
        "parameter_key": parameter_key, "parameter_carrier": parameter.carrier, "parameter_coordinates": _uint(parameter_coordinates),
        "parameter_states": _STATES, "input_channels": _names(input_channels),
        "availability": _availability(availability, primary["size"]), "time_coordinates": _uint(time_coordinates),
        "initial_parameter_knowledge_time": _uint(initial_parameter_knowledge_time), "max_lanes": _uint(max_lanes),
        "max_history_bytes": _uint(max_history_bytes), "objective": "hamming-count", "tie_policy": "lowest-truth-table",
        "input_profile": _PROFILE, "training_profile": "canonical-binary-gate-fit-v1"}
    state = ModelState("cpp_binary_gate", "1", reference, _space(reference, grade, carrier), (), config,
                       {"truth_table": table, "last_hamming_error": None, "history": ()}, "rational")
    enable_cpp_binary_gate_provider()
    _remember(state, binding.NativeBinaryGate(table))
    return state


def _remember(state, native):
    with _lock:
        _cache[state.coefficient_digest] = native
        _cache.move_to_end(state.coefficient_digest)
        while len(_cache) > 8: _cache.popitem(last=False)


def prepare_binary_gate(state, source):
    """Check the source contract and replay retained fits through the canonical kernel."""
    state.bind(source)
    validate_binary_gate_state(state)
    binding = _binding(); c = state.configuration
    if dict(c["engine_receipt"]) != binding.binary_build_info():
        raise ValueError("Boolean model requires a different native semantic implementation")
    with _lock:
        cached = _cache.get(state.coefficient_digest)
        if cached is not None:
            _cache.move_to_end(state.coefficient_digest)
            return cached
    initial = cpp_binary_gate_model(source, c["carrier"], c["parameter_key"], c["availability"],
        parameter_coordinates=c["parameter_coordinates"], time_coordinates=c["time_coordinates"],
        input_channels=c["input_channels"], grade=c["grade"], initial_parameter_knowledge_time=c["initial_parameter_knowledge_time"],
        max_lanes=c["max_lanes"], max_history_bytes=c["max_history_bytes"], reference=state.source)
    if state.configuration != initial.configuration:
        raise ValueError("Boolean model differs from its original source parameter declarations")
    gate = binding.NativeBinaryGate(initial.payload["truth_table"]); loss = None
    time = c["initial_parameter_knowledge_time"]
    for entry in state.payload["history"]:
        a, b, target, count, time = _epoch(state, entry, time)
        gate, loss = binding.fit_binary_gate(binding.NativeBinaryInput(a, b, count, max_lanes=c["max_lanes"]), target)
    if gate.truth_table != state.payload["truth_table"] or loss != state.payload["last_hamming_error"]:
        raise ValueError("Boolean parameters or loss differ from their replayed fitting history")
    _remember(state, gate)
    return gate


def cpp_binary_gate_input(state, source, first, second, *, prepare=False):
    """Bind canonical packed planes to every declared original source row."""
    if type(prepare) is not bool: raise TypeError("Boolean input preparation must be boolean")
    state.bind(source)
    validate_binary_gate_state(state)
    record = freeze_tree({"profile": _PROFILE, "source": state.source.as_record(),
        "primary_space": (state.space.name, state.space.keys), "channels": state.configuration["input_channels"],
        "lanes": len(state.space.keys), "first": first, "second": second})
    _input(state, record)
    if prepare:
        native = _binding().NativeBinaryInput(first, second, len(state.space.keys), max_lanes=state.configuration["max_lanes"])
        with _lock:
            _input_cache[id(record)] = (record, native)
            _input_cache.move_to_end(id(record))
            while len(_input_cache) > 8: _input_cache.popitem(last=False)
    return record


def cpp_binary_gate_batch(state, source, first, second, targets, *, event_cutoff, knowledge_cutoff, observed=None):
    """Capture exact Boolean labels and declared learner event and knowledge cutoffs."""
    inputs = dict(cpp_binary_gate_input(state, source, first, second))
    inputs.update(event_cutoff=_uint(event_cutoff), knowledge_cutoff=_uint(knowledge_cutoff))
    labels = _plane(targets, len(state.space.keys)).astype(object)
    observed = np.ones(len(state.space.keys), dtype=bool) if observed is None else observed
    return ModelBatch(state.bind(source), state.space, labels, observed, inputs)


def infer_binary_gate(state, source, inputs):
    """Apply the current native truth table on the complete declared source grade."""
    gate = prepare_binary_gate(state, source); record = _input(state, inputs); c = state.configuration
    with _lock:
        cached = _input_cache.get(id(inputs))
        native = cached[1] if cached is not None and cached[0] is inputs else None
        if native is not None: _input_cache.move_to_end(id(inputs))
    if native is None:
        native = _binding().NativeBinaryInput(record["first"], record["second"], record["lanes"], max_lanes=c["max_lanes"])
    raw = gate.forward(native)
    receipt = {"profile": _METHOD, "scalar_policy": "packed-boolean", "engine": c["engine_receipt"],
        "input": record, "truth_table": gate.truth_table, "parameter_carrier": c["parameter_carrier"],
        "parameter_coordinates": c["parameter_coordinates"], "parameter_states": _STATES,
        "parameter_step": state.step, "time_coordinates": c["time_coordinates"],
        "parameter_knowledge_time": state.payload["history"][-1]["knowledge_cutoff"]
            if state.payload["history"] else c["initial_parameter_knowledge_time"],
        "grade": c["grade"], "carrier": c["carrier"], "output_plane": raw, "lanes": native.lanes, "word_count": native.word_count}
    return ModelOutput(_plane(raw, native.lanes), state.space, (), state.source.bind(source), state.coefficient_digest,
                       "rational", _METHOD, grade=c["grade"], execution_receipt=receipt)


def validate_binary_gate_output(output):
    """Check retained packed outputs, original rows and exact truth table coordinates."""
    r = output.execution_receipt
    names = {"profile", "scalar_policy", "engine", "input", "truth_table", "parameter_carrier", "parameter_coordinates",
             "parameter_states", "parameter_step", "time_coordinates", "parameter_knowledge_time", "grade", "carrier", "output_plane", "lanes", "word_count"}
    if (not isinstance(r, Mapping) or set(r) != names or r["profile"] != _METHOD or r["scalar_policy"] != "packed-boolean"
            or output.method != _METHOD or output.arithmetic != "rational" or output.axes or output.dependencies
            or output.variance != "coordinate" or not isinstance(output.source, CppSourceReference)):
        raise ValueError("unsupported packed Boolean output declaration")
    _engine(r["engine"])
    count = len(output.space.keys); record = r["input"]
    if (not count or not isinstance(record, Mapping) or set(record) != _INPUT_KEYS or record["profile"] != _PROFILE or record["source"] != output.source.as_record()
            or record["primary_space"] != (output.space.name, output.space.keys) or _uint(record["lanes"]) != count
            or _uint(r["lanes"]) != count or _uint(r["word_count"]) != (count + 63) // 64
            or output.grade != _uint(r["grade"], 16) or output.grade >= len(output.source.native.coordinates)
            or output.space != _space(output.source, output.grade, _uint(r["carrier"])) or r["parameter_states"] != _STATES):
        raise ValueError("Boolean output differs from its original source or parameter states")
    _names(record["channels"])
    for key in ("parameter_carrier", "parameter_coordinates", "parameter_step", "time_coordinates", "parameter_knowledge_time"):
        _uint(r[key])
    if not r["parameter_coordinates"] or not r["time_coordinates"] or r["parameter_step"] > 1024:
        raise ValueError("Boolean output parameter or time declarations are invalid")
    table = _uint(r["truth_table"])
    if table > 15: raise ValueError("binary truth table must fit four bits")
    a, b, values = _plane(record["first"], count), _plane(record["second"], count), _plane(r["output_plane"], count)
    expected = ((table >> (2 * a + b)) & 1).astype(np.uint8)
    if not np.array_equal(values, expected) or not np.array_equal(output.values, values):
        raise ValueError("Boolean output differs from its retained input planes and truth table")


def train_binary_gate(state, source, batch, *, steps):
    """Return an immutable child after bounded canonical truth table fitting."""
    prepare_binary_gate(state, source)
    if state.step + steps > 1024: raise ValueError("Boolean retained fit count exceeded")
    if batch.dependencies or batch.target_input is not None or batch.targets.shape != (len(state.space.keys),):
        raise ValueError("Boolean fit requires one label on each declared source row")
    supplied = batch.inputs
    if not isinstance(supplied, Mapping) or set(supplied) != _INPUT_KEYS | {"event_cutoff", "knowledge_cutoff"}:
        raise ValueError("unsupported Boolean fit input declaration")
    record = freeze_tree({key: supplied[key] for key in _INPUT_KEYS})
    entry = freeze_tree({"inputs": record, "targets": pack_binary_plane(batch.targets),
        "observed": tuple(bool(v) for v in batch.observed), "event_cutoff": _uint(supplied["event_cutoff"]),
        "knowledge_cutoff": _uint(supplied["knowledge_cutoff"])})
    history = (*state.payload["history"], *(entry,) * steps)
    if len(pack_value(thaw_tree(history))) > state.configuration["max_history_bytes"]:
        raise ValueError("Boolean fit history byte admission exceeded")
    time = state.payload["history"][-1]["knowledge_cutoff"] if state.step else state.configuration["initial_parameter_knowledge_time"]
    a, b, target, count, _ = _epoch(state, entry, time)
    binding = _binding()
    native = binding.NativeBinaryInput(a, b, count, max_lanes=state.configuration["max_lanes"])
    gate, loss = binding.fit_binary_gate(native, target)
    child = state.child(payload={"truth_table": gate.truth_table, "last_hamming_error": loss, "history": history}, step=state.step + steps)
    _remember(child, gate)
    return child


def enable_cpp_binary_gate_provider():
    """Register canonical Boolean fitting with explicit exact model contracts."""
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            def create(source, *args, adapter="cpp_binary_gate", **kwargs):
                if adapter != "cpp_binary_gate": raise ValueError("unsupported Boolean model adapter")
                return cpp_binary_gate_model(source, *args, **kwargs)
            _provider = ModelProvider("rexgraph.cpp.binary", "1", (("cpp_binary_gate", "1"),),
                infer=infer_binary_gate, create=create, prepare=prepare_binary_gate, train=train_binary_gate)
            register_model_provider(_provider)
        return _provider
