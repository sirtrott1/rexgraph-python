"""Source bound sparse integer classification with exact checkpoint replay."""
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock

import numpy as np

from rexgraph.cpp_learning import _int64, _uint
from rexgraph.cpp_model import _binding, _limits, _space, enable_cpp_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference

_cache = OrderedDict()
_input_cache = OrderedDict()
_lock = RLock()
_input_keys = frozenset({"profile", "source", "primary_space", "feature_carrier", "feature_coordinates", "rows"})


def _remember(state, native):
    digest = state.coefficient_digest
    with _lock:
        _cache[digest] = native
        _cache.move_to_end(digest)
        while len(_cache) > 8:
            _cache.popitem(last=False)


def _payload(native, history=(), storage="values"):
    if storage == "packed":
        snapshot = native.packed_snapshot
        return {"parameters": np.asarray(snapshot["parameters"], dtype=np.int64),
                "mistakes": snapshot["mistakes"], "history": tuple(history)}
    snapshot = native.snapshot
    return {"parameters": (*snapshot["weights"], snapshot["bias"]),
            "mistakes": snapshot["mistakes"], "history": tuple(history)}


def cpp_perceptron_model(source, carrier, weights_key, bias_key, availability, *,
                         feature_coordinates, time_coordinates, grade=1,
                         initial_parameter_knowledge_time=0, limits=_limits, reference=None,
                         parameter_storage="values"):
    """Declare sparse feature coordinates and label availability on native source cells."""
    binding = _binding()
    if parameter_storage not in {"values", "packed"}:
        raise ValueError("integer parameter storage must be values or packed")
    if not isinstance(source, binding.NativeSource):
        raise TypeError("integer classification requires an owning NativeSource")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    if not isinstance(ref, CppSourceReference):
        raise TypeError("integer classification requires a C++ source reference")
    carrier, grade = _uint(carrier), _uint(grade, 16)
    selected = source.carrier(carrier)
    if selected["kind"] != "source_grade" or selected["grade"] != grade or not selected["size"]:
        raise ValueError("classifier rows require a nonempty declared SourceGrade carrier")
    limits = tuple(_uint(v) for v in limits)
    if len(limits) != 5 or selected["size"] > limits[0]:
        raise ValueError("integer classification row bound exceeded")
    weights_key, bias_key = _uint(weights_key), _uint(bias_key)
    weights, bias = source.field(weights_key), source.field(bias_key)
    features, scalar = source.carrier(weights.carrier), source.carrier(bias.carrier)
    feature_coordinates, time_coordinates = _uint(feature_coordinates), _uint(time_coordinates)
    if (features["kind"] != "explicit" or scalar["kind"] != "explicit"
            or features["coordinates"] != feature_coordinates or not feature_coordinates
            or not time_coordinates or scalar["size"] != 1 or not 0 < features["size"] <= limits[0]):
        raise ValueError("integer classifier requires explicit feature and scalar parameter carriers")
    if parameter_storage == "values":
        if weights.support != tuple(range(features["size"])) or bias.support != (0,):
            raise ValueError("integer parameters require full declared support")
        initial = tuple(_int64(v) for v in weights.values)
        b = _int64(bias.values[0])
        native = binding.NativePerceptron(initial, b, limits=limits)
    else:
        native = source.perceptron(weights_key, bias_key, limits=limits)
    availability = tuple(availability)
    if len(availability) != selected["size"]:
        raise ValueError("label availability must match the declared primary row order")
    times = []
    for item in availability:
        if not isinstance(item, Mapping) or set(item) != {
                "event_time", "knowledge_time", "label_available_time", "partition"}:
            raise ValueError("unsupported integer classifier availability declaration")
        if item["partition"] not in {"training", "evaluation"}:
            raise ValueError("unsupported integer classifier partition")
        times.append({**{name: _uint(item[name]) for name in (
            "event_time", "knowledge_time", "label_available_time")}, "partition": item["partition"]})
    config = {"execution_provider": {"name": "rexgraph.cpp", "version": "1"},
              "engine_receipt": binding.build_info(), "grade": grade, "carrier": carrier,
              "weights_key": weights_key, "bias_key": bias_key, "feature_carrier": features["key"],
              "feature_coordinates": feature_coordinates, "feature_count": features["size"],
              "time_coordinates": time_coordinates, "availability": tuple(times),
              "initial_parameter_knowledge_time": _uint(initial_parameter_knowledge_time),
              "limits": limits, "scalar_policy": "checked-int64",
              "training_profile": "ordered-sparse-integer-perceptron-v1"}
    if parameter_storage == "packed":
        config["parameter_storage"] = "packed"
    state = ModelState("cpp_perceptron", "1", ref, _space(ref, grade, carrier), (),
                       config, _payload(native, storage=parameter_storage), "rational")
    enable_cpp_provider()
    _remember(state, native)
    return state


def _rows(state, rows):
    if len(rows) != len(state.space.keys):
        raise ValueError("sparse feature rows must match the declared primary coordinates")
    result, count = [], 0
    for row in rows:
        count += len(row)
        if count > state.configuration["limits"][2]:
            raise ValueError("integer feature entry bound exceeded")
        entries = []
        for item in row:
            if len(item) != 2:
                raise ValueError("integer feature requires index and value")
            index = _uint(item[0], 32)
            if index >= state.configuration["feature_count"]:
                raise ValueError("integer feature lies outside its declared carrier")
            entries.append((index, _int64(item[1])))
        result.append(tuple(entries))
    return tuple(result)


def _cpu_policy(workers, rows_per_tile):
    workers, rows_per_tile = _uint(workers), _uint(rows_per_tile)
    if not 0 < workers <= 64 or not 0 < rows_per_tile <= 1000000:
        raise ValueError("CPU scoring requires 1 to 64 workers and 1 to 1000000 rows per tile")
    return {"workers": workers, "rows_per_tile": rows_per_tile}


def _input(state, rows, cpu_workers=None, rows_per_tile=256, *, normalized=False):
    if normalized and len(rows) != len(state.space.keys):
        raise ValueError("sparse feature rows must match the declared primary coordinates")
    result = {"profile": "sparse-int64-features-v1", "source": state.source.as_record(),
            "primary_space": (state.space.name, state.space.keys),
            "feature_carrier": state.configuration["feature_carrier"],
            "feature_coordinates": state.configuration["feature_coordinates"], "rows": rows if normalized else _rows(state, rows)}
    if cpu_workers is not None:
        result["cpu"] = _cpu_policy(cpu_workers, rows_per_tile)
    elif rows_per_tile != 256:
        raise ValueError("rows_per_tile requires an explicit cpu_workers count")
    return result


def cpp_perceptron_input(state, source, rows, *, cpu_workers=None, rows_per_tile=256, prepare=False):
    """Bind sparse integer features and optionally retain owned native rows."""
    if type(prepare) is not bool:
        raise TypeError("sparse input preparation must be boolean")
    native = prepare_perceptron(state, source)
    result = freeze_tree(_input(state, rows, cpu_workers, rows_per_tile))
    if prepare:
        _prepare_input_rows(native, result)
    return result


def _prepare_input_rows(native, inputs):
    owned = native.prepare_rows(inputs["rows"])
    with _lock:
        _input_cache[id(inputs["rows"])] = (inputs["rows"], owned)
        _input_cache.move_to_end(id(inputs["rows"]))
        while len(_input_cache) > 8:
            _input_cache.popitem(last=False)


def cpp_perceptron_batch(state, source, rows, labels, *, rate=1, event_cutoff,
                         knowledge_cutoff, observed=None):
    """Capture exact signed labels, sparse features and explicit causal cutoffs."""
    inputs = dict(cpp_perceptron_input(state, source, rows))
    inputs.update(rate=_int64(rate), event_cutoff=_uint(event_cutoff), knowledge_cutoff=_uint(knowledge_cutoff))
    targets = np.asarray(tuple(_int64(v) for v in labels), dtype=object)
    observed = np.ones(len(state.space.keys), dtype=bool) if observed is None else observed
    return ModelBatch(state.bind(source), state.space, targets, observed, inputs)


def _epoch(state, record, parameter_time):
    names = {"rows", "targets", "observed", "rate", "event_cutoff", "knowledge_cutoff"}
    if not isinstance(record, Mapping) or set(record) not in (names, names | {"coefficient_features"}):
        raise ValueError("unsupported integer epoch checkpoint")
    rows = _rows(state, record["rows"])
    if "coefficient_features" in record:
        from rexgraph.cpp_coefficient_features import validate_coefficient_features
        if validate_coefficient_features(record["coefficient_features"], state=state) != rows:
            raise ValueError("integer epoch differs from its retained coefficient features")
    labels = tuple(_int64(v) for v in record["targets"])
    observed = tuple(record["observed"])
    rate = _int64(record["rate"])
    events, knowledge = _uint(record["event_cutoff"]), _uint(record["knowledge_cutoff"])
    if (len(labels) != len(rows) or len(observed) != len(rows) or rate < 1
            or any(v not in {-1, 1} for v in labels) or any(type(v) is not bool for v in observed)
            or not any(observed) or parameter_time > knowledge):
        raise ValueError("invalid signed labels, observation mask, rate or parameter availability")
    chosen, targets = [], []
    for row, label, use, timing in zip(rows, labels, observed, state.configuration["availability"], strict=True):
        if not use:
            continue
        if (timing["partition"] != "training" or timing["event_time"] > events
                or max(timing["knowledge_time"], timing["label_available_time"]) > knowledge):
            raise ValueError("evaluation or future observations cannot update integer parameters")
        chosen.append(row); targets.append(label)
    return tuple(chosen), tuple(targets), rate, knowledge


def prepare_perceptron(state, source):
    """Validate the declaration and replay a bounded integer update history."""
    if state.adapter != "cpp_perceptron":
        raise TypeError("integer classifier requires a cpp_perceptron model")
    state.bind(source)
    c = state.configuration
    if dict(c["engine_receipt"]) != _binding().build_info():
        raise ValueError("model requires a different native semantic implementation")
    digest = state.coefficient_digest
    with _lock:
        cached = _cache.get(digest)
        if cached is not None:
            _cache.move_to_end(digest)
            return cached
    initial = cpp_perceptron_model(source, c["carrier"], c["weights_key"], c["bias_key"], c["availability"],
        feature_coordinates=c["feature_coordinates"], time_coordinates=c["time_coordinates"], grade=c["grade"],
        initial_parameter_knowledge_time=c["initial_parameter_knowledge_time"], limits=c["limits"], reference=state.source,
        parameter_storage=c.get("parameter_storage", "values"))
    offset = 0
    if "transport" in state.payload:
        from rexgraph.cpp_transport import restore_transport
        initial, offset = restore_transport(state, source)
    if (state.configuration != initial.configuration or state.space != initial.space
            or state.output_axes or state.dependencies or state.adapter_version != "1"
            or state.arithmetic != "rational"
            or ("transport" not in state.payload and (state.step == 0) != (state.parent is None))):
        raise ValueError("integer model differs from its declared native profile")
    history = state.payload["history"]
    if len(history) != state.step - offset or state.step > 1024:
        raise ValueError("integer checkpoint step or history bound differs")
    storage = c.get("parameter_storage", "values")
    native = (source.perceptron(c["weights_key"], c["bias_key"], limits=c["limits"])
              if storage == "packed" else _binding().NativePerceptron(
                  initial.payload["parameters"][:-1], initial.payload["parameters"][-1], limits=c["limits"]))
    parameter_time, entries = c["initial_parameter_knowledge_time"], 0
    for record in history:
        entries += sum(len(row) for row in record["rows"]) + len(state.space.keys)
        if entries > c["limits"][2]:
            raise ValueError("integer checkpoint history entry bound exceeded")
        rows, labels, rate, parameter_time = _epoch(state, record, parameter_time)
        native = native.advance(rows, labels, rate=rate)
    from rexgraph.cpp_transport import retained_payload
    expected = retained_payload(state, _payload(native, history, storage))
    actual = dict(state.payload)
    if storage == "packed":
        parameters = actual.pop("parameters", None)
        expected = dict(expected)
        values = expected.pop("parameters")
        if (not isinstance(parameters, np.ndarray) or parameters.dtype != np.dtype(np.int64)
                or not parameters.flags.c_contiguous or parameters.shape != values.shape
                or not np.array_equal(parameters, values)):
            raise ValueError("integer parameter bank differs from its replayed update history")
    if actual != expected:
        raise ValueError("integer parameters differ from their replayed update history")
    _remember(state, native)
    return native


def infer_perceptron(state, source, inputs):
    """Predict signed labels on declared source rows using exact sparse integer scores."""
    native = prepare_perceptron(state, source)
    expected = _input_keys
    if not isinstance(inputs, Mapping):
        raise ValueError("integer inference requires its declared sparse input profile")
    features = inputs.get("coefficient_features")
    if "coefficient_features" in inputs:
        from rexgraph.cpp_coefficient_features import validate_coefficient_features
        if validate_coefficient_features(features, state=state) != inputs["rows"]:
            raise ValueError("integer inference differs from its retained coefficient features")
        expected = expected | {"coefficient_features"}
    if set(inputs) not in (expected, expected | {"cpu"}):
        raise ValueError("integer inference requires its declared sparse input profile")
    workers, tile_rows = None, 256
    if "cpu" in inputs:
        cpu = inputs["cpu"]
        if not isinstance(cpu, Mapping) or set(cpu) != {"workers", "rows_per_tile"}:
            raise ValueError("unsupported CPU scoring policy")
        policy = _cpu_policy(cpu["workers"], cpu["rows_per_tile"])
        workers, tile_rows = policy["workers"], policy["rows_per_tile"]
    with _lock:
        cached = _input_cache.get(id(inputs["rows"]))
        prepared = cached[1] if cached is not None and cached[0] is inputs["rows"] else None
        if prepared is not None:
            _input_cache.move_to_end(id(inputs["rows"]))
    validated = _input(state, inputs["rows"], workers, tile_rows, normalized=prepared is not None)
    if features is not None:
        validated["coefficient_features"] = features
    if inputs != validated:
        raise ValueError("integer input uses another source, primary order or feature carrier")
    rows = validated["rows"] if prepared is None else prepared
    output = (native.forward(rows) if workers is None else
              native.forward(rows, cpu_workers=workers, rows_per_tile=tile_rows))
    c = state.configuration
    from rexgraph.cpp_transport import transport_receipt
    receipt = {"engine": _binding().build_info(), "scalar_policy": c["scalar_policy"],
               "implementation": "ExactIntegerPerceptron/checked-int64/sparse-serial-v1",
               "feature_carrier": c["feature_carrier"], "feature_coordinates": c["feature_coordinates"],
               "parameter_step": state.step, "parameter_history_step": len(state.payload["history"]),
               "scores": output["scores"], "limits": c["limits"],
               "time_coordinates": c["time_coordinates"],
               "parameter_knowledge_time": state.payload["history"][-1]["knowledge_cutoff"]
                    if state.payload["history"] else c["initial_parameter_knowledge_time"],
               "namespace": state.source.native.namespace, "sequence": state.source.native.sequence}
    receipt.update(transport_receipt(state))
    if "cpu_execution" in output:
        receipt.update(implementation="ExactIntegerPerceptron/checked-int64/sparse-cpu-tiles-v1",
                       cpu_execution=output["cpu_execution"])
    if c.get("parameter_storage") == "packed":
        receipt["parameter_storage"] = "packed"
    if features is not None:
        receipt["coefficient_features"] = features
    return ModelOutput(output["values"], state.space, (), state.source.bind(source), state.coefficient_digest,
                       "rational", "checked-int64-perceptron-v1", grade=c["grade"], variance="coordinate",
                       execution_receipt=receipt)


def train_perceptron(state, source, batch, *, steps):
    """Return an immutable child after complete ordered integer training epochs."""
    native = prepare_perceptron(state, source)
    if state.step + steps > 1024:
        raise ValueError("integer training retained epoch bound exceeded")
    supplied = batch.inputs
    expected_keys = {*_input_keys, "rate", "event_cutoff", "knowledge_cutoff"}
    if isinstance(supplied, Mapping) and "coefficient_features" in supplied:
        expected_keys.add("coefficient_features")
    if (not isinstance(supplied, Mapping) or set(supplied) != expected_keys
            or batch.targets.shape != (len(state.space.keys),) or batch.dependencies or batch.target_input is not None):
        raise ValueError("integer training requires declared sparse observations")
    validated = _input(state, supplied["rows"])
    if "coefficient_features" in supplied:
        from rexgraph.cpp_coefficient_features import validate_coefficient_features
        if validate_coefficient_features(supplied["coefficient_features"], state=state) != validated["rows"]:
            raise ValueError("integer training differs from its retained coefficient features")
        validated["coefficient_features"] = supplied["coefficient_features"]
    if {k: v for k, v in supplied.items() if k not in {"rate", "event_cutoff", "knowledge_cutoff"}} != validated:
        raise ValueError("integer batch uses another source or feature coordinate order")
    rows = validated["rows"]
    record = {"rows": rows, "targets": tuple(_int64(v) for v in batch.targets),
              "observed": tuple(bool(v) for v in batch.observed), "rate": _int64(supplied["rate"]),
              "event_cutoff": _uint(supplied["event_cutoff"]), "knowledge_cutoff": _uint(supplied["knowledge_cutoff"])}
    if "coefficient_features" in supplied:
        record["coefficient_features"] = supplied["coefficient_features"]
    history = tuple(state.payload["history"])
    entries = sum(sum(len(row) for row in r["rows"]) + len(state.space.keys) for r in (*history, *(record,) * steps))
    if entries > state.configuration["limits"][2]:
        raise ValueError("integer checkpoint history entry bound exceeded")
    if "coefficient_features" in record or any("coefficient_features" in item for item in history):
        from rexgraph.cpp_coefficient_features import validate_coefficient_feature_history
        validate_coefficient_feature_history(state, (*history, *(record,) * steps))
    parameter_time = history[-1]["knowledge_cutoff"] if history else state.configuration["initial_parameter_knowledge_time"]
    for _ in range(steps):
        selected, labels, rate, parameter_time = _epoch(state, record, parameter_time)
        native = native.advance(selected, labels, rate=rate)
        history = (*history, record)
    from rexgraph.cpp_transport import remember_transport_child, retained_payload
    child = state.child(payload=retained_payload(state, _payload(
        native, history, state.configuration.get("parameter_storage", "values"))), step=state.step + steps)
    remember_transport_child(state, child)
    _remember(child, native)
    return child
