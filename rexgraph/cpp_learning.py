"""Integer networks and bounded rational training on owning C++ sources."""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from numbers import Integral

import numpy as np

from rexgraph.cpp_model import _binding, _limits, _space, cpp_model, enable_cpp_provider
from rexgraph.exact_value import exact_fraction
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value


def _uint(value, bits=64):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 1 << bits:
        raise ValueError(f"native identifier requires an unsigned {bits} bit integer")
    return int(value)


def _int64(value):
    if type(value) is not int:
        value = exact_fraction(value)
        if value.denominator != 1:
            raise TypeError("integer network coefficients must be integral")
        value = value.numerator
    if not -(1 << 63) <= value < 1 << 63:
        raise OverflowError("integer coefficient is outside signed 64 bit storage")
    return value


def cpp_integer_model(source, carrier, layers, *, grade=1, limits=_limits, reference=None):
    """Declare ordered integral lattice channels with checked int64 arithmetic."""
    binding = _binding()
    if not isinstance(source, binding.NativeSource):
        raise TypeError("integer network requires an owning NativeSource")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    if not isinstance(ref, CppSourceReference):
        raise TypeError("integer model requires a C++ source reference")
    grade, carrier = _uint(grade, 16), _uint(carrier)
    declarations = []
    for layer in layers:
        if not isinstance(layer, Mapping) or set(layer) != {"channels", "gains", "bias", "activation"}:
            raise ValueError("integer layer requires channels, gains, bias and activation")
        declarations.append({"channels": tuple(layer["channels"]),
                             "gains": tuple(_int64(v) for v in layer["gains"]),
                             "bias": _int64(layer["bias"]), "activation": layer["activation"]})
    declarations, limits = tuple(declarations), tuple(limits)
    source.integer_network(carrier, declarations, grade=grade, limits=limits)
    config = {"execution_provider": {"name": "rexgraph.cpp", "version": "1"},
              "engine_receipt": binding.build_info(), "grade": grade, "carrier": carrier,
              "layers": declarations, "limits": limits,
              "scalar_policy": "checked-int64", "channel_profile": "integral-lattice",
              "input_profile": "source-bound-int64-coordinates-v1",
              "batch_input_profile": "packed-int64-lattice-batch-v1"}
    state = ModelState("cpp_integer", "1", ref, _space(ref, grade, carrier), (), config, {}, "rational")
    enable_cpp_provider()
    return state


def cpp_integer_input(state, source, values):
    """Bind integral input coordinates to the integer model's source and carrier."""
    if state.adapter != "cpp_integer":
        raise TypeError("integer input requires a cpp_integer model")
    prepare_learning_model(state, source)
    values = tuple(_int64(value) for value in values)
    source.integer_input(state.configuration["carrier"], values, grade=state.configuration["grade"])
    return TensorField(state.space, values, source=state.bind(source),
                       grade=state.configuration["grade"], variance="coordinate")


def cpp_integer_batch_input(state, source, values, samples):
    """Capture contiguous native int64 coefficients on source cells and named sample columns."""
    if state.adapter != "cpp_integer":
        raise TypeError("integer batch input requires a cpp_integer model")
    prepare_learning_model(state, source)
    if not isinstance(samples, CoordinateSpace):
        raise TypeError("integer batch requires a named sample CoordinateSpace")
    return _integer_batch_record(state, source, {
        "profile": "packed-int64-lattice-batch-v1", "source": state.source.as_record(),
        "samples": {"name": samples.name, "keys": samples.keys}, "values": np.asarray(values),
    })


def _integer_batch_record(state, source, record):
    if (not isinstance(record, Mapping) or set(record) != {"profile", "source", "samples", "values"}
            or record["profile"] != "packed-int64-lattice-batch-v1"):
        raise ValueError("unsupported packed integer batch declaration")
    ref = source_from_record(record["source"], allow_cpp=True)
    if not isinstance(ref, CppSourceReference) or not ref.matches(state.bind(source)):
        raise ValueError("integer batch uses another source publication")
    axis = record["samples"]
    if not isinstance(axis, Mapping) or set(axis) != {"name", "keys"}:
        raise ValueError("integer batch requires named ordered sample coordinates")
    samples = CoordinateSpace(axis["name"], axis["keys"])
    values = record["values"]
    if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("int64")
            or not values.flags.c_contiguous or values.ndim != 2):
        raise TypeError("integer batch requires a contiguous native int64 matrix")
    if (not len(state.space.keys) or not samples.keys or samples.name == state.space.name
            or values.shape != (len(state.space.keys), len(samples.keys))):
        raise ValueError("integer batch dimensions differ from the declared cell and sample axes")
    if values.size > min(state.configuration["limits"][:3]):
        raise ValueError("integer batch coefficient bound exceeded")
    return freeze_tree({**dict(record), "samples": {"name": samples.name, "keys": samples.keys}})


def _training_samples(value):
    samples = []
    names = ("input", "target", "event_time", "knowledge_time", "label_available_time")
    for sample in value:
        if not isinstance(sample, Mapping) or set(sample) != {*names, "partition"}:
            raise ValueError("unsupported native training sample declaration")
        samples.append({**{name: _uint(sample[name]) for name in names}, "partition": sample["partition"]})
    return tuple(samples)


def _declaration(value):
    names = ("gains", "bias", "channel_coordinates", "bias_coordinates", "history_namespace",
             "metadata_carrier", "metadata_tensor", "time_coordinates", "metadata_coordinates",
             "initial_parameter_knowledge_time", "publication_cutoff")
    expected = {*names, "grade", "channels", "samples", "model"}
    if not isinstance(value, Mapping) or set(value) not in (expected, expected | {"metadata_capacity"}):
        raise ValueError("unsupported native training declaration")
    result = {name: _uint(value[name]) for name in names}
    if "metadata_capacity" in value:
        result["metadata_capacity"] = _uint(value["metadata_capacity"])
    result.update(grade=_uint(value["grade"], 16), channels=tuple(value["channels"]), model=value["model"])
    result["samples"] = _training_samples(value["samples"])
    return result


def _snapshot(trainer):
    return {key: value for key, value in trainer.snapshot.items() if key != "step"}


def cpp_trainable_model(source, grade_carriers, declaration, *, metric_keys=(),
                        metric_selection="unit", limits=_limits, reference=None, training_cache_samples=8):
    """Declare ordered metric channels and an exact squared error training contract.

    Initial gains and bias are full support fields on explicit native carriers.
    Samples retain their declared order, partition and availability times.
    The native contract bounds training to 1024 samples and 1024 updates, with
    a declared metadata_capacity from 8 through 65536 bytes, default 4096.
    Complete checkpoint metadata must fit that capacity at every update.
    Fixed metric pairings are cached for at most training_cache_samples samples.
    Repeated channels share cached responses. When both Hodge sectors are
    declared, the Hodge channel uses their sum. Every original gain and bias
    retains its own exact gradient and checkpoint coordinate.
    Zero selects the direct field gradient executor.
    """
    if metric_selection not in {"unit", "diagonal"}:
        raise ValueError("exact training supports the finite unit or diagonal metric profile")
    declaration = _declaration(declaration)
    base = cpp_model(source, grade_carriers, grade=declaration["grade"], operation="hodge",
                     metric_keys=metric_keys, metric_selection=metric_selection, limits=limits,
                     reference=reference)
    c = base.configuration
    plan = source.prepare(c["grade_carriers"], metric_keys=c["metric_keys"],
                          metric_selection=c["metric_selection"], limits=c["limits"])
    training_cache_samples = _uint(training_cache_samples, 16)
    trainer = source.trainer(plan, declaration, cache_samples=training_cache_samples)
    config = {"execution_provider": c["execution_provider"], "engine_receipt": c["engine_receipt"],
              "grade_carriers": c["grade_carriers"], "metric_keys": c["metric_keys"],
              "metric_selection": c["metric_selection"], "limits": c["limits"],
              "declaration": declaration, "objective": "half-metric-squared-error",
              "input_profile": "stored-native-field-v1", "training_profile": "native-exact-trainer-v1",
              "training_cache_samples": training_cache_samples,
              "batch_input_profile": "stored-native-field-batch-v1"}
    return ModelState("cpp_metric", "1", base.source, base.space, (), config, _snapshot(trainer), "rational")


def prepare_learning_model(state, source):
    """Validate the declaration and restore or reuse its owning native execution state."""
    binding = _binding()
    state.bind(source)
    c = state.configuration
    if dict(c["engine_receipt"]) != binding.build_info():
        raise ValueError("model requires a different native semantic implementation")
    if state.adapter == "cpp_integer":
        expected = cpp_integer_model(source, c["carrier"], c["layers"], grade=c["grade"],
                                     limits=c["limits"], reference=state.source)
        if "transport" in state.payload:
            from rexgraph.cpp_transport import restore_transport, retained_payload
            expected, offset = restore_transport(state, source)
            if state.step != offset or state.payload != retained_payload(state, dict(expected.payload)):
                raise ValueError("transported integer model differs from its declared fixed network")
        elif expected.coefficient_digest != state.coefficient_digest:
            raise ValueError("integer model differs from its declared fixed network")
        return source.integer_network(c["carrier"], c["layers"], grade=c["grade"], limits=c["limits"])
    if state.adapter != "cpp_metric":
        raise ValueError("unsupported native learning adapter")
    expected = cpp_trainable_model(source, c["grade_carriers"], c["declaration"],
                                   metric_keys=c["metric_keys"], metric_selection=c["metric_selection"],
                                   limits=c["limits"], reference=state.source,
                                   training_cache_samples=c["training_cache_samples"])
    offset = 0
    if "transport" in state.payload:
        from rexgraph.cpp_transport import restore_transport
        expected, offset = restore_transport(state, source)
    if (state.configuration != expected.configuration or state.space != expected.space
            or state.adapter_version != "1" or state.arithmetic != "rational"
            or state.output_axes or state.dependencies
            or ("transport" not in state.payload and (state.step == 0) != (state.parent is None))):
        raise ValueError("trainable model differs from its declared native contract")
    plan = source.prepare(c["grade_carriers"], metric_keys=c["metric_keys"],
                          metric_selection=c["metric_selection"], limits=c["limits"])
    trainer = source.trainer(plan, c["declaration"], archive=state.payload["archive"],
                             cache_samples=c["training_cache_samples"])
    from rexgraph.cpp_transport import retained_payload
    if state.step != offset + trainer.snapshot["step"] or state.payload != retained_payload(state, _snapshot(trainer)):
        raise ValueError("model parameters or step differ from the validated native history")
    return trainer


def infer_learning_model(state, source, inputs):
    """Evaluate the declared integer network or the current exact metric parameters."""
    binding = _binding()
    native = prepare_learning_model(state, source)
    c = state.configuration
    if state.adapter == "cpp_integer":
        if isinstance(inputs, Mapping):
            record = _integer_batch_record(state, source, inputs)
            batch = source.integer_batch_input(c["carrier"], record["values"], grade=c["grade"], limits=c["limits"])
            result = native.forward_batch(batch)
            samples = CoordinateSpace(record["samples"]["name"], record["samples"]["keys"])
            receipt = {key: value for key, value in result.items()
                       if key not in {"values", "workspace_capacity", "workspace_slots"}}
            receipt.update(engine=binding.build_info(), limits=c["limits"], layers=c["layers"],
                           scalar_policy=c["scalar_policy"], channel_profile=c["channel_profile"],
                           input_digest=sha256(pack_value(thaw_tree(record))).hexdigest())
            from rexgraph.cpp_transport import transport_receipt
            receipt.update(transport_receipt(state))
            return ModelOutput(result["values"], state.space, (samples,), state.source.bind(source), state.coefficient_digest,
                               "rational", result["profile"], grade=result["grade"], variance="coordinate",
                               execution_receipt=receipt)
        if isinstance(inputs, TensorField):
            inputs.check_state()
            if (inputs.source is None or not inputs.source.matches(state.source)
                    or inputs.space != state.space or inputs.axes or inputs.dependencies
                    or inputs.grade != c["grade"] or inputs.variance != "coordinate"):
                raise ValueError("integer input differs from the declared source coordinates")
            inputs = source.integer_input(c["carrier"], tuple(_int64(v) for v in inputs.values), grade=c["grade"])
        if not isinstance(inputs, binding.NativeIntegerInput):
            raise TypeError("integer inference requires bound integral coordinates")
        result = native.forward(inputs)
        additions = {"layers": c["layers"], "scalar_policy": c["scalar_policy"],
                     "channel_profile": c["channel_profile"], "input_values": inputs.values}
        variance = "coordinate"
    else:
        additions = {"channels": c["declaration"]["channels"], "objective": c["objective"],
                     "metric_keys": c["metric_keys"], "metric_selection": c["metric_selection"],
                     "parameter_history_namespace": state.payload["history_namespace"],
                     "parameter_step": state.step, "parameter_history_step": native.snapshot["step"],
                     "parameter_knowledge_time": native.parameter_knowledge_time,
                     "parameter_archive_digest": sha256(state.payload["archive"]).hexdigest()}
        if isinstance(inputs, Mapping):
            if "field_key" not in inputs:
                from rexgraph.cpp_model import _rational_batch_output, _rational_batch_record
                record = _rational_batch_record(state, source, inputs)
                result = native.forward_batch(tuple(source.field(key) for key in record["field_keys"]))
                return _rational_batch_output(state, source, record, result, additions)
            if set(inputs) != {"field_key"}:
                raise ValueError("native input record requires only field_key")
            inputs = source.field(_uint(inputs["field_key"]))
        if not isinstance(inputs, binding.NativeField):
            raise TypeError("metric inference requires a stored NativeField or field_key record")
        result = native.forward(inputs)
        variance = "chain"
    receipt = {key: value for key, value in result.items() if key != "values"}
    receipt.update(additions, engine=binding.build_info(), limits=c["limits"])
    from rexgraph.cpp_transport import transport_receipt
    receipt.update(transport_receipt(state))
    return ModelOutput(result["values"], state.space, (), state.source.bind(source), state.coefficient_digest,
                       "rational", result["profile"], grade=result["grade"], variance=variance,
                       execution_receipt=receipt)


def _columns(state, source, name):
    return np.asarray([source.field(sample[name]).values
                       for sample in state.configuration["declaration"]["samples"]], dtype=object).T


def _training_inputs(state, source, *, rate, event_cutoff, knowledge_cutoff, capture):
    inputs = {"profile": "stored-native-training-samples-v1" if capture == "stored" else "native-exact-trainer-v1",
              "samples": state.configuration["declaration"]["samples"],
              "rate": exact_fraction(rate), "event_cutoff": _uint(event_cutoff),
              "knowledge_cutoff": _uint(knowledge_cutoff)}
    if capture == "values":
        inputs["values"] = tuple(tuple(row) for row in _columns(state, source, "input"))
    return inputs


def cpp_training_batch(state, source, *, rate, event_cutoff, knowledge_cutoff, capture="stored"):
    """Capture ordered stored sample references, an exact rate and causal cutoffs.

    The default stored profile retains field keys and availability on the batch's
    exact source publication. Targets stay in the owning native source; the
    batch's target array has source rows and zero columns. Use capture="values"
    to copy every declared input and target coefficient into the batch instead.
    Both profiles require complete observed rows and the declared sample order.
    """
    if state.adapter != "cpp_metric":
        raise TypeError("exact training batch requires a cpp_metric model")
    if capture not in {"stored", "values"}:
        raise ValueError("training capture must be stored or values")
    prepare_learning_model(state, source)
    inputs = _training_inputs(state, source, rate=rate, event_cutoff=event_cutoff,
                              knowledge_cutoff=knowledge_cutoff, capture=capture)
    targets = (_columns(state, source, "target") if capture == "values" else
               np.empty((len(state.space.keys), 0), dtype=object))
    return ModelBatch(state.bind(source), state.space, targets,
                      np.ones(len(state.space.keys), dtype=bool), inputs)


def train_learning_model(state, source, batch, *, steps):
    """Return a child after all requested native updates complete successfully."""
    native = prepare_learning_model(state, source)
    if state.step + steps > 1024:
        raise ValueError("exact training cumulative step bound exceeded")
    supplied = batch.inputs
    profiles = {"stored-native-training-samples-v1": "stored", "native-exact-trainer-v1": "values"}
    if (not isinstance(supplied, Mapping) or not isinstance(supplied.get("profile"), str)
            or supplied["profile"] not in profiles):
        raise ValueError("exact training requires its declared batch profile")
    capture = profiles[supplied["profile"]]
    names = {"profile", "samples", "rate", "event_cutoff", "knowledge_cutoff"}
    if set(supplied) != (names | {"values"} if capture == "values" else names):
        raise ValueError("exact training requires its declared batch profile")
    if not np.all(batch.observed) or batch.dependencies or batch.target_input is not None:
        raise ValueError("training batch differs from the declared exact observations")
    expected = _training_inputs(state, source, rate=supplied["rate"],
                                event_cutoff=supplied["event_cutoff"], knowledge_cutoff=supplied["knowledge_cutoff"],
                                capture=capture)
    if _training_samples(supplied["samples"]) != expected["samples"]:
        raise ValueError("training batch differs from the declared exact observations")
    if capture == "stored":
        if batch.targets.dtype != np.dtype(object) or batch.targets.shape != (len(state.space.keys), 0):
            raise ValueError("stored training targets must remain in the declared source fields")
    else:
        targets = _columns(state, source, "target")
        values = tuple(tuple(exact_fraction(v) for v in row) for row in supplied["values"])
        if (values != expected["values"] or batch.targets.shape != targets.shape
                or tuple(exact_fraction(v) for v in batch.targets.flat) != tuple(targets.flat)):
            raise ValueError("training batch differs from the declared exact observations")
    candidate = native.advance(expected["rate"], event_cutoff=expected["event_cutoff"],
                               knowledge_cutoff=expected["knowledge_cutoff"], steps=steps)
    source.remember_trainer(candidate)
    from rexgraph.cpp_transport import remember_transport_child, retained_payload
    child = state.child(payload=retained_payload(state, _snapshot(candidate)), step=state.step + steps)
    remember_transport_child(state, child)
    return child
