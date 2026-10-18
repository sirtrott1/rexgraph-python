"""Certified native bijections and retained model transport evidence."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

import numpy as np

from rexgraph.cpp_learning import _uint
from rexgraph.cpp_model import _binding, _create, _limits, _prepare
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_state import ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.value_codec import pack_value

_profile = "native-bijection-transport-v1"
_proof_limit = 32 * 1024 * 1024
_keys = {"profile", "source", "destination", "engine_receipt", "cell_maps", "carrier_links",
         "tensor_links", "configuration", "limits", "time_policy"}
_cache = OrderedDict()
_lock = RLock()


def cpp_transport(source, destination, cell_maps, carrier_links, tensor_links, *,
                  configuration, limits=_limits, source_reference=None, destination_reference=None):
    """Declare endpoint identities, complete grade bijections and destination factory arguments."""
    return _declare(source, destination, cell_maps, carrier_links, tensor_links,
        configuration=configuration, limits=limits, source_reference=source_reference,
        destination_reference=destination_reference)[0]


def _declare(source, destination, cell_maps, carrier_links, tensor_links, *,
             configuration, limits, source_reference, destination_reference):
    src = CppSourceReference(source) if source_reference is None else source_reference.bind(source)
    dst = CppSourceReference(destination) if destination_reference is None else destination_reference.bind(destination)
    if not isinstance(src, CppSourceReference) or not isinstance(dst, CppSourceReference):
        raise TypeError("native transport requires C++ publication references")
    bounds = tuple(_uint(v) for v in limits)
    native = source.transport(destination, cell_maps, carrier_links, limits=bounds)
    links = tuple({"source": _uint(v["source"]), "target": _uint(v["target"]),
                   "indices": tuple(_uint(i, 32) for i in v["indices"])} for v in carrier_links)
    tensors = []
    for item in tensor_links:
        if not isinstance(item, Mapping) or set(item) != {"source", "target"}:
            raise ValueError("transport tensor link requires source and target")
        tensors.append({"source": _uint(item["source"]), "target": _uint(item["target"])})
    if len(tensors) > bounds[0] or len({v["source"] for v in tensors}) != len(tensors) or len({v["target"] for v in tensors}) != len(tensors):
        raise ValueError("transport tensor links must be bounded and distinct")
    if not isinstance(configuration, Mapping) or "reference" in configuration or "adapter" in configuration:
        raise ValueError("transport requires destination factory configuration")
    return freeze_tree({"profile": _profile, "source": src.as_record(), "destination": dst.as_record(),
        "engine_receipt": _binding().build_info(), "cell_maps": tuple(tuple(_uint(i, 32) for i in row) for row in cell_maps),
        "carrier_links": links, "tensor_links": tuple(tensors), "configuration": dict(configuration),
        "limits": bounds, "time_policy": "shared-coordinate-times-v1"}), native


def _mapping(record, source, destination, src, dst):
    if not isinstance(record, Mapping) or set(record) != _keys or record["profile"] != _profile:
        raise ValueError("unsupported native transport declaration")
    a = source_from_record(record["source"], allow_cpp=True)
    b = source_from_record(record["destination"], allow_cpp=True)
    if not a.matches(src) or not b.matches(dst):
        raise ValueError("transport uses another source or destination publication")
    expected, native = _declare(source, destination, record["cell_maps"], record["carrier_links"],
        record["tensor_links"], configuration=record["configuration"], limits=record["limits"],
        source_reference=src, destination_reference=dst)
    if record != expected:
        raise ValueError("transport differs from its exact declaration or compiled receipt")
    return native


def _permuted(values, indices):
    out = [None] * len(values)
    for i, j in enumerate(indices):
        out[j] = values[i]
    return tuple(out)


def _base(origin, source, destination, dst, mapping, native, prepared):
    if "transport" in origin.payload:
        raise NotImplementedError("this native profile permits one transport per model lineage")
    base = _create(destination, adapter=origin.adapter, reference=dst, **mapping["configuration"])
    c = thaw_tree(origin.configuration)
    carriers = {v["source"]: v for v in mapping["carrier_links"]}
    tensors = {v["source"]: v["target"] for v in mapping["tensor_links"]}
    def carrier(key):
        if key not in carriers:
            raise ValueError("model transport is missing a selected carrier link")
        return carriers[key]["target"]
    def tensor(key):
        if key not in tensors:
            raise ValueError("model transport is missing a selected tensor link")
        return tensors[key]
    parameters = {}
    if origin.adapter in {"cpp_field", "cpp_metric"}:
        c["grade_carriers"] = tuple(carrier(key) for key in c["grade_carriers"])
        c["metric_keys"] = tuple(tensor(key) for key in c["metric_keys"])
    else:
        c["carrier"] = carrier(c["carrier"])
    if origin.adapter == "cpp_perceptron":
        old = origin.configuration
        weights = old["weights_key"]; bias = old["bias_key"]
        if weights == bias:
            raise ValueError("transport requires distinct weight and bias fields")
        c.update(weights_key=tensor(weights), bias_key=tensor(bias), feature_carrier=carrier(old["feature_carrier"]),
                 feature_coordinates=destination.carrier(carrier(old["feature_carrier"]))["coordinates"],
                 availability=_permuted(old["availability"], mapping["cell_maps"][old["grade"]]),
                 initial_parameter_knowledge_time=origin.payload["history"][-1]["knowledge_cutoff"]
                    if origin.payload["history"] else old["initial_parameter_knowledge_time"])
        parameters[weights] = origin.payload["parameters"][:-1]
        parameters[bias] = origin.payload["parameters"][-1:]
    elif origin.adapter == "cpp_metric":
        old = origin.configuration["declaration"]
        d = c["declaration"]
        for name in ("gains", "bias"):
            d[name] = tensor(old[name])
        gain_carrier = source.field(old["gains"]).carrier
        carrier(gain_carrier)
        gain_link = carriers[gain_carrier]
        d["channels"] = _permuted(old["channels"], gain_link["indices"])
        d["channel_coordinates"] = destination.carrier(gain_link["target"])["coordinates"]
        d["bias_coordinates"] = destination.carrier(carrier(source.field(old["bias"]).carrier))["coordinates"]
        d["initial_parameter_knowledge_time"] = prepared.parameter_knowledge_time
        supplied = base.configuration["declaration"]
        if supplied["history_namespace"] == old["history_namespace"]:
            raise ValueError("transport requires a separate destination parameter history namespace")
        for name in ("history_namespace", "metadata_carrier", "metadata_tensor", "metadata_coordinates", "publication_cutoff"):
            d[name] = supplied[name]
        d.pop("metadata_capacity", None)
        if "metadata_capacity" in supplied:
            d["metadata_capacity"] = supplied["metadata_capacity"]
        offset = origin.step % len(old["samples"])
        samples = (*old["samples"][offset:], *old["samples"][:offset])
        d["samples"] = tuple({**dict(item), "input": tensor(item["input"]), "target": tensor(item["target"])} for item in samples)
        parameters[old["gains"]] = origin.payload["parameters"][:-1]
        parameters[old["bias"]] = origin.payload["parameters"][-1:]
    if base.configuration != freeze_tree(c):
        raise ValueError("destination changes the declared model, time coordinates or sample continuation")
    for key, value in parameters.items():
        field = source.field(key)
        result = native.push_values(field.carrier, value)
        target = destination.field(tensor(key))
        if (target.carrier != result["output_carrier"] or target.support != tuple(range(len(value)))
                or tuple(target.values) != result["values"]):
            raise ValueError("destination fields differ from transported current parameters")
    for key, target in tensors.items():
        if key not in parameters:
            native.check_tensor(key, target)
    return base


def _proof_bytes(tensors, archive):
    if not isinstance(archive, bytes) or not isinstance(tensors, Mapping):
        raise ValueError("transport requires retained origin bytes and model tensors")
    count = len(archive)
    for name, value in tensors.items():
        if (not isinstance(value, np.ndarray) or value.ndim != 1 or not value.flags.c_contiguous
                or not (value.dtype == np.uint8 or (
                    value.dtype == np.dtype(np.int64) and isinstance(name, str)
                    and name.startswith("tensor/") and name[7:].isdecimal()))):
            raise ValueError("transport origin requires native model byte or packed int64 tensors")
        count += value.nbytes
    if count > _proof_limit:
        raise ValueError("native transport retained proof byte bound exceeded")


def transport_cpp_model(state, source, destination, mapping, *, optimizer="reset"):
    """Return a certified child retaining the original source and complete checkpoint."""
    if optimizer != "reset":
        raise ValueError("native transport declares no optimizer covariance carry")
    if not isinstance(destination, CppSourceReference):
        raise TypeError("native transport requires a live CppSourceReference destination")
    if "transport" in state.payload:
        raise NotImplementedError("this native profile permits one transport per model lineage")
    prepared = _prepare(state, source)
    target = destination.source
    native = _mapping(mapping, source, target, state.source, destination)
    base = _base(state, source, target, destination, mapping, native, prepared)
    tensors = pack_model(state)
    _proof_bytes(tensors, source.archive)
    proof = {"profile": _profile, "origin_model": tensors, "origin_archive": source.archive,
             "mapping": mapping, "base_step": state.step, "origin_digest": state.coefficient_digest}
    child = state.child(payload={**dict(base.payload), "transport": proof}, configuration=base.configuration,
                        source=base.source, space=base.space, dependencies=base.dependencies)
    _prepare(child, target)
    return child


def restore_transport(state, source):
    """Replay the original checkpoint and verify the declared destination initialization."""
    state.bind(source)
    if dict(state.configuration["engine_receipt"]) != _binding().build_info():
        raise ValueError("transport requires its compiled semantic implementation")
    digest = state.coefficient_digest
    with _lock:
        found = _cache.get(digest)
        if found is not None:
            _cache.move_to_end(digest)
            return found
    proof = state.payload["transport"]
    keys = {"profile", "origin_model", "origin_archive", "mapping", "base_step", "origin_digest"}
    if not isinstance(proof, Mapping) or set(proof) != keys or proof["profile"] != _profile:
        raise ValueError("unsupported native model transport proof")
    _proof_bytes(proof["origin_model"], proof["origin_archive"])
    origin = unpack_model(proof["origin_model"])
    if (not isinstance(origin, ModelState) or "transport" in origin.payload or origin.adapter != state.adapter
            or origin.coefficient_digest != proof["origin_digest"] or origin.step != _uint(proof["base_step"])):
        raise ValueError("transport origin checkpoint or learning step differs")
    old = _binding().NativeSource(proof["origin_archive"], max_bytes=_proof_limit)
    prepared = _prepare(origin, old)
    native = _mapping(proof["mapping"], old, source, origin.source, state.source)
    base = _base(origin, old, source, state.source, proof["mapping"], native, prepared)
    if (state.configuration != base.configuration or state.space != base.space or state.output_axes
            or state.dependencies or state.adapter_version != "1" or state.arithmetic != "rational"
            or not origin.step <= state.step <= 1024 or state.parent is None
            or (state.step == origin.step and state.parent != origin.coefficient_digest)):
        raise ValueError("transported model differs from its declared initialization")
    result = (base, origin.step)
    _remember(state, result)
    return result


def _remember(state, result):
    digest = state.coefficient_digest
    with _lock:
        _cache[digest] = result
        _cache.move_to_end(digest)
        while len(_cache) > 8:
            _cache.popitem(last=False)


def remember_transport_child(state, child):
    """Reuse certified destination initialization after a functional native update."""
    digest = state.coefficient_digest
    with _lock:
        found = _cache.get(digest)
    if found is not None:
        _remember(child, found)


def retained_payload(state, payload):
    """Keep validated transport evidence through subsequent local training updates."""
    if "transport" in state.payload:
        return {**payload, "transport": state.payload["transport"]}
    return payload


def transport_receipt(state):
    """Describe the certified handoff and its cumulative learning offset."""
    if "transport" not in state.payload:
        return {}
    proof = state.payload["transport"]
    return {"transport_profile": _profile, "transport_base_step": proof["base_step"],
            "transport_origin_model": proof["origin_digest"],
            "transport_map_digest": sha256(pack_value(thaw_tree(proof["mapping"]))).hexdigest()}
