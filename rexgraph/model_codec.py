"""Finite model state encoding without executable objects or pickle."""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from threading import RLock
from types import MappingProxyType

import numpy as np

from rexgraph.model_state import (
    ModelBatch,
    ModelInput,
    ModelOutput,
    ModelState,
    ModelTimeline,
    _metadata_cache_limit,
    _primitive_map_key,
)
from rexgraph.source_binding import (
    CppAlgebraicSourceReference,
    CppSourceReference,
    source_from_record,
)
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_spec, unpack_spec

_storage = sys.modules.get("rexgraph_cpp._model_storage")
_PayloadEncoder = None if _storage is None else _storage.PayloadEncoder
_PayloadDecoder = None if _storage is None else _storage.PayloadDecoder
_StorageTraversalLimit = None if _storage is None else _storage.StorageTraversalLimit

MODEL_VALUES = (ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput)
_checkpoint_lock = RLock()
_checkpoint_cache = OrderedDict()
_checkpoint_limit = 4


@dataclass(frozen=True)
class _Checkpoint:
    witness: object
    value: object
    tensors: object = None


def _checkpoint_key(tensors):
    if not isinstance(tensors, Mapping):
        return None
    encoded = tensors.get("model_digest")
    if type(encoded) is not np.ndarray or encoded.dtype != np.uint8 or encoded.shape != (64,):
        return None
    return encoded.tobytes()


def _remember_checkpoint(key, entry):
    _checkpoint_cache[key] = entry
    _checkpoint_cache.move_to_end(key)
    while len(_checkpoint_cache) > _checkpoint_limit:
        _checkpoint_cache.popitem(last=False)


def _checkpoint_model(tensors):
    """Reuse a decoded owner only after checking the complete checkpoint content."""
    from rexgraph.model_integrity import capture_content
    key = _checkpoint_key(tensors)
    with _checkpoint_lock:
        entry = _checkpoint_cache.get(key)
        if entry is not None and entry.witness.matches(tensors):
            entry.value.check_state()
            _checkpoint_cache.move_to_end(key)
            return entry.value
    witness = capture_content(tensors)
    value = unpack_model(tensors)
    if witness is not None:
        if not witness.matches(tensors):
            raise ValueError("model checkpoint changed during decoding")
        if key is not None:
            with _checkpoint_lock:
                _remember_checkpoint(key, _Checkpoint(witness, value))
    return value


def _model_checkpoint(value):
    """Reuse a checked, detached checkpoint for retained execution receipts."""
    from rexgraph.model_integrity import capture_content
    from rexgraph.model_state import freeze_tree
    value.check_state()
    key = value._digest.encode()
    with _checkpoint_lock:
        entry = _checkpoint_cache.get(key)
        if entry is not None and entry.value is value and entry.tensors is not None:
            if not entry.witness.matches(entry.tensors):
                raise ValueError("retained model checkpoint changed")
            _checkpoint_cache.move_to_end(key)
            return entry.tensors
        tensors = freeze_tree(pack_model(value))
        witness = capture_content(tensors)
        if witness is not None:
            _remember_checkpoint(key, _Checkpoint(witness, value, tensors))
        return tensors


def model_payload(value, *, native=False):
    return _model_payload(value, native=native)


def _model_payload(value, *, native=False, include_digest=False):
    if _PayloadEncoder is None:
        return _encode_model_payload(value, native=native, include_digest=include_digest, compiled=False)
    try:
        return _encode_model_payload(value, native=native, include_digest=include_digest, compiled=True)
    except _StorageTraversalLimit:
        return _encode_model_payload(value, native=native, include_digest=include_digest, compiled=False)


def _encode_model_payload(value, *, native=False, include_digest=False, compiled=False):
    tensors = {}
    cpp_sources = []
    rationals = {}
    def put(v):
        if isinstance(v, ModelInput):
            return ["input", put(v.original), put(v.values), [v.space.name, list(v.space.keys)],
                    [[a.name, list(a.keys)] for a in v.axes], source(v.source),
                    v.origin_digest, v.origin_arithmetic, [source(d) for d in v.dependencies]]
        if isinstance(v, np.ndarray):
            name = f"tensor/{len(tensors)}"
            a = np.array(v, copy=True, order="C")
            stored = a.astype(np.uint8) if a.dtype == bool else a
            tensors[name] = stored.reshape(1) if a.ndim == 0 else stored
            return ["array", name, a.dtype.str, list(a.shape)]
        if v is None: return ["none"]
        if isinstance(v, bool): return ["bool", v]
        if isinstance(v, str): return ["str", v]
        if isinstance(v, bytes):
            name = f"tensor/{len(tensors)}"
            tensors[name] = np.frombuffer(v, dtype=np.uint8).copy()
            return ["bytes", name]
        if isinstance(v, int): return ["int", hex(v)]
        if isinstance(v, Fraction):
            parts = v.numerator, v.denominator
            declaration = rationals.get(parts)
            if declaration is None:
                declaration = ["rational", hex(parts[0]), hex(parts[1])]
                if len(rationals) < _metadata_cache_limit:
                    rationals[parts] = declaration
            return declaration
        if isinstance(v, float): return ["float", v.hex()]
        if isinstance(v, Mapping):
            items = sorted(v.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0])))
            return ["map", [[put(k), put(x)] for k, x in items]]
        if isinstance(v, (tuple, list)):
            if len(v) > 1 and type(v[0]) is MappingProxyType:
                owners, repeated = {}, set()
                for x in v:
                    if type(x) is MappingProxyType:
                        identity = id(x)
                        if identity in owners:
                            repeated.add(identity)
                        elif len(owners) < _metadata_cache_limit:
                            owners[identity] = x
                shared = {identity for identity in repeated
                          if _primitive_map_key(owners[identity]) is not None}
                if shared:
                    declarations, items = {}, []
                    for x in v:
                        identity = id(x)
                        if identity not in shared:
                            items.append(put(x))
                        else:
                            if identity not in declarations:
                                declarations[identity] = put(x)
                            items.append(declarations[identity])
                    return ["tuple", items]
            return ["tuple", [put(x) for x in v]]
        raise TypeError("unsupported model payload value")
    if compiled:
        encoder = _PayloadEncoder(_metadata_cache_limit)
        tensors, cpp_sources, put = encoder.tensors, encoder.cpp_sources, encoder.put
    def space(v): return [v.name, list(v.keys)]
    def source(v):
        if isinstance(v, (CppSourceReference, CppAlgebraicSourceReference)): cpp_sources.append(v)
        return v.as_record()
    if isinstance(value, ModelState):
        spec = {"kind": "model", "adapter": value.adapter, "adapter_version": value.adapter_version,
                "source": source(value.source), "space": space(value.space),
                "axes": [space(v) for v in value.output_axes], "configuration": put(value.configuration),
                "payload": put(value.payload), "arithmetic": value.arithmetic, "step": value.step,
                "parent": value.parent, "dependencies": [source(v) for v in value.dependencies]}
    elif isinstance(value, ModelOutput):
        spec = {"kind": "output", "source": source(value.source), "space": space(value.space),
                "axes": [space(v) for v in value.axes], "values": put(value.values),
                "model_digest": value.model_digest, "arithmetic": value.arithmetic, "method": value.method,
                "grade": value.grade, "variance": value.variance,
                "dependencies": [source(v) for v in value.dependencies]}
        if value.execution_receipt is not None:
            spec["execution_receipt"] = put(value.execution_receipt)
        if value.field_archive is not None:
            spec["field_archive"] = put(value.field_archive)
    elif isinstance(value, ModelInput):
        spec = {"kind": "input", "value": put(value)}
    elif isinstance(value, ModelBatch):
        spec = {"kind": "batch", "source": source(value.source), "space": space(value.space),
                "targets": put(value.targets), "observed": put(value.observed), "inputs": put(value.inputs),
                "target_input": put(value.target_input),
                "dependencies": [source(v) for v in value.dependencies]}
    elif isinstance(value, ModelTimeline):
        spec = {"kind": "timeline", "times": put(value.times), "model_digests": list(value.model_digests),
                "source_digests": list(value.source_digests), "time_axis": value.time_axis, "time_unit": value.time_unit,
                "temporal_header": put(value.temporal_header),
                "temporal_tensors": put(value.temporal_tensors)}
    else:
        raise TypeError("unsupported native model object")
    spec["version"] = 3 if "field_archive" in spec else 2 if cpp_sources or "execution_receipt" in spec else 1
    declared = {*tensors, "model_digest"} if include_digest else tensors
    tensors["spec"] = pack_spec(spec, declared, native=native)
    return tensors


def pack_model(value, *, native=False):
    value.check_state()
    tensors = _model_payload(value, native=native, include_digest=native)
    tensors["model_digest"] = np.frombuffer(value.coefficient_digest.encode(), dtype=np.uint8).copy()
    return tensors


def unpack_model(tensors):
    if _PayloadDecoder is None:
        return _decode_model(tensors, compiled=False)
    try:
        return _decode_model(tensors, compiled=True)
    except _StorageTraversalLimit:
        return _decode_model(tensors, compiled=False)


def _decode_model(tensors, *, compiled=False):
    used = {"spec", "model_digest"}
    def take(v):
        if not isinstance(v, list) or not v: raise ValueError("invalid model value")
        tag = v[0]
        if tag == "input" and len(v) == 9:
            return ModelInput(take(v[1]), take(v[2]), space(v[3]), tuple(space(a) for a in v[4]),
                              source(v[5]), v[6], v[7], tuple(source(a) for a in v[8]))
        if (tag == "array" and len(v) == 4) or (tag == "bytes" and len(v) == 2):
            used.add(v[1])
            a = np.asarray(tensors[v[1]])
            if tag == "bytes":
                if a.dtype != np.uint8 or a.ndim != 1: raise ValueError("invalid model bytes")
                return a.tobytes()
            dtype, shape = np.dtype(v[2]), tuple(v[3])
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
                raise ValueError("invalid model tensor shape")
            if dtype == np.dtype(bool):
                if a.dtype != np.uint8 or not np.isin(a, (0, 1)).all():
                    raise ValueError("invalid model boolean tensor")
                a = a.astype(bool)
            elif a.dtype != dtype:
                raise ValueError("model tensor dtype differs")
            return a.reshape(shape)
        if tag == "none" and len(v) == 1: return None
        if tag == "bool" and len(v) == 2 and isinstance(v[1], bool): return v[1]
        if tag == "str" and len(v) == 2 and isinstance(v[1], str): return v[1]
        if tag == "int" and len(v) == 2: return int(v[1], 16)
        if tag == "rational" and len(v) == 3: return Fraction(int(v[1], 16), int(v[2], 16))
        if tag == "float" and len(v) == 2: return float.fromhex(v[1])
        if tag == "tuple" and len(v) == 2: return tuple(take(x) for x in v[1])
        if tag == "map" and len(v) == 2:
            out = {}
            for key, item in v[1]:
                key = take(key)
                if key in out: raise ValueError("duplicate model mapping key")
                out[key] = take(item)
            return out
        raise ValueError("unsupported model value tag")
    def source(v): return source_from_record(v, allow_cpp=spec["version"] in {2, 3})
    def space(v): return CoordinateSpace(v[0], tuple(v[1]))
    try:
        raw = np.asarray(tensors["spec"])
        if raw.dtype != np.uint8 or raw.ndim != 1: raise ValueError("invalid model specification")
        spec = unpack_spec(tensors)
        if spec.get("version") not in {1, 2, 3}: raise ValueError("unsupported model schema")
        if spec["version"] == 3:
            if spec.get("kind") != "output" or "field_archive" not in spec:
                raise ValueError("model schema 3 requires a whole-field output archive")
        elif "field_archive" in spec:
            raise ValueError("field archive requires model schema 3")
        if spec["version"] == 1 and "execution_receipt" in spec:
            raise ValueError("execution receipt requires model schema 2")
        if compiled:
            take = _PayloadDecoder(tensors, used, source, space).take
        kind = spec["kind"]
        if kind == "model":
            result = ModelState(spec["adapter"], spec["adapter_version"], source(spec["source"]), space(spec["space"]),
                tuple(space(v) for v in spec["axes"]), take(spec["configuration"]), take(spec["payload"]),
                spec["arithmetic"], spec["step"], spec["parent"], tuple(source(v) for v in spec["dependencies"]))
        elif kind == "output":
            archive = take(spec["field_archive"]) if spec["version"] == 3 else None
            if spec["version"] == 3 and archive is None:
                raise ValueError("model schema 3 requires a whole-field output archive")
            result = ModelOutput(take(spec["values"]), space(spec["space"]), tuple(space(v) for v in spec["axes"]),
                source(spec["source"]), spec["model_digest"], spec["arithmetic"], spec["method"],
                tuple(source(v) for v in spec["dependencies"]), spec["grade"], spec["variance"],
                take(spec["execution_receipt"]) if "execution_receipt" in spec else None,
                archive)
        elif kind == "input":
            result = take(spec["value"])
        elif kind == "batch":
            result = ModelBatch(source(spec["source"]), space(spec["space"]), take(spec["targets"]),
                take(spec["observed"]), take(spec["inputs"]), tuple(source(v) for v in spec["dependencies"]), take(spec["target_input"]))
        elif kind == "timeline":
            header = take(spec["temporal_header"])
            def lists(v):
                if isinstance(v, dict): return {k: lists(x) for k, x in v.items()}
                if isinstance(v, tuple): return [lists(x) for x in v]
                return v
            result = ModelTimeline(take(spec["times"]), tuple(spec["model_digests"]), tuple(spec["source_digests"]),
                                   lists(header), take(spec["temporal_tensors"]), spec["time_axis"], spec["time_unit"])
        else: raise ValueError("unknown model object kind")
        encoded_digest = np.asarray(tensors["model_digest"])
        if encoded_digest.dtype != np.uint8 or encoded_digest.ndim != 1: raise ValueError("invalid model seal")
        if set(tensors) != used: raise ValueError("unclaimed model tensor")
        if result.coefficient_digest != encoded_digest.tobytes().decode():
            raise ValueError("model identity mismatch")
        return result
    except (KeyError, UnicodeError, TypeError, IndexError, json.JSONDecodeError, ZeroDivisionError) as exc:
        raise ValueError("invalid model tensor payload") from exc
