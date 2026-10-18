"""Bounded content witnesses for sealed model declarations."""
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import sys

import numpy as np

_MAX_NODES = 262144
_MAX_BYTES = 32*1024*1024
_ATOMS = {type(None), bool, int, str, bytes}

_storage = sys.modules.get("rexgraph_cpp._model_storage")
_native_capture = None if _storage is None else _storage.capture_content
_native_freeze = None if _storage is None else _storage.freeze_model_tree
NativeContentWitness = None if _storage is None else _storage.NativeContentWitness


def _content_matches(value, tree, ordered=True):
    if NativeContentWitness is not None and type(tree) is NativeContentWitness:
        return tree.matches(value, ordered=ordered)
    return _matches(value, tree, {}, ordered)


def _content_tree(value):
    if _native_capture is not None:
        return _native_capture(value, _MAX_NODES, _MAX_BYTES)
    return _Capture().node(value)


def _fraction_parts(value):
    return type(value.numerator), int(value.numerator), type(value.denominator), int(value.denominator)


def _fractions_match(values, parts):
    return all(type(value) is Fraction and type(value.numerator) is old[0] and value.numerator == old[1] and
               type(value.denominator) is old[2] and value.denominator == old[3]
               for value, old in zip(values, parts, strict=True))


class _Unavailable(Exception):
    pass


def _fields(value):
    from rexgraph.model_state import ModelState, ModelOutput, ModelInput, ModelBatch, ModelTimeline
    declarations = (
        (ModelState, ("adapter", "adapter_version", "source", "space", "output_axes", "configuration",
                      "payload", "arithmetic", "step", "parent", "dependencies")),
        (ModelOutput, ("source", "space", "axes", "values", "model_digest", "arithmetic", "method", "grade",
                       "variance", "dependencies", "execution_receipt", "field_archive")),
        (ModelInput, ("original", "values", "space", "axes", "source", "origin_digest", "origin_arithmetic", "dependencies")),
        (ModelBatch, ("source", "space", "targets", "observed", "inputs", "target_input", "dependencies")),
        (ModelTimeline, ("times", "model_digests", "source_digests", "time_axis", "time_unit", "temporal_header", "temporal_tensors")),
    )
    for kind, names in declarations:
        if isinstance(value, kind):
            return kind, tuple(getattr(value, name) for name in names)
    raise _Unavailable


class _Capture:
    def __init__(self):
        self.nodes = 0
        self.bytes = 0
        self.owners = {}

    def charge(self, count=1, size=0):
        self.nodes += count; self.bytes += size
        if self.nodes > _MAX_NODES or self.bytes > _MAX_BYTES:
            raise _Unavailable

    def node(self, value):
        self.charge()
        kind = type(value)
        if kind in _ATOMS:
            return ("atom", value)
        if kind is float:
            return ("float", value.hex())
        if kind is Fraction:
            return ("fraction", type(value.numerator), int(value.numerator), type(value.denominator), int(value.denominator))
        if isinstance(value, np.integer):
            return ("integer", kind, int(value))
        known = self.owners.get(id(value))
        if known is not None and known[0] is value:
            return known[1]
        result = self.container(value)
        self.owners[id(value)] = (value, result)
        return result

    def container(self, value):
        from rexgraph.model_state import ModelInput
        from rexgraph.tensor_field import FieldSource
        from rexgraph.type_accession import CoordinateSpace
        if type(value) is np.ndarray:
            if value.dtype.hasobject:
                if value.size > _MAX_NODES - self.nodes:
                    raise _Unavailable
                if all(type(v) is Fraction for v in value.flat):
                    self.charge(count=value.size)
                    return ("fraction-array", value.dtype.str, value.shape, tuple(_fraction_parts(v) for v in value.flat))
                return ("exact-array", value.dtype.str, value.shape, tuple(self.node(v) for v in value.flat))
            self.charge(size=value.nbytes)
            return ("array", value.dtype.str, value.shape, value.tobytes(order="C"))
        if isinstance(value, Mapping):
            if any(type(k) not in {str, int} for k in value):
                raise _Unavailable
            return ("map", tuple((k, self.node(v)) for k, v in value.items()))
        if type(value) is tuple and all(type(v) in _ATOMS for v in value):
            return ("atom-tuple", value)
        if type(value) is tuple:
            if len(value) > _MAX_NODES - self.nodes:
                raise _Unavailable
            if all(type(v) is Fraction for v in value):
                self.charge(count=len(value))
                return ("fraction-tuple", tuple(_fraction_parts(v) for v in value))
        if type(value) in {tuple, list}:
            return ("sequence", tuple(self.node(v) for v in value))
        if isinstance(value, CoordinateSpace):
            return ("space", self.node(value.name), self.node(value.keys))
        if isinstance(value, FieldSource):
            return ("source", type(value), self.node(value.as_record()))
        if isinstance(value, ModelInput):
            kind, fields = _fields(value)
            return ("input", kind, self.node(fields))
        raise _Unavailable


def _matches(value, node, visited, ordered=True):
    tag = node[0]
    if tag == "atom":
        return type(value) is type(node[1]) and (value is node[1] or value == node[1])
    if tag == "float":
        return type(value) is float and value.hex() == node[1]
    if tag == "fraction":
        return (type(value) is Fraction and type(value.numerator) is node[1] and value.numerator == node[2] and
                type(value.denominator) is node[3] and value.denominator == node[4])
    if tag == "integer":
        return type(value) is node[1] and int(value) == node[2]
    if tag == "atom-tuple":
        if type(value) is not tuple:
            return False
        return value is node[1] or len(value) == len(node[1]) and all(
            type(v) is type(old) and (v is old or v == old) for v, old in zip(value, node[1], strict=True))
    key = (id(value), id(node))
    if visited.get(key) is value:
        return True
    if tag == "fraction-tuple":
        valid = type(value) is tuple and len(value) == len(node[1]) and _fractions_match(value, node[1])
    elif tag in {"array", "exact-array", "fraction-array"}:
        if type(value) is not np.ndarray or value.dtype.str != node[1] or value.shape != node[2]:
            return False
        if tag == "array":
            valid = value.tobytes(order="C") == node[3]
        elif tag == "fraction-array":
            valid = _fractions_match(value.flat, node[3])
        else:
            valid = all(_matches(v, old, visited, ordered) for v, old in zip(value.flat, node[3], strict=True))
    elif tag == "map":
        if not isinstance(value, Mapping) or len(value) != len(node[1]):
            return False
        if ordered:
            valid = all(type(k) is type(old) and k == old and _matches(value[k], child, visited, ordered)
                        for k, (old, child) in zip(value, node[1], strict=True))
        else:
            keys = {k: type(k) for k in value}
            valid = all(keys.get(old) is type(old) and _matches(value[old], child, visited, ordered)
                        for old, child in node[1])
    elif tag == "sequence":
        valid = type(value) in {tuple, list} and len(value) == len(node[1]) and all(
            _matches(v, old, visited, ordered) for v, old in zip(value, node[1], strict=True))
    elif tag == "space":
        from rexgraph.type_accession import CoordinateSpace
        valid = isinstance(value, CoordinateSpace) and _matches(value.name, node[1], visited, ordered) and _matches(value.keys, node[2], visited, ordered)
    elif tag == "source":
        valid = type(value) is node[1] and _matches(value.as_record(), node[2], visited, ordered)
    elif tag == "input":
        kind, fields = _fields(value)
        valid = kind is node[1] and _matches(fields, node[2], visited, ordered)
    else:
        return False
    if valid:
        visited[key] = value
    return valid


@dataclass(frozen=True)
class _Witness:
    kind: type
    tree: object
    digest: str | None = None

    def matches(self, value, ordered=True):
        try:
            kind, fields = _fields(value)
            return kind is self.kind and _content_matches(fields, self.tree, ordered)
        except (AttributeError, TypeError, ValueError, _Unavailable):
            return False


def _capture(value):
    try:
        kind, fields = _fields(value)
        tree = _content_tree(fields)
        return None if tree is None else _Witness(kind, tree)
    except _Unavailable:
        return None


@dataclass(frozen=True)
class _ContentWitness:
    tree: tuple

    def matches(self, value):
        try:
            return _matches(value, self.tree, {})
        except (AttributeError, KeyError, TypeError, ValueError, _Unavailable):
            return False


def capture_content(value):
    """Retain a bounded, type sensitive witness of supported container content."""
    try:
        tree = _content_tree(value)
        if tree is None or NativeContentWitness is not None and type(tree) is NativeContentWitness:
            return tree
        return _ContentWitness(tree)
    except _Unavailable:
        return None


def seal_model(value, digest, error):
    """Compute the canonical seal and retain a bounded witness of the same content."""
    witness = _capture(value)
    identity = digest(value)
    if witness is not None and not witness.matches(value):
        raise ValueError(error)
    object.__setattr__(value, "_digest", identity)
    object.__setattr__(value, "_integrity_witness", None if witness is None else _Witness(witness.kind, witness.tree, identity))


def check_model(value, digest, error):
    """Check every retained value before reusing its canonical seal."""
    witness = getattr(value, "_integrity_witness", None)
    if type(witness) is _Witness and witness.digest == value._digest and witness.matches(value):
        return
    if type(witness) is _Witness and not witness.matches(value, ordered=False):
        raise ValueError(error)
    if witness is None and hasattr(value, "_integrity_witness"):
        if digest(value) != value._digest:
            raise ValueError(error)
        return
    current = _capture(value)
    if digest(value) != value._digest or current is not None and not current.matches(value):
        raise ValueError(error)
    object.__setattr__(value, "_integrity_witness", None if current is None else _Witness(current.kind, current.tree, value._digest))
