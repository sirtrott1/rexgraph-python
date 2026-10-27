"""Immutable native relational programs, functions and complete fit families."""
from __future__ import annotations

from hashlib import sha256
from importlib.resources import files

from .value_codec import pack_value, unpack_value

__all__ = ["RelationalRuntime", "RelationalFieldFunction"]
_LIMITS = (65536, 128, 256, 1000000, 65536, 4194304, 16777216)
_MAX_BYTES = 132 << 20
_WRAPPER_DIGEST = sha256(files("rexgraph").joinpath("relational_runtime.py").read_bytes()).hexdigest()


def _backend():
    try:
        from rexgraph_cpp._temporal_program import NativeRelationalRuntime, temporal_program_build_info
    except ImportError as exc:
        raise ImportError("relational program execution requires rexgraph_cpp built with RexTemporal support") from exc
    implementation = temporal_program_build_info()
    implementation["core_wrapper"] = _WRAPPER_DIGEST
    return NativeRelationalRuntime, implementation


class RelationalRuntime:
    """Own a native source snapshot; extensions produce new source snapshots.

    Inputs and observations are caller declarations. The runtime does not infer
    external evidence availability from a function domain or a provenance label.
    Parent digests record lineage; ``extends`` checks the actual retained source.
    """

    __slots__ = ("_owner", "_parent", "_packet")

    def __new__(cls, *args, **kwargs):
        raise TypeError("use RelationalRuntime.create or RelationalRuntime.from_bytes")

    def __setattr__(self, name, value):
        raise AttributeError("relational runtime snapshots are immutable")

    @classmethod
    def _wrap(cls, owner, parent=None, *, lineage=None):
        _, implementation = _backend()
        source, endpoint = owner.encode()
        if parent is not None:
            if not owner.extends(parent._owner):
                raise ValueError("native child does not preserve its actual parent source")
            lineage = parent._lineage_identity()
        packet = pack_value({"profile": "rexgraph.relational-runtime.v1", "implementation": implementation,
            "source": source, "endpoint": endpoint, "selected": owner.selected,
            "limits": owner.limits, "parent": lineage})
        if len(packet) > _MAX_BYTES:
            raise ValueError("relational runtime packet exceeds admission")
        out = object.__new__(cls)
        object.__setattr__(out, "_owner", owner)
        object.__setattr__(out, "_parent", lineage)
        object.__setattr__(out, "_packet", packet)
        return out

    @classmethod
    def create(cls, namespace, *, limits=_LIMITS):
        native, _ = _backend()
        return cls._wrap(native.create(namespace, limits=limits))

    @classmethod
    def from_bytes(cls, packet, *, restore_limits=_LIMITS):
        if type(packet) is not bytes:
            raise TypeError("relational runtime packet requires bytes")
        if len(packet) > _MAX_BYTES:
            raise ValueError("relational runtime packet exceeds admission")
        data = unpack_value(packet)
        keys = {"profile", "implementation", "source", "endpoint", "selected", "limits", "parent"}
        if type(data) is not dict or set(data) != keys or data["profile"] != "rexgraph.relational-runtime.v1":
            raise ValueError("unsupported relational runtime packet")
        native, implementation = _backend()
        if data["implementation"] != implementation:
            raise ValueError("relational runtime implementation differs; use its original runtime or an explicit migration")
        bounds = data["limits"]
        if (type(bounds) is not tuple or len(bounds) != 7 or len(restore_limits) != 7
                or any(type(n) is not int or n < 0 for n in (*bounds, *restore_limits))
                or any(n > cap for n, cap in zip(bounds, restore_limits, strict=True))):
            raise ValueError("relational runtime learning bounds exceed restore admission")
        if type(data["selected"]) is not int or not 0 <= data["selected"] < 1 << 64:
            raise ValueError("invalid selected relational object")
        parent = data["parent"]
        if parent is not None:
            if (type(parent) is not dict or set(parent) != {"packet", "source", "endpoint", "namespace", "sequence", "objects", "applications", "correspondence"}
                    or parent["correspondence"] != "retained-object-application-and-ordered-occurrence-identities-v1"
                    or any(type(parent[n]) is not str or len(parent[n]) != 64 or any(c not in "0123456789abcdef" for c in parent[n])
                           for n in ("packet", "source", "endpoint"))
                    or any(type(parent[n]) is not int or not 0 <= parent[n] < 1 << 64
                           for n in ("namespace", "sequence", "objects", "applications"))):
                raise ValueError("invalid relational parent correspondence")
        owner = native.decode(data["source"], data["endpoint"], selected=data["selected"], limits=bounds)
        out = cls._wrap(owner, lineage=parent)
        if out.to_bytes() != packet:
            raise ValueError("noncanonical relational runtime packet")
        return out

    @property
    def selected(self):
        return self._owner.selected

    @property
    def digest(self):
        return sha256(self._packet).hexdigest()

    @property
    def parent_digest(self):
        return None if self._parent is None else self._parent["packet"]

    def _lineage_identity(self):
        source, endpoint = self._owner.encode()
        ns, sequence, objects, applications = self._owner.identity()
        return {"packet": self.digest, "source": sha256(source).hexdigest(), "endpoint": sha256(endpoint).hexdigest(),
            "namespace": ns, "sequence": sequence, "objects": objects, "applications": applications,
            "correspondence": "retained-object-application-and-ordered-occurrence-identities-v1"}

    def to_bytes(self):
        return self._packet

    def extend(self, operation, specification):
        """Execute a declared native extension, retaining all intermediate objects."""
        owner = self._owner.extend(operation, specification)
        return type(self)._wrap(owner, self)

    def extends(self, parent):
        if not isinstance(parent, RelationalRuntime):
            raise TypeError("actual parent runtime required")
        return self._parent == parent._lineage_identity() and self._owner.extends(parent._owner)

    def inspect(self, *, max_records=4096):
        info = self._owner.inspect(max_records=max_records)
        info.update(digest=self.digest, parent_digest=self.parent_digest,
                    parent_correspondence=None if self._parent is None else dict(self._parent),
                    source_evolution="immutable native source extension", arithmetic="rational")
        return info

    def function(self, identity=None):
        return self._owner.function(self.selected if identity is None else identity)

    def fit(self, identity=None):
        return self._owner.fit(self.selected if identity is None else identity)

    def evolution(self, identity=None):
        return self._owner.evolution(self.selected if identity is None else identity)

    def selection(self, identity=None):
        return self._owner.selection(self.selected if identity is None else identity)

    def layer(self, identity=None):
        return self._owner.layer(self.selected if identity is None else identity)

    def support(self, identity=None):
        return self._owner.support(self.selected if identity is None else identity)

    def memberships(self, identity=None):
        return self._owner.memberships(self.selected if identity is None else identity)

    def program(self, identity=None):
        return self._owner.program(self.selected if identity is None else identity)

    def evaluate(self, parameter, identity=None, *, tensor=False):
        return self._owner.evaluate(self.selected if identity is None else identity, parameter, tensor=tensor)


class RelationalFieldFunction:
    """A whole source-owned field function bound from a native model envelope."""

    __slots__ = ("_owner", "_record", "_polynomial", "_packet")

    def __new__(cls, *args, **kwargs):
        raise TypeError("use RelationalFieldFunction.from_envelope or from_bytes")

    def __setattr__(self, name, value):
        raise AttributeError("relational field functions are immutable")

    @classmethod
    def from_envelope(cls, archive, record, *, polynomial=True):
        _, implementation = _backend()
        from rexgraph_cpp._temporal_program import NativeRelationalFieldFunction
        owner = NativeRelationalFieldFunction.from_envelope(archive, record, polynomial=polynomial)
        out = object.__new__(cls)
        object.__setattr__(out, "_owner", owner)
        object.__setattr__(out, "_record", record)
        object.__setattr__(out, "_polynomial", polynomial)
        object.__setattr__(out, "_packet", pack_value({"profile": "rexgraph.relational-field-function.v1",
            "implementation": implementation, "envelope": owner.archive(), "record": record, "polynomial": polynomial}))
        return out

    @classmethod
    def from_bytes(cls, packet):
        if type(packet) is not bytes:
            raise TypeError("field function packet requires bytes")
        if len(packet) > 33 << 20:
            raise ValueError("field function packet exceeds admission")
        data = unpack_value(packet)
        if (type(data) is not dict or set(data) != {"profile", "implementation", "envelope", "record", "polynomial"}
                or data["profile"] != "rexgraph.relational-field-function.v1"):
            raise ValueError("unsupported field function packet")
        if data["implementation"] != _backend()[1]:
            raise ValueError("field function implementation differs; use its original runtime or an explicit migration")
        out = cls.from_envelope(data["envelope"], data["record"], polynomial=data["polynomial"])
        if out.to_bytes() != packet:
            raise ValueError("noncanonical field function packet")
        return out

    def to_bytes(self):
        return self._packet

    def evaluate(self, parameter):
        return self._owner.evaluate(parameter)

    def inspect(self, *, max_records=4096):
        result = {"record": self._record, "polynomial": self._polynomial, "ports": self._owner.ports(),
                  "evidence_scope": "declared-inputs", "digest": sha256(self._packet).hexdigest()}
        if self._polynomial:
            result["coefficients"] = self._owner.coefficients(max_records=max_records)
        return result
