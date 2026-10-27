"""Immutable ontology declarations over complete relational source snapshots."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from .registry import Registry
from .tensor_field import FieldSource
from .value import ExactTime, TimeRange
from .value_codec import pack_value, unpack_value

_KEY = "relational_ontology"
_PROFILE = "rexgraph.ontology.v1"
_PROVIDERS = Registry("ontology provider")
_CHANGES = {"root", "source-extension", "declaration-recode", "ontology-revision", "restriction"}
_UNSET = object()
_MAX_DECLARATION_BYTES = 64 << 20


def _state_bytes(state):
    from .io.rcbf import _write_state_stream
    stream = BytesIO()
    _write_state_stream(stream, state, max_bytes=256 << 20)
    return stream.getvalue()


@dataclass(frozen=True)
class OntologyProvider:
    """An explicitly installed mathematical validator, never imported from a file."""
    name: str
    version: int
    verify: object

    def __post_init__(self):
        _name(self.name, "provider")
        if type(self.version) is not int or self.version < 1 or not callable(self.verify):
            raise TypeError("ontology provider requires a positive version and verifier")


def register_ontology_provider(provider):
    if not isinstance(provider, OntologyProvider):
        raise TypeError("an installed OntologyProvider is required")
    key = f"{len(provider.name)}:{provider.name}:{provider.version}"
    existing = _PROVIDERS.get(key)
    if existing is not None and existing != provider:
        raise ValueError("ontology provider version is already installed")
    _PROVIDERS.register(key, provider)
    return provider


def _name(value, label):
    if type(value) is not str or not value or "\x00" in value or len(value.encode("utf-8")) > 4096:
        raise ValueError(f"ontology {label} requires bounded nonempty text")
    return value


def _digest(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("ontology parent requires a canonical digest")


def _anchor(graph, anchor):
    from .cells import cell_count
    if (type(anchor) is not tuple or len(anchor) != 2
            or any(type(n) is not int or n < 0 for n in anchor)
            or anchor[1] >= cell_count(graph, anchor[0])):
        raise ValueError("ontology anchor must identify an actual retained cell")
    return anchor


def ontology_dependencies(declaration):
    """Read explicit source pins without expanding ontology fields or native archives."""
    if (not isinstance(declaration, Mapping) or declaration.get("profile") != _PROFILE
            or type(declaration.get("dependencies")) is not tuple
            or len(declaration["dependencies"]) > 100_000):
        raise ValueError("invalid ontology dependency declaration")
    refs = []
    for dependency in declaration["dependencies"]:
        if type(dependency) is not dict or set(dependency) != {"record_id", "version", "state_digest"}:
            raise ValueError("ontology dependency requires a complete source reference")
        source = FieldSource(None, **dependency)
        if source.record_id is None or source.version is None or source.version < 1:
            raise ValueError("ontology evidence requires an explicit positive record version")
        refs.append(source)
    return tuple(refs)


def _contract(value):
    keys = {"profile", "identity", "context", "revision", "known_at", "validity", "parent",
            "change", "signature", "universe", "axioms", "interpretations", "dependencies", "provider", "annotations"}
    if not isinstance(value, Mapping) or set(value) != keys or value["profile"] != _PROFILE:
        raise ValueError("unsupported ontology declaration")
    encoded = pack_value(dict(value))
    if len(encoded) > _MAX_DECLARATION_BYTES:
        raise ValueError("ontology declaration exceeds byte admission")
    value = unpack_value(encoded)
    _name(value["identity"], "identity"); _name(value["context"], "context")
    if type(value["revision"]) is not int or value["revision"] < 1:
        raise ValueError("ontology revision requires a positive integer")
    if not isinstance(value["known_at"], ExactTime):
        raise TypeError("ontology knowledge horizon requires ExactTime")
    if value["validity"] is not None and not isinstance(value["validity"], TimeRange):
        raise TypeError("ontology validity requires a declared TimeRange")
    change = value["change"]
    if (type(change) is not dict or set(change) != {"kind", "maps"}
            or change["kind"] not in _CHANGES or type(change["maps"]) is not tuple):
        raise ValueError("ontology revision requires an explicit change and ordered maps")
    parent = value["parent"]
    if parent is None:
        if value["revision"] != 1 or change != {"kind": "root", "maps": ()}:
            raise ValueError("a root ontology requires revision one and no parent maps")
    else:
        if (type(parent) is not dict or set(parent) != {"digest", "revision"}
                or type(parent["revision"]) is not int or parent["revision"] < 1
                or value["revision"] != parent["revision"] + 1 or change["kind"] == "root"):
            raise ValueError("ontology child requires an explicit immediate parent")
        _digest(parent["digest"])
        if not change["maps"]:
            raise ValueError("ontology revisions require declared comparison or transport maps")
    if any(type(value[k]) is not dict for k in ("signature", "universe", "annotations")):
        raise TypeError("ontology signature, universe and annotations require closed records")
    if any(type(value[k]) is not tuple for k in ("axioms", "interpretations", "dependencies")):
        raise TypeError("ontology axioms, interpretations and dependencies require ordered tuples")
    ontology_dependencies(value)
    provider = value["provider"]
    if provider is not None:
        if type(provider) is not dict or set(provider) != {"name", "version"}:
            raise ValueError("ontology provider requires its exact name and version")
        _name(provider["name"], "provider")
        if type(provider["version"]) is not int or provider["version"] < 1:
            raise ValueError("ontology provider version must be positive")
    return value


class OntologyState:
    """Own the whole source and its declarations, without granting execution access.

    ``known_at`` is a declared information horizon. It does not establish the
    transaction cutoff or access policy of an external evidence store. Signature,
    axioms and maps retain their closed data; an installed provider checks their
    mathematical meaning when ``verify`` is explicitly requested.
    """
    __slots__ = ("_state", "_codec", "_anchor", "_declaration", "_digest", "_ontology_digest", "_summary")

    def __new__(cls, *args, **kwargs):
        raise TypeError("use OntologyState.create, from_record or from_bytes")

    def __setattr__(self, name, value):
        raise AttributeError("ontology snapshots are immutable")

    @classmethod
    def create(cls, graph, *, identity, context, known_at, signature, universe, axioms=(),
               interpretations=(), dependencies=(), provider=None, validity=None,
               annotations=None, anchor=(1, 0)):
        declaration = {"profile": _PROFILE, "identity": identity, "context": context,
            "revision": 1, "known_at": known_at, "validity": validity, "parent": None,
            "change": {"kind": "root", "maps": ()}, "signature": signature, "universe": universe, "axioms": axioms,
            "interpretations": interpretations, "dependencies": tuple(d.as_record() for d in dependencies),
            "provider": provider, "annotations": {} if annotations is None else annotations}
        return cls._declare(graph, declaration, anchor, replace=False)

    @classmethod
    def _declare(cls, graph, declaration, anchor, *, replace):
        from .graph import RexGraph
        from .state import from_state, to_state
        if not isinstance(graph, RexGraph):
            raise TypeError("ontology requires a complete RexGraph source")
        _anchor(graph, anchor)
        declaration = _contract(declaration)
        owned = from_state(to_state(graph))
        if not replace and owned.get_metadata(*anchor, _KEY) is not None:
            raise ValueError("ontology anchor already has a declaration; revise it explicitly")
        owned.attach_metadata(*anchor, _KEY, declaration)
        return cls.from_record(owned, anchor=anchor)

    @classmethod
    def from_record(cls, graph, *, anchor=(1, 0)):
        from .graph import RexGraph
        from .object_identity import state_object_digest
        from .state import to_state
        if not isinstance(graph, RexGraph):
            raise TypeError("ontology record requires RexGraph")
        _anchor(graph, anchor)
        declaration = _contract(graph.get_metadata(*anchor, _KEY))
        state = to_state(graph)
        encoded = pack_value(declaration)
        source_digest = state_object_digest(state)
        out = object.__new__(cls)
        object.__setattr__(out, "_state", _state_bytes(state))
        object.__setattr__(out, "_codec", "rexgraph.rcbf-v6")
        object.__setattr__(out, "_anchor", anchor)
        object.__setattr__(out, "_declaration", encoded)
        object.__setattr__(out, "_digest", source_digest)
        object.__setattr__(out, "_ontology_digest",
            sha256(pack_value((_PROFILE, anchor, source_digest, encoded))).hexdigest())
        summary = {key: declaration[key] for key in
                   ("identity", "context", "revision", "known_at", "validity", "parent", "provider", "dependencies")}
        summary.update(change={"kind": declaration["change"]["kind"], "map_count": len(declaration["change"]["maps"])},
                       axiom_count=len(declaration["axioms"]), interpretation_count=len(declaration["interpretations"]))
        object.__setattr__(out, "_summary", pack_value(summary))
        return out

    @property
    def declaration(self):
        return unpack_value(self._declaration)

    @property
    def digest(self):
        return self._ontology_digest

    @property
    def source_digest(self):
        """Identity of the complete graph, separate from the selected ontology anchor."""
        return self._digest

    @property
    def dependencies(self):
        return ontology_dependencies(dict(unpack_value(self._summary), profile=_PROFILE))

    def record(self):
        """Return a detached, complete graph for RCBF, RCBD or ordinary RCDB records."""
        from .io.rcbo import _source_record
        return _source_record(self._state, self._codec)

    def require_horizon(self, cutoff):
        if not isinstance(cutoff, ExactTime):
            raise TypeError("ontology cutoff requires ExactTime")
        if unpack_value(self._summary)["known_at"].seconds > cutoff.seconds:
            raise ValueError("ontology lies beyond the declared knowledge horizon")

    def revise(self, graph, *, known_at, kind, maps, signature=None, universe=None, axioms=None,
               interpretations=None, dependencies=None, validity=_UNSET, annotations=None, provider=_UNSET):
        """Declare a child; mathematical validation is an explicit provider operation."""
        parent = self.declaration
        if not isinstance(known_at, ExactTime) or known_at.seconds < parent["known_at"].seconds:
            raise ValueError("ontology child cannot precede its parent's knowledge horizon")
        child = dict(parent, revision=parent["revision"] + 1,
            parent={"digest": self.digest, "revision": parent["revision"]}, known_at=known_at,
            change={"kind": kind, "maps": maps})
        for name, value in (("signature", signature), ("universe", universe), ("axioms", axioms),
                            ("interpretations", interpretations),
                            ("annotations", annotations)):
            if value is not None:
                child[name] = value
        if validity is not _UNSET:
            child["validity"] = validity
        if provider is not _UNSET:
            child["provider"] = provider
        if dependencies is not None:
            child["dependencies"] = tuple(d.as_record() for d in dependencies)
        return type(self)._declare(graph, child, self._anchor, replace=True)

    def is_child_of(self, parent):
        if not isinstance(parent, OntologyState):
            raise TypeError("the actual parent ontology is required")
        a, b = unpack_value(self._summary), unpack_value(parent._summary)
        return (a["parent"] == {"digest": parent.digest, "revision": b["revision"]}
                and a["identity"] == b["identity"] and a["context"] == b["context"]
                and a["known_at"].seconds >= b["known_at"].seconds)

    def verify(self, *, sources, parent=None):
        """Run the explicitly installed provider against current application authority."""
        summary = unpack_value(self._summary)
        ref = summary["provider"]
        if ref is None:
            raise LookupError("ontology has no declared mathematical validation provider")
        provider = _PROVIDERS.get(f"{len(ref['name'])}:{ref['name']}:{ref['version']}")
        if provider is None:
            raise LookupError("required ontology provider is not installed")
        if summary["parent"] is not None:
            if not isinstance(parent, OntologyState) or not self.is_child_of(parent):
                raise ValueError("ontology verification requires its actual parent snapshot")
        elif parent is not None:
            raise ValueError("root ontology verification does not declare a parent")
        return provider.verify(self, sources=sources, parent=parent)

    def inspect(self):
        return dict(unpack_value(self._summary), digest=self.digest, source_digest=self.source_digest,
                    anchor=self._anchor, mathematical_validation="requires-explicit-provider")

    def to_bytes(self):
        from .io.rcbo import ontology_bytes
        return ontology_bytes(self)

    @classmethod
    def from_bytes(cls, data):
        from io import BytesIO
        from .io.rcbo import read_rcbo
        if type(data) is not bytes:
            raise TypeError("ontology archive requires bytes")
        return read_rcbo(BytesIO(data))


__all__ = ["OntologyState", "OntologyProvider", "register_ontology_provider"]
