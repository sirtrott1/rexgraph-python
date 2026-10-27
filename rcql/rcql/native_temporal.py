"""Finite retained native function sources, without serialized live authority."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import cached_property
import hashlib

from rexgraph.value_codec import pack_value, unpack_value

_MAGIC = b"RGTF\x01"
_LIMIT = 64*1024*1024
from pathlib import Path
_WRAPPER_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OPCODES = {"input": 1, "constant": 2, "derivative": 3, "pullback": 4,
           "add": 5, "subtract": 6, "multiply": 7, "divide": 8, "restrict": 9}
TEMPORAL_OPERATIONS = {"interval": 1, "allen": 2, "atom_support": 3,
    "support_union": 4, "support_intersection": 5, "support_difference": 6,
    "interval_from_support": 7, "law": 8, "mixture_probability": 9}
KINDS = {"source", "coordinate", "function", "program", "support", "context", "set", "fit", "joint", "product", "interval", "relation", "law"}


def _backend():
    try:
        from rexgraph_temporal import NativeTemporal, build_info
    except ModuleNotFoundError as error:
        if error.name != "rexgraph_temporal":
            raise
        raise ImportError("whole-function execution requires rexgraph-temporal-bindings") from error
    return NativeTemporal, lambda: {**build_info(), "rcql_wrapper": _WRAPPER_DIGEST}


def _frame(value):
    body = pack_value(value)
    result = _MAGIC+hashlib.sha256(body).digest()+body
    if len(result) > _LIMIT:
        raise ValueError("retained temporal source exceeds its byte limit")
    return result


def _declaration(raw):
    if type(raw) is not bytes or len(raw) > _LIMIT or not raw.startswith(_MAGIC) or len(raw) < 37:
        raise ValueError("invalid retained temporal source framing")
    body = raw[37:]
    if hashlib.sha256(body).digest() != raw[5:37]:
        raise ValueError("retained temporal source digest mismatch")
    value = unpack_value(body)
    if (not isinstance(value, dict) or set(value) != {"version", "source", "endpoint", "limits", "build", "selected", "kind", "parent", "dependencies"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["source"]) is not bytes or type(value["endpoint"]) is not bytes
            or type(value["selected"]) is not int or not 0 <= value["selected"] < 2**64
            or value["kind"] not in KINDS or not isinstance(value["limits"], tuple)
            or len(value["limits"]) not in (0, 9)
            or any(type(v) is not int or not 0 <= v < 2**64 for v in value["limits"])
            or not isinstance(value["build"], dict)
            or value["parent"] is not None and (not isinstance(value["parent"], str) or len(value["parent"]) != 64)):
        raise ValueError("invalid retained temporal source declaration")
    from rexgraph.source_binding import source_from_record
    for reference in value["dependencies"]:
        source_from_record(reference, allow_cpp=True).check()
    if (value["kind"] == "source") != (value["selected"] == 0):
        raise ValueError("native selection kind and identity differ")
    if type(value["dependencies"]) is not tuple or len(value["dependencies"]) > 4096:
        raise ValueError("bounded native dependency declaration required")
    if _frame(value) != raw:
        raise ValueError("noncanonical retained temporal source")
    return value


@dataclass(frozen=True)
class TemporalModel:
    """An immutable native source and selected object; IDs are local to this source."""
    data: bytes

    def __post_init__(self):
        _declaration(self.data)

    @classmethod
    def create(cls, name_space=1, limits=(), dependencies=()):
        native, _ = _backend()
        return cls._from_native(native(name_space=name_space, limits=limits), 0, "source").with_dependencies(dependencies)

    @classmethod
    def from_archive(cls, source, endpoint, *, selected, kind, limits=(), dependencies=()):
        """Receive complete native registry archives under this implementation.

        Native validation checks the retained declarations and their producers.
        Dependencies describe exact sources; execution still requires their
        current bindings and policies. External application companions require
        their own admission before they can supply those bindings.
        """
        if type(source) is not bytes or type(endpoint) is not bytes:
            raise TypeError("native source and endpoint bytes required")
        if (type(selected) is not int or not 0 <= selected < 2**64
                or type(kind) is not str or kind not in KINDS
                or (kind == "source") != (selected == 0)):
            raise ValueError("typed native object selection required")
        native, _ = _backend()
        owner = native(source, endpoint, limits=limits)
        model = cls._from_native(owner, selected, kind).with_dependencies(dependencies)
        model.verify()
        declaration = model.declaration
        if (declaration["source"], declaration["endpoint"]) != (source, endpoint):
            raise ValueError("native archive changed during adoption")
        return model

    @classmethod
    def from_relational(cls, runtime, *, kind, limits=(), dependencies=()):
        """Explicitly adopt the complete archive from Core's relational runtime.

        This is a checked archival conversion, never an ID-only owner transfer.
        The original runtime and its implementation receipt remain unchanged.
        """
        from rexgraph.relational_runtime import RelationalRuntime
        if type(runtime) is not RelationalRuntime:
            raise TypeError("actual canonical RelationalRuntime required")
        packet = unpack_value(runtime.to_bytes())
        return cls.from_archive(packet["source"], packet["endpoint"], selected=runtime.selected,
                                kind=kind, limits=limits, dependencies=dependencies)

    def to_relational(self, *, limits=None):
        """Export a checked mathematical archive to the canonical native runtime."""
        if self.dependencies:
            raise ValueError("retain source-bound models as FUNCTION_RECORD; an unqualified packet cannot discard dependencies")
        self.verify()
        from rexgraph.relational_runtime import RelationalRuntime, _backend, _LIMITS
        _, implementation = _backend()
        d = self.declaration
        packet = pack_value({"profile": "rexgraph.relational-runtime.v1", "implementation": implementation,
            "source": d["source"], "endpoint": d["endpoint"], "selected": self.selected,
            "limits": _LIMITS if limits is None else tuple(limits), "parent": None})
        runtime = RelationalRuntime.from_bytes(packet)
        value = unpack_value(runtime.to_bytes())
        if (value["source"],value["endpoint"]) != (d["source"],d["endpoint"]):
            raise ValueError("native archive changed during relational export")
        return runtime

    @classmethod
    def _from_native(cls, native, selected, kind, parent=None, dependencies=()):
        _, info = _backend()
        return cls(_frame({"version": 1, "source": native.source_bytes, "endpoint": native.endpoint_bytes,
            "limits": native.limits, "build": info(), "selected": selected, "kind": kind, "parent": parent, "dependencies": tuple(v.as_record() for v in dependencies)}))

    @property
    def declaration(self):
        return _declaration(self.data)

    @property
    def coefficient_digest(self):
        return hashlib.sha256(self.data).hexdigest()

    @property
    def dependencies(self):
        from rexgraph.source_binding import source_from_record
        return tuple(source_from_record(v, allow_cpp=True) for v in self.declaration["dependencies"])

    def with_dependencies(self, references):
        from rexgraph.tensor_field import FieldSource
        refs = {v.coefficient_digest: v.as_record() for v in self.dependencies}
        for ref in references:
            if not isinstance(ref, FieldSource):
                raise TypeError("native model dependencies require exact source references")
            ref.check(); refs[ref.coefficient_digest] = ref.as_record()
        return TemporalModel(_frame({**self.declaration, "dependencies": tuple(refs.values())}))

    @property
    def selected(self):
        return self.declaration["selected"]

    @property
    def kind(self):
        return self.declaration["kind"]

    @cached_property
    def _native(self):
        native, info = _backend()
        d = self.declaration
        if d["build"] != info():
            raise ValueError("retained function model requires its declared native implementation")
        return native(d["source"], d["endpoint"], limits=d["limits"])

    def replay(self, expected_build, limits, sources=None):
        """Explicitly reconstruct original function declarations in this runtime.

        A producer receipt selects data, not old authority. Current policy and
        nine receiving limits apply before complete native verification.
        """
        d=self.declaration
        if type(expected_build) is not dict or expected_build!=d['build']:
            raise ValueError("explicit original temporal implementation receipt required")
        native,info=_backend()
        current=info()
        if set(expected_build)!=set(current) or expected_build.get('profile')!='retained-temporal-v1':
            raise ValueError("unsupported temporal declaration profile")
        for key in ('temporal_digest','rcql_wrapper'):
            value=expected_build[key]
            if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError("exact temporal implementation digests required")
        if len(limits)!=9 or any(type(v) is not int or not 0<=v<2**64 for v in limits):
            raise ValueError("nine explicit current temporal receiving limits required")
        from .layer_product import _execution_sources
        _execution_sources(self,{} if sources is None else sources)
        admitted=tuple(map(min,zip(d['limits'],limits,strict=True))) if d['limits'] else tuple(limits)
        owner=native(d['source'],d['endpoint'],limits=admitted)
        owner.verify()
        result=self._from_native(owner,self.selected,self.kind,parent=self.coefficient_digest,dependencies=self.dependencies)
        result.verify()
        if (result.declaration['source'],result.declaration['endpoint'])!=(d['source'],d['endpoint']):
            raise ValueError("temporal replay changed original source declarations")
        return result

    def declare(self, kind, specification):
        if not isinstance(specification, dict):
            raise TypeError("native construction requires an explicit specification")
        kwargs = dict(specification)
        if kind == "program":
            opcode = kwargs.get("opcode", "input")
            if isinstance(opcode, str):
                if opcode not in OPCODES:
                    raise ValueError("unsupported whole-function opcode")
                kwargs["opcode"] = OPCODES[opcode]
        methods = {"coordinate": "coordinate", "function": "function", "program": "program",
            "support": "support", "context": "context", "set": "function_set"}
        if kind in methods:
            source, identity = getattr(self._native, methods[kind])(**kwargs)
        elif kind in TEMPORAL_OPERATIONS:
            source, identity = self._native.temporal(TEMPORAL_OPERATIONS[kind], **kwargs)
            kind = {"allen": "relation", "atom_support": "support", "support_union": "support",
                "support_intersection": "support", "support_difference": "support",
                "interval_from_support": "interval", "mixture_probability": "function"}.get(kind, kind)
        else:
            raise ValueError("unsupported retained native constructor")
        return self._from_native(source, identity, kind, self.coefficient_digest, self.dependencies)

    def apply(self, program, bindings, learned_family=0):
        source, identity = self._native.apply(program, bindings, learned_family)
        return self._from_native(source, identity, "function", self.coefficient_digest, self.dependencies)

    def fit(self, function_set, context, candidates, expected, observation_domain=None):
        source, identity = self._native.fit(function_set, context, candidates, expected, observation_domain)
        return self._from_native(source, identity, "fit", self.coefficient_digest, self.dependencies)

    def joint(self, families, transitions=()):
        source, identity = self._native.joint(families, transitions)
        return self._from_native(source, identity, "joint", self.coefficient_digest, self.dependencies)

    def product(self, profile, layers, transitions=(), name="layer product", previous=0):
        if profile not in {"functions", "families"}:
            raise ValueError("native layer profile must be functions or families")
        source, result = self._native.product(profile == "functions", layers, transitions, name, previous)
        return self._from_native(source, result["diagram"], "product", self.coefficient_digest, self.dependencies)

    def revise_program(self, program, changes):
        if not isinstance(changes, dict):
            raise TypeError("program changes require a finite mapping")
        original = self._native.program_info(program)
        if set(changes)-set(original):
            raise ValueError("unknown native program field")
        return self.declare("program", {**original, **changes})

    def select(self, identity, kind):
        if kind not in KINDS or type(identity) is not int or not 0 < identity < 2**64:
            raise ValueError("typed native object selection required")
        self._check_kind(identity, kind)
        value = {**self.declaration, "selected": identity, "kind": kind}
        return TemporalModel(_frame(value))

    def _check_kind(self, identity, kind):
        expected = {"function": (1,6), "program": (3,), "support": (4,), "context": (9,),
                    "set": (12,), "fit": (8,), "joint": (8,), "product": (13,),
                    "interval": (5,), "relation": (5,), "law": (6,), "coordinate": (10,)}
        matches = [o for o in self._native.objects() if o["id"] == identity]
        if len(matches) != 1 or matches[0]["kind"] not in expected.get(kind, ()):
            raise ValueError("native selection differs from its declared object kind")
        if kind in {"fit", "joint", "product"}:
            getattr(self._native, {"fit": "fit_info", "joint": "joint_info", "product": "product_info"}[kind])(identity)

    def info(self):
        if self.kind == "function":
            return self._native.function_info(self.selected)
        if self.kind == "program":
            return self._native.program_info(self.selected)
        if self.kind == "fit":
            return self._native.fit_info(self.selected)
        if self.kind == "joint":
            return self._native.joint_info(self.selected)
        if self.kind == "product":
            return self._native.product_info(self.selected)
        if self.kind == "support":
            return self._native.support_info(self.selected)
        return {"selected": self.selected, "kind": self.kind, "objects": self._native.objects()}

    def value(self, at, function=None):
        return self._native.value(self.selected if function is None else function, at)

    def verify(self):
        self._native.verify()
        if self.kind != "source":
            self._check_kind(self.selected, self.kind)
        return True

    def to_record(self):
        from rexgraph import RexGraph
        record = RexGraph.from_cells([1, []])
        record.attach_metadata(0, 0, "rcql_temporal", base64.b64encode(self.data).decode("ascii"))
        if self.dependencies:
            record._agent_meta = {"rcql_evidence": [v.as_record() for v in self.dependencies]}
        return record

    @classmethod
    def from_record(cls, record):
        from rexgraph.object_identity import object_digest
        value = record.get_metadata(0, 0, "rcql_temporal")
        if not isinstance(value, str):
            raise ValueError("record has no retained temporal model")
        result = cls(base64.b64decode(value, validate=True))
        if object_digest(result.to_record()) != object_digest(record):
            raise ValueError("temporal record carrier differs from its declaration")
        return result
