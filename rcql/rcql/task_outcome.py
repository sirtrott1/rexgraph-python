"""Portable task outcomes with separately admitted native execution owners."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.value_codec import pack_value, unpack_value

_LIMITS = (4096, 16000000, 64000000, 16384)
_RECORD = {"schema", "archive", "archive_sha256", "provider", "source", "dependencies", "limits", "max_bytes", "response_origin"}


def _binding():
    try:
        import rexgraph_cpp
    except ImportError as exc:
        raise ImportError("native task outcomes require rexgraph-cpp-bindings with the task outcome provider") from exc
    if not hasattr(rexgraph_cpp, "NativeTaskOutcome"):
        raise ImportError("installed rexgraph-cpp-bindings has no native task outcome provider")
    return rexgraph_cpp


@dataclass(frozen=True, eq=False)
class TaskOutcome:
    """A whole native archive, source declarations and an optional actual owner.

    Portable decoding is passive and grants no execution authority. Assessment,
    native restoration and serialization require a separately supplied current
    availability context. Callback receipts remain with the Hive executor.
    """

    archive: bytes
    provider: tuple
    source: CppSourceReference
    dependencies: tuple = ()
    limits: tuple = _LIMITS
    max_bytes: int = 256 * 1024 * 1024
    response_origin: bytes = b""
    _owner: object = field(default=None, init=False, repr=False)
    _digest: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("task outcome byte admission must be positive")
        if type(self.archive) is not bytes or not 32 <= len(self.archive) <= self.max_bytes:
            raise ValueError("task outcome requires a bounded whole native archive")
        if type(self.response_origin) is not bytes or len(self.response_origin) > self.max_bytes:
            raise ValueError("task response origin requires bounded canonical content")
        if self.response_origin:
            origin = unpack_value(self.response_origin)
            if (not isinstance(origin, dict) or set(origin) != {"schema", "model_digest", "output_digest", "method", "native_archive_sha256", "execution_receipt"}
                    or origin["schema"] != "rcql.task-response-origin.v1"):
                raise ValueError("invalid retained model response origin")
        limits = tuple(self.limits)
        if len(limits) != 4 or any(type(n) is not int or n <= 0 for n in limits):
            raise ValueError("task outcome requires four positive native action bounds")
        provider = tuple(tuple(row) for row in self.provider)
        if (any(len(row) != 2 or any(type(v) is not str or not v for v in row) for row in provider)
                or tuple(sorted(dict(provider).items())) != provider
                or set(dict(provider)) != {"engine", "engine_version", "engine_headers", "binding", "profile", "scalar_policy"}):
            raise ValueError("task outcome requires its complete pinned provider receipt")
        dependencies = tuple(self.dependencies)
        if not isinstance(self.source, CppSourceReference) or any(not isinstance(r, CppSourceReference) for r in dependencies):
            raise TypeError("task outcome requires declared native source publications")
        object.__setattr__(self, "source", self.source.detached())
        object.__setattr__(self, "dependencies", tuple(r.detached() for r in dependencies))
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "_digest", sha256(pack_value(self.to_record())).hexdigest())

    @property
    def coefficient_digest(self):
        return self._digest

    def to_record(self):
        """Return passive immutable content; this does not admit an execution."""
        return {"schema": "rcql.native-task-outcome.v1", "archive": self.archive,
                "archive_sha256": sha256(self.archive).hexdigest(), "provider": self.provider,
                "source": self.source.as_record(), "dependencies": tuple(r.as_record() for r in self.dependencies),
                "limits": self.limits, "max_bytes": self.max_bytes, "response_origin": self.response_origin}

    @classmethod
    def from_record(cls, record):
        if (not isinstance(record, dict) or set(record) != _RECORD
                or record["schema"] != "rcql.native-task-outcome.v1"
                or type(record["archive"]) is not bytes
                or sha256(record["archive"]).hexdigest() != record["archive_sha256"]):
            raise ValueError("invalid portable native task outcome")
        return cls(record["archive"], record["provider"], source_from_record(record["source"], allow_cpp=True),
            tuple(source_from_record(r, allow_cpp=True) for r in record["dependencies"]),
            record["limits"], record["max_bytes"], record["response_origin"])

    @classmethod
    def from_native(cls, owner, execution_context, *, source=None):
        module = _binding()
        if not isinstance(owner, module.NativeTaskOutcome):
            raise TypeError("an actual native task outcome owner is required")
        owner.admit(execution_context)
        image = owner.archive(execution_context)
        primary = CppSourceReference(module.NativeSource(owner.source_archive(execution_context))) if source is None else source
        dependencies = tuple(CppSourceReference(module.NativeSource(image))
            for image in owner.dependency_archives(execution_context))
        value = cls(image, tuple(sorted(module.task_outcome_build_info().items())), primary, dependencies,
                    owner.limits, owner.max_bytes)
        object.__setattr__(value, "_owner", owner)
        return value

    def native(self, execution_context, *, limits=_LIMITS, max_bytes=256 * 1024 * 1024):
        """Admit the retained live owner or explicitly restore a cold packet."""
        if (len(limits) != 4 or any(type(n) is not int or n <= 0 for n in limits)
                or type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError("current task read policy requires positive action/byte bounds")
        if any(n > cap for n, cap in zip(self.limits, limits, strict=True)) or len(self.archive) > max_bytes:
            raise ValueError("task outcome exceeds the current caller action/byte policy")
        module = _binding()
        if dict(self.provider) != module.task_outcome_build_info():
            raise ValueError("task outcome requires its pinned native implementation")
        if not isinstance(execution_context, module.NativeAvailabilityContext):
            raise TypeError("task outcome requires a separate native availability context")
        if self._owner is not None:
            self._owner.admit(execution_context)
            return self._owner
        owner = module.NativeTaskOutcome.from_archive(self.archive, execution_context,
            limits=self.limits, max_bytes=min(self.max_bytes, max_bytes))
        actual = CppSourceReference(module.NativeSource(owner.source_archive(execution_context)))
        if actual.native != self.source.native:
            raise ValueError("task outcome source differs from its actual native program")
        dependencies = tuple(CppSourceReference(module.NativeSource(image)) for image in
                             owner.dependency_archives(execution_context))
        if len(dependencies) != len(self.dependencies) or any(
                not a.matches(b) for a, b in zip(dependencies, self.dependencies, strict=True)):
            raise ValueError("task outcome dependencies differ from the complete native owner")
        if self.response_origin:
            origin = unpack_value(self.response_origin)
            actual_response = owner.response(execution_context).archive()
            if sha256(actual_response).hexdigest() != origin["native_archive_sha256"]:
                raise ValueError("task outcome response differs from its retained ModelOutput origin")
        return owner

    def bind(self, execution_context):
        """Retain one admitted cold owner for explicit repeated observations."""
        from dataclasses import replace
        owner = self.native(execution_context)
        value = replace(self)
        object.__setattr__(value, "_owner", owner)
        return value

    def inspect(self, execution_context):
        owner = self.native(execution_context)
        return {**owner.inspect(execution_context), "definition": owner.definition(execution_context)}

    def certificate(self, execution_context):
        return self.native(execution_context).certificate(execution_context)

    def verdict(self, execution_context):
        value = self.native(execution_context).inspect(execution_context)
        if value["status"] == "refuted":
            return False
        return True if value["executed_response"] and value["status"] == "satisfied" else None

    def to_bytes(self, execution_context):
        """Serialize through native current-origin checks, preserving the whole packet."""
        return self.native(execution_context).archive(execution_context)


def create_task_outcome(output, task, asset, binding, execution_context):
    from dataclasses import replace

    from rexgraph.cpp_relational_field_admission import read_available_field_result
    from rexgraph.cpp_relational_field_evidence_admission import (
        read_available_field_learning_result,
    )
    from rexgraph.field_archive import (
        AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD,
        AVAILABLE_FIELD_ARCHIVE_METHOD,
    )
    from rexgraph.model_state import ModelOutput, thaw_tree
    if not isinstance(output, ModelOutput):
        raise TypeError("task construction requires an actual whole available ModelOutput")
    if output.method == AVAILABLE_FIELD_ARCHIVE_METHOD:
        response = read_available_field_result(output, execution_context=execution_context)
    elif output.method == AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD:
        response = read_available_field_learning_result(output, execution_context=execution_context)
    else:
        raise TypeError("task construction requires a complete available execution or affine family")
    module = _binding()
    original = module.NativeInputUseAsset.from_archive(asset)
    owner = module.NativeTaskOutcome(task, original, binding, response, execution_context)
    value = TaskOutcome.from_native(owner, execution_context, source=output.source)
    origin = pack_value({"schema": "rcql.task-response-origin.v1", "model_digest": output.model_digest,
        "output_digest": output.coefficient_digest, "method": output.method,
        "native_archive_sha256": output.field_archive["sha256"], "execution_receipt": thaw_tree(output.execution_receipt)})
    retained = replace(value, response_origin=origin)
    object.__setattr__(retained, "_owner", owner)
    return retained


def pack_task_outcome(value, *, native=False):
    if not isinstance(value, TaskOutcome):
        raise TypeError("task codec requires a TaskOutcome")
    return value.to_record()
