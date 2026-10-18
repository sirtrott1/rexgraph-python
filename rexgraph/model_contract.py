"""Model/runtime contracts that are independent of training implementations and IO."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from threading import RLock

_checkpoint_capture: Callable | None = None
_model_providers: dict[tuple[str, str], ModelProvider] = {}
_provider_lock = RLock()
_provider_operations = ("create", "prepare", "infer", "train", "transport", "export", "restore")


@dataclass(frozen=True)
class ModelProvider:
    """Versioned callbacks for declared adapters and arithmetic profiles."""
    name: str
    version: str
    adapters: tuple[tuple[str, str], ...]
    infer: Callable
    arithmetic: tuple[str, ...] = ("rational",)
    create: Callable | None = None
    prepare: Callable | None = None
    train: Callable | None = None
    transport: Callable | None = None
    export: Callable | None = None
    restore: Callable | None = None
    target_profiles: tuple[str, ...] = ()
    context_operations: tuple[str, ...] = ()

    def __post_init__(self):
        if any(not isinstance(v, str) or not v or v.strip() != v
               for v in (self.name, self.version)):
            raise ValueError("model provider name and version must be declared strings")
        if isinstance(self.adapters, (str, bytes)):
            raise TypeError("model provider requires adapter and version pairs")
        adapters = []
        for pair in self.adapters:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError("model provider requires adapter and version pairs")
            if any(not isinstance(v, str) or not v or v.strip() != v for v in pair):
                raise ValueError("model provider adapter and version must be declared strings")
            if pair[0] in {"native_field", "online_flow"}:
                raise ValueError("model provider cannot replace a built in model adapter")
            adapters.append(tuple(pair))
        if not adapters or len(set(adapters)) != len(adapters):
            raise ValueError("model provider requires distinct supported adapters")
        if isinstance(self.arithmetic, (str, bytes)):
            raise TypeError("model provider arithmetic requires a sequence")
        arithmetic = tuple(self.arithmetic)
        if (not arithmetic or any(v not in {"rational", "algebraic", "approximate"} for v in arithmetic)
                or len(set(arithmetic)) != len(arithmetic)):
            raise ValueError("model provider arithmetic profiles must be distinct and supported")
        if not callable(self.infer):
            raise TypeError("model provider requires an inference callback")
        for operation in _provider_operations:
            callback = getattr(self, operation)
            if callback is not None and not callable(callback):
                raise TypeError(f"model provider {operation} callback must be callable")
        if isinstance(self.target_profiles,(str,bytes)):raise TypeError("encoded target profiles must be a sequence")
        profiles=tuple(self.target_profiles)
        if (len(set(profiles))!=len(profiles) or
                any(type(v) is not str or not v or v.strip()!=v for v in profiles)):
            raise ValueError("distinct declared encoded target profiles required")
        object.__setattr__(self,"target_profiles",profiles)
        if isinstance(self.context_operations, (str, bytes)):
            raise TypeError("execution context operations must be a sequence")
        contextual = tuple(self.context_operations)
        if (len(set(contextual)) != len(contextual) or any(
                name not in _provider_operations or getattr(self, name) is None
                for name in contextual)):
            raise ValueError("distinct supported execution context operations required")
        object.__setattr__(self, "context_operations", contextual)
        object.__setattr__(self, "adapters", tuple(adapters))
        object.__setattr__(self, "arithmetic", arithmetic)

    @property
    def operations(self):
        return tuple(name for name in _provider_operations if getattr(self, name) is not None)


def register_model_provider(provider):
    """Register one provider version without replacing a previous implementation."""
    if not isinstance(provider, ModelProvider):
        raise TypeError("register a ModelProvider")
    key = (provider.name, provider.version)
    with _provider_lock:
        previous = _model_providers.get(key)
        if previous is not None and previous is not provider:
            raise RuntimeError("a different model provider implementation is already registered")
        _model_providers[key] = provider


def model_provider_inventory():
    """Return registered versions, adapters, arithmetic profiles and operations."""
    with _provider_lock:
        return tuple({"name": provider.name, "version": provider.version,
                      "adapters": provider.adapters, "arithmetic": provider.arithmetic,
                      "operations": provider.operations,
                      **({"context_operations": provider.context_operations}
                         if provider.context_operations else {})}
                     for _, provider in sorted(_model_providers.items()))


def invoke_model_provider(provider, operation, *args, execution_context=None, **kwargs):
    """Pass separately supplied authority only to a declared context operation."""
    if not isinstance(provider, ModelProvider):
        raise TypeError("model execution requires a ModelProvider")
    if operation not in _provider_operations:
        raise ValueError("unknown model provider operation")
    callback = getattr(provider, operation)
    if callback is None:
        raise NotImplementedError(f"model provider {provider.name!r} does not support {operation}")
    if operation in provider.context_operations:
        if execution_context is None:
            raise ValueError(f"model provider {operation} requires a separate execution context")
        return callback(*args, execution_context=execution_context, **kwargs)
    if execution_context is not None:
        raise TypeError(f"model provider {operation} does not admit an execution context")
    return callback(*args, **kwargs)


def resolve_model_provider(state, operation):
    """Resolve an explicit execution_provider declaration or retain legacy dispatch."""
    if operation not in _provider_operations:
        raise ValueError("unknown model provider operation")
    if "execution_provider" not in state.configuration:
        from rexgraph.source_binding import CppAlgebraicSourceReference, CppSourceReference
        if isinstance(state.source, (CppSourceReference, CppAlgebraicSourceReference)):
            raise LookupError("native source models require an explicit execution provider")
        return None
    declaration = state.configuration["execution_provider"]
    if not isinstance(declaration, Mapping) or set(declaration) != {"name", "version"}:
        raise ValueError("execution provider requires exactly a name and version")
    name, version = declaration["name"], declaration["version"]
    if any(not isinstance(v, str) or not v or v.strip() != v for v in (name, version)):
        raise ValueError("execution provider name and version must be declared strings")
    with _provider_lock:
        provider = _model_providers.get((name, version))
    if provider is None:
        raise LookupError(f"model provider {name!r} version {version!r} is not registered")
    if (state.adapter, state.adapter_version) not in provider.adapters:
        raise ValueError("model adapter version is not supported by the selected provider")
    if state.arithmetic not in provider.arithmetic:
        raise ValueError("model arithmetic is not supported by the selected provider")
    if getattr(provider, operation) is None:
        raise NotImplementedError(f"model provider {name!r} does not support {operation}")
    return provider


def register_checkpoint_capture(callback: Callable) -> None:
    """Register the concrete training layer checkpoint capture implementation.

    The flow/model layers depend on this contract rather than importing ``nn.lifecycle``.
    Importing again the same implementation is harmless; a different provider is rejected so
    one process cannot silently change checkpoint semantics.
    """
    global _checkpoint_capture
    if not callable(callback):
        raise TypeError("checkpoint capture implementation must be callable")
    if _checkpoint_capture is not None and _checkpoint_capture is not callback:
        previous = (_checkpoint_capture.__module__, _checkpoint_capture.__qualname__)
        incoming = (callback.__module__, callback.__qualname__)
        if previous != incoming:
            raise RuntimeError("a checkpoint capture implementation is already registered")
    _checkpoint_capture = callback


def capture_model_checkpoint(*args, **kwargs):
    """Capture a model checkpoint through the registered lifecycle implementation.

    ``nn.lifecycle`` is imported only as the default provider bootstrap when an application
    has not imported the training layer yet.  That keeps the dependency direction at the
    contract boundary while preserving the historical ``model.checkpoint()`` convenience.
    """
    if _checkpoint_capture is None:
        import_module("rexgraph.nn.lifecycle")
    if _checkpoint_capture is None:  # pragma: no cover: defensive against broken providers
        raise RuntimeError("no model checkpoint capture implementation is registered")
    return _checkpoint_capture(*args, **kwargs)


def model_coordinates(source, grade=1):
    """Return the declared native coordinate space for a model grade."""
    from rexgraph.chain_map import CoordinateComplex

    complex_ = CoordinateComplex.from_rex(source)
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade < len(complex_.spaces):
        raise ValueError("model grade is outside the native tower")
    return complex_.spaces[grade]


__all__ = ["model_coordinates", "register_checkpoint_capture", "capture_model_checkpoint",
           "ModelProvider", "register_model_provider", "resolve_model_provider",
           "model_provider_inventory"]
