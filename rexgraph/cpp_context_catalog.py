"""Exact interpretation counts fitted from retained byte context annotations."""
from collections import OrderedDict
from copy import copy
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.cpp_correspondence import (
    _attachment, _closed, _uint, prepare_correspondence,
)
from rexgraph.cpp_model import _binding, _space, cpp_source_record, restore_cpp_source
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, _digest, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value

_PROVIDER = {"name": "rexgraph.cpp.context_catalog", "version": "1"}
_CONFIG = {"execution_provider", "engine_receipt", "carrier", "features", "labels",
           "unresolved", "pattern_source", "max_bytes", "max_samples", "max_steps"}
_SAMPLE = {"key", "feature", "context", "chart_source", "label", "partition",
           "event_time", "knowledge_time", "label_available_time"}
_provider = None
_lock = RLock()
_prepared = OrderedDict()
_contexts = OrderedDict()
_catalogs = OrderedDict()
_qualified_origins = OrderedDict()


def context_feature(scope):
    """Return the literal mention, positional role and literal context key."""
    return scope["mention_bytes"], scope["mention_side"], scope["context_bytes"]


def _feature(value):
    if (not isinstance(value, tuple) or len(value) != 3 or value[1] not in {"left", "right"}
            or any(type(value[i]) is not bytes or not value[i] for i in (0, 2))):
        raise ValueError("context feature requires two nonempty UTF8 literals and a mention role")
    for i in (0, 2):
        value[i].decode("utf-8")
    return value


def _context(raw):
    if type(raw) is not bytes:
        raise ValueError("context annotation requires a complete serialized model")
    with _lock:
        if raw in _contexts:
            _contexts.move_to_end(raw)
            return _contexts[raw]
    state = unpack_model(unpack_value(raw))
    if (not isinstance(state, ModelState) or state.adapter != "cpp_correspondence"
            or state.adapter_version != "1" or any(state.payload["counts"])):
        raise ValueError("annotation context must contain unfitted correspondence templates")
    with _lock:
        if len(raw) <= 16 << 20:
            _contexts[raw] = state
            while len(_contexts) > 64 or sum(map(len, _contexts)) > 16 << 20:
                _contexts.popitem(last=False)
    return state


def _samples(catalog, samples, event_cutoff, knowledge_cutoff, *, native=False):
    c = catalog.configuration
    event_cutoff, knowledge_cutoff = _uint(event_cutoff), _uint(knowledge_cutoff)
    if not isinstance(samples, tuple) or not samples or len(samples) > c["max_samples"]:
        raise ValueError("context training requires bounded ordered annotations")
    if len(pack_value(thaw_tree(samples))) > c["max_bytes"]:
        raise ValueError("context annotation batch byte admission exceeded")
    keys, features, labels = [], [], []
    for sample in samples:
        _closed(sample, _SAMPLE)
        key = sample["key"]
        if not isinstance(key, str) or not key or len(key) > 256 or key in keys:
            raise ValueError("context annotations require distinct declared sample identities")
        keys.append(key)
        feature, label = _uint(sample["feature"]), _uint(sample["label"])
        if feature >= len(c["features"]) or label >= len(c["labels"]):
            raise ValueError("annotation feature or interpretation differs from the catalog")
        if sample["partition"] != "training":
            raise ValueError("evaluation labels cannot update context parameters")
        event, knowledge, available = (_uint(sample[k]) for k in
                                      ("event_time", "knowledge_time", "label_available_time"))
        if event > event_cutoff or max(knowledge, available) > knowledge_cutoff:
            raise ValueError("future event, knowledge or label is unavailable")
        context = _context(sample["context"])
        if (context_feature(context.configuration["scope"]) != c["features"][feature]
                or context.configuration["labels"] != c["labels"]
                or context.configuration["unresolved"] != c["unresolved"]
                or not context.dependencies[1].matches(catalog.dependencies[0])
                or context.configuration["engine_receipt"] != c["engine_receipt"]):
            raise ValueError("annotation context differs from the declared feature or pattern owner")
        _attachment(sample["chart_source"], context.source, c["max_bytes"])
        if native:
            source = restore_cpp_source(sample["chart_source"], context.source, max_bytes=c["max_bytes"])
            prepare_correspondence(context, source)
        features.append(feature)
        labels.append(label)
    return tuple(keys), tuple(features), tuple(labels)


def validate_context_catalog(state):
    """Replay every annotation and check the retained exact parameter counts."""
    c = _closed(state.configuration, _CONFIG)
    if (state.adapter != "cpp_context_catalog" or state.adapter_version != "1"
            or state.arithmetic != "rational" or not isinstance(state.source, CppSourceReference)
            or len(state.dependencies) != 1 or not isinstance(state.dependencies[0], CppSourceReference)
            or dict(c["execution_provider"]) != _PROVIDER):
        raise ValueError("unsupported exact context catalog profile")
    _closed(c["engine_receipt"], ("engine_version", "engine_digest", "binding_digest", "binding_version", "scalar_policy"))
    from rexgraph.source_binding import _sha
    for name in ("engine_digest", "binding_digest"):
        _sha(c["engine_receipt"][name])
    for name, bound in (("max_bytes", 256 << 20), ("max_samples", 1000000), ("max_steps", 1024)):
        if not _uint(c[name], bound):
            raise ValueError("context catalog resource limits must be positive")
    features, labels = c["features"], c["labels"]
    if (not isinstance(features, tuple) or not features or len(features) > 20000
            or len(set(features)) != len(features)):
        raise ValueError("context catalog requires distinct bounded features")
    for feature in features:
        _feature(feature)
    if (not isinstance(labels, tuple) or not labels or len(labels) > 64
            or any(type(v) is not str or not v or len(v) > 256 for v in labels)
            or len(set(labels)) != len(labels) or _uint(c["unresolved"]) >= len(labels)):
        raise ValueError("context catalog requires distinct interpretations and an unresolved label")
    carrier = _uint(c["carrier"])
    if (len(state.source.native.coordinates) < 3 or len(state.source.native.coordinates[2]) != len(features)
            or state.space != _space(state.source, 2, carrier)
            or state.output_axes != (CoordinateSpace("context_interpretations", labels),)):
        raise ValueError("catalog parameter carrier differs from its declared feature columns")
    _attachment(c["pattern_source"], state.dependencies[0], c["max_bytes"])
    p = _closed(state.payload, ("counts", "history"))
    if (not isinstance(p["history"], tuple) or len(p["history"]) != state.step
            or state.step > c["max_steps"] or (state.step == 0) != (state.parent is None)):
        raise ValueError("context catalog step differs from its observation history")
    if len(pack_value(thaw_tree(p))) > c["max_bytes"]:
        raise ValueError("context annotation history byte admission exceeded")
    counts = [[0] * len(labels) for _ in features]
    used, previous, observations = set(), (0, 0), 0
    previous_parent, verification_bytes = None, 0
    configuration_bytes = len(pack_value(thaw_tree(c)))
    for position, entry in enumerate(p["history"]):
        _closed(entry, ("samples", "event_cutoff", "knowledge_cutoff", "parent"))
        _sha(entry["parent"])
        prefix_payload = {"counts": tuple(tuple(row) for row in counts), "history": p["history"][:position]}
        verification_bytes += configuration_bytes + len(pack_value(thaw_tree(prefix_payload)))
        if verification_bytes > 4*c["max_bytes"]:
            raise ValueError("context history verification work byte admission exceeded")
        prefix = copy(state)
        object.__setattr__(prefix, "payload", freeze_tree(prefix_payload))
        object.__setattr__(prefix, "step", position)
        object.__setattr__(prefix, "parent", previous_parent)
        if entry["parent"] != _digest(prefix):
            raise ValueError("context history does not follow its exact parent checkpoints")
        cutoffs = entry["event_cutoff"], entry["knowledge_cutoff"]
        if any(a < b for a, b in zip(cutoffs, previous, strict=True)):
            raise ValueError("context parameter availability cannot move backwards")
        keys, selected, targets = _samples(state, entry["samples"], *cutoffs)
        if used.intersection(keys):
            raise ValueError("context annotation was already fitted")
        used.update(keys)
        observations += len(keys)
        if observations > c["max_samples"]:
            raise ValueError("context cumulative observation limit exceeded")
        for feature, label in zip(selected, targets, strict=True):
            counts[feature][label] += 1
        previous = cutoffs
        previous_parent = entry["parent"]
    if p["counts"] != tuple(tuple(row) for row in counts):
        raise ValueError("context parameters differ from their retained annotations")
    if state.step and p["history"][-1]["parent"] != state.parent:
        raise ValueError("context child differs from its declared parent")


def prepare_context_catalog(state, source):
    """Validate native annotations and replay the checked count kernel."""
    state.bind(source)
    c = state.configuration
    if dict(c["engine_receipt"]) != _binding().build_info():
        raise ValueError("context catalog requires a different native semantic implementation")
    key = id(state), id(source)
    with _lock:
        cached = _prepared.get(key)
        if cached is not None and cached[0] is state and cached[1] is source:
            _prepared.move_to_end(key)
            return cached[2]
    validate_context_catalog(state)
    _catalog_carrier(source, c["carrier"])
    restore_cpp_source(c["pattern_source"], state.dependencies[0], max_bytes=c["max_bytes"])
    counts = tuple((0,) * len(c["labels"]) for _ in c["features"])
    for entry in state.payload["history"]:
        _, features, labels = _samples(state, entry["samples"], entry["event_cutoff"],
                                      entry["knowledge_cutoff"], native=True)
        counts = _binding().fit_context_counts(counts, features, labels, maximum=c["max_samples"])
    if counts != state.payload["counts"]:
        raise ValueError("native context parameters differ from their retained annotations")
    size = len(source.archive) + len(pack_value(thaw_tree(state.payload)))
    with _lock:
        if size <= 32 << 20:
            _prepared[key] = state, source, counts, size
            while len(_prepared) > 4 or sum(v[3] for v in _prepared.values()) > 32 << 20:
                _prepared.popitem(last=False)
    return counts


def _catalog_carrier(source, carrier):
    actual = source.carrier(carrier)
    if actual["kind"] != "source_grade" or actual["grade"] != 2 or actual["size"] != len(source.identity["coordinates"][2]):
        raise ValueError("catalog requires its exact native grade two carrier")


def cpp_context_catalog(source, pattern, *, carrier, features, labels, unresolved,
                        max_bytes=64 << 20, max_samples=1000000, max_steps=1024,
                        adapter="cpp_context_catalog", reference=None):
    """Declare a count field on explicit context columns and interpretation coordinates."""
    if adapter != "cpp_context_catalog":
        raise ValueError("unsupported context catalog adapter")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    target = CppSourceReference(pattern)
    _catalog_carrier(source, carrier)
    features, labels = tuple(tuple(v) for v in features), tuple(labels)
    c = dict(execution_provider=_PROVIDER, engine_receipt=_binding().build_info(), carrier=carrier,
        features=features, labels=labels, unresolved=unresolved, pattern_source=cpp_source_record(pattern, target),
        max_bytes=max_bytes, max_samples=max_samples, max_steps=max_steps)
    state = ModelState(adapter, "1", ref, _space(ref, 2, carrier),
        (CoordinateSpace("context_interpretations", labels),), c,
        {"counts": tuple((0,) * len(labels) for _ in features), "history": ()}, "rational", dependencies=(target,))
    prepare_context_catalog(state, source)
    enable_cpp_context_catalog_provider()
    return state


def cpp_context_annotation(key, context, source, *, feature, label, event_time,
                           knowledge_time, label_available_time, partition="training"):
    """Capture a labeled occurrence with its original chart and byte attachments."""
    context.bind(source)
    return freeze_tree(dict(key=key, feature=feature, label=label, partition=partition,
        event_time=event_time, knowledge_time=knowledge_time, label_available_time=label_available_time,
        context=pack_value(pack_model(context, native=True)), chart_source=cpp_source_record(source, context.source)))


def cpp_context_training_batch(state, source, samples, *, event_cutoff, knowledge_cutoff):
    """Capture ordered annotations without copying targets into the parameter carrier."""
    state.bind(source)
    inputs = dict(profile="exact-context-annotations-v1", samples=tuple(samples),
                  event_cutoff=event_cutoff, knowledge_cutoff=knowledge_cutoff)
    return ModelBatch(state.source, state.space, np.empty((len(state.space.keys), 0), dtype=object),
                      np.ones(len(state.space.keys), dtype=bool), inputs)


def _train(state, source, batch, *, steps):
    if steps != 1:
        raise ValueError("each context annotation batch creates exactly one parameter update")
    counts = prepare_context_catalog(state, source)
    r = _closed(batch.inputs, ("profile", "samples", "event_cutoff", "knowledge_cutoff"))
    if (r["profile"] != "exact-context-annotations-v1" or batch.dependencies or batch.target_input is not None
            or batch.targets.dtype != np.dtype(object) or batch.targets.shape != (len(state.space.keys), 0)
            or not np.all(batch.observed)):
        raise ValueError("context training batch differs from its stored annotation profile")
    _, features, labels = _samples(state, r["samples"], r["event_cutoff"], r["knowledge_cutoff"], native=True)
    candidate = _binding().fit_context_counts(counts, features, labels, maximum=state.configuration["max_samples"])
    entry = {k: r[k] for k in ("samples", "event_cutoff", "knowledge_cutoff")}
    entry["parent"] = state.coefficient_digest
    return state.child(payload={"counts": candidate, "history": (*state.payload["history"], entry)}, step=state.step+1)


def _infer(state, source, inputs):
    counts = prepare_context_catalog(state, source)
    r = _closed(inputs, ("profile", "source", "event_cutoff", "knowledge_cutoff"))
    if r["profile"] != "exact-context-catalog-input-v1" or r["source"] != state.source.as_record():
        raise ValueError("catalog inference uses another parameter owner")
    event_cutoff, knowledge_cutoff = _uint(r["event_cutoff"]), _uint(r["knowledge_cutoff"])
    if state.step:
        entry = state.payload["history"][-1]
        if event_cutoff < entry["event_cutoff"] or knowledge_cutoff < entry["knowledge_cutoff"]:
            raise ValueError("fitted context parameters are unavailable at the requested cutoff")
    c = state.configuration
    values = tuple(tuple(Fraction(n, sum(row)) for n in row) if sum(row) else
                   tuple(Fraction(int(i == c["unresolved"])) for i in range(len(row))) for row in counts)
    receipt = {"engine": c["engine_receipt"], "counts": counts, "labels": c["labels"],
               "unresolved": c["unresolved"], "carrier": c["carrier"], "event_cutoff": event_cutoff,
               "knowledge_cutoff": knowledge_cutoff}
    return ModelOutput(np.asarray(values, dtype=object), state.space, state.output_axes, state.source,
        state.coefficient_digest, "rational", "exact-context-counts-v1", state.dependencies,
        grade=2, variance="coordinate", execution_receipt=receipt)


def cpp_context_catalog_input(state, source, *, event_cutoff, knowledge_cutoff):
    """Select an available immutable context parameter publication."""
    state.bind(source)
    return freeze_tree(dict(profile="exact-context-catalog-input-v1", source=state.source.as_record(),
                            event_cutoff=event_cutoff, knowledge_cutoff=knowledge_cutoff))


def enable_cpp_context_catalog_provider():
    """Register exact context fitting and count field inference."""
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", (("cpp_context_catalog", "1"),),
                _infer, prepare=prepare_context_catalog, train=_train, create=cpp_context_catalog)
        register_model_provider(_provider)
        return _provider


def learned_context_origin(state, source, scope, *, event_cutoff, knowledge_cutoff):
    """Retain the fitted model and the selected literal context column."""
    _infer(state, source, cpp_context_catalog_input(state, source, event_cutoff=event_cutoff,
                                                   knowledge_cutoff=knowledge_cutoff))
    feature = context_feature(scope)
    index = state.configuration["features"].index(feature) if feature in state.configuration["features"] else None
    return {"catalog": pack_value(pack_model(state, native=True)), "catalog_source": cpp_source_record(source, state.source),
            "feature": index, "event_cutoff": event_cutoff, "knowledge_cutoff": knowledge_cutoff}


def validate_context_origin(origin, scope, counts, labels, unresolved, pattern, *, native=False):
    """Check learned counts against the full retained catalog and selected context."""
    _closed(origin, ("catalog", "catalog_source", "feature", "event_cutoff", "knowledge_cutoff"))
    if type(origin["catalog"]) is not bytes or len(origin["catalog"]) > 256 << 20:
        raise ValueError("learned correspondence requires a bounded complete parameter origin")
    raw = origin["catalog"]
    with _lock:
        state = _catalogs.get(raw)
        if state is not None:
            _catalogs.move_to_end(raw)
    if state is None:
        state = unpack_model(unpack_value(raw))
    if not isinstance(state, ModelState) or state.adapter != "cpp_context_catalog":
        raise ValueError("learned correspondence requires an exact context catalog")
    state.check_state()
    with _lock:
        if len(raw) <= 16 << 20:
            _catalogs[raw] = state
            while len(_catalogs) > 4 or sum(map(len, _catalogs)) > 16 << 20:
                _catalogs.popitem(last=False)
    c = state.configuration
    _attachment(origin["catalog_source"], state.source, c["max_bytes"])
    if c["labels"] != labels or c["unresolved"] != unresolved or not state.dependencies[0].matches(pattern):
        raise ValueError("learned correspondence differs from its parameter interpretation owner")
    feature = context_feature(scope)
    index = c["features"].index(feature) if feature in c["features"] else None
    if origin["feature"] != index or index is not None and type(origin["feature"]) is not int:
        raise ValueError("learned correspondence selects a different context feature")
    expected = (0,) * len(labels) if index is None else state.payload["counts"][index]
    if counts != expected:
        raise ValueError("learned correspondence parameters differ from their context observations")
    event, knowledge = _uint(origin["event_cutoff"]), _uint(origin["knowledge_cutoff"])
    if state.step:
        last = state.payload["history"][-1]
        if event < last["event_cutoff"] or knowledge < last["knowledge_cutoff"]:
            raise ValueError("learned correspondence uses unavailable fitted parameters")
    if native:
        if dict(c["engine_receipt"]) != _binding().build_info():
            raise ValueError("context origin requires a different native semantic implementation")
        with _lock:
            retained = _qualified_origins.get(raw)
            if retained is not None:
                _qualified_origins.move_to_end(raw)
        if retained is None:
            owner = restore_cpp_source(origin["catalog_source"], state.source, max_bytes=c["max_bytes"])
            prepare_context_catalog(state, owner)
            size = len(raw) + len(owner.archive)
            if size <= 32 << 20:
                with _lock:
                    _qualified_origins[raw] = state, owner, size
                    while len(_qualified_origins) > 4 or sum(v[2] for v in _qualified_origins.values()) > 32 << 20:
                        _qualified_origins.popitem(last=False)
        else:
            retained[0].bind(retained[1])
    return state


def cpp_learned_correspondence(catalog, parameter_source, source, raw, pattern, *,
                               event_cutoff, knowledge_cutoff, **declaration):
    """Prepare complete occurrence maps from the available fitted context column."""
    from rexgraph.cpp_correspondence import cpp_correspondence_model
    if set(declaration).intersection({"counts", "parameter_origin"}):
        raise ValueError("learned correspondence counts come from the selected catalog")
    origin = learned_context_origin(catalog, parameter_source, declaration["scope"],
        event_cutoff=event_cutoff, knowledge_cutoff=knowledge_cutoff)
    index = origin["feature"]
    counts = (0,) * len(catalog.configuration["labels"]) if index is None else catalog.payload["counts"][index]
    return cpp_correspondence_model(source, raw, pattern, counts=counts, parameter_origin=origin, **declaration)


def validate_context_catalog_output(output):
    """Check exact count frequencies on the original parameter carrier."""
    r = _closed(output.execution_receipt, ("engine", "counts", "labels", "unresolved", "carrier", "event_cutoff", "knowledge_cutoff"))
    if (output.arithmetic != "rational" or output.grade != 2 or output.variance != "coordinate"
            or not isinstance(output.source, CppSourceReference) or len(output.dependencies) != 1
            or not isinstance(output.dependencies[0], CppSourceReference)
            or output.space != _space(output.source, 2, _uint(r["carrier"]))
            or output.axes != (CoordinateSpace("context_interpretations", r["labels"]),)):
        raise ValueError("context output differs from its declared native parameter carrier")
    rows, labels = r["counts"], r["labels"]
    if (not isinstance(rows, tuple) or len(rows) != len(output.space.keys) or not labels
            or _uint(r["unresolved"]) >= len(labels)):
        raise ValueError("context output counts differ from its declared interpretations")
    values = []
    for row in rows:
        if len(row) != len(labels):
            raise ValueError("context count output width differs")
        for n in row:
            _uint(n)
        total = _uint(sum(row))
        values.append(tuple(Fraction(n, total) for n in row) if total else
                      tuple(Fraction(int(i == r["unresolved"])) for i in range(len(labels))))
    if output.values.shape != (len(rows), len(labels)) or tuple(output.values.flat) != tuple(v for row in values for v in row):
        raise ValueError("context frequencies differ from their retained integer counts")
    _uint(r["event_cutoff"])
    _uint(r["knowledge_cutoff"])
