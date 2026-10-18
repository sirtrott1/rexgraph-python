"""Exact learned assertion fields and native responses on retained relations."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState
from rexgraph.source_binding import CppSourceReference
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace

_PROVIDER = {"name": "rexgraph.cpp.relation_response", "version": "1"}
_PROFILE = "retained-assertion-fields-green-moments-v1"
_TRANSPORT_PROFILE = "retained-role-transport-fields-green-moments-v2"
_cache = OrderedDict()
_lock = RLock()
_provider = None


def validate_relation_response_state(state):
    """Check complete source axes and sparse learned relation fields."""
    c = state.configuration
    transported = state.adapter_version == "2"
    if (state.adapter != "cpp_relation_response" or state.adapter_version not in {"1", "2"}
            or state.arithmetic != "rational" or not isinstance(state.source, CppSourceReference)
            or c.get("profile") != (_TRANSPORT_PROFILE if transported else _PROFILE) or dict(c.get("execution_provider", {})) != _PROVIDER):
        raise ValueError("unsupported retained relation response model")
    keys = {"profile", "execution_provider", "engine_receipt", "lower_axes", "occurrence_axes",
            "parameter", "limits", "max_bytes", "labels", "training_rows", "source_accession"}
    if transported:
        keys.add("role_transports")
    if set(c) != keys:
        raise ValueError("incomplete relation response declaration")
    labels = tuple(c["labels"])
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(s, str) or not s for s in labels):
        raise ValueError("distinct semantic label identities required")
    if (state.space.keys != state.source.native.coordinates[1] or state.output_axes
            or len(c["lower_axes"]) != len(state.source.native.coordinates[0])):
        raise ValueError("relation model must retain original source coordinates")
    if not isinstance(c["parameter"], Fraction) or c["parameter"] <= 0:
        raise ValueError("positive rational Green parameter required")
    if (len(c["limits"]) != 4 or any(type(v) is not int or v <= 0 for v in c["limits"])
            or type(c["max_bytes"]) is not int or c["max_bytes"] <= 0):
        raise ValueError("positive action and archive bounds required")
    if transported:
        from rexgraph.cpp_role_transports import validate_role_transports
        validate_role_transports(c["role_transports"], c)
    accession = c["source_accession"]
    if accession is not None and (not isinstance(accession, Mapping)
            or set(accession) != {"lower", "relations", "occurrences", "charts"}
            or len(accession["lower"]) != len(c["lower_axes"])
            or len(accession["relations"]) != len(state.space.keys)
            or len(accession["occurrences"]) != len(c["occurrence_axes"])):
        raise ValueError("source accession must cover every retained cell and occurrence")
    rows = tuple(c["training_rows"])
    if len(set(rows)) != len(rows) or any(type(i) is not int or not 0 <= i < len(state.space.keys) for i in rows):
        raise ValueError("training rows must name original primary relations")
    if set(state.payload) != {"fields", "batches"} or set(state.payload["fields"]) != set(labels):
        raise ValueError("one learned relation field required per declared label")
    for entries in state.payload["fields"].values():
        indices = []
        for i, count in entries:
            if type(i) is not int or i not in rows or type(count) is not int or count <= 0:
                raise ValueError("learned field outside its retained training assertions")
            indices.append(i)
        if indices != sorted(set(indices)):
            raise ValueError("learned relation field coordinates must be unique and ordered")
    if not isinstance(state.payload["batches"], tuple) or len(state.payload["batches"]) != state.step:
        raise ValueError("learning step and retained annotation batches differ")


def _prepare(state, source):
    from rexgraph_cpp import PreparedRelationResponse, relation_build_info
    validate_relation_response_state(state)
    state.bind(source)
    c = state.configuration
    if dict(c["engine_receipt"]) != relation_build_info():
        raise ValueError("relation model requires its retained native implementation")
    if len(source.archive) > c["max_bytes"]:
        raise ValueError("model source exceeds its retained byte admission")
    key = state.source.native.archive_digest, tuple(c["lower_axes"]), tuple(c["occurrence_axes"]), c["parameter"], tuple(c["limits"]), c["max_bytes"], c.get("role_transports")
    with _lock:
        plan = _cache.get(key)
        if plan is None:
            options = {"role_transports": c["role_transports"]} if state.adapter_version == "2" else {}
            plan = PreparedRelationResponse(source.archive, c["lower_axes"], c["occurrence_axes"],
                parameter=c["parameter"], limits=c["limits"], max_bytes=c["max_bytes"], **options)
            _cache[key] = plan
            while len(_cache) > 4:
                _cache.popitem(last=False)
        _cache.move_to_end(key)
        return plan


def clear_relation_response_cache():
    """Release prepared response owners without changing learned fields."""
    with _lock:
        _cache.clear()


def _train(state, source, batch, *, steps):
    _prepare(state, source)
    if steps != 1:
        raise ValueError("one observation batch is one assertion-field update")
    if (not isinstance(batch.inputs, Mapping) or dict(batch.inputs) != {
            "profile": "retained-relation-annotations-v1", "partition": "training"}):
        raise ValueError("training requires a declared annotation partition")
    if batch.targets.shape != (len(state.space.keys),):
        raise ValueError("one semantic label per observed primary relation required")
    fields = {label: dict(entries) for label, entries in state.payload["fields"].items()}
    admitted = set(state.configuration["training_rows"])
    labels = state.configuration["labels"]
    for i in np.flatnonzero(batch.observed):
        i = int(i)
        label_index = batch.targets[i]
        if (i not in admitted or isinstance(label_index, (bool, np.bool_))
                or not isinstance(label_index, (int, np.integer)) or not 0 <= label_index < len(labels)):
            raise ValueError("annotation outside declared training relations or semantic labels")
        label = labels[int(label_index)]
        fields[label][i] = fields[label].get(i, 0) + 1
    return state.child(payload={"fields": {s: tuple(sorted(entries.items())) for s, entries in fields.items()},
        "batches": (*state.payload["batches"], batch.coefficient_digest)}, step=state.step + 1)


def _infer(state, source, inputs):
    plan = _prepare(state, source)
    if not isinstance(inputs, TensorField) or inputs.source is None or not inputs.source.matches(state.source):
        raise TypeError("response input requires an exact source-bound relation field")
    inputs.bind(source)
    if inputs.space != state.space or inputs.axes or inputs.grade != 1 or inputs.variance != "chain":
        raise ValueError("response input must cover the original primary relation stalks")
    fields = [tuple(inputs.values)]
    for label in state.configuration["labels"]:
        values = [0] * len(state.space.keys)
        for i, count in state.payload["fields"][label]:
            values[i] = count
        fields.append(tuple(values))
    result = plan.moments(fields)
    gram = result["gram"]
    scores = tuple(Fraction(0) if gram[0][0] == 0 or gram[i][i] == 0
                   else gram[0][i] ** 2 / (gram[0][0] * gram[i][i]) for i in range(1, len(fields)))
    if any(s < 0 or s > 1 for s in scores):
        raise RuntimeError("Green moment coupling violates its exact quadrance bound")
    receipt = dict(result, readout="normalized-squared-Green-source-pairing-v1",
        label_fields=state.payload["fields"], metric_interpretation="declared-unit-stalk-metrics",
        truth_status="prospective", input_digest=inputs.coefficient_digest,
        source_reference=state.source.as_record())
    return ModelOutput(scores, CoordinateSpace("relational-semantic-labels", state.configuration["labels"]), (),
        state.source, state.coefficient_digest, "rational", state.configuration["profile"], execution_receipt=receipt)


def enable_relation_response_provider():
    """Register exact relational field learning, response and preparation."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], _PROVIDER["version"], (("cpp_relation_response", "1"),("cpp_relation_response", "2")),
                _infer, create=_create, prepare=_prepare, train=_train)
        register_model_provider(_provider)
        return _provider


def cpp_relation_response_model(source, lower_axes, occurrence_axes, labels, training_rows, *,
                                parameter=Fraction(1), limits=(4096, 16000000, 64000000, 1 << 24),
                                max_bytes=256 * 1024 * 1024, source_accession=None, reference=None,
                                role_transports=None):
    """Declare learned fields on primary relations with original incidence roles."""
    from rexgraph_cpp import relation_build_info
    reference = CppSourceReference(source) if reference is None else reference.bind(source)
    labels = tuple(labels)
    c = {"profile": _PROFILE, "execution_provider": _PROVIDER, "engine_receipt": relation_build_info(),
         "lower_axes": tuple(tuple(row) for row in lower_axes), "occurrence_axes": tuple(occurrence_axes),
         "labels": labels, "training_rows": tuple(training_rows), "parameter": parameter,
         "limits": tuple(limits), "max_bytes": max_bytes, "source_accession": source_accession}
    if role_transports is not None:
        from rexgraph.cpp_role_transports import normalize_role_transports
        c.update(profile=_TRANSPORT_PROFILE, role_transports=normalize_role_transports(role_transports))
    state = ModelState("cpp_relation_response", "2" if role_transports is not None else "1", reference,
        CoordinateSpace("retained-context-relations", reference.native.coordinates[1]), (), c,
        {"fields": {s: () for s in labels}, "batches": ()}, "rational")
    enable_relation_response_provider()
    validate_relation_response_state(state)
    return state


def _create(source, *, adapter="cpp_relation_response", reference=None, **configuration):
    if adapter != "cpp_relation_response":
        raise ValueError("unsupported relation response adapter")
    return cpp_relation_response_model(source, reference=reference, **configuration)


def relation_annotation_batch(state, annotations):
    """Bind semantic observations to retained relation indices without token features."""
    values = np.full(len(state.space.keys), -1, dtype=np.int64)
    observed = np.zeros(len(values), dtype=bool)
    for i, label in annotations:
        if type(i) is not int or not 0 <= i < len(values) or observed[i]:
            raise ValueError("annotation coordinates must be original and unique")
        if not isinstance(label, str) or label not in state.configuration["labels"]:
            raise ValueError("annotation must name a declared semantic label")
        values[i] = state.configuration["labels"].index(label)
        observed[i] = True
    return ModelBatch(state.source, state.space, values, observed,
        {"profile": "retained-relation-annotations-v1", "partition": "training"})
