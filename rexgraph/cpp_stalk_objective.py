"""Conditional exact field actions within complete observed transport families."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.cpp_stalk_transport import (
    _exact, _field, _prepare as _transport_plan, _query, _receipt_matches,
    _train as _transport_train, validate_stalk_transport_state,
)
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "conditional-stalk-field-objective-v1"
_METHOD = "exact-conditional-stalk-field-objective-v1"
_PROVIDER = dict(name="rexgraph.cpp.stalk_objective", version="1")
_lock = RLock()
_cache = OrderedDict()
_provider = None


def _evidence(state):
    from rexgraph.model_codec import _checkpoint_model
    evidence = _checkpoint_model(state.payload["evidence_checkpoint"])
    if not isinstance(evidence, ModelState) or evidence.adapter != "cpp_stalk_transport":
        raise ValueError("original affine transport evidence checkpoint required")
    return evidence


def validate_stalk_objective_state(state):
    """Check declared views, probes and the separate observed transport checkpoint."""
    c = state.configuration
    if (state.adapter != "cpp_stalk_objective" or state.adapter_version != "1" or state.arithmetic != "rational" or
            state.dependencies or set(c) != {"profile", "execution_provider", "engine_receipt", "probes", "views",
                "weights", "channel_coefficients", "mass", "c_reading", "trace_normalized"} or
            c["profile"] != _PROFILE or dict(c["execution_provider"]) != _PROVIDER or
            set(state.payload) != {"evidence_checkpoint", "selection"} or
            not isinstance(state.payload["evidence_checkpoint"], Mapping)):
        raise ValueError("closed conditional field objective model required")
    evidence = _evidence(state)
    if (not state.source.matches(evidence.source) or state.space != evidence.space or
            state.output_axes != evidence.output_axes or state.step != evidence.step):
        raise ValueError("objective and evidence must retain the same actual source and stalks")
    bounds = evidence.configuration["limits"]
    if (not c["probes"] or not c["views"] or len(c["probes"]) > bounds[0] or len(c["views"]) > bounds[0] or
            type(c["trace_normalized"]) is not bool or c["c_reading"] not in {"share", "count"} or
            len(c["weights"]) != len(evidence.configuration["stalk_axes"][1]) or len(c["channel_coefficients"]) != 4):
        raise ValueError("bounded whole probes, views and declared channel readings required")
    for field in c["probes"]:
        _field(evidence.configuration, 1, field)
    for value in (*c["weights"], *c["channel_coefficients"], c["mass"]):
        _exact(value, bounds[3])
    if c["mass"] < 0 or any(v < 0 for v in c["channel_coefficients"]):
        raise ValueError("nonnegative field action coefficients required")
    cells = evidence.payload["fit"]["occurrence_cells"][1]
    names = set(); entries = sum(sum(map(len, p)) for p in c["probes"])
    for name, terms in c["views"]:
        if type(name) is not str or not name.strip() or len(name.encode()) > 256 or name in names or not terms:
            raise ValueError("distinct bounded view names and explicit occurrence support required")
        names.add(name); support = set()
        for occurrence, axis, value in terms:
            if (type(occurrence) is not int or type(axis) is not int or not 0 <= occurrence < len(cells) or
                    not 0 <= axis < len(evidence.configuration["stalk_axes"][0][cells[occurrence][1]]) or
                    (occurrence, axis) in support):
                raise ValueError("distinct original occurrence stalk directions required")
            _exact(value, bounds[3]); support.add((occurrence, axis)); entries += 3
    if entries > bounds[1]:
        raise ValueError("objective declarations exceed entry admission")
    r = state.payload["selection"]
    if (not isinstance(r, Mapping) or r.get("interpretation") != "conditional-field-action-family-v1" or
            not _receipt_matches(evidence.payload["fit"], r.get("evidence_fit"))):
        raise ValueError("conditional selection must retain the complete observed family")
    for key in ("probes", "views", "weights", "channel_coefficients", "mass", "c_reading", "trace_normalized"):
        if not _receipt_matches(c[key], r.get(key)):
            raise ValueError("conditional selection differs from its declared field objective")


def _native(evidence, source, c):
    from rexgraph_cpp import PreparedStalkFieldObjective
    return PreparedStalkFieldObjective(_transport_plan(evidence, source), c["probes"], c["views"],
        **{k:c[k] for k in ("weights", "channel_coefficients", "mass", "c_reading", "trace_normalized")})


def _prepare(state, source):
    from rexgraph_cpp import stalk_objective_build_info
    state.bind(source)
    if dict(state.configuration["engine_receipt"]) != stalk_objective_build_info():
        raise ValueError("field objective requires its retained native implementation")
    digest = state._digest
    with _lock:
        plan = _cache.get(digest)
        if plan is None:
            validate_stalk_objective_state(state)
            plan = _native(_evidence(state), source, state.configuration)
            if not _receipt_matches(plan.receipt, state.payload["selection"]):
                raise ValueError("field selection differs from exact native replay")
            _cache[digest] = plan
            while len(_cache) > 4:
                _cache.popitem(last=False)
        _cache.move_to_end(digest)
        return plan


def cpp_stalk_objective_model(source, evidence, probes, views, *, weights=None,
                              channel_coefficients=(0, 0, 0, 0), mass=0,
                              c_reading="share", trace_normalized=False, reference=None):
    """Select a complete conditional family without changing the observed family."""
    from rexgraph.model_codec import _model_checkpoint
    from rexgraph_cpp import stalk_objective_build_info
    validate_stalk_transport_state(evidence); evidence.bind(source)
    if reference is not None and not reference.bind(source).matches(evidence.source):
        raise ValueError("objective must use the actual observed source reference")
    bits = evidence.configuration["limits"][3]
    c = dict(profile=_PROFILE, execution_provider=_PROVIDER, engine_receipt=stalk_objective_build_info(),
        probes=tuple(_field(evidence.configuration, 1, p) for p in probes),
        views=tuple((name, tuple((p, a, _exact(q, bits)) for p, a, q in terms)) for name, terms in views),
        weights=tuple(_exact(v, bits) for v in ((1,)*len(evidence.configuration["stalk_axes"][1]) if weights is None else weights)),
        channel_coefficients=tuple(_exact(v, bits) for v in channel_coefficients), mass=_exact(mass, bits),
        c_reading=c_reading, trace_normalized=trace_normalized)
    plan = _native(evidence, source, c)
    state = ModelState("cpp_stalk_objective", "1", evidence.source, evidence.space, evidence.output_axes,
        c, dict(evidence_checkpoint=_model_checkpoint(evidence), selection=plan.receipt), "rational", evidence.step)
    enable_stalk_objective_provider()
    with _lock:
        _cache[state.coefficient_digest] = plan
        while len(_cache) > 4:
            _cache.popitem(last=False)
    return state


def _responses(state, source, inputs):
    evidence = _evidence(state); query = _query(evidence, inputs)
    original = _transport_plan(evidence, source).apply(query["field"])
    selected = _prepare(state, source).apply(query["field"])
    axes = (*state.output_axes, CoordinateSpace("conditional-affine-component", selected["components"]))
    names = evidence.configuration["stalk_axes"][0]; width = len(axes[0].keys)
    values = np.full((len(names), width, len(selected["components"])), Fraction(0), dtype=object)
    for j, family in enumerate(selected["boundary_family"]):
        for i, vector in enumerate(family):
            values[i, :len(vector), j] = vector
    masks = dict(stalk_support=tuple(tuple(i < len(row) for i in range(width)) for row in names))
    for label, response in (("evidence", original), ("conditional", selected)):
        for carrier in ("boundary", "occurrence"):
            family = response[carrier+"_family"]
            masks[label+"_"+carrier+"_fixed"] = tuple(tuple(
                all(f[row][axis] == 0 for f in family[1:]) for axis in range(len(vector)))
                for row, vector in enumerate(family[0]))
    return query, original, selected, axes, values, masks


def _infer(state, source, inputs):
    from rexgraph.model_codec import _model_checkpoint
    query, original, selected, axes, values, masks = _responses(state, source, inputs)
    receipt = dict(profile=_PROFILE, model_checkpoint=_model_checkpoint(state), source_archive=source.archive,
        query_declaration=query, evidence_response=original, conditional_response=selected, field_masks=masks,
        interpretation="objective-selection-with-retained-evidence-v1")
    return ModelOutput(values, state.space, axes, state.source, state.coefficient_digest,
        "rational", _METHOD, grade=0, variance="chain", execution_receipt=receipt)


def validate_stalk_objective_output(output):
    """Replay both complete response families on their original source."""
    from rexgraph.model_codec import _checkpoint_model
    from rexgraph_cpp import NativeSource
    r = output.execution_receipt
    if (not isinstance(r, Mapping) or set(r) != {"profile", "model_checkpoint", "source_archive", "query_declaration",
            "evidence_response", "conditional_response", "field_masks", "interpretation"} or r["profile"] != _PROFILE or
            r["interpretation"] != "objective-selection-with-retained-evidence-v1"):
        raise ValueError("complete observed and conditional field response receipt required")
    state = _checkpoint_model(r["model_checkpoint"])
    if (not isinstance(state, ModelState) or state.adapter != "cpp_stalk_objective" or
            output.arithmetic != "rational" or output.dependencies or output.method != _METHOD or
            output.grade != 0 or output.variance != "chain" or not output.source.matches(state.source) or
            output.model_digest != state.coefficient_digest or output.space != state.space):
        raise ValueError("conditional output differs from its retained source and model")
    maximum = _evidence(state).configuration["max_bytes"]
    if type(r["source_archive"]) is not bytes or len(r["source_archive"]) > maximum:
        raise ValueError("bounded original source archive required")
    source = NativeSource(r["source_archive"], max_bytes=maximum)
    _, original, selected, axes, values, masks = _responses(state, source, r["query_declaration"])
    if (axes != output.axes or not np.array_equal(values, output.values) or
            not _receipt_matches(original, r["evidence_response"]) or
            not _receipt_matches(selected, r["conditional_response"]) or not _receipt_matches(masks, r["field_masks"])):
        raise ValueError("conditional output differs from complete native field replay")


def stalk_objective_field(output, row=None, *, interpretation):
    """Expose a fixed whole field from the explicitly selected response family."""
    from hashlib import sha256
    from rexgraph.model_codec import _checkpoint_model
    output.check_state(); validate_stalk_objective_output(output)
    if interpretation not in {"evidence", "conditional"}:
        raise ValueError("explicit evidence or conditional interpretation required")
    r = output.execution_receipt
    names = _evidence(_checkpoint_model(r["model_checkpoint"])).configuration["stalk_axes"][0]
    if row is not None and (type(row) is not int or not 0 <= row < len(names)):
        raise ValueError("select an original lower stalk row")
    selected = range(len(names)) if row is None else (row,)
    if not all(all(r["field_masks"][interpretation+"_boundary_fixed"][i]) for i in selected):
        raise TypeError("selected whole field still depends on unresolved transport directions")
    fields = r[interpretation+"_response"]["boundary_family"][0]
    if row is None:
        local = names[0] if names else ()
        if any(n != local for n in names):
            raise TypeError("ragged stalks require an explicit original lower row")
        space = output.space; values = np.asarray(fields, dtype=object).reshape(len(names), len(local))
    else:
        local = names[row]; space = CoordinateSpace(output.space.name, (output.space.keys[row],))
        values = np.asarray((fields[row],), dtype=object)
    marker = sha256(("stalk-field-interpretation-v1:"+interpretation).encode()).hexdigest()
    return TensorField(space, values, (CoordinateSpace("named-stalk-axis", local),), output.source,
        grade=0, variance="chain", provenance=(output.model_digest, output.coefficient_digest, marker))


def _train(state, source, batch, *, steps):
    from rexgraph.model_codec import _model_checkpoint
    _prepare(state, source)
    evidence = _transport_train(_evidence(state), source, batch, steps=steps)
    plan = _native(evidence, source, state.configuration)
    child = state.child(payload=dict(evidence_checkpoint=_model_checkpoint(evidence), selection=plan.receipt), step=evidence.step)
    with _lock:
        _cache[child.coefficient_digest] = plan
        while len(_cache) > 4:
            _cache.popitem(last=False)
    return child


def _create(source, *, adapter="cpp_stalk_objective", **configuration):
    if adapter != "cpp_stalk_objective":
        raise ValueError("unsupported conditional field objective adapter")
    return cpp_stalk_objective_model(source, **configuration)


def enable_stalk_objective_provider():
    """Register observed field acquisition and conditional whole field inference."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", (("cpp_stalk_objective", "1"),),
                _infer, create=_create, prepare=_prepare, train=_train)
            register_model_provider(_provider)
        return _provider
