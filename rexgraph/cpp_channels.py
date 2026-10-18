"""Exact scalar channel models on retained native primary relations."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.cpp_stalk_fields import _exact
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "rational-relational-channels-v1"
_METHOD = "exact-relational-channels-v1"
_PROVIDER = {"name": "rexgraph.cpp.channels", "version": "1"}
_CHANNELS = ("T", "G", "F", "C")
_lock = RLock()
_cache = OrderedDict()
_provider = None


def _space(reference):
    return CoordinateSpace("primary-relations", reference.native.coordinates[1])


def _axes():
    return (CoordinateSpace("relational-channel", _CHANNELS),)


def _factors(configuration, reference):
    p = configuration["native_plan"]
    required = {"build", "source_archive", "grade", "field_metric", "geometry", "coparticipation", "weights",
                "dimensions", "occurrences", "limits", "retained_entries", "preparation_steps", "traces", "diagonals",
                "signed_incidence", "occurrence_cells", "occurrence_keys", "occurrence_activity", "channels"}
    n = len(reference.native.coordinates[1]); m = len(reference.native.coordinates[0])
    bits = configuration["limits"][3]
    if (not isinstance(p, Mapping) or set(p) != required or dict(p["build"]) != dict(configuration["engine_receipt"]) or
            p["source_archive"] != reference.native.archive_digest or type(p["grade"]) is not int or p["grade"] != 1 or
            p["field_metric"] != "unit" or p["geometry"] != "raw" or p["channels"] != _CHANNELS or
            p["weights"] != configuration["weights"] or p["coparticipation"] != configuration["coparticipation"] or
            p["limits"] != configuration["limits"] or p["dimensions"] != (m, n) or
            any(type(i) is not int for i in p["dimensions"]) or
            type(p["occurrences"]) is not int or not 0 <= p["occurrences"] <= configuration["limits"][1] or
            any(type(p[k]) is not int or not 0 <= p[k] <= configuration["limits"][i]
                for k, i in (("retained_entries", 1), ("preparation_steps", 2)))):
        raise ValueError("native channel plan differs from its source and exact profile")
    for q in p["weights"]:
        _exact(q,bits)
    count = p["occurrences"]
    if (any(len(p[k]) != count for k in ("signed_incidence", "occurrence_cells", "occurrence_keys", "occurrence_activity")) or
            max(m,n) > configuration["limits"][0] or p["retained_entries"] != 2*count+6*n+4 or
            len(set(p["occurrence_keys"])) != count or
            any(type(v) is not int or not 0 <= v < (1 << 64) for v in p["occurrence_keys"]) or
            any(type(v) is not bool for v in p["occurrence_activity"])):
        raise ValueError("complete original channel occurrences required")
    seen = set(); original = []
    for cell, q, active in zip(p["occurrence_cells"], p["signed_incidence"], p["occurrence_activity"], strict=True):
        if (len(cell) != 2 or any(type(i) is not int for i in cell) or
                not 0 <= cell[0] < n or not 0 <= cell[1] < m or cell in seen):
            raise ValueError("distinct original participants and valid grade coordinates required")
        seen.add(cell); q = _exact(q, bits)
        if not active and q:
            raise ValueError("inactive original flag has a nonzero realized channel coefficient")
        original.append((cell[0], cell[1], q))
    if len(p["traces"]) != 4 or len(p["diagonals"]) != 4 or any(len(row) != n for row in p["diagonals"]):
        raise ValueError("four complete original channel diagonals required")
    for row, trace in zip(p["diagonals"], p["traces"], strict=True):
        values = tuple(_exact(q, bits) for q in row)
        if any(q < 0 for q in values) or _exact(trace, bits) != _exact(sum(values, Fraction(0)), bits):
            raise ValueError("channel trace differs from its nonnegative exact diagonal")
    positive = [Fraction(0)]*m; negative = [Fraction(0)]*m; c_mass = [Fraction(0)]*m
    expected = [[Fraction(0)]*n for _ in range(4)]; weights = configuration["weights"]
    for up, lo, q in original:
        a = abs(q); share = Fraction(bool(q)) if configuration["coparticipation"] == "count" else a
        mass = negative if q < 0 else positive
        mass[lo] = _exact(mass[lo]+_exact(a*abs(weights[up]),bits),bits)
        c_mass[lo] = _exact(c_mass[lo]+share,bits)
        weighted = _exact(weights[up]*q,bits)
        expected[0][up] = _exact(expected[0][up]+_exact(weighted*weighted,bits),bits)
    expected[1] = expected[0].copy()
    for up, lo, q in original:
        a = abs(q); share = Fraction(bool(q)) if configuration["coparticipation"] == "count" else a
        opposite = positive[lo] if q < 0 else negative[lo]
        value = _exact(_exact(2*abs(weights[up])*a,bits)*opposite,bits)
        expected[2][up] = _exact(expected[2][up]+value,bits)
        expected[3][up] = _exact(expected[3][up]+_exact(share*(c_mass[lo]-share),bits),bits)
    if tuple(tuple(row) for row in expected) != p["diagonals"]:
        raise ValueError("channel diagonal differs from the retained signed incidence and weights")
    return tuple(original)


def validate_channel_state(state):
    """Check the exact profile, original source and complete native preparation receipt."""
    c = state.configuration
    if (state.adapter != "cpp_channels" or state.adapter_version != "1" or state.arithmetic != "rational" or
            not isinstance(state.source, CppSourceReference) or len(state.source.native.coordinates) < 2 or
            set(c) != {"profile", "execution_provider", "engine_receipt", "weights", "coparticipation", "limits",
                       "max_bytes", "trace_normalized", "source_reference", "native_plan"} or
            c["profile"] != _PROFILE or dict(c["execution_provider"]) != _PROVIDER or
            dict(c["source_reference"]) != state.source.as_record() or state.space != _space(state.source) or
            state.output_axes != _axes() or state.step != 0 or state.payload):
        raise ValueError("closed exact scalar channel model required")
    if (len(c["limits"]) != 4 or any(type(v) is not int or not 0 < v < (1 << 64) for v in c["limits"]) or
            type(c["max_bytes"]) is not int or not 0 < c["max_bytes"] < (1 << 64) or
            type(c["trace_normalized"]) is not bool or c["coparticipation"] not in {"share", "count"} or
            len(c["weights"]) != len(state.space.keys)):
        raise ValueError("positive admissions and complete original relation weights required")
    for q in c["weights"]:
        _exact(q, c["limits"][3])
    _factors(c, state.source)


def _prepare(state, source):
    from rexgraph_cpp import PreparedRelationalChannels, channels_build_info
    validate_channel_state(state); state.bind(source); c = state.configuration
    if dict(c["engine_receipt"]) != channels_build_info():
        raise ValueError("channel model requires its retained native implementation")
    key = (state.source.native.archive_digest, c["weights"], c["coparticipation"], c["limits"], c["max_bytes"])
    with _lock:
        plan = _cache.get(key)
        if plan is None:
            plan = PreparedRelationalChannels(source.archive, c["weights"], coparticipation=c["coparticipation"],
                                             limits=c["limits"], max_bytes=c["max_bytes"])
            _cache[key] = plan
            while len(_cache) > 4:
                _cache.popitem(last=False)
        if freeze_tree(plan.receipt) != c["native_plan"]:
            raise ValueError("channel preparation differs from the retained original source")
        return plan


def _field(reference, field, bits):
    values = []; width = len(reference.native.coordinates[1])
    for value in field:
        if len(values) >= width:
            raise ValueError("field exceeds original relation carrier")
        values.append(_exact(value,bits))
    if len(values) != width:
        raise ValueError("complete original relation field required")
    return tuple(values)


def cpp_channel_model(source, weights=None, *, coparticipation="share", trace_normalized=False,
                      limits=(4096, 16000000, 64000000, 16384), max_bytes=256*1024*1024, reference=None):
    """Declare raw rational T/G/F/C over the original grade one carrier and unit field metric."""
    from rexgraph_cpp import PreparedRelationalChannels, channels_build_info
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    limits = tuple(limits)
    if len(limits) != 4 or any(type(v) is not int or v <= 0 for v in limits):
        raise ValueError("four positive exact action limits required")
    if len(ref.native.coordinates) < 2:
        raise ValueError("grade-one relational source required")
    weights = tuple(Fraction(1) for _ in ref.native.coordinates[1]) if weights is None else _field(ref, weights, limits[3])
    plan = PreparedRelationalChannels(source.archive, weights, coparticipation=coparticipation, limits=limits, max_bytes=max_bytes)
    c = dict(profile=_PROFILE, execution_provider=_PROVIDER, engine_receipt=channels_build_info(),
             weights=weights, coparticipation=coparticipation, trace_normalized=trace_normalized,
             limits=limits, max_bytes=max_bytes, source_reference=ref.as_record(), native_plan=plan.receipt)
    state = ModelState("cpp_channels", "1", ref, _space(ref), _axes(), c, {}, "rational")
    enable_channel_provider(); _prepare(state, source)
    return state


def relational_channel_query(state, field):
    """Bind a complete rational field to the original relation coordinates."""
    validate_channel_state(state)
    values = _field(state.source, field, state.configuration["limits"][3])
    return dict(profile="primary-relational-channel-query-v1", source_reference=state.source.as_record(), values=values)


def _query(state, query):
    if (not isinstance(query, Mapping) or set(query) != {"profile", "source_reference", "values"} or
            query["profile"] != "primary-relational-channel-query-v1" or
            dict(query["source_reference"]) != state.source.as_record()):
        raise ValueError("channel query requires the actual original source")
    return freeze_tree(relational_channel_query(state, query["values"]))


def _infer(state, source, inputs):
    from rexgraph.model_codec import pack_model
    plan = _prepare(state, source); query = _query(state, inputs)
    result = plan.apply(query["values"], trace_normalized=state.configuration["trace_normalized"])
    values = np.empty((len(state.space.keys), 4), dtype=object)
    for k, field in enumerate(result["fields"]):
        values[:, k] = field
    receipt = dict(profile=_PROFILE, model_checkpoint=pack_model(state), query_declaration=query, response=result)
    return ModelOutput(values, state.space, state.output_axes, state.source, state.coefficient_digest, "rational", _METHOD,
                       grade=1, variance="chain", execution_receipt=receipt)


def validate_channel_output(output):
    """Verify the retained factored equations and producing model without executing a backend."""
    from rexgraph.model_codec import unpack_model
    r = output.execution_receipt
    if not isinstance(r, Mapping) or set(r) != {"profile", "model_checkpoint", "query_declaration", "response"} or r["profile"] != _PROFILE:
        raise ValueError("complete channel output receipt required")
    state = unpack_model(r["model_checkpoint"]); validate_channel_state(state)
    if (output.arithmetic != "rational" or output.grade != 1 or output.variance != "chain" or
            output.model_digest != state.coefficient_digest or not output.source.matches(state.source) or
            output.space != state.space or output.axes != state.output_axes):
        raise ValueError("channel output differs from its producing model")
    c = state.configuration; query = _query(state, r["query_declaration"]); result = r["response"]
    if (set(result) != {"profile", "source_archive", "fields", "energies", "trace_normalized", "steps", "channels"} or
            result["profile"] != "rational-relational-channel-response-v1" or
            result["source_archive"] != state.source.native.archive_digest or result["channels"] != _CHANNELS or
            result["trace_normalized"] is not c["trace_normalized"] or type(result["steps"]) is not int or
            not 0 <= result["steps"] <= c["limits"][2] or len(result["fields"]) != 4 or len(result["energies"]) != 4):
        raise ValueError("native response differs from the declared channel profile")
    n = len(state.space.keys); m = c["native_plan"]["dimensions"][0]; bits = c["limits"][3]
    x = query["values"]; w = c["weights"]; flags = _factors(c, state.source)
    rows = [[Fraction(0)]*m for _ in range(3)]
    expected = [[Fraction(0)]*n for _ in range(4)]; cmass = [Fraction(0)]*m
    for up, lo, q in flags:
        a = abs(q); share = Fraction(bool(q)) if c["coparticipation"] == "count" else a
        for k, value in enumerate((q*w[up]*x[up], a*w[up]*x[up], share*x[up])):
            rows[k][lo] = _exact(rows[k][lo]+_exact(value,bits),bits)
        cmass[lo] = _exact(cmass[lo]+share,bits)
    for up, lo, q in flags:
        a = abs(q); share = Fraction(bool(q)) if c["coparticipation"] == "count" else a
        for k, value in ((0,q*w[up]*rows[0][lo]), (1,a*w[up]*rows[1][lo]),
                         (3,share*(cmass[lo]*x[up]-rows[2][lo]))):
            expected[k][up] = _exact(expected[k][up]+_exact(value,bits),bits)
    for i in range(n):
        expected[2][i] = _exact(expected[0][i]-expected[1][i]+c["native_plan"]["diagonals"][2][i]*x[i],bits)
    for k, field in enumerate(expected):
        trace = c["native_plan"]["traces"][k]
        if c["trace_normalized"] and trace:
            field[:] = [_exact(q/trace,bits) for q in field]
        if (len(result["fields"][k]) != n or tuple(field) != tuple(result["fields"][k]) or
                _exact(result["energies"][k],bits) != _exact(sum((a*b for a,b in zip(x,field,strict=True)),Fraction(0)),bits) or
                tuple(output.values[:,k]) != tuple(field)):
            raise ValueError("channel field or energy differs from its retained exact equations")
        for q in result["fields"][k]:
            _exact(q,bits)


def _create(source, *, adapter="cpp_channels", **configuration):
    if adapter != "cpp_channels":
        raise ValueError("unsupported relational channel adapter")
    return cpp_channel_model(source, **configuration)


def enable_channel_provider():
    """Register owning exact scalar channel preparation and full field inference."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], _PROVIDER["version"], (("cpp_channels", "1"),),
                                      _infer, create=_create, prepare=_prepare)
        register_model_provider(_provider)
        return _provider
