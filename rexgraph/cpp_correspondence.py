"""Exact context correspondences over retained byte and pattern publications."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from math import lcm
from threading import RLock

import numpy as np

from rexgraph.cpp_model import _binding, _space, cpp_source_record, restore_cpp_source
from rexgraph.exact_value import exact_fraction
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.value_codec import pack_value

_PROVIDER = {"name": "rexgraph.cpp.correspondence", "version": "1"}
_CONFIG = {"execution_provider", "engine_receipt", "grade", "carriers", "raw_carriers",
           "pattern_carriers", "raw_source", "pattern_source", "provenance", "scope",
           "templates", "labels", "unresolved", "limits", "max_bytes"}
_provider = None
_learned_provider = None
_lock = RLock()
_cache = OrderedDict()


def _uint(value, maximum=(1 << 64)-1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("correspondence identifiers and counts require bounded integers")
    return value


def _closed(value, names):
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError("unsupported correspondence declaration fields")
    return value


def _family(value, source, target, limits):
    if isinstance(value, (str, bytes)) or len(value) != len(source.native.coordinates):
        raise ValueError("correspondence requires its complete grade family")
    if len(target.native.coordinates) < len(value):
        raise ValueError("correspondence target has insufficient grades")
    result, records = [], len(value)
    for k, row in enumerate(value):
        if len(row) != len(source.native.coordinates[k]):
            raise ValueError("correspondence map differs from source coordinates")
        records += len(row)
        images = []
        for image in row:
            terms = []
            records += len(image)
            if records > limits[0]:
                raise ValueError("correspondence map record bound exceeded")
            for term in image:
                if len(term) != 2:
                    raise ValueError("correspondence term requires target and coefficient")
                target_index = _uint(term[0], (1 << 32)-1)
                if target_index >= len(target.native.coordinates[k]):
                    raise ValueError("correspondence map target index differs")
                q = exact_fraction(term[1])
                if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > limits[2]:
                    raise ValueError("correspondence coefficient bit bound exceeded")
                terms.append((target_index, q))
            images.append(tuple(terms))
        result.append(tuple(images))
    return tuple(result)


def _attachment(record, reference, max_bytes):
    _closed(record, ("kind", "version", "reference", "archive"))
    actual = source_from_record(record["reference"], allow_cpp=True)
    if not isinstance(actual, CppSourceReference) or not actual.matches(reference):
        raise ValueError("correspondence attachment differs from its declared owner")
    archive = np.asarray(record["archive"])
    if (record["kind"] != "cpp_source_archive" or type(record["version"]) is not int
            or record["version"] != 1 or archive.dtype != np.uint8 or archive.ndim != 1
            or archive.size > max_bytes or sha256(archive.tobytes()).hexdigest() != reference.state_digest):
        raise ValueError("invalid correspondence source attachment")


def _weights(state):
    counts = state.payload["counts"]
    n = sum(counts)
    if n:
        return tuple(Fraction(v, n) for v in counts), n
    return tuple(Fraction(int(i == state.configuration["unresolved"]))
                 for i in range(len(counts))), 1


def _mix(templates, weights):
    result = []
    for k, row in enumerate(templates[0]):
        images = []
        for i in range(len(row)):
            terms = []
            for template, weight in zip(templates, weights, strict=True):
                terms.extend((target, q * weight) for target, q in template[k][i])
            images.append(tuple(terms))
        result.append(tuple(images))
    return tuple(result)


def _denominator(family, sample_count, bits):
    denominator = sample_count
    for row in family:
        for image in row:
            for _, q in image:
                denominator = lcm(denominator, q.denominator)
                _uint(denominator)
                if denominator.bit_length() > bits:
                    raise ValueError("correspondence common denominator bit bound exceeded")
    return denominator


def _span(value, size):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("byte scope requires an ordered start and end")
    a, b = (_uint(v) for v in value)
    if not 0 <= a < b <= size:
        raise ValueError("byte span is outside its retained source")
    return a, b


def _provenance(scope, size):
    a, b = _span(scope["left_span"], size)
    c, d = _span(scope["right_span"], size)
    if b > c:
        raise ValueError("context spans must follow their declared physical order")
    endpoints = tuple(((i, Fraction(1)),) for i in (a, b, c, d))
    edges = tuple(tuple((i, Fraction(1)) for i in range(x, y))
                  for x, y in ((a, b), (b, c), (c, d), (a, d)))
    return endpoints, edges, ((),)


def validate_correspondence_state(state):
    """Validate portable scope, map, source and exact count declarations."""
    c = state.configuration
    learned = state.adapter_version == "2"
    if (state.adapter != "cpp_correspondence" or state.adapter_version not in {"1", "2"}
            or state.arithmetic != "rational" or state.output_axes
            or not isinstance(state.source, CppSourceReference) or len(state.dependencies) != 2
            or any(not isinstance(v, CppSourceReference) for v in state.dependencies)):
        raise ValueError("unsupported correspondence model profile")
    _closed(c, _CONFIG | {"parameter_origin"} if learned else _CONFIG)
    if dict(c["execution_provider"]) != {**_PROVIDER, "version": "2" if learned else "1"}:
        raise ValueError("unsupported correspondence provider version")
    _closed(c["engine_receipt"], ("engine_version", "engine_digest", "binding_digest", "binding_version", "scalar_policy"))
    for key in ("engine_digest", "binding_digest"):
        from rexgraph.source_binding import _sha
        _sha(c["engine_receipt"][key])
    limits = tuple(_uint(v) for v in c["limits"])
    if len(limits) != 3 or not all(limits):
        raise ValueError("correspondence requires three positive resource limits")
    maximum = _uint(c["max_bytes"], 256 << 20)
    if not maximum:
        raise ValueError("correspondence source byte limit must be positive")
    grade = _uint(c["grade"], 2)
    refs = (state.source, *state.dependencies)
    for key, ref in zip(("carriers", "raw_carriers", "pattern_carriers"), refs, strict=True):
        carriers = tuple(_uint(v) for v in c[key])
        if len(carriers) != len(ref.native.coordinates) or len(set(carriers)) != len(carriers):
            raise ValueError("correspondence requires distinct complete grade carriers")
    if tuple(map(len, state.source.native.coordinates)) != (4, 4, 1):
        raise ValueError("correspondence requires the declared two-span composition chart")
    if state.space != _space(state.source, grade, c["carriers"][grade]):
        raise ValueError("correspondence input coordinates differ")
    raw, pattern = state.dependencies
    _attachment(c["raw_source"], raw, maximum)
    _attachment(c["pattern_source"], pattern, maximum)
    scope = _closed(c["scope"], ("codec", "data_field", "left_span", "right_span", "mention_side", "mention_bytes", "context_bytes"))
    if scope["codec"] != "utf-8" or scope["mention_side"] not in {"left", "right"}:
        raise ValueError("unsupported correspondence byte codec or mention role")
    _uint(scope["data_field"])
    if any(type(scope[k]) is not bytes for k in ("mention_bytes", "context_bytes")):
        raise ValueError("literal correspondence spans require exact bytes")
    sizes = tuple(map(len, raw.native.coordinates))
    if len(sizes) != 3 or sizes[0] != sizes[1] + 1 or sizes[2]:
        raise ValueError("raw correspondence target requires the complete byte interval profile")
    family = _family(c["provenance"], state.source, raw, limits)
    if family != _provenance(scope, sizes[1]):
        raise ValueError("correspondence provenance differs from its declared byte spans")
    labels = c["labels"]
    if (not isinstance(labels, tuple) or not labels or len(set(labels)) != len(labels)
            or any(not isinstance(v, str) or not v or len(v) > 256 for v in labels)):
        raise ValueError("correspondence requires distinct declared interpretation labels")
    if _uint(c["unresolved"]) >= len(labels) or len(c["templates"]) != len(labels):
        raise ValueError("correspondence requires an explicit unresolved template")
    total_records = sum(len(image) + 1 for t in c["templates"] for row in t for image in row)
    if total_records > limits[0]:
        raise ValueError("correspondence template record bound exceeded")
    for family in c["templates"]:
        _family(family, state.source, pattern, limits)
    _closed(state.payload, ("counts",))
    if not isinstance(state.payload["counts"], tuple) or len(state.payload["counts"]) != len(labels):
        raise ValueError("correspondence counts differ from declared interpretations")
    for n in state.payload["counts"]:
        _uint(n)
    _uint(sum(state.payload["counts"]))
    if state.step or state.parent is not None:
        raise ValueError("restored correspondence parameters require their declared fixed profile")
    if learned:
        from rexgraph.cpp_context_catalog import validate_context_origin
        validate_context_origin(c["parameter_origin"], c["scope"], state.payload["counts"],
                                c["labels"], c["unresolved"], pattern)


def _carriers(source, keys):
    keys = tuple(_uint(v) for v in keys)
    if len(keys) != len(source.identity["coordinates"]):
        raise ValueError("correspondence requires its complete carrier family")
    for k, key in enumerate(keys):
        c = source.carrier(key)
        if c["kind"] != "source_grade" or c["grade"] != k or c["size"] != len(source.identity["coordinates"][k]):
            raise ValueError("correspondence carrier differs from its source grade")
    return keys


def prepare_correspondence(state, source):
    """Own the restored sources and native complete chain maps."""
    state.bind(source)
    binding = _binding()
    c = state.configuration
    if dict(c["engine_receipt"]) != binding.build_info():
        raise ValueError("correspondence requires a different native semantic implementation")
    key = (id(state), id(source))
    with _lock:
        known = _cache.get(key)
        if known is not None and known[0] is state and known[1] is source:
            _cache.move_to_end(key)
            return known[2]
    validate_correspondence_state(state)
    catalog_info = None
    if state.adapter_version == "2":
        from rexgraph.cpp_context_catalog import validate_context_origin
        catalog = validate_context_origin(c["parameter_origin"], c["scope"], state.payload["counts"],
                                         c["labels"], c["unresolved"], state.dependencies[1], native=True)
        origin = c["parameter_origin"]
        catalog_info = (catalog.source, catalog.coefficient_digest, origin["feature"],
                        origin["event_cutoff"], origin["knowledge_cutoff"])
    raw = restore_cpp_source(c["raw_source"], state.dependencies[0], max_bytes=c["max_bytes"])
    pattern = restore_cpp_source(c["pattern_source"], state.dependencies[1], max_bytes=c["max_bytes"])
    for owner, keys in ((source, c["carriers"]), (raw, c["raw_carriers"]), (pattern, c["pattern_carriers"])):
        _carriers(owner, keys)
    data = raw.field(c["scope"]["data_field"])
    if data.carrier != c["raw_carriers"][1] or data.support != tuple(range(len(data.values))):
        raise ValueError("raw byte field requires complete interval support")
    if any(q.denominator != 1 or not 0 <= q <= 255 for q in data.values):
        raise ValueError("raw byte field contains a coefficient outside uint8")
    raw_bytes = bytes(int(q) for q in data.values)
    raw_bytes.decode("utf-8")
    scope = c["scope"]
    spans = (scope["left_span"], scope["right_span"])
    mention = int(scope["mention_side"] == "right")
    for span, literal in ((spans[mention], scope["mention_bytes"]), (spans[1-mention], scope["context_bytes"])):
        if raw_bytes[span[0]:span[1]] != literal:
            raise ValueError("correspondence literal differs from its original byte source")
    provenance = source.chain_map(raw, c["provenance"], limits=c["limits"])
    for family in c["templates"]:
        source.chain_map(pattern, family, limits=c["limits"])
    weights, sample_count = _weights(state)
    family = _mix(c["templates"], weights)
    denominator = _denominator(family, sample_count, c["limits"][2])
    interpretation = source.chain_map(pattern, family,
                                      denominator=denominator, limits=c["limits"])
    result = (provenance, interpretation, raw_bytes, denominator)
    if catalog_info is not None:
        result += (catalog_info,)
    size = len(source.archive) + len(raw.archive) + len(pattern.archive)
    if size <= 32 << 20:
        with _lock:
            _cache[key] = (state, source, result, size)
            while len(_cache) > 4 or sum(v[3] for v in _cache.values()) > 32 << 20:
                _cache.popitem(last=False)
    return result


def _infer(state, source, inputs):
    native = prepare_correspondence(state, source)
    c = state.configuration
    record = _closed(inputs, ("profile", "source", "grade", "side", "integer", "values"))
    if (record["profile"] != "exact-correspondence-input-v1" or record["source"] != state.source.as_record()
            or type(record["grade"]) is not int or record["grade"] != c["grade"]
            or record["side"] not in {"raw", "pattern"} or type(record["integer"]) is not bool):
        raise ValueError("correspondence input differs from its declared source, grade or side")
    index = int(record["side"] == "pattern")
    if record["integer"] and not index:
        raise ValueError("raw provenance uses the rational chain action")
    values = native[index].push(source, c["grade"], record["values"], integer=record["integer"])
    if record["integer"]:
        values = tuple(Fraction(v, native[3]) for v in values)
    reference = state.dependencies[index]
    carriers = c["pattern_carriers"] if index else c["raw_carriers"]
    receipt = {"engine": c["engine_receipt"], "side": record["side"], "target": reference.as_record(),
               "denominator": native[3] if index else 1, "integer": record["integer"],
               "target_carrier": carriers[c["grade"]],
               "scope": c["scope"], "input_digest": sha256(pack_value(thaw_tree(inputs))).hexdigest()}
    if state.adapter_version == "2":
        catalog_source, catalog_digest, feature, event_cutoff, knowledge_cutoff = native[4]
        receipt["parameter_origin"] = {"catalog_digest": catalog_digest,
            "catalog_source": catalog_source.as_record(), "feature": feature,
            "event_cutoff": event_cutoff, "knowledge_cutoff": knowledge_cutoff}
    dependencies = state.dependencies
    if state.adapter_version == "2":
        dependencies += (catalog_source,)
    return ModelOutput(values, _space(reference, c["grade"], carriers[c["grade"]]), (), state.source,
                       state.coefficient_digest, "rational", "exact-context-correspondence-v2" if state.adapter_version == "2" else "exact-context-correspondence-v1",
                       dependencies, grade=c["grade"], variance="chain", execution_receipt=receipt)


def enable_cpp_correspondence_provider():
    """Register owned byte provenance and interpretation map execution."""
    global _provider, _learned_provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], _PROVIDER["version"],
                (("cpp_correspondence", "1"),), _infer, prepare=prepare_correspondence,
                create=cpp_correspondence_model)
        register_model_provider(_provider)
        if _learned_provider is None:
            _learned_provider = ModelProvider(_PROVIDER["name"], "2",
                (("cpp_correspondence", "2"),), _infer, prepare=prepare_correspondence,
                create=cpp_correspondence_model)
        register_model_provider(_learned_provider)
        return _provider


def cpp_correspondence_model(source, raw, pattern, *, carriers, raw_carriers, pattern_carriers,
                             scope, templates, labels, unresolved, counts, grade=1,
                             limits=(1000000, 64000000, 16384), max_bytes=64 << 20,
                             adapter="cpp_correspondence", reference=None, parameter_origin=None):
    """Retain exact count parameters and every grade of a contextual correspondence."""
    if adapter != "cpp_correspondence":
        raise ValueError("unsupported correspondence adapter")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    refs = (ref, CppSourceReference(raw), CppSourceReference(pattern))
    scope = freeze_tree(scope)
    grade = _uint(grade, 2)
    carriers, raw_carriers, pattern_carriers = (_carriers(v, keys) for v, keys in
        ((source, carriers), (raw, raw_carriers), (pattern, pattern_carriers)))
    config = {"execution_provider": _PROVIDER, "engine_receipt": _binding().build_info(),
              "grade": grade, "carriers": carriers, "raw_carriers": raw_carriers,
              "pattern_carriers": pattern_carriers, "raw_source": cpp_source_record(raw, refs[1]),
              "pattern_source": cpp_source_record(pattern, refs[2]), "scope": scope,
              "provenance": _provenance(scope, len(refs[1].native.coordinates[1])),
              "templates": templates, "labels": labels, "unresolved": unresolved,
              "limits": limits, "max_bytes": max_bytes}
    version = "1"
    if parameter_origin is not None:
        version = "2"
        config["execution_provider"] = {**_PROVIDER, "version": "2"}
        config["parameter_origin"] = parameter_origin
    state = ModelState("cpp_correspondence", version, refs[0], _space(refs[0], grade, carriers[grade]), (),
                       config, {"counts": tuple(counts)}, "rational", dependencies=refs[1:])
    prepare_correspondence(state, source)
    enable_cpp_correspondence_provider()
    return state


def validate_correspondence_output(output):
    """Check the selected target carrier and both retained dependency references."""
    names = {"engine", "side", "target", "denominator", "integer", "scope", "input_digest", "target_carrier"}
    learned = output.method == "exact-context-correspondence-v2"
    receipt = _closed(output.execution_receipt, names | {"parameter_origin"} if learned else names)
    if (output.arithmetic != "rational" or output.variance != "chain" or output.axes
            or not isinstance(output.source, CppSourceReference) or len(output.dependencies) != (3 if learned else 2)
            or any(not isinstance(v, CppSourceReference) for v in output.dependencies)
            or receipt["side"] not in {"raw", "pattern"} or type(receipt["integer"]) is not bool):
        raise ValueError("unsupported correspondence output profile")
    index = int(receipt["side"] == "pattern")
    target = source_from_record(receipt["target"], allow_cpp=True)
    if not isinstance(target, CppSourceReference) or not target.matches(output.dependencies[index]):
        raise ValueError("correspondence output uses another target owner")
    grade = _uint(output.grade, 2)
    if grade >= len(target.native.coordinates) or output.space != _space(target, grade, _uint(receipt["target_carrier"])):
        raise ValueError("correspondence output target coordinates differ")
    if not _uint(receipt["denominator"]) or not index and (receipt["integer"] or receipt["denominator"] != 1):
        raise ValueError("correspondence output denominator or integer profile differs")
    from rexgraph.source_binding import _sha
    _sha(receipt["input_digest"])
    if learned:
        origin = _closed(receipt["parameter_origin"], ("catalog_digest", "catalog_source", "feature", "event_cutoff", "knowledge_cutoff"))
        _sha(origin["catalog_digest"])
        catalog = source_from_record(origin["catalog_source"], allow_cpp=True)
        if not isinstance(catalog, CppSourceReference) or not catalog.matches(output.dependencies[2]):
            raise ValueError("learned correspondence requires its native parameter source")
        if origin["feature"] is not None:
            _uint(origin["feature"])
        _uint(origin["event_cutoff"])
        _uint(origin["knowledge_cutoff"])


def correspondence_output_field(output):
    """Bind a derived field to its target and retain the observed chart as evidence."""
    from rexgraph.tensor_field import TensorField
    if not isinstance(output, ModelOutput) or output.method not in {"exact-context-correspondence-v1", "exact-context-correspondence-v2"}:
        raise TypeError("correspondence field extraction requires its declared model output")
    output.check_state()
    validate_correspondence_output(output)
    target = source_from_record(output.execution_receipt["target"], allow_cpp=True)
    dependencies = (output.source, *(v for v in output.dependencies if not v.matches(target)))
    return TensorField(output.space, output.values, output.axes, target, output.grade, output.variance,
                       provenance=(output.model_digest, output.coefficient_digest), dependencies=dependencies)


def cpp_correspondence_input(state, source, values, *, side="pattern", integer=False):
    """Bind a chain field to the retained context chart and selected target layer."""
    prepare_correspondence(state, source)
    return freeze_tree({"profile": "exact-correspondence-input-v1", "source": state.source.as_record(),
                       "grade": state.configuration["grade"], "side": side,
                       "integer": integer, "values": tuple(values)})
