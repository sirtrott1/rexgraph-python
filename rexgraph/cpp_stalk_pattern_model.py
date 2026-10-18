"""Source bound serving models for native joint transport pattern banks."""
from collections import OrderedDict
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from sys import maxsize
from threading import RLock

import numpy as np

from rexgraph.cpp_stalk_fields import _exact
from rexgraph.cpp_stalk_transport import _receipt_matches
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "exact-context-transport-pattern-bank-v1"
_QUERY = "stalk-pattern-bank-query-v1"
_SPARSE_QUERY = "stalk-pattern-bank-sparse-query-v1"
_CANDIDATE_QUERY = "stalk-pattern-bank-sparse-candidates-query-v1"
_PROGRAM_QUERY = "stalk-pattern-bank-program-query-v1"
_METHOD = "exact-joint-stalk-pattern-bank-v1"
_PROVIDER = {"name": "rexgraph.cpp.stalk_pattern_bank", "version": "1"}
_lock = RLock()
_cache = OrderedDict()
_receipts = OrderedDict()
_provider = None


def _admission(limits, maximum):
    if isinstance(limits, (str, bytes)):
        raise TypeError("four native action limits required")
    limits = tuple(limits)
    if (len(limits) != 4 or any(type(v) is not int or not 0 < v < 1 << 64 for v in limits) or
            type(maximum) is not int or not 0 < maximum < 1 << 64):
        raise ValueError("positive integer native action and archive limits required")
    return limits


def stalk_pattern_bank_source(archive, *, limits=(4096, 16000000, 64000000, 16384),
                              max_bytes=256 * 1024 * 1024):
    """Load bank tensor domains within their declared native storage limits."""
    from rexgraph_cpp import NativeSource
    limits = _admission(limits, max_bytes)
    return NativeSource(archive, max_bytes=max_bytes, **_archive_read_limits(limits))


def _archive_read_limits(limits):
    maximum = max(limits[:2])
    return dict(max_dimension=maximum, max_records=limits[1],
                max_logical_size=max(maximum, min(limits[0] * limits[0], 2 * maxsize + 1)))


def _boundary_guards(bank):
    from rexgraph_cpp import PreparedRelationGuards
    return PreparedRelationGuards(bank.archive(), limits=tuple(bank.limits[1:]),
                                 max_logical_size=_archive_read_limits(bank.limits)["max_logical_size"])


def validate_stalk_pattern_bank_state(state):
    """Validate the serving declaration without retaining training observations."""
    c = state.configuration
    if (state.adapter != "cpp_stalk_pattern_bank" or state.adapter_version != "1" or
            not isinstance(state.source, CppSourceReference) or state.arithmetic != "rational" or
            state.output_axes or state.dependencies or state.step != 0 or state.parent is not None or
            not isinstance(c, Mapping) or set(c) != {
                "profile", "execution_provider", "engine_receipt", "limits", "max_bytes"} or
            c["profile"] != _PROFILE or not _receipt_matches(_PROVIDER, c["execution_provider"]) or
            not isinstance(state.payload, Mapping) or state.payload):
        raise ValueError("closed immutable native pattern bank model required")
    _admission(c["limits"], c["max_bytes"])
    from rexgraph_cpp import stalk_transport_build_info
    if not _receipt_matches(stalk_transport_build_info(), c["engine_receipt"]):
        raise ValueError("pattern bank model requires its retained native implementation")
    if (len(state.source.native.coordinates) != 2 or not state.source.native.namespace or
            state.space != CoordinateSpace("stalk-pattern-bank", state.source.native.coordinates[1])):
        raise ValueError("model coordinates must be the retained original pattern keys")


def _prepare(state, source):
    from rexgraph_cpp import PreparedStalkTransportPatternBank
    state.bind(source)
    validate_stalk_pattern_bank_state(state)
    c = state.configuration
    if len(source.archive) > c["max_bytes"]:
        raise ValueError("pattern bank archive exceeds its model admission")
    key = (state.source.state_digest, tuple(c["limits"]), c["max_bytes"])
    with _lock:
        cached = _cache.get(key)
        if cached is None:
            bank = PreparedStalkTransportPatternBank(source.archive,
                model_namespace=state.source.native.namespace, limits=c["limits"], max_bytes=c["max_bytes"])
            _cache[key] = (source, bank)
            while len(_cache) > 4:
                _cache.popitem(last=False)
        else:
            _, bank = cached
        _cache.move_to_end(key)
    return bank


def cpp_stalk_pattern_bank_model(source, *, limits=(4096, 16000000, 64000000, 16384),
                                 max_bytes=256*1024*1024, reference=None):
    """Bind one bank archive containing its source, weights and common input images."""
    from rexgraph_cpp import stalk_transport_build_info
    bounds = _admission(limits, max_bytes)
    enable_stalk_pattern_bank_provider()
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    state = ModelState("cpp_stalk_pattern_bank", "1", ref,
        CoordinateSpace("stalk-pattern-bank", ref.native.coordinates[1]), (),
        dict(profile=_PROFILE, execution_provider=_PROVIDER, engine_receipt=stalk_transport_build_info(),
             limits=bounds, max_bytes=max_bytes), {}, "rational")
    _prepare(state, source)
    return state


def _create(source, *, adapter, **configuration):
    if adapter != "cpp_stalk_pattern_bank":
        raise ValueError("native pattern bank adapter required")
    return cpp_stalk_pattern_bank_model(source, **configuration)


def clear_stalk_pattern_bank_cache():
    """Release cached owning banks without changing model declarations."""
    with _lock:
        _cache.clear()
        _receipts.clear()


def inspect_stalk_pattern_bank_model(state, source):
    """Read original occurrences, observable weights, joint directions and provenance."""
    return freeze_tree(_prepare(state, source).inspect())


def select_stalk_pattern_bank_model(state, source, patterns, *, model_namespace, coordinate_ids):
    """Bind an independent bank containing selected stored stars and their joint weights."""
    bank = _prepare(state, source).select(patterns, model_namespace=model_namespace,
                                         coordinate_ids=coordinate_ids)
    selected = stalk_pattern_bank_source(bank.archive(), limits=state.configuration["limits"],
                                         max_bytes=state.configuration["max_bytes"])
    model = cpp_stalk_pattern_bank_model(selected, limits=state.configuration["limits"],
                                        max_bytes=state.configuration["max_bytes"])
    return model, selected


def _comparisons(state, comparisons):
    from rexgraph_cpp._stalk_boundary import _natural
    bounds = state.configuration["limits"]
    required = {"name", "left", "right", "left_frame", "right_frame", "interface_frame", "left_map", "right_map"}
    result = []; count = 0
    for item in comparisons:
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError("complete declared interface comparison required")
        row = dict(item)
        for key in ("left", "right", "left_frame", "right_frame", "interface_frame"):
            row[key] = _natural(row[key])
        for side in ("left_map", "right_map"):
            matrix = row[side]
            if not isinstance(matrix, Mapping) or set(matrix) != {"shape", "entries"} or len(matrix["shape"]) != 2:
                raise ValueError("closed interface matrix shape and sparse entries required")
            shape = tuple(_natural(v) for v in matrix["shape"])
            if max(shape) > bounds[0]:
                raise ValueError("interface dimensions exceed admission")
            entries = []
            for i, j, coefficient in matrix["entries"]:
                count += 1
                if count > bounds[1]:
                    raise ValueError("interface coefficients exceed admission")
                entries.append((_natural(i), _natural(j), _exact(coefficient, bounds[3])))
            row[side] = dict(shape=shape, entries=tuple(entries))
        result.append(row)
    return tuple(result)


def _declaration(state, inputs, operation, shared, comparisons, *, sparse=False):
    state.check_state()
    if state.adapter != "cpp_stalk_pattern_bank":
        raise ValueError("native serving pattern bank model required")
    if type(operation) is not str or operation not in {"response", "glue", "candidates"} or operation=="candidates" and not sparse:
        raise ValueError("select a native response or compatible section operation")
    if type(shared) is not bool:
        raise TypeError("shared boundary selection must be boolean")
    if not isinstance(inputs, Mapping) or len(inputs) > state.configuration["limits"][0]:
        raise ValueError("bounded original bank indices mapped to common input vectors required")
    rows = {}; count = 0
    for key, vector in inputs.items():
        if sparse:
            if (type(key) is not int or not 0 <= key < len(state.space.keys) or
                    not isinstance(vector, Mapping) or set(vector) != {"profile", "frame", "size", "entries"} or
                    vector["profile"] != "complete-sparse-stalk-field-v1" or
                    type(vector["frame"]) is not int or not 0 < vector["frame"] < 1 << 64 or
                    type(vector["size"]) is not int or not 0 <= vector["size"] <= state.configuration["limits"][0]):
                raise ValueError("complete sparse field on a declared common frame required")
            entries = []; previous = -1
            for i, coefficient in vector["entries"]:
                count += 1
                if count > state.configuration["limits"][1]:
                    raise ValueError("sparse input coefficients exceed admission")
                q = _exact(coefficient, state.configuration["limits"][3])
                if type(i) is not int or not previous < i < vector["size"] or q == 0:
                    raise ValueError("ordered distinct nonzero sparse field coefficients required")
                entries.append((i, q)); previous = i
            rows[key] = dict(profile=vector["profile"], frame=vector["frame"], size=vector["size"], entries=tuple(entries))
            continue
        if (type(key) is not int or not 0 <= key < len(state.space.keys) or
                isinstance(vector, (str, bytes)) or len(vector) > state.configuration["limits"][0]):
            raise ValueError("admitted bank index and bounded common input vector required")
        rows[key] = tuple(_exact(v, state.configuration["limits"][3]) for v in vector)
    if isinstance(comparisons, (str, bytes)) or len(comparisons) > state.configuration["limits"][0]:
        raise ValueError("bounded explicit interface comparison sequence required")
    if operation != "glue" and comparisons:
        raise ValueError("interface comparisons require the gluing operation")
    return freeze_tree(dict(profile=_CANDIDATE_QUERY if operation=="candidates" else _SPARSE_QUERY if sparse else _QUERY, source_reference=state.source.as_record(),
        model_digest=state.coefficient_digest, inputs=tuple(sorted(rows.items())), operation=operation,
        shared=shared, comparisons=_comparisons(state, comparisons)))


def stalk_pattern_bank_query(state, inputs, *, operation="response", shared=True, comparisons=()):
    """Declare bank local pattern indices, whole common inputs and named interfaces."""
    return thaw_tree(_declaration(state, inputs, operation, shared, comparisons))


def stalk_pattern_bank_sparse_query(state, inputs, *, operation="response", shared=True, comparisons=()):
    """Declare complete common fields using ordered nonzero exact coefficients."""
    return thaw_tree(_declaration(state, inputs, operation, shared, comparisons, sparse=True))


def stalk_pattern_bank_program_query(state, nodes):
    """Declare ordered stages with complete sparse occurrence maps and inputs."""
    state.check_state()
    if state.adapter != "cpp_stalk_pattern_bank":
        raise ValueError("native serving pattern bank model required")
    bounds = state.configuration["limits"]
    if isinstance(nodes, (str, bytes)) or not 0 < len(nodes) <= bounds[0]:
        raise ValueError("bounded nonempty ordered program required")
    from rexgraph_cpp._stalk_boundary import _natural
    result = []; names = set(); count = 0
    for item in nodes:
        if not isinstance(item, Mapping) or set(item) != {"name", "pattern", "input", "edges"}:
            raise ValueError("closed named program stage required")
        name = item["name"]; pattern = item["pattern"]
        if (type(name) is not str or not name or len(name.encode()) > 256 or name in names or
                type(pattern) is not int or not 0 <= pattern < len(state.space.keys)):
            raise ValueError("distinct bounded stage names and original pattern indices required")
        names.add(name)
        field = item["input"]
        if field is not None:
            field = thaw_tree(_declaration(state, {pattern: field}, "response", False, (), sparse=True))["inputs"][0][1]
            count += len(field["entries"])
        edges = []; seen = set()
        if isinstance(item["edges"], (str, bytes)) or len(item["edges"]) > bounds[0]:
            raise ValueError("bounded complete incoming map sequence required")
        if (field is not None) == bool(item["edges"]):
            raise ValueError("each stage requires an external input or incoming maps")
        for edge in item["edges"]:
            if not isinstance(edge, Mapping) or set(edge) != {"node", "occurrence", "source_frame", "target_frame", "map"}:
                raise ValueError("closed original occurrence map required")
            row = {k: _natural(edge[k]) for k in ("node", "occurrence", "source_frame", "target_frame")}
            identity = (row["node"], row["occurrence"])
            if row["node"] >= len(result) or identity in seen:
                raise ValueError("distinct incoming occurrences from preceding stages required")
            seen.add(identity)
            m = edge["map"]
            if not isinstance(m, Mapping) or set(m) != {"shape", "entries"} or len(m["shape"]) != 2:
                raise ValueError("complete sparse stage map required")
            shape = tuple(_natural(v) for v in m["shape"])
            if max(shape) > bounds[0]:
                raise ValueError("program map exceeds dimension admission")
            entries = []
            for i,j,q in m["entries"]:
                count += 1
                if count > bounds[1]:
                    raise ValueError("program entries exceed admission")
                entries.append((_natural(i),_natural(j),_exact(q,bounds[3])))
            row["map"] = dict(shape=shape, entries=tuple(entries)); edges.append(row)
        if count > bounds[1]:
            raise ValueError("program entries exceed admission")
        result.append(dict(name=name, pattern=pattern, input=field, edges=tuple(edges)))
    return thaw_tree(freeze_tree(dict(profile=_PROGRAM_QUERY, source_reference=state.source.as_record(),
        model_digest=state.coefficient_digest, inputs=tuple(result), operation="program", shared=False, comparisons=())))


def _json_integer(value, bits):
    if isinstance(value, Mapping):
        if set(value) != {"kind", "decimal"} or value["kind"] != "Integer":
            raise ValueError("declared exact integer JSON record required")
        text = value["decimal"]
        if (type(text) is not str or not text.isascii() or len(text) > bits*30103//100000+2 or
                not text.removeprefix("-").isdecimal() or text in {"-0", ""} or
                text.removeprefix("-").startswith("0") and text != "0"):
            raise ValueError("bounded canonical decimal integer required")
        value = int(Decimal(text))
    if type(value) is not int or abs(value).bit_length() > bits:
        raise TypeError("bounded exact JSON integer required")
    return value


def _json_coefficient(value, bits):
    if isinstance(value, Mapping) and set(value) == {"numerator", "denominator"}:
        numerator = _json_integer(value["numerator"], bits)
        denominator = _json_integer(value["denominator"], bits)
        if denominator <= 0:
            raise ValueError("positive exact JSON denominator required")
        return Fraction(numerator, denominator)
    return Fraction(_json_integer(value, bits))


def _json_query(state, query):
    """Decode the exact scalar records and arrays emitted by System serialization."""
    c = state.configuration
    if len(query["inputs"]) > c["limits"][0]:
        raise ValueError("bank input count exceeds admission")
    q = dict(query)
    ref = dict(q["source_reference"]); native = dict(ref["native"])
    for key in ("namespace", "sequence", "first_sequence"):
        native[key] = _json_integer(native[key], 64)
    ref["native"] = native
    if ref["version"] is not None:
        ref["version"] = _json_integer(ref["version"], 64)
    q["source_reference"] = source_from_record(ref, allow_cpp=True).as_record()
    if q["profile"] == _PROGRAM_QUERY:
        if type(q["comparisons"]) is not list or q["comparisons"]:
            raise ValueError("program JSON requires an empty comparison sequence")
        nodes = []; count = 0
        for node in q["inputs"]:
            if not isinstance(node, Mapping) or set(node) != {"name", "pattern", "input", "edges"}:
                raise ValueError("closed JSON program stage required")
            node = dict(node); node["pattern"] = _json_integer(node["pattern"], 64)
            if node["input"] is not None:
                f = node["input"]
                if not isinstance(f, Mapping) or set(f) != {"profile", "frame", "size", "entries"}:
                    raise ValueError("closed JSON program field required")
                entries = []
                for i,v in f["entries"]:
                    count += 1
                    if count > c["limits"][1]:
                        raise ValueError("JSON program entries exceed admission")
                    entries.append((_json_integer(i,64),_json_coefficient(v,c["limits"][3])))
                node["input"] = dict(profile=f["profile"], frame=_json_integer(f["frame"],64), size=_json_integer(f["size"],64),entries=tuple(entries))
            edges = []
            if type(node["edges"]) is not list or len(node["edges"]) > c["limits"][0]:
                raise ValueError("bounded JSON stage maps required")
            for edge in node["edges"]:
                if not isinstance(edge, Mapping) or set(edge) != {"node", "occurrence", "source_frame", "target_frame", "map"}:
                    raise ValueError("closed JSON occurrence map required")
                row = {key: _json_integer(edge[key],64) for key in ("node", "occurrence", "source_frame", "target_frame")}
                m = edge["map"]
                if not isinstance(m, Mapping) or set(m) != {"shape", "entries"}:
                    raise ValueError("closed JSON stage matrix required")
                if len(m["shape"]) != 2:
                    raise ValueError("complete JSON stage matrix shape required")
                entries = []
                for i,j,v in m["entries"]:
                    count += 1
                    if count > c["limits"][1]:
                        raise ValueError("JSON program entries exceed admission")
                    entries.append((_json_integer(i,64),_json_integer(j,64),_json_coefficient(v,c["limits"][3])))
                row["map"] = dict(shape=tuple(_json_integer(v,64) for v in m["shape"]),entries=tuple(entries));edges.append(row)
            node["edges"] = tuple(edges);nodes.append(node)
        q["inputs"] = tuple(nodes); q["comparisons"] = tuple(q["comparisons"])
        return q
    rows = []; count = 0
    for key, vector in q["inputs"]:
        if q["profile"] in {_SPARSE_QUERY,_CANDIDATE_QUERY}:
            if not isinstance(vector, Mapping) or set(vector) != {"profile", "frame", "size", "entries"}:
                raise ValueError("closed complete sparse JSON field required")
            row = dict(vector); entries = []
            for i, value in row["entries"]:
                count += 1
                if count > c["limits"][1]:
                    raise ValueError("JSON sparse input coefficients exceed admission")
                entries.append((_json_integer(i, 64), _json_coefficient(value, c["limits"][3])))
            row["frame"] = _json_integer(row["frame"], 64); row["size"] = _json_integer(row["size"], 64)
            row["entries"] = tuple(entries)
            rows.append((_json_integer(key, 64), row))
            continue
        if type(vector) is not list or len(vector) > c["limits"][0]:
            raise ValueError("bounded whole JSON input vector required")
        rows.append((_json_integer(key, 64), tuple(_json_coefficient(v, c["limits"][3]) for v in vector)))
    q["inputs"] = tuple(rows)
    if type(q["comparisons"]) is not list or len(q["comparisons"]) > c["limits"][0]:
        raise ValueError("bounded JSON interface sequence required")
    comparisons = []; count = 0
    for item in q["comparisons"]:
        item = dict(item)
        for key in ("left", "right", "left_frame", "right_frame", "interface_frame"):
            item[key] = _json_integer(item[key], 64)
        for side in ("left_map", "right_map"):
            matrix = dict(item[side]); entries = []
            for i, j, coefficient in matrix["entries"]:
                count += 1
                if count > c["limits"][1]:
                    raise ValueError("JSON interface entries exceed admission")
                entries.append((_json_integer(i, 64), _json_integer(j, 64),
                    _json_coefficient(coefficient, c["limits"][3])))
            matrix["entries"] = tuple(entries)
            matrix["shape"] = tuple(_json_integer(v, 64) for v in matrix["shape"])
            item[side] = matrix
        comparisons.append(item)
    q["comparisons"] = tuple(comparisons)
    return q


def _query(state, query):
    if (not isinstance(query, Mapping) or set(query) != {
            "profile", "source_reference", "model_digest", "inputs", "operation", "shared", "comparisons"} or
            query["profile"] not in {_QUERY, _SPARSE_QUERY, _CANDIDATE_QUERY, _PROGRAM_QUERY} or query["model_digest"] != state.coefficient_digest or
            (query["profile"]==_CANDIDATE_QUERY)!=(query["operation"]=="candidates") or
            (query["profile"]==_PROGRAM_QUERY)!=(query["operation"]=="program")):
        raise ValueError("closed pattern query bound to its serving model and source required")
    if type(query["inputs"]) is list:
        query = _json_query(state, query)
    if not _receipt_matches(state.source.as_record(), query["source_reference"]):
        raise ValueError("closed pattern query bound to its serving model and source required")
    if type(query["inputs"]) is not tuple:
        raise ValueError("ordered distinct bank input declarations required")
    if query["profile"] == _PROGRAM_QUERY:
        if query["shared"] is not False or query["comparisons"]:
            raise ValueError("program stages use original unscaled occurrence fields")
        return freeze_tree(stalk_pattern_bank_program_query(state,query["inputs"]))
    rows = dict(query["inputs"])
    if len(rows) != len(query["inputs"]):
        raise ValueError("distinct original bank input indices required")
    return _declaration(state, rows, query["operation"], query["shared"], query["comparisons"],
                        sparse=query["profile"] in {_SPARSE_QUERY,_CANDIDATE_QUERY})


def _analysis(state, source, query):
    bank = _prepare(state, source)
    q = _query(state, query)
    if q["operation"] == "program":
        result = bank.response_sparse_program(q["inputs"])
    elif q["operation"] == "glue":
        executor = bank.glue_sparse if q["profile"] == _SPARSE_QUERY else bank.glue
        result = executor(dict(q["inputs"]), q["comparisons"], shared=q["shared"])
    elif q["operation"] == "candidates":
        result = bank.response_sparse_candidates(dict(q["inputs"]),shared=q["shared"])
    else:
        executor = bank.response_sparse if q["profile"] == _SPARSE_QUERY else bank.response
        result = executor(dict(q["inputs"]), shared=q["shared"])
    witness = _field_witness(state, q, result)
    with _lock:
        _receipts[state.coefficient_digest] = witness
        _receipts.move_to_end(state.coefficient_digest)
        while len(_receipts) > 4:
            _receipts.popitem(last=False)
    return q, result, witness


def _field_witness(state, query, result):
    from rexgraph_cpp._model_storage import capture_content
    c = state.configuration
    witness = capture_content((query, result),
        max_nodes=min(maxsize, 128+32*(c["limits"][0]+c["limits"][1])),
        max_bytes=min(maxsize, c["max_bytes"]+32*c["limits"][1]))
    if witness is None:
        raise ValueError("native field receipt exceeds its content admission")
    return witness


def _infer(state, source, inputs):
    from rexgraph.model_codec import _model_checkpoint
    query, result, _ = _analysis(state, source, inputs)
    receipt = dict(profile=_METHOD, model_checkpoint=_model_checkpoint(state), source_archive=source.archive,
        query_declaration=query, result=result, interpretation="complete-joint-native-field-family-v1")
    return ModelOutput(np.empty((0,), dtype=object), CoordinateSpace("stalk-pattern-bank-families", ()), (),
        state.source, state.coefficient_digest, "rational", _METHOD,
        grade=None, variance="coordinate", execution_receipt=receipt)


def validate_stalk_pattern_bank_output(output):
    """Check native field witnesses and replay uncached families and contradictions."""
    from rexgraph.model_codec import _checkpoint_model
    r = output.execution_receipt
    if (not isinstance(r, Mapping) or set(r) != {"profile", "model_checkpoint", "source_archive",
            "query_declaration", "result", "interpretation"} or r["profile"] != _METHOD or
            r["interpretation"] != "complete-joint-native-field-family-v1"):
        raise ValueError("closed native pattern bank output receipt required")
    state = _checkpoint_model(r["model_checkpoint"])
    if (not isinstance(state, ModelState) or state.adapter != "cpp_stalk_pattern_bank" or
            output.method != _METHOD or output.arithmetic != "rational" or output.dependencies or
            output.grade is not None or output.variance != "coordinate" or
            not output.source.matches(state.source) or output.model_digest != state.coefficient_digest or
            output.space != CoordinateSpace("stalk-pattern-bank-families", ()) or output.axes or output.values.size):
        raise ValueError("pattern bank output differs from its original model and field contract")
    if type(r["source_archive"]) is not bytes or len(r["source_archive"]) > state.configuration["max_bytes"]:
        raise ValueError("bounded serving bank archive required")
    key = (state.source.state_digest, tuple(state.configuration["limits"]), state.configuration["max_bytes"])
    with _lock:
        cached = _cache.get(key)
        source = None if cached is None else cached[0]
    if source is None or source.archive != r["source_archive"]:
        source = stalk_pattern_bank_source(r["source_archive"], limits=state.configuration["limits"],
                                           max_bytes=state.configuration["max_bytes"])
    _prepare(state, source)
    with _lock:
        witness = _receipts.get(state.coefficient_digest)
    if witness is not None and witness.matches((r["query_declaration"], r["result"]), ordered=False):
        return
    _, _, witness = _analysis(state, source, r["query_declaration"])
    if not witness.matches((r["query_declaration"], r["result"]), ordered=False):
        raise ValueError("pattern bank output differs from complete native field replay")


def stalk_pattern_bank_result(output):
    """Read the complete response or gluing family without selecting free coordinates."""
    output.check_state()
    if output.method != _METHOD:
        raise ValueError("native pattern bank output required")
    return output.execution_receipt["result"]


def enable_stalk_pattern_bank_provider():
    """Register immutable serving and source preparation through the shared model API."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", (("cpp_stalk_pattern_bank", "1"),),
                _infer, create=_create, prepare=_prepare)
        register_model_provider(_provider)
        return _provider
