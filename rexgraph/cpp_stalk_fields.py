"""Acquired native relation columns and exact fields on named local stalks."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from numbers import Integral
from threading import RLock

import numpy as np

from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "acquired-relational-stalk-fields-v1"
_METHOD = "exact-relational-stalk-green-v1"
_PROVIDER = {"name": "rexgraph.cpp.stalk_fields", "version": "1"}
_lock = RLock()
_cache = OrderedDict()
_checked = OrderedDict()
_provider = None


def _exact(value, bits):
    if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
        raise TypeError("stalk coefficients require integers or Fractions")
    value = Fraction(int(value)) if isinstance(value, Integral) else Fraction(int(value.numerator), int(value.denominator))
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > bits:
        raise ValueError("stalk coefficient exceeds integer admission")
    return value


def _axes(rows):
    rows = tuple(tuple(row) for row in rows)
    if any(not row or len(set(row)) != len(row) or
           any(not isinstance(name, str) or not name or len(name.encode()) > 256 for name in row) for row in rows):
        raise ValueError("every original stalk requires distinct bounded named axes")
    return rows


def _zeros(c):
    return tuple((Fraction(0),) * len(row) for row in c["upper_axes"])


def _field(c, values):
    rows = tuple(tuple(_exact(v, c["limits"][3]) for v in row) for row in values)
    if tuple(map(len, rows)) != tuple(map(len, c["upper_axes"])):
        raise ValueError("field must cover every original named upper stalk")
    return rows


def _acquisition(c, observations):
    """Declare one original relation column per acquired whole stalk observation."""
    from rexgraph_cpp import NativeSource
    if not observations:
        return None
    offsets = [0]
    for axes in c["upper_axes"]:
        offsets.append(offsets[-1] + len(axes))
    if offsets[-1] + len(observations) > c["limits"][0]:
        raise ValueError("acquisition dimensions exceed action admission")
    if offsets[-1] + len(observations) + sum(len(o["values"]) for o in observations) > c["limits"][1]:
        raise ValueError("acquisition columns exceed entry admission")
    relations = []; key = 0
    for ordinal, observation in enumerate(observations):
        occurrences = []
        for axis, value in enumerate(observation["values"]):
            key += 1
            occurrences.append(dict(key=key, lower=offsets[observation["row"]]+axis,
                exists=True, negative=value < 0, magnitude=abs(value), aspects=(0,)))
        relations.append(dict(grade=1, key=ordinal+1, type=0, occurrences=tuple(occurrences)))
    declaration = dict(profile="declared-role-source-v1", namespace=941109, semantic_scale=2,
        max_grade=1, role_aspects=("field-coordinate",),
        types=(dict(name="acquired-whole-stalk", min_arity=0, max_arity=0),),
        witnesses=tuple(range(offsets[-1])), relations=tuple(relations), fields=(), metrics=(),
        carriers=(dict(key=10, kind="source_grade", grade=0, coordinates=0, size=offsets[-1]),
                  dict(key=11, kind="source_grade", grade=1, coordinates=0, size=len(relations))))
    return NativeSource.from_declaration(declaration, max_bytes=c["max_bytes"]).archive


def _literal_accessions(c, source):
    """Replay retained literal byte fields and complete token provenance maps."""
    from rexgraph_cpp import NativeSource
    from rexgraph.cpp_relational_sections import _context
    accession = c["source_accession"]
    if accession is None:
        return
    if accession.get("profile") == "original-structural-language-source-v1":
        from rexgraph.linguistic_accession import validate_linguistic_accession
        validate_linguistic_accession(accession, source, max_bytes=c["max_bytes"])
        return
    if not accession["charts"]:
        return
    selected = set()
    for row, relation in enumerate(accession["relations"]):
        if not isinstance(relation, Mapping) or relation.get("kind") != "observed-context":
            continue
        raw, chart = _context({"field": {"source_accession": accession}}, row)
        selected.add(relation["provenance"]["query"])
        literal = NativeSource(chart["byte_source_archive"], max_bytes=c["max_bytes"])
        token = NativeSource(chart["token_source_archive"], max_bytes=c["max_bytes"])
        if tuple(literal.field(1001).values) != tuple(bytes.fromhex(raw)):
            raise ValueError("literal accession differs from its retained byte field")
        token.chain_map(literal, chart["token_to_byte_family"])
    if selected != set(range(len(accession["charts"]))):
        raise ValueError("literal charts require their original context accessions")


def validate_stalk_fields_state(state):
    c = state.configuration
    if (state.adapter != "cpp_stalk_fields" or state.adapter_version != "1" or
            state.arithmetic != "rational" or not isinstance(state.source, CppSourceReference) or
            c.get("profile") != _PROFILE or dict(c.get("execution_provider", {})) != _PROVIDER):
        raise ValueError("unsupported acquired stalk field model")
    if set(c) != {"profile", "execution_provider", "engine_receipt", "upper_axes", "lower_axes",
                  "flag_maps", "training_rows", "limits", "parameter", "max_bytes", "source_reference", "source_accession"}:
        raise ValueError("closed stalk field declaration required")
    upper = _axes(c["upper_axes"]); lower = _axes(c["lower_axes"])
    width = max(map(len, upper), default=0)
    if (not width or state.space.keys != state.source.native.coordinates[1] or
            len(upper) != len(state.space.keys) or len(lower) != len(state.source.native.coordinates[0]) or
            state.output_axes != (CoordinateSpace("stalk-slot", tuple(str(i) for i in range(width))),) or
            dict(c["source_reference"]) != state.source.as_record()):
        raise ValueError("stalk declaration differs from its actual source or carrier")
    if (len(c["limits"]) != 4 or any(type(i) is not int or i <= 0 for i in c["limits"]) or
            type(c["max_bytes"]) is not int or c["max_bytes"] <= 0 or
            not isinstance(c["parameter"], Fraction) or c["parameter"] <= 0):
        raise ValueError("positive exact stalk and byte admissions required")
    if sum(map(len, upper)) + sum(map(len, lower)) > c["limits"][0]:
        raise ValueError("whole stalk dimensions exceed action admission")
    accession = c["source_accession"]
    if accession is not None:
        if not isinstance(accession, Mapping):
            raise ValueError("source accessions must retain original cells and occurrences")
        structural = accession.get("profile") == "original-structural-language-source-v1"
        keys = {"profile", "declaration", "lower", "rows", "charts"} if structural else {"lower", "relations", "occurrences", "charts"}
        if (set(accession) != keys or len(accession["lower"]) != len(lower)
                or not structural and (len(accession["relations"]) != len(upper)
                or len(accession["occurrences"]) != len(c["flag_maps"]))):
            raise ValueError("source accessions must retain original cells and occurrences")
    if len(c["flag_maps"]) + sum(map(len, c["flag_maps"])) > c["limits"][1]:
        raise ValueError("flag maps exceed entry admission")
    for block in c["flag_maps"]:
        for row, column, value in block:
            if type(row) is not int or type(column) is not int or row < 0 or column < 0:
                raise ValueError("flag map requires natural local coordinates")
            _exact(value, c["limits"][3])
    training = c["training_rows"]
    if len(set(training)) != len(training) or any(type(i) is not int or not 0 <= i < len(upper) for i in training):
        raise ValueError("training rows must name distinct original relations")
    if set(state.payload) != {"observations", "batches", "acquisition_archive"} or len(state.payload["batches"]) != state.step:
        raise ValueError("stalk model requires original acquired columns and batches")
    grouped = [[] for _ in state.payload["batches"]]
    for observation in state.payload["observations"]:
        if (set(observation) != {"row", "values", "batch", "step"} or
                type(observation["row"]) is not int or observation["row"] not in training or
                type(observation["step"]) is not int or not 0 <= observation["step"] < state.step or
                observation["batch"] != state.payload["batches"][observation["step"]] or
                len(observation["values"]) != len(upper[observation["row"]])):
            raise ValueError("acquired column differs from its original training observation")
        for value in observation["values"]:
            _exact(value, c["limits"][3])
        grouped[observation["step"]].append(observation["row"])
    if any(not rows or rows != sorted(set(rows)) for rows in grouped):
        raise ValueError("each acquired batch requires unique ordered original rows")


def _prepare(state, source):
    from rexgraph_cpp import PreparedStalkFields, stalk_fields_build_info
    validate_stalk_fields_state(state); state.bind(source)
    c = state.configuration
    if dict(c["engine_receipt"]) != stalk_fields_build_info():
        raise ValueError("stalk model requires its retained native implementation")
    with _lock:
        key = (state.source.native.archive_digest, c["lower_axes"], c["upper_axes"], c["flag_maps"],
               c["parameter"], c["limits"], c["max_bytes"])
        if state.coefficient_digest not in _checked:
            _literal_accessions(c, source)
            if _acquisition(c, state.payload["observations"]) != state.payload["acquisition_archive"]:
                raise ValueError("acquisition source differs from the original whole stalk observations")
        plan = _cache.get(key)
        if plan is None:
            plan = PreparedStalkFields(source.archive, c["lower_axes"], c["upper_axes"], c["flag_maps"],
                parameter=c["parameter"], limits=c["limits"], max_bytes=c["max_bytes"])
            _cache[key] = plan
            while len(_cache) > 4:
                _cache.popitem(last=False)
        if state.coefficient_digest not in _checked:
            if state.payload["acquisition_archive"] is not None:
                plan.inject_acquisition(state.payload["acquisition_archive"],
                    tuple(range(sum(map(len, c["upper_axes"])))), (1,)*len(state.payload["observations"]))
            _checked[state.coefficient_digest] = True
            while len(_checked) > 16:
                _checked.popitem(last=False)
        _cache.move_to_end(key); _checked.move_to_end(state.coefficient_digest)
        return plan


def cpp_stalk_fields_model(source, lower_axes, upper_axes, flag_maps, training_rows, *,
                           parameter=Fraction(1), limits=(4096, 16000000, 64000000, 16384),
                           max_bytes=256*1024*1024, reference=None, source_accession=None):
    """Declare complete local stalks and exact underlying maps for original flags."""
    from rexgraph_cpp import stalk_fields_build_info
    reference = CppSourceReference(source) if reference is None else reference.bind(source)
    lower = _axes(lower_axes); upper = _axes(upper_axes); limits = tuple(limits)
    if len(limits) != 4 or any(type(i) is not int or i <= 0 for i in limits):
        raise ValueError("four positive action admissions required")
    maps = tuple(tuple((row, column, _exact(value, limits[3])) for row, column, value in block) for block in flag_maps)
    c = dict(profile=_PROFILE, execution_provider=_PROVIDER, engine_receipt=stalk_fields_build_info(),
        lower_axes=lower, upper_axes=upper, flag_maps=maps, training_rows=tuple(training_rows),
        parameter=_exact(parameter, limits[3]), limits=limits, max_bytes=max_bytes,
        source_reference=reference.as_record(), source_accession=source_accession)
    state = ModelState("cpp_stalk_fields", "1", reference,
        CoordinateSpace("retained-stalk-relations", reference.native.coordinates[1]),
        (CoordinateSpace("stalk-slot", tuple(str(i) for i in range(max(map(len, upper), default=0)))),), c,
        dict(observations=(), batches=(), acquisition_archive=None), "rational")
    enable_stalk_fields_provider(); _prepare(state, source)
    return state


def stalk_annotation_batch(state, annotations):
    """Bind complete exact stalk observations to selected original relation rows."""
    c = state.configuration
    values = np.full((len(state.space.keys), len(state.output_axes[0].keys)), Fraction(0), dtype=object)
    observed = np.zeros(len(values), dtype=bool)
    for row, field in annotations:
        if type(row) is not int or row not in c["training_rows"] or observed[row]:
            raise ValueError("annotation must name a unique original training relation")
        field = tuple(_exact(v, c["limits"][3]) for v in field)
        if len(field) != len(c["upper_axes"][row]):
            raise ValueError("annotation must cover every named branch of its stalk")
        values[row, :len(field)] = field; observed[row] = True
    return ModelBatch(state.source, state.space, values, observed,
        dict(profile="whole-stalk-annotations-v1", partition="training"))


def _train(state, source, batch, *, steps):
    _prepare(state, source); c = state.configuration
    if (steps != 1 or dict(batch.inputs or {}) != dict(profile="whole-stalk-annotations-v1", partition="training") or
            batch.targets.shape != (len(state.space.keys), len(state.output_axes[0].keys))):
        raise ValueError("one declared complete stalk observation batch required")
    observations = list(state.payload["observations"]); added = 0
    for row, target in enumerate(batch.targets):
        values = tuple(_exact(v, c["limits"][3]) for v in target); size = len(c["upper_axes"][row])
        if any(values[size:]):
            raise ValueError("padding is outside the declared stalk")
        if batch.observed[row]:
            if row not in c["training_rows"]:
                raise ValueError("observation lies outside the original training relations")
            observations.append(dict(row=row, values=values[:size], batch=batch.coefficient_digest, step=state.step)); added += 1
        elif any(values):
            raise ValueError("unobserved rows cannot contain acquisition values")
    if not added:
        raise ValueError("stalk learning requires an observed relation")
    observations = tuple(observations)
    child = state.child(payload=dict(observations=observations,
        batches=(*state.payload["batches"], batch.coefficient_digest), acquisition_archive=_acquisition(c, observations)), step=state.step+1)
    _prepare(child, source)
    return child


def stalk_query(state, fields=None, *, include_acquisition=True):
    """Declare a whole source bound stalk field and whether acquired columns inject."""
    return dict(profile="whole-stalk-query-v1", fields=_zeros(state.configuration) if fields is None else _field(state.configuration, fields),
        include_acquisition=include_acquisition, source_reference=state.source.as_record())


def _infer(state, source, inputs):
    return infer_stalk_fields_many(state, source, (inputs,))[0]


def infer_stalk_fields_many(state, source, queries):
    """Execute complete queries together while retaining their original order."""
    plan = _prepare(state, source); c = state.configuration
    queries = tuple(queries)
    if len(queries) > c["limits"][0]:
        raise ValueError("whole query count exceeds field admission")
    unique = OrderedDict(); order = []; injections = {}
    for query in queries:
        inputs = freeze_tree(stalk_query(state) if query is None else query)
        _check_query(state, inputs)
        field = _field(c, inputs["fields"])
        key = (field, inputs["include_acquisition"])
        if key not in unique:
            enabled = inputs["include_acquisition"]
            if enabled not in injections:
                injections[enabled] = (plan.inject_acquisition(state.payload["acquisition_archive"],
                    tuple(range(sum(map(len, c["upper_axes"])))), (1,)*len(state.payload["observations"]))
                    if enabled and state.payload["acquisition_archive"] is not None else None)
            injection = injections[enabled]
            if injection is not None:
                field = tuple(tuple(_exact(a+b, c["limits"][3]) for a,b in zip(left,right,strict=True))
                    for left,right in zip(field,injection["field"],strict=True))
            unique[key] = (len(unique), field, injection)
        order.append((inputs,key))
    if not queries:
        return ()
    result = plan.apply(tuple(v[1] for v in unique.values())); outputs = []; retained = {}
    for inputs,key in order:
        index,field,injection = unique[key]
        output = _output(state, inputs, field, injection, result, index)
        digest = output.coefficient_digest
        outputs.append(retained.setdefault(digest,output))
    state.check_state()
    return tuple(outputs)


def _check_query(state, inputs):
    inputs = freeze_tree(stalk_query(state) if inputs is None else inputs)
    if (not isinstance(inputs, Mapping) or set(inputs) != {"profile", "fields", "include_acquisition", "source_reference"} or
            inputs["profile"] != "whole-stalk-query-v1" or type(inputs["include_acquisition"]) is not bool or
            dict(inputs["source_reference"]) != state.source.as_record()):
        raise ValueError("closed query bound to the actual model source required")


def _output(state, inputs, field, injection, result, index):
    c = state.configuration; response = result["fields"][index]
    width = len(state.output_axes[0].keys)
    values = np.full((len(response), width), Fraction(0), dtype=object)
    support = []
    for row, stalk in enumerate(response):
        values[row, :len(stalk)] = stalk
        support.append(tuple(i < len(stalk) for i in range(width)))
    receipt = dict(profile=_PROFILE, source_reference=state.source.as_record(),
        upper_axes=c["upper_axes"], lower_axes=c["lower_axes"], stalk_support=tuple(support),
        response_fields=response, boundary_fields=result["boundaries"][index], occurrence_fields=result["occurrence_fields"][index],
        native_receipt=result["receipt"], input_fields=field, query_declaration=inputs, acquisition_injection=injection,
        acquisition_archive=state.payload["acquisition_archive"], acquired_columns=state.payload["observations"],
        execution_steps=result["execution_steps"], residual_exact_zero=result["residual_exact_zero"],
        truth_status="prospective", readout="complete-named-stalk-fields-v1")
    return ModelOutput(values, state.space, state.output_axes, state.source, state.coefficient_digest,
        "rational", _METHOD, grade=1, variance="chain", execution_receipt=receipt)


def validate_stalk_output(output):
    """Check the complete named response and its explicitly absent padding."""
    r = output.execution_receipt
    if (not isinstance(r, Mapping) or r.get("profile") != _PROFILE or output.arithmetic != "rational" or
            not isinstance(output.source, CppSourceReference) or
            output.grade != 1 or output.variance != "chain" or r.get("readout") != "complete-named-stalk-fields-v1" or
            dict(r["source_reference"]) != output.source.as_record() or r.get("residual_exact_zero") is not True):
        raise ValueError("invalid whole stalk response declaration")
    upper = _axes(r["upper_axes"]); lower = _axes(r["lower_axes"])
    width = max(map(len, upper), default=0)
    if (len(upper) != len(output.space.keys) or output.values.shape != (len(upper), width) or
            output.axes != (CoordinateSpace("stalk-slot", tuple(str(i) for i in range(width))),) or
            r["stalk_support"] != tuple(tuple(i < len(row) for i in range(width)) for row in upper) or
            tuple(map(len, r["response_fields"])) != tuple(map(len, upper)) or
            tuple(map(len, r["boundary_fields"])) != tuple(map(len, lower))):
        raise ValueError("stalk response carrier or structural support differs")
    for i, row in enumerate(r["response_fields"]):
        if tuple(output.values[i, :len(row)]) != row or any(output.values[i, len(row):]):
            raise ValueError("stalk response differs from its named field or absent padding")
    native = r["native_receipt"]
    if native["upper_axes"] != upper or native["lower_axes"] != lower or native["archive_digest"] != output.source.native.archive_digest:
        raise ValueError("stalk response uses another native source or local carrier")
    if (len(upper) != len(output.source.native.coordinates[1]) or len(lower) != len(output.source.native.coordinates[0]) or
            output.space.keys != output.source.native.coordinates[1] or
            native["upper_dimension"] != sum(map(len, upper)) or native["lower_dimension"] != sum(map(len, lower))):
        raise ValueError("stalk fields differ from the original source coordinates")
    bits = native["limits"][3]
    parameter = _exact(native["parameter"], bits)
    if parameter <= 0 or native["metrics"] != "declared-unit-stalk-metrics":
        raise ValueError("stalk response requires positive Green parameter and unit metrics")
    for key, axes in (("response_fields", upper), ("input_fields", upper), ("boundary_fields", lower)):
        if tuple(map(len, r[key])) != tuple(map(len, axes)):
            raise ValueError("whole field differs from its original stalk axes")
        for values in r[key]:
            for value in values:
                _exact(value, bits)
    count = len(native["occurrence_cells"])
    if (len(r["occurrence_fields"]) != count or native["retained_occurrences"] != count or
            any(len(native[key]) != count for key in ("flag_maps", "occurrence_coefficients", "occurrence_activity"))):
        raise ValueError("stalk response omits original occurrence vectors")
    boundary = [[Fraction(0)]*len(names) for names in lower]
    residual = [list(values) for values in r["response_fields"]]
    for slot, (up, lo) in enumerate(native["occurrence_cells"]):
        if (type(up) is not int or type(lo) is not int or not 0 <= up < len(upper) or not 0 <= lo < len(lower) or
                type(native["occurrence_activity"][slot]) is not bool):
            raise ValueError("occurrence must name its actual original cells and activity")
        values = r["occurrence_fields"][slot]
        if len(values) != len(lower[lo]):
            raise ValueError("occurrence vector differs from its original lower stalk")
        coefficient = _exact(native["occurrence_coefficients"][slot], bits)
        if not native["occurrence_activity"][slot]:
            coefficient = Fraction(0)
        contribution = [Fraction(0)]*len(lower[lo]); seen = set()
        for row, column, value in native["flag_maps"][slot]:
            if (type(row) is not int or type(column) is not int or not 0 <= row < len(lower[lo]) or
                    not 0 <= column < len(upper[up]) or (row, column) in seen):
                raise ValueError("flag block differs from its original stalk coordinates")
            seen.add((row, column)); realized = _exact(coefficient*_exact(value, bits), bits)
            contribution[row] = _exact(contribution[row] + realized*r["response_fields"][up][column], bits)
            residual[up][column] = _exact(residual[up][column] + parameter*realized*r["boundary_fields"][lo][row], bits)
        if tuple(_exact(v, bits) for v in values) != tuple(contribution):
            raise ValueError("occurrence vector differs from its retained flag map")
        for i, value in enumerate(contribution):
            boundary[lo][i] = _exact(boundary[lo][i]+value, bits)
    if tuple(map(tuple, boundary)) != r["boundary_fields"]:
        raise ValueError("original occurrence fields do not assemble to the retained boundary")
    if tuple(map(tuple, residual)) != r["input_fields"]:
        raise ValueError("retained whole field has a nonzero exact Green residual")
    query = r["query_declaration"]
    if (set(query) != {"profile", "fields", "include_acquisition", "source_reference"} or
            query["profile"] != "whole-stalk-query-v1" or type(query["include_acquisition"]) is not bool or
            dict(query["source_reference"]) != output.source.as_record()):
        raise ValueError("stalk response requires its original source-bound query")
    base = _field({"upper_axes": upper, "limits": native["limits"]}, query["fields"])
    acquisition = [[Fraction(0)]*len(names) for names in upper]
    observations = r["acquired_columns"]
    for observation in observations:
        row = observation["row"]
        if type(row) is not int or not 0 <= row < len(upper) or len(observation["values"]) != len(upper[row]):
            raise ValueError("acquired column differs from its original named stalk")
        for i, value in enumerate(observation["values"]):
            acquisition[row][i] = _exact(acquisition[row][i] + _exact(value, bits), bits)
    archive = r["acquisition_archive"]
    if (not observations and archive is not None) or (observations and not isinstance(archive, bytes)):
        raise ValueError("acquired observations require their retained original source archive")
    injection = r["acquisition_injection"]
    if query["include_acquisition"] and observations:
        from hashlib import sha256
        if (not isinstance(injection, Mapping) or injection["profile"] != "declared-acquisition-boundary-injection-v1" or
                injection["acquisition_archive_digest"] != sha256(archive).hexdigest() or
                injection["columns"] != len(observations) or injection["mapping"] != tuple(range(sum(map(len, upper)))) or
                injection["amplitudes"] != (Fraction(1),)*len(observations) or
                injection["retained_occurrences"] != sum(len(o["values"]) for o in observations) or
                injection["field"] != tuple(map(tuple, acquisition)) or
                injection["boundary"] != tuple(v for row in acquisition for v in row)):
            raise ValueError("acquisition injection differs from the retained whole columns")
        expected = tuple(tuple(_exact(a+b, bits) for a,b in zip(left,right,strict=True))
                         for left,right in zip(base,acquisition,strict=True))
    else:
        if injection is not None:
            raise ValueError("query did not admit an acquisition injection")
        expected = base
    if expected != r["input_fields"]:
        raise ValueError("executed field differs from its original query and acquisition")


def stalk_output_field(output, row=None):
    """Select one original stalk or expose a field with one common named axis."""
    output.check_state(); validate_stalk_output(output); axes = output.execution_receipt["upper_axes"]
    if row is None:
        if any(names != axes[0] for names in axes):
            raise TypeError("ragged stalks require an explicit original relation row")
        values = output.values; space = output.space; names = axes[0]
    else:
        if type(row) is not int or not 0 <= row < len(axes):
            raise ValueError("select an original relation row")
        names = axes[row]; values = output.values[row:row+1, :len(names)]
        space = CoordinateSpace(output.space.name, (output.space.keys[row],))
    return TensorField(space, values, (CoordinateSpace("named-stalk-axis", names),), output.source,
        grade=1, variance="chain", provenance=(output.model_digest, output.coefficient_digest))


def _create(source, *, adapter="cpp_stalk_fields", **configuration):
    if adapter != "cpp_stalk_fields":
        raise ValueError("unsupported stalk adapter")
    return cpp_stalk_fields_model(source, **configuration)


def enable_stalk_fields_provider():
    """Register exact stalk acquisition, native preparation and whole field inference."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], _PROVIDER["version"], (("cpp_stalk_fields", "1"),),
                _infer, create=_create, prepare=_prepare, train=_train)
        register_model_provider(_provider)
        return _provider
