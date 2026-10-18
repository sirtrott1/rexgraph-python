"""Acquired local tensor charts and native sections on typed source maps."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.cpp_stalk_fields import _exact
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "acquired-local-tensor-sections-v1"
_METHOD = "native-local-tensor-sections-v1"
_PROVIDER = {"name": "rexgraph.cpp.local_sections", "version": "1"}
_lock = RLock()
_owners = OrderedDict()
_provider = None


def _name(value):
    if not isinstance(value, str) or not value or len(value.encode()) > 256:
        raise ValueError("local coordinates require bounded nonempty names")
    return value


def _indices(values, size, description):
    values = tuple(values)
    if len(set(values)) != len(values) or any(type(i) is not int or not 0 <= i < size for i in values):
        raise ValueError(description)
    return values


def _same_tree(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and set(left) == set(right) and all(_same_tree(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _stalks(stalks, reference, bits):
    result = []; identities = set()
    for stalk in stalks:
        if not isinstance(stalk, Mapping) or set(stalk) != {"kind", "index", "axes", "alternatives"}:
            raise ValueError("closed local stalk declaration required")
        kind, index = stalk["kind"], stalk["index"]
        if kind not in {"lower", "relation", "occurrence"} or type(index) is not int or index < 0:
            raise ValueError("stalk must name an original source cell or occurrence")
        if kind != "occurrence" and index >= len(reference.native.coordinates[kind == "relation"]):
            raise ValueError("stalk lies outside its original source grade")
        if (kind, index) in identities:
            raise ValueError("one whole stalk per original cell or occurrence required")
        identities.add((kind, index))
        axes = tuple(_name(a) for a in stalk["axes"])
        if not axes or len(set(axes)) != len(axes):
            raise ValueError("distinct local tensor axes required")
        alternatives = tuple((_name(name), tuple(_exact(v, bits) for v in values))
                             for name, values in stalk["alternatives"])
        if (not alternatives or len({name for name, _ in alternatives}) != len(alternatives)
                or any(len(values) != len(axes) for _, values in alternatives)):
            raise ValueError("whole named alternatives must retain their local shape")
        result.append(dict(kind=kind, index=index, axes=axes, alternatives=alternatives))
    if not result:
        raise ValueError("local sections require original stalk declarations")
    return tuple(result)


def validate_local_sections_state(state):
    c = state.configuration
    if (state.adapter != "cpp_local_sections" or state.adapter_version != "1"
            or state.arithmetic != "rational" or not isinstance(state.source, CppSourceReference)
            or c.get("profile") != _PROFILE or dict(c.get("execution_provider", {})) != _PROVIDER
            or state.output_axes or len(state.source.native.coordinates) != 2):
        raise ValueError("unsupported local tensor section model")
    if set(c) != {"profile", "execution_provider", "stalks", "families", "training_rows",
                  "guard_limits", "section_limits", "max_bytes", "guard_receipt", "sections_receipt", "source_accession",
                  "field_model", "field_constraints"}:
        raise ValueError("closed local tensor model declaration required")
    if (len(c["guard_limits"]) != 4 or len(c["section_limits"]) != 3
            or any(type(v) is not int or v <= 0 for v in (*c["guard_limits"], *c["section_limits"], c["max_bytes"]))):
        raise ValueError("positive native map, section and byte admissions required")
    stalks = _stalks(c["stalks"], state.source, c["guard_limits"][3])
    entries = sum(1+len(s["axes"])+sum(1+len(v) for _, v in s["alternatives"]) for s in stalks)
    if entries > c["guard_limits"][1]:
        raise ValueError("local tensor declaration entry admission exceeded")
    if state.space != CoordinateSpace("original-local-stalk", tuple(s["kind"]+":"+str(s["index"]) for s in stalks)):
        raise ValueError("model coordinates differ from their original source accessions")
    if not c["families"] or len(set(c["families"])) != len(c["families"]):
        raise ValueError("distinct acquired constraint families required")
    for family in c["families"]:
        _name(family)
    _field_program(state)
    _indices(c["training_rows"], len(state.source.native.coordinates[1]), "training rows outside original relations")
    p = state.payload
    if set(p) != {"observations", "batches", "acquisition_archive"} or len(p["batches"]) != state.step:
        raise ValueError("local charts require retained observations and batch identities")
    grouped = [[] for _ in p["batches"]]
    if len(p["observations"]) > c["guard_limits"][1]:
        raise ValueError("acquired chart entry admission exceeded")
    for observation in p["observations"]:
        _observation(c, observation, stored=True)
        step = observation["step"]
        if type(step) is not int or not 0 <= step < state.step or observation["batch"] != p["batches"][step]:
            raise ValueError("acquired chart differs from its producing batch")
        grouped[step].append(observation)
    if any(not group for group in grouped):
        raise ValueError("each acquisition step requires a local observation")


def _observation(c, observation, *, stored=False):
    keys = {"family", "rows", "support", "values", "anchors"}
    if stored:
        keys |= {"batch", "step"}
    if not isinstance(observation, Mapping) or set(observation) != keys or observation["family"] not in c["families"]:
        raise ValueError("closed declared local chart required")
    rows = tuple(observation["rows"])
    if not rows or len(set(rows)) != len(rows) or any(type(r) is not int or r not in c["training_rows"] for r in rows):
        raise ValueError("acquisition must name complete original training relation rows")
    support = _indices(observation["support"], len(c["stalks"]), "local chart support outside original stalks")
    if not support or len(observation["values"]) != len(support):
        raise ValueError("local chart values must cover their whole declared support")
    for variable, name in zip(support, observation["values"], strict=True):
        if name not in dict(c["stalks"][variable]["alternatives"]):
            raise ValueError("observed tensor outside its whole local alternative family")
    anchors = tuple(observation["anchors"])
    if len(set(anchors)) != len(anchors) or any(type(i) is not int or i < 0 for i in anchors):
        raise ValueError("anchors must be distinct original lower identities")


def _scope(owner, rows):
    lower = set(); occurrences = set()
    for row in rows:
        for flag in owner.boundary(row)["occurrences"]:
            lower.add(flag["lower"]); occurrences.add(flag["slot"])
    return {"lower": lower, "occurrence": occurrences, "relation": set(rows)}


def _check_scope(c, owner, observation):
    scope = _scope(owner, observation["rows"])
    if any(s["index"] not in scope[s["kind"]] for s in (c["stalks"][i] for i in observation["support"])):
        raise ValueError("local chart must retain its full original supporting subcomplex")
    if not set(observation["anchors"]).issubset(scope["lower"]):
        raise ValueError("fixed meaning anchors must belong to the original boundary")


def _field_program(state):
    from rexgraph.model_codec import unpack_model
    c = state.configuration
    if c["field_model"] is None:
        if c["field_constraints"]:
            raise ValueError("native field constraints require their declared whole field model")
        return None
    field = unpack_model(c["field_model"])
    if field.adapter != "cpp_stalk_fields" or field.source.native != state.source.native:
        raise ValueError("local field program must bind the original whole stalk source")
    for stalk in c["stalks"]:
        if stalk["kind"] != "relation" or stalk["axes"] != field.configuration["upper_axes"][stalk["index"]]:
            raise ValueError("local field injection requires the original relation stalk basis")
    for constraint in c["field_constraints"]:
        if not isinstance(constraint, Mapping) or set(constraint) != {"kind", "cells"}:
            raise ValueError("closed whole native field constraint required")
        if constraint["kind"] not in {"response-boundary-zero", "response-occurrence-equality"}:
            raise ValueError("unsupported whole native field constraint")
        size = len(field.configuration["lower_axes"]) if constraint["kind"] == "response-boundary-zero" else len(field.configuration["flag_maps"])
        cells = _indices(constraint["cells"], size, "field constraint outside its original source carrier")
        if not cells or constraint["kind"] == "response-occurrence-equality" and len(cells) < 2:
            raise ValueError("whole field constraint requires its complete declared interface")
    return field


def _filter_fields(state, source, required, native_sections):
    from rexgraph.cpp_stalk_fields import infer_stalk_fields_many, stalk_query
    from rexgraph.model_codec import pack_model
    field = _field_program(state)
    if field is None:
        return tuple(range(len(native_sections))), (), ()
    from rexgraph.cpp_stalk_fields import _prepare as prepare_fields
    plan = prepare_fields(field, source); c = state.configuration; constraints = c["field_constraints"]
    occurrence_cells = plan.receipt["occurrence_cells"]
    for constraint in constraints:
        if constraint["kind"] == "response-occurrence-equality":
            if len({occurrence_cells[i][1] for i in constraint["cells"]}) != 1:
                raise ValueError("occurrence equality requires one actual shared lower stalk")
    queries = []
    for case in native_sections:
        values = [list(row) for row in stalk_query(field, include_acquisition=False)["fields"]]
        for variable, index in zip(required, case, strict=True):
            stalk = c["stalks"][variable]
            values[stalk["index"]] = list(stalk["alternatives"][index][1])
        queries.append(stalk_query(field, values, include_acquisition=True))
    outputs = infer_stalk_fields_many(field, source, queries)
    accepted = []; checks = []
    for index, output in enumerate(outputs):
        readings = []
        for constraint in constraints:
            r = output.execution_receipt
            if constraint["kind"] == "response-boundary-zero":
                vectors = tuple(r["boundary_fields"][i] for i in constraint["cells"])
                compatible = all(all(v == 0 for v in vector) for vector in vectors)
            else:
                vectors = tuple(r["occurrence_fields"][i] for i in constraint["cells"])
                compatible = all(vector == vectors[0] for vector in vectors[1:])
            readings.append(dict(constraint=constraint, vectors=vectors, compatible=compatible))
        admitted = all(reading["compatible"] for reading in readings)
        if admitted:
            accepted.append(index)
        checks.append(dict(finite_case=index, readings=tuple(readings), admitted=admitted))
    return tuple(accepted), tuple(pack_model(output) for output in outputs), tuple(checks)


def _acquisition(c, observations):
    from rexgraph_cpp import NativeSource
    if not observations:
        return None
    offsets = [0]
    for stalk in c["stalks"]:
        offsets.append(offsets[-1]+len(stalk["axes"]))
    relations = []; key = 0
    for observation in observations:
        flags = []
        for variable, name in zip(observation["support"], observation["values"], strict=True):
            values = dict(c["stalks"][variable]["alternatives"])[name]
            for component, value in enumerate(values):
                key += 1
                flags.append(dict(key=key, lower=offsets[variable]+component, exists=True,
                                  negative=value < 0, magnitude=abs(value), aspects=(0,)))
        relations.append(dict(grade=1, key=len(relations)+1, type=c["families"].index(observation["family"]),
                              occurrences=tuple(flags)))
    return NativeSource.from_declaration(dict(profile="declared-role-source-v1", namespace=941117,
        semantic_scale=2, max_grade=1, role_aspects=("observed-tensor-coordinate",),
        types=tuple(dict(name=name, min_arity=0, max_arity=0) for name in c["families"]),
        witnesses=tuple(range(offsets[-1])), relations=tuple(relations), carriers=(), fields=(), metrics=()),
        max_bytes=c["max_bytes"]).archive


def _check_accession(state, source, owner):
    from rexgraph.linguistic_accession import validate_linguistic_accession
    validate_linguistic_accession(state.configuration["source_accession"], source,
        max_bytes=state.configuration["max_bytes"], owner=owner)


def _prepare(state, source):
    from rexgraph_cpp import PreparedRelationGuards, guards_build_info, sections_build_info
    state.bind(source); c = state.configuration
    if dict(c["guard_receipt"]) != guards_build_info() or dict(c["sections_receipt"]) != sections_build_info():
        raise ValueError("local charts require their retained native implementations")
    key = state._digest
    with _lock:
        if key not in _owners:
            validate_local_sections_state(state)
            owner = PreparedRelationGuards(source.archive, limits=c["guard_limits"][1:], max_bytes=c["max_bytes"])
            _check_accession(state, source, owner)
            field = _field_program(state)
            if field is not None:
                from rexgraph.cpp_stalk_fields import _prepare as prepare_fields
                prepare_fields(field, source)
            occurrences = _scope(owner, range(len(state.source.native.coordinates[1])))["occurrence"]
            if any(s["kind"] == "occurrence" and s["index"] not in occurrences for s in c["stalks"]):
                raise ValueError("stalk occurrence outside its retained original source")
            for observation in state.payload["observations"]:
                _check_scope(c, owner, observation)
            if _acquisition(c, state.payload["observations"]) != state.payload["acquisition_archive"]:
                raise ValueError("acquired tensor columns differ from the retained local observations")
            _owners[key] = owner
            while len(_owners) > 4:
                _owners.popitem(last=False)
        _owners.move_to_end(key)
        return _owners[key]


def cpp_local_sections_model(source, stalks, families, training_rows, *,
                             guard_limits=(4096, 1000000, 64000000, 16384),
                             section_limits=(65536, 1000000, 64000000),
                             max_bytes=256*1024*1024, reference=None, dependencies=(), source_accession=None,
                             field_model=None, field_constraints=()):
    """Declare whole local alternatives on original cells and use occurrences."""
    from rexgraph_cpp import guards_build_info, sections_build_info
    reference = CppSourceReference(source) if reference is None else reference.bind(source)
    if len(reference.native.coordinates) != 2:
        raise ValueError("local chart map profile requires a grade one source")
    if len(guard_limits) != 4 or any(type(v) is not int or v <= 0 for v in guard_limits):
        raise ValueError("four positive native map admissions required")
    stalks = _stalks(stalks, reference, guard_limits[3])
    from rexgraph.model_codec import pack_model
    configuration = dict(profile=_PROFILE, execution_provider=_PROVIDER, stalks=stalks,
        families=tuple(families), training_rows=tuple(training_rows), guard_limits=tuple(guard_limits),
        section_limits=tuple(section_limits), max_bytes=max_bytes,
        guard_receipt=guards_build_info(), sections_receipt=sections_build_info(), source_accession=source_accession,
        field_model=None if field_model is None else pack_model(field_model), field_constraints=tuple(field_constraints))
    state = ModelState("cpp_local_sections", "1", reference,
        CoordinateSpace("original-local-stalk", tuple(s["kind"]+":"+str(s["index"]) for s in stalks)), (),
        configuration, dict(observations=(), batches=(), acquisition_archive=None), "rational", dependencies=dependencies)
    _prepare(state, source)
    return state


def local_section_batch(state, observations):
    """Retain partial chart scopes, whole tensor targets and explicit anchors."""
    observations = freeze_tree(tuple(observations))
    if not observations:
        raise ValueError("nonempty local chart acquisition required")
    c = state.configuration; targets = np.zeros((len(state.space.keys),), dtype=np.int64)
    observed = np.zeros((len(state.space.keys),), dtype=bool)
    for observation in observations:
        _observation(c, observation)
        for variable in observation["support"]:
            observed[variable] = True
    return ModelBatch(state.source, state.space, targets, observed,
        dict(profile=_PROFILE, observations=observations), dependencies=state.dependencies)


def _train(state, source, batch, *, steps):
    owner = _prepare(state, source); c = state.configuration
    if steps != 1:
        raise ValueError("local charts acquire one declared observation batch per step")
    if not isinstance(batch.inputs, Mapping) or set(batch.inputs) != {"profile", "observations"} or batch.inputs["profile"] != _PROFILE:
        raise ValueError("local acquisition requires its original observation charts")
    expected = local_section_batch(state, batch.inputs["observations"])
    if expected.coefficient_digest != batch.coefficient_digest:
        raise ValueError("local batch differs from its retained chart observations")
    observations = list(state.payload["observations"])
    for observation in batch.inputs["observations"]:
        _check_scope(c, owner, observation)
        observations.append(dict(observation, batch=batch.coefficient_digest, step=state.step))
    if len(observations) > c["guard_limits"][1]:
        raise ValueError("acquired chart entry admission exceeded")
    child = state.child(payload=dict(observations=tuple(observations),
        batches=(*state.payload["batches"], batch.coefficient_digest),
        acquisition_archive=_acquisition(c, observations)), step=state.step+1)
    _prepare(child, source)
    return child


def local_section_query(state, variables, patches, *, restrictions=()):
    """Select original stalks and complete target relation patches for transfer."""
    variables = _indices(variables, len(state.space.keys), "query variables outside original stalks")
    patches = tuple(_indices(rows, len(state.source.native.coordinates[1]), "query patch outside original relations") for rows in patches)
    if not variables or any(not rows for rows in patches) or len(set(patches)) != len(patches):
        raise ValueError("nonempty variables and distinct complete target patches required")
    restrictions = tuple((i, tuple(names)) for i, names in restrictions)
    _indices((i for i, _ in restrictions), len(state.space.keys), "repeated or invalid query restriction")
    for i, names in restrictions:
        if not names or len(set(names)) != len(names) or any(n not in dict(state.configuration["stalks"][i]["alternatives"]) for n in names):
            raise ValueError("query restriction outside its whole local alternative family")
    return dict(profile=_PROFILE, source_reference=state.source.as_record(), variables=variables,
                patches=patches, restrictions=restrictions)


def _transfers(state, owner, query):
    c = state.configuration; stalks = c["stalks"]
    lookup = {(s["kind"], s["index"]): i for i, s in enumerate(stalks)}
    observations = state.payload["observations"]
    attempts = sum(len(o["rows"]) == len(rows) for o in observations for rows in query["patches"])
    if attempts > c["guard_limits"][0]:
        raise ValueError("local chart map candidate admission exceeded")
    work = owner.receipt["preparation_steps"]; cache = {}; charts = OrderedDict(); proofs = []
    for index, observation in enumerate(observations):
        for rows in query["patches"]:
            if len(rows) != len(observation["rows"]):
                continue
            key = observation["rows"], rows, observation["anchors"]
            if key not in cache:
                remaining = c["guard_limits"][2]-work
                if remaining <= 0:
                    raise ValueError("local chart map work admission exceeded")
                cache[key] = owner.match(observation["rows"], rows, fixed_lowers=observation["anchors"], work_limit=remaining)
                work += cache[key]["execution_steps"]
            witness = cache[key]
            if not witness["matched"]:
                continue
            maps = {"lower": dict(witness["cell_map"]), "relation": dict(witness["relation_map"]),
                    "occurrence": dict(witness["occurrence_map"])}
            support = []
            for original in observation["support"]:
                a = stalks[original]; target = lookup.get((a["kind"], maps[a["kind"]][a["index"]]))
                if target is None:
                    raise ValueError("transferred chart requires every original dependent stalk")
                b = stalks[target]
                if a["axes"] != b["axes"] or a["alternatives"] != b["alternatives"]:
                    raise ValueError("local field transfer requires the same declared tensor basis and alternatives")
                support.append(target)
            key = observation["family"], rows, tuple(support)
            chart = charts.setdefault(key, dict(family=observation["family"], rows=rows, support=tuple(support), allowed=[], proofs=[]))
            if observation["values"] not in chart["allowed"]:
                chart["allowed"].append(observation["values"])
            chart["proofs"].append(len(proofs))
            proofs.append(dict(observation=index, target_rows=rows, support=tuple(support), witness=witness))
    return tuple(charts.values()), tuple(proofs), dict(attempts=attempts, native_calls=len(cache), execution_steps=work)


def _solve(state, source, inputs):
    from rexgraph_cpp import NativeSource, PreparedRelationalSections
    owner = _prepare(state, source); query = freeze_tree(inputs)
    if not isinstance(query, Mapping) or set(query) != {"profile", "source_reference", "variables", "patches", "restrictions"}:
        raise ValueError("closed local tensor query required")
    expected = freeze_tree(local_section_query(state, query["variables"], query["patches"], restrictions=query["restrictions"]))
    if query != expected:
        raise ValueError("local query differs from its actual source declaration")
    acquired, proofs, admissions = _transfers(state, owner, query)
    required = tuple(dict.fromkeys((*query["variables"], *(i for chart in acquired for i in chart["support"]),
                                  *(i for i, _ in query["restrictions"]))))
    slots = {variable: i for i, variable in enumerate(required)}
    stalks = state.configuration["stalks"]
    domains = tuple(tuple(range(len(stalks[i]["alternatives"]))) for i in required)
    relations = []; charts = []; key = 0
    def add(name, support, allowed):
        nonlocal key
        flags = []
        for i in support:
            key += 1
            flags.append(dict(key=key, lower=i, exists=True, negative=False, magnitude=1))
        row = len(relations)
        relations.append(dict(grade=1, key=row+1, type=0, occurrences=tuple(flags)))
        charts.append(dict(name=name, relation=row, support=tuple(support), allowed=tuple(sorted(set(allowed)))))
    for index, chart in enumerate(acquired):
        add("acquired:"+str(index), tuple(slots[i] for i in chart["support"]),
            tuple(tuple(tuple(n for n, _ in stalks[i]["alternatives"]).index(value)
                        for i, value in zip(chart["support"], row, strict=True)) for row in chart["allowed"]))
    for i, names in query["restrictions"]:
        labels = tuple(n for n, _ in stalks[i]["alternatives"])
        add("restriction:"+str(i), (slots[i],), tuple((labels.index(n),) for n in names))
    covered = set(); enumeration = []
    for i in sorted(range(len(charts)), key=lambda i: (len(charts[i]["allowed"]), -len(charts[i]["support"]), i)):
        if not set(charts[i]["support"]).issubset(covered):
            enumeration.append(i); covered.update(charts[i]["support"])
    for slot, domain in enumerate(domains):
        if slot not in covered:
            enumeration.append(len(charts)); add("uncovered:"+str(required[slot]), (slot,), tuple((v,) for v in domain))
    c = state.configuration
    carrier = NativeSource.from_declaration(dict(profile="declared-exact-source-v1", namespace=941118,
        semantic_scale=2, max_grade=1, types=(dict(name="local-tensor-chart", min_arity=0, max_arity=0),),
        witnesses=tuple(range(len(required))), relations=tuple(relations), fields=(), metrics=(), carriers=()), max_bytes=c["max_bytes"])
    solved = PreparedRelationalSections(carrier.archive, domains, charts, limits=c["section_limits"],
        max_bytes=c["max_bytes"], enumeration_charts=tuple(enumeration)).sections()
    accepted, field_outputs, checks = _filter_fields(state, source, required, solved["sections"])
    assignments = tuple(tuple(stalks[i]["alternatives"][solved["sections"][index][slots[i]]][0] for i in query["variables"]) for index in accepted)
    components = tuple(str(i)+":"+axis for i in query["variables"] for axis in stalks[i]["axes"])
    values = np.empty((len(assignments), len(components)), dtype=object)
    for case, names in enumerate(assignments):
        values[case] = tuple(v for i, name in zip(query["variables"], names, strict=True)
                             for v in dict(stalks[i]["alternatives"])[name])
    supported = tuple(any(i in chart["support"] for chart in acquired) for i in query["variables"])
    receipt = dict(profile=_PROFILE, query=query, assignments=assignments, supported=supported,
        original_source_archive=source.archive,
        required_variables=required, acquired_charts=acquired, map_proofs=proofs, map_admissions=admissions,
        section_source_archive=carrier.archive, section_domains=domains, section_charts=tuple(charts),
        enumeration_charts=tuple(enumeration), sections=solved,
        admitted_finite_cases=accepted, native_field_outputs=field_outputs, native_field_checks=checks,
        readout="all-whole-local-tensor-sections", truth_status="prospective")
    return values, components, receipt


def _infer(state, source, inputs):
    from rexgraph.model_codec import pack_model
    values, components, receipt = _solve(state, source, inputs)
    receipt["model_checkpoint"] = pack_model(state)
    return ModelOutput(values, CoordinateSpace("compatible-local-sections", tuple(str(i) for i in range(len(values)))),
        (CoordinateSpace("local-tensor-component", components),), state.source, state.coefficient_digest,
        "rational", _METHOD, dependencies=state.dependencies, execution_receipt=receipt)


def validate_local_sections_output(output):
    from rexgraph.model_codec import unpack_model
    r = output.execution_receipt
    if not isinstance(r, Mapping) or r.get("profile") != _PROFILE or r.get("readout") != "all-whole-local-tensor-sections":
        raise ValueError("whole local tensor section receipt required")
    state = unpack_model(r["model_checkpoint"])
    if state.adapter != "cpp_local_sections" or state.coefficient_digest != output.model_digest or not output.source.matches(state.source):
        raise ValueError("local sections require their actual producing checkpoint and source")
    if (output.arithmetic != "rational" or output.grade is not None or output.variance != "coordinate"
            or tuple(d.coefficient_digest for d in output.dependencies) != tuple(d.coefficient_digest for d in state.dependencies)):
        raise ValueError("local sections differ from their declared field contract")
    from rexgraph_cpp import NativeSource
    source = NativeSource(r["original_source_archive"], max_bytes=state.configuration["max_bytes"])
    expected, components, receipt = _solve(state, source, r["query"])
    if (not _same_tree(freeze_tree(receipt), {k: v for k, v in r.items() if k != "model_checkpoint"})
            or not np.array_equal(expected, output.values)
            or output.space != CoordinateSpace("compatible-local-sections", tuple(str(i) for i in range(len(expected))))
            or output.axes != (CoordinateSpace("local-tensor-component", components),)):
        raise ValueError("local tensor output differs from its whole native assignments")


def local_section_fields(output, case):
    """Return each named ragged stalk of one retained compatible interpretation."""
    from rexgraph.model_codec import unpack_model
    output.check_state(); validate_local_sections_output(output)
    if type(case) is not int or not 0 <= case < len(output.space.keys):
        raise ValueError("select an original compatible section")
    state = unpack_model(output.execution_receipt["model_checkpoint"]); result = []
    for variable, name, supported in zip(output.execution_receipt["query"]["variables"],
            output.execution_receipt["assignments"][case], output.execution_receipt["supported"], strict=True):
        stalk = state.configuration["stalks"][variable]
        result.append(dict(kind=stalk["kind"], index=stalk["index"], axes=stalk["axes"],
            alternative=name, values=dict(stalk["alternatives"])[name], supported=supported))
    return freeze_tree(tuple(result))


def replay_local_sections_output(output, source):
    """Reconstruct source maps, chart gluing and whole tensors on the actual source."""
    from rexgraph.model_codec import unpack_model
    output.check_state(); validate_local_sections_output(output)
    state = unpack_model(output.execution_receipt["model_checkpoint"])
    replay = _infer(state, source, output.execution_receipt["query"])
    if replay.coefficient_digest != output.coefficient_digest:
        raise ValueError("local output differs from its complete native map and section replay")
    return replay


def _create(source, *, adapter="cpp_local_sections", **configuration):
    if adapter != "cpp_local_sections":
        raise ValueError("unsupported local section adapter")
    return cpp_local_sections_model(source, **configuration)


def extend_local_sections_model(state, previous, source, stalks, training_rows, *, source_accession=None, field_model=None):
    """Retain acquired charts through a checked original source prefix inclusion."""
    from rexgraph_cpp import PreparedRelationGuards
    old = _prepare(state, previous)
    reference = CppSourceReference(source)
    if (reference.native.namespace != state.source.native.namespace
            or len(reference.native.coordinates) != 2
            or any(tuple(b[:len(a)]) != a for a, b in zip(state.source.native.coordinates, reference.native.coordinates, strict=True))):
        raise ValueError("source extension must retain every original primary identity")
    c = state.configuration
    new = PreparedRelationGuards(source.archive, limits=c["guard_limits"][1:], max_bytes=c["max_bytes"])
    for row in range(len(state.source.native.coordinates[1])):
        if new.boundary(row) != old.boundary(row):
            raise ValueError("source extension changes an original typed occurrence column")
    family = tuple(tuple(((i, Fraction(1)),) for i in range(len(grade))) for grade in state.source.native.coordinates)
    previous.chain_map(source, family)
    stalks = _stalks(stalks, reference, c["guard_limits"][3])
    if freeze_tree(stalks[:len(c["stalks"])]) != c["stalks"]:
        raise ValueError("source extension changes an original local tensor stalk")
    rows = tuple(training_rows)
    if not set(c["training_rows"]).issubset(rows):
        raise ValueError("source extension removes original acquisition scopes")
    if c["field_model"] is not None and field_model is None:
        raise ValueError("source extension requires a complete declared field program on new directions")
    if c["field_model"] is not None:
        from rexgraph.model_codec import unpack_model
        previous_field = unpack_model(c["field_model"])
        if field_model.adapter != "cpp_stalk_fields":
            raise ValueError("source extension requires its original native field adapter")
        a, b = previous_field.configuration, field_model.configuration
        if (any(b[k][:len(a[k])] != a[k] for k in ("upper_axes", "lower_axes", "flag_maps"))
                or any(b[k] != a[k] for k in ("parameter", "limits", "max_bytes"))
                or field_model.step != previous_field.step
                or any(not _same_tree(previous_field.payload[k], field_model.payload[k]) for k in ("observations", "batches"))):
            raise ValueError("source extension changes an original field program or acquisition")
    base = cpp_local_sections_model(source, stalks, c["families"], rows,
        guard_limits=c["guard_limits"], section_limits=c["section_limits"], max_bytes=c["max_bytes"],
        dependencies=(*state.dependencies, state.source), source_accession=source_accession,
        field_model=field_model, field_constraints=c["field_constraints"])
    if c["source_accession"] is not None:
        if source_accession is None:
            raise ValueError("source extension must retain original byte and linguistic accessions")
        old_accession = c["source_accession"]; new_accession = base.configuration["source_accession"]
        if (new_accession["lower"][:len(old_accession["lower"])] != old_accession["lower"]
                or new_accession["charts"][:len(old_accession["charts"])] != old_accession["charts"]
                or any(new_accession["rows"].get(k) != v for k, v in old_accession["rows"].items())):
            raise ValueError("source extension changes an original byte or linguistic accession")
    child = state.child(source=base.source, space=base.space, configuration=base.configuration,
        payload=dict(state.payload, acquisition_archive=_acquisition(base.configuration, state.payload["observations"])),
        dependencies=base.dependencies)
    _prepare(child, source)
    return child


def enable_local_sections_provider():
    """Register native typed local acquisition and complete compatible tensors."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], _PROVIDER["version"], (("cpp_local_sections", "1"),),
                _infer, create=_create, prepare=_prepare, train=_train)
        register_model_provider(_provider)
        return _provider
