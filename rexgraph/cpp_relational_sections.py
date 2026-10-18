"""Joint annotation fields and compatible finite relational interpretations."""
from collections.abc import Mapping
from collections import OrderedDict
from fractions import Fraction
from threading import RLock
from types import SimpleNamespace

import numpy as np

from rexgraph.cpp_relation_response import cpp_relation_response_model, _prepare as _prepare_field
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "joint-annotation-fields-relational-sections-v1"
_GUARDED_PROFILE = "guarded-joint-annotation-fields-relational-sections-v2"
_DICTIONARY_PROFILE = "dictionary-bound-joint-relational-sections-v3"
_TRANSPORT_PROFILE = "learned-role-dictionary-relational-sections-v4"
_PROVIDER = {"name": "rexgraph.cpp.relational_sections", "version": "1"}
_lock = RLock()
_provider = None
_prepared = OrderedDict()


def _field_state(state):
    return ModelState("cpp_relation_response", "2" if state.adapter_version == "4" else "1", state.source, state.space, (),
        state.configuration["field"], {"fields": state.payload["fields"], "batches": state.payload["batches"]},
        "rational", step=state.step, parent=state.parent, dependencies=state.dependencies)


def _context(c, row):
    accession = c["field"]["source_accession"]
    if type(row) is not int or not 0 <= row < len(accession["relations"]):
        raise ValueError("context must name an original relation")
    relation = accession["relations"][row]
    if relation["kind"] != "observed-context":
        raise ValueError("joint interpretation requires an observed context relation")
    qid = relation["provenance"]["query"]
    if type(qid) is not int or not 0 <= qid < len(accession["charts"]):
        raise ValueError("context has no literal chart accession")
    chart = accession["charts"][qid]
    raw = bytes.fromhex(chart["raw_utf8_hex"])
    raw.decode("utf-8")
    target = chart["target"]
    if (len(target) != 2 or any(type(i) is not int for i in target)
            or not 0 <= target[0] < target[1] <= len(raw) or tuple(target) != tuple(relation["provenance"]["target"])):
        raise ValueError("literal target differs from its retained context accession")
    previous = 0
    for word, span in zip(chart["words"], chart["spans"], strict=True):
        if (len(span) != 2 or any(type(i) is not int for i in span)
                or not previous <= span[0] < span[1] <= len(raw) or raw[slice(*span)].decode("utf-8") != word):
            raise ValueError("literal word differs from its ordered byte interval")
        previous = span[1]
    return raw.hex(), chart


def _source(nodes, relations, roles, namespace, max_bytes=256*1024*1024):
    from rexgraph_cpp import NativeSource
    return NativeSource.from_declaration(dict(profile="declared-role-source-v1", namespace=namespace,
        semantic_scale=2, max_grade=1, role_aspects=tuple(roles),
        types=(dict(name="joint-context-correspondence", min_arity=0, max_arity=0),
               dict(name="declared-compatibility", min_arity=0, max_arity=0)),
        witnesses=tuple(range(len(nodes))), relations=tuple(relations), fields=(), metrics=(),
        carriers=(dict(key=10, kind="source_grade", grade=0, coordinates=0, size=len(nodes)),
                  dict(key=11, kind="source_grade", grade=1, coordinates=0, size=len(relations)))), max_bytes=max_bytes)


def _acquisition(c, fields, observations=(), dictionary_observations=()):
    """Retain each acquired context and complete annotation tuple as one column."""
    nodes = {}; relations = []; records = []; key = 0
    roles = ("source-context", "literal-encoding", *("layer:"+n for n in c["layers"]))
    if c["profile"] in {_GUARDED_PROFILE, _DICTIONARY_PROFILE, _TRANSPORT_PROFILE}:
        roles += ("guard-anchor",)
    if c["profile"] in {_DICTIONARY_PROFILE, _TRANSPORT_PROFILE}:
        roles += ("dictionary-record", "dictionary-synset", "dictionary-route-column")
    for label, values in zip(c["field"]["labels"], c["joint_values"], strict=True):
        entries = fields[label]
        for row, count in entries:
            raw, chart = _context(c, row)
            members = (("context", row), ("literal-utf8", raw),
                       *((name, value) for name, value in zip(c["layers"], values, strict=True)))
            occurrences = []
            for role, identity in enumerate(members):
                lower = nodes.setdefault(identity, len(nodes)); key += 1
                occurrences.append(dict(key=key, lower=lower, exists=True, negative=False,
                                        magnitude=1, aspects=(role,)))
            relations.append(dict(grade=1, key=len(relations)+1, type=0, occurrences=tuple(occurrences)))
            records.append(dict(source_row=row, joint_values=values, count=count, raw_utf8_hex=raw,
                target=chart["target"], source_archive_digest=c["source_archive_digest"]))
    for observation in observations:
        members = tuple(("context", row) for row in dict.fromkeys(row for row, _ in observation["support"]))
        slots = (0,)*len(members)
        members += tuple((c["layers"][a], value) for (_, a), value in zip(observation["support"], observation["values"], strict=True))
        slots += tuple(a+2 for _, a in observation["support"])
        if "guard" in observation:
            from rexgraph.cpp_context_guards import guard_anchor_cells
            pins = guard_anchor_cells(c, observation["support"], observation["guard"])
            members += tuple(("source-lower", i) for i in pins)
            slots += (roles.index("guard-anchor"),)*len(pins)
        occurrences = []
        for role, identity in zip(slots, members, strict=True):
            lower = nodes.setdefault(identity, len(nodes)); key += 1
            occurrences.append(dict(key=key, lower=lower, exists=True, negative=False, magnitude=1, aspects=(role,)))
        relations.append(dict(grade=1, key=len(relations)+1, type=1, occurrences=tuple(occurrences)))
        records.append(dict(observation, kind="acquired-compatibility", source_archive_digest=c["source_archive_digest"]))
    for observation in dictionary_observations:
        for binding, record, synset, route in observation["branches"]:
            members = (("context", observation["source_row"]),
                *((name, observation["joint_values"][c["layers"].index(name)])
                    for name in dict.fromkeys(link[0] for link in c["dictionary_links"])),
                ("source-lower", record), ("source-lower", synset),
                *(("source-relation", row) for row in route))
            slots = (0, *(c["layers"].index(name)+2 for name in dict.fromkeys(link[0] for link in c["dictionary_links"])),
                roles.index("dictionary-record"), roles.index("dictionary-synset"),
                *((roles.index("dictionary-route-column"),)*len(route)))
            occurrences = []
            for role, identity in zip(slots, members, strict=True):
                lower = nodes.setdefault(identity, len(nodes)); key += 1
                occurrences.append(dict(key=key, lower=lower, exists=True, negative=False, magnitude=1, aspects=(role,)))
            relations.append(dict(grade=1, key=len(relations)+1, type=1, occurrences=tuple(occurrences)))
            records.append(dict(observation, kind="acquired-dictionary-branch", binding=binding,
                branch=(binding,record,synset,route), source_archive_digest=c["source_archive_digest"],
                interpretation="explicit-source-accession"))
    if not relations:
        return None, ()
    return _source(nodes, relations, roles, 941107, c["field"]["max_bytes"]).archive, tuple(records)


def validate_relational_sections_state(state):
    c = state.configuration
    guarded = state.adapter_version in {"2", "3", "4"}
    dictionary_bound = state.adapter_version in {"3", "4"}
    transported = state.adapter_version == "4"
    if (state.adapter != "cpp_relational_sections" or state.adapter_version not in {"1", "2", "3", "4"}
            or state.arithmetic != "rational" or not isinstance(state.source, CppSourceReference)
            or state.output_axes or c.get("profile") != (_TRANSPORT_PROFILE if transported else _DICTIONARY_PROFILE if dictionary_bound else _GUARDED_PROFILE if guarded else _PROFILE)
            or dict(c.get("execution_provider", {})) != _PROVIDER):
        raise ValueError("unsupported joint relational model")
    keys = {"profile", "execution_provider", "field", "layers", "joint_values", "section_limits",
            "sections_receipt", "source_archive_digest", "compatibility_types"}
    if guarded:
        keys |= {"guard_limits", "guard_receipt"}
        if len(c.get("guard_limits", ())) != 4 or any(type(i) is not int or i <= 0 for i in c["guard_limits"]):
            raise ValueError("four positive context guard admissions required")
    if dictionary_bound:
        keys.add("dictionary_links")
    if transported:
        keys.add("transport_search")
    if set(c) != keys:
        raise ValueError("closed joint model declaration required")
    names, values = c["layers"], c["joint_values"]
    if (not names or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not n or len(n.encode()) > 200 for n in names)
            or not values or len(set(values)) != len(values)
            or any(len(row) != len(names) or any(not isinstance(v, str) or not v for v in row) for row in values)):
        raise ValueError("joint annotations require distinct named layers and complete symbolic tuples")
    if tuple(c["field"]["labels"]) != tuple("joint:"+str(i) for i in range(len(values))):
        raise ValueError("joint tuples differ from their source field coordinates")
    if (len(c["section_limits"]) != 3 or any(type(i) is not int or i <= 0 for i in c["section_limits"])
            or c["source_archive_digest"] != state.source.native.archive_digest):
        raise ValueError("joint model bounds or source identity differ")
    payload_keys = {"fields", "batches", "acquisition_archive", "acquisition_accession", "compatibility_observations"}
    if dictionary_bound:
        payload_keys.add("dictionary_observations")
    if transported:
        payload_keys.add("transport_fit")
    if set(state.payload) != payload_keys:
        raise ValueError("joint model must retain its acquired correspondence columns")
    from rexgraph.cpp_relation_response import validate_relation_response_state
    validate_relation_response_state(SimpleNamespace(adapter="cpp_relation_response",adapter_version="2" if transported else "1",
        source=state.source,space=state.space,output_axes=(),configuration=c["field"],
        payload={"fields":state.payload["fields"],"batches":state.payload["batches"]},arithmetic="rational",step=state.step))
    if c["field"]["source_accession"] is None:
        raise ValueError("joint model needs complete literal and dictionary accessions")
    if dictionary_bound:
        from rexgraph.cpp_dictionary_branches import validate_dictionary_links
        validate_dictionary_links(c)
    if transported:
        from rexgraph.cpp_role_transports import validate_transport_search
        validate_transport_search(state)
    for row in c["field"]["training_rows"]:
        _context(c, row)
    if not isinstance(c["compatibility_types"], Mapping):
        raise ValueError("compatibility types must name their ordered layer signature")
    for name, axes in c["compatibility_types"].items():
        if not isinstance(name, str) or not name or not axes or any(type(a) is not int or not 0 <= a < len(names) for a in axes):
            raise ValueError("compatibility type outside declared layers")
    for observation in state.payload["compatibility_observations"]:
        if set(observation) != ({"type", "support", "values", "batch", "guard"} if guarded else
                                {"type", "support", "values", "batch"}) or observation["type"] not in c["compatibility_types"]:
            raise ValueError("closed acquired compatibility assertion required")
        support = observation["support"]
        if (not support or len(set(support)) != len(support)
                or tuple(a for _, a in support) != c["compatibility_types"][observation["type"]]
                or len(observation["values"]) != len(support) or observation["batch"] not in state.payload["batches"]):
            raise ValueError("acquired compatibility differs from its declared type or batch")
        for (row, a), value in zip(support, observation["values"], strict=True):
            if type(row) is not int or type(a) is not int or row not in c["field"]["training_rows"]:
                raise ValueError("acquired compatibility outside training relations")
            if not any(values[a] == value and row in dict(state.payload["fields"][label])
                       for label, values in zip(c["field"]["labels"], c["joint_values"], strict=True)):
                raise ValueError("compatibility value has no acquired annotation")
        if guarded:
            from rexgraph.cpp_context_guards import guard_anchor_cells
            guard_anchor_cells(c, support, observation["guard"])


def _prepare(state, source, *, with_guards=False, with_field=True):
    from rexgraph_cpp import NativeSource, sections_build_info, relation_build_info
    state.bind(source)
    if (dict(state.configuration["sections_receipt"]) != sections_build_info()
            or dict(state.configuration["field"]["engine_receipt"]) != relation_build_info()):
        raise ValueError("joint model requires its retained native implementation")
    identity = state._digest
    with _lock:
        if identity not in _prepared:
            validate_relational_sections_state(state)
            c = state.configuration
            for entries in state.payload["fields"].values():
                if any(count.bit_length() > c["field"]["limits"][3] for _, count in entries):
                    raise ValueError("acquired field count exceeds integer admission")
            for row in range(len(c["field"]["source_accession"]["relations"])):
                if c["field"]["source_accession"]["relations"][row]["kind"] != "observed-context":
                    continue
                raw, chart = _context(c, row)
                literal = NativeSource(chart["byte_source_archive"],max_bytes=c["field"]["max_bytes"])
                token = NativeSource(chart["token_source_archive"],max_bytes=c["field"]["max_bytes"])
                if tuple(literal.field(1001).values) != tuple(bytes.fromhex(raw)):
                    raise ValueError("literal accession differs from its retained byte field")
                token.chain_map(literal,chart["token_to_byte_family"])
            guard_owner = None
            if state.adapter_version in {"2", "3", "4"}:
                from rexgraph.cpp_context_guards import prepare_context_guards
                guard_owner = prepare_context_guards(state, source)
            if state.adapter_version in {"3", "4"}:
                from rexgraph.cpp_dictionary_branches import context_dictionary_branches, validate_dictionary_observations
                rows = tuple(row for entries in state.payload["fields"].values() for row, _ in entries)
                branches = context_dictionary_branches(c, rows, guard_owner)[0] if rows else {}
                validate_dictionary_observations(state, branches)
            if state.adapter_version == "4" and state.payload["transport_fit"] is not None:
                from rexgraph.cpp_role_transports import fit_role_transports
                if freeze_tree(fit_role_transports(state, source)) != state.payload["transport_fit"]:
                    raise ValueError("role transport fit differs from the acquired field replay")
            archive, accession = _acquisition(c, state.payload["fields"], state.payload["compatibility_observations"],
                state.payload.get("dictionary_observations", ()))
            if archive != state.payload["acquisition_archive"] or freeze_tree(accession) != state.payload["acquisition_accession"]:
                raise ValueError("acquired native correspondence differs from learned assertion fields")
            _prepared[identity] = None, guard_owner
            while len(_prepared) > 4:
                _prepared.popitem(last=False)
        if with_field and _prepared[identity][0] is None:
            _prepared[identity] = _prepare_field(_field_state(state), source), _prepared[identity][1]
        _prepared.move_to_end(identity)
        result = _prepared[identity] if with_field else (None, _prepared[identity][1])
        return result if with_guards else result[0]


def cpp_relational_sections_model(source, lower_axes, occurrence_axes, layers, joint_values,
                                  training_rows, *, source_accession, parameter=Fraction(1),
                                  limits=(4096,16000000,64000000,1 << 24),
                                  section_limits=(65536,1000000,64000000),
                                  max_bytes=256*1024*1024, compatibility_types=None, reference=None, guard_limits=None,
                                  dictionary_links=None, transport_candidates=None,
                                  transport_limits=(16,64000000)):
    """Declare whole annotation tuples on original observed context relations."""
    from rexgraph_cpp import sections_build_info
    values = tuple(tuple(row) for row in joint_values)
    candidates = None
    if transport_candidates is not None:
        from rexgraph.cpp_role_transports import normalize_role_transports
        if dictionary_links is None:
            raise ValueError("learned role sections require declared dictionary links")
        if len(transport_limits) != 2 or any(type(v) is not int or v <= 0 for v in transport_limits):
            raise ValueError("two positive role transport search limits required")
        candidates = []
        for table in transport_candidates:
            if len(candidates) >= transport_limits[0]:
                raise ValueError("role transport candidates exceed admission")
            candidates.append(normalize_role_transports(table))
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("nonempty role transport candidate family required")
    options = {"role_transports": candidates[0]} if candidates is not None else {}
    base = cpp_relation_response_model(source, lower_axes, occurrence_axes,
        tuple("joint:"+str(i) for i in range(len(values))), training_rows,
        source_accession=source_accession, parameter=parameter, limits=limits, max_bytes=max_bytes,
        reference=reference, **options)
    configuration = dict(profile=_PROFILE, execution_provider=_PROVIDER, field=base.configuration,
             layers=tuple(layers), joint_values=values, section_limits=tuple(section_limits),
             sections_receipt=sections_build_info(), source_archive_digest=base.source.native.archive_digest,
             compatibility_types=dict(compatibility_types or {}))
    if dictionary_links is not None and guard_limits is None:
        guard_limits = (4096,1000000,64000000,16384)
    if guard_limits is not None:
        from rexgraph_cpp import guards_build_info
        configuration.update(profile=_GUARDED_PROFILE, guard_limits=tuple(guard_limits), guard_receipt=guards_build_info())
    payload = dict(base.payload, acquisition_archive=None, acquisition_accession=(), compatibility_observations=())
    if dictionary_links is not None:
        configuration.update(profile=_DICTIONARY_PROFILE, dictionary_links=tuple(tuple(link) for link in dictionary_links))
        payload["dictionary_observations"] = ()
    if candidates is not None:
        configuration.update(profile=_TRANSPORT_PROFILE, transport_search=dict(
            profile="leave-context-out-role-transports-v1", candidates=candidates, limits=tuple(transport_limits)))
        payload["transport_fit"] = None
    state = ModelState("cpp_relational_sections", "4" if candidates is not None else "3" if dictionary_links is not None else "2" if guard_limits is not None else "1",
        base.source, base.space, (), configuration, payload, "rational")
    enable_relational_sections_provider()
    validate_relational_sections_state(state)
    return state


def joint_annotation_batch(state, annotations, *, groups=()):
    """Bind complete symbolic annotation tuples to admitted source rows."""
    c = state.configuration
    targets = np.full((len(state.space.keys), len(c["layers"])), -1, dtype=np.int64)
    observed = np.zeros(len(state.space.keys), dtype=bool)
    domains = tuple(tuple(dict.fromkeys(row[a] for row in c["joint_values"])) for a in range(len(c["layers"])))
    for row, values in annotations:
        values = tuple(values)
        if type(row) is not int or not 0 <= row < len(targets) or observed[row] or values not in c["joint_values"]:
            raise ValueError("annotation requires a unique original row and a declared complete tuple")
        targets[row] = tuple(domain.index(value) for domain, value in zip(domains, values, strict=True))
        observed[row] = True
    return ModelBatch(state.source, state.space, targets, observed,
        dict(profile="joint-relational-annotations-v1", partition="training", groups=tuple(groups)))


def _train(state, source, batch, *, steps, with_field=True):
    if not with_field and state.adapter_version == "4":
        raise ValueError("scalar transport fitting requires its declared field executor")
    _, guard_owner = _prepare(state, source, with_guards=True, with_field=with_field)
    c = state.configuration
    if (steps != 1 or not isinstance(batch.inputs, Mapping) or set(batch.inputs) != {"profile", "partition", "groups"}
            or batch.inputs["profile"] != "joint-relational-annotations-v1" or batch.inputs["partition"] != "training"):
        raise ValueError("one declared joint training batch required")
    if batch.targets.shape != (len(state.space.keys), len(c["layers"])) or batch.targets.dtype.kind not in "iu":
        raise ValueError("joint targets require symbolic integer coordinates for every layer")
    domains = tuple(tuple(dict.fromkeys(row[a] for row in c["joint_values"])) for a in range(len(c["layers"])))
    fields = {name:dict(entries) for name, entries in state.payload["fields"].items()}
    annotated = {}
    for row in np.flatnonzero(batch.observed):
        row = int(row)
        if row not in c["field"]["training_rows"]:
            raise ValueError("annotation outside declared training relations")
        indices = tuple(int(i) for i in batch.targets[row])
        if any(not 0 <= i < len(domain) for i, domain in zip(indices, domains, strict=True)):
            raise ValueError("annotation outside declared layer domains")
        values = tuple(domain[i] for domain, i in zip(domains, indices, strict=True))
        if values not in c["joint_values"]:
            raise ValueError("annotation tuple was not declared jointly")
        name = c["field"]["labels"][c["joint_values"].index(values)]
        fields[name][row] = fields[name].get(row, 0)+1
        annotated[row] = values
    observations = list(state.payload["compatibility_observations"])
    dictionary_observations = list(state.payload.get("dictionary_observations", ()))
    if state.adapter_version in {"3", "4"} and annotated:
        from rexgraph.cpp_dictionary_branches import context_dictionary_branches, branch_accepts
        branches, _ = context_dictionary_branches(c, tuple(annotated), guard_owner)
        for row, values in annotated.items():
            retained = tuple(b["identity"] for b in branches[row] if branch_accepts(c, values, b["identity"]))
            if not retained:
                raise ValueError("training annotation has no compatible target dictionary branch")
            dictionary_observations.append(dict(source_row=row, joint_values=values, branches=retained, batch=batch.coefficient_digest))
    for group in batch.inputs["groups"]:
        guarded = state.adapter_version in {"2", "3", "4"}
        if set(group) != ({"type", "support", "guard"} if guarded else {"type", "support"}) or group["type"] not in c["compatibility_types"]:
            raise ValueError("training group requires a declared compatibility type")
        support = tuple(tuple(pair) for pair in group["support"])
        if (not support or len(set(support)) != len(support)
                or tuple(a for _, a in support) != c["compatibility_types"][group["type"]]
                or any(type(row) is not int or type(a) is not int or row not in annotated for row, a in support)):
            raise ValueError("compatibility group must use complete annotations in this batch")
        observation = dict(type=group["type"], support=support,
            values=tuple(annotated[row][a] for row, a in support), batch=batch.coefficient_digest)
        if guarded:
            from rexgraph.cpp_context_guards import guard_anchor_cells
            guard_anchor_cells(c, support, group["guard"])
            observation["guard"] = group["guard"]
        observations.append(observation)
    fields = {name:tuple(sorted(entries.items())) for name, entries in fields.items()}
    archive, accession = _acquisition(c, fields, observations, dictionary_observations)
    payload = dict(fields=fields, batches=(*state.payload["batches"], batch.coefficient_digest),
        acquisition_archive=archive, acquisition_accession=accession,
        compatibility_observations=tuple(observations))
    if state.adapter_version in {"3", "4"}:
        payload["dictionary_observations"] = tuple(dictionary_observations)
    if state.adapter_version == "4":
        from rexgraph.cpp_role_transports import fit_role_transports
        fit = fit_role_transports(state, source, field_entries=fields)
        configuration = dict(c, field=dict(c["field"], role_transports=c["transport_search"]["candidates"][fit["selected_index"]]))
        return state.child(configuration=configuration, payload=dict(payload, transport_fit=fit), step=state.step+1)
    return state.child(payload=payload, step=state.step+1)


def joint_query(state, contexts, *, domains=(), charts=(), discover_guarded=None):
    """Declare original context rows and optional exact compatibility charts."""
    query = dict(profile="joint-relational-query-v1", contexts=tuple(contexts), domains=tuple(domains), charts=tuple(charts))
    if state.adapter_version in {"2", "3", "4"}:
        discover_guarded = True if discover_guarded is None else discover_guarded
        if type(discover_guarded) is not bool:
            raise ValueError("guard discovery must be explicitly boolean")
        query.update(profile="joint-relational-query-v2", discover_guarded=discover_guarded)
    elif discover_guarded is not None:
        raise ValueError("guard discovery requires joint model version 2")
    return query


def _query_carrier(state, inputs, *, guard_owner=None):
    from rexgraph_cpp import PreparedRelationalSections
    c = state.configuration
    guarded = state.adapter_version in {"2", "3", "4"}
    keys = {"profile", "contexts", "domains", "charts"} | ({"discover_guarded"} if guarded else set())
    if (not isinstance(inputs, Mapping) or set(inputs) != keys or
            inputs["profile"] != ("joint-relational-query-v2" if guarded else "joint-relational-query-v1")
            or guarded and type(inputs["discover_guarded"]) is not bool):
        raise ValueError("closed joint relational query required")
    contexts = tuple(inputs["contexts"])
    if not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("distinct observed context rows required")
    learned = tuple(i for i, label in enumerate(c["field"]["labels"]) if state.payload["fields"][label])
    if not learned:
        raise ValueError("joint interpretation needs acquired annotation tuples")
    names = c["layers"]
    axes = tuple(tuple(dict.fromkeys(row[a] for row in c["joint_values"])) for a in range(len(names)))
    restrictions = tuple(inputs["domains"])
    if restrictions and len(restrictions) != len(contexts):
        raise ValueError("candidate domains must cover every queried context")
    nodes = {}; domains = []; symbols = {}; relations = []; charts = []; accession = []; variables = []; selectors = []; key = 0
    roles = ("source-context", "literal-encoding", *("layer:"+n for n in names))
    dictionary_bound = state.adapter_version in {"3", "4"}
    branches = {}; branch_variables = []; branch_report = None
    if dictionary_bound:
        from rexgraph.cpp_dictionary_branches import context_dictionary_branches, branch_accepts
        if guard_owner is None:
            raise ValueError("dictionary-bound query requires its checked native source owner")
        branches, branch_report = context_dictionary_branches(c, contexts, guard_owner)
        roles += ("dictionary-branch",)
    def symbol(value):
        return symbols.setdefault(value, len(symbols))
    def node(identity, values):
        if identity not in nodes:
            nodes[identity] = len(nodes)
            domains.append(tuple(symbol(v) for v in values))
        return nodes[identity]
    def relation(kind, support, role_slots, name, allowed, provenance):
        nonlocal key
        occurrences = []
        for lower, role in zip(support, role_slots, strict=True):
            key += 1
            occurrences.append(dict(key=key, lower=lower, exists=True, negative=False, magnitude=1, aspects=(role,)))
        index = len(relations)
        relations.append(dict(grade=1, key=index+1, type=kind, occurrences=tuple(occurrences)))
        charts.append(dict(name=name, relation=index, support=tuple(support), allowed=tuple(allowed)))
        accession.append(provenance)
    for q, row in enumerate(contexts):
        raw, literal = _context(c, row)
        fixed = (node(("context", row), (("context", row),)),
                 node(("literal-utf8", raw), (("literal-utf8", raw),)))
        cells = tuple(node(("use", row, name), tuple((name, v) for v in values)) for name, values in zip(names, axes, strict=True))
        variables.append(cells)
        allowed = tuple((domains[fixed[0]][0], domains[fixed[1]][0],
                         *(symbol((name, v)) for name, v in zip(names, c["joint_values"][i], strict=True))) for i in learned)
        support = (*fixed, *cells)
        provenance = dict(kind="learned-joint-correspondence", source_row=row, raw_utf8_hex=raw,
            target=literal["target"], source_archive_digest=c["source_archive_digest"])
        if dictionary_bound:
            choices = tuple(b["identity"] for b in branches[row])
            branch_cell = node(("dictionary-use", row), tuple(("dictionary-branch", b) for b in choices))
            branch_variables.append(branch_cell)
            allowed = tuple((*values, symbol(("dictionary-branch", b)))
                for i, values in zip(learned, allowed, strict=True) for b in choices
                if branch_accepts(c, c["joint_values"][i], b))
            if not allowed:
                raise ValueError("target dictionary routes admit no acquired annotation tuple")
            support += (branch_cell,)
            provenance.update(dictionary_branches=branches[row], dictionary_links=c["dictionary_links"])
        selectors.append(len(charts))
        relation(0, support, range(len(roles)), "context:"+str(q), allowed, provenance)
        if restrictions:
            if len(restrictions[q]) != len(names):
                raise ValueError("candidate domains must cover every layer")
            for a, values in enumerate(restrictions[q]):
                values = tuple(values)
                if not values or len(set(values)) != len(values) or any(v not in axes[a] for v in values):
                    raise ValueError("candidate domain outside declared symbolic layer")
                relation(1, (cells[a],), (a+2,), "domain:"+str(q)+":"+str(a),
                    tuple((symbol((names[a], v)),) for v in values), dict(kind="declared-candidate-domain", query=q, layer=names[a]))
    guard_matches = {}; guard_report = None
    declarations = list(inputs["charts"])
    if guarded:
        from rexgraph.cpp_context_guards import discover_context_charts
        if guard_owner is None:
            raise ValueError("guarded query requires its checked native context owner")
        if inputs["discover_guarded"] or any("type" in chart for chart in declarations):
            guard_matches, guard_report = discover_context_charts(state, contexts, guard_owner,
                reserved_work=branch_report["native_work"]-guard_owner.receipt["preparation_steps"] if branch_report else 0,
                reserved_candidates=branch_report["candidate_count"] if branch_report else 0)
            guard_report["discovery_applied"] = True
        else:
            guard_report = dict(profile="ordered-context-boundary-v1", candidate_count=0, native_calls=0,
                native_work=0, matched_groups=0, discovery_applied=False, limits=c["guard_limits"], receipt=guard_owner.receipt)
        if dictionary_bound:
            from rexgraph.cpp_dictionary_branches import guarded_branch_tuples
            spent = (guard_report["candidate_count"]+guard_report.get("dictionary_union_candidates",0)
                if guard_report["discovery_applied"] else branch_report["candidate_count"])
            retained = {}; branch_candidates = 0
            for group, witnesses in guard_matches.items():
                records = []
                for witness in witnesses:
                    uses, choices, count, certificates = guarded_branch_tuples(state, witness, group[1], contexts, branches,
                        candidate_limit=c["guard_limits"][0]-spent-branch_candidates)
                    branch_candidates += count
                    if choices:
                        records.append(dict(witness, branch_uses=uses, branch_tuples=choices, branch_certificates=certificates))
                if records:
                    retained[group] = records
            guard_matches = retained
            guard_report.update(branch_matched_groups=len(retained), branch_rule_candidates=branch_candidates)
        if inputs["discover_guarded"]:
            declarations += [dict(name="guarded:"+name+":"+str(i),type=name,support=support)
                for i, (name, support) in enumerate(guard_matches)]
    for ordinal, chart in enumerate(declarations):
        if set(chart) not in ({"name", "support", "allowed"}, {"name", "support", "type"}):
            raise ValueError("closed compatibility chart required")
        support = tuple(chart["support"])
        if not support or len(set(support)) != len(support):
            raise ValueError("compatibility chart requires distinct query-layer cells")
        for q, a in support:
            if type(q) is not int or type(a) is not int or not 0 <= q < len(variables) or not 0 <= a < len(names):
                raise ValueError("compatibility support outside query-layer cells")
        allowed = []
        if "type" in chart:
            if (chart["type"] not in c["compatibility_types"]
                    or tuple(a for _, a in support) != c["compatibility_types"][chart["type"]]):
                raise ValueError("query compatibility differs from its acquired type")
            if guarded:
                witnesses = guard_matches.get((chart["type"], support), ())
                indices = tuple(w["observation_index"] for w in witnesses)
                tuples = tuple(dict.fromkeys(state.payload["compatibility_observations"][i]["values"] for i in indices))
            else:
                indices = tuple(i for i, o in enumerate(state.payload["compatibility_observations"]) if o["type"] == chart["type"])
                tuples = tuple(dict.fromkeys(state.payload["compatibility_observations"][i]["values"] for i in indices))
            if not tuples:
                raise ValueError("query compatibility has no acquired tuples")
        else:
            tuples = chart["allowed"]
        for values in tuples:
            if len(values) != len(support) or any(v not in axes[a] for (_, a), v in zip(support, values, strict=True)):
                raise ValueError("compatibility tuple outside declared domains")
            allowed.append(tuple(symbol((names[a], v)) for (_, a), v in zip(support, values, strict=True)))
        provenance = dict(kind="declared-compatibility", ordinal=ordinal, name=chart["name"])
        if "type" in chart:
            provenance.update(kind="acquired-compatibility",type=chart["type"],observation_indices=indices)
            if guarded:
                provenance.update(kind="guarded-acquired-compatibility",guard_witnesses=tuple(witnesses))
        cells = tuple(variables[q][a] for q, a in support)
        slots = tuple(a+2 for _, a in support)
        if dictionary_bound and "type" in chart:
            uses = tuple(dict.fromkeys(q for q, _ in support))
            allowed = tuple(dict.fromkeys((*tuple(symbol((names[a], v)) for (_, a), v in
                zip(support, state.payload["compatibility_observations"][w["observation_index"]]["values"], strict=True)),
                *(symbol(("dictionary-branch", b)) for b in choice)) for w in witnesses for choice in w["branch_tuples"]))
            cells += tuple(branch_variables[q] for q in uses)
            slots += (roles.index("dictionary-branch"),)*len(uses)
            provenance.update(branch_uses=uses, interpretation="annotation-and-retained-route-compatibility")
        relation(1, cells, slots,
            chart["name"], allowed, provenance)
    source = _source(nodes, relations, roles, 941108, c["field"]["max_bytes"])
    plan = PreparedRelationalSections(source.archive, domains, charts, limits=c["section_limits"],
        max_bytes=c["field"]["max_bytes"], enumeration_charts=tuple(selectors))
    return (source, plan, tuple(variables), tuple(symbols), tuple(nodes), tuple(accession), tuple(charts), tuple(domains),
        guard_report, tuple(branch_variables), branch_report)


def _infer(state, source, inputs):
    inputs = freeze_tree(inputs)
    plan, guard_owner = _prepare(state, source, with_guards=True)
    carrier, sections_plan, variables, symbols, nodes, accession, charts, domains, guard_report, branch_variables, branch_report = (
        _query_carrier(state, inputs, guard_owner=guard_owner) if guard_owner is not None else _query_carrier(state, inputs))
    c = state.configuration
    sections = sections_plan.sections()
    assignments = tuple(tuple(tuple(symbols[section[cell]][1] for cell in cells) for cells in variables) for section in sections["sections"])
    fields = []
    for label in c["field"]["labels"]:
        values = [0]*len(state.space.keys)
        for row, count in state.payload["fields"][label]:
            values[row] = count
        fields.append(tuple(values))
    responses = []; couplings = []
    for row in inputs["contexts"]:
        query = [0]*len(state.space.keys); query[row] = 1
        result = plan.moments((tuple(query), *fields)); gram = result["gram"]
        scores = tuple(Fraction(0) if not gram[0][0] or not gram[i][i] else
            gram[0][i]**2/(gram[0][0]*gram[i][i]) for i in range(1, len(gram)))
        if any(not 0 <= s <= 1 for s in scores):
            raise RuntimeError("joint Green coupling exceeds its exact bound")
        couplings.append(scores); responses.append(result)
    joint_index = {values:i for i,values in enumerate(c["joint_values"])}
    scores = tuple(sum((couplings[q][joint_index[values]] for q, values in enumerate(assignment)), Fraction(0)) for assignment in assignments)
    maximum = max(scores) if scores else None
    best = tuple(i for i, score in enumerate(scores) if score == maximum)
    receipt = dict(profile=c["profile"], layers=c["layers"], joint_values=c["joint_values"],
        assignments=assignments, best_indices=best, context_rows=tuple(inputs["contexts"]),
        couplings=tuple(couplings), field_responses=tuple(responses), sections=sections,
        section_source_archive=carrier.archive, section_nodes=nodes, section_accession=accession,
        section_charts=charts, section_domains=domains, section_symbols=symbols,
        query_declaration=inputs, truth_status="prospective", learned_transports=state.adapter_version == "4",
        readout="sum-of-exact-normalized-squared-Green-couplings-v1",
        accession_policy="explicit-reference-no-semantic-scale-map", source_reference=state.source.as_record())
    if guard_report is not None:
        receipt["context_guards"] = guard_report
    if branch_report is not None:
        receipt.update(dictionary_branches=branch_report, dictionary_links=c["dictionary_links"],
            branch_assignments=tuple(tuple(symbols[section[cell]][1] for cell in branch_variables) for section in sections["sections"]),
            semantic_maxima=tuple(dict.fromkeys(assignments[i] for i in best)))
    if state.adapter_version == "4":
        receipt.update(role_transports=c["field"]["role_transports"], transport_fit=state.payload["transport_fit"],
            transport_selection="declared-finite-family-with-training-context-exclusion")
    return ModelOutput(scores, CoordinateSpace("compatible-joint-sections", tuple("section:"+str(i) for i in range(len(scores)))), (),
        state.source, state.coefficient_digest, "rational", c["profile"], execution_receipt=receipt)


def _create(source, *, adapter="cpp_relational_sections", reference=None, **configuration):
    if adapter != "cpp_relational_sections":
        raise ValueError("unsupported joint relational adapter")
    return cpp_relational_sections_model(source, reference=reference, **configuration)


def enable_relational_sections_provider():
    """Register joint field acquisition and native compatible section inference."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], _PROVIDER["version"], (("cpp_relational_sections", "1"),("cpp_relational_sections", "2"),("cpp_relational_sections", "3"),("cpp_relational_sections", "4")),
                _infer, create=_create, prepare=_prepare, train=_train)
        register_model_provider(_provider)
        return _provider


def clear_relational_sections_cache():
    """Release joint and shared field preparations while preserving acquired data."""
    from rexgraph.cpp_relation_response import clear_relation_response_cache
    from rexgraph.cpp_context_guards import clear_context_guard_cache
    from rexgraph.cpp_dictionary_branches import clear_dictionary_branch_cache
    with _lock:
        _prepared.clear()
        clear_relation_response_cache()
        clear_dictionary_branch_cache()
        clear_context_guard_cache()
