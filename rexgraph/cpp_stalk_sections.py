"""Complete native interpretation sections and their declared stalk fields."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.cpp_relational_sections import (
    _prepare as _prepare_sections, _query_carrier, _train as _train_sections,
    joint_annotation_batch, joint_query,
)
from rexgraph.cpp_stalk_fields import (
    _prepare as _prepare_fields, _exact, _field, infer_stalk_fields_many,
    stalk_annotation_batch, stalk_output_field, stalk_query,
)
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_runtime import train_model
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "compatible-sections-whole-stalk-fields-v1"
_PROVIDER = {"name": "rexgraph.cpp.stalk_sections", "version": "1"}
_lock = RLock()
_prepared = OrderedDict()
_provider = None


def _states(state):
    return unpack_model(state.payload["sections"]), unpack_model(state.payload["fields"])


def _contexts(sections):
    return tuple(i for i, row in enumerate(sections.configuration["field"]["source_accession"]["relations"])
                 if row["kind"] == "observed-context")


def _ports(sections, fields, declarations, routes=False):
    result = []; seen = set(); contexts = _contexts(sections)
    for declaration in declarations:
        if len(declaration) != (4 if routes else 3):
            raise ValueError("port requires an original context, complete annotation and whole field")
        row, annotation = declaration[:2]; annotation = tuple(annotation)
        if type(row) is not int or row not in contexts or annotation not in sections.configuration["joint_values"]:
            raise ValueError("port lies outside original context and annotation coordinates")
        vector = tuple(_exact(v, fields.configuration["limits"][3]) for v in declaration[-1])
        if len(vector) != len(fields.configuration["upper_axes"][row]):
            raise ValueError("port must cover the complete named local stalk")
        key = (row, annotation, tuple(declaration[2])) if routes else (row, annotation)
        if key in seen:
            raise ValueError("port declaration repeats an original interpretation")
        seen.add(key); result.append((*key, vector))
    if not routes and seen != {(row, annotation) for row in contexts for annotation in sections.configuration["joint_values"]}:
        raise ValueError("ports must cover every original context and complete annotation")
    return tuple(result)


def _interfaces(fields, declarations):
    result = []; names = set()
    for name, axes, restrictions in declarations:
        axes = tuple(axes); blocks = []; rows = set()
        if (not isinstance(name, str) or not name or len(name.encode()) > 256 or name in names or not axes or len(set(axes)) != len(axes) or
                any(not isinstance(a, str) or not a or len(a.encode()) > 256 for a in axes)):
            raise ValueError("interfaces require distinct names and complete named axes")
        names.add(name)
        for row, block in restrictions:
            if type(row) is not int or not 0 <= row < len(fields.space.keys) or row in rows:
                raise ValueError("restriction must name a distinct original relation")
            rows.add(row); seen = set(); entries = []
            for i,j,value in block:
                if (type(i) is not int or type(j) is not int or not 0 <= i < len(axes) or
                        not 0 <= j < len(fields.configuration["upper_axes"][row]) or (i,j) in seen):
                    raise ValueError("restriction differs from its declared interface and local stalk")
                seen.add((i,j)); entries.append((i,j,_exact(value,fields.configuration["limits"][3])))
            blocks.append((row,tuple(entries)))
        if len(blocks) < 2:raise ValueError("an interface requires at least two original local restrictions")
        result.append((name,axes,tuple(blocks)))
    return tuple(result)


def _interface_readings(fields, declarations, whole_field):
    result = []
    for name, axes, restrictions in declarations:
        readings = []
        for row, block in restrictions:
            values = [Fraction(0)]*len(axes)
            for i,j,value in block:values[i] = _exact(values[i]+value*whole_field[row][j],fields.configuration["limits"][3])
            readings.append(tuple(values))
        residuals = tuple(tuple(_exact(a-b,fields.configuration["limits"][3]) for a,b in zip(v,readings[0],strict=True)) for v in readings[1:])
        result.append(dict(name=name,axes=axes,rows=tuple(r for r,_ in restrictions),readings=tuple(readings),
                           residuals=residuals,compatible=not any(v for row in residuals for v in row)))
    return tuple(result)


def validate_stalk_sections_state(state):
    c = state.configuration
    if (state.adapter != "cpp_stalk_sections" or state.adapter_version != "1" or state.arithmetic != "rational" or
            not isinstance(state.source,CppSourceReference) or
            set(c) != {"profile", "execution_provider", "ports", "route_ports", "interfaces", "initial_steps", "max_case_entries"} or
            c["profile"] != _PROFILE or dict(c["execution_provider"]) != _PROVIDER or
            set(state.payload) != {"sections", "fields", "batches"} or
            len(state.payload["batches"]) != state.step):
        raise ValueError("closed whole-stalk section model required")
    sections, fields = _states(state)
    if (sections.adapter != "cpp_relational_sections" or sections.adapter_version not in {"1", "2", "3"} or
            fields.adapter != "cpp_stalk_fields" or fields.adapter_version != "1" or
            sections.source.native != fields.source.native or sections.source.native != state.source.native or
            sections.configuration["field"]["source_accession"] != fields.configuration["source_accession"] or
            sections.configuration["field"]["training_rows"] != fields.configuration["training_rows"] or
            state.space != sections.space or state.output_axes != fields.output_axes):
        raise ValueError("section and stalk checkpoints require their actual common source and accessions")
    if (len(c["initial_steps"]) != 2 or any(type(i) is not int or i < 0 for i in c["initial_steps"]) or
            (sections.step, fields.step) != tuple(i+state.step for i in c["initial_steps"]) or
            type(c["max_case_entries"]) is not int or c["max_case_entries"] <= 0):
        raise ValueError("coupled acquisition history or case admission differs")
    if _ports(sections, fields, c["ports"]) != c["ports"] or _ports(sections, fields, c["route_ports"], True) != c["route_ports"]:
        raise ValueError("whole-stalk ports differ from their retained declarations")
    if c["route_ports"] and sections.adapter_version != "3":
        raise ValueError("route ports require native dictionary interpretations")
    if _interfaces(fields,c["interfaces"]) != c["interfaces"]:
        raise ValueError("tensor restrictions differ from their declared local stalks")
    declared = sum(len(p[-1])+len(p)-1 for p in (*c["ports"], *c["route_ports"]))
    declared += sum(len(axes)+sum(1+len(block) for _,block in restrictions) for _,axes,restrictions in c["interfaces"])
    if declared > c["max_case_entries"]:
        raise ValueError("port and restriction declarations exceed case entry admission")


def _prepare(state, source):
    state.bind(source)
    key = state.coefficient_digest
    with _lock:
        if key not in _prepared:
            validate_stalk_sections_state(state)
            sections, fields = _states(state)
            _, guards = _prepare_sections(sections, source, with_guards=True, with_field=False)
            plan = _prepare_fields(fields, source)
            if state.configuration["route_ports"]:
                from rexgraph.cpp_dictionary_branches import context_dictionary_branches, branch_accepts
                rows = tuple(dict.fromkeys(p[0] for p in state.configuration["route_ports"]))
                branches, _ = context_dictionary_branches(sections.configuration, rows, guards)
                for row, annotation, branch, _ in state.configuration["route_ports"]:
                    if branch not in tuple(b["identity"] for b in branches[row]) or not branch_accepts(sections.configuration, annotation, branch):
                        raise ValueError("route port requires its actual compatible native dictionary branch")
            _prepared[key] = sections, fields, guards, plan
            while len(_prepared) > 4:
                _prepared.popitem(last=False)
        _prepared.move_to_end(key)
        return _prepared[key]


def cpp_stalk_sections_model(source, sections, fields, ports, *, route_ports=(), interfaces=(), max_case_entries=1000000, reference=None):
    """Bind declared interpretation ports to retained annotation and field checkpoints."""
    c = dict(profile=_PROFILE, execution_provider=_PROVIDER, ports=_ports(sections, fields, ports),
             route_ports=_ports(sections, fields, route_ports, True), interfaces=_interfaces(fields,interfaces), initial_steps=(sections.step, fields.step),
             max_case_entries=max_case_entries)
    state = ModelState("cpp_stalk_sections", "1", sections.source if reference is None else reference.bind(source),
        sections.space, fields.output_axes, c,
        dict(sections=pack_model(sections), fields=pack_model(fields), batches=()), "rational")
    enable_stalk_sections_provider(); _prepare(state, source)
    return state


def stalk_section_annotation_batch(state, annotations, *, groups=()):
    """Declare one annotation batch for both finite restrictions and whole stalk acquisition."""
    sections, _ = _states(state)
    batch = joint_annotation_batch(sections, annotations, groups=groups)
    return ModelBatch(state.source, state.space, batch.targets, batch.observed,
                      dict(profile="whole-stalk-section-annotations-v1", annotation=pack_model(batch)))


def _train(state, source, batch, *, steps):
    sections, fields, _, _ = _prepare(state, source)
    if steps != 1 or not isinstance(batch.inputs, Mapping) or set(batch.inputs) != {"profile", "annotation"} or batch.inputs["profile"] != "whole-stalk-section-annotations-v1":
        raise ValueError("one coupled whole-stalk annotation batch required")
    annotation = unpack_model(batch.inputs["annotation"])
    if (not isinstance(annotation, ModelBatch) or not annotation.source.matches(sections.source) or annotation.space != sections.space or
            not np.array_equal(annotation.targets, batch.targets) or not np.array_equal(annotation.observed, batch.observed)):
        raise ValueError("coupled annotations differ from their retained original batch")
    domains = tuple(tuple(dict.fromkeys(v[i] for v in sections.configuration["joint_values"])) for i in range(len(sections.configuration["layers"])))
    ports = {(p[0], p[1]): p[2] for p in state.configuration["ports"]}
    acquired = []
    for row in np.flatnonzero(annotation.observed):
        row = int(row); indices = tuple(int(i) for i in annotation.targets[row])
        if any(not 0 <= index < len(domain) for index, domain in zip(indices, domains, strict=True)):
            raise ValueError("annotation lies outside its declared symbolic layer")
        values = tuple(domain[index] for index, domain in zip(indices, domains, strict=True))
        if (row, values) not in ports:
            raise ValueError("annotation has no declared complete stalk port")
        acquired.append((row, ports[row, values]))
    child_sections = _train_sections(sections, source, annotation, steps=1, with_field=False)
    child_fields = train_model(fields, source, stalk_annotation_batch(fields, acquired))
    child = state.child(payload=dict(sections=pack_model(child_sections), fields=pack_model(child_fields),
        batches=(*state.payload["batches"], batch.coefficient_digest)), step=state.step+1)
    _prepare(child, source)
    return child


def stalk_section_query(state, contexts, *, domains=(), charts=(), discover_guarded=None, fields=None, include_acquisition=True):
    """Declare complete interpretation restrictions and a whole source field."""
    sections, field_state = _states(state)
    return dict(profile="whole-stalk-section-query-v1", source_reference=state.source.as_record(),
        interpretations=joint_query(sections, contexts, domains=domains, charts=charts, discover_guarded=discover_guarded),
        field=stalk_query(field_state, fields, include_acquisition=include_acquisition))


def _case_queries(fields, query, assignments, branches, ports, route_ports):
    result = []
    for case, choices in enumerate(assignments):
        values = [list(v) for v in query["field"]["fields"]]
        for q, (row, annotation) in enumerate(zip(query["interpretations"]["contexts"], choices, strict=True)):
            port = route_ports.get((row, annotation, branches[case][q]), ports[row, annotation]) if branches else ports[row, annotation]
            values[row] = [_exact(a+b, fields.configuration["limits"][3]) for a,b in zip(values[row],port,strict=True)]
        result.append(stalk_query(fields, values, include_acquisition=query["field"]["include_acquisition"]))
    return tuple(result)


def _infer(state, source, inputs):
    sections, fields, guards, plan = _prepare(state, source); inputs = freeze_tree(inputs)
    if (not isinstance(inputs, Mapping) or set(inputs) != {"profile", "source_reference", "interpretations", "field"} or
            inputs["profile"] != "whole-stalk-section-query-v1" or dict(inputs["source_reference"]) != state.source.as_record()):
        raise ValueError("closed query bound to the actual coupled source required")
    from rexgraph.cpp_stalk_fields import _check_query
    _check_query(fields, inputs["field"]); _field(fields.configuration, inputs["field"]["fields"])
    carrier, native, variables, symbols, nodes, accession, charts, domains, guard_report, branch_variables, branch_report = (
        _query_carrier(sections, inputs["interpretations"], guard_owner=guards))
    solved = native.sections()
    assignments = tuple(tuple(tuple(symbols[s[cell]][1] for cell in cells) for cells in variables) for s in solved["sections"])
    branches = tuple(tuple(symbols[s[cell]][1] for cell in branch_variables) for s in solved["sections"]) if branch_variables else ()
    dimension = sum(map(len, fields.configuration["upper_axes"]))
    width = len(fields.output_axes[0].keys); rows = len(fields.space.keys)
    per_case = 3*dimension+rows*width+plan.receipt["lower_dimension"]+sum(len(fields.configuration["lower_axes"][lo]) for _,lo in plan.receipt["occurrence_cells"])
    per_case += sum(2*len(axes)*(2*len(restrictions)-1) for _,axes,restrictions in state.configuration["interfaces"])
    if len(assignments)*per_case > state.configuration["max_case_entries"]:
        raise ValueError("complete interpretation fields exceed case entry admission")
    ports = {(p[0],p[1]):p[2] for p in state.configuration["ports"]}
    route_ports = {(p[0],p[1],p[2]):p[3] for p in state.configuration["route_ports"]}
    queries = _case_queries(fields, inputs, assignments, branches, ports, route_ports)
    outputs = infer_stalk_fields_many(fields, source, queries)
    unique = {}; indices = []
    for output in outputs:
        digest = output.coefficient_digest
        if digest not in unique:unique[digest] = (len(unique), pack_model(output))
        indices.append(unique[digest][0])
    values = np.empty((len(outputs), rows, width), dtype=object)
    for i, output in enumerate(outputs):values[i] = output.values
    interface_readings = tuple(dict(input=_interface_readings(fields,state.configuration["interfaces"],o.execution_receipt["input_fields"]),
                                   response=_interface_readings(fields,state.configuration["interfaces"],o.execution_receipt["response_fields"])) for o in outputs)
    receipt = dict(profile=_PROFILE, source_reference=state.source.as_record(), query=inputs, model_checkpoint=pack_model(state),
        ports=state.configuration["ports"], route_ports=state.configuration["route_ports"],
        section_source_archive=carrier.archive, sections=solved, section_variables=variables,
        section_symbols=symbols, section_nodes=nodes, section_accession=accession, section_charts=charts, section_domains=domains,
        branch_variables=branch_variables, assignments=assignments, branch_assignments=branches,
        context_guards=guard_report, dictionary_branches=branch_report,
        field_outputs=tuple(v[1] for v in unique.values()), case_output_indices=tuple(indices),
        case_queries=queries, interface_readings=interface_readings, max_case_entries=state.configuration["max_case_entries"],
        readout="all-compatible-sections-with-whole-stalk-fields-v1", truth_status="prospective")
    return ModelOutput(values, CoordinateSpace("compatible-stalk-sections", tuple("section:"+str(i) for i in range(len(outputs)))),
        (fields.space, fields.output_axes[0]), state.source, state.coefficient_digest, "rational", _PROFILE,
        execution_receipt=receipt)


def validate_stalk_sections_output(output):
    """Check every finite interpretation and its complete retained field output."""
    r = output.execution_receipt
    if not isinstance(r, Mapping) or r.get("profile") != _PROFILE or r.get("readout") != "all-compatible-sections-with-whole-stalk-fields-v1":
        raise ValueError("complete interpretation field receipt required")
    state = unpack_model(r["model_checkpoint"]); sections,fields = _states(state)
    if (state.coefficient_digest != output.model_digest or r["ports"] != state.configuration["ports"] or
            r["route_ports"] != state.configuration["route_ports"] or r["max_case_entries"] != state.configuration["max_case_entries"]):
        raise ValueError("complete interpretation fields require their actual producing checkpoint")
    if (not isinstance(output.source,CppSourceReference) or output.source.native != fields.source.native or sections.source.native != fields.source.native or
            dict(r["source_reference"]) != output.source.as_record() or output.arithmetic != "rational" or
            output.grade is not None or output.variance != "coordinate" or output.axes != (fields.space, fields.output_axes[0])):
        raise ValueError("interpretation fields differ from their actual source and carriers")
    solved = r["sections"]; count = len(solved["sections"])
    if (output.space != CoordinateSpace("compatible-stalk-sections", tuple("section:"+str(i) for i in range(count))) or
            len(r["case_output_indices"]) != count or len(r["case_queries"]) != count or len(r["interface_readings"]) != count or
            solved["satisfiable"] != bool(count) or not solved["constraint_complete"]):
        raise ValueError("whole fields must retain every native compatible section")
    from hashlib import sha256
    if solved["receipt"]["archive_digest"] != sha256(r["section_source_archive"]).hexdigest():
        raise ValueError("interpretation sections require their actual retained native carrier")
    for name,value in sections.configuration["sections_receipt"].items():
        if solved["receipt"][name] != value:
            raise ValueError("interpretation sections require their retained native executor")
    native_sections = solved["sections"]
    if native_sections != tuple(sorted(set(native_sections))):
        raise ValueError("original native sections must retain their distinct declared order")
    for case in native_sections:
        if len(case) != len(r["section_domains"]) or any(v not in domain for v,domain in zip(case,r["section_domains"],strict=True)):
            raise ValueError("native section lies outside its complete finite domains")
        for chart in r["section_charts"]:
            if tuple(case[cell] for cell in chart["support"]) not in chart["allowed"]:
                raise ValueError("native section violates an original compatibility chart")
    query = r["query"]
    if (set(query) != {"profile", "source_reference", "interpretations", "field"} or
            query["profile"] != "whole-stalk-section-query-v1" or dict(query["source_reference"]) != state.source.as_record()):
        raise ValueError("interpretation field query uses another producing source")
    from rexgraph.cpp_stalk_fields import _check_query
    _check_query(fields,query["field"])
    symbols = r["section_symbols"]; variables = r["section_variables"]
    expected = tuple(tuple(tuple(symbols[s[cell]][1] for cell in cells) for cells in variables) for s in solved["sections"])
    branches = tuple(tuple(symbols[s[cell]][1] for cell in r["branch_variables"]) for s in solved["sections"]) if r["branch_variables"] else ()
    if expected != r["assignments"] or branches != r["branch_assignments"]:
        raise ValueError("interpretation assignments differ from original native variables")
    ports = {(p[0],p[1]):p[2] for p in _ports(sections, fields, r["ports"])}
    route_ports = {(p[0],p[1],p[2]):p[3] for p in _ports(sections, fields, r["route_ports"], True)}
    if _case_queries(fields, r["query"], expected, branches, ports, route_ports) != r["case_queries"]:
        raise ValueError("case fields differ from the declared interpretation ports")
    retained = tuple(unpack_model(p) for p in r["field_outputs"])
    if set(r["case_output_indices"]) != set(range(len(retained))):
        raise ValueError("retained case outputs contain an omitted or unused field")
    for i, index in enumerate(r["case_output_indices"]):
        if type(index) is not int or not 0 <= index < len(retained):
            raise ValueError("invalid original case field index")
        field = retained[index]
        if (field.model_digest != fields.coefficient_digest or not field.source.matches(fields.source) or
                field.execution_receipt["query_declaration"] != r["case_queries"][i] or
                not np.array_equal(output.values[i],field.values)):
            raise ValueError("complete case field differs from its retained native output")
        expected_readings = dict(input=_interface_readings(fields,state.configuration["interfaces"],field.execution_receipt["input_fields"]),
                                 response=_interface_readings(fields,state.configuration["interfaces"],field.execution_receipt["response_fields"]))
        if r["interface_readings"][i] != expected_readings:
            raise ValueError("tensor interface readings differ from their original whole fields")


def stalk_section_field(output, case, row=None):
    """Select one retained interpretation's whole field or original local stalk."""
    output.check_state(); validate_stalk_sections_output(output)
    if type(case) is not int or not 0 <= case < len(output.space.keys):
        raise ValueError("select an original compatible section")
    r = output.execution_receipt
    return stalk_output_field(unpack_model(r["field_outputs"][r["case_output_indices"][case]]), row)


def replay_stalk_sections_output(output, source):
    """Reconstruct the complete native carrier and all fields on the actual source."""
    output.check_state(); validate_stalk_sections_output(output)
    state = unpack_model(output.execution_receipt["model_checkpoint"])
    replay = _infer(state, source, output.execution_receipt["query"])
    if replay.coefficient_digest != output.coefficient_digest:
        raise ValueError("retained interpretations differ from full native field replay")
    return replay


def _create(source, *, adapter="cpp_stalk_sections", **configuration):
    if adapter != "cpp_stalk_sections":raise ValueError("unsupported coupled stalk adapter")
    return cpp_stalk_sections_model(source, **configuration)


def enable_stalk_sections_provider():
    """Register coupled finite acquisition and complete native interpretation fields."""
    from rexgraph.cpp_stalk_fields import enable_stalk_fields_provider
    enable_stalk_fields_provider()
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], _PROVIDER["version"], (("cpp_stalk_sections", "1"),),
                _infer, create=_create, prepare=_prepare, train=_train)
        register_model_provider(_provider)
        return _provider
