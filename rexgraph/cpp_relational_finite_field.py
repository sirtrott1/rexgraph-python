"""Admitted finite map instances with whole heterogeneous field alternatives."""
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

from rexgraph.cpp_relational_field_program import _LIMITS, _admission, _publication
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_evidence_admission import _references, _sha
from rexgraph.field_archive import FINITE_FIELD_ARCHIVE_METHOD, FINITE_FIELD_ARCHIVE_SCHEMA, FINITE_EVALUATION_FIELD_ARCHIVE_METHOD, FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_ADAPTER = "cpp_relational_finite_field"
_PROFILE = "rexgraph.available-relational-finite-model.v1"
_QUERY = "rexgraph.available-relational-finite-query.v1"
_PROVIDER = {"name": "rexgraph.cpp.relational_finite_field", "version": "1"}
_READOUT = CoordinateSpace("native-finite-field-readout", ())
_lock = RLock()
_provider = None


def _binding():
    from rexgraph.cpp_model import _binding as binding
    module = binding()
    if not hasattr(module, "PreparedAvailableFiniteFieldMap"):
        raise ImportError("finite field models require their native available finite provider")
    return module


def _space(reference, key):
    return CoordinateSpace(f"cpp:{reference.native.namespace}:finite:{key}:literal-source-cells",
        tuple(f"{grade}:{cell}" for grade, cells in enumerate(reference.native.coordinates) for cell in cells))


def _targets(checkpoints, digests):
    if len(checkpoints) != len(digests):
        raise ValueError("finite model target checkpoints and ordering differ")
    targets = tuple(unpack_model(value) for value in checkpoints)
    for target, digest in zip(targets, digests, strict=True):
        if (not isinstance(target, ModelState) or target.adapter != "cpp_relational_field_program"
                or target.adapter_version not in {"5", "6"} or target.arithmetic != "rational"
                or not isinstance(target.source, CppSourceReference) or target.coefficient_digest != digest):
            raise ValueError("whole admitted fixed or evidence target checkpoint required")
    return targets


def validate_finite_field_state(state):
    c, p = state.configuration, state.payload
    required = {"profile", "execution_provider", "engine_receipt", "limits", "max_bytes", "key", "revision", "status",
                "finite_key", "finite_revision", "finite_source_sha256", "asset_sha256", "target_digests", "map_sha256"}
    if (state.adapter != _ADAPTER or state.adapter_version != "1" or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or set(c) != required or c["profile"] != _PROFILE
            or dict(c["execution_provider"]) != _PROVIDER or state.output_axes or state.space != _space(state.source, c["key"])
            or set(p) != {"map_archive", "target_checkpoints"} or state.step != 0 or state.parent is not None):
        raise ValueError("closed immutable admitted finite map model required")
    _admission(c["limits"], c["max_bytes"])
    if (type(p["map_archive"]) is not bytes or not p["map_archive"] or len(p["map_archive"]) > c["max_bytes"]
            or sha256(p["map_archive"]).hexdigest() != c["map_sha256"] or c["status"] not in {"observed", "prospective"}
            or not isinstance(c["engine_receipt"], Mapping) or not c["engine_receipt"]
            or any(not _sha(c[name]) for name in ("map_sha256", "finite_source_sha256", "asset_sha256"))
            or any(type(c[name]) is not int or not 0 <= c[name] < 1 << 64 for name in ("key", "revision", "finite_key", "finite_revision"))
            or not c["key"] or not c["finite_key"]):
        raise ValueError("finite map bytes, source identity or relationship differs")
    _targets(p["target_checkpoints"], c["target_digests"])


def _check_pins(module, targets):
    for target in targets:
        receipt = module.field_admission_build_info() if target.adapter_version == "5" else module.field_evidence_admission_build_info()
        if dict(target.configuration["engine_receipt"]) != receipt:
            raise ValueError("finite target requires its pinned native implementation")


def _check_target(module, native, target, authority):
    c = target.configuration
    kind = module.PreparedAvailableFieldProgram if target.adapter_version == "5" else module.PreparedAvailableFieldEvidence
    if not isinstance(native, kind):
        raise ValueError("finite target kind differs")
    native.admit(authority)
    program = native.program() if target.adapter_version == "5" else native.program().program()
    _publication(program, target.source)
    identity = program.identity()
    if ((target.adapter_version == "6" and program.archive() != c["program_archive"])
            or sha256(program.archive()).hexdigest() != c["program_sha256"]
            or identity["key"] != c["key"] or identity["revision"] != c["revision"]):
        raise ValueError("finite target complete program or source carrier differs")
    if target.adapter_version == "5":
        if native.archive() != c["available_archive"]:
            raise ValueError("finite fixed target publisher closure differs")
        from rexgraph.cpp_relational_field_admission import _dependencies as dependencies
        deps = dependencies(module, native, c["limits"])
        if len(deps) != len(target.dependencies) or any(not a.matches(b) for a, b in zip(deps, target.dependencies, strict=True)):
            raise ValueError("finite fixed target source dependencies differ")
    else:
        from rexgraph.cpp_relational_field_evidence_admission import _state
        ancestors = []; current = native
        while current is not None:
            ancestors.append(current)
            if len(ancestors) > target.step + 1:
                raise ValueError("finite target has extra evidence parents")
            current = current.parent()
        if len(ancestors) != target.step + 1 or sha256(ancestors[-1].archive()).hexdigest() != c["initial_evidence_sha256"]:
            raise ValueError("finite target original evidence parent differs")
        reconstructed = None
        for step, evidence in enumerate(reversed(ancestors)):
            e = evidence.evidence().identity()
            if any(e[name] != c[key] for name, key in (("key", "parameter_key"), ("node", "learned_node"), ("grade", "learned_grade"))):
                raise ValueError("finite target evidence parameter carrier differs")
            reconstructed = _state(evidence, target.source, c, None if reconstructed is None else reconstructed.coefficient_digest, step)
        if reconstructed.coefficient_digest != target.coefficient_digest:
            raise ValueError("finite target complete evidence or immutable checkpoint lineage differs")


def _check_map(module, native, targets, reference, c, authority):
    native.admit(authority)
    actual = native.targets(authority, max_records=c["limits"][0])
    if len(actual) != len(targets):
        raise ValueError("finite target count differs")
    for value, target in zip(actual, targets, strict=True):
        _check_target(module, value, target, authority)
    finite = native.source(authority); literal = finite.raw().asset()
    source = module.NativeSource(literal.source_archive(), max_dimension=max(c["limits"][:2]), max_records=c["limits"][1])
    reference.bind(source)
    info = native.inspect(authority, max_records=c["limits"][1]); f = finite.inspect(authority, max_records=c["limits"][1])
    if (sha256(native.archive()).hexdigest() != c["map_sha256"] or sha256(finite.archive()).hexdigest() != c["finite_source_sha256"]
            or sha256(literal.archive()).hexdigest() != c["asset_sha256"]
            or any(info[name] != c[key] for name, key in (("key", "key"), ("revision", "revision"), ("status", "status")))
            or f["key"] != c["finite_key"] or f["revision"] != c["finite_revision"]):
        raise ValueError("finite original map, literal asset or constraint source differs")


def _dependencies(module, value, targets, authority, limits):
    deps = list(_references(module, value.dependency_archives(authority, max_records=limits[1]), limits))
    for target in targets:
        deps.append(target.source); deps.extend(target.dependencies)
    if len(deps) > limits[1]:
        raise ValueError("finite complete source dependency count exceeds admission")
    return tuple(deps)


def cpp_available_finite_field_model(source, *, available_map, targets, execution_context,
                                     limits=_LIMITS, max_bytes=256*1024*1024, reference=None):
    module = _binding(); authority = _context(module, execution_context); limits = _admission(limits, max_bytes)
    targets = tuple(targets)
    if not isinstance(source, module.NativeSource) or not isinstance(available_map, module.PreparedAvailableFiniteFieldMap):
        raise TypeError("actual literal source and admitted finite map owners required")
    checkpoints = tuple(pack_model(value, native=True) for value in targets)
    digests = tuple(value.coefficient_digest for value in targets); targets = _targets(checkpoints, digests)
    _check_pins(module, targets)
    if available_map.admission() != {"limits": limits, "max_bytes": max_bytes}:
        raise ValueError("finite map admission differs from its model declaration")
    available_map.admit(authority); info = available_map.inspect(authority, max_records=limits[1])
    finite = available_map.source(authority); f = finite.inspect(authority, max_records=limits[1]); image = available_map.archive()
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    c = {"profile": _PROFILE, "execution_provider": _PROVIDER, "engine_receipt": module.finite_field_admission_build_info(),
         "limits": limits, "max_bytes": max_bytes, "key": info["key"], "revision": info["revision"], "status": info["status"],
         "finite_key": f["key"], "finite_revision": f["revision"], "finite_source_sha256": sha256(finite.archive()).hexdigest(),
         "asset_sha256": sha256(finite.raw().asset().archive()).hexdigest(), "target_digests": digests, "map_sha256": sha256(image).hexdigest()}
    _check_map(module, available_map, targets, ref, c, authority)
    state = ModelState(_ADAPTER, "1", ref, _space(ref, c["key"]), (), c,
                       {"map_archive": image, "target_checkpoints": checkpoints}, "rational",
                       dependencies=_dependencies(module, available_map, targets, authority, limits))
    enable_finite_field_provider()
    return state


def _prepare(state, source, *, execution_context):
    validate_finite_field_state(state); state.bind(source)
    module = _binding(); authority = _context(module, execution_context); c = state.configuration
    if dict(c["engine_receipt"]) != module.finite_field_admission_build_info():
        raise ValueError("finite model requires its pinned native implementation")
    targets = _targets(state.payload["target_checkpoints"], c["target_digests"]); _check_pins(module, targets)
    native = module.PreparedAvailableFiniteFieldMap.from_archive(state.payload["map_archive"], authority, limits=c["limits"], max_bytes=c["max_bytes"])
    _check_map(module, native, targets, state.source, c, authority)
    deps = _dependencies(module, native, targets, authority, c["limits"])
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a, b in zip(deps, state.dependencies, strict=True)):
        raise ValueError("finite complete source or target dependency ordering differs")
    return native


def finite_field_query(state):
    validate_finite_field_state(state)
    return freeze_tree({"profile": _QUERY, "model_digest": state.coefficient_digest, "source": state.source.as_record(),
                        "map_sha256": state.configuration["map_sha256"]})


def _output(state, value, authority, *, selected=False):
    c = state.configuration; module = _binding(); image = value.archive()
    schema, method = ((FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA, FINITE_EVALUATION_FIELD_ARCHIVE_METHOD) if selected else
                      (FINITE_FIELD_ARCHIVE_SCHEMA, FINITE_FIELD_ARCHIVE_METHOD))
    targets = _targets(state.payload["target_checkpoints"], c["target_digests"])
    family = value.family(authority) if selected else value
    deps = _dependencies(module, value if selected else family.model(authority), targets, authority, c["limits"])
    archive = {"schema": schema, "data": image, "sha256": sha256(image).hexdigest(), "map_sha256": c["map_sha256"], "engine_receipt": c["engine_receipt"]}
    receipt = {"profile": method, "limits": c["limits"], "max_bytes": c["max_bytes"], "status": c["status"],
               "interpretation": "explicit-prospective-finite-choice" if selected else "complete-native-finite-family",
               "finite_key": c["finite_key"], "finite_revision": c["finite_revision"], "key": c["key"], "revision": c["revision"],
               "finite_source_sha256": c["finite_source_sha256"], "asset_sha256": c["asset_sha256"],
               "target_digests": c["target_digests"], "target_checkpoints": state.payload["target_checkpoints"], "dependency_count": len(deps)}
    value.admit(authority)
    return ModelOutput([], _READOUT, (), state.source, state.coefficient_digest, "rational", method,
                       dependencies=deps, execution_receipt=receipt, field_archive=archive)


def _infer(state, source, inputs, *, execution_context):
    native = _prepare(state, source, execution_context=execution_context)
    if (not isinstance(inputs, Mapping) or set(inputs) != {"profile", "model_digest", "source", "map_sha256"}
            or inputs["profile"] != _QUERY or inputs["model_digest"] != state.coefficient_digest
            or inputs["map_sha256"] != state.configuration["map_sha256"]
            or not source_from_record(inputs["source"], allow_cpp=True).matches(state.source)):
        raise ValueError("closed query for the original immutable finite map instance required")
    return _output(state, native.execute(execution_context), execution_context)


def read_finite_field_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if (not isinstance(output, ModelOutput) or output.method not in {FINITE_FIELD_ARCHIVE_METHOD, FINITE_EVALUATION_FIELD_ARCHIVE_METHOD}
            or output.space != _READOUT or output.axes or output.values.size or output.grade is not None or output.variance != "coordinate"):
        raise TypeError("complete native finite family or explicit finite choice required")
    output.check_state(); a, r = output.field_archive, output.execution_receipt
    selected = output.method == FINITE_EVALUATION_FIELD_ARCHIVE_METHOD
    schema = FINITE_EVALUATION_FIELD_ARCHIVE_SCHEMA if selected else FINITE_FIELD_ARCHIVE_SCHEMA
    required = {"profile", "limits", "max_bytes", "status", "interpretation", "finite_key", "finite_revision", "key", "revision",
                "finite_source_sha256", "asset_sha256", "target_digests", "target_checkpoints", "dependency_count"}
    if (not isinstance(r, Mapping) or set(r) != required or r["profile"] != output.method or r["status"] not in {"observed", "prospective"}
            or r["interpretation"] != ("explicit-prospective-finite-choice" if selected else "complete-native-finite-family")
            or r["dependency_count"] != len(output.dependencies)):
        raise ValueError("closed finite field execution receipt required")
    admitted = _admission(r["limits"], r["max_bytes"]); maximum = r["max_bytes"]
    if limits is not None: admitted = tuple(min(a,b) for a,b in zip(admitted, _admission(limits,maximum), strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes); maximum = min(maximum,max_bytes)
    module = _binding(); authority = _context(module, execution_context)
    if a["schema"] != schema or dict(a["engine_receipt"]) != module.finite_field_admission_build_info():
        raise ValueError("finite output requires its pinned native implementation")
    targets = _targets(r["target_checkpoints"], r["target_digests"]); _check_pins(module, targets)
    if state is not None:
        validate_finite_field_state(state)
        if (output.model_digest != state.coefficient_digest or not output.source.matches(state.source)
                or a["map_sha256"] != state.configuration["map_sha256"] or tuple(r["target_digests"]) != tuple(state.configuration["target_digests"])):
            raise ValueError("finite output belongs to another model")
    reader = module.NativeAvailableFiniteFieldEvaluation if selected else module.NativeAvailableFiniteFieldResult
    result = reader.from_archive(a["data"], authority, limits=admitted, max_bytes=maximum)
    family = result.family(authority) if selected else result
    c = dict(r, map_sha256=a["map_sha256"]); _check_map(module, family.model(authority), targets, output.source, c, authority)
    deps = _dependencies(module, result if selected else family.model(authority), targets, authority, admitted)
    if len(deps) != len(output.dependencies) or any(not a.matches(b) for a,b in zip(deps,output.dependencies,strict=True)):
        raise ValueError("finite output complete original dependencies differ")
    result.admit(authority)
    return result


def evaluate_finite_field(output, ordinal, values, parameter_setting, choice_origin, *, execution_context, state):
    if output.method != FINITE_FIELD_ARCHIVE_METHOD:
        raise TypeError("a complete unresolved finite family is required")
    family = read_finite_field_result(output, execution_context=execution_context, state=state)
    return _output(state, family.evaluate(ordinal, values, parameter_setting, choice_origin, execution_context), execution_context, selected=True)


def finite_field_branches(output, begin, count, *, execution_context):
    value = read_finite_field_result(output, execution_context=execution_context)
    family = value.family(execution_context) if output.method == FINITE_EVALUATION_FIELD_ARCHIVE_METHOD else value
    targets = _targets(output.execution_receipt["target_checkpoints"], output.execution_receipt["target_digests"])
    records = []
    for branch in family.select(begin, count, execution_context):
        info = branch.inspect(execution_context); raw = branch.raw().result(); index = info["target"]
        records.append(dict(info, target_source=None if index is None else targets[index].source.as_record(),
                            fields=() if raw is None else raw.fields(), origin_output=output.coefficient_digest,
                            family_sha256=sha256(family.archive()).hexdigest()))
    return tuple(records)


def finite_field_info(output, *, execution_context):
    value = read_finite_field_result(output, execution_context=execution_context)
    selected = output.method == FINITE_EVALUATION_FIELD_ARCHIVE_METHOD
    family = value.family(execution_context) if selected else value
    return {"map_sha256": output.field_archive["map_sha256"], "archive_sha256": output.field_archive["sha256"],
            "branch_count": family.inspect(execution_context)["branch_count"],
            "target_count": len(output.execution_receipt["target_digests"]),
            "selected_ordinal": value.inspect(execution_context)["ordinal"] if selected else None,
            "relationship_status": output.execution_receipt["status"],
            "availability": "independent-native-context-admitted", "retains_complete_alternatives": True}


def finite_field_stalks(output, node, cells, *, execution_context, coordinate_limit=256):
    if output.method != FINITE_EVALUATION_FIELD_ARCHIVE_METHOD:
        raise TypeError("finite stalk selection requires an explicit prospective finite completion")
    value = read_finite_field_result(output, execution_context=execution_context)
    info = value.inspect(execution_context); ordinal = info["ordinal"]
    branch = value.family(execution_context).select(ordinal, 1, execution_context)[0].inspect(execution_context)
    target = _targets(output.execution_receipt["target_checkpoints"], output.execution_receipt["target_digests"])[branch["target"]]
    return tuple(dict(row, finite_ordinal=ordinal, target_source=target.source.as_record(),
                      origin_output=output.coefficient_digest, map_sha256=output.field_archive["map_sha256"])
                 for row in value.raw().result().select(node, cells, coordinate_limit=coordinate_limit))


def inspect_finite_field_model(state, source, *, execution_context):
    return _prepare(state, source, execution_context=execution_context)


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError("finite field adapter required")
    return cpp_available_finite_field_model(source, **configuration)


def enable_finite_field_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", ((_ADAPTER, "1"),), _infer, create=_create, prepare=_prepare,
                                      context_operations=("create", "prepare", "infer"))
        register_model_provider(_provider)
    return _provider
