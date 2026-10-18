"""Versioned faithful source extensions with complete input completion families."""
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

from rexgraph.cpp_relational_field_program import _ADAPTER, _LIMITS, _admission, _publication, _space
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_evidence_admission import _binding as _evidence_binding, _dependencies, _input_rows, _sha
from rexgraph.field_archive import EXTENSION_FIELD_ARCHIVE_METHOD, EXTENSION_FIELD_ARCHIVE_SCHEMA, EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD, EXTENSION_EVALUATION_FIELD_ARCHIVE_SCHEMA
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "rexgraph.available-relational-field-extension.v1"
_QUERY = "rexgraph.available-relational-extension-query.v1"
_PROVIDER = {"name": "rexgraph.cpp.relational_field_program", "version": "7"}
_provider = None
_lock = RLock()


def _binding():
    module = _evidence_binding()
    if not hasattr(module, "PreparedAvailableFieldExtension"):
        raise ImportError("faithful source extension requires its native extension provider")
    return module


def _parent(state):
    parent = unpack_model(state.payload["parent_checkpoint"])
    if (not isinstance(parent, ModelState) or parent.adapter != _ADAPTER or parent.adapter_version not in {"5", "6", "7"}
            or parent.coefficient_digest != state.parent or state.step != parent.step + 1 or parent.output_axes != state.output_axes):
        raise ValueError("extension requires its complete immutable parent model")
    return parent


def validate_field_extension_state(state):
    c, p = state.configuration, state.payload
    required = {"profile", "execution_provider", "engine_receipt", "program_archive", "program_sha256", "source_program_archive",
                "source_program_sha256", "source_reference", "key", "revision", "limits", "max_bytes", "extension_key", "extension_revision", "status"}
    if (state.adapter != _ADAPTER or state.adapter_version != "7" or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or set(c) != required or c["profile"] != _PROFILE
            or dict(c["execution_provider"]) != _PROVIDER or state.output_axes or state.space != _space(state.source, c["key"])
            or set(p) != {"extension_archive", "extension_sha256", "parent_checkpoint"} or state.parent is None):
        raise ValueError("closed versioned source-extension model required")
    _admission(c["limits"], c["max_bytes"])
    for image, digest in ((c["program_archive"], c["program_sha256"]), (c["source_program_archive"], c["source_program_sha256"]),
                          (p["extension_archive"], p["extension_sha256"])):
        if type(image) is not bytes or not image or len(image) > c["max_bytes"] or sha256(image).hexdigest() != digest:
            raise ValueError("extension whole archive differs")
    if (not isinstance(c["engine_receipt"], Mapping) or c["status"] not in {"observed", "prospective"}
            or any(type(c[name]) is not int or not 0 <= c[name] < 1 << 64 for name in ("key", "revision", "extension_key", "extension_revision"))
            or not c["key"] or not c["extension_key"] or not c["extension_revision"]):
        raise ValueError("extension relationship identity differs")
    parent = _parent(state)
    if not source_from_record(c["source_reference"], allow_cpp=True).matches(parent.source):
        raise ValueError("extension original source lineage differs")


def _check_parent_pins(module, parent):
    current = parent
    while True:
        version = current.adapter_version
        receipt = {"5": module.field_admission_build_info, "6": module.field_evidence_admission_build_info,
                   "7": module.field_extension_build_info}[version]()
        if dict(current.configuration["engine_receipt"]) != receipt:
            raise ValueError("extension parent requires its pinned native implementation")
        if version != "7": return
        current = _parent(current)


def _check_native_parent(module, extension, parent, authority):
    _check_parent_pins(module, parent)
    version = parent.adapter_version; c = parent.configuration
    _publication(extension.source(), parent.source)
    previous, root = extension.parent(), extension.root()
    if version == "7":
        valid = previous is not None and previous.archive() == parent.payload["extension_archive"]
    elif version == "5":
        valid = previous is None and isinstance(root, module.PreparedAvailableFieldProgram) and root.archive() == parent.configuration["available_archive"]
    else:
        valid = (previous is None and isinstance(root, module.NativeAvailableAffineEvaluation)
                 and root.family().evidence().archive() == parent.payload["evidence_archive"])
    if not valid:
        raise ValueError("extension actual native parent differs from its parent model")
    if version == "7":
        checked = module.PreparedAvailableFieldExtension.from_archive(previous.archive(), authority, limits=c["limits"], max_bytes=c["max_bytes"])
        _check_native_parent(module, previous, _parent(parent), authority)
        program = checked.target().program(); info = checked.info()
        if (checked.source().archive() != c["source_program_archive"]
                or any(info[a] != c[b] for a,b in (("key","extension_key"),("revision","extension_revision"),("status","status")))):
            raise ValueError("extension parent relationship metadata differs")
    elif version == "5":
        checked = module.PreparedAvailableFieldProgram.from_archive(root.archive(), authority, limits=c["limits"], max_bytes=c["max_bytes"])
        program = checked.program()
    else:
        checked = module.PreparedAvailableFieldEvidence.from_archive(root.family().evidence().archive(), authority, limits=c["limits"], max_bytes=c["max_bytes"])
        program = checked.program().program()
        from rexgraph.cpp_relational_field_evidence_admission import _state
        ancestors = []; current = checked
        while current is not None:
            ancestors.append(current)
            if len(ancestors) > parent.step+1: raise ValueError("extension parent has extra evidence stages")
            current = current.parent()
        if len(ancestors) != parent.step+1 or sha256(ancestors[-1].archive()).hexdigest() != c["initial_evidence_sha256"]:
            raise ValueError("extension original evidence ancestry differs")
        reconstructed = None
        for step, evidence in enumerate(reversed(ancestors)):
            identity = evidence.evidence().identity()
            if any(identity[a] != c[b] for a,b in (("key","parameter_key"),("node","learned_node"),("grade","learned_grade"))):
                raise ValueError("extension original evidence identity differs")
            reconstructed = _state(evidence, parent.source, c, None if reconstructed is None else reconstructed.coefficient_digest, step)
        if reconstructed.coefficient_digest != parent.coefficient_digest:
            raise ValueError("extension complete learned parent checkpoint differs")
    identity = program.identity(); _publication(program, parent.source)
    if (sha256(program.archive()).hexdigest() != c["program_sha256"]
            or identity["key"] != c["key"] or identity["revision"] != c["revision"]
            or (version != "5" and program.archive() != c["program_archive"])):
        raise ValueError("extension parent program metadata differs")
    deps = _dependencies(module, checked, c["limits"])
    if len(deps) != len(parent.dependencies) or any(not a.matches(b) for a,b in zip(deps,parent.dependencies,strict=True)):
        raise ValueError("extension complete parent dependencies differ")


def cpp_available_field_extension_model(source, *, parent, available_extension, execution_context,
                                        limits=_LIMITS, max_bytes=256*1024*1024, reference=None):
    """Create an explicit structural child retaining the original parent checkpoint."""
    module = _binding(); authority = _context(module, execution_context); limits = _admission(limits, max_bytes)
    if (not isinstance(source, module.NativeSource) or not isinstance(parent, ModelState)
            or parent.adapter != _ADAPTER or parent.adapter_version not in {"5", "6", "7"}
            or not isinstance(available_extension, module.PreparedAvailableFieldExtension)):
        raise TypeError("actual native extension, target source and admitted parent model required")
    parent.check_state(); available_extension.admit(authority)
    if available_extension.admission() != {"limits": limits, "max_bytes": max_bytes}:
        raise ValueError("native extension admission differs from its model declaration")
    _check_native_parent(module, available_extension, parent, authority)
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    target = available_extension.target().program(); _publication(target, ref)
    identity, info = target.identity(), available_extension.info()
    image, before = target.archive(), available_extension.source().archive()
    c = {"profile": _PROFILE, "execution_provider": _PROVIDER, "engine_receipt": module.field_extension_build_info(),
         "program_archive": image, "program_sha256": sha256(image).hexdigest(), "source_program_archive": before,
         "source_program_sha256": sha256(before).hexdigest(), "source_reference": parent.source.as_record(),
         "key": identity["key"], "revision": identity["revision"], "limits": limits, "max_bytes": max_bytes,
         "extension_key": info["key"], "extension_revision": info["revision"], "status": info["status"]}
    archive = available_extension.archive()
    p = {"extension_archive": archive, "extension_sha256": sha256(archive).hexdigest(), "parent_checkpoint": pack_model(parent, native=True)}
    state = ModelState(_ADAPTER, "7", ref, _space(ref, identity["key"]), (), c, p, "rational", step=parent.step+1,
                       parent=parent.coefficient_digest, dependencies=_dependencies(module, available_extension, limits))
    enable_field_extension_provider()
    _prepare(state, source, execution_context=authority)
    return state


def _prepare(state, source, *, execution_context):
    validate_field_extension_state(state); state.bind(source)
    module = _binding(); authority = _context(module, execution_context); c = state.configuration
    if dict(c["engine_receipt"]) != module.field_extension_build_info():
        raise ValueError("extension model requires its pinned native implementation")
    parent = _parent(state); _check_parent_pins(module, parent)
    extension = module.PreparedAvailableFieldExtension.from_archive(state.payload["extension_archive"], authority,
                                                                   limits=c["limits"], max_bytes=c["max_bytes"])
    _check_native_parent(module, extension, parent, authority); target = extension.target().program(); _publication(target, state.source)
    info, identity = extension.info(), target.identity()
    if (target.archive() != c["program_archive"] or extension.source().archive() != c["source_program_archive"]
            or identity["key"] != c["key"] or identity["revision"] != c["revision"]
            or any(info[a] != c[b] for a, b in (("key", "extension_key"), ("revision", "extension_revision"), ("status", "status")))):
        raise ValueError("extension complete source/program relationship differs")
    deps = _dependencies(module, extension, c["limits"])
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a,b in zip(deps,state.dependencies,strict=True)):
        raise ValueError("extension original parent and publication dependencies differ")
    return extension


def field_extension_query(state, inputs):
    validate_field_extension_state(state)
    return freeze_tree({"profile": _QUERY, "model_digest": state.coefficient_digest, "source": state.source.as_record(),
                        "input_source": state.configuration["source_reference"], "inputs": _input_rows(inputs)})


def _output(state, result, *, selected=False):
    module = _binding(); c = state.configuration; family = result.family() if selected else result; image = result.archive()
    schema, method = ((EXTENSION_EVALUATION_FIELD_ARCHIVE_SCHEMA, EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD) if selected else
                      (EXTENSION_FIELD_ARCHIVE_SCHEMA, EXTENSION_FIELD_ARCHIVE_METHOD))
    deps = _dependencies(module, result, c["limits"])
    archive = {"schema": schema, "data": image, "sha256": sha256(image).hexdigest(), "program_sha256": c["program_sha256"], "engine_receipt": c["engine_receipt"]}
    receipt = {"profile": method, "limits": c["limits"], "max_bytes": c["max_bytes"],
               "interpretation": "explicit-input-prospective-completion" if selected else "complete-correlated-extension-family",
               "extension_sha256": sha256(family.model().archive()).hexdigest(), "source_program_sha256": c["source_program_sha256"],
               "input_source": c["source_reference"], "status": c["status"], "dependency_count": len(deps)}
    return ModelOutput([], CoordinateSpace("native-program-field-readout", ()), (), state.source, state.coefficient_digest,
                       "rational", method, dependencies=deps, execution_receipt=receipt, field_archive=archive)


def _infer(state, source, inputs, *, execution_context):
    extension = _prepare(state, source, execution_context=execution_context)
    if (not isinstance(inputs, Mapping) or set(inputs) != {"profile", "model_digest", "source", "input_source", "inputs"}
            or inputs["profile"] != _QUERY or inputs["model_digest"] != state.coefficient_digest
            or not source_from_record(inputs["source"], allow_cpp=True).matches(state.source)
            or not source_from_record(inputs["input_source"], allow_cpp=True).matches(_parent(state).source)):
        raise ValueError("closed independently supplied extension input query required")
    return _output(state, extension.execute_inputs(_input_rows(inputs["inputs"]), execution_context))


def read_field_extension_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if (not isinstance(output, ModelOutput) or output.method not in {EXTENSION_FIELD_ARCHIVE_METHOD, EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD}
            or output.space != CoordinateSpace("native-program-field-readout", ()) or output.axes or output.values.size
            or output.grade is not None or output.variance != "coordinate"):
        raise TypeError("complete extension family or explicit completion required")
    output.check_state(); a, r = output.field_archive, output.execution_receipt
    selected = output.method == EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD
    schema = EXTENSION_EVALUATION_FIELD_ARCHIVE_SCHEMA if selected else EXTENSION_FIELD_ARCHIVE_SCHEMA
    meaning = "explicit-input-prospective-completion" if selected else "complete-correlated-extension-family"
    if (not isinstance(r, Mapping) or set(r) != {"profile", "limits", "max_bytes", "interpretation", "extension_sha256", "source_program_sha256", "input_source", "status", "dependency_count"}
            or r["profile"] != output.method or r["interpretation"] != meaning or not _sha(r["extension_sha256"])
            or not _sha(r["source_program_sha256"]) or r["status"] not in {"observed", "prospective"} or r["dependency_count"] != len(output.dependencies)):
        raise ValueError("closed complete extension execution receipt required")
    admitted = _admission(r["limits"], r["max_bytes"]); maximum = r["max_bytes"]
    if limits is not None: admitted = tuple(min(a,b) for a,b in zip(admitted,_admission(limits,maximum),strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes); maximum = min(maximum,max_bytes)
    module = _binding(); authority = _context(module, execution_context)
    if a["schema"] != schema or dict(a["engine_receipt"]) != module.field_extension_build_info():
        raise ValueError("extension output requires its pinned native implementation")
    reader = module.NativeAvailableExtensionEvaluation if selected else module.NativeAvailableExtensionFamily
    result = reader.from_archive(a["data"], authority, limits=admitted, max_bytes=maximum)
    family = result.family() if selected else result; extension = family.model(); target = extension.target().program()
    _publication(target, output.source); _publication(extension.source(), source_from_record(r["input_source"], allow_cpp=True))
    if (sha256(extension.archive()).hexdigest() != r["extension_sha256"] or sha256(target.archive()).hexdigest() != a["program_sha256"]
            or sha256(extension.source().archive()).hexdigest() != r["source_program_sha256"] or extension.info()["status"] != r["status"]):
        raise ValueError("extension output original source or relationship differs")
    deps = _dependencies(module, result, admitted)
    if len(deps) != len(output.dependencies) or any(not a.matches(b) for a,b in zip(deps,output.dependencies,strict=True)):
        raise ValueError("extension output complete dependencies differ")
    if state is not None:
        validate_field_extension_state(state)
        if (output.model_digest != state.coefficient_digest or not output.source.matches(state.source)
                or r["extension_sha256"] != state.payload["extension_sha256"] or a["program_sha256"] != state.configuration["program_sha256"]):
            raise ValueError("extension output belongs to another model")
    return result


def evaluate_field_extension(output, values, origin, *, execution_context, state):
    family = read_field_extension_result(output, execution_context=execution_context, state=state)
    if output.method != EXTENSION_FIELD_ARCHIVE_METHOD: raise TypeError("a complete unresolved extension family is required")
    return _output(state, family.evaluate(values, origin, execution_context), selected=True)


def inspect_field_extension_model(state, source, *, execution_context):
    return _prepare(state, source, execution_context=execution_context)


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError("whole-field extension adapter required")
    return cpp_available_field_extension_model(source, **configuration)


def enable_field_extension_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "7", ((_ADAPTER, "7"),), _infer, create=_create, prepare=_prepare,
                                      context_operations=("create", "prepare", "infer"))
        register_model_provider(_provider)
    return _provider
