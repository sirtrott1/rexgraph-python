"""Immutable operator checkpoints over explicitly admitted independent inputs."""
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

from rexgraph.cpp_relational_field_program import _LIMITS, _admission
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_evidence_admission import _references, _sha
from rexgraph.cpp_relational_finite_field import (
    validate_finite_field_state, _prepare as _prepare_reference,
    _targets, _check_pins as _check_target_pins,
)
from rexgraph.field_archive import (
    OPERATOR_FIELD_ARCHIVE_SCHEMA, OPERATOR_FIELD_ARCHIVE_METHOD,
    OPERATOR_EVALUATION_FIELD_ARCHIVE_SCHEMA, OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD,
)
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelState, ModelOutput, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_ADAPTER = "cpp_relational_finite_operator"
_PROFILE = "rexgraph.available-relational-finite-operator-model.v1"
_QUERY = "rexgraph.available-relational-finite-operator-query.v1"
_PROVIDER = {"name": "rexgraph.cpp.relational_finite_operator", "version": "1"}
_READOUT = CoordinateSpace("native-finite-operator-readout", ())
_CONFIGURATION = {"profile", "execution_provider", "engine_receipt", "limits", "max_bytes",
                  "key", "revision", "status", "operator_sha256", "reference_digest"}
_provider = None
_lock = RLock()


def _binding():
    from rexgraph.cpp_model import _binding as binding
    module = binding()
    if not hasattr(module, "PreparedAvailableFiniteFieldOperator"):
        raise ImportError("finite operator models require their native available operator provider")
    return module


def _reference(checkpoint):
    value = unpack_model(checkpoint)
    if not isinstance(value, ModelState):
        raise TypeError("complete original finite checkpoint required")
    validate_finite_field_state(value)
    return value


def validate_finite_operator_state(state):
    c, p = state.configuration, state.payload
    if (state.adapter != _ADAPTER or state.adapter_version != "1" or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or set(c) != _CONFIGURATION
            or c["profile"] != _PROFILE or dict(c["execution_provider"]) != _PROVIDER
            or set(p) != {"operator_archive", "reference_checkpoint"}):
        raise ValueError("closed immutable admitted operator model required")
    _admission(c["limits"], c["max_bytes"])
    if (type(p["operator_archive"]) is not bytes or not p["operator_archive"]
            or len(p["operator_archive"]) > c["max_bytes"]
            or sha256(p["operator_archive"]).hexdigest() != c["operator_sha256"]
            or not _sha(c["operator_sha256"]) or not _sha(c["reference_digest"])
            or not isinstance(c["engine_receipt"], Mapping) or not c["engine_receipt"]
            or c["status"] not in {"observed", "prospective"}
            or any(type(c[name]) is not int or not 0 < c[name] < 1 << 64 for name in ("key", "revision"))):
        raise ValueError("operator whole bytes, identity or admission differs")
    parent = _reference(p["reference_checkpoint"])
    if (c["reference_digest"] != parent.coefficient_digest or state.parent != parent.coefficient_digest
            or state.step != parent.step + 1 or state.space != parent.space or state.output_axes != parent.output_axes
            or not state.source.matches(parent.source)):
        raise ValueError("operator original checkpoint, source, coordinates or child lineage differs")
    return parent


def _pins(module, c, parent):
    if dict(c["engine_receipt"]) != module.finite_operator_admission_build_info():
        raise ValueError("operator requires its pinned native implementation")
    if dict(parent.configuration["engine_receipt"]) != module.finite_field_admission_build_info():
        raise ValueError("operator reference requires its pinned native implementation")
    targets = _targets(parent.payload["target_checkpoints"], parent.configuration["target_digests"])
    _check_target_pins(module, targets)


def _check_operator(value, parent, source, c, authority):
    value.admit(authority)
    original = _prepare_reference(parent, source, execution_context=authority)
    info = value.inspect(authority)
    if (value.reference(authority).archive() != original.archive()
            or sha256(value.archive()).hexdigest() != c["operator_sha256"]
            or any(info[name] != c[name] for name in ("key", "revision", "status"))):
        raise ValueError("operator complete reference or declared identity differs")


def _dependencies(module, value, parent, authority, limits):
    rows = list(_references(module, value.dependency_archives(authority, max_records=limits[1]), limits))
    rows.append(parent.source)
    rows.extend(parent.dependencies)
    if len(rows) > limits[1]:
        raise ValueError("operator complete dependency count exceeds admission")
    return tuple(rows)


def _state(module, value, parent, source, c, authority):
    _check_operator(value, parent, source, c, authority)
    return ModelState(_ADAPTER, "1", parent.source, parent.space, parent.output_axes, c,
                      {"operator_archive": value.archive(), "reference_checkpoint": pack_model(parent, native=True)},
                      "rational", step=parent.step + 1, parent=parent.coefficient_digest,
                      dependencies=_dependencies(module, value, parent, authority, c["limits"]))


def cpp_available_finite_operator_model(source, *, reference_state, available_operator, execution_context,
                                        limits=_LIMITS, max_bytes=256*1024*1024, reference=None):
    module = _binding(); authority = _context(module, execution_context); limits = _admission(limits, max_bytes)
    if (not isinstance(source, module.NativeSource) or not isinstance(reference_state, ModelState)
            or not isinstance(available_operator, module.PreparedAvailableFiniteFieldOperator)):
        raise TypeError("actual original source, finite checkpoint and admitted operator required")
    parent = _reference(pack_model(reference_state, native=True)); parent.bind(source)
    if reference is not None and not reference.bind(source).matches(parent.source):
        raise ValueError("operator original source reference differs")
    if available_operator.admission() != {"limits": limits, "max_bytes": max_bytes}:
        raise ValueError("operator admission differs from its model declaration")
    info = available_operator.inspect(authority); image = available_operator.archive()
    c = {"profile": _PROFILE, "execution_provider": _PROVIDER, "engine_receipt": module.finite_operator_admission_build_info(),
         "limits": limits, "max_bytes": max_bytes, "key": info["key"], "revision": info["revision"],
         "status": info["status"], "operator_sha256": sha256(image).hexdigest(), "reference_digest": parent.coefficient_digest}
    _pins(module, c, parent)
    result = _state(module, available_operator, parent, source, c, authority)
    enable_finite_operator_provider()
    return result


def _prepare(state, source, *, execution_context):
    parent = validate_finite_operator_state(state); state.bind(source)
    module = _binding(); authority = _context(module, execution_context); c = state.configuration
    _pins(module, c, parent)
    value = module.PreparedAvailableFiniteFieldOperator.from_archive(state.payload["operator_archive"], authority,
                                                                    limits=c["limits"], max_bytes=c["max_bytes"])
    restored = _state(module, value, parent, source, c, authority)
    if restored.coefficient_digest != state.coefficient_digest:
        raise ValueError("operator original complete checkpoint or dependency ordering differs")
    return value


def inspect_finite_operator_model(state, source, *, execution_context):
    return _prepare(state, source, execution_context=execution_context)


def finite_operator_query(state, instance, *, execution_context):
    validate_finite_operator_state(state); module = _binding(); authority = _context(module, execution_context)
    if not isinstance(instance, module.NativeAvailableFiniteOperatorInstance):
        raise TypeError("actual independently admitted operator instance required")
    instance.admit(authority)
    if (instance.operator_program(authority).archive() != state.payload["operator_archive"]
            or instance.admission() != {"limits": tuple(state.configuration["limits"]), "max_bytes": state.configuration["max_bytes"]}):
        raise ValueError("input instance belongs to another operator or admission")
    image = instance.archive()
    return freeze_tree({"profile": _QUERY, "model_digest": state.coefficient_digest, "source": state.source.as_record(),
                        "operator_sha256": state.configuration["operator_sha256"], "instance_archive": image,
                        "instance_sha256": sha256(image).hexdigest()})


def _actual(module, instance, authority, limits, max_bytes):
    info = instance.inspect(authority); finite = instance.source(authority); asset = finite.raw().asset()
    source = CppSourceReference(module.NativeSource(asset.source_archive(), max_bytes=max_bytes,
                                                  max_dimension=max(limits[:2]), max_records=limits[1]))
    return {"source": source.as_record(), "finite_sha256": sha256(finite.archive()).hexdigest(),
            "asset_sha256": sha256(asset.archive()).hexdigest(), "instance_sha256": sha256(instance.archive()).hexdigest(),
            "key": info["key"], "revision": info["revision"], "origins": info["origins"]}


def _output(state, value, authority, *, selected=False):
    c = state.configuration; module = _binding(); parent = _reference(state.payload["reference_checkpoint"])
    family = value.family(authority) if selected else value; instance = family.instance(authority)
    deps = _dependencies(module, value if selected else instance, parent, authority, c["limits"])
    schema, method = ((OPERATOR_EVALUATION_FIELD_ARCHIVE_SCHEMA, OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD) if selected else
                      (OPERATOR_FIELD_ARCHIVE_SCHEMA, OPERATOR_FIELD_ARCHIVE_METHOD))
    image = value.archive()
    archive = {"schema": schema, "data": image, "sha256": sha256(image).hexdigest(),
               "operator_sha256": c["operator_sha256"], "engine_receipt": c["engine_receipt"]}
    receipt = {"profile": method, "limits": c["limits"], "max_bytes": c["max_bytes"], "model_configuration": c,
               "reference_checkpoint": state.payload["reference_checkpoint"], "actual_input": _actual(module, instance, authority, c["limits"], c["max_bytes"]),
               "interpretation": "explicit-prospective-operator-choice" if selected else "complete-native-operator-family",
               "dependency_count": len(deps)}
    value.admit(authority)
    return ModelOutput([], _READOUT, (), state.source, state.coefficient_digest, "rational", method,
                       dependencies=deps, execution_receipt=receipt, field_archive=archive)


def _infer(state, source, inputs, *, execution_context):
    operator = _prepare(state, source, execution_context=execution_context); c = state.configuration
    required = {"profile", "model_digest", "source", "operator_sha256", "instance_archive", "instance_sha256"}
    if (not isinstance(inputs, Mapping) or set(inputs) != required or inputs["profile"] != _QUERY
            or inputs["model_digest"] != state.coefficient_digest or inputs["operator_sha256"] != c["operator_sha256"]
            or not source_from_record(inputs["source"], allow_cpp=True).matches(state.source)
            or type(inputs["instance_archive"]) is not bytes or not inputs["instance_archive"]
            or len(inputs["instance_archive"]) > c["max_bytes"]
            or sha256(inputs["instance_archive"]).hexdigest() != inputs["instance_sha256"]):
        raise ValueError("closed query for an explicitly admitted independent operator instance required")
    module = _binding(); authority = _context(module, execution_context)
    instance = module.NativeAvailableFiniteOperatorInstance.from_archive(inputs["instance_archive"], authority,
                                                                       limits=c["limits"], max_bytes=c["max_bytes"])
    if instance.operator_program(authority).archive() != operator.archive():
        raise ValueError("independent input instance changes the original operator")
    return _output(state, instance.execute(authority), authority)


def read_finite_operator_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if (not isinstance(output, ModelOutput) or output.method not in {OPERATOR_FIELD_ARCHIVE_METHOD, OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD}
            or output.space != _READOUT or output.axes or output.values.size or output.grade is not None or output.variance != "coordinate"):
        raise TypeError("complete native operator family or explicit operator choice required")
    output.check_state(); a, r = output.field_archive, output.execution_receipt
    selected = output.method == OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD
    schema = OPERATOR_EVALUATION_FIELD_ARCHIVE_SCHEMA if selected else OPERATOR_FIELD_ARCHIVE_SCHEMA
    required = {"profile", "limits", "max_bytes", "model_configuration", "reference_checkpoint", "actual_input", "interpretation", "dependency_count"}
    if (not isinstance(r, Mapping) or set(r) != required or r["profile"] != output.method
            or r["interpretation"] != ("explicit-prospective-operator-choice" if selected else "complete-native-operator-family")
            or r["dependency_count"] != len(output.dependencies)):
        raise ValueError("closed operator execution receipt required")
    c = r["model_configuration"]
    if (not isinstance(c, Mapping) or set(c) != _CONFIGURATION or c["limits"] != r["limits"] or c["max_bytes"] != r["max_bytes"]
            or c["operator_sha256"] != a["operator_sha256"] or a["schema"] != schema):
        raise ValueError("operator output declaration or envelope differs")
    admitted = _admission(r["limits"], r["max_bytes"]); maximum = r["max_bytes"]
    if limits is not None: admitted = tuple(min(x,y) for x,y in zip(admitted, _admission(limits,maximum), strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes); maximum = min(maximum,max_bytes)
    module = _binding(); authority = _context(module, execution_context); parent = _reference(r["reference_checkpoint"])
    _pins(module, c, parent)
    if dict(a["engine_receipt"]) != dict(c["engine_receipt"]):
        raise ValueError("operator output requires its pinned native implementation")
    if not parent.source.matches(output.source):
        raise ValueError("operator output original model source differs")
    reader = module.NativeAvailableFiniteOperatorEvaluation if selected else module.NativeAvailableFiniteOperatorResult
    value = reader.from_archive(a["data"], authority, limits=admitted, max_bytes=maximum)
    family = value.family(authority) if selected else value; instance = family.instance(authority)
    operator = instance.operator_program(authority)
    original_asset = operator.reference(authority).source(authority).raw().asset()
    source = module.NativeSource(original_asset.source_archive(), max_bytes=maximum,
                                 max_dimension=max(admitted[:2]), max_records=admitted[1])
    reconstructed = _state(module, operator, parent, source, c, authority)
    if reconstructed.coefficient_digest != output.model_digest:
        raise ValueError("operator output complete original checkpoint or model lineage differs")
    if state is not None:
        validate_finite_operator_state(state)
        if reconstructed.coefficient_digest != state.coefficient_digest:
            raise ValueError("operator output belongs to another immutable model")
    actual = freeze_tree(_actual(module, instance, authority, admitted, maximum))
    if actual != r["actual_input"]:
        raise ValueError("operator actual input source, transport, roles or instance lineage differs")
    deps = _dependencies(module, value if selected else instance, parent, authority, admitted)
    if len(deps) != len(output.dependencies) or any(not x.matches(y) for x,y in zip(deps, output.dependencies, strict=True)):
        raise ValueError("operator whole original and input dependency ordering differs")
    value.admit(authority)
    return value


def evaluate_finite_operator(output, ordinal, values, parameter_setting, choice_origin, *, execution_context, state):
    if output.method != OPERATOR_FIELD_ARCHIVE_METHOD:
        raise TypeError("complete unresolved operator family required")
    family = read_finite_operator_result(output, execution_context=execution_context, state=state)
    value = family.evaluate(ordinal, values, parameter_setting, choice_origin, execution_context)
    return _output(state, value, execution_context, selected=True)


def finite_operator_branches(output, begin, count, *, execution_context):
    value = read_finite_operator_result(output, execution_context=execution_context)
    family = value.family(execution_context) if output.method == OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD else value
    parent = _reference(output.execution_receipt["reference_checkpoint"])
    targets = _targets(parent.payload["target_checkpoints"], parent.configuration["target_digests"])
    rows = []
    for branch in family.select(begin, count, execution_context):
        info = branch.inspect(execution_context); raw = branch.raw().raw().result(); index = info["target"]
        rows.append(dict(info, target_source=None if index is None else targets[index].source.as_record(),
                         fields=() if raw is None else raw.fields(), origin_output=output.coefficient_digest,
                         operator_sha256=output.field_archive["operator_sha256"], input_instance_sha256=output.execution_receipt["actual_input"]["instance_sha256"]))
    return tuple(rows)


def finite_operator_info(output, *, execution_context):
    value = read_finite_operator_result(output, execution_context=execution_context)
    selected = output.method == OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD
    family = value.family(execution_context) if selected else value
    parent = _reference(output.execution_receipt["reference_checkpoint"])
    return {"operator_sha256": output.field_archive["operator_sha256"], "archive_sha256": output.field_archive["sha256"],
            "branch_count": family.inspect(execution_context)["branch_count"], "target_count": len(parent.configuration["target_digests"]),
            "selected_ordinal": value.inspect(execution_context)["ordinal"] if selected else None,
            "relationship_status": output.execution_receipt["model_configuration"]["status"],
            "original_model_source": output.source.as_record(), "actual_input": output.execution_receipt["actual_input"],
            "availability": "independent-native-context-admitted", "retains_complete_alternatives": True}


def finite_operator_stalks(output, node, cells, *, execution_context, coordinate_limit=256):
    if output.method != OPERATOR_EVALUATION_FIELD_ARCHIVE_METHOD:
        raise TypeError("operator stalk selection requires an explicit prospective completion")
    value = read_finite_operator_result(output, execution_context=execution_context)
    ordinal = value.inspect(execution_context)["ordinal"]
    branch = value.family(execution_context).select(ordinal, 1, execution_context)[0].inspect(execution_context)
    parent = _reference(output.execution_receipt["reference_checkpoint"])
    target = _targets(parent.payload["target_checkpoints"], parent.configuration["target_digests"])[branch["target"]]
    return tuple(dict(row, finite_ordinal=ordinal, target_source=target.source.as_record(),
                      actual_input_source=output.execution_receipt["actual_input"]["source"], origin_output=output.coefficient_digest,
                      operator_sha256=output.field_archive["operator_sha256"])
                 for row in value.raw().raw().result().select(node, cells, coordinate_limit=coordinate_limit))


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError("finite operator adapter required")
    return cpp_available_finite_operator_model(source, **configuration)


def enable_finite_operator_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", ((_ADAPTER, "1"),), _infer, create=_create, prepare=_prepare,
                                      context_operations=("create", "prepare", "infer"))
        register_model_provider(_provider)
    return _provider
