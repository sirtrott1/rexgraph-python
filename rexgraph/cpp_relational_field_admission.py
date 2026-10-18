"""Whole native field execution with separately supplied value authority."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

from rexgraph.cpp_relational_field_program import (
    _ADAPTER, _LIMITS, _admission, _binding as _program_binding, _publication,
    _space, _source_dependencies,
)
from rexgraph.field_archive import AVAILABLE_FIELD_ARCHIVE_METHOD, AVAILABLE_FIELD_ARCHIVE_SCHEMA
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "rexgraph.available-relational-field-program.v1"
_QUERY = "rexgraph.available-relational-field-query.v1"
_PROVIDER = {"name": "rexgraph.cpp.relational_field_program", "version": "5"}
_lock = RLock()
_cache = OrderedDict()
_provider = None


def _binding():
    module = _program_binding()
    if not hasattr(module, "PreparedAvailableFieldProgram"):
        raise ImportError("available whole fields require the native value admission provider")
    return module


def _context(module, value):
    if not isinstance(value, module.NativeAvailabilityContext):
        raise TypeError("separately supplied actual native availability context required")
    return value


def _dependencies(module, result, limits):
    return tuple(CppSourceReference(module.NativeSource(image, max_dimension=max(limits[:2]), max_records=limits[1]))
                 for image in result.dependency_archives(max_records=limits[1]))


def validate_available_field_state(state):
    c = state.configuration
    required = {"profile", "execution_provider", "engine_receipt", "available_archive", "available_sha256",
                "program_sha256", "key", "revision", "limits", "max_bytes"}
    if (state.adapter != _ADAPTER or state.adapter_version != "5" or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or set(c) != required
            or c["profile"] != _PROFILE or dict(c["execution_provider"]) != _PROVIDER
            or state.payload or state.output_axes or state.step or state.parent is not None
            or state.space != _space(state.source, c["key"])):
        raise ValueError("closed available whole-field model declaration required")
    _admission(c["limits"], c["max_bytes"])
    if (type(c["available_archive"]) is not bytes or not c["available_archive"]
            or len(c["available_archive"]) > c["max_bytes"]
            or sha256(c["available_archive"]).hexdigest() != c["available_sha256"]
            or not isinstance(c["engine_receipt"], Mapping)
            or type(c["key"]) is not int or not 0 < c["key"] < 1 << 64
            or type(c["revision"]) is not int or not 0 <= c["revision"] < 1 << 64
            or type(c["program_sha256"]) is not str or len(c["program_sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in c["program_sha256"])
            or len(state.dependencies) != 3):
        raise ValueError("available program archive, identity or origin dependencies differ")


def cpp_available_field_program_model(source, *, available_program, execution_context,
                                     limits=_LIMITS, max_bytes=256*1024*1024, reference=None):
    """Retain a checked native program and its complete publisher obligations."""
    module = _binding(); authority = _context(module, execution_context); limits = _admission(limits, max_bytes)
    if not isinstance(source, module.NativeSource): raise TypeError("owning native source required")
    if not isinstance(available_program, module.PreparedAvailableFieldProgram):
        raise TypeError("owning native available program required")
    available_program.admit(authority)
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    program = available_program.program(); _publication(program, ref); identity = program.identity()
    image = available_program.archive()
    c = {"profile": _PROFILE, "execution_provider": _PROVIDER, "engine_receipt": module.field_admission_build_info(),
         "available_archive": image, "available_sha256": sha256(image).hexdigest(),
         "program_sha256": sha256(program.archive()).hexdigest(), "key": identity["key"], "revision": identity["revision"],
         "limits": limits, "max_bytes": max_bytes}
    state = ModelState(_ADAPTER, "5", ref, _space(ref, identity["key"]), (), c, {}, "rational",
                       dependencies=_dependencies(module, available_program, limits))
    enable_available_field_program_provider()
    _prepare(state, source, execution_context=authority)
    return state


def _prepare(state, source, *, execution_context):
    validate_available_field_state(state); state.bind(source)
    module = _binding(); authority = _context(module, execution_context); c = state.configuration
    if dict(c["engine_receipt"]) != module.field_admission_build_info():
        raise ValueError("available whole-field model requires its pinned native implementation")
    key = (id(source), state.coefficient_digest, id(authority))
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] is source and cached[1] is authority:
            program = cached[2]; _cache.move_to_end(key)
        else:
            program = None
    if program is None:
        program = module.PreparedAvailableFieldProgram.from_archive(c["available_archive"], authority,
                    limits=c["limits"], max_bytes=c["max_bytes"])
    program.admit(authority)
    actual = program.program(); _publication(actual, state.source); identity = actual.identity()
    if (identity["key"] != c["key"] or identity["revision"] != c["revision"]
            or sha256(actual.archive()).hexdigest() != c["program_sha256"]
            or sha256(program.archive()).hexdigest() != c["available_sha256"]):
        raise ValueError("available native program identity or complete origin receipts differ")
    deps = _dependencies(module, program, c["limits"])
    if len(deps) != len(state.dependencies) or any(not a.matches(b) for a,b in zip(deps,state.dependencies,strict=True)):
        raise ValueError("available program authentic origin histories differ")
    with _lock:
        _cache[key] = (source, authority, program)
        while len(_cache) > 4: _cache.popitem(last=False)
    return program


def available_field_query(state, inputs):
    """Retain closed input declarations without publishing their value origins."""
    validate_available_field_state(state); rows = []
    for value in inputs:
        if (not isinstance(value, Mapping) or set(value) != {"asset", "accession", "use", "field", "origins"}
                or not isinstance(value["accession"], Mapping) or not isinstance(value["origins"], Mapping)
                or set(value["origins"]) != {"asset", "value", "accession", "reconstruction"}
                or any(type(value[k]) is not bytes or not value[k] for k in ("asset", "field"))
                or type(value["use"]) is not int or not 0 < value["use"] < 1 << 64
                or any(type(v) is not int or not 0 < v < 1 << 64 for v in value["origins"].values())):
            raise ValueError("complete native input and actual immutable origin keys required")
        rows.append(dict(value))
    return freeze_tree({"profile": _QUERY, "model_digest": state.coefficient_digest,
                       "source": state.source.as_record(), "inputs": tuple(rows)})


def _infer(state, source, inputs, *, execution_context):
    program = _prepare(state, source, execution_context=execution_context)
    if (not isinstance(inputs, Mapping) or set(inputs) != {"profile", "model_digest", "source", "inputs"}
            or inputs["profile"] != _QUERY or inputs["model_digest"] != state.coefficient_digest
            or not source_from_record(inputs["source"], allow_cpp=True).matches(state.source)):
        raise ValueError("available field query uses another model or source")
    query = available_field_query(state, inputs["inputs"])
    result = program.execute_inputs(query["inputs"], execution_context); c = state.configuration; module = _binding()
    image = result.archive(); raw = result.result()
    dependencies = _dependencies(module, result, c["limits"]) + _source_dependencies(
        module, raw, len(raw.input_uses(max_records=c["limits"][1])), c["limits"], c["max_bytes"])
    archive = {"schema": AVAILABLE_FIELD_ARCHIVE_SCHEMA, "data": image, "sha256": sha256(image).hexdigest(),
               "program_sha256": c["program_sha256"], "engine_receipt": c["engine_receipt"]}
    receipt = {"profile": AVAILABLE_FIELD_ARCHIVE_METHOD, "limits": c["limits"], "max_bytes": c["max_bytes"],
               "available_program_sha256": c["available_sha256"], "dependency_count": len(dependencies),
               "interpretation": "prospective-complete-native-field-response"}
    return ModelOutput([], CoordinateSpace("native-program-field-readout", ()), (), state.source, state.coefficient_digest,
                       "rational", AVAILABLE_FIELD_ARCHIVE_METHOD, dependencies=dependencies,
                       execution_receipt=receipt, field_archive=archive)


def read_available_field_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    """Replay complete fields under independently supplied current authority."""
    if (not isinstance(output, ModelOutput) or output.method != AVAILABLE_FIELD_ARCHIVE_METHOD
            or output.space != CoordinateSpace("native-program-field-readout", ()) or output.axes or output.values.size
            or output.grade is not None or output.variance != "coordinate" or output.arithmetic != "rational"):
        raise TypeError("available whole-field output required")
    output.check_state(); module = _binding(); authority = _context(module, execution_context)
    a, r = output.field_archive, output.execution_receipt
    if (a["schema"] != AVAILABLE_FIELD_ARCHIVE_SCHEMA or not isinstance(r, Mapping)
            or set(r) != {"profile", "limits", "max_bytes", "available_program_sha256", "dependency_count", "interpretation"}
            or r["profile"] != AVAILABLE_FIELD_ARCHIVE_METHOD or dict(a["engine_receipt"]) != module.field_admission_build_info()
            or r["interpretation"] != "prospective-complete-native-field-response"):
        raise ValueError("available field output admission receipt differs")
    admitted = _admission(r["limits"] if limits is None else limits, r["max_bytes"] if max_bytes is None else max_bytes)
    maximum = r["max_bytes"] if max_bytes is None else max_bytes
    result = module.NativeAvailableFieldResult.from_archive(a["data"], authority, limits=admitted, max_bytes=maximum)
    raw = result.result(); program = raw.program(); _publication(program, output.source)
    available_sha = sha256(result.program_archive(authority)).hexdigest()
    if sha256(program.archive()).hexdigest() != a["program_sha256"] or available_sha != r["available_program_sha256"]:
        raise ValueError("available result's actual program or publisher receipts differ")
    deps = _dependencies(module, result, admitted) + _source_dependencies(
        module, raw, len(raw.input_uses(max_records=admitted[1])), admitted, maximum)
    if (type(r["dependency_count"]) is not int or r["dependency_count"] != len(deps) or len(deps) != len(output.dependencies)
            or any(not x.matches(y) for x,y in zip(deps,output.dependencies,strict=True))):
        raise ValueError("available result authentic value/input histories differ")
    if state is not None:
        validate_available_field_state(state)
        if (output.model_digest != state.coefficient_digest or not output.source.matches(state.source)
                or available_sha != state.configuration["available_sha256"]
                or a["program_sha256"] != state.configuration["program_sha256"]):
            raise ValueError("available result targets another model or origin closure")
    return result


def inspect_available_field_model(state, source, *, execution_context):
    return _prepare(state, source, execution_context=execution_context).program().inspect(max_records=state.configuration["limits"][1])


def clear_available_field_cache():
    with _lock: _cache.clear()


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError("available whole-field adapter required")
    return cpp_available_field_program_model(source, **configuration)


def enable_available_field_program_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "5", ((_ADAPTER, "5"),), _infer,
                                     create=_create, prepare=_prepare, context_operations=("create", "prepare", "infer"))
            register_model_provider(_provider)
    return _provider
