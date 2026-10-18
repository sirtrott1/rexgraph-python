"""Source bound native whole field programs and bounded result inspection."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

from rexgraph.field_archive import (
    FIELD_ARCHIVE_METHOD, FIELD_ARCHIVE_SCHEMA,
    ACCESSION_FIELD_ARCHIVE_METHOD, ACCESSION_FIELD_ARCHIVE_SCHEMA,
)
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_SCHEMA = "rexgraph.relational-field-program.v1"
_QUERY = "rexgraph.relational-field-program-query.v1"
_ADAPTER = "cpp_relational_field_program"
_PROVIDER = {"name": "rexgraph.cpp.relational_field_program", "version": "1"}
_ACCESSION_PROVIDER = {"name": _PROVIDER["name"], "version": "2"}
_ACCESSION_SCHEMA = "rexgraph.relational-accession-field-program.v1"
_ACCESSION_QUERY = "rexgraph.relational-accession-field-program-query.v1"
_LIMITS = (4096, 16000000, 64000000, 16384)
_lock = RLock()
_cache = OrderedDict()
_results = OrderedDict()
_provider = None
_accession_provider = None


def _binding():
    from rexgraph.cpp_model import _binding as binding
    module = binding()
    if not hasattr(module, "PreparedFieldProgram"):
        raise ImportError("this model requires the owning field program native provider")
    return module


def _admission(limits, maximum):
    limits = tuple(limits)
    if (len(limits) != 4 or any(type(v) is not int or not 0 < v < 1 << 64 for v in limits)
            or type(maximum) is not int or not 0 < maximum < 1 << 64):
        raise ValueError("positive field program action and archive admissions required")
    return limits


def _space(reference, key):
    return CoordinateSpace(f"cpp:{reference.native.namespace}:program:{key}:source-cells",
                           tuple(f"{grade}:{cell}" for grade, cells in enumerate(reference.native.coordinates) for cell in cells))


def _profile(version):
    if version == "1":
        return _SCHEMA, _PROVIDER, _QUERY, FIELD_ARCHIVE_SCHEMA, FIELD_ARCHIVE_METHOD
    if version == "2":
        return _ACCESSION_SCHEMA, _ACCESSION_PROVIDER, _ACCESSION_QUERY, ACCESSION_FIELD_ARCHIVE_SCHEMA, ACCESSION_FIELD_ARCHIVE_METHOD
    raise ValueError("unsupported whole-field program adapter version")


def _receipt(module, version):
    if version == "2" and not hasattr(module, "field_interface_build_info"):
        raise ImportError("actual source-use programs require their native interface provider")
    return module.field_program_build_info() if version == "1" else module.field_interface_build_info()


def validate_field_program_state(state):
    """Validate immutable declarations without claiming native program replay."""
    if state.adapter_version == "7":
        from rexgraph.cpp_relational_field_extension import validate_field_extension_state
        return validate_field_extension_state(state)
    if state.adapter_version == "6":
        from rexgraph.cpp_relational_field_evidence_admission import validate_available_field_learning_state
        return validate_available_field_learning_state(state)
    if state.adapter_version == "5":
        from rexgraph.cpp_relational_field_admission import validate_available_field_state
        return validate_available_field_state(state)
    if state.adapter_version == "3":
        from rexgraph.cpp_relational_field_learning import validate_field_learning_state
        return validate_field_learning_state(state)
    if state.adapter_version == "4":
        from rexgraph.cpp_relational_field_comparison import validate_field_comparison_state
        return validate_field_comparison_state(state)
    c = state.configuration
    profile, provider, _, _, _ = _profile(state.adapter_version)
    required = {"profile", "execution_provider", "engine_receipt", "program_archive", "program_sha256",
                "key", "revision", "limits", "max_bytes"}
    if (state.adapter != _ADAPTER or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or set(c) != required
            or c["profile"] != profile or dict(c["execution_provider"]) != provider
            or state.payload or state.step != 0 or state.parent is not None or state.output_axes or state.dependencies):
        raise ValueError("closed fixed whole-field program model required")
    _admission(c["limits"], c["max_bytes"])
    if (type(c["program_archive"]) is not bytes or not c["program_archive"]
            or len(c["program_archive"]) > c["max_bytes"]
            or sha256(c["program_archive"]).hexdigest() != c["program_sha256"]
            or type(c["key"]) is not int or not 0 < c["key"] < 1 << 64
            or type(c["revision"]) is not int or not 0 <= c["revision"] < 1 << 64
            or not isinstance(c["engine_receipt"], Mapping) or not c["engine_receipt"]
            or state.space != _space(state.source, c["key"])):
        raise ValueError("field program bytes, identity or original coordinate order differ")


def _publication(program, reference):
    publication = program.publication()
    expected = {"namespace": reference.native.namespace, "sequence": reference.native.sequence,
                "endpoint_digest": reference.native.endpoint_digest, "coordinates": reference.native.coordinates}
    if any(publication[name] != value for name, value in expected.items()):
        raise ValueError("field program uses another source namespace, cutoff, endpoint or coordinate order")
    return publication


def _prepare(state, source):
    validate_field_program_state(state)
    state.bind(source)
    module = _binding()
    c = state.configuration
    if dict(c["engine_receipt"]) != _receipt(module, state.adapter_version):
        raise ValueError("field program model requires a different native implementation")
    key = (id(source), state.coefficient_digest)
    with _lock:
        known = _cache.get(key)
        if known is not None and known[0] is source:
            _cache.move_to_end(key)
            return known[1]
        program = (module.PreparedFieldProgram.from_archive(c["program_archive"], limits=c["limits"], max_bytes=c["max_bytes"])
                   if state.adapter_version == "1" else
                   module.PreparedAccessionFieldProgram(c["program_archive"], limits=c["limits"], max_bytes=c["max_bytes"]))
        _publication(program, state.source)
        identity = program.identity()
        if identity["key"] != c["key"] or identity["revision"] != c["revision"]:
            raise ValueError("field program declaration identity differs")
        _cache[key] = (source, program)
        _cache.move_to_end(key)
        while len(_cache) > 4:
            _cache.popitem(last=False)
        return program


def cpp_relational_field_program_model(source, declaration=None, *, program_archive=None,
                                      limits=_LIMITS, max_bytes=256 * 1024 * 1024, reference=None):
    """Declare a fixed ordered native program on an authentic source publication."""
    return _make_model(source, declaration, program_archive, limits, max_bytes, reference, "1")


def cpp_accession_field_program_model(source, declaration=None, *, program_archive=None,
                                     limits=_LIMITS, max_bytes=256 * 1024 * 1024, reference=None):
    """Declare a program whose inputs retain actual original assets and faithful maps."""
    return _make_model(source, declaration, program_archive, limits, max_bytes, reference, "2")


def _make_model(source, declaration, program_archive, limits, max_bytes, reference, version):
    module = _binding()
    profile, provider, _, _, _ = _profile(version)
    implementation = _receipt(module, version)
    if not isinstance(source, module.NativeSource):
        raise TypeError("field program model requires an owning NativeSource")
    limits = _admission(limits, max_bytes)
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    if not isinstance(ref, CppSourceReference) or (declaration is None) == (program_archive is None):
        raise ValueError("declare one native program or supply its canonical archive")
    program = (module.PreparedFieldProgram(source.archive, declaration, limits=limits, max_bytes=max_bytes)
               if declaration is not None else module.PreparedFieldProgram.from_archive(program_archive, limits=limits, max_bytes=max_bytes))
    _publication(program, ref)
    image = program.archive()
    identity = program.identity()
    if version == "2":
        program = module.PreparedAccessionFieldProgram(image, limits=limits, max_bytes=max_bytes)
    configuration = {"profile": profile, "execution_provider": provider,
                     "engine_receipt": implementation, "program_archive": image,
                     "program_sha256": sha256(image).hexdigest(), "key": identity["key"],
                     "revision": identity["revision"], "limits": limits, "max_bytes": max_bytes}
    state = ModelState(_ADAPTER, version, ref, _space(ref, identity["key"]), (), configuration, {}, "rational")
    (enable_field_program_provider if version == "1" else enable_accession_field_program_provider)()
    with _lock:
        _cache[(id(source), state.coefficient_digest)] = (source, program)
        while len(_cache) > 4:
            _cache.popitem(last=False)
    return state


def field_program_query(state, inputs):
    """Capture source bound input uses with native complete support tensor bytes."""
    validate_field_program_state(state)
    if state.adapter_version == "7":
        from rexgraph.cpp_relational_field_extension import field_extension_query
        return field_extension_query(state, inputs)
    if state.adapter_version == "6":
        from rexgraph.cpp_relational_field_evidence_admission import available_field_learning_query
        return available_field_learning_query(state, inputs)
    if state.adapter_version == "5":
        from rexgraph.cpp_relational_field_admission import available_field_query
        return available_field_query(state, inputs)
    if state.adapter_version == "3":
        from rexgraph.cpp_relational_field_learning import field_learning_query
        return field_learning_query(state, inputs)
    if state.adapter_version == "4":
        from rexgraph.cpp_relational_field_comparison import field_comparison_query
        return field_comparison_query(state, inputs)
    if state.adapter_version == "2":
        return accession_field_program_query(state, inputs)
    rows = []
    for value in inputs:
        if (not isinstance(value, Mapping) or set(value) != {"node", "use", "field"}
                or type(value["node"]) is not int or not 0 < value["node"] < 1 << 64
                or type(value["use"]) is not int or not 0 < value["use"] < 1 << 64
                or type(value["field"]) is not bytes):
            raise ValueError("original input node/use and immutable sparse field bytes required")
        rows.append(dict(value))
    return freeze_tree({"profile": _QUERY, "model_digest": state.coefficient_digest,
                        "source": state.source.as_record(), "inputs": tuple(rows)})


def accession_field_program_query(state, inputs):
    """Capture actual original asset archives, complete accessions and source fields."""
    validate_field_program_state(state)
    if state.adapter_version != "2":
        raise ValueError("source-backed accession program model required")
    rows = []
    for value in inputs:
        if (not isinstance(value, Mapping) or set(value) != {"asset", "accession", "use", "field"}
                or type(value["asset"]) is not bytes or not value["asset"]
                or not isinstance(value["accession"], Mapping)
                or type(value["field"]) is not bytes or not value["field"]
                or type(value["use"]) is not int or not 0 < value["use"] < 1 << 64):
            raise ValueError("closed actual input asset/accession/use/native-field declaration required")
        rows.append(dict(value))
    return freeze_tree({"profile": _ACCESSION_QUERY, "model_digest": state.coefficient_digest,
                       "source": state.source.as_record(), "inputs": tuple(rows)})


def _source_dependencies(module, result, count, limits, maximum):
    return tuple(CppSourceReference(module.NativeSource(
        module.NativeInputUseAsset.from_archive(value["asset"], limits=limits, max_bytes=maximum).source_archive(),
        max_dimension=max(limits[:2]), max_records=limits[1]))
        for value in result.input_uses(max_records=count))


def _infer(state, source, inputs):
    program = _prepare(state, source)
    _, _, query_profile, archive_schema, method = _profile(state.adapter_version)
    if not isinstance(inputs, Mapping) or set(inputs) != {"profile", "model_digest", "source", "inputs"}:
        raise ValueError("closed source-bound field program query required")
    if (inputs["profile"] != query_profile or inputs["model_digest"] != state.coefficient_digest
            or not source_from_record(inputs["source"], allow_cpp=True).matches(state.source)):
        raise ValueError("field program query uses another model or source publication")
    query = field_program_query(state, inputs["inputs"])
    if state.adapter_version == "1":
        result = program.execute(query["inputs"])
        dependencies = ()
    else:
        module = _binding(); c = state.configuration
        native_inputs = []
        for value in query["inputs"]:
            asset = module.NativeInputUseAsset.from_archive(value["asset"], limits=c["limits"], max_bytes=c["max_bytes"])
            accession = module.NativeInputAccession(asset, program, value["accession"], limits=c["limits"], max_bytes=c["max_bytes"])
            native_inputs.append(accession.admit(value["use"], value["field"]))
        result = program.execute(native_inputs)
        dependencies = _source_dependencies(module, result, len(native_inputs), c["limits"], c["max_bytes"])
    image = result.archive()
    c = state.configuration
    archive = {"schema": archive_schema, "data": image, "sha256": sha256(image).hexdigest(),
               "program_sha256": c["program_sha256"], "engine_receipt": c["engine_receipt"]}
    key = (archive_schema, archive["sha256"], tuple(c["limits"]), c["max_bytes"])
    with _lock:
        _results[key] = result
        _results.move_to_end(key)
        while len(_results) > 4:
            _results.popitem(last=False)
    receipt = {"profile": method, "limits": c["limits"], "max_bytes": c["max_bytes"],
               "interpretation": "prospective-complete-native-field-response"}
    if state.adapter_version == "2":
        receipt["input_count"] = len(dependencies)
    return ModelOutput([], CoordinateSpace("native-program-field-readout", ()), (), state.source,
                       state.coefficient_digest, "rational", method, dependencies=dependencies,
                       execution_receipt=receipt, field_archive=archive)


def read_field_program_result(output, *, state=None, limits=None, max_bytes=None, execution_context=None):
    """Replay whole native fields independently of parent model loading."""
    if isinstance(output, ModelOutput) and output.method == "exact-conditioned-occurrence-program-v1":
        from rexgraph.cpp_conditioned_occurrence_program import read_conditioned_occurrence_result
        return read_conditioned_occurrence_result(output, state=state, execution_context=execution_context, limits=limits, max_bytes=max_bytes)
    if isinstance(output, ModelOutput) and output.method == "exact-conditioned-joint-field-family-v1":
        from rexgraph.cpp_conditioned_field_family import read_conditioned_family_result
        return read_conditioned_family_result(output, state=state, execution_context=execution_context, limits=limits, max_bytes=max_bytes)
    from rexgraph.field_archive import CONDITIONED_FIELD_ARCHIVE_METHOD
    if isinstance(output, ModelOutput) and output.method == CONDITIONED_FIELD_ARCHIVE_METHOD:
        from rexgraph.cpp_conditioned_field_program import read_conditioned_field_result
        return read_conditioned_field_result(output, state=state, execution_context=execution_context, limits=limits, max_bytes=max_bytes)
    from rexgraph.field_archive import AFFINE_FIELD_ARCHIVE_METHOD, EVALUATION_FIELD_ARCHIVE_METHOD
    from rexgraph.field_archive import COMPARISON_FIELD_ARCHIVE_METHOD
    from rexgraph.field_archive import AVAILABLE_FIELD_ARCHIVE_METHOD
    from rexgraph.field_archive import AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD, AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD
    from rexgraph.field_archive import EXTENSION_FIELD_ARCHIVE_METHOD, EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD
    if isinstance(output, ModelOutput) and output.method in {EXTENSION_FIELD_ARCHIVE_METHOD, EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD}:
        from rexgraph.cpp_relational_field_extension import read_field_extension_result
        return read_field_extension_result(output, state=state, limits=limits, max_bytes=max_bytes, execution_context=execution_context)
    from rexgraph.field_archive import LOCAL_FIELD_ARCHIVE_METHOD
    if isinstance(output, ModelOutput) and output.method == LOCAL_FIELD_ARCHIVE_METHOD:
        from rexgraph.cpp_relational_field_local import read_local_field_result
        return read_local_field_result(output,state=state,limits=limits,max_bytes=max_bytes,execution_context=execution_context)
    if isinstance(output, ModelOutput) and output.method in {AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD, AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD}:
        from rexgraph.cpp_relational_field_evidence_admission import read_available_field_learning_result
        return read_available_field_learning_result(output, state=state, limits=limits, max_bytes=max_bytes,
                                                   execution_context=execution_context)
    if isinstance(output, ModelOutput) and output.method == AVAILABLE_FIELD_ARCHIVE_METHOD:
        from rexgraph.cpp_relational_field_admission import read_available_field_result
        return read_available_field_result(output, state=state, limits=limits, max_bytes=max_bytes,
                                           execution_context=execution_context)
    if execution_context is not None:
        raise TypeError("field result profile does not admit an execution context")
    if isinstance(output, ModelOutput) and output.method == COMPARISON_FIELD_ARCHIVE_METHOD:
        from rexgraph.cpp_relational_field_comparison import read_field_comparison_result
        return read_field_comparison_result(output, state=state, limits=limits, max_bytes=max_bytes)
    if isinstance(output, ModelOutput) and output.method in {AFFINE_FIELD_ARCHIVE_METHOD, EVALUATION_FIELD_ARCHIVE_METHOD}:
        from rexgraph.cpp_relational_field_learning import read_field_learning_result
        return read_field_learning_result(output, state=state, limits=limits, max_bytes=max_bytes)
    if (not isinstance(output, ModelOutput) or output.method not in {FIELD_ARCHIVE_METHOD, ACCESSION_FIELD_ARCHIVE_METHOD}
            or output.space != CoordinateSpace("native-program-field-readout", ())
            or output.axes or output.values.size or output.grade is not None
            or output.variance != "coordinate"):
        raise TypeError("whole-field program output required")
    output.check_state()
    archive, receipt = output.field_archive, output.execution_receipt
    version = "1" if output.method == FIELD_ARCHIVE_METHOD else "2"
    _, _, _, schema, method = _profile(version)
    expected = {"profile", "limits", "max_bytes", "interpretation"}
    if version == "2": expected.add("input_count")
    if (not isinstance(receipt, Mapping) or set(receipt) != expected or receipt["profile"] != method
            or receipt["interpretation"] != "prospective-complete-native-field-response"):
        raise ValueError("closed field program result receipt required")
    admitted = _admission(receipt["limits"], receipt["max_bytes"])
    if limits is not None:
        requested = _admission(limits, receipt["max_bytes"])
        admitted = tuple(min(a, b) for a, b in zip(admitted, requested, strict=True))
    maximum = receipt["max_bytes"]
    if max_bytes is not None:
        _admission(admitted, max_bytes)
        maximum = min(max_bytes, maximum)
    if len(archive["data"]) > maximum:
        raise ValueError("field program result exceeds its archive byte admission")
    module = _binding()
    if archive["schema"] != schema or dict(archive["engine_receipt"]) != _receipt(module, version):
        raise ValueError("field program result requires a different native implementation")
    if state is not None:
        validate_field_program_state(state)
        state.check_state()
        if (output.model_digest != state.coefficient_digest or not output.source.matches(state.source)
                or archive["program_sha256"] != state.configuration["program_sha256"]):
            raise ValueError("whole-field result uses another declared model or source")
    key = (schema, archive["sha256"], admitted, maximum)
    with _lock:
        result = _results.get(key)
        if result is None:
            reader = module.NativeFieldProgramResult if version == "1" else module.NativeAccessionFieldResult
            result = reader.from_archive(archive["data"], limits=admitted, max_bytes=maximum)
            _results[key] = result
            while len(_results) > 4:
                _results.popitem(last=False)
        else:
            _results.move_to_end(key)
    program = result.program()
    if sha256(program.archive()).hexdigest() != archive["program_sha256"]:
        raise ValueError("field result canonical program differs from its envelope")
    _publication(program, output.source)
    if version == "1":
        if output.dependencies: raise ValueError("initial program declares no external input dependency profile")
    else:
        count = receipt["input_count"]
        if type(count) is not int or count < 0 or count != len(output.dependencies):
            raise ValueError("source-backed result dependency count differs")
        actual = _source_dependencies(module, result, count, admitted, maximum)
        if any(not a.matches(b) for a, b in zip(actual, output.dependencies, strict=True)):
            raise ValueError("source-backed result actual input publications differ")
    return result


def field_program_stalks(output, node, cells, *, coordinate_limit=256, state=None, execution_context=None):
    """Inspect complete selected stalks with actual basis and metric declarations."""
    result = read_field_program_result(output, state=state, execution_context=execution_context)
    from rexgraph.field_archive import AVAILABLE_FIELD_ARCHIVE_METHOD
    from rexgraph.field_archive import AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD, AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD
    if output.method in {"exact-conditioned-occurrence-program-v1", "exact-conditioned-field-program-v1", AVAILABLE_FIELD_ARCHIVE_METHOD, AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD, AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD}:
        result = result.result()
    return result.select(node, cells, coordinate_limit=coordinate_limit)


def clear_field_program_cache():
    with _lock:
        _cache.clear(); _results.clear()


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER:
        raise ValueError("whole-field program adapter required")
    return cpp_relational_field_program_model(source, **configuration)


def enable_field_program_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", ((_ADAPTER, "1"),), _infer,
                                      create=_create, prepare=_prepare)
        register_model_provider(_provider)
        return _provider


def _create_accession(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError("whole-field program adapter required")
    return cpp_accession_field_program_model(source, **configuration)


def enable_accession_field_program_provider():
    global _accession_provider
    module = _binding()
    if not hasattr(module, "PreparedAccessionFieldProgram"):
        raise ImportError("actual source-use programs require their native interface provider")
    with _lock:
        if _accession_provider is None:
            _accession_provider = ModelProvider(_ACCESSION_PROVIDER["name"], "2", ((_ADAPTER, "2"),), _infer,
                                                create=_create_accession, prepare=_prepare)
        register_model_provider(_accession_provider)
        return _accession_provider
