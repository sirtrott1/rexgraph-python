"""Exact algebraic archive actions with declared module and polynomial coordinates."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from numbers import Integral
from threading import RLock

import numpy as np

from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference, _sha, source_from_record
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value

__all__ = ["cpp_algebraic_model", "cpp_algebraic_input", "enable_cpp_algebraic_provider"]
_profile = "algebraic-archive-action-v1"
_graph_profile = "algebraic-graph-archive-action-v1"
_input_profile = "declared-algebraic-vector-v1"
_origin_input_profile = "declared-algebraic-origin-vector-v1"
_origin_profiles = {_profile: "algebraic-origin-archive-action-v1",
                    _graph_profile: "algebraic-origin-graph-archive-action-v1"}
_operations = frozenset({"apply", "solve", "evaluate", "quadrance", "as_covector",
                         "accession", "reconstruct", "project"})
_pairs = frozenset({"accession", "reconstruct", "project"})
_limit_names = frozenset({
    "max_file_bytes", "max_records", "max_rational_coordinates", "max_integer_bytes",
    "field.max_degree", "field.max_input_coefficients", "field.max_integer_bits",
    "field.max_scalar_steps", "field.max_matrix_coordinates", "modules.max_cells",
    "modules.max_rank", "modules.max_retained_coordinates", "modules.max_matrix_coordinates",
    "fibers.max_occurrences", "fibers.max_retained_map_coordinates", "real_root.max_bisections",
    "complex_root.max_refinements", "complex_root.max_rounding_bits", "complex_root.max_trace_coordinates",
    "metric.max_sign_refinements", "metric.max_factor_steps"})
_graph_limit_names = frozenset("archive." + name for name in _limit_names) | {
    "max_file_bytes", "max_owner_records", "max_edges", "max_rational_coordinates",
    "measurement.max_coefficient_products"}
_provider = None
_lock = RLock()
_prepared = OrderedDict()
_cache_bytes = 64 << 20


def _binding():
    from rexgraph.cpp_model import _binding as load
    return load()


def _key(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 1 << 64:
        raise ValueError("algebraic record keys require unsigned 64 bit integers")
    return int(value)


def _carrier(reference, value):
    if not isinstance(value, Mapping) or set(value) != {"grade", "index", "rank", "field_degree"}:
        raise ValueError("algebraic module requires its grade, index, rank and field degree")
    result = {name: _key(v) for name, v in value.items()}
    n = reference.native
    if (result["grade"] >= len(n.source_sizes) or result["index"] >= n.source_sizes[result["grade"]]
            or result["field_degree"] != n.field_degree):
        raise ValueError("algebraic module differs from the declared source and field")
    return result


def _space(reference, carrier):
    if carrier is None:
        return CoordinateSpace(f"cpp-algebraic:{reference.native.coordinate_scope}:scalar", ("value",))
    c = _carrier(reference, carrier)
    return CoordinateSpace(f"cpp-algebraic:{reference.native.coordinate_scope}:module:{c['grade']}:{c['index']}",
                           tuple(str(i) for i in range(c["rank"])))


def _polynomial_axis(reference):
    return CoordinateSpace(f"cpp-algebraic:{reference.native.coordinate_scope}:polynomial",
                           tuple(str(i) for i in range(reference.native.field_degree)))


def validate_algebraic_state(state):
    """Check portable field and module declarations without loading the native extension."""
    if state.adapter == "cpp_algebraic_grade_metric":
        from rexgraph.cpp_algebraic_grade_metric_model import validate_grade_metric_model_state
        return validate_grade_metric_model_state(state)
    if state.adapter == "cpp_algebraic_grade":
        from rexgraph.cpp_algebraic_grade_model import validate_grade_model_state
        return validate_grade_model_state(state)
    if not isinstance(state.source, CppAlgebraicSourceReference):
        raise TypeError("algebraic model requires an algebraic archive source reference")
    c = state.configuration
    names = {"execution_provider", "engine_receipt", "profile", "input_profile", "operator_key",
             "input_key", "input_side", "operation", "limits", "input_carrier", "output_carrier", "output_kind"}
    profile = _profile
    if state.source.native.graph_digest is not None:
        names.add("graph_limits")
        profile = _graph_profile
    if (state.adapter != "cpp_algebraic" or state.adapter_version != "1" or set(c) != names
            or c["profile"] != profile or c["input_profile"] != _input_profile
            or dict(c["execution_provider"]) != {"name": "rexgraph.cpp.algebraic", "version": "1"}):
        raise ValueError("unsupported algebraic model declaration")
    _key(c["operator_key"]); _key(c["input_key"])
    if c["operation"] not in _operations or c["input_side"] not in {"source", "target"}:
        raise ValueError("unsupported algebraic action or input carrier side")
    carrier = _carrier(state.source, c["input_carrier"])
    if (len(state.space.keys) != carrier["rank"] or len(state.output_axes) != 1
            or len(state.output_axes[0].keys) != state.source.native.field_degree):
        raise ValueError("algebraic model dimensions differ from its retained coordinates")
    if state.space != _space(state.source, carrier) or state.output_axes != (_polynomial_axis(state.source),):
        raise ValueError("algebraic model axes differ from its declared module and coefficient basis")
    _output_kind(state.source, c["output_kind"], c["output_carrier"])
    _portable_limits(c["limits"])
    if profile == _graph_profile:
        _portable_limits(c["graph_limits"], graph=True)
        effective = _archive_graph_limits(c["graph_limits"])
        if any(c["limits"][name] < effective[name] for name in _limit_names):
            raise ValueError("algebraic graph archive admission exceeds its retained limits")
    for module in (carrier, c["output_carrier"]):
        if module is not None and module["rank"] > c["limits"]["modules.max_rank"]:
            raise ValueError("algebraic model module rank exceeds its retained native admission")
    if state.source.native.field_degree > c["limits"]["field.max_degree"]:
        raise ValueError("algebraic model field degree exceeds its retained native admission")
    if state.payload or state.step != 0 or state.parent is not None or state.dependencies:
        raise ValueError("fixed algebraic actions have no training or continuation payload")


def _portable_limits(limits, *, graph=False):
    if not isinstance(limits, Mapping) or set(limits) != (_graph_limit_names if graph else _limit_names):
        raise ValueError("algebraic model requires complete native admission limits")
    for value in limits.values():
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= np.iinfo(np.uintp).max:
            raise ValueError("invalid algebraic model admission limit")


def _archive_graph_limits(limits):
    result = {name: limits["archive." + name] for name in _limit_names}
    result["max_file_bytes"] = limits["max_file_bytes"]
    result["max_rational_coordinates"] = limits["max_rational_coordinates"]
    return result


def _output_kind(reference, kind, carrier):
    if kind == "scalar":
        if carrier is not None:
            raise ValueError("algebraic scalar has no module carrier")
        return 1, None, ()
    if kind not in {"vector", "covector"}:
        raise ValueError("algebraic action output must be scalar, vector or covector")
    c = _carrier(reference, carrier)
    shape = (c["rank"], 1) if kind == "vector" else (1, c["rank"])
    return c["rank"], c["grade"], shape


def _output_declaration(reference, kind, carrier):
    _, grade, shape = _output_kind(reference, kind, carrier)
    return _space(reference, carrier), grade, shape


def validate_algebraic_output(output):
    """Validate the exact field, coefficient basis and retained native value kind."""
    if output.method == "algebraic-grade-metric-action-v1":
        from rexgraph.cpp_algebraic_grade_metric_model import validate_grade_metric_model_output
        return validate_grade_metric_model_output(output)
    if output.method == "algebraic-grade-metric-stalk-v1":
        from rexgraph.cpp_algebraic_metric_stalk import validate_metric_stalk_output
        return validate_metric_stalk_output(output)
    if output.method == "algebraic-grade-stalk-v1":
        from rexgraph.cpp_algebraic_stalk import validate_stalk_output
        return validate_stalk_output(output)
    if output.method == "algebraic-grade-boundary-v1":
        from rexgraph.cpp_algebraic_grade_model import validate_grade_model_output
        return validate_grade_model_output(output)
    if not isinstance(output.source, CppAlgebraicSourceReference):
        raise TypeError("algebraic output requires an algebraic archive source reference")
    if output.method == "algebraic-graph-comparison-v1":
        from rexgraph.cpp_algebraic_graph import validate_comparison_output
        return validate_comparison_output(output)
    r = output.execution_receipt
    names = {"profile", "engine", "archive_digest", "source_digest", "field_degree", "minimal_polynomial",
             "coefficient_basis", "kind", "native_shape", "input_carrier", "output_carrier", "operator_key",
             "operation", "input_digest", "limits", "declared_limits"}
    profile = _profile
    if output.source.native.graph_digest is not None:
        profile = _graph_profile
        names |= {"graph_digest", "archive_key", "graph_limits", "declared_graph_limits"}
    has_origin = output.method in _origin_profiles.values()
    if has_origin:
        profile = _origin_profiles[profile]
        names |= {"input_origin", "origin_depth"}
    if not isinstance(r, Mapping) or set(r) != names or r.get("profile") != profile or output.method != profile:
        raise ValueError("algebraic output requires its native execution receipt")
    n = output.source.native
    degree = _key(r["field_degree"])
    from rexgraph.exact_value import exact_fraction
    polynomial = tuple(exact_fraction(q) for q in r["minimal_polynomial"])
    if (r.get("archive_digest") != n.archive_digest or r.get("source_digest") != n.source_digest
            or degree != n.field_degree or polynomial != n.polynomial
            or r.get("coefficient_basis") != "power" or len(output.axes) != 1
            or len(output.axes[0].keys) != degree or output.axes != (_polynomial_axis(output.source),)):
        raise ValueError("algebraic output differs from its declared field coefficient basis")
    rank, _, expected_shape = _output_kind(output.source, r["kind"], r["output_carrier"])
    if len(output.space.keys) != rank or tuple(_key(v) for v in r["native_shape"]) != expected_shape:
        raise ValueError("algebraic output dimensions differ from its retained coordinates")
    space, grade, shape = _output_declaration(output.source, r.get("kind"), r.get("output_carrier"))
    if output.space != space or output.grade != grade or output.variance != "coordinate" or r.get("native_shape") != shape:
        raise ValueError("algebraic output differs from its native module and value kind")
    _carrier(output.source, r["input_carrier"])
    _key(r["operator_key"]); _sha(r["input_digest"])
    if r["operation"] not in _operations:
        raise ValueError("unknown algebraic execution receipt operation")
    _portable_limits(r["limits"]); _portable_limits(r["declared_limits"])
    if any(r["limits"][name] > r["declared_limits"][name] for name in _limit_names):
        raise ValueError("algebraic execution exceeded its retained native policy")
    if n.graph_digest is not None:
        if r["graph_digest"] != n.graph_digest or _key(r["archive_key"]) != n.archive_key:
            raise ValueError("algebraic execution graph reference differs")
        _portable_limits(r["graph_limits"], graph=True); _portable_limits(r["declared_graph_limits"], graph=True)
        if (dict(r["limits"]) != _archive_graph_limits(r["graph_limits"]) or any(
                r["graph_limits"][name] > r["declared_graph_limits"][name] for name in _graph_limit_names)):
            raise ValueError("algebraic graph execution exceeded its retained native policy")
    if has_origin:
        if _key(r["origin_depth"]) != _key(r["input_origin"]["origin_depth"]):
            raise ValueError("algebraic action origin depth differs")
        _check_input_origin(r["input_origin"], output.source, _space(output.source, r["input_carrier"]),
                            output.axes, r["declared_limits"]["max_file_bytes"])
        if sha256(pack_value(thaw_tree(r["input_origin"]))).hexdigest() != r["input_digest"]:
            raise ValueError("algebraic action input differs from its complete origin declaration")


def _check_input_origin(record, source, space, axes, maximum):
    names = {"profile", "source", "carrier_record", "carrier_side", "coefficients", "origin_output", "origin_depth"}
    if not isinstance(record, Mapping) or set(record) != names or record["profile"] != _origin_input_profile:
        raise ValueError("algebraic input requires its complete origin profile")
    depth = _key(record["origin_depth"])
    if not 1 <= depth <= 32:
        raise ValueError("algebraic input origin depth exceeds its admission")
    raw = record["origin_output"]
    if not isinstance(raw, bytes) or len(raw) > maximum:
        raise ValueError("algebraic input origin bytes exceed retained admission")
    from rexgraph.model_codec import unpack_model
    origin = unpack_model(unpack_value(raw))
    if not isinstance(origin, ModelOutput) or origin.arithmetic != "algebraic":
        raise TypeError("algebraic input origin requires a complete exact model output")
    origin.check_state()
    reference = source_from_record(record["source"], allow_cpp=True)
    if (origin.method not in {"algebraic-grade-stalk-v1", "algebraic-grade-metric-stalk-v1", *_origin_profiles.values()}
            or not reference.matches(source) or not origin.source.matches(source)
            or origin.space != space or origin.axes != axes or origin.execution_receipt["kind"] != "vector"
            or tuple(tuple(row) for row in record["coefficients"]) != tuple(tuple(row) for row in origin.values)):
        raise ValueError("algebraic input origin differs from its declared source or vector coordinates")
    _key(record["carrier_record"])
    if record["carrier_side"] not in {"source", "target"}:
        raise ValueError("algebraic input origin carrier side differs")
    if depth != origin.execution_receipt.get("origin_depth", 0) + 1:
        raise ValueError("algebraic input origin chain depth differs")


def _limits(values, *, defaults=None, complete=False):
    policy = _binding().NativeAlgebraicArchive.default_limits() if defaults is None else dict(defaults)
    if (not isinstance(values, Mapping) or set(values) - set(policy)
            or complete and set(values) != set(policy)):
        raise ValueError("invalid algebraic native admission fields")
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("algebraic native admission requires integers")
        if not 0 <= value <= np.iinfo(np.uintp).max:
            raise OverflowError("algebraic native admission is outside its declared width")
        policy[name] = int(value)
    return policy


def algebraic_restore_limits(state, limits=None):
    """Intersect independent reader admission with the retained model's native policy."""
    state.check_state()
    validate_algebraic_state(state)
    graph = state.source.native.graph_digest is not None
    defaults = _binding().NativeAlgebraicArchiveGraph.default_limits() if graph else None
    reader = _limits({} if limits is None else limits, defaults=defaults)
    declared = _limits(state.configuration["graph_limits" if graph else "limits"], defaults=defaults, complete=True)
    return {name: min(value, declared[name]) for name, value in reader.items()}


def _execute(archive, operation, key, value):
    if operation in _pairs:
        return getattr(archive, operation)(key, value)
    return getattr(archive.record(key), operation)(value)


def cpp_algebraic_model(archive, *, operator_key, input_key, operation="apply", input_side="source",
                        reference=None, limits=None, graph_limits=None):
    """Declare one fixed native action and the exact module carrier accepted by its inputs."""
    binding = _binding()
    if type(archive) is not binding.NativeAlgebraicArchive:
        raise TypeError("algebraic model requires an owning NativeAlgebraicArchive")
    if reference is not None and not isinstance(reference, CppAlgebraicSourceReference):
        raise TypeError("algebraic model requires an algebraic archive source reference")
    ref = CppAlgebraicSourceReference(archive) if reference is None else reference.bind(archive)
    operator_key, input_key = _key(operator_key), _key(input_key)
    if operation not in _operations or input_side not in {"source", "target"}:
        raise ValueError("unsupported algebraic action or input carrier side")
    actual = archive.limits
    declared = actual if limits is None else _limits(limits, complete=True)
    if any(actual[name] > declared[name] for name in actual):
        raise ValueError("native archive policy exceeds the model's retained admission limits")
    carrier = _carrier(ref, archive.record(input_key).preview(0)[input_side + "_carrier"])
    zero = tuple((0,) * ref.native.field_degree for _ in range(carrier["rank"]))
    prototype = archive.vector(input_key, zero, carrier_side=input_side)
    result = _execute(archive, operation, operator_key, prototype).preview(0)
    configuration = {"execution_provider": {"name": "rexgraph.cpp.algebraic", "version": "1"},
                     "engine_receipt": binding.build_info(), "profile": _profile,
                     "input_profile": _input_profile, "operator_key": operator_key, "input_key": input_key,
                     "input_side": input_side, "operation": operation, "limits": declared,
                     "input_carrier": carrier, "output_carrier": result["source_carrier"],
                     "output_kind": result["kind"]}
    if archive.graph_owner is not None:
        actual_graph = archive.graph_owner.limits
        declared_graph = actual_graph if graph_limits is None else _limits(
            graph_limits, defaults=binding.NativeAlgebraicArchiveGraph.default_limits(), complete=True)
        if any(actual_graph[name] > declared_graph[name] for name in actual_graph):
            raise ValueError("native graph policy exceeds the model's retained admission limits")
        configuration.update(profile=_graph_profile, graph_limits=declared_graph)
    elif graph_limits is not None:
        raise ValueError("standalone algebraic model has no graph admission policy")
    state = ModelState("cpp_algebraic", "1", ref, _space(ref, carrier), (_polynomial_axis(ref),),
                       configuration, {}, "algebraic")
    enable_cpp_algebraic_provider()
    return state


def _prepare(state, archive):
    state.check_state()
    validate_algebraic_state(state)
    state.bind(archive)
    c = state.configuration
    if dict(c["engine_receipt"]) != _binding().build_info():
        raise ValueError("algebraic model requires a different native semantic implementation")
    key = (archive, state.coefficient_digest)
    with _lock:
        if key in _prepared:
            _prepared.move_to_end(key)
            return archive
    expected = cpp_algebraic_model(archive, operator_key=c["operator_key"], input_key=c["input_key"],
                                  input_side=c["input_side"], operation=c["operation"],
                                  reference=state.source, limits=c["limits"], graph_limits=c.get("graph_limits"))
    if expected.coefficient_digest != state.coefficient_digest:
        raise ValueError("algebraic model differs from its declared native action")
    size = len(archive.archive) if archive.graph_owner is None else len(archive.graph_owner.image)
    if size <= _cache_bytes:
        with _lock:
            _prepared[key] = size
            _prepared.move_to_end(key)
            while len(_prepared) > 8 or sum(_prepared.values()) > _cache_bytes:
                _prepared.popitem(last=False)
    return archive


def _input(state, archive, inputs):
    binding = _binding()
    c = state.configuration
    origin = None
    if isinstance(inputs, ModelOutput):
        if inputs.method not in {"algebraic-grade-stalk-v1", "algebraic-grade-metric-stalk-v1", *_origin_profiles.values()}:
            raise TypeError("direct algebraic model input requires a complete retained vector origin")
        inputs = cpp_algebraic_input(state, archive, inputs)
    if inputs is None:
        value = archive.record(c["input_key"])
        if value.kind != "vector":
            raise TypeError("this input carrier requires explicit algebraic vector coefficients")
    elif type(inputs) is binding.NativeAlgebraicValue:
        value = inputs
        if value.archive_owner is not archive:
            raise ValueError("algebraic input belongs to another native archive owner")
    elif isinstance(inputs, Mapping):
        names = {"profile", "source", "carrier_record", "carrier_side", "coefficients"}
        if inputs.get("profile") == _origin_input_profile:
            names |= {"origin_output", "origin_depth"}
            _check_input_origin(inputs, state.source, state.space, state.output_axes, c["limits"]["max_file_bytes"])
            origin = inputs
        if (set(inputs) != names
                or inputs["profile"] not in {_input_profile, _origin_input_profile} or _key(inputs["carrier_record"]) != c["input_key"]
                or inputs["carrier_side"] != c["input_side"]):
            raise ValueError("algebraic input differs from its declared carrier profile")
        reference = source_from_record(inputs["source"], allow_cpp=True)
        if not isinstance(reference, CppAlgebraicSourceReference) or not reference.matches(state.source):
            raise ValueError("algebraic input source differs from its model")
        value = archive.vector(c["input_key"], inputs["coefficients"], carrier_side=c["input_side"])
    else:
        raise TypeError("algebraic inference requires an owning value or a declared algebraic input")
    if value.kind != "vector" or value.preview(0)["source_carrier"] != c["input_carrier"]:
        raise ValueError("algebraic input differs from its declared native module carrier")
    record = {"profile": _input_profile, "source": state.source.as_record(),
              "carrier_record": c["input_key"], "carrier_side": c["input_side"],
              "coefficients": value.coefficients}
    if origin is not None:
        record.update(profile=_origin_input_profile, origin_output=origin["origin_output"], origin_depth=origin["origin_depth"])
    return value, freeze_tree(record)


def cpp_algebraic_input(state, archive, coefficients):
    """Capture exact coefficients or explicitly rebind a retained vector to its declared carrier."""
    _prepare(state, archive)
    origin = None
    if isinstance(coefficients, ModelOutput):
        coefficients.check_state()
        if (coefficients.arithmetic != "algebraic" or not coefficients.source.matches(state.source)
                or coefficients.space != state.space or coefficients.axes != state.output_axes
                or coefficients.execution_receipt["kind"] != "vector"):
            raise ValueError("algebraic output does not match the model's declared vector carrier")
        if coefficients.method in {"algebraic-grade-stalk-v1", "algebraic-grade-metric-stalk-v1", *_origin_profiles.values()}:
            depth = coefficients.execution_receipt.get("origin_depth", 0) + 1
            if depth > 32:
                raise ValueError("algebraic input origin depth exceeds its admission")
            from rexgraph.model_codec import pack_model
            origin = (pack_value(pack_model(coefficients, native=True)), depth)
        coefficients = coefficients.values
    elif type(coefficients) is _binding().NativeAlgebraicValue:
        return _input(state, archive, coefficients)[1]
    record = {"profile": _input_profile, "source": state.source.as_record(),
              "carrier_record": state.configuration["input_key"],
              "carrier_side": state.configuration["input_side"], "coefficients": coefficients}
    if origin is not None:
        record.update(profile=_origin_input_profile, origin_output=origin[0], origin_depth=origin[1])
    return _input(state, archive, record)[1]


def _infer(state, archive, inputs):
    _prepare(state, archive)
    value, record = _input(state, archive, inputs)
    c = state.configuration
    result = _execute(archive, c["operation"], c["operator_key"], value)
    info = result.preview(0)
    ref = state.source.bind(archive)
    space, grade, shape = _output_declaration(ref, info["kind"], info["source_carrier"])
    rows = np.array(result.coefficients, dtype=object).reshape((len(space.keys), ref.native.field_degree))
    receipt = {"profile": c["profile"], "engine": _binding().build_info(),
               "archive_digest": ref.native.archive_digest, "source_digest": ref.native.source_digest,
               "field_degree": ref.native.field_degree, "minimal_polynomial": ref.native.polynomial,
               "coefficient_basis": "power", "kind": info["kind"], "native_shape": shape,
               "input_carrier": c["input_carrier"], "output_carrier": info["source_carrier"],
               "operator_key": c["operator_key"], "operation": c["operation"],
               "input_digest": sha256(pack_value(thaw_tree(record))).hexdigest(),
               "limits": archive.limits, "declared_limits": c["limits"]}
    if archive.graph_owner is not None:
        receipt.update(graph_digest=ref.native.graph_digest, archive_key=ref.native.archive_key,
                       graph_limits=archive.graph_owner.limits, declared_graph_limits=c["graph_limits"])
    if record["profile"] == _origin_input_profile:
        receipt.update(profile=_origin_profiles[c["profile"]], input_origin=thaw_tree(record), origin_depth=record["origin_depth"])
    return ModelOutput(rows, space, state.output_axes, ref, state.coefficient_digest,
                       "algebraic", receipt["profile"], grade=grade, execution_receipt=receipt)


def _create(archive, *, adapter="cpp_algebraic", **options):
    if adapter != "cpp_algebraic":
        raise ValueError("unsupported algebraic model adapter")
    return cpp_algebraic_model(archive, **options)


def enable_cpp_algebraic_provider():
    """Register exact archive inference and its declared operations."""
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider("rexgraph.cpp.algebraic", "1", (("cpp_algebraic", "1"),),
                                      _infer, arithmetic=("algebraic",), create=_create, prepare=_prepare)
        register_model_provider(_provider)
        return _provider
