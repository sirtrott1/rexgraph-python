"""Fixed whole grade metric models on explicitly retained algebraic publications."""
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock

import numpy as np

from rexgraph.cpp_algebraic import _key, _polynomial_axis, _portable_limits
from rexgraph.cpp_algebraic import _space as _scalar_space
from rexgraph.cpp_algebraic_grade import (
    algebraic_grade_observation,
    restore_algebraic_grade_observation,
    validate_algebraic_grade_observation,
)
from rexgraph.cpp_algebraic_grade_model import (
    _build,
    _cancel,
    _engine,
    _layout,
    _layout_check,
    _policy,
    _space,
    bind_grade_model_source,
)
from rexgraph.cpp_algebraic_metric import (
    algebraic_grade_metric_observation,
    validate_algebraic_grade_metric_observation,
)
from rexgraph.cpp_algebraic_query import _check_publication, _closed, _cutoff, _limits, _publication
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference
from rexgraph.value_codec import pack_value, unpack_value

__all__ = ["cpp_algebraic_grade_metric_model", "cpp_algebraic_grade_metric_input",
           "infer_algebraic_grade_metric_model", "enable_cpp_algebraic_grade_metric_provider",
           "clear_algebraic_grade_metric_model_cache"]
_PROFILE = "algebraic-grade-metric-action-v1"
_INPUT = "owning-algebraic-grade-metric-v1"
_ADAPTER = "cpp_algebraic_grade_metric"
_PROVIDER = {"name": "rexgraph.cpp.algebraic_grade_metric", "version": "1"}
_ACTIONS = {"hodge", "coboundary", "inner", "adjacent_quadrance"}
_cache = OrderedDict()
_lock = RLock()
_provider = None


def _request(value, sizes):
    _closed(value, ("grade", "record_keys", "boundary_steps"))
    grade, steps = _key(value["grade"]), _key(value["boundary_steps"])
    keys = value["record_keys"]
    if (grade >= len(sizes) or steps > grade or isinstance(keys, (str, bytes, Mapping))
            or len(keys) != sizes[grade] or any(not _key(key) for key in keys)):
        raise ValueError("metric model requires complete original input keys and boundary steps")
    return {"grade": grade, "record_keys": tuple(keys), "boundary_steps": steps}


def _build_request(snapshot, request, availability, limits, maximum, cancellation):
    return _build(snapshot, request["grade"], request["record_keys"], request["boundary_steps"],
                  availability, limits, maximum, cancellation)


def _output_layout(field, input_layout, operation):
    if operation not in {"hodge", "coboundary"}:
        return None
    grade = input_layout["grade"] + (operation == "coboundary")
    archive = field.snapshot_owner.archive_owner
    sizes = archive.manifest["source_sizes"]
    if grade >= len(sizes):
        raise ValueError("top grade has no coboundary target")
    ranks = {}
    for key in field.metric_keys:
        carrier = archive.record(key).preview(0)["source_carrier"]
        if carrier["grade"] == grade:
            ranks[carrier["index"]] = carrier["rank"]
    if set(ranks) != set(range(sizes[grade])):
        raise ValueError("metric model result requires every original grade carrier")
    offsets = [0]
    for cell in range(sizes[grade]):
        offsets.append(offsets[-1] + ranks[cell])
    return {"grade": grade, "cell_count": sizes[grade], "domain": (offsets[-1],), "stalk_offsets": tuple(offsets)}


def _check_input_declaration(request, layouts, requirements, source, limits, maximum):
    request = _request(request, source.native.source_sizes)
    if (isinstance(layouts, (str, bytes, Mapping)) or isinstance(requirements, (str, bytes, Mapping))
            or len(layouts) != request["boundary_steps"] + 1 or len(layouts) != len(requirements)):
        raise ValueError("metric model requires complete input boundary layouts and admission")
    for i, (layout, required) in enumerate(zip(layouts, requirements, strict=True)):
        _layout_check(layout, source.native.source_sizes)
        if layout["grade"] != request["grade"] - i or layout["domain"][0] * source.native.field_degree > maximum:
            raise ValueError("metric model input grade or coordinate admission differs")
        required = _limits(required, complete=True)
        if any(required[key] > limits[key] for key in limits):
            raise ValueError("metric model input exceeds its retained execution admission")


def validate_grade_metric_model_state(state):
    """Check a closed metric model declaration without importing the native extension."""
    c = state.configuration
    _closed(c, ("execution_provider", "profile", "input_profile", "engine_receipt", "publication", "availability",
               "operation", "metric_keys", "max_retained_coordinates", "request", "other_request", "layouts",
               "other_layouts", "requirements", "other_requirements", "output_kind", "output_layout",
               "execution_limits", "archive_limits", "max_coordinates"))
    if (state.adapter != _ADAPTER or state.adapter_version != "1" or state.arithmetic != "algebraic"
            or c["profile"] != _PROFILE or c["input_profile"] != _INPUT or dict(c["execution_provider"]) != _PROVIDER):
        raise ValueError("unsupported whole-grade metric model declaration")
    if not isinstance(state.source, CppAlgebraicSourceReference):
        raise TypeError("metric model requires its complete algebraic source reference")
    source = _check_publication(c["publication"])
    if state.source.native != source.native:
        raise ValueError("metric model archive differs from its retained publication")
    _cutoff(c["publication"], c["availability"]); _engine(c["engine_receipt"])
    operation = c["operation"]
    if type(operation) is not str or operation not in _ACTIONS:
        raise ValueError("unsupported metric model operation")
    keys = c["metric_keys"]
    if (isinstance(keys, (str, bytes, Mapping)) or len(keys) != sum(source.native.source_sizes)
            or any(not _key(key) for key in keys) or len(set(keys)) != len(keys)):
        raise ValueError("metric model requires distinct explicit metrics for all original cells")
    _key(c["max_retained_coordinates"])
    limits, maximum = _limits(c["execution_limits"], complete=True), _key(c["max_coordinates"])
    _check_input_declaration(c["request"], c["layouts"], c["requirements"], source, limits, maximum)
    if operation == "inner":
        _check_input_declaration(c["other_request"], c["other_layouts"], c["other_requirements"], source, limits, maximum)
        if c["layouts"][-1] != c["other_layouts"][-1]:
            raise ValueError("metric pairing input carrier layouts differ")
    elif c["other_request"] is not None or c["other_layouts"] or c["other_requirements"]:
        raise ValueError("unary metric actions have no second input declaration")
    vector = operation in {"hodge", "coboundary"}
    if c["output_kind"] != ("field" if vector else "scalar"):
        raise ValueError("metric model output kind differs")
    if vector:
        output = c["output_layout"]
        _layout_check(output, source.native.source_sizes)
        if (output["grade"] != c["layouts"][-1]["grade"] + (operation == "coboundary")
                or operation == "hodge" and output != c["layouts"][-1]
                or output["domain"][0] * source.native.field_degree > maximum):
            raise ValueError("metric model result grade or coordinate admission differs")
    elif c["output_layout"] is not None or source.native.field_degree > maximum:
        raise ValueError("metric scalar result declaration or coordinate admission differs")
    _portable_limits(c["archive_limits"], graph=source.native.graph_digest is not None)
    if state.space != _space(state.source, c["layouts"][-1]) or state.output_axes != (_polynomial_axis(state.source),):
        raise ValueError("metric model axes differ from its original input stalks and polynomial coordinates")
    if state.payload or state.step or state.parent is not None or state.dependencies:
        raise ValueError("fixed metric models have no learning or continuation payload")


def _metric_policy(configuration, maximum):
    saved = _key(configuration["max_retained_coordinates"])
    return saved if maximum is None else min(saved, _key(maximum))


def _cache_key(state, snapshot, policy, maximum):
    return state.coefficient_digest, id(snapshot), tuple(policy.items()), maximum


def _remember(state, snapshot, field, tensor, other, policy, maximum):
    key = _cache_key(state, snapshot, policy, maximum)
    with _lock:
        entry = _cache.setdefault(key, (snapshot, field, tensor, other))
        _cache.move_to_end(key)
        while len(_cache) > 8:
            _cache.popitem(last=False)
        return entry[1:]


def clear_algebraic_grade_metric_model_cache():
    """Release the bounded cache of actual native metric fields and input owners."""
    with _lock:
        _cache.clear()


def cpp_algebraic_grade_metric_model(snapshot, *, request, metric_keys, operation="hodge", other_request=None,
                                     availability, execution_limits=None, max_retained_coordinates=1000000,
                                     max_coordinates=65536, reference=None, cancellation=None):
    """Declare a fixed exact metric action with complete default original grade inputs."""
    from rexgraph_cpp import NativeAlgebraicMetricField, NativeAlgebraicSnapshot, build_info
    if type(snapshot) is not NativeAlgebraicSnapshot:
        raise TypeError("metric model construction requires an owning native snapshot")
    _cancel(cancellation)
    if type(operation) is not str or operation not in _ACTIONS:
        raise ValueError("unsupported metric model operation")
    sizes = snapshot.archive_owner.manifest["source_sizes"]
    request = _request(request, sizes)
    if (operation == "inner") != (other_request is not None):
        raise ValueError("metric inner product requires an explicit second input request")
    other_request = None if other_request is None else _request(other_request, sizes)
    policy, maximum, retained = _limits(execution_limits), _key(max_coordinates), _key(max_retained_coordinates)
    field = NativeAlgebraicMetricField(snapshot, metric_keys, availability=availability,
                                       max_retained_coordinates=retained, cancellation=cancellation)
    _, tensor, layouts, requirements = _build_request(snapshot, request, availability, policy, maximum, cancellation)
    other, other_layouts, other_requirements = None, (), ()
    if other_request is not None:
        _, other, other_layouts, other_requirements = _build_request(snapshot, other_request, availability, policy, maximum, cancellation)
    output = _output_layout(field, layouts[-1], operation)
    reference = CppAlgebraicSourceReference(snapshot.archive_owner) if reference is None else reference.bind(snapshot.archive_owner)
    archive = snapshot.archive_owner
    configuration = {"execution_provider": _PROVIDER, "profile": _PROFILE, "input_profile": _INPUT,
        "engine_receipt": build_info(), "publication": _publication(snapshot), "availability": tuple(availability),
        "operation": operation, "metric_keys": field.metric_keys, "max_retained_coordinates": retained,
        "request": request, "other_request": other_request, "layouts": layouts, "other_layouts": other_layouts,
        "requirements": requirements, "other_requirements": other_requirements,
        "output_kind": "scalar" if output is None else "field", "output_layout": output,
        "execution_limits": tensor.execution_limits,
        "archive_limits": archive.limits if archive.graph_owner is None else archive.graph_owner.limits,
        "max_coordinates": maximum}
    state = ModelState(_ADAPTER, "1", reference, _space(reference, layouts[-1]),
                       (_polynomial_axis(reference),), configuration, {}, "algebraic")
    enable_cpp_algebraic_grade_metric_provider()
    _remember(state, snapshot, field, tensor, other, policy, retained)
    return state


def _prepare(state, snapshot, *, execution_limits=None, max_retained_coordinates=None, cancellation=None):
    if not isinstance(state, ModelState) or state.adapter != _ADAPTER:
        raise TypeError("metric preparation requires a whole-grade metric ModelState")
    _cancel(cancellation); state.check_state(); bind_grade_model_source(state, snapshot)
    c = state.configuration
    policy, retained = _policy(c, execution_limits), _metric_policy(c, max_retained_coordinates)
    from rexgraph_cpp._bridge import _query_admit
    for required in (*c["requirements"], *c["other_requirements"]):
        _query_admit(tuple(required[key] for key in policy), tuple(policy.values()))
    key = _cache_key(state, snapshot, policy, retained)
    with _lock:
        for cached_key, entry in tuple(_cache.items()):
            if cached_key[0] == key[0] and cached_key[2:] == key[2:] and entry[0].same_publication(snapshot):
                _cache.move_to_end(cached_key)
                return entry[1:]
    from rexgraph_cpp import NativeAlgebraicMetricField
    field = NativeAlgebraicMetricField(snapshot, c["metric_keys"], availability=c["availability"],
                                       max_retained_coordinates=retained, cancellation=cancellation)
    _, tensor, layouts, requirements = _build_request(snapshot, c["request"], c["availability"], policy, c["max_coordinates"], cancellation)
    other, other_layouts, other_requirements = None, (), ()
    if c["other_request"] is not None:
        _, other, other_layouts, other_requirements = _build_request(snapshot, c["other_request"], c["availability"], policy, c["max_coordinates"], cancellation)
    if (layouts != thaw_tree(c["layouts"]) or requirements != thaw_tree(c["requirements"])
            or other_layouts != thaw_tree(c["other_layouts"]) or other_requirements != thaw_tree(c["other_requirements"])
            or _output_layout(field, layouts[-1], c["operation"]) != thaw_tree(c["output_layout"])):
        raise ValueError("metric model declaration differs from actual native source and carrier layouts")
    _cancel(cancellation)
    return _remember(state, snapshot, field, tensor, other, policy, retained)


def _input(state, snapshot, value, layout, policy, availability, cancellation):
    from rexgraph_cpp import NativeAlgebraicGradeTensor
    if isinstance(value, ModelOutput):
        value.check_state()
        if value.method != "algebraic-grade-boundary-v1" or not value.source.matches(state.source):
            raise ValueError("metric input requires a complete matching original-grade model output")
        value = restore_algebraic_grade_observation(snapshot, thaw_tree(value.execution_receipt["observation"]),
            execution_limits=policy, max_coordinates=state.configuration["max_coordinates"], cancellation=cancellation)
    if type(value) is not NativeAlgebraicGradeTensor:
        raise TypeError("metric input requires an owning original-grade tensor or complete boundary model output")
    if not value.matches_source(snapshot) or _layout(value) != thaw_tree(layout):
        raise ValueError("metric input actual publication or original stalk layout differs")
    observation = algebraic_grade_observation(value, availability=availability, max_coordinates=state.configuration["max_coordinates"])
    validate_algebraic_grade_observation(observation, execution_limits=policy,
                                         max_coordinates=state.configuration["max_coordinates"])
    return value


def cpp_algebraic_grade_metric_input(state, snapshot, inputs=None, *, execution_limits=None,
                                     max_retained_coordinates=None, cancellation=None):
    """Prepare checked original inputs, including both explicitly declared pairing operands."""
    _, default, other = _prepare(state, snapshot, execution_limits=execution_limits,
                                 max_retained_coordinates=max_retained_coordinates, cancellation=cancellation)
    c, policy = state.configuration, _policy(state.configuration, execution_limits)
    if inputs is None:
        return default if other is None else {"input": default, "other": other}
    if c["operation"] == "inner":
        _closed(inputs, ("input", "other"))
        return {"input": _input(state, snapshot, inputs["input"], c["layouts"][-1], policy, c["availability"], cancellation),
                "other": _input(state, snapshot, inputs["other"], c["other_layouts"][-1], policy, c["availability"], cancellation)}
    return _input(state, snapshot, inputs, c["layouts"][-1], policy, c["availability"], cancellation)


def _origin(value):
    if not isinstance(value, ModelOutput):
        return None
    from rexgraph.model_codec import pack_model
    value.check_state()
    return pack_value(pack_model(value, native=True))


def _origins(inputs, pairing, maximum):
    if inputs is None:
        origins = {"input": None, "other": None}
    elif pairing:
        _closed(inputs, ("input", "other"))
        origins = {key: _origin(inputs[key]) for key in ("input", "other")}
    else:
        origins = {"input": _origin(inputs), "other": None}
    if sum(len(raw) for raw in origins.values() if raw is not None) > maximum:
        raise ValueError("metric model input origins exceed retained byte admission")
    return origins


def infer_algebraic_grade_metric_model(state, snapshot, inputs=None, *, availability=None, execution_limits=None,
                                       max_retained_coordinates=None, cancellation=None):
    """Run the declared canonical metric operation on complete owning grade inputs."""
    _cancel(cancellation)
    if not isinstance(state, ModelState) or state.adapter != _ADAPTER:
        raise TypeError("metric inference requires a whole-grade metric ModelState")
    c = state.configuration
    availability = c["availability"] if availability is None else availability
    _cutoff(c["publication"], availability)
    field, default, other = _prepare(state, snapshot, execution_limits=execution_limits,
                                     max_retained_coordinates=max_retained_coordinates, cancellation=cancellation)
    policy = _policy(c, execution_limits)
    origins = _origins(inputs, c["operation"] == "inner", c["archive_limits"]["max_file_bytes"])
    tensor = default
    if inputs is not None:
        selected = cpp_algebraic_grade_metric_input(state, snapshot, inputs, execution_limits=policy,
            max_retained_coordinates=field.max_retained_coordinates, cancellation=cancellation)
        if c["operation"] == "inner": tensor, other = selected["input"], selected["other"]
        else: tensor = selected
    for value in (tensor,) if other is None else (tensor, other):
        observed = algebraic_grade_observation(value, availability=availability, max_coordinates=c["max_coordinates"])
        validate_algebraic_grade_observation(observed, execution_limits=policy, max_coordinates=c["max_coordinates"])
    observation = algebraic_grade_metric_observation(field, tensor, c["operation"], availability=availability,
        other=other, cancellation=cancellation, max_coordinates=c["max_coordinates"])
    _cancel(cancellation)
    detail, layout = observation["detail"], thaw_tree(c["output_layout"])
    if layout is not None and {key: detail[key] for key in layout} != layout:
        raise ValueError("metric output differs from the declared original result carriers")
    space = _scalar_space(state.source, None) if layout is None else _space(state.source, layout)
    receipt = {"profile": _PROFILE, "engine": observation["engine"], "operation": c["operation"],
               "observation": observation, "execution_limits": policy, "declared_limits": thaw_tree(c["execution_limits"]),
               "max_retained_coordinates": field.max_retained_coordinates,
               "declared_max_retained_coordinates": c["max_retained_coordinates"],
               "input_origins": origins, "origin_max_bytes": c["archive_limits"]["max_file_bytes"]}
    values = np.array(detail["coefficients"], dtype=object).reshape(len(space.keys), state.source.native.field_degree)
    return ModelOutput(values, space, state.output_axes, state.source, state.coefficient_digest, "algebraic", _PROFILE,
        grade=None if layout is None else layout["grade"], variance="coordinate", execution_receipt=receipt)


def validate_grade_metric_model_output(output):
    """Check full metric output coordinates and complete original input observations."""
    r = thaw_tree(output.execution_receipt)
    _closed(r, ("profile", "engine", "operation", "observation", "execution_limits", "declared_limits",
               "max_retained_coordinates", "declared_max_retained_coordinates", "input_origins", "origin_max_bytes"))
    if output.method != _PROFILE or r["profile"] != _PROFILE or output.arithmetic != "algebraic":
        raise ValueError("unsupported whole-grade metric model output")
    if not isinstance(output.source, CppAlgebraicSourceReference):
        raise TypeError("metric output requires its retained algebraic source")
    _engine(r["engine"])
    policy, declared = _limits(r["execution_limits"], complete=True), _limits(r["declared_limits"], complete=True)
    if any(policy[key] > declared[key] for key in policy) or _key(r["max_retained_coordinates"]) > _key(r["declared_max_retained_coordinates"]):
        raise ValueError("metric model inference enlarged its retained admission")
    value = r["observation"]
    maximum = max(output.values.size, len(value["input"]["detail"]["coefficients"]) * output.source.native.field_degree,
                  0 if value["other"] is None else len(value["other"]["detail"]["coefficients"]) * output.source.native.field_degree)
    validate_algebraic_grade_metric_observation(value, execution_limits=policy, max_coordinates=maximum)
    if (r["operation"] != value["operation"] or r["engine"] != value["engine"]
            or r["max_retained_coordinates"] != value["max_retained_coordinates"]
            or _check_publication(value["publication"]).native != output.source.native):
        raise ValueError("metric model output receipt or source differs")
    origins, bound = r["input_origins"], _key(r["origin_max_bytes"])
    _closed(origins, ("input", "other"))
    if any(raw is not None and type(raw) is not bytes for raw in origins.values()):
        raise TypeError("metric input origins require complete encoded model outputs")
    if sum(len(raw) for raw in origins.values() if raw is not None) > bound:
        raise ValueError("metric input origins exceed their retained byte admission")
    if origins["other"] is not None and value["other"] is None:
        raise ValueError("unary metric output cannot retain a second origin")
    from rexgraph.cpp_algebraic_stalk import _same_field
    from rexgraph.model_codec import unpack_model
    for key, raw in origins.items():
        if raw is None:
            continue
        parent = unpack_model(unpack_value(raw))
        if (not isinstance(parent, ModelOutput) or parent.method != "algebraic-grade-boundary-v1"
                or not parent.source.matches(output.source)
                or not _same_field(thaw_tree(parent.execution_receipt["observation"]), value[key])):
            raise ValueError("metric input origin differs from its complete original grade observation")
    scalar = value["operation"] in {"inner", "adjacent_quadrance"}
    layout = None if scalar else {key: value["detail"][key] for key in ("grade", "cell_count", "domain", "stalk_offsets")}
    space = _scalar_space(output.source, None) if scalar else _space(output.source, layout)
    if (output.space != space or output.axes != (_polynomial_axis(output.source),)
            or output.grade != (None if scalar else layout["grade"]) or output.variance != "coordinate"
            or tuple(tuple(row) for row in output.values) != value["detail"]["coefficients"]):
        raise ValueError("metric model output differs from its exact original carrier coordinates")


def _create(snapshot, *, adapter=_ADAPTER, **options):
    if adapter != _ADAPTER:
        raise ValueError("unsupported whole-grade metric adapter")
    return cpp_algebraic_grade_metric_model(snapshot, **options)


def enable_cpp_algebraic_grade_metric_provider():
    """Register fixed whole grade metric construction, preparation and exact inference."""
    global _provider
    from rexgraph_cpp import build_info
    build_info()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", ((_ADAPTER, "1"),),
                infer_algebraic_grade_metric_model, arithmetic=("algebraic",), create=_create, prepare=_prepare)
        register_model_provider(_provider)
        return _provider
