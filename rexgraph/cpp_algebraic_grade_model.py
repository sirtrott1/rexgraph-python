"""Exact whole grade model actions on owning algebraic stalk fields."""
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock

import numpy as np

from rexgraph.cpp_algebraic import _key, _polynomial_axis, _portable_limits
from rexgraph.cpp_algebraic_grade import (
    _matches_publication,
    algebraic_grade_observation,
    restore_algebraic_grade_observation,
    validate_algebraic_grade_observation,
)
from rexgraph.cpp_algebraic_query import _check_publication, _closed, _cutoff, _limits, _publication
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference, _sha
from rexgraph.type_accession import CoordinateSpace

__all__ = ["cpp_algebraic_grade_model", "cpp_algebraic_grade_input", "infer_algebraic_grade_model",
           "enable_cpp_algebraic_grade_provider", "clear_algebraic_grade_model_cache"]
_PROFILE = "algebraic-grade-boundary-v1"
_INPUT = "owning-algebraic-grade-v1"
_PROVIDER = {"name": "rexgraph.cpp.algebraic_grade", "version": "1"}
_LAYOUT = {"grade", "cell_count", "domain", "stalk_offsets"}
_cache = OrderedDict()
_lock = RLock()
_provider = None


def _layout(tensor):
    return {"grade": tensor.grade, "cell_count": tensor.cell_count,
            "domain": tensor.domain, "stalk_offsets": tensor.stalk_offsets}


def _space(reference, layout):
    offsets = layout["stalk_offsets"]
    keys = tuple(f"{cell}:{local}" for cell in range(layout["cell_count"])
                 for local in range(offsets[cell + 1] - offsets[cell]))
    return CoordinateSpace(f"cpp-algebraic:{reference.native.coordinate_scope}:grade:{layout['grade']}:stalk", keys)


def _engine(receipt):
    _closed(receipt, ("engine_version", "engine_digest", "binding_digest", "binding_version", "scalar_policy"))
    _sha(receipt["engine_digest"]); _sha(receipt["binding_digest"])
    if (not isinstance(receipt["engine_version"], str) or not receipt["engine_version"]
            or receipt["binding_version"] != "1" or receipt["scalar_policy"] != "gmp-rational-exact"):
        raise ValueError("grade models require an exact native engine receipt")


def _layout_check(layout, sizes):
    _closed(layout, _LAYOUT)
    grade, count = _key(layout["grade"]), _key(layout["cell_count"])
    if grade >= len(sizes) or count != sizes[grade]:
        raise ValueError("grade model layout differs from the original source cells")
    domain, offsets = layout["domain"], layout["stalk_offsets"]
    if isinstance(domain, (str, bytes, Mapping)) or len(domain) != 1:
        raise ValueError("grade model requires one complete flattened stalk domain")
    if isinstance(offsets, (str, bytes, Mapping)) or len(offsets) != count + 1:
        raise ValueError("grade model requires an offset for every original stalk")
    values = tuple(_key(i) for i in offsets)
    if (values[0] != 0 or values[-1] != _key(domain[0])
            or any(a > b for a, b in zip(values, values[1:], strict=False))):
        raise ValueError("grade model stalk offsets differ")


def validate_grade_model_state(state):
    """Check a closed whole grade model declaration without loading the native extension."""
    c = state.configuration
    _closed(c, ("execution_provider", "profile", "input_profile", "engine_receipt", "publication", "availability",
               "grade", "record_keys", "boundary_steps", "layouts", "requirements", "execution_limits", "archive_limits",
               "max_coordinates"))
    if (state.adapter != "cpp_algebraic_grade" or state.adapter_version != "1" or state.arithmetic != "algebraic"
            or c["profile"] != _PROFILE or c["input_profile"] != _INPUT or dict(c["execution_provider"]) != _PROVIDER):
        raise ValueError("unsupported whole-grade model declaration")
    if not isinstance(state.source, CppAlgebraicSourceReference):
        raise TypeError("grade model requires its complete algebraic source reference")
    source = _check_publication(c["publication"])
    if state.source.native != source.native:
        raise ValueError("grade model archive differs from its source publication")
    _cutoff(c["publication"], c["availability"]); _engine(c["engine_receipt"])
    grade, steps = _key(c["grade"]), _key(c["boundary_steps"])
    if grade >= len(source.native.source_sizes) or steps > grade:
        raise ValueError("grade model boundary count exceeds its original grade")
    if isinstance(c["record_keys"], (str, bytes, Mapping)) or len(c["record_keys"]) != source.native.source_sizes[grade]:
        raise ValueError("grade model requires one vector record per original input cell")
    if any(not _key(k) for k in c["record_keys"]):
        raise ValueError("grade model input record keys must be nonzero")
    layouts, requirements = c["layouts"], c["requirements"]
    if (isinstance(layouts, (str, bytes, Mapping)) or len(layouts) != steps + 1
            or isinstance(requirements, (str, bytes, Mapping)) or len(requirements) != len(layouts)):
        raise ValueError("grade model requires its complete boundary layouts and admission")
    limits, maximum = _limits(c["execution_limits"], complete=True), _key(c["max_coordinates"])
    for i, (layout, required) in enumerate(zip(layouts, requirements, strict=True)):
        _layout_check(layout, source.native.source_sizes)
        if layout["grade"] != grade - i:
            raise ValueError("grade model boundary order differs")
        if layout["domain"][0] * source.native.field_degree > maximum:
            raise ValueError("grade model coordinate envelope exceeds its retained admission")
        required = _limits(required, complete=True)
        if any(required[k] > limits[k] for k in limits):
            raise ValueError("grade model action exceeds its retained execution admission")
    _portable_limits(c["archive_limits"], graph=source.native.graph_digest is not None)
    if (len(state.space.keys) != layouts[0]["domain"][0] or state.space != _space(state.source, layouts[0])
            or state.output_axes != (_polynomial_axis(state.source),)):
        raise ValueError("grade model axes differ from its original stalk and polynomial coordinates")
    if state.payload or state.step or state.parent is not None or state.dependencies:
        raise ValueError("fixed grade actions have no learning or continuation payload")


def bind_grade_model_source(state, snapshot):
    """Bind a model to an explicitly selected native publication and its actual archive."""
    from rexgraph_cpp import NativeAlgebraicSnapshot, build_info
    if type(snapshot) is not NativeAlgebraicSnapshot:
        raise TypeError("grade model execution requires an owning NativeAlgebraicSnapshot")
    if not _matches_publication(state.configuration["publication"], _publication(snapshot)):
        raise ValueError("grade model belongs to another source publication")
    engine = state.configuration["engine_receipt"]
    current = build_info()
    if any(engine[key] != current[key] for key in ("engine_version", "engine_digest", "scalar_policy")):
        raise ValueError("grade model requires its declared canonical engine implementation")
    archive = snapshot.archive_owner
    live = archive.limits if archive.graph_owner is None else archive.graph_owner.limits
    if any(live[key] > saved for key, saved in state.configuration["archive_limits"].items()):
        raise ValueError("grade model source exceeds its retained archive admission")
    return state.source.bind(archive)


def grade_model_restore_limits(state, reader=None):
    """Intersect saved archive/history admission with independent reader bounds."""
    if state.adapter == "cpp_algebraic_grade_metric":
        from rexgraph.cpp_algebraic_grade_metric_model import validate_grade_metric_model_state
        validate_grade_metric_model_state(state)
    else:
        validate_grade_model_state(state)
    policy = dict(state.configuration["archive_limits"])
    publication = state.configuration["publication"]
    if "graph_history" in publication:
        for key, value in publication["graph_history"]["limits"].items():
            policy[key] = min(policy.get(key, value), value)
    if reader is not None:
        if not isinstance(reader, Mapping) or set(reader) - set(policy):
            raise ValueError("unknown grade model source reader admission")
        for key, value in reader.items():
            policy[key] = min(policy[key], _key(value))
    return policy


def _build(snapshot, grade, keys, steps, availability, limits, maximum, cancellation=None):
    from rexgraph_cpp import NativeAlgebraicGradeTensor
    root = NativeAlgebraicGradeTensor(snapshot, grade, keys, availability=availability,
                                     execution_limits=limits, cancellation=cancellation)
    layouts, requirements = [], []
    tensor = root
    for i in range(steps + 1):
        if tensor.domain[0] * snapshot.archive_owner.field_degree > maximum:
            raise ValueError("grade model coordinate envelope exceeds its admission")
        layouts.append(_layout(tensor)); requirements.append(tensor.requirements)
        if i < steps:
            tensor = tensor.boundary(availability=availability, cancellation=cancellation)
    return root, tensor, tuple(layouts), tuple(requirements)


def _policy(configuration, overrides=None):
    limits = _limits(configuration["execution_limits"], complete=True)
    if overrides is not None:
        requested = _limits(overrides)
        for key in overrides:
            limits[key] = min(limits[key], requested[key])
    return limits


def _cancel(cancellation):
    from rexgraph_cpp import CancellationToken, ExecutionCancelled
    if cancellation is not None:
        if type(cancellation) is not CancellationToken:
            raise TypeError("grade execution cancellation requires a native CancellationToken")
        if cancellation.cancelled:
            raise ExecutionCancelled("exact execution cancelled at a phase boundary")


def _remember(state, snapshot, root, result):
    key = (state.coefficient_digest, id(snapshot), tuple(sorted(root.execution_limits.items())))
    with _lock:
        entry = _cache.setdefault(key, (snapshot, root, result))
        _cache.move_to_end(key)
        while len(_cache) > 8:
            _cache.popitem(last=False)
        return entry[1:]


def clear_algebraic_grade_model_cache():
    """Release the bounded cache of actual prepared publication owners."""
    with _lock:
        _cache.clear()


def cpp_algebraic_grade_model(snapshot, *, grade, record_keys, boundary_steps=1, availability,
                              execution_limits=None, max_coordinates=65536, reference=None, cancellation=None):
    """Declare a fixed native boundary action on the full original grade stalk sum."""
    from rexgraph_cpp import NativeAlgebraicSnapshot, build_info
    if type(snapshot) is not NativeAlgebraicSnapshot:
        raise TypeError("grade model construction requires an owning native snapshot")
    _cancel(cancellation)
    grade, steps, maximum = _key(grade), _key(boundary_steps), _key(max_coordinates)
    if steps > grade:
        raise ValueError("grade model boundary count exceeds its original grade")
    limits = _limits(execution_limits)
    root, result, layouts, requirements = _build(snapshot, grade, record_keys, steps, availability, limits, maximum, cancellation)
    reference = CppAlgebraicSourceReference(snapshot.archive_owner) if reference is None else reference.bind(snapshot.archive_owner)
    archive = snapshot.archive_owner
    configuration = {"execution_provider": _PROVIDER, "profile": _PROFILE, "input_profile": _INPUT,
        "engine_receipt": build_info(), "publication": _publication(snapshot), "availability": tuple(availability),
        "grade": grade, "record_keys": root.record_keys, "boundary_steps": steps, "layouts": layouts,
        "requirements": requirements, "execution_limits": root.execution_limits,
        "archive_limits": archive.limits if archive.graph_owner is None else archive.graph_owner.limits,
        "max_coordinates": maximum}
    state = ModelState("cpp_algebraic_grade", "1", reference, _space(reference, layouts[0]),
                       (_polynomial_axis(reference),), configuration, {}, "algebraic")
    enable_cpp_algebraic_grade_provider()
    _remember(state, snapshot, root, result)
    return state


def _prepare(state, snapshot, *, execution_limits=None, cancellation=None):
    if not isinstance(state, ModelState) or state.adapter != "cpp_algebraic_grade":
        raise TypeError("grade preparation requires a whole-grade ModelState")
    _cancel(cancellation)
    state.check_state()
    bind_grade_model_source(state, snapshot)
    c = state.configuration
    policy = _policy(c, execution_limits)
    from rexgraph_cpp._bridge import _query_admit
    for required in c["requirements"]:
        _query_admit(tuple(required[key] for key in policy), tuple(policy.values()))
    key = (state.coefficient_digest, id(snapshot), tuple(sorted(policy.items())))
    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            if not entry[0].same_publication(snapshot):
                raise ValueError("prepared grade model changed its actual publication owner")
            _cache.move_to_end(key)
            return entry[1:]
        for cached_key, entry in tuple(_cache.items()):
            if cached_key[0] == key[0] and cached_key[2] == key[2] and entry[0].same_publication(snapshot):
                _cache.move_to_end(cached_key)
                return entry[1:]
    root, result, layouts, requirements = _build(snapshot, c["grade"], c["record_keys"], c["boundary_steps"],
        c["availability"], policy, c["max_coordinates"], cancellation)
    if layouts != thaw_tree(c["layouts"]) or requirements != thaw_tree(c["requirements"]):
        raise ValueError("grade model declaration differs from its actual source stalks and boundary action")
    return _remember(state, snapshot, root, result)


def cpp_algebraic_grade_input(state, snapshot, inputs=None, *, execution_limits=None, cancellation=None):
    """Select or replay complete native input fields on the model's actual publication."""
    from rexgraph_cpp import NativeAlgebraicGradeTensor
    root, _ = _prepare(state, snapshot, execution_limits=execution_limits, cancellation=cancellation)
    policy = _policy(state.configuration, execution_limits)
    if inputs is None:
        return root
    if isinstance(inputs, ModelOutput):
        inputs.check_state()
        if inputs.method != _PROFILE or not inputs.source.matches(state.source):
            raise ValueError("grade model input requires a matching whole-grade algebraic output")
        inputs = restore_algebraic_grade_observation(snapshot, thaw_tree(inputs.execution_receipt["observation"]),
            execution_limits=policy, max_coordinates=state.configuration["max_coordinates"], cancellation=cancellation)
    if type(inputs) is not NativeAlgebraicGradeTensor:
        raise TypeError("grade model input requires an owning native field or complete grade-model output")
    if not inputs.matches_source(snapshot) or _layout(inputs) != thaw_tree(state.configuration["layouts"][0]):
        raise ValueError("grade model input differs from its actual publication or original stalk layout")
    observation = algebraic_grade_observation(inputs, availability=state.configuration["availability"],
                                             max_coordinates=state.configuration["max_coordinates"])
    validate_algebraic_grade_observation(observation, execution_limits=policy,
                                         max_coordinates=state.configuration["max_coordinates"])
    return inputs


def infer_algebraic_grade_model(state, snapshot, inputs=None, *, availability=None, execution_limits=None, cancellation=None):
    """Run the declared native boundary action without converting polynomial coefficients."""
    from rexgraph_cpp import build_info
    _cancel(cancellation)
    if not isinstance(state, ModelState) or state.adapter != "cpp_algebraic_grade":
        raise TypeError("grade inference requires a whole-grade ModelState")
    c = state.configuration
    availability = c["availability"] if availability is None else availability
    _cutoff(c["publication"], availability)
    limits = _policy(c, execution_limits)
    root, default = _prepare(state, snapshot, execution_limits=limits, cancellation=cancellation)
    tensor = root if inputs is None else cpp_algebraic_grade_input(state, snapshot, inputs,
                                                                 execution_limits=limits, cancellation=cancellation)
    input_observation = algebraic_grade_observation(tensor, availability=availability, max_coordinates=c["max_coordinates"])
    validate_algebraic_grade_observation(input_observation, execution_limits=limits, max_coordinates=c["max_coordinates"])
    if inputs is None:
        result = default
    else:
        result = tensor
        for _ in range(c["boundary_steps"]):
            result = result.boundary(availability=availability, execution_limits=limits, cancellation=cancellation)
    observation = algebraic_grade_observation(result, availability=availability, max_coordinates=c["max_coordinates"])
    _cancel(cancellation)
    validate_algebraic_grade_observation(observation, execution_limits=limits, max_coordinates=c["max_coordinates"])
    receipt = {"profile": _PROFILE, "engine": build_info(), "boundary_steps": c["boundary_steps"],
               "input_observation": input_observation, "observation": observation,
               "execution_limits": limits, "declared_limits": thaw_tree(c["execution_limits"])}
    values = np.array(observation["detail"]["coefficients"], dtype=object).reshape(
        (result.domain[0], state.source.native.field_degree))
    return ModelOutput(values, _space(state.source, _layout(result)), state.output_axes, state.source,
                       state.coefficient_digest, "algebraic", _PROFILE, grade=result.grade,
                       variance="coordinate", execution_receipt=receipt)


def validate_grade_model_output(output):
    """Check complete stalk coordinates, source publication and retained boundary lineage."""
    r = thaw_tree(output.execution_receipt)
    _closed(r, ("profile", "engine", "boundary_steps", "input_observation", "observation", "execution_limits", "declared_limits"))
    if output.method != _PROFILE or r["profile"] != _PROFILE or output.arithmetic != "algebraic":
        raise ValueError("unsupported grade model output profile")
    _engine(r["engine"])
    limits, declared = _limits(r["execution_limits"], complete=True), _limits(r["declared_limits"], complete=True)
    if any(limits[key] > declared[key] for key in limits):
        raise ValueError("grade inference enlarged its declared execution admission")
    original, result = r["input_observation"], r["observation"]
    maximum = max(len(output.values.flat), sum(len(row) for row in original["detail"]["coefficients"]))
    for value in (original, result):
        validate_algebraic_grade_observation(value, execution_limits=limits, max_coordinates=maximum)
        if value["operation"] != "field":
            raise ValueError("grade model output requires a complete native field")
    if original["publication"] != result["publication"] or original["availability"] != result["availability"]:
        raise ValueError("grade inference changed its publication or availability")
    if _check_publication(result["publication"]).native != output.source.native:
        raise ValueError("grade model output differs from its retained algebraic source")
    steps = _key(r["boundary_steps"])
    if (tuple(result["lineage"][:len(original["lineage"])]) != tuple(original["lineage"])
            or len(result["lineage"]) != len(original["lineage"]) + steps):
        raise ValueError("grade model output lost its complete input boundary lineage")
    layout = {key: result["detail"][key] for key in _LAYOUT}
    if (output.space != _space(output.source, layout) or output.axes != (_polynomial_axis(output.source),)
            or output.grade != layout["grade"] or output.variance != "coordinate"
            or tuple(tuple(row) for row in output.values) != tuple(result["detail"]["coefficients"])):
        raise ValueError("grade model output differs from its original stalk polynomial coordinates")


def _create(snapshot, *, adapter="cpp_algebraic_grade", **options):
    if adapter != "cpp_algebraic_grade":
        raise ValueError("unsupported whole-grade model adapter")
    return cpp_algebraic_grade_model(snapshot, **options)


def enable_cpp_algebraic_grade_provider():
    """Register fixed exact whole grade model construction, preparation and inference."""
    global _provider
    from rexgraph_cpp import build_info
    build_info()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", (("cpp_algebraic_grade", "1"),),
                infer_algebraic_grade_model, arithmetic=("algebraic",), create=_create, prepare=_prepare)
        register_model_provider(_provider)
        return _provider
