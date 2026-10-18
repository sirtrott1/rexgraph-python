"""Exact transport learning at one original grade with complete chain constraints."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.cpp_stalk_boundary import _space, _slots, _field, _zeros
from rexgraph.cpp_stalk_fields import _axes, _exact
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "single-grade-chain-transport-model-v1"
_METHOD = "exact-graded-affine-stalk-transport-v1"
_BATCH = "whole-graded-transport-observation-v1"
_QUERY = "whole-graded-transport-query-v1"
_PROVIDER = dict(name="rexgraph.cpp.graded_stalk_transport", version="1")
_lock = RLock()
_cache = OrderedDict()
_provider = None
_affine_provider = None
_AFFINE_PROFILE = "single-grade-affine-completion-model-v1"
_AFFINE_PROVIDER = dict(name="rexgraph.cpp.graded_stalk_transport", version="2")


def _profile(version):
    if version == "1": return _PROFILE, _PROVIDER, _METHOD
    if version == "2": return _AFFINE_PROFILE, _AFFINE_PROVIDER, _METHOD
    raise ValueError("unsupported graded transport adapter version")


def _observation(c, observation):
    if not isinstance(observation, Mapping) or set(observation) != {"input", "target", "observed"}:
        raise ValueError("closed original graded observation required")
    field = _field(c, c["grade"], observation["input"])
    target = _field(c, c["grade"]-1, observation["target"])
    mask = tuple(observation["observed"])
    if len(mask) != len(target) or not any(mask) or any(type(v) is not bool for v in mask):
        raise ValueError("nonempty whole original lower-stalk Boolean mask required")
    if any(any(row) for row, seen in zip(target, mask, strict=True) if not seen):
        raise ValueError("unobserved original stalk has nonzero target data")
    return dict(input=field, target=target, observed=mask)


def _batch(state, observation):
    c = state.configuration; target = observation["target"]
    width = len(state.output_axes[0].keys)
    if len(target)*width > c["limits"][1]:
        raise ValueError("padded original target field exceeds entry admission")
    values = np.full((len(target), width), Fraction(0), dtype=object)
    for row, vector in enumerate(target):
        values[row, :len(vector)] = vector
    return ModelBatch(state.source, state.space, values, np.asarray(observation["observed"], dtype=bool),
        dict(profile=_BATCH, partition="training", input=observation["input"]))


def _training_observation(state, batch):
    c = state.configuration; axes = c["stalk_axes"][c["grade"]-1]; b = batch.inputs
    batch.check_state()
    if (batch.space != state.space or not batch.source.matches(state.source) or batch.dependencies or
            not isinstance(b, Mapping) or set(b) != {"profile", "partition", "input"} or
            b["profile"] != _BATCH or b["partition"] != "training" or
            batch.targets.shape != (len(axes), len(state.output_axes[0].keys))):
        raise ValueError("original whole graded transport training batch required")
    target = []
    for i, row in enumerate(batch.targets):
        values = tuple(_exact(v, c["limits"][3]) for v in row)
        if any(values[len(axes[i]):]) or batch.observed[i] and i not in c["training_rows"]:
            raise ValueError("target lies outside the original declared training stalks")
        target.append(values[:len(axes[i])])
    return _observation(c, dict(input=b["input"], target=tuple(target), observed=tuple(bool(v) for v in batch.observed)))


def validate_graded_stalk_transport_state(state):
    """Validate original grade carriers, parameter declarations and whole field history."""
    c = state.configuration
    profile, provider, _ = _profile(state.adapter_version)
    if (state.adapter != "cpp_graded_stalk_transport" or
            state.arithmetic != "rational" or state.dependencies or not isinstance(state.source, CppSourceReference) or
            set(c) != {"profile", "execution_provider", "engine_receipt", "stalk_axes", "fixed_maps", "parameter_slots",
                       "parameter_space", "parameter_basis", "training_rows", "grade", "limits", "max_bytes", "source_reference"} or
            c["profile"] != profile or dict(c["execution_provider"]) != provider):
        raise ValueError("closed original graded transport model required")
    axes = tuple(_axes(rows) for rows in c["stalk_axes"])
    if (len(c["limits"]) != 4 or any(type(v) is not int or v <= 0 for v in c["limits"]) or
            type(c["max_bytes"]) is not int or c["max_bytes"] <= 0 or
            type(c["grade"]) is not int or not 0 < c["grade"] < len(axes) or
            len(axes) != len(state.source.native.coordinates) or
            tuple(map(len, axes)) != tuple(map(len, state.source.native.coordinates)) or
            state.space != _space(state.source, c["grade"]-1) or state.output_axes != _slots(axes[c["grade"]-1]) or
            dict(c["source_reference"]) != state.source.as_record()):
        raise ValueError("original named stalk grades and positive admissions required")
    if sum(len(row) for family in axes for row in family) > c["limits"][0]:
        raise ValueError("complete original stalk dimensions exceed admission")
    if len(c["fixed_maps"]) != len(axes) or c["fixed_maps"][0]:
        raise ValueError("complete original graded fixed map declarations required")
    for family in c["fixed_maps"]:
        for block in family:
            seen = set()
            for i, j, q in block:
                if type(i) is not int or type(j) is not int or min(i, j) < 0 or (i, j) in seen:
                    raise ValueError("distinct natural original map coordinates required")
                seen.add((i, j)); _exact(q, c["limits"][3])
    support = c["parameter_slots"]
    if len(support) > c["limits"][0] or len(set(support)) != len(support):
        raise ValueError("distinct admitted original parameter support required")
    if any(len(s) != 3 or any(type(v) is not int or v < 0 for v in s) for s in support):
        raise ValueError("original occurrence, row and column coordinates required")
    if c["parameter_space"] is None:
        if c["parameter_basis"] is not None:
            raise ValueError("parameter basis requires a named parameter space")
        n = len(support)
    else:
        space = c["parameter_space"]
        if not isinstance(space, Mapping) or set(space) != {"name", "keys"} or c["parameter_basis"] is None:
            raise ValueError("complete named parameter declaration required")
        CoordinateSpace(space["name"], space["keys"]); n = len(space["keys"])
        seen = set()
        for i, j, q in c["parameter_basis"]:
            if type(i) is not int or type(j) is not int or not 0 <= i < len(support) or not 0 <= j < n or (i, j) in seen:
                raise ValueError("distinct original parameter basis coordinates required")
            seen.add((i, j)); _exact(q, c["limits"][3])
    training = c["training_rows"]
    if len(set(training)) != len(training) or any(type(i) is not int or not 0 <= i < len(axes[c["grade"]-1]) for i in training):
        raise ValueError("distinct original lower training stalks required")
    p = state.payload
    if set(p) != {"observations", "fit"} or len(p["observations"]) != state.step:
        raise ValueError("complete original graded observation history required")
    for record in p["observations"]:
        if set(record) != {"input", "target", "observed", "batch", "batch_checkpoint"}:
            raise ValueError("complete original batch receipt required")
        observation = _observation(c, {k:record[k] for k in ("input", "target", "observed")})
        if any(seen and i not in training for i, seen in enumerate(observation["observed"])):
            raise ValueError("observation exceeds original training rows")
    fit = p["fit"].get("fit", {})
    if (fit.get("kernel_shape") != (n, len(fit.get("free_columns", ()))) or len(fit.get("particular", ())) != n or
            type(fit.get("rank")) is not int or not 0 <= fit["rank"] <= n or
            fit["rank"]+len(fit.get("free_columns", ())) != n or not p["fit"].get("strict_chain_family")):
        raise ValueError("complete original graded affine parameter family required")


def _native(c, source, observations):
    from rexgraph_cpp import PreparedStalkBoundary
    owner_type = PreparedStalkBoundary
    if c["profile"] == _AFFINE_PROFILE:
        from rexgraph_cpp import PreparedStalkDeclaration
        owner_type = PreparedStalkDeclaration
    owner = owner_type(source.archive, c["stalk_axes"], c["fixed_maps"], limits=c["limits"], max_bytes=c["max_bytes"])
    return owner.fit_transports(c["grade"], c["parameter_slots"],
        tuple({k:o[k] for k in ("input", "target", "observed")} for o in observations),
        parameter_space=c["parameter_space"], parameter_basis=c["parameter_basis"])


def _remember(state, plan):
    with _lock:
        _cache[state.coefficient_digest] = plan
        while len(_cache) > 4:
            _cache.popitem(last=False)


def _prepare(state, source):
    from rexgraph_cpp import graded_transport_build_info
    from rexgraph.model_codec import unpack_model
    validate_graded_stalk_transport_state(state); state.bind(source)
    if dict(state.configuration["engine_receipt"]) != graded_transport_build_info():
        raise ValueError("graded model requires its retained native implementation")
    with _lock:
        plan = _cache.get(state.coefficient_digest)
        if plan is not None:
            _cache.move_to_end(state.coefficient_digest)
            return plan
    for record in state.payload["observations"]:
        batch = unpack_model(record["batch_checkpoint"])
        if not isinstance(batch, ModelBatch) or batch.coefficient_digest != record["batch"] or freeze_tree(_training_observation(state, batch)) != freeze_tree(
                {k:record[k] for k in ("input", "target", "observed")}):
            raise ValueError("retained original observation differs from its complete batch checkpoint")
    plan = _native(state.configuration, source, state.payload["observations"])
    if freeze_tree(plan.receipt) != state.payload["fit"]:
        raise ValueError("retained graded fit differs from complete native chain and evidence replay")
    _remember(state, plan)
    return plan


def _model(source, stalk_axes, fixed_maps, parameter_slots, training_rows=(), *, version, grade,
                                     parameter_space=None, parameter_basis=None, limits=(4096,16000000,64000000,16384),
                                     max_bytes=256*1024*1024, reference=None):
    """Declare one trainable transport grade and fixed maps at every original grade."""
    from rexgraph_cpp import graded_transport_build_info
    limits = tuple(limits)
    if len(limits) != 4 or any(type(v) is not int or v <= 0 for v in limits):
        raise ValueError("four positive action admissions required")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    axes = tuple(_axes(rows) for rows in stalk_axes)
    maps = tuple(tuple(tuple((i,j,_exact(q,limits[3])) for i,j,q in block) for block in family) for family in fixed_maps)
    if isinstance(parameter_space, CoordinateSpace):
        parameter_space = dict(name=parameter_space.name, keys=parameter_space.keys)
    profile, provider, _ = _profile(version)
    c = dict(profile=profile, execution_provider=provider, engine_receipt=graded_transport_build_info(),
        stalk_axes=axes, fixed_maps=maps, parameter_slots=tuple(tuple(s) for s in parameter_slots),
        parameter_space=parameter_space, parameter_basis=None if parameter_basis is None else tuple(tuple(s) for s in parameter_basis),
        training_rows=tuple(training_rows), grade=grade, limits=limits, max_bytes=max_bytes, source_reference=ref.as_record())
    plan = _native(c, source, ())
    state = ModelState("cpp_graded_stalk_transport", version, ref, _space(ref,grade-1), _slots(axes[grade-1]),
        c, dict(observations=(), fit=plan.receipt), "rational")
    enable_graded_stalk_transport_provider(); _remember(state,plan); _prepare(state,source)
    return state


def cpp_graded_stalk_transport_model(source, stalk_axes, fixed_maps, parameter_slots, training_rows=(), *, grade,
                                     parameter_space=None, parameter_basis=None, limits=(4096,16000000,64000000,16384),
                                     max_bytes=256*1024*1024, reference=None):
    """Fit transports from an already strict original boundary tower."""
    return _model(source, stalk_axes, fixed_maps, parameter_slots, training_rows, version="1", grade=grade,
                  parameter_space=parameter_space, parameter_basis=parameter_basis, limits=limits,
                  max_bytes=max_bytes, reference=reference)


def cpp_affine_stalk_transport_model(source, stalk_axes, fixed_maps, parameter_slots, training_rows=(), *, grade,
                                    parameter_space=None, parameter_basis=None, limits=(4096,16000000,64000000,16384),
                                    max_bytes=256*1024*1024, reference=None):
    """Complete declared fixed maps through a native affine family at one grade."""
    return _model(source, stalk_axes, fixed_maps, parameter_slots, training_rows, version="2", grade=grade,
                  parameter_space=parameter_space, parameter_basis=parameter_basis, limits=limits,
                  max_bytes=max_bytes, reference=reference)


def graded_stalk_transport_observation_batch(state, field, annotations):
    """Observe whole lower stalks for a complete field at the learned original grade."""
    validate_graded_stalk_transport_state(state); c = state.configuration
    target = list(_zeros(c,c["grade"]-1)); mask = [False]*len(target)
    for i, vector in annotations:
        if type(i) is not int or i not in c["training_rows"] or mask[i]:
            raise ValueError("unique declared original lower training stalk required")
        target[i] = tuple(_exact(v,c["limits"][3]) for v in vector); mask[i] = True
    return _batch(state,_observation(c,dict(input=field,target=tuple(target),observed=tuple(mask))))


def _train(state, source, batch, *, steps):
    from rexgraph.model_codec import pack_model
    if steps != 1:
        raise ValueError("one original whole-field learning step required")
    plan = _prepare(state,source); observation = _training_observation(state,batch)
    child_plan = plan.append((observation,))
    acquired = dict(observation,batch=batch.coefficient_digest,batch_checkpoint=pack_model(batch))
    child = state.child(payload=dict(observations=(*state.payload["observations"],acquired),fit=child_plan.receipt),step=state.step+1)
    _remember(child,child_plan); _prepare(child,source)
    return child


def graded_stalk_transport_query(state, field=None):
    """Bind a complete field to the selected original input grade."""
    c = state.configuration
    return dict(profile=_QUERY,source_reference=state.source.as_record(),
        field=_zeros(c,c["grade"]) if field is None else _field(c,c["grade"],field))


def _response(state, source, inputs):
    plan = _prepare(state,source); c = state.configuration
    q = freeze_tree(graded_stalk_transport_query(state) if inputs is None else inputs)
    if (not isinstance(q,Mapping) or set(q) != {"profile","source_reference","field"} or q["profile"] != _QUERY or
            dict(q["source_reference"]) != state.source.as_record()):
        raise ValueError("closed graded query bound to its original source required")
    response = plan.apply(q["field"]); lower = c["stalk_axes"][c["grade"]-1]
    used = sorted({j for _,j,_ in response["boundary_directions"]})
    names = response["receipt"]["parameter_space"][1]
    components = ("origin",*("free:"+names[response["free_columns"][j]] for j in used))
    axes = (state.output_axes[0],CoordinateSpace("affine-components",components))
    width = len(axes[0].keys)
    if len(lower)*width*len(components) > c["limits"][1]:
        raise ValueError("complete query response display exceeds entry admission")
    values = np.full((len(lower),width,len(components)),Fraction(0),dtype=object)
    positions = tuple((i,j) for i,row in enumerate(lower) for j in range(len(row)))
    for i,row in enumerate(response["boundary"]):values[i,:len(row),0] = row
    index = {j:i+1 for i,j in enumerate(used)}; unknown = set()
    for r,j,value in response["boundary_directions"]:
        i,axis = positions[r]; values[i,axis,index[j]] = value; unknown.add((i,axis))
    identified = tuple(tuple((i,j) not in unknown for j in range(len(row))) for i,row in enumerate(lower))
    return q,response,axes,values,identified


def _infer(state, source, inputs):
    from rexgraph.model_codec import pack_model
    q,response,axes,values,identified = _response(state,source,inputs)
    receipt = dict(profile=state.configuration["profile"],model_checkpoint=pack_model(state),source_archive=source.archive,
        query_declaration=q,response=response,boundary_identified=identified,readout="complete-graded-affine-family-v1")
    return ModelOutput(values,state.space,axes,state.source,state.coefficient_digest,"rational",_profile(state.adapter_version)[2],
        grade=state.configuration["grade"]-1,variance="chain",execution_receipt=receipt)


def validate_graded_stalk_transport_output(output):
    """Reproduce the complete response family from its original model and native source."""
    from rexgraph_cpp import NativeSource
    from rexgraph.model_codec import unpack_model
    r = output.execution_receipt
    if (not isinstance(r,Mapping) or set(r) != {"profile","model_checkpoint","source_archive","query_declaration",
            "response","boundary_identified","readout"} or r["profile"] not in {_PROFILE, _AFFINE_PROFILE} or r["readout"] != "complete-graded-affine-family-v1"):
        raise ValueError("complete original graded transport output required")
    state = unpack_model(r["model_checkpoint"])
    if r["profile"] != state.configuration["profile"]:
        raise ValueError("graded output profile differs from its model")
    if not isinstance(r["source_archive"],bytes) or len(r["source_archive"]) > state.configuration["max_bytes"]:
        raise ValueError("retained original source archive exceeds admission")
    source = NativeSource(r["source_archive"],max_bytes=state.configuration["max_bytes"])
    q,response,axes,values,identified = _response(state,source,r["query_declaration"])
    if (not output.source.matches(state.source) or output.model_digest != state.coefficient_digest or
            output.arithmetic != "rational" or output.method != _profile(state.adapter_version)[2] or output.space != state.space or
            output.grade != state.configuration["grade"]-1 or output.variance != "chain" or output.axes != axes or
            freeze_tree(response) != r["response"] or freeze_tree(identified) != r["boundary_identified"] or
            not np.array_equal(output.values,values)):
        raise ValueError("graded response differs from complete native family replay")


def graded_stalk_transport_maps(state,source,*,free_parameters=None):
    """Materialize the full strict tower with explicit values for every free parameter."""
    return _prepare(state,source).materialize(free_parameters=free_parameters)


def replay_graded_stalk_transport_model(state, source):
    """Replay original graded observations and certify the complete current chain family."""
    from rexgraph.model_codec import unpack_model
    from rexgraph.model_runtime import train_model
    from rexgraph.cpp_stalk_transport import _receipt_matches
    state.check_state(); validate_graded_stalk_transport_state(state); state.bind(source)
    c = state.configuration
    constructor = cpp_graded_stalk_transport_model if state.adapter_version == "1" else cpp_affine_stalk_transport_model
    current = constructor(source, c["stalk_axes"], c["fixed_maps"], c["parameter_slots"],
        c["training_rows"], grade=c["grade"], parameter_space=c["parameter_space"], parameter_basis=c["parameter_basis"],
        limits=c["limits"], max_bytes=c["max_bytes"], reference=state.source)
    for observation in state.payload["observations"]:
        batch = unpack_model(observation["batch_checkpoint"])
        if (not isinstance(batch, ModelBatch) or batch.coefficient_digest != observation["batch"] or
                freeze_tree(_training_observation(state, batch)) !=
                freeze_tree({k: observation[k] for k in ("input", "target", "observed")})):
            raise ValueError("graded observation differs from its original batch")
        current = train_model(current, source, batch)
    def stable(receipt):
        value = {k: v for k, v in receipt.items() if k not in {"engine_headers", "binding"}}
        value["source"] = {k: v for k, v in receipt["source"].items() if k not in {"engine_headers", "binding"}}
        value["fit"] = {k: v for k, v in receipt["fit"].items() if k not in {"preparation_steps", "retained_entries"}}
        return value
    if not _receipt_matches(stable(state.payload["fit"]), stable(current.payload["fit"])) or not _receipt_matches(
            state.payload["observations"], current.payload["observations"]):
        raise ValueError("current graded replay differs from the complete retained transport family")
    return current


def graded_stalk_transport_field(output, row=None):
    """Expose a named original stalk only when its complete response is identified."""
    from rexgraph.model_codec import unpack_model
    validate_graded_stalk_transport_output(output)
    state = unpack_model(output.execution_receipt["model_checkpoint"]); axes = state.configuration["stalk_axes"][output.grade]
    selected = tuple(range(len(axes))) if row is None else (row,)
    if any(type(i) is not int or not 0 <= i < len(axes) for i in selected):
        raise ValueError("select an original output grade stalk")
    if any(not all(output.execution_receipt["boundary_identified"][i]) for i in selected):
        raise TypeError("unidentified graded responses require their complete affine family")
    names = axes[selected[0]] if selected else ()
    if any(axes[i] != names for i in selected):
        raise ValueError("select one original stalk when named bases differ")
    values = np.asarray([output.values[i,:len(names),0] for i in selected],dtype=object).reshape(len(selected),len(names))
    space = CoordinateSpace(output.space.name,tuple(output.space.keys[i] for i in selected))
    return TensorField(space,values,(CoordinateSpace("named-stalk-axis",names),),output.source,
        grade=output.grade,variance="chain",provenance=(output.model_digest,output.coefficient_digest))


def _create(source, *, adapter="cpp_graded_stalk_transport", **configuration):
    if adapter != "cpp_graded_stalk_transport":
        raise ValueError("unsupported graded transport adapter")
    return cpp_graded_stalk_transport_model(source,**configuration)


def _create_affine(source, *, adapter="cpp_graded_stalk_transport", **configuration):
    if adapter != "cpp_graded_stalk_transport":
        raise ValueError("unsupported graded transport adapter")
    return cpp_affine_stalk_transport_model(source, **configuration)


def enable_graded_stalk_transport_provider(version="1"):
    """Register whole field learning with complete native transport chain constraints."""
    global _provider, _affine_provider
    _profile(version)
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"],_PROVIDER["version"],(("cpp_graded_stalk_transport","1"),),
                _infer,create=_create,prepare=_prepare,train=_train)
        if _affine_provider is None:
            _affine_provider = ModelProvider(_AFFINE_PROVIDER["name"],_AFFINE_PROVIDER["version"],
                (("cpp_graded_stalk_transport","2"),), _infer, create=_create_affine, prepare=_prepare, train=_train)
        register_model_provider(_provider)
        register_model_provider(_affine_provider)
        return _provider if version == "1" else _affine_provider
