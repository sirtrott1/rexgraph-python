"""Exact affine learning of supported original incidence transport maps."""
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock
from fractions import Fraction

import numpy as np

from rexgraph.cpp_stalk_fields import _axes, _exact
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "acquired-affine-stalk-transports-v1"
_METHOD = "exact-affine-stalk-transport-v1"
_PROVIDER = {"name": "rexgraph.cpp.stalk_transport", "version": "1"}
_TIED_PROFILE = "acquired-tied-stalk-transports-v2"
_TIED_METHOD = "exact-tied-stalk-transport-v2"
_TIED_PROVIDER = {"name": "rexgraph.cpp.stalk_transport", "version": "2"}
_COLUMN_PROFILE = "acquired-column-stalk-transports-v3"
_COLUMN_METHOD = "exact-column-stalk-transport-v3"
_COLUMN_PROVIDER = {"name": "rexgraph.cpp.stalk_transport", "version": "3"}
_TIED_COLUMN_PROFILE = "acquired-tied-column-stalk-transports-v4"
_TIED_COLUMN_METHOD = "exact-tied-column-stalk-transport-v4"
_TIED_COLUMN_PROVIDER = {"name": "rexgraph.cpp.stalk_transport", "version": "4"}
_PROFILES = {"1": _PROFILE, "2": _TIED_PROFILE, "3": _COLUMN_PROFILE, "4": _TIED_COLUMN_PROFILE}
_METHODS = {"1": _METHOD, "2": _TIED_METHOD, "3": _COLUMN_METHOD, "4": _TIED_COLUMN_METHOD}
_PROVIDERS = {"1": _PROVIDER, "2": _TIED_PROVIDER, "3": _COLUMN_PROVIDER, "4": _TIED_COLUMN_PROVIDER}
_GROUP_PROFILE = "whole-stalk-transport-observation-group-v1"
_lock = RLock()
_cache = OrderedDict()
_provider = None
_tied_provider = None


def _space(reference):
    return CoordinateSpace("retained-stalk-grade:0", reference.native.coordinates[0])


def _slots(axes):
    return (CoordinateSpace("stalk-slot", tuple(str(i) for i in range(max(map(len, axes), default=0)))),)


def _field(c, grade, rows):
    values = tuple(tuple(_exact(v, c["limits"][3]) for v in row) for row in rows)
    if tuple(map(len, values)) != tuple(map(len, c["stalk_axes"][grade])):
        raise ValueError("complete fields on the original named stalks required")
    return values


def _zeros(c, grade):
    return tuple((Fraction(0),)*len(row) for row in c["stalk_axes"][grade])


def _observation(c, observation):
    if not isinstance(observation, Mapping) or set(observation) != {"input", "target", "observed"}:
        raise ValueError("closed original whole-field observation required")
    field = _field(c, 1, observation["input"]); target = _field(c, 0, observation["target"])
    mask = observation["observed"]
    if (len(mask) != len(target) or any(type(v) is not bool for v in mask) or not any(mask) or
            any(any(row) for row, seen in zip(target, mask, strict=True) if not seen)):
        raise ValueError("observed original lower stalks and zero unobserved targets required")
    return dict(input=field, target=target, observed=tuple(mask))


def _parameter_family(c):
    record = c["parameter_space"]
    if not isinstance(record, Mapping) or set(record) != {"name", "keys"}:
        raise ValueError("closed named transport parameter space required")
    space = CoordinateSpace(record["name"], record["keys"])
    if (not space.keys or len(space.keys)+1 > c["limits"][0] or len(space.name.encode()) > 256 or
            any(len(k.encode()) > 256 for k in space.keys)):
        raise ValueError("bounded nonempty named transport parameter coordinates required")
    seen = set(); basis = []
    for row, col, value in c["parameter_basis"]:
        if (type(row) is not int or type(col) is not int or not 0 <= row < len(c["parameter_slots"]) or
                not 0 <= col < len(space.keys) or (row,col) in seen):
            raise ValueError("distinct original entry and named parameter basis coordinates required")
        value = _exact(value, c["limits"][3])
        if not value:
            raise ValueError("sparse parameter basis entries must be nonzero")
        seen.add((row,col)); basis.append((row,col,value))
        if len(basis) > c["limits"][1]:
            raise ValueError("parameter basis exceeds admission")
    return space, tuple(basis)


def validate_stalk_transport_state(state):
    """Check original parameter support, whole observations and affine fit coordinates."""
    c = state.configuration
    tied = state.adapter_version in {"2", "4"}
    columns = state.adapter_version in {"3", "4"}
    fields = {"profile", "execution_provider", "engine_receipt", "stalk_axes", "fixed_maps",
              "parameter_slots", "training_rows", "limits", "max_bytes", "source_reference", "source_accession"}
    if tied: fields.update(("parameter_space", "parameter_basis"))
    if (state.adapter != "cpp_stalk_transport" or state.adapter_version not in _PROFILES or state.arithmetic != "rational" or
            state.dependencies or
            not isinstance(state.source, CppSourceReference) or
            set(c) != fields or c["profile"] != _PROFILES[state.adapter_version] or
            dict(c["execution_provider"]) != _PROVIDERS[state.adapter_version]):
        raise ValueError("closed exact original stalk transport model required")
    if (len(c["limits"]) != 4 or any(type(v) is not int or v <= 0 for v in c["limits"]) or
            type(c["max_bytes"]) is not int or c["max_bytes"] <= 0):
        raise ValueError("positive action and source byte admissions required")
    axes = tuple(_axes(family) for family in c["stalk_axes"])
    if (len(axes) < 2 or len(axes) != len(state.source.native.coordinates) or any(axes[2:]) or
            tuple(map(len, axes)) != tuple(map(len, state.source.native.coordinates)) or
            state.space != _space(state.source) or state.output_axes != _slots(axes[0]) or
            dict(c["source_reference"]) != state.source.as_record()):
        raise ValueError("transport model requires its original grade-one source and complete named stalks")
    if sum(len(row) for family in axes for row in family) > c["limits"][0]:
        raise ValueError("whole original stalk dimensions exceed admission")
    if c["source_accession"] is not None and not isinstance(c["source_accession"], Mapping):
        raise ValueError("original source accession must be a declared mapping")
    maps = c["fixed_maps"]
    if len(maps) != len(axes) or maps[0] or any(maps[2:]):
        raise ValueError("complete original grade-one flag declarations required")
    support = set()
    for slot, block in enumerate(maps[1]):
        for row, col, value in block:
            if (type(row) is not int or type(col) is not int or row < 0 or col < 0 or (slot, row, col) in support):
                raise ValueError("distinct natural fixed map coordinates required")
            support.add((slot, row, col)); _exact(value, c["limits"][3])
    parameters = c["parameter_slots"]
    if len(parameters)+1 > c["limits"][0]:
        raise ValueError("parameter dimensions exceed admission")
    for slot in parameters:
        if (len(slot) != 3 or any(type(v) is not int or v < 0 for v in slot) or slot in support):
            raise ValueError("distinct supported parameter slots disjoint from fixed entries required")
        support.add(slot)
    parameter_count = len(parameters)
    if tied:
        parameter_space, basis = _parameter_family(c)
        parameter_count = len(parameter_space.keys)
    training = c["training_rows"]
    if len(set(training)) != len(training) or any(type(v) is not int or not 0 <= v < len(axes[0]) for v in training):
        raise ValueError("distinct original lower training stalks required")
    p = state.payload
    if set(p) != {"observations", "fit"} or len(p["observations"]) != state.step:
        raise ValueError("complete whole-field acquisition history required")
    if len(p["observations"]) > c["limits"][0]:
        raise ValueError("observation count exceeds admission")
    entries = len(support)*3+sum(map(len, maps))
    if tied: entries += len(basis)+parameter_count
    constraints = 0
    if columns:
        from rexgraph.cpp_column_transport import validate_column_history
        acquired, constraints = validate_column_history(state)
        entries += acquired
    else:
        for o in p["observations"]:
            if not isinstance(o, Mapping) or set(o) != {"input", "target", "observed", "batch", "batch_checkpoint"}:
                raise ValueError("whole observation and original batch identity required")
            _observation(c, {k: o[k] for k in ("input", "target", "observed")})
            if any(seen and row not in training for row, seen in enumerate(o["observed"])):
                raise ValueError("observation lies outside declared training stalks")
            if not isinstance(o["batch"], str) or len(o["batch"]) != 64:
                raise ValueError("original observation batch digest required")
            entries += sum(map(len, o["input"]))+sum(map(len, o["target"]))+len(o["observed"])
            constraints += sum(len(row) for row, seen in zip(axes[0], o["observed"], strict=True) if seen)
    if entries > c["limits"][1] or constraints > c["limits"][0]:
        raise ValueError("whole observation coefficients exceed admission")
    fit = p["fit"]
    if (not isinstance(fit, Mapping) or fit.get("parameter_slots") != parameters or
            len(fit.get("particular", ())) != parameter_count or
            fit.get("rank") != len(fit.get("pivots", ())) or
            len(fit.get("free_columns", ()))+fit["rank"] != parameter_count or
            fit.get("kernel_shape") != (parameter_count,len(fit["free_columns"]))):
        raise ValueError("complete affine fit on original parameter coordinates required")
    if tied and (fit.get("parameter_space") != (parameter_space.name,parameter_space.keys) or
            fit.get("parameter_basis") != basis or fit.get("basis_shape") != (len(parameters),parameter_count) or
            type(fit.get("basis_rank")) is not int or not 0 <= fit["basis_rank"] <= min(len(parameters),parameter_count) or
            fit.get("parameterization") != "named-original-flag-templates-v2"):
        raise ValueError("complete retained named incidence templates required")
    seen = set()
    for row,col,value in fit["kernel"]:
        if (type(row) is not int or type(col) is not int or not 0 <= row < parameter_count or
                not 0 <= col < len(fit["free_columns"]) or (row,col) in seen or not _exact(value,c["limits"][3])):
            raise ValueError("distinct nonzero sparse affine kernel entries required")
        seen.add((row,col))


def _batch(state, observation):
    c = state.configuration; axes = c["stalk_axes"][0]
    target = np.full((len(axes), len(state.output_axes[0].keys)), Fraction(0), dtype=object)
    for row, values in enumerate(observation["target"]):
        target[row, :len(values)] = values
    return ModelBatch(state.source, state.space, target, np.asarray(observation["observed"], dtype=bool),
        dict(profile="whole-stalk-transport-observation-v1", partition="training", input=observation["input"]))


def _native(c, source, observations):
    from rexgraph_cpp import PreparedStalkTransport
    if c["source_accession"] is not None:
        from rexgraph.linguistic_accession import validate_linguistic_accession
        validate_linguistic_accession(c["source_accession"], source, max_bytes=c["max_bytes"])
    extra = {k:c[k] for k in ("parameter_space","parameter_basis")} if c["profile"] in {_TIED_PROFILE, _TIED_COLUMN_PROFILE} else {}
    if c["profile"] in {_COLUMN_PROFILE, _TIED_COLUMN_PROFILE}:
        plan = PreparedStalkTransport(source.archive,c["stalk_axes"],c["fixed_maps"],c["parameter_slots"],
            limits=c["limits"],max_bytes=c["max_bytes"],column_fields=True,**extra)
        return plan.append_columns(tuple({k:o[k] for k in ("input","target")} for o in observations)) if observations else plan
    return PreparedStalkTransport(source.archive, c["stalk_axes"], c["fixed_maps"], c["parameter_slots"],
        tuple({k: o[k] for k in ("input", "target", "observed")} for o in observations),
        limits=c["limits"], max_bytes=c["max_bytes"], **extra)


def _training_observation(state, batch):
    c = state.configuration; b = batch.inputs
    batch.check_state()
    if (batch.space != state.space or not batch.source.matches(state.source) or batch.dependencies or
            not isinstance(b, Mapping) or set(b) != {"profile", "partition", "input"} or
            b["profile"] != "whole-stalk-transport-observation-v1" or b["partition"] != "training" or
            batch.targets.shape != (len(c["stalk_axes"][0]), len(state.output_axes[0].keys))):
        raise ValueError("one declared whole-field training observation required")
    target = []
    for row, names in zip(batch.targets, c["stalk_axes"][0], strict=True):
        values = tuple(_exact(v, c["limits"][3]) for v in row)
        if any(values[len(names):]):
            raise ValueError("padding lies outside the original stalk")
        target.append(values[:len(names)])
    o = _observation(c, dict(input=b["input"], target=tuple(target), observed=tuple(bool(v) for v in batch.observed)))
    if any(seen and row not in c["training_rows"] for row, seen in enumerate(o["observed"])):
        raise ValueError("observation lies outside declared original training stalks")
    return o


def _remember(state, plan):
    with _lock:
        _cache[state.coefficient_digest] = plan
        while len(_cache) > 4:
            _cache.popitem(last=False)


def _receipt_matches(expected, retained):
    """Compare closed native receipts with exact scalar kinds and coordinates."""
    if isinstance(expected, Mapping):
        return (isinstance(retained, Mapping) and len(expected) == len(retained) and
                all(type(k) is str for k in retained) and
                all(k in retained and _receipt_matches(v, retained[k]) for k, v in expected.items()))
    if type(expected) is tuple:
        return (type(retained) is tuple and len(expected) == len(retained) and
                all(_receipt_matches(v, old) for v, old in zip(expected, retained, strict=True)))
    if type(expected) is Fraction:
        return (type(retained) is Fraction and type(retained.numerator) is int and
                type(retained.denominator) is int and expected == retained)
    if type(expected) is np.ndarray:
        if type(retained) is not np.ndarray or expected.dtype != retained.dtype or expected.shape != retained.shape:
            return False
        if expected.dtype.hasobject:
            return all(_receipt_matches(a, b) for a, b in zip(expected.flat, retained.flat, strict=True))
        return expected.tobytes(order="C") == retained.tobytes(order="C")
    return type(expected) in {type(None), bool, int, str, bytes} and type(retained) is type(expected) and expected == retained


def _prepare(state, source):
    from rexgraph_cpp import stalk_transport_build_info
    state.bind(source)
    c = state.configuration
    if dict(c["engine_receipt"]) != stalk_transport_build_info(int(state.adapter_version)):
        raise ValueError("transport model requires its retained native implementation")
    digest = state._digest
    with _lock:
        plan = _cache.get(digest)
        if plan is None:
            validate_stalk_transport_state(state)
            from rexgraph.model_codec import unpack_model
            if state.adapter_version in {"3", "4"}:
                from rexgraph.cpp_column_transport import verify_column_history
                verify_column_history(state)
            else:
                for o in state.payload["observations"]:
                    batch = unpack_model(o["batch_checkpoint"])
                    if (not isinstance(batch, ModelBatch) or batch.coefficient_digest != o["batch"] or
                            freeze_tree(_training_observation(state, batch)) != freeze_tree({k:o[k] for k in ("input", "target", "observed")})):
                        raise ValueError("transport observation differs from its original batch")
            plan = _native(c, source, state.payload["observations"])
            if not _receipt_matches(plan.receipt, state.payload["fit"]):
                raise ValueError("transport fit differs from exact native observation replay")
            _cache[digest] = plan
            while len(_cache) > 4:
                _cache.popitem(last=False)
        _cache.move_to_end(digest)
        return plan


def cpp_stalk_transport_model(source, stalk_axes, fixed_maps, parameter_slots, training_rows=(), *,
                              limits=(4096, 16000000, 64000000, 16384), max_bytes=256*1024*1024,
                              source_accession=None, reference=None, parameter_space=None, parameter_basis=None, column_fields=False):
    """Declare fixed maps and independent or named sparse parameter fields on original flags."""
    from rexgraph_cpp import stalk_transport_build_info
    limits = tuple(limits)
    if len(limits) != 4 or any(type(v) is not int or v <= 0 for v in limits):
        raise ValueError("four positive action admissions required")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    axes = tuple(_axes(family) for family in stalk_axes)
    if len(axes) < 2:
        raise ValueError("original lower and upper stalk families required")
    maps = tuple(tuple(tuple((r, col, _exact(v, limits[3])) for r, col, v in block) for block in family) for family in fixed_maps)
    tied = parameter_space is not None
    if type(column_fields) is not bool:raise TypeError("boolean column field profile required")
    if not tied and parameter_basis is not None:
        raise ValueError("named parameter space required with a sparse basis")
    version = "4" if column_fields and tied else "3" if column_fields else "2" if tied else "1"
    c = dict(profile=_PROFILES[version], execution_provider=_PROVIDERS[version],
        engine_receipt=stalk_transport_build_info(int(version)),
        stalk_axes=axes, fixed_maps=maps, parameter_slots=tuple(tuple(s) for s in parameter_slots),
        training_rows=tuple(training_rows), limits=limits, max_bytes=max_bytes, source_reference=ref.as_record(),
        source_accession=freeze_tree(source_accession))
    if tied:
        if isinstance(parameter_space, CoordinateSpace):
            parameter_space = dict(name=parameter_space.name,keys=parameter_space.keys)
        if parameter_basis is None:
            raise ValueError("named parameter space and sparse basis must be declared together")
        c.update(parameter_space=parameter_space,parameter_basis=tuple(tuple(s) for s in parameter_basis))
        space, basis = _parameter_family(c)
        c.update(parameter_space=dict(name=space.name,keys=space.keys),parameter_basis=basis)
    plan = _native(c, source, ())
    state = ModelState("cpp_stalk_transport", version, ref, _space(ref), _slots(axes[0]), c,
        dict(observations=(), fit=plan.receipt), "rational")
    enable_stalk_transport_provider(state.adapter_version); _remember(state, plan); _prepare(state, source)
    return state


def stalk_transport_observation_batch(state, field, annotations):
    """Observe complete lower stalk vectors for one complete upper field."""
    validate_stalk_transport_state(state); c = state.configuration
    target = list(_zeros(c, 0)); mask = [False]*len(target)
    for row, vector in annotations:
        if type(row) is not int or row not in c["training_rows"] or mask[row]:
            raise ValueError("unique declared original training stalk required")
        target[row] = tuple(_exact(v, c["limits"][3]) for v in vector); mask[row] = True
    o = _observation(c, dict(input=field, target=tuple(target), observed=tuple(mask)))
    return _batch(state, o)


def _group_checkpoint_entries(checkpoints, limit):
    entries = 0
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping) or not checkpoint:
            raise TypeError("complete original batch checkpoint required")
        for tensor in checkpoint.values():
            if type(tensor) is not np.ndarray:
                raise TypeError("original checkpoint tensors required")
            entries += tensor.size
            if entries > limit:
                raise ValueError("observation group exceeds retained entry admission")
    return entries


def _group_admission(state, count, checkpoint_entries):
    c = state.configuration
    if not count or state.step+count > c["limits"][0]:
        raise ValueError("nonempty admitted original observation group required")
    entries = len(state.space.keys)*(1+count*len(state.output_axes[0].keys))+checkpoint_entries
    if entries > c["limits"][1]:
        raise ValueError("observation group exceeds retained entry admission")


def stalk_transport_observation_group(state, batches):
    """Group ordered original batches for one joint native fit and child checkpoint."""
    from rexgraph.model_codec import pack_model
    if not isinstance(state, ModelState) or state.adapter != "cpp_stalk_transport":
        raise TypeError("original transport ModelState required")
    state.check_state(); validate_stalk_transport_state(state)
    if isinstance(batches, (str, bytes, Mapping)):
        raise TypeError("ordered original whole-field batches required")
    checkpoints = []; targets = []; masks = []; entries = 0
    for batch in batches:
        if state.step+len(checkpoints) >= state.configuration["limits"][0]:
            raise ValueError("observation count exceeds admission")
        if not isinstance(batch, ModelBatch):
            raise TypeError("original whole-field ModelBatch required")
        _training_observation(state, batch)
        checkpoint = pack_model(batch)
        entries += _group_checkpoint_entries((checkpoint,), state.configuration["limits"][1])
        _group_admission(state, len(checkpoints)+1, entries)
        checkpoints.append(checkpoint)
        targets.append(batch.targets); masks.append(batch.observed)
    _group_admission(state, len(checkpoints), entries)
    return ModelBatch(state.source, state.space, np.stack(targets, axis=1), np.any(masks, axis=0),
        dict(profile=_GROUP_PROFILE, partition="training", batches=tuple(checkpoints)))


def _training_group(state, batch):
    from rexgraph.model_codec import unpack_model
    b = batch.inputs; c = state.configuration
    batch.check_state()
    if (batch.space != state.space or not batch.source.matches(state.source) or batch.dependencies or
            batch.target_input is not None or not isinstance(b, Mapping) or
            set(b) != {"profile", "partition", "batches"} or b["profile"] != _GROUP_PROFILE or
            b["partition"] != "training" or type(b["batches"]) is not tuple):
        raise ValueError("closed original whole-field observation group required")
    count = len(b["batches"])
    _group_admission(state, count, 0)
    _group_admission(state, count, _group_checkpoint_entries(b["batches"], c["limits"][1]))
    if batch.targets.shape != (len(state.space.keys), count, len(state.output_axes[0].keys)):
        raise ValueError("group targets require original rows, observation order and stalk slots")
    records = []; union = np.zeros(len(state.space.keys), dtype=bool)
    for i, checkpoint in enumerate(b["batches"]):
        original = unpack_model(checkpoint)
        if not isinstance(original, ModelBatch):
            raise TypeError("group checkpoint must contain an original ModelBatch")
        o = _training_observation(state, original)
        for row, values in enumerate(o["target"]):
            retained = tuple(_exact(value, c["limits"][3]) for value in batch.targets[row, i])
            expected = values+(Fraction(0),)*(len(state.output_axes[0].keys)-len(values))
            if retained != expected:
                raise ValueError("group targets differ from their original batch checkpoint")
        union |= original.observed
        records.append(dict(o, batch=original.coefficient_digest, batch_checkpoint=checkpoint))
    if not np.array_equal(batch.observed, union):
        raise ValueError("group observation mask differs from its original batches")
    return tuple(records)


def _train(state, source, batch, *, steps):
    from rexgraph.model_codec import pack_model
    parent_plan = _prepare(state, source)
    if isinstance(batch.inputs, Mapping) and batch.inputs.get("profile") == _GROUP_PROFILE:
        acquired = _training_group(state, batch)
        if steps != len(acquired):
            raise ValueError("group learning steps must equal its original observation count")
    else:
        if steps != 1:
            raise ValueError("one declared whole-field training observation required")
        o = _training_observation(state, batch)
        acquired = (dict(o, batch=batch.coefficient_digest, batch_checkpoint=pack_model(batch)),)
    observations = (*state.payload["observations"], *acquired)
    plan = parent_plan.append(tuple({k: o[k] for k in ("input", "target", "observed")} for o in acquired))
    child = state.child(payload=dict(observations=observations, fit=plan.receipt), step=state.step+len(acquired))
    _remember(child, plan); _prepare(child, source)
    return child


def stalk_transport_query(state, field=None):
    """Bind a complete upper field without selecting unidentified map parameters."""
    return dict(profile="whole-stalk-transport-query-v1", source_reference=state.source.as_record(),
        field=_zeros(state.configuration, 1) if field is None else _field(state.configuration, 1, field))


def stalk_transport_maps(state, source, *, free_parameters=None):
    """Materialize complete maps with explicit choices for any unidentified parameters."""
    return freeze_tree(_prepare(state, source).materialize(free_parameters))


def replay_stalk_transport_model(state, source):
    """Create a current implementation checkpoint by replaying original observations."""
    from rexgraph.model_codec import unpack_model
    from rexgraph.model_runtime import train_model
    state.check_state(); validate_stalk_transport_state(state); state.bind(source)
    if state.adapter_version in {"3", "4"}:
        from rexgraph.cpp_column_transport import replay_column_model
        return replay_column_model(state,source)
    c = state.configuration
    extra = {k: c[k] for k in ("parameter_space", "parameter_basis")} if state.adapter_version == "2" else {}
    current = cpp_stalk_transport_model(source, c["stalk_axes"], c["fixed_maps"], c["parameter_slots"],
        c["training_rows"], limits=c["limits"], max_bytes=c["max_bytes"], source_accession=c["source_accession"],
        reference=state.source, **extra)
    batches = []
    for observation in state.payload["observations"]:
        batch = unpack_model(observation["batch_checkpoint"])
        if (not isinstance(batch, ModelBatch) or batch.coefficient_digest != observation["batch"] or
                freeze_tree(_training_observation(state, batch)) !=
                freeze_tree({k: observation[k] for k in ("input", "target", "observed")})):
            raise ValueError("transport observation differs from its original batch")
        batches.append(batch)
    if batches:
        current = train_model(current, source, stalk_transport_observation_group(current, batches), steps=len(batches))
    changing = {"engine_headers", "binding", "preparation_steps", "retained_entries"}
    original = {k: v for k, v in state.payload["fit"].items() if k not in changing}
    replayed = {k: v for k, v in current.payload["fit"].items() if k not in changing}
    if not _receipt_matches(original, replayed) or not _receipt_matches(
            state.payload["observations"], current.payload["observations"]):
        raise ValueError("current native replay differs from the complete retained transport family")
    return current


def _query(state, query):
    q = freeze_tree(stalk_transport_query(state) if query is None else query)
    if (not isinstance(q, Mapping) or set(q) != {"profile", "source_reference", "field"} or
            q["profile"] != "whole-stalk-transport-query-v1" or dict(q["source_reference"]) != state.source.as_record()):
        raise ValueError("closed query bound to the actual transport source required")
    _field(state.configuration, 1, q["field"])
    return q


def _response(state, source, query):
    plan = _prepare(state, source); q = _query(state, query); response = plan.apply(q["field"])
    axes = (*state.output_axes, CoordinateSpace("affine-component", response["components"]))
    names = state.configuration["stalk_axes"][0]; width = len(axes[0].keys)
    values = np.full((len(names), width, len(response["components"])), Fraction(0), dtype=object)
    for component, family in enumerate(response["boundary_family"]):
        for row, vector in enumerate(family):
            values[row, :len(vector), component] = vector
    identified = tuple(tuple(all(family[row][col] == 0 for family in response["boundary_family"][1:])
        for col in range(len(local))) for row, local in enumerate(names))
    occurrence_identified = tuple(tuple(all(family[row][col] == 0 for family in response["occurrence_family"][1:])
        for col in range(len(local))) for row, local in enumerate(response["occurrence_family"][0]))
    return q, response, axes, values, identified, occurrence_identified


def _infer(state, source, inputs):
    from rexgraph.model_codec import _model_checkpoint
    if isinstance(inputs, Mapping) and inputs.get("profile") in (
            "whole-stalk-parameter-relations-query-v1", "whole-stalk-context-hypotheses-query-v1",
            "whole-stalk-context-views-query-v1", "whole-stalk-context-diagrams-query-v1",
            "whole-stalk-context-rules-query-v1", "whole-stalk-context-view-rules-query-v1",
            "whole-stalk-observed-context-view-rules-query-v1", "whole-stalk-declared-template-view-rules-query-v1"):
        from rexgraph.cpp_stalk_parameter_relations import _infer as infer_relations
        return infer_relations(state, source, inputs)
    q, response, axes, values, identified, occurrence_identified = _response(state, source, inputs)
    receipt = dict(profile=_PROFILES[state.adapter_version], model_checkpoint=_model_checkpoint(state), source_archive=source.archive,
        query_declaration=q, response=response,
        boundary_identified=identified, occurrence_identified=occurrence_identified,
        stalk_support=tuple(tuple(i < len(row) for i in range(len(axes[0].keys))) for row in state.configuration["stalk_axes"][0]),
        readout="whole-stalk-affine-family-v1")
    return ModelOutput(values, state.space, axes, state.source, state.coefficient_digest,
        "rational", _METHODS[state.adapter_version], grade=0, variance="chain", execution_receipt=receipt)


def validate_stalk_transport_output(output):
    """Replay the complete field family and its separate original occurrence responses."""
    from rexgraph.model_codec import _checkpoint_model
    from rexgraph_cpp import NativeSource
    r = output.execution_receipt
    version = next((v for v, method in _METHODS.items() if method == output.method), None)
    if (not isinstance(r, Mapping) or set(r) != {"profile", "model_checkpoint", "source_archive", "query_declaration", "response",
            "boundary_identified", "occurrence_identified", "stalk_support", "readout"} or
            version is None or r["profile"] != _PROFILES[version] or r["readout"] != "whole-stalk-affine-family-v1"):
        raise ValueError("complete original affine response receipt required")
    state = _checkpoint_model(r["model_checkpoint"])
    if (state.adapter != "cpp_stalk_transport" or state.adapter_version != version or
            output.arithmetic != "rational" or output.dependencies or
            output.method != _METHODS[version] or output.grade != 0 or output.variance != "chain" or
            not output.source.matches(state.source) or output.model_digest != state.coefficient_digest or output.space != state.space):
        raise ValueError("affine response differs from its source and model")
    if not isinstance(r["source_archive"], bytes) or len(r["source_archive"]) > state.configuration["max_bytes"]:
        raise ValueError("bounded original native source archive required")
    source = NativeSource(r["source_archive"], max_bytes=state.configuration["max_bytes"])
    _, response, axes, values, identified, occurrence_identified = _response(state, source, r["query_declaration"])
    support = tuple(tuple(i < len(row) for i in range(len(axes[0].keys))) for row in state.configuration["stalk_axes"][0])
    if (axes != output.axes or not np.array_equal(values, output.values) or not _receipt_matches(response, r["response"]) or
            not _receipt_matches(identified, r["boundary_identified"]) or
            not _receipt_matches(occurrence_identified, r["occurrence_identified"]) or
            not _receipt_matches(support, r["stalk_support"])):
        raise ValueError("affine response differs from exact native field replay")


def stalk_transport_field(output, row=None):
    """Expose a single boundary field only when every supported coefficient is identified."""
    output.check_state(); validate_stalk_transport_output(output)
    r = output.execution_receipt
    from rexgraph.model_codec import _checkpoint_model
    names = _checkpoint_model(r["model_checkpoint"]).configuration["stalk_axes"][0]
    if row is not None and (type(row) is not int or not 0 <= row < len(names)):
        raise ValueError("select an original lower stalk row")
    selected = range(len(names)) if row is None else (row,)
    if not all(all(r["boundary_identified"][i]) for i in selected):
        raise TypeError("boundary field still depends on unidentified transport directions")
    if row is None:
        local = names[0] if names else ()
        if any(n != local for n in names):
            raise TypeError("ragged stalks require an explicit original lower row")
        space = output.space; values = output.values[:, :len(local), 0]
    else:
        local = names[row]; space = CoordinateSpace(output.space.name, (output.space.keys[row],))
        values = output.values[row:row+1, :len(local), 0]
    return TensorField(space, values, (CoordinateSpace("named-stalk-axis", local),), output.source, grade=0, variance="chain",
        provenance=(output.model_digest, output.coefficient_digest))


def _create(source, *, adapter="cpp_stalk_transport", **configuration):
    if adapter != "cpp_stalk_transport":
        raise ValueError("unsupported original stalk transport adapter")
    if configuration.get("parameter_space") is not None:
        raise ValueError("named templates require transport provider version 2")
    if configuration.get("column_fields", False):
        raise ValueError("encoded columns require transport provider version 3 or 4")
    return cpp_stalk_transport_model(source, **configuration)


def _create_tied(source, *, adapter="cpp_stalk_transport", **configuration):
    if adapter != "cpp_stalk_transport" or configuration.get("parameter_space") is None:
        raise ValueError("transport provider version 2 requires named templates")
    if configuration.get("column_fields", False):
        raise ValueError("named encoded columns require transport provider version 4")
    return cpp_stalk_transport_model(source, **configuration)


def enable_stalk_transport_provider(version="1"):
    """Register whole field learning and exact affine family inference."""
    if version in {"3", "4"}:
        from rexgraph.cpp_column_transport import enable_column_transport_provider
        return enable_column_transport_provider(version)
    if version not in {"1","2"}:
        raise ValueError("supported transport provider version required")
    global _provider, _tied_provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "1", (("cpp_stalk_transport", "1"),),
                _infer, create=_create, prepare=_prepare, train=_train)
            register_model_provider(_provider)
        if _tied_provider is None:
            _tied_provider = ModelProvider(_TIED_PROVIDER["name"], "2", (("cpp_stalk_transport", "2"),),
                _infer, create=_create_tied, prepare=_prepare, train=_train)
            register_model_provider(_tied_provider)
        return _tied_provider if version == "2" else _provider
