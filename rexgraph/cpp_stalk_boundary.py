"""Exact acquired fields and boundary traces on original graded stalks."""
from collections import OrderedDict
from collections.abc import Mapping
from fractions import Fraction
from threading import RLock

import numpy as np

from rexgraph.cpp_stalk_fields import _axes, _exact, _acquisition
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.tensor_field import TensorField
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "acquired-graded-stalk-boundary-v1"
_METHOD = "exact-graded-stalk-boundary-v1"
_PROVIDER = {"name": "rexgraph.cpp.stalk_boundary", "version": "1"}
_lock = RLock()
_cache = OrderedDict()
_checked = OrderedDict()
_provider = None


def _space(reference, grade):
    return CoordinateSpace("retained-stalk-grade:"+str(grade), reference.native.coordinates[grade])


def _slots(axes):
    return (CoordinateSpace("stalk-slot", tuple(str(i) for i in range(max(map(len, axes), default=0)))),)


def _field(c, grade, rows):
    result = tuple(tuple(_exact(v, c["limits"][3]) for v in row) for row in rows)
    if tuple(map(len, result)) != tuple(map(len, c["stalk_axes"][grade])):
        raise ValueError("complete fields on the original named grade stalks required")
    return result


def _zeros(c, grade):
    return tuple((Fraction(0),)*len(row) for row in c["stalk_axes"][grade])


def _acquisition_config(c):
    return dict(upper_axes=c["stalk_axes"][c["grade"]], limits=c["limits"], max_bytes=c["max_bytes"])


def validate_stalk_boundary_state(state):
    """Check original grade carriers, complete declarations and acquisition history."""
    c = state.configuration
    if (state.adapter != "cpp_stalk_boundary" or state.adapter_version != "1" or
            state.arithmetic != "rational" or not isinstance(state.source, CppSourceReference) or
            set(c) != {"profile", "execution_provider", "engine_receipt", "stalk_axes", "flag_maps",
                       "training_rows", "grade", "boundary_steps", "limits", "max_bytes", "source_reference"} or
            c["profile"] != _PROFILE or dict(c["execution_provider"]) != _PROVIDER):
        raise ValueError("closed exact graded stalk boundary model required")
    axes = tuple(_axes(rows) for rows in c["stalk_axes"])
    if (len(c["limits"]) != 4 or any(type(v) is not int or v <= 0 for v in c["limits"]) or
            type(c["max_bytes"]) is not int or c["max_bytes"] <= 0 or
            type(c["grade"]) is not int or not 0 <= c["grade"] < len(axes) or
            type(c["boundary_steps"]) is not int or not 0 <= c["boundary_steps"] <= c["grade"]):
        raise ValueError("positive admissions and original adjacent boundary grades required")
    grade = c["grade"]; target = grade-c["boundary_steps"]
    if (len(axes) != len(state.source.native.coordinates) or
            tuple(map(len, axes)) != tuple(map(len, state.source.native.coordinates)) or
            state.space != _space(state.source, grade) or state.output_axes != _slots(axes[target]) or
            dict(c["source_reference"]) != state.source.as_record()):
        raise ValueError("stalk model differs from its actual source grade carriers")
    if sum(len(row) for family in axes for row in family) > c["limits"][0]:
        raise ValueError("all grade stalk dimensions exceed admission")
    maps = c["flag_maps"]
    if len(maps) != len(axes) or maps[0]:
        raise ValueError("complete maps and an empty grade-zero occurrence family required")
    if sum(1+len(block) for family in maps for block in family) > c["limits"][1]:
        raise ValueError("original flag declarations exceed admission")
    for family in maps:
        for block in family:
            seen = set()
            for row, column, value in block:
                if (type(row) is not int or type(column) is not int or row < 0 or column < 0 or
                        (row, column) in seen):
                    raise ValueError("distinct natural flag map coordinates required")
                seen.add((row, column)); _exact(value, c["limits"][3])
    training = c["training_rows"]
    if len(set(training)) != len(training) or any(type(i) is not int or not 0 <= i < len(axes[grade]) for i in training):
        raise ValueError("training rows must name distinct original input grade cells")
    p = state.payload
    if set(p) != {"observations", "batches", "acquisition_archive"} or len(p["batches"]) != state.step:
        raise ValueError("original acquisition columns and complete batch history required")
    grouped = [[] for _ in p["batches"]]
    for o in p["observations"]:
        if (set(o) != {"row", "values", "step", "batch"} or type(o["row"]) is not int or o["row"] not in training or
                type(o["step"]) is not int or not 0 <= o["step"] < state.step or
                o["batch"] != p["batches"][o["step"]] or len(o["values"]) != len(axes[grade][o["row"]])):
            raise ValueError("acquired column differs from its original grade stalk observation")
        for value in o["values"]:
            _exact(value, c["limits"][3])
        grouped[o["step"]].append(o["row"])
    if any(not rows or rows != sorted(set(rows)) for rows in grouped):
        raise ValueError("acquired batches require unique ordered original rows")
    input_dimension = sum(map(len, axes[grade]))
    if input_dimension+len(p["observations"]) > c["limits"][0]:
        raise ValueError("acquisition dimensions exceed action admission")
    if input_dimension+len(p["observations"])+sum(len(o["values"]) for o in p["observations"]) > c["limits"][1]:
        raise ValueError("acquisition columns exceed entry admission")
    if (not p["observations"] and p["acquisition_archive"] is not None or
            p["observations"] and not isinstance(p["acquisition_archive"], bytes)):
        raise ValueError("acquired columns require their native source archive")


def _prepare(state, source):
    from rexgraph_cpp import PreparedStalkBoundary, stalk_boundary_build_info
    validate_stalk_boundary_state(state); state.bind(source)
    c = state.configuration
    if dict(c["engine_receipt"]) != stalk_boundary_build_info():
        raise ValueError("graded boundary model requires its retained native implementation")
    with _lock:
        key = (state.source.native.archive_digest, c["stalk_axes"], c["flag_maps"], c["limits"], c["max_bytes"])
        digest = state.coefficient_digest
        if digest not in _checked and _acquisition(_acquisition_config(c), state.payload["observations"]) != state.payload["acquisition_archive"]:
            raise ValueError("acquisition source differs from the original grade stalk observations")
        plan = _cache.get(key)
        if plan is None:
            plan = PreparedStalkBoundary(source.archive, c["stalk_axes"], c["flag_maps"],
                limits=c["limits"], max_bytes=c["max_bytes"])
            _cache[key] = plan
            while len(_cache) > 4:
                _cache.popitem(last=False)
        if digest not in _checked:
            if state.payload["acquisition_archive"] is not None:
                plan.inject_acquisition(c["grade"], state.payload["acquisition_archive"],
                    tuple(range(sum(map(len, c["stalk_axes"][c["grade"]])))), (1,)*len(state.payload["observations"]))
            _checked[digest] = True
            while len(_checked) > 16:
                _checked.popitem(last=False)
        _cache.move_to_end(key); _checked.move_to_end(digest)
        return plan


def cpp_stalk_boundary_model(source, stalk_axes, flag_maps, training_rows=(), *, grade,
                             boundary_steps=1, limits=(4096, 16000000, 64000000, 16384),
                             max_bytes=256*1024*1024, reference=None):
    """Declare every original stalk and flag block, then check the full chain family."""
    from rexgraph_cpp import stalk_boundary_build_info
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    axes = tuple(_axes(rows) for rows in stalk_axes); limits = tuple(limits)
    if len(limits) != 4 or any(type(v) is not int or v <= 0 for v in limits):
        raise ValueError("four positive action admissions required")
    if type(grade) is not int or type(boundary_steps) is not int or not 0 <= boundary_steps <= grade < len(axes):
        raise ValueError("original input grade and adjacent boundary steps required")
    maps = tuple(tuple(tuple((r, col, _exact(v, limits[3])) for r, col, v in block) for block in family) for family in flag_maps)
    c = dict(profile=_PROFILE, execution_provider=_PROVIDER, engine_receipt=stalk_boundary_build_info(),
        stalk_axes=axes, flag_maps=maps, training_rows=tuple(training_rows), grade=grade,
        boundary_steps=boundary_steps, limits=limits, max_bytes=max_bytes, source_reference=ref.as_record())
    state = ModelState("cpp_stalk_boundary", "1", ref, _space(ref, grade), _slots(axes[grade-boundary_steps]),
        c, dict(observations=(), batches=(), acquisition_archive=None), "rational")
    enable_stalk_boundary_provider(); _prepare(state, source)
    return state


def stalk_boundary_annotation_batch(state, annotations):
    """Bind complete input grade vectors to explicit original training cells."""
    validate_stalk_boundary_state(state); c = state.configuration; axes = c["stalk_axes"][c["grade"]]
    values = np.full((len(axes), max(map(len, axes), default=0)), Fraction(0), dtype=object)
    observed = np.zeros(len(axes), dtype=bool)
    for row, field in annotations:
        if type(row) is not int or row not in c["training_rows"] or observed[row]:
            raise ValueError("annotation requires a unique declared input grade training cell")
        field = tuple(_exact(v, c["limits"][3]) for v in field)
        if len(field) != len(axes[row]):
            raise ValueError("annotation must cover the complete named original stalk")
        values[row, :len(field)] = field; observed[row] = True
    return ModelBatch(state.source, state.space, values, observed,
        dict(profile="whole-graded-stalk-annotations-v1", partition="training"))


def _train(state, source, batch, *, steps):
    _prepare(state, source); c = state.configuration; axes = c["stalk_axes"][c["grade"]]
    if (steps != 1 or dict(batch.inputs or {}) != dict(profile="whole-graded-stalk-annotations-v1", partition="training") or
            batch.targets.shape != (len(axes), max(map(len, axes), default=0))):
        raise ValueError("one complete original grade stalk observation batch required")
    observations = list(state.payload["observations"]); added = 0
    for row, target in enumerate(batch.targets):
        values = tuple(_exact(v, c["limits"][3]) for v in target); size = len(axes[row])
        if any(values[size:]):
            raise ValueError("padding lies outside the actual named stalk")
        if batch.observed[row]:
            if row not in c["training_rows"]:
                raise ValueError("observed row lies outside declared original training cells")
            observations.append(dict(row=row, values=values[:size], batch=batch.coefficient_digest, step=state.step)); added += 1
        elif any(values):
            raise ValueError("unobserved rows cannot contain acquisition values")
    if not added:
        raise ValueError("acquisition requires an observed original grade cell")
    observations = tuple(observations)
    child = state.child(payload=dict(observations=observations, batches=(*state.payload["batches"], batch.coefficient_digest),
        acquisition_archive=_acquisition(_acquisition_config(c), observations)), step=state.step+1)
    _prepare(child, source)
    return child


def stalk_boundary_query(state, fields=None, *, include_acquisition=True):
    """Declare a complete field on the actual input grade without changing its basis."""
    c = state.configuration
    return dict(profile="whole-graded-stalk-query-v1", source_reference=state.source.as_record(),
        fields=_zeros(c, c["grade"]) if fields is None else _field(c, c["grade"], fields), include_acquisition=include_acquisition)


def _query(state, inputs):
    q = freeze_tree(stalk_boundary_query(state) if inputs is None else inputs)
    if (not isinstance(q, Mapping) or set(q) != {"profile", "source_reference", "fields", "include_acquisition"} or
            q["profile"] != "whole-graded-stalk-query-v1" or type(q["include_acquisition"]) is not bool or
            dict(q["source_reference"]) != state.source.as_record()):
        raise ValueError("closed query bound to the actual model source required")
    _field(state.configuration, state.configuration["grade"], q["fields"])
    return q


def _infer(state, source, inputs):
    from rexgraph.model_codec import pack_model
    plan = _prepare(state, source); c = state.configuration; q = _query(state, inputs)
    field = q["fields"]; injection = None
    if q["include_acquisition"] and state.payload["acquisition_archive"] is not None:
        injection = plan.inject_acquisition(c["grade"], state.payload["acquisition_archive"],
            tuple(range(sum(map(len, c["stalk_axes"][c["grade"]])))), (1,)*len(state.payload["observations"]))
        field = tuple(tuple(_exact(a+b, c["limits"][3]) for a,b in zip(left,right,strict=True))
            for left,right in zip(field,injection["field"],strict=True))
    trace = plan.apply(c["grade"], field, boundary_steps=c["boundary_steps"])
    final = trace["fields"][-1]; target = c["grade"]-c["boundary_steps"]; axes = c["stalk_axes"][target]
    width = len(state.output_axes[0].keys); values = np.full((len(final),width), Fraction(0), dtype=object)
    for row, vector in enumerate(final):
        values[row, :len(vector)] = vector
    receipt = dict(profile=_PROFILE, model_checkpoint=pack_model(state), query_declaration=q,
        trace=trace, acquisition_injection=injection,
        stalk_support=tuple(tuple(i < len(row) for i in range(width)) for row in axes),
        readout="complete-original-grade-stalks-v1")
    return ModelOutput(values, _space(state.source,target), state.output_axes, state.source,
        state.coefficient_digest, "rational", _METHOD, grade=target, variance="chain", execution_receipt=receipt)


def _retained_blocks(c, native):
    """Realize every declared occurrence block with its signed active coefficient."""
    axes = c["stalk_axes"]; bits = c["limits"][3]; blocks = [()]
    for k in range(1,len(axes)):
        family = []; cells = native["occurrence_cells"][k]
        if any(len(native[name][k]) != len(c["flag_maps"][k]) for name in
               ("occurrence_cells", "occurrence_coefficients", "occurrence_activity", "occurrence_keys")):
            raise ValueError("receipt omits original occurrence declarations")
        keys = native["occurrence_keys"][k]
        if len(set(keys)) != len(keys) or any(type(key) is not int or key < 0 for key in keys):
            raise ValueError("distinct original occurrence keys required")
        for slot,(up,lo) in enumerate(cells):
            if (type(up) is not int or type(lo) is not int or not 0 <= up < len(axes[k]) or
                    not 0 <= lo < len(axes[k-1]) or type(native["occurrence_activity"][k][slot]) is not bool):
                raise ValueError("occurrence must name actual adjacent grade cells")
            coefficient = _exact(native["occurrence_coefficients"][k][slot],bits)
            if not native["occurrence_activity"][k][slot]:
                coefficient = Fraction(0)
            block = []
            for row,col,value in c["flag_maps"][k][slot]:
                if row >= len(axes[k-1][lo]) or col >= len(axes[k][up]):
                    raise ValueError("flag map exceeds its named original stalk directions")
                block.append((row,col,_exact(coefficient*value,bits)))
            family.append((up,lo,tuple(block)))
        blocks.append(tuple(family))
    return tuple(blocks)


def _boundary(c, blocks, grade, field, work):
    out = [list(row) for row in _zeros(c,grade-1)]; occurrences = []; bits = c["limits"][3]
    for up,lo,block in blocks[grade]:
        contribution = [Fraction(0)]*len(out[lo])
        for row,col,value in block:
            work[0] += 1
            if work[0] > c["limits"][2]:
                raise ValueError("portable boundary verification exceeds work admission")
            contribution[row] = _exact(contribution[row]+_exact(value*field[up][col],bits),bits)
        for row,value in enumerate(contribution):
            out[lo][row] = _exact(out[lo][row]+value,bits)
        occurrences.append(tuple(contribution))
    return tuple(map(tuple,out)),tuple(occurrences)


def validate_stalk_boundary_output(output):
    """Verify complete trace equations, local channels and the producing checkpoint."""
    from hashlib import sha256
    from rexgraph.model_codec import unpack_model
    r = output.execution_receipt
    if (not isinstance(r,Mapping) or set(r) != {"profile", "model_checkpoint", "query_declaration", "trace",
            "acquisition_injection", "stalk_support", "readout"} or r["profile"] != _PROFILE or
            r["readout"] != "complete-original-grade-stalks-v1" or output.arithmetic != "rational" or
            not isinstance(output.source,CppSourceReference) or output.variance != "chain"):
        raise ValueError("complete exact graded boundary output required")
    state = unpack_model(r["model_checkpoint"]); validate_stalk_boundary_state(state); c = state.configuration
    target = c["grade"]-c["boundary_steps"]
    if (state.coefficient_digest != output.model_digest or not state.source.matches(output.source) or
            output.grade != target or output.space != _space(state.source,target) or output.axes != state.output_axes):
        raise ValueError("boundary output differs from its producing model or original grade")
    trace = r["trace"]; native = trace["receipt"]; count = c["boundary_steps"]; axes = c["stalk_axes"]
    if (set(trace) != {"grades", "fields", "occurrence_fields", "intermediate_channels", "execution_steps", "receipt"} or
            trace["grades"] != tuple(range(c["grade"],target-1,-1)) or len(trace["fields"]) != count+1 or
            len(trace["occurrence_fields"]) != count or len(trace["intermediate_channels"]) != count or
            native["stalk_axes"] != axes or native["flag_maps"] != c["flag_maps"] or
            native["limits"] != c["limits"] or native["archive_digest"] != state.source.native.archive_digest or
            native["dimensions"] != tuple(sum(map(len,family)) for family in axes) or
            native["strict_chain_family"] is not True or native["readout"] != "original-grades-and-intermediate-channels-v1" or
            any(native.get(key) != value for key,value in c["engine_receipt"].items()) or
            any(len(native[key]) != len(axes) or native[key][0] for key in
                ("occurrence_cells", "occurrence_coefficients", "occurrence_activity", "occurrence_keys"))):
        raise ValueError("native boundary trace differs from the original complete declaration")
    fields = tuple(_field(c,k,v) for k,v in zip(trace["grades"],trace["fields"],strict=True))
    blocks = _retained_blocks(c,native); work = [0]
    for i in range(count):
        grade = c["grade"]-i
        boundary,occurrences = _boundary(c,blocks,grade,fields[i],work)
        if boundary != fields[i+1] or occurrences != trace["occurrence_fields"][i]:
            raise ValueError("original occurrence vectors do not assemble to the complete boundary")
        expected = []
        if i+1 < count:
            for cell in range(len(axes[grade-1])):
                selected = list(_zeros(c,grade-1)); selected[cell] = boundary[cell]
                expected.append(_boundary(c,blocks,grade-1,selected,work)[0])
            total = tuple(tuple(_exact(sum(v[row][col] for v in expected),c["limits"][3]) for col in range(len(names)))
                          for row,names in enumerate(axes[grade-2]))
            if total != _zeros(c,grade-2):
                raise ValueError("retained intermediate channels have a nonzero strict chain sum")
        if tuple(expected) != trace["intermediate_channels"][i]:
            raise ValueError("trace changes an original intermediate-cell channel")
    final = fields[-1]; width = len(output.axes[0].keys)
    if r["stalk_support"] != tuple(tuple(i < len(row) for i in range(width)) for row in axes[target]):
        raise ValueError("output changes original stalk support")
    for row,values in enumerate(final):
        if tuple(output.values[row,:len(values)]) != values or any(output.values[row,len(values):]):
            raise ValueError("output differs from the complete final field or absent padding")
    q = _query(state,r["query_declaration"]); acquired = [list(row) for row in _zeros(c,c["grade"])]
    for o in state.payload["observations"]:
        for i,v in enumerate(o["values"]):
            acquired[o["row"]][i] = _exact(acquired[o["row"]][i]+v,c["limits"][3])
    injection = r["acquisition_injection"]
    if q["include_acquisition"] and state.payload["observations"]:
        acquired = tuple(map(tuple,acquired)); observations = state.payload["observations"]
        if (not isinstance(injection,Mapping) or injection["profile"] != "declared-graded-stalk-acquisition-v1" or
                injection["grade"] != c["grade"] or injection["columns"] != len(observations) or
                injection["retained_occurrences"] != sum(len(o["values"]) for o in observations) or
                injection["acquisition_archive_digest"] != sha256(state.payload["acquisition_archive"]).hexdigest() or
                injection["mapping"] != tuple(range(sum(map(len,axes[c["grade"]])))) or
                injection["amplitudes"] != (Fraction(1),)*len(observations) or injection["field"] != acquired or
                injection["boundary"] != tuple(v for row in acquired for v in row)):
            raise ValueError("acquisition injection differs from the retained original columns")
        initial = tuple(tuple(_exact(a+b,c["limits"][3]) for a,b in zip(left,right,strict=True))
            for left,right in zip(q["fields"],acquired,strict=True))
    else:
        if injection is not None:
            raise ValueError("query did not admit acquired columns")
        initial = q["fields"]
    if initial != fields[0]:
        raise ValueError("executed field differs from its original query and acquisition")


def stalk_boundary_stage(output, stage=-1, row=None):
    """Select a retained trace grade and optionally one original local stalk."""
    output.check_state(); validate_stalk_boundary_output(output)
    trace = output.execution_receipt["trace"]
    if type(stage) is not int or not -len(trace["grades"]) <= stage < len(trace["grades"]):
        raise ValueError("select a retained original boundary trace stage")
    grade = trace["grades"][stage]; axes = trace["receipt"]["stalk_axes"][grade]; fields = trace["fields"][stage]
    space = _space(output.source,grade)
    if row is None:
        if axes and any(names != axes[0] for names in axes):
            raise TypeError("ragged stalks require an explicit original cell row")
        names = axes[0] if axes else (); values = np.asarray(fields,dtype=object).reshape(len(axes),len(names))
    else:
        if type(row) is not int or not 0 <= row < len(axes):
            raise ValueError("select an original cell at the retained trace grade")
        names = axes[row]; values = np.asarray((fields[row],),dtype=object)
        space = CoordinateSpace(space.name,(space.keys[row],))
    return TensorField(space,values,(CoordinateSpace("named-stalk-axis",names),),output.source,
        grade=grade,variance="chain",provenance=(output.model_digest,output.coefficient_digest))


def replay_stalk_boundary_output(output, source):
    """Recheck the full native chain family and reproduce every retained trace value."""
    from rexgraph.model_codec import unpack_model
    validate_stalk_boundary_output(output)
    state = unpack_model(output.execution_receipt["model_checkpoint"])
    actual = _infer(state,source,output.execution_receipt["query_declaration"])
    if actual.coefficient_digest != output.coefficient_digest:
        raise ValueError("saved boundary trace differs from native execution")
    return actual


def _create(source, *, adapter="cpp_stalk_boundary", **configuration):
    if adapter != "cpp_stalk_boundary":
        raise ValueError("unsupported graded boundary adapter")
    return cpp_stalk_boundary_model(source,**configuration)


def enable_stalk_boundary_provider():
    """Register exact whole field acquisition and owning graded boundary execution."""
    global _provider
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"],_PROVIDER["version"],(("cpp_stalk_boundary","1"),),
                _infer,create=_create,prepare=_prepare,train=_train)
        register_model_provider(_provider)
        return _provider
