"""Source preserving native whole field evidence and correlated execution."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

import numpy as np

from rexgraph.cpp_relational_field_program import (
    _ADAPTER, _LIMITS, _admission, _binding as _program_binding, _publication,
    _space, _source_dependencies,
)
from rexgraph.field_archive import AFFINE_FIELD_ARCHIVE_METHOD, AFFINE_FIELD_ARCHIVE_SCHEMA, EVALUATION_FIELD_ARCHIVE_METHOD, EVALUATION_FIELD_ARCHIVE_SCHEMA
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "rexgraph.relational-field-learning.v1"
_QUERY = "rexgraph.relational-affine-field-query.v1"
FIELD_PROGRAM_TARGETS = "rexgraph.relational-field-training-targets.v1"
_PROVIDER = {"name": "rexgraph.cpp.relational_field_program", "version": "3"}
_lock = RLock()
_cache = OrderedDict()
_results = OrderedDict()
_provider = None


def _binding():
    module = _program_binding()
    if not hasattr(module, "PreparedFieldEvidence"):
        raise ImportError("whole-field learning requires its native evidence provider")
    return module


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_field_learning_state(state):
    """Check the portable closed declaration; native preparation replays evidence."""
    c, p = state.configuration, state.payload
    required = {"profile", "execution_provider", "engine_receipt", "program_archive", "program_sha256", "key", "revision",
                "limits", "max_bytes", "initial_evidence_sha256", "parameter_key", "parameter_revision", "learned_node", "learned_grade"}
    if (state.adapter != _ADAPTER or state.adapter_version != "3" or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or set(c) != required
            or c["profile"] != _PROFILE or dict(c["execution_provider"]) != _PROVIDER
            or set(p) != {"evidence_archive", "evidence_sha256", "revision", "observation_count"} or state.output_axes
            or state.space != _space(state.source, c["key"])):
        raise ValueError("closed source-preserving whole-field learning state required")
    _admission(c["limits"], c["max_bytes"])
    for image, digest in [(c["program_archive"], c["program_sha256"]), (p["evidence_archive"], p["evidence_sha256"])]:
        if type(image) is not bytes or not image or len(image) > c["max_bytes"] or sha256(image).hexdigest() != digest:
            raise ValueError("whole-field learning archive differs from its declaration")
    if (not _sha(c["initial_evidence_sha256"]) or not isinstance(c["engine_receipt"], Mapping)
            or any(type(c[k]) is not int or not 0 <= c[k] < 1 << 64 for k in
                   ("key", "revision", "parameter_key", "parameter_revision", "learned_node", "learned_grade"))
            or not c["key"] or not c["parameter_key"] or not c["parameter_revision"] or not c["learned_node"]
            or type(p["revision"]) is not int or p["revision"] != c["parameter_revision"] + state.step
            or type(p["observation_count"]) is not int or p["observation_count"] < state.step
            or (state.step == 0) != (state.parent is None)
            or (state.step == 0 and (state.dependencies or p["observation_count"] or p["evidence_sha256"] != c["initial_evidence_sha256"]))):
        raise ValueError("whole-field learning identity, revision or immutable lineage differs")


def _input_rows(inputs):
    rows = []
    for value in inputs:
        if (not isinstance(value, Mapping) or set(value) != {"asset", "accession", "use", "field"}
                or not isinstance(value["accession"], Mapping)
                or any(type(value[k]) is not bytes or not value[k] for k in ("asset", "field"))
                or type(value["use"]) is not int or not 0 < value["use"] < 1 << 64):
            raise ValueError("actual asset/accession/use and complete native source field required")
        rows.append(dict(value))
    return tuple(rows)


def _references(module, archives, limits):
    return tuple(CppSourceReference(module.NativeSource(image, max_dimension=max(limits[:2]), max_records=limits[1])) for image in archives)


def _dependencies(module, evidence, limits):
    return _references(module, evidence.dependency_archives(max_records=limits[1]), limits)


def _state(evidence, source, configuration, parent=None, step=0):
    module = _binding(); identity = evidence.identity(); image = evidence.archive()
    payload = {"evidence_archive": image, "evidence_sha256": sha256(image).hexdigest(),
               "revision": identity["revision"], "observation_count": identity["observation_count"]}
    return ModelState(_ADAPTER, "3", source, _space(source, configuration["key"]), (), configuration, payload, "rational",
                      step=step, parent=parent, dependencies=_dependencies(module, evidence, configuration["limits"]))


def cpp_field_learning_model(source, declaration=None, *, parameters, program_archive=None,
                             limits=_LIMITS, max_bytes=256*1024*1024, reference=None):
    """Declare one learned native boundary with actual shared Phi parameter maps."""
    module = _binding(); limits = _admission(limits, max_bytes)
    if not isinstance(source, module.NativeSource): raise TypeError("owning native source required")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    if (declaration is None) == (program_archive is None): raise ValueError("one complete native program declaration or archive required")
    fixed = (module.PreparedFieldProgram(source.archive, declaration, limits=limits, max_bytes=max_bytes) if declaration is not None else
             module.PreparedFieldProgram.from_archive(program_archive, limits=limits, max_bytes=max_bytes))
    _publication(fixed, ref); image = fixed.archive(); identity = fixed.identity()
    program = module.PreparedAccessionFieldProgram(image, limits=limits, max_bytes=max_bytes)
    evidence = module.PreparedFieldEvidence(program, parameters, limits=limits, max_bytes=max_bytes)
    e = evidence.identity()
    configuration = {"profile": _PROFILE, "execution_provider": _PROVIDER, "engine_receipt": module.field_evidence_build_info(),
                     "program_archive": image, "program_sha256": sha256(image).hexdigest(), "key": identity["key"],
                     "revision": identity["revision"], "limits": limits, "max_bytes": max_bytes,
                     "initial_evidence_sha256": sha256(evidence.archive()).hexdigest(), "parameter_key": e["key"],
                     "parameter_revision": e["revision"], "learned_node": e["node"], "learned_grade": e["grade"]}
    state = _state(evidence, ref, configuration)
    enable_field_learning_provider()
    with _lock:
        _cache[(id(source), state.coefficient_digest)] = (source, evidence)
        while len(_cache) > 4: _cache.popitem(last=False)
    return state


def _prepare(state, source):
    validate_field_learning_state(state); state.bind(source)
    module = _binding(); c = state.configuration
    if dict(c["engine_receipt"]) != module.field_evidence_build_info():
        raise ValueError("whole-field learning requires its pinned native implementation")
    key = (id(source), state.coefficient_digest)
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] is source:
            _cache.move_to_end(key); return cached[1]
    evidence = module.PreparedFieldEvidence.from_archive(state.payload["evidence_archive"], limits=c["limits"], max_bytes=c["max_bytes"])
    program = evidence.program(); _publication(program, state.source)
    if sha256(program.archive()).hexdigest() != c["program_sha256"]: raise ValueError("learning evidence changes its original program")
    ancestors = []; current = evidence
    while current is not None:
        ancestors.append(current)
        if len(ancestors) > state.step + 1: raise ValueError("native evidence has extra parent stages")
        current = current.parent()
    if len(ancestors) != state.step + 1 or sha256(ancestors[-1].archive()).hexdigest() != c["initial_evidence_sha256"]:
        raise ValueError("native evidence parent lineage differs")
    reconstructed = None
    for step, native in enumerate(reversed(ancestors)):
        identity = native.identity()
        if identity["key"] != c["parameter_key"] or identity["node"] != c["learned_node"] or identity["grade"] != c["learned_grade"]:
            raise ValueError("native evidence parameter owner or target grade differs")
        reconstructed = _state(native, state.source, c, None if reconstructed is None else reconstructed.coefficient_digest, step)
    if reconstructed.coefficient_digest != state.coefficient_digest:
        raise ValueError("wrapper evidence, original dependencies or exact parent seal differ")
    with _lock:
        _cache[key] = (source, evidence)
        while len(_cache) > 4: _cache.popitem(last=False)
    return evidence


def field_learning_query(state, inputs):
    """Capture independent original source uses before any prospective choice."""
    validate_field_learning_state(state)
    return freeze_tree({"profile": _QUERY, "model_digest": state.coefficient_digest, "source": state.source.as_record(),
                        "inputs": _input_rows(inputs)})


def validate_field_training_targets(value, source, space, mask):
    """Check whole stalk observation metadata without native numerical replay."""
    if (not isinstance(source, CppSourceReference) or set(value) != {"profile", "model_digest", "node", "grade", "observations"}
            or value["profile"] != FIELD_PROGRAM_TARGETS or not _sha(value["model_digest"])
            or type(value["node"]) is not int or not 0 < value["node"] < 1 << 64
            or type(value["grade"]) is not int or not 0 <= value["grade"] < len(source.native.coordinates)):
        raise ValueError("closed whole-field training target declaration required")
    positions = {key: i for i, key in enumerate(space.keys)}; selected = set(); keys = set()
    if not value["observations"]: raise ValueError("nonempty whole-field evidence required")
    for sample in value["observations"]:
        if (not isinstance(sample, Mapping) or set(sample) != {"key", "inputs", "target_asset", "target_use", "target", "observed"}
                or type(sample["key"]) is not int or not 0 < sample["key"] < 1 << 64 or sample["key"] in keys
                or type(sample["target_use"]) is not int or not 0 < sample["target_use"] < 1 << 64
                or any(type(sample[k]) is not bytes or not sample[k] for k in ("target_asset", "target"))):
            raise ValueError("original input/target asset, use and complete measurement required")
        keys.add(sample["key"]); _input_rows(sample["inputs"]); ordered = []
        for cell in sample["observed"]:
            if len(cell) != 2: raise ValueError("whole observed stalk grade and index required")
            grade, index = cell
            if (type(grade) is not int or grade != value["grade"] or type(index) is not int
                    or not 0 <= index < len(source.native.coordinates[grade]) or (ordered and index <= ordered[-1])):
                raise ValueError("ordered whole original target stalks required")
            ordered.append(index); name = f"{grade}:{source.native.coordinates[grade][index]}"
            if name not in positions: raise ValueError("training changes original source coordinate order")
            selected.add(positions[name])
    if not np.array_equal(mask, np.fromiter((i in selected for i in range(len(space.keys))), dtype=bool)):
        raise ValueError("batch mask differs from its complete observed stalk metadata")
    return value


def field_program_batch(state, observations):
    """Retain complete native targets and actual input/target dependency cutoffs."""
    validate_field_learning_state(state); module = _binding(); c = state.configuration; rows = []; dependencies = []
    for value in observations:
        if not isinstance(value, Mapping) or set(value) != {"key", "inputs", "target_asset", "target_use", "target", "observed"}:
            raise ValueError("closed whole-field observation declaration required")
        row = dict(value); row["inputs"] = _input_rows(value["inputs"]); rows.append(row)
        assets = [v["asset"] for v in row["inputs"]] + [row["target_asset"]]
        archives = [module.NativeInputUseAsset.from_archive(image, limits=c["limits"], max_bytes=c["max_bytes"]).source_archive() for image in assets]
        dependencies.extend(_references(module, archives, c["limits"]))
    targets = {"profile": FIELD_PROGRAM_TARGETS, "model_digest": state.coefficient_digest, "node": c["learned_node"],
               "grade": c["learned_grade"], "observations": tuple(rows)}
    names = {f"{grade}:{state.source.native.coordinates[grade][index]}" for row in rows for grade, index in row["observed"]}
    mask = np.fromiter((key in names for key in state.space.keys), dtype=bool)
    return ModelBatch(state.source, state.space, targets, mask, dependencies=tuple(dependencies))


def _execute_inputs(module, program, rows, limits, maximum):
    inputs = []
    for row in rows:
        asset = module.NativeInputUseAsset.from_archive(row["asset"], limits=limits, max_bytes=maximum)
        accession = module.NativeInputAccession(asset, program, row["accession"], limits=limits, max_bytes=maximum)
        inputs.append(accession.admit(row["use"], row["field"]))
    return program.execute(inputs)


def _train(state, source, batch, *, steps=1):
    if steps != 1: raise ValueError("whole-field evidence append is one immutable update")
    evidence = _prepare(state, source); c = state.configuration; t = batch.targets
    if (t["model_digest"] != state.coefficient_digest or t["node"] != c["learned_node"] or t["grade"] != c["learned_grade"]
            or batch.inputs is not None or batch.target_input is not None):
        raise ValueError("whole-field observations target another model or native measurement contract")
    module = _binding(); program = evidence.program(); observations = []
    for row in t["observations"]:
        observations.append({"key": row["key"], "input": _execute_inputs(module, program, row["inputs"], c["limits"], c["max_bytes"]),
                             "target_asset": module.NativeInputUseAsset.from_archive(row["target_asset"], limits=c["limits"], max_bytes=c["max_bytes"]),
                             "target_use": row["target_use"], "target": row["target"], "observed": row["observed"]})
    child = evidence.append(observations)
    if child.parent().archive() != evidence.archive(): raise ValueError("native evidence changed its actual parent")
    result = _state(child, state.source, c, state.coefficient_digest, state.step+1)
    expected = state.dependencies + batch.dependencies
    if len(expected) != len(result.dependencies) or any(not a.matches(b) for a,b in zip(expected, result.dependencies, strict=True)):
        raise ValueError("whole-field evidence actual original input/target cutoffs differ")
    with _lock:
        _cache[(id(source), result.coefficient_digest)] = (source, child)
        while len(_cache) > 4: _cache.popitem(last=False)
    return result


def _output(state, result, interpretation):
    module = _binding(); c = state.configuration
    family = result if interpretation == "complete-correlated-field-family" else result.family()
    evidence = family.evidence(); image = result.archive(); held = family.input()
    dependencies = _dependencies(module, evidence, c["limits"]) + _source_dependencies(module, held, len(held.input_uses(max_records=c["limits"][1])), c["limits"], c["max_bytes"])
    schema, method = ((AFFINE_FIELD_ARCHIVE_SCHEMA, AFFINE_FIELD_ARCHIVE_METHOD) if interpretation == "complete-correlated-field-family" else
                      (EVALUATION_FIELD_ARCHIVE_SCHEMA, EVALUATION_FIELD_ARCHIVE_METHOD))
    archive = {"schema": schema, "data": image, "sha256": sha256(image).hexdigest(), "program_sha256": c["program_sha256"],
               "engine_receipt": c["engine_receipt"]}
    receipt = {"profile": method, "limits": c["limits"], "max_bytes": c["max_bytes"], "interpretation": interpretation,
               "evidence_sha256": sha256(evidence.archive()).hexdigest(), "dependency_count": len(dependencies)}
    return ModelOutput([], CoordinateSpace("native-program-field-readout", ()), (), state.source, state.coefficient_digest,
                       "rational", method, dependencies=dependencies, execution_receipt=receipt, field_archive=archive)


def _infer(state, source, inputs):
    evidence = _prepare(state, source); c = state.configuration
    if (not isinstance(inputs, Mapping) or set(inputs) != {"profile", "model_digest", "source", "inputs"}
            or inputs["profile"] != _QUERY or inputs["model_digest"] != state.coefficient_digest
            or not source_from_record(inputs["source"], allow_cpp=True).matches(state.source)):
        raise ValueError("closed independent source-bound affine query required")
    module = _binding(); original = _execute_inputs(module, evidence.program(), _input_rows(inputs["inputs"]), c["limits"], c["max_bytes"])
    return _output(state, evidence.execute(original), "complete-correlated-field-family")


def read_field_learning_result(output, *, state=None, limits=None, max_bytes=None):
    """Cold native replay preserves whole families without parent model loading."""
    if (not isinstance(output, ModelOutput) or output.method not in {AFFINE_FIELD_ARCHIVE_METHOD, EVALUATION_FIELD_ARCHIVE_METHOD}
            or output.space != CoordinateSpace("native-program-field-readout", ()) or output.axes or output.values.size
            or output.grade is not None or output.variance != "coordinate"):
        raise TypeError("whole-field affine output required")
    output.check_state(); a, r = output.field_archive, output.execution_receipt
    required = {"profile", "limits", "max_bytes", "interpretation", "evidence_sha256", "dependency_count"}
    family_mode = output.method == AFFINE_FIELD_ARCHIVE_METHOD
    schema = AFFINE_FIELD_ARCHIVE_SCHEMA if family_mode else EVALUATION_FIELD_ARCHIVE_SCHEMA
    meaning = "complete-correlated-field-family" if family_mode else "explicit-prospective-free-parameter-choice"
    if (not isinstance(r, Mapping) or set(r) != required or r["profile"] != output.method or r["interpretation"] != meaning
            or not _sha(r["evidence_sha256"]) or type(r["dependency_count"]) is not int or r["dependency_count"] != len(output.dependencies)):
        raise ValueError("closed affine family/choice execution receipt required")
    admitted = _admission(r["limits"], r["max_bytes"]); maximum = r["max_bytes"]
    if limits is not None: admitted = tuple(min(a,b) for a,b in zip(admitted, _admission(limits,maximum), strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes); maximum = min(maximum,max_bytes)
    module = _binding()
    if a["schema"] != schema or dict(a["engine_receipt"]) != module.field_evidence_build_info():
        raise ValueError("affine archive requires a different pinned native implementation")
    key = (schema, a["sha256"], admitted, maximum)
    with _lock:
        result = _results.get(key)
    if result is None:
        reader = module.NativeAffineFieldResult if family_mode else module.NativeAffineEvaluation
        result = reader.from_archive(a["data"], limits=admitted, max_bytes=maximum)
        with _lock:
            _results[key] = result
            while len(_results) > 4: _results.popitem(last=False)
    family = result if family_mode else result.family(); evidence = family.evidence(); program = evidence.program()
    if sha256(program.archive()).hexdigest() != a["program_sha256"] or sha256(evidence.archive()).hexdigest() != r["evidence_sha256"]:
        raise ValueError("affine archive parent program or evidence differs")
    _publication(program, output.source); held = family.input()
    actual = _dependencies(module, evidence, admitted) + _source_dependencies(module, held, len(held.input_uses(max_records=admitted[1])), admitted, maximum)
    if len(actual) != len(output.dependencies) or any(not a.matches(b) for a,b in zip(actual,output.dependencies,strict=True)):
        raise ValueError("affine original evidence/input source dependencies differ")
    if state is not None:
        validate_field_learning_state(state); state.check_state()
        if (state.coefficient_digest != output.model_digest or not state.source.matches(output.source)
                or state.payload["evidence_sha256"] != r["evidence_sha256"] or state.configuration["program_sha256"] != a["program_sha256"]):
            raise ValueError("affine response uses another immutable learning model")
    return result


def evaluate_field_program(output, free_values, *, state):
    """Select a complete prospective xi explicitly while retaining its full family."""
    family = read_field_learning_result(output, state=state)
    if output.method != AFFINE_FIELD_ARCHIVE_METHOD: raise ValueError("an unselected correlated family is required")
    return _output(state, family.evaluate(free_values), "explicit-prospective-free-parameter-choice")


def inspect_field_learning_model(state, source):
    return _prepare(state, source)


def clear_field_learning_cache():
    with _lock: _cache.clear(); _results.clear()


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError("whole-field program adapter required")
    return cpp_field_learning_model(source, **configuration)


def enable_field_learning_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"], "3", ((_ADAPTER,"3"),), _infer,
                                      create=_create, prepare=_prepare, train=_train, target_profiles=(FIELD_PROGRAM_TARGETS,))
        register_model_provider(_provider)
        return _provider
