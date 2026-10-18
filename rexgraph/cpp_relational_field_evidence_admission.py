"""Source preserving admitted whole field evidence and prospective interpretation."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

import numpy as np

from rexgraph.cpp_relational_field_program import _ADAPTER, _LIMITS, _admission, _publication, _space
from rexgraph.cpp_relational_field_admission import _binding as _fixed_binding, _context
from rexgraph.cpp_relational_field_learning import validate_field_training_targets, FIELD_PROGRAM_TARGETS
from rexgraph.field_archive import AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD, AVAILABLE_AFFINE_FIELD_ARCHIVE_SCHEMA, AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD, AVAILABLE_EVALUATION_FIELD_ARCHIVE_SCHEMA
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "rexgraph.available-relational-field-learning.v1"
_QUERY = "rexgraph.available-relational-affine-field-query.v1"
AVAILABLE_FIELD_PROGRAM_TARGETS = "rexgraph.available-relational-field-training-targets.v1"
_PROVIDER = {"name":"rexgraph.cpp.relational_field_program", "version":"6"}
_lock = RLock()
_cache = OrderedDict()
_provider = None


def _binding():
    module = _fixed_binding()
    if not hasattr(module, "PreparedAvailableFieldEvidence"):
        raise ImportError("available whole-field learning requires its native evidence provider")
    return module


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _origin_keys(value, names):
    if (not isinstance(value, Mapping) or set(value) != set(names)
            or any(type(value[name]) is not int or not 0 < value[name] < 1 << 64 for name in names)):
        raise ValueError("complete original dependency origin keys required")
    return dict(value)


def _input_rows(inputs):
    rows = []
    for value in inputs:
        if (not isinstance(value, Mapping) or set(value) != {"asset","accession","use","field","origins"}
                or not isinstance(value["accession"], Mapping)
                or any(type(value[key]) is not bytes or not value[key] for key in ("asset","field"))
                or type(value["use"]) is not int or not 0 < value["use"] < 1 << 64):
            raise ValueError("original asset/accession/use/whole-field and dependency origins required")
        row = dict(value); row["origins"] = _origin_keys(value["origins"], ("asset","value","accession","reconstruction")); rows.append(row)
    return tuple(rows)


def validate_available_field_learning_state(state):
    c, p = state.configuration, state.payload
    required = {"profile","execution_provider","engine_receipt","program_archive","program_sha256","key","revision",
                "limits","max_bytes","initial_evidence_sha256","parameter_key","parameter_revision","learned_node","learned_grade"}
    if (state.adapter != _ADAPTER or state.adapter_version != "6" or state.arithmetic != "rational"
            or not isinstance(state.source, CppSourceReference) or set(c) != required or c["profile"] != _PROFILE
            or dict(c["execution_provider"]) != _PROVIDER or state.output_axes or state.space != _space(state.source,c["key"])
            or set(p) != {"evidence_archive","evidence_sha256","revision","observation_count","origin_count"}):
        raise ValueError("closed source-preserving available whole-field learning state required")
    _admission(c["limits"],c["max_bytes"])
    for image, expected in ((c["program_archive"],c["program_sha256"]),(p["evidence_archive"],p["evidence_sha256"])):
        if type(image) is not bytes or not image or len(image) > c["max_bytes"] or sha256(image).hexdigest() != expected:
            raise ValueError("available whole-field learning archive differs")
    if (not _sha(c["initial_evidence_sha256"]) or not isinstance(c["engine_receipt"],Mapping)
            or any(type(c[name]) is not int or not 0 <= c[name] < 1 << 64 for name in
                   ("key","revision","parameter_key","parameter_revision","learned_node","learned_grade"))
            or any(not c[name] for name in ("key","parameter_key","parameter_revision","learned_node"))
            or type(p["revision"]) is not int or p["revision"] != c["parameter_revision"] + state.step
            or type(p["observation_count"]) is not int or p["observation_count"] < state.step
            or type(p["origin_count"]) is not int or not 4 <= p["origin_count"] <= len(state.dependencies)
            or (state.step == 0) != (state.parent is None)
            or (state.step == 0 and (p["observation_count"] or p["evidence_sha256"] != c["initial_evidence_sha256"]))):
        raise ValueError("available learning identity, dependency closure or immutable lineage differs")


def _references(module, archives, limits):
    return tuple(CppSourceReference(module.NativeSource(image,max_dimension=max(limits[:2]),max_records=limits[1])) for image in archives)


def _dependencies(module, value, limits):
    return _references(module,value.dependency_archives(max_records=limits[1]),limits)


def _state(evidence, source, configuration, parent=None, step=0):
    native = evidence.evidence(); identity = native.identity(); image = evidence.archive()
    payload = {"evidence_archive":image,"evidence_sha256":sha256(image).hexdigest(),"revision":identity["revision"],
               "observation_count":identity["observation_count"],"origin_count":evidence.origin_count()}
    return ModelState(_ADAPTER,"6",source,_space(source,configuration["key"]),(),configuration,payload,"rational",
                      step=step,parent=parent,dependencies=_dependencies(_binding(),evidence,configuration["limits"]))


def cpp_available_field_learning_model(source, *, available_evidence, execution_context,
                                       limits=_LIMITS, max_bytes=256*1024*1024, reference=None):
    module = _binding(); authority = _context(module,execution_context); limits = _admission(limits,max_bytes)
    if not isinstance(source,module.NativeSource) or not isinstance(available_evidence,module.PreparedAvailableFieldEvidence):
        raise TypeError("actual native source and available evidence owners required")
    if available_evidence.admission() != {"limits":limits,"max_bytes":max_bytes}:
        raise ValueError("available evidence admission differs from its model declaration")
    available_evidence.admit(authority); evidence = available_evidence.evidence(); e = evidence.identity()
    if evidence.parent() is not None or e["observation_count"]: raise ValueError("model creation requires original root evidence")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    program = available_evidence.program().program(); _publication(program,ref); identity = program.identity(); image = program.archive()
    c = {"profile":_PROFILE,"execution_provider":_PROVIDER,"engine_receipt":module.field_evidence_admission_build_info(),
         "program_archive":image,"program_sha256":sha256(image).hexdigest(),"key":identity["key"],"revision":identity["revision"],
         "limits":limits,"max_bytes":max_bytes,"initial_evidence_sha256":sha256(available_evidence.archive()).hexdigest(),
         "parameter_key":e["key"],"parameter_revision":e["revision"],"learned_node":e["node"],"learned_grade":e["grade"]}
    state = _state(available_evidence,ref,c); enable_available_field_learning_provider()
    with _lock: _cache[(id(source),state.coefficient_digest,id(authority))] = (source,authority,available_evidence)
    return state


def _prepare(state, source, *, execution_context):
    validate_available_field_learning_state(state); state.bind(source)
    module = _binding(); authority = _context(module,execution_context); c = state.configuration
    if dict(c["engine_receipt"]) != module.field_evidence_admission_build_info():
        raise ValueError("available learning requires its pinned native implementation")
    key = (id(source),state.coefficient_digest,id(authority))
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] is source and cached[1] is authority:
            cached[2].admit(authority); _cache.move_to_end(key); return cached[2]
    evidence = module.PreparedAvailableFieldEvidence.from_archive(state.payload["evidence_archive"],authority,limits=c["limits"],max_bytes=c["max_bytes"])
    evidence.admit(authority); program = evidence.program().program(); _publication(program,state.source)
    if sha256(program.archive()).hexdigest() != c["program_sha256"]: raise ValueError("available evidence changes its original program")
    ancestors = []; current = evidence
    while current is not None:
        ancestors.append(current)
        if len(ancestors) > state.step+1: raise ValueError("available evidence has extra parent stages")
        current = current.parent()
    if len(ancestors) != state.step+1 or sha256(ancestors[-1].archive()).hexdigest() != c["initial_evidence_sha256"]:
        raise ValueError("available evidence parent lineage differs")
    reconstructed = None
    for step, native in enumerate(reversed(ancestors)):
        identity = native.evidence().identity()
        if any(identity[name] != c[key] for name,key in (("key","parameter_key"),("node","learned_node"),("grade","learned_grade"))):
            raise ValueError("available native evidence parameter owner or target grade differs")
        reconstructed = _state(native,state.source,c,None if reconstructed is None else reconstructed.coefficient_digest,step)
    if reconstructed.coefficient_digest != state.coefficient_digest: raise ValueError("available evidence full dependencies or parent seal differ")
    with _lock:
        _cache[key] = (source,authority,evidence)
        while len(_cache) > 4: _cache.popitem(last=False)
    return evidence


def available_field_learning_query(state, inputs):
    validate_available_field_learning_state(state)
    return freeze_tree({"profile":_QUERY,"model_digest":state.coefficient_digest,"source":state.source.as_record(),"inputs":_input_rows(inputs)})


def validate_available_field_training_targets(value, source, space, mask):
    if (not isinstance(value,Mapping) or set(value) != {"profile","model_digest","node","grade","observations","origin_count"}
            or value["profile"] != AVAILABLE_FIELD_PROGRAM_TARGETS or type(value["origin_count"]) is not int):
        raise ValueError("closed available whole-field observation declaration required")
    plain = []
    count = 0
    for row in value["observations"]:
        if not isinstance(row,Mapping) or set(row) != {"key","inputs","target_asset","target_use","target","observed","origins"}:
            raise ValueError("complete original observation and target origins required")
        _origin_keys(row["origins"],("asset","value")); inputs = _input_rows(row["inputs"]); count += 5+4*len(inputs)
        original = {key:item for key,item in row.items() if key != "origins"}
        original["inputs"] = tuple({key:item for key,item in item.items() if key != "origins"} for item in inputs); plain.append(original)
    if count != value["origin_count"]: raise ValueError("observation original publisher count differs")
    legacy = {key:item for key,item in value.items() if key not in {"origin_count","profile"}}
    legacy.update(profile=FIELD_PROGRAM_TARGETS,observations=tuple(plain))
    validate_field_training_targets(legacy,source,space,mask)
    return value


def available_field_program_batch(state, source, observations, *, execution_context):
    evidence = _prepare(state,source,execution_context=execution_context); rows = tuple(dict(row) for row in observations); c = state.configuration
    info = evidence.observation_dependencies(rows,execution_context,max_records=c["limits"][1])
    targets = {"profile":AVAILABLE_FIELD_PROGRAM_TARGETS,"model_digest":state.coefficient_digest,"node":c["learned_node"],
               "grade":c["learned_grade"],"observations":rows,"origin_count":info["origin_count"]}
    names = {f"{grade}:{state.source.native.coordinates[grade][index]}" for row in rows for grade,index in row["observed"]}
    mask = np.fromiter((key in names for key in state.space.keys),dtype=bool)
    return ModelBatch(state.source,state.space,targets,mask,dependencies=_references(_binding(),info["dependencies"],c["limits"]))


def _train(state, source, batch, *, steps=1, execution_context):
    if steps != 1: raise ValueError("available whole-field append is one immutable update")
    evidence = _prepare(state,source,execution_context=execution_context); c = state.configuration; t = batch.targets
    if (t["profile"] != AVAILABLE_FIELD_PROGRAM_TARGETS or t["model_digest"] != state.coefficient_digest
            or t["node"] != c["learned_node"] or t["grade"] != c["learned_grade"] or batch.inputs is not None or batch.target_input is not None):
        raise ValueError("available observations target another model or source field")
    info = evidence.observation_dependencies(t["observations"],execution_context,max_records=c["limits"][1])
    deps = _references(_binding(),info["dependencies"],c["limits"])
    if info["origin_count"] != t["origin_count"] or len(deps) != len(batch.dependencies) or any(not a.matches(b) for a,b in zip(deps,batch.dependencies,strict=True)):
        raise ValueError("available batch original dependencies differ")
    child = evidence.append_inputs(t["observations"],execution_context)
    if child.parent().archive() != evidence.archive(): raise ValueError("available native append changed its actual parent")
    result = _state(child,state.source,c,state.coefficient_digest,state.step+1)
    previous, added = state.payload["origin_count"], info["origin_count"]
    expected = state.dependencies[:previous]+deps[:added]+state.dependencies[previous:]+deps[added:]
    if len(expected) != len(result.dependencies) or any(not a.matches(b) for a,b in zip(expected,result.dependencies,strict=True)):
        raise ValueError("available child original publisher/asset order differs")
    with _lock:
        _cache[(id(source),result.coefficient_digest,id(execution_context))] = (source,execution_context,child)
        while len(_cache) > 4: _cache.popitem(last=False)
    return result


def _output(state, result, *, selected=False):
    module = _binding(); c = state.configuration; family = result.family() if selected else result
    image = result.archive(); dependencies = _dependencies(module,result,c["limits"])
    schema, method = ((AVAILABLE_EVALUATION_FIELD_ARCHIVE_SCHEMA,AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD) if selected else
                      (AVAILABLE_AFFINE_FIELD_ARCHIVE_SCHEMA,AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD))
    archive = {"schema":schema,"data":image,"sha256":sha256(image).hexdigest(),"program_sha256":c["program_sha256"],"engine_receipt":c["engine_receipt"]}
    receipt = {"profile":method,"limits":c["limits"],"max_bytes":c["max_bytes"],"interpretation":"explicit-global-prospective-parameter-choice" if selected else "complete-admitted-correlated-field-family",
               "evidence_sha256":sha256(family.evidence().archive()).hexdigest(),"dependency_count":len(dependencies)}
    return ModelOutput([],CoordinateSpace("native-program-field-readout",()),(),state.source,state.coefficient_digest,"rational",method,
                       dependencies=dependencies,execution_receipt=receipt,field_archive=archive)


def _infer(state, source, inputs, *, execution_context):
    evidence = _prepare(state,source,execution_context=execution_context)
    if (not isinstance(inputs,Mapping) or set(inputs) != {"profile","model_digest","source","inputs"}
            or inputs["profile"] != _QUERY or inputs["model_digest"] != state.coefficient_digest
            or not source_from_record(inputs["source"],allow_cpp=True).matches(state.source)):
        raise ValueError("closed independently supplied available affine query required")
    result = evidence.execute_inputs(_input_rows(inputs["inputs"]),execution_context)
    return _output(state,result)


def read_available_field_learning_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    if (not isinstance(output,ModelOutput) or output.method not in {AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD,AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD}
            or output.space != CoordinateSpace("native-program-field-readout",()) or output.axes or output.values.size
            or output.grade is not None or output.variance != "coordinate"):
        raise TypeError("available whole-field affine output required")
    output.check_state(); a,r = output.field_archive,output.execution_receipt
    required = {"profile","limits","max_bytes","interpretation","evidence_sha256","dependency_count"}
    selected = output.method == AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD
    schema = AVAILABLE_EVALUATION_FIELD_ARCHIVE_SCHEMA if selected else AVAILABLE_AFFINE_FIELD_ARCHIVE_SCHEMA
    meaning = "explicit-global-prospective-parameter-choice" if selected else "complete-admitted-correlated-field-family"
    if (not isinstance(r,Mapping) or set(r) != required or r["profile"] != output.method or r["interpretation"] != meaning
            or not _sha(r["evidence_sha256"]) or r["dependency_count"] != len(output.dependencies)):
        raise ValueError("closed admitted affine execution receipt required")
    admitted = _admission(r["limits"],r["max_bytes"]); maximum = r["max_bytes"]
    if limits is not None: admitted = tuple(min(a,b) for a,b in zip(admitted,_admission(limits,maximum),strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes); maximum = min(maximum,max_bytes)
    module = _binding(); authority = _context(module,execution_context)
    if a["schema"] != schema or dict(a["engine_receipt"]) != module.field_evidence_admission_build_info():
        raise ValueError("available affine archive requires its pinned native implementation")
    reader = module.NativeAvailableAffineEvaluation if selected else module.NativeAvailableAffineResult
    result = reader.from_archive(a["data"],authority,limits=admitted,max_bytes=maximum); family = result.family() if selected else result
    evidence = family.evidence(); program = evidence.program().program(); _publication(program,output.source)
    if sha256(evidence.archive()).hexdigest() != r["evidence_sha256"] or sha256(program.archive()).hexdigest() != a["program_sha256"]:
        raise ValueError("available output original evidence or program differs")
    deps = _dependencies(module,result,admitted)
    if len(deps) != len(output.dependencies) or any(not a.matches(b) for a,b in zip(deps,output.dependencies,strict=True)):
        raise ValueError("available output original dependencies differ")
    if state is not None:
        validate_available_field_learning_state(state)
        if (output.model_digest != state.coefficient_digest or not output.source.matches(state.source)
                or a["program_sha256"] != state.configuration["program_sha256"] or r["evidence_sha256"] != state.payload["evidence_sha256"]):
            raise ValueError("available output belongs to another model")
    return result


def evaluate_available_field_program(output, values, origin, *, execution_context, state):
    family = read_available_field_learning_result(output,execution_context=execution_context,state=state)
    if output.method != AVAILABLE_AFFINE_FIELD_ARCHIVE_METHOD: raise TypeError("a full available family is required")
    return _output(state,family.evaluate(values,origin,execution_context),selected=True)


def inspect_available_field_learning_model(state, source, *, execution_context):
    return _prepare(state,source,execution_context=execution_context)


def clear_available_field_learning_cache():
    with _lock: _cache.clear()


def _create(source, *, adapter=_ADAPTER, **configuration):
    if adapter != _ADAPTER: raise ValueError("available whole-field learning adapter required")
    return cpp_available_field_learning_model(source,**configuration)


def enable_available_field_learning_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider = ModelProvider(_PROVIDER["name"],_PROVIDER["version"],((_ADAPTER,"6"),),_infer,create=_create,
                                     prepare=_prepare,train=_train,target_profiles=(AVAILABLE_FIELD_PROGRAM_TARGETS,),
                                     context_operations=("create","prepare","infer","train"))
            register_model_provider(_provider)
    return _provider
