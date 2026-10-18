"""Fixed native comparison actions retaining complete source backed parents."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock

from rexgraph.cpp_relational_field_program import _ADAPTER, _LIMITS, _admission, _binding as _program_binding, _publication, _space, _source_dependencies
from rexgraph.cpp_relational_field_learning import _input_rows, _execute_inputs
from rexgraph.field_archive import COMPARISON_FIELD_ARCHIVE_SCHEMA, COMPARISON_FIELD_ARCHIVE_METHOD
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_PROFILE = "rexgraph.relational-field-comparison-model.v1"
_QUERY = "rexgraph.relational-field-comparison-query.v1"
_PROVIDER = {"name":"rexgraph.cpp.relational_field_program","version":"4"}
_lock = RLock()
_cache = OrderedDict()
_results = OrderedDict()
_provider = None


def _binding():
    module = _program_binding()
    if not hasattr(module,"PreparedFieldComparison"):
        raise ImportError("whole-field comparisons require their native comparison provider")
    return module


def validate_field_comparison_state(state):
    c = state.configuration
    required = {"profile","execution_provider","engine_receipt","program_archive","program_sha256","key","revision",
                "comparison_archive","comparison_sha256","comparison_key","comparison_revision","grade","node","channel","limits","max_bytes"}
    if (state.adapter != _ADAPTER or state.adapter_version != "4" or state.arithmetic != "rational"
            or not isinstance(state.source,CppSourceReference) or set(c) != required or c["profile"] != _PROFILE
            or dict(c["execution_provider"]) != _PROVIDER or state.payload or state.step or state.parent is not None
            or state.output_axes or state.dependencies or state.space != _space(state.source,c["key"])):
        raise ValueError("closed fixed source-backed comparison state required")
    _admission(c["limits"],c["max_bytes"])
    for image,digest in ((c["program_archive"],c["program_sha256"]),(c["comparison_archive"],c["comparison_sha256"])):
        if type(image) is not bytes or not image or len(image)>c["max_bytes"] or sha256(image).hexdigest()!=digest:
            raise ValueError("comparison archive differs from its immutable declaration")
    if (c["channel"] not in {"T","rawG","C","F"} or not isinstance(c["engine_receipt"],Mapping)
            or any(type(c[k]) is not int or not 0<c[k]<1<<64 for k in ("key","comparison_key","comparison_revision","grade","node"))
            or type(c["revision"]) is not int or not 0<=c["revision"]<1<<64):
        raise ValueError("comparison identity, grade or channel differs")


def cpp_field_comparison_model(source,declaration=None,*,comparison,node,channel,program_archive=None,
                               limits=_LIMITS,max_bytes=256*1024*1024,reference=None):
    module = _binding(); limits = _admission(limits,max_bytes)
    if not isinstance(source,module.NativeSource): raise TypeError("owning native source required")
    if (declaration is None)==(program_archive is None): raise ValueError("one complete program declaration or canonical archive required")
    if channel not in {"T","rawG","C","F"}: raise ValueError("explicit T/rawG/C/F channel required")
    ref = CppSourceReference(source) if reference is None else reference.bind(source)
    fixed = (module.PreparedFieldProgram(source.archive,declaration,limits=limits,max_bytes=max_bytes) if declaration is not None else
             module.PreparedFieldProgram.from_archive(program_archive,limits=limits,max_bytes=max_bytes))
    _publication(fixed,ref)
    program = module.PreparedAccessionFieldProgram(fixed.archive(),limits=limits,max_bytes=max_bytes)
    forms = module.PreparedFieldComparison(program,comparison,limits=limits,max_bytes=max_bytes)
    forms.validate_node(node)
    identity, info = fixed.identity(),forms.identity(); image = forms.archive()
    c = {"profile":_PROFILE,"execution_provider":_PROVIDER,"engine_receipt":module.field_comparison_build_info(),
         "program_archive":fixed.archive(),"program_sha256":sha256(fixed.archive()).hexdigest(),"key":identity["key"],"revision":identity["revision"],
         "comparison_archive":image,"comparison_sha256":sha256(image).hexdigest(),"comparison_key":info["key"],"comparison_revision":info["revision"],
         "grade":info["grade"],"node":node,"channel":channel,"limits":limits,"max_bytes":max_bytes}
    state = ModelState(_ADAPTER,"4",ref,_space(ref,c["key"]),(),c,{},"rational")
    enable_field_comparison_provider()
    with _lock:
        _cache[(id(source),state.coefficient_digest)] = (source,forms)
        while len(_cache)>4: _cache.popitem(last=False)
    return state


def _prepare(state,source):
    validate_field_comparison_state(state);state.bind(source)
    module = _binding();c = state.configuration
    if dict(c["engine_receipt"])!=module.field_comparison_build_info(): raise ValueError("comparison model requires its pinned native implementation")
    key = (id(source),state.coefficient_digest)
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] is source: _cache.move_to_end(key);return cached[1]
    forms = module.PreparedFieldComparison.from_archive(c["comparison_archive"],limits=c["limits"],max_bytes=c["max_bytes"])
    program = forms.program();_publication(program,state.source);info = forms.identity();program_identity = program.identity()
    forms.validate_node(c["node"])
    if (sha256(program.archive()).hexdigest()!=c["program_sha256"] or info["key"]!=c["comparison_key"]
            or info["revision"]!=c["comparison_revision"] or info["grade"]!=c["grade"]
            or program_identity["key"]!=c["key"] or program_identity["revision"]!=c["revision"]):
        raise ValueError("comparison archive uses another complete source/program/declaration")
    with _lock:
        _cache[key] = (source,forms)
        while len(_cache)>4: _cache.popitem(last=False)
    return forms


def field_comparison_query(state,inputs):
    validate_field_comparison_state(state)
    return freeze_tree({"profile":_QUERY,"model_digest":state.coefficient_digest,"source":state.source.as_record(),"inputs":_input_rows(inputs)})


def _infer(state,source,inputs):
    forms = _prepare(state,source);c = state.configuration
    if (not isinstance(inputs,Mapping) or set(inputs)!={"profile","model_digest","source","inputs"}
            or inputs["profile"]!=_QUERY or inputs["model_digest"]!=state.coefficient_digest
            or not source_from_record(inputs["source"],allow_cpp=True).matches(state.source)):
        raise ValueError("closed actual-source comparison query required")
    module = _binding();rows = _input_rows(inputs["inputs"])
    parent = _execute_inputs(module,forms.program(),rows,c["limits"],c["max_bytes"])
    result = forms.apply(parent,node=c["node"],channel=c["channel"]);image = result.archive()
    dependencies = _source_dependencies(module,parent,len(rows),c["limits"],c["max_bytes"])
    archive = {"schema":COMPARISON_FIELD_ARCHIVE_SCHEMA,"data":image,"sha256":sha256(image).hexdigest(),
               "program_sha256":c["program_sha256"],"engine_receipt":c["engine_receipt"]}
    receipt = {"profile":COMPARISON_FIELD_ARCHIVE_METHOD,"limits":c["limits"],"max_bytes":c["max_bytes"],
               "interpretation":"complete-fixed-native-comparison-observation","input_count":len(rows),
               "node":c["node"],"channel":c["channel"],"comparison_sha256":c["comparison_sha256"]}
    return ModelOutput([],CoordinateSpace("native-comparison-field-readout",()),(),state.source,state.coefficient_digest,
                       "rational",COMPARISON_FIELD_ARCHIVE_METHOD,dependencies=dependencies,execution_receipt=receipt,field_archive=archive)


def read_field_comparison_result(output,*,state=None,limits=None,max_bytes=None):
    if (not isinstance(output,ModelOutput) or output.method!=COMPARISON_FIELD_ARCHIVE_METHOD
            or output.space!=CoordinateSpace("native-comparison-field-readout",()) or output.axes or output.values.size
            or output.grade is not None or output.variance!="coordinate"):
        raise TypeError("complete native comparison output required")
    output.check_state();a,r = output.field_archive,output.execution_receipt
    if (not isinstance(r,Mapping) or set(r)!={"profile","limits","max_bytes","interpretation","input_count","node","channel","comparison_sha256"}
            or r["profile"]!=output.method or r["interpretation"]!="complete-fixed-native-comparison-observation"
            or type(r["input_count"]) is not int or r["input_count"]!=len(output.dependencies)):
        raise ValueError("closed comparison execution receipt required")
    admitted = _admission(r["limits"],r["max_bytes"]);maximum = r["max_bytes"]
    if limits is not None: admitted = tuple(min(a,b) for a,b in zip(admitted,_admission(limits,maximum),strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes);maximum = min(maximum,max_bytes)
    module = _binding()
    if a["schema"]!=COMPARISON_FIELD_ARCHIVE_SCHEMA or dict(a["engine_receipt"])!=module.field_comparison_build_info():
        raise ValueError("comparison result requires its pinned native implementation")
    key = (a["sha256"],admitted,maximum)
    with _lock: result = _results.get(key)
    if result is None:
        result = module.NativeComparisonFieldResult.from_archive(a["data"],limits=admitted,max_bytes=maximum)
        with _lock:
            _results[key] = result
            while len(_results)>4: _results.popitem(last=False)
    parent = result.input();program = parent.program();info = result.inspect();_publication(program,output.source)
    if (sha256(program.archive()).hexdigest()!=a["program_sha256"] or sha256(result.forms_archive()).hexdigest()!=r["comparison_sha256"]
            or info["node"]!=r["node"] or info["channel"]!=r["channel"]):
        raise ValueError("comparison action/program/declaration differs from its envelope")
    actual = _source_dependencies(module,parent,r["input_count"],admitted,maximum)
    if len(actual)!=len(output.dependencies) or any(not a.matches(b) for a,b in zip(actual,output.dependencies,strict=True)):
        raise ValueError("comparison actual original source dependencies differ")
    if state is not None:
        validate_field_comparison_state(state);state.check_state()
        if (state.coefficient_digest!=output.model_digest or not state.source.matches(output.source)
                or state.configuration["comparison_sha256"]!=r["comparison_sha256"] or state.configuration["program_sha256"]!=a["program_sha256"]):
            raise ValueError("comparison result uses another fixed model")
        if state.configuration["node"]!=r["node"] or state.configuration["channel"]!=r["channel"]:
            raise ValueError("comparison result uses another declared node or channel")
    return result


def clear_field_comparison_cache():
    with _lock: _cache.clear();_results.clear()


def _create(source,*,adapter=_ADAPTER,**configuration):
    if adapter!=_ADAPTER: raise ValueError("whole-field program adapter required")
    return cpp_field_comparison_model(source,**configuration)


def enable_field_comparison_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None: _provider = ModelProvider(_PROVIDER["name"],"4",((_ADAPTER,"4"),),_infer,create=_create,prepare=_prepare)
        register_model_provider(_provider)
        return _provider
