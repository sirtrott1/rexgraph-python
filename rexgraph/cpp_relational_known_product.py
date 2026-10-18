"""Owning native product learning and complete prospective field responses."""
from collections import OrderedDict
from collections.abc import Mapping
from hashlib import sha256
from threading import RLock
from uuid import uuid4

import numpy as np

from rexgraph.cpp_relational_field_program import _admission, _publication, _space
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_evidence_admission import _references
from rexgraph.field_archive import KNOWN_PRODUCT_FIELD_ARCHIVE_METHOD, KNOWN_PRODUCT_FIELD_ARCHIVE_SCHEMA
from rexgraph.model_contract import ModelProvider, register_model_provider
from rexgraph.model_state import ModelBatch, ModelOutput, ModelState, freeze_tree
from rexgraph.source_binding import CppSourceReference, source_from_record
from rexgraph.type_accession import CoordinateSpace

_ADAPTER = "cpp_relational_known_product"
_PROFILE = "rexgraph.available-known-product-learning.v1"
_QUERY = "rexgraph.available-known-product-query.v1"
TARGETS = "rexgraph.available-known-product-training.v1"
_PROVIDER = {"name":"rexgraph.cpp.relational_known_product","version":"1"}
_lock = RLock()
_cache = OrderedDict()
_batch_cache = OrderedDict()
_query_cache = OrderedDict()
_provider = None


def _binding():
    import rexgraph_cpp as module
    if not hasattr(module,"PreparedAvailableKnownProductEvidence"):
        raise ImportError("known-product models require their native owning learning provider")
    return module


def _sha(value):
    return type(value) is str and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def _sharing(value):
    result=tuple(value)
    if len(result)!=4 or any(type(v) is not int or not 0<v<1<<64 for v in result):
        raise ValueError("four positive explicit shared storage bounds required")
    return result


def _policy(c):
    return {"limits":c["limits"],"max_bytes":c["max_bytes"],"shared_limits":c["shared_limits"],"teacher_grammar":c["teacher_grammar"],"expected_root_payload":c["expected_root_payload"]}


def validate_known_product_state(state):
    c,p=state.configuration,state.payload
    required={"profile","execution_provider","engine_receipt","program_archive","program_sha256","key","revision","limits","max_bytes",
              "shared_limits","teacher_grammar","expected_root_payload","expected_root_sha256","initial_evidence_sha256","parameter_schema_sha256","parameter_key","parameter_revision","learned_node","learned_grade"}
    if (state.adapter!=_ADAPTER or state.adapter_version!="1" or state.arithmetic!="rational" or not isinstance(state.source,CppSourceReference)
            or set(c)!=required or c["profile"]!=_PROFILE or dict(c["execution_provider"])!=_PROVIDER
            or state.output_axes or state.space!=_space(state.source,c["key"])
            or set(p)!={"evidence_archive","evidence_sha256","revision","observation_count","origin_count"}):
        raise ValueError("closed source-preserving known-product learning state required")
    _admission(c["limits"],c["max_bytes"]);shared=_sharing(c["shared_limits"])
    expected=c["expected_root_payload"]
    if (type(expected) is not bytes or len(expected)>c["max_bytes"] or bool(expected)!=c["teacher_grammar"]
            or not _sha(c["expected_root_sha256"]) or sha256(expected).hexdigest()!=c["expected_root_sha256"]):
        raise ValueError("teacher moment model requires its complete immutable factory-derived root payload")
    for image,digest,maximum in ((c["program_archive"],c["program_sha256"],c["max_bytes"]),(p["evidence_archive"],p["evidence_sha256"],shared[0])):
        if type(image) is not bytes or not image or len(image)>maximum or sha256(image).hexdigest()!=digest:
            raise ValueError("known-product archive differs from its declared immutable digest")
    if (not _sha(c["initial_evidence_sha256"]) or not _sha(c["parameter_schema_sha256"]) or not isinstance(c["engine_receipt"],Mapping)
            or type(c["teacher_grammar"]) is not bool or any(type(c[name]) is not int or not 0<=c[name]<1<<64 for name in
                ("key","revision","parameter_key","parameter_revision","learned_node","learned_grade"))
            or any(not c[name] for name in ("key","revision","parameter_key","parameter_revision","learned_node"))
            or type(p["revision"]) is not int or p["revision"]!=c["parameter_revision"]+state.step
            or type(p["observation_count"]) is not int or p["observation_count"]<state.step
            or type(p["origin_count"]) is not int or not 4<=p["origin_count"]<=len(state.dependencies)
            or (state.step==0)!=(state.parent is None)
            or (state.step==0 and (p["observation_count"] or p["evidence_sha256"]!=c["initial_evidence_sha256"]))):
        raise ValueError("known-product identity, dependency closure or immutable parent lineage differs")


def _dependencies(module,owner,limits):
    return _references(module,owner.dependency_archives(max_records=limits[1]),limits)


def _state(owner,source,c,parent=None,step=0):
    identity=owner.evidence().identity();image=owner.archive()
    payload={"evidence_archive":image,"evidence_sha256":sha256(image).hexdigest(),"revision":identity["revision"],
             "observation_count":identity["observation_count"],"origin_count":owner.origin_count()}
    return ModelState(_ADAPTER,"1",source,_space(source,c["key"]),(),c,payload,"rational",step=step,parent=parent,
                      dependencies=_dependencies(_binding(),owner,c["limits"]))


def _remember(state,source,owner):
    with _lock:
        _cache[(id(source),state.coefficient_digest)]=(source,owner)
        while len(_cache)>4:_cache.popitem(last=False)


def cpp_known_product_model(source,*,available_evidence,execution_context,reference=None):
    module=_binding();authority=_context(module,execution_context)
    if not isinstance(source,module.NativeSource) or not isinstance(available_evidence,module.PreparedAvailableKnownProductEvidence):
        raise TypeError("actual source and owning known-product evidence required")
    available_evidence.admit(authority);policy=available_evidence.admission();identity=available_evidence.evidence().identity()
    if available_evidence.parent() is not None or identity["observation_count"]:
        raise ValueError("known-product model creation requires original root evidence")
    ref=CppSourceReference(source) if reference is None else reference.bind(source)
    program=available_evidence.program().program();_publication(program,ref);spec=program.identity();image=program.archive()
    c={"profile":_PROFILE,"execution_provider":_PROVIDER,"engine_receipt":module.known_product_learning_build_info(),
       "program_archive":image,"program_sha256":sha256(image).hexdigest(),"key":spec["key"],"revision":spec["revision"],**policy,
       "initial_evidence_sha256":sha256(available_evidence.archive()).hexdigest(),
       "expected_root_sha256":sha256(policy["expected_root_payload"]).hexdigest(),
       "parameter_schema_sha256":sha256(available_evidence.parameter_schema_archive()).hexdigest(),
       "parameter_key":identity["key"],"parameter_revision":identity["revision"],"learned_node":identity["node"],"learned_grade":identity["grade"]}
    state=_state(available_evidence,ref,c);enable_known_product_provider();_remember(state,source,available_evidence)
    return state


def _prepare(state,source,*,execution_context):
    validate_known_product_state(state);state.bind(source);module=_binding();authority=_context(module,execution_context);c=state.configuration
    if dict(c["engine_receipt"])!=module.known_product_learning_build_info():
        raise ValueError("known-product model requires its pinned native implementation")
    key=(id(source),state.coefficient_digest)
    with _lock:cached=_cache.get(key)
    if cached is not None and cached[0] is source:
        cached[1].admit(authority);return cached[1]
    owner=module.PreparedAvailableKnownProductEvidence.from_archive(state.payload["evidence_archive"],authority,**_policy(c))
    owner.admit(authority);program=owner.program().program();_publication(program,state.source)
    if sha256(program.archive()).hexdigest()!=c["program_sha256"] or sha256(owner.parameter_schema_archive()).hexdigest()!=c["parameter_schema_sha256"]:
        raise ValueError("restored known-product program or parameter basis differs")
    ancestors=[];current=owner
    while current is not None:
        ancestors.append(current)
        if len(ancestors)>state.step+1:raise ValueError("known-product evidence has extra parent stages")
        current=current.parent()
    if len(ancestors)!=state.step+1 or sha256(ancestors[-1].archive()).hexdigest()!=c["initial_evidence_sha256"]:
        raise ValueError("known-product original parent lineage differs")
    reconstructed=None
    for step,native in enumerate(reversed(ancestors)):
        identity=native.evidence().identity()
        if any(identity[name]!=c[key] for name,key in (("key","parameter_key"),("node","learned_node"),("grade","learned_grade"))):
            raise ValueError("known-product parameter owner or original grade differs")
        reconstructed=_state(native,state.source,c,None if reconstructed is None else reconstructed.coefficient_digest,step)
    if reconstructed.coefficient_digest!=state.coefficient_digest:raise ValueError("known-product dependency or parent seal differs")
    _remember(state,source,owner);return owner


def _portable_input(input,lineage,c):
    module=_binding()
    if not isinstance(input,module.NativeAvailableKnownTensorSquareResult):raise TypeError("actual owning product input required")
    if lineage is not None and not isinstance(lineage,module.NativeAvailableInitialInputLineage):raise TypeError("actual initial-source lineage required")
    if c["teacher_grammar"] and lineage is None:raise ValueError("teacher role input requires actual original source lineage")
    return {"input":module.shared_known_product_input(input,shared_limits=c["shared_limits"]),"lineage":None if lineage is None else lineage.archive()}


def known_product_query(state,input,*,lineage=None,restore=False):
    validate_known_product_state(state)
    if type(restore) is not bool:raise TypeError("explicit cold query restoration flag required")
    portable=_portable_input(input,lineage,state.configuration)
    token=None if restore else uuid4().hex
    key=(state.coefficient_digest,sha256(portable["input"]).hexdigest(),None if portable["lineage"] is None else sha256(portable["lineage"]).hexdigest(),token)
    if not restore:
        with _lock:
            _query_cache[key]=(input,lineage)
            while len(_query_cache)>8:_query_cache.popitem(last=False)
    return freeze_tree({"profile":_QUERY,"model_digest":state.coefficient_digest,"source":state.source.as_record(),"restore":restore,"dispatch_token":token,**portable})


def _restore_input(owner,row,context,c):
    if c["teacher_grammar"] and row["lineage"] is None:raise ValueError("teacher role restoration requires actual original lineage")
    return owner.restore_query(row["input"],row["lineage"],context)


def validate_known_product_targets(value,source,space,mask):
    keys={"profile","model_digest","node","grade","observations","origin_count"}
    rowkeys={"key","input","lineage","target_asset","target_use","target","observed","origins","interface_origin","evidence_origin"}
    if not isinstance(value,Mapping) or set(value)!=keys or value["profile"]!=TARGETS or not _sha(value["model_digest"]):
        raise ValueError("closed immutable known-product observations required")
    expected=np.zeros(len(space.keys),dtype=bool);coordinates=source.native.coordinates
    if not value["observations"]:raise ValueError("complete nonempty native observation suffix required")
    for row in value["observations"]:
        if (not isinstance(row,Mapping) or set(row)!=rowkeys or any(type(row[key]) is not bytes or not row[key] for key in ("input","target_asset","target"))
                or (row["lineage"] is not None and (type(row["lineage"]) is not bytes or not row["lineage"]))
                or any(type(row[key]) is not int or not 0<row[key]<1<<64 for key in ("key","target_use","interface_origin","evidence_origin"))
                or not isinstance(row["origins"],Mapping) or set(row["origins"])!={"asset","value"}
                or any(type(v) is not int or not 0<v<1<<64 for v in row["origins"].values())):
            raise ValueError("original compound, target and observer origin declaration required")
        for grade,index in row["observed"]:
            if type(grade) is not int or type(index) is not int or not 0<=grade<len(coordinates) or not 0<=index<len(coordinates[grade]):
                raise ValueError("observed original cell scope differs")
            name=f"{grade}:{coordinates[grade][index]}"
            if name not in space.keys:raise ValueError("whole observation cell absent from original model space")
            expected[space.keys.index(name)]=True
    if not np.array_equal(mask,expected):raise ValueError("observed original cells differ from the batch mask")
    return value


def known_product_batch(state,source,observations,*,execution_context):
    owner=_prepare(state,source,execution_context=execution_context);module=_binding();c=state.configuration
    rows=tuple(dict(row,origins=dict(row["origins"]),observed=tuple(tuple(cell) for cell in row["observed"])) for row in observations)
    info=owner.observation_dependencies(rows,execution_context,max_records=c["limits"][1]);portable=[]
    for row in rows:
        owner.observation_payload(row,execution_context)
        item=dict(row);item.update(_portable_input(row["input"],row["lineage"],c));item["target_asset"]=row["target_asset"].archive();portable.append(item)
    names={f"{grade}:{state.source.native.coordinates[grade][index]}" for row in rows for grade,index in row["observed"]}
    mask=np.fromiter((key in names for key in state.space.keys),dtype=bool)
    targets={"profile":TARGETS,"model_digest":state.coefficient_digest,"node":c["learned_node"],"grade":c["learned_grade"],"observations":tuple(portable),"origin_count":info["origin_count"]}
    batch=ModelBatch(state.source,state.space,targets,mask,dependencies=_references(module,info["dependencies"],c["limits"]))
    with _lock:
        _batch_cache[(batch.coefficient_digest,id(execution_context))]=(execution_context,rows)
        while len(_batch_cache)>2:_batch_cache.popitem(last=False)
    return batch


def _train(state,source,batch,*,steps=1,execution_context):
    if steps!=1:raise ValueError("native product append is one immutable model update")
    owner=_prepare(state,source,execution_context=execution_context);c=state.configuration;t=batch.targets
    if t["profile"]!=TARGETS or t["model_digest"]!=state.coefficient_digest or t["node"]!=c["learned_node"] or t["grade"]!=c["learned_grade"] or batch.inputs is not None or batch.target_input is not None:
        raise ValueError("known-product observation suffix belongs to another original model")
    with _lock:cached=_batch_cache.get((batch.coefficient_digest,id(execution_context)))
    if cached is not None and cached[0] is execution_context:rows=cached[1]
    else:
        module=_binding();rows=[]
        for portable in t["observations"]:
            input,lineage=_restore_input(owner,portable,execution_context,c);row=dict(portable);row.update(input=input,lineage=lineage,
                target_asset=module.NativeInputUseAsset.from_archive(portable["target_asset"],limits=c["limits"],max_bytes=c["max_bytes"]))
            rows.append(row)
    info=owner.observation_dependencies(rows,execution_context,max_records=c["limits"][1])
    deps=_references(_binding(),info["dependencies"],c["limits"])
    if info["origin_count"]!=t["origin_count"] or len(deps)!=len(batch.dependencies) or any(not a.matches(b) for a,b in zip(deps,batch.dependencies,strict=True)):
        raise ValueError("known-product batch publisher and source dependencies differ")
    child=owner.append(rows,execution_context)
    if child.parent().archive()!=owner.archive() or child.parameter_schema_archive()!=owner.parameter_schema_archive():
        raise ValueError("native product append changes original parent or parameter basis")
    result=_state(child,state.source,c,state.coefficient_digest,state.step+1);_remember(result,source,child)
    return result


def _output(state,result):
    c=state.configuration;image=result.archive();dependencies=_dependencies(_binding(),result,c["limits"])
    archive={"schema":KNOWN_PRODUCT_FIELD_ARCHIVE_SCHEMA,"data":image,"sha256":sha256(image).hexdigest(),
             "program_sha256":c["program_sha256"],"engine_receipt":c["engine_receipt"]}
    receipt={"profile":KNOWN_PRODUCT_FIELD_ARCHIVE_METHOD,**_policy(c),"interpretation":"complete-admitted-correlated-field-family",
             "evidence_sha256":state.payload["evidence_sha256"],"dependency_count":len(dependencies)}
    return ModelOutput([],CoordinateSpace("native-program-field-readout",()),(),state.source,state.coefficient_digest,"rational",KNOWN_PRODUCT_FIELD_ARCHIVE_METHOD,
                       dependencies=dependencies,execution_receipt=receipt,field_archive=archive)


def _infer(state,source,query,*,execution_context):
    owner=_prepare(state,source,execution_context=execution_context)
    if (not isinstance(query,Mapping) or set(query)!={"profile","model_digest","source","input","lineage","restore","dispatch_token"} or query["profile"]!=_QUERY
            or query["model_digest"]!=state.coefficient_digest or not source_from_record(query["source"],allow_cpp=True).matches(state.source)):
        raise ValueError("closed independent known-product query required")
    if type(query["restore"]) is not bool:raise ValueError("explicit checked query restoration required")
    if query["restore"]:
        if query["dispatch_token"] is not None:raise ValueError("cold query cannot retain a warm dispatch token")
        input,lineage=_restore_input(owner,query,execution_context,state.configuration)
    else:
        token=query["dispatch_token"]
        if type(token) is not str or len(token)!=32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("warm query requires its unique actual-owner dispatch token")
        key=(state.coefficient_digest,sha256(query["input"]).hexdigest(),None if query["lineage"] is None else sha256(query["lineage"]).hexdigest(),token)
        with _lock:cached=_query_cache.get(key)
        if cached is None:raise ValueError("warm query requires its actual native owner; request explicit cold restoration")
        input,lineage=cached
    return _output(state,owner.execute(input,execution_context,lineage=lineage))


def read_known_product_result(output,*,execution_context,state=None):
    if (not isinstance(output,ModelOutput) or output.method!=KNOWN_PRODUCT_FIELD_ARCHIVE_METHOD or output.space!=CoordinateSpace("native-program-field-readout",())
            or output.axes or output.values.size or output.grade is not None or output.variance!="coordinate"):
        raise TypeError("complete known-product field output required")
    output.check_state();a,r=output.field_archive,output.execution_receipt;module=_binding();authority=_context(module,execution_context)
    if dict(a["engine_receipt"])!=module.known_product_learning_build_info():raise ValueError("known-product output requires its pinned native implementation")
    if not isinstance(r,Mapping) or set(r)!={"profile","limits","max_bytes","shared_limits","teacher_grammar","expected_root_payload","interpretation","evidence_sha256","dependency_count"} or r["profile"]!=output.method:
        raise ValueError("complete known-product execution receipt required")
    result=module.NativeKnownProductFamily.from_archive(a["data"],authority,**_policy(r));owner=result.evidence();program=owner.program().program();_publication(program,output.source)
    if sha256(program.archive()).hexdigest()!=a["program_sha256"] or sha256(owner.archive()).hexdigest()!=r["evidence_sha256"]:
        raise ValueError("known-product output changes original evidence or program")
    deps=_dependencies(module,result,r["limits"])
    if r["dependency_count"]!=len(deps) or len(deps)!=len(output.dependencies) or any(not x.matches(y) for x,y in zip(deps,output.dependencies,strict=True)):
        raise ValueError("known-product output dependency closure differs")
    if state is not None and (output.model_digest!=state.coefficient_digest or not output.source.matches(state.source) or r["evidence_sha256"]!=state.payload["evidence_sha256"]):
        raise ValueError("known-product output belongs to another model")
    return result


def inspect_known_product_model(state,source,*,execution_context):
    return _prepare(state,source,execution_context=execution_context)


def clear_known_product_cache():
    with _lock:_cache.clear();_batch_cache.clear();_query_cache.clear()


def _create(source,*,adapter=_ADAPTER,**configuration):
    if adapter!=_ADAPTER:raise ValueError("known-product model adapter required")
    return cpp_known_product_model(source,**configuration)


def enable_known_product_provider():
    global _provider
    _binding()
    with _lock:
        if _provider is None:
            _provider=ModelProvider(_PROVIDER["name"],_PROVIDER["version"],((_ADAPTER,"1"),),_infer,create=_create,prepare=_prepare,train=_train,
                                    target_profiles=(TARGETS,),context_operations=("create","prepare","infer","train"))
            register_model_provider(_provider)
    return _provider
