"""Native local response packets with complete source and RHS lineage."""
from collections.abc import Mapping
from hashlib import sha256

from rexgraph.cpp_relational_field_program import _admission, _publication, read_field_program_result
from rexgraph.cpp_relational_field_admission import _context
from rexgraph.cpp_relational_field_evidence_admission import _binding, _dependencies, _sha
from rexgraph.field_archive import (
    AVAILABLE_FIELD_ARCHIVE_METHOD, AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD,
    LOCAL_FIELD_ARCHIVE_METHOD, LOCAL_FIELD_ARCHIVE_SCHEMA, EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD,
)
from rexgraph.model_state import ModelOutput
from rexgraph.type_accession import CoordinateSpace


def _native():
    module = _binding()
    if not hasattr(module, "NativeLocalFieldResult"):
        raise ImportError("exact local serving requires its native local provider")
    return module


def _program(rhs):
    if isinstance(rhs, _native().NativeAvailableExtensionEvaluation):
        return rhs.family().model().target().program()
    if isinstance(rhs, _native().NativeAvailableAffineEvaluation):
        return rhs.family().evidence().program().program()
    return rhs.result().program()


def _output(parent, native, limits, maximum):
    module = _native(); info = native.info(); image = native.archive()
    if sha256(native.rhs().archive()).hexdigest() != parent.field_archive["sha256"]:
        raise ValueError("native local result changed its original RHS")
    archive = {"schema": LOCAL_FIELD_ARCHIVE_SCHEMA, "data": image, "sha256": sha256(image).hexdigest(),
               "program_sha256": parent.field_archive["program_sha256"], "engine_receipt": module.field_local_build_info()}
    dependencies = _dependencies(module, native, limits)
    receipt = {"profile": LOCAL_FIELD_ARCHIVE_METHOD, "limits": limits, "max_bytes": maximum,
               "interpretation": "complete-global-solve-with-faithful-local-readout", "packet_context": "complete-global-context",
               "rhs_sha256": parent.field_archive["sha256"], "rhs_method": parent.method,
               "problem_key": info["key"], "problem_revision": info["revision"], "node": info["node"], "grade": info["grade"],
               "dependency_count": len(dependencies)}
    return ModelOutput([], CoordinateSpace("native-program-field-readout", ()), (), parent.source, parent.model_digest,
                       "rational", LOCAL_FIELD_ARCHIVE_METHOD, dependencies=dependencies,
                       execution_receipt=receipt, field_archive=archive)


class LocalFieldRun:
    """Own one native run; cancellation joins before returning."""
    def __init__(self, parent, work, limits, maximum):
        self._parent, self._work, self._limits, self._maximum = parent, work, limits, maximum

    def cancel(self):
        self._work.cancel()

    def receipt(self):
        return self._work.receipt()

    def wait(self, *, execution_context):
        native = self._work.wait(_context(_native(), execution_context))
        return _output(self._parent, native, self._limits, self._maximum)


def submit_available_field_solve(output, node, declaration, readout, origins, *, execution_context,
                                 state=None, limits=None, max_bytes=None, work=(1,1,1,256*1024*1024,256*1024*1024)):
    """Queue an admitted fixed or explicitly selected RHS in a native WorkGraph."""
    if not isinstance(output, ModelOutput) or output.method not in {AVAILABLE_FIELD_ARCHIVE_METHOD, AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD, EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD}:
        raise TypeError("local solving requires a fixed or explicitly completed whole-field output")
    module = _native(); authority = _context(module, execution_context)
    native = read_field_program_result(output, state=state, execution_context=authority, limits=limits, max_bytes=max_bytes)
    admitted = _admission(output.execution_receipt["limits"], output.execution_receipt["max_bytes"])
    maximum = output.execution_receipt["max_bytes"]
    if limits is not None: admitted = tuple(min(a,b) for a,b in zip(admitted, _admission(limits,maximum), strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes); maximum = min(maximum,max_bytes)
    task = module.submit_local_field_solve(native,node,declaration,readout,origins,authority,
                                          limits=admitted,max_bytes=maximum,work=work)
    return LocalFieldRun(output,task,admitted,maximum)


def solve_available_field_output(output, node, declaration, readout, origins, *, execution_context, **options):
    run = submit_available_field_solve(output,node,declaration,readout,origins,execution_context=execution_context,**options)
    return run.wait(execution_context=execution_context)


def read_local_field_result(output, *, execution_context, state=None, limits=None, max_bytes=None):
    """Replay the complete packet without requiring a separate model file."""
    if (not isinstance(output,ModelOutput) or output.method != LOCAL_FIELD_ARCHIVE_METHOD
            or output.space != CoordinateSpace("native-program-field-readout",()) or output.axes or output.values.size
            or output.grade is not None or output.variance != "coordinate"):
        raise TypeError("whole local field output required")
    output.check_state(); receipt, archive = output.execution_receipt, output.field_archive
    required = {"profile","limits","max_bytes","interpretation","packet_context","rhs_sha256","rhs_method",
                "problem_key","problem_revision","node","grade","dependency_count"}
    if (not isinstance(receipt,Mapping) or set(receipt)!=required or receipt["profile"]!=output.method
            or receipt["interpretation"]!="complete-global-solve-with-faithful-local-readout"
            or receipt["packet_context"]!="complete-global-context" or not _sha(receipt["rhs_sha256"])
            or receipt["rhs_method"] not in {AVAILABLE_FIELD_ARCHIVE_METHOD,AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD,EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD}
            or receipt["dependency_count"]!=len(output.dependencies)
            or any(type(receipt[k]) is not int or not 0<=receipt[k]<1<<64 for k in ("problem_key","problem_revision","node","grade"))):
        raise ValueError("closed local packet receipt required")
    module = _native(); authority = _context(module,execution_context)
    if archive["schema"]!=LOCAL_FIELD_ARCHIVE_SCHEMA or dict(archive["engine_receipt"])!=module.field_local_build_info():
        raise ValueError("local packet requires its pinned native implementation")
    admitted = _admission(receipt["limits"],receipt["max_bytes"]); maximum=receipt["max_bytes"]
    if limits is not None: admitted=tuple(min(a,b) for a,b in zip(admitted,_admission(limits,maximum),strict=True))
    if max_bytes is not None: _admission(admitted,max_bytes); maximum=min(maximum,max_bytes)
    native = module.NativeLocalFieldResult.from_archive(archive["data"],authority,limits=admitted,max_bytes=maximum)
    info=native.info(); rhs=native.rhs(); selected=isinstance(rhs,module.NativeAvailableAffineEvaluation)
    extended=isinstance(rhs,module.NativeAvailableExtensionEvaluation)
    expected_method=EXTENSION_EVALUATION_FIELD_ARCHIVE_METHOD if extended else AVAILABLE_EVALUATION_FIELD_ARCHIVE_METHOD if selected else AVAILABLE_FIELD_ARCHIVE_METHOD
    if receipt["rhs_method"] != expected_method:
        raise ValueError("local packet RHS kind differs")
    program=_program(rhs); _publication(program,output.source)
    if (sha256(rhs.archive()).hexdigest()!=receipt["rhs_sha256"] or sha256(program.archive()).hexdigest()!=archive["program_sha256"]
            or any(info[a]!=receipt[b] for a,b in (("key","problem_key"),("revision","problem_revision"),("node","node"),("grade","grade")))):
        raise ValueError("local packet complete RHS or problem identity differs")
    deps=_dependencies(module,native,admitted)
    if len(deps)!=len(output.dependencies) or any(not a.matches(b) for a,b in zip(deps,output.dependencies,strict=True)):
        raise ValueError("local packet original dependencies differ")
    if state is not None:
        if (output.model_digest!=state.coefficient_digest or not output.source.matches(state.source)
                or archive["program_sha256"]!=state.configuration.get("program_sha256")):
            raise ValueError("local packet belongs to another model")
        if selected and sha256(rhs.family().evidence().archive()).hexdigest()!=state.payload.get("evidence_sha256"):
            raise ValueError("local packet learned parent differs")
        if extended and sha256(rhs.family().model().archive()).hexdigest()!=state.payload.get("extension_sha256"):
            raise ValueError("local packet extension parent differs")
    return native
