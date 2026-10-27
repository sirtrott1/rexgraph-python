# distutils: language=c++
"""Owning exact assignment evidence, immutable solves and conditioned responses."""
from ._finite_field cimport NativeFiniteSource
from ._field_interface cimport _bounds, _maximum, _image, _matrix
from ._field_admission cimport NativeAvailabilityContext, NativeAvailableFieldResult, NativeValueOrigin
from ._field_evidence_admission cimport PreparedAvailableFieldEvidence, NativeAvailableAffineResult
from ._task_outcome cimport NativeTaskOutcome, _binding, _Sink, _write
from ._known_tensor_graph cimport _Check, _check, _callback, _raise_check
from ._field_program import _closed
from ._stalk_fields import _natural, _scalar, _values

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_ASSIGNMENT_BINDING_DIGEST

_LIMITS = (4096, 16000000, 64000000, 1 << 24)
_STATUS = ('inconsistent', 'unresolved', 'identified')
_CERTIFICATE = ('satisfied', 'open', 'refuted')


def assignment_build_info():
    return dict(engine='rexgraph-cpp', engine_version=RFC_ENGINE_VERSION.decode(),
        engine_headers=RFC_ENGINE_DIGEST.decode(), binding=RFC_ASSIGNMENT_BINDING_DIGEST.decode(),
        profile='owning-relational-assignment-evidence-v1', scalar_policy='gmp-rational-exact')


cdef vector[uint64_t] _keys(value, size_t maximum):
    cdef vector[uint64_t] out
    for item in value:
        if out.size() >= maximum: raise ValueError('assignment key list exceeds admission')
        out.push_back(_natural(item))
    return out


cdef AssignmentDeclaration _declaration(value, vector[size_t] bounds):
    _closed(value, ('key', 'revision', 'origin', 'parameter_carrier', 'parameters'))
    cdef AssignmentDeclaration out
    out.key = _natural(value['key']); out.revision = _natural(value['revision']); out.origin = _natural(value['origin'])
    out.parameter_carrier = _natural(value['parameter_carrier']); out.parameters = _keys(value['parameters'], bounds[0])
    return out


cdef vector[AssignmentCandidate] _candidates(values, vector[size_t] bounds):
    cdef vector[AssignmentCandidate] out
    cdef AssignmentCandidate row
    for value in values:
        if out.size() >= bounds[0]: raise ValueError('assignment candidate count exceeds admission')
        _closed(value, ('key', 'assignment', 'binding', 'evidence', 'shift', 'injection'))
        if not isinstance(value['evidence'], PreparedAvailableFieldEvidence): raise TypeError('actual available evidence root required')
        row.key = _natural(value['key']); row.assignment = _keys(value['assignment'], bounds[0])
        row.binding = _binding(value['binding'], bounds); row.evidence = (<PreparedAvailableFieldEvidence>value['evidence'])._pin()
        row.shift.clear()
        for q in value['shift']:
            if row.shift.size() >= bounds[0]: raise ValueError('assignment shift exceeds admission')
            row.shift.push_back(_scalar(q, bounds[3]))
        row.injection = _matrix(value['injection'], bounds[3]); out.push_back(row)
    return out


cdef AssignmentObservation _observation(value, vector[size_t] bounds):
    _closed(value, ('key', 'origin', 'binding', 'measured', 'inputs'))
    if not isinstance(value['measured'], NativeAvailableFieldResult): raise TypeError('actual returned whole measurement required')
    cdef AssignmentObservation out
    cdef AssignmentInput row
    out.key = _natural(value['key']); out.origin = _natural(value['origin']); out.binding = _binding(value['binding'], bounds)
    out.measured = (<NativeAvailableFieldResult>value['measured'])._pin()
    for item in value['inputs']:
        _closed(item, ('candidate', 'input'))
        if out.inputs.size() >= bounds[0]: raise ValueError('assignment input count exceeds admission')
        if not isinstance(item['input'], NativeAvailableFieldResult): raise TypeError('actual candidate whole input required')
        row.candidate = _natural(item['candidate']); row.input = (<NativeAvailableFieldResult>item['input'])._pin()
        out.inputs.push_back(row)
    return out


cdef object _coordinates(AssignmentCoordinates value):
    return dict(shift=<bytes>value.shift, map=<bytes>value.map,
                original_dimension=value.original_dimension, free_dimension=value.free_dimension)


cdef object _certificate(TaskCertificate value):
    return dict(status=_CERTIFICATE[value.status], rank=value.rank, residual=<bytes>value.residual,
        directions=<bytes>value.directions, feasible_shift=<bytes>value.feasible_shift,
        feasible_kernel=<bytes>value.feasible_kernel, contradiction=<bytes>value.contradiction)


cdef object _matrix_info(ProgramMatrix value):
    return dict(shape=(value.rows, value.columns), entries=tuple(
        (entry.row, entry.column, _values((<bytes>entry.value,))[0]) for entry in value.entries))


cdef object _binding_info(TaskBinding value):
    return dict(profile='rexgraph.relational-worker-binding.v1', key=value.key, revision=value.revision,
        worker=value.worker, machine=value.machine, operation=value.operation, origin=value.origin,
        interface=dict(key=value.interface.key, branch=value.interface.branch,
            cells=tuple((v.node, (v.grade, v.cell)) for v in value.interface.cells)),
        view=dict(profile='rexgraph.relational-field-view.v1', key=value.view.key, basis=value.view.basis,
            directions=tuple(value.view.directions), accession=_matrix_info(value.view.accession),
            reconstruction=_matrix_info(value.view.reconstruction),
            physical=_matrix_info(value.view.physical) if value.view.has_physical else None))


cdef NativeAssignmentEvidence _model(shared_ptr[AssignmentModelBridge] value, vector[size_t] bounds, size_t maximum):
    cdef NativeAssignmentEvidence out = NativeAssignmentEvidence.__new__(NativeAssignmentEvidence)
    out._handle = value; out._limits = bounds; out._maximum = maximum
    return out


cdef NativeAvailableFieldResult _input(shared_ptr[AvailableResultBridge] value, vector[size_t] bounds, size_t maximum):
    cdef NativeAvailableFieldResult out = NativeAvailableFieldResult.__new__(NativeAvailableFieldResult)
    out._handle = value; out._limits = bounds; out._maximum = maximum
    return out


cdef NativeAvailableAffineResult _original(shared_ptr[AvailableAffineBridge] value, vector[size_t] bounds, size_t maximum):
    cdef NativeAvailableAffineResult out = NativeAvailableAffineResult.__new__(NativeAvailableAffineResult)
    out._handle = value; out._limits = bounds; out._maximum = maximum
    return out


def assignment_root_origin_payload(declaration, NativeFiniteSource finite not None, candidates,
                                   *, limits=_LIMITS, max_bytes=256*1024*1024):
    cdef vector[size_t] bounds = _bounds(limits)
    cdef size_t maximum = _maximum(max_bytes)
    cdef AssignmentDeclaration d = _declaration(declaration, bounds)
    cdef vector[AssignmentCandidate] values = _candidates(candidates, bounds)
    cdef shared_ptr[FiniteSourceBridge] source = finite._pin()
    cdef string image
    with nogil: image = AssignmentModelBridge.root_payload(d, source, values, bounds, maximum)
    return <bytes>image


def assignment_measurement_origin_payload(observation, *, limits=_LIMITS, max_bytes=256*1024*1024):
    cdef vector[size_t] bounds = _bounds(limits)
    cdef size_t maximum = _maximum(max_bytes)
    cdef AssignmentObservation value = _observation(observation, bounds)
    cdef string image
    with nogil: image = AssignmentModelBridge.measurement_payload(value, bounds, maximum)
    return <bytes>image


def assignment_input_origins(NativeAvailableFieldResult input not None, NativeAvailabilityContext execution_context not None,
                            *, max_records=4096, max_bytes=256*1024*1024):
    cdef shared_ptr[AvailableResultBridge] owner = input._pin()
    cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
    cdef size_t maximum = _natural(max_records), byte_limit = _maximum(max_bytes)
    cdef GraphCheck callback
    cdef vector[shared_ptr[ValueOriginBridge]] values
    cdef NativeValueOrigin value
    with nogil: values = AssignmentModelBridge.input_origins(owner, authority, maximum, byte_limit, callback)
    out = []
    for handle in values:
        value = NativeValueOrigin.__new__(NativeValueOrigin); value._handle = handle; out.append(value)
    return tuple(out)


cdef class NativeAssignmentEvidence:
    cdef shared_ptr[AssignmentModelBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('assignment evidence is not initialized')
        return self._handle

    def __init__(self, declaration, NativeFiniteSource finite not None, candidates,
                 NativeAvailabilityContext execution_context not None, *, limits=_LIMITS, max_bytes=256*1024*1024, check=None):
        if self._loading or self._handle.get() != NULL: raise TypeError('assignment evidence is immutable')
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef AssignmentDeclaration d = _declaration(declaration, bounds)
        cdef vector[AssignmentCandidate] values = _candidates(candidates, bounds)
        cdef shared_ptr[FiniteSourceBridge] source = finite._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        self._loading = True
        try:
            with nogil: self._handle = shared_ptr[AssignmentModelBridge](new AssignmentModelBridge(d, source, values, authority, bounds, maximum, callback))
            self._limits = bounds; self._maximum = maximum
        finally:
            self._loading = False
            _raise_check(state)

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS, max_bytes=256*1024*1024, check=None):
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        cdef shared_ptr[AssignmentModelBridge] value
        try:
            with nogil: value = AssignmentModelBridge.restore(image, authority, bounds, maximum, callback)
        finally: _raise_check(state)
        return _model(value, bounds, maximum)

    def admission(self):
        self._pin()
        return dict(limits=tuple(self._limits), max_bytes=self._maximum)

    def admit(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        try:
            with nogil: owner.get().admit(authority, callback)
        finally: _raise_check(state)

    def same_owner(self, NativeAssignmentEvidence other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

    def inspect(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef AssignmentSummary value
        cdef AssignmentDeclaration d
        with nogil: value = owner.get().summary(authority, maximum, callback)
        d = value.declaration
        return dict(key=d.key, root_revision=d.revision, origin=d.origin, parameter_carrier=d.parameter_carrier,
            parameters=tuple(d.parameters), revision=value.revision, candidate_count=value.candidates,
            observation_count=value.observations, remainder_count=value.remainder, retained_entries=value.retained_entries,
            status=_STATUS[value.status], identified_candidate=value.identified_candidate if value.identified else None,
            has_parent=bool(value.has_parent))

    def archive(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        cdef string image
        try:
            with nogil: image = owner.get().archive(authority, callback)
        finally: _raise_check(state)
        return <bytes>image

    def stream(self, consume, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        cdef _Sink sink = _Sink()
        sink.write = consume; sink.error = None
        cdef TaskSink target
        target.write = _write; target.context = <void*><object>sink
        cdef size_t count
        try:
            with nogil: count = owner.get().stream(target, authority, callback)
        finally:
            if sink.error is not None: raise sink.error
            _raise_check(state)
        return count

    def parent(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin(), value
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        with nogil: value = owner.get().parent(authority, callback)
        return _model(value, self._limits, self._maximum) if value.get() != NULL else None

    def finite_source(self, NativeAvailabilityContext execution_context not None):
        cdef NativeFiniteSource out = NativeFiniteSource.__new__(NativeFiniteSource)
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        with nogil: out._handle = owner.get().finite_source(authority, callback)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def candidate(self, index, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t selected = _natural(index), maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef AssignmentCandidate value
        cdef PreparedAvailableFieldEvidence evidence
        with nogil: value = owner.get().candidate(selected, authority, maximum, callback)
        evidence = PreparedAvailableFieldEvidence.__new__(PreparedAvailableFieldEvidence)
        evidence._handle = value.evidence; evidence._limits = self._limits; evidence._maximum = self._maximum
        return dict(key=value.key, assignment=tuple(value.assignment), binding=_binding_info(value.binding), evidence=evidence,
                    shift=tuple(_values(tuple(<bytes>q for q in value.shift))), injection=_matrix_info(value.injection))

    def branch(self, index, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t selected = _natural(index)
        cdef GraphCheck callback
        cdef AssignmentBranchInfo value
        with nogil: value = owner.get().branch(selected, authority, callback)
        return dict(rejected=bool(value.rejected), solve_count=value.solves, coordinates=_coordinates(value.coordinates))

    def observation(self, index, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t selected = _natural(index), maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef AssignmentObservation value
        with nogil: value = owner.get().observation(selected, authority, maximum, callback)
        return dict(key=value.key, origin=value.origin, binding=_binding_info(value.binding),
            measured=_input(value.measured, self._limits, self._maximum),
            inputs=tuple(dict(candidate=row.candidate, input=_input(row.input, self._limits, self._maximum)) for row in value.inputs))

    def remainder(self, begin, count, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t start = _natural(begin), length = _natural(count), maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef vector[vector[uint64_t]] rows
        with nogil: rows = owner.get().remainder(start, length, authority, maximum, callback)
        return tuple(tuple(row) for row in rows)

    def origins(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef vector[shared_ptr[ValueOriginBridge]] values
        cdef NativeValueOrigin value
        with nogil: values = owner.get().origins(authority, maximum, callback)
        out = []
        for handle in values:
            value = NativeValueOrigin.__new__(NativeValueOrigin); value._handle = handle; out.append(value)
        return tuple(out)

    def append(self, observations, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin(), result
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef vector[AssignmentObservation] values
        for row in observations:
            if values.size() >= self._limits[0]: raise ValueError('assignment observation count exceeds admission')
            values.push_back(_observation(row, self._limits))
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        try:
            with nogil: result = owner.get().append(values, authority, callback)
        finally: _raise_check(state)
        return _model(result, self._limits, self._maximum)

    def dependency_archives(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef vector[string] values
        with nogil: values = owner.get().dependencies(authority, maximum, callback)
        return tuple(<bytes>value for value in values)

    def observation_dependencies(self, observations, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[AssignmentObservation] values
        for row in observations:
            if values.size() >= self._limits[0]: raise ValueError('assignment observation count exceeds admission')
            values.push_back(_observation(row, self._limits))
        cdef GraphCheck callback
        cdef AssignmentBatchInfo value
        with nogil: value = owner.get().observation_dependencies(values, authority, maximum, callback)
        return dict(origin_count=value.origin_count, dependencies=tuple(<bytes>item for item in value.dependencies))

    def restore_input(self, candidate, archive, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t selected = _natural(candidate)
        cdef string image = _image(archive, self._maximum)
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        cdef shared_ptr[AvailableResultBridge] result
        try:
            with nogil: result = owner.get().restore_input(selected, image, authority, callback)
        finally: _raise_check(state)
        return _input(result, self._limits, self._maximum)

    def solve(self, candidate, index, NativeAvailabilityContext execution_context not None):
        cdef NativeAssignmentSolve out = NativeAssignmentSolve.__new__(NativeAssignmentSolve)
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t selected = _natural(candidate), ordinal = _natural(index)
        cdef GraphCheck callback
        with nogil: out._handle = owner.get().solve(selected, ordinal, authority, callback)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def execute(self, candidate, NativeAvailableFieldResult input not None, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef NativeAssignmentResponse out = NativeAssignmentResponse.__new__(NativeAssignmentResponse)
        cdef shared_ptr[AssignmentModelBridge] owner = self._pin()
        cdef shared_ptr[AvailableResultBridge] value = input._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t selected = _natural(candidate)
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        try:
            with nogil: out._handle = owner.get().execute(selected, value, authority, callback)
        finally: _raise_check(state)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAssignmentSolve:
    cdef shared_ptr[AssignmentSolveBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('assignment solve is not initialized')
        return self._handle

    def same_owner(self, NativeAssignmentSolve other not None):
        return bool(self._pin().get().same_owner(other._pin().get()[0]))

    def inspect(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentSolveBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        cdef AssignmentCoordinates value
        cdef uint64_t observation
        with nogil:
            value = owner.get().coordinates(authority, callback); observation = owner.get().observation(authority, callback)
        return dict(observation=observation, child=_coordinates(value))

    def certificate(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentSolveBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        cdef TaskCertificate value
        with nogil: value = owner.get().certificate(authority, callback)
        return _certificate(value)

    def original(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentSolveBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        cdef shared_ptr[AvailableAffineBridge] value
        with nogil: value = owner.get().original(authority, callback)
        return _original(value, self._limits, self._maximum)


cdef class NativeAssignmentResponse:
    cdef shared_ptr[AssignmentResponseBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('assignment response is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None, *, limits=_LIMITS, max_bytes=256*1024*1024, check=None):
        cdef NativeAssignmentResponse out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        try:
            with nogil: out._handle = AssignmentResponseBridge.restore(image, authority, bounds, maximum, callback)
        finally: _raise_check(state)
        out._limits = bounds; out._maximum = maximum
        return out

    def admit(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        try:
            with nogil: owner.get().admit(authority, callback)
        finally: _raise_check(state)

    def model(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        cdef shared_ptr[AssignmentModelBridge] value
        with nogil: value = owner.get().model(authority, callback)
        return _model(value, self._limits, self._maximum)

    def inspect(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        cdef AssignmentCoordinates value
        with nogil: value = owner.get().coordinates(authority, callback)
        return dict(candidate=owner.get().candidate(), coordinates=_coordinates(value))

    def origins(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef vector[shared_ptr[ValueOriginBridge]] values
        cdef NativeValueOrigin value
        with nogil: values = owner.get().origins(authority, maximum, callback)
        out = []
        for handle in values:
            value = NativeValueOrigin.__new__(NativeValueOrigin); value._handle = handle; out.append(value)
        return tuple(out)

    def dependency_archives(self, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t maximum = _natural(max_records)
        cdef GraphCheck callback
        cdef vector[string] values
        with nogil: values = owner.get().dependencies(authority, maximum, callback)
        return tuple(<bytes>value for value in values)

    def original(self, NativeAvailabilityContext execution_context not None):
        """Return the unconditioned original response for lineage inspection."""
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        cdef shared_ptr[AvailableAffineBridge] value
        with nogil: value = owner.get().original(authority, callback)
        return _original(value, self._limits, self._maximum)

    def field_archive(self, node, NativeAvailabilityContext execution_context not None, *, occurrences=False):
        if type(occurrences) is not bool: raise TypeError('occurrence selection requires bool')
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef uint64_t key = _natural(node)
        cdef cppbool flags = occurrences
        cdef GraphCheck callback
        cdef AffineFieldInfo value
        with nogil: value = owner.get().field(key, flags, authority, callback)
        return dict(particular=<bytes>value.particular, directions=<bytes>value.directions, dimension=value.dimension,
            supported=value.supported, nonzeros=value.nonzeros, free_parameter_count=value.free_parameters,
            direction_nonzeros=value.direction_nonzeros)

    def coefficient_independent_map(self, grade, flag, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef size_t selected_grade = _natural(grade), selected_flag = _natural(flag)
        cdef GraphCheck callback
        cdef AssignmentMapInfo value
        with nogil: value = owner.get().map(selected_grade, selected_flag, authority, callback)
        return dict(particular=<bytes>value.particular, directions=tuple(<bytes>v for v in value.directions))

    def postcondition(self, NativeTaskOutcome goal not None, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[TaskOutcomeBridge] requested = goal._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef GraphCheck callback
        cdef TaskCertificate value
        with nogil: value = owner.get().postcondition(requested, authority, callback)
        return _certificate(value)

    def restore_goal(self, archive, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef NativeTaskOutcome out = NativeTaskOutcome.__new__(NativeTaskOutcome)
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef string image = _image(archive, self._maximum)
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        try:
            with nogil: out._handle = owner.get().restore_goal(image, authority, callback)
        finally: _raise_check(state)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def archive(self, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        cdef string value
        try:
            with nogil: value = owner.get().archive(authority, callback)
        finally: _raise_check(state)
        return <bytes>value

    def stream(self, consume, NativeAvailabilityContext execution_context not None, *, check=None):
        cdef shared_ptr[AssignmentResponseBridge] owner = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef _Check state = _check(check)
        cdef GraphCheck callback = _callback(state)
        cdef _Sink sink = _Sink()
        sink.write = consume; sink.error = None
        cdef TaskSink target
        target.write = _write; target.context = <void*><object>sink
        cdef size_t count
        try:
            with nogil: count = owner.get().stream(target, authority, callback)
        finally:
            if sink.error is not None: raise sink.error
            _raise_check(state)
        return count
