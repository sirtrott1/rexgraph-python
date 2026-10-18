# distutils: language=c++
"""Owning admitted evidence, full correlated fields and global parameter settings."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport FieldProgramBridge, PreparedAccessionFieldProgram, _bounds, _maximum, _image, _indices
from ._field_evidence cimport (
    EvidenceDeclaration, FieldEvidenceBridge, AffineResultBridge, AffineEvaluationBridge,
    PreparedFieldEvidence, NativeAffineFieldResult, NativeAffineEvaluation, _evidence_declaration,
    NativeOccurrenceTargetPort, OccurrenceTargetBridge,
)
from ._field_admission cimport (
    AdmissionInput, AvailableProgramBridge, AvailableResultBridge, AvailabilityContextBridge,
    PreparedAvailableFieldProgram, NativeAvailableFieldResult, NativeAvailabilityContext, _input, _keys,
    NativeValueOrigin, ValueOriginBridge,
)
from ._stalk_fields import _natural, _scalar
from ._field_program import _closed


cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_EVIDENCE_ADMISSION_BINDING_DIGEST

_LIMITS = (4096, 16000000, 64000000, 16384)


def field_evidence_admission_build_info():
    return {'engine':'rexgraph-cpp', 'engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(), 'binding':RFC_FIELD_EVIDENCE_ADMISSION_BINDING_DIGEST.decode(),
            'profile':'owning-relational-field-evidence-admission-v1', 'scalar_policy':'gmp-rational-exact'}


cdef vector[AdmissionInput] _inputs(values, vector[size_t] bounds, size_t maximum):
    cdef vector[AdmissionInput] out
    for value in values:
        out.push_back(_input(value, bounds, maximum, True))
        if out.size() > bounds[0]: raise ValueError('native input count exceeds admission')
    return out


cdef vector[string] _free(values, vector[size_t] bounds):
    cdef vector[string] out
    for value in values:
        out.push_back(_scalar(value, bounds[3]))
        if out.size() > bounds[0]: raise ValueError('native parameter count exceeds admission')
    return out


cdef vector[AvailableEvidenceObservation] _observations(observations, vector[size_t] bounds, size_t maximum):
    cdef vector[AvailableEvidenceObservation] rows
    cdef AvailableEvidenceObservation row
    cdef NativeOccurrenceTargetPort port
    for value in observations:
        row.port.reset();row.observed.clear();row.observed_coordinates.clear()
        if 'port' in value:
            _closed(value, ('key','inputs','target_asset','target_use','target','port','observed_coordinates','origins'))
            if not isinstance(value['port'],NativeOccurrenceTargetPort):raise TypeError('actual native target port required')
            port=value['port'];row.port=port._pin()
            for k in value['observed_coordinates']:
                row.observed_coordinates.push_back(_natural(k))
                if row.observed_coordinates.size()>bounds[0]:raise ValueError('target mask admission exceeded')
            row.origins=_keys(value['origins'],('asset','value','port'))
        else:
            _closed(value, ('key','inputs','target_asset','target_use','target','observed','origins'))
            row.observed=_indices(value['observed']);row.origins=_keys(value['origins'],('asset','value'))
        row.key = _natural(value['key']); row.inputs = _inputs(value['inputs'], bounds, maximum)
        row.target_asset = _image(value['target_asset'], maximum); row.target_use = _natural(value['target_use'])
        row.target = _image(value['target'], maximum);rows.push_back(row)
        if rows.size() > bounds[0]: raise ValueError('native observation count exceeds admission')
    return rows


cdef object _origin_owners(vector[shared_ptr[ValueOriginBridge]] values):
    result=[]
    cdef NativeValueOrigin out
    for value in values:
        out=NativeValueOrigin.__new__(NativeValueOrigin);out._handle=value;result.append(out)
    return tuple(result)


def field_occurrence_target_origin_payload(NativeOccurrenceTargetPort port not None, asset, use, target, observed_coordinates, inputs,
                                          *, limits=_LIMITS, max_bytes=256*1024*1024):
    cdef vector[size_t] bounds=_bounds(limits)
    cdef size_t maximum=_maximum(max_bytes)
    cdef shared_ptr[OccurrenceTargetBridge] actual=port._pin()
    cdef string a=_image(asset,maximum), t=_image(target,maximum), image
    cdef uint64_t u=_natural(use)
    cdef vector[uint64_t] mask
    cdef vector[AdmissionInput] rows=_inputs(inputs,bounds,maximum)
    for k in observed_coordinates:
        mask.push_back(_natural(k))
        if mask.size()>bounds[0]:raise ValueError('target mask admission exceeded')
    with nogil:image=AvailableEvidenceBridge.occurrence_target_payload(actual,a,u,t,rows,mask,bounds,maximum)
    return <bytes>image


def field_evidence_origin_payload(PreparedAccessionFieldProgram program not None, parameters,
                                  *, limits=_LIMITS, max_bytes=256*1024*1024):
    cdef vector[size_t] bounds = _bounds(limits)
    cdef size_t maximum = _maximum(max_bytes)
    cdef EvidenceDeclaration declaration = _evidence_declaration(parameters, bounds)
    cdef shared_ptr[FieldProgramBridge] actual = program._pin()
    cdef string image
    with nogil: image = AvailableEvidenceBridge.root_payload(actual, declaration, bounds, maximum)
    return <bytes>image


def field_target_origin_payload(PreparedAccessionFieldProgram program not None, node, asset, use, target,
                                *, limits=_LIMITS, max_bytes=256*1024*1024):
    cdef vector[size_t] bounds = _bounds(limits)
    cdef size_t maximum = _maximum(max_bytes)
    cdef uint64_t selected = _natural(node), original = _natural(use)
    cdef string source = _image(asset, maximum), field = _image(target, maximum), image
    cdef shared_ptr[FieldProgramBridge] actual = program._pin()
    with nogil: image = AvailableEvidenceBridge.target_payload(actual, selected, source, original, field, bounds, maximum)
    return <bytes>image


cdef class PreparedAvailableFieldEvidence:
    cdef shared_ptr[AvailableEvidenceBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native available evidence is not initialized')
        return self._handle

    def __init__(self, PreparedAvailableFieldProgram program not None, parameters, origin,
                 NativeAvailabilityContext execution_context not None, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native available evidence owners are immutable')
        self._loading = True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef uint64_t key
        cdef EvidenceDeclaration declaration
        cdef shared_ptr[AvailableProgramBridge] actual
        cdef shared_ptr[AvailabilityContextBridge] authority
        try:
            bounds = _bounds(limits); maximum = _maximum(max_bytes); key = _natural(origin)
            declaration = _evidence_declaration(parameters, bounds); actual = program._pin(); authority = execution_context._pin()
            with nogil: self._handle = shared_ptr[AvailableEvidenceBridge](new AvailableEvidenceBridge(actual, declaration, key, authority, bounds, maximum))
            self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None,
                     *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef PreparedAvailableFieldEvidence out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        with nogil: out._handle = AvailableEvidenceBridge.restore(image, authority, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef string image
        with nogil: image = actual.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        with nogil: actual.get().admit(authority)

    def evidence(self):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef PreparedFieldEvidence out = PreparedFieldEvidence.__new__(PreparedFieldEvidence)
        with nogil: out._handle = actual.get().evidence()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def program(self):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef PreparedAvailableFieldProgram out = PreparedAvailableFieldProgram.__new__(PreparedAvailableFieldProgram)
        with nogil: out._handle = actual.get().program()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def parent(self):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin(), parent
        with nogil: parent = actual.get().parent()
        if parent.get() == NULL: return None
        cdef PreparedAvailableFieldEvidence out = PreparedAvailableFieldEvidence.__new__(PreparedAvailableFieldEvidence)
        out._handle = parent; out._limits = self._limits; out._maximum = self._maximum
        return out

    def origin_count(self):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef size_t count
        with nogil: count = actual.get().origin_count()
        return count

    def admission(self):
        self._pin()
        return {'limits':tuple(self._limits), 'max_bytes':self._maximum}

    def dependency_archives(self, *, max_records=4096):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[string] images
        with nogil: images = actual.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)

    def parameter_setting_payload(self, values):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef vector[string] free = _free(values, self._limits)
        cdef string image
        with nogil: image = actual.get().setting_payload(free)
        return <bytes>image

    def append_inputs(self, observations, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef vector[AvailableEvidenceObservation] rows = _observations(observations, self._limits, self._maximum)
        cdef PreparedAvailableFieldEvidence out = PreparedAvailableFieldEvidence.__new__(PreparedAvailableFieldEvidence)
        with nogil: out._handle = actual.get().append(rows, authority)
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def observation_dependencies(self, observations, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef vector[AvailableEvidenceObservation] rows = _observations(observations, self._limits, self._maximum)
        cdef size_t maximum = _natural(max_records)
        cdef AvailableEvidenceBatchInfo info
        with nogil: info = actual.get().batch_info(rows, authority, maximum)
        return {'origin_count':int(info.origin_count), 'dependencies':tuple(<bytes>image for image in info.dependencies)}

    def typed_observation_dependencies(self, observations, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AvailableEvidenceBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef vector[AvailableEvidenceObservation] rows=_observations(observations,self._limits,self._maximum)
        cdef size_t maximum=_natural(max_records)
        cdef AvailableEvidenceBatchInfo info
        with nogil:info=actual.get().batch_info(rows,authority,maximum)
        return {'origin_count':int(info.origin_count),'dependencies':tuple(<bytes>b for b in info.dependencies),'origins':_origin_owners(info.origins)}

    def observation_origins(self, index, NativeAvailabilityContext execution_context not None, *, max_records=4096):
        cdef shared_ptr[AvailableEvidenceBridge] actual=self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority=execution_context._pin()
        cdef size_t i=_natural(index), maximum=_natural(max_records)
        cdef vector[shared_ptr[ValueOriginBridge]] values
        with nogil:values=actual.get().observation_origins(i,authority,maximum)
        return _origin_owners(values)

    def execute_inputs(self, inputs, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableEvidenceBridge] actual = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef vector[AdmissionInput] rows = _inputs(inputs, self._limits, self._maximum)
        cdef NativeAvailableAffineResult out = NativeAvailableAffineResult.__new__(NativeAvailableAffineResult)
        with nogil: out._handle = actual.get().execute(rows, authority)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAvailableAffineResult:
    cdef shared_ptr[AvailableAffineBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native available affine result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None,
                     *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeAvailableAffineResult out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        with nogil: out._handle = AvailableAffineBridge.restore(image, authority, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AvailableAffineBridge] actual = self._pin()
        cdef string image
        with nogil: image = actual.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableAffineBridge] actual = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        with nogil: actual.get().admit(authority)

    def result(self):
        cdef shared_ptr[AvailableAffineBridge] actual = self._pin()
        cdef NativeAffineFieldResult out = NativeAffineFieldResult.__new__(NativeAffineFieldResult)
        with nogil: out._handle = actual.get().result()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def evidence(self):
        cdef shared_ptr[AvailableAffineBridge] actual = self._pin()
        cdef PreparedAvailableFieldEvidence out = PreparedAvailableFieldEvidence.__new__(PreparedAvailableFieldEvidence)
        with nogil: out._handle = actual.get().evidence()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def input(self):
        cdef shared_ptr[AvailableAffineBridge] actual = self._pin()
        cdef NativeAvailableFieldResult out = NativeAvailableFieldResult.__new__(NativeAvailableFieldResult)
        with nogil: out._handle = actual.get().input()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def dependency_archives(self, *, max_records=4096):
        cdef shared_ptr[AvailableAffineBridge] actual = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[string] images
        with nogil: images = actual.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)

    def evaluate(self, values, origin, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableAffineBridge] actual = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        cdef vector[string] free = _free(values, self._limits)
        cdef uint64_t key = _natural(origin)
        cdef NativeAvailableAffineEvaluation out = NativeAvailableAffineEvaluation.__new__(NativeAvailableAffineEvaluation)
        with nogil: out._handle = actual.get().evaluate(free, key, authority)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAvailableAffineEvaluation:
    cdef shared_ptr[AvailableEvaluationBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native available evaluation is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, NativeAvailabilityContext execution_context not None,
                     *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeAvailableAffineEvaluation out = cls.__new__(cls)
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string image = _image(archive, maximum)
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        with nogil: out._handle = AvailableEvaluationBridge.restore(image, authority, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AvailableEvaluationBridge] actual = self._pin()
        cdef string image
        with nogil: image = actual.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext execution_context not None):
        cdef shared_ptr[AvailableEvaluationBridge] actual = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = execution_context._pin()
        with nogil: actual.get().admit(authority)

    def result(self):
        cdef shared_ptr[AvailableEvaluationBridge] actual = self._pin()
        cdef NativeAffineEvaluation out = NativeAffineEvaluation.__new__(NativeAffineEvaluation)
        with nogil: out._handle = actual.get().result()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def family(self):
        cdef shared_ptr[AvailableEvaluationBridge] actual = self._pin()
        cdef NativeAvailableAffineResult out = NativeAvailableAffineResult.__new__(NativeAvailableAffineResult)
        with nogil: out._handle = actual.get().family()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def dependency_archives(self, *, max_records=4096):
        cdef shared_ptr[AvailableEvaluationBridge] actual = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[string] images
        with nogil: images = actual.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)
