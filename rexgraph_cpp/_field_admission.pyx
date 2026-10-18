# distutils: language=c++
"""Native value publishers and independently supplied field execution authority."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport (
    FieldProgramBridge, AccessionResultBridge, InterfaceAccession,
    PreparedAccessionFieldProgram, NativeAccessionFieldResult,
    _bounds, _maximum, _image, _accession_declaration,
)
from ._stalk_fields import _natural
from ._field_program import _closed


cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_ADMISSION_BINDING_DIGEST

_LIMITS = (4096, 16000000, 64000000, 16384)
_ADMISSION = (4096, 64*1024*1024, 256*1024*1024)
_ROLES = ('program', 'maps', 'metrics', 'input-asset', 'input-value', 'input-accession',
          'reconstruction', 'target-asset', 'target-value', 'interface', 'evidence', 'prospective')
_PROGRAM_KEYS = ('program', 'maps', 'metrics')
_INPUT_KEYS = ('asset', 'value', 'accession', 'reconstruction')


def field_admission_build_info():
    return {'engine': 'rexgraph-cpp', 'engine_version': RFC_ENGINE_VERSION.decode(),
            'engine_headers': RFC_ENGINE_DIGEST.decode(), 'binding': RFC_FIELD_ADMISSION_BINDING_DIGEST.decode(),
            'profile': 'owning-relational-field-admission-v1', 'scalar_policy': 'gmp-rational-exact'}


cdef vector[size_t] _admission_bounds(value):
    value = tuple(value)
    if len(value) != 3: raise ValueError('three native origin admission bounds required')
    cdef vector[size_t] out
    for n in value:
        if not _natural(n): raise ValueError('positive native origin admission bounds required')
        out.push_back(_natural(n))
    return out


cdef vector[uint64_t] _keys(value, names):
    _closed(value, names)
    cdef vector[uint64_t] out
    for name in names:
        if not _natural(value[name]): raise ValueError('positive actual origin key required')
        out.push_back(_natural(value[name]))
    return out


cdef AdmissionInput _input(value, vector[size_t] bounds, size_t maximum, bint require_origins):
    _closed(value, ('asset', 'accession', 'use', 'field', 'origins') if require_origins else
                  ('asset', 'accession', 'use', 'field'))
    cdef AdmissionInput out
    out.asset = _image(value['asset'], maximum); out.field = _image(value['field'], maximum)
    out.accession = _accession_declaration(value['accession'], bounds[3]); out.use = _natural(value['use'])
    if require_origins: out.origins = _keys(value['origins'], _INPUT_KEYS)
    return out


def field_input_origin_payloads(PreparedAccessionFieldProgram program not None, submission, *,
                              limits=_LIMITS, max_bytes=256*1024*1024):
    """Passively encode complete native dependencies for an explicit publisher."""
    cdef shared_ptr[FieldProgramBridge] parent = program._pin()
    cdef vector[size_t] bounds = _bounds(limits)
    cdef size_t maximum = _maximum(max_bytes)
    cdef AdmissionInput value = _input(submission, bounds, maximum, False)
    cdef vector[string] images
    with nogil: images = AvailableProgramBridge.input_payloads(parent, value, bounds, maximum)
    return dict(zip(_INPUT_KEYS, tuple(<bytes>image for image in images), strict=True))


cdef class NativeValueOrigin:

    cdef shared_ptr[ValueOriginBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native value origin is not initialized')
        return self._handle

    def __init__(self, key, history, cutoff, role, payload, *, limits=_LIMITS,
                 max_bytes=256*1024*1024, admission_limits=_ADMISSION):
        """Publish an immutable value at an explicitly declared authentic prefix."""
        if self._loading or self._handle.get() != NULL: raise TypeError('native value origins are immutable')
        if role not in _ROLES: raise ValueError('explicit native dependency role required')
        self._loading = True
        cdef shared_ptr[ValueOriginBridge] handle
        cdef vector[size_t] bounds, admission
        cdef uint64_t original, sequence
        cdef size_t maximum
        cdef unsigned selected
        cdef string source, content
        try:
            bounds = _bounds(limits); admission = _admission_bounds(admission_limits); maximum = _maximum(max_bytes)
            original = _natural(key); sequence = _natural(cutoff); selected = _ROLES.index(role)
            source = _image(history, maximum); content = _image(payload, maximum)
            with nogil: handle = shared_ptr[ValueOriginBridge](new ValueOriginBridge(
                original, source, sequence, selected, content, bounds, maximum, admission))
            self._handle = handle
        finally: self._loading = False

    def inspect(self):
        cdef shared_ptr[ValueOriginBridge] handle = self._pin()
        cdef AdmissionOriginInfo info
        with nogil: info = handle.get().info()
        return {'key': int(info.key), 'namespace': int(info.name_space), 'first_sequence': int(info.first_sequence),
                'sequence': int(info.sequence), 'role': _ROLES[info.role], 'payload_bytes': int(info.payload_bytes)}

    def payload(self):
        cdef shared_ptr[ValueOriginBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().payload()
        return <bytes>image

    def receipt(self):
        cdef shared_ptr[ValueOriginBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().receipt()
        return <bytes>image

    def source_archive(self):
        cdef shared_ptr[ValueOriginBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().source_archive()
        return <bytes>image

    def same_owner(self, NativeValueOrigin other not None):
        cdef shared_ptr[ValueOriginBridge] left = self._pin(), right = other._pin()
        cdef cppbool same
        with nogil: same = left.get().same_owner(right.get()[0])
        return bool(same)


cdef class NativeAvailabilityContext:

    cdef shared_ptr[AvailabilityContextBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native availability context is not initialized')
        return self._handle

    def __init__(self, grants, *, admission_limits=_ADMISSION):
        if self._loading or self._handle.get() != NULL: raise TypeError('native availability contexts are immutable')
        self._loading = True
        cdef vector[AdmissionGrant] values
        cdef AdmissionGrant value
        cdef vector[size_t] admission
        cdef shared_ptr[AvailabilityContextBridge] handle
        try:
            admission = _admission_bounds(admission_limits)
            for grant in tuple(grants):
                _closed(grant, ('origin', 'namespace', 'sequence'))
                if not isinstance(grant['origin'], NativeValueOrigin): raise TypeError('actual native value origin required')
                value.origin = (<NativeValueOrigin>grant['origin'])._pin()
                value.name_space = _natural(grant['namespace']); value.sequence = _natural(grant['sequence'])
                values.push_back(value)
            with nogil: handle = shared_ptr[AvailabilityContextBridge](new AvailabilityContextBridge(values, admission))
            self._handle = handle
        finally: self._loading = False

    def require(self, key, role, payload, *, max_bytes=256*1024*1024):
        """Return the actual granted native origin for complete typed content."""
        if role not in _ROLES: raise ValueError('explicit native dependency role required')
        cdef shared_ptr[AvailabilityContextBridge] owner = self._pin()
        cdef uint64_t original = _natural(key)
        cdef unsigned selected = _ROLES.index(role)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string content = _image(payload, maximum)
        cdef NativeValueOrigin out = NativeValueOrigin.__new__(NativeValueOrigin)
        with nogil: out._handle = owner.get().require(original, selected, content, maximum)
        return out

    def recheck(self, NativeValueOrigin origin not None):
        """Recheck the same actual origin owner against current authority."""
        cdef shared_ptr[AvailabilityContextBridge] owner = self._pin()
        cdef shared_ptr[ValueOriginBridge] expected = origin._pin()
        with nogil: owner.get().recheck(expected)


cdef class PreparedAvailableFieldProgram:

    cdef shared_ptr[AvailableProgramBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native available program is not initialized')
        return self._handle

    def __init__(self, PreparedAccessionFieldProgram program not None, origins,
                 NativeAvailabilityContext context not None, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native available programs are immutable')
        self._loading = True
        cdef shared_ptr[FieldProgramBridge] parent
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef shared_ptr[AvailableProgramBridge] handle
        cdef vector[uint64_t] keys
        cdef vector[size_t] bounds
        cdef size_t maximum
        try:
            parent = program._pin(); authority = context._pin(); keys = _keys(origins, _PROGRAM_KEYS)
            bounds = _bounds(limits); maximum = _maximum(max_bytes)
            with nogil: handle = shared_ptr[AvailableProgramBridge](new AvailableProgramBridge(parent, keys, authority, bounds, maximum))
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    @classmethod
    def from_archive(cls, image, NativeAvailabilityContext context not None, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef PreparedAvailableFieldProgram out = cls.__new__(cls)
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string value = _image(image, maximum)
        with nogil: out._handle = AvailableProgramBridge.restore(value, authority, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AvailableProgramBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableProgramBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        with nogil: handle.get().admit(authority)

    def program(self):
        cdef shared_ptr[AvailableProgramBridge] handle = self._pin()
        cdef PreparedAccessionFieldProgram out = PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil: out._handle = handle.get().program()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def dependency_archives(self, *, max_records=256):
        cdef shared_ptr[AvailableProgramBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[string] images
        with nogil: images = handle.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)

    def execute_inputs(self, inputs, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableProgramBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef vector[AdmissionInput] declarations
        for value in tuple(inputs): declarations.push_back(_input(value, self._limits, self._maximum, True))
        cdef NativeAvailableFieldResult out = NativeAvailableFieldResult.__new__(NativeAvailableFieldResult)
        with nogil: out._handle = handle.get().execute(declarations, authority)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAvailableFieldResult:

    cdef shared_ptr[AvailableResultBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native available result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, image, NativeAvailabilityContext context not None, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeAvailableFieldResult out = cls.__new__(cls)
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string value = _image(image, maximum)
        with nogil: out._handle = AvailableResultBridge.restore(value, authority, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AvailableResultBridge] handle = self._pin()
        cdef string image
        with nogil: image = handle.get().archive()
        return <bytes>image

    def admit(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableResultBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        with nogil: handle.get().admit(authority)

    def result(self):
        cdef shared_ptr[AvailableResultBridge] handle = self._pin()
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle = handle.get().result()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def program_archive(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableResultBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef string image
        with nogil: image = handle.get().program_archive(authority)
        return <bytes>image

    def dependency_archives(self, *, max_records=256):
        cdef shared_ptr[AvailableResultBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[string] images
        with nogil: images = handle.get().dependency_archives(maximum)
        return tuple(<bytes>image for image in images)
