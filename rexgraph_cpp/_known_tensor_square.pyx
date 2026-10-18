# distutils: language=c++
"""Owning known context products with independent native execution authority."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_admission cimport (
    AvailableProgramBridge, AvailabilityContextBridge, AdmissionInput,
    PreparedAvailableFieldProgram, NativeAvailabilityContext, _input, _keys,
)
from ._field_interface cimport (
    FieldProgramBridge, InterfaceAccession, InterfaceInputInfo, AccessionResultBridge,
    PreparedAccessionFieldProgram, NativeAccessionFieldResult,
    _bounds, _maximum, _image, _accession_declaration, _input_info,
)
from ._stalk_fields import _natural
from ._field_program import _closed

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_KNOWN_TENSOR_SQUARE_BINDING_DIGEST


_LIMITS = (70000, 16000000, 64000000, 16384)
_INPUT_KEYS = ('asset', 'value', 'accession', 'reconstruction')


def known_tensor_square_build_info():
    return {'engine': 'rexgraph-cpp', 'engine_version': RFC_ENGINE_VERSION.decode(),
            'engine_headers': RFC_ENGINE_DIGEST.decode(), 'binding': RFC_KNOWN_TENSOR_SQUARE_BINDING_DIGEST.decode(),
            'profile': 'available-known-context-tensor-square-v1', 'scalar_policy': 'gmp-rational-exact'}


cdef KnownSquareDeclaration _declaration(value, size_t maximum):
    _closed(value, ('profile', 'key', 'revision', 'lifted_basis', 'context', 'pair_directions'))
    if value['profile'] != 'rexgraph.known-context-tensor-square.v1':
        raise ValueError('declared native known context product required')
    cdef KnownSquareDeclaration out
    out.key = _natural(value['key']); out.revision = _natural(value['revision'])
    out.basis = _natural(value['lifted_basis'])
    if not out.key or not out.revision or not out.basis:
        raise ValueError('positive declared identity, revision and enlarged basis required')
    if len(value['context']) != 2:
        raise ValueError('actual original context grade and cell required')
    out.grade = _natural(value['context'][0]); out.cell = _natural(value['context'][1])
    if len(value['pair_directions']) > maximum:
        raise ValueError('bounded complete ordered pair identities required')
    for key in value['pair_directions']: out.pairs.push_back(_natural(key))
    return out


def known_tensor_square_origin_payload(PreparedAccessionFieldProgram program not None, declaration, *,
                                      max_pairs=16000000, max_bytes=256*1024*1024):
    """Passive complete declaration payload for an actual Interface publisher."""
    cdef shared_ptr[FieldProgramBridge] parent = program._pin()
    cdef size_t maximum = _maximum(max_bytes)
    cdef KnownSquareDeclaration d = _declaration(declaration, _natural(max_pairs))
    cdef string value
    with nogil: value = AvailableKnownSquareProducerBridge.declaration_payload(parent, d, maximum)
    return <bytes>value


cdef class PreparedAvailableKnownTensorSquare:

    cdef shared_ptr[AvailableKnownSquareProducerBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native known-product producer is not initialized')
        return self._handle

    def __init__(self, PreparedAvailableFieldProgram program not None, declaration, origin,
                 NativeAvailabilityContext context not None, *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get() != NULL: raise TypeError('native known-product producers are immutable')
        self._loading = True
        cdef shared_ptr[AvailableProgramBridge] parent
        cdef shared_ptr[AvailabilityContextBridge] authority
        cdef KnownSquareDeclaration d
        cdef uint64_t key
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef shared_ptr[AvailableKnownSquareProducerBridge] handle
        try:
            parent = program._pin(); authority = context._pin(); bounds = _bounds(limits); maximum = _maximum(max_bytes)
            d = _declaration(declaration, bounds[1]); key = _natural(origin)
            with nogil: handle = shared_ptr[AvailableKnownSquareProducerBridge](new AvailableKnownSquareProducerBridge(
                parent, d, key, authority, bounds, maximum))
            self._handle = handle; self._limits = bounds; self._maximum = maximum
        finally: self._loading = False

    def admit(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableKnownSquareProducerBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        with nogil: handle.get().admit(authority)

    def lift(self, submission, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableKnownSquareProducerBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef AdmissionInput row = _input(submission, self._limits, self._maximum, True)
        cdef NativeAvailableKnownTensorSquare out = NativeAvailableKnownTensorSquare.__new__(NativeAvailableKnownTensorSquare)
        with nogil: out._handle = handle.get().lift(row, authority)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAvailableKnownTensorSquare:

    cdef shared_ptr[AvailableKnownSquareBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native whole known product is not initialized')
        return self._handle

    def admit(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        with nogil: handle.get().admit(authority)

    def field_archive(self, kind='lifted'):
        if kind not in ('context', 'square', 'lifted'): raise ValueError('whole known field kind required')
        cdef unsigned selected = ('context', 'square', 'lifted').index(kind)
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef string value
        with nogil: value = handle.get().field(selected)
        return <bytes>value

    def metric_archive(self):
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef string value
        with nogil: value = handle.get().metric()
        return <bytes>value

    def archive(self):
        """Raw mathematical inspection packet; supplies no executable authority."""
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef string value
        with nogil: value = handle.get().archive()
        return <bytes>value

    def parent(self):
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef InterfaceInputInfo value
        with nogil: value = handle.get().parent()
        return _input_info(value)

    def pair_coordinate(self, index):
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef size_t selected = _natural(index)
        cdef KnownSquarePair value
        with nogil: value = handle.get().pair(selected)
        return {'context': (value.grade, value.cell), 'original_basis': value.basis,
                'left': value.left, 'right': value.right, 'lifted_direction': value.direction}

    def require_image(self, field, *, max_bytes=256*1024*1024):
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef string value = _image(field, _maximum(max_bytes))
        with nogil: handle.get().require_image(value)

    def input_origin_payloads(self, PreparedAccessionFieldProgram program not None, accession, *, max_records=16000000):
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef shared_ptr[FieldProgramBridge] target = program._pin()
        cdef InterfaceAccession d = _accession_declaration(accession, _natural(max_records))
        cdef vector[string] images
        with nogil: images = handle.get().input_payloads(target, d)
        return dict(zip(_INPUT_KEYS, tuple(<bytes>image for image in images), strict=True))

    def admit_input(self, PreparedAvailableFieldProgram program not None, accession, origins,
                    NativeAvailabilityContext context not None, *, max_records=16000000):
        cdef shared_ptr[AvailableKnownSquareBridge] handle = self._pin()
        cdef shared_ptr[AvailableProgramBridge] target = program._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef InterfaceAccession d = _accession_declaration(accession, _natural(max_records))
        cdef vector[uint64_t] keys = _keys(origins, _INPUT_KEYS)
        cdef NativeAvailableKnownTensorSquareInput out = NativeAvailableKnownTensorSquareInput.__new__(NativeAvailableKnownTensorSquareInput)
        with nogil: out._handle = handle.get().admit_input(target, d, keys, authority)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAvailableKnownTensorSquareInput:

    cdef shared_ptr[AvailableKnownSquareInputBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native compound known input is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, image, NativeAvailabilityContext context not None, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string value = _image(image, maximum)
        cdef NativeAvailableKnownTensorSquareInput out = cls.__new__(cls)
        with nogil: out._handle = AvailableKnownSquareInputBridge.restore(value, authority, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AvailableKnownSquareInputBridge] handle = self._pin()
        cdef string value
        with nogil: value = handle.get().archive()
        return <bytes>value

    def admit(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableKnownSquareInputBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        with nogil: handle.get().admit(authority)

    def parent(self):
        cdef shared_ptr[AvailableKnownSquareInputBridge] handle = self._pin()
        cdef InterfaceInputInfo value
        with nogil: value = handle.get().parent()
        return _input_info(value)

    def input(self):
        """Inspect the complete derived field and source maps without detaching its owner."""
        cdef shared_ptr[AvailableKnownSquareInputBridge] handle = self._pin()
        cdef InterfaceInputInfo value
        with nogil: value = handle.get().input()
        return _input_info(value)

    def dependency_receipts(self, *, max_records=4096):
        cdef shared_ptr[AvailableKnownSquareInputBridge] handle = self._pin()
        cdef size_t maximum = _natural(max_records)
        cdef vector[string] images
        with nogil: images = handle.get().dependencies(maximum)
        return tuple(<bytes>image for image in images)

    def execute(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableKnownSquareInputBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef NativeAvailableKnownTensorSquareResult out = NativeAvailableKnownTensorSquareResult.__new__(NativeAvailableKnownTensorSquareResult)
        with nogil: out._handle = handle.get().execute(authority)
        out._limits = self._limits; out._maximum = self._maximum
        return out


cdef class NativeAvailableKnownTensorSquareResult:

    cdef shared_ptr[AvailableKnownSquareResultBridge] _pin(self) except *:
        if self._handle.get() == NULL: raise RuntimeError('native compound result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, image, NativeAvailabilityContext context not None, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        cdef vector[size_t] bounds = _bounds(limits)
        cdef size_t maximum = _maximum(max_bytes)
        cdef string value = _image(image, maximum)
        cdef NativeAvailableKnownTensorSquareResult out = cls.__new__(cls)
        with nogil: out._handle = AvailableKnownSquareResultBridge.restore(value, authority, bounds, maximum)
        out._limits = bounds; out._maximum = maximum
        return out

    def archive(self):
        cdef shared_ptr[AvailableKnownSquareResultBridge] handle = self._pin()
        cdef string value
        with nogil: value = handle.get().archive()
        return <bytes>value

    def admit(self, NativeAvailabilityContext context not None):
        cdef shared_ptr[AvailableKnownSquareResultBridge] handle = self._pin()
        cdef shared_ptr[AvailabilityContextBridge] authority = context._pin()
        with nogil: handle.get().admit(authority)

    def input(self):
        cdef shared_ptr[AvailableKnownSquareResultBridge] handle = self._pin()
        cdef NativeAvailableKnownTensorSquareInput out = NativeAvailableKnownTensorSquareInput.__new__(NativeAvailableKnownTensorSquareInput)
        with nogil: out._handle = handle.get().input()
        out._limits = self._limits; out._maximum = self._maximum
        return out

    def result(self):
        """Mathematical inspection; retain this compound owner for serving and replay."""
        cdef shared_ptr[AvailableKnownSquareResultBridge] handle = self._pin()
        cdef NativeAccessionFieldResult out = NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle = handle.get().result()
        out._limits = self._limits; out._maximum = self._maximum
        return out
