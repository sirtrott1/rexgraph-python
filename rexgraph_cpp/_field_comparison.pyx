# distutils: language=c++
"""Actual source native factored comparison actions and complete observations."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport (
    FieldProgramBridge, AccessionResultBridge, ProgramSelected, PreparedAccessionFieldProgram,
    NativeAccessionFieldResult, _bounds, _maximum, _image, _indices, _selected,
)
from ._stalk_fields import _natural, _scalar, _values
from ._field_program import _closed

cdef extern from "field_comparison_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ComparisonWeight:
        size_t grade, cell
        string value
    cdef cppclass ComparisonDeclaration:
        uint64_t key, revision
        size_t grade
        unsigned participation
        vector[ComparisonWeight] weights
    cdef cppclass ComparisonInfo:
        uint64_t key, revision, node
        size_t grade, dimension, supported, nonzeros, occurrences
        unsigned participation, channel
        string quadrance
    cdef cppclass ComparisonPairInfo:
        size_t first, second, grade, cell
        unsigned channel
        cppbool admitted
        string residual, metric, weight, quadrance
    cdef cppclass ComparisonResultBridge
    cdef cppclass ComparisonPairBridge
    cdef cppclass FieldComparisonBridge:
        FieldComparisonBridge(shared_ptr[FieldProgramBridge], const ComparisonDeclaration&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FieldComparisonBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FieldProgramBridge] program() except +
        ComparisonInfo identity() except +
        void validate_node(uint64_t) except +
        ComparisonDeclaration declaration(size_t) except +
        string operator_matrix(unsigned, size_t) except +
        shared_ptr[ComparisonResultBridge] apply(shared_ptr[AccessionResultBridge], uint64_t, unsigned) except +
    cdef cppclass ComparisonResultBridge:
        @staticmethod
        shared_ptr[ComparisonResultBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        string native_archive() except +
        string forms_archive() except +
        shared_ptr[AccessionResultBridge] input() except +
        ComparisonInfo info() except +
        string field(unsigned) except +
        string occurrence_field(size_t) except +
        vector[ProgramSelected] select(const vector[vector[size_t]]&, size_t) except +
        shared_ptr[ComparisonPairBridge] pair(size_t, size_t, unsigned) except +
    cdef cppclass ComparisonPairBridge:
        ComparisonPairInfo info() except +
        shared_ptr[ComparisonResultBridge] parent() except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_FIELD_COMPARISON_BINDING_DIGEST

_LIMITS = (4096,16000000,64000000,16384)
_CHANNELS = ('T','rawG','C','F')
_POLICIES = ('share','legacy_count')
_ROLES = ('input','output','signed_boundary','unsigned_boundary')


def field_comparison_build_info():
    return {'engine':'rexgraph-cpp','engine_version':RFC_ENGINE_VERSION.decode(),
            'engine_headers':RFC_ENGINE_DIGEST.decode(),'binding':RFC_FIELD_COMPARISON_BINDING_DIGEST.decode(),
            'profile':'owning-relational-field-comparison-v1','scalar_policy':'gmp-rational-exact'}


cdef object _rational(string value):
    cdef vector[string] values
    values.push_back(value)
    return _values(values)[0]


cdef unsigned _channel(value) except *:
    if value not in _CHANNELS: raise ValueError('explicit T/rawG/C/F channel required')
    return _CHANNELS.index(value)


cdef class PreparedFieldComparison
cdef class NativeComparisonFieldResult
cdef class NativeComparisonPair


cdef class PreparedFieldComparison:
    cdef shared_ptr[FieldComparisonBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading

    cdef shared_ptr[FieldComparisonBridge] _pin(self) except *:
        if self._handle.get()==NULL: raise RuntimeError('native comparison forms are not initialized')
        return self._handle

    def __init__(self, PreparedAccessionFieldProgram program not None, declaration,
                 *, limits=_LIMITS, max_bytes=256*1024*1024):
        if self._loading or self._handle.get()!=NULL: raise TypeError('native comparison owners are immutable')
        self._loading=True
        cdef vector[size_t] bounds
        cdef size_t maximum
        cdef ComparisonDeclaration d
        cdef ComparisonWeight w
        cdef shared_ptr[FieldProgramBridge] actual
        cdef shared_ptr[FieldComparisonBridge] handle
        try:
            _closed(declaration,('profile','key','revision','grade','participation','weights'))
            if declaration['profile']!='rexgraph.relational-comparison-forms.v1': raise ValueError('unknown native comparison profile')
            if declaration['participation'] not in _POLICIES: raise ValueError('explicit share/legacy_count participation required')
            bounds=_bounds(limits);maximum=_maximum(max_bytes);actual=program._pin()
            d.key=_natural(declaration['key']);d.revision=_natural(declaration['revision']);d.grade=_natural(declaration['grade'])
            d.participation=_POLICIES.index(declaration['participation'])
            for value in declaration['weights']:
                _closed(value,('cell','value'));w.grade=_natural(value['cell'][0]);w.cell=_natural(value['cell'][1])
                if len(value['cell'])!=2: raise ValueError('actual upper weight grade and cell required')
                w.value=_scalar(value['value'],bounds[3]);d.weights.push_back(w)
                if d.weights.size()>bounds[0]: raise ValueError('comparison weight count exceeds admission')
            with nogil: handle=shared_ptr[FieldComparisonBridge](new FieldComparisonBridge(actual,d,bounds,maximum))
            self._handle=handle;self._limits=bounds;self._maximum=maximum
        finally: self._loading=False

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef PreparedFieldComparison out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil: out._handle=FieldComparisonBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum
        return out

    def archive(self):
        cdef shared_ptr[FieldComparisonBridge] handle=self._pin()
        cdef string image
        with nogil: image=handle.get().archive()
        return <bytes>image

    def identity(self):
        cdef shared_ptr[FieldComparisonBridge] handle=self._pin()
        cdef ComparisonInfo d
        with nogil: d=handle.get().identity()
        return {'key':int(d.key),'revision':int(d.revision),'grade':int(d.grade),
                'participation':_POLICIES[d.participation],'dimension':int(d.dimension),'occurrences':int(d.occurrences)}

    def validate_node(self,node):
        cdef shared_ptr[FieldComparisonBridge] handle=self._pin()
        cdef uint64_t original=_natural(node)
        with nogil: handle.get().validate_node(original)

    def inspect(self, *, max_records=256):
        cdef shared_ptr[FieldComparisonBridge] handle=self._pin()
        cdef size_t maximum=_natural(max_records)
        cdef ComparisonDeclaration d
        with nogil: d=handle.get().declaration(maximum)
        return {'key':int(d.key),'revision':int(d.revision),'grade':int(d.grade),'participation':_POLICIES[d.participation],
                'weights':tuple({'cell':(int(w.grade),int(w.cell)),'value':_rational(w.value)} for w in d.weights)}

    def program(self):
        cdef shared_ptr[FieldComparisonBridge] handle=self._pin()
        cdef PreparedAccessionFieldProgram out=PreparedAccessionFieldProgram.__new__(PreparedAccessionFieldProgram)
        with nogil: out._handle=handle.get().program()
        out._limits=self._limits;out._maximum=self._maximum
        return out

    def operator_matrix(self, channel, *, max_entries):
        cdef shared_ptr[FieldComparisonBridge] handle=self._pin()
        cdef unsigned selected=_channel(channel)
        cdef size_t maximum=_natural(max_entries)
        cdef string image
        with nogil: image=handle.get().operator_matrix(selected,maximum)
        return <bytes>image

    def apply(self, NativeAccessionFieldResult input not None, *, node, channel):
        cdef shared_ptr[FieldComparisonBridge] handle=self._pin()
        cdef shared_ptr[AccessionResultBridge] actual=input._pin()
        cdef uint64_t original=_natural(node)
        cdef unsigned selected=_channel(channel)
        cdef NativeComparisonFieldResult out=NativeComparisonFieldResult.__new__(NativeComparisonFieldResult)
        with nogil: out._handle=handle.get().apply(actual,original,selected)
        out._limits=self._limits;out._maximum=self._maximum
        return out


cdef class NativeComparisonFieldResult:
    cdef shared_ptr[ComparisonResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum

    cdef shared_ptr[ComparisonResultBridge] _pin(self) except *:
        if self._handle.get()==NULL: raise RuntimeError('native source-backed comparison result is not initialized')
        return self._handle

    @classmethod
    def from_archive(cls, archive, *, limits=_LIMITS, max_bytes=256*1024*1024):
        cdef NativeComparisonFieldResult out=cls.__new__(cls)
        cdef vector[size_t] bounds=_bounds(limits)
        cdef size_t maximum=_maximum(max_bytes)
        cdef string image=_image(archive,maximum)
        with nogil: out._handle=ComparisonResultBridge.restore(image,bounds,maximum)
        out._limits=bounds;out._maximum=maximum
        return out

    def archive(self):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef string image
        with nogil: image=handle.get().archive()
        return <bytes>image

    def native_archive(self):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef string image
        with nogil: image=handle.get().native_archive()
        return <bytes>image

    def forms_archive(self):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef string image
        with nogil: image=handle.get().forms_archive()
        return <bytes>image

    def input(self):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef NativeAccessionFieldResult out=NativeAccessionFieldResult.__new__(NativeAccessionFieldResult)
        with nogil: out._handle=handle.get().input()
        out._limits=self._limits;out._maximum=self._maximum
        return out

    def inspect(self):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef ComparisonInfo d
        with nogil: d=handle.get().info()
        return {'key':int(d.key),'revision':int(d.revision),'node':int(d.node),'grade':int(d.grade),
                'channel':_CHANNELS[d.channel],'participation':_POLICIES[d.participation],
                'dimension':int(d.dimension),'supported':int(d.supported),'nonzeros':int(d.nonzeros),
                'occurrences':int(d.occurrences),'quadrance':_rational(d.quadrance)}

    def field_archive(self, role='output'):
        if role not in _ROLES: raise ValueError('explicit input/output/signed_boundary/unsigned_boundary role required')
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef unsigned selected=_ROLES.index(role)
        cdef string image
        with nogil: image=handle.get().field(selected)
        return <bytes>image

    def occurrence_field_archive(self, occurrence):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef size_t original=_natural(occurrence)
        cdef string image
        with nogil: image=handle.get().occurrence_field(original)
        return <bytes>image

    def select(self, cells, *, coordinate_limit=256):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef vector[vector[size_t]] requested=_indices(cells)
        cdef size_t maximum=_natural(coordinate_limit)
        cdef vector[ProgramSelected] values
        with nogil: values=handle.get().select(requested,maximum)
        return tuple(_selected(v) for v in values)

    def pair(self, first, second, *, channel):
        cdef shared_ptr[ComparisonResultBridge] handle=self._pin()
        cdef size_t p=_natural(first),q=_natural(second)
        cdef unsigned selected=_channel(channel)
        cdef NativeComparisonPair out=NativeComparisonPair.__new__(NativeComparisonPair)
        with nogil: out._handle=handle.get().pair(p,q,selected)
        out._limits=self._limits;out._maximum=self._maximum
        return out


cdef class NativeComparisonPair:
    cdef shared_ptr[ComparisonPairBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum

    cdef shared_ptr[ComparisonPairBridge] _pin(self) except *:
        if self._handle.get()==NULL: raise RuntimeError('native comparison pair is not initialized')
        return self._handle

    def inspect(self):
        cdef shared_ptr[ComparisonPairBridge] handle=self._pin()
        cdef ComparisonPairInfo d
        with nogil: d=handle.get().info()
        return {'first':int(d.first),'second':int(d.second),'lower':(int(d.grade),int(d.cell)),
                'channel':_CHANNELS[d.channel],'admitted':bool(d.admitted),'residual':<bytes>d.residual,'metric':<bytes>d.metric,
                'weight':_rational(d.weight),'quadrance':_rational(d.quadrance)}

    def parent(self):
        cdef shared_ptr[ComparisonPairBridge] handle=self._pin()
        cdef NativeComparisonFieldResult out=NativeComparisonFieldResult.__new__(NativeComparisonFieldResult)
        with nogil: out._handle=handle.get().parent()
        out._limits=self._limits;out._maximum=self._maximum
        return out
