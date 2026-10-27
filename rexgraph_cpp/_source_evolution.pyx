# distutils: language = c++
"""Owning native series forks and source-declaration authoring."""
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint16_t, uint64_t
from libc.stddef cimport size_t

cdef extern from "source_evolution_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass SourceAspectValue:
        uint16_t grade
        uint64_t occurrence
        bint value
    cdef cppclass SourceAspectAddition:
        string name
        unsigned role
        bint builder_default
        bint has_fill
        bint fill
        vector[SourceAspectValue] assignments
    cdef cppclass SourceTypeAddition:
        string name
        size_t minimum
        size_t maximum
    cdef cppclass SourceEvolutionBridge:
        @staticmethod
        shared_ptr[SourceEvolutionBridge] decode(const string&, const vector[size_t]&) except +
        string archive() except +
        vector[size_t] limits() except +
        vector[uint64_t] identity() except +
        shared_ptr[SourceEvolutionBridge] append(const string&) except +
        shared_ptr[SourceEvolutionBridge] add_declarations(const vector[SourceAspectAddition]&, const vector[SourceTypeAddition]&) except +

cdef extern from *:
    const char* RFC_ENGINE_DIGEST
    const char* RFC_ENGINE_VERSION
    const char* RFC_SOURCE_EVOLUTION_BINDING_DIGEST


def source_evolution_build_info():
    return {"engine": (<bytes>RFC_ENGINE_DIGEST).decode("ascii"),
            "version": (<bytes>RFC_ENGINE_VERSION).decode("ascii"),
            "binding": (<bytes>RFC_SOURCE_EVOLUTION_BINDING_DIGEST).decode("ascii"),
            "profile": "native-source-evolution-v1"}


cdef class NativeSourceEvolution:
    cdef shared_ptr[SourceEvolutionBridge] _owner
    cdef bint _loading

    def __init__(self, bytes archive, *, limits=(64 << 20, 1000000, 1000000, 16000000, 100000, 1000000, 64 << 20)):
        from ._source_evolution_spec import limits as checked_limits
        if self._loading or self._owner.get() != NULL:
            raise RuntimeError("native source evolution snapshots are immutable")
        bounds = checked_limits(limits)
        if not archive or len(archive) > bounds[0]:
            raise ValueError("native source archive exceeds byte admission")
        cdef string raw = archive
        cdef vector[size_t] policy = bounds
        self._loading = True
        try:
            with nogil:
                self._owner = SourceEvolutionBridge.decode(raw, policy)
        finally:
            self._loading = False

    cdef SourceEvolutionBridge* _pin(self) except NULL:
        if self._owner.get() == NULL:
            raise RuntimeError("native source evolution is not initialized")
        return self._owner.get()

    cdef NativeSourceEvolution _wrap(self, shared_ptr[SourceEvolutionBridge] value):
        cdef NativeSourceEvolution out = NativeSourceEvolution.__new__(NativeSourceEvolution)
        out._owner = value
        return out

    @property
    def archive(self):
        return <bytes>self._pin().archive()

    @property
    def identity(self):
        return tuple(self._pin().identity())

    @property
    def limits(self):
        return tuple(self._pin().limits())

    def append(self, bytes frame):
        if not frame or len(frame) > self.limits[0]:
            raise ValueError("native source frame exceeds byte admission")
        cdef string raw = frame
        cdef SourceEvolutionBridge* owner = self._pin()
        cdef shared_ptr[SourceEvolutionBridge] result
        with nogil:
            result = owner.append(raw)
        return self._wrap(result)

    def add_declarations(self, specification):
        from ._source_evolution_spec import additions
        aspects, types = additions(specification, self.limits)
        cdef vector[SourceAspectAddition] aa
        cdef vector[SourceTypeAddition] tt
        cdef SourceAspectAddition a
        cdef SourceAspectValue value
        cdef SourceTypeAddition t
        for row in aspects:
            a.name, a.role, a.builder_default, a.has_fill, a.fill = row[:5]
            a.assignments.clear()
            for item in row[5]:
                value.grade, value.occurrence, value.value = item
                a.assignments.push_back(value)
            aa.push_back(a)
        for row in types:
            t.name, t.minimum, t.maximum = row
            tt.push_back(t)
        cdef SourceEvolutionBridge* owner = self._pin()
        cdef shared_ptr[SourceEvolutionBridge] result
        with nogil:
            result = owner.add_declarations(aa, tt)
        return self._wrap(result)

    def source(self):
        """Return the existing owning NativeSource view of this complete publication."""
        from ._bridge import NativeSource
        bounds = self.limits
        return NativeSource(self.archive, max_bytes=bounds[0], max_records=bounds[1],
                            max_dimension=bounds[2], max_logical_size=bounds[3])
