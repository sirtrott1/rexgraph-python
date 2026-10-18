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
cdef extern from "field_admission_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass AdmissionOriginInfo:
        uint64_t key, name_space, first_sequence, sequence
        unsigned role
        size_t payload_bytes
    cdef cppclass ValueOriginBridge:
        ValueOriginBridge(uint64_t, const string&, uint64_t, unsigned, const string&,
                          const vector[size_t]&, size_t, const vector[size_t]&) except +
        AdmissionOriginInfo info() except +
        string payload() except +
        string receipt() except +
        string source_archive() except +
        cppbool same_owner(const ValueOriginBridge&) except +
    cdef cppclass AdmissionGrant:
        shared_ptr[ValueOriginBridge] origin
        uint64_t name_space, sequence
    cdef cppclass AvailabilityContextBridge:
        AvailabilityContextBridge(const vector[AdmissionGrant]&, const vector[size_t]&) except +
        shared_ptr[ValueOriginBridge] require(uint64_t, unsigned, const string&, size_t) except +
        void recheck(shared_ptr[ValueOriginBridge]) except +
    cdef cppclass AdmissionInput:
        string asset, field
        InterfaceAccession accession
        uint64_t use
        vector[uint64_t] origins
    cdef cppclass AvailableResultBridge:
        @staticmethod
        shared_ptr[AvailableResultBridge] restore(const string&, shared_ptr[AvailabilityContextBridge],
                                                  const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AccessionResultBridge] result() except +
        string program_archive(shared_ptr[AvailabilityContextBridge]) except +
        vector[string] dependency_archives(size_t) except +
    cdef cppclass AvailableProgramBridge:
        AvailableProgramBridge(shared_ptr[FieldProgramBridge], const vector[uint64_t]&,
                               shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[AvailableProgramBridge] restore(const string&, shared_ptr[AvailabilityContextBridge],
                                                   const vector[size_t]&, size_t) except +
        @staticmethod
        vector[string] input_payloads(shared_ptr[FieldProgramBridge], const AdmissionInput&,
                                     const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FieldProgramBridge] program() except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[AvailableResultBridge] execute(const vector[AdmissionInput]&, shared_ptr[AvailabilityContextBridge]) except +


cdef class NativeValueOrigin:
    cdef shared_ptr[ValueOriginBridge] _handle
    cdef bint _loading
    cdef shared_ptr[ValueOriginBridge] _pin(self) except *

cdef class NativeAvailabilityContext:
    cdef shared_ptr[AvailabilityContextBridge] _handle
    cdef bint _loading
    cdef shared_ptr[AvailabilityContextBridge] _pin(self) except *

cdef class PreparedAvailableFieldProgram:
    cdef shared_ptr[AvailableProgramBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableProgramBridge] _pin(self) except *

cdef class NativeAvailableFieldResult:
    cdef shared_ptr[AvailableResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableResultBridge] _pin(self) except *

cdef vector[uint64_t] _keys(value, names)
cdef AdmissionInput _input(value, vector[size_t] bounds, size_t maximum, bint require_origins)
