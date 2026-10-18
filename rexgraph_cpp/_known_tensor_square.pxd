from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_admission cimport AvailableProgramBridge, AvailabilityContextBridge, AdmissionInput
from ._field_interface cimport FieldProgramBridge, InterfaceAccession, InterfaceInputInfo, AccessionResultBridge

cdef extern from "known_tensor_square_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass KnownSquareDeclaration:
        uint64_t key, revision, basis
        size_t grade, cell
        vector[uint64_t] pairs
    cdef cppclass KnownSquarePair:
        size_t grade, cell
        uint64_t basis, left, right, direction
    cdef cppclass AvailableKnownSquareBridge
    cdef cppclass AvailableKnownSquareInputBridge
    cdef cppclass AvailableKnownSquareResultBridge
    cdef cppclass AvailableKnownSquareProducerBridge:
        AvailableKnownSquareProducerBridge(shared_ptr[AvailableProgramBridge], const KnownSquareDeclaration&,
            uint64_t, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string declaration_payload(shared_ptr[FieldProgramBridge], const KnownSquareDeclaration&, size_t) except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableKnownSquareBridge] lift(const AdmissionInput&, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableKnownSquareBridge:
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        string field(unsigned) except +
        string metric() except +
        string archive() except +
        InterfaceInputInfo parent() except +
        KnownSquarePair pair(size_t) except +
        void require_image(const string&) except +
        vector[string] input_payloads(shared_ptr[FieldProgramBridge], const InterfaceAccession&) except +
        shared_ptr[AvailableKnownSquareInputBridge] admit_input(shared_ptr[AvailableProgramBridge],
            const InterfaceAccession&, const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableKnownSquareInputBridge:
        @staticmethod
        shared_ptr[AvailableKnownSquareInputBridge] restore(const string&, shared_ptr[AvailabilityContextBridge],
            const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        InterfaceInputInfo parent() except +
        InterfaceInputInfo input() except +
        vector[string] dependencies(size_t) except +
        shared_ptr[AvailableKnownSquareResultBridge] execute(shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableKnownSquareResultBridge:
        @staticmethod
        shared_ptr[AvailableKnownSquareResultBridge] restore(const string&, shared_ptr[AvailabilityContextBridge],
            const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[AvailableKnownSquareInputBridge] input() except +
        shared_ptr[AccessionResultBridge] result() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +


cdef class PreparedAvailableKnownTensorSquare:
    cdef shared_ptr[AvailableKnownSquareProducerBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableKnownSquareProducerBridge] _pin(self) except *


cdef class NativeAvailableKnownTensorSquare:
    cdef shared_ptr[AvailableKnownSquareBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableKnownSquareBridge] _pin(self) except *


cdef class NativeAvailableKnownTensorSquareInput:
    cdef shared_ptr[AvailableKnownSquareInputBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableKnownSquareInputBridge] _pin(self) except *


cdef class NativeAvailableKnownTensorSquareResult:
    cdef shared_ptr[AvailableKnownSquareResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableKnownSquareResultBridge] _pin(self) except *
