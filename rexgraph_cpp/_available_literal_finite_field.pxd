from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_admission cimport AvailabilityContextBridge
from ._finite_field_admission cimport AvailableFiniteMapBridge
from ._literal_finite_field cimport LiteralFiniteMapBridge
from ._finite_field cimport FiniteResultBridge, FiniteBranchBridge, FiniteEvaluationBridge

cdef extern from "available_literal_finite_field_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass AvailableLiteralMapBridge
    cdef cppclass AvailableLiteralResultBridge
    cdef cppclass AvailableLiteralBranchBridge
    cdef cppclass AvailableLiteralEvaluationBridge
    cdef cppclass AvailableLiteralMapBridge:
        AvailableLiteralMapBridge(shared_ptr[AvailableFiniteMapBridge], shared_ptr[LiteralFiniteMapBridge], const vector[vector[uint64_t]]&, uint64_t, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string interface_payload(shared_ptr[AvailableFiniteMapBridge], shared_ptr[LiteralFiniteMapBridge], const vector[vector[uint64_t]]&, shared_ptr[AvailabilityContextBridge], size_t) except +
        @staticmethod
        shared_ptr[AvailableLiteralMapBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        cppbool same_owner(const AvailableLiteralMapBridge&) except +
        shared_ptr[AvailableFiniteMapBridge] base() except +
        shared_ptr[LiteralFiniteMapBridge] reader() except +
        vector[vector[uint64_t]] inputs(size_t) except +
        uint64_t interface() except +
        vector[string] dependencies(size_t) except +
        shared_ptr[AvailableLiteralResultBridge] execute(shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableLiteralResultBridge:
        @staticmethod
        shared_ptr[AvailableLiteralResultBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableLiteralMapBridge] model() except +
        shared_ptr[FiniteResultBridge] raw() except +
        size_t size() except +
        vector[shared_ptr[AvailableLiteralBranchBridge]] select(size_t, size_t, shared_ptr[AvailabilityContextBridge]) except +
        string completion_payload(size_t, const vector[string]&, uint64_t, shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableLiteralEvaluationBridge] evaluate(size_t, const vector[string]&, uint64_t, uint64_t, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableLiteralBranchBridge:
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableLiteralResultBridge] family() except +
        shared_ptr[FiniteBranchBridge] raw() except +
    cdef cppclass AvailableLiteralEvaluationBridge:
        @staticmethod
        shared_ptr[AvailableLiteralEvaluationBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableLiteralResultBridge] family() except +
        shared_ptr[FiniteEvaluationBridge] raw() except +
        string setting_receipt() except +
        string choice_receipt() except +
        vector[string] dependencies(size_t) except +

cdef class PreparedAvailableLiteralFiniteFieldMap:
    cdef shared_ptr[AvailableLiteralMapBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableLiteralMapBridge] _pin(self) except *

cdef class NativeAvailableLiteralFiniteFieldResult:
    cdef shared_ptr[AvailableLiteralResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableLiteralResultBridge] _pin(self) except *

cdef class NativeAvailableLiteralFiniteFieldBranch:
    cdef shared_ptr[AvailableLiteralBranchBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableLiteralBranchBridge] _pin(self) except *

cdef class NativeAvailableLiteralFiniteFieldEvaluation:
    cdef shared_ptr[AvailableLiteralEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableLiteralEvaluationBridge] _pin(self) except *
