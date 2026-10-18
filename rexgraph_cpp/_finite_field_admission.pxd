from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport InputUseAssetBridge, InterfaceAccession
from ._field_admission cimport AvailableProgramBridge, AvailabilityContextBridge
from ._field_evidence_admission cimport AvailableEvidenceBridge
from ._finite_field cimport FiniteDeclaration, FiniteSourceBridge, FiniteSectionsBridge, FiniteMapBridge, FiniteResultBridge, FiniteBranchBridge, FiniteEvaluationBridge

cdef extern from "finite_field_admission_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass AvailableFiniteTarget:
        shared_ptr[AvailableProgramBridge] fixed
        shared_ptr[AvailableEvidenceBridge] affine
    cdef cppclass AvailableFiniteInput:
        shared_ptr[InputUseAssetBridge] asset
        InterfaceAccession accession
        uint64_t use
        string field
        vector[uint64_t] origins
    cdef cppclass AvailableFiniteRule:
        uint64_t key
        vector[uint64_t] bindings, symbols
        cppbool has_target
        size_t target
        string unsupported
        vector[AvailableFiniteInput] inputs
    cdef cppclass AvailableFiniteMapDeclaration:
        uint64_t key, revision
        vector[AvailableFiniteRule] rules
    cdef cppclass AvailableFiniteSourceBridge
    cdef cppclass AvailableFiniteSectionsBridge
    cdef cppclass AvailableFiniteSectionBridge
    cdef cppclass AvailableFiniteMapBridge
    cdef cppclass AvailableFiniteResultBridge
    cdef cppclass AvailableFiniteBranchBridge
    cdef cppclass AvailableFiniteEvaluationBridge
    cdef cppclass AvailableFiniteSourceBridge:
        @staticmethod
        shared_ptr[AvailableFiniteSourceBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteSourceBridge] raw() except +
        AvailableFiniteSourceBridge(shared_ptr[InputUseAssetBridge], const FiniteDeclaration&, unsigned, const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string constraint_payload(shared_ptr[InputUseAssetBridge], const FiniteDeclaration&, unsigned, size_t) except +
        unsigned status() except +
        vector[uint64_t] origins() except +
        cppbool same_owner(const AvailableFiniteSourceBridge&) except +
        vector[string] receipts(size_t) except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[AvailableFiniteSectionsBridge] sections(shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableFiniteSectionsBridge:
        @staticmethod
        shared_ptr[AvailableFiniteSectionsBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteSectionsBridge] raw() except +
        shared_ptr[AvailableFiniteSourceBridge] source() except +
        vector[shared_ptr[AvailableFiniteSectionBridge]] select(size_t, size_t, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableFiniteSectionBridge:
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableFiniteSectionsBridge] family() except +
        size_t ordinal() except +
        vector[uint64_t] assignment(size_t) except +
    cdef cppclass AvailableFiniteMapBridge:
        @staticmethod
        shared_ptr[AvailableFiniteMapBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteMapBridge] raw() except +
        AvailableFiniteMapBridge(shared_ptr[AvailableFiniteSourceBridge], const vector[AvailableFiniteTarget]&, const AvailableFiniteMapDeclaration&, unsigned, const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string interpretation_payload(shared_ptr[AvailableFiniteSourceBridge], const vector[AvailableFiniteTarget]&, const AvailableFiniteMapDeclaration&, unsigned, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        unsigned status() except +
        vector[uint64_t] origins() except +
        cppbool same_owner(const AvailableFiniteMapBridge&) except +
        shared_ptr[AvailableFiniteSourceBridge] source() except +
        vector[AvailableFiniteTarget] targets(size_t) except +
        vector[string] receipts(size_t) except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[AvailableFiniteResultBridge] execute(shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableFiniteResultBridge:
        @staticmethod
        shared_ptr[AvailableFiniteResultBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteResultBridge] raw() except +
        shared_ptr[AvailableFiniteMapBridge] model() except +
        size_t size() except +
        vector[shared_ptr[AvailableFiniteBranchBridge]] select(size_t, size_t, shared_ptr[AvailabilityContextBridge]) except +
        string completion_payload(size_t, const vector[string]&, uint64_t, shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableFiniteEvaluationBridge] evaluate(size_t, const vector[string]&, uint64_t, uint64_t, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableFiniteBranchBridge:
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteBranchBridge] raw() except +
        shared_ptr[AvailableFiniteResultBridge] family() except +
        size_t ordinal() except +
    cdef cppclass AvailableFiniteEvaluationBridge:
        @staticmethod
        shared_ptr[AvailableFiniteEvaluationBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FiniteEvaluationBridge] raw() except +
        shared_ptr[AvailableFiniteResultBridge] family() except +
        size_t ordinal() except +
        string setting_receipt() except +
        string choice_receipt() except +
        vector[string] dependency_archives(size_t) except +

cdef class PreparedAvailableFiniteSource:
    cdef shared_ptr[AvailableFiniteSourceBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteSourceBridge] _pin(self) except *

cdef class NativeAvailableFiniteSections:
    cdef shared_ptr[AvailableFiniteSectionsBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteSectionsBridge] _pin(self) except *

cdef class NativeAvailableFiniteSection:
    cdef shared_ptr[AvailableFiniteSectionBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteSectionBridge] _pin(self) except *

cdef class PreparedAvailableFiniteFieldMap:
    cdef shared_ptr[AvailableFiniteMapBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteMapBridge] _pin(self) except *

cdef class NativeAvailableFiniteFieldResult:
    cdef shared_ptr[AvailableFiniteResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteResultBridge] _pin(self) except *

cdef class NativeAvailableFiniteFieldBranch:
    cdef shared_ptr[AvailableFiniteBranchBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteBranchBridge] _pin(self) except *

cdef class NativeAvailableFiniteFieldEvaluation:
    cdef shared_ptr[AvailableFiniteEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableFiniteEvaluationBridge] _pin(self) except *

cdef unsigned _status(value) except *
cdef vector[string] _free(value,vector[size_t] bounds)
