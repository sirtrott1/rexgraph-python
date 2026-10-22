from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport InputUseAssetBridge, AccessionInputBridge, AccessionResultBridge, FieldProgramBridge
from ._field_evidence cimport FieldEvidenceBridge, AffineResultBridge, AffineEvaluationBridge

cdef extern from "finite_field_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass FiniteSymbol:
        uint64_t key
        string definition
    cdef cppclass FiniteVariable:
        size_t grade, cell
        vector[uint64_t] domain
    cdef cppclass FiniteChart:
        uint64_t key
        string name
        vector[vector[size_t]] patch, support
        vector[vector[uint64_t]] allowed
    cdef cppclass FiniteBinding:
        uint64_t key, use
        size_t grade, cell
    cdef cppclass FiniteDeclaration:
        uint64_t key, revision
        vector[FiniteSymbol] symbols
        vector[FiniteVariable] variables
        vector[FiniteChart] charts
        vector[FiniteBinding] bindings
    cdef cppclass FiniteAssignmentValue:
        size_t grade, cell
        uint64_t symbol
    cdef cppclass FiniteSearchInfo:
        unsigned status
        cppbool has_witness
        vector[FiniteAssignmentValue] witness
        size_t steps, value_trials, chart_tests, tuple_tests, coordinate_tests
        size_t variables, peak_assigned, peak_entries, domain_reductions, search_nodes
    cdef cppclass FiniteTarget:
        shared_ptr[FieldProgramBridge] fixed
        shared_ptr[FieldEvidenceBridge] affine
    cdef cppclass FiniteRule:
        uint64_t key
        vector[uint64_t] bindings, symbols
        cppbool has_target
        size_t target
        string unsupported
        vector[shared_ptr[AccessionInputBridge]] inputs
    cdef cppclass FiniteMapDeclaration:
        uint64_t key, revision
        vector[FiniteRule] rules
    cdef cppclass FiniteBranchInfo:
        size_t ordinal, section, target
        uint64_t rule
        unsigned status
        cppbool has_rule, has_target, affine
        vector[uint64_t] assignment
    cdef cppclass FiniteSectionsBridge
    cdef cppclass FiniteResultBridge
    cdef cppclass FiniteBranchBridge
    cdef cppclass FiniteEvaluationBridge
    cdef cppclass FiniteSourceBridge:
        FiniteSourceBridge(shared_ptr[InputUseAssetBridge], const FiniteDeclaration&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FiniteSourceBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        FiniteDeclaration info(size_t) except +
        shared_ptr[InputUseAssetBridge] asset() except +
        cppbool same_owner(const FiniteSourceBridge&) except +
        shared_ptr[FiniteSectionsBridge] sections() except +
        FiniteSearchInfo solve(const vector[FiniteAssignmentValue]&, size_t, size_t, size_t, cppbool) except +
    cdef cppclass FiniteSectionsBridge:
        shared_ptr[FiniteSourceBridge] source() except +
        size_t size() except +
        size_t candidates() except +
        vector[vector[uint64_t]] select(size_t, size_t) except +
    cdef cppclass FiniteMapBridge:
        FiniteMapBridge(shared_ptr[FiniteSourceBridge], const vector[FiniteTarget]&, const FiniteMapDeclaration&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FiniteMapBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FiniteSourceBridge] source() except +
        vector[FiniteTarget] targets(size_t) except +
        FiniteMapDeclaration info(size_t) except +
        shared_ptr[FiniteResultBridge] execute() except +
    cdef cppclass FiniteResultBridge:
        @staticmethod
        shared_ptr[FiniteResultBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FiniteMapBridge] model() except +
        shared_ptr[FiniteSectionsBridge] sections() except +
        size_t size() except +
        vector[shared_ptr[FiniteBranchBridge]] select(size_t, size_t) except +
        shared_ptr[FiniteEvaluationBridge] evaluate(size_t, const vector[string]&) except +
    cdef cppclass FiniteBranchBridge:
        shared_ptr[FiniteResultBridge] family() except +
        FiniteBranchInfo info() except +
        shared_ptr[AccessionResultBridge] fixed() except +
        shared_ptr[AffineResultBridge] affine() except +
    cdef cppclass FiniteEvaluationBridge:
        @staticmethod
        shared_ptr[FiniteEvaluationBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FiniteResultBridge] family() except +
        size_t ordinal() except +
        shared_ptr[AccessionResultBridge] result() except +
        shared_ptr[AffineEvaluationBridge] affine() except +
        string free_values() except +

cdef class NativeFiniteSource:
    cdef shared_ptr[FiniteSourceBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FiniteSourceBridge] _pin(self) except *
cdef class NativeFiniteSections:
    cdef shared_ptr[FiniteSectionsBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteSectionsBridge] _pin(self) except *
cdef class PreparedFiniteFieldMap:
    cdef shared_ptr[FiniteMapBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FiniteMapBridge] _pin(self) except *
cdef class NativeFiniteFieldResult:
    cdef shared_ptr[FiniteResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteResultBridge] _pin(self) except *
cdef class NativeFiniteFieldBranch:
    cdef shared_ptr[FiniteBranchBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteBranchBridge] _pin(self) except *
cdef class NativeFiniteFieldEvaluation:
    cdef shared_ptr[FiniteEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteEvaluationBridge] _pin(self) except *

cdef FiniteDeclaration _declaration(value, vector[size_t] bounds, size_t maximum)

cdef object _declaration_info(FiniteDeclaration value)
