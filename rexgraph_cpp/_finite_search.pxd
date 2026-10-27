from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._finite_field cimport FiniteSourceBridge, FiniteAssignmentValue

cdef extern from "rfc/local/constraint_system.hpp" namespace "rfc::local" nogil:
    cdef cppclass FiniteSearchReceipt:
        size_t steps, value_trials, chart_tests, tuple_tests, coordinate_tests
        size_t variables, peak_assigned, peak_entries, domain_reductions, search_nodes
        size_t distinct_groups, matching_tests
cdef extern from "rfc/inference/relational_finite_family.hpp" namespace "rfc::inference" nogil:
    cdef cppclass RelationalFiniteFamilyWork:
        size_t steps, peak_entries, candidates, completed_queries, preparations, source_queries, rows, witness_values
        size_t reused_preparations, prior_preparation_steps, reused_propagations, prior_propagation_steps, continued_source_queries
        size_t propagations, pruned_values
        FiniteSearchReceipt native
    cdef cppclass RelationalFiniteImageQuery:
        vector[uint64_t] tuple
        unsigned status
        size_t source_queries, native_steps
        optional[size_t] decisive_leaf

cdef extern from "finite_search_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass FiniteSearchStopBridge:
        FiniteSearchStopBridge() except +
        void cancel() except +
    cdef cppclass FiniteDomainRow:
        size_t grade, cell
        vector[uint64_t] values
    cdef cppclass FiniteProjectionInfo:
        uint64_t key, revision
        vector[vector[size_t]] cells, relations
        cppbool has_window
        vector[vector[uint64_t]] window
        vector[FiniteAssignmentValue] fixed
    cdef cppclass FiniteComparisonInfo:
        uint64_t key, revision
        vector[size_t] positions
        vector[vector[uint64_t]] symbols
    cdef cppclass FiniteFamilyInfo:
        uint64_t key, revision
        size_t leaves, retained_entries
        vector[vector[size_t]] cells
        vector[vector[uint64_t]] carriers, window
    cdef cppclass FiniteQuestionInfo:
        unsigned status
        size_t answers, observations, max_answers, checks, retained_values, left, right
        cppbool has_answer, has_obstruction
        vector[uint64_t] answer
    cdef cppclass FiniteClaimVerdict:
        unsigned status
        size_t satisfying, refuting
    cdef cppclass FinitePropagationEvidenceInfo:
        size_t leaf, variables
        unsigned status
        cppbool reused, has_snapshot
        FiniteSearchReceipt work
    cdef cppclass FinitePruningInfo:
        cppbool present, complete, has_unsatisfiable_leaf
        size_t unsatisfiable_leaf
        unsigned status
        vector[size_t] limits
        vector[vector[uint64_t]] window
        vector[FinitePropagationEvidenceInfo] propagations
    cdef cppclass FinitePreparedSearchBridge
    cdef cppclass FinitePropagationBridge
    cdef cppclass FiniteFamilyBridge
    cdef cppclass FiniteExtensionBridge
    cdef cppclass FiniteImageBridge
    cdef cppclass FiniteImageQueriesBridge
    cdef cppclass FinitePreparedReply:
        unsigned status
        shared_ptr[FinitePreparedSearchBridge] prepared
        FiniteSearchReceipt work
    cdef cppclass FinitePropagationReply:
        unsigned status
        shared_ptr[FinitePropagationBridge] snapshot
        FiniteSearchReceipt work
    cdef cppclass FiniteSectionReply:
        unsigned status
        cppbool has_witness
        vector[FiniteAssignmentValue] witness
        FiniteSearchReceipt work
    cdef cppclass FiniteExtensionReply:
        unsigned status
        shared_ptr[FiniteExtensionBridge] extension
        cppbool has_decisive_leaf
        size_t decisive_leaf
        RelationalFiniteFamilyWork work
    cdef cppclass FiniteImageReply:
        unsigned status
        shared_ptr[FiniteImageBridge] image
        shared_ptr[FiniteImageQueriesBridge] queries
        RelationalFiniteFamilyWork work
    cdef cppclass FinitePreparedSearchBridge:
        @staticmethod
        FinitePreparedReply prepare(shared_ptr[FiniteSourceBridge], const vector[vector[size_t]]&, const vector[size_t]&, shared_ptr[FiniteSearchStopBridge], const vector[size_t]&, size_t) except +
        shared_ptr[FiniteSourceBridge] source() except +
        cppbool same_owner(const FinitePreparedSearchBridge&) except +
        size_t retained_entries() except +
        FiniteSearchReceipt work() except +
        vector[vector[size_t]] groups(size_t) except +
        FiniteSectionReply solve(const vector[FiniteAssignmentValue]&, const vector[size_t]&, shared_ptr[FiniteSearchStopBridge]) except +
        FinitePropagationReply propagate(const vector[FiniteAssignmentValue]&, const vector[size_t]&, shared_ptr[FiniteSearchStopBridge]) except +
    cdef cppclass FinitePropagationBridge:
        shared_ptr[FinitePreparedSearchBridge] prepared() except +
        shared_ptr[FiniteSourceBridge] source() except +
        cppbool same_owner(const FinitePropagationBridge&) except +
        size_t retained_entries() except +
        size_t size() except +
        FiniteSearchReceipt work() except +
        vector[FiniteAssignmentValue] fixed(size_t) except +
        vector[FiniteDomainRow] domains(size_t, size_t, size_t) except +
        FiniteSectionReply solve(const vector[FiniteAssignmentValue]&, const vector[size_t]&, shared_ptr[FiniteSearchStopBridge]) except +
    cdef cppclass FiniteFamilyBridge:
        @staticmethod
        shared_ptr[FiniteFamilyBridge] projection(shared_ptr[FiniteSourceBridge], shared_ptr[FinitePreparedSearchBridge], shared_ptr[FinitePropagationBridge], const FiniteProjectionInfo&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FiniteFamilyBridge] intersect(shared_ptr[FiniteFamilyBridge], shared_ptr[FiniteFamilyBridge], const FiniteComparisonInfo&, const vector[size_t]&, size_t) except +
        cppbool same_owner(const FiniteFamilyBridge&) except +
        shared_ptr[FiniteSourceBridge] source(size_t) except +
        shared_ptr[FinitePropagationBridge] propagation(size_t) except +
        shared_ptr[FiniteFamilyBridge] parent(cppbool) except +
        FiniteProjectionInfo declaration(size_t, size_t) except +
        cppbool has_comparison() except +
        FiniteComparisonInfo comparison(size_t) except +
        FiniteFamilyInfo source_mapping(size_t, size_t) except +
        FiniteFamilyInfo info(size_t) except +
        FiniteExtensionReply extend(const vector[uint64_t]&, const vector[size_t]&, shared_ptr[FiniteSearchStopBridge]) except +
        FiniteImageReply materialize(const vector[size_t]&, shared_ptr[FiniteSearchStopBridge]) except +
    cdef cppclass FiniteExtensionBridge:
        shared_ptr[FiniteFamilyBridge] family() except +
        vector[uint64_t] tuple(size_t) except +
        size_t leaf_count() except +
        size_t field_size(size_t) except +
        vector[FiniteAssignmentValue] witness(size_t, size_t, size_t, size_t) except +
    cdef cppclass FiniteImageBridge:
        FiniteQuestionInfo analyze_question(const vector[vector[size_t]]&, const vector[vector[size_t]]&, size_t, shared_ptr[FiniteSearchStopBridge]) except +
        vector[FiniteClaimVerdict] classify_equalities(const vector[vector[FiniteAssignmentValue]]&, size_t) except +
        string proof_archive(size_t) except +
        FinitePruningInfo pruning(size_t) except +
        shared_ptr[FinitePropagationBridge] pruning_root(size_t) except +
        shared_ptr[FiniteFamilyBridge] family() except +
        size_t size() except +
        shared_ptr[FiniteExtensionBridge] row(size_t) except +
        size_t query_count() except +
        vector[RelationalFiniteImageQuery] queries(size_t, size_t, size_t) except +
    cdef cppclass FiniteImageQueriesBridge:
        string proof_archive(size_t) except +
        FinitePruningInfo pruning(size_t) except +
        shared_ptr[FinitePropagationBridge] pruning_root(size_t) except +
        shared_ptr[FiniteFamilyBridge] family() except +
        size_t query_count() except +
        vector[RelationalFiniteImageQuery] queries(size_t, size_t, size_t) except +
    cdef cppclass FiniteSourceMapInfo:
        vector[vector[size_t]] cells, occurrences
    cdef cppclass FiniteGluingInfo:
        uint64_t key, revision, symbol_key, symbol_revision
        FiniteSourceMapInfo common_left, common_right, left_glued, right_glued
        vector[vector[uint64_t]] left_symbols, right_symbols
    cdef cppclass FiniteGluingBridge
    cdef cppclass FiniteGluedSectionBridge
    cdef cppclass FiniteGluingReply:
        unsigned status
        shared_ptr[FiniteGluedSectionBridge] section
        RelationalFiniteFamilyWork work
    cdef cppclass FiniteGluingBridge:
        FiniteGluingBridge(const string&, shared_ptr[FiniteSourceBridge], shared_ptr[FiniteSourceBridge], shared_ptr[FiniteSourceBridge], const FiniteGluingInfo&, const vector[size_t]&, size_t) except +
        cppbool same_owner(const FiniteGluingBridge&) except +
        size_t retained_entries() except +
        size_t preparation_steps() except +
        FiniteGluingInfo maps(size_t) except +
        shared_ptr[FiniteSourceBridge] source(unsigned) except +
        FiniteGluingReply glue(shared_ptr[FiniteExtensionBridge], size_t, shared_ptr[FiniteExtensionBridge], size_t, const vector[size_t]&, shared_ptr[FiniteSearchStopBridge]) except +
    cdef cppclass FiniteGluedSectionBridge:
        shared_ptr[FiniteGluingBridge] gluing() except +
        shared_ptr[FiniteExtensionBridge] parent(cppbool) except +
        size_t leaf(cppbool) except +
        size_t size() except +
        vector[FiniteAssignmentValue] field(size_t, size_t, size_t) except +

cdef class FiniteSearchCancellation:
    cdef shared_ptr[FiniteSearchStopBridge] _handle
    cdef shared_ptr[FiniteSearchStopBridge] _pin(self) except *
cdef class PreparedFiniteSearch:
    cdef shared_ptr[FinitePreparedSearchBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FinitePreparedSearchBridge] _pin(self) except *
cdef class NativeFinitePropagation:
    cdef shared_ptr[FinitePropagationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FinitePropagationBridge] _pin(self) except *
cdef class NativeFiniteFamily:
    cdef shared_ptr[FiniteFamilyBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteFamilyBridge] _pin(self) except *
cdef class NativeFiniteExtension:
    cdef shared_ptr[FiniteExtensionBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteExtensionBridge] _pin(self) except *
cdef class NativeFiniteImage:
    cdef shared_ptr[FiniteImageBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteImageBridge] _pin(self) except *
cdef class NativeFiniteImageQueries:
    cdef shared_ptr[FiniteImageQueriesBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteImageQueriesBridge] _pin(self) except *
cdef class NativeFiniteGluing:
    cdef shared_ptr[FiniteGluingBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FiniteGluingBridge] _pin(self) except *
cdef class NativeFiniteGluedSection:
    cdef shared_ptr[FiniteGluedSectionBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FiniteGluedSectionBridge] _pin(self) except *
