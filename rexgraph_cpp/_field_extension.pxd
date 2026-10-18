from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport ProgramMatrix, ProgramFieldInfo, FieldProgramBridge, FieldProgramResultBridge, InputAccessionBridge, AccessionResultBridge
from ._field_evidence cimport AffineFieldInfo, AffineSelected
from ._field_admission cimport AdmissionInput, AvailableProgramBridge, AvailabilityContextBridge
from ._field_evidence_admission cimport AvailableEvaluationBridge

cdef extern from "field_extension_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass ExtensionStalk:
        size_t grade, cell
        uint64_t source_basis, target_basis
        ProgramMatrix inclusion, reconstruction
    cdef cppclass ExtensionDeclaration:
        uint64_t key, revision, free_carrier, free_basis
        vector[uint64_t] free_directions
        vector[vector[size_t]] cells, occurrences
        vector[ExtensionStalk] stalks
        cppbool isometric
    cdef cppclass ExtensionInfo:
        uint64_t key, revision, free_carrier, free_basis
        unsigned status
        cppbool has_parent, isometric
        vector[uint64_t] free_directions
        vector[vector[size_t]] cells, occurrences
        vector[vector[string]] grade_maps
    cdef cppclass ExtensionFamilyBridge
    cdef cppclass ExtensionEvaluationBridge
    cdef cppclass ExtensionModelBridge:
        ExtensionModelBridge(shared_ptr[AvailableProgramBridge], shared_ptr[AvailableEvaluationBridge], shared_ptr[AvailableProgramBridge],
            const ExtensionDeclaration&, unsigned, uint64_t, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[ExtensionModelBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string root_payload(shared_ptr[AvailableProgramBridge], shared_ptr[AvailableEvaluationBridge], shared_ptr[AvailableProgramBridge],
            const ExtensionDeclaration&, unsigned, const vector[size_t]&, size_t) except +
        string extension_payload(shared_ptr[AvailableProgramBridge], const ExtensionDeclaration&, unsigned) except +
        shared_ptr[ExtensionModelBridge] extend(shared_ptr[AvailableProgramBridge], const ExtensionDeclaration&, unsigned, uint64_t, shared_ptr[AvailabilityContextBridge]) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FieldProgramBridge] source() except +
        shared_ptr[AvailableProgramBridge] target() except +
        shared_ptr[ExtensionModelBridge] parent() except +
        shared_ptr[AvailableProgramBridge] fixed_parent() except +
        shared_ptr[AvailableEvaluationBridge] selected_parent() except +
        ExtensionInfo info() except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[InputAccessionBridge] transport(shared_ptr[InputAccessionBridge], uint64_t, shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[ExtensionFamilyBridge] execute(const vector[AdmissionInput]&, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass ExtensionFamilyBridge:
        @staticmethod
        shared_ptr[ExtensionFamilyBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[ExtensionModelBridge] model() except +
        shared_ptr[AccessionResultBridge] input() except +
        AffineFieldInfo field(uint64_t, cppbool) except +
        vector[AffineSelected] select(uint64_t, const vector[vector[size_t]]&, size_t) except +
        vector[ProgramFieldInfo] fields(size_t) except +
        string completion_payload(const vector[string]&) except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[ExtensionEvaluationBridge] evaluate(const vector[string]&, uint64_t, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass ExtensionEvaluationBridge:
        @staticmethod
        shared_ptr[ExtensionEvaluationBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[ExtensionFamilyBridge] family() except +
        shared_ptr[FieldProgramResultBridge] result() except +
        string free_values() except +
        vector[string] dependency_archives(size_t) except +

cdef class PreparedAvailableFieldExtension:
    cdef shared_ptr[ExtensionModelBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[ExtensionModelBridge] _pin(self) except *

cdef class NativeAvailableExtensionFamily:
    cdef shared_ptr[ExtensionFamilyBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[ExtensionFamilyBridge] _pin(self) except *

cdef class NativeAvailableExtensionEvaluation:
    cdef shared_ptr[ExtensionEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[ExtensionEvaluationBridge] _pin(self) except *
