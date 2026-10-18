from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport FieldProgramBridge
from ._field_evidence cimport EvidenceDeclaration, FieldEvidenceBridge, AffineResultBridge, AffineEvaluationBridge, OccurrenceTargetBridge
from ._field_admission cimport AdmissionInput, AvailableProgramBridge, AvailableResultBridge, AvailabilityContextBridge, ValueOriginBridge

cdef extern from "field_evidence_admission_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass AvailableEvidenceObservation:
        uint64_t key, target_use
        vector[AdmissionInput] inputs
        string target_asset, target
        vector[vector[size_t]] observed
        vector[uint64_t] origins
        shared_ptr[OccurrenceTargetBridge] port
        vector[uint64_t] observed_coordinates
    cdef cppclass AvailableEvidenceBatchInfo:
        size_t origin_count
        vector[string] dependencies
        vector[shared_ptr[ValueOriginBridge]] origins
    cdef cppclass AvailableAffineBridge
    cdef cppclass AvailableEvaluationBridge
    cdef cppclass AvailableEvidenceBridge:
        AvailableEvidenceBridge(shared_ptr[AvailableProgramBridge], const EvidenceDeclaration&, uint64_t,
            shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[AvailableEvidenceBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string root_payload(shared_ptr[FieldProgramBridge], const EvidenceDeclaration&, const vector[size_t]&, size_t) except +
        @staticmethod
        string target_payload(shared_ptr[FieldProgramBridge], uint64_t, const string&, uint64_t, const string&, const vector[size_t]&, size_t) except +
        @staticmethod
        string occurrence_target_payload(shared_ptr[OccurrenceTargetBridge], const string&, uint64_t, const string&, const vector[AdmissionInput]&, const vector[uint64_t]&, const vector[size_t]&, size_t) except +
        string setting_payload(const vector[string]&) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[FieldEvidenceBridge] evidence() except +
        shared_ptr[AvailableProgramBridge] program() except +
        shared_ptr[AvailableEvidenceBridge] parent() except +
        size_t origin_count() except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[AvailableEvidenceBridge] append(const vector[AvailableEvidenceObservation]&, shared_ptr[AvailabilityContextBridge]) except +
        AvailableEvidenceBatchInfo batch_info(const vector[AvailableEvidenceObservation]&, shared_ptr[AvailabilityContextBridge], size_t) except +
        vector[shared_ptr[ValueOriginBridge]] observation_origins(size_t, shared_ptr[AvailabilityContextBridge], size_t) except +
        shared_ptr[AvailableAffineBridge] execute(const vector[AdmissionInput]&, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableAffineBridge:
        @staticmethod
        shared_ptr[AvailableAffineBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableEvidenceBridge] evidence() except +
        shared_ptr[AffineResultBridge] result() except +
        shared_ptr[AvailableResultBridge] input() except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[AvailableEvaluationBridge] evaluate(const vector[string]&, uint64_t, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass AvailableEvaluationBridge:
        @staticmethod
        shared_ptr[AvailableEvaluationBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        string archive() except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        shared_ptr[AvailableAffineBridge] family() except +
        shared_ptr[AffineEvaluationBridge] result() except +
        vector[string] dependency_archives(size_t) except +


cdef class PreparedAvailableFieldEvidence:
    cdef shared_ptr[AvailableEvidenceBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableEvidenceBridge] _pin(self) except *

cdef class NativeAvailableAffineResult:
    cdef shared_ptr[AvailableAffineBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableAffineBridge] _pin(self) except *

cdef class NativeAvailableAffineEvaluation:
    cdef shared_ptr[AvailableEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AvailableEvaluationBridge] _pin(self) except *
