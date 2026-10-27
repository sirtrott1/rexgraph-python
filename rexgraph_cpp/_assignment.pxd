from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._finite_field cimport FiniteSourceBridge
from ._field_interface cimport ProgramMatrix
from ._field_evidence cimport AffineFieldInfo
from ._field_admission cimport AvailabilityContextBridge, AvailableResultBridge, ValueOriginBridge
from ._field_evidence_admission cimport AvailableEvidenceBridge, AvailableAffineBridge
from ._task_outcome cimport TaskBinding, TaskCertificate, TaskOutcomeBridge, TaskSink
from ._known_tensor_graph cimport GraphCheck

cdef extern from "assignment_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass AssignmentDeclaration:
        uint64_t key, revision, origin, parameter_carrier
        vector[uint64_t] parameters
    cdef cppclass AssignmentCandidate:
        uint64_t key
        vector[uint64_t] assignment
        TaskBinding binding
        shared_ptr[AvailableEvidenceBridge] evidence
        vector[string] shift
        ProgramMatrix injection
    cdef cppclass AssignmentInput:
        uint64_t candidate
        shared_ptr[AvailableResultBridge] input
    cdef cppclass AssignmentObservation:
        uint64_t key, origin
        TaskBinding binding
        shared_ptr[AvailableResultBridge] measured
        vector[AssignmentInput] inputs
    cdef cppclass AssignmentCoordinates:
        string shift, map
        size_t original_dimension, free_dimension
    cdef cppclass AssignmentSummary:
        AssignmentDeclaration declaration
        uint64_t revision
        size_t candidates, observations, remainder, retained_entries, identified_candidate
        unsigned status
        cppbool identified, has_parent
    cdef cppclass AssignmentBranchInfo:
        cppbool rejected
        size_t solves
        AssignmentCoordinates coordinates
    cdef cppclass AssignmentMapInfo:
        string particular
        vector[string] directions
    cdef cppclass AssignmentBatchInfo:
        vector[string] dependencies
        size_t origin_count
    cdef cppclass AssignmentModelBridge
    cdef cppclass AssignmentResponseBridge
    cdef cppclass AssignmentSolveBridge
    cdef cppclass AssignmentModelBridge:
        AssignmentModelBridge(const AssignmentDeclaration&, shared_ptr[FiniteSourceBridge], const vector[AssignmentCandidate]&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        @staticmethod
        shared_ptr[AssignmentModelBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        @staticmethod
        string root_payload(const AssignmentDeclaration&, shared_ptr[FiniteSourceBridge], const vector[AssignmentCandidate]&, const vector[size_t]&, size_t) except +
        @staticmethod
        string measurement_payload(const AssignmentObservation&, const vector[size_t]&, size_t) except +
        @staticmethod
        vector[shared_ptr[ValueOriginBridge]] input_origins(shared_ptr[AvailableResultBridge], shared_ptr[AvailabilityContextBridge], size_t, size_t, GraphCheck) except +
        void admit(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        cppbool same_owner(const AssignmentModelBridge&) except +
        AssignmentSummary summary(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        shared_ptr[AssignmentModelBridge] parent(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        string archive(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        size_t stream(TaskSink, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[FiniteSourceBridge] finite_source(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        AssignmentCandidate candidate(size_t, shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        AssignmentBranchInfo branch(size_t, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        AssignmentObservation observation(size_t, shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        vector[vector[uint64_t]] remainder(size_t, size_t, shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        vector[shared_ptr[ValueOriginBridge]] origins(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        vector[string] dependencies(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        AssignmentBatchInfo observation_dependencies(const vector[AssignmentObservation]&, shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        shared_ptr[AssignmentModelBridge] append(const vector[AssignmentObservation]&, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableResultBridge] restore_input(size_t, const string&, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AssignmentSolveBridge] solve(size_t, size_t, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AssignmentResponseBridge] execute(size_t, shared_ptr[AvailableResultBridge], shared_ptr[AvailabilityContextBridge], GraphCheck) except +
    cdef cppclass AssignmentSolveBridge:
        cppbool same_owner(const AssignmentSolveBridge&) except +
        uint64_t observation(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        TaskCertificate certificate(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        AssignmentCoordinates coordinates(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableAffineBridge] original(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
    cdef cppclass AssignmentResponseBridge:
        @staticmethod
        shared_ptr[AssignmentResponseBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        void admit(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AssignmentModelBridge] model(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        size_t candidate() except +
        vector[shared_ptr[ValueOriginBridge]] origins(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        vector[string] dependencies(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        AssignmentCoordinates coordinates(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableAffineBridge] original(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        AffineFieldInfo field(uint64_t, cppbool, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        AssignmentMapInfo map(size_t, size_t, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        TaskCertificate postcondition(shared_ptr[TaskOutcomeBridge], shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[TaskOutcomeBridge] restore_goal(const string&, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        string archive(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        size_t stream(TaskSink, shared_ptr[AvailabilityContextBridge], GraphCheck) except +

cdef class NativeAssignmentEvidence:
    cdef shared_ptr[AssignmentModelBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AssignmentModelBridge] _pin(self) except *
cdef class NativeAssignmentSolve:
    cdef shared_ptr[AssignmentSolveBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AssignmentSolveBridge] _pin(self) except *
cdef class NativeAssignmentResponse:
    cdef shared_ptr[AssignmentResponseBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AssignmentResponseBridge] _pin(self) except *
