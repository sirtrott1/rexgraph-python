from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport ProgramMatrix, InterfacePatch, InterfaceView, InterfaceNodeCell, InputUseAssetBridge, ProgramSourceInfo
from ._field_admission cimport AvailableResultBridge, AvailabilityContextBridge, ValueOriginBridge
from ._field_evidence_admission cimport AvailableAffineBridge
from ._known_tensor_graph cimport GraphCheck

cdef extern from "task_outcome_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass TaskDeclaration:
        uint64_t key, revision, use, origin, basis
        vector[uint64_t] directions
        ProgramMatrix metric
        string target
    cdef cppclass TaskBinding:
        uint64_t key, revision, worker, machine, operation, origin
        InterfacePatch interface
        InterfaceView view
    cdef cppclass TaskCertificate:
        unsigned status
        size_t rank
        string residual, directions, feasible_shift, feasible_kernel, contradiction
    cdef cppclass TaskSink:
        int (*write)(void*, const char*, size_t) noexcept
        void* context
    cdef cppclass TaskDetails:
        uint64_t interface_key, branch, view
        vector[uint64_t] directions
        vector[InterfaceNodeCell] cells
        string target, metric, accession, reconstruction
    cdef cppclass TaskOutcomeBridge:
        TaskOutcomeBridge(const TaskDeclaration&, shared_ptr[InputUseAssetBridge], const TaskBinding&,
            shared_ptr[AvailableResultBridge], shared_ptr[AvailableAffineBridge], shared_ptr[AvailabilityContextBridge],
            const vector[size_t]&, size_t, GraphCheck) except +
        @staticmethod
        shared_ptr[TaskOutcomeBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        @staticmethod
        string goal_payload(const TaskDeclaration&, shared_ptr[InputUseAssetBridge], shared_ptr[AvailableResultBridge], shared_ptr[AvailableAffineBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string binding_payload(const TaskBinding&, shared_ptr[AvailableResultBridge], shared_ptr[AvailableAffineBridge], const vector[size_t]&, size_t) except +
        void admit(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        unsigned status(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        cppbool executed() except +
        string archive(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        size_t stream(TaskSink, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        TaskCertificate certificate(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        vector[uint64_t] identity(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[InputUseAssetBridge] asset(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableResultBridge] field(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableAffineBridge] family(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        ProgramSourceInfo source(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        TaskDetails details(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        vector[string] dependencies(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        vector[shared_ptr[ValueOriginBridge]] origins(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +

cdef class NativeTaskOutcome:
    cdef shared_ptr[TaskOutcomeBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[TaskOutcomeBridge] _pin(self) except *
