from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._known_tensor_graph cimport GraphCheck
from ._literal_field_family cimport LiteralFamilyInput, LiteralJointResultBridge, LiteralJointEvaluationBridge
from ._field_evidence cimport FieldFamilyBridge
from ._field_interface cimport FieldProgramBridge
from ._field_evidence_admission cimport AvailableEvidenceBridge
from ._field_admission cimport ValueOriginBridge, AvailabilityContextBridge

cdef extern from "available_literal_field_family_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass JointReaderInput:
        LiteralFamilyInput input
        vector[uint64_t] origins
    cdef cppclass JointObservationReaders:
        uint64_t observation
        vector[JointReaderInput] inputs
    cdef cppclass JointMemberOrigins:
        vector[uint64_t] program
        uint64_t root
        vector[vector[uint64_t]] targets
    cdef cppclass JointSink:
        int (*write)(void*, const char*, size_t)
        void* context
    cdef cppclass AvailableJointPacketBridge:
        size_t size() except +
        string bytes(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        size_t stream(JointSink, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[ValueOriginBridge] publish(uint64_t, const string&, uint64_t, unsigned int,
            shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, const vector[size_t]&, GraphCheck) except +
    cdef cppclass AvailableJointResultBridge
    cdef cppclass AvailableJointEvaluationBridge
    cdef cppclass AvailableJointFamilyBridge:
        AvailableJointFamilyBridge(shared_ptr[FieldFamilyBridge], const vector[shared_ptr[AvailableEvidenceBridge]]&,
            const vector[vector[JointObservationReaders]]&, const vector[uint64_t]&, uint64_t,
            shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        @staticmethod
        shared_ptr[AvailableJointFamilyBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        void admit(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        cppbool same_owner(const AvailableJointFamilyBridge&) except +
        shared_ptr[FieldFamilyBridge] family(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        vector[shared_ptr[AvailableEvidenceBridge]] members(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        vector[vector[JointObservationReaders]] training_readers(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        vector[uint64_t] origin_keys(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        vector[shared_ptr[ValueOriginBridge]] obligations(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        shared_ptr[AvailableJointPacketBridge] prepare(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableJointPacketBridge] setting(const vector[string]&, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        @staticmethod
        shared_ptr[AvailableJointPacketBridge] interface_payload(shared_ptr[FieldFamilyBridge], size_t, size_t, GraphCheck) except +
        @staticmethod
        shared_ptr[AvailableJointPacketBridge] model_payload(shared_ptr[FieldFamilyBridge],
            const vector[shared_ptr[AvailableEvidenceBridge]]&, const vector[vector[JointObservationReaders]]&,
            const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        shared_ptr[AvailableJointResultBridge] execute(size_t, const vector[JointReaderInput]&, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
    cdef cppclass AvailableJointResultBridge:
        @staticmethod
        shared_ptr[AvailableJointResultBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        void admit(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableJointFamilyBridge] model(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[LiteralJointResultBridge] raw(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        vector[JointReaderInput] inputs(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        vector[shared_ptr[ValueOriginBridge]] obligations(shared_ptr[AvailabilityContextBridge], size_t, GraphCheck) except +
        shared_ptr[AvailableJointPacketBridge] prepare(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableJointPacketBridge] completion(const vector[string]&, uint64_t, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableJointEvaluationBridge] evaluate(const vector[string]&, uint64_t, uint64_t, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
    cdef cppclass AvailableJointEvaluationBridge:
        @staticmethod
        shared_ptr[AvailableJointEvaluationBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +
        void admit(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableJointResultBridge] parent(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[LiteralJointEvaluationBridge] raw(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[ValueOriginBridge] origin(unsigned int, shared_ptr[AvailabilityContextBridge], GraphCheck) except +
        shared_ptr[AvailableJointPacketBridge] prepare(shared_ptr[AvailabilityContextBridge], GraphCheck) except +
    cdef cppclass JointAvailableReplay:
        shared_ptr[FieldFamilyBridge] family
        vector[shared_ptr[AvailableEvidenceBridge]] members
    LiteralFamilyInput _replay_joint_reader_input "rexgraph::python::replay_joint_reader_input"(const LiteralFamilyInput&, shared_ptr[FieldProgramBridge],
        const vector[size_t]&, size_t, GraphCheck) except +
    JointAvailableReplay rebuild_available_field_family(shared_ptr[FieldFamilyBridge],
        const vector[JointMemberOrigins]&, const vector[vector[JointObservationReaders]]&,
        shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, GraphCheck) except +

cdef class _Sink:
    cdef object write, error

cdef class NativeJointFieldPacket:
    cdef shared_ptr[AvailableJointPacketBridge] _handle
    cdef shared_ptr[AvailableJointPacketBridge] _pin(self) except *

cdef class PreparedAvailableLiteralFieldFamily:
    cdef shared_ptr[AvailableJointFamilyBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableJointFamilyBridge] _pin(self) except *

cdef class NativeAvailableLiteralJointFieldResult:
    cdef shared_ptr[AvailableJointResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableJointResultBridge] _pin(self) except *

cdef class NativeAvailableLiteralJointFieldEvaluation:
    cdef shared_ptr[AvailableJointEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[AvailableJointEvaluationBridge] _pin(self) except *
