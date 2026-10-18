from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_admission cimport AvailableProgramBridge, AvailabilityContextBridge, AdmissionInput
from ._field_interface cimport FieldProgramBridge, InputUseAssetBridge
from ._field_evidence cimport EvidenceDeclaration, FieldEvidenceBridge, AffineResultBridge
from ._known_tensor_square cimport AvailableKnownSquareResultBridge

cdef extern from "known_product_learning_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass MomentPair:
        size_t relation, head, dependent, row0, row1
        uint64_t head_flag, dependent_flag
    cdef cppclass InitialInputLineageBridge:
        InitialInputLineageBridge(shared_ptr[InputUseAssetBridge], shared_ptr[AvailableKnownSquareResultBridge], uint64_t, uint64_t,
            const vector[vector[uint64_t]]&, const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        string payload(shared_ptr[InputUseAssetBridge], shared_ptr[AvailableKnownSquareResultBridge], uint64_t, uint64_t,
            const vector[vector[uint64_t]]&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[InitialInputLineageBridge] restore(const string&, shared_ptr[AvailableKnownSquareResultBridge],
            shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        string archive() except +
        string asset(cppbool) except +
        vector[string] dependency_archives(size_t) except +
    cdef cppclass FourUseMomentBasisBridge:
        FourUseMomentBasisBridge(shared_ptr[FieldProgramBridge], shared_ptr[AvailableKnownSquareResultBridge], uint64_t, uint64_t,
            uint64_t, const vector[uint64_t]&, const vector[MomentPair]&, const vector[size_t]&, size_t) except +
        vector[size_t] counts() except +
        string matrix(size_t, unsigned) except +
    cdef cppclass KnownProductObservation:
        uint64_t key, use, asset_origin, value_origin, interface_origin, evidence_origin
        shared_ptr[AvailableKnownSquareResultBridge] input
        shared_ptr[InitialInputLineageBridge] lineage
        shared_ptr[InputUseAssetBridge] target_asset
        string target
        vector[vector[size_t]] observed
    cdef cppclass KnownProductBatchInfo:
        size_t origin_count
        vector[string] dependencies
    cdef cppclass KnownProductFamilyBridge
    cdef cppclass KnownProductQueryOwners:
        shared_ptr[AvailableKnownSquareResultBridge] input
        shared_ptr[InitialInputLineageBridge] lineage
    cdef cppclass KnownProductEvidenceBridge:
        KnownProductEvidenceBridge(shared_ptr[AvailableProgramBridge], const EvidenceDeclaration&, uint64_t,
            shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, const vector[size_t]&) except +
        @staticmethod
        shared_ptr[KnownProductEvidenceBridge] from_basis(shared_ptr[AvailableProgramBridge], shared_ptr[FourUseMomentBasisBridge], uint64_t,
            shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, const vector[size_t]&) except +
        @staticmethod
        string basis_root_payload(shared_ptr[FourUseMomentBasisBridge], const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[KnownProductEvidenceBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, const vector[size_t]&, cppbool, const string&) except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        string archive() except +
        string parameter_schema() except +
        shared_ptr[FieldEvidenceBridge] evidence() except +
        shared_ptr[AvailableProgramBridge] program() except +
        shared_ptr[KnownProductEvidenceBridge] parent() except +
        size_t origin_count() except +
        vector[string] dependency_archives(size_t) except +
        string descriptor(const KnownProductObservation&, shared_ptr[AvailabilityContextBridge]) except +
        KnownProductBatchInfo batch_info(const vector[KnownProductObservation]&, shared_ptr[AvailabilityContextBridge], size_t) except +
        shared_ptr[KnownProductEvidenceBridge] append(const vector[KnownProductObservation]&, shared_ptr[AvailabilityContextBridge], cppbool) except +
        shared_ptr[AvailableKnownSquareResultBridge] restore_input(const string&, shared_ptr[AvailabilityContextBridge]) except +
        KnownProductQueryOwners restore_query(const string&, const string&, shared_ptr[AvailabilityContextBridge], cppbool) except +
        shared_ptr[KnownProductFamilyBridge] execute(shared_ptr[AvailableKnownSquareResultBridge], shared_ptr[InitialInputLineageBridge], shared_ptr[AvailabilityContextBridge], cppbool) except +
    cdef cppclass KnownProductFamilyBridge:
        @staticmethod
        shared_ptr[KnownProductFamilyBridge] restore(const string&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t, const vector[size_t]&, cppbool, const string&) except +
        void admit(shared_ptr[AvailabilityContextBridge]) except +
        string archive() except +
        shared_ptr[KnownProductEvidenceBridge] evidence() except +
        shared_ptr[AvailableKnownSquareResultBridge] input() except +
        shared_ptr[InitialInputLineageBridge] lineage() except +
        shared_ptr[AffineResultBridge] result() except +
        vector[string] dependency_archives(size_t) except +
    void require_four_use_grammar(shared_ptr[AvailableProgramBridge], const AdmissionInput&, shared_ptr[AvailabilityContextBridge], const vector[size_t]&, size_t) except +
    string shared_product_input(shared_ptr[AvailableKnownSquareResultBridge], const vector[size_t]&) except +

cdef class NativeAvailableInitialInputLineage:
    cdef shared_ptr[InitialInputLineageBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[InitialInputLineageBridge] _pin(self) except *

cdef class NativeFourUseMomentBasis:
    cdef shared_ptr[FourUseMomentBasisBridge] _handle
    cdef bint _loading
    cdef shared_ptr[FourUseMomentBasisBridge] _pin(self) except *

cdef class PreparedAvailableKnownProductEvidence:
    cdef shared_ptr[KnownProductEvidenceBridge] _handle
    cdef vector[size_t] _limits, _shared
    cdef size_t _maximum
    cdef bint _loading, _teacher
    cdef string _expected
    cdef shared_ptr[KnownProductEvidenceBridge] _pin(self) except *

cdef class NativeKnownProductFamily:
    cdef shared_ptr[KnownProductFamilyBridge] _handle
    cdef vector[size_t] _limits, _shared
    cdef size_t _maximum
    cdef bint _teacher
    cdef string _expected
    cdef shared_ptr[KnownProductFamilyBridge] _pin(self) except *

cdef extern from "four_use_role_application_bridge.hpp" namespace "rexgraph::python" nogil:
    vector[shared_ptr[FieldProgramBridge]] role_programs_native "rexgraph::python::four_use_role_programs"(shared_ptr[InputUseAssetBridge], const string&, const vector[size_t]&, size_t) except +
    string role_observed_tensor_native "rexgraph::python::four_use_role_observed_tensor"(const string&, const vector[size_t]&, size_t) except +
    string role_publisher_history_native "rexgraph::python::four_use_role_publisher_history"(const string&, uint64_t, uint64_t, const vector[size_t]&, size_t) except +
    cdef cppclass FourUseRoleLiftBridge:
        vector[string] input_payloads() except +
        shared_ptr[AvailableKnownSquareResultBridge] admit_input(const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge]) except +
    cdef cppclass FourUseRoleInputBridge:
        FourUseRoleInputBridge(shared_ptr[AvailableProgramBridge], shared_ptr[AvailableProgramBridge], uint64_t,
            shared_ptr[AvailabilityContextBridge], vector[size_t], size_t) except +
        @staticmethod
        string interface_payload(shared_ptr[FieldProgramBridge], size_t) except +
        vector[string] input_payloads(shared_ptr[InputUseAssetBridge], const string&, uint64_t) except +
        shared_ptr[FourUseRoleLiftBridge] lift(shared_ptr[InputUseAssetBridge], const string&, uint64_t,
            const vector[uint64_t]&, shared_ptr[AvailabilityContextBridge]) except +

cdef class PreparedFourUseRoleInput:
    cdef shared_ptr[FourUseRoleInputBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FourUseRoleInputBridge] _pin(self) except *

cdef class NativeFourUseRoleLift:
    cdef shared_ptr[FourUseRoleLiftBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[FourUseRoleLiftBridge] _pin(self) except *
