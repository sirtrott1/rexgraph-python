from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from ._field_interface cimport (
    ProgramMatrix, ProgramFieldInfo, FieldProgramBridge, AccessionResultBridge,
    InputUseAssetBridge, ProgramSelected, PreparedAccessionFieldProgram,
    NativeAccessionFieldResult, NativeInputUseAsset, _bounds, _maximum, _image,
    _matrix, _indices, _selected,
)
cdef extern from "occurrence_target_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass OccurrenceTargetDeclaration:
        uint64_t key, revision, node, frame
        vector[vector[uint64_t]] coordinates
        ProgramMatrix accession, reconstruction
    cdef cppclass OccurrenceTargetInfo:
        uint64_t key, revision, node, frame
        vector[vector[uint64_t]] coordinates
        string accession, reconstruction, selection, action, coordinate_metric, represented_metric
        size_t retained_entries
    cdef cppclass OccurrenceTargetBridge:
        OccurrenceTargetBridge(shared_ptr[FieldProgramBridge], const OccurrenceTargetDeclaration&, const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[OccurrenceTargetBridge] restore(const string&, const vector[size_t]&, size_t) except +
        shared_ptr[OccurrenceTargetBridge] rebind(shared_ptr[FieldProgramBridge]) except +
        string archive() except +
        shared_ptr[FieldProgramBridge] program() except +
        cppbool same_owner(const shared_ptr[OccurrenceTargetBridge]&) except +
        OccurrenceTargetInfo info(size_t) except +

cdef class NativeOccurrenceTargetPort:
    cdef shared_ptr[OccurrenceTargetBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[OccurrenceTargetBridge] _pin(self) except *

cdef object _wrap_port(shared_ptr[OccurrenceTargetBridge], vector[size_t], size_t)
cdef extern from "field_evidence_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass EvidenceDeclaration:
        uint64_t key, revision, node
        vector[vector[size_t]] variables
        ProgramMatrix basis
        vector[uint64_t] parameters
    cdef cppclass EvidenceIdentity:
        uint64_t key, revision, node
        size_t grade, parameters, free_parameters, observations
        cppbool has_parent
    cdef cppclass EvidenceDetails:
        vector[vector[size_t]] variables, chain_coordinates, observation_coordinates
        vector[uint64_t] parameters, free_parameters
        string basis, particular, kernel, chain, design, augmented, target, parent_shift, parent_map
        cppbool has_parent
        vector[vector[size_t]] typed_coordinates
    cdef cppclass EvidenceObservation:
        uint64_t key, target_use
        shared_ptr[AccessionResultBridge] input
        shared_ptr[InputUseAssetBridge] target_asset
        string target
        vector[vector[size_t]] observed
        shared_ptr[OccurrenceTargetBridge] port
        vector[uint64_t] observed_coordinates
    cdef cppclass EvidenceObservationInfo:
        uint64_t key, target_use
        string input, target_asset, target
        vector[vector[size_t]] observed
        shared_ptr[OccurrenceTargetBridge] port
        vector[uint64_t] observed_coordinates
    cdef cppclass AffineFieldInfo:
        string particular, directions
        size_t dimension, supported, nonzeros, free_parameters, direction_nonzeros
    cdef cppclass AffineSelected:
        ProgramSelected particular
        string directions
        vector[cppbool] identified
    cdef cppclass AffineMapInfo:
        size_t grade, occurrence
        string particular
        vector[string] directions
    cdef cppclass AffineResultBridge
    cdef cppclass EvidenceResidualBridge
    cdef cppclass EvidenceMapBridge
    cdef cppclass AffineEvaluationBridge
    cdef cppclass FieldEvidenceBridge:
        FieldEvidenceBridge(shared_ptr[FieldProgramBridge], const EvidenceDeclaration&,
            const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FieldEvidenceBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        EvidenceIdentity identity() except +
        EvidenceDetails details(size_t) except +
        cppbool has_occurrence_observations() except +
        vector[size_t] update() except +
        EvidenceObservationInfo observation(size_t, size_t) except +
        vector[string] dependency_archives(size_t) except +
        shared_ptr[FieldProgramBridge] program() except +
        shared_ptr[FieldEvidenceBridge] parent() except +
        shared_ptr[FieldEvidenceBridge] append(const vector[EvidenceObservation]&) except +
        shared_ptr[AffineResultBridge] execute(shared_ptr[AccessionResultBridge]) except +
        shared_ptr[EvidenceResidualBridge] residual(size_t) except +
        shared_ptr[EvidenceMapBridge] occurrence_map(size_t, size_t) except +
    cdef cppclass AffineResultBridge:
        @staticmethod
        shared_ptr[AffineResultBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FieldEvidenceBridge] evidence() except +
        shared_ptr[AccessionResultBridge] input() except +
        vector[ProgramFieldInfo] fields(size_t) except +
        AffineFieldInfo field(uint64_t, cppbool) except +
        cppbool identified(uint64_t, size_t) except +
        vector[AffineSelected] select(uint64_t, const vector[vector[size_t]]&, size_t) except +
        shared_ptr[AffineEvaluationBridge] evaluate(const vector[string]&) except +
    cdef cppclass AffineEvaluationBridge:
        @staticmethod
        shared_ptr[AffineEvaluationBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[AffineResultBridge] family() except +
        shared_ptr[AccessionResultBridge] result() except +
        string free_values() except +
    cdef cppclass EvidenceResidualBridge:
        AffineFieldInfo field() except +
        shared_ptr[AffineResultBridge] response() except +
        size_t observation() except +
        shared_ptr[OccurrenceTargetBridge] target_port() except +
    cdef cppclass EvidenceMapBridge:
        AffineMapInfo info(size_t) except +
        shared_ptr[FieldEvidenceBridge] evidence() except +


cdef class PreparedFieldEvidence:
    cdef shared_ptr[FieldEvidenceBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FieldEvidenceBridge] _pin(self) except *

cdef class NativeAffineFieldResult:
    cdef shared_ptr[AffineResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AffineResultBridge] _pin(self) except *

cdef class NativeAffineEvaluation:
    cdef shared_ptr[AffineEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[AffineEvaluationBridge] _pin(self) except *

cdef class NativeFieldResidual:
    cdef shared_ptr[EvidenceResidualBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[EvidenceResidualBridge] _pin(self) except *

cdef class NativeOccurrenceMapFamily:
    cdef shared_ptr[EvidenceMapBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[EvidenceMapBridge] _pin(self) except *

cdef EvidenceDeclaration _evidence_declaration(value, vector[size_t] bounds)


cdef extern from "field_family_bridge.hpp" namespace "rexgraph::python" nogil:
    cdef cppclass FamilyView:
        uint64_t frame
        ProgramMatrix accession, reconstruction
    cdef cppclass FamilyInterface:
        uint64_t key, revision, cutoff
        size_t left, right
        string common
        vector[vector[size_t]] left_cells, right_cells, left_occurrences, right_occurrences
        vector[vector[FamilyView]] left_views, right_views
    cdef cppclass FamilyDeclaration:
        uint64_t key, revision
        vector[FamilyInterface] interfaces
    cdef cppclass FamilyIdentity:
        uint64_t key, revision
        size_t members, interfaces, free_parameters, rank, retained_entries
        cppbool consistent
        vector[size_t] limits
    cdef cppclass FamilyProof:
        string particular, kernel, coordinate_origin, coordinate_kernel, residual, equations, contradiction
        vector[size_t] parameter_offsets, member_free_offsets, free_columns
        vector[vector[size_t]] equation_coordinates
        vector[unsigned int] statuses
    cdef cppclass FamilyWitness:
        size_t interface, grade, occurrence, output_dimension
        string common_input, left_input, right_input
        vector[vector[size_t]] response_shapes, equation_coordinates
    cdef cppclass FamilyCoordinates:
        string shift, map
    cdef cppclass JointResultBridge
    cdef cppclass JointEvaluationBridge
    cdef cppclass JointMapBridge
    cdef cppclass FieldFamilyBridge:
        FieldFamilyBridge(const FamilyDeclaration&, const vector[shared_ptr[FieldEvidenceBridge]]&,
                          const vector[size_t]&, size_t) except +
        @staticmethod
        shared_ptr[FieldFamilyBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        FamilyIdentity identity() except +
        FamilyProof proof(size_t) except +
        FamilyWitness witness(size_t, size_t) except +
        size_t witnesses() except +
        FamilyCoordinates coordinates(size_t) except +
        shared_ptr[FieldEvidenceBridge] member(size_t) except +
        cppbool same_owner(shared_ptr[FieldFamilyBridge]) except +
        shared_ptr[JointResultBridge] execute(size_t, shared_ptr[AccessionResultBridge]) except +
        shared_ptr[JointMapBridge] occurrence_map(size_t, size_t, size_t) except +
    cdef cppclass JointResultBridge:
        @staticmethod
        shared_ptr[JointResultBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[FieldFamilyBridge] family() except +
        shared_ptr[AffineResultBridge] original() except +
        shared_ptr[AccessionResultBridge] input() except +
        size_t member() except +
        vector[ProgramFieldInfo] fields(size_t) except +
        AffineFieldInfo field(uint64_t, cppbool) except +
        vector[AffineSelected] select(uint64_t, const vector[vector[size_t]]&, size_t) except +
        cppbool identified(uint64_t, size_t) except +
        shared_ptr[JointEvaluationBridge] evaluate(const vector[string]&) except +
    cdef cppclass JointEvaluationBridge:
        @staticmethod
        shared_ptr[JointEvaluationBridge] restore(const string&, const vector[size_t]&, size_t) except +
        string archive() except +
        shared_ptr[JointResultBridge] family() except +
        shared_ptr[AccessionResultBridge] result() except +
        string free_values() except +
    cdef cppclass JointMapBridge:
        AffineMapInfo info(size_t) except +
        shared_ptr[FieldFamilyBridge] family() except +
        size_t member() except +


cdef class PreparedFieldFamily:
    cdef shared_ptr[FieldFamilyBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef bint _loading
    cdef shared_ptr[FieldFamilyBridge] _pin(self) except *


cdef class NativeJointFieldResult:
    cdef shared_ptr[JointResultBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[JointResultBridge] _pin(self) except *


cdef class NativeJointFieldEvaluation:
    cdef shared_ptr[JointEvaluationBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[JointEvaluationBridge] _pin(self) except *


cdef class NativeJointOccurrenceMap:
    cdef shared_ptr[JointMapBridge] _handle
    cdef vector[size_t] _limits
    cdef size_t _maximum
    cdef shared_ptr[JointMapBridge] _pin(self) except *
