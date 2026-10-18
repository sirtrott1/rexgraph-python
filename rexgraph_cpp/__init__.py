"""Compiled exact source and field bindings for the RexGraph C++ engine."""
from ._binary import NativeBinaryGate, NativeBinaryInput, binary_build_info, fit_binary_gate
from ._relation import PreparedRelationResponse, relation_build_info
from ._sections import PreparedRelationalSections, sections_build_info
from ._guards import PreparedLexicalRoutes, PreparedLexicalContextFields
from ._stalk_fields import PreparedStalkFields, stalk_fields_build_info
from ._stalk_boundary import PreparedStalkDeclaration, PreparedStalkBoundary, stalk_boundary_build_info
from ._stalk_boundary import PreparedGradedStalkTransport, graded_transport_build_info
from ._stalk_transport import PreparedStalkTransport, PreparedStalkTransportPattern, PreparedStalkTransportPatternBank, PreparedStalkTransportPatternTraining, PreparedStalkTransportPatternScope, stalk_transport_build_info, encode_stalk_column_tensor
from ._model_storage import model_storage_build_info
from ._field_program import PreparedFieldProgram, NativeFieldProgramResult, field_program_build_info
from ._known_tensor_square import (
    PreparedAvailableKnownTensorSquare, NativeAvailableKnownTensorSquare,
    NativeAvailableKnownTensorSquareInput, NativeAvailableKnownTensorSquareResult,
    known_tensor_square_origin_payload, known_tensor_square_build_info,
)
from ._field_interface import (
    PreparedAccessionFieldProgram, NativeInputUseAsset, NativeInputAccession,
    NativeAccessionInput, NativeLiteralInput, NativeAccessionFieldResult, NativeFieldInterface,
    NativeFieldReading, NativeFieldSection, field_interface_build_info,
)
from ._field_evidence import (
    PreparedFieldEvidence, NativeAffineFieldResult, NativeAffineEvaluation,
    NativeFieldResidual, NativeOccurrenceMapFamily, field_evidence_build_info,
)
from ._field_comparison import (
    PreparedFieldComparison, NativeComparisonFieldResult, NativeComparisonPair,
    field_comparison_build_info,
)
from ._field_admission import (
    NativeValueOrigin as NativeValueOrigin, NativeAvailabilityContext as NativeAvailabilityContext, PreparedAvailableFieldProgram as PreparedAvailableFieldProgram,
    NativeAvailableFieldResult as NativeAvailableFieldResult, field_input_origin_payloads as field_input_origin_payloads, field_admission_build_info as field_admission_build_info,
)
from ._bridge import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeAlgebraicArchive,
    NativeAlgebraicArchiveGraph,
    NativeAlgebraicGradeTensor,
    NativeAlgebraicGradeTensorWindow,
    NativeAlgebraicGraphDatabase,
    NativeAlgebraicGraphRevision,
    NativeAlgebraicMetricField,
    NativeAlgebraicMetricValues,
    NativeAlgebraicQueryCache,
    NativeAlgebraicQueryResult,
    NativeAlgebraicRecord,
    NativeAlgebraicSnapshot,
    NativeAlgebraicTensorWindow,
    NativeAlgebraicValue,
    NativeField,
    NativeChainMap,
    NativeIntegerBatch,
    NativeIntegerInput,
    NativePerceptron,
    NativePerceptronRows,
    NativePhaseFoldArchive,
    NativeSource,
    NativeTrainer,
    NativeTransport,
    PreparedAlgebraicQuery,
    PreparedField,
    PreparedIntegerNetwork,
    build_info,
    fit_context_counts,
    literal_token_chart,
    literal_whitespace_chart,
    literal_whitespace_spans,
)

__all__ = ["NativeSource", "NativeField", "NativeAlgebraicArchive", "NativeAlgebraicArchiveGraph", "NativeAlgebraicValue", "PreparedField", "NativeIntegerInput", "NativeIntegerBatch",
           "PreparedIntegerNetwork", "NativeTrainer", "NativeTransport", "NativePerceptron", "NativePerceptronRows", "CancellationToken",
           "ExecutionCancelled", "ExecutionLimitError", "build_info"]

__all__ += ["NativeAlgebraicSnapshot", "NativeAlgebraicRecord", "NativeAlgebraicTensorWindow",
            "PreparedAlgebraicQuery", "NativeAlgebraicQueryResult", "NativeAlgebraicQueryCache"]

__all__ += ["NativePhaseFoldArchive"]

__all__ += ["NativeAlgebraicGraphRevision", "NativeAlgebraicGraphDatabase"]

__all__ += ["NativeAlgebraicGradeTensor", "NativeAlgebraicGradeTensorWindow"]

__all__ += ["NativeAlgebraicMetricField", "NativeAlgebraicMetricValues"]

__all__ += ["NativeBinaryGate", "NativeBinaryInput", "binary_build_info", "fit_binary_gate"]
__all__ += ["NativeChainMap"]
__all__ += ["fit_context_counts"]
__all__ += ["literal_token_chart"]
__all__ += ["literal_whitespace_chart"]
__all__ += ["literal_whitespace_spans"]
__all__ += ["PreparedRelationResponse", "relation_build_info"]
__all__ += ["PreparedRelationalSections", "sections_build_info"]
__all__ += ["PreparedLexicalRoutes", "PreparedLexicalContextFields"]
__all__ += ["PreparedStalkFields", "stalk_fields_build_info"]
__all__ += ["PreparedStalkDeclaration", "PreparedStalkBoundary", "stalk_boundary_build_info"]
__all__ += ["PreparedGradedStalkTransport", "graded_transport_build_info"]
__all__ += ["PreparedStalkTransport", "encode_stalk_column_tensor", "PreparedStalkTransportPattern", "PreparedStalkTransportPatternBank", "PreparedStalkTransportPatternTraining", "PreparedStalkTransportPatternScope", "stalk_transport_build_info"]
from ._stalk_transport import PreparedStalkFieldObjective, stalk_objective_build_info
__all__ += ["PreparedStalkFieldObjective", "stalk_objective_build_info"]
from ._stalk_transport import PreparedStalkParameterRelations, stalk_parameter_relations_build_info
__all__ += ["PreparedStalkParameterRelations", "stalk_parameter_relations_build_info"]
from ._guards import PreparedRelationGuards, PreparedDictionaryGuards, guards_build_info
__all__ += ["PreparedRelationGuards", "PreparedDictionaryGuards", "guards_build_info"]
from ._channels import PreparedRelationalChannels, channels_build_info
__all__ += ["PreparedRelationalChannels", "channels_build_info"]
__all__ += ["model_storage_build_info"]
__all__ += ["PreparedFieldProgram", "NativeFieldProgramResult", "field_program_build_info"]
__all__ += ["PreparedAvailableKnownTensorSquare", "NativeAvailableKnownTensorSquare",
            "NativeAvailableKnownTensorSquareInput", "NativeAvailableKnownTensorSquareResult",
            "known_tensor_square_origin_payload", "known_tensor_square_build_info"]
__all__ += ["PreparedAccessionFieldProgram", "NativeInputUseAsset", "NativeInputAccession",
            "NativeAccessionInput", "NativeLiteralInput", "NativeAccessionFieldResult", "NativeFieldInterface",
            "NativeFieldReading", "NativeFieldSection", "field_interface_build_info"]
__all__ += ["PreparedFieldEvidence", "NativeAffineFieldResult", "NativeAffineEvaluation",
            "NativeFieldResidual", "NativeOccurrenceMapFamily", "field_evidence_build_info"]
__all__ += ["PreparedFieldComparison", "NativeComparisonFieldResult", "NativeComparisonPair",
            "field_comparison_build_info"]


from ._field_evidence_admission import (
    PreparedAvailableFieldEvidence, NativeAvailableAffineResult, NativeAvailableAffineEvaluation,
    field_evidence_origin_payload, field_target_origin_payload, field_evidence_admission_build_info,
)
__all__ += ["PreparedAvailableFieldEvidence", "NativeAvailableAffineResult", "NativeAvailableAffineEvaluation",
            "field_evidence_origin_payload", "field_target_origin_payload", "field_evidence_admission_build_info"]


def _bind_model_storage():
    """Attach compiled storage paths to Core modules that are already loaded."""
    import sys
    storage = sys.modules[__name__ + "._model_storage"]
    bindings = {
        "rexgraph.value_codec": {"_native_pack": "pack_native_value", "_native_unpack": "unpack_native_value"},
        "rexgraph.model_integrity": {"_native_capture": "capture_content", "_native_freeze": "freeze_model_tree",
                                    "NativeContentWitness": "NativeContentWitness"},
        "rexgraph.model_state": {"_native_freeze": "freeze_model_tree"},
        "rexgraph.model_codec": {"_PayloadEncoder": "PayloadEncoder", "_PayloadDecoder": "PayloadDecoder",
                                "_StorageTraversalLimit": "StorageTraversalLimit"},
    }
    for name, attributes in bindings.items():
        module = sys.modules.get(name)
        if module is not None:
            for target, source in attributes.items():
                setattr(module, target, getattr(storage, source))


_bind_model_storage()

from ._field_local import (
    NativeLocalFieldResult, NativeLocalFieldWork, submit_local_field_solve,
    local_field_origin_payloads, field_local_build_info,
)
__all__ += ["NativeLocalFieldResult", "NativeLocalFieldWork", "submit_local_field_solve",
            "local_field_origin_payloads", "field_local_build_info"]

from ._field_extension import (
    PreparedAvailableFieldExtension, NativeAvailableExtensionFamily, NativeAvailableExtensionEvaluation,
    field_extension_origin_payload, field_extension_build_info,
)
__all__ += ["PreparedAvailableFieldExtension", "NativeAvailableExtensionFamily", "NativeAvailableExtensionEvaluation",
            "field_extension_origin_payload", "field_extension_build_info"]

from ._finite_field import (
    NativeFiniteSource, NativeFiniteSections, PreparedFiniteFieldMap,
    NativeFiniteFieldResult, NativeFiniteFieldBranch, NativeFiniteFieldEvaluation,
    finite_field_build_info,
)
__all__ += ["NativeFiniteSource", "NativeFiniteSections", "PreparedFiniteFieldMap",
            "NativeFiniteFieldResult", "NativeFiniteFieldBranch", "NativeFiniteFieldEvaluation", "finite_field_build_info"]

from ._finite_field_admission import (
    PreparedAvailableFiniteSource as PreparedAvailableFiniteSource, NativeAvailableFiniteSections as NativeAvailableFiniteSections, NativeAvailableFiniteSection as NativeAvailableFiniteSection,
    PreparedAvailableFiniteFieldMap as PreparedAvailableFiniteFieldMap, NativeAvailableFiniteFieldResult as NativeAvailableFiniteFieldResult,
    NativeAvailableFiniteFieldBranch as NativeAvailableFiniteFieldBranch, NativeAvailableFiniteFieldEvaluation as NativeAvailableFiniteFieldEvaluation,
    finite_constraint_origin_payload as finite_constraint_origin_payload, finite_interpretation_origin_payload as finite_interpretation_origin_payload,
    finite_field_admission_build_info as finite_field_admission_build_info,
)

from ._finite_operator import (
    PreparedFiniteFieldOperator, NativeFiniteSourceTransport, NativeFiniteInputSchema,
    NativeFiniteOperatorInstance, NativeFiniteOperatorResult, NativeFiniteOperatorBranch,
    NativeFiniteOperatorEvaluation, finite_operator_build_info,
)
__all__ += [
    "PreparedFiniteFieldOperator", "NativeFiniteSourceTransport", "NativeFiniteInputSchema",
    "NativeFiniteOperatorInstance", "NativeFiniteOperatorResult", "NativeFiniteOperatorBranch",
    "NativeFiniteOperatorEvaluation", "finite_operator_build_info",
]

from ._finite_operator_admission import (
    PreparedAvailableFiniteFieldOperator, NativeAvailableFiniteOperatorInstance,
    NativeAvailableFiniteOperatorResult, NativeAvailableFiniteOperatorBranch,
    NativeAvailableFiniteOperatorEvaluation, finite_operator_origin_payload,
    finite_operator_input_origin_payloads, finite_operator_admission_build_info,
)
__all__ += [
    "PreparedAvailableFiniteFieldOperator", "NativeAvailableFiniteOperatorInstance",
    "NativeAvailableFiniteOperatorResult", "NativeAvailableFiniteOperatorBranch",
    "NativeAvailableFiniteOperatorEvaluation", "finite_operator_origin_payload",
    "finite_operator_input_origin_payloads", "finite_operator_admission_build_info",
]

from ._known_product_learning import (
    NativeAvailableInitialInputLineage, NativeFourUseMomentBasis,
    PreparedAvailableKnownProductEvidence, NativeKnownProductFamily,
    require_four_use_grammar_image, initial_input_lineage_payload,
    known_product_root_payload, shared_known_product_input, known_product_learning_build_info,
    PreparedFourUseRoleInput, NativeFourUseRoleLift, four_use_role_programs,
    four_use_role_interface_payload, four_use_role_observed_tensor,
    four_use_role_publisher_history,
)
__all__ += [
    "PreparedFourUseRoleInput", "NativeFourUseRoleLift", "four_use_role_programs",
    "four_use_role_interface_payload", "four_use_role_observed_tensor",
    "four_use_role_publisher_history",
    "NativeAvailableInitialInputLineage", "NativeFourUseMomentBasis",
    "PreparedAvailableKnownProductEvidence", "NativeKnownProductFamily",
    "require_four_use_grammar_image", "initial_input_lineage_payload",
    "known_product_root_payload", "shared_known_product_input", "known_product_learning_build_info",
]

from ._field_evidence import (NativeOccurrenceTargetPort, PreparedFieldFamily, NativeJointFieldResult, NativeJointFieldEvaluation,
    NativeJointOccurrenceMap, field_family_build_info)
__all__ += ["NativeOccurrenceTargetPort", "PreparedFieldFamily", "NativeJointFieldResult", "NativeJointFieldEvaluation",
            "NativeJointOccurrenceMap", "field_family_build_info"]

from ._field_evidence_admission import field_occurrence_target_origin_payload
__all__ += ["field_occurrence_target_origin_payload"]

from ._literal_finite_field import (
    PreparedLiteralFiniteFieldMap, NativeLiteralFiniteFieldResult,
    encode_literal_use_value, literal_finite_field_build_info,
)
__all__ += ["PreparedLiteralFiniteFieldMap", "NativeLiteralFiniteFieldResult",
            "encode_literal_use_value", "literal_finite_field_build_info"]

from ._available_literal_finite_field import (
    PreparedAvailableLiteralFiniteFieldMap, NativeAvailableLiteralFiniteFieldResult,
    NativeAvailableLiteralFiniteFieldBranch, NativeAvailableLiteralFiniteFieldEvaluation,
    literal_finite_interface_origin_payload, available_literal_finite_field_build_info,
)

__all__ += ["PreparedAvailableLiteralFiniteFieldMap", "NativeAvailableLiteralFiniteFieldResult",
            "NativeAvailableLiteralFiniteFieldBranch", "NativeAvailableLiteralFiniteFieldEvaluation",
            "literal_finite_interface_origin_payload", "available_literal_finite_field_build_info"]

from ._known_tensor_graph import NativeKnownTensorGraph, NativeKnownTensorGraphInput, known_tensor_graph_build_info
from ._literal_field_family import NativeLiteralJointFieldResult, NativeLiteralJointFieldEvaluation, literal_field_family_build_info
__all__ += ["NativeKnownTensorGraph", "NativeKnownTensorGraphInput", "known_tensor_graph_build_info",
            "NativeLiteralJointFieldResult", "NativeLiteralJointFieldEvaluation", "literal_field_family_build_info"]

from ._available_literal_field_family import (
    PreparedAvailableLiteralFieldFamily, NativeAvailableLiteralJointFieldResult,
    NativeAvailableLiteralJointFieldEvaluation, NativeJointFieldPacket,
    joint_family_interface_payload, joint_family_origin_payload, joint_packet_value_origin,
    replay_available_joint_family, replay_joint_reader_input, available_literal_field_family_build_info,
)
__all__ += ["PreparedAvailableLiteralFieldFamily", "NativeAvailableLiteralJointFieldResult",
            "NativeAvailableLiteralJointFieldEvaluation", "NativeJointFieldPacket",
            "joint_family_interface_payload", "joint_family_origin_payload", "joint_packet_value_origin",
            "replay_available_joint_family", "replay_joint_reader_input", "available_literal_field_family_build_info"]
