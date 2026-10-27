"""JSON values returned by the System API."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields, is_dataclass
from decimal import Decimal
from fractions import Fraction
from math import isfinite
from typing import Any

import numpy as np
from rcql.capabilities import SourcePolicy
from rexgraph.channel_operator import ChannelOperator
from rexgraph.cochain import Chain, Cochain, Field
from rexgraph.exact_array import ExactArray
from rexgraph.graph import RexGraph, TemporalRex
from rexgraph.green import GreenOperator
from rexgraph.linear_operator import RexOperator
from rexgraph.operator_bracket import GradedOperatorBracket, OperatorBracket
from rexgraph.sheaf import ExactGlueResult, ExactSectionCheck
from rexgraph.value import Absent, Approx, ExactTime, TimeRange
from rexgraph.weighted_dirac import GradedChain, WeightedDiracOperator


def _array(value: Any, max_values: int) -> dict[str, Any]:
    arr = np.asarray(value)
    out = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.size <= max_values:
        out["values"] = json_value(arr.tolist(), max_values=max_values)
    elif arr.size:
        flat = arr.ravel()
        out["sample"] = json_value(flat[:max_values].tolist(), max_values=max_values)
        if arr.dtype.kind in "fiu" and np.any(np.isfinite(arr)):
            finite = arr[np.isfinite(arr)]
            out["min"] = json_value(np.min(finite).item(), max_values=max_values)
            out["max"] = json_value(np.max(finite).item(), max_values=max_values)
    return out


def _cell_inventory(indices, max_values):
    return [{"grade": grade, "count": len(selected), "indices": list(selected[:max_values]),
             "truncated": len(selected) > max_values}
            for grade, selected in enumerate(indices)]


def json_value(value: Any, *, max_values: int = 256) -> Any:
    """Render one RCQL result as JSON, bounding array previews by max_values."""
    if isinstance(max_values, bool) or not isinstance(max_values, int) or max_values < 0:
        raise ValueError("max_values must be a nonnegative integer")
    from rexgraph.ontology import OntologyState
    if isinstance(value, OntologyState):
        info = value.inspect()
        dependencies = info.pop("dependencies")
        return {"kind": "OntologyState", **json_value(info, max_values=max_values),
                "dependency_count": len(dependencies),
                "dependencies": json_value(dependencies[:max_values], max_values=max_values),
                "dependencies_truncated": len(dependencies) > max_values}
    if value is Absent:
        return {"kind": "Absent"}
    if isinstance(value, Approx):
        return {"kind": "Approx", "value": value.value, "source": value.source}
    if isinstance(value, ExactTime):
        return {"kind": "ExactTime", "seconds": json_value(value.seconds, max_values=max_values)}
    if isinstance(value, TimeRange):
        return {"kind": "TimeRange", "start": json_value(value.start, max_values=max_values),
                "end": json_value(value.end, max_values=max_values)}
    if isinstance(value, ExactArray):
        count = min(value.size, max_values)
        preview = ExactArray(*(getattr(value, name).ravel()[:count] for name in
                               ("numerator", "denominator", "presence", "kind")),
                             tuple(entry for entry in value.bigint if entry[0] < count))
        values = preview.values()
        if count == value.size:
            values = values.reshape(value.shape)
        return {"kind": "ExactArray", "shape": list(value.shape),
                "values" if count == value.size else "sample": json_value(values.tolist(), max_values=max_values),
                "truncated": count < value.size}
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53-1:
        # Decimal(int) is exact and does not depend on Python's configurable
        # int to text digit limit. Browser clients receive text, never a rounded Number.
        return {"kind": "Integer", "decimal": str(Decimal(value))}
    if isinstance(value, float) and not isfinite(value):
        return {"nonfinite_float": str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        import hashlib
        return {"kind": "ArtifactBytes", "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(), "payload": "omitted"}
    if isinstance(value, Fraction):
        return {"numerator": json_value(value.numerator, max_values=max_values),
                "denominator": json_value(value.denominator, max_values=max_values)}
    from rcql.program import ProgramResult
    if isinstance(value, ProgramResult):
        return {"kind": "ProgramResult", "program_digest": value.program_digest,
                "output_count": len(value.values), "aliases": list(value.aliases[:max_values]),
                "values": [json_value(v, max_values=max_values) for v in value.values[:max_values]],
                "outputs_truncated": len(value.values) > max_values,
                "step_count": len(value.steps),
                "steps": [{"name": name, "output_count": len(result.values),
                           "aliases": list(result.aliases[:max_values]),
                           "execution_records": len(result.execution)}
                          for name, result in value.steps[:max_values]],
                "steps_truncated": len(value.steps) > max_values,
                "source_count": len(value.sources),
                "sources": json_value(value.sources[:max_values], max_values=max_values),
                "sources_truncated": len(value.sources) > max_values,
                "dependency_count": len(value.dependencies),
                "dependencies": [ref.as_record() for ref in value.dependencies[:max_values]],
                "dependencies_truncated": len(value.dependencies) > max_values}
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            if value.dtype.kind == "f":
                if not np.isfinite(value):
                    return {"nonfinite_float": str(value)}
                from rexgraph.exact_value import binary_fraction
                return {"kind": "BinaryFloat", "dtype": str(value.dtype),
                        "exact": json_value(binary_fraction(value), max_values=max_values)}
            if value.dtype.kind == "c":
                return {"real": json_value(value.real, max_values=max_values),
                        "imaginary": json_value(value.imag, max_values=max_values)}
            raise TypeError(f"unsupported NumPy result scalar {value.dtype}")
        return json_value(scalar, max_values=max_values)
    if isinstance(value, complex):
        return {"real": json_value(value.real, max_values=max_values),
                "imaginary": json_value(value.imag, max_values=max_values)}
    if isinstance(value, np.ndarray):
        return _array(value, max_values)
    import sys
    native = sys.modules.get("rexgraph_cpp")
    if native is not None:
        native_types = tuple(getattr(native, name) for name in
                             ("NativeAlgebraicArchive", "NativeAlgebraicArchiveGraph", "NativeAlgebraicValue",
                              "NativeAlgebraicGraphRevision", "NativeAlgebraicGraphDatabase",
                              "NativeAlgebraicGradeTensor", "NativeAlgebraicGradeTensorWindow", "NativeAlgebraicMetricValues",
                              "NativeAlgebraicSnapshot", "NativeAlgebraicRecord", "NativeAlgebraicTensorWindow",
                              "PreparedAlgebraicQuery", "NativeAlgebraicQueryResult", "NativePhaseFoldArchive") if hasattr(native, name))
        if isinstance(value, native_types):
            details = value.preview(max_values)
            if "kind" in details:
                details["value_kind"] = details.pop("kind")
            return {"kind": type(value).__name__,
                    **json_value(details, max_values=max_values)}
    from rexgraph.partition_state import RexPartition
    from rexgraph.selection import Lineage, Selection
    if isinstance(value, Selection):
        value.check_state()
        return {"kind": "Selection", "digest": value.digest, "source_state": value.source_state,
                "source_sizes": list(value.source_sizes),
                "indices": _cell_inventory(value.indices, max_values)}
    if isinstance(value, Lineage):
        return {"kind": "Lineage", "digest": value.digest, "source_state": value.source_state,
                "result_state": value.result_state, "source_sizes": list(value.source_sizes),
                "result_sizes": list(value.result_sizes), "carried_state": value.carried_state,
                "selection_digest": value.selection_digest, "policy_digest": value.policy_digest,
                "requested": _cell_inventory(value.requested, max_values),
                "cell_maps": _cell_inventory(value.cell_maps, max_values),
                "parents": list(value.parents[:max_values]), "parent_count": len(value.parents),
                "parents_truncated": len(value.parents) > max_values}
    if isinstance(value, RexPartition):
        value.check_state()
        return {"kind": "RexPartition", "rex": json_value(value.rex, max_values=max_values),
                "state": value.manifest, "digest": value.digest,
                "cell_maps": _cell_inventory(value.cell_maps, max_values),
                "lineage": json_value(value.lineage, max_values=max_values) if value.source_sizes else None}
    from rexgraph.model_state import ModelBatch, ModelInput, ModelOutput, ModelState, ModelTimeline
    if isinstance(value, (ModelState, ModelOutput, ModelBatch, ModelTimeline, ModelInput)):
        out = {"kind": type(value).__name__, "digest": value.coefficient_digest}
        if isinstance(value, ModelTimeline):
            out.update(count=len(value.times), times=json_value(value.times[:max_values], max_values=max_values),
                       model_digests=list(value.model_digests[:max_values]),
                       source_digests=list(value.source_digests[:max_values]),
                       truncated=len(value.times) > max_values, temporal_index="ordinal",
                       time_axis=value.time_axis, time_unit=value.time_unit)
            return out
        out.update(source=value.source.as_record(),
                   coordinates={"name": value.space.name, "count": len(value.space.keys),
                                "keys": list(value.space.keys[:max_values]),
                                "truncated": len(value.space.keys) > max_values},
                   dependencies=[v.as_record() for v in value.dependencies[:max_values]],
                   dependencies_truncated=len(value.dependencies) > max_values)
        if isinstance(value, ModelState):
            parameters = value.payload.get("weights", {})
            out.update(adapter=value.adapter, adapter_version=value.adapter_version,
                       arithmetic=value.arithmetic, step=value.step, parent=value.parent,
                       resumable="optimizer" in value.payload,
                       axes=[{"name": a.name, "size": len(a.keys)} for a in value.output_axes],
                       parameter_count=len(parameters),
                       parameters=[{"name": name, "shape": list(v.shape), "dtype": str(v.dtype)}
                                   for name, v in list(parameters.items())[:max_values]
                                   if isinstance(v, np.ndarray)],
                       parameters_truncated=len(parameters) > max_values)
            if value.adapter == "cpp_relation_response":
                counts = {label: len(entries) for label, entries in value.payload["fields"].items()}
                out.update(operation="green", input_grade=1, trainable=True,
                           resumable=True,
                           learned_field_entries=counts, parameter_count=sum(counts.values()),
                           relation_count=len(value.space.keys),
                           metrics="declared-unit-stalk-metrics", learned_transports=False)
                if value.adapter_version == "2":
                    out.update(role_transports=json_value(value.configuration["role_transports"], max_values=max_values))
            elif value.adapter == "cpp_assignment_evidence":
                out.update(operation="exact-native-assignment-learning", trainable=True, resumable=True,
                           parameter_storage="immutable-native-assignment-evidence",
                           retains_whole_fields=True, retains_correlated_families=True,
                           native_model_sha256=value.payload["sha256"], native_revision=value.payload["revision"],
                           assignment_status=value.payload["status"], observation_count=value.payload["observation_count"],
                           parameter_carrier=value.configuration["parameter_carrier"],
                           availability="independent-native-context-required")
            elif value.adapter == "cpp_finite_search":
                out.update(operation="native-correlated-finite-projection", trainable=False, resumable=False,
                           recipe=json_value(value.configuration["recipe"], max_values=max_values),
                           parameter_storage="admitted-source-and-complete-request-reexecution",
                           retains_whole_fields=True, retains_correlated_families=True,
                           availability="independent-native-context-required",
                           source_sha256=value.configuration["source_sha256"],
                           available_sha256=value.configuration["available_sha256"])
            elif value.adapter == "cpp_relational_finite_operator":
                out.update(operation="native-available-finite-operator",trainable=False,resumable=False,
                           operator_sha256=value.configuration["operator_sha256"],operator_bytes=len(value.payload["operator_archive"]),
                           operator_key=value.configuration["key"],revision=value.configuration["revision"],
                           relationship_status=value.configuration["status"],reference_digest=value.configuration["reference_digest"],
                           parameter_storage="immutable-native-operator-and-original-checkpoint",retains_whole_fields=True,
                           retains_correlated_families=True,availability="independent-native-context-required",
                           operations=["create","prepare","infer"],context_operations=["create","prepare","infer"])
            elif value.adapter == "cpp_relational_literal_finite_field":
                out.update(operation="native-available-literal-finite-reader", trainable=False, resumable=False,
                           map_sha256=value.configuration["map_sha256"], reader_sha256=value.configuration["reader_sha256"],
                           base_digest=value.configuration["base_digest"], map_bytes=len(value.payload["map_archive"]),
                           parameter_storage="complete-native-reader-and-target-checkpoints", retains_whole_fields=True,
                           retains_correlated_families=True, availability="independent-native-context-required",
                           operations=["create","prepare","infer"], context_operations=["create","prepare","infer"])
            elif value.adapter == "cpp_relational_finite_field":
                out.update(operation="native-available-finite-map-instance",trainable=False,resumable=False,
                           map_sha256=value.configuration["map_sha256"],map_bytes=len(value.payload["map_archive"]),
                           map_key=value.configuration["key"],revision=value.configuration["revision"],
                           relationship_status=value.configuration["status"],target_count=len(value.configuration["target_digests"]),
                           target_digests=list(value.configuration["target_digests"]),
                           parameter_storage="immutable-native-finite-map-and-target-checkpoints",retains_whole_fields=True,
                           retains_correlated_families=True,availability="independent-native-context-required",
                           operations=["create","prepare","infer"],context_operations=["create","prepare","infer"])
            elif value.adapter == "cpp_conditioned_occurrence_program":
                out.update(operation="native-conditioned-occurrence-field", trainable=True, resumable=True,
                           source_backed=True, parameter_count=8, target_coordinates=24,
                           observations=value.payload["observation_count"],
                           parameter_storage="immutable-native-typed-evidence-and-literal-proof-history",
                           retains_whole_fields=True, retains_correlated_families=True,
                           availability="independent-native-context-required",
                           operations=["create", "prepare", "infer", "train"],
                           context_operations=["create", "prepare", "infer", "train"])
            elif value.adapter == "cpp_conditioned_field_family":
                out.update(operation="native-conditioned-joint-whole-field", trainable=False, resumable=True,
                           source_backed=True, member_count=value.configuration["member_count"],
                           parameter_storage="immutable-native-members-interfaces-and-complete-literal-proof-history",
                           retains_whole_fields=True, retains_correlated_families=True,
                           availability="independent-native-context-required", operations=["create", "prepare", "infer"],
                           context_operations=["create", "prepare", "infer"])
            elif value.adapter == "cpp_conditioned_field_program":
                out.update(operation="native-conditioned-whole-field", trainable=True, resumable=True,
                           source_backed=True, parameter_count=8, observations=len(value.payload["history"]),
                           parameter_storage="immutable-native-evidence-and-complete-literal-proof-history",
                           retains_whole_fields=True, retains_correlated_families=True,
                           availability="independent-native-context-required",
                           operations=["create", "prepare", "infer", "train"],
                           context_operations=["create", "prepare", "infer", "train"])
            elif value.adapter == "cpp_relational_field_program":
                learned = value.adapter_version in {"3", "6"}
                out.update(operation="native-relational-field-program", trainable=learned, resumable=learned,
                           source_backed=value.adapter_version in {"2", "3", "4", "5", "6", "7"},
                           program_key=value.configuration["key"], revision=value.configuration["revision"],
                           program_sha256=value.configuration["program_sha256"],
                           program_bytes=len(value.configuration["available_archive"] if value.adapter_version == "5"
                                             else value.configuration["program_archive"]),
                           parameter_storage="immutable-native-program-archive", retains_whole_fields=True,
                           operations=["create", "prepare", "infer", "train"] if learned else ["create", "prepare", "infer"])
                if learned:
                    out.update(evidence_sha256=value.payload["evidence_sha256"], evidence_bytes=len(value.payload["evidence_archive"]),
                               evidence_revision=value.payload["revision"], observations=value.payload["observation_count"],
                               retains_correlated_families=True)
                if value.adapter_version == "4":
                    out.update(operation="native-fixed-whole-field-comparison", comparison_sha256=value.configuration["comparison_sha256"],
                               comparison_grade=value.configuration["grade"], comparison_channel=value.configuration["channel"],
                               comparison_node=value.configuration["node"], comparison_storage="immutable-native-forms-archive")
                if value.adapter_version == "5":
                    out.update(operation="native-available-whole-field-program", available_program_sha256=value.configuration["available_sha256"],
                               availability="independent-native-context-required", context_operations=["create", "prepare", "infer"])
                if value.adapter_version == "6":
                    out.update(operation="native-available-whole-field-learning", original_publisher_count=value.payload["origin_count"],
                               availability="independent-native-context-required", context_operations=["create", "prepare", "infer", "train"])
                if value.adapter_version == "7":
                    out.update(operation="native-faithful-source-extension",extension_sha256=value.payload["extension_sha256"],
                               extension_key=value.configuration["extension_key"],extension_revision=value.configuration["extension_revision"],
                               relationship_status=value.configuration["status"],input_source=json_value(value.configuration["source_reference"],max_values=max_values),
                               availability="independent-native-context-required",context_operations=["create","prepare","infer"],
                               retains_correlated_families=True)
            elif value.adapter == "cpp_stalk_pattern_bank":
                out["source"] = json_value(value.source.as_record(), max_values=max_values)
                out.update(operation="joint-pattern-bank", trainable=False, resumable=False,
                           learned_transports=True, retains_affine_family=True,
                           parameter_storage="native-source-archive", parameter_count=None,
                           pattern_count=len(value.space.keys), profile=value.configuration["profile"])
            elif value.adapter == "cpp_channels":
                out.update(operation="raw-relational-channels", trainable=False, input_grade=1,
                           relation_count=len(value.space.keys), channels=["T", "G", "F", "C"], field_metric="unit",
                           geometry="raw", coparticipation=value.configuration["coparticipation"],
                           trace_normalized=value.configuration["trace_normalized"])
            elif value.adapter == "cpp_local_sections":
                out.update(operation="native-local-tensor-sections", trainable=True, resumable=True,
                           learned_transports=False, local_stalk_count=len(value.space.keys),
                           stalk_dimensions=[len(s["axes"]) for s in value.configuration["stalks"]],
                           acquired_charts=len(value.payload["observations"]), retains_all_interpretations=True)
            elif value.adapter == "cpp_stalk_fields":
                out.update(operation="whole-stalk-green", input_grade=1, trainable=True, resumable=True,
                           relation_count=len(value.space.keys),
                           stalk_dimensions=list(map(len, value.configuration["upper_axes"])),
                           acquired_columns=len(value.payload["observations"]), learned_transports=False,
                           parameter_count=sum(len(o["values"]) for o in value.payload["observations"]))
            elif value.adapter == "cpp_stalk_objective":
                fit = value.payload["selection"]
                out.update(operation="conditional-stalk-field-objective", trainable=True, resumable=True,
                           evidence_rank=fit["evidence_fit"]["rank"], objective_rank=fit["rank"],
                           conditional_directions=len(fit["free_columns"]), retains_evidence_family=True)
            elif value.adapter == "cpp_stalk_transport":
                fit = value.payload["fit"]
                out.update(operation="affine-stalk-transport", trainable=True, resumable=True, learned_transports=True,
                           input_grade=1, output_grade=0, parameter_count=len(fit["particular"]),
                           identified_rank=fit["rank"], unidentified_directions=len(fit["free_columns"]),
                           acquired_fields=value.step, retains_affine_family=True, parameter_relations=True,
                           parameter_relation_query="whole-stalk-parameter-relations-query-v1",
                           context_hypotheses=True, context_hypothesis_query="whole-stalk-context-hypotheses-query-v1",
                           context_views=True, context_view_query="whole-stalk-context-views-query-v1",
                           context_diagrams=True, context_diagram_query="whole-stalk-context-diagrams-query-v1",
                           observation_groups=True, observation_group_profile="whole-stalk-transport-observation-group-v1")
                if value.adapter_version in {"3", "4"}:
                    out.update(observation_group_profile="column-stalk-transport-observation-group-v1",
                        input_storage="original-column-tensors",target_storage="encoded-original-stalk-targets-v1")
                if value.adapter_version in {"2", "4"}:
                    out.update(parameter_space=json_value(value.configuration["parameter_space"],max_values=max_values),
                               basis_rank=fit["basis_rank"],supported_map_entries=len(value.configuration["parameter_slots"]))
            elif value.adapter == "cpp_graded_stalk_transport":
                c = value.configuration; fit = value.payload["fit"]["fit"]
                out.update(operation="graded-affine-stalk-transport",trainable=True,resumable=True,learned_transports=True,
                           input_grade=c["grade"],output_grade=c["grade"]-1,strict_chain_family=True,
                           parameter_count=len(fit["particular"]),identified_rank=fit["rank"],chain_rank=fit["chain_rank"],
                           observation_rank=fit["observation_rank"],unidentified_directions=len(fit["free_columns"]),
                           acquired_fields=value.step,retains_affine_family=True)
            elif value.adapter == "cpp_stalk_boundary":
                c = value.configuration
                out.update(operation="graded-stalk-boundary", trainable=True, resumable=True,
                           input_grade=c["grade"], output_grade=c["grade"]-c["boundary_steps"],
                           boundary_steps=c["boundary_steps"], strict_chain_family=True, learned_transports=False,
                           grade_stalk_dimensions=[list(map(len,family)) for family in c["stalk_axes"]],
                           acquired_columns=len(value.payload["observations"]),
                           parameter_count=sum(len(o["values"]) for o in value.payload["observations"]))
            elif value.adapter == "cpp_stalk_sections":
                from rexgraph.cpp_stalk_sections import _states
                sections, fields = _states(value)
                out.update(operation="compatible-sections-whole-stalk-fields", trainable=True, resumable=True,
                           layers=list(sections.configuration["layers"]), joint_states=len(sections.configuration["joint_values"]),
                           acquired_columns=len(fields.payload["observations"]), interfaces=len(value.configuration["interfaces"]),
                           retains_all_interpretations=True, learned_transports=False,
                           parameter_count=sum(len(o["values"]) for o in fields.payload["observations"]))
            elif value.adapter == "cpp_relational_sections":
                counts = {label:len(entries) for label,entries in value.payload["fields"].items()}
                out.update(operation="green-and-compatible-sections", input_grade=1, trainable=True,
                           resumable=True,
                           layers=list(value.configuration["layers"]), joint_states=len(value.configuration["joint_values"]),
                           learned_compatibility_assertions=len(value.payload["compatibility_observations"]),
                           learned_field_entries=counts, parameter_count=sum(counts.values()), learned_transports=value.adapter_version == "4")
                if value.adapter_version in {"2", "3", "4"}:
                    out.update(context_guard_profile="ordered-context-boundary-v1", automatic_guarded_links=True,
                               guard_limits=list(value.configuration["guard_limits"]),
                               context_guard_conditions=list(dict.fromkeys(o["guard"]["profile"]
                                   for o in value.payload["compatibility_observations"])))
                if value.adapter_version in {"3", "4"}:
                    out.update(dictionary_links=json_value(value.configuration["dictionary_links"], max_values=max_values),
                               acquired_dictionary_observations=len(value.payload["dictionary_observations"]))
                if value.adapter_version == "4":
                    out.update(role_transports=json_value(value.configuration["field"]["role_transports"], max_values=max_values),
                               transport_candidates=len(value.configuration["transport_search"]["candidates"]),
                               transport_fit=json_value(value.payload["transport_fit"], max_values=max_values))
            elif value.adapter == "cpp_algebraic_grade":
                layouts = value.configuration["layouts"]
                out.update(operation="boundary", boundary_steps=value.configuration["boundary_steps"],
                           input_grade=layouts[0]["grade"], output_grade=layouts[-1]["grade"],
                           input_stalk_offsets=json_value(layouts[0]["stalk_offsets"], max_values=max_values),
                           output_stalk_offsets=json_value(layouts[-1]["stalk_offsets"], max_values=max_values),
                           field_degree=value.source.native.field_degree, coefficient_basis="power", trainable=False)
            elif value.adapter == "cpp_algebraic_grade_metric":
                c = value.configuration
                layout = c["output_layout"]
                out.update(operation=c["operation"], input_grade=c["layouts"][-1]["grade"],
                           output_kind=c["output_kind"], output_grade=None if layout is None else layout["grade"],
                           metric_keys=json_value(c["metric_keys"], max_values=max_values),
                           max_retained_coordinates=c["max_retained_coordinates"],
                           input_stalk_offsets=json_value(c["layouts"][-1]["stalk_offsets"], max_values=max_values),
                           output_stalk_offsets=None if layout is None else json_value(layout["stalk_offsets"], max_values=max_values),
                           field_degree=value.source.native.field_degree, coefficient_basis="power", trainable=False)
            elif value.adapter == "cpp_algebraic":
                out.update(operation=value.configuration["operation"],
                           operator_key=value.configuration["operator_key"],
                           field_degree=value.source.native.field_degree,
                           coefficient_basis="power", trainable=False,
                           input_carrier=dict(value.configuration["input_carrier"]),
                           output_kind=value.configuration["output_kind"],
                           output_carrier=json_value(value.configuration["output_carrier"], max_values=max_values))
            elif value.adapter == "cpp_metric":
                parameters = value.payload["parameters"]
                out.update(resumable=True, parameter_count=len(parameters),
                           parameters=json_value(parameters[:max_values], max_values=max_values),
                           parameters_truncated=len(parameters) > max_values,
                           objective=value.configuration["objective"],
                           parameter_history_namespace=value.payload["history_namespace"],
                           last_loss=json_value(value.payload["last_loss"], max_values=max_values))
            elif value.adapter == "cpp_perceptron":
                parameters = value.payload["parameters"]
                preview = parameters[:max_values]
                if isinstance(preview, np.ndarray):
                    preview = preview.tolist()
                out.update(resumable=True, parameter_count=len(parameters),
                           parameters=json_value(preview, max_values=max_values),
                           parameters_truncated=len(parameters) > max_values,
                           feature_count=value.configuration["feature_count"],
                           scalar_policy=value.configuration["scalar_policy"],
                           mistakes=value.payload["mistakes"])
            elif value.adapter == "cpp_integer":
                layers = value.configuration["layers"]
                parameters = tuple(v for layer in layers for v in (*layer["gains"], layer["bias"]))
                out.update(parameter_count=len(parameters),
                           parameters=json_value(parameters[:max_values], max_values=max_values),
                           parameters_truncated=len(parameters) > max_values, layer_count=len(layers),
                           scalar_policy=value.configuration["scalar_policy"],
                           channel_profile=value.configuration["channel_profile"])
            if value.adapter in {"cpp_field", "cpp_integer", "cpp_metric", "cpp_perceptron"} and "transport" in value.payload:
                proof = value.payload["transport"]
                out.update(transport={"profile": proof["profile"], "base_step": proof["base_step"],
                                      "origin_model": proof["origin_digest"]},
                           parameter_history_step=value.step-proof["base_step"])
        elif isinstance(value, ModelInput):
            out.update(arithmetic=value.arithmetic, origin_digest=value.origin_digest,
                       original_arithmetic=value.origin_arithmetic,
                       axes=[{"name": a.name, "size": len(a.keys)} for a in value.axes],
                       values=_array(value.values, max_values), original="retained in the model input")
        elif isinstance(value, ModelOutput):
            out.update(arithmetic=value.arithmetic, model_digest=value.model_digest, method=value.method,
                       grade=value.grade, variance=value.variance,
                       axes=[{"name": a.name, "size": len(a.keys)} for a in value.axes],
                       values=_array(value.values, max_values))
            if value.method in {"algebraic-grade-stalk-v1", "algebraic-grade-metric-stalk-v1"}:
                receipt = value.execution_receipt
                out.update(original_cell=receipt["cell"], source_slice=list(receipt["source_slice"]),
                           original_grade=value.grade,
                           coefficient_basis="power", origin_output=receipt["origin_output"])
                if value.method == "algebraic-grade-metric-stalk-v1":
                    out.update(metric_operation=receipt["operation"], max_retained_coordinates=receipt["max_retained_coordinates"])
            if value.execution_receipt is not None:
                out["execution_receipt"] = json_value(value.execution_receipt, max_values=max_values)
            if value.field_archive is not None:
                archive = value.field_archive
                carrier = "model_sha256" if "model_sha256" in archive else "operator_sha256" if "operator_sha256" in archive else "map_sha256" if "map_sha256" in archive else "program_sha256"
                out["field_archive"] = {"schema": archive["schema"], "bytes": len(archive["data"]),
                                        "sha256": archive["sha256"], carrier: archive[carrier],
                                        "engine_receipt": json_value(archive["engine_receipt"], max_values=max_values)}
                out.update(value_storage="immutable-native-field-archive", retains_whole_fields=True,
                           selection="explicit node and original stalk")
                if value.method == "exact-conditioned-assignment-response-v1":
                    out.update(value_storage="immutable-native-conditioned-assignment-response",
                               selection="conditioned whole fields and maps on the retained common parameter family",
                               retains_complete_alternatives=True, executed_external_worker=False,
                               availability="independent-native-context-required")
                if value.method == "exact-available-finite-projection-replay-v1":
                    out.update(value_storage="admitted-source-and-complete-request-reexecution",
                               selection="explicit image row and original source coordinates",
                               retains_complete_alternatives=True,
                               availability="independent-native-context-required")
                if value.method in {"exact-available-relational-finite-operator-family-v1", "exact-available-relational-finite-operator-evaluation-v1"}:
                    out.update(selection="explicit operator choice, target node and original target stalk",
                               retains_complete_alternatives=True,availability="independent-native-context-required",
                               actual_input=json_value(value.execution_receipt["actual_input"],max_values=max_values))
                if value.method in {"exact-available-relational-finite-field-family-v1", "exact-available-relational-finite-field-evaluation-v1"}:
                    out.update(selection="explicit finite choice, target node and original target stalk",
                               retains_complete_alternatives=True,availability="independent-native-context-required",
                               target_count=len(value.execution_receipt["target_digests"]))
                if value.method == "exact-available-relational-known-product-family-v1":
                    out.update(retains_complete_alternatives=True, availability="independent-native-context-required",
                               retains_original_source_lineage=bool(value.execution_receipt["teacher_grammar"]))
                if value.method == "exact-available-relational-local-solve-v1":
                    out.update(packet_context="complete-global-context",
                               selection="explicit solved node and original stalk",
                               retains_global_response=True,retains_interface_and_reconstruction=True)
                out.pop("values")
            if value.method == "exact-joint-stalk-pattern-bank-v1":
                receipt = value.execution_receipt
                out["source"] = json_value(value.source.as_record(), max_values=max_values)
                out.update(operation=receipt["query_declaration"]["operation"],
                           value_storage="execution-receipt", retains_affine_family=True)
                out.pop("values")
            if value.method == "native-local-tensor-sections-v1":
                receipt = value.execution_receipt
                out.update(case_count=len(value.space.keys), retains_all_interpretations=True,
                           supported=json_value(receipt["supported"], max_values=max_values),
                           assignments=json_value(receipt["assignments"], max_values=max_values))
            if value.method == "exact-graded-stalk-boundary-v1":
                trace = value.execution_receipt["trace"]
                out.update(trace_grades=list(trace["grades"]),
                           stalk_axes=json_value(trace["receipt"]["stalk_axes"], max_values=max_values),
                           intermediate_channels=json_value(trace["intermediate_channels"], max_values=max_values))
            if value.method == "compatible-sections-whole-stalk-fields-v1":
                from rexgraph.cpp_stalk_sections import _states
                from rexgraph.model_codec import unpack_model
                receipt = value.execution_receipt
                _, fields = _states(unpack_model(receipt["model_checkpoint"]))
                out.update(case_count=len(value.space.keys), retains_all_interpretations=True,
                           stalk_axes=json_value(fields.configuration["upper_axes"], max_values=max_values),
                           assignments=json_value(receipt["assignments"], max_values=max_values),
                           branch_assignments=json_value(receipt["branch_assignments"], max_values=max_values),
                           interface_readings=json_value(receipt["interface_readings"], max_values=max_values))
        else:
            stored = (isinstance(value.inputs, Mapping)
                      and isinstance(value.inputs.get("profile"), str)
                      and value.inputs.get("profile") == "stored-native-training-samples-v1"
                      and isinstance(value.inputs.get("samples"), tuple))
            if isinstance(value.targets,Mapping):
                if value.targets["profile"] == "rexgraph.relational-field-training-targets.v1":
                    observations = value.targets["observations"]
                    targets = dict(storage=value.targets["profile"], samples=len(observations),
                        node=value.targets["node"], grade=value.targets["grade"],
                        observed_stalks=sum(len(o["observed"]) for o in observations),
                        native_target_bytes=sum(len(o["target"]) for o in observations),
                        input_fields=sum(len(o["inputs"]) for o in observations),
                        retains_complete_measurements=True)
                else:
                    targets = dict(storage=value.targets["profile"],samples=value.targets["samples"],
                        observed_stalks=len(value.targets["stalks"]),
                        tensors=sum(len(fields) for _,fields in value.targets["stalks"]))
            else:
                targets = ({"shape": list(value.targets.shape), "dtype": str(value.targets.dtype)} if stored else
                           _array(value.targets, max_values))
            out.update(targets=targets, observed=_array(value.observed, max_values),
                       observed_count=int(value.observed.sum()), inputs="query explicitly")
            if stored:
                out.update(training_profile=value.inputs["profile"],
                           sample_count=len(value.inputs["samples"]), target_storage="native source fields")
        return out
    from rexgraph.chain_map import SymmetryGroup
    if isinstance(value, SymmetryGroup):
        value.check_state()
        return {"kind": "SymmetryGroup", "metric": "euclidean", "product_order": "rightmost-first",
                "generator_count": value.generator_count, "word": list(value.word[:max_values]),
                "word_length": len(value.word), "sizes": list(value.sizes[:max_values]),
                "grades": len(value.sizes), "map": "lazy",
                "truncated": len(value.word) > max_values or len(value.sizes) > max_values}
    from rexgraph.void_state import VoidState
    if isinstance(value, VoidState):
        value.check_state()
        return {"kind": "VoidState", "source_state": value.source_state, "shape": list(value.shape),
                "n_voids": value.n_voids, "n_potential": value.n_potential, "strain": value.strain,
                "region": list(value.region[:max_values]), "region_truncated": len(value.region) > max_values,
                "potential": json_value(value.potential[:max_values], max_values=max_values),
                "potential_truncated": value.n_potential > max_values,
                "void_indices": list(value.void_indices[:max_values]),
                "columns": json_value(value.columns[:max_values], max_values=max_values),
                "columns_truncated": value.n_voids > max_values, "homology": "query explicitly"}
    from rexgraph.relative_quotient import RelativeQuotient
    if isinstance(value, RelativeQuotient):
        value.check_state()
        return {"kind": "RelativeQuotient", "sizes": list(value.sizes),
                "source_boundary_digest": value.source_boundary_digest,
                "coefficient_digest": value.coefficient_digest,
                "residuals": json_value(value.residuals, max_values=max_values),
                "boundaries": [{"shape": [value.sizes[k], value.sizes[k+1]], "nnz": len(entries),
                    "entries": json_value(entries[:max_values], max_values=max_values),
                    "entries_truncated": len(entries) > max_values}
                    for k, entries in enumerate(value.boundaries)],
                "coordinates": [{"kept_count": len(kept), "removed_count": len(removed),
                    "kept": list(kept[:max_values]), "removed": list(removed[:max_values]),
                    "truncated": max(len(kept), len(removed)) > max_values}
                    for kept, removed in zip(value.cell_maps, value.removed_cells, strict=True)],
                "homology": "query readings explicitly"}
    from rexgraph.boundary_difference import BoundaryDifference
    if isinstance(value, BoundaryDifference):
        value.check_state()
        return {"kind": "BoundaryDifference", "shape": list(value.shape), "nnz": value.nnz,
                "matching": value.matching, "source_digests": list(value.source_digests),
                "entries": json_value(value.entries[:max_values], max_values=max_values),
                "entries_truncated": value.nnz > max_values}
    if isinstance(value, RexGraph):
        return {"kind": "RexGraph", "nV": value.nV, "nE": value.nE, "nF": value.nF,
                "g_channel": value.g_channel, "c_channel": value.c_channel}
    if isinstance(value, TemporalRex):
        return {"kind": "TemporalRex", "T": value.T}
    if isinstance(value, GradedChain):
        return {"kind": "GradedChain", "sizes": list(value.sizes),
                "components": [json_value(c, max_values=max_values) for c in value.components]}
    if isinstance(value, WeightedDiracOperator):
        return {"kind": "GradedOperator", "name": value.name, "shape": list(value.shape),
                "sizes": list(value.sizes), "exact_action": value.exact,
                "metric_self_adjoint": value.metric_self_adjoint,
                "metric_skew_adjoint": value.metric_skew_adjoint}
    if isinstance(value, (OperatorBracket, GradedOperatorBracket)):
        graded = isinstance(value, GradedOperatorBracket)
        return {"kind": "GradedOperatorBracket" if graded else "OperatorBracket",
                "name": value.name, "shape": list(value.shape),
                "spaces": {"sizes": list(value.sizes), "variance": "chain"} if graded else {
                    "domain_grade": value.domain_grade, "codomain_grade": value.codomain_grade,
                    "variance": value.variance},
                "operands": [json_value(op, max_values=max_values) for op in (value.left, value.right)],
                "metric_self_adjoint": value.metric_self_adjoint,
                "metric_skew_adjoint": value.metric_skew_adjoint,
                "euclidean_skew_adjoint": None if graded else value.euclidean_skew_adjoint,
                "exact_action": value.exact if graded else value.exact_matvec is not None,
                "psd": False}
    if isinstance(value, (Chain, Cochain)):
        return {
            "kind": "Chain" if isinstance(value, Chain) else "Cochain",
            "grade": value.grade,
            "cells": value.n_cells,
            "cell_keys": json_value(value.cell_keys, max_values=max_values),
            "values": _array(value.values, max_values),
        }
    if isinstance(value, Field):
        return {
            "kind": "Field",
            "field": value.kind,
            "grade": value.grade,
            "values": _array(value.values, max_values),
            "operator": json_value(value.operator, max_values=max_values),
        }
    if isinstance(value, RexOperator):
        out = {
            "kind": "RexOperator",
            "name": value.name,
            "shape": list(value.shape),
            "domain_grade": value.domain_grade,
            "codomain_grade": value.codomain_grade,
            "symmetric": value.symmetric,
            "psd": value.psd,
            "arithmetic": value.arithmetic,
        }
        if isinstance(value, ChannelOperator):
            out.update(channel=value.channel, g_channel=value.g_channel, c_channel=value.c_channel,
                       exact_action=value.exact_matvec is not None,
                       exact_transpose=value.exact_transpose_matvec is not None,
                       exact_diagonal=True, trace_normalized=False, frustration_reference="raw-G")
        from rexgraph.rational_operator import ResolventGroup
        if isinstance(value, ResolventGroup):
            value.check_state()
            out.update(kind="ResolventGroup", word=list(value.word[:max_values]),
                       scales=json_value(value.scales[:max_values], max_values=max_values),
                       word_length=len(value.word), generators=len(value.operators),
                       truncated=len(value.word) > max_values or len(value.scales) > max_values,
                       exact_action=True, product_order="rightmost-first")
        return out
    if isinstance(value, GreenOperator):
        return {
            "kind": "GreenOperator",
            "green": value.kind,
            "operator": json_value(value.operator, max_values=max_values),
            "parameters": json_value(value.parameters, max_values=max_values),
            "solve_form": "positive-diagonal-metric" if value.metric is not None else "euclidean",
            "metric_digest": None if value.metric is None else value.metric.coefficient_digest,
            "action_variance": value.operator.variance,
        }
    if isinstance(value, Mapping):
        return {str(k): json_value(v, max_values=max_values) for k, v in value.items()}
    if isinstance(value, SourcePolicy):
        return {"permissions": sorted(value.permissions),
                "record_fields": None if value.record_fields is None else sorted(value.record_fields),
                "digest": value.digest}
    if isinstance(value, (tuple, list)):
        return [json_value(v, max_values=max_values) for v in value]
    if is_dataclass(value):
        # Do not deep copy a dataclass's live source graph just to render it.
        out = {field.name: json_value(getattr(value, field.name), max_values=max_values)
               for field in dataclass_fields(value) if not field.name.startswith("_") and field.name != "source"}
        if isinstance(value, ExactSectionCheck):
            out.update(kind="ExactSectionCheck", compatible=value.compatible,
                       diagnostic="anchor-incidence-residuals")
        elif isinstance(value, ExactGlueResult):
            out.update(kind="ExactGlueResult", ratio=json_value(value.ratio),
                       obstruction_count=value.obstruction_count,
                       agreement_components=value.h0, diagnostic="all-pair-incidence-residuals")
        return out
    if hasattr(value, "_asdict"):
        return json_value(value._asdict(), max_values=max_values)
    # Opaque handles are not an invitation to expose their repr, paths or state.
    return {"python_type": type(value).__name__}
