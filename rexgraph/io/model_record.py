"""Native model records and full histories over existing storage formats."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rexgraph.model_state import ModelState, ModelTimeline
from rexgraph.source_binding import CppAlgebraicSourceReference, CppSourceReference

__all__ = ["model_record", "read_model_record", "model_history", "read_model_history",
           "save_model", "load_model"]


def _copy(source):
    from rexgraph.state import from_state, to_state
    return from_state(to_state(source))


def model_record(source, state):
    """Keep a model assertion separate from the native source it observes."""
    from rexgraph.graph import RexGraph
    if not isinstance(state, ModelState): raise TypeError("model record requires ModelState")
    state.bind(source)
    record = RexGraph.from_hypergraph([0, 1], [0])
    record.attach_metadata(1, 0, "model", state)
    if state.adapter in {"cpp_algebraic_grade", "cpp_algebraic_grade_metric"}:
        from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
        record.attach_metadata(1, 0, "model_source", algebraic_snapshot_record(source))
    elif isinstance(state.source, CppAlgebraicSourceReference):
        from rexgraph.io.algebraic_record import algebraic_record
        record.attach_metadata(1, 0, "model_source", algebraic_record(source))
    elif isinstance(state.source, CppSourceReference):
        from rexgraph.cpp_model import cpp_source_record
        record.attach_metadata(1, 0, "model_source", cpp_source_record(source, state.source))
    else:
        record.attach_metadata(1, 0, "model_source", _copy(source))
    return record


def read_model_record(record, index=0, *, _source_cache=None, algebraic_limits=None, _algebraic_cache=None, execution_context=None, _prepare=True):
    state = record.get_metadata(1, index, "model")
    source = record.get_metadata(1, index, "model_source")
    if not isinstance(state, ModelState) or source is None:
        raise TypeError("selected record does not carry a native model and its source")
    if state.adapter in {"cpp_algebraic_grade", "cpp_algebraic_grade_metric"}:
        from rexgraph.cpp_algebraic_grade_model import (
            enable_cpp_algebraic_grade_provider,
            grade_model_restore_limits,
        )
        from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record
        source = read_algebraic_snapshot_record(source, limits=grade_model_restore_limits(state, algebraic_limits),
                                               _archive_cache=_algebraic_cache)
        if state.adapter == "cpp_algebraic_grade_metric":
            from rexgraph.cpp_algebraic_grade_metric_model import (
                enable_cpp_algebraic_grade_metric_provider,
            )
            enable_cpp_algebraic_grade_metric_provider().prepare(state, source)
        else:
            enable_cpp_algebraic_grade_provider().prepare(state, source)
    elif isinstance(state.source, CppAlgebraicSourceReference):
        from rexgraph.cpp_algebraic import algebraic_restore_limits, enable_cpp_algebraic_provider
        from rexgraph.io.algebraic_record import read_algebraic_record
        source = read_algebraic_record(source, limits=algebraic_restore_limits(state, algebraic_limits),
                                       _archive_cache=_algebraic_cache)
        enable_cpp_algebraic_provider().prepare(state, source)
    elif isinstance(state.source, CppSourceReference):
        from rexgraph.cpp_model import restore_cpp_source
        from rexgraph.model_contract import invoke_model_provider, resolve_model_provider
        maximum = state.configuration["max_bytes"] if state.adapter in {"cpp_conditioned_occurrence_program", "cpp_conditioned_field_family", "cpp_conditioned_field_program", "cpp_stalk_pattern_bank", "cpp_relational_field_program", "cpp_relational_finite_field", "cpp_relational_literal_finite_field", "cpp_relational_finite_operator", "cpp_relational_known_product", "cpp_finite_search", "cpp_assignment_evidence"} else 256*1024*1024
        read_limits = {}
        if state.adapter == "cpp_stalk_pattern_bank":
            from rexgraph.cpp_stalk_pattern_model import _admission, _archive_read_limits
            limits = _admission(state.configuration["limits"], maximum)
            read_limits = _archive_read_limits(limits)
        if state.adapter in {"cpp_conditioned_occurrence_program", "cpp_conditioned_field_family", "cpp_conditioned_field_program", "cpp_relational_field_program", "cpp_relational_finite_field", "cpp_relational_literal_finite_field", "cpp_relational_finite_operator", "cpp_relational_known_product", "cpp_finite_search", "cpp_assignment_evidence"}:
            limits = state.configuration["limits"]
            read_limits = {"max_dimension": max(limits[:2]), "max_records": limits[1],
                           "max_logical_size": max(limits[:2])}
        source = restore_cpp_source(source, state.source, max_bytes=maximum, _source_cache=_source_cache, **read_limits)
        if not _prepare:
            # Structural source binding precedes operation-specific authority.
            state.bind(source)
            return state, source
        if state.adapter == "cpp_binary_gate":
            from rexgraph.cpp_binary_gate import enable_cpp_binary_gate_provider
            enable_cpp_binary_gate_provider()
        if state.adapter == "cpp_conditioned_occurrence_program":
            from rexgraph.cpp_conditioned_occurrence_program import enable_conditioned_occurrence_provider
            from rexgraph.model_codec import pack_model, unpack_model
            state = unpack_model(pack_model(state))
            enable_conditioned_occurrence_provider()
        if state.adapter == "cpp_conditioned_field_family":
            from rexgraph.cpp_conditioned_field_family import enable_conditioned_family_provider
            from rexgraph.model_codec import pack_model, unpack_model
            state = unpack_model(pack_model(state))
            enable_conditioned_family_provider()
        if state.adapter == "cpp_conditioned_field_program":
            from rexgraph.cpp_conditioned_field_program import enable_conditioned_field_provider
            from rexgraph.model_codec import pack_model, unpack_model
            state = unpack_model(pack_model(state))
            enable_conditioned_field_provider()
        if state.adapter == "cpp_relational_known_product":
            from rexgraph.cpp_relational_known_product import enable_known_product_provider
            enable_known_product_provider()
        if state.adapter == "cpp_relational_literal_finite_field":
            from rexgraph.cpp_relational_literal_finite_field import enable_literal_finite_field_provider
            from rexgraph.model_codec import pack_model, unpack_model
            state = unpack_model(pack_model(state))
            enable_literal_finite_field_provider()
        if state.adapter == "cpp_relational_finite_field":
            from rexgraph.cpp_relational_finite_field import enable_finite_field_provider
            enable_finite_field_provider()
        if state.adapter == "cpp_finite_search":
            from rexgraph.cpp_finite_search import enable_finite_search_provider
            enable_finite_search_provider()
        if state.adapter == "cpp_assignment_evidence":
            from rexgraph.cpp_assignment import enable_assignment_provider
            enable_assignment_provider()
        if state.adapter == "cpp_relational_finite_operator":
            from rexgraph.cpp_relational_finite_operator import enable_finite_operator_provider
            enable_finite_operator_provider()
        if state.adapter == "cpp_correspondence":
            from rexgraph.cpp_correspondence import enable_cpp_correspondence_provider
            enable_cpp_correspondence_provider()
        if state.adapter == "cpp_context_catalog":
            from rexgraph.cpp_context_catalog import enable_cpp_context_catalog_provider
            enable_cpp_context_catalog_provider()
        if state.adapter == "cpp_relation_response":
            from rexgraph.cpp_relation_response import enable_relation_response_provider
            enable_relation_response_provider()
        if state.adapter == "cpp_channels":
            from rexgraph.cpp_channels import enable_channel_provider
            enable_channel_provider()
        if state.adapter == "cpp_relational_sections":
            from rexgraph.cpp_relational_sections import enable_relational_sections_provider
            enable_relational_sections_provider()
        if state.adapter == "cpp_stalk_fields":
            from rexgraph.cpp_stalk_fields import enable_stalk_fields_provider
            enable_stalk_fields_provider()
        if state.adapter == "cpp_stalk_boundary":
            from rexgraph.cpp_stalk_boundary import enable_stalk_boundary_provider
            enable_stalk_boundary_provider()
        if state.adapter == "cpp_graded_stalk_transport":
            from rexgraph.cpp_graded_stalk_transport import enable_graded_stalk_transport_provider
            enable_graded_stalk_transport_provider()
        if state.adapter == "cpp_stalk_transport":
            from rexgraph.cpp_stalk_transport import enable_stalk_transport_provider
            enable_stalk_transport_provider(state.adapter_version)
        if state.adapter == "cpp_stalk_pattern_bank":
            from rexgraph.cpp_stalk_pattern_model import enable_stalk_pattern_bank_provider
            enable_stalk_pattern_bank_provider()
        if state.adapter == "cpp_relational_field_program":
            from rexgraph.cpp_relational_field_program import enable_field_program_provider, enable_accession_field_program_provider
            if state.adapter_version == "7":
                from rexgraph.cpp_relational_field_extension import enable_field_extension_provider
                enable_field_extension_provider()
            elif state.adapter_version == "6":
                from rexgraph.cpp_relational_field_evidence_admission import enable_available_field_learning_provider
                enable_available_field_learning_provider()
            elif state.adapter_version == "5":
                from rexgraph.cpp_relational_field_admission import enable_available_field_program_provider
                enable_available_field_program_provider()
            elif state.adapter_version == "4":
                from rexgraph.cpp_relational_field_comparison import enable_field_comparison_provider
                enable_field_comparison_provider()
            elif state.adapter_version == "3":
                from rexgraph.cpp_relational_field_learning import enable_field_learning_provider
                enable_field_learning_provider()
            else:
                (enable_field_program_provider if state.adapter_version == "1" else enable_accession_field_program_provider)()
        if state.adapter == "cpp_stalk_objective":
            from rexgraph.cpp_stalk_objective import enable_stalk_objective_provider
            enable_stalk_objective_provider()
        if state.adapter == "cpp_stalk_sections":
            from rexgraph.cpp_stalk_sections import enable_stalk_sections_provider
            enable_stalk_sections_provider()
        if state.adapter == "cpp_local_sections":
            from rexgraph.cpp_local_sections import enable_local_sections_provider
            enable_local_sections_provider()
        provider = resolve_model_provider(state, "prepare")
        if provider is None:
            raise ValueError("native model record requires an execution provider")
        invoke_model_provider(provider, "prepare", state, source,
                              execution_context=execution_context if "prepare" in provider.context_operations else None)
    state.bind(source)
    return state, source


def model_history(states, sources, times, *, time_axis="model_observation", time_unit="step"):
    """Store complete snapshots beside an ordinal TemporalState acceleration index."""
    from rexgraph.graph import RexGraph, TemporalRex
    from rexgraph.temporal_state import to_temporal_state
    states, sources, times = tuple(states), tuple(sources), tuple(times)
    if not states or not len(states) == len(sources) == len(times):
        raise ValueError("model history requires aligned states, sources and exact times")
    if any(not isinstance(state, ModelState) for state in states):
        raise TypeError("history requires ModelState entries")
    if any(isinstance(state.source, (CppSourceReference, CppAlgebraicSourceReference)) for state in states):
        raise NotImplementedError("native model history requires its declared C++ timeline profile")
    history = TemporalRex([])
    for i, (state, source) in enumerate(zip(states, sources, strict=True)):
        state.bind(source)
        if i and state.parent != states[i - 1].coefficient_digest:
            raise ValueError("model history must follow its declared parent chain")
        history.append_snapshot(_copy(source), at=i)
    temporal = to_temporal_state(history)
    timeline = ModelTimeline(times, tuple(v.coefficient_digest for v in states),
        tuple(v.source.state_digest for v in states), temporal.header, temporal.tensors, time_axis, time_unit)
    record = RexGraph.from_hypergraph(np.arange(len(states) + 1), np.arange(len(states)))
    for i, (state, source) in enumerate(zip(states, sources, strict=True)):
        record.attach_metadata(1, i, "model", state)
        record.attach_metadata(1, i, "model_source", _copy(source))
    record.attach_metadata(1, 0, "model_timeline", timeline)
    return record


def _index_projection(source):
    import json

    from rexgraph.state import CODEC_TENSOR, decode_tensors, to_state
    state = to_state(source)
    tensors = dict(state.tensors)
    if CODEC_TENSOR in tensors:
        decode_tensors(tensors, json.loads(tensors[CODEC_TENSOR].tobytes().decode()))
    names = ("boundary_ptr", "boundary_idx", "w_E", "signs", "B2_col_ptr", "B2_row_idx", "B2_vals", "relation_ids")
    return {k: tensors[k] for k in names if k in tensors}, {
        k: state.header[k] for k in ("nV", "nE", "nF", "directed", "g_channel", "c_channel")}


def _verify_index(timeline, snapshots):
    from rexgraph.temporal_state import from_temporal_state
    indexed = from_temporal_state(timeline.temporal_state())
    for i, (_, source) in enumerate(snapshots):
        actual, header = _index_projection(indexed.reconstruct_at(i))
        expected, wanted = _index_projection(source)
        if header != wanted or actual.keys() != expected.keys() or any(
                not np.array_equal(actual[k], expected[k]) for k in expected):
            raise ValueError("temporal index differs from its complete source snapshot")


def read_model_history(record, time=None):
    timeline = record.get_metadata(1, 0, "model_timeline")
    if not isinstance(timeline, ModelTimeline): raise TypeError("record has no model timeline")
    timeline.check_state()
    snapshots = tuple(read_model_record(record, i) for i in range(len(timeline.times)))
    if tuple(s.coefficient_digest for s, _ in snapshots) != timeline.model_digests:
        raise ValueError("model timeline and complete snapshots differ")
    if tuple(s.source.state_digest for s, _ in snapshots) != timeline.source_digests:
        raise ValueError("model timeline source identities differ")
    for i in range(1, len(snapshots)):
        if snapshots[i][0].parent != snapshots[i - 1][0].coefficient_digest:
            raise ValueError("model history parent chain differs")
    _verify_index(timeline, snapshots)
    return (timeline, snapshots) if time is None else snapshots[timeline.at(time)]


def save_model(path, record, *, algebraic_limits=None, execution_context=None, **options):
    """Use the canonical bundle or safetensors writer for a native model record."""
    path = Path(path)
    if record.get_metadata(1, 0, "model_timeline") is not None: read_model_history(record)
    else: read_model_record(record, algebraic_limits=algebraic_limits, execution_context=execution_context)
    if path.suffix == ".rcbd":
        from rexgraph.io.bundle import save_rcbd
        return save_rcbd(path, record, **options)
    if path.suffix == ".safetensors":
        from rexgraph.io.safetensors_bridge import rex_to_safetensors
        return rex_to_safetensors(record, path, **options)
    raise ValueError("model transport requires .rcbd or .safetensors")


def load_model(path, *, algebraic_limits=None, execution_context=None, **options):
    path = Path(path)
    if path.suffix == ".rcbd":
        from rexgraph.io.bundle import load_rcbd
        record = load_rcbd(path, **options)
    elif path.suffix == ".safetensors":
        from rexgraph.io.safetensors_bridge import safetensors_to_rex
        record = safetensors_to_rex(path, **options)
    else: raise ValueError("model transport requires .rcbd or .safetensors")
    if record.get_metadata(1, 0, "model_timeline") is not None: read_model_history(record)
    else: read_model_record(record, algebraic_limits=algebraic_limits, execution_context=execution_context)
    return record
