"""Local model queries over selected native data and immutable checkpoints."""
from __future__ import annotations

from .binding import Binding
from .capabilities import SourcePolicy
from .execution_trace import current_binding, current_policy, current_query_resources, record_method


def _algebraic_cache():
    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
    resources = current_query_resources()
    if resources is None:
        return None
    cache = resources.get("algebraic_model_sources")
    if cache is None:
        cache = resources.setdefault("algebraic_model_sources", _AlgebraicRestoreCache())
    return cache


def _read_model_record(source, index=0, *, execution_context=None):
    from rexgraph.io.model_record import read_model_record
    from rexgraph.model_state import ModelState
    from rexgraph.source_binding import CppSourceReference
    resources = current_query_resources()
    state = source.get_metadata(1, index, "model")
    cache = None
    if resources is not None and isinstance(state, ModelState) and isinstance(state.source, CppSourceReference):
        from rexgraph.cpp_model import _SourceRestoreCache
        cache = resources.get("native_model_sources")
        if cache is None:
            cache = resources.setdefault("native_model_sources", _SourceRestoreCache())
    return read_model_record(source, index, _source_cache=cache, _algebraic_cache=_algebraic_cache(),
                             execution_context=execution_context)


def source_for(source, state=None, *, algebraic_snapshot=False, execution_context=None):
    from rexgraph.io.model_record import read_model_history
    from rexgraph.model_state import ModelState
    first = source.get_metadata(1, 0, "model") if source.nE else None
    if isinstance(first, ModelState):
        if source.get_metadata(1, 0, "model_timeline") is not None:
            _, snapshots = read_model_history(source)
            if state is None: return snapshots[-1][1]
            for saved, data in snapshots:
                if saved.source.matches(state.source):
                    state.bind(data)
                    return data
            raise ValueError("model source is not present in the selected history")
        _, data = _read_model_record(source, execution_context=execution_context)
    else:
        if source.nE and (algebraic_snapshot or state is not None and state.adapter in {"cpp_algebraic_grade", "cpp_algebraic_grade_metric"}):
            from rexgraph.cpp_algebraic_grade_model import grade_model_restore_limits
            from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record
            limits = None if state is None else grade_model_restore_limits(state)
            data = read_algebraic_snapshot_record(source, limits=limits, _archive_cache=_algebraic_cache())
        elif source.nE and source.get_metadata(1, 0, "algebraic_archive") is not None:
            from rexgraph.cpp_algebraic import algebraic_restore_limits
            from rexgraph.io.algebraic_record import read_algebraic_record
            limits = None if state is None else algebraic_restore_limits(state)
            data = read_algebraic_record(source, limits=limits, _archive_cache=_algebraic_cache())
        else:
            data = source
    if state is not None: state.bind(data)
    return data


def authorize(source, state, binding=None, *, execution_context=None):
    data = source_for(source, state, execution_context=execution_context)
    if binding is not None:
        binding.source.require("read")
        if data is source:
            if state.source.record_id is not None and state.source.record_id != binding.ref.record_id:
                raise ValueError("model source record identity differs")
            if state.source.version is not None and state.source.version != binding.ref.record_version:
                raise ValueError("model source version differs")
    return data


def _reference(source):
    from rexgraph.tensor_field import FieldSource
    binding = current_binding()
    if binding is None: return FieldSource(source)
    return FieldSource(source, binding.ref.record_id, binding.ref.record_version)


def model_init(source, adapter="native_field", configuration=None, axes=None, optimizer=None, seed=0):
    from rexgraph.model_runtime import native_model
    config = dict(configuration or {})
    if adapter in {"cpp_conditioned_occurrence_program", "cpp_conditioned_field_family", "cpp_conditioned_field_program", "cpp_field", "cpp_integer", "cpp_metric", "cpp_perceptron", "cpp_binary_gate", "cpp_correspondence", "cpp_context_catalog", "cpp_relation_response", "cpp_channels", "cpp_relational_sections", "cpp_local_sections", "cpp_stalk_fields", "cpp_stalk_sections", "cpp_stalk_boundary", "cpp_graded_stalk_transport", "cpp_stalk_transport", "cpp_stalk_pattern_bank", "cpp_relational_field_program", "cpp_relational_finite_field", "cpp_relational_literal_finite_field", "cpp_relational_finite_operator", "cpp_relational_known_product", "cpp_stalk_objective", "cpp_algebraic", "cpp_algebraic_grade", "cpp_algebraic_grade_metric"}:
        if optimizer is not None or axes is not None or seed != 0:
            raise ValueError("native profiles declare no neural optimizer, feature axes or random seed")
        data = source_for(source, algebraic_snapshot=adapter in {"cpp_algebraic_grade", "cpp_algebraic_grade_metric"},
                          execution_context=config.get("execution_context"))
        binding = current_binding()
        if adapter == "cpp_algebraic_grade_metric":
            from rexgraph.cpp_algebraic_grade_metric_model import (
                enable_cpp_algebraic_grade_metric_provider as enable,
            )
            from rexgraph.source_binding import CppAlgebraicSourceReference as Reference
        elif adapter == "cpp_algebraic_grade":
            from rexgraph.cpp_algebraic_grade_model import (
                enable_cpp_algebraic_grade_provider as enable,
            )
            from rexgraph.source_binding import CppAlgebraicSourceReference as Reference
        elif adapter == "cpp_conditioned_occurrence_program":
            from rexgraph.cpp_conditioned_occurrence_program import enable_conditioned_occurrence_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_conditioned_field_family":
            from rexgraph.cpp_conditioned_field_family import enable_conditioned_family_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_conditioned_field_program":
            from rexgraph.cpp_conditioned_field_program import enable_conditioned_field_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_binary_gate":
            from rexgraph.cpp_binary_gate import enable_cpp_binary_gate_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_correspondence":
            from rexgraph.cpp_correspondence import enable_cpp_correspondence_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_context_catalog":
            from rexgraph.cpp_context_catalog import enable_cpp_context_catalog_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_relation_response":
            from rexgraph.cpp_relation_response import enable_relation_response_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_channels":
            from rexgraph.cpp_channels import enable_channel_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_relational_sections":
            from rexgraph.cpp_relational_sections import enable_relational_sections_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_local_sections":
            from rexgraph.cpp_local_sections import enable_local_sections_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_stalk_fields":
            from rexgraph.cpp_stalk_fields import enable_stalk_fields_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_stalk_boundary":
            from rexgraph.cpp_stalk_boundary import enable_stalk_boundary_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_graded_stalk_transport":
            from rexgraph.cpp_graded_stalk_transport import enable_graded_stalk_transport_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_stalk_objective":
            from rexgraph.cpp_stalk_objective import enable_stalk_objective_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_stalk_transport":
            from rexgraph.cpp_stalk_transport import enable_stalk_transport_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_stalk_pattern_bank":
            from rexgraph.cpp_stalk_pattern_model import enable_stalk_pattern_bank_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_relational_literal_finite_field":
            from rexgraph.cpp_relational_literal_finite_field import enable_literal_finite_field_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_relational_finite_field":
            from rexgraph.cpp_relational_finite_field import enable_finite_field_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_relational_known_product":
            from rexgraph.cpp_relational_known_product import enable_known_product_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_relational_finite_operator":
            from rexgraph.cpp_relational_finite_operator import enable_finite_operator_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_relational_field_program":
            from rexgraph.cpp_relational_field_program import enable_field_program_provider, enable_accession_field_program_provider
            source_backed = config.pop("source_backed", False)
            learning = config.pop("learning", False)
            comparison_fields = config.pop("comparison_fields", False)
            available_fields = config.pop("available_fields", False)
            if any(type(v) is not bool for v in (source_backed, learning, comparison_fields, available_fields)):
                raise TypeError("source_backed, learning, comparison_fields and available_fields require explicit booleans")
            if available_fields:
                if not source_backed or comparison_fields:
                    raise ValueError("available programs require their explicit source-backed profile")
                if learning:
                    from rexgraph.cpp_relational_field_evidence_admission import enable_available_field_learning_provider as enable
                else:
                    from rexgraph.cpp_relational_field_admission import enable_available_field_program_provider as enable
            elif comparison_fields:
                if learning or not source_backed: raise ValueError("fixed comparisons require their explicit source-backed profile")
                from rexgraph.cpp_relational_field_comparison import enable_field_comparison_provider as enable
            elif learning:
                if not source_backed: raise ValueError("whole-field learning requires actual source-backed inputs")
                from rexgraph.cpp_relational_field_learning import enable_field_learning_provider as enable
            else:
                enable = enable_accession_field_program_provider if source_backed else enable_field_program_provider
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_stalk_sections":
            from rexgraph.cpp_stalk_sections import enable_stalk_sections_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        elif adapter == "cpp_algebraic":
            from rexgraph.cpp_algebraic import enable_cpp_algebraic_provider as enable
            from rexgraph.source_binding import CppAlgebraicSourceReference as Reference
        else:
            from rexgraph.cpp_model import enable_cpp_provider as enable
            from rexgraph.source_binding import CppSourceReference as Reference
        owner = data.archive_owner if adapter in {"cpp_algebraic_grade", "cpp_algebraic_grade_metric"} else data
        reference = Reference(owner) if binding is None else Reference(
            owner, binding.ref.record_id, binding.ref.record_version)
        if adapter == "cpp_graded_stalk_transport":
            affine_completion = config.pop("affine_completion", False)
            if type(affine_completion) is not bool:
                raise TypeError("affine_completion requires an explicit boolean")
            provider = enable("2" if affine_completion else "1")
        elif adapter == "cpp_stalk_transport":
            column_fields = config.pop("column_fields", False)
            if type(column_fields) is not bool:
                raise TypeError("boolean column field profile required")
            version = "2" if config.get("parameter_space") is not None else "1"
            if column_fields:
                version = "4" if config.get("parameter_space") is not None else "3"
            provider = enable(version)
        else:
            provider = enable()
        from rexgraph.model_contract import invoke_model_provider
        context = config.pop("execution_context", None)
        state = invoke_model_provider(provider, "create", data, adapter=adapter, reference=reference,
                                      execution_context=context, **config)
    elif adapter == "native_field":
        if optimizer is not None or axes is not None:
            raise ValueError("native field action has no neural optimizer or fixed output feature axes")
        state = native_model(source, reference=_reference(source), **config)
    elif adapter == "online_flow":
        if optimizer is not None or axes is not None:
            raise ValueError("online correction declares no neural optimizer")
        from rexgraph.flow.model import online_model
        state = online_model(source, reference=_reference(source), **config)
    else:
        from rexgraph.nn.lifecycle import create_checkpoint
        state = create_checkpoint(source, adapter=adapter, configuration=config, axes=axes,
                                  optimizer=optimizer, reference=_reference(source), seed=seed)
    record_method("native-local-model-declaration", adapter=state.adapter, model_digest=state.coefficient_digest,
                  arithmetic=state.arithmetic)
    return state


def model_state(source, index=0, execution_context=None):
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("model cell index must be nonnegative")
    state, _ = _read_model_record(source, index, execution_context=execution_context)
    record_method("selected-native-model-state", model_digest=state.coefficient_digest)
    return state


def model_at(source, time):
    from rexgraph.io.model_record import read_model_history
    state, _ = read_model_history(source, time)
    record_method("exact-model-history-selection", model_digest=state.coefficient_digest)
    return state


def model_batch(source, state, targets, observed, inputs=None, contributors=()):
    from rexgraph.model_state import ModelBatch
    authorize(source, state, current_binding())
    if state.adapter == "cpp_relational_field_program" and state.adapter_version == "3":
        from rexgraph.cpp_relational_field_learning import field_program_batch
        import numpy as np
        if inputs is not None or contributors:
            raise ValueError("whole-field batches retain their actual dependencies inside each observation")
        result = field_program_batch(state, targets["observations"])
        from rexgraph.model_state import freeze_tree
        if result.targets != freeze_tree(targets) or not np.array_equal(result.observed, observed):
            raise ValueError("whole-field batch target/model/mask declarations differ")
        record_method("declared-whole-native-field-observations", dependencies=len(result.dependencies))
        return result
    refs = []
    for binding in contributors:
        if not isinstance(binding, Binding): raise TypeError("model evidence requires explicit source Bindings")
        binding.source.require("read")
        if SourcePolicy.intersection(current_policy(), binding.source.policy).digest != current_policy().digest:
            raise PermissionError("model query policy exceeds an evidence contributor policy")
        from rexgraph.tensor_field import FieldSource
        refs.append(FieldSource(binding.value, binding.ref.record_id, binding.ref.record_version))
    record_method("declared-model-observations")
    return ModelBatch(state.source, state.space, targets, observed, inputs, tuple(refs))


def model_infer(source, state, inputs=None, execution_context=None):
    from rexgraph.model_runtime import infer_model
    data = authorize(source, state, current_binding(), execution_context=execution_context)
    output = infer_model(state, data, inputs, execution_context=execution_context)
    record_method(output.method, arithmetic=output.arithmetic, model_digest=output.model_digest,
                  retained_axes=tuple((a.name, a.keys) for a in output.axes), source=state.source.as_record())
    return output


def model_train(source, state, batch, steps=1, execution_context=None):
    from rexgraph.model_runtime import train_model
    if not current_policy().permits("train"): raise PermissionError("model learning requires train permission")
    data = authorize(source, state, current_binding(), execution_context=execution_context)
    result = train_model(state, data, batch, steps=steps, execution_context=execution_context)
    record_method("native-model-observation-update", arithmetic=result.arithmetic,
                  parent=state.coefficient_digest, model_digest=result.coefficient_digest, step=result.step)
    return result


def model_transport(source, state, destination, mapping, optimizer="reset", execution_context=None):
    from rexgraph.model_runtime import transport_model
    from rexgraph.tensor_field import FieldSource
    data = authorize(source, state, current_binding(), execution_context=execution_context)
    if state.adapter in {"cpp_algebraic", "cpp_algebraic_grade", "cpp_algebraic_grade_metric"}:
        raise NotImplementedError("fixed algebraic actions do not declare parameter transport")
    if not isinstance(destination, Binding): raise TypeError("model transport requires a destination Binding")
    destination.source.require("read")
    if SourcePolicy.intersection(current_policy(), destination.source.policy).digest != current_policy().digest:
        raise PermissionError("model transport exceeds the destination policy")
    from rexgraph.source_binding import CppSourceReference
    if isinstance(state.source, CppSourceReference):
        reference = CppSourceReference(source_for(destination.value), destination.ref.record_id, destination.ref.record_version)
    else:
        reference = FieldSource(destination.value, destination.ref.record_id, destination.ref.record_version)
    result = transport_model(state, data, reference, mapping, optimizer=optimizer, execution_context=execution_context)
    record_method("declared-model-parameter-transport", arithmetic=result.arithmetic,
                  model_digest=result.coefficient_digest, optimizer_policy=optimizer)
    return result


def model_record(source, state):
    from rexgraph.io.model_record import model_record as make_record
    data = authorize(source, state, current_binding())
    record_method("native-model-record-construction", model_digest=state.coefficient_digest)
    return make_record(data, state)


def model_history(source, states, sources, times, time_axis="model_observation", time_unit="step"):
    from rexgraph.io.model_record import model_history as make_history
    states, sources = tuple(states), tuple(sources)
    if not states or len(states) != len(sources):
        raise ValueError("model history requires one binding per checkpoint")
    data = []
    for state, binding in zip(states, sources, strict=True):
        if not isinstance(binding, Binding):
            raise TypeError("model history sources require explicit Bindings")
        binding.source.require("read")
        if SourcePolicy.intersection(current_policy(), binding.source.policy).digest != current_policy().digest:
            raise PermissionError("model history query exceeds a contributor policy")
        data.append(authorize(binding.value, state, binding))
    result = make_history(states, data, times, time_axis=time_axis, time_unit=time_unit)
    record_method("native-model-history-construction", snapshots=len(states))
    return result


def model_field(source, output):
    from rexgraph.model_state import ModelOutput
    if not isinstance(output, ModelOutput): raise TypeError("model field requires ModelOutput")
    field = output.tensor()
    if output.method in {"exact-context-correspondence-v1", "exact-context-correspondence-v2"}:
        from rexgraph.cpp_correspondence import prepare_correspondence
        state, data = _read_model_record(source)
        if not output.source.matches(state.source):
            raise ValueError("correspondence field observes another context chart")
        prepared = prepare_correspondence(state, data)
        data = prepared[int(output.execution_receipt["side"] == "pattern")].destination
    else:
        data = source_for(source)
    field = field.bind(data)
    record_method("exact-model-field", model_digest=output.model_digest, arithmetic="rational")
    return field


def model_extend(source, state, destination, available_extension, execution_context):
    from rexgraph.cpp_relational_field_extension import cpp_available_field_extension_model
    from rexgraph.source_binding import CppSourceReference
    authorize(source,state,current_binding(),execution_context=execution_context)
    if not isinstance(destination,Binding): raise TypeError("extension target requires an explicit source Binding")
    destination.source.require("read")
    if SourcePolicy.intersection(current_policy(),destination.source.policy).digest != current_policy().digest:
        raise PermissionError("extension query exceeds its target source policy")
    target=source_for(destination.value,execution_context=execution_context)
    reference=CppSourceReference(target,destination.ref.record_id,destination.ref.record_version)
    result=cpp_available_field_extension_model(target,parent=state,available_extension=available_extension,
                                               execution_context=execution_context,reference=reference,
                                               limits=available_extension.admission()["limits"],max_bytes=available_extension.admission()["max_bytes"])
    record_method("explicit-versioned-native-source-extension",arithmetic="rational",parent=state.coefficient_digest,model_digest=result.coefficient_digest)
    return result


def model_complete(source, output, values, origin, state, execution_context):
    from rexgraph.cpp_relational_field_extension import evaluate_field_extension
    authorize(source,state,current_binding(),execution_context=execution_context)
    result=evaluate_field_extension(output,values,origin,execution_context=execution_context,state=state)
    record_method("explicit-input-prospective-field-completion",arithmetic="rational",model_digest=result.model_digest)
    return result


def model_finite_complete(source, output, ordinal, values, parameter_setting, choice_origin, state, execution_context):
    if output.method == "exact-available-relational-literal-finite-field-family-v1":
        from rexgraph.cpp_relational_literal_finite_field import evaluate_literal_finite_field as evaluate
    elif output.method == "exact-available-relational-finite-operator-family-v1":
        from rexgraph.cpp_relational_finite_operator import evaluate_finite_operator as evaluate
    else:
        from rexgraph.cpp_relational_finite_field import evaluate_finite_field as evaluate
    authorize(source,state,current_binding(),execution_context=execution_context)
    result=evaluate(output,ordinal,values,parameter_setting,choice_origin,execution_context=execution_context,state=state)
    record_method("explicit-prospective-finite-field-choice",arithmetic="rational",model_digest=result.model_digest,ordinal=ordinal)
    return result


def model_finite_branches(source, output, begin=0, count=16, execution_context=None):
    if output.method in {"exact-available-relational-literal-finite-field-family-v1", "exact-available-relational-literal-finite-field-evaluation-v1"}:
        from rexgraph.cpp_relational_literal_finite_field import literal_finite_field_branches as branches
    elif output.method in {"exact-available-relational-finite-operator-family-v1", "exact-available-relational-finite-operator-evaluation-v1"}:
        from rexgraph.cpp_relational_finite_operator import finite_operator_branches as branches
    else:
        from rexgraph.cpp_relational_finite_field import finite_field_branches as branches
    output.source.bind(source_for(source,execution_context=execution_context))
    result=branches(output,begin,count,execution_context=execution_context)
    record_method("bounded-complete-family-finite-branch-metadata",arithmetic="rational",origin_output=output.coefficient_digest,begin=begin,count=count)
    return result


def model_local(source, output, node, declaration, readout, origins, execution_context,
                work=(1,1,1,256*1024*1024,256*1024*1024)):
    from rexgraph.cpp_relational_field_local import solve_available_field_output
    output.source.bind(source_for(source,execution_context=execution_context))
    result=solve_available_field_output(output,node,declaration,readout,origins,execution_context=execution_context,work=work)
    record_method("complete-available-native-local-solve",arithmetic="rational",model_digest=output.model_digest)
    return result


def model_values(source, output, execution_context=None):
    if output.method == "exact-conditioned-occurrence-program-v1":
        from rexgraph.cpp_conditioned_occurrence_program import read_conditioned_occurrence_result
        output.source.bind(source_for(source, execution_context=execution_context))
        result = read_conditioned_occurrence_result(output, execution_context=execution_context).result()
        record_method("complete-conditioned-occurrence-field-metadata", arithmetic="rational", model_digest=output.model_digest)
        return {"program_sha256": output.field_archive["program_sha256"], "archive_sha256": output.field_archive["sha256"],
                "fields": result.fields(), "availability": "independent-native-context-admitted"}
    if output.method == "exact-conditioned-joint-field-family-v1":
        from rexgraph.cpp_conditioned_field_family import read_conditioned_family_result
        output.source.bind(source_for(source, execution_context=execution_context))
        result = read_conditioned_family_result(output, execution_context=execution_context)
        record_method("complete-conditioned-native-joint-field-metadata", arithmetic="rational", model_digest=output.model_digest)
        return {"program_sha256": output.field_archive["program_sha256"], "archive_sha256": output.field_archive["sha256"],
                "fields": result.fields(), "availability": "independent-native-context-admitted"}
    if output.method == "exact-conditioned-field-program-v1":
        from rexgraph.cpp_conditioned_field_program import read_conditioned_field_result
        output.source.bind(source_for(source, execution_context=execution_context))
        result = read_conditioned_field_result(output, execution_context=execution_context).result()
        record_method("complete-conditioned-native-field-metadata", arithmetic="rational", model_digest=output.model_digest)
        return {"program_sha256": output.field_archive["program_sha256"], "archive_sha256": output.field_archive["sha256"],
                "fields": result.fields(), "availability": "independent-native-context-admitted"}
    if output.method == "exact-available-relational-known-product-family-v1":
        from rexgraph.cpp_relational_known_product import read_known_product_result
        output.source.bind(source_for(source,execution_context=execution_context))
        owner=read_known_product_result(output,execution_context=execution_context)
        record_method("complete-available-native-known-product-metadata",arithmetic="rational",model_digest=output.model_digest)
        return {"program_sha256":output.field_archive["program_sha256"],"archive_sha256":output.field_archive["sha256"],
                "fields":owner.result().fields(),"availability":"independent-native-context-admitted"}
    if output.method in {"exact-available-relational-finite-operator-family-v1", "exact-available-relational-finite-operator-evaluation-v1"}:
        from rexgraph.cpp_relational_finite_operator import finite_operator_info
        output.source.bind(source_for(source,execution_context=execution_context))
        result=finite_operator_info(output,execution_context=execution_context)
        record_method("complete-available-native-operator-metadata",arithmetic="rational",model_digest=output.model_digest)
        return result
    if output.method in {"exact-available-relational-literal-finite-field-family-v1", "exact-available-relational-literal-finite-field-evaluation-v1"}:
        from rexgraph.cpp_relational_literal_finite_field import literal_finite_field_info
        output.source.bind(source_for(source,execution_context=execution_context))
        result=literal_finite_field_info(output,execution_context=execution_context)
        record_method("complete-available-native-finite-field-metadata",arithmetic="rational",model_digest=output.model_digest)
        return result
    if output.method in {"exact-available-relational-finite-field-family-v1", "exact-available-relational-finite-field-evaluation-v1"}:
        from rexgraph.cpp_relational_finite_field import finite_field_info
        output.source.bind(source_for(source,execution_context=execution_context))
        result=finite_field_info(output,execution_context=execution_context)
        record_method("complete-available-native-finite-field-metadata",arithmetic="rational",model_digest=output.model_digest)
        return result
    if output.method in {"exact-available-relational-extension-family-v1", "exact-available-relational-extension-evaluation-v1"}:
        from rexgraph.cpp_relational_field_extension import read_field_extension_result
        output.source.bind(source_for(source,execution_context=execution_context))
        result=read_field_extension_result(output,execution_context=execution_context)
        family=result.family() if output.method.endswith("evaluation-v1") else result
        record_method("complete-available-native-extension-field-metadata",arithmetic="rational",model_digest=output.model_digest)
        return {"program_sha256":output.field_archive["program_sha256"],"archive_sha256":output.field_archive["sha256"],
                "fields":result.fields(),"relationship":family.model().info(),"availability":"independent-native-context-admitted"}
    if output.method == "exact-available-relational-local-solve-v1":
        from rexgraph.cpp_relational_field_local import read_local_field_result
        output.source.bind(source_for(source,execution_context=execution_context))
        result=read_local_field_result(output,execution_context=execution_context)
        info=result.info()
        record_method("complete-available-native-local-field-metadata",arithmetic="rational",model_digest=output.model_digest)
        return {"program_sha256":output.field_archive["program_sha256"],"archive_sha256":output.field_archive["sha256"],
                "node":info["node"],"grade":info["grade"],"kind":info["kind"],"patch":info["patch"],
                "availability":"independent-native-context-admitted","packet_context":"complete-global-context",
                "fields":("global","patch","visible","reconstructed","augmented_rhs","augmented_response","interface_rhs","interface_response","residual")}
    if output.method in {"exact-available-relational-affine-field-program-v1", "exact-available-relational-affine-evaluation-v1"}:
        from rexgraph.cpp_relational_field_evidence_admission import read_available_field_learning_result
        output.source.bind(source_for(source, execution_context=execution_context))
        result = read_available_field_learning_result(output, execution_context=execution_context).result()
        record_method("complete-available-native-affine-field-metadata", arithmetic="rational", model_digest=output.model_digest)
        return {"program_sha256":output.field_archive["program_sha256"], "archive_sha256":output.field_archive["sha256"],
                "fields":result.fields(), "availability":"independent-native-context-admitted"}
    if output.method == "exact-available-relational-field-program-v1":
        from rexgraph.cpp_relational_field_admission import read_available_field_result
        output.source.bind(source_for(source, execution_context=execution_context))
        result = read_available_field_result(output, execution_context=execution_context)
        record_method("complete-available-native-field-metadata", arithmetic="rational", model_digest=output.model_digest)
        return {"program_sha256": output.field_archive["program_sha256"],
                "archive_sha256": output.field_archive["sha256"], "fields": result.result().fields(),
                "availability": "independent-native-context-admitted"}
    if execution_context is not None: raise TypeError("model output profile does not admit an execution context")
    if output.method == "exact-relational-comparison-field-program-v1":
        from rexgraph.cpp_relational_field_comparison import read_field_comparison_result
        output.source.bind(source_for(source))
        result = read_field_comparison_result(output)
        record_method("complete-native-comparison-field-metadata", arithmetic="rational", model_digest=output.model_digest)
        return {"program_sha256": output.field_archive["program_sha256"], "archive_sha256": output.field_archive["sha256"],
                "comparison": result.inspect(), "fields": ("input", "output", "signed_boundary", "unsigned_boundary", "original_Phi_fields")}
    if output.method in {"exact-relational-field-program-v1", "exact-relational-accession-field-program-v1", "exact-relational-affine-field-program-v1", "exact-relational-affine-evaluation-v1"}:
        from rexgraph.cpp_relational_field_program import read_field_program_result
        output.source.bind(source_for(source))
        result = read_field_program_result(output)
        record_method("complete-native-program-field-metadata", arithmetic="rational", model_digest=output.model_digest)
        return {"program_sha256": output.field_archive["program_sha256"],
                "archive_sha256": output.field_archive["sha256"], "fields": result.fields()}
    output.check_state()
    if output.method == "exact-joint-stalk-pattern-bank-v1":
        from rexgraph.cpp_stalk_pattern_model import stalk_pattern_bank_result
        from rexgraph.model_state import thaw_tree
        result = thaw_tree(stalk_pattern_bank_result(output))
        record_method("complete-joint-native-field-family", arithmetic="rational", model_digest=output.model_digest)
        return result
    if output.arithmetic == "algebraic":
        record_method("algebraic-polynomial-coordinate-observation", arithmetic="rational",
                      model_digest=output.model_digest, field=output.source.native.as_record(),
                      coefficient_basis="power", retained_axes=tuple((a.name, a.keys) for a in output.axes))
    else:
        record_method("model-value-observation", arithmetic=output.arithmetic, model_digest=output.model_digest)
    return output.values.copy()


def model_input(source, value, dtype="float64"):
    from rexgraph.model_state import model_input as convert
    from rexgraph.tensor_field import TensorField
    if not isinstance(value, TensorField):
        value.check_state()
    result = convert(value, dtype)
    record_method("declared-model-input-conversion", arithmetic=result.arithmetic,
                  origin=result.origin_digest, conversion_digest=result.coefficient_digest,
                  dtype=str(result.values.dtype))
    return result


def model_certify(source, state, inputs, candidate, recorded_binary=False):
    from rexgraph.model_runtime import certify_native_response
    data = authorize(source, state, current_binding())
    result = certify_native_response(state, data, inputs, candidate, recorded_binary=recorded_binary)
    record_method("exact-native-response-certificate", candidate_arithmetic=result["candidate_arithmetic"],
                  interpretation=result["interpretation"], model_digest=state.coefficient_digest)
    return result


def model_stalk(source, output, cell, node=None, max_coordinates=256, execution_context=None):
    if output.method in {"exact-available-relational-finite-operator-family-v1", "exact-available-relational-finite-operator-evaluation-v1"}:
        from rexgraph.cpp_relational_finite_operator import finite_operator_stalks
        output.source.bind(source_for(source,execution_context=execution_context))
        binding=current_binding()
        if binding is not None: binding.source.require("read")
        if node is None: raise ValueError("operator field selection requires an explicit target node")
        result=finite_operator_stalks(output,node,(cell,),execution_context=execution_context,coordinate_limit=max_coordinates)[0]
        record_method("available-native-operator-target-stalk-selection",node=node,cell=cell,origin_output=output.coefficient_digest)
        return result
    if output.method in {"exact-available-relational-literal-finite-field-family-v1", "exact-available-relational-literal-finite-field-evaluation-v1"}:
        from rexgraph.cpp_relational_literal_finite_field import literal_finite_field_stalks
        output.source.bind(source_for(source,execution_context=execution_context))
        binding=current_binding()
        if binding is not None: binding.source.require("read")
        if node is None: raise ValueError("finite field selection requires an explicit target node")
        result=literal_finite_field_stalks(output,node,(cell,),execution_context=execution_context,coordinate_limit=max_coordinates)[0]
        record_method("available-native-finite-target-stalk-selection",node=node,cell=cell,origin_output=output.coefficient_digest)
        return result
    if output.method in {"exact-available-relational-finite-field-family-v1", "exact-available-relational-finite-field-evaluation-v1"}:
        from rexgraph.cpp_relational_finite_field import finite_field_stalks
        output.source.bind(source_for(source,execution_context=execution_context))
        binding=current_binding()
        if binding is not None: binding.source.require("read")
        if node is None: raise ValueError("finite field selection requires an explicit target node")
        result=finite_field_stalks(output,node,(cell,),execution_context=execution_context,coordinate_limit=max_coordinates)[0]
        record_method("available-native-finite-target-stalk-selection",node=node,cell=cell,origin_output=output.coefficient_digest)
        return result
    if output.method == "exact-available-relational-known-product-family-v1":
        from rexgraph.cpp_relational_known_product import read_known_product_result
        output.source.bind(source_for(source,execution_context=execution_context))
        binding=current_binding()
        if binding is not None:binding.source.require("read")
        if node is None:raise ValueError("known-product stalk selection requires an explicit node")
        result=read_known_product_result(output,execution_context=execution_context).result()
        record_method("complete-available-native-known-product-field",arithmetic="rational",model_digest=output.model_digest)
        return result.select(node,(cell,),coordinate_limit=max_coordinates)[0]
    if output.method in {"exact-conditioned-occurrence-program-v1", "exact-conditioned-joint-field-family-v1", "exact-conditioned-field-program-v1", "exact-available-relational-field-program-v1", "exact-available-relational-affine-field-program-v1", "exact-available-relational-affine-evaluation-v1", "exact-available-relational-local-solve-v1", "exact-available-relational-extension-family-v1", "exact-available-relational-extension-evaluation-v1"}:
        from rexgraph.cpp_relational_field_program import field_program_stalks
        output.source.bind(source_for(source, execution_context=execution_context))
        binding = current_binding()
        if binding is not None: binding.source.require("read")
        if node is None: raise ValueError("whole-field program selection requires its explicit node")
        result = field_program_stalks(output, node, (cell,), coordinate_limit=max_coordinates,
                                      execution_context=execution_context)[0]
        record_method("available-native-program-stalk-selection", node=node, cell=cell, origin_output=output.coefficient_digest)
        return result
    if execution_context is not None: raise TypeError("model output profile does not admit an execution context")
    if output.method == "exact-relational-comparison-field-program-v1":
        from rexgraph.cpp_relational_field_comparison import read_field_comparison_result
        output.source.bind(source_for(source))
        if node is not None: raise ValueError("comparison output uses its declared original grade; node selects program fields")
        binding = current_binding()
        if binding is not None: binding.source.require("read")
        result = read_field_comparison_result(output).select((cell,), coordinate_limit=max_coordinates)[0]
        record_method("native-comparison-stalk-selection", cell=cell, origin_output=output.coefficient_digest)
        return result
    if output.method in {"exact-relational-field-program-v1", "exact-relational-accession-field-program-v1", "exact-relational-affine-field-program-v1", "exact-relational-affine-evaluation-v1"}:
        from rexgraph.cpp_relational_field_program import field_program_stalks
        output.source.bind(source_for(source))
        binding = current_binding()
        if binding is not None:
            binding.source.require("read")
        if node is None:
            raise ValueError("whole-field program selection requires its explicit node")
        result = field_program_stalks(output, node, (cell,), coordinate_limit=max_coordinates)[0]
        record_method("native-program-stalk-selection", node=node, cell=cell,
                      origin_output=output.coefficient_digest, program_sha256=output.field_archive["program_sha256"])
        return result
    if node is not None or max_coordinates != 256:
        raise ValueError("node/export bound applies to the whole-field program selection")
    from rexgraph.cpp_algebraic_stalk import cpp_algebraic_grade_stalk
    binding = current_binding()
    if binding is not None:
        binding.source.require("read")
    snapshot = source_for(source, algebraic_snapshot=True)
    result = cpp_algebraic_grade_stalk(output, snapshot, cell=cell)
    record_method("original-algebraic-stalk-selection", grade=result.grade, cell=cell,
                  origin_output=output.coefficient_digest, selection_digest=result.coefficient_digest)
    return result


def model_info(source, state, execution_context=None):
    authorize(source, state, current_binding(), execution_context=execution_context)
    record_method("native-model-descriptor")
    value = {"adapter": state.adapter, "adapter_version": state.adapter_version, "step": state.step,
            "digest": state.coefficient_digest, "parent": state.parent, "arithmetic": state.arithmetic,
            "source": state.source.as_record(), "resumable": "optimizer" in state.payload,
            "axes": tuple((a.name, a.keys) for a in (state.space, *state.output_axes))}
    if state.adapter == "cpp_relational_known_product":
        value.update(profile=state.configuration["profile"],program_sha256=state.configuration["program_sha256"],
                     operations=("create","prepare","infer","train"),trainable=True,source_backed=True,
                     storage="immutable-native-compound-and-field-archives",retains_correlated_families=True,
                     context_operations=("create","prepare","infer","train"),availability="independent-native-context-required",
                     observations=state.payload["observation_count"],original_publisher_count=state.payload["origin_count"])
    elif state.adapter == "cpp_relational_finite_operator":
        value.update(profile=state.configuration["profile"],operator_sha256=state.configuration["operator_sha256"],
                     operator_key=state.configuration["key"],revision=state.configuration["revision"],
                     operation="native-available-finite-operator",operations=("create","prepare","infer"),trainable=False,
                     source_backed=True,storage="immutable-native-operator-and-original-checkpoint",retains_correlated_families=True,
                     reference_digest=state.configuration["reference_digest"],relationship_status=state.configuration["status"],
                     context_operations=("create","prepare","infer"),availability="independent-native-context-required")
    elif state.adapter == "cpp_relational_literal_finite_field":
        value.update(profile=state.configuration["profile"], map_sha256=state.configuration["map_sha256"],
                     reader_sha256=state.configuration["reader_sha256"], base_digest=state.configuration["base_digest"],
                     operation="native-available-literal-finite-reader", operations=("create","prepare","infer"),
                     trainable=False, source_backed=True, retains_correlated_families=True,
                     storage="complete-native-reader-and-target-checkpoints", retains_whole_fields=True,
                     context_operations=("create","prepare","infer"), availability="independent-native-context-required")
    elif state.adapter == "cpp_relational_finite_field":
        value.update(profile=state.configuration["profile"],map_sha256=state.configuration["map_sha256"],
                     map_key=state.configuration["key"],revision=state.configuration["revision"],
                     operation="native-available-finite-map-instance",operations=("create","prepare","infer"),trainable=False,
                     source_backed=True,storage="immutable-native-map-and-target-checkpoints",retains_correlated_families=True,
                     target_count=len(state.configuration["target_digests"]),target_digests=state.configuration["target_digests"],
                     relationship_status=state.configuration["status"],context_operations=("create","prepare","infer"),
                     availability="independent-native-context-required")
    elif state.adapter == "cpp_conditioned_occurrence_program":
        value.update(profile=state.configuration["profile"], operation="native-conditioned-occurrence-whole-field",
                     operations=("create", "prepare", "infer", "train"), trainable=True, source_backed=True,
                     storage="immutable-native-typed-evidence-and-complete-literal-proof-history",
                     observations=state.payload["observation_count"], retains_correlated_families=True,
                     target_coordinate_domain="original-occurrence", role_target_coordinate_count=24,
                     context_operations=("create", "prepare", "infer", "train"),
                     availability="independent-native-context-required")
    elif state.adapter == "cpp_conditioned_field_family":
        value.update(profile=state.configuration["profile"], operation="native-conditioned-joint-whole-field",
                     operations=("create", "prepare", "infer"), trainable=False, source_backed=True,
                     member_count=state.configuration["member_count"], retains_correlated_families=True,
                     storage="immutable-native-members-interfaces-and-complete-literal-proof-history",
                     context_operations=("create", "prepare", "infer"), availability="independent-native-context-required")
    elif state.adapter == "cpp_conditioned_field_program":
        value.update(profile=state.configuration["profile"], operation="native-conditioned-whole-field",
                     operations=("create", "prepare", "infer", "train"), trainable=True, source_backed=True,
                     storage="immutable-native-evidence-and-complete-literal-proof-history",
                     observations=len(state.payload["history"]), retains_correlated_families=True,
                     context_operations=("create", "prepare", "infer", "train"),
                     availability="independent-native-context-required")
    elif state.adapter == "cpp_relational_field_program":
        learned = state.adapter_version in {"3", "6"}
        value.update(profile=state.configuration["profile"], program_sha256=state.configuration["program_sha256"],
                     program_key=state.configuration["key"], revision=state.configuration["revision"],
                     operations=("create", "prepare", "infer", "train") if learned else ("create", "prepare", "infer"), trainable=learned,
                     source_backed=state.adapter_version in {"2", "3", "4", "5", "6", "7"},
                     storage="immutable-native-program-and-field-archives")
        if learned:
            value.update(evidence_sha256=state.payload["evidence_sha256"], evidence_bytes=len(state.payload["evidence_archive"]),
                         evidence_revision=state.payload["revision"], observations=state.payload["observation_count"],
                         learned_node=state.configuration["learned_node"], retains_correlated_families=True, resumable=True)
        if state.adapter_version == "4":
            value.update(comparison_sha256=state.configuration["comparison_sha256"], comparison_grade=state.configuration["grade"],
                         comparison_channel=state.configuration["channel"], comparison_node=state.configuration["node"],
                         operation="native-fixed-whole-field-comparison")
        if state.adapter_version == "5":
            value.update(available_program_sha256=state.configuration["available_sha256"],
                         context_operations=("create", "prepare", "infer"),
                         availability="independent-native-context-required", operation="native-available-whole-field-program")
        if state.adapter_version == "6":
            value.update(context_operations=("create","prepare","infer","train"),availability="independent-native-context-required",
                         original_publisher_count=state.payload["origin_count"])
        if state.adapter_version == "7":
            value.update(operation="native-faithful-source-extension",extension_sha256=state.payload["extension_sha256"],
                         extension_key=state.configuration["extension_key"],extension_revision=state.configuration["extension_revision"],
                         relationship_status=state.configuration["status"],input_source=state.configuration["source_reference"],
                         context_operations=("create","prepare","infer"),availability="independent-native-context-required",
                         retains_correlated_families=True)
    elif state.adapter == "cpp_relation_response":
        value.update(profile=state.configuration["profile"], operation="green", trainable=True,
                     resumable=True,
                     input_grade=1, relation_count=len(state.space.keys), learned_transports=False,
                     learned_field_entries={s:len(v) for s,v in state.payload["fields"].items()})
        if state.adapter_version == "2":
            value.update(role_transports=state.configuration["role_transports"])
    elif state.adapter == "cpp_stalk_pattern_bank":
        from rexgraph.cpp_stalk_pattern_model import inspect_stalk_pattern_bank_model
        bank = inspect_stalk_pattern_bank_model(state, source_for(source))
        value.update(profile=bank["profile"], operation="joint-pattern-bank", trainable=False,
                     resumable=False, retains_affine_family=True, learned_transports=True,
                     parameter_count=len(bank["weight_origin"]),
                     unidentified_directions=bank["parameter_directions"]["shape"][1],
                     provenance_scope=bank["provenance_scope"], source_sha256=bank["source_sha256"],
                     evidence_sha256=bank["evidence_sha256"], observations=bank["observations"], peers=bank["peers"])
    elif state.adapter == "cpp_channels":
        value.update(profile=state.configuration["profile"], operation="raw-relational-channels", trainable=False,
                     input_grade=1, relation_count=len(state.space.keys), field_metric="unit",
                     channels=("T", "G", "F", "C"), geometry="raw",
                     coparticipation=state.configuration["coparticipation"],
                     trace_normalized=state.configuration["trace_normalized"])
    elif state.adapter == "cpp_local_sections":
        value.update(profile=state.configuration["profile"], operation="native-local-tensor-sections",
                     trainable=True, resumable=True, learned_transports=False,
                     local_stalk_count=len(state.space.keys),
                     stalk_dimensions=tuple(len(s["axes"]) for s in state.configuration["stalks"]),
                     acquired_charts=len(state.payload["observations"]),
                     retains_all_interpretations=True)
    elif state.adapter == "cpp_stalk_fields":
        value.update(profile=state.configuration["profile"], operation="whole-stalk-green", trainable=True,
                     resumable=True, input_grade=1, relation_count=len(state.space.keys),
                     stalk_dimensions=tuple(map(len, state.configuration["upper_axes"])),
                     acquired_columns=len(state.payload["observations"]), learned_transports=False)
    elif state.adapter == "cpp_stalk_objective":
        fit = state.payload["selection"]
        value.update(profile=state.configuration["profile"], operation="conditional-stalk-field-objective",
                     trainable=True, resumable=True, input_grade=1, output_grade=0,
                     evidence_rank=fit["evidence_fit"]["rank"], objective_rank=fit["rank"],
                     conditional_directions=len(fit["free_columns"]), retains_evidence_family=True)
    elif state.adapter == "cpp_stalk_transport":
        fit = state.payload["fit"]
        value.update(profile=state.configuration["profile"], operation="affine-stalk-transport", trainable=True,
                     resumable=True, learned_transports=True, input_grade=1, output_grade=0,
                     parameter_count=len(fit["particular"]), identified_rank=fit["rank"],
                     unidentified_directions=len(fit["free_columns"]), acquired_fields=state.step,
                     retains_affine_family=True, parameter_relations=True,
                     parameter_relation_query="whole-stalk-parameter-relations-query-v1",
                     context_hypotheses=True, context_hypothesis_query="whole-stalk-context-hypotheses-query-v1",
                     context_views=True, context_view_query="whole-stalk-context-views-query-v1",
                     context_diagrams=True, context_diagram_query="whole-stalk-context-diagrams-query-v1",
                     observation_groups=True, observation_group_profile="whole-stalk-transport-observation-group-v1")
        if state.adapter_version == "2":
            value.update(parameter_space=state.configuration["parameter_space"],basis_rank=fit["basis_rank"],
                         supported_map_entries=len(state.configuration["parameter_slots"]))
    elif state.adapter == "cpp_graded_stalk_transport":
        c = state.configuration; fit = state.payload["fit"]["fit"]
        value.update(profile=c["profile"],operation="graded-affine-stalk-transport",trainable=True,resumable=True,
                     learned_transports=True,input_grade=c["grade"],output_grade=c["grade"]-1,strict_chain_family=True,
                     parameter_count=len(fit["particular"]),identified_rank=fit["rank"],chain_rank=fit["chain_rank"],
                     observation_rank=fit["observation_rank"],unidentified_directions=len(fit["free_columns"]),
                     acquired_fields=state.step,retains_affine_family=True)
    elif state.adapter == "cpp_stalk_boundary":
        c = state.configuration
        value.update(profile=c["profile"], operation="graded-stalk-boundary", trainable=True, resumable=True,
                     input_grade=c["grade"], output_grade=c["grade"]-c["boundary_steps"],
                     boundary_steps=c["boundary_steps"], strict_chain_family=True, learned_transports=False,
                     grade_stalk_dimensions=tuple(tuple(map(len,family)) for family in c["stalk_axes"]),
                     acquired_columns=len(state.payload["observations"]))
    elif state.adapter == "cpp_stalk_sections":
        from rexgraph.cpp_stalk_sections import _states
        sections, fields = _states(state)
        value.update(profile=state.configuration["profile"], operation="compatible-sections-whole-stalk-fields",
                     trainable=True, resumable=True, learned_transports=False,
                     layers=sections.configuration["layers"], joint_states=len(sections.configuration["joint_values"]),
                     acquired_columns=len(fields.payload["observations"]), interfaces=len(state.configuration["interfaces"]),
                     retains_all_interpretations=True,
                     parameter_count=sum(len(o["values"]) for o in fields.payload["observations"]))
    elif state.adapter == "cpp_relational_sections":
        value.update(profile=state.configuration["profile"], operation="green-and-compatible-sections", trainable=True,
                     resumable=True,
                     input_grade=1, relation_count=len(state.space.keys), learned_transports=state.adapter_version == "4",
                     layers=state.configuration["layers"], joint_states=len(state.configuration["joint_values"]),
                     learned_compatibility_assertions=len(state.payload["compatibility_observations"]))
        if state.adapter_version in {"2", "3", "4"}:
            value.update(context_guard_profile="ordered-context-boundary-v1", automatic_guarded_links=True,
                         guard_limits=state.configuration["guard_limits"],
                         context_guard_conditions=tuple(dict.fromkeys(o["guard"]["profile"]
                             for o in state.payload["compatibility_observations"])))
        if state.adapter_version in {"3", "4"}:
            value.update(dictionary_links=state.configuration["dictionary_links"],
                         acquired_dictionary_observations=len(state.payload["dictionary_observations"]))
        if state.adapter_version == "4":
            value.update(role_transports=state.configuration["field"]["role_transports"],
                         transport_candidates=len(state.configuration["transport_search"]["candidates"]),
                         transport_fit=state.payload["transport_fit"])
    elif state.adapter == "cpp_algebraic_grade":
        c = state.configuration
        value.update(profile=c["profile"], input_profile=c["input_profile"], trainable=False,
                     input_grade=c["grade"], output_grade=c["layouts"][-1]["grade"],
                     boundary_steps=c["boundary_steps"], field_degree=state.source.native.field_degree,
                     coefficient_basis="power", publication=c["publication"])
    elif state.adapter == "cpp_algebraic_grade_metric":
        c = state.configuration
        value.update(profile=c["profile"], input_profile=c["input_profile"], operation=c["operation"],
                     trainable=False, input_grade=c["layouts"][-1]["grade"], output_kind=c["output_kind"],
                     output_grade=None if c["output_layout"] is None else c["output_layout"]["grade"],
                     metric_keys=c["metric_keys"], max_retained_coordinates=c["max_retained_coordinates"],
                     field_degree=state.source.native.field_degree, coefficient_basis="power", publication=c["publication"])
    return value


def install(register):
    for name, fn in (("MODEL_INIT", model_init), ("MODEL_STATE", model_state), ("MODEL_AT", model_at),
                     ("MODEL_BATCH", model_batch), ("MODEL_INFER", model_infer), ("MODEL_TRAIN", model_train),
                     ("MODEL_TRANSPORT", model_transport), ("MODEL_RECORD", model_record), ("MODEL_HISTORY", model_history),
                     ("MODEL_FIELD", model_field), ("MODEL_VALUES", model_values), ("MODEL_INPUT", model_input), ("MODEL_INFO", model_info), ("MODEL_CERTIFY", model_certify),
                     ("MODEL_STALK", model_stalk), ("MODEL_LOCAL", model_local),
                     ("MODEL_EXTEND", model_extend), ("MODEL_COMPLETE", model_complete),
                     ("MODEL_FINITE_COMPLETE", model_finite_complete), ("MODEL_FINITE_BRANCHES", model_finite_branches)):
        register(name)(fn)
