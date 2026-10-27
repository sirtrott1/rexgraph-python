"""Type and capability declarations for persistent local models."""
from .types import Domain, Exactness, PredicateResult, RCType, ValueKind

ARGUMENTS = {
    "MODEL_INIT": (("adapter", "configuration", "axes", "optimizer", "seed"), ("native_field", None, None, None, 0)),
    "MODEL_STATE": (("index", "execution_context"), (0, None)),
    "MODEL_AT": (("time",), ()),
    "MODEL_BATCH": (("state", "targets", "observed", "inputs", "contributors"), (None, ())),
    "MODEL_INFER": (("state", "inputs", "execution_context"), (None, None)),
    "MODEL_EXTEND": (("state", "destination", "available_extension", "execution_context"), ()),
    "MODEL_COMPLETE": (("output", "values", "origin", "state", "execution_context"), ()),
    "MODEL_FINITE_COMPLETE": (("output", "ordinal", "values", "parameter_setting", "choice_origin", "state", "execution_context"), ()),
    "MODEL_FINITE_BRANCHES": (("output", "begin", "count", "execution_context"), (0, 16, None)),
    "MODEL_LOCAL": (("output", "node", "declaration", "readout", "origins", "execution_context", "work"), ((1,1,1,256*1024*1024,256*1024*1024),)),
    "MODEL_TRAIN": (("state", "batch", "steps", "execution_context"), (1, None)),
    "MODEL_TRANSPORT": (("state", "destination", "mapping", "optimizer", "execution_context"), ("reset", None)),
    "MODEL_RECORD": (("state",), ()),
    "MODEL_HISTORY": (("states", "sources", "times", "time_axis", "time_unit"), ("model_observation", "step")),
    "MODEL_FIELD": (("output",), ()),
    "MODEL_VALUES": (("output", "execution_context"), (None,)),
    "MODEL_STALK": (("output", "cell", "node", "max_coordinates", "execution_context"), (None, 256, None)),
    "MODEL_INPUT": (("value", "dtype"), ("float64",)),
    "MODEL_INFO": (("state", "execution_context"), (None,)),
    "MODEL_CERTIFY": (("state", "inputs", "candidate", "recorded_binary"), (False,)),
}


def _type(kind, arithmetic=None):
    return RCType(kind.value, kind=kind,
                  domain=Domain.RATIONAL if arithmetic == "rational" else Domain.ALGEBRAIC if arithmetic == "algebraic" else Domain.REAL if arithmetic == "approximate" else Domain.METADATA,
                  exactness=Exactness.RATIONAL if arithmetic == "rational" else Exactness.ALGEBRAIC if arithmetic == "algebraic" else Exactness.APPROXIMATE if arithmetic == "approximate" else Exactness.STRUCTURAL)


def model_literal(binding, value):
    from rexgraph.model_state import ModelBatch, ModelInput, ModelOutput, ModelState, ModelTimeline
    for cls, kind in ((ModelState, ValueKind.MODEL_STATE), (ModelOutput, ValueKind.MODEL_OUTPUT),
                      (ModelInput, ValueKind.MODEL_INPUT), (ModelBatch, ValueKind.MODEL_BATCH), (ModelTimeline, ValueKind.MODEL_TIMELINE)):
        if isinstance(value, cls):
            arithmetic = getattr(value, "arithmetic", None) if isinstance(value, ModelOutput) else None
            return _type(kind, arithmetic).with_(source=binding.ref, declaration_digest=value.coefficient_digest)
    return None


def refine(typed, children, context):
    if not context.native: raise TypeError("model operations require a selected native record")
    from rexgraph.model_state import ModelBatch, ModelOutput, ModelState

    from .model_operators import authorize
    name = typed.operator
    labels, defaults = ARGUMENTS[name]
    values = tuple(context.known_value(c) for c in children)
    missing = len(labels) - len(values)
    if missing: values += defaults[-missing:]
    first = values[0] if values else None
    arithmetic = None
    if isinstance(first, ModelState):
        authority = values[labels.index("execution_context")] if "execution_context" in labels else None
        authorize(context.binding.value, first, context.binding, execution_context=authority)
        arithmetic = first.arithmetic
    if name == "MODEL_INIT" and isinstance(first, str):
        if first not in {"native_field", "online_flow", "cpp_field", "cpp_integer", "cpp_metric", "cpp_perceptron", "cpp_binary_gate", "cpp_correspondence", "cpp_context_catalog", "cpp_relation_response", "cpp_channels", "cpp_relational_sections", "cpp_local_sections", "cpp_stalk_fields", "cpp_stalk_sections", "cpp_stalk_boundary", "cpp_graded_stalk_transport", "cpp_stalk_transport", "cpp_stalk_pattern_bank", "cpp_conditioned_occurrence_program", "cpp_conditioned_field_family", "cpp_conditioned_field_program", "cpp_relational_field_program", "cpp_assignment_evidence", "cpp_finite_search", "cpp_relational_finite_field", "cpp_relational_literal_finite_field", "cpp_relational_finite_operator", "cpp_relational_known_product", "cpp_stalk_objective", "cpp_algebraic", "cpp_algebraic_grade", "cpp_algebraic_grade_metric"}:
            from rexgraph.nn.lifecycle import _adapter
            _adapter(first)
        arithmetic = "algebraic" if first in {"cpp_algebraic", "cpp_algebraic_grade", "cpp_algebraic_grade_metric"} else "rational" if first in {"native_field", "cpp_field", "cpp_integer", "cpp_metric", "cpp_perceptron", "cpp_binary_gate", "cpp_correspondence", "cpp_context_catalog", "cpp_relation_response", "cpp_channels", "cpp_relational_sections", "cpp_local_sections", "cpp_stalk_fields", "cpp_stalk_sections", "cpp_stalk_boundary", "cpp_graded_stalk_transport", "cpp_stalk_transport", "cpp_stalk_pattern_bank", "cpp_conditioned_occurrence_program", "cpp_conditioned_field_family", "cpp_conditioned_field_program", "cpp_relational_field_program", "cpp_assignment_evidence", "cpp_finite_search", "cpp_relational_finite_field", "cpp_relational_literal_finite_field", "cpp_relational_finite_operator", "cpp_relational_known_product", "cpp_stalk_objective"} else "approximate"
    if (name in {"MODEL_TRAIN", "MODEL_TRANSPORT"} and isinstance(first, ModelState)
            and first.adapter == "cpp_correspondence"):
        raise NotImplementedError("fixed correspondence actions do not declare training or parameter transport")
    if (name in {"MODEL_TRAIN", "MODEL_TRANSPORT"} and isinstance(first, ModelState)
            and first.adapter in {"cpp_algebraic", "cpp_algebraic_grade", "cpp_algebraic_grade_metric"}):
        raise NotImplementedError("fixed algebraic actions do not declare training or parameter transport")
    if name == "MODEL_TRAIN":
        if not context.binding.source.policy.permits("train"):
            raise PermissionError("model training requires train permission")
        if isinstance(first, ModelState) and first.adapter == "native_field":
            raise TypeError("the native exact action is evaluated rather than trained")
        if isinstance(first, ModelState) and isinstance(values[1], ModelBatch):
            batch = values[1]; batch.check_state()
            if batch.space != first.space or not batch.source.matches(first.source):
                raise ValueError("training batch source or coordinates disagree")
        if isinstance(values[2], int) and (isinstance(values[2], bool) or values[2] < 1):
            raise ValueError("training steps must be positive")
    if name == "MODEL_TRANSPORT" and isinstance(first, ModelState) and values[2] is not None:
        from collections.abc import Mapping

        from rexgraph.coordinate_map import CoordinateMap
        from rexgraph.source_binding import CppSourceReference
        if isinstance(first.source, CppSourceReference):
            if not isinstance(values[2], Mapping):
                raise TypeError("native transport requires a declared endpoint and carrier map record")
        elif not isinstance(values[2], CoordinateMap):
            raise TypeError("model transport requires a CoordinateMap")
    if name == "MODEL_FIELD" and isinstance(first, ModelOutput) and first.arithmetic != "rational":
        if first.arithmetic == "algebraic":
            raise TypeError("algebraic model output retains polynomial coordinates and its field declaration")
        raise TypeError("numerical output cannot be labeled an exact field")
    if name == "MODEL_INPUT" and isinstance(first, ModelOutput) and first.arithmetic == "algebraic":
        raise TypeError("algebraic coefficients require an explicitly selected field embedding")
    if name == "MODEL_STALK":
        if isinstance(first, ModelOutput) and first.method == "exact-conditioned-assignment-response-v1":
            raise TypeError("conditioned assignment inspection requires MODEL_ASSIGNMENT_FIELD on its complete native family")
        program = (isinstance(first, ModelOutput) and first.method in {"exact-relational-field-program-v1", "exact-relational-accession-field-program-v1", "exact-relational-affine-field-program-v1", "exact-relational-affine-evaluation-v1", "exact-relational-comparison-field-program-v1", "exact-available-relational-field-program-v1", "exact-available-relational-affine-field-program-v1", "exact-available-relational-affine-evaluation-v1", "exact-available-relational-local-solve-v1", "exact-available-relational-extension-family-v1", "exact-available-relational-extension-evaluation-v1", "exact-available-relational-finite-field-family-v1", "exact-available-relational-literal-finite-field-family-v1", "exact-available-relational-finite-field-evaluation-v1", "exact-available-relational-literal-finite-field-evaluation-v1", "exact-available-relational-finite-operator-family-v1", "exact-available-relational-finite-operator-evaluation-v1"}
                   or len(values) > 2 and values[2] is not None)
        if isinstance(first, ModelOutput) and not program and first.method not in {"algebraic-grade-boundary-v1", "algebraic-grade-metric-action-v1"}:
            raise TypeError("MODEL_STALK requires a complete whole-grade algebraic output")
        if values[1] is not None and not program:
            from rexgraph.cpp_algebraic import _key
            _key(values[1])
        arithmetic = "rational" if program else "algebraic"
    if name == "MODEL_FIELD": arithmetic = "rational"
    if name == "MODEL_COMPLETE":
        if isinstance(first,ModelOutput) and first.method != "exact-available-relational-extension-family-v1":
            raise TypeError("MODEL_COMPLETE requires a complete unresolved extension family")
        arithmetic = "rational"
    if name in {"MODEL_FINITE_COMPLETE", "MODEL_FINITE_BRANCHES"}:
        if isinstance(first,ModelOutput) and first.method not in {"exact-available-relational-finite-field-family-v1", "exact-available-relational-literal-finite-field-family-v1", "exact-available-relational-finite-field-evaluation-v1", "exact-available-relational-literal-finite-field-evaluation-v1", "exact-available-relational-finite-operator-family-v1", "exact-available-relational-finite-operator-evaluation-v1", "exact-available-finite-projection-replay-v1"}:
            raise TypeError("finite operations require their complete finite field output")
        if name == "MODEL_FINITE_COMPLETE" and isinstance(first,ModelOutput) and first.method not in {"exact-available-relational-finite-field-family-v1", "exact-available-relational-literal-finite-field-family-v1", "exact-available-relational-finite-operator-family-v1"}:
            raise TypeError("finite completion requires a complete unresolved finite family")
        arithmetic = "rational"
    if name == "MODEL_LOCAL":
        if isinstance(first,ModelOutput) and first.method not in {"exact-available-relational-field-program-v1", "exact-available-relational-affine-evaluation-v1", "exact-available-relational-extension-evaluation-v1"}:
            raise TypeError("MODEL_LOCAL requires an admitted fixed or explicitly selected whole-field output")
        arithmetic = "rational"
    if name == "MODEL_VALUES" and isinstance(first, ModelOutput):
        arithmetic = "rational" if first.arithmetic == "algebraic" else first.arithmetic
    result = typed.result
    if arithmetic is None and name in {"MODEL_INFER", "MODEL_VALUES"}:
        result = result.with_(exactness=None)
    if arithmetic is not None and name in {"MODEL_INFER", "MODEL_LOCAL", "MODEL_COMPLETE", "MODEL_FINITE_COMPLETE", "MODEL_FINITE_BRANCHES", "MODEL_FIELD", "MODEL_VALUES", "MODEL_STALK"}:
        result = _type(result.kind, arithmetic)
    if name == "MODEL_VALUES" and isinstance(first, ModelOutput) and first.method in {"exact-joint-stalk-pattern-bank-v1", "exact-relational-field-program-v1", "exact-relational-accession-field-program-v1", "exact-relational-affine-field-program-v1", "exact-relational-affine-evaluation-v1", "exact-relational-comparison-field-program-v1", "exact-available-relational-field-program-v1", "exact-available-relational-affine-field-program-v1", "exact-available-relational-affine-evaluation-v1", "exact-available-relational-local-solve-v1", "exact-available-relational-extension-family-v1", "exact-available-relational-extension-evaluation-v1", "exact-available-relational-finite-field-family-v1", "exact-available-relational-literal-finite-field-family-v1", "exact-available-relational-finite-field-evaluation-v1", "exact-available-relational-literal-finite-field-evaluation-v1", "exact-available-relational-finite-operator-family-v1", "exact-available-relational-finite-operator-evaluation-v1", "exact-available-finite-projection-replay-v1", "exact-conditioned-assignment-response-v1"}:
        result = _type(ValueKind.RECORD)
    if name == "MODEL_STALK" and arithmetic == "rational":
        result = _type(ValueKind.RECORD, "rational")
    result_source = context.binding.ref
    if name in {"MODEL_TRANSPORT", "MODEL_EXTEND"}:
        from .binding import Binding
        if isinstance(values[1], Binding): result_source = values[1].ref
    return result.with_(source=result_source), [
        PredicateResult("model_identity", "verified" if isinstance(first, ModelState) else "deferred", "native source and complete model digest"),
        PredicateResult("model_execution", "deferred", "declared exact action or registered numerical model")]


def install(register):
    from .signatures import OperatorSignature, TypePattern
    v = ValueKind
    inputs = {
        "MODEL_INIT": (str, None, None, None, int), "MODEL_STATE": (int, None), "MODEL_AT": (None,),
        "MODEL_BATCH": (v.MODEL_STATE, None, None, None, None),
        "MODEL_INFER": (v.MODEL_STATE, None, None), "MODEL_TRAIN": (v.MODEL_STATE, v.MODEL_BATCH, int, None),
        "MODEL_EXTEND": (v.MODEL_STATE, None, None, None),
        "MODEL_COMPLETE": (v.MODEL_OUTPUT, None, int, v.MODEL_STATE, None),
        "MODEL_FINITE_COMPLETE": (v.MODEL_OUTPUT, int, None, int, int, v.MODEL_STATE, None),
        "MODEL_FINITE_BRANCHES": (v.MODEL_OUTPUT, int, int, None),
        "MODEL_LOCAL": (v.MODEL_OUTPUT, int, None, None, None, None, None),
        "MODEL_TRANSPORT": (v.MODEL_STATE, None, None, str, None),
        "MODEL_HISTORY": (None, None, None, str, str), "MODEL_RECORD": (v.MODEL_STATE,), "MODEL_FIELD": (v.MODEL_OUTPUT,),
        "MODEL_CERTIFY": (v.MODEL_STATE, v.TENSOR_FIELD, None, bool), "MODEL_INPUT": (None,str), "MODEL_VALUES": (v.MODEL_OUTPUT, None), "MODEL_INFO": (v.MODEL_STATE, None),
        "MODEL_STALK": (v.MODEL_OUTPUT, None, (int, type(None)), int, None)}
    outputs = {"MODEL_INIT": v.MODEL_STATE, "MODEL_STATE": v.MODEL_STATE, "MODEL_AT": v.MODEL_STATE,
        "MODEL_BATCH": v.MODEL_BATCH, "MODEL_INFER": v.MODEL_OUTPUT, "MODEL_TRAIN": v.MODEL_STATE,
        "MODEL_EXTEND": v.MODEL_STATE, "MODEL_COMPLETE": v.MODEL_OUTPUT,
        "MODEL_FINITE_COMPLETE": v.MODEL_OUTPUT, "MODEL_FINITE_BRANCHES": v.RECORD,
        "MODEL_LOCAL": v.MODEL_OUTPUT,
        "MODEL_TRANSPORT": v.MODEL_STATE, "MODEL_RECORD": v.REX, "MODEL_FIELD": v.TENSOR_FIELD,
        "MODEL_VALUES": v.MODEL_VALUES, "MODEL_INFO": v.RECORD, "MODEL_HISTORY": v.REX, "MODEL_INPUT": v.MODEL_INPUT, "MODEL_CERTIFY": v.RECORD,
        "MODEL_STALK": v.MODEL_OUTPUT}
    for name, (labels, defaults) in ARGUMENTS.items():
        patterns = []
        for i, (label, kind) in enumerate(zip(labels, inputs[name], strict=True)):
            optional = i >= len(labels) - len(defaults)
            patterns.append(TypePattern(label, kind=kind, optional=optional,
                            literal=type(None) if optional else None) if isinstance(kind, ValueKind)
                            else TypePattern(label, literal=kind, optional=optional))
        requires = frozenset({"read", "train"}) if name == "MODEL_TRAIN" else frozenset({"read"})
        register(OperatorSignature(name=name, source_kind=v.REX, inputs=tuple(patterns), result=_type(outputs[name]),
            implementation_key="rexgraph.model_runtime." + name.lower(), requires=requires,
            memoizable=name not in {"MODEL_LOCAL", "MODEL_EXTEND", "MODEL_COMPLETE", "MODEL_FINITE_COMPLETE", "MODEL_FINITE_BRANCHES"},
            preconditions=("native model and observations name their complete source and coordinates",
                           "publication remains an explicit RCDB mutation")))
