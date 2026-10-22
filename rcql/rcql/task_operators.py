"""Explicit native task construction and whole-field observations."""
def _owner(value):
    from .task_outcome import TaskOutcome
    if not isinstance(value, TaskOutcome):
        raise TypeError("task observation requires a TaskOutcome")
    return value


def task_outcome(source, output, task, asset, binding, execution_context):
    from .task_outcome import create_task_outcome
    return create_task_outcome(output, task, asset, binding, execution_context)


def task_status(source, outcome, execution_context):
    return _owner(outcome).native(execution_context).inspect(execution_context)["status"]


def task_verdict(source, outcome, execution_context):
    return _owner(outcome).verdict(execution_context)


def task_info(source, outcome, execution_context):
    return _owner(outcome).inspect(execution_context)


def task_certificate(source, outcome, execution_context):
    return _owner(outcome).certificate(execution_context)


def task_field(source, outcome, node, execution_context, occurrences=False):
    value = _owner(outcome)
    owner = value.native(execution_context)
    response = owner.response(execution_context)
    return {"field": response.result().field_archive(node, occurrences=occurrences),
            "source": value.source.as_record(), "node": node, "occurrences": occurrences,
            "executed_response": owner.inspect(execution_context)["executed_response"]}


def install(register):
    for name, fn in (("TASK_OUTCOME", task_outcome), ("TASK_STATUS", task_status),
                     ("TASK_VERDICT", task_verdict), ("TASK_INFO", task_info),
                     ("TASK_CERTIFICATE", task_certificate), ("TASK_FIELD", task_field)):
        register(name)(fn)


def install_signatures(register):
    from .signatures import OperatorSignature, TypePattern
    from .types import Domain, Exactness, RCType
    from .types import ValueKind as K
    outcome = TypePattern("outcome", kind=K.TASK_OUTCOME)
    context = TypePattern("execution_context")
    record = RCType("Record", kind=K.RECORD, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
    specs = (
        ("TASK_OUTCOME", (TypePattern("output", kind=K.MODEL_OUTPUT), TypePattern("task"),
            TypePattern("asset", literal=bytes), TypePattern("binding"), context),
            RCType("TaskOutcome", kind=K.TASK_OUTCOME, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)),
        ("TASK_STATUS", (outcome, context), RCType("Text", kind=K.TEXT, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)),
        ("TASK_VERDICT", (outcome, context), RCType("TaskVerdict", kind=K.UNKNOWN, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)),
        ("TASK_INFO", (outcome, context), record), ("TASK_CERTIFICATE", (outcome, context), record),
        ("TASK_FIELD", (outcome, TypePattern("node", literal=int), context, TypePattern("occurrences", literal=bool, optional=True)), record),
    )
    for name, inputs, result in specs:
        register(OperatorSignature(name=name, inputs=inputs, result=result, source_kind=K.REX,
            memoizable=False, implementation_key="native-whole-task-outcome"))
