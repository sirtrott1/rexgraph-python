"""Exact observations over explicitly labelled owning algebraic source records."""


def _snapshot(source):
    from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record

    from .model_operators import _algebraic_cache
    return read_algebraic_snapshot_record(source, _archive_cache=_algebraic_cache())


def algebraic_query(source, request, max_sequence, execution_limits=None, cpu_workers=1, max_coordinates=65536):
    from rexgraph.cpp_algebraic_query import algebraic_query_observation, prepare_algebraic_query

    from .execution_trace import record_method
    snapshot = _snapshot(source)
    availability = (snapshot.namespace, max_sequence)
    query = prepare_algebraic_query(snapshot, request, availability=availability, execution_limits=execution_limits)
    result = query.execute(availability=availability, cpu_workers=cpu_workers)
    value = algebraic_query_observation(result, max_coordinates=max_coordinates)
    record_method("prepared-algebraic-query-v1", arithmetic="algebraic", source=value["publication"],
                  availability=availability, requirements=query.requirements, cpu_execution=result.cpu_execution)
    return value


def algebraic_window(source, key, lower, upper, max_sequence, execution_limits=None, max_coordinates=65536):
    from rexgraph.cpp_algebraic_query import algebraic_window_observation
    from rexgraph_cpp import NativeAlgebraicTensorWindow

    from .execution_trace import record_method
    snapshot = _snapshot(source)
    availability = (snapshot.namespace, max_sequence)
    window = NativeAlgebraicTensorWindow(snapshot.record(key), lower, upper, availability=availability,
                                        execution_limits=execution_limits)
    value = algebraic_window_observation(window, availability=availability, max_coordinates=max_coordinates)
    record_method("algebraic-coefficient-window-v1", arithmetic="algebraic", source=value["publication"],
                  availability=availability, requirements=window.requirements)
    return value


def algebraic_grade(source, request, max_sequence, execution_limits=None, max_coordinates=65536):
    from rexgraph.cpp_algebraic_grade import algebraic_grade_observation, prepare_algebraic_grade

    from .execution_trace import record_method
    snapshot = _snapshot(source)
    availability = (snapshot.namespace, max_sequence)
    tensor = prepare_algebraic_grade(snapshot, request, availability=availability, execution_limits=execution_limits)
    value = algebraic_grade_observation(tensor, availability=availability, max_coordinates=max_coordinates)
    record_method("original-algebraic-grade-field-v1", arithmetic="algebraic", source=value["publication"],
                  availability=availability, requirements=tensor.requirements)
    return value


def algebraic_grade_window(source, request, shape, lower, upper, max_sequence, execution_limits=None, max_coordinates=65536):
    from rexgraph.cpp_algebraic_grade import (
        algebraic_grade_window_observation,
        prepare_algebraic_grade,
    )
    from rexgraph_cpp import NativeAlgebraicGradeTensorWindow

    from .execution_trace import record_method
    snapshot = _snapshot(source)
    availability = (snapshot.namespace, max_sequence)
    tensor = prepare_algebraic_grade(snapshot, request, availability=availability, execution_limits=execution_limits)
    window = NativeAlgebraicGradeTensorWindow(tensor, shape, lower, upper, availability=availability)
    value = algebraic_grade_window_observation(window, availability=availability, max_coordinates=max_coordinates)
    record_method("original-algebraic-grade-window-v1", arithmetic="algebraic", source=value["publication"],
                  availability=availability, requirements=window.requirements)
    return value


def algebraic_grade_metric(source, request, metric_keys, operation, max_sequence, execution_limits=None,
                           max_coordinates=65536, max_retained_coordinates=1000000, other_request=None):
    from rexgraph.cpp_algebraic_grade import prepare_algebraic_grade
    from rexgraph.cpp_algebraic_metric import algebraic_grade_metric_observation
    from rexgraph_cpp import NativeAlgebraicMetricField

    from .execution_trace import record_method
    snapshot = _snapshot(source)
    availability = (snapshot.namespace, max_sequence)
    field = NativeAlgebraicMetricField(snapshot, metric_keys, availability=availability,
                                       max_retained_coordinates=max_retained_coordinates)
    tensor = prepare_algebraic_grade(snapshot, request, availability=availability, execution_limits=execution_limits)
    other = None if other_request is None else prepare_algebraic_grade(snapshot, other_request,
        availability=availability, execution_limits=execution_limits)
    value = algebraic_grade_metric_observation(field, tensor, operation, availability=availability,
                                               other=other, max_coordinates=max_coordinates)
    record_method("original-algebraic-grade-metric-v1", arithmetic="algebraic", source=value["publication"],
                  availability=availability, operation=operation, metric_keys=field.metric_keys,
                  max_retained_coordinates=field.max_retained_coordinates)
    return value


def algebraic_at(source, sequence, history_max_bytes=256 << 20, archive_limits=None):
    from rexgraph.cpp_algebraic_history import algebraic_history_at

    from .execution_trace import record_method
    value = algebraic_history_at(source, sequence, limits=archive_limits, history_max_bytes=history_max_bytes)
    publication = value.get_metadata(1, 0, "algebraic_snapshot")["publication"]
    record_method("native-algebraic-history-prefix-v1", source=publication, sequence=sequence)
    return value


def algebraic_history_info(source, history_max_bytes=256 << 20):
    from copy import deepcopy

    from rexgraph.cpp_algebraic_query import _check_retained_snapshot
    declared = _check_retained_snapshot(source, 0, 256 << 20, history_max_bytes)
    if "graph_history" in declared["publication"]:
        return deepcopy(declared["publication"]["graph_history"])
    if "history" not in declared["publication"]:
        raise ValueError("selected record does not retain native algebraic history")
    return deepcopy(declared["publication"]["history"])


def install(register):
    register("ALGEBRAIC_QUERY")(algebraic_query)
    register("ALGEBRAIC_WINDOW")(algebraic_window)
    register("ALGEBRAIC_GRADE")(algebraic_grade)
    register("ALGEBRAIC_GRADE_WINDOW")(algebraic_grade_window)
    register("ALGEBRAIC_GRADE_METRIC")(algebraic_grade_metric)
    register("ALGEBRAIC_AT")(algebraic_at)
    register("ALGEBRAIC_HISTORY_INFO")(algebraic_history_info)


def install_signatures(register):
    from .signatures import OperatorSignature, TypePattern
    from .types import Domain, Exactness, RCType, ValueKind
    declarations = {
        "ALGEBRAIC_QUERY": (("request", None), ("max_sequence", int), ("execution_limits", None),
                            ("cpu_workers", int), ("max_coordinates", int)),
        "ALGEBRAIC_WINDOW": (("key", int), ("lower", None), ("upper", None), ("max_sequence", int),
                             ("execution_limits", None), ("max_coordinates", int)),
        "ALGEBRAIC_GRADE": (("request", None), ("max_sequence", int), ("execution_limits", None), ("max_coordinates", int)),
        "ALGEBRAIC_GRADE_WINDOW": (("request", None), ("shape", None), ("lower", None), ("upper", None),
                                   ("max_sequence", int), ("execution_limits", None), ("max_coordinates", int)),
        "ALGEBRAIC_GRADE_METRIC": (("request", None), ("metric_keys", None), ("operation", str),
                                    ("max_sequence", int), ("execution_limits", None), ("max_coordinates", int),
                                    ("max_retained_coordinates", int), ("other_request", None)),
        "ALGEBRAIC_AT": (("sequence", int), ("history_max_bytes", int), ("archive_limits", None)),
        "ALGEBRAIC_HISTORY_INFO": (("history_max_bytes", int),),
    }
    for name, arguments in declarations.items():
        required = {"ALGEBRAIC_QUERY": 2, "ALGEBRAIC_WINDOW": 4, "ALGEBRAIC_GRADE": 2,
                    "ALGEBRAIC_GRADE_WINDOW": 5, "ALGEBRAIC_GRADE_METRIC": 4, "ALGEBRAIC_AT": 1,
                    "ALGEBRAIC_HISTORY_INFO": 0}[name]
        inputs = tuple(TypePattern(label, literal=kind, optional=i >= required)
                       for i, (label, kind) in enumerate(arguments))
        result = (RCType("RexGraph", kind=ValueKind.REX, exactness=Exactness.STRUCTURAL) if name == "ALGEBRAIC_AT" else
                  RCType("AlgebraicHistory", kind=ValueKind.RECORD, domain=Domain.METADATA, exactness=Exactness.STRUCTURAL)
                  if name == "ALGEBRAIC_HISTORY_INFO" else
                  RCType("AlgebraicObservation", kind=ValueKind.RECORD, domain=Domain.ALGEBRAIC, exactness=Exactness.ALGEBRAIC))
        preconditions = ("selected record declares its algebraic publication namespace and sequence",
                         "availability and execution admission are checked by the canonical native operation",
                         "results retain complete polynomial coordinates and supported zeros")
        if name == "ALGEBRAIC_GRADE_METRIC":
            preconditions = ("every retained cell has an explicitly selected certified metric record",
                             "native grade inputs retain their complete original boundary lineage",
                             "metric coordinate admission and cancellation between native phases are explicit")
        elif name == "ALGEBRAIC_AT":
            preconditions = ("selected record retains a complete native publication prefix and channel receipt",
                             "sequence selects an endpoint retained by that prefix",
                             "saved and reader byte and archive limits apply to restoration")
        elif name == "ALGEBRAIC_HISTORY_INFO":
            preconditions = ("selected record retains a complete native publication prefix and channel receipt",
                             "portable declarations and complete prefix and archive bytes are checked",
                             "reading the declaration does not restore native proof or carrier owners")
        register(OperatorSignature(name=name, source_kind=ValueKind.REX, inputs=inputs, result=result,
            implementation_key="rcql.algebraic_operators." + name.lower(), requires=frozenset({"read"}),
            memoizable=True, preconditions=preconditions))
