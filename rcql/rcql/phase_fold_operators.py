"""Exact operation word reads and prospective edits on native phase/fold records."""


def _archive(source):
    from rexgraph.cpp_phase_fold import read_phase_fold_record

    from .model_operators import _algebraic_cache

    return read_phase_fold_record(source, _archive_cache=_algebraic_cache())


def phase_fold_info(source):
    from copy import deepcopy

    from rexgraph.cpp_phase_fold import _check_record

    return deepcopy(_check_record(source)[0])


def phase_fold_prefix(source, count):
    from rexgraph.cpp_phase_fold import phase_fold_record

    return phase_fold_record(_archive(source).observed_prefix(count))


def phase_fold_observe(source, count, max_coordinates=65536):
    from rexgraph.cpp_phase_fold import phase_fold_observation

    from .execution_trace import record_method

    value = phase_fold_observation(_archive(source), count, max_coordinates=max_coordinates)
    record_method(
        "native-phase-fold-prefix-observation-v1", source=value["source"], logical_count=count
    )
    return value


def phase_fold_propose(source, key, position, coordinates, active=True):
    from rexgraph.cpp_phase_fold import phase_fold_record

    return phase_fold_record(_archive(source).propose(key, position, coordinates, active=active))


def phase_fold_apply(source, key):
    from rexgraph.cpp_phase_fold import phase_fold_record

    return phase_fold_record(_archive(source).apply(key))


def phase_fold_at(source, sequence, history_max_bytes=256 << 20, archive_limits=None):
    from rexgraph.cpp_phase_fold import phase_fold_history_at

    return phase_fold_history_at(
        source, sequence, limits=archive_limits, history_max_bytes=history_max_bytes
    )


def install(register):
    for name, function in [
        ("PHASE_FOLD_INFO", phase_fold_info),
        ("PHASE_FOLD_PREFIX", phase_fold_prefix),
        ("PHASE_FOLD_OBSERVE", phase_fold_observe),
        ("PHASE_FOLD_PROPOSE", phase_fold_propose),
        ("PHASE_FOLD_APPLY", phase_fold_apply),
        ("PHASE_FOLD_AT", phase_fold_at),
    ]:
        register(name)(function)


def install_signatures(register):
    from .signatures import OperatorSignature, TypePattern
    from .types import Domain, Exactness, RCType, ValueKind

    declarations = {
        "PHASE_FOLD_INFO": (0, ()),
        "PHASE_FOLD_PREFIX": (1, (("count", int),)),
        "PHASE_FOLD_OBSERVE": (1, (("count", int), ("max_coordinates", int))),
        "PHASE_FOLD_PROPOSE": (
            3,
            (("key", int), ("position", int), ("coordinates", None), ("active", bool)),
        ),
        "PHASE_FOLD_APPLY": (1, (("key", int),)),
        "PHASE_FOLD_AT": (
            1,
            (("sequence", int), ("history_max_bytes", int), ("archive_limits", None)),
        ),
    }
    for name, (required, arguments) in declarations.items():
        output = name in {"PHASE_FOLD_INFO", "PHASE_FOLD_OBSERVE"}
        result = (
            RCType(
                "PhaseFoldRecord",
                kind=ValueKind.RECORD,
                domain=Domain.METADATA,
                exactness=Exactness.STRUCTURAL,
            )
            if output
            else RCType("RexGraph", kind=ValueKind.REX, exactness=Exactness.STRUCTURAL)
        )
        register(
            OperatorSignature(
                name=name,
                source_kind=ValueKind.REX,
                inputs=tuple(
                    TypePattern(label, literal=kind, optional=i >= required)
                    for i, (label, kind) in enumerate(arguments)
                ),
                result=result,
                implementation_key="rcql.phase_fold_operators." + name.lower(),
                requires=frozenset({"read"}),
                memoizable=True,
                preconditions=(
                    "source retains a complete checked RFCXW word",
                    "logical operation counts and native publication sequences are selected explicitly",
                    "prospective replacements remain separate from retained operations",
                ),
            )
        )
