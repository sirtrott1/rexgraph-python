"""Syntax argument declarations for exact native source observations."""

ARGUMENTS = {
    "ALGEBRAIC_QUERY": (("request", "max_sequence", "execution_limits", "cpu_workers", "max_coordinates"),
                        (None, 1, 65536)),
    "ALGEBRAIC_WINDOW": (("key", "lower", "upper", "max_sequence", "execution_limits", "max_coordinates"),
                         (None, 65536)),
    "ALGEBRAIC_GRADE": (("request", "max_sequence", "execution_limits", "max_coordinates"),
                        (None, 65536)),
    "ALGEBRAIC_GRADE_WINDOW": (("request", "shape", "lower", "upper", "max_sequence", "execution_limits", "max_coordinates"),
                               (None, 65536)),
    "ALGEBRAIC_GRADE_METRIC": (("request", "metric_keys", "operation", "max_sequence", "execution_limits",
                                "max_coordinates", "max_retained_coordinates", "other_request"),
                               (None, 65536, 1000000, None)),
    "ALGEBRAIC_AT": (("sequence", "history_max_bytes", "archive_limits"), (256 << 20, None)),
    "ALGEBRAIC_HISTORY_INFO": (("history_max_bytes",), (256 << 20,)),
    "PHASE_FOLD_INFO": ((), ()),
    "PHASE_FOLD_PREFIX": (("count",), ()),
    "PHASE_FOLD_OBSERVE": (("count", "max_coordinates"), (65536,)),
    "PHASE_FOLD_PROPOSE": (("key", "position", "coordinates", "active"), (True,)),
    "PHASE_FOLD_APPLY": (("key",), ()),
    "PHASE_FOLD_AT": (("sequence", "history_max_bytes", "archive_limits"), (256 << 20, None)),
}
