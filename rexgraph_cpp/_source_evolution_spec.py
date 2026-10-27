"""Bounded data declarations for native source-signature additions."""
from collections.abc import Mapping

ROLES = {"relation_role": 2, "direction": 3, "type": 4, "visibility": 5,
         "activity": 6, "provenance": 7, "custom": 8}
LIMITS = (64 << 20, 1_000_000, 1_000_000, 16_000_000, 100_000, 1_000_000, 64 << 20)


def unsigned(value, maximum=(1 << 64)-1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("source evolution requires bounded unsigned integers")
    return value


def limits(values):
    if type(values) is not tuple or len(values) != len(LIMITS):
        raise ValueError("source evolution requires seven explicit resource limits")
    if any(unsigned(v) == 0 or v > cap for v, cap in zip(values, LIMITS, strict=True)):
        raise ValueError("source evolution limits exceed admission")
    return values


def additions(specification, bounds):
    def closed(value, keys):
        if not isinstance(value, Mapping) or set(value) != set(keys):
            raise ValueError("unsupported source evolution declaration fields")
    def rows(value, cap):
        if type(value) is not tuple or len(value) > cap:
            raise ValueError("source evolution requires a bounded ordered tuple")
        return value
    def name(value):
        if type(value) is not str or not value or "\x00" in value or len(value.encode("utf-8")) > 4096:
            raise ValueError("source declaration names require bounded nonempty text")
        return value.encode("utf-8")
    def flag(value):
        if type(value) is not bool:
            raise TypeError("source declaration aspects require explicit booleans")
        return value
    closed(specification, ("profile", "aspects", "types"))
    if specification["profile"] != "source-signature-additions-v1":
        raise ValueError("unsupported source signature additions profile")
    aspects, types = [], []
    total = 0
    for row in rows(specification["aspects"], min(bounds[1], 65535)):
        closed(row, ("name", "role", "builder_default", "fill", "assignments"))
        if row["role"] not in ROLES:
            raise ValueError("unsupported source aspect role")
        assignments = rows(row["assignments"], bounds[5])
        total += len(assignments)
        if total > bounds[1]:
            raise ValueError("source assignment records exceed admission")
        values = []
        for item in assignments:
            if type(item) is not tuple or len(item) != 3:
                raise ValueError("source aspect assignment requires grade, occurrence and value")
            values.append((unsigned(item[0], 65535), unsigned(item[1]), flag(item[2])))
        fill = row["fill"]
        if fill is not None: flag(fill)
        aspects.append((name(row["name"]), ROLES[row["role"]], flag(row["builder_default"]),
                        fill is not None, False if fill is None else fill, tuple(values)))
    for row in rows(specification["types"], bounds[1]):
        closed(row, ("name", "min_arity", "max_arity"))
        types.append((name(row["name"]), unsigned(row["min_arity"]), unsigned(row["max_arity"])))
    return tuple(aspects), tuple(types)
