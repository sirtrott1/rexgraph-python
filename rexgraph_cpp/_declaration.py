"""Bounded exact declarations for canonical native source construction."""
from collections.abc import Mapping
from fractions import Fraction
from numbers import Integral


def _uint(value, bits=64):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("native declaration identifiers require integers")
    if not 0 <= value < 1 << bits:
        raise ValueError("native declaration integer exceeds its width")
    return int(value)


def normalise(declaration, max_records, max_slots, max_integer_bits):
    """Validate closed declarations and convert exact coefficients to hexadecimal."""
    budget = [0, 0, 0]
    limits = tuple(_uint(v) for v in (max_records, max_slots, max_integer_bits))
    def charge(index, amount):
        budget[index] += amount
        if budget[index] > limits[index]:
            raise ValueError("native construction exceeds its declared resource bound")
    def records(values):
        charge(0, len(values))
        return tuple(values)
    def closed(value, keys):
        if not isinstance(value, Mapping) or set(value) != set(keys):
            raise ValueError("unsupported native source declaration fields")
        return value
    def scalar(value):
        if isinstance(value, bool) or not isinstance(value, (Integral, Fraction)):
            raise TypeError("native construction requires integer or Fraction coefficients")
        value = Fraction(int(value)) if isinstance(value, Integral) else Fraction(
            int(value.numerator), int(value.denominator))
        charge(2, abs(value.numerator).bit_length() + value.denominator.bit_length())
        return (format(value.numerator, "x") + "/" + format(value.denominator, "x")).encode("ascii")
    def flag(value):
        if type(value) is not bool:
            raise TypeError("native occurrence aspects require boolean values")
        return value
    typed = isinstance(declaration, Mapping) and declaration.get("profile") == "declared-role-source-v1"
    d = closed(declaration, ("profile", "namespace", "semantic_scale", "max_grade", "types",
                             "witnesses", "relations", "carriers", "fields", "metrics") + (("role_aspects",) if typed else ()))
    if d["profile"] not in {"declared-exact-source-v1", "declared-role-source-v1"}:
        raise ValueError("unsupported native source construction profile")
    result = {"namespace": _uint(d["namespace"]), "semantic_scale": _uint(d["semantic_scale"], 16),
              "max_grade": _uint(d["max_grade"], 16)}
    if not result["namespace"]:
        raise ValueError("native source namespace must be nonzero")
    charge(0, result["max_grade"] + 1)
    roles = records(d["role_aspects"]) if typed else ()
    if any(not isinstance(name, str) or not name or len(name.encode("utf-8")) > 256 for name in roles):
        raise ValueError("role aspect names require 1 to 256 UTF8 bytes")
    if len(set(roles)) != len(roles) or len(roles) > 65532:
        raise ValueError("duplicate or excessive role aspect names")
    result["role_aspects"] = tuple(name.encode("utf-8") for name in roles)
    types = []
    for t in records(d["types"]):
        closed(t, ("name", "min_arity", "max_arity"))
        if not isinstance(t["name"], str) or not t["name"] or len(t["name"].encode("utf-8")) > 256:
            raise ValueError("native type name requires 1 to 256 UTF8 bytes")
        types.append((t["name"].encode("utf-8"), _uint(t["min_arity"]), _uint(t["max_arity"])))
    result["types"] = tuple(types)
    result["witnesses"] = tuple(_uint(v) for v in records(d["witnesses"]))
    if len(set(result["witnesses"])) != len(result["witnesses"]):
        raise ValueError("duplicate witness identity declaration")
    relations = []
    for r in records(d["relations"]):
        closed(r, ("grade", "key", "type", "occurrences"))
        occurrences = []
        for o in records(r["occurrences"]):
            closed(o, ("key", "lower", "exists", "negative", "magnitude") + (("aspects",) if typed else ()))
            aspects = tuple(_uint(a, 16) for a in records(o["aspects"])) if typed else ()
            if len(set(aspects)) != len(aspects) or any(a >= len(roles) for a in aspects):
                raise ValueError("duplicate or foreign occurrence role aspect")
            occurrences.append((_uint(o["key"]), _uint(o["lower"]), flag(o["exists"]),
                                flag(o["negative"]), scalar(o["magnitude"]), aspects))
        relations.append((_uint(r["grade"], 16), _uint(r["key"]), _uint(r["type"], 32), tuple(occurrences)))
    result["relations"] = tuple(relations)
    carriers, sizes = [], {}
    for c in records(d["carriers"]):
        closed(c, ("key", "kind", "grade", "coordinates", "size"))
        if c["kind"] not in {"source_grade", "explicit"}:
            raise ValueError("unsupported native construction carrier kind")
        key, size = _uint(c["key"]), _uint(c["size"])
        charge(1, size)
        if key in sizes:
            raise ValueError("duplicate native carrier key")
        sizes[key] = size
        carriers.append((key, 1 if c["kind"] == "source_grade" else 0,
                         _uint(c["grade"], 16), _uint(c["coordinates"]), size))
    result["carriers"] = tuple(carriers)
    fields = []
    for f in records(d["fields"]):
        closed(f, ("key", "carrier", "entries", "support"))
        carrier = _uint(f["carrier"])
        if carrier not in sizes:
            raise ValueError("native field references an undeclared carrier")
        charge(1, sizes[carrier])
        entries = []
        for entry in records(f["entries"]):
            if len(entry) != 2:
                raise ValueError("native sparse field entry requires index and value")
            entries.append((_uint(entry[0]), scalar(entry[1])))
        support = None if f["support"] is None else tuple(_uint(v) for v in records(f["support"]))
        fields.append((_uint(f["key"]), carrier, tuple(entries), support))
    result["fields"] = tuple(fields)
    metrics = []
    for m in records(d["metrics"]):
        closed(m, ("key", "carrier", "diagonal"))
        carrier = _uint(m["carrier"])
        if carrier not in sizes:
            raise ValueError("native metric references an undeclared carrier")
        charge(1, sizes[carrier] ** 2)
        metrics.append((_uint(m["key"]), carrier, tuple(scalar(v) for v in records(m["diagonal"]))))
    result["metrics"] = tuple(metrics)
    return result
