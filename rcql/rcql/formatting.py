"""RCQL expression display and checked query text."""
from dataclasses import fields, is_dataclass
from fractions import Fraction
import json
from math import isfinite

from .ast import (Alias, Call, Comparison, ListExpr, Literal, Member,
                  MutationQuery, Parameter, Query, Reference, StructuralEdit)


def _expression(expr, *, strict=False):
    def render(value):
        return _expression(value, strict=strict)
    if isinstance(expr, Comparison):
        if expr.operation == "NOT":
            return f"(NOT {render(expr.left)})"
        return f"({render(expr.left)} {expr.operation} {render(expr.right)})"
    if isinstance(expr, StructuralEdit):
        return f"{render(expr.state)} {expr.operation} {render(expr.value)}"
    if isinstance(expr, Alias):
        return f"{render(expr.value)} AS {expr.name}"
    if isinstance(expr, Member):
        return f"{render(expr.value)}.{expr.name}"
    if isinstance(expr, ListExpr):
        return "[" + ", ".join(render(item) for item in expr.items) + "]"
    if isinstance(expr, Literal):
        value = expr.value
        if isinstance(value, Fraction):
            return f"{value.numerator}/{value.denominator}"
        if strict:
            if value is not None and type(value) not in (bool, str, int, float):
                raise TypeError("query text requires scalar literals; bind other values as parameters")
            if type(value) is float and not isfinite(value):
                raise ValueError("query text requires finite floating literals")
        else:
            from .planning import _plain_literal
            value = _plain_literal(value)
        return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else repr(value)
    if isinstance(expr, Parameter):
        return "$" + expr.name
    if isinstance(expr, Reference):
        return expr.name
    if isinstance(expr, Call):
        return f"{expr.name}({', '.join(render(a) for a in expr.args)})"
    if strict:
        raise TypeError("query text requires supported RCQL syntax nodes")
    return repr(expr)


def format_expr(expr):
    """Return a compact RCQL expression string."""
    return _expression(expr)


def _same(left, right):
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(_same(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float):
        return left.hex() == right.hex()
    return left == right


def format_query(query):
    """Render a complete query and check that parsing preserves its syntax tree.

    Values without RCQL literal syntax must remain bound parameters.
    """
    if type(query) not in (Query, MutationQuery):
        raise TypeError("query formatting requires Query or MutationQuery")
    def render(expr):
        return _expression(expr, strict=True)
    parts = ["EXPLAIN FROM" if query.explain else "FROM", render(query.source)]
    if query.source_alias is not None:
        parts.extend(("AS", query.source_alias))
    for binding in query.bindings:
        parts.extend(("LET", binding.name, "=", render(binding.value)))
    if isinstance(query, MutationQuery):
        parts.extend(("MUTATE", render(query.record_id), "SET"))
        assignments = [("actor", query.actor), ("valid_from", query.valid_from),
                       ("valid_to", query.valid_to), ("expected_version", query.expected_version),
                       ("expected_hash", query.expected_hash)]
        state, edits = query.resulting, []
        while isinstance(state, StructuralEdit):
            edits.append((state.operation, state.value))
            state = state.state
        assignments.append(("state", state))
        parts.append(", ".join(f"{name} = {render(value)}" for name, value in assignments))
        for operation, value in reversed(edits):
            parts.extend((operation, render(value)))
        parts.append("COMMIT")
    else:
        if query.matches:
            parts.extend(("MATCH", ", ".join(f"{b.name} IN {render(b.value)}" for b in query.matches)))
        if query.where is not None:
            parts.extend(("WHERE", render(query.where)))
        parts.extend(("RETURN", ", ".join(render(value) for value in query.returns)))
        if query.order:
            parts.extend(("ORDER BY", ", ".join(render(value) + (" DESC" if descending else " ASC")
                                               for value, descending in query.order)))
        if query.limit is not None:
            parts.extend(("LIMIT", str(query.limit)))
        if query.offset:
            parts.extend(("OFFSET", str(query.offset)))
    text = " ".join(parts)
    text.encode("utf8")
    from .parser import parse
    try:
        restored = parse(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("query cannot be represented by the current RCQL text syntax") from exc
    if not _same(query, restored):
        raise ValueError("query text would change its declared syntax tree")
    return text
