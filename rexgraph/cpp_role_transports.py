"""Finite exact role transport selection on acquired primary relation fields."""
from fractions import Fraction
from dataclasses import replace
from collections.abc import Mapping


def normalize_role_transports(table):
    """Retain distinct named nonnegative membership and role gains."""
    rows = []
    for entry in table:
        if (len(entry) != 3 or not isinstance(entry[0], str) or not entry[0]
                or any(isinstance(v, bool) or not isinstance(v, (int, Fraction)) or v < 0 for v in entry[1:])):
            raise ValueError("role transports require a name and two nonnegative exact gains")
        rows.append((entry[0], Fraction(entry[1]), Fraction(entry[2])))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("distinct role transports required")
    return tuple(sorted(rows))


def validate_role_transports(table, field):
    """Check gains against the declared original role axes and integer admission."""
    roles = {name for axes in field["lower_axes"] for name in axes[1:]}
    if table != normalize_role_transports(table) or any(not isinstance(v, Fraction) for row in table for v in row[1:]):
        raise ValueError("canonical rational role transports required")
    if len(table) > field["limits"][1]:
        raise ValueError("role transports exceed entry admission")
    for name, membership, role in table:
        if name not in roles:
            raise ValueError("role transport must name an original retained role")
        if any(max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > field["limits"][3] for q in (membership, role)):
            raise ValueError("role transport exceeds integer admission")


def validate_transport_search(state):
    """Require a closed candidate family and complete fit witness."""
    c = state.configuration
    search = c["transport_search"]
    if (not isinstance(search, Mapping) or set(search) != {"profile", "candidates", "limits"}
            or search["profile"] != "leave-context-out-role-transports-v1"
            or len(search["limits"]) != 2 or any(type(v) is not int or v <= 0 for v in search["limits"])):
        raise ValueError("closed role transport search and two positive limits required")
    candidates = search["candidates"]
    if not candidates or len(candidates) > search["limits"][0] or len(set(candidates)) != len(candidates):
        raise ValueError("distinct role transport candidates within admission required")
    for table in candidates:
        validate_role_transports(table, c["field"])
    if c["field"]["role_transports"] not in candidates:
        raise ValueError("selected role transport is outside the declared family")
    fit = state.payload["transport_fit"]
    if fit is None:
        if state.step or c["field"]["role_transports"] != candidates[0]:
            raise ValueError("unfitted role transport must use the initial candidate")
    elif (not isinstance(fit, Mapping) or set(fit) != {"profile", "source_archive_digest", "rows", "objectives",
                "best_indices", "selected_index", "candidate_receipts", "native_work", "contraction_work"}
            or fit["profile"] != search["profile"] or fit["source_archive_digest"] != state.source.native.archive_digest
            or type(fit["selected_index"]) is not int or not 0 <= fit["selected_index"] < len(candidates)
            or candidates[fit["selected_index"]] != c["field"]["role_transports"]):
        raise ValueError("role transport fit differs from its source or selected candidate")


def fit_role_transports(state, source, *, field_entries=None, field_model=None):
    """Select declared incidence gains by exact leave context out margins."""
    from rexgraph.cpp_relation_response import _prepare
    from rexgraph.cpp_relational_sections import _field_state
    search = state.configuration["transport_search"]
    base = _field_state(state) if field_model is None else field_model
    field = base.configuration
    labels = field["labels"]
    acquired = state.payload["fields"] if field_entries is None else field_entries
    fields = tuple(dict(acquired[label]) for label in labels)
    rows = tuple(sorted({row for entries in fields for row in entries}))
    size = len(base.space.keys)
    if (len(rows)+len(labels) > field["limits"][0]
            or (len(rows)+len(labels))*size > field["limits"][1]):
        raise ValueError("role transport training fields exceed native admission")
    admitted = set(field["training_rows"])
    if any(type(row) is not int or row not in admitted or type(count) is not int or count <= 0
            or count.bit_length() > field["limits"][3] for entries in fields for row,count in entries.items()):
        raise ValueError("role transport training fields exceed coordinate or integer admission")
    vectors = []
    for row in rows:
        values = [0]*size; values[row] = 1; vectors.append(tuple(values))
    for entries in fields:
        values = [0]*size
        for row, count in entries.items():
            values[row] = count
        vectors.append(tuple(values))
    work = contractions = 0
    bits = field["limits"][3]
    def scalar(value):
        nonlocal contractions
        contractions += 1
        if contractions + work > search["limits"][1]:
            raise ValueError("role transport search exceeds work admission")
        q = Fraction(value)
        if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > bits:
            raise ValueError("role transport objective exceeds integer admission")
        return q
    objectives = []; receipts = []
    for table in search["candidates"]:
        candidate = replace(base, configuration=dict(base.configuration, role_transports=table))
        plan = _prepare(candidate, source)
        work += plan.receipt["preparation_steps"]
        if work + contractions > search["limits"][1]:
            raise ValueError("role transport search exceeds work admission")
        result = plan.moments(tuple(vectors))
        work += result["execution_steps"]
        if work + contractions > search["limits"][1]:
            raise ValueError("role transport search exceeds work admission")
        gram = result["gram"]; n = len(rows)
        errors = total = skipped = 0; margin = Fraction(0)
        for i, row in enumerate(rows):
            scores = []; available = []
            for j, entries in enumerate(fields):
                count = entries.get(row, 0)
                inner = scalar(gram[i][n+j] - scalar(count*gram[i][i]))
                norm = scalar(gram[n+j][n+j] - scalar(2*count*gram[i][n+j]) + scalar(count*count*gram[i][i]))
                if norm < 0:
                    raise RuntimeError("negative exact leave-context-out quadrance")
                score = scalar(scalar(inner*inner)/scalar(gram[i][i]*norm)) if norm else Fraction(0)
                if not 0 <= score <= 1:
                    raise RuntimeError("role transport Green coupling exceeds its exact bound")
                scores.append(score); available.append(bool(norm))
            for j, entries in enumerate(fields):
                count = entries.get(row, 0)
                if not count:
                    continue
                others = tuple(score for k, score in enumerate(scores) if k != j and available[k])
                if not available[j] or not others:
                    skipped += count; continue
                gap = scalar(scores[j] - max(others))
                errors += count if gap <= 0 else 0
                total += count; margin = scalar(margin + scalar(count*gap))
        objectives.append(dict(errors=errors, eligible=total, skipped=skipped,
            mean_margin=scalar(margin/total) if total else Fraction(0)))
        receipts.append(dict(result["receipt"], execution_steps=result["execution_steps"],
            residual_exact_zero=result["residual_exact_zero"]))
    ordering = tuple((-o["errors"], o["mean_margin"]) for o in objectives)
    maximum = max(ordering)
    best = tuple(i for i, value in enumerate(ordering) if value == maximum)
    return dict(profile=search["profile"], source_archive_digest=base.source.native.archive_digest,
        rows=rows, objectives=tuple(objectives), best_indices=best, selected_index=best[0],
        candidate_receipts=tuple(receipts), native_work=work, contraction_work=contractions)
