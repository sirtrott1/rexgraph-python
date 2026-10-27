"""Whole native function programs retain source identities, domains and families."""
from fractions import Fraction as Q

import pytest

native = pytest.importorskip("rexgraph_cpp._temporal_program")
Runtime = native.NativeRelationalRuntime
UNBOUNDED = (None, None, False, False)


def coordinate(runtime, name="time"):
    child = runtime.extend("coordinate", dict(name=name, clock=name + "-clock", unit="seconds", origin="declared-zero"))
    return child, child.selected


def function(runtime, coord, coefficients, *, domain=UNBOUNDED, denominator=(1,), label="function"):
    child = runtime.extend("function", dict(coordinate=coord, name=label, provenance="synthetic declaration",
        pieces=[dict(identity=1, domain=domain, numerator=coefficients, denominator=denominator, provenance=label)]))
    return child, child.selected


def program(runtime, opcode, operands=(), **options):
    spec = dict(opcode=opcode, role=1, constant=0, operands=operands, scale=1, shift=0,
                domain=None, coordinate=0, meaning=opcode + " declared whole function")
    spec.update(options)
    child = runtime.extend("program", spec)
    return child, child.selected


def words(runtime=None):
    r, t = coordinate(Runtime.create(89001) if runtime is None else runtime)
    r, arrangement = coordinate(r, "arrangement")
    r, x = program(r, "input")
    r, d = program(r, "derivative", (x,))
    r, p = program(r, "pullback", (x,), scale=2, shift=1)
    r, dp = program(r, "derivative", (p,))
    r, pd = program(r, "pullback", (d,), scale=2, shift=1)
    return r, t, arrangement, dp, pd


def fit_fixture(runtime=None):
    r, t, s, dp, pd = words(runtime)
    r, quadratic = function(r, t, (0, 0, 1))
    r, observed = function(r, t, (4, 8))
    r = r.extend("support", dict(coordinate=s, domains=[(Q(0), Q(2), True, True)], label="full validity"))
    support = r.selected
    r = r.extend("context", dict(support=support, meaning="observed context")); context = r.selected
    r = r.extend("function_set", dict(coordinate=s,
        members=[(101, 1, quadratic, support, True), (102, 1, quadratic, support, False)], label="observed participants"))
    population = r.selected
    parent = r
    r = r.extend("fit", dict(function_set=population, context=context, candidates=[dp, pd], expected=observed, observed_domain=None))
    return parent, r, t, s, dp, pd, context, support


def test_learned_word_transfers_whole_unseen_function_and_reopens():
    parent, r, t, _, dp, _, _, _ = fit_fixture()
    family = r.selected
    assert r.fit(family)["forced"] and r.fit(family)["matches"] == ({"program": dp, "inputs": ((1, 8, 101),)},)
    assert r.extends(parent)
    old_bytes = r.encode()
    r, cubic = function(r, t, (2, -1, 3, 1), label="independent held input")
    r = r.extend("apply_learned", dict(family=family, bindings=[(1, cubic)], label="held complete response"))
    output = r.selected
    assert r.function(output)["pieces"][0]["numerator"] == (Q(16), Q(48), Q(24))
    assert r.evaluate(output, Q(1, 3)) == Q(104, 3)
    r = r.extend("tensor", dict(function=output, piece=1, label="whole coefficient view")); tensor = r.selected
    assert r.evaluate(tensor, Q(1, 3), tensor=True) == Q(104, 3)
    reopened = Runtime.decode(*r.encode(), selected=tensor)
    assert not reopened.same_owner(r) and reopened.encode() == r.encode()
    assert reopened.fit(family) == r.fit(family)
    assert reopened.evaluate(tensor, Q(1, 3), tensor=True) == Q(104, 3)
    assert Runtime.decode(*old_bytes, selected=family).fit(family)["forced"]


def test_equal_functions_keep_distinct_participant_and_program_alternatives():
    r, t, s, dp, pd = words()
    r, constant = function(r, t, (3, 0))
    r, equal = function(r, t, (3, 0))
    r, zero = function(r, t, (0, 0))
    r = r.extend("support", dict(coordinate=s, domains=[(0, 1, True, True)], label="validity")); support = r.selected
    r = r.extend("context", dict(support=support, meaning="ambiguous context")); context = r.selected
    r = r.extend("function_set", dict(coordinate=s, members=[(11, 1, constant, support, True),
        (12, 1, equal, support, True)], label="distinct uses")); population = r.selected
    r = r.extend("fit", dict(function_set=population, context=context, candidates=[dp, pd], expected=zero, observed_domain=None))
    fit = r.fit(r.selected)
    assert fit["complete"] and not fit["forced"] and len(fit["matches"]) == 4
    assert {m["inputs"][0][1] for m in fit["matches"]} == {constant, equal}
    assert r.function(zero)["pieces"][0]["numerator"] == (Q(0), Q(0))


def test_gluing_retains_charts_and_original_rational_holes():
    r, t = coordinate(Runtime.create(89002))
    r, u = coordinate(r, "chart")
    r, f = function(r, t, (2, 1), domain=(0, Q(3, 2), True, True))
    r, g = function(r, u, (Q(3, 2), Q(1, 2)), domain=(3, 5, True, True))
    r = r.extend("support", dict(coordinate=t, domains=[(0, Q(3, 2), True, True)], label="left")); left = r.selected
    r = r.extend("support", dict(coordinate=t, domains=[(1, 2, True, True)], label="right")); right = r.selected
    child = r.extend("glue", dict(sections=[(f, left, 1, 0), (g, right, 2, 1)], label="compatible sections"))
    assert child.evaluate(child.selected, Q(7, 4)) == Q(15, 4)
    assert child.extends(r)
    r, hole = function(r, t, (0, 1), denominator=(0, 1), domain=(0, 1, True, True))
    with pytest.raises((ValueError, RuntimeError)):r.evaluate(hole, 0)
    r = r.extend("support", dict(coordinate=t, domains=[(0, 1, True, True)], label="includes hole")); support = r.selected
    before = r.encode()
    with pytest.raises(ValueError, match="incompatible"):
        r.extend("glue", dict(sections=[(hole, support, 1, 0)], label="unsupported extension"))
    assert r.encode() == before


def test_allen_support_is_exact_open_cell_with_closed_outer_endpoint():
    r, t = coordinate(Runtime.create(89003))
    r, s = coordinate(r, "parameter")
    r, a = function(r, s, (0,))
    r, b = function(r, s, (2,))
    r, c = function(r, s, (0, 1))
    r, d = function(r, s, (2, 1))
    r = r.extend("interval", dict(start=a, end=b, event_coordinate=t, domain=(-1, 3, True, True), label="fixed")); fixed = r.selected
    r = r.extend("interval", dict(start=c, end=d, event_coordinate=t, domain=(-1, 3, True, True), label="moving")); moving = r.selected
    r = r.extend("allen", dict(left=fixed, right=moving, label="exact crossing partition")); relation = r.selected
    r = r.extend("atom_support", dict(relation=relation, atoms=[0], label="Before"))
    assert r.support(r.selected) == ((Q(2), Q(3), False, True),)
    assert Runtime.decode(*r.encode(), selected=r.selected).support(r.selected) == r.support(r.selected)


def test_bad_values_and_uninitialized_owners_are_rejected():
    r = Runtime.create(89004)
    with pytest.raises((TypeError, ValueError)):coordinate(r, 1)
    with pytest.raises(ValueError):Runtime.create(True)
    r, t = coordinate(r)
    with pytest.raises(TypeError):function(r, t, (0.5,))
    with pytest.raises(RuntimeError):r.inspect(max_records=0)
    with pytest.raises(RuntimeError):Runtime.__new__(Runtime).encode()


def test_joint_evolution_retains_correlations_and_explicit_empty_pair_relation():
    _, r, t, s, dp, pd, context, support = fit_fixture()
    first = r.selected
    r, constant = function(r, t, (3,))
    r, zero = function(r, t, (0,))
    r = r.extend("context", dict(support=support, meaning="second actual use")); second_context = r.selected
    r = r.extend("function_set", dict(coordinate=s, members=[(201, 1, constant, support, True)], label="constant context")); population = r.selected
    r = r.extend("fit", dict(function_set=population, context=second_context, candidates=[dp, pd], expected=zero, observed_domain=None)); second = r.selected
    assert len(r.fit(second)["matches"]) == 2
    transition = dict(left=context, right=second_context, same_program=True, allowed_pairs=None, participation=[])
    child = r.extend("join", dict(families=[first, second], transitions=[transition]))
    result = child.evolution(child.selected)
    assert result["complete"] and result["forced"] and result["status"] == "satisfiable"
    assert len(result["witnesses"]) == 2
    assert Runtime.decode(*child.encode(), selected=child.selected).evolution(child.selected) == result
    transition["allowed_pairs"] = []
    impossible = r.extend("join", dict(families=[first, second], transitions=[transition]))
    assert impossible.evolution(impossible.selected)["status"] == "unsatisfiable"


def test_candidate_exhaustion_is_incomplete_and_does_not_force_a_program():
    limits = list(native.DEFAULT_LIMITS);limits[0] = 1
    r, t, s, dp, pd = words(Runtime.create(89005, limits=limits))
    r, constant = function(r, t, (3,))
    r, zero = function(r, t, (0,))
    r = r.extend("support", dict(coordinate=s, domains=[(0, 1, True, True)], label="context validity")); support = r.selected
    r = r.extend("context", dict(support=support, meaning="bounded observation")); context = r.selected
    r = r.extend("function_set", dict(coordinate=s, members=[(11, 1, constant, support, True)], label="constant")); population = r.selected
    r = r.extend("fit", dict(function_set=population, context=context, candidates=[dp, pd], expected=zero, observed_domain=None))
    assert not r.fit(r.selected)["complete"] and not r.fit(r.selected)["forced"]


def test_shared_immutable_owner_supports_concurrent_readers():
    from concurrent.futures import ThreadPoolExecutor
    _, r, _, _, _, _, _, _ = fit_fixture()
    expected = r.fit(r.selected)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(value == expected for value in pool.map(lambda _: r.fit(r.selected), range(32)))


def test_whole_field_function_uses_actual_source_pairing_and_complete_coefficients():
    import json
    import os
    from pathlib import Path
    root = os.environ.get("REXGRAPH_FIELD_FUNCTION_FIXTURE")
    if root is None:pytest.skip("synthetic native field-function envelope not supplied")
    root = Path(root)
    raw = (root / "whole-field.envelope").read_bytes()
    expected = json.loads((root / "independent.json").read_text())
    polynomial = native.NativeRelationalFieldFunction.from_envelope(raw, 56000)
    direct = native.NativeRelationalFieldFunction.from_envelope(raw, 56002, polynomial=False)
    assert polynomial.archive() == raw == direct.archive()
    coefficients = polynomial.coefficients()
    assert coefficients["power_axis"] == (0, 1, 2, 3)
    tensor = coefficients["tensor"]
    assert tensor["shape"] == (8, 4) and tensor["full_support"] and tensor["support_count"] == 32
    assert dict(tensor["entries"]) == {4 * i + j: Q(q) for i, row in enumerate(expected["coefficients"]) for j, q in enumerate(row) if Q(q)}
    ports = polynomial.ports()
    assert [p["source_cell_key"] for p in ports] == [101, 102, 103, 104]
    assert ports[0]["coordinates"] == (10301, 10302) and ports[-1]["coordinates"] == (10601, 10602)
    assert not polynomial.same_owner(direct)
    held = []
    for row in expected["evaluations"]:
        x = polynomial.evaluate(Q(row["parameter"]));y = direct.evaluate(Q(row["parameter"]))
        assert x.inspect() == y.inspect()
        assert dict(x.inspect()["entries"]) == {i: Q(q) for i, q in enumerate(row["field"]) if Q(q)}
        assert x.quadrance() == y.quadrance()
        held.append(x)
    assert held[0].same_owner(held[-1])
    del polynomial
    assert held[0].inspect()["shape"] == (8,)


def test_supported_zero_field_tensor_and_missing_coefficients_are_distinct():
    import os
    from pathlib import Path
    root = os.environ.get("REXGRAPH_FIELD_FUNCTION_CONTROLS")
    if root is None:pytest.skip("synthetic zero/missing field envelopes not supplied")
    root = Path(root)
    complete = (root / "supported-zero.envelope").read_bytes()
    owner = native.NativeRelationalFieldFunction.from_envelope(complete, 60000)
    info = owner.coefficients()["tensor"]
    assert info["full_support"] and info["support_count"] == 32 and info["entries"] == ()
    field = owner.evaluate(Q(1, 3))
    assert field.inspect()["full_support"] and field.inspect()["support_count"] == 8
    assert field.inspect()["entries"] == () and field.quadrance() == 0
    cold = native.NativeRelationalFieldFunction.from_envelope(owner.archive(), 60000)
    assert cold.coefficients() == owner.coefficients() and not cold.same_owner(owner)
    with pytest.raises(ValueError, match="missing polynomial coefficients"):
        native.NativeRelationalFieldFunction.from_envelope((root / "missing-coefficients.envelope").read_bytes(), 60000)


@pytest.mark.parametrize("opcode,expected", [("input", Q(3)), ("constant", Q(3)),
    ("derivative", Q(3)), ("pullback", Q(7)), ("add", Q(6)),
    ("subtract", Q(0)), ("multiply", Q(9)), ("divide", Q(1)), ("restrict", Q(3))])
def test_every_declared_function_opcode_executes_and_replays(opcode, expected):
    r, t = coordinate(Runtime.create(89008))
    r, f = function(r, t, (2, 3))
    r, x = program(r, "input")
    operands = () if opcode in ("input", "constant") else (x, x) if opcode in ("add", "subtract", "multiply", "divide") else (x,)
    options = {"constant": f} if opcode == "constant" else {"scale": 2, "shift": 1} if opcode == "pullback" else {"domain": (0, 1, True, False)} if opcode == "restrict" else {}
    r, p = program(r, opcode, operands, **options)
    assert r.program(p)["opcode"] == opcode
    child = r.extend("apply", dict(program=p, bindings=[] if opcode == "constant" else [(1, f)], label="full operation result"))
    assert child.evaluate(child.selected, Q(1, 3)) == expected
    cold = Runtime.decode(*child.encode(), selected=child.selected)
    assert cold.function(cold.selected) == child.function(child.selected)
    if opcode == "restrict":
        with pytest.raises((ValueError, RuntimeError)):child.evaluate(child.selected, 1)
    if opcode == "divide":
        with pytest.raises((ValueError, RuntimeError)):child.evaluate(child.selected, Q(-2, 3))


@pytest.mark.parametrize("operation,expected", [("support_union", (0, 3, True, True)),
    ("support_intersection", (1, 2, True, True)), ("support_difference", (0, 1, True, False))])
def test_each_boolean_support_operation_preserves_exact_endpoints(operation, expected):
    r, t = coordinate(Runtime.create(89009))
    r = r.extend("support", dict(coordinate=t, domains=[(0, 2, True, True)], label="left")); left = r.selected
    r = r.extend("support", dict(coordinate=t, domains=[(1, 3, True, True)], label="right")); right = r.selected
    r = r.extend(operation, dict(left=left, right=right, label=operation))
    assert r.support(r.selected) == (expected,)
    assert Runtime.decode(*r.encode(), selected=r.selected).support(r.selected) == (expected,)


def test_nonlinear_boundary_is_rejected_without_replacing_it_by_samples():
    r, t = coordinate(Runtime.create(89010))
    r, s = coordinate(r, "arrangement")
    r, start = function(r, s, (0, 0, 1))
    r, end = function(r, s, (1, 0, 1))
    before = r.encode()
    with pytest.raises(ValueError, match="affine"):
        r.extend("interval", dict(start=start, end=end, event_coordinate=t, domain=(0, 1, True, True), label="nonlinear"))
    assert r.encode() == before
