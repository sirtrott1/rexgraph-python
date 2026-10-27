"""Installed native ownership, exact actions and execution limits."""
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeField,
    NativeSource,
    PreparedField,
    build_info,
)

OPERATIONS = {0: "boundary", 1: "adjoint", 2: "hodge_down", 3: "hodge_up",
              4: "hodge", 6: "green", 7: "harmonic"}
CASES = [(source, profile, kind, grade)
         for source, profile in (("diagonal", "finite"), ("diagonal", "global"), ("positive", "global"))
         for kind in OPERATIONS if profile == "global" or kind not in (6, 7)
         for grade in range(3) if grade or kind not in (0, 1)]


@pytest.mark.parametrize("name,profile,kind,grade", CASES)
def test_canonical_exact_actions(native_fixtures, name, profile, kind, grade):
    root, oracle = native_fixtures
    source = NativeSource((root / f"{name}.rfcxs").read_bytes())
    plan = source.prepare((10, 11, 12), metric_keys=(2000, 2001, 2002),
                          metric_selection=name, profile=profile)
    expected = oracle[name, kind, grade]
    result = plan.execute(source.field(expected["input"]), operation=OPERATIONS[kind],
                          grade=grade, parameter=Fraction(1, 2) if kind == 6 else 0)
    assert result["values"] == tuple(map(Fraction, expected["values"]))
    assert result["namespace"] == 88 and result["sequence"] == 0
    assert result["input_field"] == expected["input"]
    assert result["grade"] == (grade - 1 if kind == 0 else grade)
    assert result["integer_bits"] > 0 and result["scalar_work"] > 0


def test_exact_supported_zero_and_large_integer(native_source):
    field = native_source.field(1001)
    assert field.owner is native_source and field.key == 1001 and field.carrier == 11
    assert field.values == (Fraction(1, 3), Fraction(0), Fraction(-2, 5))
    assert field.support == (0, 1, 2)
    assert native_source.field(3002).support == (0,)
    assert native_source.field(3003).values[0] == 1 << 512
    result = native_source.prepare((10, 11, 12)).execute(native_source.field(3003))
    assert all(isinstance(value, Fraction) for value in result["values"])


def test_large_rationals_do_not_depend_on_decimal_string_limits(native_source):
    expected = Fraction(1 << 16384, 7)
    field = native_source.field(3004)
    assert field.values == (expected, 0, 0)
    result = native_source.prepare((10, 11, 12)).execute(field, operation="boundary")
    assert result["values"] == (expected, -expected, 0)


def test_large_exact_green_parameter(native_source):
    coefficient = Fraction(1 << 16384, 7)
    plan = native_source.prepare((10, 11, 12), profile="global",
                                 limits=(4000000, 16000000, 16000000, 64000000, 3000000000))
    field = native_source.field(1001)
    values = plan.execute(field, operation="green", parameter=coefficient)["values"]
    matrix = ((6, -2, -1), (-2, 6, -1), (-1, -1, 2))
    assert tuple(values[i] + coefficient * sum(entry * value for entry, value in zip(row, values, strict=True))
                 for i, row in enumerate(matrix)) == field.values


@pytest.mark.parametrize("kwargs", [
    {"grade_carriers": (10, 10, 12)},
    {"grade_carriers": (10, 21, 12), "metric_keys": (2000, 2001, 2002), "metric_selection": "diagonal"},
    {"metric_keys": (2001, 2000, 2002), "metric_selection": "diagonal"},
    {"metric_keys": (1000, 2001, 2002), "metric_selection": "diagonal"},
    {"metric_keys": (2000, 2001, 2002)},
    {"metric_keys": (2000, 2001, 2002), "metric_selection": "positive"},
    {"limits": (4000000, 16000000, 16000000, 64000000)},
])
def test_preparation_rejects_carrier_metric_and_policy_mismatches(native_source, kwargs):
    options = {"grade_carriers": (10, 11, 12), **kwargs}
    with pytest.raises(ValueError):
        native_source.prepare(**options)


def test_equivalent_archive_is_a_distinct_publication(native_source):
    other = NativeSource(native_source.archive)
    assert other.identity == native_source.identity
    plan = native_source.prepare((10, 11, 12))
    with pytest.raises(ValueError, match="same authentic publication"):
        plan.execute(other.field(1001))


@pytest.mark.parametrize("field,error", [(3001, "carrier identity"), (1000, "carrier identity"),
                                        (3002, "full retained grade support")])
def test_input_carrier_and_support_are_explicit(native_source, field, error):
    with pytest.raises(ValueError, match=error):
        native_source.prepare((10, 11, 12)).execute(native_source.field(field))


def test_handles_outlive_source_variable(native_source):
    plan = native_source.prepare((10, 11, 12))
    field = native_source.field(1001)
    expected = plan.execute(field)
    del native_source
    gc.collect()
    assert plan.execute(field) == expected
    del field
    gc.collect()


def test_native_source_publication_cannot_be_reinitialized(native_source):
    identity = native_source.identity
    with pytest.raises(RuntimeError, match="immutable"):
        native_source.__init__(native_source.archive)
    assert native_source.identity == identity


def test_uninitialized_and_none_handles_reject(native_source):
    plan = native_source.prepare((10, 11, 12))
    with pytest.raises(TypeError):
        plan.execute(None)
    with pytest.raises(RuntimeError, match="not initialized"):
        plan.execute(NativeField.__new__(NativeField))
    with pytest.raises(RuntimeError, match="not initialized"):
        PreparedField.__new__(PreparedField).execute(native_source.field(1001))
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = NativeSource.__new__(NativeSource).identity


@pytest.mark.parametrize("mode", ["empty", "truncated", "trailing", "corrupt", "bounded"])
def test_archive_validation(native_source, mode):
    archive = native_source.archive
    if mode == "empty": archive = b""
    elif mode == "truncated": archive = archive[:-1]
    elif mode == "trailing": archive += b"x"
    elif mode == "corrupt": archive = archive[:70] + bytes([archive[70] ^ 1]) + archive[71:]
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        NativeSource(archive, max_bytes=len(archive) - 1 if mode == "bounded" else 256 * 1024 * 1024)


def test_retained_cutoff_excludes_future_frames(native_fixtures):
    root, _ = native_fixtures
    history = NativeSource((root / "diagonal-history.rfcxs").read_bytes())
    old = history.at(0)
    assert old.archive == (root / "diagonal.rfcxs").read_bytes()
    assert old.identity["sequence"] == 0 and history.identity["sequence"] == 1
    assert old.field(1001).values[1] == 0 and history.field(1001).values[1] == 5
    with pytest.raises((ValueError, IndexError)):
        history.at(2)


@pytest.mark.parametrize("profile", ["finite", "global"])
def test_empty_retained_grade(native_fixtures, profile):
    source = NativeSource((native_fixtures[0] / "empty.rfcxs").read_bytes())
    assert source.identity["coordinates"][-1] == ()
    field = source.field(1003)
    assert field.values == () and field.support == ()
    assert NativeSource(source.archive).identity == source.identity
    plan = source.prepare((10, 11, 12, 13), profile=profile)
    for operation in ("hodge_down", "hodge_up", "hodge"):
        result = plan.execute(field, operation=operation, grade=3)
        assert result["values"] == () and result["grade"] == 3
        assert result["input_carrier"] == result["output_carrier"] == 13
        assert result["input_field"] == 1003 and result["namespace"] == 88
    boundary = plan.execute(field, operation="boundary", grade=3)
    assert boundary["values"] == (0, 0) and boundary["grade"] == 2
    assert boundary["input_carrier"] == 13 and boundary["output_carrier"] == 12
    adjoint = plan.execute(source.field(1002), operation="adjoint", grade=3)
    assert adjoint["values"] == () and adjoint["grade"] == 3
    assert adjoint["input_carrier"] == 12 and adjoint["output_carrier"] == 13
    if profile == "global":
        for operation in ("green", "harmonic"):
            result = plan.execute(field, operation=operation, grade=3,
                                  parameter=Fraction(2, 7) if operation == "green" else 0)
            assert result["values"] == () and result["output_carrier"] == 13
    with pytest.raises(ValueError):
        source.prepare((10, 11, 12), profile=profile)
    with pytest.raises(ValueError, match="carrier"):
        plan.execute(source.field(1002), operation="hodge", grade=3)
    other = NativeSource(source.archive)
    with pytest.raises(ValueError, match="same authentic publication"):
        plan.execute(other.field(1003), operation="hodge", grade=3)
    assert NativeSource(source.archive).identity == source.identity


def test_prepared_plan_reuse_and_concurrent_calls(native_source):
    plan = native_source.prepare((10, 11, 12))
    assert native_source.prepare((10, 11, 12)) is plan
    assert native_source.prepare((10, 11, 12), limits=(4000000, 16000000, 16000000, 64000000, 10000)) is not plan
    field = native_source.field(1001)
    expected = plan.execute(field)
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(result == expected for result in executor.map(lambda _: plan.execute(field), range(20)))


def test_concurrent_preparation_reuses_one_plan(native_source):
    def prepare(_):
        return native_source.prepare((10, 11, 12), profile="global")
    with ThreadPoolExecutor(max_workers=4) as executor:
        plans = tuple(executor.map(prepare, range(20)))
    assert all(plan is plans[0] for plan in plans)


def test_cancellation_is_owned_and_does_not_damage_plan(native_source):
    plan = native_source.prepare((10, 11, 12))
    token = CancellationToken()
    token.cancel()
    assert token.cancelled
    with pytest.raises(ExecutionCancelled):
        plan.execute(native_source.field(1001), cancellation=token)
    assert plan.execute(native_source.field(1001))["values"]


def test_execution_reports_declared_resource_limit(native_source):
    plan = native_source.prepare((10, 11, 12), limits=(4000000, 16000000, 16000000, 64000000, 100))
    with pytest.raises(ExecutionLimitError) as failure:
        plan.execute(native_source.field(1001))
    assert failure.value.required > failure.value.limit == 100
    assert failure.value.resource == "exact integer growth bound"


@pytest.mark.parametrize("value", [-1, True, 1.5, 1 << 64])
def test_native_key_widths(native_source, value):
    with pytest.raises((TypeError, ValueError)):
        native_source.field(value)


@pytest.mark.parametrize("parameter", [0.5, True])
def test_native_parameters_do_not_round(native_source, parameter):
    with pytest.raises(TypeError, match="exact integer"):
        native_source.prepare((10, 11, 12)).execute(native_source.field(1001), parameter=parameter)


def test_build_receipt_identifies_engine_and_bridge():
    receipt = build_info()
    expected = json.loads((Path(__file__).resolve().parents[1] / "native-sdk.json").read_text())
    assert receipt["engine_version"] == expected["version"]
    assert receipt["engine_digest"] == expected["engine_digest"]
    assert len(receipt["engine_digest"]) == len(receipt["binding_digest"]) == 64
    assert receipt["scalar_policy"] == "gmp-rational-exact"
