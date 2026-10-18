"""Exact CPU row scoring, input scope and application routes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from rexgraph_cpp import CancellationToken, ExecutionCancelled, ExecutionLimitError, NativePerceptron, NativeSource
from test_construction import ROWS, batch, declaration, model

from rexgraph.cpp_perceptron import cpp_perceptron_input
from rexgraph.cpp_learning import _int64
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value, unpack_value


@pytest.mark.parametrize("value", [0, -(1 << 63), (1 << 63) - 1, np.int64(-7), np.uint64(7),
                                 Fraction(7), Decimal("7.0")])
def test_integer_boundary_preserves_exact_carriers(value):
    assert _int64(value) == int(value)
    assert type(_int64(value)) is int


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.0, np.float64(1), Fraction(1, 2),
                                 Decimal("0.5"), 1 << 63, -(1 << 63) - 1, np.uint64(1 << 63)])
def test_integer_boundary_rejects_approximate_fractional_boolean_or_overflow(value):
    with pytest.raises((TypeError, OverflowError)): _int64(value)


@pytest.mark.parametrize("workers", [1, 2, 4, 64])
@pytest.mark.parametrize("tile_rows", [1, 3, 256])
def test_native_scoring_matches_ordered_integer_oracle(workers, tile_rows):
    weights, bias = (7, -9, 11), -5
    rows = tuple(((), ((0, 2), (1, -1)), ((2, 0), (1, 3), (1, -2)),
                  ((0, 1), (2, -1), (0, -1)))[i % 4] for i in range(1031))
    native = NativePerceptron(weights, bias)
    before = native.snapshot
    expected = tuple(bias + sum(weights[i] * v for i, v in row) for row in rows)
    out = native.forward(rows, cpu_workers=workers, rows_per_tile=tile_rows)
    assert out["scores"] == expected
    assert out["values"] == tuple(1 if s >= 0 else -1 for s in expected)
    serial = native.forward(rows)
    assert serial == {k: out[k] for k in ("values", "scores")}
    assert native.snapshot == before
    report = out["cpu_execution"]
    assert report["requested_workers"] == workers
    assert report["worker_slots"] == min(workers, report["tiles"])
    assert report["launched_threads"] == report["worker_slots"] - 1


def test_zero_margin_and_ordered_overflow_are_preserved():
    native = NativePerceptron((1,))
    maximum = (1 << 63) - 1
    assert native.forward(((),), cpu_workers=4)["values"] == (1,)
    rows = (((0, maximum), (0, -1), (0, 1)),)
    assert native.forward(rows, cpu_workers=4)["scores"] == (maximum,)
    before = native.snapshot
    for workers in (1, 4):
        with pytest.raises(OverflowError):
            native.forward(((), ((0, maximum), (0, 1), (0, -1))),
                           cpu_workers=workers, rows_per_tile=1)
    assert native.snapshot == before
    assert native.forward(rows, cpu_workers=4)["scores"] == (maximum,)


@pytest.mark.parametrize("kwargs", [{"cpu_workers": 0}, {"cpu_workers": 65}, {"cpu_workers": -1},
                                  {"cpu_workers": True}, {"cpu_workers": 1.0},
                                  {"cpu_workers": 2, "rows_per_tile": 0},
                                  {"cpu_workers": 2, "rows_per_tile": 1000001},
                                  {"cpu_workers": 2, "rows_per_tile": False},
                                  {"rows_per_tile": 1}])
def test_native_policy_rejects_invalid_counts(kwargs):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        NativePerceptron((1,)).forward(((),), **kwargs)


def test_cancellation_bounds_and_bad_indices_return_no_partial_result():
    native = NativePerceptron((1,))
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): native.forward(((),), cpu_workers=4, cancellation=token)
    with pytest.raises(IndexError): native.forward(((), ((1, 2),)), cpu_workers=4, rows_per_tile=1)
    with pytest.raises(ValueError): native.forward((), cpu_workers=4)
    empty = NativePerceptron.__new__(NativePerceptron)
    with pytest.raises(RuntimeError): empty.forward(((),), cpu_workers=4)
    bounded = NativePerceptron((1,), limits=(5, 100, 100, 100, 64))
    assert bounded.forward(((), ()))["values"] == (1, 1)
    with pytest.raises(ExecutionLimitError): bounded.forward(((), ()), cpu_workers=2, rows_per_tile=1)
    assert native.snapshot == {"weights": (1,), "bias": 0, "mistakes": 0}


def test_concurrent_calls_on_one_model_are_independent():
    native = NativePerceptron((1, -2), 3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda _: native.forward(ROWS * 1000, cpu_workers=2, rows_per_tile=64), range(9)))
    assert all(o == outputs[0] for o in outputs)
    assert native.snapshot == {"weights": (1, -2), "bias": 3, "mistakes": 0}


@pytest.mark.parametrize("storage", ["values", "packed"])
def test_core_record_restore_and_application_routes(storage):
    from rcql import Executor, parse
    from system.serialize import json_value
    owner = NativeSource.from_declaration(declaration())
    state = model(owner, parameter_storage=storage)
    child = train_model(state, owner, batch(state, owner), steps=4)
    restored, source = read_model_record(model_record(owner, child))
    inputs = cpp_perceptron_input(restored, source, ROWS, cpu_workers=2, rows_per_tile=1)
    assert unpack_value(pack_value(thaw_tree(inputs))) == inputs
    output = infer_model(restored, source, inputs)
    serial = infer_model(restored, source, cpp_perceptron_input(restored, source, ROWS))
    assert np.array_equal(output.values, serial.values)
    assert output.execution_receipt["scores"] == serial.execution_receipt["scores"]
    assert output.model_digest == serial.model_digest == restored.coefficient_digest
    assert output.execution_receipt["cpu_execution"]["worker_slots"] == 2
    record = model_record(source, restored)
    result = Executor(sources={"record": record}, params={"input": inputs}).execute(
        parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
    assert result.values[0].coefficient_digest == output.coefficient_digest
    assert json_value(output)["execution_receipt"]["cpu_execution"]["worker_slots"] == 2
    assert train_model(restored, source, batch(restored, source)).coefficient_digest == train_model(
        child, owner, batch(child, owner)).coefficient_digest


@pytest.mark.parametrize("cpu", [{"workers": 0, "rows_per_tile": 1}, {"workers": True, "rows_per_tile": 1},
                                 {"workers": 65, "rows_per_tile": 1}, {"workers": 2, "rows_per_tile": 0},
                                 {"workers": 2}, {"workers": 2, "rows_per_tile": 1, "extra": 0}, None])
def test_core_rejects_forged_cpu_input(cpu):
    owner = NativeSource.from_declaration(declaration()); state = model(owner)
    inputs = {**cpp_perceptron_input(state, owner, ROWS), "cpu": cpu}
    with pytest.raises((TypeError, ValueError)): infer_model(state, owner, inputs)


@pytest.mark.parametrize("name", ["source", "primary_space", "feature_carrier", "feature_coordinates"])
def test_cpu_input_keeps_source_and_coordinate_scope(name):
    owner = NativeSource.from_declaration(declaration()); state = model(owner)
    inputs = thaw_tree(cpp_perceptron_input(state, owner, ROWS, cpu_workers=2, rows_per_tile=1))
    inputs[name] = ({} if name == "source" else (state.space.name, tuple(reversed(state.space.keys)))
                    if name == "primary_space" else 9999)
    with pytest.raises(ValueError): infer_model(state, owner, inputs)


def test_cpu_inference_policy_cannot_change_training_contract():
    owner = NativeSource.from_declaration(declaration()); state = model(owner)
    b = batch(state, owner)
    with pytest.raises(ValueError):
        train_model(state, owner, replace(b, inputs={**b.inputs, "cpu": {"workers": 2, "rows_per_tile": 1}}))
    assert state.step == 0
