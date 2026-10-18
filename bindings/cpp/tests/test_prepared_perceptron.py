"""Owned sparse observations, exact reuse and portable input records."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken, ExecutionCancelled, ExecutionLimitError,
    NativePerceptron, NativePerceptronRows, NativeSource,
)
from test_construction import ROWS, batch, declaration, model

from rexgraph.cpp_perceptron import cpp_perceptron_input
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value, unpack_value


@pytest.mark.parametrize("workers", [None, 1, 2, 4, 64])
def test_owned_rows_preserve_order_zero_and_integer_oracle(workers):
    native = NativePerceptron((7, -9, 11), -5)
    rows = [[], [[0, 2], [1, -1]], [[2, 0], [1, 3], [1, -2]], [[0, 1], [2, -1], [0, -1]]]
    expected = tuple(-5 + sum((7, -9, 11)[i] * v for i, v in row) for row in rows)
    owned = native.prepare_rows(rows)
    assert owned.shape == (4, 3) and owned.entries == 8
    kwargs = {} if workers is None else {"cpu_workers": workers, "rows_per_tile": 1}
    ordinary = native.forward(rows, **kwargs)
    rows[1][0][1] = 999; rows.clear()
    assert native.forward(owned, **kwargs) == ordinary
    assert ordinary["scores"] == expected
    with pytest.raises(AttributeError): owned.entries = 0
    with pytest.raises(TypeError): owned.__init__()
    assert native.forward(owned, **kwargs) == ordinary


def test_owner_survives_model_and_applies_current_parameters():
    native = NativePerceptron((0, 0))
    owned = native.prepare_rows((((0, 1),), ((1, 1),)))
    child = native.advance((((0, 1),), ((1, 1),)), (1, -1))
    expected = child.forward((((0, 1),), ((1, 1),)))
    del native
    assert child.forward(owned) == expected
    assert NativePerceptron((3, 7), 5).forward(owned)["scores"] == (8, 12)
    with pytest.raises(ValueError): NativePerceptron((3,)).forward(owned)


@pytest.mark.parametrize("rows", [(), (((1, 0),),), (((-1, 0),),), (((True, 0),),),
                                  (((0, 1.0),),), (((0, Fraction(1, 2)),),),
                                  (((0, True),),), (((0, 1 << 63),),), (((0,),),)])
def test_preparation_rejects_invalid_rows(rows):
    with pytest.raises((ValueError, TypeError, OverflowError, IndexError)):
        NativePerceptron((1,)).prepare_rows(rows)


def test_uninitialized_handles_reject_and_cannot_be_subclassed():
    with pytest.raises(TypeError): NativePerceptronRows()
    empty = NativePerceptronRows.__new__(NativePerceptronRows)
    with pytest.raises(RuntimeError): _ = empty.shape
    with pytest.raises(RuntimeError): _ = empty.entries
    with pytest.raises(RuntimeError): NativePerceptron((1,)).forward(empty)
    with pytest.raises(TypeError): type("MutableRows", (NativePerceptronRows,), {})


def test_cancellation_and_limits_are_rechecked_on_every_use():
    native = NativePerceptron((1,))
    rows = ((), (), ())
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): native.prepare_rows(rows, cancellation=token)
    owned = native.prepare_rows(rows)
    for kwargs in ({}, {"cpu_workers": 2, "rows_per_tile": 1}):
        with pytest.raises(ExecutionCancelled): native.forward(owned, cancellation=token, **kwargs)
        assert native.forward(owned, **kwargs)["scores"] == (0, 0, 0)
    limited = NativePerceptron((1,), limits=(3, 100, 100, 100, 64))
    with pytest.raises(ExecutionLimitError): limited.forward(owned)
    cpu_limited = NativePerceptron((1,), limits=(5, 100, 100, 100, 64))
    assert cpu_limited.forward(owned)["scores"] == (0, 0, 0)
    with pytest.raises(ExecutionLimitError): cpu_limited.forward(owned, cpu_workers=2, rows_per_tile=1)
    with pytest.raises(ExecutionLimitError): NativePerceptron((1,), limits=(100, 100, 1, 100, 64)).prepare_rows((((0, 1),),))


@pytest.mark.parametrize("workers", [None, 1, 4])
def test_ordered_overflow_and_concurrent_reuse(workers):
    native = NativePerceptron((1,))
    maximum = (1 << 63) - 1
    valid = native.prepare_rows((((0, maximum), (0, -1), (0, 1)),))
    invalid = native.prepare_rows(((), ((0, maximum), (0, 1), (0, -1))))
    kwargs = {} if workers is None else {"cpu_workers": workers, "rows_per_tile": 1}
    with pytest.raises(OverflowError): native.forward(invalid, **kwargs)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: native.forward(valid, **kwargs), range(9)))
    assert all(v["scores"] == (maximum,) for v in results)
    assert native.snapshot == {"weights": (1,), "bias": 0, "mistakes": 0}


@pytest.mark.parametrize("storage", ["values", "packed"])
@pytest.mark.parametrize("workers", [None, 2])
def test_core_preparation_matches_default_training_and_portable_routes(storage, workers, monkeypatch):
    import rexgraph.cpp_perceptron as module
    from rcql import Executor, parse
    from system.serialize import json_value
    source = NativeSource.from_declaration(declaration())
    state = model(source, parameter_storage=storage)
    child = train_model(state, source, batch(state, source), steps=4)
    ordinary = cpp_perceptron_input(state, source, ROWS, cpu_workers=workers)
    prepared = cpp_perceptron_input(state, source, ROWS, prepare=True, cpu_workers=workers)
    assert prepared == ordinary
    expected = infer_model(child, source, ordinary)
    original = module._rows
    def reject(*args):
        raise AssertionError("captured immutable rows were revalidated")
    monkeypatch.setattr(module, "_rows", reject)
    output = infer_model(child, source, prepared)
    assert output.coefficient_digest == expected.coefficient_digest
    record = model_record(source, child)
    result = Executor(sources={"record": record}, params={"input": prepared}).execute(
        parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
    assert result.values[0].coefficient_digest == output.coefficient_digest
    assert json_value(output) == json_value(expected)
    monkeypatch.setattr(module, "_rows", original)
    restored, owner = read_model_record(record)
    portable = unpack_value(pack_value(thaw_tree(prepared)))
    assert portable == ordinary
    assert infer_model(restored, owner, portable).coefficient_digest == output.coefficient_digest
    assert np.array_equal(output.values, expected.values)


def test_cache_eviction_restores_ordinary_validation(monkeypatch):
    import rexgraph.cpp_perceptron as module
    source = NativeSource.from_declaration(declaration()); state = model(source)
    inputs = [cpp_perceptron_input(state, source, ROWS, prepare=True) for _ in range(9)]
    assert len(module._input_cache) == 8
    original = module._rows; calls = []
    def counting(*args):
        calls.append(1); return original(*args)
    monkeypatch.setattr(module, "_rows", counting)
    last = infer_model(state, source, inputs[-1])
    assert calls == []
    first = infer_model(state, source, inputs[0])
    assert len(calls) == 1 and first.coefficient_digest == last.coefficient_digest


@pytest.mark.parametrize("name", ["source", "primary_space", "feature_carrier", "feature_coordinates"])
def test_cached_rows_do_not_bypass_source_or_carrier_validation(name):
    source = NativeSource.from_declaration(declaration()); state = model(source)
    prepared = cpp_perceptron_input(state, source, ROWS, prepare=True)
    forged = dict(prepared)
    forged[name] = ({} if name == "source" else (state.space.name, tuple(reversed(state.space.keys)))
                    if name == "primary_space" else 9999)
    with pytest.raises(ValueError): infer_model(state, source, forged)


def test_cached_rows_do_not_bypass_parameter_seal():
    source = NativeSource.from_declaration(declaration()); state = model(source, parameter_storage="packed")
    prepared = cpp_perceptron_input(state, source, ROWS, prepare=True)
    state.payload["parameters"].flags.writeable = True
    state.payload["parameters"][0] += 1
    with pytest.raises(ValueError, match="changed after construction"): infer_model(state, source, prepared)


@pytest.mark.parametrize("prepare", [None, 1, "yes"])
def test_preparation_requires_explicit_boolean(prepare):
    source = NativeSource.from_declaration(declaration()); state = model(source)
    with pytest.raises(TypeError): cpp_perceptron_input(state, source, ROWS, prepare=prepare)
