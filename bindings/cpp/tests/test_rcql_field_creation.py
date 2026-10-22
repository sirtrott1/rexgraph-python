"""RCQL forwards explicit field actions through the provider creation dispatcher."""
from fractions import Fraction as Q

import numpy as np
import pytest

pytest.importorskip("rcql")
from rcql import Executor, parse
from rexgraph.cpp_model import cpp_model, cpp_rational_batch_input
from rexgraph.io.model_record import model_record
from rexgraph.model_runtime import infer_model
from rexgraph.type_accession import CoordinateSpace
from rexgraph_cpp import NativeSource
from test_construction import declaration


@pytest.mark.parametrize("operation", ["hodge", "boundary", "hodge_up"])
def test_model_init_forwards_native_action_and_replays_whole_batch(operation):
    source = NativeSource.from_declaration(declaration())
    initial = cpp_model(source, (10, 11, 12))
    inputs = cpp_rational_batch_input(initial, source, (1001,), CoordinateSpace("samples", ("one",)))
    record = model_record(source, initial)
    config = {"grade_carriers": (10, 11, 12), "operation": operation}
    result = Executor(sources={"record": record}, params={"config": config, "inputs": inputs}).execute(parse(
        'FROM $record LET m=MODEL_INIT("cpp_field",$config) '
        'RETURN m,MODEL_INFER(m,$inputs),MODEL_RECORD(m)'))
    created, actual, saved = result.values
    expected_state = cpp_model(source, (10, 11, 12), operation=operation)
    expected = infer_model(expected_state, source, inputs)
    assert created.coefficient_digest == expected_state.coefficient_digest
    assert actual.space == expected.space and actual.axes == expected.axes
    assert np.array_equal(actual.values, expected.values)
    assert actual.execution_receipt["operation"] == operation
    assert actual.execution_receipt["engine"] == expected.execution_receipt["engine"]
    # Restore from a record and resolve the stored fields through its actual new owner.
    replay = Executor(sources={"record": saved}, params={"inputs": inputs}).execute(parse(
        'FROM $record RETURN MODEL_INFER(MODEL_STATE(),$inputs)')).values[0]
    assert replay.coefficient_digest == actual.coefficient_digest


def test_model_init_preserves_explicit_global_action_and_parameter():
    source = NativeSource.from_declaration(declaration())
    record = model_record(source, cpp_model(source, (10, 11, 12)))
    config = {"grade_carriers": (10, 11, 12), "operation": "green",
              "profile": "global", "parameter": Q(1, 2)}
    result = Executor(sources={"record": record}, params={"config": config}).execute(parse(
        'FROM $record LET m=MODEL_INIT("cpp_field",$config) RETURN m,MODEL_RECORD(m)'))
    expected = cpp_model(source, (10, 11, 12), operation="green", profile="global", parameter=Q(1, 2))
    assert result.values[0].coefficient_digest == expected.coefficient_digest
    restored = Executor(sources={"record": result.values[1]}).execute(parse(
        'FROM $record RETURN MODEL_STATE()')).values[0]
    assert restored.coefficient_digest == expected.coefficient_digest
