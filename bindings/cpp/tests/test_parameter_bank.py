"""Packed exact parameter storage, native admission and checkpoint replay."""
from dataclasses import replace
from fractions import Fraction
import gc

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken, ExecutionCancelled, ExecutionLimitError, NativePerceptron, NativeSource,
)
from test_construction import LABELS, ROWS, availability, batch, declaration, model

from rexgraph.cpp_perceptron import cpp_perceptron_input, prepare_perceptron
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.source_binding import CppSourceReference


def source():
    return NativeSource.from_declaration(declaration())


def test_native_packed_snapshot_is_owned_readonly_and_exact():
    owner = source()
    native = owner.perceptron(4000, 4001)
    view = native.packed_snapshot["parameters"]
    assert view.format == "q" and view.readonly and view.shape == (3,)
    assert tuple(view) == (1, -2, 3)
    with pytest.raises(TypeError): view[0] = 0
    child = native.advance(ROWS, LABELS, rate=2)
    snapshot = child.snapshot
    assert tuple(child.packed_snapshot["parameters"]) == (*snapshot["weights"], snapshot["bias"])
    assert tuple(view) == (1, -2, 3)
    del owner, native, child
    gc.collect()
    assert tuple(view) == (1, -2, 3)


def test_packed_initialization_avoids_python_parameter_conversion(monkeypatch):
    import rexgraph.cpp_perceptron as module
    owner = source()
    def reject(value):
        raise AssertionError("parameter converted in Python")
    monkeypatch.setattr(module, "_int64", reject)
    state = model(owner, parameter_storage="packed")
    assert state.payload["parameters"].dtype == np.int64
    assert state.payload["parameters"].nbytes == 24
    assert not state.payload["parameters"].flags.writeable
    assert tuple(state.payload["parameters"]) == (1, -2, 3)
    with pytest.raises(AssertionError): model(owner)


@pytest.mark.parametrize("values", [(0, 0, 0), (-(1 << 63), (1 << 63)-1, 0),
                                    ((1 << 53)+1, -((1 << 53)+3), -(1 << 63))])
def test_source_loader_preserves_supported_zero_and_int64_extremes(values):
    d = declaration()
    d["fields"][-2]["entries"] = tuple(enumerate(values[:2]))
    d["fields"][-1]["entries"] = ((0, values[-1]),)
    owner = NativeSource.from_declaration(d)
    assert tuple(owner.perceptron(4000, 4001).packed_snapshot["parameters"]) == values
    state = model(owner, parameter_storage="packed")
    restored, _ = read_model_record(model_record(owner, state))
    assert np.array_equal(restored.payload["parameters"], values)


@pytest.mark.parametrize("mode", ["fraction", "overflow", "underflow", "partial", "bias_fraction",
                                 "bias_overflow", "source_grade", "missing"])
def test_native_source_loader_rejects_invalid_parameter_fields(mode):
    d = declaration()
    if mode == "fraction": d["fields"][-2]["entries"] = ((0, Fraction(1, 2)),)
    elif mode == "overflow": d["fields"][-2]["entries"] = ((0, 1 << 63),)
    elif mode == "underflow": d["fields"][-2]["entries"] = ((0, -(1 << 63)-1),)
    elif mode == "partial":
        d["fields"][-2].update(entries=((0, 1),), support=(0,))
    elif mode == "bias_fraction": d["fields"][-1]["entries"] = ((0, Fraction(1, 2)),)
    elif mode == "bias_overflow": d["fields"][-1]["entries"] = ((0, 1 << 63),)
    owner = NativeSource.from_declaration(d)
    key = 1001 if mode == "source_grade" else 99999 if mode == "missing" else 4000
    with pytest.raises((ValueError, OverflowError, IndexError, RuntimeError)):
        owner.perceptron(key, 4001)


@pytest.mark.parametrize("limits", [(1, 10, 10, 100, 64), (10, 10, 1, 100, 64),
                                   (10, 10, 10, 1, 64), (10, 10, 10, 100, 63)])
def test_native_source_loader_resource_admission(limits):
    with pytest.raises(ExecutionLimitError): source().perceptron(4000, 4001, limits=limits)


def test_native_loader_cancellation_and_uninitialized_handles():
    owner = source()
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): owner.perceptron(4000, 4001, cancellation=token)
    for value in (NativePerceptron.__new__(NativePerceptron),):
        with pytest.raises(RuntimeError): _ = value.packed_snapshot
    empty = NativeSource.__new__(NativeSource)
    with pytest.raises(RuntimeError): empty.perceptron(4000, 4001)


@pytest.mark.parametrize("mode", ["values", "dtype", "shape", "tuple", "mistakes", "history", "extra", "storage"])
def test_packed_replay_rejects_forged_banks(mode):
    owner = source()
    state = model(owner, parameter_storage="packed")
    state = train_model(state, owner, batch(state, owner), steps=2)
    payload, config = dict(state.payload), dict(state.configuration)
    if mode == "values": payload["parameters"] = np.zeros(3, dtype=np.int64)
    elif mode == "dtype": payload["parameters"] = state.payload["parameters"].astype(np.int32)
    elif mode == "shape": payload["parameters"] = state.payload["parameters"].reshape(1, 3)
    elif mode == "tuple": payload["parameters"] = tuple(state.payload["parameters"])
    elif mode == "mistakes": payload["mistakes"] = 999
    elif mode == "history": payload["history"] = payload["history"][:1]
    elif mode == "extra": payload["extra"] = True
    elif mode == "storage": config["parameter_storage"] = "inferred"
    changed = replace(state, payload=payload, configuration=config)
    with pytest.raises(ValueError): read_model_record(model_record(owner, changed))


def test_packed_training_matches_tuple_storage_and_cold_replay():
    import rexgraph.cpp_perceptron as module
    owner = source()
    old, packed = model(owner), model(owner, parameter_storage="packed")
    parent_parameters = packed.payload["parameters"].copy()
    for _ in range(4):
        old = train_model(old, owner, batch(old, owner))
        packed = train_model(packed, owner, batch(packed, owner))
        assert tuple(packed.payload["parameters"]) == old.payload["parameters"]
        assert packed.payload["history"] == old.payload["history"]
        assert packed.payload["mistakes"] == old.payload["mistakes"]
    module._cache.clear()
    assert tuple(prepare_perceptron(packed, owner).packed_snapshot["parameters"]) == old.payload["parameters"]
    output = infer_model(packed, owner, cpp_perceptron_input(packed, owner, ROWS))
    assert output.execution_receipt["parameter_storage"] == "packed"
    assert np.array_equal(parent_parameters, (1, -2, 3))


@pytest.mark.parametrize("mode", ["dtype", "dimensions", "metadata", "byte_bound"])
def test_packed_transport_proof_tensor_admission(monkeypatch, mode):
    import rexgraph.cpp_transport as module
    from rexgraph.model_codec import pack_model
    owner = source()
    tensors = pack_model(model(owner, parameter_storage="packed"))
    if mode == "dtype": tensors["tensor/0"] = tensors["tensor/0"].astype(np.float64)
    elif mode == "dimensions": tensors["tensor/0"] = tensors["tensor/0"].reshape(1, 3)
    elif mode == "metadata": tensors["spec"] = tensors["spec"].astype(np.int64)
    elif mode == "byte_bound": monkeypatch.setattr(module, "_proof_limit", sum(v.nbytes for v in tensors.values())-1)
    with pytest.raises(ValueError): module._proof_bytes(tensors, b"")


def test_packed_system_preview_keeps_exact_bounded_values():
    from system.serialize import json_value
    d = declaration()
    d["fields"][-2]["entries"] = ((0, (1 << 53)+1), (1, -2))
    owner = NativeSource.from_declaration(d)
    state = model(owner, parameter_storage="packed")
    preview = json_value(state, max_values=2)
    assert preview["parameter_count"] == 3 and preview["parameters_truncated"]
    assert preview["parameters"] == [{"kind": "Integer", "decimal": str((1 << 53)+1)}, -2]


def test_rcql_creates_and_trains_packed_profile():
    from rcql import Executor, parse
    owner = source()
    initial = model(owner, parameter_storage="packed")
    record = model_record(owner, initial)
    config = dict(carrier=11, weights_key=4000, bias_key=4001, availability=availability(),
                  feature_coordinates=900, time_coordinates=999, parameter_storage="packed")
    created = Executor(sources={"record": record}, params={"config": config}).execute(parse(
        'FROM $record RETURN MODEL_INIT("cpp_perceptron",$config)')).values[0]
    assert created.coefficient_digest == initial.coefficient_digest
    b = batch(created, owner)
    trained = Executor(sources={"record": record}, params={"model": created, "batch": b}).execute(parse(
        'FROM $record RETURN MODEL_TRAIN($model,$batch,2)')).values[0]
    assert trained.coefficient_digest == train_model(initial, owner, b, steps=2).coefficient_digest


@pytest.mark.parametrize("destination_storage", ["packed", "values"])
def test_packed_parameter_transport_and_destination_resume(destination_storage):
    from test_transport import MAPS, links, permute, relabel, rows_after, tensor_links
    from rexgraph.cpp_transport import cpp_transport
    from rexgraph.cpp_perceptron import cpp_perceptron_batch
    owner = source()
    initial = model(owner, parameter_storage="packed")
    trained = train_model(initial, owner, batch(initial, owner), steps=3)
    target = NativeSource.from_declaration(relabel(declaration(), trained.payload["parameters"]))
    config = dict(carrier=111, weights_key=5000, bias_key=5001,
                  availability=permute(trained.configuration["availability"], MAPS[1]),
                  feature_coordinates=1900, time_coordinates=999, initial_parameter_knowledge_time=20,
                  parameter_storage=destination_storage)
    mapping = cpp_transport(owner, target, MAPS, links(owner), tensor_links(owner), configuration=config)
    if destination_storage == "values":
        with pytest.raises(ValueError): transport_model(trained, owner, CppSourceReference(target), mapping)
        return
    child = transport_model(trained, owner, CppSourceReference(target), mapping)
    restored, destination = read_model_record(model_record(target, child))
    a = infer_model(trained, owner, cpp_perceptron_input(trained, owner, ROWS))
    b = infer_model(restored, destination, cpp_perceptron_input(restored, destination, rows_after()))
    assert tuple(b.values) == permute(tuple(a.values), MAPS[1])
    data = cpp_perceptron_batch(restored, destination, rows_after(), permute(LABELS, MAPS[1]),
                                rate=2, event_cutoff=10, knowledge_cutoff=20)
    advanced = train_model(restored, destination, data)
    resumed, _ = read_model_record(model_record(destination, advanced))
    assert resumed.coefficient_digest == advanced.coefficient_digest
