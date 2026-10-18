"""Packed integer batch execution, exact references and application records."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeIntegerBatch,
    NativeSource,
)
from test_construction import declaration
from test_transport import MAPS, family, permute

from rexgraph.cpp_learning import cpp_integer_batch_input, cpp_integer_model
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, transport_model
from rexgraph.model_state import thaw_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value


def layers(activation="relu"):
    return (dict(channels=("identity", "hodge_down", "hodge_up", "hodge", "hodge_down"),
                 gains=(2, -1, 1, 1, 0), bias=3, activation=activation),
            dict(channels=("identity",), gains=(2,), bias=-1, activation="identity"))


def values(n=3, b=8):
    return (np.arange(n*b, dtype=np.int64) % 7-3).reshape(n, b)


def sample_space(b=8):
    return CoordinateSpace("observations", tuple(f"row-{i}" for i in range(b)))


@pytest.mark.parametrize("grade", [0, 1, 2])
@pytest.mark.parametrize("activation", ["identity", "relu", "sign"])
@pytest.mark.parametrize("b", [1, 2, 8])
def test_native_batch_matches_independent_gmp_and_scalar(grade, activation, b, native_fixtures):
    source = NativeSource.from_declaration(declaration())
    n = len(source.identity["coordinates"][grade])
    array = values(n, b)
    native = source.integer_batch_input(10+grade, array, grade=grade)
    plan = source.integer_network(10+grade, layers(activation), grade=grade)
    result = plan.forward_batch(native)
    output = np.asarray(result["values"])
    oracle = next(row for row in json.loads((native_fixtures[0]/"integer-batch.json").read_text())
                  if row["grade"] == grade and row["activation"] == {"identity": 0, "relu": 1, "sign": 2}[activation]
                  and row["batch"] == b)
    assert tuple(output.flat) == tuple(int(v) for v in oracle["values"])
    for j in range(b):
        assert tuple(output[:, j]) == plan.forward(source.integer_input(10+grade, tuple(array[:, j]), grade=grade))["values"]
    assert output.dtype == np.int64 and not output.flags.writeable
    assert result["batch_size"] == b and result["grade"] == grade
    assert result["layout"] == "cell-major-samples-contiguous-v1"
    array[:] = 100
    assert np.asarray(native.values)[0, 0] == -3
    with pytest.raises(TypeError): result["values"][0, 0] = 9


def test_workspace_reuse_growth_and_failure_recovery():
    source = NativeSource.from_declaration(declaration())
    plan = source.integer_network(11, layers())
    assert plan is source.integer_network(11, layers())
    for b, expected in ((2, 2), (8, 8), (1, 8), (8, 8)):
        result = plan.forward_batch(source.integer_batch_input(11, values(b=b)))
        assert result["workspace_capacity"] == expected
        assert result["workspace_slots"] == (5*3+3+2)*expected
    overflowing = np.full((3, 2), (1 << 63)-1, dtype=np.int64)
    with pytest.raises(OverflowError): plan.forward_batch(source.integer_batch_input(11, overflowing))
    restored = plan.forward_batch(source.integer_batch_input(11, values(b=2)))
    assert tuple(np.asarray(restored["values"])[:, 0]) == plan.forward(source.integer_input(11, tuple(values(b=2)[:, 0])))["values"]
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): plan.forward_batch(source.integer_batch_input(11, values()), cancellation=token)
    assert plan.forward_batch(source.integer_batch_input(11, values()))["batch_size"] == 8
    other = NativeSource(source.archive)
    with pytest.raises(ValueError): plan.forward_batch(other.integer_batch_input(11, values()))
    del source, other
    assert np.asarray(restored["values"]).shape == (3, 2)


@pytest.mark.parametrize("change", ["float", "bool", "unsigned", "small", "endian", "object", "strided", "empty", "rows", "axis", "source", "extra"])
def test_core_batch_rejects_wrong_storage_and_coordinates(change):
    source = NativeSource.from_declaration(declaration())
    state = cpp_integer_model(source, 11, layers())
    record = thaw_tree(cpp_integer_batch_input(state, source, values(), sample_space()))
    if change in {"float", "bool", "unsigned", "small", "endian", "object"}:
        dtype = {"float": "float64", "bool": "bool", "unsigned": "uint64", "small": "int32", "endian": ">i8", "object": object}[change]
        record["values"] = record["values"].astype(dtype)
    elif change == "strided": record["values"] = record["values"][:, ::2]
    elif change == "empty": record["values"] = np.empty((3, 0), dtype=np.int64); record["samples"]["keys"] = ()
    elif change == "rows": record["values"] = np.zeros((2, 8), dtype=np.int64)
    elif change == "axis": record["samples"]["keys"] = ("duplicate",)*8
    elif change == "source": record["source"]["record_id"] = "another"
    else: record["extra"] = 1
    with pytest.raises((ValueError, TypeError)): infer_model(state, source, record)


def test_native_limits_and_uninitialized_handles():
    source = NativeSource.from_declaration(declaration())
    with pytest.raises(ValueError): source.integer_batch_input(11, values(), limits=(1, 1, 1, 1, 64))
    with pytest.raises(ExecutionLimitError): source.integer_batch_input(11, values(), limits=(100, 100, 100, 100, 63))
    plan = source.integer_network(11, layers(), limits=(1000, 100, 100, 100000, 64))
    with pytest.raises(ExecutionLimitError): plan.forward_batch(source.integer_batch_input(11, values()))
    assert plan.forward_batch(source.integer_batch_input(11, values(b=1)))["batch_size"] == 1
    with pytest.raises(ValueError): source.integer_batch_input(40, np.ones((2, 2), dtype=np.int64))
    with pytest.raises(ValueError): source.integer_batch_input(11, np.ones((2, 2), dtype=np.int64))
    empty = NativeIntegerBatch.__new__(NativeIntegerBatch)
    with pytest.raises(RuntimeError): _ = empty.values
    with pytest.raises(RuntimeError): plan.forward_batch(empty)


def test_concurrent_owned_batch_calls_and_detached_outputs():
    source = NativeSource.from_declaration(declaration())
    plan = source.integer_network(11, layers())
    inputs = [source.integer_batch_input(11, values(b=b)) for b in (1, 2, 8, 4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(plan.forward_batch, inputs))
    for batch, output in zip(inputs, outputs, strict=True):
        assert np.asarray(output["values"]).shape == batch.shape
        for j in range(batch.shape[1]):
            assert tuple(np.asarray(output["values"])[:, j]) == plan.forward(source.integer_input(11, tuple(np.asarray(batch.values)[:, j])))["values"]
    del source, plan, inputs
    assert np.asarray(outputs[-1]["values"]).shape == (3, 4)


@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_core_output_and_model_roundtrip(suffix, tmp_path):
    source = NativeSource.from_declaration(declaration())
    state = cpp_integer_model(source, 11, layers())
    record = cpp_integer_batch_input(state, source, values(b=2), sample_space(2))
    output = infer_model(state, source, record)
    assert output.values.dtype == np.int64 and output.arithmetic == "rational"
    assert output.axes == (sample_space(2),)
    assert output.tensor().axes == output.axes
    restored_output = unpack_model(pack_model(output))
    assert restored_output.coefficient_digest == output.coefficient_digest
    path = tmp_path/f"model.{suffix}"
    envelope = model_record(source, state)
    envelope.attach_metadata(1, 0, "packed_batch", thaw_tree(record))
    envelope.attach_metadata(1, 0, "batch_output", output)
    save_model(path, envelope)
    loaded = load_model(path)
    restored, data = read_model_record(loaded)
    assert loaded.get_metadata(1, 0, "batch_output").coefficient_digest == output.coefficient_digest
    assert infer_model(restored, data, unpack_value(pack_value(thaw_tree(record)))).coefficient_digest == output.coefficient_digest
    infer_model(state, source, cpp_integer_batch_input(state, source, values(), sample_space()))
    assert infer_model(state, source, record).coefficient_digest == output.coefficient_digest


def test_transported_model_preserves_batch_samples():
    source, state, target, mapping = family("cpp_integer")
    child = transport_model(state, source, CppSourceReference(target), mapping)
    data = values(b=2)
    a = infer_model(state, source, cpp_integer_batch_input(state, source, data, sample_space(2)))
    b = infer_model(child, target, cpp_integer_batch_input(child, target, np.array(permute(tuple(data), MAPS[1])), sample_space(2)))
    assert tuple(map(tuple, b.values)) == permute(tuple(map(tuple, a.values)), MAPS[1])
    assert b.axes == a.axes and b.execution_receipt["transport_base_step"] == state.step


def test_rcql_agent_and_system_use_the_common_batch_route():
    from agent.hive import Hive
    from system.serialize import json_value

    from rcql import Executor, parse
    source = NativeSource.from_declaration(declaration())
    state = cpp_integer_model(source, 11, layers())
    inputs = cpp_integer_batch_input(state, source, values(b=2), sample_space(2))
    expected = infer_model(state, source, inputs)
    query = Executor(sources={"record": model_record(source, state)}, params={"batch": inputs})
    actual = query.execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$batch)')).values[0]
    assert actual.coefficient_digest == expected.coefficient_digest
    with patch("agent.activity.record"), patch("agent.activity.open_use", return_value=0), patch("agent.activity.close_use"):
        hive = Hive("private-batch")
        hive.add_worker("batch-model", lambda value, **kw: infer_model(state, source, value), capability="predict")
        assert hive.invoke("batch-model", inputs, record=False).coefficient_digest == expected.coefficient_digest
    preview = json_value(actual)
    assert preview["execution_receipt"]["batch_size"] == 2
    assert preview["axes"] == [{"name": "observations", "size": 2}]


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_fresh_process_batch_model_and_input_restore(backend, tmp_path):
    from test_models import NO_TORCH

    from rcdb import open_store
    source = NativeSource.from_declaration(declaration())
    state = cpp_integer_model(source, 11, layers())
    inputs = cpp_integer_batch_input(state, source, values(b=2), sample_space(2))
    expected = infer_model(state, source, inputs)
    record = model_record(source, state)
    record.attach_metadata(1, 0, "packed_batch", thaw_tree(inputs))
    record.attach_metadata(1, 0, "batch_output", expected)
    url = {"file": "file://"+str(tmp_path/"file"), "rex": "rex://"+str(tmp_path/"rex"),
           "sqlite": "sqlite:///"+str(tmp_path/"records.sqlite")}[backend]
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    store.commit_mutation("batched", record, expected_version=0)
    assert store.verify_commits("batched"); store.close()
    script = NO_TORCH+'''
import numpy as np
from rcdb import open_store
from rcql import Executor,parse
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
store=open_store(sys.argv[1])
record=store.read_record("batched",version=1).value
state,source=read_model_record(record)
inputs=record.get_metadata(1,0,"packed_batch")
output=infer_model(state,source,inputs)
assert output.values.dtype==np.int64 and not output.values.flags.writeable
assert record.get_metadata(1,0,"batch_output").coefficient_digest==output.coefficient_digest
result=Executor(sources={"record":record},params={"input":inputs}).execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
assert result.values[0].coefficient_digest==output.coefficient_digest
assert not any(name=="torch" or name.startswith("torch.") for name in sys.modules)
print(output.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected.coefficient_digest


def test_packed_output_width_and_legacy_exact_storage():
    source = NativeSource.from_declaration(declaration())
    state = cpp_integer_model(source, 11, layers())
    output = infer_model(state, source, cpp_integer_batch_input(state, source, values(b=2), sample_space(2)))
    for dtype in (np.float64, np.uint64, np.int32, object, ">i8"):
        with pytest.raises(TypeError): replace(output, values=output.values.astype(dtype))
    legacy = replace(output, method="legacy-exact")
    assert legacy.values.dtype == object and all(isinstance(v, Fraction) for v in legacy.values.flat)
    extremes = np.array([[-(1 << 63), (1 << 63)-1]]*3, dtype=np.int64)
    packed = replace(output, values=extremes)
    restored = unpack_model(pack_model(packed))
    assert restored.coefficient_digest == packed.coefficient_digest
    assert restored.values[0, 0] == -(1 << 63) and restored.values[0, 1] == (1 << 63)-1
