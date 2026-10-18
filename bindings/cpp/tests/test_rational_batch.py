"""Finite rational field batches, fitted channels and exact application records."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from unittest.mock import patch

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeField,
    NativeSource,
    NativeTrainer,
    PreparedField,
)
from test_construction import declaration
from test_learning import declaration as training_declaration
from test_models import NO_TORCH
from test_transport import MAPS, family

from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch, prepare_learning_model
from rexgraph.cpp_model import cpp_model, cpp_rational_batch_input
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.model_state import thaw_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_value

OPERATIONS = {0: "boundary", 1: "adjoint", 2: "hodge_down", 3: "hodge_up", 4: "hodge"}
CASES = [(kind, grade) for kind in OPERATIONS for grade in range(3) if grade or kind not in (0, 1)]


def axis(b):
    return CoordinateSpace("observations", tuple(f"sample-{s}" for s in range(b)))


def keys_for(kind, grade, b):
    k = grade-1 if kind == 1 else grade
    return tuple(3003 if k == 1 and s % 3 == 1 else 1000+k for s in range(b))


@pytest.mark.parametrize("metric", ["unit", "diagonal"])
@pytest.mark.parametrize("kind,grade", CASES)
@pytest.mark.parametrize("b", [1, 3, 8])
def test_field_batch_matches_scalar_and_independent_reference(metric, kind, grade, b, native_fixtures):
    root = native_fixtures[0]
    source = NativeSource((root/"diagonal.rfcxs").read_bytes())
    options = dict(metric_selection=metric, metric_keys=(2000, 2001, 2002) if metric == "diagonal" else ())
    plan = source.prepare((10, 11, 12), **options)
    keys = keys_for(kind, grade, b)
    result = plan.execute_batch(tuple(source.field(key) for key in keys), operation=OPERATIONS[kind], grade=grade)
    values = np.asarray(result["values"], dtype=object)
    oracle = {row["input"]: tuple(map(Q, row["values"]))
              for row in json.loads((root/"rational-batch.json").read_text())
              if row["metric"] == metric and row["kind"] == kind and row["grade"] == grade}
    for s, key in enumerate(keys):
        assert tuple(values[:, s]) == oracle[key]
        assert tuple(values[:, s]) == plan.execute(source.field(key), operation=OPERATIONS[kind], grade=grade)["values"]
    assert result["input_fields"] == keys and result["input_field"] == 0
    assert result["batch_size"] == b and result["namespace"] == 88
    assert result["grade"] == (grade-1 if kind == 0 else grade)
    assert result["layout"] == "cell-major-samples-contiguous-v1"
    assert all(isinstance(v, Q) for v in values.flat)
    state = cpp_model(source, (10, 11, 12), grade=grade, operation=OPERATIONS[kind], **options)
    record = cpp_rational_batch_input(state, source, keys, axis(b))
    output = infer_model(state, source, record)
    assert np.array_equal(output.values, values)
    assert output.axes == (axis(b),) and output.variance == "chain"
    assert output.space.keys == source.identity["coordinates"][output.grade]
    assert output.values.dtype == object and not output.values.flags.writeable
    assert output.execution_receipt["input_fields"] == keys
    assert "workspace_capacity" not in output.execution_receipt


def test_workspace_reuse_grade_replacement_concurrency_and_detached_values():
    source = NativeSource.from_declaration(declaration())
    plan = source.prepare((10, 11, 12))
    fields = (source.field(1001),)
    for b, capacity in ((2, 2), (8, 8), (1, 8), (3, 8)):
        result = plan.execute_batch(fields*b)
        assert result["workspace_capacity"] == capacity
        assert result["workspace_slots"] == 17*capacity
    replaced = plan.execute_batch((source.field(1002),)*2, grade=2)
    assert replaced["workspace_capacity"] == 2 and replaced["workspace_slots"] == 12*2
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(plan.execute_batch, [fields*b for b in (1, 4, 8, 2)]))
    expected = plan.execute(fields[0])["values"]
    for output in outputs:
        assert all(tuple(row[s] for row in output["values"]) == expected for s in range(output["batch_size"]))
    del source, plan, fields
    assert outputs[-1]["values"] == tuple((value, value) for value in expected)


def test_boundary_capture_rejects_output_coordinate_axis_name():
    source = NativeSource.from_declaration(declaration())
    state = cpp_model(source, (10, 11, 12), grade=2, operation="boundary")
    samples = CoordinateSpace(f"cpp:{state.source.native.namespace}:grade:1:carrier:11", ("first",))
    with pytest.raises(ValueError): cpp_rational_batch_input(state, source, (1002,), samples)


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_metric"])
@pytest.mark.parametrize("mode", ["profile", "source", "axis", "same_axis", "count", "keys", "carrier", "support", "extra"])
def test_rejects_invalid_batch_records(adapter, mode):
    source, state, _, _ = family(adapter)
    record = thaw_tree(cpp_rational_batch_input(state, source, (1001, 1003), axis(2)))
    if mode == "profile": record["profile"] = "other"
    elif mode == "source": record["source"]["record_id"] = "foreign"
    elif mode == "axis": record["samples"]["keys"] = ("a", "a")
    elif mode == "same_axis": record["samples"]["name"] = state.space.name
    elif mode == "count": record["field_keys"] = (1001,)
    elif mode == "keys": record["field_keys"] = (1001, True)
    elif mode == "carrier": record["field_keys"] = (1001, 3001)
    elif mode == "support": record["field_keys"] = (1001, 3002)
    else: record["extra"] = 1
    with pytest.raises((ValueError, TypeError)): infer_model(state, source, record)


def test_native_rejections_cancellation_and_admission_recovery(native_source):
    plan = native_source.prepare((10, 11, 12))
    fields = (native_source.field(1001),)
    for bad in ((), (1,), (NativeField.__new__(NativeField),), (native_source.field(3002),),
                (NativeSource(native_source.archive).field(1001),)):
        with pytest.raises((TypeError, ValueError, RuntimeError)): plan.execute_batch(bad)
    for operation in ("green", "harmonic", "dirac"):
        with pytest.raises(ValueError): plan.execute_batch(fields, operation=operation)
    with pytest.raises(ValueError): plan.execute_batch(fields, parameter=Q(1, 2))
    with pytest.raises(TypeError): plan.execute_batch(fields, parameter=0.5)
    with pytest.raises(RuntimeError): PreparedField.__new__(PreparedField).execute_batch(fields)
    with pytest.raises(RuntimeError): NativeTrainer.__new__(NativeTrainer).forward_batch(fields)
    global_plan = native_source.prepare((10, 11, 12), profile="global")
    with pytest.raises(ValueError): global_plan.execute_batch(fields)
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): plan.execute_batch(fields*8, cancellation=token)
    assert plan.execute_batch(fields)["batch_size"] == 1
    bounded = native_source.prepare((10, 11, 12), limits=(1000, 140, 1000, 1000000, 1 << 24))
    assert bounded.execute_batch(fields)["workspace_capacity"] == 1
    with pytest.raises(ExecutionLimitError): bounded.execute_batch(fields*8)
    assert bounded.execute_batch(fields)["workspace_capacity"] == 1


@pytest.mark.parametrize("metric", ["unit", "diagonal"])
def test_fitted_metric_batches_retain_training_history(metric, native_fixtures):
    source = NativeSource((native_fixtures[0]/f"learning-{metric}.rfcxs").read_bytes())
    state = cpp_trainable_model(source, (10, 11, 12), training_declaration(), metric_selection=metric,
                               metric_keys=(2000, 2001, 2002) if metric == "diagonal" else ())
    fields = (source.field(1001), source.field(4002), source.field(1001))
    for step in (0, 3, 5):
        if step != state.step:
            batch = cpp_training_batch(state, source, rate=Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20)
            state = train_model(state, source, batch, steps=step-state.step)
        native = prepare_learning_model(state, source)
        before = native.snapshot
        record = cpp_rational_batch_input(state, source, (1001, 4002, 1001), axis(3))
        output = infer_model(state, source, record)
        for s, field in enumerate(fields):
            assert tuple(output.values[:, s]) == native.forward(field)["values"]
        assert output.execution_receipt["parameter_step"] == step
        assert output.execution_receipt["parameter_knowledge_time"] == native.parameter_knowledge_time
        assert native.snapshot == before
        token = CancellationToken(); token.cancel()
        with pytest.raises(ExecutionCancelled): native.forward_batch(fields, cancellation=token)
        with pytest.raises(ValueError): native.forward_batch((source.field(3002),))
        with pytest.raises(ValueError): native.forward_batch((NativeSource(source.archive).field(1001),))
        assert native.snapshot == before
        assert native.forward_batch(fields)["values"] == tuple(tuple(row) for row in output.values)


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_metric"])
@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_model_batch_output_roundtrip_and_cache_independent_seals(adapter, suffix, tmp_path):
    source, state, _, _ = family(adapter)
    inputs = cpp_rational_batch_input(state, source, (1001, 1003), axis(2))
    output = infer_model(state, source, inputs)
    assert unpack_model(pack_model(output)).coefficient_digest == output.coefficient_digest
    record = model_record(source, state)
    record.attach_metadata(1, 0, "batch_input", thaw_tree(inputs))
    record.attach_metadata(1, 0, "batch_output", output)
    path = tmp_path/f"rational.{suffix}"
    save_model(path, record)
    loaded = load_model(path)
    restored, data = read_model_record(loaded)
    captured = loaded.get_metadata(1, 0, "batch_input")
    assert infer_model(restored, data, captured).coefficient_digest == output.coefficient_digest
    assert loaded.get_metadata(1, 0, "batch_output").coefficient_digest == output.coefficient_digest
    infer_model(state, source, cpp_rational_batch_input(state, source, (1001,)*8, axis(8)))
    assert infer_model(state, source, unpack_value(pack_value(thaw_tree(inputs)))).coefficient_digest == output.coefficient_digest


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_metric"])
def test_transport_and_shared_application_routes(adapter):
    from agent.hive import Hive
    from system.serialize import json_value

    from rcql import Executor, parse

    source, state, target, mapping = family(adapter)
    child = transport_model(state, source, CppSourceReference(target), mapping)
    inputs = cpp_rational_batch_input(state, source, (1001, 1003), axis(2))
    target_inputs = cpp_rational_batch_input(child, target, (2001, 2003), axis(2))
    expected = infer_model(state, source, inputs)
    actual = infer_model(child, target, target_inputs)
    for i, j in enumerate(MAPS[1]):
        assert tuple(expected.values[i]) == tuple(actual.values[j])
    assert actual.axes == expected.axes
    query = Executor(sources={"record": model_record(source, state)}, params={"batch": inputs})
    result = query.execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$batch)')).values[0]
    assert result.coefficient_digest == expected.coefficient_digest
    with patch("agent.activity.record"), patch("agent.activity.open_use", return_value=0), patch("agent.activity.close_use"):
        hive = Hive("private-rational-batch")
        hive.add_worker("batch-model", lambda value, **kw: infer_model(state, source, value), capability="predict")
        assert hive.invoke("batch-model", inputs, record=False).coefficient_digest == expected.coefficient_digest
    preview = json_value(result)
    assert preview["execution_receipt"]["batch_size"] == 2
    assert preview["axes"] == [{"name": "observations", "size": 2}]


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_metric"])
@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_fresh_process_stored_fields_and_fitted_batch_restore(adapter, backend, tmp_path):
    from rcdb import open_store

    source, state, _, _ = family(adapter)
    inputs = cpp_rational_batch_input(state, source, (1001, 1003), axis(2))
    expected = infer_model(state, source, inputs)
    record = model_record(source, state)
    record.attach_metadata(1, 0, "batch_input", thaw_tree(inputs))
    record.attach_metadata(1, 0, "batch_output", expected)
    url = {"file": "file://"+str(tmp_path/"file"), "rex": "rex://"+str(tmp_path/"rex"),
           "sqlite": "sqlite:///"+str(tmp_path/"records.sqlite")}[backend]
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    store.commit_mutation("batched", record, expected_version=0)
    assert store.verify_commits("batched"); store.close()
    script = NO_TORCH+'''
from fractions import Fraction
from rcdb import open_store
from rcql import Executor,parse
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
store=open_store(sys.argv[1])
record=store.read_record("batched",version=1).value
state,source=read_model_record(record)
inputs=record.get_metadata(1,0,"batch_input")
output=infer_model(state,source,inputs)
assert all(isinstance(v,Fraction) for v in output.values.flat)
assert record.get_metadata(1,0,"batch_output").coefficient_digest==output.coefficient_digest
result=Executor(sources={"record":record},params={"input":inputs}).execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
assert result.values[0].coefficient_digest==output.coefficient_digest
assert not any(name=="torch" or name.startswith("torch.") for name in sys.modules)
print(output.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected.coefficient_digest
