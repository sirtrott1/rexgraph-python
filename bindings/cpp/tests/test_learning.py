"""Canonical integer inference and exact training through native model records."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph_cpp import CancellationToken, ExecutionCancelled, ExecutionLimitError, NativeSource

from rexgraph.cpp_learning import (
    cpp_integer_input,
    cpp_integer_model,
    cpp_trainable_model,
    cpp_training_batch,
    prepare_learning_model,
)
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import infer_model, train_model

LAYERS = ({"channels": ("identity", "hodge_down", "hodge_up", "hodge"),
           "gains": (2, -1, 1, 0), "bias": 3, "activation": "relu"},
          {"channels": ("identity", "hodge"), "gains": (1, -1), "bias": -2, "activation": "identity"})


def declaration():
    return dict(grade=1, channels=("identity", "hodge_down", "hodge_up", "hodge"),
                model="bridge-training", gains=4000, bias=4001, channel_coordinates=900,
                bias_coordinates=901, history_namespace=700, metadata_carrier=9900,
                metadata_tensor=9901, time_coordinates=9902, metadata_coordinates=9903,
                initial_parameter_knowledge_time=2, publication_cutoff=0,
                samples=({"input": 1001, "target": 1001, "event_time": 3, "knowledge_time": 4,
                          "label_available_time": 5, "partition": "training"},
                         {"input": 4002, "target": 4003, "event_time": 6, "knowledge_time": 7,
                          "label_available_time": 8, "partition": "training"}))


@pytest.fixture(params=["diagonal", "unit"])
def learning_case(native_fixtures, request):
    root = native_fixtures[0]
    name = "learning-" + request.param
    source = NativeSource((root / (name + ".rfcxs")).read_bytes())
    options = dict(metric_selection=request.param, metric_keys=(2000, 2001, 2002) if request.param == "diagonal" else ())
    oracle = json.loads((root / (name + ".json")).read_text())
    return source, options, oracle, root, name


def model(case, d=None):
    return cpp_trainable_model(case[0], (10, 11, 12), declaration() if d is None else d, **case[1])


def batch(state, source, **kwargs):
    return cpp_training_batch(state, source, rate=Q(1, 1000000), event_cutoff=10,
                              knowledge_cutoff=20, **kwargs)


def test_integer_network_matches_canonical_cpp(learning_case):
    source, _, _, root, name = learning_case
    state = cpp_integer_model(source, 11, LAYERS)
    inputs = cpp_integer_input(state, source, (1, -2, 3))
    output = infer_model(state, source, inputs)
    assert tuple(output.values) == tuple(json.loads((root / (name + "-integer.json")).read_text()))
    assert output.variance == "coordinate"
    assert output.execution_receipt["channel_profile"] == "integral-lattice"
    assert output.execution_receipt["scalar_policy"] == "checked-int64"
    assert output.execution_receipt["input_values"] == (1, -2, 3)
    plan = prepare_learning_model(state, source)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: plan.forward(source.integer_input(11, (1, -2, 3))), range(8)))
    assert all(row["values"] == tuple(output.values) for row in results)


@pytest.mark.parametrize("value", [0.5, True, np.bool_(True), Q(1, 2), 1 << 63, -(1 << 63) - 1])
def test_integer_input_rejects_lossy_coefficients(native_source, value):
    state = cpp_integer_model(native_source, 11, LAYERS)
    with pytest.raises((TypeError, OverflowError)):
        cpp_integer_input(state, native_source, (value, 1, 2))


def test_integer_checked_overflow_activation_and_ownership(native_source):
    layers = ({"channels": ("identity",), "gains": (2,), "bias": 0, "activation": "identity"},)
    plan = native_source.integer_network(11, layers)
    with pytest.raises(OverflowError):
        plan.forward(native_source.integer_input(11, (1 << 62, 0, 0)))
    foreign = NativeSource(native_source.archive)
    with pytest.raises(ValueError):
        plan.forward(foreign.integer_input(11, (1, 0, 0)))
    with pytest.raises(ValueError):
        plan.forward(native_source.integer_input(21, (1, 0, 0)))
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled):
        plan.forward(native_source.integer_input(11, (1, 0, 0)), cancellation=token)
    signed = native_source.integer_network(11, ({**layers[0], "gains": (1,), "activation": "sign"},))
    assert signed.forward(native_source.integer_input(11, (-1, 0, 1)))["values"] == (-1, 1, 1)


@pytest.mark.parametrize("value", [np.bool_(True), True, 0.5, Q(1, 2)])
def test_low_level_integer_input_rejects_non_integer_carriers(native_source, value):
    with pytest.raises(TypeError):
        native_source.integer_input(11, (value, 0, 0))


@pytest.mark.parametrize("mode", ["layers", "channels", "activation", "count", "policy"])
def test_integer_declaration_and_policy_bounds(native_source, mode):
    layers = [dict(LAYERS[0])]
    limits = (4000000, 16000000, 16000000, 64000000, 1 << 24)
    if mode == "layers": layers *= 65
    elif mode == "channels": layers[0]["channels"] = ("green",) * 4
    elif mode == "activation": layers[0]["activation"] = "sigmoid"
    elif mode == "count": layers[0]["gains"] = (1,)
    else: limits = (4000000, 16000000, 16000000, 64000000, 63)
    with pytest.raises((ValueError, RuntimeError)):
        cpp_integer_model(native_source, 11, layers, limits=limits)


def test_exact_updates_and_parameters_match_canonical_cpp(learning_case):
    source, _, oracle, root, name = learning_case
    state = model(learning_case)
    initial = state.coefficient_digest
    for row in oracle:
        assert state.step == row["step"]
        assert state.payload["parameters"] == tuple(map(Q, row["parameters"]))
        assert state.payload["last_loss"] == (None if row["loss"] is None else Q(row["loss"]))
        output = infer_model(state, source, {"field_key": 1001})
        assert tuple(output.values) == tuple(map(Q, row["values"]))
        assert output.execution_receipt["namespace"] == 88
        assert output.execution_receipt["parameter_history_namespace"] == 700
        assert output.execution_receipt["parameter_step"] == state.step
        if state.step == 4:
            assert state.payload["archive"] == (root / (name + "-trained.rfcxs")).read_bytes()
        if state.step < 6:
            parent = state
            state = train_model(parent, source, batch(parent, source))
            assert parent.coefficient_digest == state.parent
            assert prepare_learning_model(state, source) is prepare_learning_model(state, source)
    assert model(learning_case).coefficient_digest == initial


def test_exact_gradient_independent_central_difference(learning_case):
    source, options, _, _, _ = learning_case
    state = model(learning_case)
    plan = source.prepare((10, 11, 12), **options)
    field = source.field(1001)
    responses = (field.values, *(plan.execute(field, operation=op)["values"]
                                for op in ("hodge_down", "hodge_up", "hodge")))
    metric = (2, 3, 4) if options["metric_selection"] == "diagonal" else (1, 1, 1)
    def loss(parameters):
        prediction = [parameters[-1] + sum(g * x[i] for g, x in zip(parameters[:-1], responses, strict=True))
                      for i in range(3)]
        return sum(m * (p - y) ** 2 for m, p, y in zip(metric, prediction, field.values, strict=True)) / 2
    before = state.payload["parameters"]
    expected = list(before)
    for i in range(len(before)):
        plus, minus = list(before), list(before)
        plus[i] += Q(1, 13); minus[i] -= Q(1, 13)
        expected[i] -= Q(1, 1000000) * (loss(plus) - loss(minus)) / Q(2, 13)
    after = train_model(state, source, batch(state, source))
    assert after.payload["parameters"] == tuple(expected)
    assert after.payload["last_loss"] == loss(before)


@pytest.mark.parametrize("mode", ["event", "label", "parameter", "evaluation", "late_evaluation"])
def test_causal_and_partition_failures_leave_parent_unchanged(learning_case, mode):
    source = learning_case[0]
    d = declaration()
    if mode in {"evaluation", "late_evaluation"}:
        index = 1 if mode == "late_evaluation" else 0
        d["samples"][index]["partition"] = "evaluation"
    state = model(learning_case, d)
    archive, digest = state.payload["archive"], state.coefficient_digest
    b = cpp_training_batch(state, source, rate=Q(1, 1000000),
                           event_cutoff=2 if mode == "event" else 10,
                           knowledge_cutoff=1 if mode == "parameter" else 4 if mode == "label" else 20)
    with pytest.raises(ValueError):
        train_model(state, source, b, steps=2 if mode == "late_evaluation" else 1)
    assert state.coefficient_digest == digest and state.payload["archive"] == archive and state.step == 0


@pytest.mark.parametrize("mode", ["values", "samples", "targets", "float_targets", "mask", "extra"])
def test_training_does_not_ignore_batch_content(learning_case, mode):
    source = learning_case[0]; state = model(learning_case); b = batch(state, source, capture="values")
    inputs = dict(b.inputs)
    if mode == "values": inputs["values"] = ((Q(99), Q(99)),) * 3
    elif mode == "samples": inputs["samples"] = tuple(reversed(inputs["samples"]))
    elif mode == "extra": inputs["extra"] = True
    targets = np.zeros((3, 2), dtype=float if mode == "float_targets" else object) if mode in {"targets", "float_targets"} else b.targets
    observed = np.zeros(3, dtype=bool) if mode == "mask" else b.observed
    with pytest.raises((ValueError, TypeError)):
        train_model(state, source, replace(b, inputs=inputs, targets=targets, observed=observed))


@pytest.mark.parametrize("mode", ["parameters", "loss", "step", "archive", "channels", "extra", "parent"])
def test_training_restore_rejects_forged_state(learning_case, mode):
    source = learning_case[0]; initial = model(learning_case)
    state = train_model(initial, source, batch(initial, source), steps=2)
    payload, config = dict(state.payload), dict(state.configuration)
    if mode == "parameters": payload["parameters"] = (Q(0),) * 5
    elif mode == "loss": payload["last_loss"] = Q(0)
    elif mode == "archive": payload["archive"] = payload["archive"][:-1]
    elif mode == "extra": payload["extra"] = True
    elif mode == "channels":
        d = dict(config["declaration"]); d["channels"] = tuple(reversed(d["channels"])); config["declaration"] = d
    state = replace(state, payload=payload, configuration=config,
                    step=3 if mode == "step" else state.step, parent=None if mode == "parent" else state.parent)
    with pytest.raises((ValueError, RuntimeError)):
        read_model_record(model_record(source, state))


def test_native_training_cancellation_and_trusted_cache(learning_case):
    source = learning_case[0]; state = model(learning_case); native = prepare_learning_model(state, source)
    before = native.snapshot
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled):
        native.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20, cancellation=token)
    assert native.snapshot == before
    candidate = native.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20)
    source.remember_trainer(candidate)
    plan = source.prepare((10, 11, 12), **learning_case[1])
    assert source.trainer(plan, declaration(), archive=candidate.snapshot["archive"]) is candidate
    with pytest.raises(ValueError):
        NativeSource(source.archive).remember_trainer(candidate)
    with pytest.raises(ValueError):
        native.advance(Q(1), event_cutoff=10, knowledge_cutoff=20, steps=1025)


@pytest.mark.parametrize("mode", ["foreign", "carrier", "partial", "policy"])
def test_training_inference_rejects_incompatible_input(learning_case, mode):
    source = learning_case[0]; state = model(learning_case)
    native = prepare_learning_model(state, source)
    inputs = NativeSource(source.archive).field(1001) if mode == "foreign" else source.field(
        3001 if mode == "carrier" else 3002 if mode == "partial" else 3004)
    if mode == "policy":
        limits = (*state.configuration["limits"][:4], 10000)
        state = cpp_trainable_model(source, (10, 11, 12), declaration(), limits=limits, **learning_case[1])
        native = prepare_learning_model(state, source)
    with pytest.raises((ValueError, ExecutionLimitError)):
        native.forward(inputs)


@pytest.mark.parametrize("mode", ["metadata", "coordinates", "gain_source_grade", "positive", "foreign_plan"])
def test_native_training_declaration_bounds(learning_case, mode):
    source = learning_case[0]; d = declaration(); options = dict(learning_case[1])
    if mode == "metadata": d["samples"] *= 100
    elif mode == "coordinates": d["metadata_coordinates"] = d["time_coordinates"]
    elif mode == "gain_source_grade": d["gains"] = 1001
    elif mode == "positive": options["metric_selection"] = "positive"
    if mode == "foreign_plan":
        plan = NativeSource(source.archive).prepare((10, 11, 12), **options)
        with pytest.raises(ValueError): source.trainer(plan, d)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            cpp_trainable_model(source, (10, 11, 12), d, **options)


def test_chunked_updates_and_concurrent_children(learning_case):
    source = learning_case[0]; state = model(learning_case); b = batch(state, source)
    with ThreadPoolExecutor(max_workers=3) as pool:
        children = list(pool.map(lambda _: train_model(state, source, b, steps=4), range(3)))
    assert all(child.coefficient_digest == children[0].coefficient_digest for child in children)
    serial = state
    for _ in range(4): serial = train_model(serial, source, b)
    assert serial.payload == children[0].payload
    assert state.step == 0


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_integer_and_training_records_roundtrip(learning_case, tmp_path, suffix):
    source = learning_case[0]
    integer = cpp_integer_model(source, 11, LAYERS)
    initial = model(learning_case)
    trained = train_model(initial, source, batch(initial, source), steps=4)
    for state in (integer, trained):
        path = tmp_path / (state.adapter + suffix)
        save_model(path, model_record(source, state))
        restored, loaded = read_model_record(load_model(path))
        assert restored.coefficient_digest == state.coefficient_digest
        if state.adapter == "cpp_integer":
            actual = infer_model(restored, loaded, cpp_integer_input(restored, loaded, (1, -2, 3)))
            expected = infer_model(state, source, cpp_integer_input(state, source, (1, -2, 3)))
        else:
            actual = train_model(restored, loaded, batch(restored, loaded), steps=2)
            expected = train_model(trained, source, batch(trained, source), steps=2)
        assert actual.coefficient_digest == expected.coefficient_digest


NO_TORCH = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("native model imported Torch")
sys.meta_path.insert(0, NoTorch())
'''


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_fresh_process_committed_training_resume(learning_case, tmp_path, backend):
    from rcdb import open_store
    source = learning_case[0]; initial = model(learning_case)
    state = train_model(initial, source, batch(initial, source), steps=4)
    expected = train_model(state, source, batch(state, source), steps=2)
    url = {"file": "file://" + str(tmp_path / "file"), "rex": "rex://" + str(tmp_path / "rex"),
           "sqlite": "sqlite:///" + str(tmp_path / "records.sqlite")}[backend]
    options = {} if backend == "sqlite" else {"read_only": False}
    store = open_store(url, **options).configure_security(require_commits=True)
    store.commit_mutation("exact-training", model_record(source, state), expected_version=0)
    assert store.verify_commits("exact-training"); store.close()
    script = NO_TORCH + '''
from fractions import Fraction
from rcdb import open_store
from rexgraph.cpp_learning import cpp_training_batch
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import train_model
store = open_store(sys.argv[1])
state, source = read_model_record(store.read_record("exact-training", version=1).value)
batch = cpp_training_batch(state, source, rate=Fraction(1,1000000), event_cutoff=10, knowledge_cutoff=20)
child = train_model(state, source, batch, steps=2)
from rcql import Executor, parse
from rexgraph.io.model_record import model_record
from rexgraph.model_runtime import infer_model
from system.serialize import json_value
from agent.hive import Hive
from unittest.mock import patch
record = model_record(source, child)
result = Executor(sources={"record": record}, params={"input": {"field_key":1001}}).execute(
    parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
expected = infer_model(child, source, {"field_key":1001})
assert result.values[0].coefficient_digest == expected.coefficient_digest
assert json_value(child)["resumable"]
assert json_value(expected)["execution_receipt"]["parameter_step"] == 6
with patch("agent.activity.record"), patch("agent.activity.open_use", return_value=0), patch("agent.activity.close_use"):
    hive = Hive("private-native-test")
    hive.add_worker("metric-model", lambda value, **kw: infer_model(child, source, value), capability="predict")
    assert hive.invoke("metric-model", {"field_key":1001}, record=False).coefficient_digest == expected.coefficient_digest
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
print(child.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == expected.coefficient_digest


def test_rcql_existing_native_model_routes(learning_case):
    from rcql import Executor, parse
    source = learning_case[0]; state = model(learning_case); b = batch(state, source)
    record = model_record(source, state)
    result = Executor(sources={"record": record}, params={"batch": b, "input": {"field_key": 1001}}).execute(
        parse('FROM $record LET m = MODEL_STATE() LET n = MODEL_TRAIN(m,$batch,2) RETURN n, MODEL_INFER(n,$input), MODEL_RECORD(n)'))
    child, output, saved = result.values
    expected = train_model(state, source, b, steps=2)
    assert child.coefficient_digest == expected.coefficient_digest
    assert output.coefficient_digest == infer_model(expected, source, {"field_key": 1001}).coefficient_digest
    restored, loaded = read_model_record(saved)
    assert restored.coefficient_digest == child.coefficient_digest
    assert loaded.archive == source.archive


@pytest.mark.parametrize("adapter", ["cpp_field", "cpp_integer", "cpp_metric"])
def test_rcql_native_model_creation_and_exact_types(learning_case, adapter):
    from rcql import Executor, parse
    source = learning_case[0]; original = model(learning_case)
    record = model_record(source, original)
    config = {"carrier": 11, "layers": LAYERS} if adapter == "cpp_integer" else {
        "grade_carriers": (10, 11, 12), **learning_case[1]}
    if adapter == "cpp_metric": config["declaration"] = declaration()
    result = Executor(sources={"record": record}, params={"config": config}).execute(parse(
        'FROM $record RETURN MODEL_INIT("' + adapter + '",$config)'))
    state = result.values[0]
    assert state.adapter == adapter and state.arithmetic == "rational"
    assert state.source.native == original.source.native
    assert read_model_record(model_record(source, state))[0].coefficient_digest == state.coefficient_digest
    invalid = parse('FROM $record RETURN MODEL_INIT("' + adapter + '",$config,$none,$none,1)')
    with pytest.raises(ValueError, match="seed"):
        Executor(sources={"record": record}, params={"config": config, "none": None}).execute(invalid)


def test_system_model_preview_retains_parameters_and_receipt(learning_case):
    from system.serialize import json_value
    source = learning_case[0]; state = model(learning_case)
    trained = train_model(state, source, batch(state, source), steps=2)
    preview = json_value(trained, max_values=2)
    assert preview["resumable"] and preview["parameter_count"] == 5
    assert len(preview["parameters"]) == 2 and preview["parameters_truncated"]
    assert preview["parameter_history_namespace"] == 700
    output = json_value(infer_model(trained, source, {"field_key": 1001}))
    assert output["execution_receipt"]["engine"] == dict(state.configuration["engine_receipt"])
    assert output["execution_receipt"]["parameter_step"] == 2
    integer = cpp_integer_model(source, 11, LAYERS)
    preview = json_value(integer, max_values=2)
    assert preview["parameter_count"] == 8 and not preview["resumable"]
    assert preview["parameters"] == [2, -1] and preview["parameters_truncated"]
    json.dumps(output, allow_nan=False)
