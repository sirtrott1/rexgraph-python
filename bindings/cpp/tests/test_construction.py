"""Exact native source construction and sparse integer model lifecycle."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest
from rexgraph_cpp import CancellationToken, ExecutionCancelled, NativePerceptron, NativeSource

from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch
from rexgraph.cpp_model import cpp_model
from rexgraph.cpp_perceptron import (
    cpp_perceptron_batch,
    cpp_perceptron_input,
    cpp_perceptron_model,
    prepare_perceptron,
)
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model

ROWS = (((0, 2), (1, 1)), ((0, -2), (1, -1)), ((0, 1), (0, 1), (1, 2)))
LABELS = (1, -1, 1)


def declaration():
    def occ(key, lower, negative=False, magnitude=1, exists=True):
        return dict(key=key, lower=lower, exists=exists, negative=negative, magnitude=Q(magnitude))
    def rel(grade, key, values):
        return dict(grade=grade, key=key, type=0, occurrences=values)
    carriers = [dict(key=10+k, kind="source_grade", grade=k, coordinates=0, size=n)
                for k, n in enumerate((3, 3, 2))]
    carriers += [dict(key=21, kind="source_grade", grade=1, coordinates=0, size=3),
                 dict(key=40, kind="explicit", grade=0, coordinates=900, size=2),
                 dict(key=41, kind="explicit", grade=0, coordinates=901, size=1)]
    fields = [dict(key=1000+k, carrier=10+k, entries=((0, Q(1, 3)), (n-1, Q(-2, 5))), support=None)
              for k, n in enumerate((3, 3, 2))]
    fields += [dict(key=3001, carrier=21, entries=((0, Q(1, 3)), (2, Q(-2, 5))), support=None),
               dict(key=3002, carrier=11, entries=((0, Q(1, 3)),), support=(0,)),
               dict(key=4000, carrier=40, entries=((0, 1), (1, -2)), support=None),
               dict(key=4001, carrier=41, entries=((0, 3),), support=None)]
    return dict(profile="declared-exact-source-v1", namespace=88, semantic_scale=0, max_grade=2,
                types=(dict(name="bridge-fixture", min_arity=0, max_arity=10),), witnesses=(),
                relations=[rel(1, 100, (occ(0, 10), occ(1, 11, True), occ(2, 10, magnitude=2),
                                       occ(3, 10, True, 2), occ(4, 12, magnitude=7, exists=False))),
                           rel(1, 101, (occ(5, 10), occ(6, 11, True))),
                           rel(1, 102, (occ(7, 11), occ(8, 12, True))),
                           rel(2, 200, (occ(9, 100, magnitude=2), occ(10, 101, True, 2))), rel(2, 201, ())],
                carriers=carriers, fields=fields,
                metrics=[dict(key=2000+k, carrier=10+k, diagonal=tuple(range(2, n+2)))
                         for k, n in enumerate((3, 3, 2))])


@pytest.fixture
def built_source():
    return NativeSource.from_declaration(declaration())


def availability():
    return tuple(dict(event_time=2+i, knowledge_time=3+i, label_available_time=4+i,
                      partition="training") for i in range(3))


def model(source, **kwargs):
    return cpp_perceptron_model(source, 11, 4000, 4001, availability(),
                                feature_coordinates=900, time_coordinates=999, **kwargs)


def batch(state, source, **kwargs):
    return cpp_perceptron_batch(state, source, ROWS, LABELS, rate=2,
                                event_cutoff=10, knowledge_cutoff=20, **kwargs)


def test_constructed_source_matches_canonical_bytes_and_exact_actions(native_fixtures, built_source):
    assert built_source.archive == (native_fixtures[0] / "constructed.rfcxs").read_bytes()
    assert NativeSource(built_source.archive).identity == built_source.identity
    assert built_source.carrier(40) == dict(key=40, kind="explicit", grade=0, coordinates=900, size=2)
    assert built_source.field(1001).support == (0, 1, 2)
    assert built_source.field(1001).values[1] == 0
    assert built_source.field(3002).support == (0,)
    state = cpp_model(built_source, (10, 11, 12), metric_keys=(2000, 2001, 2002), metric_selection="diagonal")
    assert tuple(infer_model(state, built_source, built_source.field(1001)).values) == (Q(83, 30), Q(-17, 45), Q(-19, 20))


def test_construction_preserves_witnesses_empty_grades_and_big_coefficients():
    d = declaration(); d["witnesses"] = (999,)
    d["carriers"][0]["size"] = 4; d["metrics"][0]["diagonal"] = (2, 3, 4, 5)
    d["fields"][0]["entries"] = ((1, Q(1 << 16384, 7)),)
    d["max_grade"] = 3
    source = NativeSource.from_declaration(d)
    assert source.identity["coordinates"][0] == ("999", "10", "11", "12")
    assert source.identity["coordinates"][3] == ()
    assert source.field(1000).values[1] == Q(1 << 16384, 7)
    assert NativeSource(source.archive).archive == source.archive


@pytest.mark.parametrize("mode", ["profile", "extra", "occurrence", "closure", "future_lower", "arity",
    "negative", "float", "boolean", "aspect", "carrier", "shape", "tensor", "support", "field_carrier", "metric_carrier"])
def test_source_construction_rejects_invalid_declarations(mode):
    d = declaration()
    if mode == "profile": d["profile"] = "inferred"
    elif mode == "extra": d["notes"] = "ignored"
    elif mode == "occurrence": d["relations"][1]["occurrences"][0]["key"] = 0
    elif mode == "closure": d["relations"][3]["occurrences"][1]["negative"] = False
    elif mode == "future_lower": d["relations"][3]["occurrences"][0]["lower"] = 999
    elif mode == "arity": d["types"][0]["max_arity"] = 1
    elif mode == "negative": d["relations"][0]["occurrences"][0]["magnitude"] = -1
    elif mode == "float": d["fields"][0]["entries"] = ((0, 0.1),)
    elif mode == "boolean": d["fields"][0]["entries"] = ((0, np.bool_(True)),)
    elif mode == "aspect": d["relations"][0]["occurrences"][0]["exists"] = 1
    elif mode == "carrier": d["carriers"][1]["key"] = 10
    elif mode == "shape": d["carriers"][1]["size"] = 2
    elif mode == "tensor": d["fields"][1]["key"] = 1000
    elif mode == "support": d["fields"][0]["support"] = (1,)
    elif mode == "field_carrier": d["fields"][0]["carrier"] = 999
    elif mode == "metric_carrier": d["metrics"][0]["carrier"] = 999
    with pytest.raises((ValueError, TypeError, IndexError, RuntimeError)):
        NativeSource.from_declaration(d)


@pytest.mark.parametrize("bounds", [{"max_records": 5}, {"max_slots": 5}, {"max_integer_bits": 5}, {"max_bytes": 5}])
def test_source_construction_bounds(bounds):
    with pytest.raises((ValueError, RuntimeError)): NativeSource.from_declaration(declaration(), **bounds)


def test_python_constructed_parameters_train_rational_model(built_source):
    d = dict(grade=1, channels=("identity", "hodge"), model="constructed-channel-model", gains=4000,
             bias=4001, channel_coordinates=900, bias_coordinates=901, history_namespace=700,
             metadata_carrier=9900, metadata_tensor=9901, time_coordinates=9902, metadata_coordinates=9903,
             initial_parameter_knowledge_time=0, publication_cutoff=0,
             samples=(dict(input=1001, target=1001, event_time=1, knowledge_time=1,
                           label_available_time=1, partition="training"),))
    state = cpp_trainable_model(built_source, (10, 11, 12), d)
    b = cpp_training_batch(state, built_source, rate=Q(1, 1000), event_cutoff=1, knowledge_cutoff=1)
    child = train_model(state, built_source, b)
    assert child.step == 1 and child.payload["parameters"] != state.payload["parameters"]
    restored, source = read_model_record(model_record(built_source, child))
    assert restored.coefficient_digest == child.coefficient_digest
    assert source.archive == built_source.archive


@pytest.mark.parametrize("storage", ["values", "packed"])
def test_perceptron_matches_canonical_cpp_and_independent_integer_updates(built_source, native_fixtures, storage):
    state = model(built_source, parameter_storage=storage)
    oracle = json.loads((native_fixtures[0] / "perceptron.json").read_text())
    weights, bias = [1, -2], 3
    for row in oracle:
        assert state.step == row["step"]
        assert tuple(state.payload["parameters"]) == tuple(row["parameters"]) == (*weights, bias)
        assert state.payload["mistakes"] == row["mistakes"]
        output = infer_model(state, built_source, cpp_perceptron_input(state, built_source, ROWS))
        assert tuple(output.values) == tuple(row["values"])
        assert output.execution_receipt["scores"] == tuple(row["scores"])
        if state.step < 6:
            mistakes = 0
            for features, label in zip(ROWS, LABELS, strict=True):
                score = bias + sum(weights[i] * value for i, value in features)
                if label * score <= 0:
                    mistakes += 1
                    for i, value in features: weights[i] += 2 * label * value
                    bias += 2 * label
            parent = state
            state = train_model(state, built_source, batch(state, built_source))
            assert state.parent == parent.coefficient_digest and state.payload["mistakes"] == mistakes


def test_zero_margin_empty_rows_and_checked_atomic_overflow():
    native = NativePerceptron((0, 0))
    assert native.forward(((),))["values"] == (1,)
    child = native.advance(((),), (1,))
    assert child.snapshot["bias"] == 1 and child.snapshot["mistakes"] == 1
    assert native.snapshot["bias"] == 0
    before = native.snapshot
    with pytest.raises(OverflowError):
        native.advance((((0, 1),), ((1, (1 << 63) - 1),)), (-1, 1), rate=2)
    assert native.snapshot == before
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): native.advance(((),), (-1,), cancellation=token)
    with pytest.raises(ExecutionCancelled): native.forward(((),), cancellation=token)


@pytest.mark.parametrize("mode", ["features", "float", "labels", "source", "coordinates", "mask", "future", "evaluation", "history_bound"])
@pytest.mark.parametrize("storage", ["values", "packed"])
def test_perceptron_rejects_invalid_or_unavailable_observations(built_source, mode, storage):
    state = model(built_source, parameter_storage=storage)
    if mode == "history_bound": state = model(built_source, parameter_storage=storage,
                                               limits=(4000000, 16000000, 20, 64000000, 1 << 24))
    if mode == "evaluation":
        a = availability(); a[1]["partition"] = "evaluation"
        state = cpp_perceptron_model(built_source, 11, 4000, 4001, a, feature_coordinates=900,
                                    time_coordinates=999, parameter_storage=storage)
    b = batch(state, built_source)
    inputs = dict(b.inputs)
    targets, observed = b.targets, b.observed
    if mode == "features": inputs["rows"] = (((2, 1),),) * 3
    elif mode == "float": inputs["rows"] = (((0, 0.5),),) * 3
    elif mode == "labels": targets = np.array([1, 0, 1])
    elif mode == "source": inputs["source"] = {}
    elif mode == "coordinates": inputs["feature_coordinates"] = 901
    elif mode == "mask": observed = np.zeros(3, dtype=bool)
    elif mode == "future": inputs["knowledge_cutoff"] = 1
    before = state.coefficient_digest
    with pytest.raises((ValueError, TypeError)):
        train_model(state, built_source, replace(b, inputs=inputs, targets=targets, observed=observed),
                    steps=3 if mode == "history_bound" else 1)
    assert state.coefficient_digest == before and state.step == 0


def test_masked_evaluation_rows_and_future_fitted_parameters(built_source):
    a = availability(); a[1]["partition"] = "evaluation"
    state = cpp_perceptron_model(built_source, 11, 4000, 4001, a, feature_coordinates=900, time_coordinates=999)
    b = batch(state, built_source, observed=np.array([True, False, True]))
    child = train_model(state, built_source, b)
    changed = dict(b.inputs); changed["knowledge_cutoff"] = 10
    with pytest.raises(ValueError): train_model(child, built_source, replace(b, inputs=changed))


@pytest.mark.parametrize("mode", ["parameters", "mistakes", "history", "step", "parent", "features", "extra"])
def test_perceptron_checkpoint_replay_rejects_forgery(built_source, mode):
    state = model(built_source)
    state = train_model(state, built_source, batch(state, built_source), steps=2)
    payload, config = dict(state.payload), dict(state.configuration)
    if mode == "parameters": payload["parameters"] = (0, 0, 0)
    elif mode == "mistakes": payload["mistakes"] = 99
    elif mode == "history": payload["history"] = payload["history"][:1]
    elif mode == "features": config["feature_coordinates"] = 901
    elif mode == "extra": payload["extra"] = True
    changed = replace(state, payload=payload, configuration=config,
                      step=3 if mode == "step" else state.step, parent=None if mode == "parent" else state.parent)
    with pytest.raises(ValueError): read_model_record(model_record(built_source, changed))


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
@pytest.mark.parametrize("storage", ["values", "packed"])
def test_source_perceptron_and_batch_roundtrip(built_source, tmp_path, suffix, storage):
    state = model(built_source, parameter_storage=storage); b = batch(state, built_source)
    assert unpack_model(pack_model(b)).coefficient_digest == b.coefficient_digest
    state = train_model(state, built_source, b, steps=4)
    path = tmp_path / ("integer" + suffix)
    save_model(path, model_record(built_source, state))
    restored, source = read_model_record(load_model(path))
    assert restored.coefficient_digest == state.coefficient_digest
    assert train_model(restored, source, batch(restored, source), steps=2).coefficient_digest == train_model(
        state, built_source, batch(state, built_source), steps=2).coefficient_digest


def test_integer_parallel_children_and_source_scope(built_source):
    state = model(built_source); b = batch(state, built_source)
    with ThreadPoolExecutor(max_workers=3) as pool:
        children = list(pool.map(lambda _: train_model(state, built_source, b, steps=4), range(3)))
    assert all(child.coefficient_digest == children[0].coefficient_digest for child in children)
    source = NativeSource(built_source.archive)
    assert prepare_perceptron(state, source).snapshot == prepare_perceptron(state, built_source).snapshot
    d = declaration(); d["namespace"] = 89
    with pytest.raises(ValueError): prepare_perceptron(state, NativeSource.from_declaration(d))


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
@pytest.mark.parametrize("storage", ["values", "packed"])
def test_fresh_process_integer_resume_and_application_routes(built_source, tmp_path, backend, storage):
    from rcdb import open_store
    state = model(built_source, parameter_storage=storage)
    trained = train_model(state, built_source, batch(state, built_source), steps=4)
    expected = train_model(trained, built_source, batch(trained, built_source), steps=2)
    url = {"file": "file://" + str(tmp_path / "file"), "rex": "rex://" + str(tmp_path / "rex"),
           "sqlite": "sqlite:///" + str(tmp_path / "records.sqlite")}[backend]
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    store.commit_mutation("integer-model", model_record(built_source, trained), expected_version=0)
    assert store.verify_commits("integer-model"); store.close()
    from test_learning import NO_TORCH
    script = NO_TORCH + '''
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor,parse
from agent.hive import Hive
from system.serialize import json_value
from rexgraph.io.model_record import read_model_record, model_record
from rexgraph.cpp_perceptron import cpp_perceptron_batch,cpp_perceptron_input
from rexgraph.model_runtime import infer_model,train_model
rows=(((0,2),(1,1)),((0,-2),(1,-1)),((0,1),(0,1),(1,2)))
store=open_store(sys.argv[1])
state,source=read_model_record(store.read_record("integer-model",version=1).value)
batch=cpp_perceptron_batch(state,source,rows,(1,-1,1),rate=2,event_cutoff=10,knowledge_cutoff=20)
child=train_model(state,source,batch,steps=2)
inputs=cpp_perceptron_input(child,source,rows,cpu_workers=2,rows_per_tile=1,prepare=True)
out=infer_model(child,source,inputs)
serial=infer_model(child,source,cpp_perceptron_input(child,source,rows))
assert tuple(serial.values)==tuple(out.values)
assert out.execution_receipt["cpu_execution"]["worker_slots"]==2
assert json_value(child)["resumable"] and json_value(child)["parameter_count"]==3
record=model_record(source,child)
result=Executor(sources={"record":record},params={"input":inputs}).execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
assert result.values[0].coefficient_digest==out.coefficient_digest
with patch("agent.activity.record"),patch("agent.activity.open_use",return_value=0),patch("agent.activity.close_use"):
    hive=Hive("private-perceptron")
    hive.add_worker("integer-model",lambda value,**kw:infer_model(child,source,value),capability="predict")
    assert hive.invoke("integer-model",inputs,record=False).coefficient_digest==out.coefficient_digest
assert not any(name=="torch" or name.startswith("torch.") for name in sys.modules)
print(child.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected.coefficient_digest


def test_rcql_integer_model_creation_and_training(built_source):
    from rcql import Executor, parse
    state = model(built_source)
    record = model_record(built_source, state)
    config = dict(carrier=11, weights_key=4000, bias_key=4001, availability=availability(),
                  feature_coordinates=900, time_coordinates=999)
    result = Executor(sources={"record": record}, params={"config": config}).execute(parse(
        'FROM $record RETURN MODEL_INIT("cpp_perceptron",$config)'))
    created = result.values[0]
    b = batch(created, built_source)
    result = Executor(sources={"record": record}, params={"model": created, "batch": b}).execute(parse(
        'FROM $record RETURN MODEL_TRAIN($model,$batch,2)'))
    assert result.values[0].coefficient_digest == train_model(created, built_source, b, steps=2).coefficient_digest
