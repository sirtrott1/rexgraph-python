"""Exact prepared updates, bounded pairing caches and canonical checkpoint replay."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q

import pytest
from rexgraph_cpp import CancellationToken, ExecutionCancelled, NativeSource

from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch, prepare_learning_model
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import train_model


def declaration():
    return dict(grade=1, channels=("identity", "hodge_down", "hodge_up", "hodge"),
                model="prepared-training", gains=4000, bias=4001, channel_coordinates=900,
                bias_coordinates=901, history_namespace=700, metadata_carrier=9900,
                metadata_tensor=9901, time_coordinates=9902, metadata_coordinates=9903,
                initial_parameter_knowledge_time=2, publication_cutoff=0,
                samples=({"input": 1001, "target": 1001, "event_time": 3, "knowledge_time": 4,
                          "label_available_time": 5, "partition": "training"},
                         {"input": 4002, "target": 4003, "event_time": 6, "knowledge_time": 7,
                          "label_available_time": 8, "partition": "training"}))


@pytest.fixture(params=["unit", "diagonal"])
def prepared_case(native_fixtures, request):
    source = NativeSource((native_fixtures[0] / f"learning-{request.param}.rfcxs").read_bytes())
    options = dict(metric_selection=request.param,
                   metric_keys=(2000, 2001, 2002) if request.param == "diagonal" else ())
    return source, options


def batch(state, source):
    return cpp_training_batch(state, source, rate=Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20)


@pytest.mark.parametrize("capacity", [0, 1, 2, 8])
@pytest.mark.parametrize("duplicate", [False, True])
def test_cached_updates_preserve_parameters_loss_time_and_complete_bytes(prepared_case, capacity, duplicate):
    source, options = prepared_case
    d = declaration()
    if duplicate:
        d["channels"] = ("identity", "hodge_down", "hodge_down", "hodge")
    plan = source.prepare((10, 11, 12), **options)
    scalar = source.trainer(plan, d, cache_samples=0)
    cached = source.trainer(plan, d, cache_samples=capacity)
    initial = cached.snapshot
    for step in range(6):
        rate = Q(1, 1000000 + 17*step)
        scalar = scalar.advance(rate, event_cutoff=10, knowledge_cutoff=20+step)
        cached = cached.advance(rate, event_cutoff=10, knowledge_cutoff=20+step)
        assert cached.snapshot == scalar.snapshot
        assert cached.parameter_knowledge_time == scalar.parameter_knowledge_time
    assert initial["step"] == 0
    info = cached.training_execution
    assert info["capacity"] == min(capacity, 2)
    assert info["retained_slots"] == info["cached_samples"] * 21
    expected_channels = ("identity", "hodge_down", "hodge" if duplicate else "hodge_up", "scalar_bias")
    assert info["response_channels"] == (expected_channels if capacity else ())
    expected_coordinates = ((0,), (1,), (1,) if duplicate else (2,), (2,) if duplicate else (1, 2), (3,))
    assert info["parameter_coordinates"] == (expected_coordinates if capacity else ())
    assert info["hits"] == (4 if capacity >= 2 else 0)
    assert info["misses"] == (2 if capacity >= 2 else 6 if capacity else 0)
    assert set(cached.snapshot) == {"archive", "parameters", "last_loss", "step", "history_namespace"}


@pytest.mark.parametrize("capacity", [1, 2, 8])
def test_cold_replay_reuses_pairings_and_preserves_next_update(prepared_case, capacity):
    source, options = prepared_case
    d = declaration()
    plan = source.prepare((10, 11, 12), **options)
    direct = source.trainer(plan, d, cache_samples=0).advance(Q(1, 1000000), event_cutoff=10,
                                                          knowledge_cutoff=20, steps=8)
    loaded = NativeSource(source.archive)
    replay = loaded.trainer(loaded.prepare((10, 11, 12), **options), d,
                            archive=direct.snapshot["archive"], cache_samples=capacity)
    assert replay.snapshot == direct.snapshot
    assert replay.training_execution["hits"] == (6 if capacity >= 2 else 0)
    assert replay.advance(Q(1, 999999), event_cutoff=10, knowledge_cutoff=21).snapshot == direct.advance(
        Q(1, 999999), event_cutoff=10, knowledge_cutoff=21).snapshot


@pytest.mark.parametrize("capacity", [0, 2])
@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_model_records_retain_execution_selection_and_exact_continuation(prepared_case, capacity, suffix, tmp_path):
    source, options = prepared_case
    initial = cpp_trainable_model(source, (10, 11, 12), declaration(), training_cache_samples=capacity, **options)
    trained = train_model(initial, source, batch(initial, source), steps=4)
    path = tmp_path / ("model"+suffix)
    save_model(path, model_record(source, trained))
    restored, loaded = read_model_record(load_model(path))
    assert restored.coefficient_digest == trained.coefficient_digest
    assert restored.configuration["training_cache_samples"] == capacity
    actual = train_model(restored, loaded, batch(restored, loaded), steps=2)
    expected = train_model(trained, source, batch(trained, source), steps=2)
    assert actual.coefficient_digest == expected.coefficient_digest
    assert actual.payload == expected.payload
    assert "training_execution" not in actual.payload


@pytest.mark.parametrize("capacity", [True, -1, 1025, 0.5, Q(1, 2)])
def test_cache_selection_rejects_invalid_capacities(prepared_case, capacity):
    source, options = prepared_case
    with pytest.raises((TypeError, ValueError, OverflowError)):
        source.trainer(source.prepare((10, 11, 12), **options), declaration(), cache_samples=capacity)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        cpp_trainable_model(source, (10, 11, 12), declaration(), training_cache_samples=capacity, **options)


def test_availability_evaluation_and_cancellation_precede_cache_use(prepared_case):
    source, options = prepared_case
    d = declaration()
    plan = source.prepare((10, 11, 12), **options)
    native = source.trainer(plan, d)
    before = native.snapshot
    for events, knowledge in [(2, 20), (10, 4), (10, 1)]:
        with pytest.raises(ValueError):
            native.advance(Q(1), event_cutoff=events, knowledge_cutoff=knowledge)
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled):
        native.advance(Q(1), event_cutoff=10, knowledge_cutoff=20, cancellation=token)
    assert native.snapshot == before
    assert native.training_execution["cached_samples"] == 0
    trained = native.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20, steps=2)
    info = trained.training_execution
    with pytest.raises(ValueError):
        trained.advance(Q(1), event_cutoff=10, knowledge_cutoff=19)
    with pytest.raises(ExecutionCancelled):
        trained.advance(Q(1), event_cutoff=10, knowledge_cutoff=20, cancellation=token)
    assert trained.training_execution == info
    d = declaration(); d["samples"][1]["partition"] = "evaluation"
    evaluation = source.trainer(plan, d).advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20)
    saved = evaluation.snapshot
    with pytest.raises(ValueError):
        evaluation.advance(Q(1), event_cutoff=10, knowledge_cutoff=20)
    assert evaluation.snapshot == saved
    assert evaluation.training_execution["cached_samples"] == 1


def test_shared_cache_concurrent_children_preserve_parent_and_seals(prepared_case):
    source, options = prepared_case
    initial = cpp_trainable_model(source, (10, 11, 12), declaration(), **options)
    b = batch(initial, source)
    with ThreadPoolExecutor(max_workers=3) as pool:
        children = list(pool.map(lambda _: train_model(initial, source, b, steps=4), range(3)))
    assert all(child.coefficient_digest == children[0].coefficient_digest for child in children)
    assert initial.step == 0
    native = prepare_learning_model(children[0], source)
    assert native.training_execution["cached_samples"] == 2
    assert native.training_execution["hits"] >= 10


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
@pytest.mark.parametrize("capacity", [0, 2])
def test_fresh_process_store_resume_retains_training_selection(prepared_case, tmp_path, backend, capacity):
    import subprocess
    import sys

    from rcdb import open_store

    source, options = prepared_case
    initial = cpp_trainable_model(source, (10, 11, 12), declaration(), training_cache_samples=capacity, **options)
    trained = train_model(initial, source, batch(initial, source), steps=4)
    expected = train_model(trained, source, batch(trained, source), steps=2)
    url = {"file": "file://"+str(tmp_path/"file"), "rex": "rex://"+str(tmp_path/"rex"),
           "sqlite": "sqlite:///"+str(tmp_path/"records.sqlite")}[backend]
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    store.commit_mutation("prepared-training", model_record(source, trained), expected_version=0)
    assert store.verify_commits("prepared-training"); store.close()
    script = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."): raise AssertionError("unexpected Torch import")
sys.meta_path.insert(0, NoTorch())
from fractions import Fraction
from rcdb import open_store
from rexgraph.cpp_learning import cpp_training_batch, prepare_learning_model
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import train_model
store = open_store(sys.argv[1])
state, source = read_model_record(store.read_record("prepared-training",version=1).value)
assert state.configuration["training_cache_samples"] == int(sys.argv[2])
observations = cpp_training_batch(state,source,rate=Fraction(1,1000000),event_cutoff=10,knowledge_cutoff=20)
child = train_model(state,source,observations,steps=2)
info = prepare_learning_model(child,source).training_execution
assert info["capacity"] == int(sys.argv[2])
assert info["hits"] == (4 if int(sys.argv[2]) else 0)
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
print(child.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url, str(capacity)], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == expected.coefficient_digest
