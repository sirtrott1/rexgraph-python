"""Ordered training passes, archive admission and owning parameter time reads."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
from pathlib import Path

import pytest
from rexgraph_cpp import NativeSource, NativeTrainer

from test_learning import declaration, learning_case as _learning_case
from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch, prepare_learning_model
from rexgraph.model_runtime import infer_model, train_model

learning_case = _learning_case


@pytest.mark.parametrize("cache", [0, 2])
def test_parameter_time_follows_immutable_forks_and_resume(learning_case, cache):
    source, options, _, _, _ = learning_case
    plan = source.prepare((10, 11, 12), **options)
    parent = source.trainer(plan, declaration(), cache_samples=cache)
    rate = Q(1, 1000000)
    first = parent.advance(rate, event_cutoff=10, knowledge_cutoff=20, steps=4)
    sibling = parent.advance(rate, event_cutoff=10, knowledge_cutoff=21, steps=4)
    child = first.advance(rate, event_cutoff=10, knowledge_cutoff=22)
    with ThreadPoolExecutor(max_workers=4) as pool:
        times = list(pool.map(lambda trainer: trainer.parameter_knowledge_time,
                              [parent, first, sibling, child]*32))
    assert times == [2, 20, 21, 22]*32
    archive = child.snapshot["archive"]
    fresh = NativeSource(source.archive)
    loaded = fresh.trainer(fresh.prepare((10, 11, 12), **options), declaration(),
                           archive=archive, cache_samples=cache)
    assert loaded.parameter_knowledge_time == 22 and loaded.snapshot == child.snapshot
    returned = loaded.snapshot
    returned["archive"] = b""; returned["step"] = 0
    del fresh, first, sibling, child
    assert loaded.parameter_knowledge_time == 22 and loaded.snapshot["archive"] == archive
    assert parent.parameter_knowledge_time == 2 and parent.snapshot["step"] == 0


def test_parameter_time_rejects_an_uninitialized_owner():
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = NativeTrainer.__new__(NativeTrainer).parameter_knowledge_time


@pytest.mark.parametrize("cache", [0, 2])
def test_archive_byte_admission_applies_to_cold_and_cached_restore(learning_case, cache):
    source, options, _, _, _ = learning_case
    plan = source.prepare((10, 11, 12), **options)
    parent = source.trainer(plan, declaration(), cache_samples=cache)
    child = parent.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20, steps=4)
    archive = child.snapshot["archive"]
    fresh = NativeSource(source.archive)
    fresh_plan = fresh.prepare((10, 11, 12), **options)
    with pytest.raises(ValueError, match="byte limit"):
        fresh.trainer(fresh_plan, declaration(), archive=archive, max_bytes=len(archive)-1, cache_samples=cache)
    restored = fresh.trainer(fresh_plan, declaration(), archive=archive, max_bytes=len(archive), cache_samples=cache)
    assert restored.snapshot == child.snapshot
    with pytest.raises(ValueError, match="byte limit"):
        fresh.trainer(fresh_plan, declaration(), archive=archive, max_bytes=len(archive)-1, cache_samples=cache)
    assert fresh.trainer(fresh_plan, declaration(), archive=archive, max_bytes=len(archive), cache_samples=cache) is restored
    assert restored.snapshot == child.snapshot and parent.snapshot["step"] == 0


@pytest.mark.parametrize("metric", ["unit", "diagonal"])
@pytest.mark.parametrize("cache", [0, 1, 8])
def test_complete_distinct_data_passes_match_updates_and_fresh_continuation(monkeypatch, metric, cache):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from prepared_training import make_task

    source, options, contract, _, rate, samples, _ = make_task(4, metric, 8, 2)
    contract["metadata_capacity"] = 2003
    plan = source.prepare((10, 11), **options)
    initial = source.trainer(plan, contract, cache_samples=cache)
    controls = dict(event_cutoff=20, knowledge_cutoff=20)
    one_call = initial.advance(rate, steps=2*samples, **controls)
    individual = initial
    for _ in range(2*samples):
        individual = individual.advance(rate, **controls)
    assert individual.snapshot == one_call.snapshot
    state = cpp_trainable_model(source, (10, 11), contract, training_cache_samples=cache, **options)
    observations = cpp_training_batch(state, source, rate=rate, **controls)
    child = train_model(state, source, observations, steps=2*samples)
    assert child.payload["archive"] == one_call.snapshot["archive"]
    fresh = NativeSource(source.archive)
    restored = prepare_learning_model(child, fresh)
    assert restored.snapshot == one_call.snapshot
    assert restored.parameter_knowledge_time == 20
    assert infer_model(child, fresh, {"field_key": 1008}).values.tolist() == list(
        one_call.forward(source.field(1008))["values"])
    continued = train_model(child, fresh, cpp_training_batch(child, fresh, rate=rate, **controls))
    assert continued.payload["archive"] == one_call.advance(rate, **controls).snapshot["archive"]
    assert state.step == 0 and child.step == 16 and continued.step == 17
