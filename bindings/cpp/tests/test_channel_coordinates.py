"""Exact finite response coordinates preserve every original parameter update."""
from pathlib import Path

import pytest
from rexgraph_cpp import NativeSource

from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch, prepare_learning_model
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import train_model


@pytest.fixture
def task(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from channel_training import make_task
    return make_task


@pytest.mark.parametrize("metric", ["unit", "diagonal"])
@pytest.mark.parametrize("seed", [0, 7, 23])
def test_sixty_four_channels_preserve_original_gradients_histories_and_heldout_quality(task, metric, seed):
    from channel_training import heldout_loss

    source, options, contract, diagonal, rate, responses = task(8, 64, metric, seed)
    plan = source.prepare((10, 11), **options)
    direct = source.trainer(plan, contract, cache_samples=0)
    reduced = source.trainer(plan, contract, cache_samples=2)
    parameters = direct.snapshot["parameters"]
    identities = [i for i, name in enumerate(contract["channels"]) if name == "identity"]
    down, up, hodge = (contract["channels"].index(name) for name in ("hodge_down", "hodge_up", "hodge"))
    original_difference = parameters[identities[0]] - parameters[identities[1]]
    original_dependence = parameters[hodge] - parameters[down] - parameters[up]
    target = source.field(5000).values
    error = tuple(sum(p*column[i] for p, column in zip(parameters, responses[0], strict=True))-target[i]
                  for i in range(len(diagonal)))
    expected = tuple(p-rate*sum(value*weight*e for value, weight, e in zip(column, diagonal, error, strict=True))
                     for p, column in zip(parameters, responses[0], strict=True))
    controls = dict(event_cutoff=20, knowledge_cutoff=20)
    first = reduced.advance(rate, **controls)
    assert first.snapshot["parameters"] == expected
    assert first.snapshot["last_loss"] == sum(weight*e*e for weight, e in zip(diagonal, error, strict=True))/2
    before = heldout_loss(reduced, source, diagonal)
    for _ in range(8):
        direct, reduced = direct.advance(rate, **controls), reduced.advance(rate, **controls)
        assert reduced.snapshot == direct.snapshot
        retained = reduced.snapshot["parameters"]
        assert retained[identities[0]] - retained[identities[1]] == original_difference
        assert retained[hodge] - retained[down] - retained[up] == original_dependence
    assert heldout_loss(reduced, source, diagonal) < before
    info = reduced.training_execution
    assert info["response_channels"] == ("identity", "hodge_down", "hodge_up", "scalar_bias")
    assert len(info["parameter_coordinates"]) == 65
    expected_coordinates = {"identity": (0,), "hodge_down": (1,), "hodge_up": (2,), "hodge": (1, 2)}
    assert info["parameter_coordinates"] == tuple(expected_coordinates[name] for name in contract["channels"]) + ((3,),)
    assert info["retained_slots"] == 42 and info["misses"] == 2 and info["hits"] >= 6
    fresh = NativeSource(source.archive)
    restored = fresh.trainer(fresh.prepare((10, 11), **options), contract,
                             archive=reduced.snapshot["archive"], cache_samples=1)
    assert restored.snapshot == reduced.snapshot
    assert restored.advance(rate, **controls).snapshot == direct.advance(rate, **controls).snapshot
    assert len(reduced.snapshot["parameters"]) == 65 and direct.snapshot["step"] == 8


@pytest.mark.parametrize("channels", [("hodge",)*64, ("identity",)*64,
                                     ("hodge", "hodge_down")*32, ("hodge_up", "hodge_down")*32])
def test_coordinates_preserve_repeated_and_missing_sectors(task, channels):
    source, options, contract, _, rate, _ = task(4, 64, "diagonal", 7, channels)
    plan = source.prepare((10, 11), **options)
    direct = source.trainer(plan, contract, cache_samples=0)
    cached = source.trainer(plan, contract, cache_samples=2)
    controls = dict(event_cutoff=20, knowledge_cutoff=20)
    cached = cached.advance(rate, steps=3, **controls)
    assert cached.snapshot == direct.advance(rate, steps=3, **controls).snapshot
    info = cached.training_execution
    assert set(info["response_channels"]) == {*channels, "scalar_bias"}
    assert all(len(row) == 1 for row in info["parameter_coordinates"])
    size = len(info["response_channels"])
    assert info["retained_slots"] == 2*(size*size+size+1)


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_model_io_retains_all_original_coordinates_and_deterministic_resume(task, suffix, tmp_path):
    source, options, contract, _, rate, _ = task(4, 64, "diagonal", 23)
    initial = cpp_trainable_model(source, (10, 11), contract, training_cache_samples=2, **options)
    controls = dict(event_cutoff=20, knowledge_cutoff=20)
    batch = cpp_training_batch(initial, source, rate=rate, **controls)
    trained = train_model(initial, source, batch, steps=4)
    path = tmp_path/("original-coordinates"+suffix)
    save_model(path, model_record(source, trained))
    restored, fresh = read_model_record(load_model(path))
    assert restored.coefficient_digest == trained.coefficient_digest
    resumed = train_model(restored, fresh, cpp_training_batch(restored, fresh, rate=rate, **controls), steps=2)
    expected = train_model(trained, source, batch, steps=2)
    assert resumed.payload == expected.payload and resumed.coefficient_digest == expected.coefficient_digest
    assert len(resumed.payload["parameters"]) == 65
    assert prepare_learning_model(resumed, fresh).training_execution["retained_slots"] == 42
