"""Declared checkpoint capacity, complete sample metadata and exact continuation."""
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from rexgraph_cpp import NativeSource

from test_learning import declaration
from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch, prepare_learning_model
from rexgraph.cpp_transport import cpp_transport
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import train_model, transport_model
from rexgraph.model_state import thaw_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.value_codec import pack_value


@pytest.fixture(params=["unit", "diagonal"])
def training_source(native_fixtures, request):
    source = NativeSource((native_fixtures[0] / f"learning-{request.param}.rfcxs").read_bytes())
    return source, dict(metric_selection=request.param,
                        metric_keys=(2000, 2001, 2002) if request.param == "diagonal" else ())


def observations(state, source):
    return cpp_training_batch(state, source, rate=Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20)


@pytest.mark.parametrize("capacity", [1024, 2003, 4096, 8192, 65536])
def test_capacity_preserves_exact_updates_and_complete_native_resume(training_source, capacity):
    source, options = training_source
    plan = source.prepare((10, 11, 12), **options)
    d = declaration()
    original = source.trainer(plan, d)
    explicit_default = source.trainer(plan, {**d, "metadata_capacity": 4096})
    assert explicit_default is original
    selected = {**d, "metadata_capacity": capacity}
    direct = source.trainer(plan, selected, cache_samples=0)
    cached = source.trainer(plan, selected, cache_samples=2)
    initial = cached.snapshot
    direct = direct.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20, steps=2)
    cached = cached.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20, steps=2)
    reference = original.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20, steps=2)
    assert cached.snapshot == direct.snapshot
    assert cached.snapshot["parameters"] == reference.snapshot["parameters"]
    assert cached.snapshot["last_loss"] == reference.snapshot["last_loss"]
    assert cached.parameter_knowledge_time == reference.parameter_knowledge_time
    if capacity == 4096:
        assert cached.snapshot == reference.snapshot
    else:
        with pytest.raises(ValueError):
            source.trainer(plan, d, archive=cached.snapshot["archive"])
    loaded = NativeSource(source.archive)
    restored = loaded.trainer(loaded.prepare((10, 11, 12), **options), selected,
                              archive=cached.snapshot["archive"])
    assert restored.snapshot == cached.snapshot
    assert restored.advance(Q(1, 999999), event_cutoff=10, knowledge_cutoff=21).snapshot == cached.advance(
        Q(1, 999999), event_cutoff=10, knowledge_cutoff=21).snapshot
    assert initial["step"] == 0


@pytest.mark.parametrize("value", [True, np.bool_(True), -1, 0, 7, 65537, 1 << 64, 4096.0, Q(4096)])
def test_capacity_rejects_invalid_selection(training_source, value):
    source, options = training_source
    d = {**declaration(), "metadata_capacity": value}
    with pytest.raises((TypeError, ValueError, OverflowError)):
        source.trainer(source.prepare((10, 11, 12), **options), d)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        cpp_trainable_model(source, (10, 11, 12), d, **options)


def test_capacity_rejects_incomplete_metadata_and_failed_growth_is_atomic(training_source):
    source, options = training_source
    plan = source.prepare((10, 11, 12), **options)
    with pytest.raises((ValueError, RuntimeError)):
        source.trainer(plan, {**declaration(), "metadata_capacity": 8})
    native = source.trainer(plan, {**declaration(), "metadata_capacity": 1024})
    before, counters = native.snapshot, native.training_execution
    with pytest.raises(RuntimeError):
        native.advance(Q(1 << 8192), event_cutoff=10, knowledge_cutoff=20)
    assert native.snapshot == before
    assert native.training_execution == counters
    assert native.advance(Q(1, 1000000), event_cutoff=10, knowledge_cutoff=20).snapshot["step"] == 1


def test_full_sample_limit_fits_explicit_capacity_and_retains_later_declarations(training_source):
    source, options = training_source
    d = declaration()
    d["samples"] *= 512
    with pytest.raises((ValueError, RuntimeError)):
        cpp_trainable_model(source, (10, 11, 12), d, **options)
    d["metadata_capacity"] = 65536
    state = cpp_trainable_model(source, (10, 11, 12), d, **options)
    child = train_model(state, source, observations(state, source), steps=1)
    replay = prepare_learning_model(child, NativeSource(source.archive))
    assert replay.snapshot["archive"] == child.payload["archive"]
    changed = thaw_tree(child.configuration)
    changed["declaration"]["samples"][-1]["label_available_time"] += 1
    with pytest.raises(ValueError):
        prepare_learning_model(replace(child, configuration=changed), NativeSource(source.archive))
    d["samples"] = (*d["samples"], d["samples"][0])
    with pytest.raises(ValueError):
        cpp_trainable_model(source, (10, 11, 12), d, **options)


def test_example_constructs_1024_distinct_samples_without_metric_key_collisions(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from prepared_training import input_key, make_task

    source, options, d, _, rate, samples, heldout = make_task(2, "diagonal", 1024, 1)
    assert samples == 1024 and heldout == 1
    assert len({sample["input"] for sample in d["samples"]}) == 1024
    assert source.field(input_key(1000)).key == 101000
    d["metadata_capacity"] = 65536
    native = source.trainer(source.prepare((10, 11), **options), d)
    child = native.advance(rate, event_cutoff=1026, knowledge_cutoff=1028)
    assert child.snapshot["step"] == 1 and native.snapshot["step"] == 0


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
@pytest.mark.parametrize("capacity", [1024, 16384])
def test_model_records_retain_capacity_and_exact_next_child(training_source, tmp_path, suffix, capacity):
    source, options = training_source
    d = {**declaration(), "metadata_capacity": capacity}
    if capacity == 16384:
        d["samples"] *= 64
    initial = cpp_trainable_model(source, (10, 11, 12), d, **options)
    child = train_model(initial, source, observations(initial, source), steps=2)
    path = tmp_path / ("trained" + suffix)
    save_model(path, model_record(source, child))
    restored, loaded = read_model_record(load_model(path))
    assert restored.configuration["declaration"]["metadata_capacity"] == capacity
    actual = train_model(restored, loaded, observations(restored, loaded))
    expected = train_model(child, source, observations(child, source))
    assert actual.coefficient_digest == expected.coefficient_digest
    assert pack_value(thaw_tree(actual.payload)) == pack_value(thaw_tree(expected.payload))


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_fresh_process_large_sample_records_use_common_model_routes(training_source, tmp_path, backend):
    from rcdb import open_store

    source, options = training_source
    d = {**declaration(), "metadata_capacity": 16384}
    d["samples"] *= 64
    initial = cpp_trainable_model(source, (10, 11, 12), d, **options)
    child = train_model(initial, source, observations(initial, source), steps=2)
    expected = train_model(child, source, observations(child, source))
    url = {"file": "file://" + str(tmp_path / "file"), "rex": "rex://" + str(tmp_path / "rex"),
           "sqlite": "sqlite:///" + str(tmp_path / "records.sqlite")}[backend]
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    store.commit_mutation("large-samples", model_record(source, child), expected_version=0)
    assert store.verify_commits("large-samples")
    store.close()
    script = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."): raise AssertionError("unexpected Torch import")
sys.meta_path.insert(0, NoTorch())
from fractions import Fraction
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor, parse
from rexgraph.cpp_learning import cpp_training_batch
from rexgraph.io.model_record import read_model_record, model_record
from rexgraph.model_runtime import train_model, infer_model
from agent.hive import Hive
from system.serialize import json_value
store = open_store(sys.argv[1])
state, source = read_model_record(store.read_record("large-samples", version=1).value)
assert state.configuration["declaration"]["metadata_capacity"] == 16384
assert len(state.configuration["declaration"]["samples"]) == 128
batch = cpp_training_batch(state, source, rate=Fraction(1,1000000),event_cutoff=10,knowledge_cutoff=20)
record = model_record(source,state)
result = Executor(sources={"record":record},params={"batch":batch}).execute(
    parse('FROM $record RETURN MODEL_TRAIN(MODEL_STATE(),$batch)'))
child = train_model(state,source,batch)
assert result.values[0].coefficient_digest == child.coefficient_digest
prediction = infer_model(child,source,{"field_key":1001})
assert json_value(child)["resumable"]
with patch("agent.activity.record"),patch("agent.activity.open_use",return_value=0),patch("agent.activity.close_use"):
    hive = Hive("checkpoint-test")
    hive.add_worker("trained",lambda value,**kw:infer_model(child,source,value),capability="predict")
    assert hive.invoke("trained",{"field_key":1001},record=False).coefficient_digest == prediction.coefficient_digest
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
print(child.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == expected.coefficient_digest


@pytest.mark.parametrize("capacity", [None, 1024, 8192])
@pytest.mark.parametrize("origin_capacity", [None, 1024])
def test_transport_selects_destination_checkpoint_capacity(tmp_path, capacity, origin_capacity):
    from test_transport import family, next_batch

    source, state, target, mapping = family("cpp_metric", metadata_capacity=origin_capacity)
    config = thaw_tree(mapping["configuration"])
    config["declaration"].pop("metadata_capacity", None)
    if capacity is not None:
        config["declaration"]["metadata_capacity"] = capacity
    new_mapping = cpp_transport(source, target, mapping["cell_maps"], mapping["carrier_links"],
                                mapping["tensor_links"], configuration=config)
    child = transport_model(state, source, CppSourceReference(target), new_mapping)
    assert child.configuration["declaration"].get("metadata_capacity", 4096) == (4096 if capacity is None else capacity)
    path = tmp_path / "transported.rcbd"
    save_model(path, model_record(target, child))
    restored, loaded = read_model_record(load_model(path))
    actual = train_model(restored, loaded, next_batch(restored, loaded))
    expected = train_model(child, target, next_batch(child, target))
    assert actual.coefficient_digest == expected.coefficient_digest
    assert pack_value(thaw_tree(actual.payload)) == pack_value(thaw_tree(expected.payload))
