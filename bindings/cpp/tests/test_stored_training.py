"""Source bound training observations, exact continuation and retained records."""
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from rexgraph_cpp import NativeSource

from test_learning import declaration, learning_case as _learning_case
from rexgraph.cpp_learning import cpp_trainable_model, cpp_training_batch
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_runtime import train_model, transport_model
from rexgraph.model_state import thaw_tree
from rexgraph.source_binding import CppSourceReference
from rexgraph.value_codec import pack_value

learning_case = _learning_case


def model(case, *, cache=2, contract=None):
    return cpp_trainable_model(case[0], (10, 11, 12), declaration() if contract is None else contract,
                               training_cache_samples=cache, **case[1])


def batch(state, source, **kwargs):
    return cpp_training_batch(state, source, rate=Q(1, 1000000), event_cutoff=10,
                              knowledge_cutoff=20, **kwargs)


@pytest.mark.parametrize("cache", [0, 2])
@pytest.mark.parametrize("steps", [1, 4])
def test_stored_and_materialized_observations_produce_identical_children(learning_case, cache, steps):
    source = learning_case[0]
    initial = model(learning_case, cache=cache)
    stored, materialized = batch(initial, source), batch(initial, source, capture="values")
    assert stored.inputs["profile"] == "stored-native-training-samples-v1"
    assert materialized.inputs["profile"] == "native-exact-trainer-v1"
    assert stored.targets.shape == (3, 0) and stored.targets.dtype == object
    assert stored.targets.nbytes == 0 and "values" not in stored.inputs
    assert stored.inputs["samples"] == initial.configuration["declaration"]["samples"]
    assert not stored.targets.flags.writeable and not stored.observed.flags.writeable
    expected = train_model(initial, source, materialized, steps=steps)
    actual = train_model(initial, NativeSource(source.archive), stored, steps=steps)
    assert actual.coefficient_digest == expected.coefficient_digest
    assert actual.payload == expected.payload
    if steps == 4:
        assert actual.payload["archive"] == (learning_case[3] / (learning_case[4]+"-trained.rfcxs")).read_bytes()
    assert initial.step == 0 and actual.parent == initial.coefficient_digest


def test_stored_capture_and_validation_do_not_materialize_columns(learning_case, monkeypatch):
    from rexgraph import cpp_learning

    source = learning_case[0]
    initial = model(learning_case)
    legacy = batch(initial, source, capture="values")
    expected = train_model(initial, source, legacy, steps=4)
    def forbidden(*args):
        raise AssertionError("training copied stored field coefficients")
    monkeypatch.setattr(cpp_learning, "_columns", forbidden)
    stored = batch(initial, source)
    assert train_model(initial, source, stored, steps=4).coefficient_digest == expected.coefficient_digest
    assert train_model(initial, NativeSource(source.archive), stored).step == 1


@pytest.mark.parametrize("capture", ["stored", "values"])
def test_exact_integer_and_rational_scalar_representations_agree(learning_case, capture):
    source = learning_case[0]
    initial = model(learning_case)
    b = cpp_training_batch(initial, source, rate=1, event_cutoff=10, knowledge_cutoff=20, capture=capture)
    inputs = thaw_tree(b.inputs); inputs["rate"] = 1
    if capture == "values":
        inputs["values"] = tuple(tuple(v.numerator if v.denominator == 1 else v for v in row)
                                 for row in inputs["values"])
    expected = train_model(initial, source, b)
    actual = train_model(initial, source, replace(b, inputs=inputs))
    assert actual.coefficient_digest == expected.coefficient_digest and actual.payload == expected.payload


@pytest.mark.parametrize("capture", [None, True, "other", [], {}])
def test_capture_rejects_unknown_modes(learning_case, capture):
    with pytest.raises((ValueError, TypeError)):
        batch(model(learning_case), learning_case[0], capture=capture)


@pytest.mark.parametrize("samples", [None, 42])
def test_preview_handles_incomplete_stored_profile_metadata(learning_case, samples):
    from system.serialize import json_value

    b = batch(model(learning_case), learning_case[0])
    incomplete = replace(b, inputs={"profile": b.inputs["profile"], "samples": samples})
    preview = json_value(incomplete, max_values=1)
    assert "target_storage" not in preview and preview["targets"]["shape"] == [3, 0]


@pytest.mark.parametrize("profile", [None, np.array([1, 2])])
def test_preview_preserves_generic_batches_with_non_string_profile_metadata(learning_case, profile):
    from system.serialize import json_value

    b = batch(model(learning_case), learning_case[0])
    generic = replace(b, inputs={"profile": profile, "samples": b.inputs["samples"]})
    preview = json_value(generic, max_values=1)
    assert "target_storage" not in preview and preview["kind"] == "ModelBatch"


@pytest.mark.parametrize("mode", ["input", "target", "last_time", "order", "duplicate", "partition",
    "extra_sample", "bool_sample", "float_sample", "rational_sample", "bool_rate", "float_rate",
    "bool_cutoff", "float_cutoff", "negative_cutoff", "unknown_profile", "legacy_profile",
    "extra_values", "missing_rate", "mask", "target_values", "target_dtype", "target_axis",
    "no_inputs", "dependency"])
def test_stored_training_rejects_replaced_evidence_without_changing_parent(learning_case, mode):
    source = learning_case[0]
    d = declaration(); d["samples"][0]["event_time"] = 1
    initial = model(learning_case, contract=d)
    b = batch(initial, source)
    inputs, targets, observed, dependencies = thaw_tree(b.inputs), b.targets, b.observed, b.dependencies
    samples = inputs["samples"]
    if mode == "input": samples[0]["input"] = 4002
    elif mode == "target": samples[0]["target"] = 4003
    elif mode == "last_time": samples[-1]["label_available_time"] += 1
    elif mode == "order": inputs["samples"] = tuple(reversed(samples))
    elif mode == "duplicate": inputs["samples"] = (samples[0], samples[0])
    elif mode == "partition": samples[-1]["partition"] = "evaluation"
    elif mode == "extra_sample": samples[0]["other"] = 1
    elif mode == "bool_sample": samples[0]["event_time"] = True
    elif mode == "float_sample": samples[0]["event_time"] = 1.0
    elif mode == "rational_sample": samples[0]["event_time"] = Q(1)
    elif mode == "bool_rate": inputs["rate"] = True
    elif mode == "float_rate": inputs["rate"] = 0.000001
    elif mode == "bool_cutoff": inputs["knowledge_cutoff"] = True
    elif mode == "float_cutoff": inputs["knowledge_cutoff"] = 20.0
    elif mode == "negative_cutoff": inputs["event_cutoff"] = -1
    elif mode == "unknown_profile": inputs["profile"] = "other"
    elif mode == "legacy_profile": inputs["profile"] = "native-exact-trainer-v1"
    elif mode == "extra_values": inputs["values"] = ()
    elif mode == "missing_rate": inputs.pop("rate")
    elif mode == "mask": observed = np.zeros(3, dtype=bool)
    elif mode == "target_values": targets = np.zeros((3, 1), dtype=object)
    elif mode == "target_dtype": targets = np.empty((3, 0), dtype=float)
    elif mode == "target_axis": targets = np.empty((3, 0, 1), dtype=object)
    elif mode == "no_inputs": inputs = None
    elif mode == "dependency": dependencies = (initial.source,)
    before = initial.coefficient_digest, initial.payload["archive"]
    with pytest.raises((ValueError, TypeError)):
        train_model(initial, source, replace(b, inputs=inputs, targets=targets,
                                            observed=observed, dependencies=dependencies))
    assert (initial.coefficient_digest, initial.payload["archive"]) == before
    assert train_model(initial, source, b).step == 1


@pytest.mark.parametrize("capture", ["stored", "values"])
@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_batch_records_restore_declared_fields_and_exact_next_update(learning_case, capture, suffix, tmp_path):
    source = learning_case[0]
    initial = model(learning_case)
    b = batch(initial, source, capture=capture)
    child = train_model(initial, source, b, steps=2)
    record = model_record(source, child)
    record.attach_metadata(1, 0, "observations", b)
    path = tmp_path / ("observations" + suffix)
    save_model(path, record)
    restored_record = load_model(path)
    restored, loaded = read_model_record(restored_record)
    captured = restored_record.get_metadata(1, 0, "observations")
    assert captured.coefficient_digest == b.coefficient_digest
    assert captured.source.source is None
    actual = train_model(restored, loaded, captured, steps=2)
    expected = train_model(child, source, b, steps=2)
    assert actual.coefficient_digest == expected.coefficient_digest
    assert pack_value(thaw_tree(actual.payload)) == pack_value(thaw_tree(expected.payload))


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_fresh_process_records_use_stored_batches_across_package_routes(learning_case, backend, tmp_path):
    from rcdb import open_store

    source = learning_case[0]
    initial = model(learning_case)
    b = batch(initial, source)
    child = train_model(initial, source, b, steps=2)
    expected = train_model(child, source, b)
    record = model_record(source, child)
    record.attach_metadata(1, 0, "observations", b)
    url = {"file": "file://"+str(tmp_path/"file"), "rex": "rex://"+str(tmp_path/"rex"),
           "sqlite": "sqlite:///"+str(tmp_path/"records.sqlite")}[backend]
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    store.commit_mutation("stored-training", record, expected_version=0)
    assert store.verify_commits("stored-training"); store.close()
    script = '''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."): raise AssertionError("unexpected Torch import")
sys.meta_path.insert(0, NoTorch())
from unittest.mock import patch
from rcdb import open_store
from rcql import Executor, parse
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import train_model
from agent.hive import Hive
from system.serialize import json_value
store = open_store(sys.argv[1])
record = store.read_record("stored-training",version=1).value
state, source = read_model_record(record)
batch = record.get_metadata(1,0,"observations")
with patch("rexgraph.cpp_learning._columns",side_effect=AssertionError("unexpected materialization")):
    child = train_model(state,source,batch)
    result = Executor(sources={"record":record},params={"batch":batch}).execute(
        parse('FROM $record RETURN MODEL_TRAIN(MODEL_STATE(),$batch)'))
    assert result.values[0].coefficient_digest == child.coefficient_digest
    with patch("agent.activity.record"),patch("agent.activity.open_use",return_value=0),patch("agent.activity.close_use"):
        hive = Hive("stored-training")
        hive.add_worker("train",lambda value,**kw:train_model(state,source,value),capability="transform")
        assert hive.invoke("train",batch,record=False).coefficient_digest == child.coefficient_digest
preview = json_value(batch,max_values=1)
assert preview["training_profile"] == "stored-native-training-samples-v1"
assert preview["target_storage"] == "native source fields" and preview["sample_count"] == 2
assert preview["targets"]["shape"] == [3,0] and preview["observed_count"] == 3
assert json_value(child)["resumable"]
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
print(child.coefficient_digest)
store.close()
'''
    result = subprocess.run([sys.executable, "-c", script, url], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == expected.coefficient_digest


@pytest.mark.parametrize("capacity", [None, 1024, 8192])
def test_transport_requires_destination_observations_and_preserves_both_profiles(capacity):
    from test_transport import family

    source, state, target, mapping = family("cpp_metric", metadata_capacity=capacity)
    original = cpp_training_batch(state, source, rate=Q(1,1000),event_cutoff=10,knowledge_cutoff=21)
    child = transport_model(state, source, CppSourceReference(target), mapping)
    with pytest.raises(ValueError):
        train_model(child, target, original)
    stored = cpp_training_batch(child,target,rate=Q(1,1000),event_cutoff=10,knowledge_cutoff=21)
    copied = cpp_training_batch(child,target,rate=Q(1,1000),event_cutoff=10,knowledge_cutoff=21,capture="values")
    actual, expected = train_model(child,target,stored,steps=2), train_model(child,target,copied,steps=2)
    assert actual.coefficient_digest == expected.coefficient_digest
    assert pack_value(thaw_tree(actual.payload)) == pack_value(thaw_tree(expected.payload))


@pytest.mark.parametrize("metric", ["unit", "diagonal"])
def test_complete_1024_distinct_samples_use_source_bound_core_training(metric, monkeypatch):
    from rexgraph import cpp_learning

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from prepared_training import make_task

    source, options, d, _, rate, *_ = make_task(2, metric, 1024, 1)
    d["metadata_capacity"] = 65536
    initial = cpp_trainable_model(source,(10,11),d,**options)
    expected = source.trainer(source.prepare((10,11),**options),d).advance(rate,event_cutoff=1026,knowledge_cutoff=1028)
    def forbidden(*args):
        raise AssertionError("training copied stored field coefficients")
    monkeypatch.setattr(cpp_learning,"_columns",forbidden)
    observations = cpp_training_batch(initial,source,rate=rate,event_cutoff=1026,knowledge_cutoff=1028)
    assert len(observations.inputs["samples"]) == 1024 and observations.targets.shape == (2,0)
    actual = train_model(initial,source,observations)
    assert actual.payload["archive"] == expected.snapshot["archive"] and actual.step == 1
