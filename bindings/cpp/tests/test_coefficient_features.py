"""Exact declared coefficient features retain producer and learner identities."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest
from rexgraph_cpp import NativeSource
from test_algebraic_metric_stalk import result
from test_construction import declaration, model

from rexgraph.cpp_algebraic import cpp_algebraic_model
from rexgraph.cpp_algebraic_stalk import prepare_algebraic_grade_stalks
from rexgraph.cpp_coefficient_features import (
    cpp_coefficient_perceptron_batch,
    cpp_coefficient_perceptron_input,
    validate_coefficient_features,
)
from rexgraph.cpp_perceptron import cpp_perceptron_input, prepare_perceptron
from rexgraph.io import load, save
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value, unpack_value


@pytest.fixture(scope="session")
def files():
    return Path(os.environ["REXGRAPH_CPP_METRIC_FIXTURES"])


def case(files, profile=13, operation="hodge", storage="values", origins=True):
    parent, snapshot, _ = result(files, profile, 2, operation, 1, origins)
    selection = prepare_algebraic_grade_stalks(parent, snapshot)
    outputs = tuple(selection.select(i % selection.values_owner.cell_count) for i in range(3))
    learner = NativeSource.from_declaration(declaration())
    state = model(learner, parameter_storage=storage)
    features = tuple(tuple((index, 0, index, Q(output.values[0, index]).denominator)
                          for index in range(2)) for output in outputs)
    return state, learner, outputs, features, snapshot


def inputs(state, source, outputs, features, **kwargs):
    return cpp_coefficient_perceptron_input(state, source, outputs, row_keys=state.space.keys,
                                           features=features, **kwargs)


def batch(state, source, outputs, features, **kwargs):
    return cpp_coefficient_perceptron_batch(state, source, outputs, (1, -1, 1),
        row_keys=state.space.keys, features=features, rate=2, event_cutoff=10,
        knowledge_cutoff=20, **kwargs)


@pytest.mark.parametrize("profile", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14])
@pytest.mark.parametrize("operation", ["hodge", "coboundary"])
@pytest.mark.parametrize("storage", ["values", "packed"])
def test_coefficient_features_and_ordered_training_match_independent_integer_updates(files, profile, operation, storage):
    state, source, outputs, features, _ = case(files, profile, operation, storage)
    captured = inputs(state, source, outputs, features, prepare=True)
    rows = tuple(tuple((index, int(Q(output.values[coordinate, power]) * scale))
                      for index, coordinate, power, scale in row)
                 for output, row in zip(outputs, features, strict=True))
    assert captured["rows"] == rows
    assert validate_coefficient_features(captured["coefficient_features"], state=state) == rows
    prediction = infer_model(state, source, captured)
    direct = infer_model(state, source, cpp_perceptron_input(state, source, rows))
    assert tuple(prediction.values) == tuple(direct.values)
    assert prediction.execution_receipt["scores"] == direct.execution_receipt["scores"]
    weights = [1, -2]; bias = 3
    initial_digest = state.coefficient_digest
    for epoch in range(3):
        mistakes = 0
        for row, label in zip(rows, (1, -1, 1), strict=True):
            score = bias + sum(weights[index] * value for index, value in row)
            if (1 if score >= 0 else -1) != label:
                for index, value in row:
                    weights[index] += 2 * label * value
                bias += 2 * label
                mistakes += 1
        child = train_model(state, source, batch(state, source, outputs, features))
        assert tuple(child.payload["parameters"]) == (*weights, bias)
        assert child.payload["mistakes"] == mistakes and child.step == epoch + 1
        assert "coefficient_features" in child.payload["history"][-1]
        assert unpack_model(pack_model(child)).coefficient_digest == child.coefficient_digest
        state = child
    assert state.parent is not None and initial_digest != state.coefficient_digest
    assert prepare_perceptron(state, source).snapshot["weights"] == tuple(weights)


@pytest.mark.parametrize("workers", [None, 1, 2, 4])
def test_prepared_cpu_features_retain_current_parameters_and_concurrent_producers(files, workers):
    state, source, outputs, features, _ = case(files)
    captured = inputs(state, source, outputs, features, prepare=True, cpu_workers=workers,
                      rows_per_tile=256 if workers is None else 1)
    child = train_model(state, source, batch(state, source, outputs, features), steps=2)
    expected = infer_model(child, source, captured)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: infer_model(child, source, captured), range(8)))
    assert all(v.coefficient_digest == expected.coefficient_digest for v in values)
    assert expected.execution_receipt["coefficient_features"] == captured["coefficient_features"]


@pytest.mark.parametrize("mode", ["solve", "as_covector", "quadrance"])
def test_composed_vector_covector_and_scalar_features_keep_complete_metric_origins(files, mode):
    state, source, outputs, _, snapshot = case(files)
    producer = cpp_algebraic_model(snapshot.archive_owner, operator_key=5100, input_key=1100,
                                  operation=mode)
    output = infer_model(producer, snapshot.archive_owner, outputs[0])
    mapped = tuple((index, 0, index, Q(output.values[0, index]).denominator) for index in range(2))
    captured = inputs(state, source, (output,) * 3, (mapped,) * 3)
    assert len(captured["coefficient_features"]["origins"]) == 1
    saved = unpack_model(unpack_value(captured["coefficient_features"]["origins"][0]))
    assert saved.coefficient_digest == output.coefficient_digest
    assert saved.execution_receipt["input_origin"]["origin_output"]
    assert infer_model(state, source, captured).execution_receipt["coefficient_features"] == captured["coefficient_features"]


@pytest.mark.parametrize("mode", ["row_order", "row_count", "features_count", "parameter", "coordinate", "power",
    "repeat", "zero_scale", "float_scale", "bool_scale", "fractional", "overflow", "bytes", "whole_grade", "empty_features"])
def test_declared_features_reject_invalid_or_lossy_links(files, mode):
    state, source, outputs, features, snapshot = case(files)
    features = [list(row) for row in features]; keys = state.space.keys; options = {}
    if mode == "row_order": keys = keys[::-1]
    elif mode == "row_count": outputs = outputs[:-1]
    elif mode == "features_count": features = features[:-1]
    elif mode == "whole_grade": outputs = (result(files)[0],) * 3
    elif mode == "bytes": options["max_origin_bytes"] = 1
    elif mode == "repeat": features[0].append(features[0][0])
    elif mode == "empty_features":
        captured = cpp_coefficient_perceptron_input(state, source, outputs, row_keys=keys, features=((),) * 3)
        assert captured["rows"] == ((),) * 3
        assert infer_model(state, source, captured).execution_receipt["scores"] == (3,) * 3
        return
    else:
        entry = list(features[0][0])
        if mode == "parameter": entry[0] = 2
        elif mode == "coordinate": entry[1] = 1 << 32
        elif mode == "power": entry[2] = 1 << 32
        elif mode == "zero_scale": entry[3] = 0
        elif mode == "float_scale": entry[3] = 1.0
        elif mode == "bool_scale": entry[3] = True
        elif mode == "fractional": entry[3] = Q(1, 1 + abs(Q(outputs[0].values[0, 0]).numerator))
        elif mode == "overflow": entry[3] *= 1 << 128
        features[0][0] = tuple(entry)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        cpp_coefficient_perceptron_input(state, source, outputs, row_keys=keys, features=features, **options)


@pytest.mark.parametrize("mode", ["source", "primary_space", "feature_carrier", "feature_coordinates", "feature_count", "grade", "carrier",
    "extra", "rows", "producer_order", "producer_extra", "binding_row", "binding_origin", "binding_extra", "origin_bytes"])
def test_inference_and_training_reject_changed_feature_receipts(files, mode):
    state, source, outputs, features, _ = case(files)
    captured = thaw_tree(inputs(state, source, outputs, features))
    record = captured["coefficient_features"]
    if mode == "source": record["source"] = outputs[0].source.as_record()
    elif mode == "primary_space": record[mode] = ("another", state.space.keys)
    elif mode in {"feature_carrier", "feature_coordinates", "feature_count", "grade", "carrier"}: record[mode] += 1
    elif mode == "extra": record["ignored"] = True
    elif mode == "rows": captured["rows"] = ((),) * 3
    elif mode == "producer_order": record["origins"] = record["origins"][::-1]
    elif mode == "producer_extra": record["origins"] = (*record["origins"], record["origins"][0])
    elif mode == "origin_bytes": record["origins"] = (b"invalid", *record["origins"][1:])
    elif mode.startswith("binding_"):
        entry = dict(record["bindings"][0])
        if mode == "binding_row": entry["row"] = "foreign"
        elif mode == "binding_origin": entry["origin"] = 999
        else: entry["ignored"] = True
        record["bindings"] = (entry, *record["bindings"][1:])
    with pytest.raises((ValueError, TypeError, KeyError)):
        infer_model(state, source, captured)
    original = batch(state, source, outputs, features)
    captured.update(rate=2, event_cutoff=10, knowledge_cutoff=20)
    with pytest.raises((ValueError, TypeError, KeyError)):
        train_model(state, source, replace(original, inputs=captured))


@pytest.mark.parametrize("mode", ["rows", "features", "producer", "byte_limit", "extra"])
def test_portable_checkpoint_construction_rejects_changed_provenance(files, mode):
    state, source, outputs, features, _ = case(files)
    child = train_model(state, source, batch(state, source, outputs, features))
    payload = thaw_tree(child.payload); entry = payload["history"][0]
    if mode == "rows": entry["rows"] = ((),) * 3
    elif mode == "features": entry["coefficient_features"]["feature_coordinates"] += 1
    elif mode == "producer": entry["coefficient_features"]["origins"] = (b"invalid",)
    elif mode == "extra": entry["ignored"] = True
    else: entry["coefficient_features"]["max_origin_bytes"] = 1
    with pytest.raises((ValueError, TypeError)):
        replace(child, payload=payload)


def test_prepared_inputs_keep_the_actual_owned_sparse_row_cache(files, monkeypatch):
    import rexgraph.cpp_perceptron as module
    state, source, outputs, features, _ = case(files)
    captured = inputs(state, source, outputs, features, prepare=True)
    assert module._input_cache[id(captured["rows"])][0] is captured["rows"]
    expected = infer_model(state, source, captured)
    def reject(*args):
        raise AssertionError("prepared rows were normalized again")
    monkeypatch.setattr(module, "_rows", reject)
    assert infer_model(state, source, captured).coefficient_digest == expected.coefficient_digest


def test_scaled_rational_features_and_explicit_parameter_order(files):
    state, source, outputs, _, _ = case(files)
    features = tuple(((1, 0, 0, 1 / Q(output.values[0, 0])),
                      (0, 0, 0, -2 / Q(output.values[0, 0]))) for output in outputs)
    captured = inputs(state, source, outputs, features)
    assert captured["rows"] == (((1, 1), (0, -2)),) * 3
    assert infer_model(state, source, captured).execution_receipt["scores"] == (-1,) * 3


@pytest.mark.parametrize("mode", ["event", "knowledge", "label", "evaluation", "mask"])
def test_coefficient_training_obeys_declared_learner_availability(files, mode):
    state, source, outputs, features, _ = case(files)
    config = thaw_tree(state.configuration)
    times = config["availability"]
    entry = dict(times[1])
    if mode == "evaluation": entry["partition"] = "evaluation"
    elif mode != "mask": entry[{"event": "event_time", "knowledge": "knowledge_time", "label": "label_available_time"}[mode]] = 999
    config["availability"] = (times[0], entry, times[2])
    state = replace(state, configuration=config)
    captured = batch(state, source, outputs, features, observed=(True, False, True) if mode == "mask" else None)
    if mode == "mask":
        assert train_model(state, source, captured).step == 1
    else:
        with pytest.raises(ValueError): train_model(state, source, captured)


def test_aggregate_feature_history_admission_precedes_native_updates(files, monkeypatch):
    import rexgraph.cpp_coefficient_features as module
    import rexgraph.cpp_perceptron as classifier
    state, source, outputs, features, _ = case(files)
    probe = inputs(state, source, outputs, features)
    size = len(pack_value(thaw_tree(probe["coefficient_features"])))
    captured = batch(state, source, outputs, features, max_origin_bytes=size + 128)
    monkeypatch.setattr(module, "_MAX_BYTES", (size + 128) * 3 // 2)
    def reject(*args):
        raise AssertionError("native training phase entered before byte admission")
    monkeypatch.setattr(classifier, "_epoch", reject)
    with pytest.raises(ValueError, match="history byte admission"):
        train_model(state, source, captured, steps=2)
    assert state.step == 0


def test_retained_producer_cache_is_bounded_and_owns_decoded_coordinates(files):
    import rexgraph.cpp_coefficient_features as module
    for profile in (1, 2, 3, 4):
        state, source, outputs, features, _ = case(files, profile)
        captured = inputs(state, source, outputs, features)
        assert len(module._origin_cache) <= 8
        assert module._origin_cache_bytes == sum(len(raw) for raw in module._origin_cache)
        assert module._origin_cache_bytes <= 32 << 20
    before = validate_coefficient_features(captured["coefficient_features"], state=state)
    outputs[0].values.flags.writeable = True
    outputs[0].values[0, 0] += 1
    assert validate_coefficient_features(captured["coefficient_features"], state=state) == before
    with pytest.raises(ValueError, match="changed"):
        inputs(state, source, outputs, features)


@pytest.mark.parametrize("storage", ["values", "packed"])
@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_complete_training_records_and_outputs_roundtrip_in_fresh_processes(files, tmp_path, storage, suffix):
    state, source, outputs, features, _ = case(files, storage=storage)
    child = train_model(state, source, batch(state, source, outputs, features), steps=2)
    captured = inputs(child, source, outputs, features)
    output = infer_model(child, source, captured)
    record = model_record(source, child)
    record.attach_metadata(1, 0, "model_output", output)
    record.attach_metadata(1, 0, "coefficient_input", thaw_tree(captured))
    path = tmp_path / ("trained" + suffix); save(path, record)
    script = '''
import sys,json
from rexgraph.io import load
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
r=load(sys.argv[1]);state,source=read_model_record(r)
out=infer_model(state,source,r.get_metadata(1,0,"coefficient_input"))
assert out.coefficient_digest==r.get_metadata(1,0,"model_output").coefficient_digest
print(json.dumps({"state":state.coefficient_digest,"output":out.coefficient_digest}))
'''
    run = subprocess.run([sys.executable, "-c", script, str(path)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == {"state": child.coefficient_digest, "output": output.coefficient_digest}
    restored, owner = read_model_record(load(path))
    continued = train_model(restored, owner, batch(restored, owner, outputs, features))
    assert continued.step == 3


def test_rcql_agent_and_system_use_existing_integer_model_routes(files):
    from agent.hive import Hive
    from rcql.executor import Executor
    from system.serialize import json_value

    from rcql import parse
    state, source, outputs, features, _ = case(files)
    captured = inputs(state, source, outputs, features)
    training = batch(state, source, outputs, features)
    record = model_record(source, state)
    query = Executor(sources={"record": record}, params={"model": state, "input": captured, "batch": training})
    actual = query.execute(parse('FROM $record RETURN MODEL_INFER($model,$input),MODEL_TRAIN($model,$batch,2)')).values
    assert actual[0].coefficient_digest == infer_model(state, source, captured).coefficient_digest
    assert actual[1].coefficient_digest == train_model(state, source, training, steps=2).coefficient_digest
    hive = Hive("coefficient-features")
    hive.add_worker("classify", lambda value: infer_model(state, source, value), capability="predict")
    assert hive.invoke("classify", captured, record=False).coefficient_digest == actual[0].coefficient_digest
    preview = json_value(actual[0])
    assert preview["arithmetic"] == "rational" and "coefficient_features" in preview["execution_receipt"]


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_copy_record_preserves_training_observations_in_a_fresh_store_process(files, tmp_path, backend):
    from rcdb import copy_record, open_store
    state, source, outputs, features, _ = case(files, storage="packed")
    child = train_model(state, source, batch(state, source, outputs, features), steps=2)
    captured = inputs(child, source, outputs, features)
    output = infer_model(child, source, captured)
    record = model_record(source, child)
    record.attach_metadata(1, 0, "coefficient_input", thaw_tree(captured))
    record.attach_metadata(1, 0, "model_output", output)
    urls = [f"{backend}:///{tmp_path / name}" for name in ("first", "second")]
    stores = [open_store(url, **({} if backend == "sqlite" else {"read_only": False})) for url in urls]
    try:
        stores[0].put("classifier", record)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("classifier"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
    finally:
        for store in stores: store.close()
    code = '''
import json,sys
from rcdb import open_store
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
store=open_store(sys.argv[1])
try:
 record=store.get_version('classifier',1);state,source=read_model_record(record)
 output=infer_model(state,source,record.get_metadata(1,0,'coefficient_input'))
 assert output.coefficient_digest==record.get_metadata(1,0,'model_output').coefficient_digest
 assert len(state.payload['history'])==2
 print(json.dumps({'state':state.coefficient_digest,'output':output.coefficient_digest}))
finally:store.close()
'''
    actual = subprocess.check_output([sys.executable, "-I", "-c", code, urls[1]], cwd=tmp_path, text=True)
    assert json.loads(actual) == {"state": child.coefficient_digest, "output": output.coefficient_digest}
