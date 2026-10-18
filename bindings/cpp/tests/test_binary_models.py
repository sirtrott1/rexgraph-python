"""Canonical Boolean model fitting preserves source, parameter and causal contracts."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from rexgraph_cpp import NativeSource
from test_construction import declaration

from rexgraph.cpp_binary_gate import (
    cpp_binary_gate_batch,
    cpp_binary_gate_input,
    cpp_binary_gate_model,
    pack_binary_plane,
    prepare_binary_gate,
)
from rexgraph.io import load, save
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import resolve_model_provider
from rexgraph.model_runtime import infer_model, train_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value

A = (0, 0, 1, 1)
B = (0, 1, 0, 1)


def source(table=0):
    d = declaration()
    d["witnesses"] = (13,)
    d["carriers"][0]["size"] = 4
    d["metrics"][0]["diagonal"] = (2, 3, 4, 5)
    next(c for c in d["carriers"] if c["key"] == 40)["size"] = 4
    next(f for f in d["fields"] if f["key"] == 4000)["entries"] = tuple((i, (table >> i) & 1) for i in range(4))
    return NativeSource.from_declaration(d)


def availability(n=4):
    return tuple(dict(event_time=i + 1, knowledge_time=i + 1, label_available_time=i + 1,
                      partition="training") for i in range(n))


def model(owner, **kwargs):
    return cpp_binary_gate_model(owner, 10, 4000, availability(), grade=0,
        parameter_coordinates=900, time_coordinates=999, **kwargs)


def inputs(state, owner, a=A, b=B, **kwargs):
    return cpp_binary_gate_input(state, owner, pack_binary_plane(a), pack_binary_plane(b), **kwargs)


def batch(state, owner, labels, **kwargs):
    return cpp_binary_gate_batch(state, owner, pack_binary_plane(A), pack_binary_plane(B),
        pack_binary_plane(labels), event_cutoff=10, knowledge_cutoff=20, **kwargs)


@pytest.mark.parametrize("initial", range(16))
@pytest.mark.parametrize("fitted", [0, 6, 8, 9, 15])
def test_operator_parameters_recover_boolean_rules_on_original_source_rows(initial, fitted):
    owner = source(initial); state = model(owner)
    observed = inputs(state, owner, prepare=True)
    before = state.coefficient_digest
    assert state.payload["truth_table"] == initial
    assert tuple(infer_model(state, owner, observed).values) == tuple((initial >> i) & 1 for i in range(4))
    labels = tuple((fitted >> i) & 1 for i in range(4))
    child = train_model(state, owner, batch(state, owner, labels), steps=2)
    assert child.payload["truth_table"] == fitted and child.payload["last_hamming_error"] == 0
    assert state.coefficient_digest == before and state.step == 0 and child.step == 2
    assert child.configuration["parameter_states"] == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert len(child.payload["history"]) == 2
    output = infer_model(child, owner, observed)
    assert output.values.dtype == np.dtype(np.uint8) and tuple(output.values) == labels
    assert output.values.nbytes == 4 and output.execution_receipt["output_plane"] == pack_binary_plane(labels)
    assert output.execution_receipt["time_coordinates"] == child.configuration["time_coordinates"]
    assert output.grade == 0 and output.space.keys == owner.identity["coordinates"][0]
    assert unpack_model(pack_model(child)).coefficient_digest == child.coefficient_digest
    assert unpack_model(pack_model(output)).coefficient_digest == output.coefficient_digest
    assert prepare_binary_gate(child, owner).truth_table == fitted


@pytest.mark.parametrize("grade", [1, 2])
@pytest.mark.parametrize("table", range(16))
def test_higher_grade_fitting_preserves_the_complete_attaching_source(grade, table):
    owner = source(15); original = owner.archive
    count = len(owner.identity["coordinates"][grade])
    state = cpp_binary_gate_model(owner, 10 + grade, 4000, availability(count), grade=grade,
        parameter_coordinates=900, time_coordinates=999)
    first, second = pack_binary_plane(A[:count]), pack_binary_plane(B[:count])
    labels = tuple((table >> i) & 1 for i in range(count))
    captured = cpp_binary_gate_batch(state, owner, first, second, pack_binary_plane(labels),
        event_cutoff=10, knowledge_cutoff=20)
    child = train_model(state, owner, captured)
    output = infer_model(child, owner, cpp_binary_gate_input(child, owner, first, second, prepare=True))
    assert child.payload["truth_table"] == table & ((1 << count) - 1)
    assert tuple(output.values) == labels and output.grade == grade
    assert output.space.keys == owner.identity["coordinates"][grade]
    assert owner.archive == original and child.source.native == state.source.native
    restored = unpack_model(pack_model(child))
    assert prepare_binary_gate(restored, NativeSource(original)).truth_table == child.payload["truth_table"]


@pytest.mark.parametrize("mask", [(True, False, False, False), (False, True, True, False), (True, True, True, False)])
def test_partial_observations_choose_declared_lowest_table_and_preserve_absent_labels(mask):
    owner = source(15); state = model(owner)
    child = train_model(state, owner, batch(state, owner, (1, 1, 1, 1), observed=mask))
    assert child.payload["truth_table"] == sum(int(use) << i for i, use in enumerate(mask))
    assert child.payload["history"][0]["observed"] == mask


@pytest.mark.parametrize("mode", ["event", "knowledge", "label", "evaluation", "parameter", "none"])
def test_unavailable_or_unobserved_labels_cannot_fit_parameters(mode):
    owner = source(); state = model(owner)
    c = thaw_tree(state.configuration)
    if mode == "parameter": c["initial_parameter_knowledge_time"] = 99
    elif mode != "none":
        entry = dict(c["availability"][0])
        if mode == "evaluation": entry["partition"] = "evaluation"
        else: entry[{"event": "event_time", "knowledge": "knowledge_time", "label": "label_available_time"}[mode]] = 99
        c["availability"] = (entry, *c["availability"][1:])
    state = replace(state, configuration=c)
    with pytest.raises(ValueError):
        train_model(state, owner, batch(state, owner, (0, 1, 1, 0), observed=(False,) * 4 if mode == "none" else None))


@pytest.mark.parametrize("mode", ["profile", "source", "primary_space", "channels", "lanes", "padding", "bytes", "extra"])
def test_complete_input_declarations_reject_wrong_rows_channels_or_packed_shape(mode):
    owner = source(); state = model(owner); observed = thaw_tree(inputs(state, owner))
    if mode == "profile": observed[mode] = "inferred"
    elif mode == "source": observed[mode] = model(source(1)).source.as_record()
    elif mode == "primary_space": observed[mode] = (state.space.name, state.space.keys[::-1])
    elif mode == "channels": observed[mode] = ("b", "a")
    elif mode == "lanes": observed[mode] = 3
    elif mode == "padding": observed["first"] = b"\xff"
    elif mode == "bytes": observed["first"] = b""
    else: observed["ignored"] = True
    with pytest.raises(ValueError): infer_model(state, owner, observed)


@pytest.mark.parametrize("mode", ["table", "loss", "steps", "parent", "history", "history_extra", "engine", "states", "channels", "bytes"])
def test_portable_state_and_native_replay_reject_changed_parameter_history(mode):
    owner = source(); state = model(owner)
    child = train_model(state, owner, batch(state, owner, (0, 1, 1, 0)))
    p, c = thaw_tree(child.payload), thaw_tree(child.configuration); options = {}
    if mode == "table": p["truth_table"] = 9
    elif mode == "loss": p["last_hamming_error"] = 1
    elif mode == "steps": options["step"] = 2
    elif mode == "parent": options["parent"] = None
    elif mode == "history": p["history"] = ()
    elif mode == "history_extra": p["history"][0]["ignored"] = True
    elif mode == "engine": c["engine_receipt"]["binding_digest"] = "0" * 64
    elif mode == "states": c["parameter_states"] = c["parameter_states"][::-1]
    elif mode == "channels": c["input_channels"] = ("a", "a")
    else: c["max_history_bytes"] = 1
    with pytest.raises(ValueError):
        altered = replace(child, payload=p, configuration=c, **options)
        prepare_binary_gate(altered, owner)


@pytest.mark.parametrize("mode", ["values", "dtype", "truth_table", "plane", "grade", "source", "extra",
    "parameter_coordinates", "time_coordinates", "parameter_step"])
def test_portable_output_construction_checks_exact_boolean_values_and_provenance(mode):
    owner = source(6); state = model(owner); output = infer_model(state, owner, inputs(state, owner))
    receipt = thaw_tree(output.execution_receipt); options = {}
    if mode == "values": options["values"] = np.zeros(4, dtype=np.uint8)
    elif mode == "dtype": options["values"] = output.values.astype(np.int64)
    elif mode == "truth_table": receipt[mode] = 9
    elif mode == "plane": receipt["output_plane"] = b"\0"
    elif mode == "grade": options["grade"] = 1
    elif mode == "source": options["source"] = model(source(1)).source
    elif mode in {"parameter_coordinates", "time_coordinates"}: receipt[mode] = 0
    elif mode == "parameter_step": receipt[mode] = 1025
    else: receipt["ignored"] = True
    with pytest.raises((ValueError, TypeError)):
        replace(output, execution_receipt=receipt, **options)


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_complete_model_io_and_native_fit_continuation_in_fresh_processes(tmp_path, suffix):
    owner = source(); state = model(owner); child = train_model(state, owner, batch(state, owner, (0, 1, 1, 0)))
    observed = inputs(child, owner); output = infer_model(child, owner, observed)
    record = model_record(owner, child)
    record.attach_metadata(1, 0, "binary_input", thaw_tree(observed)); record.attach_metadata(1, 0, "model_output", output)
    path = tmp_path / ("gate" + suffix); save(path, record)
    code = '''
import json,sys
from rexgraph.io import load
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
record=load(sys.argv[1]);state,source=read_model_record(record)
output=infer_model(state,source,record.get_metadata(1,0,'binary_input'))
assert output.coefficient_digest==record.get_metadata(1,0,'model_output').coefficient_digest
print(json.dumps({'state':state.coefficient_digest,'output':output.coefficient_digest}))
'''
    result = subprocess.check_output([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, text=True)
    assert json.loads(result) == {"state": child.coefficient_digest, "output": output.coefficient_digest}
    restored, source_owner = read_model_record(load(path))
    assert train_model(restored, source_owner, batch(restored, source_owner, (1, 1, 1, 1))).payload["truth_table"] == 15


def test_prepared_planes_are_shared_and_original_complex_bytes_do_not_change():
    owner = source(); state = model(owner); original = owner.archive
    observed = inputs(state, owner, prepare=True)
    child = train_model(state, owner, batch(state, owner, (0, 1, 1, 0)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: infer_model(child, owner, observed), range(8)))
    assert len({output.coefficient_digest for output in outputs}) == 1
    assert owner.archive == original
    with pytest.raises(NotImplementedError): resolve_model_provider(child, "transport")


def test_declared_byte_and_lane_admission_precedes_fit():
    owner = source()
    with pytest.raises(ValueError): model(owner, max_lanes=3)
    with pytest.raises(ValueError, match="history byte admission"):
        model(owner, max_history_bytes=1)
    state = model(owner, max_history_bytes=len(pack_value(())))
    with pytest.raises(ValueError, match="history byte admission"):
        train_model(state, owner, batch(state, owner, (0, 1, 1, 0)))


@pytest.mark.parametrize("channels", ["ab", b"ab", ("a", "a"), ("a", ""), ("a",)])
def test_channels_require_two_explicit_distinct_names(channels):
    with pytest.raises(ValueError):
        model(source(), input_channels=channels)


def test_rcql_agent_and_system_use_the_existing_model_routes():
    from agent.hive import Hive
    from system.serialize import json_value

    from rcql import Executor, parse
    owner = source(); state = model(owner); observed = inputs(state, owner)
    training = batch(state, owner, (0, 1, 1, 0)); record = model_record(owner, state)
    config = dict(carrier=10, parameter_key=4000, availability=availability(), grade=0, parameter_coordinates=900, time_coordinates=999)
    created = Executor(sources={"record": record}, params={"config": config}).execute(parse(
        'FROM $record RETURN MODEL_INIT("cpp_binary_gate",$config)')).values[0]
    assert created.adapter == "cpp_binary_gate"
    actual = Executor(sources={"record": record}, params={"model": state, "input": observed, "batch": training}).execute(parse(
        'FROM $record RETURN MODEL_INFER($model,$input),MODEL_TRAIN($model,$batch)')).values
    assert actual[0].coefficient_digest == infer_model(state, owner, observed).coefficient_digest
    assert actual[1].coefficient_digest == train_model(state, owner, training).coefficient_digest
    hive = Hive("binary-gate"); hive.add_worker("gate", lambda value: infer_model(state, owner, value), capability="predict")
    assert hive.invoke("gate", observed, record=False).coefficient_digest == actual[0].coefficient_digest
    assert json_value(actual[0])["execution_receipt"]["truth_table"] == 0
    assert json_value(actual[1])["adapter"] == "cpp_binary_gate"


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_copy_record_preserves_operator_fits_in_a_fresh_store_process(tmp_path, backend):
    from rcdb import copy_record, open_store
    owner = source(); state = model(owner); child = train_model(state, owner, batch(state, owner, (0, 1, 1, 0)))
    observed = inputs(child, owner); output = infer_model(child, owner, observed)
    record = model_record(owner, child); record.attach_metadata(1, 0, "binary_input", thaw_tree(observed)); record.attach_metadata(1, 0, "model_output", output)
    urls = [f"{backend}:///{tmp_path / name}" for name in ("first", "second")]
    stores = [open_store(url, **({} if backend == "sqlite" else {"read_only": False})) for url in urls]
    try:
        stores[0].put("gate", record)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("gate"), return_receipt=True)
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
 record=store.get_version('gate',1);state,source=read_model_record(record)
 output=infer_model(state,source,record.get_metadata(1,0,'binary_input'))
 assert output.coefficient_digest==record.get_metadata(1,0,'model_output').coefficient_digest
 print(json.dumps({'state':state.coefficient_digest,'output':output.coefficient_digest}))
finally:store.close()
'''
    result = subprocess.check_output([sys.executable, "-I", "-c", code, urls[1]], cwd=tmp_path, text=True)
    assert json.loads(result) == {"state": child.coefficient_digest, "output": output.coefficient_digest}
