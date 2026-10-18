"""Exact metric model lifecycle on complete original algebraic grade inputs."""
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeAlgebraicArchive,
    NativeAlgebraicGraphDatabase,
    NativeAlgebraicGraphRevision,
    NativeAlgebraicSnapshot,
    NativeSource,
)
from test_algebraic_grade_model import AVAILABLE, oracle, snapshot
from test_algebraic_metric import ACTIONS, exact_actions, metrics

from rexgraph.cpp_algebraic_grade_metric_model import (
    _cache,
    _prepare,
    clear_algebraic_grade_metric_model_cache,
    cpp_algebraic_grade_metric_input,
    cpp_algebraic_grade_metric_model,
    enable_cpp_algebraic_grade_metric_provider,
    infer_algebraic_grade_metric_model,
)
from rexgraph.cpp_algebraic_grade_model import cpp_algebraic_grade_model, grade_model_restore_limits
from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import model_provider_inventory, resolve_model_provider
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value, unpack_value


@pytest.fixture(scope="session")
def files():
    return Path(os.environ["REXGRAPH_CPP_METRIC_FIXTURES"])


def request(files, profile=13, grade=1, steps=0):
    return {"grade": grade, "record_keys": oracle(files, profile)[grade]["keys"], "boundary_steps": steps}


def model(files, profile=13, grade=1, operation="hodge", steps=0, **options):
    source = snapshot(files, profile)
    req = request(files, profile, grade, steps)
    state = cpp_algebraic_grade_metric_model(source, request=req, metric_keys=metrics(source),
        operation=operation, other_request=req if operation == "inner" else None,
        availability=AVAILABLE, **options)
    return state, source


@pytest.mark.parametrize("profile,grade,operation", ACTIONS)
def test_complete_metric_model_matches_independent_exact_oracle_and_roundtrips(files, profile, grade, operation):
    state, source = model(files, profile, grade, operation)
    output = infer_model(state, source)
    x = tuple(tuple(Fraction(q) for q in row) for row in oracle(files, profile)[grade]["coordinates"])
    expected = exact_actions(files, profile, grade, x)[operation]
    assert tuple(tuple(row) for row in output.values) == expected
    assert all(type(q) is Fraction for q in output.values.flat)
    assert output.values.shape[1] == source.archive_owner.field_degree
    assert output.grade == (None if operation in {"inner", "adjacent_quadrance"} else grade + (operation == "coboundary"))
    for value in (state, output):
        restored = unpack_model(unpack_value(pack_value(pack_model(value, native=True))))
        assert restored.coefficient_digest == value.coefficient_digest
    saved_state, saved_source = read_model_record(model_record(source, state))
    assert infer_model(saved_state, saved_source).coefficient_digest == output.coefficient_digest


@pytest.mark.parametrize("profile", [1, 9, 13, 14, 15])
@pytest.mark.parametrize("operation", ["hodge", "coboundary", "inner", "adjacent_quadrance"])
def test_boundary_inputs_keep_full_portable_model_origins(files, profile, operation):
    state, source = model(files, profile, 2, operation, 1)
    upstream = cpp_algebraic_grade_model(source, grade=2, record_keys=oracle(files, profile)[2]["keys"],
        boundary_steps=1, availability=AVAILABLE)
    origin = infer_model(upstream, source)
    inputs = {"input": origin, "other": origin} if operation == "inner" else origin
    value = infer_model(state, source, inputs)
    x = tuple(tuple(row) for row in origin.values)
    assert tuple(tuple(row) for row in value.values) == exact_actions(files, profile, 1, x)[operation]
    r = value.execution_receipt
    decoded = unpack_model(unpack_value(r["input_origins"]["input"]))
    assert decoded.coefficient_digest == origin.coefficient_digest
    assert len(r["observation"]["input"]["lineage"]) == 2
    assert unpack_model(pack_model(value)).coefficient_digest == value.coefficient_digest
    default = infer_model(state, source)
    native = cpp_algebraic_grade_metric_input(state, source)
    assert infer_model(state, source, native).coefficient_digest == default.coefficient_digest
    assert value.coefficient_digest != default.coefficient_digest


@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
@pytest.mark.parametrize("operation", ["hodge", "coboundary", "inner", "adjacent_quadrance"])
def test_storage_and_core_only_decode_preserve_state_output_and_origins(files, tmp_path, suffix, operation):
    state, source = model(files, operation=operation, grade=2, steps=1)
    parent = cpp_algebraic_grade_model(source, grade=2, record_keys=[1200], boundary_steps=1, availability=AVAILABLE)
    origin = infer_model(parent, source)
    inputs = {"input": origin, "other": origin} if operation == "inner" else origin
    output = infer_model(state, source, inputs)
    record = model_record(source, state); record.attach_metadata(1, 0, "output", output)
    path = tmp_path / f"metric-model.{suffix}"
    save_model(path, record)
    restored = load_model(path)
    restored_state, restored_source = read_model_record(restored)
    assert restored_state.coefficient_digest == state.coefficient_digest
    assert restored.get_metadata(1, 0, "output").coefficient_digest == output.coefficient_digest
    assert infer_model(restored_state, restored_source, inputs).coefficient_digest == output.coefficient_digest
    code = '''
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rexgraph_cpp','torch'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from rexgraph.io import load
record=load(sys.argv[1]); state=record.get_metadata(1,0,'model'); output=record.get_metadata(1,0,'output')
state.check_state(); output.check_state()
print(json.dumps({'digest':output.coefficient_digest,'adapter':state.adapter}))
'''
    actual = subprocess.check_output([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, text=True)
    assert json.loads(actual)["digest"] == output.coefficient_digest


@pytest.mark.parametrize("case", ["request", "steps", "layout", "other", "kind", "output", "metrics", "provider", "profile", "policy", "boolean", "payload", "extra", "axes", "parent"])
def test_closed_state_rejects_false_declarations(files, case):
    state, _ = model(files)
    c = thaw_tree(state.configuration); options = {"configuration": c}
    if case == "request": c["request"]["record_keys"] = ()
    elif case == "steps": c["request"]["boundary_steps"] = 2
    elif case == "layout": c["layouts"][0]["stalk_offsets"] = (0, 99)
    elif case == "other": c["other_request"] = c["request"]
    elif case == "kind": c["output_kind"] = "scalar"
    elif case == "output": c["output_layout"]["grade"] = 0
    elif case == "metrics": c["metric_keys"] = (5000,) * len(c["metric_keys"])
    elif case == "provider": c["execution_provider"]["version"] = "2"
    elif case == "profile": c["profile"] = "unknown"
    elif case == "policy": c["execution_limits"]["records"] = 0
    elif case == "boolean": c["max_retained_coordinates"] = True
    elif case == "payload": options["payload"] = {"trainable": True}
    elif case == "extra": c["notes"] = "unknown"
    elif case == "axes": options["output_axes"] = ()
    else: options["parent"] = state.coefficient_digest
    with pytest.raises((TypeError, ValueError)): replace(state, **options)


@pytest.mark.parametrize("case", ["values", "grade", "operation", "engine", "policy", "retained", "basis", "extra", "origin", "origin_type", "origin_bound", "other_origin"])
def test_closed_output_rejects_false_receipts_and_origins(files, case):
    state, source = model(files)
    output = infer_model(state, source)
    r = thaw_tree(output.execution_receipt); options = {"execution_receipt": r}
    if case == "values": options["values"] = output.values + 1
    elif case == "grade": options["grade"] = 0
    elif case == "operation": r["operation"] = "inner"
    elif case == "engine": r["engine"]["engine_digest"] = "0" * 64
    elif case == "policy": r["execution_limits"]["scalar_work"] += 1
    elif case == "retained": r["max_retained_coordinates"] += 1
    elif case == "basis": r["observation"]["input"]["coefficient_basis"] = "floating"
    elif case == "extra": r["notes"] = "unknown"
    elif case == "origin": r["input_origins"]["input"] = pack_value({"type": "unknown"})
    elif case == "origin_type": r["input_origins"]["input"] = "bytes"
    elif case == "origin_bound":
        r["input_origins"]["input"] = b"bad"; r["origin_max_bytes"] = 0
    else: r["input_origins"]["other"] = b"bad"
    with pytest.raises((TypeError, ValueError, KeyError)): replace(output, **options)


def test_actual_sources_input_layouts_and_explicit_pairing_are_required(files):
    state, source = model(files)
    native = cpp_algebraic_grade_metric_input(state, source)
    with pytest.raises(ValueError): infer_model(state, snapshot(files), native)
    for invalid in (np.ones((6, 3)), [1, 2, 3], {"coefficients": native.coefficients}):
        with pytest.raises(TypeError): infer_model(state, source, invalid)
    with pytest.raises(ValueError): cpp_algebraic_grade_metric_model(source, request=request(files), metric_keys=metrics(source), operation="inner", availability=AVAILABLE)
    pairing, _ = model(files, operation="inner")
    with pytest.raises(ValueError): infer_model(pairing, source, {"input": native})
    wrong = cpp_algebraic_grade_model(source, grade=0, record_keys=oracle(files)[0]["keys"], boundary_steps=0, availability=AVAILABLE)
    with pytest.raises(ValueError): infer_model(state, source, infer_model(wrong, source))


def test_preparation_cache_checks_policy_cancellation_and_actual_publications(files):
    revision = NativeAlgebraicGraphRevision(NativeSource((files / "profile-13.rfcxs").read_bytes()), (9631, 1, 2), sequence=31)
    db = NativeAlgebraicGraphDatabase(revision, availability=AVAILABLE)
    source = db.archive(9, availability=AVAILABLE)
    state = cpp_algebraic_grade_metric_model(source, request=request(files), metric_keys=metrics(source), availability=AVAILABLE)
    field, tensor, _ = _prepare(state, source)
    assert _prepare(state, source)[0] is field
    resources = {"records": "binding records", "rational_slots": "reference rational slots",
                 "sparse_entries": "reference sparse entries", "scalar_work": "reference scalar work envelope",
                 "integer_bits": "exact integer growth bound"}
    for name, resource in resources.items():
        needed = max(row[name] for row in state.configuration["requirements"])
        if needed:
            with pytest.raises(ExecutionLimitError) as error: infer_algebraic_grade_metric_model(state, source, execution_limits={name: needed - 1})
            assert error.value.resource == resource
    with pytest.raises(RuntimeError): infer_algebraic_grade_metric_model(state, source, max_retained_coordinates=0)
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): infer_algebraic_grade_metric_model(state, source, cancellation=token)
    with pytest.raises(ExecutionCancelled): cpp_algebraic_grade_metric_input(state, source, cancellation=token)
    with pytest.raises(TypeError): infer_algebraic_grade_metric_model(state, source, cancellation=True)
    with pytest.raises(ValueError): infer_algebraic_grade_metric_model(state, source, availability=(9603, 30))
    fresh = snapshot(files)
    assert _prepare(state, fresh)[0] is not field
    alias = db.archive(9, availability=AVAILABLE)
    assert _prepare(state, alias)[0] is field
    narrowed = _prepare(state, source, execution_limits={"scalar_work": 10000})
    assert narrowed[0] is not field and narrowed[1].execution_limits["scalar_work"] == 10000
    with ThreadPoolExecutor(max_workers=4) as pool:
        digests = list(pool.map(lambda _: infer_model(state, source).coefficient_digest, range(12)))
    assert len(set(digests)) == 1
    for _ in range(10): _prepare(state, snapshot(files))
    assert len(_cache) <= 8
    clear_algebraic_grade_metric_model_cache(); gc.collect()
    assert tensor.matches_source(source)
    assert infer_model(state, source).coefficient_digest == digests[0]


def test_reader_policy_intersects_and_fixed_provider_has_no_learning_callbacks(files):
    state, source = model(files)
    record = model_record(source, state)
    with pytest.raises(ValueError): grade_model_restore_limits(state, {"unknown": 1})
    with pytest.raises((ValueError, ExecutionLimitError)): read_model_record(record, algebraic_limits={"max_source_coordinates": 0})
    smaller = state.configuration["archive_limits"]["max_file_bytes"] - 1
    restored, owner = read_model_record(record, algebraic_limits={"max_file_bytes": smaller})
    assert infer_model(restored, owner).values.tolist() == infer_model(state, source).values.tolist()
    inventory = next(row for row in model_provider_inventory() if row["name"] == "rexgraph.cpp.algebraic_grade_metric")
    assert set(inventory["operations"]) == {"create", "prepare", "infer"}
    enable_cpp_algebraic_grade_metric_provider()
    with pytest.raises(NotImplementedError): resolve_model_provider(state, "train")
    with pytest.raises(NotImplementedError): resolve_model_provider(state, "transport")


def test_rcql_agent_and_system_preserve_exact_metric_outputs(files):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    state, source = model(files)
    saved = algebraic_snapshot_record(source)
    config = {"request": request(files), "metric_keys": metrics(source), "availability": AVAILABLE}
    initialized = Executor(sources={"record": saved}, params={"config": config}).execute(parse('FROM $record RETURN MODEL_INIT("cpp_algebraic_grade_metric",$config)')).values[0]
    query = parse("FROM $record RETURN MODEL_INFER($state), MODEL_RECORD($state), MODEL_INFO($state)")
    result = Executor(sources={"record": saved}, params={"state": initialized}).execute(query)
    assert result.values[0].values.tolist() == infer_model(state, source).values.tolist()
    restored, owner = read_model_record(result.values[1])
    assert infer_model(restored, owner).coefficient_digest == result.values[0].coefficient_digest
    assert result.values[2]["operation"] == "hodge" and result.values[2]["trainable"] is False
    runtime = RCQLRuntime(); runtime.register("record", saved, policy=SourcePolicy.allow("read"))
    assert runtime.execute(query, params={"state": initialized}).values[0].coefficient_digest == result.values[0].coefficient_digest
    denied = RCQLRuntime(); denied.register("record", saved, policy=SourcePolicy.allow())
    with pytest.raises(PermissionError): denied.execute(query, params={"state": initialized})
    hive = Hive("metric-model"); hive.add_worker("metric", lambda inputs: infer_model(state, source, inputs), capability="transform")
    assert hive.invoke("metric", None, record=False).values.tolist() == result.values[0].values.tolist()
    for value in (state, result.values[0]): json.dumps(json_value(value, max_values=2), allow_nan=False)
    for name in ("MODEL_FIELD", "MODEL_INPUT"):
        text = f"FROM $record RETURN {name}($output" + (",0)" if name == "MODEL_STALK" else ")")
        with pytest.raises((TypeError, ValueError)): Executor(sources={"record": saved}, params={"output": result.values[0]}).execute(parse(text))


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_rcdb_copy_and_fresh_process_restore_metric_model(files, tmp_path, backend):
    from rcdb import copy_record, open_store
    state, source = model(files, operation="adjacent_quadrance")
    expected = infer_model(state, source).coefficient_digest
    urls = [f"{backend}:///{tmp_path / name}" for name in ("first", "second")]
    stores = [open_store(url, **({} if backend == "sqlite" else {"read_only": False})) for url in urls]
    try:
        stores[0].put("model", model_record(source, state))
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("model"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        restored, owner = read_model_record(stores[1].get_version("model", 1))
        assert infer_model(restored, owner).coefficient_digest == expected
    finally:
        for store in stores: store.close()
    code = '''
import json,sys
from rcdb import open_store
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
store=open_store(sys.argv[1])
try:
 state,source=read_model_record(store.get_version('model',1))
 print(json.dumps({'digest':infer_model(state,source).coefficient_digest}))
finally: store.close()
'''
    actual = subprocess.check_output([sys.executable, "-I", "-c", code, urls[1]], cwd=tmp_path, text=True)
    assert json.loads(actual)["digest"] == expected


@pytest.mark.parametrize("profile", [1, 9, 13, 14, 15])
def test_pairing_requests_can_prepare_different_boundary_lineages(files, profile):
    from test_algebraic_metric import add, conjugate, declared_oracle, matvec, multiply
    source = snapshot(files, profile)
    state = cpp_algebraic_grade_metric_model(source, request=request(files, profile, 1, 1),
        other_request=request(files, profile, 0), metric_keys=metrics(source), operation="inner", availability=AVAILABLE)
    inputs = cpp_algebraic_grade_metric_input(state, source)
    p, involution, _, matrices, _ = declared_oracle(str(files), profile)
    expected = (Fraction(0),) * source.archive_owner.field_degree
    for a, b in zip(inputs["input"].coefficients, matvec(matrices[0], inputs["other"].coefficients, p), strict=True):
        expected = add(expected, multiply(conjugate(a, involution, p), b, p))
    value = infer_model(state, source)
    assert tuple(value.values[0]) == expected
    assert len(value.execution_receipt["observation"]["input"]["lineage"]) == 2
    assert len(value.execution_receipt["observation"]["other"]["lineage"]) == 1


def test_valid_but_unrelated_model_origins_and_changed_saved_prefix_are_rejected(files):
    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
    state, source = model(files)
    output = infer_model(state, source)
    parent = cpp_algebraic_grade_model(source, grade=2, record_keys=[1200], boundary_steps=1, availability=AVAILABLE)
    false_origin = infer_model(parent, source)
    receipt = thaw_tree(output.execution_receipt)
    receipt["input_origins"]["input"] = pack_value(pack_model(false_origin, native=True))
    with pytest.raises(ValueError): replace(output, execution_receipt=receipt)
    record = model_record(source, state); cache = _AlgebraicRestoreCache()
    read_model_record(record, _algebraic_cache=cache)
    record.get_metadata(1, 0, "model_source").get_metadata(1, 0, "algebraic_archive")["archive"][-1] ^= 1
    with pytest.raises(ValueError): read_model_record(record, _algebraic_cache=cache)


@pytest.mark.parametrize("case", ["engine", "keys", "output"])
def test_cache_miss_checks_semantic_engine_and_actual_declared_carriers(files, case):
    state, source = model(files, operation="coboundary")
    c = thaw_tree(state.configuration)
    if case == "engine": c["engine_receipt"]["engine_digest"] = "0" * 64
    elif case == "keys": c["request"]["record_keys"] = (1000, 1001, 1002)
    else:
        c["output_layout"]["domain"] = (2,); c["output_layout"]["stalk_offsets"] = (0, 2)
    forged = replace(state, configuration=c)
    with pytest.raises((ValueError, RuntimeError)): infer_model(forged, source)


def test_standalone_metric_publication_roundtrip_keeps_original_source_kind(files):
    source = NativeAlgebraicSnapshot(NativeAlgebraicArchive((files / "profile-13.rfcxa").read_bytes()), namespace=9603, sequence=31)
    state = cpp_algebraic_grade_metric_model(source, request=request(files), metric_keys=metrics(source), availability=AVAILABLE)
    restored, owner = read_model_record(model_record(source, state))
    assert owner.graph_origin is None
    assert infer_model(restored, owner).coefficient_digest == infer_model(state, source).coefficient_digest


def test_reordered_execution_policies_keep_canonical_resource_order(files):
    state, source = model(files, grade=2, steps=1)
    c = thaw_tree(state.configuration); c["execution_limits"] = dict(reversed(tuple(c["execution_limits"].items())))
    state = replace(state, configuration=c)
    for name, resource in {"records": "binding records", "rational_slots": "reference rational slots",
        "sparse_entries": "reference sparse entries", "scalar_work": "reference scalar work envelope",
        "integer_bits": "exact integer growth bound"}.items():
        required = max(row[name] for row in c["requirements"])
        assert required > 0
        with pytest.raises(ExecutionLimitError) as error: infer_algebraic_grade_metric_model(state, source, execution_limits={name: required - 1})
        assert error.value.resource == resource
