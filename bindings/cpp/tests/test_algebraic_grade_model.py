"""Whole grade model lifecycle with exact native ownership and polynomial coordinates."""
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

from rexgraph.cpp_algebraic_grade_model import (
    _cache,
    _prepare,
    clear_algebraic_grade_model_cache,
    cpp_algebraic_grade_input,
    cpp_algebraic_grade_model,
    grade_model_restore_limits,
    infer_algebraic_grade_model,
)
from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_contract import model_provider_inventory
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree

AVAILABLE = (9603, 31)
CASES = [(profile, grade) for profile in range(1, 16) for grade in range(5 if profile <= 10 else 3)]


@pytest.fixture(scope="session")
def files():
    return Path(os.environ["REXGRAPH_CPP_GRADE_FIXTURES"])


def snapshot(files, profile=13):
    native = NativeSource((files / f"profile-{profile}.rfcxs").read_bytes())
    revision = NativeAlgebraicGraphRevision(native, (9631, 1, 2), sequence=31)
    return NativeAlgebraicGraphDatabase(revision, availability=AVAILABLE).archive(9, availability=AVAILABLE)


def oracle(files, profile=13):
    return [json.loads(line) for line in (files / f"profile-{profile}.jsonl").read_text().splitlines()][1:]


def model(files, profile=13, grade=2, steps=1, **kwargs):
    source = snapshot(files, profile)
    state = cpp_algebraic_grade_model(source, grade=grade, record_keys=oracle(files, profile)[grade]["keys"],
        boundary_steps=steps, availability=AVAILABLE, **kwargs)
    return state, source


def fractions(rows):
    return tuple(tuple(Fraction(q) for q in row) for row in rows)


@pytest.mark.parametrize("profile,grade", CASES)
def test_all_original_grades_match_independent_occurrence_oracle_and_model_codec(files, profile, grade):
    rows = oracle(files, profile)
    steps = (min(grade, 2) if "boundary" in rows[grade] else 0)
    state, source = model(files, profile, grade, steps)
    output = infer_model(state, source)
    expected = rows[grade]["coordinates" if steps == 0 else "boundary" if steps == 1 else "twice"]
    assert tuple(tuple(row) for row in output.values) == fractions(expected)
    assert output.grade == grade - steps and output.arithmetic == "algebraic"
    assert output.values.shape == (state.configuration["layouts"][-1]["domain"][0], source.archive_owner.field_degree)
    assert all(type(q) is Fraction for q in output.values.flat)
    assert output.axes == state.output_axes
    receipt = output.execution_receipt
    assert receipt["observation"]["publication"] == state.configuration["publication"]
    assert len(receipt["observation"]["lineage"]) == steps + 1
    assert receipt["observation"]["lineage"][0] == receipt["input_observation"]["lineage"][0]
    for value in (state, output):
        restored = unpack_model(pack_model(value))
        assert restored.coefficient_digest == value.coefficient_digest
    restored_state, restored_source = read_model_record(model_record(source, state))
    assert infer_model(restored_state, restored_source).coefficient_digest == output.coefficient_digest


@pytest.mark.parametrize("profile", [1, 9, 13, 14, 15])
def test_composable_outputs_replay_complete_input_lineage(files, profile):
    first, source = model(files, profile)
    row = oracle(files, profile)[1]
    second = cpp_algebraic_grade_model(source, grade=1, record_keys=row["keys"], boundary_steps=1, availability=AVAILABLE)
    intermediate = infer_model(first, source)
    output = infer_model(second, source, intermediate)
    assert tuple(tuple(row) for row in output.values) == fractions(oracle(files, profile)[2]["twice"])
    assert len(output.execution_receipt["input_observation"]["lineage"]) == 2
    assert len(output.execution_receipt["observation"]["lineage"]) == 3
    rebound = cpp_algebraic_grade_input(second, source, intermediate)
    assert rebound.matches_source(source) and rebound.input_owner is not None
    assert infer_model(second, source, rebound).coefficient_digest == output.coefficient_digest


@pytest.mark.parametrize("grade", range(5))
def test_supported_zero_and_empty_grades_preserve_exact_axes(files, grade):
    state, source = model(files, 2, grade, 0)
    result = infer_model(state, source)
    assert all(q == 0 for q in result.values.flat)
    assert len(result.space.keys) == result.values.shape[0]
    assert len(result.axes[0].keys) == 3
    if grade <= 2:
        empty, empty_source = model(files, 15, grade, grade)
        value = infer_model(empty, empty_source)
        assert value.values.shape == (0, 3)
        assert len(value.execution_receipt["observation"]["lineage"]) == grade + 1


@pytest.mark.parametrize("profile", [1, 9, 13, 14, 15])
@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_model_storage_reconstructs_selected_publication_and_exact_outputs(files, tmp_path, profile, suffix):
    state, source = model(files, profile)
    result = infer_model(state, source)
    saved = model_record(source, state)
    path = tmp_path / f"grade-model.{suffix}"; save_model(path, saved)
    record = load_model(path)
    restored_state, restored_source = read_model_record(record)
    assert restored_source.graph_origin == source.graph_origin
    assert restored_source.graph_revision_owner.image == source.graph_revision_owner.image
    assert restored_state.coefficient_digest == state.coefficient_digest
    assert infer_model(restored_state, restored_source).coefficient_digest == result.coefficient_digest
    with pytest.raises(ValueError): load_model(path, algebraic_limits={"max_file_bytes": 1})


def test_actual_owners_and_standalone_publications_remain_distinct(files):
    state, source = model(files)
    fresh = snapshot(files)
    field = cpp_algebraic_grade_input(state, source)
    with pytest.raises(ValueError): infer_model(state, fresh, field)
    archive = NativeAlgebraicArchive((files.parent / "grade-fixtures/profile-13.rfcxa").read_bytes())
    standalone = NativeAlgebraicSnapshot(archive, namespace=9603, sequence=31)
    with pytest.raises(ValueError): infer_model(state, standalone)
    simple = cpp_algebraic_grade_model(standalone, grade=2, record_keys=[1200], availability=AVAILABLE)
    before = infer_model(simple, standalone)
    saved = model_record(standalone, simple)
    restored_state, restored_source = read_model_record(saved)
    assert restored_source.graph_origin is None
    assert infer_model(restored_state, restored_source).coefficient_digest == before.coefficient_digest
    for invalid in ([1, 2, 3], np.ones((1, 3)), {"coefficients": before.values}):
        with pytest.raises(TypeError): infer_model(simple, standalone, invalid)


@pytest.mark.parametrize("case", ["layout", "keys", "grade", "step", "profile", "provider", "policy", "bits", "publication", "payload", "extra", "axes"])
def test_closed_model_state_rejects_inconsistent_declarations(files, case):
    state, _ = model(files)
    c = thaw_tree(state.configuration)
    options = {"configuration": c}
    if case == "layout": c["layouts"][0]["stalk_offsets"] = (0, 99)
    elif case == "keys": c["record_keys"] = ()
    elif case == "grade": c["grade"] = 1
    elif case == "step": c["boundary_steps"] = 3
    elif case == "profile": c["profile"] = "unknown"
    elif case == "provider": c["execution_provider"]["version"] = "2"
    elif case == "policy": c["execution_limits"]["scalar_work"] = 0
    elif case == "bits": c["execution_limits"]["integer_bits"] = True
    elif case == "publication": c["publication"]["namespace"] = 999
    elif case == "payload": options["payload"] = {"trainable": True}
    elif case == "extra": c["private_notes"] = "unknown"
    else: options["output_axes"] = ()
    with pytest.raises((ValueError, TypeError)): replace(state, **options)


@pytest.mark.parametrize("case", ["values", "grade", "lineage", "steps", "basis", "policy", "extra"])
def test_model_output_requires_complete_exact_observation_and_lineage(files, case):
    state, source = model(files)
    output = infer_model(state, source)
    r = thaw_tree(output.execution_receipt)
    options = {"execution_receipt": r}
    if case == "values": options["values"] = output.values + 1
    elif case == "grade": options["grade"] = 0
    elif case == "lineage": r["input_observation"] = r["observation"]
    elif case == "steps": r["boundary_steps"] = 0
    elif case == "basis": r["observation"]["coefficient_basis"] = "floating"
    elif case == "policy": r["execution_limits"]["scalar_work"] += 1
    else: r["notes"] = "unknown"
    with pytest.raises((ValueError, TypeError)): replace(output, **options)


def test_cache_and_reused_plans_recheck_reader_bounds_and_cancellation(files):
    state, source = model(files)
    first, output = _prepare(state, source)
    assert _prepare(state, source)[0] is first
    for name in ("records", "rational_slots", "sparse_entries", "scalar_work", "integer_bits"):
        requirements = max(row[name] for row in state.configuration["requirements"])
        if requirements:
            with pytest.raises(ExecutionLimitError): infer_algebraic_grade_model(state, source, execution_limits={name: requirements - 1})
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled): infer_algebraic_grade_model(state, source, cancellation=token)
    with pytest.raises(ExecutionCancelled): cpp_algebraic_grade_input(state, source, cancellation=token)
    with pytest.raises(ExecutionCancelled):
        cpp_algebraic_grade_model(source, grade=2, record_keys=[1200], availability=AVAILABLE, cancellation=token)
    assert output.matches_source(source)
    with pytest.raises(ValueError): infer_algebraic_grade_model(state, source, availability=(9603, 30))
    with pytest.raises(ValueError): infer_algebraic_grade_model(state, source, execution_limits={"unknown": 1})
    with pytest.raises(TypeError): infer_algebraic_grade_model(state, source, cancellation=True)
    policy = {"scalar_work": 10000}
    strict, _ = _prepare(state, source, execution_limits=policy)
    assert strict.execution_limits["scalar_work"] == 10000
    assert strict is not first
    assert cpp_algebraic_grade_input(state, source) is first
    assert not any(r["name"] == "rexgraph.cpp.algebraic_grade" and "train" in r["operations"] for r in model_provider_inventory())


def test_cache_miss_rechecks_actual_key_carrier_and_stalk_layout(files):
    state, source = model(files)
    c = thaw_tree(state.configuration); c["record_keys"] = (1000,)
    incorrect = replace(state, configuration=c)
    with pytest.raises(ValueError): infer_model(incorrect, source)
    c = thaw_tree(state.configuration); c["layouts"][1]["stalk_offsets"] = (0, 1, 2, 6)
    incorrect = replace(state, configuration=c)
    with pytest.raises(ValueError): infer_model(incorrect, source)


def test_resource_diagnostics_keep_native_order_after_policy_mapping_reordering(files):
    state, source = model(files)
    configuration = thaw_tree(state.configuration)
    configuration["execution_limits"] = dict(reversed(tuple(configuration["execution_limits"].items())))
    reordered = replace(state, configuration=configuration)
    resources = {"records": "binding records", "rational_slots": "reference rational slots",
                 "sparse_entries": "reference sparse entries", "scalar_work": "reference scalar work envelope",
                 "integer_bits": "exact integer growth bound"}
    for key, resource in resources.items():
        required = max(row[key] for row in state.configuration["requirements"])
        assert required > 0
        with pytest.raises(ExecutionLimitError) as caught:
            infer_algebraic_grade_model(reordered, source, execution_limits={key: required - 1})
        assert caught.value.resource == resource
        assert caught.value.required == required and caught.value.limit == required - 1


def test_cached_owner_lifetime_and_concurrent_inference(files):
    state, source = model(files)
    root = cpp_algebraic_grade_input(state, source)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: infer_model(state, source).coefficient_digest, range(8)))
    assert len(set(results)) == 1
    for _ in range(10):
        owner = snapshot(files)
        infer_model(state, owner)
    assert len(_cache) <= 8
    clear_algebraic_grade_model_cache()
    gc.collect()
    assert root.matches_source(source)
    assert infer_model(state, source).coefficient_digest == results[0]


def test_strict_source_reader_policy_is_intersected_before_restore(files):
    state, source = model(files)
    saved = model_record(source, state)
    policy = grade_model_restore_limits(state, {"max_source_coordinates": 0})
    assert policy["max_source_coordinates"] == 0
    with pytest.raises((ValueError, ExecutionLimitError)): read_model_record(saved, algebraic_limits=policy)
    with pytest.raises(ValueError): grade_model_restore_limits(state, {"unknown": 1})
    with pytest.raises(ValueError): grade_model_restore_limits(state, {"max_file_bytes": True})
    smaller = state.configuration["archive_limits"]["max_file_bytes"] - 1
    restored_state, restored_source = read_model_record(saved, algebraic_limits={"max_file_bytes": smaller})
    assert restored_source.archive_owner.graph_owner.limits["max_file_bytes"] == smaller
    assert infer_model(restored_state, restored_source).values.tolist() == infer_model(state, source).values.tolist()


def test_rcql_model_lifecycle_agent_and_system_keep_algebraic_arithmetic(files):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    state, source = model(files)
    saved = algebraic_snapshot_record(source)
    options = {"grade": 2, "record_keys": [1200], "boundary_steps": 1, "availability": AVAILABLE}
    declaration = Executor(sources={"record": saved}, params={"config": options}).execute(
        parse('FROM $record RETURN MODEL_INIT("cpp_algebraic_grade",$config)'))
    selected = declaration.values[0]
    text = "FROM $record RETURN MODEL_INFER($state), MODEL_RECORD($state), MODEL_INFO($state)"
    result = Executor(sources={"record": saved}, params={"state": selected}).execute(parse(text))
    assert result.values[0].arithmetic == "algebraic"
    assert result.values[0].values.tolist() == infer_model(state, source).values.tolist()
    loaded_state, loaded_source = read_model_record(result.values[1])
    assert infer_model(loaded_state, loaded_source).values.tolist() == result.values[0].values.tolist()
    runtime = RCQLRuntime(); runtime.register("record", saved, policy=SourcePolicy.allow("read"))
    assert runtime.execute(parse(text), params={"state": selected}).values[0].coefficient_digest == result.values[0].coefficient_digest
    hive = Hive("grade-model")
    hive.add_worker("boundary", lambda inputs: infer_model(state, source, inputs), capability="predict")
    assert hive.invoke("boundary", None, record=False).coefficient_digest == infer_model(state, source).coefficient_digest
    for value in (state, result.values[0]): json.dumps(json_value(value, max_values=2), allow_nan=False)
    denied = RCQLRuntime(); denied.register("record", saved, policy=SourcePolicy.allow())
    with pytest.raises(PermissionError): denied.execute(parse(text), params={"state": selected})
    for query in ("FROM $record RETURN MODEL_FIELD($output)", "FROM $record RETURN MODEL_INPUT($output)"):
        with pytest.raises(TypeError): Executor(sources={"record": saved}, params={"output": result.values[0]}).execute(parse(query))


def test_rcdb_copy_and_fresh_process_restore_actual_native_publication(files, tmp_path):
    from rcdb import copy_record, open_store
    state, source = model(files)
    saved = model_record(source, state)
    stores = [open_store(f"sqlite:///{tmp_path / (name + '.sqlite')}") for name in ("first", "second")]
    try:
        stores[0].put("model", saved)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("model"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        restored_state, restored_source = read_model_record(stores[1].get_version("model", 1))
        assert infer_model(restored_state, restored_source).coefficient_digest == infer_model(state, source).coefficient_digest
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
 assert source.graph_revision_owner.history_owner.identity['sequence']==31
 print(json.dumps({'digest':infer_model(state,source).coefficient_digest}))
finally: store.close()
'''
    output = subprocess.check_output([sys.executable, "-I", "-c", code, f"sqlite:///{tmp_path / 'second.sqlite'}"], cwd=tmp_path, text=True)
    assert json.loads(output)["digest"] == infer_model(state, source).coefficient_digest


def test_model_outputs_read_in_fresh_process_without_native_extension_or_torch(files, tmp_path):
    from rexgraph.io import save
    state, source = model(files)
    output = infer_model(state, source)
    saved = model_record(source, state); saved.attach_metadata(1, 0, "output", output)
    path = tmp_path / "model.rcbd"; save(path, saved)
    code = '''
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rexgraph_cpp','torch'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from rexgraph.io import load
record=load(sys.argv[1]);state=record.get_metadata(1,0,'model');output=record.get_metadata(1,0,'output')
state.check_state();output.check_state()
assert output.arithmetic=='algebraic' and output.grade==1
print(json.dumps({'digest':output.coefficient_digest}))
'''
    result = subprocess.check_output([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, text=True)
    assert json.loads(result)["digest"] == output.coefficient_digest


def test_actual_publication_aliases_share_prepared_plans_and_equal_bytes_do_not(files):
    native = NativeSource((files / "profile-13.rfcxs").read_bytes())
    revision = NativeAlgebraicGraphRevision(native, (9631, 1, 2), sequence=31)
    db = NativeAlgebraicGraphDatabase(revision, availability=AVAILABLE)
    source = db.archive(9, availability=AVAILABLE)
    state = cpp_algebraic_grade_model(source, grade=2, record_keys=oracle(files)[2]["keys"],
        availability=AVAILABLE)
    alias = db.archive(9, availability=AVAILABLE)
    assert alias.same_publication(source)
    root = cpp_algebraic_grade_input(state, source)
    assert cpp_algebraic_grade_input(state, alias) is root
    independent = snapshot(files)
    assert not independent.same_publication(source)
    fresh = cpp_algebraic_grade_input(state, independent)
    assert fresh is not root and not fresh.same_owner(root)
    assert fresh.matches_source(independent)


def test_cache_does_not_hide_changed_saved_source_bytes(files):
    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
    state, source = model(files)
    record = model_record(source, state)
    cache = _AlgebraicRestoreCache()
    read_model_record(record, _algebraic_cache=cache)
    retained = record.get_metadata(1, 0, "model_source")
    retained.get_metadata(1, 0, "algebraic_archive")["archive"][-1] ^= 1
    with pytest.raises(ValueError): read_model_record(record, _algebraic_cache=cache)


def test_declared_engine_and_selected_publication_are_required_before_execution(files):
    state, source = model(files)
    c = thaw_tree(state.configuration)
    c["engine_receipt"]["engine_digest"] = "0" * 64
    foreign = replace(state, configuration=c)
    with pytest.raises(ValueError, match="engine"): infer_model(foreign, source)
    c = thaw_tree(state.configuration)
    c["publication"]["graph_history"]["revision"]["sources"][0]["cells"] = ((2, 1, 0), (0, 1, 2), (0,))
    with pytest.raises(ValueError):
        foreign = replace(state, configuration=c)
        infer_model(foreign, source)
    policy = grade_model_restore_limits(state, {"max_file_bytes": 1 << 40})
    assert policy["max_file_bytes"] == state.configuration["archive_limits"]["max_file_bytes"]


def test_grade_record_replay_accepts_independently_narrower_archive_limits(files):
    from rexgraph.cpp_algebraic_grade import algebraic_grade_record, read_algebraic_grade_record
    state, source = model(files)
    native = cpp_algebraic_grade_input(state, source).boundary(availability=AVAILABLE)
    record = algebraic_grade_record(native, availability=AVAILABLE)
    restored = read_algebraic_grade_record(record, limits={"max_file_bytes": 67108863})
    assert restored.coefficients == native.coefficients
    assert restored.snapshot_owner.archive_owner.graph_owner.limits["max_file_bytes"] == 67108863
