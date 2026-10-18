"""Metric result stalks preserve original carriers and complete producer origins."""
import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from rexgraph_cpp import CancellationToken, ExecutionCancelled, ExecutionLimitError
from test_algebraic_grade_metric_model import model
from test_algebraic_grade_model import AVAILABLE, oracle, snapshot
from test_algebraic_metric import (
    ACTIONS,
    PROFILES,
    conjugate,
    declared_oracle,
    exact_actions,
    multiply,
    solve,
)
from test_algebraic_stalk import matrix_action

from rexgraph.cpp_algebraic import cpp_algebraic_input, cpp_algebraic_model
from rexgraph.cpp_algebraic_grade_metric_model import _prepare
from rexgraph.cpp_algebraic_grade_model import cpp_algebraic_grade_model
from rexgraph.cpp_algebraic_metric_stalk import (
    AlgebraicMetricStalkSelection,
    prepare_algebraic_metric_stalks,
)
from rexgraph.cpp_algebraic_stalk import (
    algebraic_grade_stalk_record,
    cpp_algebraic_grade_stalk,
    prepare_algebraic_grade_stalks,
    read_algebraic_grade_stalk_output,
    read_algebraic_grade_stalk_record,
)
from rexgraph.io import load, save
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value, unpack_value

VECTOR_ACTIONS = [(p, k, op) for p, k, op in ACTIONS if op in {"hodge", "coboundary"}]


@pytest.fixture(scope="session")
def files():
    return Path(os.environ["REXGRAPH_CPP_METRIC_FIXTURES"])


def result(files, profile=13, grade=1, operation="hodge", steps=0, origins=False):
    state, source = model(files, profile, grade, operation, steps)
    inputs = None
    if origins:
        upstream = cpp_algebraic_grade_model(source, grade=grade, record_keys=oracle(files, profile)[grade]["keys"],
            boundary_steps=steps, availability=AVAILABLE)
        inputs = infer_model(upstream, source)
    return infer_model(state, source, inputs), source, state


@pytest.mark.parametrize("profile,grade,operation", VECTOR_ACTIONS)
def test_each_original_result_stalk_matches_independent_metric_oracle(files, profile, grade, operation):
    output, source, state = result(files, profile, grade, operation)
    field, tensor, _ = _prepare(state, source)
    native = field.apply(tensor, operation, availability=AVAILABLE)
    plan = prepare_algebraic_grade_stalks(output, source, metric_values=native)
    assert isinstance(plan, AlgebraicMetricStalkSelection) and plan.values_owner is native
    x = tuple(tuple(Fraction(q) for q in row) for row in oracle(files, profile)[grade]["coordinates"])
    expected = exact_actions(files, profile, grade, x)[operation]
    offsets = declared_oracle(str(files), profile)[2][output.grade]
    for cell in range(native.cell_count):
        selected = plan.select(cell); vector = plan.native(cell)
        begin, end = offsets[cell:cell + 2]
        assert tuple(tuple(row) for row in selected.values) == vector.coefficients == expected[begin:end]
        assert selected.grade == output.grade and selected.execution_receipt["source_slice"] == (begin, end)
        assert vector.archive_owner is source.archive_owner
        parent = unpack_model(unpack_value(selected.execution_receipt["origin"]))
        assert parent.coefficient_digest == output.coefficient_digest
        assert unpack_model(pack_model(selected)).coefficient_digest == selected.coefficient_digest
    if native.cell_count == 0:
        assert plan.select_many(()) == ()
        with pytest.raises(ValueError): plan.select(0)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("operation", ["hodge", "coboundary"])
def test_selected_results_retain_boundary_input_producer_receipts(files, profile, operation):
    output, source, _ = result(files, profile, 2, operation, 1, True)
    plan = prepare_algebraic_metric_stalks(output, source)
    for cell in range(plan.values_owner.cell_count):
        selected = plan.select(cell)
        metric = unpack_model(unpack_value(selected.execution_receipt["origin"]))
        upstream = unpack_model(unpack_value(metric.execution_receipt["input_origins"]["input"]))
        assert len(upstream.execution_receipt["observation"]["lineage"]) == 2
        assert metric.coefficient_digest == output.coefficient_digest
        assert tuple(tuple(row) for row in selected.values) == plan.native(cell).coefficients


@pytest.mark.parametrize("profile", [p for p in PROFILES if p != 15])
@pytest.mark.parametrize("operation", ["solve", "quadrance", "as_covector"])
def test_metric_stalks_compose_with_metric_models_using_exact_independent_arithmetic(files, profile, operation):
    output, source, _ = result(files, profile)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    grade = output.grade; key = 5000 + 100 * grade
    state = cpp_algebraic_model(source.archive_owner, operator_key=key, input_key=1000 + 100 * grade, operation=operation)
    inputs = cpp_algebraic_input(state, source.archive_owner, selected)
    composed = infer_model(state, source.archive_owner, inputs)
    assert infer_model(state, source.archive_owner, selected).coefficient_digest == composed.coefficient_digest
    receipt = composed.execution_receipt
    assert receipt["origin_depth"] == 1
    origin = unpack_model(unpack_value(receipt["input_origin"]["origin_output"]))
    assert origin.coefficient_digest == selected.coefficient_digest
    assert unpack_model(pack_model(composed)).coefficient_digest == composed.coefficient_digest
    p, involution, _, matrices, _ = declared_oracle(str(files), profile)
    vector = tuple(tuple(row) for row in selected.values); n = len(vector)
    matrix = tuple(row[:n] for row in matrices[grade][:n])
    if operation == "solve": expected = solve(matrix, vector, p)
    elif operation == "quadrance":
        expected_value = (Fraction(0),) * (len(p) - 1)
        for i in range(n):
            for j in range(n): expected_value = add(expected_value, multiply(conjugate(vector[i], involution, p), multiply(matrix[i][j], vector[j], p), p))
        expected = (expected_value,)
    if operation == "as_covector":
        coordinates=[]
        for j in range(n):
            coordinate=(Fraction(0),)*(len(p)-1)
            for i in range(n): coordinate=add(coordinate,multiply(conjugate(vector[i],involution,p),matrix[i][j],p))
            coordinates.append(coordinate)
        expected=tuple(coordinates)
    assert tuple(tuple(row) for row in composed.values) == expected


def add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("profile", range(1, 11))
@pytest.mark.parametrize("operation,key", [("apply", 30), ("evaluate", 20)])
def test_metric_result_carriers_feed_existing_map_and_dual_models(files, profile, operation, key):
    output, source, _ = result(files, profile, grade=0)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    state = cpp_algebraic_model(source.archive_owner, operator_key=key, input_key=10, operation=operation)
    composed = infer_model(state, source.archive_owner, cpp_algebraic_input(state, source.archive_owner, selected))
    expected = matrix_action(source.archive_owner.record(key).coefficients, tuple(tuple(row) for row in selected.values), state.source.native.polynomial)
    assert tuple(tuple(row) for row in composed.values) == expected
    with pytest.raises(ValueError): cpp_algebraic_input(state, source.archive_owner, cpp_algebraic_grade_stalk(output, source, cell=1))


@pytest.mark.parametrize("operation", ["hodge", "coboundary"])
@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_records_restore_selected_vectors_and_full_producers_in_fresh_core_only_process(files, tmp_path, operation, suffix):
    output, source, _ = result(files, grade=2, operation=operation, steps=1, origins=True)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    path = tmp_path / ("selection." + suffix)
    save(path, algebraic_grade_stalk_record(selected, source))
    record = load(path)
    assert read_algebraic_grade_stalk_output(record).coefficient_digest == selected.coefficient_digest
    restored, owner, vector = read_algebraic_grade_stalk_record(record)
    assert restored.coefficient_digest == selected.coefficient_digest and vector.archive_owner is owner.archive_owner
    reader = owner.archive_owner.graph_owner.limits["max_file_bytes"] - 1
    assert read_algebraic_grade_stalk_record(record, limits={"max_file_bytes": reader})[2].coefficients == vector.coefficients
    with pytest.raises(ValueError): read_algebraic_grade_stalk_output(record, max_archive_bytes=1)
    with pytest.raises(RuntimeError): read_algebraic_grade_stalk_record(record, max_retained_coordinates=0)
    code = '''
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rexgraph_cpp','torch'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from rexgraph.io import load
from rexgraph.cpp_algebraic_stalk import read_algebraic_grade_stalk_output
value=read_algebraic_grade_stalk_output(load(sys.argv[1]));value.check_state()
print(json.dumps({'digest':value.coefficient_digest}))
'''
    decoded = subprocess.check_output([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, text=True)
    assert json.loads(decoded)["digest"] == selected.coefficient_digest


@pytest.mark.parametrize("case", ["values", "cell", "grade", "carrier", "slice", "model", "parent", "origin", "bound", "operation", "engine", "policy", "quota", "axes", "extra"])
def test_closed_metric_stalk_receipts_reject_false_provenance_and_coordinates(files, case):
    output, source, _ = result(files)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    r = thaw_tree(selected.execution_receipt); options = {"execution_receipt": r}
    if case == "values": options["values"] = selected.values + 1
    elif case == "cell": r["cell"] = 99
    elif case == "grade": options["grade"] = 0
    elif case == "carrier": r["output_carrier"]["index"] = 1
    elif case == "slice": r["source_slice"] = (1, 3)
    elif case == "model": r["origin_model"] = "0" * 64
    elif case == "parent": r["origin_output"] = "0" * 64
    elif case == "origin": r["origin"] = pack_value(pack_model(selected, native=True))
    elif case == "bound": r["origin_max_bytes"] = 1
    elif case == "operation": r["operation"] = "inner"
    elif case == "engine": r["engine"]["engine_digest"] = "0" * 64
    elif case == "policy": r["execution_limits"]["records"] += 1
    elif case == "quota": r["max_retained_coordinates"] += 1
    elif case == "axes": options["axes"] = ()
    else: r["notes"] = "unknown"
    with pytest.raises((TypeError, ValueError)): replace(selected, **options)


def test_prepared_native_values_reject_foreign_publications_operations_lineages_and_limits(files):
    output, source, state = result(files)
    field, tensor, _ = _prepare(state, source)
    values = field.apply(tensor, "hodge", availability=AVAILABLE)
    with pytest.raises(ValueError): prepare_algebraic_metric_stalks(output, snapshot(files), values=values)
    with pytest.raises(ValueError): prepare_algebraic_metric_stalks(output, source, values=values, max_retained_coordinates=0)
    with pytest.raises(ValueError): prepare_algebraic_metric_stalks(output, source, values=field.apply(tensor, "coboundary", availability=AVAILABLE))
    with pytest.raises(TypeError): prepare_algebraic_grade_stalks(output, source, tensor=tensor)
    with pytest.raises(ValueError): prepare_algebraic_metric_stalks(output, source, max_origin_bytes=1)
    with pytest.raises(ValueError): prepare_algebraic_metric_stalks(output, source, max_coordinates=1)
    needed = tensor.requirements["records"]
    with pytest.raises((ValueError, ExecutionLimitError)): prepare_algebraic_metric_stalks(output, source, execution_limits={"records": needed - 1})
    scalar, _, _ = result(files, operation="inner")
    with pytest.raises(TypeError): prepare_algebraic_grade_stalks(scalar, source)
    other, _, _ = result(files, grade=2, steps=1)
    with pytest.raises(ValueError): prepare_algebraic_metric_stalks(other, source, values=values)


def test_owned_results_cancellation_bounded_sequences_and_concurrent_selections(files):
    output, source, _ = result(files)
    plan = prepare_algebraic_metric_stalks(output, source)
    expected = plan.select(0).coefficient_digest
    assert [v.execution_receipt["cell"] for v in plan.select_many([2, 0, 2])] == [2, 0, 2]
    with pytest.raises(ValueError): plan.select_many([0, 1], max_selections=1)
    with pytest.raises(ValueError): plan.select(True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert set(pool.map(lambda _, prepared=plan: prepared.select(0).coefficient_digest, range(12))) == {expected}
    token=CancellationToken();token.cancel()
    with pytest.raises(ExecutionCancelled): plan.select(0, cancellation=token)
    with pytest.raises(ExecutionCancelled): plan.select_many([0], cancellation=token)
    with pytest.raises(ExecutionCancelled): prepare_algebraic_grade_stalks(output, source, cancellation=token)
    vector=plan.native(0);del plan, output, source;gc.collect()
    assert vector.kind == "vector" and vector.coefficients


def test_narrowed_reader_admission_and_two_carrier_actions_preserve_full_metric_origin(files):
    output, source, _ = result(files, grade=2, steps=1, origins=True)
    bound = len(pack_value(pack_model(output, native=True)))
    plan = prepare_algebraic_metric_stalks(output, source, max_retained_coordinates=100000, max_origin_bytes=bound)
    selected = plan.select(0)
    assert selected.execution_receipt["max_retained_coordinates"] == 100000
    assert selected.execution_receipt["origin_max_bytes"] == bound
    state = cpp_algebraic_model(source.archive_owner, operator_key=5100, input_key=1100, operation="solve")
    first = infer_model(state, source.archive_owner, cpp_algebraic_input(state, source.archive_owner, selected))
    second = infer_model(state, source.archive_owner, cpp_algebraic_input(state, source.archive_owner, first))
    assert infer_model(state, source.archive_owner, first).coefficient_digest == second.coefficient_digest
    assert second.execution_receipt["origin_depth"] == 2
    recovered_first = unpack_model(unpack_value(second.execution_receipt["input_origin"]["origin_output"]))
    recovered_selection = unpack_model(unpack_value(recovered_first.execution_receipt["input_origin"]["origin_output"]))
    recovered_metric = unpack_model(unpack_value(recovered_selection.execution_receipt["origin"]))
    assert recovered_metric.coefficient_digest == output.coefficient_digest
    record = algebraic_grade_stalk_record(selected, source)
    restored, _, vector = read_algebraic_grade_stalk_record(record, max_retained_coordinates=50000)
    assert restored.coefficient_digest == selected.coefficient_digest and vector.coefficients == tuple(tuple(row) for row in selected.values)
    token=CancellationToken();token.cancel()
    with pytest.raises(ExecutionCancelled):read_algebraic_grade_stalk_record(record,cancellation=token)


def test_rcql_agent_and_system_use_existing_stalk_composition_route(files):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
    output, source, _ = result(files)
    record = algebraic_snapshot_record(source)
    query = parse("FROM $record RETURN MODEL_STALK($output,0)")
    selected = Executor(sources={"record": record}, params={"output": output}).execute(query).values[0]
    assert selected.coefficient_digest == cpp_algebraic_grade_stalk(output, source, cell=0).coefficient_digest
    carrier = cpp_algebraic_model(source.archive_owner, operator_key=5100, input_key=1100, operation="solve")
    composed_query = parse("FROM $record RETURN MODEL_INFER($carrier,MODEL_STALK($output,0))")
    composed = Executor(sources={"record": record}, params={"output": output, "carrier": carrier}).execute(composed_query).values[0]
    assert composed.coefficient_digest == infer_model(carrier, source.archive_owner, selected).coefficient_digest
    runtime=RCQLRuntime();runtime.register("record",record,policy=SourcePolicy.allow("read"))
    assert runtime.execute(query,params={"output":output}).values[0].coefficient_digest == selected.coefficient_digest
    assert runtime.execute(composed_query,params={"output":output,"carrier":carrier}).values[0].coefficient_digest == composed.coefficient_digest
    denied=RCQLRuntime();denied.register("record",record,policy=SourcePolicy.allow())
    with pytest.raises(PermissionError):denied.execute(query,params={"output":output})
    hive=Hive("metric-stalk");hive.add_worker("stalk",lambda value:cpp_algebraic_grade_stalk(value,source,cell=0),capability="transform")
    assert hive.invoke("stalk",output,record=False).coefficient_digest == selected.coefficient_digest
    preview=json_value(selected,max_values=2)
    assert preview["original_cell"] == 0 and preview["original_grade"] == selected.grade
    assert preview["metric_operation"] == "hodge"
    json.dumps(preview,allow_nan=False)


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_copy_record_preserves_metric_producer_and_composed_model_in_fresh_store_process(files, tmp_path, backend):
    from rcdb import copy_record, open_store
    output, source, _ = result(files, grade=2, steps=1, origins=True)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    state = cpp_algebraic_model(source.archive_owner, operator_key=5100, input_key=1100, operation="solve")
    inputs = cpp_algebraic_input(state, source.archive_owner, selected)
    composed = infer_model(state, source.archive_owner, inputs)
    record = algebraic_grade_stalk_record(selected, source)
    record.attach_metadata(1, 0, "carrier_model", state)
    record.attach_metadata(1, 0, "carrier_input", thaw_tree(inputs))
    record.attach_metadata(1, 0, "carrier_output", composed)
    urls = [f"{backend}:///{tmp_path / name}" for name in ("first", "second")]
    stores = [open_store(url, **({} if backend == "sqlite" else {"read_only": False})) for url in urls]
    try:
        stores[0].put("selection", record)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("selection"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
    finally:
        for store in stores: store.close()
    code = '''
import json,sys
from rcdb import open_store
from rexgraph.cpp_algebraic_stalk import read_algebraic_grade_stalk_record
from rexgraph.cpp_algebraic import enable_cpp_algebraic_provider
from rexgraph.model_runtime import infer_model
enable_cpp_algebraic_provider()
store=open_store(sys.argv[1])
try:
 record=store.get_version('selection',1)
 selected,source,_=read_algebraic_grade_stalk_record(record)
 actual=infer_model(record.get_metadata(1,0,'carrier_model'),source.archive_owner,record.get_metadata(1,0,'carrier_input'))
 assert actual.coefficient_digest==record.get_metadata(1,0,'carrier_output').coefficient_digest
 print(json.dumps({'digest':actual.coefficient_digest,'selection':selected.coefficient_digest}))
finally:store.close()
'''
    actual = subprocess.check_output([sys.executable, "-I", "-c", code, urls[1]], cwd=tmp_path, text=True)
    assert json.loads(actual) == {"digest": composed.coefficient_digest, "selection": selected.coefficient_digest}
