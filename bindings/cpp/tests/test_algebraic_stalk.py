"""Original stalk model composition retains native owners and exact source coordinates."""
import gc
import json
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    NativeAlgebraicGradeTensor,
    NativeAlgebraicSnapshot,
)
from test_algebraic_grade_model import AVAILABLE, CASES, model, oracle, snapshot

from rexgraph.cpp_algebraic import cpp_algebraic_input, cpp_algebraic_model
from rexgraph.cpp_algebraic_stalk import (
    AlgebraicGradeStalkSelection,
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
from rexgraph.value_codec import pack_value


@pytest.fixture(scope="session")
def files():
    return Path(os.environ["REXGRAPH_CPP_GRADE_FIXTURES"])


def result(files, profile=13, grade=2, steps=2):
    state, source = model(files, profile, grade, steps)
    return infer_model(state, source), source


@pytest.mark.parametrize("profile,grade", CASES)
def test_each_original_stalk_matches_independent_grade_oracle(files, profile, grade):
    rows = oracle(files, profile)
    steps = min(grade, 2) if "boundary" in rows[grade] else 0
    output, source = result(files, profile, grade, steps)
    plan = prepare_algebraic_grade_stalks(output, source)
    coefficients = rows[grade]["coordinates" if steps == 0 else "boundary" if steps == 1 else "twice"]
    coordinates = rows[grade - steps]["stalk_coordinates"]
    for cell in range(source.archive_owner.manifest["source_sizes"][grade - steps]):
        expected = tuple(tuple(Fraction(q) for q in row) for row, coordinate in zip(coefficients, coordinates, strict=True) if coordinate[1] == cell)
        selected = plan.select(cell)
        native = plan.native(cell)
        assert tuple(tuple(row) for row in selected.values) == native.coefficients == expected
        assert native.archive_owner is source.archive_owner
        assert selected.execution_receipt["kind"] == "vector"
        assert selected.execution_receipt["output_carrier"]["index"] == cell
        assert selected.space.keys == tuple(str(i) for i in range(len(expected)))
        assert selected.grade == grade - steps
        assert selected.execution_receipt["observation"]["lineage"] == output.execution_receipt["observation"]["lineage"]
        assert unpack_model(pack_model(selected)).coefficient_digest == selected.coefficient_digest
    if plan.tensor_owner.cell_count == 0:
        assert plan.select_many(()) == ()
        with pytest.raises(ValueError): plan.select(0)


def multiply(x, y, polynomial):
    degree = len(polynomial) - 1
    values = [Fraction(0) for _ in range(2 * degree - 1)]
    for i, a in enumerate(x):
        for j, b in enumerate(y): values[i + j] += a * b
    for i in range(len(values) - 1, degree - 1, -1):
        for j in range(degree): values[i - degree + j] -= values[i] * polynomial[j]
    return tuple(values[:degree])


def matrix_action(matrix, vector, polynomial):
    rank = len(vector); degree = len(polynomial) - 1
    values = []
    for i in range(len(matrix) // rank):
        row = [Fraction(0) for _ in range(degree)]
        for j in range(rank):
            product = multiply(matrix[i * rank + j], vector[j], polynomial)
            row = [a + b for a, b in zip(row, product, strict=True)]
        values.append(tuple(row))
    return tuple(values)


@pytest.mark.parametrize("profile", range(1, 11))
@pytest.mark.parametrize("operation,key", [("apply", 30), ("solve", 50), ("quadrance", 50), ("as_covector", 50), ("evaluate", 20)])
def test_grade_outputs_feed_existing_map_metric_and_dual_models(files, profile, operation, key):
    output, source = result(files, profile)
    plan = prepare_algebraic_grade_stalks(output, source)
    selected = plan.select(0)
    archive = source.archive_owner
    state = cpp_algebraic_model(archive, operator_key=key, input_key=1000, operation=operation)
    captured = cpp_algebraic_input(state, archive, selected)
    native = plan.native(0)
    composed = infer_model(state, archive, captured)
    direct = infer_model(state, archive, native)
    assert composed.values.tolist() == direct.values.tolist()
    assert composed.model_digest == direct.model_digest
    assert composed.coefficient_digest != direct.coefficient_digest
    assert composed.execution_receipt["origin_depth"] == 1
    assert composed.execution_receipt["input_origin"]["origin_output"] == pack_value(pack_model(selected, native=True))
    assert unpack_model(pack_model(composed)).coefficient_digest == composed.coefficient_digest
    if operation == "apply":
        assert tuple(tuple(row) for row in composed.values) == matrix_action(archive.record(key).coefficients, native.coefficients, state.source.native.polynomial)
    if operation == "solve":
        assert matrix_action(archive.record(key).coefficients, tuple(tuple(row) for row in composed.values), state.source.native.polynomial) == native.coefficients
    if operation == "apply":
        wrong = plan.select(1)
        with pytest.raises(ValueError, match="carrier"): cpp_algebraic_input(state, archive, wrong)


@pytest.mark.parametrize("profile", [1, 3, 5, 7, 9, 13])
@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_selection_records_restore_complete_source_and_exact_native_vector(files, profile, suffix, tmp_path):
    output, source = result(files, profile)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    path = tmp_path / ("selection." + suffix)
    save(path, algebraic_grade_stalk_record(selected, source))
    record = load(path)
    assert read_algebraic_grade_stalk_output(record).coefficient_digest == selected.coefficient_digest
    restored, owner, value = read_algebraic_grade_stalk_record(record)
    assert restored.coefficient_digest == selected.coefficient_digest
    assert value.archive_owner is owner.archive_owner
    assert value.coefficients == tuple(tuple(row) for row in selected.values)
    assert not owner.same_publication(source)
    reader = owner.archive_owner.graph_owner.limits["max_file_bytes"] - 1
    assert read_algebraic_grade_stalk_record(record, limits={"max_file_bytes": reader})[2].coefficients == value.coefficients
    with pytest.raises(ValueError): read_algebraic_grade_stalk_output(record, max_archive_bytes=1)
    with pytest.raises(ValueError): read_algebraic_grade_stalk_record(record, limits={"max_file_bytes": 1})


def test_prepared_selection_reuses_actual_tensor_and_retains_owners(files):
    output, source = result(files)
    tensor = NativeAlgebraicGradeTensor(source, 2, [1200], availability=AVAILABLE).boundary(availability=AVAILABLE).boundary(availability=AVAILABLE)
    plan = prepare_algebraic_grade_stalks(output, source, tensor=tensor)
    assert plan.tensor_owner is tensor
    assert [s.execution_receipt["cell"] for s in plan.select_many([2, 0, 2])] == [2, 0, 2]
    pin = plan.native(2)
    del plan, tensor, source
    gc.collect()
    assert pin.kind == "vector" and pin.coefficients


@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_composed_model_actions_retain_complete_original_grade_origin(files, suffix, tmp_path):
    from rexgraph.value_codec import unpack_value
    output, source = result(files, 1)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    archive = source.archive_owner
    state = cpp_algebraic_model(archive, operator_key=50, input_key=1000, operation="solve")
    first = infer_model(state, archive, cpp_algebraic_input(state, archive, selected))
    second = infer_model(state, archive, cpp_algebraic_input(state, archive, first))
    assert second.execution_receipt["origin_depth"] == 2
    decoded_first = unpack_model(unpack_value(second.execution_receipt["input_origin"]["origin_output"]))
    decoded_selection = unpack_model(unpack_value(decoded_first.execution_receipt["input_origin"]["origin_output"]))
    assert decoded_first.coefficient_digest == first.coefficient_digest
    assert decoded_selection.coefficient_digest == selected.coefficient_digest
    record = algebraic_grade_stalk_record(selected, source)
    record.attach_metadata(1, 0, "composed_output", second)
    record.attach_metadata(1, 0, "captured_input", thaw_tree(second.execution_receipt["input_origin"]))
    path = tmp_path / ("composed." + suffix); save(path, record)
    restored_record = load(path)
    assert restored_record.get_metadata(1, 0, "composed_output").coefficient_digest == second.coefficient_digest
    _, owner, _ = read_algebraic_grade_stalk_record(restored_record)
    restored_state = cpp_algebraic_model(owner.archive_owner, operator_key=50, input_key=1000, operation="solve")
    actual = infer_model(restored_state, owner.archive_owner, restored_record.get_metadata(1, 0, "captured_input"))
    assert actual.coefficient_digest == second.coefficient_digest


@pytest.mark.parametrize("case", ["depth", "bytes", "values", "side", "key", "notes"])
def test_composed_input_profile_rejects_incomplete_or_changed_origins(files, case):
    output, source = result(files, 1)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    state = cpp_algebraic_model(source.archive_owner, operator_key=50, input_key=1000, operation="solve")
    captured = thaw_tree(cpp_algebraic_input(state, source.archive_owner, selected))
    if case == "depth": captured["origin_depth"] = 33
    elif case == "bytes": captured["origin_output"] = b"invalid"
    elif case == "values": captured["coefficients"] = tuple(tuple(q + 1 for q in row) for row in captured["coefficients"])
    elif case == "side": captured["carrier_side"] = "target"
    elif case == "key": captured["carrier_record"] = 1001
    else: captured["notes"] = "unknown"
    with pytest.raises((ValueError, TypeError)): infer_model(state, source.archive_owner, captured)


def test_actual_publication_and_complete_boundary_lineage_are_required(files):
    output, source = result(files)
    original = NativeAlgebraicGradeTensor(source, 0, [1000, 1001, 1002], availability=AVAILABLE)
    with pytest.raises(ValueError, match="lineage"): prepare_algebraic_grade_stalks(output, source, tensor=original)
    other = snapshot(files)
    foreign = prepare_algebraic_grade_stalks(output, other)
    with pytest.raises(ValueError, match="publication"):
        prepare_algebraic_grade_stalks(output, source, tensor=foreign.tensor_owner)
    alias = NativeAlgebraicSnapshot(source.archive_owner, namespace=source.namespace, sequence=source.sequence)
    with pytest.raises(ValueError): prepare_algebraic_grade_stalks(output, alias)
    with pytest.raises(TypeError): AlgebraicGradeStalkSelection(output, source, True, None)


@pytest.mark.parametrize("cell", [True, -1, 3, 1.0, "0"])
def test_original_cell_selectors_reject_invalid_coordinates(files, cell):
    output, source = result(files)
    with pytest.raises(ValueError): cpp_algebraic_grade_stalk(output, source, cell=cell)


def test_bounds_cancellation_and_current_output_seals_apply_to_reused_selections(files):
    output, source = result(files)
    with pytest.raises(ValueError): prepare_algebraic_grade_stalks(output, source, max_coordinates=1)
    with pytest.raises(ValueError): prepare_algebraic_grade_stalks(output, source, execution_limits={"integer_bits": 0})
    with pytest.raises(ValueError): prepare_algebraic_grade_stalks(output, source, execution_limits={"unknown": 1})
    plan = prepare_algebraic_grade_stalks(output, source)
    with pytest.raises(ValueError): plan.select_many([0, 1], max_selections=1)
    with pytest.raises(ValueError): plan.select_many([0], max_selections=True)
    token = CancellationToken(); token.cancel()
    for action in (lambda: plan.native(0, cancellation=token), lambda: plan.select(0, cancellation=token),
                   lambda: plan.select_many((), cancellation=token),
                   lambda: prepare_algebraic_grade_stalks(output, source, cancellation=token)):
        with pytest.raises(ExecutionCancelled): action()
    output.values.flags.writeable = True; output.values[0, 0] += 1
    with pytest.raises(ValueError, match="changed"): plan.native(0)


@pytest.mark.parametrize("case", ["values", "slice", "cell", "carrier", "kind", "basis", "origin", "policy", "notes", "engine"])
def test_closed_selection_receipts_reject_changed_coordinates_and_declarations(files, case):
    output, source = result(files)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    receipt = thaw_tree(selected.execution_receipt); options = {"execution_receipt": receipt}
    if case == "values": options["values"] = selected.values + 1
    elif case == "slice": receipt["source_slice"] = (1, 4)
    elif case == "cell": receipt["cell"] = 1
    elif case == "carrier": receipt["output_carrier"]["index"] = 1
    elif case == "kind": receipt["kind"] = "scalar"
    elif case == "basis": receipt["coefficient_basis"] = "float"
    elif case == "origin": receipt["origin_output"] = "0" * 64
    elif case == "policy": receipt["execution_limits"]["scalar_work"] += 1
    elif case == "engine": receipt["engine"]["engine_digest"] = "0" * 64
    else: receipt["notes"] = "unknown"
    with pytest.raises((ValueError, TypeError)): replace(selected, **options)


def test_native_replay_rejects_resealed_false_source_observations(files):
    output, source = result(files)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    receipt = thaw_tree(selected.execution_receipt)
    observation = receipt["observation"]
    rows = list(observation["detail"]["coefficients"])
    rows[0] = tuple(q + 1 for q in rows[0]); observation["detail"]["coefficients"] = tuple(rows)
    observation["seal"] = sha256(pack_value({key: value for key, value in observation.items() if key != "seal"})).hexdigest()
    values = selected.values.copy(); values[0] += 1
    false = replace(selected, values=values, execution_receipt=receipt)
    with pytest.raises(ValueError, match="coordinates"): algebraic_grade_stalk_record(false, source)


def test_rcql_agent_and_system_keep_selected_cell_identity_and_read_policy(files):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
    output, source = result(files)
    record = algebraic_snapshot_record(source)
    query = parse("FROM $record RETURN MODEL_STALK($output, 1), MODEL_VALUES(MODEL_STALK($output, 1))")
    executed = Executor(sources={"record": record}, params={"output": output}).execute(query)
    selected = executed.values[0]
    assert selected.grade == 0 and selected.execution_receipt["output_carrier"]["index"] == 1
    assert executed.values[1].tolist() == selected.values.tolist()
    hive = Hive("stalk-selection")
    hive.add_worker("select", lambda cell: cpp_algebraic_grade_stalk(output, source, cell=cell), capability="transform")
    assert hive.invoke("select", 1, record=False).coefficient_digest == selected.coefficient_digest
    view = json_value(selected, max_values=1)
    assert view["original_cell"] == 1 and view["source_slice"] == [3, 5]
    json.dumps(view, allow_nan=False)
    denied = RCQLRuntime(); denied.register("record", record, policy=SourcePolicy.allow())
    with pytest.raises(PermissionError): denied.execute(query, params={"output": output})
    with pytest.raises(TypeError): Executor(sources={"record": record}, params={"output": selected}).execute(query)


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_rcdb_copy_and_fresh_process_selection_roundtrip(files, backend, tmp_path):
    from rcdb import copy_record, open_store
    output, source = result(files)
    selected = cpp_algebraic_grade_stalk(output, source, cell=0)
    paths = [f"sqlite:///{tmp_path / name}" if backend == "sqlite" else f"{backend}://{tmp_path / name}"
             for name in ("first", "second")]
    stores = [open_store(path, **({} if backend == "sqlite" else {"read_only": False})) for path in paths]
    try:
        stores[0].put("selected", algebraic_grade_stalk_record(selected, source))
        copy_record(stores[0], stores[1], stores[0].get_record("selected"))
    finally:
        for store in stores: store.close()
    code = '''from rcdb import open_store
from rexgraph.cpp_algebraic_stalk import read_algebraic_grade_stalk_record
import sys
store=open_store(sys.argv[1])
try:
 output,snapshot,value=read_algebraic_grade_stalk_record(store.get_version('selected',1))
 assert value.archive_owner is snapshot.archive_owner
 print(output.coefficient_digest)
finally:store.close()
'''
    actual = subprocess.check_output([sys.executable, "-I", "-c", code, paths[1]], cwd=tmp_path, text=True)
    assert actual.strip() == selected.coefficient_digest
