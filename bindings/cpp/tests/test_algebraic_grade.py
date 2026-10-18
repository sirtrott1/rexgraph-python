"""Whole grade fields checked against independent canonical occurrence arithmetic."""
import gc
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeAlgebraicGradeTensor,
    NativeAlgebraicGradeTensorWindow,
    NativeAlgebraicGraphDatabase,
    NativeAlgebraicGraphRevision,
    NativeSource,
)

from rexgraph.cpp_algebraic_grade import (
    algebraic_grade_observation,
    algebraic_grade_record,
    algebraic_grade_window_observation,
    prepare_algebraic_grade,
    read_algebraic_grade_observation_record,
    read_algebraic_grade_record,
    validate_algebraic_grade_observation,
)
from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
from rexgraph.value_codec import pack_value

AVAILABLE = (9603, 31)
PROFILES = range(1, 16)


@pytest.fixture(scope="session")
def files():
    return Path(os.environ["REXGRAPH_CPP_GRADE_FIXTURES"])


def snapshot(files, profile=13):
    source = NativeSource((files / f"profile-{profile}.rfcxs").read_bytes())
    revision = NativeAlgebraicGraphRevision(source, (9631, 1, 2), sequence=31)
    return NativeAlgebraicGraphDatabase(revision, availability=AVAILABLE).archive(9, availability=AVAILABLE)


def oracle(files, profile):
    return [json.loads(row) for row in (files / f"profile-{profile}.jsonl").read_text().splitlines()][1:]


def fractions(rows):
    return tuple(tuple(Fraction(q) for q in row) for row in rows)


def field(files, profile=13, grade=2):
    row = oracle(files, profile)[grade]
    return NativeAlgebraicGradeTensor(snapshot(files, profile), grade, row["keys"], availability=AVAILABLE)


def reseal(observation):
    observation["seal"] = sha256(pack_value({k: v for k, v in observation.items() if k != "seal"})).hexdigest()


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("grade", range(3))
def test_complete_fields_and_boundaries_match_original_occurrence_oracle(files, profile, grade):
    rows = oracle(files, profile)
    tensor = field(files, profile, grade)
    assert tensor.coefficients == fractions(rows[grade]["coordinates"])
    assert tensor.preview()["source_coordinates"] == tuple(map(tuple, rows[grade]["stalk_coordinates"]))
    assert tensor.cell_count == len(rows[grade]["keys"])
    assert tensor.record_keys == tuple(rows[grade]["keys"])
    assert tensor.matches_source(tensor.snapshot_owner) and tensor.same_owner(tensor)
    assert not tensor.matches_source(snapshot(files, profile))
    assert not tensor.derived and tensor.input_owner is None
    for i, coefficients in enumerate(tensor.coefficients):
        witness = tensor.source_coordinate(i)
        assert tensor.source_index(witness) == i
        assert tensor.coordinate(i).coefficients == (coefficients,)
        assert tensor.stalk(witness[1]).coefficients[witness[2]] == coefficients
    if grade and "boundary" in rows[grade]:
        first = tensor.boundary(availability=AVAILABLE)
        assert first.coefficients == fractions(rows[grade]["boundary"])
        assert first.input_owner is tensor and first.snapshot_owner is tensor.snapshot_owner
        assert first.derived and first.record_keys == ()
        if grade >= 2:
            twice = first.boundary(availability=AVAILABLE)
            assert twice.coefficients == fractions(rows[grade]["twice"])
            assert twice.input_owner is first
    else:
        with pytest.raises((ValueError, IndexError)): tensor.boundary(availability=AVAILABLE)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("grade", range(3))
def test_shape_windows_preserve_support_original_witnesses_and_exact_sum(files, profile, grade):
    tensor = field(files, profile, grade)
    size = tensor.domain[0]
    lo = min(1, size)
    window = NativeAlgebraicGradeTensorWindow(tensor, [1, size], [0, lo], [1, size], availability=AVAILABLE)
    assert window.tensor_owner is tensor
    assert window.domain == (1, size - lo)
    assert window.preview()["coefficients"] == tensor.coefficients[lo:]
    for i in range(window.supported_count):
        assert window.source_index(i) == i + lo
        assert window.window_index(i + lo) == i
        assert window.source_coordinate(i) == tensor.source_coordinate(i + lo)
        assert window.coordinate(i).coefficients[0] == tensor.coefficients[i + lo]
    if lo: assert window.window_index(0) is None
    observed = algebraic_grade_window_observation(window, availability=AVAILABLE)
    assert observed["detail"]["source_coordinates"] == tuple(tensor.source_coordinate(i) for i in range(lo, size))
    if not window.supported_count:
        assert observed["sum"] is None and window.sum(availability=AVAILABLE) is None
    else:
        expected = tuple(sum((row[j] for row in tensor.coefficients[lo:]), Fraction(0))
                         for j in range(tensor.snapshot_owner.archive_owner.field_degree))
        assert observed["sum"] == expected
        assert window.sum(availability=AVAILABLE).coefficients == (expected,)
    assert window.preview(0)["truncated"] == bool(window.supported_count)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("operation", ["field", "window"])
def test_complete_lineage_record_replays_actual_owners_after_gc(files, profile, operation):
    tensor = field(files, profile)
    steps = 2 if "twice" in oracle(files, profile)[2] else 0
    for _ in range(steps):
        tensor = tensor.boundary(availability=AVAILABLE)
    if operation == "window":
        size = tensor.domain[0]
        value = NativeAlgebraicGradeTensorWindow(tensor, [size], [0], [size], availability=AVAILABLE)
    else: value = tensor
    saved = algebraic_grade_record(value, availability=AVAILABLE)
    before = read_algebraic_grade_observation_record(saved)
    del tensor, value
    gc.collect()
    restored = read_algebraic_grade_record(saved)
    owner = restored if operation == "field" else restored.tensor_owner
    assert owner.derived == bool(steps)
    if steps:
        assert owner.input_owner.derived and not owner.input_owner.input_owner.derived
    else:
        assert owner.input_owner is None
    assert owner.snapshot_owner.graph_origin["archive_key"] == 9
    assert owner.snapshot_owner.graph_revision_owner.history_owner.identity["sequence"] == 31
    assert before == read_algebraic_grade_observation_record(algebraic_grade_record(restored, availability=AVAILABLE))
    assert len(before["lineage"]) == steps + 1
    assert all(type(q) is Fraction for row in before["detail"]["coefficients"] for q in row)


@pytest.mark.parametrize("availability", [(9603, 30), (9604, 31), (0, 31), (True, 31), (9603, 31.0)])
def test_each_operation_rechecks_temporal_availability(files, availability):
    tensor = field(files)
    window = NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [1], availability=AVAILABLE)
    for operation in (
        lambda: NativeAlgebraicGradeTensor(tensor.snapshot_owner, 2, [1200], availability=availability),
        lambda: tensor.boundary(availability=availability),
        lambda: NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [1], availability=availability),
        lambda: window.sum(availability=availability),
        lambda: algebraic_grade_observation(tensor, availability=availability),
    ):
        with pytest.raises((ValueError, TypeError)): operation()


@pytest.mark.parametrize("keys", [[], [1200, 1200], [1000], [0], [99999], [True], [1200.0], "1200", {}])
def test_selection_requires_exact_record_count_and_actual_original_stalk(files, keys):
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        NativeAlgebraicGradeTensor(snapshot(files), 2, keys, availability=AVAILABLE)


def test_selected_keys_are_owned_and_heterogeneous_offsets_remain_explicit(files):
    keys = [1000, 1001, 1002]
    tensor = NativeAlgebraicGradeTensor(snapshot(files), 0, keys, availability=AVAILABLE)
    keys.reverse()
    assert tensor.record_keys == (1000, 1001, 1002)
    assert tensor.stalk_offsets == (0, 3, 5, 7)
    with pytest.raises(ValueError): tensor.source_index((1, 0, 0))
    with pytest.raises((ValueError, IndexError)): tensor.source_index((0, 0, 3))
    with pytest.raises((ValueError, IndexError)): tensor.coordinate(7)
    with pytest.raises((ValueError, IndexError)): tensor.stalk(3)
    with pytest.raises((TypeError, ValueError)): tensor.source_index((False, 0, 0))


@pytest.mark.parametrize("shape,lower,upper", [([2], [0], [2]), ([1], [0, 0], [1]), ([1], [0], [2]),
    ([1], [1], [0]), ([True], [0], [1]), ([1.0], [0], [1]), ([1], [True], [1])])
def test_shape_windows_require_complete_grade_size_and_exact_bounds(files, shape, lower, upper):
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        NativeAlgebraicGradeTensorWindow(field(files), shape, lower, upper, availability=AVAILABLE)


def test_scalar_shape_and_empty_window_distinguish_supported_zero(files):
    tensor = field(files, 14)
    window = NativeAlgebraicGradeTensorWindow(tensor, [], [], [], availability=AVAILABLE)
    assert window.supported_count == 1 and window.domain == ()
    assert window.sum(availability=AVAILABLE).coefficients == ((Fraction(0),) * 3,)
    empty = NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [0], availability=AVAILABLE)
    assert empty.supported_count == 0 and empty.sum(availability=AVAILABLE) is None
    assert empty.window_index(0) is None
    with pytest.raises((ValueError, IndexError)): empty.window_index(1)


def test_cancellation_and_resource_admission_at_every_native_phase(files):
    tensor = field(files)
    window = NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [1], availability=AVAILABLE)
    token = CancellationToken(); token.cancel()
    for operation in (
        lambda: NativeAlgebraicGradeTensor(tensor.snapshot_owner, 2, [1200], availability=AVAILABLE, cancellation=token),
        lambda: tensor.boundary(availability=AVAILABLE, cancellation=token),
        lambda: NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [1], availability=AVAILABLE, cancellation=token),
        lambda: window.sum(availability=AVAILABLE, cancellation=token),
    ):
        with pytest.raises(ExecutionCancelled): operation()
    with pytest.raises(ExecutionLimitError):
        NativeAlgebraicGradeTensor(tensor.snapshot_owner, 2, [1200], availability=AVAILABLE, execution_limits={"records": 4})
    boundary = tensor.boundary(availability=AVAILABLE)
    for key in ("records", "rational_slots", "sparse_entries", "scalar_work", "integer_bits"):
        with pytest.raises(ExecutionLimitError):
            tensor.boundary(availability=AVAILABLE, execution_limits={key: boundary.requirements[key] - 1})
    with pytest.raises(ExecutionLimitError):
        NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [1], availability=AVAILABLE, execution_limits={"records": 4})
    with pytest.raises(ExecutionLimitError): window.sum(availability=AVAILABLE, execution_limits={"rational_slots": 0})
    strict = NativeAlgebraicGradeTensor(tensor.snapshot_owner, 2, [1200], availability=AVAILABLE, execution_limits={"scalar_work": 0})
    with pytest.raises(ExecutionLimitError): strict.boundary(availability=AVAILABLE, execution_limits={"scalar_work": 1000000})


@pytest.mark.parametrize("case", ["offset", "keys", "grade", "lineage", "basis", "partial", "literal", "extra", "seal", "growth"])
def test_closed_observations_reject_inconsistent_or_truncated_declarations(files, case):
    tensor = field(files).boundary(availability=AVAILABLE)
    observed = algebraic_grade_observation(tensor, availability=AVAILABLE)
    if case == "offset": observed["detail"]["stalk_offsets"] = (0, 2, 5, 6)
    elif case == "keys": observed["lineage"][0]["record_keys"] = ()
    elif case == "grade": observed["detail"]["source_coordinates"] = ((0, 0, 0),) * 6
    elif case == "lineage": observed["lineage"] = observed["lineage"][1:]
    elif case == "basis": observed["coefficient_basis"] = "float"
    elif case == "partial": observed["detail"]["truncated"] = True
    elif case == "literal": observed["detail"]["coefficients"] = ((1.0,) * 3,) * 6
    elif case == "extra": observed["private_notes"] = "unknown"
    elif case == "seal": observed["seal"] = "0" * 64
    else: observed["detail"]["coefficients"] = ((Fraction(1 << 300), Fraction(0), Fraction(0)),) * 6
    if case != "seal": reseal(observed)
    with pytest.raises((ValueError, TypeError)): validate_algebraic_grade_observation(observed, execution_limits={"integer_bits": 256})


@pytest.mark.parametrize("operation", ["field", "window"])
def test_native_replay_checks_values_beyond_a_portable_observation_seal(files, operation):
    tensor = field(files)
    value = tensor if operation == "field" else NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [1], availability=AVAILABLE)
    saved = algebraic_grade_record(value, availability=AVAILABLE)
    observed = saved.get_metadata(1, 0, "algebraic_grade_observation")
    observed["detail"]["coefficients"] = ((Fraction(99), Fraction(1), Fraction(0)),)
    if operation == "window": observed["sum"] = observed["detail"]["coefficients"][0]
    reseal(observed)
    assert read_algebraic_grade_observation_record(saved)["detail"]["coefficients"][0][0] == 99
    with pytest.raises(ValueError, match="native action"): read_algebraic_grade_record(saved)


@pytest.mark.parametrize("operation", ["field", "window"])
@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_exact_storage_round_trip_and_reader_admission(files, tmp_path, operation, suffix):
    from rexgraph.io import load, save
    tensor = field(files).boundary(availability=AVAILABLE)
    value = tensor if operation == "field" else NativeAlgebraicGradeTensorWindow(tensor, [2, 3], [0, 1], [2, 3], availability=AVAILABLE)
    saved = algebraic_grade_record(value, availability=AVAILABLE)
    path = tmp_path / f"grade.{suffix}"; save(path, saved)
    restored = load(path)
    assert read_algebraic_grade_observation_record(restored) == read_algebraic_grade_observation_record(saved)
    assert read_algebraic_grade_record(restored).preview() == value.preview()
    with pytest.raises(ValueError): read_algebraic_grade_observation_record(restored, max_coordinates=1)
    with pytest.raises(ValueError): read_algebraic_grade_observation_record(restored, max_archive_bytes=1)
    with pytest.raises(ValueError): read_algebraic_grade_record(restored, execution_limits={"scalar_work": 0})
    with pytest.raises(ValueError): read_algebraic_grade_record(restored, limits={"max_file_bytes": 1})


def test_fresh_process_core_only_observation_and_rcdb_copy(files, tmp_path):
    from rcdb import copy_record, open_store
    from rexgraph.io import save
    value = field(files).boundary(availability=AVAILABLE)
    saved = algebraic_grade_record(value, availability=AVAILABLE)
    stores = [open_store(f"sqlite:///{tmp_path / (name + '.sqlite')}") for name in ("first", "second")]
    try:
        stores[0].put("grade", saved)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("grade"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        assert read_algebraic_grade_record(stores[1].get_version("grade", 1)).coefficients == value.coefficients
    finally:
        for store in stores: store.close()
    path = tmp_path / "grade.rcbd"; save(path, saved)
    code = '''
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rexgraph_cpp','torch'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from rexgraph.io import load
from rexgraph.cpp_algebraic_grade import read_algebraic_grade_observation_record
print(json.dumps({'seal':read_algebraic_grade_observation_record(load(sys.argv[1]))['seal']}))
'''
    output = subprocess.check_output([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, text=True)
    assert json.loads(output)["seal"] == read_algebraic_grade_observation_record(saved)["seal"]
    code = '''
import sys
from rcdb import open_store
from rexgraph.cpp_algebraic_grade import read_algebraic_grade_record
store=open_store(sys.argv[1])
try:
 field=read_algebraic_grade_record(store.get_version('grade',1))
 assert field.derived and field.input_owner.grade==2 and field.snapshot_owner.graph_origin['sequence']==31
finally: store.close()
'''
    subprocess.run([sys.executable, "-I", "-c", code, f"sqlite:///{tmp_path / 'second.sqlite'}"], cwd=tmp_path, check=True)


def test_rcql_agent_and_system_use_the_same_complete_grade_contract(files):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    snap = snapshot(files)
    saved = algebraic_snapshot_record(snap)
    request = {"grade": 2, "record_keys": [1200], "boundary_steps": 1}
    text = "FROM $record RETURN ALGEBRAIC_GRADE($request,31), ALGEBRAIC_GRADE_WINDOW($request,[2,3],[0,1],[2,3],31)"
    result = Executor(sources={"record": saved}, params={"request": request}).execute(parse(text))
    runtime = RCQLRuntime(); runtime.register("record", saved, policy=SourcePolicy.allow("read"))
    assert runtime.execute(parse(text), params={"request": request}).values == result.values
    hive = Hive("algebraic-grade")
    hive.add_worker("boundary", lambda steps: prepare_algebraic_grade(snap, {**request, "boundary_steps": steps}, availability=AVAILABLE), capability="transform")
    tensor = hive.invoke("boundary", 1, record=False)
    window = NativeAlgebraicGradeTensorWindow(tensor, [2, 3], [0, 1], [2, 3], availability=AVAILABLE)
    assert result.values[0] == algebraic_grade_observation(tensor, availability=AVAILABLE)
    assert result.values[1] == algebraic_grade_window_observation(window, availability=AVAILABLE)
    for value in (tensor, window): json.dumps(json_value(value, max_values=1), allow_nan=False)
    denied = RCQLRuntime(); denied.register("record", saved, policy=SourcePolicy.allow())
    with pytest.raises(PermissionError): denied.execute(parse(text), params={"request": request})


def test_immutable_handles_and_concurrent_boundary_owners(files):
    tensor = field(files)
    window = NativeAlgebraicGradeTensorWindow(tensor, [1], [0], [1], availability=AVAILABLE)
    with pytest.raises(RuntimeError): tensor.__init__(tensor.snapshot_owner, 2, [1200], availability=AVAILABLE)
    with pytest.raises(RuntimeError): window.__init__(tensor, [1], [0], [1], availability=AVAILABLE)
    for klass in (NativeAlgebraicGradeTensor, NativeAlgebraicGradeTensorWindow):
        with pytest.raises(RuntimeError): klass.__new__(klass).preview()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: tensor.boundary(availability=AVAILABLE), range(8)))
    assert all(value.input_owner is tensor and value.coefficients == results[0].coefficients for value in results)
    assert not results[0].same_owner(results[1])
    with pytest.raises(ValueError): algebraic_grade_observation(tensor, availability=AVAILABLE, max_coordinates=2)


def test_new_module_is_installed_in_core():
    assert importlib.import_module("rexgraph.cpp_algebraic_grade").prepare_algebraic_grade is prepare_algebraic_grade


@pytest.mark.parametrize("case", ["source_shape", "bounds", "indices", "witness", "support", "sum", "policy"])
def test_window_observation_rejects_resealed_inconsistent_coordinates(files, case):
    tensor = field(files).boundary(availability=AVAILABLE)
    window = NativeAlgebraicGradeTensorWindow(tensor, [2, 3], [0, 1], [2, 3], availability=AVAILABLE)
    observed = algebraic_grade_window_observation(window, availability=AVAILABLE)
    detail = observed["detail"]
    if case == "source_shape": detail["source_domain"] = (2, 4)
    elif case == "bounds": detail["upper"] = (3, 3)
    elif case == "indices": detail["source_indices"] = (0, 2, 4, 5)
    elif case == "witness": detail["source_coordinates"] = ((1, 0, 0),) * 4
    elif case == "support": detail["supported_count"] = 3
    elif case == "sum": observed["sum"] = (Fraction(99), Fraction(0), Fraction(0))
    else: detail["execution_limits"]["scalar_work"] += 1
    reseal(observed)
    with pytest.raises(ValueError): validate_algebraic_grade_observation(observed)


def test_retained_source_bytes_are_checked_before_native_replay(files):
    saved = algebraic_grade_record(field(files), availability=AVAILABLE)
    attachment = saved.get_metadata(1, 0, "algebraic_archive")
    attachment["archive"][-1] ^= 1
    with pytest.raises(ValueError): read_algebraic_grade_observation_record(saved)
    with pytest.raises(ValueError): read_algebraic_grade_record(saved)


@pytest.mark.parametrize("kind", ["tensor", "window"])
def test_initialization_guard_owns_declaration_callbacks_before_releasing_gil(files, kind):
    from threading import Event
    entered, release = Event(), Event()

    class BlockingSequence:
        def __len__(self):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("declaration timeout")
            return 1

        def __iter__(self):
            return iter([1200] if kind == "tensor" else [1])

    tensor = field(files)
    if kind == "tensor":
        owner = NativeAlgebraicGradeTensor.__new__(NativeAlgebraicGradeTensor)
        first_args = (tensor.snapshot_owner, 2, BlockingSequence())
        second_args = (tensor.snapshot_owner, 2, [1200])
    else:
        owner = NativeAlgebraicGradeTensorWindow.__new__(NativeAlgebraicGradeTensorWindow)
        first_args = (tensor, BlockingSequence(), [0], [1])
        second_args = (tensor, [1], [0], [1])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.__init__, *first_args, availability=AVAILABLE)
        try:
            assert entered.wait(5)
            second = pool.submit(owner.__init__, *second_args, availability=AVAILABLE)
            with pytest.raises(RuntimeError, match="immutable"):
                second.result(timeout=2)
        finally:
            release.set()
        first.result(timeout=5)
    assert owner.preview()["coefficients"] == tensor.coefficients


@pytest.mark.parametrize("profile", range(1, 11))
@pytest.mark.parametrize("grade", [3, 4])
def test_upper_grades_preserve_original_actions_windows_and_full_lineage(files, profile, grade):
    test_complete_fields_and_boundaries_match_original_occurrence_oracle(files, profile, grade)
    test_shape_windows_preserve_support_original_witnesses_and_exact_sum(files, profile, grade)
    tensor = field(files, profile, grade)
    for _ in range(grade):
        tensor = tensor.boundary(availability=AVAILABLE)
    saved = algebraic_grade_record(tensor, availability=AVAILABLE)
    assert len(read_algebraic_grade_observation_record(saved)["lineage"]) == grade + 1
    assert read_algebraic_grade_record(saved).coefficients == tensor.coefficients
