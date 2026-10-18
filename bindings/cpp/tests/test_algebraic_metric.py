"""Whole grade metric actions preserve original carriers and complete exact inputs."""
import gc
import json
import os
import subprocess
import sys
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import pytest
from rexgraph_cpp import (
    CancellationToken,
    NativeAlgebraicMetricField,
    NativeAlgebraicMetricValues,
)
from test_algebraic_grade import AVAILABLE, oracle, snapshot
from test_algebraic_stalk import multiply

from rexgraph.cpp_algebraic_grade import prepare_algebraic_grade
from rexgraph.cpp_algebraic_grade_model import cpp_algebraic_grade_model
from rexgraph.cpp_algebraic_metric import (
    algebraic_grade_metric_observation,
    algebraic_grade_metric_record,
    read_algebraic_grade_metric_observation_record,
    read_algebraic_grade_metric_record,
    restore_algebraic_grade_metric_observation,
    validate_algebraic_grade_metric_observation,
)
from rexgraph.cpp_algebraic_query import algebraic_snapshot_record
from rexgraph.io import load, save
from rexgraph.model_runtime import infer_model
from rexgraph.model_state import thaw_tree
from rexgraph.value_codec import pack_value

PROFILES = (*range(1, 11), 13, 14, 15)
CASES = [(p, k) for p in PROFILES for k in range(5 if p <= 10 else 3)]
ACTIONS = [(p, k, op) for p, k in CASES for op in ("hodge", "coboundary", "inner", "adjacent_quadrance")
           if op != "coboundary" or k < (4 if p <= 10 else 2)]


@pytest.fixture(scope="session")
def files():
    return Path(os.environ["REXGRAPH_CPP_METRIC_FIXTURES"])


def metrics(source):
    return tuple(5000 + 100 * k + cell for k, count in enumerate(source.archive_owner.manifest["source_sizes"]) for cell in range(count))


def setup(files, profile=13, grade=1, steps=0):
    source = snapshot(files, profile)
    keys = oracle(files, profile)[grade]["keys"]
    tensor = prepare_algebraic_grade(source, {"grade": grade, "record_keys": keys, "boundary_steps": steps}, availability=AVAILABLE)
    field = NativeAlgebraicMetricField(source, metrics(source), availability=AVAILABLE)
    return field, tensor


def add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def neg(a):
    return tuple(-x for x in a)


def rational_solve(matrix, rhs):
    n = len(rhs)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if a[i][col])
        a[col], a[pivot] = a[pivot], a[col]
        factor = a[col][col]; a[col] = [x / factor for x in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]; a[row] = [x - factor * y for x, y in zip(a[row], a[col], strict=True)]
    return tuple(row[-1] for row in a)


@lru_cache(maxsize=2048)
def inverse(value, polynomial):
    degree = len(polynomial) - 1
    columns = [multiply(value, tuple(Fraction(i == j) for i in range(degree)), polynomial) for j in range(degree)]
    return rational_solve(tuple(zip(*columns, strict=True)), tuple(Fraction(i == 0) for i in range(degree)))


def solve(matrix, rhs, polynomial):
    n = len(rhs); zero = (Fraction(0),) * (len(polynomial) - 1)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if a[i][col] != zero)
        a[col], a[pivot] = a[pivot], a[col]
        factor = inverse(a[col][col], polynomial)
        a[col] = [multiply(x, factor, polynomial) for x in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [add(x, neg(multiply(factor, y, polynomial))) for x, y in zip(a[row], a[col], strict=True)]
    return tuple(row[-1] for row in a)


def matvec(matrix, vector, polynomial):
    zero = (Fraction(0),) * (len(polynomial) - 1)
    out = []
    for row in matrix:
        value = zero
        for entry, coordinate in zip(row, vector, strict=True): value = add(value, multiply(entry, coordinate, polynomial))
        out.append(value)
    return tuple(out)


def conjugate(value, involution, polynomial):
    degree = len(polynomial) - 1; power = (Fraction(1),) + (Fraction(0),) * (degree - 1)
    out = (Fraction(0),) * degree
    for coefficient in value:
        out = add(out, tuple(coefficient * x for x in power)); power = multiply(power, involution, polynomial)
    return out


@lru_cache(maxsize=32)
def declared_oracle(folder, profile):
    source = snapshot(Path(folder), profile); archive = source.archive_owner
    polynomial = archive.manifest["minimal_polynomial"]; degree = archive.field_degree
    zero = (Fraction(0),) * degree
    shapes, matrices, boundaries = [], [], []
    for grade, count in enumerate(archive.manifest["source_sizes"]):
        selected = [archive.record(5000 + 100 * grade + cell) for cell in range(count)]
        ranks = [item.shape[0] for item in selected]; offsets = [0]
        for rank in ranks: offsets.append(offsets[-1] + rank)
        shapes.append(tuple(offsets)); n = offsets[-1]
        matrix = [[zero for _ in range(n)] for _ in range(n)]
        for cell, metric in enumerate(selected):
            rank = ranks[cell]
            for row in range(rank):
                for col in range(rank): matrix[offsets[cell] + row][offsets[cell] + col] = metric.coefficients[row * rank + col]
        matrices.append(tuple(map(tuple, matrix)))
        if grade == 0: boundaries.append(None); continue
        boundary = [[zero for _ in range(n)] for _ in range(shapes[grade - 1][-1])]
        for slot in range(1000):
            try: occurrence = source.occurrence(grade, slot)
            except IndexError: break
            if not occurrence["active"]: continue
            value = occurrence["underlying"]; info = value.preview(0)
            upper, lower = info["source_carrier"], info["target_carrier"]
            for row in range(lower["rank"]):
                for col in range(upper["rank"]):
                    a = shapes[grade - 1][lower["index"]] + row; b = offsets[upper["index"]] + col
                    q = value.coefficients[row * upper["rank"] + col]
                    boundary[a][b] = add(boundary[a][b], tuple(occurrence["coefficient"] * x for x in q))
        else: raise AssertionError("fixture occurrence bound exceeded")
        boundaries.append(tuple(map(tuple, boundary)))
    return polynomial, archive.manifest["involution"], tuple(shapes), tuple(matrices), tuple(boundaries)


def exact_actions(files, profile, grade, x):
    p, involution, shapes, matrices, boundaries = declared_oracle(str(files), profile)
    zero = (Fraction(0),) * (len(p) - 1)
    def pairing(k, a, b):
        result = zero
        for u, v in zip(a, matvec(matrices[k], b, p), strict=True): result = add(result, multiply(conjugate(u, involution, p), v, p))
        return result
    def adjoint(k, y):
        weighted = matvec(matrices[k - 1], y, p)
        transpose = tuple(tuple(conjugate(boundaries[k][i][j], involution, p) for i in range(len(weighted)))
                          for j in range(shapes[k][-1]))
        return solve(matrices[k], matvec(transpose, weighted, p), p)
    energy = zero; hodge = tuple(zero for _ in x)
    if grade:
        down = matvec(boundaries[grade], x, p)
        hodge = adjoint(grade, down); energy = add(energy, pairing(grade - 1, down, down))
    up = None
    if grade + 1 < len(shapes):
        up = adjoint(grade + 1, x); energy = add(energy, pairing(grade + 1, up, up))
        hodge = tuple(add(a, b) for a, b in zip(hodge, matvec(boundaries[grade + 1], up, p), strict=True))
    assert pairing(grade, x, hodge) == energy
    return {"hodge": hodge, "coboundary": up, "inner": (pairing(grade, x, x),), "adjacent_quadrance": (energy,)}


@pytest.mark.parametrize("profile,grade,operation", ACTIONS)
def test_native_actions_match_independent_exact_polynomial_oracle(files, profile, grade, operation):
    field, tensor = setup(files, profile, grade)
    expected = exact_actions(files, profile, grade, tensor.coefficients)[operation]
    other = tensor if operation == "inner" else None
    value = algebraic_grade_metric_observation(field, tensor, operation, availability=AVAILABLE, other=other)
    assert value["detail"]["coefficients"] == expected
    assert value["input"]["detail"]["coefficients"] == tensor.coefficients
    if operation in {"hodge", "coboundary"}:
        native = field.apply(tensor, operation, availability=AVAILABLE)
        assert type(native) is NativeAlgebraicMetricValues
        assert native.input_owner is tensor and native.field_owner is field and native.snapshot_owner is tensor.snapshot_owner
        assert native.grade == grade + (operation == "coboundary")
        assert native.preview(0)["truncated"] == bool(native.domain[0])
        for cell in range(native.cell_count):
            a, b = native.stalk_offsets[cell:cell + 2]
            assert native.stalk(cell).coefficients == expected[a:b]
            assert native.stalk(cell).archive_owner is tensor.snapshot_owner.archive_owner
        if operation == "hodge":
            assert field.adjacent_quadrance(tensor, availability=AVAILABLE).coefficients == exact_actions(files, profile, grade, tensor.coefficients)["adjacent_quadrance"]


@pytest.mark.parametrize("profile", [1, 5, 9, 13, 15])
@pytest.mark.parametrize("operation", ["hodge", "coboundary", "inner", "adjacent_quadrance"])
@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_complete_metric_records_roundtrip_and_retain_boundary_inputs(files, profile, operation, suffix, tmp_path):
    field, tensor = setup(files, profile, 2, 1)
    other = tensor if operation == "inner" else None
    record = algebraic_grade_metric_record(field, tensor, operation, availability=AVAILABLE, other=other)
    expected = read_algebraic_grade_metric_observation_record(record)
    assert len(expected["input"]["lineage"]) == 2
    path = tmp_path / ("metric" + suffix); save(path, record)
    del field, tensor, record, other; gc.collect()
    observation, source, (field, tensor, other, result) = read_algebraic_grade_metric_record(load(path))
    assert observation == expected and result.coefficients == expected["detail"]["coefficients"]
    assert tensor.matches_source(source) and tensor.input_owner is not None and field.snapshot_owner is source
    assert (other is not None) == (operation == "inner")


@pytest.mark.parametrize("case", ["missing", "duplicate", "vector", "absent", "limit", "owner", "cutoff", "namespace", "cancel", "grade", "operation"])
def test_metric_admission_and_actual_owners_are_required(files, case):
    field, tensor = setup(files); source = tensor.snapshot_owner; keys = metrics(source)
    with pytest.raises((ValueError, IndexError, OverflowError, RuntimeError)):
        if case == "missing": NativeAlgebraicMetricField(source, keys[:-1], availability=AVAILABLE)
        elif case == "duplicate": NativeAlgebraicMetricField(source, (*keys[:-1], keys[0]), availability=AVAILABLE)
        elif case == "vector": NativeAlgebraicMetricField(source, (1000, *keys[1:]), availability=AVAILABLE)
        elif case == "absent": setup(files, 11)
        elif case == "limit": NativeAlgebraicMetricField(source, keys, availability=AVAILABLE, max_retained_coordinates=0)
        elif case == "owner": field.apply(setup(files)[1], "hodge", availability=AVAILABLE)
        elif case == "cutoff": field.apply(tensor, "hodge", availability=(9603, 30))
        elif case == "namespace": field.inner(tensor, tensor, availability=(9999, 31))
        elif case == "cancel":
            token = CancellationToken(); token.cancel(); field.adjacent_quadrance(tensor, availability=AVAILABLE, cancellation=token)
        elif case == "grade": field.inner(tensor, prepare_algebraic_grade(source, {"grade": 0, "record_keys": [1000, 1001, 1002], "boundary_steps": 0}, availability=AVAILABLE), availability=AVAILABLE)
        else: field.apply(tensor, "boundary", availability=AVAILABLE)


@pytest.mark.parametrize("case", ["offset", "coordinate", "coefficients", "metric_keys", "publication", "engine", "input", "extra", "count"])
def test_closed_portable_metric_receipts_reject_changed_declarations(files, case):
    field, tensor = setup(files); value = algebraic_grade_metric_observation(field, tensor, "hodge", availability=AVAILABLE)
    value = deepcopy(value)
    if case == "offset": value["detail"]["stalk_offsets"] = (0, 3, 5, 6)
    elif case == "coordinate": value["detail"]["source_coordinates"] = ((1, 9, 0), *value["detail"]["source_coordinates"][1:])
    elif case == "coefficients": value["detail"]["coefficients"] = value["detail"]["coefficients"][:-1]
    elif case == "metric_keys": value["metric_keys"] = value["metric_keys"][:-1]
    elif case == "publication": value["availability"] = (9603, 30)
    elif case == "engine": value["engine"]["engine_digest"] = "0" * 64
    elif case == "input": value["input"]["detail"]["coefficients"] = ()
    elif case == "extra": value["extra"] = True
    else: value["other"] = deepcopy(value["input"])
    value["seal"] = sha256(pack_value({k: v for k, v in value.items() if k != "seal"})).hexdigest()
    with pytest.raises((ValueError, TypeError)): validate_algebraic_grade_metric_observation(value)


def test_resealed_false_values_require_native_replay_and_reader_admission(files):
    field, tensor = setup(files); value = algebraic_grade_metric_observation(field, tensor, "hodge", availability=AVAILABLE)
    false = deepcopy(value); rows = list(false["detail"]["coefficients"]); rows[0] = tuple(q + 1 for q in rows[0]); false["detail"]["coefficients"] = tuple(rows)
    false["seal"] = sha256(pack_value({k: v for k, v in false.items() if k != "seal"})).hexdigest()
    validate_algebraic_grade_metric_observation(false)
    with pytest.raises(ValueError): restore_algebraic_grade_metric_observation(tensor.snapshot_owner, false)
    with pytest.raises((ValueError, OverflowError, RuntimeError)): restore_algebraic_grade_metric_observation(tensor.snapshot_owner, value, max_retained_coordinates=0)
    with pytest.raises(ValueError): validate_algebraic_grade_metric_observation(value, max_coordinates=0)
    with pytest.raises((ValueError, OverflowError, RuntimeError)): restore_algebraic_grade_metric_observation(tensor.snapshot_owner, value, execution_limits={"records": 0})


def test_whole_grade_model_output_feeds_canonical_metric_consumer(files):
    field, _ = setup(files); source = field.snapshot_owner
    state = cpp_algebraic_grade_model(source, grade=2, record_keys=[1200], boundary_steps=1, availability=AVAILABLE)
    output = infer_model(state, source)
    from rexgraph.cpp_algebraic_grade import restore_algebraic_grade_observation
    tensor = restore_algebraic_grade_observation(source, thaw_tree(output.execution_receipt)["observation"])
    value = algebraic_grade_metric_observation(field, tensor, "hodge", availability=AVAILABLE)
    assert value["input"]["detail"]["coefficients"] == tuple(tuple(row) for row in output.values)


def test_initialization_reentrancy_and_result_ownership(files):
    field, tensor = setup(files); result = field.apply(tensor, "hodge", availability=AVAILABLE)
    with pytest.raises(RuntimeError): field.__init__(tensor.snapshot_owner, metrics(tensor.snapshot_owner), availability=AVAILABLE)
    for value in (NativeAlgebraicMetricField.__new__(NativeAlgebraicMetricField), NativeAlgebraicMetricValues.__new__(NativeAlgebraicMetricValues)):
        with pytest.raises(RuntimeError): _ = value.snapshot_owner
    empty = NativeAlgebraicMetricField.__new__(NativeAlgebraicMetricField)
    selected_source = tensor.snapshot_owner
    selected_keys = metrics(selected_source)
    class Key(int):
        def __int__(self):
            with pytest.raises(RuntimeError): empty.__init__(selected_source, selected_keys, availability=AVAILABLE)
            return selected_keys[0]
    keys = metrics(tensor.snapshot_owner)
    empty.__init__(tensor.snapshot_owner, [Key(keys[0]), *keys[1:]], availability=AVAILABLE)
    expected = result.coefficients; del field, tensor; gc.collect()
    assert result.coefficients == expected and result.input_owner.matches_source(result.snapshot_owner)


def test_rcql_agent_system_keep_exact_metric_declarations_and_read_policy(files):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    field, tensor = setup(files); record = algebraic_snapshot_record(tensor.snapshot_owner)
    request = {"grade": 1, "record_keys": [1100, 1101, 1102], "boundary_steps": 0}
    query = parse('FROM $record RETURN ALGEBRAIC_GRADE_METRIC($request,$keys,"hodge",31)')
    params = {"request": request, "keys": field.metric_keys}
    values = Executor(sources={"record": record}, params=params).execute(query).values
    runtime = RCQLRuntime(); runtime.register("record", record, policy=SourcePolicy.allow("read"))
    assert runtime.execute(query, params=params).values == values
    denied = RCQLRuntime(); denied.register("record", record, policy=SourcePolicy.allow())
    with pytest.raises(PermissionError): denied.execute(query, params=params)
    hive = Hive("grade-metric"); hive.add_worker("hodge", lambda t: field.apply(t, "hodge", availability=AVAILABLE), capability="transform")
    native = hive.invoke("hodge", tensor, record=False)
    assert native.coefficients == values[0]["detail"]["coefficients"]
    preview = json_value(native, max_values=1)
    assert preview["kind"] == "NativeAlgebraicMetricValues"
    json.dumps(preview, allow_nan=False); json.dumps(json_value(values[0], max_values=1), allow_nan=False)


@pytest.mark.parametrize("backend", ["sqlite", "file", "rex"])
def test_metric_records_copy_and_restore_in_fresh_process(files, backend, tmp_path):
    from rcdb import copy_record, open_store
    field, tensor = setup(files)
    record = algebraic_grade_metric_record(field, tensor, "hodge", availability=AVAILABLE)
    expected = read_algebraic_grade_metric_observation_record(record)["seal"]
    urls = [f"sqlite:///{tmp_path / name}" if backend == "sqlite" else f"{backend}://{tmp_path / name}" for name in ("first", "second")]
    stores = [open_store(url, **({} if backend == "sqlite" else {"read_only": False})) for url in urls]
    try:
        stores[0].put("metric", record); copy_record(stores[0], stores[1], stores[0].get_record("metric"))
    finally:
        for store in stores: store.close()
    code = '''import sys
from rcdb import open_store
from rexgraph.cpp_algebraic_metric import read_algebraic_grade_metric_record
store=open_store(sys.argv[1])
try:
 value,source,owners=read_algebraic_grade_metric_record(store.get_version('metric',1))
 assert owners[1].matches_source(source)
 print(value['seal'])
finally:store.close()
'''
    assert subprocess.check_output([sys.executable, "-I", "-c", code, urls[1]], cwd=tmp_path, text=True).strip() == expected


def test_complete_retained_admission_counts_inactive_occurrences_and_metric_order(files):
    field, tensor = setup(files, 1, 1); source = tensor.snapshot_owner; degree = source.archive_owner.field_degree
    count = sum(source.archive_owner.record(key).preview(0)["coefficient_count"] for key in field.metric_keys)
    inactive = 0
    for grade in range(1, len(source.archive_owner.manifest["source_sizes"])):
        for slot in range(1000):
            try: occurrence = source.occurrence(grade, slot)
            except IndexError: break
            count += occurrence["underlying"].preview(0)["coefficient_count"]
            inactive += not occurrence["active"]
    assert inactive
    bound = count * degree * degree
    prepared = NativeAlgebraicMetricField(source, tuple(reversed(field.metric_keys)), availability=AVAILABLE, max_retained_coordinates=bound)
    assert prepared.apply(tensor, "hodge", availability=AVAILABLE).coefficients == field.apply(tensor, "hodge", availability=AVAILABLE).coefficients
    with pytest.raises((ValueError, OverflowError, RuntimeError)): NativeAlgebraicMetricField(source, field.metric_keys, availability=AVAILABLE, max_retained_coordinates=bound - 1)


def test_concurrent_metric_actions_share_immutable_owners(files):
    from concurrent.futures import ThreadPoolExecutor
    field, tensor = setup(files, 9, 2)
    def action(_): return field.apply(tensor, "hodge", availability=AVAILABLE)
    with ThreadPoolExecutor(max_workers=4) as executor: values = list(executor.map(action, range(12)))
    assert all(value.coefficients == values[0].coefficients and value.input_owner is tensor and value.field_owner is field for value in values)


def test_cross_pairing_matches_independent_hermitian_oracle(files):
    field, tensor = setup(files, 5, 1); source = tensor.snapshot_owner
    other = prepare_algebraic_grade(source, {"grade": 2, "record_keys": oracle(files, 5)[2]["keys"], "boundary_steps": 1}, availability=AVAILABLE)
    p, involution, _, matrices, _ = declared_oracle(str(files), 5)
    expected = (Fraction(0),) * (len(p) - 1)
    for a, b in zip(tensor.coefficients, matvec(matrices[1], other.coefficients, p), strict=True):
        expected = add(expected, multiply(conjugate(a, involution, p), b, p))
    observed = algebraic_grade_metric_observation(field, tensor, "inner", availability=AVAILABLE, other=other)
    assert observed["detail"]["coefficients"] == (expected,)
    reverse = field.inner(other, tensor, availability=AVAILABLE).coefficients[0]
    assert reverse == conjugate(expected, involution, p)
