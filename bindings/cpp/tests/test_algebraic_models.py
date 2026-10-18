"""Exact algebraic model declarations retain field owners, carriers and portable coordinates."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from rexgraph_cpp import NativeAlgebraicArchive
from test_algebraic_archive import algebraic_fixtures as algebraic_fixtures
from test_algebraic_archive import archive_case as archive_case
from test_algebraic_archive import exact_rows

from rexgraph.cpp_algebraic import (
    cpp_algebraic_input,
    cpp_algebraic_model,
    enable_cpp_algebraic_provider,
)
from rexgraph.io.algebraic_record import (
    _AlgebraicRestoreCache,
    algebraic_record,
    read_algebraic_record,
)
from rexgraph.io.model_record import (
    load_model,
    model_history,
    model_record,
    read_model_record,
    save_model,
)
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.model_state import ModelBatch, model_input, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference, source_from_record
from rexgraph.state import from_state, to_state
from rexgraph.type_accession import CoordinateSpace
from rexgraph.value_codec import pack_value, unpack_spec, unpack_value


def declaration(archive, **options):
    return cpp_algebraic_model(archive, **dict({"operator_key": 30, "input_key": 10}, **options))


@pytest.mark.parametrize("operation,key,oracle_key,kind", [
    ("apply", 30, "map", "vector"), ("solve", 50, "solve", "vector"),
    ("evaluate", 20, "evaluate", "scalar"), ("quadrance", 50, "quadrance", "scalar"),
    ("as_covector", 50, "covector", "covector"), ("accession", 100, "accession", "vector"),
    ("reconstruct", 100, "reconstruct", "vector"), ("project", 100, "accession", "vector")])
def test_model_actions_match_canonical_exact_oracle(archive_case, operation, key, oracle_key, kind):
    archive, oracle = archive_case
    physical = operation in {"reconstruct", "project"}
    state = declaration(archive, operation=operation, operator_key=key,
                        input_key=90 if physical else 10, input_side="target" if physical else "source")
    inputs = archive.accession(100, archive.record(10)) if physical else archive.record(10)
    output = infer_model(state, archive, inputs)
    assert state.arithmetic == output.arithmetic == "algebraic"
    assert tuple(map(tuple, output.values)) == exact_rows(oracle[oracle_key])
    assert output.values.shape == (3 if operation in {"accession", "project"} else 1 if kind == "scalar" else 2,
                                   archive.manifest["field_degree"])
    assert output.execution_receipt["kind"] == kind
    assert output.execution_receipt["engine"] == state.configuration["engine_receipt"]
    assert output.execution_receipt["minimal_polynomial"] == archive.manifest["minimal_polynomial"]
    assert output.variance == "coordinate"
    assert output.grade == (None if kind == "scalar" else 0)
    assert output.model_digest == state.coefficient_digest
    assert all(isinstance(q, Fraction) for q in output.values.flat)
    captured = cpp_algebraic_input(state, archive, inputs)
    assert infer_model(state, archive, captured).coefficient_digest == output.coefficient_digest


def test_exact_custom_inputs_use_native_module_and_field(archive_case):
    archive, _ = archive_case
    state = declaration(archive)
    degree = archive.manifest["field_degree"]
    coefficients = tuple(tuple(Fraction(((-1) ** j) * (2 ** 130 + i + j), 7) for j in range(degree)) for i in range(2))
    value = archive.vector(30, coefficients)
    assert value.coefficients == coefficients
    captured = cpp_algebraic_input(state, archive, coefficients)
    assert captured["coefficients"] == coefficients
    assert tuple(map(tuple, infer_model(state, archive, captured).values)) == archive.record(30).apply(value).coefficients
    assert archive.vector(50, coefficients, carrier_side="target").coefficients == coefficients
    source = state.configuration["input_carrier"]
    assert source["rank"] == 2 and source["field_degree"] == degree
    assert len(state.space.keys) == 2 and len(state.output_axes[0].keys) == degree
    assert state.space.name != state.output_axes[0].name
    assert state.space.keys != tuple(str(i) for i in range(archive.manifest["source_sizes"][0]))


@pytest.mark.parametrize("change", ["float", "boolean", "complex", "text", "incomplete", "extra", "rank", "side", "key", "bits"])
def test_vector_construction_rejects_undeclared_or_unbounded_coordinates(archive_case, change):
    archive, _ = archive_case
    rows = [list(v) for v in archive.record(10).coefficients]
    side, key = "source", 10
    if change == "float": rows[0][0] = 0.5
    elif change == "boolean": rows[0][0] = True
    elif change == "complex": rows[0][0] = 1j
    elif change == "text": rows[0][0] = "1/3"
    elif change == "incomplete": rows[0].pop()
    elif change == "extra": rows[0].append(Fraction(0))
    elif change == "rank": rows.pop()
    elif change == "side": side = "unspecified"
    elif change == "key": key = True
    elif change == "bits": rows[0][0] = 1 << archive.limits["field.max_integer_bits"]
    with pytest.raises((TypeError, ValueError, OverflowError)):
        archive.vector(key, rows, carrier_side=side)


def test_foreign_owner_and_equal_rank_carrier_reject(archive_case):
    archive, _ = archive_case
    state = declaration(archive)
    other = NativeAlgebraicArchive(archive.archive)
    with pytest.raises(ValueError, match="owner"):
        infer_model(state, archive, other.record(10))
    with pytest.raises(ValueError, match="carrier"):
        infer_model(state, archive, archive.vector(30, archive.record(10).coefficients, carrier_side="target"))
    with pytest.raises(ValueError, match="carrier"):
        infer_model(state, archive, archive.record(20))
    with pytest.raises(TypeError, match="owning"):
        infer_model(state, archive, archive.record(10).coefficients)
    assert infer_model(state, other).coefficient_digest == infer_model(state, archive).coefficient_digest


@pytest.mark.parametrize("native", [False, True])
def test_source_model_output_and_input_codecs_retain_exact_contract(archive_case, native):
    archive, _ = archive_case
    ref = CppAlgebraicSourceReference(archive, "algebraic-data", 7)
    state = declaration(archive, reference=ref)
    output = infer_model(state, archive)
    assert source_from_record(ref.as_record(), allow_cpp=True).matches(ref)
    with pytest.raises(ValueError, match="schema"):
        source_from_record(ref.as_record())
    for value in (state, output):
        encoded = pack_model(value, native=native)
        assert unpack_spec(encoded)["version"] == 2
        restored = unpack_model(encoded)
        assert restored.coefficient_digest == value.coefficient_digest
        assert isinstance(restored.source, CppAlgebraicSourceReference)
        assert restored.source.source is None and restored.source.native == ref.native
        assert restored.source.record_id == "algebraic-data" and restored.source.version == 7
    inputs = cpp_algebraic_input(state, archive, archive.record(10))
    portable = unpack_value(pack_value(thaw_tree(inputs)))
    assert infer_model(state, archive, portable).coefficient_digest == output.coefficient_digest


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_complete_model_record_roundtrip(archive_case, tmp_path, suffix):
    archive, _ = archive_case
    state = declaration(archive, operation="solve", operator_key=50)
    output = infer_model(state, archive)
    inputs = cpp_algebraic_input(state, archive, archive.record(10))
    record = from_state(to_state(model_record(archive, state)))
    record.attach_metadata(1, 0, "prediction", output)
    record.attach_metadata(1, 0, "model_input", thaw_tree(inputs))
    path = tmp_path / ("algebraic-model" + suffix)
    save_model(path, record)
    restored = load_model(path)
    saved, loaded = read_model_record(restored)
    assert loaded is not archive and loaded.archive == archive.archive and loaded.manifest == archive.manifest
    assert saved.coefficient_digest == state.coefficient_digest
    assert infer_model(saved, loaded, restored.get_metadata(1, 0, "model_input")).coefficient_digest == output.coefficient_digest
    assert restored.get_metadata(1, 0, "prediction").coefficient_digest == output.coefficient_digest


def test_vector_output_composes_through_explicit_carrier_rebinding(archive_case):
    archive, _ = archive_case
    accession = declaration(archive, operator_key=100, operation="accession")
    reconstruction = declaration(archive, operator_key=100, operation="reconstruct", input_key=90, input_side="target")
    output = unpack_model(pack_model(infer_model(accession, archive)))
    saved, owner = read_model_record(model_record(archive, reconstruction))
    captured = cpp_algebraic_input(saved, owner, output)
    assert tuple(map(tuple, infer_model(saved, owner, captured).values)) == archive.record(10).coefficients
    with pytest.raises(ValueError, match="carrier"):
        cpp_algebraic_input(accession, archive, output)
    with pytest.raises(TypeError, match="explicit"):
        infer_model(reconstruction, archive)


@pytest.mark.parametrize("route", ["tensor", "recorded_binary", "model_input", "train", "transport", "history"])
def test_unsupported_numeric_and_learning_routes_reject(archive_case, route):
    archive, _ = archive_case
    state = declaration(archive)
    output = infer_model(state, archive)
    with pytest.raises((TypeError, NotImplementedError)):
        if route == "tensor": output.tensor()
        elif route == "recorded_binary": output.recorded_binary()
        elif route == "model_input": model_input(output)
        elif route == "train":
            batch = ModelBatch(state.source, state.space, np.zeros(2, dtype=object), np.ones(2, dtype=bool))
            train_model(state, archive, batch)
        elif route == "transport": transport_model(state, archive, CppAlgebraicSourceReference(archive), {})
        else: model_history([state], [archive], [Fraction(0)])
    assert enable_cpp_algebraic_provider().operations == ("create", "prepare", "infer")


@pytest.mark.parametrize("change", ["source", "carrier", "side", "profile", "extra"])
def test_portable_inputs_reject_altered_declarations(archive_case, change):
    archive, _ = archive_case
    state = declaration(archive)
    record = thaw_tree(cpp_algebraic_input(state, archive, archive.record(10)))
    if change == "source": record["source"]["record_id"] = "other"
    elif change == "carrier": record["carrier_record"] = 11
    elif change == "side": record["carrier_side"] = "target"
    elif change == "profile": record["profile"] = "raw-coordinates"
    else: record["extra"] = True
    with pytest.raises(ValueError):
        infer_model(state, archive, record)


@pytest.mark.parametrize("change", ["basis", "payload", "step", "profile", "polynomial", "receipt", "operator"])
def test_model_and_output_tampering_reject(archive_case, change):
    archive, _ = archive_case
    state = declaration(archive)
    output = infer_model(state, archive)
    with pytest.raises((ValueError, TypeError)):
        if change == "basis": replace(state, output_axes=(CoordinateSpace("other", state.output_axes[0].keys),))
        elif change == "payload": replace(state, payload={"weights": (1, 2)})
        elif change == "step": replace(state, step=1)
        elif change == "polynomial":
            replace(output, execution_receipt=dict(output.execution_receipt, minimal_polynomial=(Fraction(0), Fraction(1))))
        else:
            config = thaw_tree(state.configuration)
            if change == "profile": config["profile"] = "unclaimed"
            elif change == "receipt": config["engine_receipt"] = {}
            elif change == "operator": config["operator_key"] = 70
            changed = replace(state, configuration=config)
            infer_model(changed, archive)
    with pytest.raises(ValueError): output.values[0, 0] = Fraction(0)
    with pytest.raises(TypeError): output.execution_receipt["field_degree"] = 100


def test_independent_reader_admission_is_intersected_with_model_policy(archive_case):
    archive, _ = archive_case
    state = declaration(archive)
    record = model_record(archive, state)
    with pytest.raises(ValueError, match="byte"):
        read_model_record(record, algebraic_limits={"max_file_bytes": len(archive.archive) - 1})
    with pytest.raises((ValueError, RuntimeError)):
        read_model_record(record, algebraic_limits={"field.max_integer_bits": 64})
    saved, owner = read_model_record(record, algebraic_limits={"max_file_bytes": len(archive.archive)})
    assert owner.limits["max_file_bytes"] == len(archive.archive)
    result = infer_model(saved, owner)
    assert np.array_equal(result.values, infer_model(state, archive).values)
    assert result.execution_receipt["limits"]["max_file_bytes"] == len(archive.archive)
    assert result.execution_receipt["declared_limits"] == archive.limits
    declared = dict(archive.limits, max_file_bytes=len(archive.archive))
    strict = NativeAlgebraicArchive(archive.archive, limits=declared)
    strict_state = declaration(strict)
    _, owner = read_model_record(model_record(strict, strict_state), algebraic_limits={"max_file_bytes": 1 << 30})
    assert owner.limits["max_file_bytes"] == len(archive.archive)


def test_bounded_restore_cache_preserves_actual_record_owners(archive_case):
    archive, _ = archive_case
    record = algebraic_record(archive)
    cache = _AlgebraicRestoreCache()
    first = read_algebraic_record(record, _archive_cache=cache)
    assert read_algebraic_record(record, _archive_cache=cache) is first
    assert read_algebraic_record(deepcopy(record), _archive_cache=cache) is not first
    assert read_algebraic_record(record, limits={"max_file_bytes": len(archive.archive)}, _archive_cache=cache) is not first
    with ThreadPoolExecutor(max_workers=4) as pool:
        owners = tuple(pool.map(lambda _: read_algebraic_record(record, _archive_cache=cache), range(16)))
    assert all(owner is first for owner in owners)
    attachment = record.get_metadata(1, 0, "algebraic_archive")
    attachment["archive"] = attachment["archive"].copy()
    assert read_algebraic_record(record, _archive_cache=cache) is not first
    for _ in range(9): read_algebraic_record(algebraic_record(archive), _archive_cache=cache)
    assert len(cache._entries) == 8 and cache._bytes <= 64 << 20
    assert read_algebraic_record(record, _archive_cache=cache) is not first
    record.get_metadata(1, 0, "algebraic_archive")["archive"][0] ^= 1
    with pytest.raises(ValueError, match="digest"):
        read_algebraic_record(record, _archive_cache=cache)


@pytest.mark.parametrize("change", ["input_rank", "output_rank", "output_degree", "limit_fields", "limit_boolean",
                                    "field_degree", "receipt_rank", "receipt_degree", "receipt_polynomial", "receipt_shape"])
def test_portable_type_guards_reject_before_coordinate_allocation(archive_case, change):
    archive, _ = archive_case
    state = declaration(archive)
    output = infer_model(state, archive)
    with pytest.raises((ValueError, TypeError)):
        if change.startswith("receipt_"):
            receipt = thaw_tree(output.execution_receipt)
            if change == "receipt_rank": receipt["output_carrier"]["rank"] = 1 << 63
            elif change == "receipt_degree": receipt["field_degree"] = True
            elif change == "receipt_polynomial": receipt["minimal_polynomial"] = tuple(float(q) for q in archive.manifest["minimal_polynomial"])
            elif change == "receipt_shape": receipt["native_shape"] = (2, True)
            replace(output, execution_receipt=receipt)
        else:
            config = thaw_tree(state.configuration)
            if change == "input_rank": config["input_carrier"]["rank"] = 1 << 63
            elif change == "output_rank": config["output_carrier"]["rank"] = 1 << 63
            elif change == "output_degree": config["output_carrier"]["field_degree"] = True
            elif change == "limit_fields": config["limits"]["other"] = config["limits"].pop("max_file_bytes")
            elif change == "limit_boolean": config["limits"]["max_file_bytes"] = True
            elif change == "field_degree": config["limits"]["field.max_degree"] = archive.manifest["field_degree"] - 1
            replace(state, configuration=config)


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_rcdb_fresh_process_restores_exact_model_input_and_output(archive_case, tmp_path, backend):
    from rcdb import open_store
    archive, _ = archive_case
    state = declaration(archive, operation="solve", operator_key=50)
    output = infer_model(state, archive)
    record = model_record(archive, state)
    record.attach_metadata(1, 0, "prediction", output)
    record.attach_metadata(1, 0, "model_input", thaw_tree(cpp_algebraic_input(state, archive, archive.record(10))))
    url = f"sqlite:///{tmp_path}/store.sqlite" if backend == "sqlite" else f"{backend}://{tmp_path}/store"
    store = open_store(url, **({} if backend == "sqlite" else {"read_only": False})).configure_security(require_commits=True)
    try:
        store.commit_mutation("algebraic-model", record, expected_version=0)
        assert store.verify_commits("algebraic-model")
    finally: store.close()
    code = '''
import importlib.abc,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('native algebraic model imported Torch')
sys.meta_path.insert(0,NoTorch())
from rcdb import open_store
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
from rcql import Executor,parse
store=open_store(sys.argv[1]).configure_security(require_commits=True)
try:
 assert store.verify_commits('algebraic-model')
 record=store.read_record('algebraic-model',version=1).value
 state,owner=read_model_record(record)
 inputs=record.get_metadata(1,0,'model_input')
 result=infer_model(state,owner,inputs)
 expected=record.get_metadata(1,0,'prediction')
 assert result.coefficient_digest==expected.coefficient_digest
 query=Executor(sources={'record':record},params={'input':inputs}).execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE(),$input)'))
 assert query.values[0].coefficient_digest==expected.coefficient_digest
 assert query.exactness[0].value=='algebraic'
 print(json.dumps({'model':state.coefficient_digest,'output':result.coefficient_digest,'archive':owner.manifest['archive_digest']}))
finally:store.close()
'''
    returned = subprocess.run([sys.executable, "-I", "-c", code, url], cwd=tmp_path, check=True, capture_output=True, text=True)
    parsed = json.loads(returned.stdout)
    assert parsed == {"model": state.coefficient_digest, "output": output.coefficient_digest,
                      "archive": archive.manifest["archive_digest"]}


def test_rcql_declares_algebraic_arithmetic_and_coordinate_observation(archive_case):
    from rcql.types import Domain, Exactness

    from rcql import Executor, call, parse, query, source
    archive, _ = archive_case
    state = declaration(archive)
    output = infer_model(state, archive)
    record = model_record(archive, state)
    result = Executor(sources={"record": record}, params={"output": output}).execute(
        parse('FROM $record RETURN MODEL_INFER(MODEL_STATE()),MODEL_VALUES($output)'))
    assert result.exactness == (Exactness.ALGEBRAIC, Exactness.RATIONAL)
    assert result.values[0].coefficient_digest == output.coefficient_digest
    np.testing.assert_array_equal(result.values[1], output.values)
    explained = Executor(sources={"record": record}).execute(
        query(source("record"), call("MODEL_INFER", state), call("MODEL_VALUES", output), explain=True)).values[0]
    contracts = [entry["result"] for entry in explained["returns"]]
    assert contracts[0]["exactness"] == "algebraic" and contracts[0]["domain"] == Domain.ALGEBRAIC.value
    assert contracts[1]["exactness"] == "rational" and contracts[1]["domain"] == Domain.RATIONAL.value


def test_rcql_initialization_and_record_share_the_same_archive(archive_case):
    from rcql import Executor, parse
    archive, _ = archive_case
    record = algebraic_record(archive)
    result = Executor(sources={"record": record}, params={"config": {"operator_key": 30, "input_key": 10}}).execute(
        parse('FROM $record LET m=MODEL_INIT("cpp_algebraic",$config) RETURN MODEL_INFER(m),MODEL_RECORD(m),MODEL_INFO(m)'))
    output, saved, info = result.values
    state, owner = read_model_record(saved)
    assert owner.archive == archive.archive
    assert output.coefficient_digest == infer_model(state, owner).coefficient_digest
    assert info["arithmetic"] == "algebraic" and not info["resumable"]


@pytest.mark.parametrize("route", ["MODEL_FIELD", "MODEL_INPUT", "MODEL_TRAIN", "MODEL_TRANSPORT"])
def test_rcql_unsupported_conversions_and_training_reject(archive_case, route):
    from rcql import Executor, call, query, source
    archive, _ = archive_case
    state = declaration(archive)
    output = infer_model(state, archive)
    args = (output,) if route in {"MODEL_FIELD", "MODEL_INPUT"} else (
        (state, ModelBatch(state.source, state.space, np.zeros(2, dtype=object), np.ones(2, dtype=bool)))
        if route == "MODEL_TRAIN" else (state, None, {}))
    for explain in (False, True):
        with pytest.raises((TypeError, NotImplementedError)):
            Executor(sources={"record": model_record(archive, state)}).execute(
                query(source("record"), call(route, *args), explain=explain))


def test_rcql_query_reuses_owners_without_skipping_attachment_validation(archive_case, monkeypatch):
    import rexgraph.io.algebraic_record as records
    from rcql import Executor, parse
    archive, _ = archive_case
    state = declaration(archive)
    record = model_record(archive, state)
    owners = []
    original = records.read_algebraic_record
    def counted(*args, **kwargs):
        owner = original(*args, **kwargs)
        owners.append(owner)
        return owner
    monkeypatch.setattr(records, "read_algebraic_record", counted)
    executor = Executor(sources={"record": record})
    text = 'FROM $record RETURN MODEL_INFO(MODEL_STATE()),MODEL_INFER(MODEL_STATE()),MODEL_RECORD(MODEL_STATE())'
    executor.execute(parse(text))
    assert len(owners) >= 3 and all(owner is owners[0] for owner in owners)
    first = owners[0]; owners.clear()
    executor.execute(parse(text))
    assert all(owner is owners[0] for owner in owners) and owners[0] is not first
    record.get_metadata(1, 0, "model_source").get_metadata(1, 0, "algebraic_archive")["archive"][0] ^= 1
    with pytest.raises(ValueError): executor.execute(parse(text))


def test_agent_and_system_use_exact_model_contract(archive_case):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import SourcePolicy, parse
    archive, _ = archive_case
    state = declaration(archive)
    expected = infer_model(state, archive)
    hive = Hive("algebraic-model-fixture")
    hive.add_worker("exact-model", lambda inputs: infer_model(state, archive, inputs), capability="predict")
    output = hive.invoke("exact-model", cpp_algebraic_input(state, archive, archive.record(10)), record=False)
    assert output.coefficient_digest == expected.coefficient_digest
    runtime = RCQLRuntime()
    runtime.register("record", model_record(archive, state), policy=SourcePolicy.allow("read"))
    assert runtime.execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE())')).values[0].coefficient_digest == expected.coefficient_digest
    view = json_value(output, max_values=1)
    assert view["kind"] == "ModelOutput" and view["arithmetic"] == "algebraic"
    assert view["values"]["shape"] == list(output.values.shape)
    assert view["execution_receipt"]["coefficient_basis"] == "power"
    descriptor = json_value(state, max_values=0)
    assert descriptor["trainable"] is False and descriptor["field_degree"] == archive.manifest["field_degree"]
    assert json.dumps(view, allow_nan=False)


def test_read_policy_is_required_for_archive_model_queries(archive_case):
    from rcql import BoundSource, Executor, SourcePolicy, parse
    archive, _ = archive_case
    record = model_record(archive, declaration(archive))
    executor = Executor(sources={"record": BoundSource(record, SourcePolicy.allow("train"))})
    with pytest.raises(PermissionError):
        executor.execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE())'))


def test_detached_algebraic_models_decode_without_optional_runtime(archive_case, tmp_path):
    from rexgraph.io.bundle import save_rcbd
    archive, _ = archive_case
    state = declaration(archive)
    record = model_record(archive, state)
    record.attach_metadata(1, 0, "prediction", infer_model(state, archive))
    path = tmp_path / "portable.rcbd"
    save_rcbd(path, record)
    code = '''
import importlib.abc,sys
class NoBridge(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='rexgraph_cpp' or fullname.startswith('rexgraph_cpp.'):
   raise ModuleNotFoundError('optional bindings unavailable',name='rexgraph_cpp')
sys.meta_path.insert(0,NoBridge())
from rexgraph.io.bundle import load_rcbd
from rexgraph.source_binding import CppAlgebraicSourceReference
record=load_rcbd(sys.argv[1])
state=record.get_metadata(1,0,'model')
output=record.get_metadata(1,0,'prediction')
assert state.arithmetic==output.arithmetic=='algebraic'
assert isinstance(state.source,CppAlgebraicSourceReference)
assert state.source.source is None
assert not any(name=='rexgraph_cpp' or name.startswith('rexgraph_cpp.') for name in sys.modules)
state.check_state();output.check_state()
'''
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, check=True, capture_output=True, text=True)
