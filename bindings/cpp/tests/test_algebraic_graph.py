"""Owning archive graphs, complete comparisons and exact model record round trips."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from rexgraph_cpp import NativeAlgebraicArchive, NativeAlgebraicArchiveGraph

from rexgraph.cpp_algebraic import cpp_algebraic_input, cpp_algebraic_model
from rexgraph.cpp_algebraic_graph import cpp_algebraic_compare
from rexgraph.io.algebraic_record import (
    _AlgebraicRestoreCache,
    algebraic_graph_record,
    algebraic_record,
    read_algebraic_graph_record,
    read_algebraic_record,
)
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.model_codec import pack_model, unpack_model
from rexgraph.model_runtime import infer_model, train_model, transport_model
from rexgraph.model_state import model_input, thaw_tree
from rexgraph.source_binding import CppAlgebraicSourceReference, source_from_record
from rexgraph.state import from_state, to_state


def exact_rows(rows):
    def q(text):
        n, _, d = text.partition("/")
        return Fraction(int(n, 16), int(d, 16) if d else 1)
    return tuple(tuple(q(v) for v in row) for row in rows)


@pytest.fixture(scope="session")
def graph_data():
    path = Path(os.environ["REXGRAPH_CPP_GRAPH_FIXTURE"])
    oracle = json.loads(subprocess.check_output([os.environ["REXGRAPH_CPP_GRAPH_ORACLE"], str(path)], text=True))
    return path.read_bytes(), oracle


@pytest.fixture
def graph(graph_data):
    return NativeAlgebraicArchiveGraph(graph_data[0])


def model(archive, operation="apply"):
    return cpp_algebraic_model(archive, operator_key=20, input_key=10, operation=operation)


@pytest.mark.parametrize("key", range(1, 7))
@pytest.mark.parametrize("operation", ["apply", "solve", "quadrance", "as_covector"])
def test_archive_view_actions_and_models_retain_graph_identity(graph, graph_data, key, operation):
    archive = graph.archive(key)
    assert graph.archive(key) is archive
    assert archive.graph_owner is graph and archive.graph_key == key
    assert archive.record(10).coefficients == exact_rows(graph_data[1]["archives"][str(key)])
    state = model(archive, operation)
    result = infer_model(state, archive)
    native = getattr(archive.record(20), operation)(archive.record(10))
    assert tuple(map(tuple, result.values)) == native.coefficients
    assert state.configuration["profile"] == result.method == "algebraic-graph-archive-action-v1"
    assert result.source.native.graph_digest == graph.manifest["graph_digest"]
    assert result.execution_receipt["archive_key"] == key
    assert result.execution_receipt["graph_limits"] == graph.limits
    for value in (state, result):
        for native_format in (False, True):
            restored = unpack_model(pack_model(value, native=native_format))
            assert restored.coefficient_digest == value.coefficient_digest
            assert restored.source.source is None
            assert restored.source.native.archive_key == key
    ref = CppAlgebraicSourceReference(archive)
    assert ref.as_record()["kind"] == "cpp_algebraic_graph_source"
    assert source_from_record(ref.as_record(), allow_cpp=True).matches(ref)


def test_declared_sharing_and_independent_equal_presentations(graph):
    a, alias, b, independent = (graph.archive(k) for k in (1, 4, 2, 3))
    assert a.archive == alias.archive
    assert a.record(20).apply(alias.record(10)).coefficients == a.record(10).coefficients
    assert a.manifest["bundle_key"] == alias.manifest["bundle_key"]
    assert b.manifest["field_key"] == independent.manifest["field_key"]
    assert b.manifest["bundle_key"] != independent.manifest["bundle_key"]
    with pytest.raises(ValueError, match="carrier"):
        b.record(20).apply(independent.record(10))
    assert not CppAlgebraicSourceReference(a).matches(CppAlgebraicSourceReference(alias))
    assert model(a).space != model(alias).space
    with pytest.raises(ValueError, match="owner"):
        infer_model(model(a), a, alias.record(10))
    with pytest.raises(ValueError, match="carrier"):
        cpp_algebraic_input(model(a), a, infer_model(model(alias), alias))


@pytest.mark.parametrize("edge,source,target", [(1, 1, 2), (2, 2, 3), (3, 1, 5), (6, 5, 1), (7, 6, 2)])
def test_complete_carrier_basis_matches_canonical_oracle(graph, graph_data, edge, source, target):
    archive = graph.archive(source)
    info = graph.comparison("carrier", edge, max_values=10000)
    rank, degree = info["source_carrier"]["rank"], archive.manifest["field_degree"]
    oracle = graph_data[1]["carriers"][str(edge)]
    for j in range(rank):
        for k in range(degree):
            rows = [[0] * degree for _ in range(rank)]
            rows[j][k] = 1
            value = archive.vector(10, rows)
            result = graph.compare("carrier", edge, value, target_archive=target)
            assert result.archive_owner is graph.archive(target)
            assert result.coefficients == exact_rows(oracle[j * degree + k])
            action = np.array(info["action"], dtype=object).reshape(info["action_shape"])
            assert tuple(action[:, j * degree + k]) == tuple(q for row in result.coefficients for q in row)
    assert info["coefficient_count"] == info["source_carrier"]["rank"] * info["target_carrier"]["rank"]
    assert not info["coefficients_truncated"] and not info["action_truncated"]


@pytest.mark.parametrize("edge,source,target", [(1, 1, 2), (2, 2, 3), (3, 1, 5), (6, 5, 1), (7, 6, 2)])
def test_model_comparison_and_destination_inference(graph, edge, source, target):
    a, b = graph.archive(source), graph.archive(target)
    origin = infer_model(model(a), a)
    transferred = cpp_algebraic_compare(origin, a, b, kind="carrier", key=edge, carrier_record=10)
    native = graph.compare("carrier", edge, a.record(10), target_archive=target)
    assert tuple(map(tuple, transferred.values)) == native.coefficients
    assert transferred.method == "algebraic-graph-comparison-v1"
    assert transferred.execution_receipt["origin_output"] == origin.coefficient_digest
    assert unpack_model(pack_model(transferred)).coefficient_digest == transferred.coefficient_digest
    destination = model(b)
    inputs = cpp_algebraic_input(destination, b, transferred)
    assert tuple(map(tuple, infer_model(destination, b, inputs).values)) == native.coefficients


@pytest.mark.parametrize("edge,source,target,scalar", [(1, 1, 2, 1), (2, 1, 5, 3), (3, 6, 2, 5)])
def test_certified_measurement_and_scalar_comparisons(graph, graph_data, edge, source, target, scalar):
    a, b = graph.archive(source), graph.archive(target)
    vector = a.record(10)
    measured = graph.compare("measurement", edge, vector, target_archive=target, operation="quadrance")
    expected = exact_rows(graph_data[1]["measurements"][str(edge)])
    assert expected[0] == expected[1] == measured.coefficients[0]
    q = a.record(20).quadrance(vector)
    assert graph.compare("scalar", scalar, q, target_archive=target).coefficients == measured.coefficients
    origin = infer_model(model(a), a)
    transferred = cpp_algebraic_compare(origin, a, b, kind="measurement", key=edge, operation="quadrance", carrier_record=10)
    assert tuple(map(tuple, transferred.values)) == measured.coefficients
    scalar_origin = infer_model(model(a, "quadrance"), a)
    scalar_output = cpp_algebraic_compare(scalar_origin, a, b, kind="scalar", key=scalar)
    assert np.array_equal(scalar_output.values, transferred.values)
    assert transferred.grade is None


@pytest.mark.parametrize("kind,key,source,target,operation", [
    ("carrier", 1, 1, 3, "apply"), ("carrier", 1, 5, 2, "apply"),
    ("carrier", 2, 2, 2, "apply"), ("measurement", 1, 1, 3, "apply"),
    ("measurement", 1, 1, 3, "quadrance"), ("scalar", 1, 5, 2, "apply"),
    ("scalar", 1, 1, 5, "apply"), ("reconstruction", 1, 1, 4, "accession")])
def test_comparisons_reject_wrong_actual_field_carrier_or_destination(graph, kind, key, source, target, operation):
    a = graph.archive(source)
    value = a.scalar([1] + [0] * (a.manifest["field_degree"] - 1)) if kind == "scalar" else a.record(10)
    with pytest.raises((ValueError, IndexError)):
        graph.compare(kind, key, value, target_archive=target, operation=operation)


@pytest.mark.parametrize("kind,operation", [("carrier", "inverse"), ("carrier", "solve"), ("scalar", "inverse"),
                                           ("domain", "apply"), ("measurement", "solve"), ("reconstruction", "apply")])
def test_comparisons_never_infer_undeclared_operations(graph, kind, operation):
    with pytest.raises(ValueError):
        graph.compare(kind, 1, graph.archive(1).record(10), target_archive=2, operation=operation)


def test_reconstruction_requires_declared_pair_and_retains_target_owner(graph):
    x = graph.archive(1).record(10)
    y = graph.compare("reconstruction", 2, x, target_archive=5, operation="accession")
    returned = graph.compare("reconstruction", 2, y, target_archive=1, operation="reconstruct")
    assert returned.coefficients == x.coefficients
    projected = graph.compare("reconstruction", 2, y, target_archive=5, operation="project")
    assert projected.coefficients == y.coefficients
    info = graph.comparison("reconstruction", 2, 0)
    assert (info["accession_key"], info["reconstruction_key"]) == (3, 6)


def test_separate_graph_loads_and_standalone_archive_never_share_live_owners(graph):
    fresh = NativeAlgebraicArchiveGraph(graph.image)
    standalone = NativeAlgebraicArchive(graph.archive(1).archive)
    for foreign in (fresh.archive(1), standalone):
        with pytest.raises(ValueError, match="owner"):
            graph.compare("carrier", 1, foreign.record(10), target_archive=2)
        with pytest.raises(ValueError, match="carrier"):
            graph.archive(1).record(20).apply(foreign.record(10))
    with pytest.raises(ValueError, match="graph owner"):
        cpp_algebraic_compare(infer_model(model(graph.archive(1)), graph.archive(1)), graph.archive(1),
                              fresh.archive(2), kind="carrier", key=1, carrier_record=10)


def test_escaped_values_pin_graph_and_parallel_views_share_declared_owner(graph_data):
    def escaped():
        owner = NativeAlgebraicArchiveGraph(graph_data[0])
        with ThreadPoolExecutor(max_workers=4) as pool:
            views = list(pool.map(lambda _: owner.archive(2), range(32)))
        assert all(value is views[0] for value in views)
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(lambda _: owner.compare("carrier", 1, owner.archive(1).record(10), target_archive=2), range(32)))
    values = escaped()
    for value in values:
        owner = value.archive_owner
        assert owner.graph_owner.archive(2) is owner
        assert owner.record(20).apply(value).coefficients == value.coefficients


@pytest.mark.parametrize("field,value", [("max_file_bytes", 32), ("max_owner_records", 0), ("max_edges", 0),
    ("max_rational_coordinates", 0), ("measurement.max_coefficient_products", 0), ("archive.max_records", 0),
    ("archive.field.max_degree", 1), ("archive.field.max_integer_bits", 1), ("archive.modules.max_rank", 1),
    ("archive.fibers.max_occurrences", 0), ("unknown", 1), ("max_edges", True), ("max_edges", -1)])
def test_graph_admission_is_explicit_and_bounded(graph_data, field, value):
    with pytest.raises((ValueError, TypeError, OverflowError, RuntimeError)):
        NativeAlgebraicArchiveGraph(graph_data[0], limits={field: value})


@pytest.mark.parametrize("offset", [0, 8, 10, 12, 16, 24, 40, 500, -1])
def test_corrupt_graph_bytes_reject(graph_data, offset):
    data = bytearray(graph_data[0]); data[offset] ^= 1
    with pytest.raises((ValueError, IndexError)):
        NativeAlgebraicArchiveGraph(bytes(data))


@pytest.mark.parametrize("size", [0, 1, 8, 24, 31, 32, 60, 500, 15271])
def test_truncated_graph_bytes_reject(graph_data, size):
    with pytest.raises(ValueError):
        NativeAlgebraicArchiveGraph(graph_data[0][:size])


@pytest.mark.parametrize("kind,key", [("carrier", True), ("scalar", -1), ("carrier", 0), ("measurement", 99), ("invalid", 1)])
def test_invalid_named_keys_reject(graph, kind, key):
    with pytest.raises((ValueError, TypeError, IndexError)):
        graph.comparison(kind, key)


def test_native_graph_is_immutable_and_previews_are_detached_and_bounded(graph):
    with pytest.raises(RuntimeError, match="immutable"):
        graph.__init__(graph.image)
    with pytest.raises(RuntimeError, match="immutable"):
        graph.archive(1).__init__(graph.archive(1).archive)
    preview = graph.preview(1)
    assert preview["tables"]["fields"] == {"keys": (1,), "count": 4, "truncated": True}
    manifest = graph.manifest; manifest["tables"]["fields"] = ()
    assert graph.manifest["tables"]["fields"] == (1, 2, 3, 4)
    for kind, key in [("scalar", 1), ("carrier", 1), ("domain", 1), ("measurement", 1), ("reconstruction", 1)]:
        value = graph.comparison(kind, key, 0)
        assert value["coefficients"] == value["action"] == ()
    blank = NativeAlgebraicArchiveGraph.__new__(NativeAlgebraicArchiveGraph)
    with pytest.raises(RuntimeError, match="initialized"):
        blank.archive(1)


@pytest.mark.parametrize("value", [True, 0.5, 1j, "1/2", None])
def test_exact_scalar_construction_rejects_implicit_conversion(graph, value):
    with pytest.raises((ValueError, TypeError)):
        graph.archive(1).scalar([value, 0])


@pytest.mark.parametrize("key", range(1, 7))
@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_graph_model_and_comparison_roundtrip_through_user_formats(graph, tmp_path, key, suffix):
    archive = graph.archive(key)
    state = model(archive)
    expected = infer_model(state, archive)
    record = model_record(archive, state)
    record.attach_metadata(1, 0, "prediction", expected)
    path = tmp_path / ("model" + suffix)
    save_model(path, record)
    saved = load_model(path)
    restored, source = read_model_record(saved)
    assert source.graph_owner.image == graph.image and source.graph_key == key
    assert source.graph_owner is not graph
    assert infer_model(restored, source).coefficient_digest == expected.coefficient_digest
    assert saved.get_metadata(1, 0, "prediction").coefficient_digest == expected.coefficient_digest
    if key == 1:
        comparison = cpp_algebraic_compare(expected, archive, graph.archive(2), kind="carrier", key=1, carrier_record=10)
        record.attach_metadata(1, 0, "comparison", comparison)
        save_model(path, record)
        retained = load_model(path).get_metadata(1, 0, "comparison")
        assert retained.coefficient_digest == comparison.coefficient_digest


def test_whole_graph_records_and_selected_views_preserve_canonical_bytes(graph):
    record = algebraic_graph_record(graph)
    fresh = read_algebraic_graph_record(from_state(to_state(record)))
    assert fresh.image == graph.image and fresh.manifest == graph.manifest
    for key in range(1, 7):
        view = read_algebraic_record(from_state(to_state(algebraic_record(graph.archive(key)))))
        assert view.graph_owner.image == graph.image and view.graph_key == key


def test_query_cache_retains_actual_graph_sharing_and_checks_each_selected_view(graph):
    from rexgraph.graph import RexGraph
    record = RexGraph.from_hypergraph([0, 1, 2], [0, 1])
    first = algebraic_record(graph.archive(1)).get_metadata(1, 0, "algebraic_archive")
    second = algebraic_record(graph.archive(2)).get_metadata(1, 0, "algebraic_archive")
    second["archive"] = first["archive"]
    record.attach_metadata(1, 0, "algebraic_archive", first)
    record.attach_metadata(1, 1, "algebraic_archive", second)
    cache = _AlgebraicRestoreCache()
    a = read_algebraic_record(record, _archive_cache=cache)
    b = read_algebraic_record(record, 1, _archive_cache=cache)
    assert a.graph_owner is b.graph_owner
    assert read_algebraic_record(record, _archive_cache=cache) is a
    assert a.graph_owner.compare("carrier", 1, a.record(10), target_archive=2).archive_owner is b
    detached = from_state(to_state(record))
    assert read_algebraic_record(detached, _archive_cache=cache).graph_owner is not a.graph_owner
    wrong = deepcopy(second); wrong["source_digest"] = "0" * 64
    record.attach_metadata(1, 1, "algebraic_archive", wrong)
    with pytest.raises(ValueError, match="source differs"):
        read_algebraic_record(record, 1, _archive_cache=cache)
    narrow = {"max_file_bytes": len(graph.image) - 1}
    with pytest.raises(ValueError, match="byte admission"):
        read_algebraic_record(record, _archive_cache=cache, limits=narrow)


@pytest.mark.parametrize("change", ["kind", "graph", "key", "key_boolean", "polynomial", "codec"])
def test_source_declarations_reject_graph_scope_confusion(graph, change):
    ref = CppAlgebraicSourceReference(graph.archive(1))
    record = deepcopy(ref.as_record())
    if change == "kind": record["kind"] = "cpp_algebraic_source"
    elif change == "graph": record["native"]["graph_digest"] = "0" * 64
    elif change == "key": record["native"]["archive_key"] = 4
    elif change == "key_boolean": record["native"]["archive_key"] = True
    elif change == "polynomial": record["native"]["polynomial"] = (("1", "1"), ("1", "1"))
    elif change == "codec": record["native"]["codec"] = "rfcxa-v1"
    with pytest.raises((ValueError, TypeError)):
        source_from_record(record, allow_cpp=True).bind(graph.archive(1))


@pytest.mark.parametrize("change", ["profile", "graph", "archive_key", "graph_fields", "graph_boolean", "rank", "degree"])
def test_model_declarations_reject_graph_policy_and_dimension_tampering(graph, change):
    archive = graph.archive(1); state = model(archive)
    config = thaw_tree(state.configuration)
    declared_source = state.source
    if change == "profile": config["profile"] = "algebraic-archive-action-v1"
    elif change == "graph":
        source = replace(state.source.native, graph_digest="0" * 64)
        declared_source = CppAlgebraicSourceReference(None, native=source)
    elif change == "archive_key":
        source = replace(state.source.native, archive_key=4)
        declared_source = CppAlgebraicSourceReference(None, native=source)
    elif change == "graph_fields": config["graph_limits"].pop("max_edges")
    elif change == "graph_boolean": config["graph_limits"]["max_edges"] = True
    elif change == "rank": config["input_carrier"]["rank"] = 1 << 63
    elif change == "degree": config["input_carrier"]["field_degree"] = 4
    with pytest.raises((ValueError, TypeError)):
        infer_model(replace(state, configuration=config, source=declared_source), archive)


@pytest.mark.parametrize("change", ["graph", "source", "key", "kind", "operation", "axis", "policy", "rank", "action"])
def test_comparison_receipts_reject_inconsistent_contracts(graph, change):
    a, b = graph.archive(1), graph.archive(2)
    result = cpp_algebraic_compare(infer_model(model(a), a), a, b, kind="carrier", key=1, carrier_record=10)
    receipt = thaw_tree(result.execution_receipt)
    if change == "graph": receipt["graph_digest"] = "0" * 64
    elif change == "source": receipt["source"]["native"]["graph_digest"] = "0" * 64
    elif change == "key": receipt["comparison_key"] = True
    elif change == "kind": receipt["kind"] = "scalar"
    elif change == "operation": receipt["operation"] = "inverse"
    elif change == "axis": receipt["coefficient_basis"] = "embedded"
    elif change == "policy": receipt["limits"]["max_edges"] = False
    elif change == "rank": receipt["output_carrier"]["rank"] = 1 << 63
    elif change == "action": receipt["comparison"]["action_shape"] = (0, 0)
    with pytest.raises((ValueError, TypeError)):
        replace(result, execution_receipt=receipt)


@pytest.mark.parametrize("route", ["tensor", "input", "train", "transport"])
def test_graph_profiles_keep_unsupported_learning_and_embeddings_explicit(graph, route):
    archive = graph.archive(1); state = model(archive); output = infer_model(state, archive)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        if route == "tensor": output.tensor()
        elif route == "input": model_input(output)
        elif route == "train": train_model(state, archive, {})
        else: transport_model(state, graph.archive(2))


@pytest.mark.parametrize("backend", ["local", "file", "rex", "sqlite"])
@pytest.mark.parametrize("source,target,kind,edge", [(1, 2, "carrier", 1), (6, 2, "measurement", 3)])
def test_rcdb_fresh_process_replays_graph_model_and_named_transfer(graph, tmp_path, backend, source, target, kind, edge):
    from rcdb import MemoryStore, copy_record, open_store
    a, b = graph.archive(source), graph.archive(target)
    state = model(a)
    output = infer_model(state, a)
    comparison = cpp_algebraic_compare(output, a, b, kind=kind, key=edge, carrier_record=10)
    record = model_record(a, state)
    record.attach_metadata(1, 0, "prediction", output)
    record.attach_metadata(1, 0, "comparison", comparison)
    url = f"sqlite:///{tmp_path}/models.sqlite" if backend == "sqlite" else f"{backend}://{tmp_path}/models"
    store = open_store(url, **({"read_only": False} if backend in {"file", "rex"} else {})).configure_security(require_commits=True)
    try:
        store.commit_mutation("graph-model", record, expected_version=0)
        assert store.verify_commits("graph-model")
        destination = MemoryStore()
        try:
            receipt = copy_record(store, destination, store.get_record("graph-model"), return_receipt=True)
            assert receipt.source_digest == receipt.destination_digest
            copied_state, copied_source = read_model_record(destination.get_version("graph-model", 1))
            assert infer_model(copied_state, copied_source).coefficient_digest == output.coefficient_digest
        finally:
            destination.close()
    finally:
        store.close()
    code = '''
import importlib.abc,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('graph model imported Torch')
sys.meta_path.insert(0,NoTorch())
from rcdb import open_store
from rexgraph.cpp_algebraic_graph import cpp_algebraic_compare
from rexgraph.io.model_record import read_model_record
from rexgraph.model_runtime import infer_model
from rcql import Executor,parse
store=open_store(sys.argv[1]).configure_security(require_commits=True)
try:
 assert store.verify_commits('graph-model')
 record=store.read_record('graph-model',version=1).value
 state,owner=read_model_record(record)
 result=infer_model(state,owner)
 expected=record.get_metadata(1,0,'prediction')
 assert result.coefficient_digest==expected.coefficient_digest
 saved=record.get_metadata(1,0,'comparison')
 r=saved.execution_receipt
 target=owner.graph_owner.archive(saved.source.native.archive_key)
 replay=cpp_algebraic_compare(result,owner,target,kind=r['comparison_kind'],key=r['comparison_key'],carrier_record=10)
 assert replay.coefficient_digest==saved.coefficient_digest
 query=Executor(sources={'record':record}).execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE())'))
 assert query.values[0].coefficient_digest==expected.coefficient_digest
 assert query.exactness[0].value=='algebraic'
 print(json.dumps({'output':result.coefficient_digest,'comparison':replay.coefficient_digest}))
finally:store.close()
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, url], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {"output": output.coefficient_digest, "comparison": comparison.coefficient_digest}


def test_agent_and_system_execute_composed_graph_models_and_show_scoped_results(graph):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import SourcePolicy, parse
    a, b = graph.archive(1), graph.archive(2)
    first, second = model(a), model(b)
    def predict(inputs):
        output = infer_model(first, a, inputs)
        compared = cpp_algebraic_compare(output, a, b, kind="measurement", key=1, carrier_record=10)
        return infer_model(second, b, cpp_algebraic_input(second, b, compared))
    hive = Hive("graph-model-fixture")
    hive.add_worker("composed-model", predict, capability="predict")
    inputs = cpp_algebraic_input(first, a, a.record(10))
    expected = predict(inputs)
    assert hive.invoke("composed-model", inputs, record=False).coefficient_digest == expected.coefficient_digest
    runtime = RCQLRuntime()
    runtime.register("record", model_record(a, first), policy=SourcePolicy.allow("read"))
    queried = runtime.execute(parse('FROM $record RETURN MODEL_INFER(MODEL_STATE())')).values[0]
    assert queried.coefficient_digest == infer_model(first, a).coefficient_digest
    preview = json_value(graph, max_values=1)
    assert preview["kind"] == "NativeAlgebraicArchiveGraph"
    assert preview["tables"]["archives"]["count"] == 6
    assert json_value(first)["source"]["kind"] == "cpp_algebraic_graph_source"
    assert json_value(expected)["execution_receipt"]["graph_digest"] == graph.manifest["graph_digest"]
    assert json.dumps(preview, allow_nan=False)


def test_graph_model_and_comparison_decode_without_optional_engine(graph, tmp_path):
    from rexgraph.io.bundle import save_rcbd
    a = graph.archive(1)
    state = model(a)
    output = infer_model(state, a)
    comparison = cpp_algebraic_compare(output, a, graph.archive(2), kind="carrier", key=1, carrier_record=10)
    record = model_record(a, state)
    record.attach_metadata(1, 0, "prediction", output)
    record.attach_metadata(1, 0, "comparison", comparison)
    path = tmp_path / "offline.rcbd"
    save_rcbd(path, record)
    code = '''
import importlib.abc,sys
class NoBridge(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='rexgraph_cpp' or fullname.startswith('rexgraph_cpp.') or fullname=='torch' or fullname.startswith('torch.'):
   raise ModuleNotFoundError('optional runtime unavailable',name=fullname)
sys.meta_path.insert(0,NoBridge())
from rexgraph.io.bundle import load_rcbd
record=load_rcbd(sys.argv[1])
for name in ('model','prediction','comparison'):
 value=record.get_metadata(1,0,name)
 value.check_state()
 assert value.source.source is None
 assert value.source.native.graph_digest is not None
assert not any(name=='rexgraph_cpp' or name.startswith('rexgraph_cpp.') for name in sys.modules)
'''
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, check=True, capture_output=True, text=True)
