"""Native graph revisions, stable database pins and complete prefix records."""
import gc
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from rexgraph_cpp import (
    CancellationToken,
    ExecutionCancelled,
    ExecutionLimitError,
    NativeAlgebraicGraphDatabase,
    NativeAlgebraicGraphRevision,
    NativeAlgebraicSnapshot,
    NativeAlgebraicTensorWindow,
    NativeSource,
)

from rexgraph.cpp_algebraic_graph_history import (
    algebraic_graph_history_at,
    algebraic_graph_history_record,
)
from rexgraph.cpp_algebraic_history import algebraic_history_at
from rexgraph.cpp_algebraic_query import (
    algebraic_query_record,
    algebraic_window_record,
    prepare_algebraic_query,
    read_algebraic_observation_record,
    read_algebraic_snapshot_record,
)
from rexgraph.io.algebraic_record import _AlgebraicRestoreCache
from rexgraph.state import from_state, to_state

CHANNEL = (9611, 1, 2)
AVAILABLE = (9601, 42)


@pytest.fixture(scope="session")
def graph_history_files():
    return Path(os.environ["REXGRAPH_CPP_GRAPH_HISTORY_FIXTURES"])


@pytest.fixture
def source(graph_history_files):
    return NativeSource((graph_history_files / "42.rfcxs").read_bytes())


def revision(source, sequence=41, **kwargs):
    return NativeAlgebraicGraphRevision(source, CHANNEL, sequence=sequence, **kwargs)


def database(source, sequence=41, **kwargs):
    return NativeAlgebraicGraphDatabase(revision(source, sequence), availability=AVAILABLE, **kwargs)


def record(source, sequence=41, archive_key=1):
    return algebraic_graph_history_record(source, CHANNEL, sequence=sequence, archive_key=archive_key)


def observed(source):
    snap = database(source).archive(1, availability=AVAILABLE)
    query = prepare_algebraic_query(snap, {"kind": "vector"}, availability=AVAILABLE)
    return algebraic_query_record(query.execute(availability=AVAILABLE, cpu_workers=2))


@pytest.mark.parametrize("sequence", [40, 41, 42])
@pytest.mark.parametrize("key", range(1, 7))
def test_actual_endpoint_stable_database_pins_and_original_maps(source, graph_history_files, sequence, key):
    owner = revision(source, sequence)
    assert owner.image == (graph_history_files / f"{sequence}.rfcxh").read_bytes()
    assert owner.history_owner.identity["sequence"] == sequence
    assert owner.history_owner.archive == (graph_history_files / f"{sequence}.rfcxs").read_bytes()
    assert owner.manifest["sources"][0]["occurrences"] == ((), (0, 1, 2, 3))
    assert owner.manifest["sources"][1]["kind"] == "anchor_morphism"
    assert owner.manifest["sources"][2]["kind"] == "independent"
    db = NativeAlgebraicGraphDatabase(owner, availability=(9601, sequence))
    first = db.archive(key, availability=AVAILABLE)
    second = db.archive(key, availability=AVAILABLE)
    assert first.same_publication(second)
    assert first.graph_revision_owner is owner
    assert first.archive_owner is owner.graph_owner.archive(key)
    assert first.graph_origin == dict(archive_key=key, bundle_key=dict(owner.manifest["archives"])[key],
                                    anchor_bundle=1, namespace=9601, sequence=sequence)
    assert db.requirements["records"] == 19
    assert first.record(10).value.coefficients == owner.graph_owner.archive(key).record(10).coefficients
    plain = NativeAlgebraicSnapshot(first.archive_owner, namespace=9601, sequence=sequence)
    assert plain.graph_revision_owner is None and plain.graph_origin is None
    assert not plain.same_publication(first)


def test_comparisons_consume_selected_actual_owners_without_decoding(source):
    db = database(source)
    left, alias, right, independent = (db.archive(key, availability=AVAILABLE) for key in (1, 4, 2, 5))
    assert not left.same_publication(alias)
    assert left.record(20).value.apply(alias.record(10).value).coefficients == alias.record(10).value.coefficients
    graph = db.revision_owner.graph_owner
    mapped = graph.compare("carrier", 1, left.record(10).value, target_archive=2)
    assert mapped.coefficients == right.record(10).value.coefficients
    with pytest.raises(ValueError):
        left.record(20).value.apply(independent.record(10).value)
    other = database(source)
    assert not db.revision_owner.same_owner(other.revision_owner)
    with pytest.raises(ValueError):
        left.record(20).value.apply(other.archive(1, availability=AVAILABLE).record(10).value)
    retained = left.record(20)
    del db, left, alias, right, independent, graph
    gc.collect()
    assert retained.snapshot_owner.graph_origin["archive_key"] == 1
    assert retained.value.preview()["coefficient_count"] > 0


@pytest.mark.parametrize("sequence", [39, 43, True, 40.0, -1, 1 << 64])
def test_selected_cutoff_requires_exact_retained_revision(source, sequence):
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        revision(source, sequence)


@pytest.mark.parametrize("channel", [(0, 1, 2), (9611, 0, 2), (9611, 1, 0), (True, 1, 2),
    (9611.0, 1, 2), (9611, 1), (9611, 1, 2, 3), "9611", {},
    (9612, 1, 2), (9611, 2, 2), (9611, 1, 999), (9611, 1, 20), (9611, 1, 21)])
def test_wrong_and_opaque_channels_do_not_change_native_source(source, channel):
    identity = source.identity
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        NativeAlgebraicGraphRevision(source, channel, sequence=41)
    assert source.identity == identity


def test_selected_revision_excludes_damaged_future_and_checks_complete_anchor(graph_history_files):
    future = NativeSource((graph_history_files / "future-corrupt.rfcxs").read_bytes())
    selected = revision(future, 42)
    assert selected.history_owner.archive == (graph_history_files / "42.rfcxs").read_bytes()
    with pytest.raises(ValueError):
        revision(future, 43)
    with pytest.raises(ValueError, match="anchor"):
        revision(NativeSource((graph_history_files / "foreign.rfcxs").read_bytes()), 42)


@pytest.mark.parametrize("name", ["max_snapshot_bytes", "max_file_bytes", "max_owner_records", "max_edges",
    "max_rational_coordinates", "max_source_bindings", "max_archive_bindings", "max_source_coordinates", "max_history_revisions"])
def test_canonical_snapshot_limits_are_enforced(source, name):
    with pytest.raises((ValueError, TypeError, OverflowError, RuntimeError)):
        revision(source, limits={name: 0})


@pytest.mark.parametrize("availability", [(9601, 40), (9602, 41), (0, 41), (True, 41), (9601, 41.0), (9601,), {}])
def test_database_and_every_archive_selection_admit_availability(source, availability):
    db = database(source)
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        NativeAlgebraicGraphDatabase(db.revision_owner, availability=availability)
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        db.archive(1, availability=availability)


def test_database_policy_cancellation_and_missing_archive_apply_on_repeated_calls(source):
    owner = revision(source)
    with pytest.raises(ExecutionLimitError):
        NativeAlgebraicGraphDatabase(owner, availability=AVAILABLE, execution_limits={"records": 18})
    token = CancellationToken(); token.cancel()
    with pytest.raises(ExecutionCancelled):
        NativeAlgebraicGraphDatabase(owner, availability=AVAILABLE, cancellation=token)
    db = NativeAlgebraicGraphDatabase(owner, availability=AVAILABLE, execution_limits={"records": 19})
    assert db.archive(1, availability=AVAILABLE).graph_origin["archive_key"] == 1
    with pytest.raises(ExecutionLimitError):
        db.archive(1, availability=AVAILABLE, execution_limits={"records": 18})
    with pytest.raises(ExecutionCancelled):
        db.archive(1, availability=AVAILABLE, cancellation=token)
    with pytest.raises(IndexError):
        db.archive(999, availability=AVAILABLE)


@pytest.mark.parametrize("sequence", [40, 41, 42])
@pytest.mark.parametrize("key", range(1, 7))
def test_complete_records_restore_fresh_pins_and_prefix_selection(source, sequence, key):
    saved = from_state(to_state(record(source, sequence, key)))
    first, second = (read_algebraic_snapshot_record(saved) for _ in range(2))
    assert first.graph_origin == second.graph_origin
    assert not first.same_publication(second)
    assert first.graph_revision_owner.image == second.graph_revision_owner.image
    assert first.graph_revision_owner.history_owner.identity["sequence"] == sequence
    prior = read_algebraic_snapshot_record(algebraic_history_at(saved, 40))
    assert prior.sequence == 40 and prior.graph_origin["archive_key"] == key
    selected = read_algebraic_snapshot_record(algebraic_graph_history_at(saved, 40, archive_key=2))
    assert selected.graph_origin["archive_key"] == 2
    with pytest.raises((ValueError, IndexError)):
        algebraic_graph_history_at(saved, sequence+1)


def test_query_cache_reuses_actual_snapshot_and_revalidates_bytes_and_limits(source):
    saved = record(source)
    cache = _AlgebraicRestoreCache()
    first = read_algebraic_snapshot_record(saved, _archive_cache=cache)
    second = read_algebraic_snapshot_record(saved, _archive_cache=cache)
    assert first is second and first.same_publication(second)
    with ThreadPoolExecutor(max_workers=4) as pool:
        handles = list(pool.map(lambda _: read_algebraic_snapshot_record(saved, _archive_cache=cache), range(8)))
    assert all(h is first for h in handles)
    limits = {"max_history_revisions": 1}
    with pytest.raises((ValueError, RuntimeError)):
        read_algebraic_snapshot_record(saved, limits=limits, _archive_cache=cache)
    limits = {"max_snapshot_bytes": 32}
    with pytest.raises((ValueError, RuntimeError)):
        read_algebraic_snapshot_record(saved, limits=limits, _archive_cache=cache)
    saved.get_metadata(1, 0, "algebraic_graph_history")["archive"][0] ^= 1
    with pytest.raises(ValueError):
        read_algebraic_snapshot_record(saved, _archive_cache=cache)


@pytest.mark.parametrize("case", ["native_bytes", "snapshot_bytes", "graph_bytes", "snapshot_shape", "snapshot_dtype",
    "history_byte_limit", "snapshot_byte_limit", "snapshot_digest", "graph_digest", "channel", "namespace", "sequence",
    "anchor", "archive", "bundle", "map", "independent_map", "limits", "extra", "dual_history"])
def test_complete_attachment_and_publication_mutations_reject_before_restore(source, case):
    saved = observed(source)
    publication = deepcopy(saved.get_metadata(1, 0, "algebraic_snapshot"))
    attachment = deepcopy(saved.get_metadata(1, 0, "algebraic_graph_history"))
    ref = publication["publication"]["graph_history"]
    if case in {"native_bytes", "snapshot_bytes"}:
        attachment["archive" if case == "native_bytes" else "snapshot"][0] ^= 1
    elif case == "graph_bytes":
        raw = saved.get_metadata(1, 0, "algebraic_archive")["archive"]; raw[0] ^= 1
    elif case == "snapshot_shape": attachment["snapshot"] = attachment["snapshot"].reshape(1, -1)
    elif case == "snapshot_dtype": attachment["snapshot"] = attachment["snapshot"].astype(np.int16)
    elif case == "history_byte_limit": ref["max_bytes"] = 1
    elif case == "snapshot_byte_limit": ref["limits"]["max_snapshot_bytes"] = 1
    elif case in {"snapshot_digest", "graph_digest"}: ref["revision"][case] = "0" * 64
    elif case == "channel": ref["channel"] = (9611, 1, 999)
    elif case in {"namespace", "sequence"}: ref["revision"][case] += 1
    elif case == "anchor": ref["revision"]["anchor_bundle"] = 999
    elif case == "archive": ref["revision"]["archives"] = ((999, 1),)
    elif case == "bundle": ref["revision"]["sources"][0]["bundle"] = 999
    elif case == "map": ref["revision"]["sources"][0]["occurrences"] = ((), (1, 0, 2, 3))
    elif case == "independent_map": ref["revision"]["sources"][2]["cells"] = ((0,),)
    elif case == "limits": ref["limits"]["unknown"] = 1
    elif case == "extra": attachment["unknown"] = 1
    elif case == "dual_history": publication["publication"]["history"] = {}
    attachment["reference"] = deepcopy(ref)
    saved.attach_metadata(1, 0, "algebraic_snapshot", publication)
    saved.attach_metadata(1, 0, "algebraic_graph_history", attachment)
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        read_algebraic_snapshot_record(saved)


@pytest.mark.parametrize("suffix", ["rcbd", "safetensors"])
def test_storage_and_copy_restore_complete_graph_origin(source, tmp_path, suffix):
    from rexgraph.io import load, save
    saved = observed(source)
    path = tmp_path / f"graph.{suffix}"
    save(path, saved); restored = load(path)
    assert read_algebraic_observation_record(restored) == read_algebraic_observation_record(saved)
    snap = read_algebraic_snapshot_record(restored)
    assert snap.graph_origin["archive_key"] == 1
    assert snap.graph_revision_owner.image == revision(source).image


def test_query_window_rcql_agent_and_system_use_actual_revision(source):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse
    snap = database(source).archive(1, availability=AVAILABLE)
    result = observed(source)
    observed_value = read_algebraic_observation_record(result)
    assert observed_value["publication"]["graph_history"]["revision"]["sequence"] == 41
    window = NativeAlgebraicTensorWindow(snap.record(10), [0], [2], availability=AVAILABLE)
    w = algebraic_window_record(window, availability=AVAILABLE)
    assert read_algebraic_snapshot_record(w).graph_origin == snap.graph_origin
    saved = record(source, 42)
    text = "FROM $record RETURN ALGEBRAIC_AT(41)"
    selected = Executor(sources={"record": saved}).execute(parse(text)).values[0]
    query = "FROM $record RETURN ALGEBRAIC_QUERY($request,41), ALGEBRAIC_HISTORY_INFO()"
    value = Executor(sources={"record": selected}, params={"request": {"kind": "vector"}}).execute(parse(query))
    assert value.values[1]["revision"]["sequence"] == 41
    runtime = RCQLRuntime(); runtime.register("record", selected, policy=SourcePolicy.allow("read"))
    assert runtime.execute(parse(query), params={"request": {"kind": "vector"}}).values == value.values
    hive = Hive("graph-history")
    hive.add_worker("select", lambda seq: algebraic_history_at(saved, seq), capability="transform")
    assert read_algebraic_snapshot_record(hive.invoke("select", 40, record=False)).sequence == 40
    for obj in (snap, snap.graph_revision_owner, database(source)):
        json.dumps(json_value(obj, max_values=1), allow_nan=False)
    assert snap.preview()["native_history"] is True


def test_rcdb_courier_fresh_process_and_optional_free_observations(source, tmp_path):
    from rcdb import copy_record, open_store
    saved = observed(source)
    stores = [open_store(f"sqlite:///{tmp_path / (name + '.sqlite')}") for name in ("first", "second")]
    try:
        stores[0].put("graph", saved)
        receipt = copy_record(stores[0], stores[1], stores[0].get_record("graph"), return_receipt=True)
        assert receipt.source_digest == receipt.destination_digest
        result = stores[1].get_version("graph", 1)
        assert read_algebraic_snapshot_record(result).graph_origin["sequence"] == 41
    finally:
        for store in stores: store.close()
    from rexgraph.io import save
    path = tmp_path / "graph.rcbd"; save(path, saved)
    code = '''
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','rexgraph_cpp'}: raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from rexgraph.io import load
from rexgraph.cpp_algebraic_query import read_algebraic_observation_record
value=read_algebraic_observation_record(load(sys.argv[1]))
print(json.dumps({'seal':value['seal']}))
'''
    output = subprocess.check_output([sys.executable, "-I", "-c", code, str(path)], cwd=tmp_path, text=True)
    assert json.loads(output)["seal"] == read_algebraic_observation_record(saved)["seal"]
    code = '''
import sys
from rcdb import open_store
from rexgraph.cpp_algebraic_query import read_algebraic_snapshot_record
store=open_store(sys.argv[1])
try:
 snapshot=read_algebraic_snapshot_record(store.get_version('graph',1))
 assert snapshot.graph_origin['sequence']==41 and snapshot.graph_revision_owner.history_owner.identity['sequence']==41
finally: store.close()
'''
    subprocess.run([sys.executable, "-I", "-c", code, f"sqlite:///{tmp_path / 'second.sqlite'}"], cwd=tmp_path, check=True)


@pytest.mark.parametrize("klass", [NativeAlgebraicGraphRevision, NativeAlgebraicGraphDatabase])
def test_immutable_and_uninitialized_native_facades(source, klass):
    value = revision(source) if klass is NativeAlgebraicGraphRevision else database(source)
    with pytest.raises(RuntimeError):
        if klass is NativeAlgebraicGraphRevision: value.__init__(source, CHANNEL, sequence=41)
        else: value.__init__(revision(source), availability=AVAILABLE)
    blank = klass.__new__(klass)
    with pytest.raises(RuntimeError): blank.preview()
    if klass is NativeAlgebraicGraphRevision:
        with pytest.raises(RuntimeError): _ = blank.image
    else:
        with pytest.raises(RuntimeError): blank.archive(1, availability=AVAILABLE)


def test_core_dependency_guard_installs_new_bridge_module():
    module = importlib.import_module("rexgraph.cpp_algebraic_graph_history")
    assert module.algebraic_graph_history_record is algebraic_graph_history_record


def test_cached_restore_distinguishes_equal_bytes_on_different_archive_keys(source):
    saved, other = record(source, archive_key=1), record(source, archive_key=4)
    cache = _AlgebraicRestoreCache()
    first = read_algebraic_snapshot_record(saved, _archive_cache=cache)
    assert saved.get_metadata(1, 0, "algebraic_archive")["archive_digest"] == other.get_metadata(1, 0, "algebraic_archive")["archive_digest"]
    saved.attach_metadata(1, 0, "algebraic_snapshot", other.get_metadata(1, 0, "algebraic_snapshot"))
    saved.get_metadata(1, 0, "algebraic_archive")["archive_key"] = 4
    second = read_algebraic_snapshot_record(saved, _archive_cache=cache)
    assert second.graph_origin["archive_key"] == 4
    assert first.graph_origin["archive_key"] == 1
    assert not first.same_publication(second)


def test_cached_restore_rechecks_nonanchor_source_binding_declarations(source):
    saved = record(source)
    cache = _AlgebraicRestoreCache()
    read_algebraic_snapshot_record(saved, _archive_cache=cache)
    publication = deepcopy(saved.get_metadata(1, 0, "algebraic_snapshot"))
    reference = publication["publication"]["graph_history"]
    reference["revision"]["sources"][1]["cells"] = ((1, 0), (0,))
    saved.attach_metadata(1, 0, "algebraic_snapshot", publication)
    saved.get_metadata(1, 0, "algebraic_graph_history")["reference"] = deepcopy(reference)
    with pytest.raises(ValueError, match="complete declaration"):
        read_algebraic_snapshot_record(saved, _archive_cache=cache)


def test_revision_initialization_owns_guard_before_native_prefix_selection(source):
    from threading import Event
    entered, release = Event(), Event()

    class BlockingSource(NativeSource):
        def at(self, sequence):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("prefix selection timeout")
            return super().at(sequence)

    blocked = BlockingSource(source.archive)
    owner = NativeAlgebraicGraphRevision.__new__(NativeAlgebraicGraphRevision)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.__init__, blocked, CHANNEL, sequence=41)
        try:
            assert entered.wait(5)
            second = pool.submit(owner.__init__, source, CHANNEL, sequence=42)
            with pytest.raises(RuntimeError, match="immutable"):
                second.result(timeout=2)
        finally:
            release.set()
        first.result(timeout=5)
    assert owner.manifest["sequence"] == 41
    assert owner.history_owner.identity["sequence"] == 41
