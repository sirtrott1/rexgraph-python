"""Exact phase/fold words, logical cutoffs and complete application records."""

import gc
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from rexgraph_cpp import NativePhaseFoldArchive, NativeSource

from rexgraph.cpp_phase_fold import (
    phase_fold_history_at,
    phase_fold_history_record,
    phase_fold_observation,
    phase_fold_observation_record,
    phase_fold_record,
    phase_fold_reference,
    read_phase_fold_observation_record,
    read_phase_fold_record,
    validate_phase_fold_observation,
    validate_phase_fold_reference,
)
from rexgraph.state import from_state, to_state
from rexgraph.value_codec import pack_value

PROFILES = ["rational", "algebraic", "fold", "rational-walk", "algebraic-walk"]
CHANNEL = (801, 1, 100)


@pytest.fixture(scope="session")
def word_fixtures():
    base = os.environ.get("REXGRAPH_CPP_PHASE_FOLD_REFERENCE")
    history = os.environ.get("REXGRAPH_CPP_PHASE_FOLD_FIXTURES")
    if not base or not history:
        raise RuntimeError(
            "phase/fold tests require canonical references and the native fixture generator output"
        )
    return Path(base), Path(history)


@pytest.fixture(params=PROFILES)
def word_case(request, word_fixtures):
    name = request.param
    base, history = word_fixtures
    raw = (history / (name + "-72.rfcxs")).read_bytes()
    return (
        name,
        NativePhaseFoldArchive((base / (name + ".rfcxw")).read_bytes()),
        NativeSource(raw, max_bytes=len(raw)),
    )


def selected(case, sequence=70):
    return NativePhaseFoldArchive.from_history(case[2], CHANNEL, sequence=sequence)


def reseal(value):
    value["seal"] = sha256(pack_value({k: v for k, v in value.items() if k != "seal"})).hexdigest()
    return value


def test_full_exact_views_keep_source_and_distinct_carrier_owners(word_case):
    _, archive, _ = word_case
    copy = NativePhaseFoldArchive(archive.archive)
    assert copy.archive == archive.archive and copy.source_rfcx == archive.source_rfcx
    assert archive.same_carrier(archive) and not archive.same_carrier(copy)
    assert archive.history_owner is None and archive.history_channel is None
    assert len(archive.limits) == 35
    manifest = archive.manifest
    view = archive.observe(max_coordinates=manifest["observation_coordinate_count"])
    assert len(view["operations"]) == manifest["operation_count"]
    assert len(view["endpoint"]) == manifest["endpoint_coordinate_count"]
    assert all(type(q) is Fraction for q in view["endpoint"])
    manifest["source_sizes"] = ()
    assert archive.manifest["source_sizes"]
    with pytest.raises(ValueError):
        archive.observe(max_coordinates=archive.manifest["observation_coordinate_count"] - 1)
    with pytest.raises(RuntimeError):
        archive.__init__(archive.archive)
    uninitialised = NativePhaseFoldArchive.__new__(NativePhaseFoldArchive)
    with pytest.raises(RuntimeError):
        uninitialised.observe()


@pytest.mark.parametrize("engine_version", ["0.2.1", "0.3.0"])
def test_portable_engine_versions_remain_readable(word_case, engine_version):
    reference = deepcopy(phase_fold_reference(word_case[1]))
    reference["sdk"]["engine_version"] = engine_version
    assert validate_phase_fold_reference(reference) == reference
    reference["sdk"]["engine_version"] = "0.4.0"
    with pytest.raises(ValueError, match="native engine profile"):
        validate_phase_fold_reference(reference)


def test_prior_engine_record_requires_its_matching_runtime(word_case):
    record = phase_fold_record(word_case[1])
    reference = record.get_metadata(1, 0, "phase_fold_archive")["reference"]
    reference["sdk"]["engine_version"] = "0.2.1"
    validate_phase_fold_reference(reference)
    with pytest.raises(ValueError, match="declaration differs"):
        read_phase_fold_record(record)


@pytest.mark.parametrize("cut", ["zero", "full"])
def test_logical_prefix_bytes_match_canonical_oracles_and_drop_all_proposals(
    word_case, word_fixtures, cut
):
    name, base, _ = word_case
    archive = selected(word_case)
    count = 0 if cut == "zero" else base.manifest["operation_count"]
    prefix = archive.observed_prefix(count)
    assert prefix.archive == (word_fixtures[1] / f"{name}-prefix-{count}.rfcxw").read_bytes()
    assert prefix.same_carrier(archive) and prefix.manifest["proposal_count"] == 0
    assert prefix.history_owner is None and prefix.history_channel is None
    record = phase_fold_observation_record(archive, count)
    assert record.get_metadata(1, 0, "phase_fold_history") is None
    expected = phase_fold_observation(archive, count)
    assert read_phase_fold_observation_record(record) == expected
    assert expected["source"]["history"] is None
    assert len(expected["endpoint"]["operations"]) == count
    assert read_phase_fold_record(record).archive == prefix.archive
    del archive, prefix
    gc.collect()
    assert read_phase_fold_observation_record(record) == expected


def test_repeated_identities_inactive_steps_ordered_groups_and_large_lifts(word_case):
    name, archive, _ = word_case
    view = archive.observe()
    ops = view["operations"]
    if name.endswith("walk"):
        assert any(not op["forward"] for op in ops)
        assert len(view["positions"]) == len(ops) + 1
        assert view["positions"][0] == (-1844674407370955161600 if name == "rational-walk" else 0)
        assert all(type(n) is int for n in view["positions"])
        if name == "rational-walk":
            assert abs(view["positions"][0]) > 1 << 64
    else:
        assert len(ops) == 3 and len({op["identity"] for op in ops}) < 3
        assert any(not op["active"] for op in ops)
        if name != "fold":
            assert any(op["identity"] == 0 for op in ops)
        if name == "fold":
            assert [op["moving_panels"] for op in ops] == [(2, 1), (2,), (1, 2)]
            assert view["endpoint_shape"] == (3, 4, 4)


@pytest.mark.parametrize("sequence", [70, 71, 72])
def test_native_revisions_retain_complete_selected_prefix_and_word(
    word_case, word_fixtures, sequence
):
    name, base, source = word_case
    owner = selected(word_case, sequence)
    assert (
        owner.history_owner.archive == (word_fixtures[1] / f"{name}-{sequence}.rfcxs").read_bytes()
    )
    assert owner.history_owner.identity["sequence"] == sequence
    assert owner.history_owner.max_bytes == source.max_bytes
    assert owner.history_channel == CHANNEL
    if sequence == 70:
        assert owner.archive == base.archive
    elif sequence == 72:
        assert owner.manifest["operation_count"] == 0
    elif name.endswith("walk"):
        assert owner.manifest["operation_count"] == base.manifest["operation_count"] - 1
    else:
        assert owner.archive == (word_fixtures[1] / f"{name}-applied.rfcxw").read_bytes()
    if sequence < 72:
        with pytest.raises((IndexError, ValueError)):
            owner.history_owner.at(sequence + 1)
    assert phase_fold_reference(owner)["history"]["source"]["native"]["sequence"] == sequence


@pytest.mark.parametrize(
    "channel",
    [
        (0, 1, 100),
        (801, 0, 100),
        (801, 1, 0),
        (True, 1, 100),
        (801.0, 1, 100),
        (801, 1),
        "801",
        {},
        (802, 1, 100),
        (801, 2, 100),
        (801, 1, 999),
        (801, 1, 200),
        (801, 1, 201),
    ],
)
def test_native_channels_require_actual_declarations_and_exact_payload(word_case, channel):
    before = word_case[2].identity
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        NativePhaseFoldArchive.from_history(word_case[2], channel, sequence=70)
    assert word_case[2].identity == before


@pytest.mark.parametrize("cut", [-1, True, 1.0, 1 << 64, 69, 73])
def test_native_cutoffs_reject_unretained_or_inexact_revisions(word_case, cut):
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        selected(word_case, cut)


def test_foreign_source_channel_rejects_even_with_a_valid_native_prefix(word_case, word_fixtures):
    foreign = NativeSource((word_fixtures[1] / (word_case[0] + "-foreign.rfcxs")).read_bytes())
    with pytest.raises((ValueError, IndexError)):
        NativePhaseFoldArchive.from_history(foreign, CHANNEL, sequence=72)


def test_apply_replays_canonical_suffix_and_keeps_proposals_separate(word_fixtures):
    for name in ["rational", "algebraic", "fold"]:
        archive = NativePhaseFoldArchive((word_fixtures[0] / (name + ".rfcxw")).read_bytes())
        branch = archive.apply(7)
        assert branch.archive == (word_fixtures[1] / (name + "-applied.rfcxw")).read_bytes()
        assert archive.manifest["proposal_count"] == 1 and branch.manifest["proposal_count"] == 0
        assert branch.same_carrier(archive)
        replacement = archive.preview()["proposals"][0]
        clean = archive.observed_prefix(archive.manifest["operation_count"])
        proposed = clean.propose(
            replacement["key"],
            replacement["position"],
            replacement["phase"],
            active=replacement["active"],
        )
        assert proposed.archive == archive.archive
        assert proposed.observe()["endpoint"] == clean.observe()["endpoint"]
        assert proposed.apply(7).archive == branch.archive
        assert proposed.history_owner is None


@pytest.mark.parametrize(
    "invalid",
    ["float", "boolean", "arity", "nonunit", "position", "duplicate", "activity", "unknown"],
)
def test_prospective_edits_reject_invalid_coordinates_and_keys(word_fixtures, invalid):
    archive = NativePhaseFoldArchive((word_fixtures[0] / "rational.rfcxw").read_bytes())
    before = archive.archive
    kwargs = {"key": 8, "position": 0, "coordinates": (Fraction(1), Fraction(0))}
    if invalid == "float":
        kwargs["coordinates"] = (1.0, 0.0)
    elif invalid == "boolean":
        kwargs["coordinates"] = (True, 0)
    elif invalid == "arity":
        kwargs["coordinates"] = (Fraction(1),)
    elif invalid == "nonunit":
        kwargs["coordinates"] = (Fraction(1), Fraction(1))
    elif invalid == "position":
        kwargs["position"] = 3
    elif invalid == "duplicate":
        kwargs["key"] = 7
    elif invalid == "activity":
        kwargs["active"] = 1
    with pytest.raises((ValueError, TypeError, IndexError, OverflowError)):
        if invalid == "unknown":
            archive.apply(8)
        else:
            archive.propose(**kwargs)
    assert archive.archive == before


def test_walks_do_not_infer_a_replacement_profile(word_fixtures):
    for name in PROFILES[-2:]:
        archive = NativePhaseFoldArchive((word_fixtures[0] / (name + ".rfcxw")).read_bytes())
        with pytest.raises(ValueError):
            archive.propose(8, 0, (Fraction(1), Fraction(0)))


@pytest.mark.parametrize(
    "limit", ["wire.max_file_bytes", "wire.max_integer_bytes", "wire.max_rational_coordinates"]
)
def test_wire_reader_limits_apply_to_both_standalone_and_history_payloads(word_case, limit):
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        NativePhaseFoldArchive(word_case[1].archive, limits={limit: 0})
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        NativePhaseFoldArchive.from_history(word_case[2], CHANNEL, sequence=70, limits={limit: 0})


def test_each_declared_limit_rejects_noninteger_admission(word_fixtures):
    raw = (word_fixtures[0] / "rational.rfcxw").read_bytes()
    for name in NativePhaseFoldArchive.default_limits():
        with pytest.raises((TypeError, ValueError)):
            NativePhaseFoldArchive(raw, limits={name: True})
    with pytest.raises(ValueError):
        NativePhaseFoldArchive(raw, limits={"unknown": 1})


def test_profile_specific_admission_is_enforced_by_canonical_load(word_case):
    name, archive, _ = word_case
    field = "cover.max_path_steps" if name.endswith("walk") else "history.max_steps"
    with pytest.raises((ValueError, RuntimeError)):
        NativePhaseFoldArchive(archive.archive, limits={field: 0})
    if name.endswith("walk"):
        with pytest.raises((ValueError, RuntimeError)):
            NativePhaseFoldArchive(archive.archive, limits={"cover.max_integer_bits": 64 if name == "rational-walk" else 0})
    elif name == "fold":
        with pytest.raises((ValueError, RuntimeError)):
            NativePhaseFoldArchive(archive.archive, limits={"panels.max_creases": 0})
    elif name == "algebraic":
        with pytest.raises((ValueError, RuntimeError)):
            NativePhaseFoldArchive(archive.archive, limits={"wire.field.max_degree": 1})
    else:
        with pytest.raises((ValueError, RuntimeError)):
            NativePhaseFoldArchive(archive.archive, limits={"max_proposals": 0})


@pytest.mark.parametrize("suffix", [".rcbd", ".safetensors"])
def test_complete_words_and_prefix_observations_roundtrip_existing_formats(
    word_case, tmp_path, suffix
):
    from rexgraph.io import load, save

    full = phase_fold_record(selected(word_case, 71))
    observed = phase_fold_observation_record(
        selected(word_case), word_case[1].manifest["operation_count"]
    )
    for label, record in [("full", full), ("observed", observed)]:
        path = tmp_path / (label + suffix)
        save(path, record)
        for restored in [from_state(to_state(record)), load(path)]:
            assert (
                read_phase_fold_record(restored).archive == read_phase_fold_record(record).archive
            )
            if label == "full":
                assert (
                    read_phase_fold_record(restored).history_owner.archive
                    == read_phase_fold_record(record).history_owner.archive
                )
            else:
                assert read_phase_fold_observation_record(
                    restored
                ) == read_phase_fold_observation_record(record)


def test_independent_native_prefix_and_word_byte_limits_remain_saved(word_case):
    record = phase_fold_record(selected(word_case, 71))
    original = read_phase_fold_record(record)
    size, history_size = len(original.archive), len(original.history_owner.archive)
    owner = read_phase_fold_record(
        record, limits={"wire.max_file_bytes": size}, history_max_bytes=history_size
    )
    assert (
        owner.limits["wire.max_file_bytes"] == size
        and owner.history_owner.max_bytes == history_size
    )
    reference = phase_fold_reference(read_phase_fold_record(phase_fold_record(owner)))
    assert (
        reference["limits"]["wire.max_file_bytes"] == size
        and reference["history"]["max_bytes"] == history_size
    )
    with pytest.raises(ValueError):
        read_phase_fold_record(record, limits={"wire.max_file_bytes": size - 1})
    with pytest.raises(ValueError):
        read_phase_fold_record(record, history_max_bytes=history_size - 1)
    old = read_phase_fold_record(phase_fold_history_at(record, 70))
    assert old.history_owner.identity["sequence"] == 70 and old.archive == word_case[1].archive
    with pytest.raises((ValueError, IndexError)):
        phase_fold_history_at(record, 72)
    with pytest.raises(ValueError):
        phase_fold_history_at(phase_fold_record(word_case[1]), 70)


@pytest.mark.parametrize("attachment", ["phase_fold_archive", "phase_fold_history"])
def test_cache_hits_recheck_both_complete_images_and_charge_all_bytes(word_case, attachment):
    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache

    cache = _AlgebraicRestoreCache()
    record = phase_fold_record(selected(word_case, 71))
    first = read_phase_fold_record(record, _archive_cache=cache)
    assert read_phase_fold_record(record, _archive_cache=cache) is first
    assert cache._bytes == len(first.archive) + len(first.history_owner.archive)
    other = read_phase_fold_record(from_state(to_state(record)), _archive_cache=cache)
    assert other is not first and not other.same_carrier(first)
    record.get_metadata(1, 0, attachment)["archive"][0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="bytes"):
        read_phase_fold_record(record, _archive_cache=cache)


def test_cache_concurrency_retains_one_actual_owner_and_survives_collection(word_case):
    from rexgraph.io.algebraic_record import _AlgebraicRestoreCache

    record = phase_fold_record(selected(word_case))
    cache = _AlgebraicRestoreCache()
    with ThreadPoolExecutor(max_workers=4) as pool:
        owners = list(
            pool.map(lambda _: read_phase_fold_record(record, _archive_cache=cache), range(12))
        )
    assert all(owner is owners[0] for owner in owners)
    pin = owners[0].observed_prefix(0)
    del owners
    cache = record = None
    gc.collect()
    assert pin.history_owner is None and pin.manifest["operation_count"] == 0
    assert pin.observe()["endpoint"]


def test_native_restore_rechecks_resealed_channel_and_manifest(word_case):
    record = phase_fold_record(selected(word_case, 71))
    reference = record.get_metadata(1, 0, "phase_fold_archive")["reference"]
    reference["history"]["channel"] = (801, 1, 101)
    record.get_metadata(1, 0, "phase_fold_history")["reference"] = deepcopy(reference["history"])
    with pytest.raises(ValueError, match="declaration"):
        read_phase_fold_record(record)


@pytest.mark.parametrize(
    "mutation", ["version", "sdk", "limits", "history", "count", "shape", "field", "extra"]
)
def test_portable_references_reject_inconsistent_domain_and_policy(word_case, mutation):
    reference = deepcopy(phase_fold_reference(word_case[1]))
    if mutation == "version":
        reference["version"] = True
    elif mutation == "sdk":
        reference["sdk"]["engine_digest"] = "G" * 64
    elif mutation == "limits":
        reference["limits"]["history.max_steps"] = True
    elif mutation == "history":
        reference["history"] = {}
    elif mutation == "count":
        reference["manifest"]["operation_count"] = True
    elif mutation == "shape":
        reference["manifest"]["endpoint_shape"] = (1,)
    elif mutation == "field":
        reference["manifest"]["minimal_polynomial"] = (1.0,)
    else:
        reference["extra"] = None
    with pytest.raises((ValueError, TypeError, OverflowError)):
        validate_phase_fold_reference(reference)


@pytest.mark.parametrize(
    "mutation",
    ["version", "endpoint", "float", "boolean", "operations", "positions", "extra", "seal"],
)
def test_resealed_observations_reject_malformed_exact_data(word_case, mutation):
    value = deepcopy(phase_fold_observation(word_case[1], word_case[1].manifest["operation_count"]))
    endpoint = value["endpoint"]
    if mutation == "version":
        value["version"] = True
    elif mutation == "endpoint":
        endpoint["endpoint"] = endpoint["endpoint"][:-1]
    elif mutation == "float":
        endpoint["endpoint"] = (1.0,) + endpoint["endpoint"][1:]
    elif mutation == "boolean":
        endpoint["endpoint"] = (True,) + endpoint["endpoint"][1:]
    elif mutation == "operations":
        endpoint["operations"] = endpoint["operations"][:-1]
    elif mutation == "positions":
        endpoint["positions"] = (True,)
    elif mutation == "extra":
        endpoint["extra"] = None
    if mutation == "seal":
        value["seal"] = "0" * 64
    else:
        reseal(value)
    with pytest.raises((ValueError, TypeError)):
        validate_phase_fold_observation(value)


@pytest.mark.parametrize("backend", ["file", "rex", "sqlite"])
def test_rcdb_commit_copy_and_fresh_process_keep_history_and_logical_prefix(
    word_case, tmp_path, backend
):
    from rcdb import MemoryStore, copy_record, open_store

    record = phase_fold_record(selected(word_case, 72))
    observed = phase_fold_observation_record(
        selected(word_case), word_case[1].manifest["operation_count"]
    )
    expected = read_phase_fold_observation_record(observed)
    url = (
        f"sqlite:///{tmp_path}/word.sqlite"
        if backend == "sqlite"
        else f"{backend}://{tmp_path}/word"
    )
    store = open_store(
        url, **({} if backend == "sqlite" else {"read_only": False})
    ).configure_security(require_commits=True)
    try:
        store.commit_mutation("word", record, expected_version=0)
        store.commit_mutation("observed", observed, expected_version=0)
        assert store.verify_commits("word") and store.verify_commits("observed")
        target = MemoryStore()
        try:
            receipt = copy_record(store, target, store.get_record("observed"), return_receipt=True)
            assert receipt.source_digest == receipt.destination_digest
            assert read_phase_fold_observation_record(target.get_version("observed", 1)) == expected
        finally:
            target.close()
    finally:
        store.close()
    code = """
import importlib.abc,json,sys
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='torch':raise AssertionError('phase/fold path imported Torch')
sys.meta_path.insert(0,NoTorch())
from rcdb import open_store
from rexgraph.cpp_phase_fold import read_phase_fold_record,read_phase_fold_observation_record,phase_fold_history_at
from rcql import Executor,parse
store=open_store(sys.argv[1])
try:
 record=store.get_version('word',1)
 assert read_phase_fold_record(record).history_owner.identity['sequence']==72
 earlier=phase_fold_history_at(record,70)
 owner=read_phase_fold_record(earlier)
 value=Executor(sources={'record':earlier}).execute(parse('FROM $record RETURN PHASE_FOLD_OBSERVE('+str(owner.manifest['operation_count'])+')')).values[0]
 assert value==read_phase_fold_observation_record(store.get_version('observed',1))
 print(json.dumps({'seal':value['seal']}))
finally:store.close()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, url],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert json.loads(result.stdout) == {"seal": expected["seal"]}


def test_rcql_agent_and_system_share_logical_and_native_selectors(word_case):
    from agent.hive import Hive
    from agent.rcql_runtime import RCQLRuntime
    from rcql.capabilities import BoundSource
    from system.serialize import json_value

    from rcql import Executor, SourcePolicy, parse

    record = phase_fold_history_record(word_case[2], CHANNEL, sequence=72)
    earlier = (
        Executor(sources={"record": record})
        .execute(parse("FROM $record RETURN PHASE_FOLD_AT(70)"))
        .values[0]
    )
    text = "FROM $record RETURN PHASE_FOLD_OBSERVE(0), PHASE_FOLD_INFO(), PHASE_FOLD_PREFIX(0)"
    result = Executor(sources={"record": earlier}).execute(parse(text))
    assert result.values[0]["source"]["manifest"]["operation_count"] == 0
    assert result.values[1]["history"]["source"]["native"]["sequence"] == 70
    assert read_phase_fold_record(result.values[2]).history_owner is None
    runtime = RCQLRuntime()
    runtime.register("record", earlier, policy=SourcePolicy.allow("read"))
    routed = runtime.execute(parse(text))
    assert routed.values[:2] == result.values[:2]
    hive = Hive("phase-fold")
    hive.add_worker(
        "select", lambda sequence: phase_fold_history_at(record, sequence), capability="transform"
    )
    assert (
        read_phase_fold_record(hive.invoke("select", 70, record=False)).archive
        == word_case[1].archive
    )
    with pytest.raises(PermissionError):
        Executor(sources={"record": BoundSource(record, SourcePolicy.allow())}).execute(
            parse("FROM $record RETURN PHASE_FOLD_INFO()")
        )
    preview = json_value(read_phase_fold_record(earlier), max_values=1)
    assert preview["native_history"] is True and len(preview["endpoint"]) <= 1
    assert len(preview["operations"]) <= 1
    json.dumps(preview, allow_nan=False)


def test_rcql_proposal_and_apply_match_native_exact_word(word_fixtures):
    from rcql import Executor, parse

    for name in ["rational", "algebraic", "fold"]:
        owner = NativePhaseFoldArchive((word_fixtures[0] / (name + ".rfcxw")).read_bytes())
        p = owner.preview()["proposals"][0]
        record = phase_fold_record(owner.observed_prefix(owner.manifest["operation_count"]))
        executor = Executor(sources={"record": record}, params={"coordinates": p["phase"]})
        proposed = executor.execute(
            parse(
                "FROM $record RETURN PHASE_FOLD_PROPOSE(7," + str(p["position"]) + ",$coordinates)"
            )
        ).values[0]
        assert read_phase_fold_record(proposed).archive == owner.archive
        branch = (
            Executor(sources={"record": proposed})
            .execute(parse("FROM $record RETURN PHASE_FOLD_APPLY(7)"))
            .values[0]
        )
        assert read_phase_fold_record(branch).archive == owner.apply(7).archive


def test_observation_read_is_offline_without_optional_runtimes(word_case, tmp_path):
    from rexgraph.io import save

    record = phase_fold_observation_record(
        selected(word_case), word_case[1].manifest["operation_count"]
    )
    path = tmp_path / "word.rcbd"
    save(path, record)
    code = """
import importlib.abc,json,sys
class NoOptional(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','rexgraph_cpp'}:raise AssertionError('offline phase/fold read imported an optional runtime')
sys.meta_path.insert(0,NoOptional())
from rexgraph.io import load
from rexgraph.cpp_phase_fold import read_phase_fold_observation_record
value=read_phase_fold_observation_record(load(sys.argv[1]))
print(json.dumps({'seal':value['seal']}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert json.loads(result.stdout)["seal"] == read_phase_fold_observation_record(record)["seal"]


def test_larger_word_keeps_every_step_and_declared_coordinate_bounds(word_fixtures):
    archive = NativePhaseFoldArchive((word_fixtures[1] / "large.rfcxw").read_bytes())
    assert archive.manifest["operation_count"] == 64
    observation = phase_fold_observation(archive, 64)
    assert len(observation["endpoint"]["operations"]) == 64
    assert (
        read_phase_fold_observation_record(
            from_state(to_state(phase_fold_observation_record(archive, 64)))
        )
        == observation
    )
    with pytest.raises((ValueError, RuntimeError)):
        NativePhaseFoldArchive(archive.archive, limits={"history.max_steps": 63})
    with pytest.raises(ValueError):
        phase_fold_observation(archive, 64, max_coordinates=129)
    assert archive.observed_prefix(32).manifest["operation_count"] == 32
