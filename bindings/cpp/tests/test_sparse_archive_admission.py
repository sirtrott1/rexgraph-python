"""Sparse tensor shape admission and exact serving archive restoration."""
from pathlib import Path
import subprocess
import sys
from sys import maxsize

import pytest

from rexgraph.cpp_model import _SourceRestoreCache, cpp_source_record, restore_cpp_source
from rexgraph.cpp_stalk_pattern_model import (
    _archive_read_limits, _boundary_guards, cpp_stalk_pattern_bank_model, stalk_pattern_bank_source,
)
from rexgraph.io.model_record import load_model, model_record, read_model_record, save_model
from rexgraph.source_binding import CppSourceReference
from rexgraph_cpp import NativeSource, PreparedRelationGuards, PreparedStalkTransportPatternBank

LIMITS = (4096, 50000, 64000000, 16384)


@pytest.fixture(scope="module")
def archive():
    return (Path(__file__).parent / "fixtures" / "sparse-pattern-bank.rfcxs").read_bytes()


def test_logical_shape_does_not_consume_the_stored_entry_budget(archive):
    assert len(archive) < LIMITS[1]
    with pytest.raises(ValueError, match="logical size"):
        NativeSource(archive, max_dimension=50000, max_records=50000)
    owner = stalk_pattern_bank_source(archive, limits=LIMITS)
    bank = PreparedStalkTransportPatternBank(owner.archive, model_namespace=49051, limits=LIMITS)
    fields = bank.inspect()
    assert fields["parameter_directions"]["shape"] == (768, 768)
    assert len(fields["parameter_directions"]["entries"]) == 768
    assert bank.archive() == archive
    assert not bank.response({0: (1,) + (0,) * 255})["identified"]


@pytest.mark.parametrize("policy, message", [
    ({"max_dimension": 767, "max_logical_size": 16777216}, "dimension"),
    ({"max_dimension": 50000, "max_logical_size": 589823}, "logical size"),
    ({"max_dimension": 50000, "max_logical_size": 16777216, "max_records": 1}, "budget"),
])
def test_axis_logical_shape_and_record_limits_remain_independent(archive, policy, message):
    with pytest.raises(ValueError, match=message):
        NativeSource(archive, **policy)


def test_restore_cache_cannot_reuse_a_more_permissive_logical_shape(archive):
    source = stalk_pattern_bank_source(archive, limits=LIMITS)
    ref = CppSourceReference(source)
    attachment = cpp_source_record(source, ref)
    cache = _SourceRestoreCache()
    policy = _archive_read_limits(LIMITS)
    first = restore_cpp_source(attachment, ref, _source_cache=cache, **policy)
    assert restore_cpp_source(attachment, ref, _source_cache=cache, **policy) is first
    policy["max_logical_size"] = 589823
    with pytest.raises(ValueError, match="logical size"):
        restore_cpp_source(attachment, ref, _source_cache=cache, **policy)


def test_boundary_reader_accepts_declared_shape_and_retains_original_source(archive):
    with pytest.raises(ValueError, match="logical size"):
        PreparedRelationGuards(archive, limits=(50000, 64000000, 16384))
    guards = PreparedRelationGuards(archive, limits=(50000, 64000000, 16384),
                                   max_logical_size=16777216)
    assert guards.receipt["max_logical_size"] == 16777216
    assert len(guards.boundary(0)["occurrences"]) == 1
    bank = PreparedStalkTransportPatternBank(archive, model_namespace=49051, limits=LIMITS)
    assert guards.boundary(0)["occurrences"][0]["key"] == int(bank.inspect()["occurrence_keys"][0])
    assert _boundary_guards(bank).boundary(0) == guards.boundary(0)


@pytest.mark.parametrize("value", [True, 1.5, -1, 1 << 100])
def test_logical_size_limit_requires_an_admitted_integer(archive, value):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        NativeSource(archive, max_logical_size=value)
    source = stalk_pattern_bank_source(archive, limits=LIMITS)
    ref = CppSourceReference(source)
    with pytest.raises((TypeError, ValueError)):
        restore_cpp_source(cpp_source_record(source, ref), ref, max_logical_size=value)


def test_derived_logical_size_limit_saturates_at_the_machine_width():
    policy = _archive_read_limits((2 * maxsize + 1, 50000, 64000000, 16384))
    assert policy["max_logical_size"] == 2 * maxsize + 1


@pytest.mark.parametrize("suffix", [".safetensors", ".rcbd"])
def test_cold_model_record_uses_the_same_sparse_archive_policy(archive, tmp_path, suffix):
    source = stalk_pattern_bank_source(archive, limits=LIMITS)
    state = cpp_stalk_pattern_bank_model(source, limits=LIMITS)
    path = tmp_path / ("bank" + suffix)
    save_model(path, model_record(source, state))
    loaded, owner = read_model_record(load_model(path))
    assert loaded.coefficient_digest == state.coefficient_digest
    assert owner.archive == archive
    bank = PreparedStalkTransportPatternBank(owner.archive, model_namespace=49051, limits=LIMITS)
    assert bank.archive() == archive
    assert bank.inspect()["parameter_directions"]["shape"] == (768, 768)
    code = """
import sys
from rexgraph.io.model_record import load_model, read_model_record
from rexgraph_cpp import PreparedStalkTransportPatternBank
state, source = read_model_record(load_model(sys.argv[1]))
assert state.coefficient_digest == sys.argv[2]
bank = PreparedStalkTransportPatternBank(source.archive, model_namespace=49051,
    limits=state.configuration['limits'])
assert bank.archive() == source.archive
assert bank.inspect()['parameter_directions']['shape'] == (768, 768)
assert not bank.response({0: (1,) + (0,) * 255})['identified']
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(path), state.coefficient_digest],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
