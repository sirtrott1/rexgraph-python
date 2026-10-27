"""Query scoped source owners retain complete archive and reference validation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from rexgraph.cpp_model import _SourceRestoreCache, cpp_model, cpp_source_record, restore_cpp_source
from rexgraph.io.model_record import model_record, read_model_record
from rexgraph.source_binding import CppSourceReference


def attachment(source):
    reference = CppSourceReference(source)
    return cpp_source_record(source, reference), reference


def test_same_attachment_reuses_owner_after_current_validation(native_source):
    record, reference = attachment(native_source)
    cache = _SourceRestoreCache()
    first = restore_cpp_source(record, reference, _source_cache=cache)
    second = restore_cpp_source(record, reference, _source_cache=cache)
    assert first is second
    assert first.archive == native_source.archive
    assert first.identity == native_source.identity
    assert len(cache._entries) == 1 and cache._archive_bytes == len(first.archive)
    assert restore_cpp_source(record, reference) is not first


@pytest.mark.parametrize("change", ["archive", "namespace", "cutoff", "shape", "dtype", "extra"])
def test_cache_hits_do_not_accept_changed_attachments(native_source, change):
    record, reference = attachment(native_source)
    cache = _SourceRestoreCache()
    restore_cpp_source(record, reference, _source_cache=cache)
    if change == "archive":
        record["archive"][0] ^= 1
    elif change == "namespace":
        record["reference"]["native"]["namespace"] += 1
    elif change == "cutoff":
        record["reference"]["native"]["sequence"] += 1
    elif change == "shape":
        record["archive"] = record["archive"].reshape(1, -1)
    elif change == "dtype":
        record["archive"] = record["archive"].astype(np.int64)
    else:
        record["extra"] = True
    with pytest.raises(ValueError):
        restore_cpp_source(record, reference, _source_cache=cache)


@pytest.mark.parametrize("limit", [0, True, 1.5, float("inf"), -1, 1 << 100])
def test_each_restore_checks_current_byte_admission(native_source, limit):
    record, reference = attachment(native_source)
    cache = _SourceRestoreCache()
    restore_cpp_source(record, reference, _source_cache=cache)
    with pytest.raises((TypeError, ValueError)):
        restore_cpp_source(record, reference, max_bytes=limit, _source_cache=cache)


def test_equal_bytes_in_distinct_attachments_do_not_identify_owners(native_source):
    record, reference = attachment(native_source)
    cache = _SourceRestoreCache()
    first = restore_cpp_source(record, reference, _source_cache=cache)
    other = deepcopy(record)
    second = restore_cpp_source(other, reference, _source_cache=cache)
    assert first is not second
    record["archive"] = record["archive"].copy()
    assert restore_cpp_source(record, reference, _source_cache=cache) is not first


@pytest.mark.parametrize("bound", ["sources", "bytes", "disabled"])
def test_restored_owner_retention_is_bounded(native_source, bound):
    record, reference = attachment(native_source)
    size = len(native_source.archive)
    cache = _SourceRestoreCache(max_sources=0 if bound == "disabled" else 1 if bound == "sources" else 8,
                               max_archive_bytes=size if bound == "bytes" else size * 4)
    first = restore_cpp_source(record, reference, _source_cache=cache)
    restore_cpp_source(deepcopy(record), reference, _source_cache=cache)
    assert len(cache._entries) <= (0 if bound == "disabled" else 1)
    assert cache._archive_bytes <= (size if bound == "bytes" else size * 4)
    assert restore_cpp_source(record, reference, _source_cache=cache) is not first


def test_cached_source_does_not_skip_model_parameter_validation(native_source):
    state = cpp_model(native_source, (10, 11, 12))
    record = model_record(native_source, state)
    cache = _SourceRestoreCache()
    read_model_record(record, _source_cache=cache)
    changed = replace(state, step=1)
    record.attach_metadata(1, 0, "model", changed)
    with pytest.raises(ValueError, match="differs"):
        read_model_record(record, _source_cache=cache)


def test_source_cache_accepts_parallel_readers_with_exact_publication(native_source):
    record, reference = attachment(native_source)
    cache = _SourceRestoreCache()
    first = restore_cpp_source(record, reference, _source_cache=cache)
    with ThreadPoolExecutor(max_workers=4) as pool:
        owners = tuple(pool.map(lambda _: restore_cpp_source(record, reference, _source_cache=cache), range(16)))
    assert all(source is first for source in owners)
    assert cache._archive_bytes == len(first.archive)


def test_query_resources_are_distinct_and_reset_on_nested_failure():
    from rcql.execution_trace import current_query_resources, query_resource_scope
    assert current_query_resources() is None
    with query_resource_scope():
        outer = current_query_resources()
        outer["retained"] = object()
        # Nested program steps reuse their enclosing owners. An explicit
        # resource dictionary declares the independent lifetime tested here.
        with pytest.raises(RuntimeError), query_resource_scope(resources={}):
            assert current_query_resources() is not outer
            raise RuntimeError("failed query")
        assert current_query_resources() is outer
    assert current_query_resources() is None


def test_nested_program_resources_share_owners_and_reset_on_failure():
    from rcql.execution_trace import current_query_resources, query_resource_scope
    assert current_query_resources() is None
    with query_resource_scope():
        outer = current_query_resources()
        retained = outer['retained'] = object()
        with pytest.raises(RuntimeError), query_resource_scope():
            assert current_query_resources() is outer
            assert current_query_resources()['retained'] is retained
            raise RuntimeError('failed program step')
        assert current_query_resources() is outer
    assert current_query_resources() is None


def test_rcql_reuses_source_only_inside_one_query(native_source, monkeypatch):
    from rcql.execution_trace import current_query_resources

    import rexgraph.cpp_model as models
    from rcql import Executor, parse
    state = cpp_model(native_source, (10, 11, 12))
    record = model_record(native_source, state)
    executor = Executor(sources={"record": record})
    query = parse("FROM $record RETURN MODEL_INFO(MODEL_STATE())")
    original = models.restore_cpp_source
    owners, scopes = [], []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        owners.append(result);scopes.append(current_query_resources())
        return result
    monkeypatch.setattr(models, "restore_cpp_source", observed)
    first = executor.execute(query)
    assert len(owners) >= 2 and all(owner is owners[0] for owner in owners)
    assert all(scope is scopes[0] for scope in scopes)
    assert current_query_resources() is None
    retained = owners[0];owners.clear();scopes.clear()
    second = executor.execute(query)
    assert all(owner is owners[0] for owner in owners) and owners[0] is not retained
    assert first.values == second.values
    assert current_query_resources() is None


def test_query_rechecks_archive_after_a_previous_restore(native_source, monkeypatch):
    from rcql.execution_trace import current_query_resources

    import rexgraph.cpp_model as models
    from rcql import Executor, parse
    state = cpp_model(native_source, (10, 11, 12))
    record = model_record(native_source, state)
    retained = record.get_metadata(1, 0, "model_source")
    original = models.restore_cpp_source
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        retained["archive"][0] ^= 1
        return result
    monkeypatch.setattr(models, "restore_cpp_source", changed)
    executor = Executor(sources={"record": record})
    with pytest.raises(ValueError, match="digest differs"):
        executor.execute(parse("FROM $record RETURN MODEL_INFO(MODEL_STATE())"))
    assert current_query_resources() is None


def test_query_source_reuse_does_not_grant_read_permission(native_source):
    from rcql.capabilities import BoundSource, SourcePolicy
    from rcql.execution_trace import current_query_resources

    from rcql import Executor, parse
    state = cpp_model(native_source, (10, 11, 12))
    record = model_record(native_source, state)
    executor = Executor(sources={"record": BoundSource(record, SourcePolicy.allow("train"))})
    with pytest.raises(PermissionError):
        executor.execute(parse("FROM $record RETURN MODEL_INFO(MODEL_STATE())"))
    assert current_query_resources() is None
