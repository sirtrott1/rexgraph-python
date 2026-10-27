"""Typed record selection delegates clocks, filters and history to RCDB."""
from contextlib import closing
import pytest
from rcql import BoundSource, Executor, SourcePolicy, parse
from rexgraph import RexGraph


@pytest.fixture(params=["memory", "local"])
def store(request, tmp_path):
    rcdb = pytest.importorskip("rcdb")
    selected = rcdb.MemoryStore() if request.param == "memory" else rcdb.LocalStore(tmp_path / "records")
    with closing(selected) as value:
        value.put("first", RexGraph.from_cells([3, [[0, 1, 2]]]), tags=["selected"], _tx_time=10.0,
                  valid_from=1.0, valid_to=5.0)
        value.put("first", RexGraph.from_cells([3, [[0, 1], [1, 2]]]), tags=["selected"], _tx_time=20.0,
                  valid_from=5.0, valid_to=9.0)
        value.put("second", RexGraph.from_cells([2, [[0, 1]]]), tags=["other"], _tx_time=30.0)
        yield value


def execute(store, *, filters=None, limit=100, offset=0, as_of=None, valid_at=None, history=False):
    return Executor(sources={"db": store}, params=locals()).execute(parse(
        'FROM $db RETURN RCDB_SELECT($filters,$limit,$offset,$as_of,$valid_at,$history)')).values[0]


def test_selects_filters_ordered_pages_and_typed_signature(store, monkeypatch):
    def no_payload(*args, **kwargs):
        raise AssertionError("metadata query decoded a payload")
    monkeypatch.setattr(store, "read_record", no_payload)
    assert [r["id"] for r in execute(store)] == ["second", "first"]
    assert [r["id"] for r in execute(store, filters={"tags_all": ["selected"]})] == ["first"]
    assert [r["id"] for r in execute(store, limit=1, offset=1)] == ["first"]
    assert execute(store, limit=0) == []
    assert execute(store, filters={"tags_all": ["selected"]}, limit=0) == []
    value = Executor(sources={"db": store}).execute(parse(
        'FROM $db MATCH r IN RCDB_SELECT() WHERE r.signature.nE >= 2 RETURN r.id, r.version'))
    assert value.values == ((("first", 2),),)


def test_selects_the_actual_cut_and_complete_record_history(store):
    assert [(r["id"], r["version"]) for r in execute(store, as_of=15.0)] == [("first", 1)]
    assert [(r["id"], r["version"]) for r in execute(store, valid_at=3.0)] == [("first", 1)]
    rows = execute(store, filters={"tags_all": ["selected"]}, history=True)
    assert [(r["id"], r["version"]) for r in rows] == [("first", 2), ("first", 1)]
    assert execute(store, as_of=5.0) == []


def test_existing_source_policy_projects_identity_and_signature(store):
    policy = SourcePolicy.allow("read", "records", "search", record_fields=["nE"])
    rows = Executor(sources={"db": BoundSource(store, policy)}).execute(parse(
        'FROM $db RETURN RCDB_SELECT()')).values[0]
    assert all("id" not in row and set(row["signature"]) == {"nE"} for row in rows)
    with pytest.raises(PermissionError):
        Executor(sources={"db": BoundSource(store, SourcePolicy.allow("read"))}).execute(parse(
            'FROM $db RETURN RCDB_SELECT()'))


@pytest.mark.parametrize("options", [
    {"filters": "DROP TABLE data"}, {"filters": {"unknown": 1}},
    {"limit": True}, {"limit": -1}, {"limit": 10001},
    {"offset": -1}, {"offset": True}, {"history": 1},
    {"as_of": float("nan")}, {"valid_at": float("inf")}, {"as_of": True},
])
def test_invalid_record_selection_is_explicit(store, options):
    with pytest.raises((ValueError, TypeError)):
        execute(store, **options)
