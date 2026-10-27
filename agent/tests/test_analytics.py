"""Live RCQL record views with no SQL dependency or independent payload copy."""
import pytest
from agent import analytics, rcdb
from rexgraph import RexGraph


@pytest.fixture(params=["memory", "local"])
def store(request, tmp_path):
    s = rcdb.MemoryStore() if request.param == "memory" else rcdb.LocalStore(tmp_path / "records")
    for i in range(12):
        s.put(str(i), RexGraph.from_cells([i+2, [[j,j+1] for j in range(i+1)]]),
              meta={"source": "even" if i%2 == 0 else "odd"})
    yield s
    s.close()


def test_live_filter_uses_rcdb_selection(store):
    view = analytics.signature_view(store)
    assert len(view) == 12
    assert set(view.ids({"min_nE": 10})) == {r.id for r in store.query(min_nE=10)}
    assert view.query('FROM $db RETURN COUNT(RCDB_SELECT())').values == (12,)
    assert len(view.query('FROM $db MATCH r IN RCDB_SELECT() WHERE r.signature.nE >= 10 RETURN r.id').values[0]) == 3


def test_source_predicates_and_counts_are_rcql(store):
    view = analytics.signature_view(store)
    for source in ["even", "odd"]:
        assert view.query('FROM $db RETURN COUNT(RCDB_SELECT($filters))',
            parameters={"filters": {"source": source}}).values == (6,)
    with pytest.raises(ValueError, match="SQL"):
        view.sql("SELECT * FROM signatures")


def test_revision_selected_payloads_remain_with_the_store(store):
    view = analytics.signature_view(store)
    assert all(value.nE >= 10 for _, value in view.complexes({"min_nE": 10}))
    store.put("later", RexGraph.from_cells([2, [[0,1]]]))
    assert len(view) == 13
    assert view.refresh() is view


def test_metadata_export_does_not_fill_absent_measurements(store):
    view = analytics.signature_view(store)
    assert view.to_arrow().num_rows == 12
    assert len(view.describe()) == len(analytics.COLUMNS)
    assert "kappa_mean" in view.columns()


def test_history_retains_actual_record_versions(tmp_path):
    s = rcdb.LocalStore(tmp_path / "records")
    s.put("same", RexGraph.from_cells([2, [[0,1]]]))
    s.put("same", RexGraph.from_cells([3, [[0,1],[1,2]]]))
    view = analytics.signature_view(s, include_history=True)
    assert len(view) == 2
    assert sorted(value.nE for _, value in view.complexes()) == [1,2]
    s.close()


def test_analytics_has_no_optional_sql_runtime(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in {"sqlalchemy", "duckdb"}:
            pytest.fail("SQL dependency was imported")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert analytics.available()
    assert len(analytics.signature_view(rcdb.MemoryStore())) == 0
