"""agent.data_complex: records as a relational complex: clusters, outliers, centrality."""

from agent.agentic_db import AgenticDB
from agent.data_complex import analyze_rows, rows_to_complex


def test_records_cluster_by_shared_value():
    rows = [{"id": f"o{i}", "cust": c} for i, c in enumerate(["A", "A", "A", "B", "B", "C"])]
    r = analyze_rows(rows, link_on="cust", id_col="id")
    assert r["n_rows"] == 6 and r["n_clusters"] == 3
    assert set(r["clusters"][0]) == {"o0", "o1", "o2"}       # the largest cluster first
    assert "o5" in r["outliers"]                             # cust C: a single record, isolated
    assert r["central"]                                      # coherence centrality reported


def test_no_shared_values_all_outliers():
    rows = [{"id": "a", "k": 1}, {"id": "b", "k": 2}]
    r = analyze_rows(rows, link_on="k", id_col="id")
    assert r["n_clusters"] == 2 and set(r["outliers"]) == {"a", "b"}
    # Unlinked records remain declared grade zero cells with no relations.
    rex, meta = rows_to_complex(rows, link_on="k")
    assert int(rex.nV) == 2 and int(rex.nE) == 0
    assert int(rex.betti[0]) == 2
    # no id_col on this call, so participants are labelled by row index
    assert meta["unattached_participants"] == ["0", "1"]


def test_agentic_db_data_complex(tmp_path):
    from agent import rcdb
    store = rcdb.LocalStore(tmp_path / "records")
    rows = [{"id": i, "customer_id": c} for i,c in [(1,10),(2,10),(3,10),(4,20),(5,30)]]
    store.put_record("orders", rows, expected_version=0)
    with AgenticDB(store=store) as db:
        result = db.data_complex("orders", link_on="customer_id", id_col="id")
        assert result["n_rows"] == 5 and len(result["clusters"][0]) == 3
        assert {"4", "5"} <= set(result["outliers"])
    store.close()
