"""Route tests for /api/v1/connectors via TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _record():
    from rexgraph import RexGraph
    from agent.rcdb import default_store
    graph = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])
    default_store().put("source", graph)
    return {"record_id": "source", "version": 1}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate native record storage and workspace handles.
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "local://" + str(tmp_path / "rcdb.sqlite"))
    from agent import rcdb
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    rcdb.reset_default_store()
    from agent.server.app import app
    from agent.server.auth import get_auth_manager
    get_auth_manager().disable_auth(persist=False)
    with TestClient(app) as client:
        yield client
    rcdb.reset_default_store()


def test_list_endpoint(client):
    r = client.get("/api/v1/connectors")
    assert r.status_code == 200
    names = {c["connector"] for c in r.json()["connectors"]}
    assert {"RCDBConnector", "NativeFileConnector", "GraphConnector"} <= names


def test_read_and_validate_endpoints(client):
    source = _record()
    result = client.post("/api/v1/connectors/read", json={"scheme": "rcdb", "source": source})
    assert result.status_code == 200, result.text
    assert (result.json()["nV"], result.json()["nE"]) == (4, 2)
    result = client.post("/api/v1/connectors/validate", json={"scheme": "rcdb", "source": source})
    assert result.status_code == 200 and result.json()["ok"], result.text


def test_ingest_lands_in_rcdb(client):
    source = _record()
    result = client.post("/api/v1/connectors/ingest", json={"scheme": "rcdb", "source": source,
                         "id": "shop", "tags": ["demo"]})
    assert result.status_code == 200 and result.json()["stored_as"] == "shop", result.text
    ids = [r["id"] for r in client.get("/api/v1/db/list").json()["records"]]
    assert "shop" in ids


def test_in_memory_edges_via_scheme(client):
    r = client.post("/api/v1/connectors/read",
                    json={"scheme": "edges", "source": [["a", "b", 3.0], ["b", "c", 1.0]]})
    assert r.status_code == 200
    assert r.json()["nV"] == 3 and r.json()["nE"] == 2


def test_missing_source_is_400(client):
    r = client.post("/api/v1/connectors/read", json={})
    assert r.status_code == 400
