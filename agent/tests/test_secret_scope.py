"""Native record routes retain socket identity and workspace isolation."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ALPHA_URI = "postgresql://alice:alicepw@alpha.example.com:5432/app"
BETA_URI = "postgresql://bob:bobpw@beta.example.com:5432/app"


@pytest.fixture
def tenants(tmp_path, monkeypatch):
    pytest.importorskip("bcrypt", reason="authentication requires the server extra")
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", f"file://{tmp_path}/connections.json")
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", f"local://{tmp_path}/rcdb")
    from agent.server import audit, auth, scope
    scope.reset_secret_store()
    auth.reset_auth_manager(); audit.reset_cache()
    from agent.server.app import app
    mgr = auth.get_auth_manager(); mgr.enable_auth()
    mgr.bootstrap_admin()
    alice = mgr.create_token("alice", ["alpha"], role="user")
    bob = mgr.create_token("bob", ["beta"], role="user")
    ah = {"Authorization": f"Bearer {alice}", "X-Workspace": "alpha"}
    bh = {"Authorization": f"Bearer {bob}", "X-Workspace": "beta"}
    yield TestClient(app), ah, bh
    scope.reset_secret_store()
    auth.reset_auth_manager(); audit.reset_cache()


def _seed():
    from agent import rcdb
    from rexgraph import RexGraph
    rcdb.reset_default_store()
    store = rcdb.default_store()
    store.put("alpha-record", RexGraph.from_cells([3, [[0, 1, 2]]]), meta={"workspace": "alpha"})
    store.put("beta-record", RexGraph.from_cells([2, [[0, 1]]]), meta={"workspace": "beta"})
    return store


def _names(client, headers):
    result = client.get("/api/v1/dbmanager/records", headers=headers)
    assert result.status_code == 200, result.text
    return {row["id"] for row in result.json()["result"]["records"]}


def test_a_tenant_only_lists_its_native_records(tenants):
    client, alice, bob = tenants
    _seed()
    assert _names(client, alice) == {"alpha-record"}
    assert _names(client, bob) == {"beta-record"}


def test_a_tenant_cannot_inspect_another_tenants_native_record(tenants):
    client, alice, bob = tenants
    _seed()
    result = client.post("/api/v1/dbmanager/inspect", headers=bob,
                         json={"record_id": "alpha-record", "version": 1})
    assert result.status_code == 404, result.text


def test_forged_workspace_cannot_bind_another_record_scope(tenants):
    client, alice, bob = tenants
    _seed()
    forged = {**bob, "X-Workspace": "alpha"}
    result = client.get("/api/v1/dbmanager/records", headers=forged)
    assert result.status_code == 403, result.text


def test_cross_workspace_publication_does_not_overwrite(tenants):
    client, alice, bob = tenants
    store = _seed()
    before = store.read_record("alpha-record").state_digest
    result = client.post("/api/v1/dbmanager/publish", headers=bob, json={
        "record_id": "alpha-record", "expected_version": 1, "cells": [2, [[0, 1]]]})
    assert result.status_code in {400, 403, 404}, result.text
    assert store.read_record("alpha-record").state_digest == before
    assert store.get_record("alpha-record").version == 1


def test_sql_source_connections_are_scoped_without_becoming_primary_stores(tenants):
    client,alice,bob=tenants
    response=client.post("/api/v1/dbmanager/connections",headers=alice,json={"name":"prod","uri":ALPHA_URI})
    assert response.status_code==200
    own=client.get("/api/v1/dbmanager/connections",headers=alice).json()["connections"]
    other=client.get("/api/v1/dbmanager/connections",headers=bob).json()["connections"]
    assert own==[{"name":"prod","kind":"sql"}] and other==[]
    response=client.post("/api/v1/dbmanager/sql/catalog",headers=bob,json={"connection":"prod"})
    assert response.status_code==404


def test_outside_a_request_saved_secrets_remain_operator_owned(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", f"file://{tmp_path}/c.json")
    from agent.secrets import open_secret_store
    store = open_secret_store()
    store.put("cli", ALPHA_URI, "sql")
    assert store.get("cli") == ALPHA_URI
    assert [c.get("name") for c in store.list()] == ["cli"]


def test_connector_native_record_reads_obey_the_same_workspace_scope(tenants):
    client, alice, bob = tenants
    _seed()
    result = client.post("/api/v1/connectors/read", headers=bob,
                         json={"scheme": "rcdb", "source": {"record_id": "alpha-record"}})
    assert result.status_code == 400
    result = client.post("/api/v1/connectors/read", headers=alice,
                         json={"scheme": "rcdb", "source": {"record_id": "alpha-record"}})
    assert result.status_code == 200 and result.json()["nE"] == 1, result.text
