"""Actual authenticated capability routes and RexClient use one relational service."""
import base64
from fractions import Fraction as Q

import pytest
from agent.client import RexClient
from fastapi.testclient import TestClient

from agent import rcdb
from rcdb import ComplexRecord, MemoryStore
from rcql import Result


@pytest.fixture
def network(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("REXGRAPH_AUDIT_JOURNAL", str(tmp_path / "audit.jsonl"))
    from agent.server import audit, auth
    auth.reset_auth_manager()
    audit.reset_cache()
    manager = auth.get_auth_manager()
    manager.enable_auth()
    manager.bootstrap_admin()
    admin = manager.create_token("alice", ["alpha"], role="user")
    bob = manager.create_token("bob", ["beta"], role="user")
    store = MemoryStore()
    monkeypatch.setattr(rcdb, "default_store", lambda: store)
    from agent.server.app import app
    with TestClient(app) as client:
        yield client, store, {"Authorization": f"Bearer {admin}", "X-Workspace": "alpha"}, {
            "Authorization": f"Bearer {bob}", "X-Workspace": "beta"}
    store.close()
    auth.reset_auth_manager()
    audit.reset_cache()


def invoke(client, name, headers, **arguments):
    return client.post("/rex/v1/call", headers=headers, json={"name": name, "arguments": arguments})


def test_tools_are_discovered_through_native_and_MCP_routes(network):
    client, _, _, headers = network
    for path in ("/rex/v1/tools", "/api/v1/mcp/tools"):
        reply = client.get(path, headers=headers)
        assert reply.status_code == 200, reply.text
        names = {t["name"] for t in reply.json()["tools"]}
        assert {"rexgraph_rcdb_publish", "rexgraph_rcql_query", "rexgraph_rcql_run_program"} <= names


def test_authenticated_record_query_history_exact_payload_and_audit(network):
    client, store, alpha, beta = network
    published = invoke(client, "rexgraph_rcdb_publish", alpha, record_id="data",
        cells=[3, [[0, 1], [1, 2]]], expected_version=0, meta={"workspace": "beta"})
    assert published.status_code == 200, published.text
    rec = ComplexRecord.from_dict(published.json()["result"]["record"])
    assert rec.meta["workspace"] == "alpha"
    answer = invoke(client, "rexgraph_rcql_query", alpha,
        query='FROM RCDB_VERSION($db,"data",1) RETURN COUNT(CELLS(1)),SUM([1/3,2/7])', save_as="answer")
    assert answer.status_code == 200, answer.text
    body = answer.json()["result"]
    assert body["values"] == [2, {"numerator": 13, "denominator": 21}]
    result = Result.from_bytes(base64.b64decode(body["result_base64"]))
    assert result.values == (2, Q(13, 21))
    assert store.read_record("answer").value.to_bytes() == result.to_bytes()
    assert invoke(client, "rexgraph_rcdb_list", beta).json()["result"]["records"] == []
    assert invoke(client, "rexgraph_rcdb_inspect", beta, record_id="data", version=1).status_code == 404
    assert invoke(client, "rexgraph_rcdb_publish", beta, record_id="data",
        cells=[2, [[0, 1]]], expected_version=1).status_code == 403
    assert invoke(client, "rexgraph_rcdb_publish", alpha, record_id="data",
        cells=[2, [[0, 1]]], expected_version=0).status_code == 400
    trail = client.get("/rex/v1/audit", headers=alpha).json()
    assert trail["integrity"]
    assert any(e.get("target") == "rexgraph_rcql_query" for e in trail["entries"])


def test_HTTP_and_MCP_run_programs_and_reject_unbound_or_untyped_requests(network):
    client, _, headers, _ = network
    response = invoke(client, "rexgraph_rcdb_publish", headers, record_id="data",
        cells=[2, [[0, 1]]], expected_version=0)
    assert response.status_code == 200, response.text
    declaration = {"name": "Double", "steps": [{"name": "twice", "query": "FROM $data RETURN SUM([$x,$x])"}],
        "inputs": [{"name": "x", "kind": "ExactRational"}]}
    response = invoke(client, "rexgraph_rcql_publish_program", headers, record_id="double",
        declaration=declaration, expected_version=0)
    assert response.status_code == 200, response.text
    arguments = {"record_id": "double", "version": 1, "parameters": {"x": {"numerator": 2, "denominator": 3}},
        "sources": {"data": {"record_id": "data", "version": 1}}}
    response = client.post("/api/v1/mcp/call", headers=headers,
        json={"name": "rexgraph_rcql_run_program", "arguments": arguments})
    assert response.status_code == 200, response.text
    assert response.json()["result"]["values"] == [{"numerator": 4, "denominator": 3}]
    arguments["parameters"] = {"x": 2}
    assert invoke(client, "rexgraph_rcql_run_program", headers, **arguments).status_code == 400
    assert invoke(client, "rexgraph_rcql_query", headers, query='FROM REX("/tmp/secret") RETURN BETTI(0)').status_code == 404
    assert invoke(client, "rexgraph_rcql_query", headers, query="FROM $db RETURN RCDB_LIST()", allow_write="yes").status_code == 400


def test_RexClient_conveniences_use_the_existing_native_tool_route(network, monkeypatch):
    client, _, alpha, _ = network
    import httpx
    requests = []
    def post(url, **options):
        requests.append(url)
        options.pop("timeout", None)
        return client.post(url.replace("http://agent", ""), **options)
    monkeypatch.setattr(httpx, "post", post)
    remote = RexClient("http://agent", api_key=alpha["Authorization"].removeprefix("Bearer "), workspace="alpha")
    remote.rcdb_publish("client-data", [2, [[0, 1]]], expected_version=0)
    answer = remote.rcql_query('FROM RCDB_VERSION($db,"client-data",1) RETURN COUNT(CELLS(1))')
    assert answer["result"]["values"] == [1]
    assert all(url == "http://agent/rex/v1/call" for url in requests)


def test_cell_admission_and_bad_output_options_refuse_before_publication(network, monkeypatch):
    client, store, headers, _ = network
    monkeypatch.setenv("REXGRAPH_MAX_CELLS", "3")
    reply = invoke(client, "rexgraph_rcdb_publish", headers, record_id="too-large", cells=[4, []], expected_version=0)
    assert reply.status_code == 429 and store.get_record("too-large") is None
    assert invoke(client, "rexgraph_rcdb_publish", headers, record_id="small", cells=[2, [[0, 1]]], expected_version=0).status_code == 200
    reply = invoke(client, "rexgraph_rcql_query", headers, allow_write=True, include_payload="yes",
        query='FROM $db MUTATE "copy" SET state=RCDB_GET("small"),expected_version=0 COMMIT')
    assert reply.status_code == 400 and store.get_record("copy") is None


def test_real_configured_default_store_composes_with_request_workspace_scope(tmp_path, monkeypatch):
    from agent.server import auth
    from agent.server.app import app
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "local://" + str(tmp_path / "records"))
    rcdb.reset_default_store()
    manager = auth.get_auth_manager()
    manager.enable_auth()
    manager.bootstrap_admin()
    token = manager.create_token("writer", ["alpha"], role="user")
    headers = {"Authorization": f"Bearer {token}", "X-Workspace": "alpha"}
    try:
        with TestClient(app) as client:
            reply = invoke(client, "rexgraph_rcdb_publish", headers, record_id="data",
                cells=[2, [[0, 1]]], expected_version=0)
            assert reply.status_code == 200, reply.text
            reply = invoke(client, "rexgraph_rcql_query", headers,
                query='FROM RCDB_VERSION($db,"data",1) RETURN COUNT(CELLS(1))', save_as="answer")
            assert reply.status_code == 200, reply.text
        assert rcdb.default_store().get_record("answer").meta["workspace"] == "alpha"
        assert rcdb.default_store().commit_history("data")[-1].transition.actor == "writer"
    finally:
        rcdb.reset_default_store()
