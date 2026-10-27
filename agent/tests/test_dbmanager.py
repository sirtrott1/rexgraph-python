"""The database manager delegates RCDB/RCQL to the existing socket tool gate."""
import pytest
from fastapi.testclient import TestClient
from agent import rcdb
from rexgraph import RexGraph
from rexgraph.object_identity import object_digest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("REXGRAPH_RCDB_URI", "local://" + str(tmp_path / "records"))
    from agent.server.app import app
    from agent.server.scope import reset_secret_store
    monkeypatch.setenv("REXGRAPH_SECRETS_URI","file://"+str(tmp_path/"secrets.json"))
    reset_secret_store()
    from agent.server.auth import get_auth_manager
    rcdb.reset_default_store()
    get_auth_manager().disable_auth(persist=False)
    with TestClient(app) as value:
        yield value
    rcdb.reset_default_store()
    reset_secret_store()


def test_current_records_and_typed_query_share_the_store(client):
    rcdb.default_store().put("source", RexGraph.from_cells([3, [[0,1,2]]]))
    response = client.get("/api/v1/dbmanager/records")
    assert response.status_code == 200 and response.json()["result"]["records"][0]["id"] == "source"
    response = client.post("/api/v1/dbmanager/query", json={"query":
        'FROM RCDB_VERSION($db,"source",1) RETURN COUNT(CELLS(1))'})
    assert response.status_code == 200
    from rcql import Result
    import base64
    assert Result.from_bytes(base64.b64decode(response.json()["result"]["result_base64"])).values == (1,)


def test_sql_sources_are_saved_without_replacing_primary_store(client):
    uri="postgresql://user:secret@host/database?token=private"
    result=client.post("/api/v1/dbmanager/connections",json={"name":"source","uri":uri})
    assert result.status_code==200
    listing=client.get("/api/v1/dbmanager/connections")
    assert listing.json()["connections"]==[{"name":"source","kind":"sql"}]
    assert "private" not in listing.text and "secret" not in listing.text
    assert rcdb.default_store().backend=="local"


def test_offline_schema_rendering_remains_available(client):
    result = client.post("/api/v1/dbmanager/ddl", json={"spec": {"tables": [{"name": "sample", "columns": ["id"]}]}})
    assert result.status_code == 200 and "sample" in result.json()["ddl"]


@pytest.mark.parametrize("format", ["safetensors", "rcbf", "rcbd"])
def test_uploaded_native_file_publishes_under_existing_guard(client, tmp_path, format):
    from rexgraph.io import load, save
    from _rexgraph_agent_tests.test_native_file_records import rcbf_bytes, zipped_bundle
    value = RexGraph.from_cells([3, [[0,1,2]]])
    path = tmp_path / ("source." + format)
    if format == "rcbf":
        path.write_bytes(rcbf_bytes())
        value = load(path)
        contents = path.read_bytes()
    elif format == "rcbd":
        save(path, value)
        contents = zipped_bundle(path)
    else:
        save(path, value)
        contents = path.read_bytes()
    uploaded = client.post("/rex/v1/upload", content=contents, headers={"X-Filename": path.name})
    assert uploaded.status_code == 200
    args = {"record_id": "imported", "expected_version": 0,
            "files": [uploaded.json()["handle"]], "format": format}
    response = client.post("/api/v1/dbmanager/import", json=args)
    assert response.status_code == 200, response.text
    assert object_digest(rcdb.default_store().read_record("imported").value) == object_digest(value)
    assert client.post("/api/v1/dbmanager/import", json=args).status_code >= 400
