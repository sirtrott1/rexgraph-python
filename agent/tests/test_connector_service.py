"""Tests for agent.connectors.service and the rexgraph connect CLI."""

from __future__ import annotations

from agent.cli import connect as cli
from agent.connectors import service as svc


def _shop_file(path):
    from rexgraph import RexGraph
    from rexgraph.io import save
    value = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])
    save(path, value, format="rcbd")
    return "rcbd://" + str(path)


# service layer

def test_list_connectors_groups_and_reports_capabilities():
    by = {r["connector"]: r for r in svc.list_connectors()}
    assert {"NativeFileConnector", "RCDBConnector"} <= by.keys()
    assert {"SQLConnector", "WarehouseConnector"} <= by.keys()
    assert {s["scheme"] for s in by["NativeFileConnector"]["schemes"]} == {"rcbd", "rcbf"}
    assert all(s["driver_available"] for s in by["NativeFileConnector"]["schemes"])


def test_native_drivers_need_no_sql_extras():
    for scheme in ("ontology", "rcdb", "rcbd", "rcbf"):
        assert svc.driver_status(scheme) == {"available": True, "hint": ""}
    assert svc.driver_status("snowflake")["available"] is False


def test_read_validate_ingest_roundtrip(tmp_path):
    uri = _shop_file(tmp_path / "source.rcbd")
    summary = svc.read(uri)
    assert (summary["nV"], summary["nE"]) == (4, 2)
    report = svc.validate(uri)
    assert report.ok, str(report)
    destination = "local://" + str(tmp_path / "records")
    assert svc.ingest(uri, "shop", store_uri=destination, tags=["demo"])["stored_as"] == "shop"
    from contextlib import closing
    from agent.rcdb import open_store
    from rexgraph.io import load
    from rexgraph.object_identity import object_digest
    with closing(open_store(destination)) as reopened:
        assert object_digest(reopened.get("shop")) == object_digest(load(tmp_path / "source.rcbd"))


def test_read_in_memory_shape():
    summ = svc.read("edges", source=[("x", "y", 2.0), ("y", "z", 1.0)])
    assert summ["nV"] == 3 and summ["nE"] == 2 and summ["weighted"]


# CLI

def test_cli_list_runs(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "NativeFileConnector" in out and "RCDBConnector" in out
    assert "SQLConnector" in out


def test_cli_read_and_validate(tmp_path, capsys):
    uri = _shop_file(tmp_path / "source.rcbd")
    assert cli.main(["read", uri]) == 0
    assert "nV=4" in capsys.readouterr().out
    assert cli.main(["validate", uri]) == 0


def test_cli_ingest(tmp_path):
    uri = _shop_file(tmp_path / "source.rcbd")
    destination = "local://" + str(tmp_path / "records")
    assert cli.main(["ingest", uri, "--store", destination, "--id", "shop", "--tags", "a,b"]) == 0
    from contextlib import closing
    from agent.rcdb import open_store
    with closing(open_store(destination)) as store:
        assert store.get("shop").nE == 2


def test_cli_resolves_saved_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("REXGRAPH_SECRETS_URI", "file://" + str(tmp_path / "conns.json"))
    from agent.secrets import open_secret_store
    uri = _shop_file(tmp_path / "source.rcbd")
    open_secret_store().put("shop", uri, "rcbd")
    assert cli._resolve("shop") == uri


def test_rcdb_ingest_uses_checked_selected_version_copy(tmp_path, monkeypatch):
    from contextlib import closing
    from agent import rcdb
    from rexgraph import RexGraph
    from rexgraph.object_identity import object_digest
    with closing(rcdb.LocalStore(str(tmp_path / "from"))) as source, \
         closing(rcdb.LocalStore(str(tmp_path / "to"))) as destination:
        old = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])
        source.put("original", old, tags=["retained"], valid_from=10.0, valid_to=20.0)
        source.put("original", RexGraph.from_cells([2, [[0, 1]]]))
        actual = rcdb.copy_record
        calls = []
        def copy(*args, **kwargs):
            calls.append((args[2].id, args[2].version))
            return actual(*args, **kwargs)
        monkeypatch.setattr(rcdb, "copy_record", copy)
        result = svc.ingest("rcdb", "copied", store=destination,
            source={"record_id": "original", "version": 1, "store": source})
        assert calls == [("original", 1)] and result["stored_as"] == "copied"
        selected = destination.read_record("copied")
        assert selected.record.valid_from == 10.0 and selected.record.valid_to == 20.0
        assert selected.record.signature["tags"] == ["retained"]
        assert object_digest(selected.value) == object_digest(old)
